# src/feature_extractor_v2.py
"""
Extended feature extractor: 23 → 40 features
Adds: conditions, rollback risk, cross-account, compliance
"""

import pandas as pd
import numpy as np
import pickle
import networkx as nx
from math import log2
from tqdm import tqdm


class ExtendedFeatureExtractor:
    """Extracts 40 features from IAM graph"""
    
    DANGEROUS_PATTERNS = [
        'iam:CreatePolicyVersion', 'iam:AttachUserPolicy',
        'iam:PutUserPolicy', 'iam:CreateUser', 'iam:CreateAccessKey',
        'iam:AddUserToGroup', 'iam:UpdateAssumeRolePolicy',
        'iam:SetDefaultPolicyVersion', 'iam:PassRole',
        'sts:AssumeRole', 'lambda:CreateFunction', 'lambda:UpdateFunctionCode',
        'glue:CreateJob', 'cloudformation:CreateStack',
        'codebuild:StartBuild', 'ec2:RunInstances',
        'iam:CreateRole', 'iam:DeletePolicyVersion',
        'iam:AttachRolePolicy', 'iam:AttachGroupPolicy'
    ]
    
    COMPLIANCE_THRESHOLDS = {
        'PCI_DSS_7_1_2': {'unused_ratio': 0.5},
        'NIST_800_53_AC6': {'dangerous_actions': 5},
        'SOC2_CC6_3': {'requires_mfa': True},
        'ISO_27001_A9_2_3': {'iam_write_max': 3}
    }
    
    def __init__(self, graph):
        self.graph = graph
    
        print("Precomputing expensive graph metrics (one-time)...")
    
        # ── FIX: Convert multigraph to simple graph for metrics ──
        # nx.clustering and some other metrics don't support MultiDiGraph
        print("Converting to simple graph for metric computation...")
        simple_graph = nx.DiGraph()
        for u, v, data in graph.edges(data=True):
            if not simple_graph.has_edge(u, v):
                simple_graph.add_edge(u, v, **data)
        for node, data in graph.nodes(data=True):
            simple_graph.add_node(node, **data)
    
        print("Computing betweenness centrality (sampled)...")
        self.betweenness = nx.betweenness_centrality(simple_graph, k=100)
    
        print("Computing PageRank...")
        self.pagerank = nx.pagerank(simple_graph, alpha=0.85, max_iter=100)
    
        print("Computing clustering coefficient...")
        # Use undirected simple graph for clustering
        self.clustering = nx.clustering(simple_graph.to_undirected())
    
        print("Metrics ready.")
    
        # Keep reference to simple graph for path queries (faster)
        self.simple_graph = simple_graph

    
    def extract(self, policy_id):
        """Extract all 40 features for one policy"""
        
        node_data = self.graph.nodes.get(policy_id, {})
        if not node_data:
            return None
        
        features = {}
        features['policy_id'] = policy_id
        
        # ── STRUCTURAL FEATURES (12) ────────────────────────────
        
        out_deg = self.graph.out_degree(policy_id)
        in_deg = self.graph.in_degree(policy_id)
        
        successors = list(self.graph.successors(policy_id))
        predecessors = list(self.graph.predecessors(policy_id))
        
        services = [n for n in successors 
                    if self.graph.nodes[n].get('type') == 'service']
        resources = [n for n in successors 
                     if self.graph.nodes[n].get('type') == 'resource']
        entities = [n for n in predecessors 
                    if self.graph.nodes[n].get('type') in ['user', 'role']]
        external = [n for n in successors 
                    if self.graph.nodes[n].get('type') == 'external']
        
        # Ego network
        try:
            ego = nx.ego_graph(self.graph, policy_id, radius=2)
            ego_density = nx.density(ego) if len(ego) > 1 else 0.0
        except:
            ego_density = 0.0
        
        # Shortest path to admin
        admin_nodes = [n for n in self.graph.nodes()
                       if 'Admin' in str(n) and 
                       self.graph.nodes[n].get('type') in ['policy', 'role']]
        shortest_to_admin = 999
        for admin in admin_nodes[:5]:
            try:
                if nx.has_path(self.graph, policy_id, admin):
                    path_len = nx.shortest_path_length(
                        self.graph, policy_id, admin
                    )
                    shortest_to_admin = min(shortest_to_admin, path_len)
            except:
                pass
        
        features['out_degree'] = out_deg
        features['in_degree'] = in_deg
        features['betweenness_centrality'] = self.betweenness.get(policy_id, 0.0)
        features['pagerank'] = self.pagerank.get(policy_id, 0.0)
        features['clustering_coefficient'] = self.clustering.get(policy_id, 0.0)
        features['ego_network_density'] = ego_density
        features['shortest_path_to_admin'] = shortest_to_admin
        features['attachment_count'] = len(entities)
        features['service_count'] = len(services)
        features['resource_count'] = len(resources)
        features['cross_account_edge_count'] = len(external)
        features['subgraph_modularity'] = ego_density * 0.5  # Approximation
        
        # ── SEMANTIC FEATURES (10) ──────────────────────────────
        
        actions = node_data.get('actions', [])
        if not actions:
            actions = []
        
        has_wildcard_action = 1 if '*' in actions else 0
        resource_list = node_data.get('resources', ['*'])
        has_wildcard_resource = 1 if '*' in resource_list else 0
        
        service_wildcards = sum(1 for a in actions if ':*' in str(a))
        wildcard_ratio = service_wildcards / max(len(actions), 1)
        wildcard_entropy = -wildcard_ratio * log2(wildcard_ratio + 1e-10)
        specificity = 1.0 / (1.0 + wildcard_entropy)
        
        dangerous_count = sum(
            1 for action in actions
            for pattern in self.DANGEROUS_PATTERNS
            if pattern.lower() in str(action).lower()
        )
        
        action_services = [str(a).split(':')[0] for a in actions if ':' in str(a)]
        from collections import Counter
        svc_counts = Counter(action_services)
        if svc_counts:
            probs = [c / sum(svc_counts.values()) for c in svc_counts.values()]
            action_diversity = -sum(p * log2(p + 1e-10) for p in probs)
        else:
            action_diversity = 0.0
        
        arn_specificity = sum(
            1 for r in resource_list 
            if r != '*' and '/*' not in str(r)
        ) / max(len(resource_list), 1)
        
        unique_services = len(set(action_services))
        overlap = 1.0 - (unique_services / max(len(actions), 1))
        
        dangerous_combos = [
            ('s3', 'lambda'), ('iam', 'sts'), ('ec2', 'iam'),
            ('dynamodb', 'lambda'), ('secretsmanager', 'lambda'),
            ('glue', 'iam'), ('codebuild', 'iam')
        ]
        cross_chain = sum(
            1 for (s1, s2) in dangerous_combos
            if s1 in action_services and s2 in action_services
        )
        
        features['wildcard_entropy'] = wildcard_entropy
        features['specificity_score'] = specificity
        features['dangerous_action_count'] = dangerous_count
        features['has_wildcard_action'] = has_wildcard_action
        features['has_wildcard_resource'] = has_wildcard_resource
        features['service_wildcard_count'] = service_wildcards
        features['action_diversity'] = action_diversity
        features['resource_arn_specificity'] = arn_specificity
        features['permission_overlap_score'] = overlap
        features['cross_service_permission_chains'] = cross_chain
        
        # ── ESCALATION FEATURES (8) ─────────────────────────────
        
        passrole = 1 if any('PassRole' in str(a) for a in actions) else 0
        createpolicyver = 1 if any('CreatePolicyVersion' in str(a) for a in actions) else 0
        attachuserpol = 1 if any('AttachUserPolicy' in str(a) for a in actions) else 0
        
        iam_write_patterns = ['Create', 'Delete', 'Put', 'Attach',
                              'Detach', 'Update', 'Set', 'Add']
        iam_write_count = sum(
            1 for a in actions
            if str(a).startswith('iam:') and
            any(p in str(a) for p in iam_write_patterns)
        )
        
        # Escalation paths (uses entity nodes we added in Step 1)
        escalation_count = 0
        min_path_length = 999
        
        for entity in entities[:3]:  # Sample first 3 entities
            try:
                admin_nodes_sample = admin_nodes[:3]
                for admin in admin_nodes_sample:
                    if nx.has_path(self.graph, entity, admin):
                        path = nx.shortest_path(self.graph, entity, admin)
                        escalation_count += 1
                        min_path_length = min(min_path_length, len(path))
            except:
                pass
        
        esc_risk = (
            escalation_count * 0.3 +
            passrole * 2.0 +
            createpolicyver * 2.5 +
            attachuserpol * 2.0 +
            iam_write_count * 0.3
        )
        
        features['escalation_path_count'] = escalation_count
        features['min_escalation_path_length'] = min_path_length
        features['escalation_techniques_enabled'] = dangerous_count
        features['passrole_chain_exists'] = passrole
        features['createpolicyversion_exists'] = createpolicyver
        features['attachuserpolicy_exists'] = attachuserpol
        features['iam_write_permission_count'] = iam_write_count
        features['privilege_escalation_risk_score'] = min(esc_risk, 10.0)
        
        # ── CONDITION FEATURES (5) — NEW ────────────────────────
        
        conditions = node_data.get('conditions', {})
        cond_str = str(conditions)
        
        has_mfa = 1 if 'MultiFactorAuth' in cond_str else 0
        has_ip = 0
        ip_range = ''
        
        if 'SourceIp' in cond_str:
            has_ip = 1
            # Check if it's actually restrictive
            if '0.0.0.0/0' in cond_str or '*' in cond_str:
                has_ip = 0  # Not actually restrictive
        
        has_time = 1 if 'CurrentTime' in cond_str else 0
        
        protection = 0.0
        if has_mfa:   protection -= 0.30
        if has_ip:    protection -= 0.20
        if has_time:  protection -= 0.10
        
        # Is bounded by permission boundary?
        is_bounded = 0
        for entity in entities[:1]:
            entity_successors = list(self.graph.successors(entity))
            if any(self.graph.nodes[n].get('type') == 'boundary' 
                   for n in entity_successors):
                is_bounded = 1
        
        features['has_mfa_condition'] = has_mfa
        features['has_ip_restriction'] = has_ip
        features['has_time_restriction'] = has_time
        features['condition_protection_score'] = protection
        features['is_bounded'] = is_bounded
        
        # ── ROLLBACK & VERSION FEATURES (3) — NEW ───────────────
        
        version_nodes = [
            n for n in self.graph.successors(policy_id)
            if self.graph.nodes[n].get('type') == 'policy_version'
        ]
        
        version_count = len(version_nodes) if version_nodes else 1
        max_historical_risk = 0  # 0=LOW, 1=MEDIUM, 2=HIGH
        
        # If entity has SetDefaultPolicyVersion → rollback risk
        rollback_risk = 0.0
        for entity in entities[:3]:
            entity_actions = []
            for role in self.graph.successors(entity):
                for pol in self.graph.successors(role):
                    pol_data = self.graph.nodes[pol]
                    entity_actions.extend(pol_data.get('actions', []))
            
            if 'iam:SetDefaultPolicyVersion' in entity_actions:
                rollback_risk = 1.0
        
        features['policy_version_count'] = version_count
        features['max_historical_risk'] = max_historical_risk
        features['rollback_risk_score'] = rollback_risk
        
        # ── USAGE & COMPLIANCE FEATURES (2) — NEW ───────────────
        
        # Unused permission ratio (approximated without CloudTrail)
        # High out_degree + low attachment = likely over-provisioned
        if len(entities) == 0:
            unused_ratio = 0.5  # Unknown
        elif out_deg > 100 and len(entities) < 3:
            unused_ratio = 0.7  # Many permissions, few users = over-provisioned
        elif out_deg > 50:
            unused_ratio = 0.4
        else:
            unused_ratio = 0.1
        
        # Compliance violations
        violations = 0
        if unused_ratio > self.COMPLIANCE_THRESHOLDS['PCI_DSS_7_1_2']['unused_ratio']:
            violations += 1
        if dangerous_count > self.COMPLIANCE_THRESHOLDS['NIST_800_53_AC6']['dangerous_actions']:
            violations += 1
        if not has_mfa and (passrole or createpolicyver or attachuserpol):
            violations += 1
        if iam_write_count > self.COMPLIANCE_THRESHOLDS['ISO_27001_A9_2_3']['iam_write_max']:
            violations += 1
        
        features['unused_permission_ratio'] = unused_ratio
        features['compliance_violation_count'] = violations
        
        return features
    
    def extract_all(self, policy_ids, sample_size=1000):
        """Extract features for all policies"""
        
        all_features = []
        sampled = policy_ids[:sample_size]
        
        for pid in tqdm(sampled, desc="Extracting 40 features"):
            feat = self.extract(pid)
            if feat:
                all_features.append(feat)
        
        df = pd.DataFrame(all_features)
        print(f"\nExtracted {len(df)} policies × {len(df.columns)-1} features")
        return df


def run_extended_extraction():
    """Run the extended feature extraction"""
    
    print("Loading enhanced graph (with entities)...")
    
    # Try enhanced graph first, fall back to original
    try:
        with open('data/iam_graph_with_entities.pkl', 'rb') as f:
            graph = pickle.load(f)
        print("Using enhanced graph (with entities)")
    except:
        with open('data/iam_graph.pkl', 'rb') as f:
            graph = pickle.load(f)
        print("Using original graph (no entities - run Step 1 first)")
    
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Get policy nodes
    policy_ids = [
        n for n in graph.nodes()
        if graph.nodes[n].get('type') == 'policy'
    ]
    print(f"Found {len(policy_ids)} policy nodes")
    
    # Extract features
    extractor = ExtendedFeatureExtractor(graph)
    df = extractor.extract_all(policy_ids, sample_size=1000)
    
    # Save
    df.to_csv('data/graph_features_v2.csv', index=False)
    print(f"Saved: data/graph_features_v2.csv")
    print(f"Shape: {df.shape}")
    print(f"\nNew features added:")
    new_features = ['has_mfa_condition', 'has_ip_restriction', 'has_time_restriction',
                    'condition_protection_score', 'is_bounded', 'policy_version_count',
                    'max_historical_risk', 'rollback_risk_score', 'unused_permission_ratio',
                    'compliance_violation_count']
    for f in new_features:
        if f in df.columns:
            print(f"  ✅ {f}: mean={df[f].mean():.3f}")
    
    return df


if __name__ == "__main__":
    run_extended_extraction()

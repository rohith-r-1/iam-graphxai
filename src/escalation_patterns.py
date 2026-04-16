# escalation_patterns.py
from typing import List, Set
from dataclasses import dataclass

@dataclass
class EscalationTechnique:
    technique_id: str
    name: str
    required_permissions: List[str]
    description: str
    severity: str  # High, Medium, Low

# Based on Rhino Security Labs research
ESCALATION_TECHNIQUES = [
    EscalationTechnique(
        technique_id="T1",
        name="CreatePolicyVersion",
        required_permissions=["iam:CreatePolicyVersion"],
        description="Attacker can modify existing policy to add admin permissions",
        severity="High"
    ),
    EscalationTechnique(
        technique_id="T2",
        name="SetDefaultPolicyVersion",
        required_permissions=["iam:SetDefaultPolicyVersion"],
        description="Switch to previously created malicious policy version",
        severity="High"
    ),
    EscalationTechnique(
        technique_id="T3",
        name="CreateAccessKey",
        required_permissions=["iam:CreateAccessKey"],
        description="Create access keys for other users including admins",
        severity="High"
    ),
    EscalationTechnique(
        technique_id="T4",
        name="CreateLoginProfile",
        required_permissions=["iam:CreateLoginProfile"],
        description="Create console password for other users",
        severity="High"
    ),
    EscalationTechnique(
        technique_id="T5",
        name="UpdateAssumeRolePolicy",
        required_permissions=["iam:UpdateAssumeRolePolicy"],
        description="Modify trust policy to allow assuming privileged roles",
        severity="High"
    ),
    EscalationTechnique(
        technique_id="T6",
        name="AttachUserPolicy",
        required_permissions=["iam:AttachUserPolicy"],
        description="Attach admin policy to own user",
        severity="High"
    ),
    EscalationTechnique(
        technique_id="T7",
        name="AttachRolePolicy",
        required_permissions=["iam:AttachRolePolicy", "sts:AssumeRole"],
        description="Attach admin policy to assumable role",
        severity="High"
    ),
    EscalationTechnique(
        technique_id="T8",
        name="PutUserPolicy",
        required_permissions=["iam:PutUserPolicy"],
        description="Embed inline admin policy to own user",
        severity="High"
    ),
    EscalationTechnique(
        technique_id="T9",
        name="PutRolePolicy",
        required_permissions=["iam:PutRolePolicy", "sts:AssumeRole"],
        description="Embed inline admin policy to assumable role",
        severity="High"
    ),
    EscalationTechnique(
        technique_id="T10",
        name="AddUserToGroup",
        required_permissions=["iam:AddUserToGroup"],
        description="Add self to admin group",
        severity="High"
    ),
    EscalationTechnique(
        technique_id="T11",
        name="UpdateFunctionCode",
        required_permissions=["lambda:UpdateFunctionCode", "iam:PassRole"],
        description="Update Lambda function with privileged role to execute arbitrary code",
        severity="High"
    ),
    EscalationTechnique(
        technique_id="T12",
        name="PassRole",
        required_permissions=["iam:PassRole", "lambda:CreateFunction"],
        description="Create Lambda with privileged role",
        severity="High"
    ),
    # Add remaining 19 techniques from research papers
]

class EscalationDetector:
    """Detect privilege escalation paths in IAM graph"""
    
    def __init__(self, graph):
        self.graph = graph
        self.techniques = {t.technique_id: t for t in ESCALATION_TECHNIQUES}
        
    def check_technique_possible(self, node_id: str, technique: EscalationTechnique) -> bool:
        """
        Check if a node has permissions for an escalation technique
        
        Algorithm:
        1. Get all policies attached to node (direct + inherited)
        2. Extract all granted actions
        3. Check if technique's required permissions are subset
        """
        granted_actions = self._get_granted_actions(node_id)
        
        for required_perm in technique.required_permissions:
            if not self._action_matches(required_perm, granted_actions):
                return False
                
        return True
    
    def _get_granted_actions(self, node_id: str) -> Set[str]:
        """Get all actions granted to a node (transitively)"""
        actions = set()
        
        # BFS to find all policies
        visited = set()
        queue = [node_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            # Get outgoing edges
            for neighbor in self.graph.successors(current):
                edge_data = self.graph.get_edge_data(current, neighbor)
                
                if not edge_data:
                    continue
                    
                # Handle multiple edges
                for key, data in edge_data.items():
                    edge_type = data.get('type')
                    
                    if edge_type == 'grants_access':
                        # This is a policy granting access to a service
                        granted_actions = data.get('actions', [])
                        actions.update(granted_actions)
                    elif edge_type in ['attached_policy', 'assume_role', 'trust']:
                        # Follow these edges
                        queue.append(neighbor)
        
        return actions
    
    def _action_matches(self, required: str, granted_set: Set[str]) -> bool:
        """Check if required action is covered by granted actions (handle wildcards)"""
        
        # Exact match
        if required in granted_set:
            return True
            
        # Wildcard match
        if "*" in granted_set or "*:*" in granted_set:
            return True
            
        # Service wildcard (e.g., "iam:*" covers "iam:CreateUser")
        service = required.split(':')[0]
        if f"{service}:*" in granted_set:
            return True
            
        return False
    
    def find_escalation_paths(self, start_node: str, max_depth: int = 5) -> List[dict]:
        """
        Find all privilege escalation paths from a starting node
        
        Returns: List of path dictionaries with:
            - path: List of nodes
            - techniques: Techniques used at each step
            - risk_score: Computed risk
        """
        paths = []
        
        def dfs(current_node, path, techniques_used, depth):
            if depth > max_depth:
                return
                
            # Check if current node has admin-equivalent permissions
            if self._is_admin_equivalent(current_node):
                paths.append({
                    'path': path.copy(),
                    'techniques': techniques_used.copy(),
                    'length': len(path),
                    'risk_score': self._compute_path_risk(path, techniques_used)
                })
                return
            
            # Try each escalation technique
            for tech_id, technique in self.techniques.items():
                if self.check_technique_possible(current_node, technique):
                    # Simulate applying this technique
                    next_nodes = self._get_reachable_after_technique(current_node, technique)
                    
                    for next_node in next_nodes:
                        if next_node not in path:  # Avoid cycles
                            dfs(
                                next_node,
                                path + [next_node],
                                techniques_used + [technique.name],
                                depth + 1
                            )
        
        dfs(start_node, [start_node], [], 0)
        return paths
    
    def _is_admin_equivalent(self, node_id: str) -> bool:
        """Check if node has admin-equivalent permissions"""
        actions = self._get_granted_actions(node_id)
        
        # Check for admin indicators
        admin_indicators = [
            "*",
            "*:*",
            "iam:*",
            "sts:AssumeRole"  # With wildcard resource
        ]
        
        for indicator in admin_indicators:
            if indicator in actions:
                return True
                
        return False
    
    def _get_reachable_after_technique(self, node_id: str, technique: EscalationTechnique) -> List[str]:
        """
        Get nodes reachable after applying an escalation technique
        
        This is a simplification - in reality you'd simulate the graph modification
        """
        # For simplicity, return nodes connected by trust/assume relationships
        reachable = []
        
        for neighbor in self.graph.successors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            
            for key, data in edge_data.items():
                if data.get('type') in ['assume_role', 'trust']:
                    reachable.append(neighbor)
                    
        return reachable
    
    def _compute_path_risk(self, path: List[str], techniques: List[str]) -> float:
        """Compute risk score for an escalation path"""
        
        # Shorter paths are more dangerous
        length_penalty = 1.0 / len(path)
        
        # More techniques used = higher risk
        technique_weight = len(techniques) * 0.2
        
        # High severity techniques increase risk
        severity_score = sum(
            1.0 if any(t.name in techniques and t.severity == "High" for t in ESCALATION_TECHNIQUES)
            else 0.5
            for _ in techniques
        )
        
        return length_penalty + technique_weight + severity_score

# Usage
if __name__ == "__main__":
    import networkx as nx
    
    # Load graph
    import os, pickle

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    GRAPH_PATH = os.path.join(BASE_DIR, "data", "iam_graph.pkl")

    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)
    
    # Create detector
    detector = EscalationDetector(graph)
    
    # Find paths from each user node
    for node, data in graph.nodes(data=True):
        if data.get('type') == 'user':
            paths = detector.find_escalation_paths(node)
            if paths:
                print(f"\nUser {node} has {len(paths)} escalation paths:")
                for path in paths[:3]:  # Show top 3
                    print(f"  Path: {' → '.join(path['path'])}")
                    print(f"  Techniques: {', '.join(path['techniques'])}")
                    print(f"  Risk: {path['risk_score']:.2f}")

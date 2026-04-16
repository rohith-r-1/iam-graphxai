# graph_builder.py
import pickle
import os
import networkx as nx
from typing import List
from policy_parser import IAMPolicy
from graph_schema import NodeType, EdgeType, GraphNode, GraphEdge

class IAMGraphBuilder:
    """Construct graph from parsed IAM policies"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self.node_counter = 0
        
    def add_node(self, node_id: str, node_type: NodeType, **attributes):
        """Add node with attributes"""
        self.graph.add_node(
            node_id,
            type=node_type.value,
            **attributes
        )
        
    def add_edge(self, source: str, target: str, edge_type: EdgeType, **attributes):
        """Add directed edge"""
        self.graph.add_edge(
            source,
            target,
            type=edge_type.value,
            **attributes
        )
        
    def build_from_policies(self, policies: List[IAMPolicy]):
        """
        Main graph construction algorithm
        
        Steps:
        1. Create policy nodes
        2. Create service nodes from actions
        3. Create resource nodes
        4. Link policies to entities (users/roles)
        5. Create trust relationships
        """
        
        # Track created nodes to avoid duplicates
        service_nodes = set()
        
        for policy in policies:
            # Add policy node
            policy_node_id = f"policy:{policy.policy_name}"
            self.add_node(
                policy_node_id,
                NodeType.POLICY,
                name=policy.policy_name,
                statement_count=len(policy.statements)
            )
            
            # Process each statement
            for stmt in policy.statements:
                # Extract services from actions
                for action in stmt.actions:
                    if action == "*":
                        # Wildcard - add edge to special "all_services" node
                        service = "all_services"
                    else:
                        # Extract service (e.g., "iam:CreateUser" → "iam")
                        service = action.split(':')[0]
                    
                    service_node_id = f"service:{service}"
                    
                    if service_node_id not in service_nodes:
                        self.add_node(
                            service_node_id,
                            NodeType.SERVICE,
                            name=service
                        )
                        service_nodes.add(service_node_id)
                    
                    # Add edge: Policy → Service
                    self.add_edge(
                        policy_node_id,
                        service_node_id,
                        EdgeType.GRANTS_ACCESS,
                        actions=stmt.actions,
                        effect=stmt.effect,
                        has_wildcard=(action == "*")
                    )
                    
                # Process resources
                for resource in stmt.resources:
                    resource_node_id = f"resource:{resource}"
                    self.add_node(
                        resource_node_id,
                        NodeType.RESOURCE,
                        arn=resource,
                        is_wildcard=(resource == "*")
                    )
                    
                    self.add_edge(
                        policy_node_id,
                        resource_node_id,
                        EdgeType.ACTS_ON
                    )
                    
                # Process principals (for trust policies)
                if stmt.principals:
                    principal_type = list(stmt.principals.keys())[0]
                    principal_values = stmt.principals[principal_type]
                    
                    if not isinstance(principal_values, list):
                        principal_values = [principal_values]
                    
                    for principal in principal_values:
                        # Parse principal (could be ARN, service, account ID)
                        principal_node_id = f"principal:{principal}"
                        
                        # Determine node type
                        if "role" in principal.lower():
                            node_type = NodeType.ROLE
                        elif "user" in principal.lower():
                            node_type = NodeType.USER
                        else:
                            node_type = NodeType.SERVICE
                            
                        self.add_node(
                            principal_node_id,
                            node_type,
                            identifier=principal
                        )
                        
                        # Trust relationship edge
                        self.add_edge(
                            principal_node_id,
                            policy_node_id,
                            EdgeType.TRUST_RELATIONSHIP,
                            effect=stmt.effect
                        )
            
            # Link policy to attached entities
            for entity in policy.attached_to:
                entity_node_id = f"entity:{entity}"
                
                # Infer entity type from name
                if "role" in entity.lower():
                    entity_type = NodeType.ROLE
                elif "group" in entity.lower():
                    entity_type = NodeType.GROUP
                else:
                    entity_type = NodeType.USER
                    
                self.add_node(
                    entity_node_id,
                    entity_type,
                    name=entity
                )
                
                self.add_edge(
                    entity_node_id,
                    policy_node_id,
                    EdgeType.ATTACHED_POLICY
                )
        
        return self.graph
    
    def save_graph(self, filename: str):
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.graph, f)

    def load_graph(self, filename: str):
        """Load graph from disk using pickle"""
        with open(filename, "rb") as f:
            self.graph = pickle.load(f)
        return self.graph
    
    def get_statistics(self):
        """Print graph statistics"""
        print(f"Nodes: {self.graph.number_of_nodes()}")
        print(f"Edges: {self.graph.number_of_edges()}")
        
        # Count by node type
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            ntype = data.get('type', 'unknown')
            node_types[ntype] = node_types.get(ntype, 0) + 1
            
        print("Node types:")
        for ntype, count in node_types.items():
            print(f"  {ntype}: {count}")

# Usage
if __name__ == "__main__":
    import os
    from policy_parser import PolicyParser

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # E:\iam-graph-xai
    POLICIES_DIR = os.path.join(
        BASE_DIR,
        "data",
        "aws-iam-managed-policies",
        "data",
        "json"
    )

    parser = PolicyParser()
    policies = parser.parse_directory(POLICIES_DIR)
    print(f"Parsed {len(policies)} policies for graph building")

    builder = IAMGraphBuilder()
    graph = builder.build_from_policies(policies)

    GRAPH_PATH = os.path.join(BASE_DIR, "data", "iam_graph.pkl")
    builder.save_graph(GRAPH_PATH)
    builder.get_statistics()

# ADD THIS TO THE END OF src/graph_builder.py

class EntityAttachmentSimulator:
    """
    Simulates realistic enterprise IAM structure.
    Adds 500+ users + 6 roles to the existing graph.
    Uses fuzzy policy name matching to handle any node ID format.
    """

    ENTERPRISE_ROLES = {
        'Developer': {
            'policies': [
                'AWSLambda_FullAccess', 'AmazonDynamoDBFullAccess',
                'AmazonS3ReadOnlyAccess', 'CloudWatchLogsReadOnlyAccess'
            ],
            'user_count': 150,
            'risk_tier': 'medium'
        },
        'DataScientist': {
            'policies': [
                'AmazonS3FullAccess', 'AmazonSageMakerFullAccess',
                'AmazonEC2ReadOnlyAccess'
            ],
            'user_count': 50,
            'risk_tier': 'medium'
        },
        'DevOps': {
            'policies': [
                'AmazonEC2FullAccess', 'AWSCloudFormationFullAccess',
                'AWSLambda_FullAccess'
            ],
            'user_count': 40,
            'risk_tier': 'high'
        },
        'SecurityAuditor': {
            'policies': [
                'SecurityAudit', 'ReadOnlyAccess'
            ],
            'user_count': 10,
            'risk_tier': 'low'
        },
        'Admin': {
            'policies': [
                'AdministratorAccess'
            ],
            'user_count': 5,
            'risk_tier': 'critical'
        },
        'ReadOnly': {
            'policies': [
                'ReadOnlyAccess', 'ViewOnlyAccess'
            ],
            'user_count': 300,
            'risk_tier': 'low'
        }
    }

    def _build_policy_lookup(self, graph):
        """
        Build a flexible lookup map: cleaned_name → node_id
        Handles all possible node ID formats:
          - "policy:AWSLambdaFullAccess"
          - "policy:arn:aws:iam::aws:policy/AWSLambdaFullAccess"
          - "AWSLambdaFullAccess"
        """
        lookup = {}

        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            if node_data.get('type') != 'policy':
                continue

            # Strategy 1: strip prefix → "AWSLambdaFullAccess"
            base = node_id

            # Remove "policy:" prefix if present
            if base.startswith('policy:'):
                base = base[len('policy:'):]

            # Remove ARN prefix if present
            # e.g. "arn:aws:iam::aws:policy/AWSLambdaFullAccess" → "AWSLambdaFullAccess"
            if 'policy/' in base:
                base = base.split('policy/')[-1]

            # Take last segment after any remaining colons
            if ':' in base:
                base = base.split(':')[-1]

            # Store lowercase → node_id
            key = base.lower().strip()
            if key:
                lookup[key] = node_id

        return lookup

    def _find_policy_node(self, policy_name, graph, lookup):
        """
        Find a policy node ID using multiple matching strategies.
        Returns the node_id string or None if not found.
        """
        search = policy_name.lower().strip()

        # Strategy 1: Direct node ID match
        candidates = [
            f"policy:{policy_name}",
            policy_name
        ]
        for c in candidates:
            if c in graph.nodes():
                return c

        # Strategy 2: Exact lowercase match in lookup
        if search in lookup:
            return lookup[search]

        # Strategy 3: Partial match — search key contained in a lookup key
        for key, node_id in lookup.items():
            if search == key:
                return node_id

        # Strategy 4: Substring match — policy name is substring of node name
        for key, node_id in lookup.items():
            if search in key:
                return node_id

        # Strategy 5: Reverse substring — node name is substring of policy name
        for key, node_id in lookup.items():
            if key in search and len(key) > 5:  # Avoid short spurious matches
                return node_id

        return None  # Not found

    def inject_into_graph(self, graph):
        """
        Injects user and role nodes into existing policy graph.

        Before:
            policy:AWSLambdaFullAccess → service:lambda
        After:
            user:Developer_0 → role:Developer → policy:AWSLambdaFullAccess → service:lambda
        """
        print("Injecting entity attachments...")

        # Build flexible policy lookup once
        policy_lookup = self._build_policy_lookup(graph)

        print(f"  Policy lookup built: {len(policy_lookup)} entries")
        print(f"  Sample lookup keys: {list(policy_lookup.keys())[:5]}")

        users_added = 0
        roles_added = 0
        total_policy_attachments = 0

        for role_name, config in self.ENTERPRISE_ROLES.items():
            role_id = f"role:{role_name}"

            # Add role node
            graph.add_node(
                role_id,
                type='role',
                risk_tier=config['risk_tier'],
                department=role_name
            )
            roles_added += 1

            # Attach policies to role using flexible matching
            attached_count = 0
            for policy_name in config['policies']:
                node_id = self._find_policy_node(policy_name, graph, policy_lookup)

                if node_id:
                    graph.add_edge(
                        role_id,
                        node_id,
                        type='assigned_to',
                        relationship='has_policy'
                    )
                    attached_count += 1
                    total_policy_attachments += 1
                else:
                    print(f"    WARNING: '{policy_name}' not found in graph")

            print(f"  Role {role_name}: {attached_count}/{len(config['policies'])} policies attached")

            # Create users for this role
            for i in range(config['user_count']):
                user_id = f"user:{role_name}_{i}"
                graph.add_node(
                    user_id,
                    type='user',
                    department=role_name,
                    user_index=i
                )
                graph.add_edge(
                    user_id,
                    role_id,
                    type='assumes_role',
                    relationship='member_of'
                )
                users_added += 1

        # ── CROSS-ROLE ACCESS (Realistic privilege accumulation) ──────────

        # 5% of Developers can also assume DevOps role
        dev_users = [
            n for n in graph.nodes()
            if graph.nodes[n].get('type') == 'user'
            and graph.nodes[n].get('department') == 'Developer'
        ][:8]

        cross_role_count = 0
        for user in dev_users:
            if 'role:DevOps' in graph.nodes():
                graph.add_edge(
                    user, 'role:DevOps',
                    type='assumes_role',
                    relationship='cross_role_access'
                )
                cross_role_count += 1

        # 2 DevOps users have break-glass Admin access
        devops_users = [
            n for n in graph.nodes()
            if graph.nodes[n].get('type') == 'user'
            and graph.nodes[n].get('department') == 'DevOps'
        ][:2]

        breakglass_count = 0
        for user in devops_users:
            if 'role:Admin' in graph.nodes():
                graph.add_edge(
                    user, 'role:Admin',
                    type='assumes_role',
                    relationship='emergency_access'
                )
                breakglass_count += 1

        # ── SUMMARY ──────────────────────────────────────────────────────

        print(f"\nEntity injection complete:")
        print(f"  Users added:              {users_added}")
        print(f"  Roles added:              {roles_added}")
        print(f"  Policy attachments:       {total_policy_attachments}")
        print(f"  Cross-role access edges:  {cross_role_count}")
        print(f"  Break-glass Admin edges:  {breakglass_count}")
        print(f"  Total graph nodes:        {graph.number_of_nodes()}")
        print(f"  Total graph edges:        {graph.number_of_edges()}")

        if total_policy_attachments == 0:
            print("\n  ⚠ No policies attached. Running diagnostic...")
            self._diagnose(graph, policy_lookup)

        return graph

    def _diagnose(self, graph, policy_lookup):
        """
        Runs when 0 policies attach. Prints debug info to find the mismatch.
        """
        print("\n  ── DIAGNOSTIC ──────────────────────────────────────")

        # Show actual policy node format
        policy_nodes = [
            n for n in graph.nodes()
            if graph.nodes[n].get('type') == 'policy'
        ]
        print(f"  Total policy nodes: {len(policy_nodes)}")
        print(f"  Sample raw node IDs:")
        for p in policy_nodes[:8]:
            print(f"    {repr(p)}")

        # Show what lookup keys look like
        print(f"\n  Sample lookup keys:")
        for k in list(policy_lookup.keys())[:8]:
            print(f"    {repr(k)}")

        # Try to find known policies manually
        test_names = ['AdministratorAccess', 'ReadOnlyAccess',
                      'AmazonS3FullAccess', 'SecurityAudit']
        print(f"\n  Manual search for known policy names:")
        for name in test_names:
            found = [n for n in policy_nodes
                     if name.lower() in n.lower()]
            print(f"    '{name}' → {found[:2] if found else 'NOT FOUND'}")

        print("  ────────────────────────────────────────────────────")
        print("  Copy the sample node IDs above and share them.")


# ─────────────────────────────────────────────────────────────────────────────
# Runner — called when you run: python src/graph_builder.py
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_graph_with_entities():
    import pickle
    import os

    print("Loading existing graph...")
    with open('data/iam_graph.pkl', 'rb') as f:
        graph = pickle.load(f)

    print(f"Original graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges")

    simulator = EntityAttachmentSimulator()
    graph = simulator.inject_into_graph(graph)

    os.makedirs('data', exist_ok=True)
    with open('data/iam_graph_with_entities.pkl', 'wb') as f:
        pickle.dump(graph, f)

    print(f"\nSaved: data/iam_graph_with_entities.pkl")
    return graph


if __name__ == "__main__":
    rebuild_graph_with_entities()


# ─────────────────────────────────────────────────────────────
# MODIFY YOUR EXISTING build_graph() function to call this
# Find your build_graph() function and add at the end:
# ─────────────────────────────────────────────────────────────

def rebuild_graph_with_entities():
    """
    Run this ONCE to rebuild your graph with entity attachments.
    Saves new graph to data/iam_graph_with_entities.pkl
    """
    import pickle
    import os
    
    print("Loading existing graph...")
    with open('data/iam_graph.pkl', 'rb') as f:
        graph = pickle.load(f)
    
    print(f"Original graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Inject entities
    simulator = EntityAttachmentSimulator()
    graph = simulator.inject_into_graph(graph)
    
    # Save enhanced graph
    with open('data/iam_graph_with_entities.pkl', 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"\nEnhanced graph saved to data/iam_graph_with_entities.pkl")
    print(f"Final: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    return graph


if __name__ == "__main__":
    rebuild_graph_with_entities()

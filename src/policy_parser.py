# policy_parser.py
import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PolicyStatement:
    effect: str  # Allow or Deny
    actions: List[str]
    resources: List[str]
    conditions: Optional[Dict] = None
    principals: Optional[Dict] = None
    
@dataclass
class IAMPolicy:
    policy_id: str
    policy_name: str
    statements: List[PolicyStatement]
    policy_type: str  # identity-based, resource-based, trust
    attached_to: List[str]  # Users, roles, or groups
    
class PolicyParser:
    """Parse AWS IAM policy JSON into structured format"""
    
    def __init__(self):
        self.policies = []
        
    def parse_policy_document(self, policy_json: dict, policy_metadata: dict) -> IAMPolicy:
        """
        Parse a single IAM policy document
        
        Args:
            policy_json: The policy document JSON
            policy_metadata: Metadata (name, attached entities, etc.)
        """
        statements = []
        
        # Handle both single statement and array
        stmt_list = policy_json.get('Statement', [])
        if not isinstance(stmt_list, list):
            stmt_list = [stmt_list]
            
        for stmt in stmt_list:
            # Parse actions
            actions = stmt.get('Action', [])
            if isinstance(actions, str):
                actions = [actions]
                
            # Parse resources
            resources = stmt.get('Resource', [])
            if isinstance(resources, str):
                resources = [resources]
                
            # Parse conditions
            conditions = stmt.get('Condition')
            
            # Parse principals (for trust policies)
            principals = stmt.get('Principal')
            
            statement = PolicyStatement(
                effect=stmt.get('Effect', 'Allow'),
                actions=actions,
                resources=resources,
                conditions=conditions,
                principals=principals
            )
            statements.append(statement)
            
        return IAMPolicy(
            policy_id=policy_metadata.get('policy_id'),
            policy_name=policy_metadata.get('policy_name'),
            statements=statements,
            policy_type=policy_metadata.get('policy_type', 'identity-based'),
            attached_to=policy_metadata.get('attached_to', [])
        )
    
    def parse_directory(self, directory: str) -> List[IAMPolicy]:
        """Parse all policies in a directory"""
        import glob
        
        policy_files = glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)
        
        for policy_file in policy_files:
            with open(policy_file, 'r') as f:
                try:
                    policy_data = json.load(f)
                    # Extract metadata from filename or file structure
                    metadata = {
                        'policy_id': policy_file,
                        'policy_name': policy_file.split('/')[-1].replace('.json', ''),
                        'attached_to': []  # Would need to infer or have separate mapping
                    }
                    policy = self.parse_policy_document(policy_data, metadata)
                    self.policies.append(policy)
                except Exception as e:
                    print(f"Error parsing {policy_file}: {e}")
                    
        return self.policies

# Test parser
if __name__ == "__main__":
    import os
    
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
    print(f"Parsed {len(policies)} policies")
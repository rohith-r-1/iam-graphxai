# graph_schema.py
from enum import Enum
from dataclasses import dataclass

class NodeType(Enum):
    USER = "user"
    ROLE = "role"
    GROUP = "group"
    POLICY = "policy"
    SERVICE = "service"
    RESOURCE = "resource"

class EdgeType(Enum):
    ASSUME_ROLE = "assume_role"         # User/Role → Role
    ATTACHED_POLICY = "attached_policy"  # User/Role/Group → Policy
    MEMBER_OF = "member_of"              # User → Group
    GRANTS_ACCESS = "grants_access"      # Policy → Service
    ACTS_ON = "acts_on"                  # Policy → Resource
    TRUST_RELATIONSHIP = "trust"         # Role → Role/User/Service

@dataclass
class GraphNode:
    node_id: str
    node_type: NodeType
    attributes: dict
    
@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    attributes: dict

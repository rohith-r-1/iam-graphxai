# src/debug_graph.py
import pickle
from collections import Counter

print("Loading graph...")
with open('D:/iam-graph-xai/data/iam_graph.pkl', 'rb') as f:
    G = pickle.load(f)

print(f"Graph type : {type(G).__name__}")
print(f"Nodes      : {G.number_of_nodes()}")
print(f"Edges      : {G.number_of_edges()}")
print()

# ── All node attribute keys ───────────────────────────────────────
node_keys = set()
for _, d in G.nodes(data=True):
    node_keys.update(d.keys())
print(f"Node attribute keys : {sorted(node_keys)}")
print()

# ── All node types ────────────────────────────────────────────────
types = Counter(
    d.get('type', d.get('node_type', 'MISSING'))
    for _, d in G.nodes(data=True)
)
print("Node types:")
for t, cnt in types.most_common():
    print(f"  {t!r:30s} : {cnt}")
print()

# ── All edge attribute keys ───────────────────────────────────────
edge_keys = set()
for _, _, d in list(G.edges(data=True))[:10000]:
    edge_keys.update(d.keys())
print(f"Edge attribute keys : {sorted(edge_keys)}")
print()

# ── Sample 3 policy nodes in full detail ─────────────────────────
policy_nodes = [
    (n, d) for n, d in G.nodes(data=True)
    if str(d.get('type', d.get('node_type', ''))).lower() == 'policy'
]
print(f"Policy nodes found : {len(policy_nodes)}")
print()

for n, d in policy_nodes[:3]:
    print(f"{'='*55}")
    print(f"  Policy node id  : {n}")
    print(f"  All node attrs  : {dict(d)}")

    out_edges = list(G.out_edges(n, data=True, keys=True))
    in_edges  = list(G.in_edges(n,  data=True, keys=True))

    print(f"  Out-edges total : {len(out_edges)}")
    for u, v, k, ed in out_edges[:5]:
        vd = dict(G.nodes[v])
        print(f"    → v={v}  edge={dict(ed)}  v_attrs={vd}")

    print(f"  In-edges total  : {len(in_edges)}")
    for u, v, k, ed in in_edges[:3]:
        ud = dict(G.nodes[u])
        print(f"    ← u={u}  edge={dict(ed)}  u_attrs={ud}")
    print()

# ── Sample 3 non-policy nodes to see action/resource nodes ───────
non_policy = [
    (n, d) for n, d in G.nodes(data=True)
    if str(d.get('type', d.get('node_type', ''))).lower() != 'policy'
]
print(f"Non-policy node sample (first 3):")
for n, d in non_policy[:3]:
    print(f"  node={n}  attrs={dict(d)}")
print()

# ── Check what keys store actions on policy nodes ─────────────────
print("Checking action-related keys on policy nodes:")
action_keys = ['actions', 'action_list', 'permissions',
               'statements', 'policy_document', 'document',
               'action', 'allowed_actions']
for n, d in policy_nodes[:5]:
    found = {k: d[k] for k in action_keys if k in d}
    if found:
        print(f"  node={n} → {found}")
    else:
        print(f"  node={n} → (none of the known action keys present)")

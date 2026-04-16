# src/merge_teammate_data.py  — FINAL CORRECTED v4
"""
CloudShield — Merge Teammate Data
===================================
Graph schema (confirmed via debug_graph.py):
  Node types : policy, service, resource, role, user
  Node attrs : type, name, arn, statement_count, conditions,
               risk_context {has_mfa,has_ip,has_time,has_org},
               attached_to, risk_label, identifier, is_wildcard
  Edge types : grants_access → d['actions'] (list), d['has_wildcard'],
                                d['conditions'], d['effect']
               acts_on       → v['arn'], v['is_wildcard']  (resource node)
  Structure  : bipartite-like (policy→service, policy→resource)
               → clustering/ego/modularity are structural zeros for policy nodes
               → replaced with pagerank-based proxies

Run: python src/merge_teammate_data.py
"""

import os, sys, pickle, warnings
from collections import Counter
import pandas as pd
import numpy as np
import networkx as nx
warnings.filterwarnings('ignore')

TEAMMATE_DATA = r"D:/iam-graph-xai/data"
OUR_DATA      = r"E:/iam-graph-xai/data"
FEAT_PKL      = r"E:/iam-graph-xai/models/feature_names_v2.pkl"
OUTPUT_CSV    = r"E:/iam-graph-xai/data/labeled_features_merged.csv"

# Full qualified dangerous actions (iam: prefix)
DANGEROUS_ACTIONS = {
    'iam:passrole',
    'iam:createpolicyversion',
    'iam:setdefaultpolicyversion',
    'iam:attachuserpolicy',
    'iam:attachrolepolicy',
    'iam:attachgrouppolicy',
    'iam:putuserinlinepolicy',
    'iam:putrolepolicy',
    'iam:putgrouppolicy',
    'iam:createaccesskey',
    'iam:updateloginprofile',
    'iam:createloginprofile',
    'iam:addusertogroup',
    'iam:updateassumerolepolicydocument',
    'iam:createuser',
    'iam:deleteuser',
    'iam:createrole',
    'iam:deleterole',
    'iam:createpolicy',
    'iam:deletepolicy',
}

# Short-form fallback (policies that omit service prefix)
DANGEROUS_SHORT = {a.split(':')[1] for a in DANGEROUS_ACTIONS}

# Tier-1: actions that are HIGH-risk by themselves
HIGH_ALONE = {
    'iam:passrole',
    'iam:createpolicyversion',
    'iam:setdefaultpolicyversion',
    'iam:attachuserpolicy',
    'iam:attachrolepolicy',
    'iam:attachgrouppolicy',
    'iam:putuserinlinepolicy',
    'iam:putrolepolicy',
    'iam:putgrouppolicy',
    'iam:createaccesskey',
    'iam:updateloginprofile',
    'iam:updateassumerolepolicydocument',
    'iam:addusertogroup',
}
HIGH_ALONE_SHORT = {a.split(':')[1] for a in HIGH_ALONE}

# Keywords that identify admin-equivalent nodes (checked against
# both the node ID string and the 'name' attribute)
ADMIN_KEYWORDS = [
    'admin', 'administrator', 'root', 'superuser',
    'fullaccess', 'full_access', 'poweruser', 'power_user',
    'securityaudit', 'privileged', 'unrestricted',
]


# ─────────────────────────────────────────────────────────────────────
# 1  LOAD
# ─────────────────────────────────────────────────────────────────────
def load_all():
    print("=" * 60)
    print("  Loading datasets")
    print("=" * 60)

    our_csv = pd.read_csv(f"{OUR_DATA}/labeled_features_v2.csv")
    tm_csv  = pd.read_csv(f"{TEAMMATE_DATA}/labeled_features.csv")

    before  = len(our_csv)
    our_csv = our_csv[our_csv['risk_label'] != -1].reset_index(drop=True)
    print(f"  Ours (-1 dropped {before - len(our_csv)} rows) : {our_csv.shape}")
    print(f"  Teammate                                       : {tm_csv.shape}")

    print(f"\n  Loading graph (~30 s)...")
    with open(f"{TEAMMATE_DATA}/iam_graph.pkl", 'rb') as f:
        G = pickle.load(f)
    print(f"  Graph : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Type  : {type(G).__name__}")

    with open(FEAT_PKL, 'rb') as f:
        feature_names = pickle.load(f)
    print(f"  Features required : {len(feature_names)}")

    return our_csv, tm_csv, G, feature_names


# ─────────────────────────────────────────────────────────────────────
# 2  BULK PRE-COMPUTATION  (all O(V+E), each runs once)
# ─────────────────────────────────────────────────────────────────────
def bulk_precompute(G, tm_csv):
    print("\n" + "=" * 60)
    print("  Bulk pre-computation")
    print("=" * 60)

    # ── Convert MultiDiGraph → simple graphs ──────────────────────
    print("  Converting MultiDiGraph → simple DiGraph + undirected...")
    G_simple = nx.DiGraph(G)
    G_und    = nx.Graph(G_simple)

    # ── Node maps ─────────────────────────────────────────────────
    print("  Building node maps...")
    policy_nodes = set()
    admin_nodes  = set()
    pid_to_node  = {}

    for n, d in G.nodes(data=True):
        ntype     = str(d.get('type', d.get('node_type', ''))).lower()
        name      = str(d.get('name', '')).lower()
        node_id   = str(n).lower()

        if ntype == 'policy':
            policy_nodes.add(n)
            pid_to_node[str(n)]   = n          # "policy:bf_AddUserToGroup"
            pid_to_node[name]     = n           # "bf_addusertogroup"
            bare = str(n).replace('policy:', '')
            pid_to_node[bare]     = n           # "bf_AddUserToGroup"

        # ── Admin detection: ALL node types, check node ID + name ──
        # IAM admin targets: AdministratorAccess policy, AdminRole,
        # root user, FullAccess policies, etc.
        if any(kw in node_id or kw in name for kw in ADMIN_KEYWORDS):
            admin_nodes.add(n)

    # Also map by policy_id column value
    if 'policy_id' in tm_csv.columns:
        for pid in tm_csv['policy_id'].astype(str):
            if pid not in pid_to_node:
                cand = f"policy:{pid}"
                if cand in pid_to_node:
                    pid_to_node[pid] = pid_to_node[cand]

    print(f"  Policy nodes : {len(policy_nodes)}")
    print(f"  Admin nodes  : {len(admin_nodes)}")

    # ── Clustering coefficient ─────────────────────────────────────
    # NOTE: policy nodes form a bipartite-like structure
    # (policy→service, policy→resource). Neighbors (services/resources)
    # are NOT connected to each other → clustering = 0 for ALL policy
    # nodes. This is a structural property, not a bug.
    # We still compute it and replace with pagerank proxy below.
    print("  Clustering coefficients (bulk, expect ~0 for policy nodes)...")
    cc = nx.clustering(G_und)

    # ── PageRank (primary importance metric for policy nodes) ──────
    print("  PageRank (bulk)...")
    try:
        pr = nx.pagerank(G_simple, alpha=0.85, max_iter=200, tol=1e-4)
    except Exception as e:
        print(f"  ⚠️  PageRank failed ({e}), using in-degree fallback")
        total = max(G_simple.number_of_edges(), 1)
        pr = {n: G_simple.in_degree(n) / total for n in G_simple.nodes()}

    # Pre-compute normalisers for structural proxies
    pr_vals   = [pr.get(n, 0.0) for n in policy_nodes]
    max_pr    = max(pr_vals)    if pr_vals    else 1.0

    # ── Degree maps ───────────────────────────────────────────────
    in_deg  = dict(G_simple.in_degree())
    out_deg = dict(G_simple.out_degree())
    und_deg = dict(G_und.degree())

    out_vals  = [out_deg.get(n, 0) for n in policy_nodes]
    max_out   = max(out_vals) if out_vals else 1

    # ── Reverse multi-source BFS for shortest_path_to_admin ───────
    print(f"  Reverse BFS from {len(admin_nodes)} admin nodes...")
    if admin_nodes:
        G_rev = G_simple.reverse()
        dist_to_admin = dict(
            nx.multi_source_dijkstra_path_length(
                G_rev, admin_nodes, cutoff=10
            )
        )
    else:
        # Fallback: use service nodes as proxy targets
        # (policies that reach more services are "closer to admin")
        service_nodes = {n for n, d in G.nodes(data=True)
                         if str(d.get('type', '')).lower() == 'service'}
        print(f"  ⚠️  No admin nodes found — using {len(service_nodes)} "
              f"service nodes as proxy")
        G_rev = G_simple.reverse()
        dist_to_admin = dict(
            nx.multi_source_dijkstra_path_length(
                G_rev, service_nodes, cutoff=5
            )
        )
    print(f"  Reachable nodes : {len(dist_to_admin)}")

    # ── Edge scan: actions / resources / conditions ────────────────
    print("  Extracting actions/resources/conditions (edge scan)...")

    policy_actions          = {n: set()  for n in policy_nodes}
    policy_resources        = {n: set()  for n in policy_nodes}
    policy_services         = {n: set()  for n in policy_nodes}
    policy_has_wildcard_act = {n: False  for n in policy_nodes}
    policy_has_wildcard_res = {n: False  for n in policy_nodes}
    policy_conditions       = {n: []     for n in policy_nodes}

    for u, v, d in G.edges(data=True):          # iterate original multigraph
        if u not in policy_nodes:
            continue

        etype = d.get('type', '')

        if etype == 'grants_access':
            acts = d.get('actions', [])
            if isinstance(acts, str):
                acts = [acts]
            for a in acts:
                a = str(a).lower().strip()
                if not a:
                    continue
                policy_actions[u].add(a)
                if ':' in a:
                    policy_services[u].add(a.split(':')[0])

            if d.get('has_wildcard', False):
                policy_has_wildcard_act[u] = True

            cond = d.get('conditions', {})
            if cond:
                policy_conditions[u].append(cond)

        elif etype == 'acts_on':
            vd  = G.nodes[v]
            arn = str(vd.get('arn', v))
            policy_resources[u].add(arn)
            if vd.get('is_wildcard', False) or arn == '*':
                policy_has_wildcard_res[u] = True

    # ── risk_context from policy node attributes ───────────────────
    print("  Extracting risk_context from policy nodes...")
    policy_risk_context = {}
    for n in policy_nodes:
        nd = G.nodes[n]
        rc = nd.get('risk_context', {})
        policy_risk_context[n] = {
            'has_mfa' : bool(rc.get('has_mfa',  False)),
            'has_ip'  : bool(rc.get('has_ip',   False)),
            'has_time': bool(rc.get('has_time', False)),
            'has_org' : bool(rc.get('has_org',  False)),
        }

    non_empty = sum(1 for v in policy_actions.values() if v)
    print(f"  Policies with actions     : {non_empty}/{len(policy_nodes)}")
    print(f"  Policies with wildcard act: {sum(policy_has_wildcard_act.values())}")
    print(f"  Policies with wildcard res: {sum(policy_has_wildcard_res.values())}")

    # ── Neighbour sets (for ego density proxy) ─────────────────────
    print("  Pre-computing neighbour sets...")
    nbr_sets = {n: set(G_und.neighbors(n)) for n in policy_nodes}

    return {
        'pid_to_node'            : pid_to_node,
        'policy_nodes'           : policy_nodes,
        'admin_nodes'            : admin_nodes,
        'cc'                     : cc,
        'pr'                     : pr,
        'max_pr'                 : max_pr,
        'in_deg'                 : in_deg,
        'out_deg'                : out_deg,
        'und_deg'                : und_deg,
        'max_out'                : max_out,
        'dist_to_admin'          : dist_to_admin,
        'policy_actions'         : policy_actions,
        'policy_res'             : policy_resources,
        'policy_svc'             : policy_services,
        'policy_has_wildcard_act': policy_has_wildcard_act,
        'policy_has_wildcard_res': policy_has_wildcard_res,
        'policy_conditions'      : policy_conditions,
        'policy_risk_context'    : policy_risk_context,
        'nbr_sets'               : nbr_sets,
        'G_und'                  : G_und,
    }


# ─────────────────────────────────────────────────────────────────────
# 3  PER-POLICY FEATURE COMPUTATION  (pure dict lookups)
# ─────────────────────────────────────────────────────────────────────
def compute_missing_features(tm_csv, pre, feature_names):
    print("\n" + "=" * 60)
    print("  Per-policy feature computation")
    print("=" * 60)

    pid_col  = 'policy_id' if 'policy_id' in tm_csv.columns else None
    tm_pids  = (tm_csv[pid_col].astype(str).tolist()
                if pid_col else [str(i) for i in tm_csv.index])

    rows         = []
    unmatched_ct = 0

    for idx, (pid, row) in enumerate(zip(tm_pids, tm_csv.itertuples())):

        # ── Node lookup ───────────────────────────────────────────
        node = (pre['pid_to_node'].get(pid)
             or pre['pid_to_node'].get(f"policy:{pid}")
             or pre['pid_to_node'].get(pid.replace('policy:', '').lower()))

        if node is None:
            unmatched_ct += 1

        actions    = pre['policy_actions'].get(node, set())          if node else set()
        res        = pre['policy_res'].get(node, set())              if node else set()
        svcs       = pre['policy_svc'].get(node, set())              if node else set()
        rc         = pre['policy_risk_context'].get(node, {})        if node else {}
        has_wc_act = pre['policy_has_wildcard_act'].get(node, False) if node else False
        has_wc_res = pre['policy_has_wildcard_res'].get(node, False) if node else False

        action_str = ' '.join(actions)

        def in_act(keyword):
            """True if keyword (full or short form) present in actions."""
            short = keyword.split(':')[-1]
            return keyword in action_str or short in action_str

        c = {'policy_id': pid}

        # ── STRUCTURAL ────────────────────────────────────────────

        # clustering_coefficient: bipartite graph → structural zero for all
        # policy nodes. Use normalised PageRank as proxy (captures node
        # importance in the permission graph — correlates with real centrality).
        raw_cc = float(pre['cc'].get(node, 0.0)) if node else 0.0
        if raw_cc > 0:
            c['clustering_coefficient'] = raw_cc
        else:
            # PageRank proxy: normalised to [0, 1]
            c['clustering_coefficient'] = (
                float(pre['pr'].get(node, 0.0)) / max(pre['max_pr'], 1e-9)
                if node else 0.0
            )

        # ego_network_density: neighbors not connected → structural zero.
        # Proxy: normalised out-degree (how many services/resources reached)
        raw_ego = 0.0
        if node:
            nbrs  = pre['nbr_sets'].get(node, set())
            k     = len(nbrs)
            inner = sum(1 for nb in nbrs if pre['nbr_sets'].get(nb, set()) & nbrs)
            raw_ego = inner / (k * (k + 1)) if k > 1 else 0.0
        if raw_ego > 0:
            c['ego_network_density'] = raw_ego
        else:
            od = pre['out_deg'].get(node, 0) if node else 0
            c['ego_network_density'] = od / max(pre['max_out'], 1)

        # subgraph_modularity: same structural issue.
        # Proxy: service concentration (1 - top_service_fraction).
        # Higher = actions spread across more services = broader blast radius.
        if actions and svcs:
            svc_counts  = Counter(a.split(':')[0] for a in actions if ':' in a)
            top_frac    = svc_counts.most_common(1)[0][1] / len(actions)
            c['subgraph_modularity'] = float(1.0 - top_frac)
        else:
            c['subgraph_modularity'] = 0.0

        c['shortest_path_to_admin'] = (
            float(pre['dist_to_admin'].get(node, 10.0)) if node else 10.0
        )

        # cross_account_edge_count: ARNs with non-empty account field
        # arn:partition:service:region:ACCOUNT:resource
        c['cross_account_edge_count'] = float(
            sum(1 for r in res
                if r.startswith('arn:') and len(r.split(':')) >= 5
                and r.split(':')[4] not in ('', '*'))
        )

        # ── SEMANTIC ──────────────────────────────────────────────

        c['action_diversity'] = float(len(svcs))

        if res:
            specific = sum(
                1 for r in res
                if r != '*' and not r.endswith('/*') and '/*' not in r
            )
            c['resource_arn_specificity'] = specific / len(res)
        else:
            c['resource_arn_specificity'] = 0.5

        danger_ct = sum(1 for kw in DANGEROUS_ACTIONS if in_act(kw))
        c['permission_overlap_score'] = min(danger_ct / 10.0, 1.0)
        c['dangerous_action_count']   = float(danger_ct)

        chains = sum(
            1 for a in actions
            if any(kw in a for kw in ['passrole', 'invoke', 'assume', 'trigger'])
            and len(svcs) > 1
        )
        c['cross_service_permission_chains'] = float(min(chains, 5))

        # ── ESCALATION ────────────────────────────────────────────

        c['passrole_chain_exists']      = float(in_act('iam:passrole'))
        c['createpolicyversion_exists'] = float(
            in_act('iam:createpolicyversion') or
            in_act('iam:setdefaultpolicyversion')
        )
        c['attachuserpolicy_exists']    = float(
            in_act('iam:attachuserpolicy') or
            in_act('iam:attachrolepolicy') or
            in_act('iam:attachgrouppolicy')
        )
        c['iam_write_permission_count'] = float(danger_ct)

        c['privilege_escalation_risk_score'] = min(
            c['passrole_chain_exists']      * 0.40 +
            c['createpolicyversion_exists'] * 0.35 +
            c['attachuserpolicy_exists']    * 0.30 +
            min(danger_ct * 0.05, 0.30),
            1.0
        )

        # ── CONDITION ─────────────────────────────────────────────

        has_mfa  = float(rc.get('has_mfa',  False))
        has_ip   = float(rc.get('has_ip',   False))
        has_time = float(rc.get('has_time', False))
        has_org  = float(rc.get('has_org',  False))

        c['has_mfa_condition']          = has_mfa
        c['condition_protection_score'] = (
            has_mfa * 0.4 + has_ip * 0.3 + has_org * 0.2 + has_time * 0.1
        )
        c['is_bounded'] = float(
            c['resource_arn_specificity'] >= 0.9 and not has_wc_res
        )

        # ── WILDCARD ──────────────────────────────────────────────

        c['has_wildcard_action']   = float(has_wc_act)
        c['has_wildcard_resource'] = float(has_wc_res)

        # ── COMPLIANCE ────────────────────────────────────────────

        c['unused_permission_ratio'] = max(
            0.0,
            1.0 - c['resource_arn_specificity'] - c['condition_protection_score']
        )
        vcount  = int(c['passrole_chain_exists'])
        vcount += int(c['createpolicyversion_exists'])
        vcount += int(has_mfa == 0)
        vcount += int(has_ip == 0 and danger_ct >= 2)
        vcount += int(has_wc_act)
        vcount += int(has_wc_res)
        c['compliance_violation_count'] = float(vcount)

        # ── ROLLBACK ──────────────────────────────────────────────

        c['policy_version_count'] = float(
            getattr(row, 'policy_version_count', 1) or 1
        )
        c['max_historical_risk']  = float(
            getattr(row, 'risk_label', 0) or 0
        ) / 2.0
        c['rollback_risk_score']  = (
            c['privilege_escalation_risk_score'] * 0.8 +
            c['createpolicyversion_exists']      * 0.2
        )

        rows.append(c)
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(tm_pids)}")

    print(f"  Done — {len(rows)} policies computed")
    print(f"  Graph-unmatched : {unmatched_ct}")

    df = pd.DataFrame(rows)

    zv = [f for f in feature_names if f in df.columns and df[f].nunique() <= 1]
    if zv:
        print(f"  ⚠️  Still zero-variance in computed set : {zv}")
    else:
        print(f"  ✅  No zero-variance in computed feature set")

    return df


# ─────────────────────────────────────────────────────────────────────
# 4  BUILD TEAMMATE DATAFRAME + LABEL CORRECTION
# ─────────────────────────────────────────────────────────────────────
def build_teammate_df(tm_csv, computed_df, feature_names):
    print("\n" + "=" * 60)
    print("  Building full teammate dataframe")
    print("=" * 60)

    overlap   = [c for c in feature_names if c in tm_csv.columns]
    meta_cols = ['policy_id', 'risk_label']
    if 'source' in tm_csv.columns:
        meta_cols.append('source')

    print(f"  Overlapping features kept : {len(overlap)}")

    base = tm_csv[meta_cols + overlap].copy()
    base['policy_id'] = base['policy_id'].astype(str)
    computed_df = computed_df.copy()
    computed_df['policy_id'] = computed_df['policy_id'].astype(str)

    # Drop cols that computed_df re-supplies (avoid _x/_y suffixes)
    overlap_in_computed = [c for c in overlap if c in computed_df.columns]
    base = base.drop(columns=overlap_in_computed, errors='ignore')

    full = base.merge(computed_df, on='policy_id', how='left')

    for f in feature_names:
        if f not in full.columns:
            full[f] = 0.0
        full[f] = full[f].fillna(0.0)

    print(f"  Shape before label fix : {full.shape}")
    print(f"\n  Labels BEFORE correction : "
          f"{full['risk_label'].value_counts().sort_index().to_dict()}")

    # ── Label correction ──────────────────────────────────────────
    # Teammate labeler marked 71% HIGH — most were wildcard-resource
    # policies with only 1 non-critical IAM action.
    #
    # Tier logic (matches AWS privilege-escalation taxonomy):
    #   TIER 1 — any single action below = HIGH regardless of resource
    #   TIER 2 — 3+ dangerous actions = HIGH
    #   TIER 3 — 2 dangerous + wildcard resource = HIGH
    #   TIER 4 — 2 dangerous actions = MEDIUM
    #   TIER 5 — 1 dangerous + wildcard resource = MEDIUM
    #   ELSE   → LOW
    def correct_label(row):
        if row['risk_label'] != 2:
            return int(row['risk_label'])

        # TIER 1: single-action escalation paths → always HIGH
        if row.get('passrole_chain_exists',      0) > 0: return 2
        if row.get('createpolicyversion_exists', 0) > 0: return 2
        if row.get('attachuserpolicy_exists',    0) > 0: return 2
        if row.get('has_wildcard_action',        0) > 0: return 2

        iaw = row.get('iam_write_permission_count', 0)
        wres = row.get('has_wildcard_resource', 0) > 0

        # TIER 2: 3+ dangerous actions
        if iaw >= 3: return 2

        # TIER 3: 2 dangerous + wildcard resource
        if iaw >= 2 and wres: return 2

        # TIER 4: 2 dangerous actions alone
        if iaw >= 2: return 1

        # TIER 5: 1 dangerous + wildcard resource
        if iaw >= 1 and wres: return 1

        # No strong signals
        return 0

    full['risk_label'] = full.apply(correct_label, axis=1)

    print(f"  Labels AFTER  correction :")
    dist = full['risk_label'].value_counts().sort_index()
    for lbl, cnt in dist.items():
        print(f"    {lbl} : {cnt:>5}  ({cnt/len(full)*100:.1f}%)")

    return full


# ─────────────────────────────────────────────────────────────────────
# 5  MERGE
# ─────────────────────────────────────────────────────────────────────
def merge_datasets(our_csv, tm_full, feature_names):
    print("\n" + "=" * 60)
    print("  Merging datasets")
    print("=" * 60)

    if 'source' not in our_csv.columns:
        our_csv = our_csv.copy()
        our_csv['source'] = 'ours'
    if 'source' not in tm_full.columns:
        tm_full = tm_full.copy()
        tm_full['source'] = 'teammate'

    keep    = ['policy_id', 'risk_label', 'source'] + feature_names
    our_aln = our_csv.reindex(columns=keep, fill_value=0.0)
    tm_aln  = tm_full.reindex(columns=keep, fill_value=0.0)

    merged = pd.concat([our_aln, tm_aln], ignore_index=True)
    print(f"  Rows before dedup : {len(merged)}")

    merged = merged.drop_duplicates(subset='policy_id', keep='first')
    print(f"  Rows after  dedup : {len(merged)}")

    merged = merged[merged['risk_label'].isin([0, 1, 2])]
    merged['risk_label'] = merged['risk_label'].astype(int)

    print(f"  Final rows : {len(merged)}")
    dist = merged['risk_label'].value_counts().sort_index()
    for lbl, cnt in dist.items():
        print(f"    {lbl} : {cnt:>5}  ({cnt/len(merged)*100:.1f}%)")

    return merged


# ─────────────────────────────────────────────────────────────────────
# 6  VALIDATION
# ─────────────────────────────────────────────────────────────────────
def validate(merged, feature_names):
    print("\n" + "=" * 60)
    print("  Validation")
    print("=" * 60)

    zv   = [f for f in feature_names
            if f in merged.columns and merged[f].nunique() <= 1]
    miss = [f for f in feature_names if f not in merged.columns]
    nans = [f for f in feature_names
            if f in merged.columns and merged[f].isnull().any()]
    dups = int(merged['policy_id'].duplicated().sum())
    high = (merged['risk_label'] == 2).mean() * 100
    med  = (merged['risk_label'] == 1).mean() * 100

    checks = [
        ("Row count ≥ 5000",
         len(merged) >= 5000,
         str(len(merged))),
        ("All 40 features present",
         len(miss) == 0,
         f"missing={miss}"),
        ("No zero-variance features",
         len(zv) == 0,
         f"zero_var={zv}"),
        ("HIGH class 10–70%",
         10.0 <= high <= 70.0,
         f"{high:.1f}%"),
        ("MEDIUM class ≥ 5%",
         med >= 5.0,
         f"{med:.1f}%"),
        ("No NaN in features",
         len(nans) == 0,
         f"nan_cols={nans}"),
        ("No duplicate policy_ids",
         dups == 0,
         f"{dups} duplicates"),
    ]

    all_ok = True
    for name, ok, detail in checks:
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {name:<40}  {detail}")
        if not ok:
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  CloudShield — Merge Teammate Data (v4)")
    print("=" * 60)

    our_csv, tm_csv, G, feature_names = load_all()

    pre         = bulk_precompute(G, tm_csv)
    computed_df = compute_missing_features(tm_csv, pre, feature_names)
    tm_full     = build_teammate_df(tm_csv, computed_df, feature_names)
    merged      = merge_datasets(our_csv, tm_full, feature_names)

    ok = validate(merged, feature_names)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved : {OUTPUT_CSV}")
    print(f"  Shape : {merged.shape}")

    if ok:
        print("\n  ✅  All checks passed — run: python run_all.py --from rf")
    else:
        print("\n  ⚠️  Fix validation errors above before retraining")
        sys.exit(1)

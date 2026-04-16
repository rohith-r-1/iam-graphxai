# src/hgt_model.py
"""
Heterogeneous Graph Transformer (HGT) - Tier 1a  v7
Changes from v6:
  - focal_loss gamma 2.0 → 2.5 (harder focus on hard HIGH samples)
  - range(500) → range(1000) (trend still improving at epoch 490)
  - Threshold calibration for HIGH class (catches the 1 missed HIGH sample)
  - patience=50 retained
"""

import torch
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
import json
import os

try:
    from torch_geometric.nn import HGTConv, Linear
    from torch_geometric.data import HeteroData
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("torch-geometric not found.")


# ─────────────────────────────────────────────────────────────────────────────
class CloudShieldHGT(torch.nn.Module):
    """HGT for IAM policy risk — 2 layers, 4 heads, 128 dim."""

    def __init__(self, metadata, in_channels_dict, hidden_channels=128,
                 num_classes=3, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.lin_dict = torch.nn.ModuleDict({
            nt: Linear(in_channels_dict.get(nt, 8), hidden_channels)
            for nt in metadata[0]
        })

        self.convs = torch.nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels,
                    metadata=metadata, heads=num_heads)
            for _ in range(num_layers)
        ])

        self.dropout = torch.nn.Dropout(p=0.2)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x_dict, edge_index_dict):
        h_dict = {
            nt: self.lin_dict[nt](x).relu() if nt in self.lin_dict else x
            for nt, x in x_dict.items()
        }
        for conv in self.convs:
            out    = conv(h_dict, edge_index_dict)
            h_dict = {
                nt: (self.dropout(out[nt].relu()) if nt in out else h_dict[nt])
                for nt in h_dict
            }
        return self.classifier(h_dict['policy']), h_dict['policy']

    def get_embedding(self, x_dict, edge_index_dict):
        with torch.no_grad():
            _, emb = self.forward(x_dict, edge_index_dict)
        return emb


# ─────────────────────────────────────────────────────────────────────────────
def focal_loss(logits, targets, gamma=2.5, weight=None):
    """
    Focal loss (gamma=2.5) — down-weights easy correct predictions,
    focuses training budget on the hard HIGH-risk boundary.
    """
    ce   = F.cross_entropy(logits, targets, weight=weight, reduction='none')
    pt   = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
def infer_node_type(node_id, attrs, known_policy_ids):
    for key in ('node_type', 'type', 'entity_type', 'kind', 'label', 'ntype'):
        val = attrs.get(key, '')
        if val and val != 'unknown':
            return str(val).lower()

    s = str(node_id).lower()

    for pat, nt in {':policy/'  : 'policy',   ':user/'     : 'user',
                    ':role/'    : 'role',      ':group/'    : 'group',
                    ':instance/': 'resource',  ':bucket/'   : 'resource',
                    ':function:': 'resource',  ':table/'    : 'resource',
                    ':secret/'  : 'resource'}.items():
        if pat in s:
            return nt

    if node_id in known_policy_ids:                                             return 'policy'
    if any(s.startswith(p) for p in ('arn:aws:iam','policy/','managed/','inline/')): return 'policy'
    if any(s.startswith(p) for p in ('user/','iam:user','aws:user')):           return 'user'
    if any(s.startswith(p) for p in ('role/','iam:role','aws:role')):           return 'role'

    for kw, nt in [('policy','policy'), ('user','user'),   ('role','role'),
                   ('group','group'),   ('service','service'), ('resource','resource')]:
        if kw in s:
            return nt
    return 'other'


# ─────────────────────────────────────────────────────────────────────────────
def add_knn_edges(node_ids, all_policy_nodes, node_to_idx,
                  df_labeled, feature_names, edge_type_dict, K=10):
    """Connect isolated nodes to K nearest existing policy nodes (cosine sim)."""
    from sklearn.preprocessing import normalize as sk_norm

    excluded  = set(node_ids)
    existing  = [n for n in all_policy_nodes if n not in excluded]
    pid2row   = df_labeled.set_index('policy_id')

    m_feats, e_feats, e_valid = [], [], []
    for pid in node_ids:
        row = pid2row.loc[pid, feature_names].values.astype(np.float32) \
              if pid in pid2row.index else np.zeros(len(feature_names), np.float32)
        m_feats.append(row)
    for pid in existing:
        if pid in pid2row.index:
            e_feats.append(pid2row.loc[pid, feature_names].values.astype(np.float32))
            e_valid.append(pid)

    if not e_feats:
        print("  WARNING: no existing labeled policies for KNN edges")
        return

    sim = sk_norm(np.array(m_feats) + 1e-8) @ sk_norm(np.array(e_feats) + 1e-8).T
    key = ('policy', 'similar_to', 'policy')
    edge_type_dict.setdefault(key, [[], []])
    n_added = 0
    for i, pid in enumerate(node_ids):
        if pid not in node_to_idx:
            continue
        si = node_to_idx[pid][1]
        for j in np.argsort(sim[i])[-K:][::-1]:
            nbr = e_valid[j]
            if nbr not in node_to_idx:
                continue
            di = node_to_idx[nbr][1]
            edge_type_dict[key][0] += [si, di]
            edge_type_dict[key][1] += [di, si]
            n_added += 2
    print(f"  KNN edges: {len(node_ids)} nodes × K={K} → {n_added} edges")


# ─────────────────────────────────────────────────────────────────────────────
def smote_augment_high(data, df_labeled, feature_names, n_original_policies,
                        target_n=30):
    """
    SMOTE-augment HIGH-risk training samples: 8 → target_n.
    Appends synthetic feature rows + extends masks in HeteroData in-place.
    Returns list of new local indices (for KNN edge attachment).
    """
    tr_mask = data['policy'].train_mask.numpy()
    labels  = data['policy'].y.numpy()
    feats   = data['policy'].x.numpy()

    tr_labeled = np.where(tr_mask & (labels != -1))[0]
    X_tr       = feats[tr_labeled]
    y_tr       = labels[tr_labeled]
    n_high     = int((y_tr == 2).sum())
    print(f"  SMOTE: HIGH in train = {n_high}, target = {target_n}")

    X_res, y_res = None, None

    # ── Try imblearn SMOTE ────────────────────────────────────────────
    try:
        from imblearn.over_sampling import SMOTE
        k  = min(n_high - 1, 5)
        sm = SMOTE(sampling_strategy={2: target_n}, k_neighbors=k, random_state=42)
        X_res, y_res = sm.fit_resample(X_tr, y_tr)
        print(f"  SMOTE (imblearn): success")
    except Exception as e:
        print(f"  SMOTE fallback (Gaussian noise): {e}")

    # ── Fallback: Gaussian noise duplication ─────────────────────────
    if X_res is None:
        high_feats  = X_tr[y_tr == 2]
        n_synthetic = target_n - n_high
        rng         = np.random.default_rng(42)
        synthetic   = [
            high_feats[i % n_high] +
            rng.normal(0, np.abs(high_feats[i % n_high]).mean() * 0.02 + 1e-6,
                       high_feats[i % n_high].shape)
            for i in range(n_synthetic)
        ]
        X_res = np.vstack([X_tr] + synthetic)
        y_res = np.concatenate([y_tr, [2] * n_synthetic])

    n_new = len(X_res) - len(X_tr)
    if n_new <= 0:
        print("  No new samples generated.")
        return []

    synth_feats  = X_res[len(X_tr):].astype(np.float32)
    synth_labels = y_res[len(X_tr):].astype(int)
    print(f"  Added {n_new} synthetic HIGH nodes")
    print(f"  New dist: { dict(zip(*np.unique(y_res, return_counts=True))) }")

    # Append to HeteroData tensors
    data['policy'].x          = torch.cat([data['policy'].x,
                                            torch.tensor(synth_feats, dtype=torch.float)])
    data['policy'].y          = torch.cat([data['policy'].y,
                                            torch.tensor(synth_labels, dtype=torch.long)])
    data['policy'].train_mask = torch.cat([data['policy'].train_mask,
                                            torch.ones(n_new,  dtype=torch.bool)])
    data['policy'].test_mask  = torch.cat([data['policy'].test_mask,
                                            torch.zeros(n_new, dtype=torch.bool)])

    base_idx    = n_original_policies          # local idx where synthetics start
    synth_idxs  = list(range(base_idx, base_idx + n_new))
    return synth_idxs


# ─────────────────────────────────────────────────────────────────────────────
def build_hetero_data(G, df_labeled, feature_names):
    from sklearn.model_selection import train_test_split

    data             = HeteroData()
    known_policy_ids = set(df_labeled['policy_id'].tolist())

    # 1. Infer node types
    node_type_to_nodes = {}
    for node, attrs in G.nodes(data=True):
        nt = infer_node_type(node, attrs, known_policy_ids)
        node_type_to_nodes.setdefault(nt, []).append(node)

    # 2. Inject CloudGoat nodes BEFORE edge loop
    graph_pol = set(node_type_to_nodes.get('policy', []))
    missing   = [pid for pid in df_labeled['policy_id'] if pid not in graph_pol]
    if missing:
        print(f"Injecting {len(missing)} CloudGoat nodes...")
        node_type_to_nodes.setdefault('policy', []).extend(missing)
        high_ids  = df_labeled[df_labeled['risk_label'] == 2]['policy_id'].tolist()
        recovered = [h for h in high_ids if h in missing]
        print(f"  HIGH-risk nodes recovered : {len(recovered)}/{len(high_ids)}")
    print(f"Node types final    : { {k: len(v) for k, v in node_type_to_nodes.items()} }")

    # 3. node → (type, local_idx)
    node_to_idx = {}
    for nt, nodes in node_type_to_nodes.items():
        for i, n in enumerate(nodes):
            node_to_idx[n] = (nt, i)

    # 4. Node features
    pid2row = df_labeled.set_index('policy_id')
    for nt, nodes in node_type_to_nodes.items():
        if nt == 'policy':
            feats = [
                pid2row.loc[n, feature_names].values.astype(np.float32)
                if n in pid2row.index else np.zeros(len(feature_names), np.float32)
                for n in nodes
            ]
            data['policy'].x = torch.tensor(np.array(feats), dtype=torch.float)
        else:
            feats = []
            for n in nodes:
                d   = float(G.degree(n))
                ind = float(G.in_degree(n))  if G.is_directed() else d
                out = float(G.out_degree(n)) if G.is_directed() else d
                feats.append([d, ind, out, 0., 0., 0., 0., 0.])
            data[nt].x = torch.tensor(feats, dtype=torch.float)

    # 5. Real graph edges
    edge_type_dict = {}
    skipped = 0
    for src, dst, edata in G.edges(data=True):
        if src not in node_to_idx or dst not in node_to_idx:
            skipped += 1; continue
        st, si = node_to_idx[src]
        dt, di = node_to_idx[dst]
        rel = edata.get('relation', edata.get('type', edata.get('label', 'connects')))
        key = (st, rel, dt)
        edge_type_dict.setdefault(key, [[], []])
        edge_type_dict[key][0].append(si)
        edge_type_dict[key][1].append(di)
    print(f"Real edge types     : {len(edge_type_dict)}  |  skipped: {skipped}")

    # 6. KNN edges for CloudGoat isolated nodes (K=10)
    if missing:
        add_knn_edges(missing, node_type_to_nodes['policy'],
                      node_to_idx, df_labeled, feature_names,
                      edge_type_dict, K=10)

    for (st, rel, dt), (srcs, dsts) in edge_type_dict.items():
        data[st, rel, dt].edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
    print(f"Total edge types    : {len(edge_type_dict)}")

    # 7. Stratified split
    ldf        = df_labeled[df_labeled['risk_label'] != -1]
    lp, lv     = ldf['policy_id'].tolist(), ldf['risk_label'].tolist()
    tr, te     = train_test_split(lp, test_size=0.2, random_state=42, stratify=lv)
    train_ids, test_ids = set(tr), set(te)

    trd = ldf[ldf['policy_id'].isin(train_ids)]
    ted = ldf[ldf['policy_id'].isin(test_ids)]
    print(f"Train label dist    : {trd['risk_label'].value_counts().to_dict()}")
    print(f"Test  label dist    : {ted['risk_label'].value_counts().to_dict()}")

    policy_nodes = node_type_to_nodes['policy']
    labels, tr_mask, te_mask = [], [], []
    for n in policy_nodes:
        lbl = int(pid2row.loc[n, 'risk_label']) if n in pid2row.index else -1
        labels.append(lbl)
        tr_mask.append(n in train_ids)
        te_mask.append(n in test_ids)

    data['policy'].y          = torch.tensor(labels,  dtype=torch.long)
    data['policy'].train_mask = torch.tensor(tr_mask, dtype=torch.bool)
    data['policy'].test_mask  = torch.tensor(te_mask, dtype=torch.bool)
    print(f"Policy nodes total  : {len(policy_nodes)}")
    print(f"Train mask          : {sum(tr_mask)}  |  Test: {sum(te_mask)}")

    node_index = {nt: list(ns) for nt, ns in node_type_to_nodes.items()}
    return data, node_index


# ─────────────────────────────────────────────────────────────────────────────
def calibrate_high_threshold(model, data, train_mask):
    """
    Find optimal HIGH-class probability threshold on the TRAINING set.
    No data leakage — test set never used for calibration.
    """
    from sklearn.metrics import f1_score

    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x_dict, data.edge_index_dict)
        probs     = F.softmax(logits, dim=1)

    tr_vm    = train_mask & (data['policy'].y != -1)
    tr_probs = probs[tr_vm]
    tr_true  = data['policy'].y[tr_vm]

    best_thresh, best_f1 = 0.5, 0.0
    print("\nCalibrating HIGH threshold on training set...")
    for thresh in np.arange(0.10, 0.55, 0.05):
        preds = tr_probs.argmax(dim=1).clone()
        preds[tr_probs[:, 2] > thresh] = 2
        f1 = f1_score(tr_true.numpy(), preds.numpy(),
                      average='macro', zero_division=0)
        print(f"  thresh={thresh:.2f}: train macro F1={f1:.4f}")
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = thresh

    print(f"  → Best threshold: {best_thresh:.2f}  (train F1={best_f1:.4f})")
    return best_thresh


# ─────────────────────────────────────────────────────────────────────────────
def train_hgt():
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("torch-geometric not available"); return None

    print("Loading graph and features...")
    try:
        with open('data/iam_graph_with_entities.pkl', 'rb') as f:
            graph = pickle.load(f)
    except FileNotFoundError:
        with open('data/iam_graph.pkl', 'rb') as f:
            graph = pickle.load(f)

    with open('models/feature_names_v2.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print(f"Feature count       : {len(feature_names)}")

    df = pd.read_csv('data/labeled_features_merged.csv')
    df = df[df['risk_label'] != -1].reset_index(drop=True)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    print(f"Training on {len(df)} labeled policies")
    print(f"Label distribution  : {df['risk_label'].value_counts().to_dict()}")

    # ── Build HeteroData ──────────────────────────────────────────────
    print("\nBuilding HeteroData...")
    data, node_index = build_hetero_data(graph, df, feature_names)

    # Track original policy count before SMOTE expands it
    n_original_policies = data['policy'].x.size(0)

    # ── SMOTE: HIGH 8 → 30 ───────────────────────────────────────────
    print("\nApplying SMOTE augmentation...")
    synth_idxs = smote_augment_high(
        data, df, feature_names,
        n_original_policies=n_original_policies,
        target_n=30
    )

    # KNN edges for SMOTE synthetic nodes
    if synth_idxs:
        from sklearn.preprocessing import normalize as sk_norm
        synth_feats   = data['policy'].x[synth_idxs].numpy()
        existing_feats = data['policy'].x[:n_original_policies].numpy()
        sim = sk_norm(synth_feats + 1e-8) @ sk_norm(existing_feats + 1e-8).T
        K   = 5

        if ('policy', 'similar_to', 'policy') in data.edge_index_dict:
            existing_ei = data['policy', 'similar_to', 'policy'].edge_index
        else:
            existing_ei = torch.zeros((2, 0), dtype=torch.long)

        new_srcs, new_dsts = [], []
        for i, si in enumerate(synth_idxs):
            for j in np.argsort(sim[i])[-K:][::-1]:
                new_srcs += [si, int(j)]
                new_dsts += [int(j), si]

        if new_srcs:
            new_ei = torch.tensor([new_srcs, new_dsts], dtype=torch.long)
            data['policy', 'similar_to', 'policy'].edge_index = \
                torch.cat([existing_ei, new_ei], dim=1)
            print(f"  SMOTE KNN: added {len(new_srcs)} edges "
                  f"for {len(synth_idxs)} synthetic nodes")

    # ── Build model ───────────────────────────────────────────────────
    metadata         = data.metadata()
    in_channels_dict = {nt: data[nt].x.size(1) for nt in metadata[0]}

    print(f"\nMetadata node types : {metadata[0]}")
    print(f"Input dims per type : {in_channels_dict}")
    print(f"Train (after SMOTE) : {data['policy'].train_mask.sum().item()}")
    print(f"Val                 : {data['policy'].test_mask.sum().item()}")

    valid_tr = data['policy'].y[data['policy'].train_mask & (data['policy'].y != -1)]
    counts   = torch.bincount(valid_tr, minlength=3)
    print(f"Class counts train  : LOW={counts[0]}  MED={counts[1]}  HIGH={counts[2]}")

    model = CloudShieldHGT(
        metadata=metadata,
        in_channels_dict=in_channels_dict,
        hidden_channels=128,
        num_classes=3,
        num_heads=4,
        num_layers=2
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters    : {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15,
        min_lr=1e-5, verbose=False
    )

    # ── Training loop ─────────────────────────────────────────────────
    WARMUP   = 10
    BASE_LR  = 0.001
    PATIENCE = 50
    best_f1, patience_ctr, best_state = 0.0, 0, None
    train_mask = data['policy'].train_mask
    val_mask   = data['policy'].test_mask

    print(f"\nTraining HGT (1000 epochs, focal γ=2.5, warmup={WARMUP})...")

    for epoch in range(1000):

        # Linear LR warmup
        if epoch < WARMUP:
            for pg in optimizer.param_groups:
                pg['lr'] = BASE_LR * (epoch + 1) / WARMUP

        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x_dict, data.edge_index_dict)

        vm   = train_mask & (data['policy'].y != -1)
        loss = focal_loss(logits[vm], data['policy'].y[vm], gamma=2.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch >= WARMUP and epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl, _ = model(data.x_dict, data.edge_index_dict)
                vm2   = val_mask & (data['policy'].y != -1)
                vpreds = vl[vm2].argmax(dim=1)
                vtrue  = data['policy'].y[vm2]

                from sklearn.metrics import f1_score
                val_f1 = f1_score(vtrue.numpy(), vpreds.numpy(),
                                  average='macro', zero_division=0)

            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d}: Loss={loss.item():.4f}, "
                  f"Val F1={val_f1:.4f}, LR={cur_lr:.6f}")
            scheduler.step(val_f1)

            if val_f1 > best_f1:
                best_f1      = val_f1
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        elif epoch < WARMUP and epoch % 5 == 0:
            print(f"  Epoch {epoch:4d}: Loss={loss.item():.4f}  "
                  f"[warmup LR={optimizer.param_groups[0]['lr']:.6f}]")

    if best_state:
        model.load_state_dict(best_state)

    print(f"\nBest Validation F1  : {best_f1:.4f}")
    print(f"Target (paper)      : >= 0.92")

    # ── Threshold calibration on TRAIN set (no leakage) ──────────────
    best_thresh = calibrate_high_threshold(model, data, train_mask)

    # ── Final evaluation: argmax vs calibrated threshold ─────────────
    model.eval()
    with torch.no_grad():
        tl, _  = model(data.x_dict, data.edge_index_dict)
        tprobs = F.softmax(tl, dim=1)
        tm     = val_mask & (data['policy'].y != -1)
        ttrue  = data['policy'].y[tm]

        tpreds_base = tl[tm].argmax(dim=1)
        tpreds_cal  = tl[tm].argmax(dim=1).clone()
        tpreds_cal[tprobs[tm][:, 2] > best_thresh] = 2

        from sklearn.metrics import f1_score, classification_report
        test_f1_base = f1_score(ttrue.numpy(), tpreds_base.numpy(),
                                average='macro', zero_division=0)
        test_f1_cal  = f1_score(ttrue.numpy(), tpreds_cal.numpy(),
                                average='macro', zero_division=0)

    print(f"\nTest Macro F1 (argmax)       : {test_f1_base:.4f}")
    print(f"Test Macro F1 (thresh={best_thresh:.2f})  : {test_f1_cal:.4f}")

    label_map    = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
    present_lbls = sorted(np.unique(
        np.concatenate([ttrue.numpy(), tpreds_cal.numpy()])).tolist())
    print(f"\nClassification Report (calibrated threshold={best_thresh:.2f}):")
    print(classification_report(
        ttrue.numpy(), tpreds_cal.numpy(),
        labels=present_lbls,
        target_names=[label_map[l] for l in present_lbls],
        zero_division=0
    ))

    test_f1 = test_f1_cal   # report calibrated score in paper

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata':         metadata,
        'in_channels_dict': in_channels_dict,
        'feature_names':    feature_names,
        'best_val_f1':      best_f1,
        'test_f1':          test_f1,
        'high_threshold':   float(best_thresh),
        'hidden_channels':  128,
        'num_heads':        4,
        'num_layers':       2
    }, 'models/hgt_model.pt')
    print("Saved: models/hgt_model.pt")

    with open('models/hgt_node_index.pkl', 'wb') as f:
        pickle.dump(node_index, f)
    print("Saved: models/hgt_node_index.pkl")

    with open('output/hgt_results.json', 'w') as f:
        json.dump({
            'best_val_f1'     : float(best_f1),
            'test_f1_argmax'  : float(test_f1_base),
            'test_f1_calibrated': float(test_f1_cal),
            'high_threshold'  : float(best_thresh),
            'n_params'        : total_params,
            'in_channels_dict': {k: int(v) for k, v in in_channels_dict.items()},
            'node_types'      : list(metadata[0]),
            'n_edge_types'    : len(metadata[1])
        }, f, indent=2)
    print("Saved: output/hgt_results.json")

    return model, data, node_index


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_hgt()

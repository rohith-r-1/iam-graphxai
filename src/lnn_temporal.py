# src/lnn_temporal.py
"""
CloudShield — LNN (Liquid Neural Network) on Raw Features
==========================================================
Fixes:
  1. Loads labeled_features_merged.csv (5539 rows)
  2. Dynamic SMOTE — never requests fewer samples than exist
  3. CloudShieldLNN accepts use_ncps param — honours checkpoint architecture
  4. 5-fold cross-validation with --cv-only / --no-cv flags
  5. Saves output/lnn_results.json with CV scores

Run:
  python src/lnn_temporal.py             # full train + CV
  python src/lnn_temporal.py --cv-only   # CV only
  python src/lnn_temporal.py --no-cv     # train only
"""

import os, sys, json, pickle, warnings, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')

DATA_CSV = 'data/labeled_features_merged.csv'

# ── ncps backend ──────────────────────────────────────────────────────────────
try:
    from ncps.torch import LTC
    from ncps.wirings import AutoNCP
    USE_NCPS = True
    print("Using ncps.torch LTC (PyTorch backend)")
except ImportError:
    USE_NCPS = False
    print("WARNING: ncps not installed — falling back to GRU")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class CloudShieldLNN(nn.Module):
    """
    Liquid Neural Network for IAM policy risk classification.
    Input  : [batch, T, input_size]
    Output : [batch, num_classes]

    use_ncps param allows loading a GRU checkpoint even when ncps is
    installed (and vice-versa) — critical for lnn_concat_model.pt.
    """
    def __init__(self, input_size=38, hidden_size=64,
                 num_classes=3, use_ncps=None):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Respect caller override; fall back to global flag
        _use = USE_NCPS if use_ncps is None else use_ncps

        if _use:
            wiring        = AutoNCP(hidden_size, num_classes)
            self.ltc      = LTC(input_size, wiring, batch_first=True)
            self.use_ncps = True
        else:
            self.gru      = nn.GRU(input_size, hidden_size,
                                   num_layers=2, batch_first=True,
                                   dropout=0.2)
            self.head     = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 64),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
            self.use_ncps = False

    def forward(self, x):
        if self.use_ncps:
            out, _ = self.ltc(x)
            return out[:, -1, :]
        else:
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
def focal_loss(logits, targets, gamma=2.0, weight=None):
    ce   = F.cross_entropy(logits, targets, weight=weight, reduction='none')
    pt   = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Temporal sequence simulation
# ─────────────────────────────────────────────────────────────────────────────
def simulate_temporal_sequences(X, y, T=5, seed=42, noise_std=0.02):
    """Simulate T snapshots per policy. Shape: [N, T, D]"""
    rng     = np.random.default_rng(seed)
    N, D    = X.shape
    seq     = np.zeros((N, T, D), dtype=np.float32)
    for t in range(T):
        seq[:, t, :] = X + rng.normal(0, noise_std, (N, D)).astype(np.float32)
    return seq


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic SMOTE
# ─────────────────────────────────────────────────────────────────────────────
def apply_smote(X_tr, y_tr):
    """
    SMOTE that adapts to actual class counts.
    Never requests fewer samples than already exist.
    """
    print("\nApplying SMOTE augmentation...")
    counts_tr = {int(c): int((y_tr == c).sum()) for c in np.unique(y_tr)}
    majority  = max(counts_tr.values())
    print(f"  Class counts before SMOTE: {counts_tr}")

    try:
        from imblearn.over_sampling import SMOTE

        smote_strategy = {
            cls: max(cnt, min(cnt * 2, majority))
            for cls, cnt in counts_tr.items()
            if cnt < majority
        }

        if not smote_strategy:
            print(f"  SMOTE skipped — classes balanced: {counts_tr}")
            return X_tr, y_tr

        min_cls_count = min(counts_tr[c] for c in smote_strategy)
        k_neighbors   = max(1, min(min_cls_count - 1, 5))

        sm           = SMOTE(sampling_strategy=smote_strategy,
                             k_neighbors=k_neighbors, random_state=42)
        X_sm, y_sm   = sm.fit_resample(X_tr, y_tr)
        new_counts   = {int(c): int((y_sm == c).sum()) for c in np.unique(y_sm)}
        print(f"  SMOTE success: {new_counts}")
        return X_sm, y_sm

    except Exception as e:
        print(f"  SMOTE fallback (Gaussian noise): {e}")
        X_aug, y_aug = [X_tr], [y_tr]
        rng          = np.random.default_rng(42)

        for cls, cnt in counts_tr.items():
            if cnt < majority:
                n_syn = max(0, min(cnt, majority) - cnt)
                if n_syn <= 0:
                    continue
                cls_idx = np.where(y_tr == cls)[0]
                chosen  = cls_idx[rng.integers(0, len(cls_idx), n_syn)]
                synth   = X_tr[chosen] + rng.normal(
                    0, 0.02, (n_syn, X_tr.shape[1])
                ).astype(np.float32)
                X_aug.append(synth)
                y_aug.append(np.full(n_syn, cls, dtype=y_tr.dtype))
                print(f"  Added {n_syn} synthetic samples for class {cls}")

        X_out      = np.vstack(X_aug)
        y_out      = np.concatenate(y_aug)
        new_counts = {int(c): int((y_out == c).sum()) for c in np.unique(y_out)}
        print(f"  Fallback result: {new_counts}")
        return X_out, y_out


# ─────────────────────────────────────────────────────────────────────────────
# Build DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
def build_loaders(X_tr, y_tr, X_te, y_te, T=5, seed_tr=42, seed_te=99):
    X_tr_seq = simulate_temporal_sequences(X_tr, y_tr, T=T, seed=seed_tr)
    X_te_seq = simulate_temporal_sequences(X_te, y_te, T=T, seed=seed_te)

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr_seq, dtype=torch.float32),
                      torch.tensor(y_tr,     dtype=torch.long)),
        batch_size=64, shuffle=True, drop_last=False
    )
    test_dl  = DataLoader(
        TensorDataset(torch.tensor(X_te_seq, dtype=torch.float32),
                      torch.tensor(y_te,     dtype=torch.long)),
        batch_size=64, shuffle=False, drop_last=False
    )
    return train_dl, test_dl, X_tr_seq.shape


# ─────────────────────────────────────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(y_tr_t):
    counts = torch.bincount(y_tr_t, minlength=3).float()
    raw_w  = 1.0 / (counts + 1e-6)
    low_w  = raw_w[0].item()
    cw     = torch.tensor([
        raw_w[0].item(),
        min(raw_w[1].item(), low_w * 2.5),
        min(raw_w[2].item(), low_w * 15.0)
    ])
    return cw / cw.min()


# ─────────────────────────────────────────────────────────────────────────────
# Core training loop (reused by train_lnn and CV)
# ─────────────────────────────────────────────────────────────────────────────
def run_training(train_dl, test_dl, n_features, cw,
                 epochs=300, warmup=5, base_lr=0.001,
                 patience=30, gamma=2.0, verbose=True):
    model     = CloudShieldLNN(input_size=n_features, hidden_size=64,
                               num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-5
    )

    best_f1, patience_ctr, best_state = 0.0, 0, None
    history = []

    for epoch in range(epochs):
        if epoch < warmup:
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * (epoch + 1) / warmup

        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = focal_loss(model(xb), yb, gamma=gamma, weight=cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in test_dl:
                    preds.extend(model(xb).argmax(1).numpy())
                    trues.extend(yb.numpy())

            val_f1 = f1_score(trues, preds, average='macro', zero_division=0)
            avg_l  = epoch_loss / len(train_dl)
            lr_cur = optimizer.param_groups[0]['lr']

            if verbose:
                print(f"  Epoch {epoch:3d}: Loss={avg_l:.4f}, "
                      f"Val F1={val_f1:.4f}, LR={lr_cur:.6f}")

            history.append({'epoch': epoch, 'loss': round(avg_l, 4),
                            'val_f1': round(val_f1, 4), 'lr': lr_cur})
            scheduler.step(val_f1)

            if val_f1 > best_f1:
                best_f1      = val_f1
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_f1, history


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    with open('models/feature_names_v2.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    df = pd.read_csv(DATA_CSV)
    df = df[df['risk_label'].isin([0, 1, 2])].reset_index(drop=True)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_names].fillna(0).values.astype(np.float32)
    y = df['risk_label'].values.astype(int)
    return X, y, feature_names, df


# ─────────────────────────────────────────────────────────────────────────────
# Full training run
# ─────────────────────────────────────────────────────────────────────────────
def train_lnn():
    X, y, feature_names, df = load_data()
    n_features = len(feature_names)

    print(f"Loading labeled features...")
    print(f"Dataset             : {len(df)} policies × {n_features} features")
    print(f"Label distribution  : {dict(zip(*np.unique(y, return_counts=True)))}")

    scaler = StandardScaler()
    X      = scaler.fit_transform(X).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")
    print(f"Train dist          : {dict(zip(*np.unique(y_tr, return_counts=True)))}")
    print(f"Test  dist          : {dict(zip(*np.unique(y_te, return_counts=True)))}")

    X_tr, y_tr = apply_smote(X_tr, y_tr)

    T                     = 5
    train_dl, test_dl, sh = build_loaders(X_tr, y_tr, X_te, y_te, T=T)

    print(f"\nSimulating T={T} temporal snapshots...")
    print(f"  Train seq shape   : {sh}")

    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    cw     = compute_class_weights(y_tr_t)

    total_p = sum(p.numel() for p in
                  CloudShieldLNN(input_size=n_features).parameters())
    print(f"\nArchitecture        : {'ncps AutoNCP LTC' if USE_NCPS else 'GRU fallback'}")
    print(f"Model parameters    : {total_p:,}")
    print(f"Class weights       : LOW={cw[0]:.2f}  MED={cw[1]:.2f}  HIGH={cw[2]:.2f}")
    print(f"\nTraining LNN (300 epochs, focal γ=2.0, warmup=5)...")

    model, best_f1, history = run_training(
        train_dl, test_dl, n_features, cw, verbose=True
    )

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds.extend(model(xb).argmax(1).numpy())
            trues.extend(yb.numpy())

    test_f1   = f1_score(trues, preds, average='macro', zero_division=0)
    label_map = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
    present   = sorted(set(trues + preds))

    print(f"\nBest Validation F1  : {best_f1:.4f}")
    print(f"Final Test Macro F1 : {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        trues, preds,
        labels=present,
        target_names=[label_map[l] for l in present],
        zero_division=0
    ))

    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict' : model.state_dict(),
        'input_size'       : n_features,
        'hidden_size'      : 64,
        'num_classes'      : 3,
        'T'                : T,
        'best_f1'          : best_f1,
        'test_f1'          : test_f1,
        'use_ncps'         : USE_NCPS,
        'feature_names'    : feature_names,
        'training_history' : history,
    }, 'models/lnn_model.pt')

    with open('models/lnn_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Saved: models/lnn_model.pt")
    print("Saved: models/lnn_scaler.pkl")

    os.makedirs('output', exist_ok=True)
    results = {
        'test_f1'           : round(test_f1,   4),
        'best_val_f1'       : round(best_f1,   4),
        'input_size'        : n_features,
        'dataset_size'      : len(df),
        'label_distribution': {
            int(k): int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        },
        'architecture'      : 'ncps_ltc' if USE_NCPS else 'gru',
        'epochs_trained'    : history[-1]['epoch'] if history else 0,
        'smote_applied'     : True,
    }
    with open('output/lnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Saved: output/lnn_results.json")
    print(f"\n✅  lnn_temporal.py complete  (best F1={best_f1:.4f})")
    return test_f1


# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────
def cross_validate_lnn(n_splits=5):
    X, y, feature_names, df = load_data()
    n_features = len(feature_names)

    print(f"\n{'='*60}")
    print(f"LNN {n_splits}-Fold Cross-Validation")
    print(f"Dataset: {len(df)} policies × {n_features} features")
    print(f"{'='*60}")

    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1s = []
    T        = 5

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
        X_te   = scaler.transform(X_te).astype(np.float32)

        X_tr, y_tr = apply_smote(X_tr, y_tr)

        train_dl, test_dl, _ = build_loaders(
            X_tr, y_tr, X_te, y_te, T=T,
            seed_tr=42 + fold, seed_te=99 + fold
        )

        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        cw     = compute_class_weights(y_tr_t)

        print(f"\n  Fold {fold+1}/{n_splits} — "
              f"train={len(y_tr)}, test={len(y_te)}")
        print(f"  Training fold {fold+1}... ", end='', flush=True)

        _, best_f1, _ = run_training(
            train_dl, test_dl, n_features, cw,
            epochs=300, patience=30, verbose=False
        )

        fold_f1s.append(best_f1)
        print(f"done  (F1={best_f1:.4f})")

    mean_f1 = float(np.mean(fold_f1s))
    std_f1  = float(np.std(fold_f1s))

    print(f"\n{'='*60}")
    print(f"LNN {n_splits}-Fold CV Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Per-fold: {[round(f, 4) for f in fold_f1s]}")
    print(f"{'='*60}")

    try:
        with open('output/lnn_results.json') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    results['cv_macro_f1']     = round(mean_f1, 4)
    results['cv_macro_f1_std'] = round(std_f1,  4)
    results['cv_fold_scores']  = [round(f, 4) for f in fold_f1s]
    results['n_splits']        = n_splits

    os.makedirs('output', exist_ok=True)
    with open('output/lnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Updated: output/lnn_results.json")
    return mean_f1, std_f1


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CloudShield LNN Training")
    parser.add_argument('--cv-only', action='store_true',
                        help='Run only cross-validation')
    parser.add_argument('--no-cv',  action='store_true',
                        help='Skip cross-validation after training')
    args = parser.parse_args()

    if args.cv_only:
        cross_validate_lnn()
    else:
        train_lnn()
        if not args.no_cv:
            cross_validate_lnn()

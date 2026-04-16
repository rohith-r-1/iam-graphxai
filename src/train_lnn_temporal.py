"""
src/train_lnn_temporal.py
=========================
Retrain CloudShieldLNN on real flaws.cloud temporal sequences.
Input : data/flaws_sequences.npz  → X(381033, 10, 38)  y(381033,)
Output: models/lnn_temporal.pt
        output/lnn_temporal_results.json

No GPU needed — trains overnight on CPU (~4-6 hrs for 20 epochs).
Run: python src/train_lnn_temporal.py
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NPZ_PATH    = 'data/flaws_sequences.npz'
MODEL_OUT   = 'models/lnn_temporal.pt'
RESULTS_OUT = 'output/lnn_temporal_results.json'

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE  = 256
EPOCHS      = 20
LR          = 1e-3
LR_PATIENCE = 4       # ReduceLROnPlateau patience
ES_PATIENCE = 7       # Early stopping patience
SEED        = 42
HIGH_OVER   = 20      # oversample HIGH class × this factor in sampler

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading sequences...")
data = np.load(NPZ_PATH)
X    = data['X'].astype(np.float32)   # (N, 10, 38)
y    = data['y'].astype(np.int64)

N, T, F = X.shape
print(f"  Shape   : {X.shape}")
print(f"  Classes : {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"  Input   : T={T} timesteps  F={F} features")

# ── Train/val/test split (stratified) ─────────────────────────────────────────
X_tmp, X_te, y_tmp, y_te = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.15, random_state=SEED, stratify=y_tmp
)

print(f"\n  Train   : {len(y_tr):,}")
print(f"  Val     : {len(y_val):,}")
print(f"  Test    : {len(y_te):,}")

# ── Class weights for loss ─────────────────────────────────────────────────────
cw     = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_tr)
cw_t   = torch.tensor(cw, dtype=torch.float32)
print(f"\n  Class weights: LOW={cw[0]:.3f}  MEDIUM={cw[1]:.3f}  HIGH={cw[2]:.3f}")

# ── Weighted sampler — oversample HIGH during training ─────────────────────────
sample_weights = np.ones(len(y_tr), dtype=np.float32)
sample_weights[y_tr == 2] *= HIGH_OVER   # HIGH gets 20× sampling weight
sample_weights[y_tr == 1] *= 2           # MEDIUM gets 2× sampling weight
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights),
    num_samples=len(y_tr),
    replacement=True
)

tr_ds  = TensorDataset(torch.tensor(X_tr),  torch.tensor(y_tr))
val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
te_ds  = TensorDataset(torch.tensor(X_te),  torch.tensor(y_te))

tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
te_loader  = DataLoader(te_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading CloudShieldLNN architecture...")
from lnn_temporal import CloudShieldLNN

attempts = [
    lambda: CloudShieldLNN(input_dim=F, num_classes=3),
    lambda: CloudShieldLNN(input_size=F, num_classes=3),
    lambda: CloudShieldLNN(F, 3),
    lambda: CloudShieldLNN(num_classes=3),
    lambda: CloudShieldLNN(F),
    lambda: CloudShieldLNN(),
]
model, last_err = None, None
for attempt in attempts:
    try:
        model = attempt()
        break
    except TypeError as e:
        last_err = e

if model is None:
    raise RuntimeError(f"Cannot instantiate CloudShieldLNN: {last_err}")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model   : {model.__class__.__name__}")
print(f"  Params  : {total_params:,}")

# ── Optimizer + scheduler ─────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=cw_t)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=LR_PATIENCE, verbose=True
)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nTraining for up to {EPOCHS} epochs "
      f"(batch={BATCH_SIZE}, lr={LR}, early_stop={ES_PATIENCE})...")
print("-" * 65)

best_val_f1   = 0.0
best_state    = None
es_counter    = 0
history       = []
train_start   = time.time()

for epoch in range(1, EPOCHS + 1):
    ep_start = time.time()

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for Xb, yb in tr_loader:
        optimizer.zero_grad()
        # unsqueeze NOT needed — X is already (B, T, F)
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss    += loss.item() * len(yb)
        tr_correct += (out.argmax(1) == yb).sum().item()
        tr_total   += len(yb)

    tr_loss /= tr_total
    tr_acc   = tr_correct / tr_total

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    val_preds, val_true = [], []
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            out       = model(Xb)
            val_loss += criterion(out, yb).item() * len(yb)
            val_preds.extend(out.argmax(1).tolist())
            val_true.extend(yb.tolist())

    val_loss /= len(y_val)
    val_f1    = f1_score(val_true, val_preds, average='macro', zero_division=0)
    val_acc   = sum(p == t for p, t in zip(val_preds, val_true)) / len(val_true)

    scheduler.step(val_f1)

    ep_time = time.time() - ep_start
    eta_min = int((EPOCHS - epoch) * ep_time / 60)

    print(f"  Ep {epoch:02d}/{EPOCHS}  "
          f"loss={tr_loss:.4f}  acc={tr_acc:.4f}  |  "
          f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  val_acc={val_acc:.4f}  "
          f"[{ep_time:.0f}s, ETA {eta_min}m]")

    history.append({
        'epoch': epoch, 'tr_loss': round(tr_loss, 4),
        'tr_acc': round(tr_acc, 4), 'val_loss': round(val_loss, 4),
        'val_f1': round(val_f1, 4), 'val_acc': round(val_acc, 4)
    })

    # Save best
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        es_counter  = 0
        print(f"    ✔ New best val_f1={best_val_f1:.4f} — checkpoint saved")
    else:
        es_counter += 1
        if es_counter >= ES_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {ES_PATIENCE} epochs)")
            break

total_time = time.time() - train_start
print(f"\nTraining complete in {total_time/60:.1f} min")

# ── Test evaluation ───────────────────────────────────────────────────────────
print("\nEvaluating on held-out test set...")
model.load_state_dict(best_state)
model.eval()

te_preds, te_true, te_proba = [], [], []
with torch.no_grad():
    for Xb, yb in te_loader:
        out = model(Xb)
        prob = torch.softmax(out, dim=1)
        te_preds.extend(out.argmax(1).tolist())
        te_true.extend(yb.tolist())
        te_proba.extend(prob.numpy())

te_proba = np.array(te_proba)
te_f1    = f1_score(te_true, te_preds, average='macro', zero_division=0)
te_f1_w  = f1_score(te_true, te_preds, average='weighted', zero_division=0)

try:
    te_auc = roc_auc_score(te_true, te_proba, multi_class='ovr', average='macro')
except Exception:
    te_auc = 0.0

print(f"\n  Test Macro F1    : {te_f1:.4f}")
print(f"  Test Weighted F1 : {te_f1_w:.4f}")
print(f"  Test ROC-AUC     : {te_auc:.4f}")
print("\nClassification Report:")
print(classification_report(te_true, te_preds,
                             target_names=['LOW','MEDIUM','HIGH'],
                             zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(te_true, te_preds))

# ── Save model ────────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
torch.save({
    'model_state_dict': best_state,
    'input_dim':        F,
    'timesteps':        T,
    'num_classes':      3,
    'val_f1':           best_val_f1,
    'test_f1':          te_f1,
    'architecture':     'CloudShieldLNN_temporal',
    'trained_on':       'flaws.cloud CloudTrail sequences',
    'epochs_trained':   epoch,
}, MODEL_OUT)
print(f"\nSaved: {MODEL_OUT}")

# ── Save results ──────────────────────────────────────────────────────────────
os.makedirs('output', exist_ok=True)
results = {
    'lnn_temporal_test_f1':      round(te_f1,    4),
    'lnn_temporal_test_f1_w':    round(te_f1_w,  4),
    'lnn_temporal_roc_auc':      round(te_auc,   4),
    'lnn_temporal_best_val_f1':  round(best_val_f1, 4),
    'lnn_temporal_epochs':       epoch,
    'lnn_temporal_train_time_min': round(total_time/60, 1),
    'dataset':                   'flaws.cloud CloudTrail',
    'n_sequences':               len(y),
    'n_train':                   len(y_tr),
    'n_val':                     len(y_val),
    'n_test':                    len(y_te),
    'timesteps':                 T,
    'features':                  F,
    'history':                   history
}
with open(RESULTS_OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS_OUT}")
print(f"\nDone. LNN now trained on REAL temporal sequences.")
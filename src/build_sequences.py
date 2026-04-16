"""
src/build_sequences.py  — Fixed v2
===================================
Fixes:
  1. Vectorized expand_to_38 (numpy, no Python loops) — 30s not 6hrs
  2. Rebalanced labels using error patterns + AssumeRole + IAM writes
     since flaws.cloud has only 116 IAM escalation events total
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_CSV  = 'data/flaws/nineteenFeaturesDf.csv'
OUT_NPZ  = 'data/flaws_sequences.npz'
OUT_META = 'data/flaws_sequence_meta.csv'
T        = 10
STRIDE   = 5
MIN_EVENTS = 10

ESC_ACTIONS = {
    'PassRole', 'CreatePolicyVersion', 'SetDefaultPolicyVersion',
    'CreateUser', 'AttachUserPolicy', 'AttachRolePolicy',
    'PutUserPolicy', 'AddUserToGroup', 'UpdateAssumeRolePolicy',
    'CreateAccessKey', 'CreateLoginProfile', 'UpdateLoginProfile'
}
IAM_WRITE_ACTIONS = {
    'CreateUser','DeleteUser','CreateRole','DeleteRole',
    'AttachUserPolicy','DetachUserPolicy','AttachRolePolicy','DetachRolePolicy',
    'PutUserPolicy','DeleteUserPolicy','PutRolePolicy','DeleteRolePolicy',
    'CreatePolicy','CreatePolicyVersion','SetDefaultPolicyVersion','DeletePolicy',
    'PassRole','CreateAccessKey','UpdateAccessKey','AddUserToGroup',
    'RemoveUserFromGroup','CreateLoginProfile','UpdateLoginProfile'
}
DANGEROUS_ACTIONS = {
    'PassRole','CreatePolicyVersion','SetDefaultPolicyVersion',
    'CreateUser','AttachUserPolicy','AttachRolePolicy',
    'CreateAccessKey','UpdateAssumeRolePolicy'
}
RECON_ACTIONS = {
    'GetCallerIdentity','ListBuckets','GetBucketAcl',
    'DescribeInstances','GetPolicyVersion','ListFunctions202224',
    'ListEntitiesForPolicy','ListPolicyVersions','GetPolicy',
    'DescribeSecurityGroups','GetAccountPasswordPolicy','ListUsers',
    'ListRoles','ListGroups','GetAccountSummary'
}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading CloudTrail data...")
df = pd.read_csv(RAW_CSV, low_memory=False)
print(f"  Rows: {len(df):,}  |  IPs: {df['sourceIPAddress'].nunique():,}")

df['eventTime'] = pd.to_datetime(df['eventTime'], errors='coerce', utc=True)
df = df.dropna(subset=['eventTime', 'sourceIPAddress'])
df = df.sort_values(['sourceIPAddress', 'eventTime']).reset_index(drop=True)

# ── Per-event binary features (10D) ──────────────────────────────────────────
print("Extracting per-event features (vectorized)...")

ename = df['eventName'].fillna('').astype(str)
esrc  = df['eventSource'].fillna('').astype(str)
etype = df['userIdentitytype'].fillna('').astype(str)
err   = df['errorCode'].fillna('NoError').astype(str)

f0  = ename.isin(ESC_ACTIONS).astype(np.float32)
f1  = ename.isin(IAM_WRITE_ACTIONS).astype(np.float32)
f2  = ename.isin(DANGEROUS_ACTIONS).astype(np.float32)
f3  = ename.isin(RECON_ACTIONS).astype(np.float32)
f4  = esrc.str.contains('iam', na=False).astype(np.float32)
f5  = etype.eq('Root').astype(np.float32)
f6  = etype.eq('IAMUser').astype(np.float32)
f7  = ename.eq('AssumeRole').astype(np.float32)
f8  = err.isin({'AccessDenied','Client.UnauthorizedOperation',
                'UnauthorizedOperation'}).astype(np.float32)
f9  = err.ne('NoError').astype(np.float32)

feat_matrix = np.stack([f0,f1,f2,f3,f4,f5,f6,f7,f8,f9], axis=1).astype(np.float32)
ips         = df['sourceIPAddress'].values
times       = df['eventTime'].values

# ── Build sliding window sequences ───────────────────────────────────────────
print("Building sliding window sequences...")

sequences, labels, meta = [], [], []

ip_vals, ip_starts, ip_counts = np.unique(ips, return_index=True, return_counts=True)

for ip, start_idx, count in zip(ip_vals, ip_starts, ip_counts):
    if count < MIN_EVENTS:
        continue
    ip_feats = feat_matrix[start_idx: start_idx + count]
    ip_times = times[start_idx: start_idx + count]

    for s in range(0, count - T + 1, STRIDE):
        win   = ip_feats[s: s + T]      # (T, 10)
        t_s   = str(ip_times[s])
        t_e   = str(ip_times[s + T - 1])

        # ── Rebalanced labeling ───────────────────────────────────────────────
        # Uses error patterns + recon + IAM writes since ESC events are rare
        esc_n     = int(win[:, 0].sum())   # escalation actions
        iam_w     = int(win[:, 1].sum())   # IAM writes
        danger_n  = int(win[:, 2].sum())   # dangerous actions
        recon_n   = int(win[:, 3].sum())   # recon actions
        iam_svc_n = int(win[:, 4].sum())   # IAM service calls
        is_root   = int(win[:, 5].any())   # any Root activity
        assume_n  = int(win[:, 7].sum())   # AssumeRole calls
        denied_n  = int(win[:, 8].sum())   # AccessDenied errors
        error_n   = int(win[:, 9].sum())   # any error

        # HIGH: direct privilege escalation evidence
        if esc_n >= 1 or danger_n >= 2 or (iam_w >= 3 and denied_n >= 2):
            label = 2
        # MEDIUM: suspicious recon + IAM activity or elevated errors
        elif (iam_w >= 1 or assume_n >= 2 or
              (recon_n >= 3 and iam_svc_n >= 2) or
              (denied_n >= 4) or
              (is_root and iam_svc_n >= 1) or
              (error_n >= 5 and iam_svc_n >= 1)):
            label = 1
        else:
            label = 0

        sequences.append(win)
        labels.append(label)
        meta.append({
            'ip': ip, 'start': t_s, 'end': t_e,
            'esc_n': esc_n, 'iam_w': iam_w, 'danger_n': danger_n,
            'recon_n': recon_n, 'denied_n': denied_n,
            'assume_n': assume_n, 'label': label,
            'label_name': ['LOW','MEDIUM','HIGH'][label]
        })

X_raw = np.array(sequences, dtype=np.float32)   # (N, T, 10)
y     = np.array(labels,    dtype=np.int64)

print(f"\nSequences: {len(y):,}  shape={X_raw.shape}")
for lbl, name in enumerate(['LOW','MEDIUM','HIGH']):
    c = int((y == lbl).sum())
    print(f"  {name:<8}: {c:>7,}  ({c/len(y)*100:.1f}%)")

# ── Vectorized expand 10D → 38D ───────────────────────────────────────────────
print("\nExpanding 10D → 38D (vectorized)...")

N, T_dim, F = X_raw.shape
X38 = np.zeros((N, T_dim, 38), dtype=np.float32)
X38[:, :, :10] = X_raw   # first 10 = raw event features

# Cumulative sums over time axis — (N, T, feature)
cum = np.cumsum(X_raw, axis=1)                   # (N, T, 10)

X38[:, :, 10] = cum[:, :, 0]   # cumulative esc actions
X38[:, :, 11] = cum[:, :, 1]   # cumulative IAM writes
X38[:, :, 12] = cum[:, :, 2]   # cumulative dangerous
X38[:, :, 13] = cum[:, :, 8]   # cumulative denials
X38[:, :, 14] = cum[:, :, 7]   # cumulative AssumeRole

# Relative timestep position: 0.0 → 1.0
steps = np.linspace(0, 1, T_dim, dtype=np.float32)
X38[:, :, 15] = steps[np.newaxis, :]

# Ever-seen flags (cummax via cumsum > 0)
X38[:, :, 16] = (cum[:, :, 0] > 0).astype(np.float32)   # ever esc
X38[:, :, 17] = (cum[:, :, 2] > 0).astype(np.float32)   # ever dangerous

# Rolling window (last 3 steps) — use uniform filter approximation
def rolling_mean(arr, w=3):
    """arr: (N, T) → (N, T) rolling mean over last w steps"""
    out = np.zeros_like(arr)
    for t in range(arr.shape[1]):
        start = max(0, t - w + 1)
        out[:, t] = arr[:, start:t+1].mean(axis=1)
    return out

X38[:, :, 18] = rolling_mean(X_raw[:, :, 0])   # recent esc rate
X38[:, :, 19] = rolling_mean(X_raw[:, :, 8])   # recent denial rate
X38[:, :, 20] = rolling_mean(X_raw[:, :, 9])   # recent error rate
X38[:, :, 21] = rolling_mean(X_raw[:, :, 4])   # recent IAM service rate
X38[:, :, 22] = rolling_mean(X_raw[:, :, 3])   # recent recon rate

# Full-window aggregates (global, broadcast to all T)
X38[:, :, 23] = X_raw[:, :, 0].mean(axis=1, keepdims=True)  # mean esc rate
X38[:, :, 24] = X_raw[:, :, 8].mean(axis=1, keepdims=True)  # mean denial rate
X38[:, :, 25] = X_raw[:, :, 9].mean(axis=1, keepdims=True)  # mean error rate
X38[:, :, 26] = X_raw[:, :, 4].mean(axis=1, keepdims=True)  # mean IAM rate
X38[:, :, 27] = X_raw[:, :, 7].mean(axis=1, keepdims=True)  # mean AssumeRole
X38[:, :, 28] = X_raw[:, :, 2].sum(axis=1,  keepdims=True)  # total dangerous
X38[:, :, 29] = X_raw[:, :, 3].sum(axis=1,  keepdims=True)  # total recon

# Identity/session context (static per sequence)
X38[:, :, 30] = X_raw[:, 0:1, 5]               # Root at start
X38[:, :, 31] = X_raw[:, :, 5].max(axis=1, keepdims=True)  # any Root in seq
X38[:, :, 32] = (X_raw[:, :, 0].sum(axis=1, keepdims=True) > 0).astype(np.float32)
X38[:, :, 33] = (X_raw[:, :, 8].sum(axis=1, keepdims=True) > 2).astype(np.float32)
X38[:, :, 34] = X_raw[:, :, 9].max(axis=1, keepdims=True)
X38[:, :, 35] = (X_raw[:, :, 7].sum(axis=1, keepdims=True) > 0).astype(np.float32)
X38[:, :, 36] = X_raw[:, :, 1].sum(axis=1, keepdims=True)  # total IAM writes
X38[:, :, 37] = X_raw[:, :, 6].mean(axis=1, keepdims=True) # mean IAMUser rate

print(f"  Expanded: {X38.shape}")

# ── Normalize per feature across (N*T) ────────────────────────────────────────
print("Normalizing...")
flat   = X38.reshape(-1, 38)
scaler = StandardScaler()
X_norm = scaler.fit_transform(flat).reshape(N, T_dim, 38).astype(np.float32)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
np.savez_compressed(OUT_NPZ, X=X_norm, y=y, X_raw=X38)
pd.DataFrame(meta).to_csv(OUT_META, index=False)

print(f"\nSaved : {OUT_NPZ}   ({X_norm.nbytes/1e6:.1f} MB uncompressed)")
print(f"Saved : {OUT_META}")
print(f"\nDone — {len(y):,} real temporal sequences ready for LNN retraining.")
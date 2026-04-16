import pandas as pd
import json
import os
import numpy as np
import pickle
import sys

# ── CHANGE THIS TO YOUR TEAMMATE'S DATA FOLDER ────────────────────
DATA = r"D:\iam-graph-xai\data"   # e.g.  r"E:\teammate-work\data"
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  1. labeled_features.csv")
print("="*60)
df = pd.read_csv(f"{DATA}/labeled_features.csv")
print(f"  Rows × Cols   : {df.shape}")
print(f"  All columns   : {df.columns.tolist()}")
if 'risk_label' in df.columns:
    vc  = df['risk_label'].value_counts().sort_index().to_dict()
    pct = df['risk_label'].value_counts(normalize=True).mul(100).round(1).to_dict()
    print(f"  Label counts  : {vc}")
    print(f"  Label %       : {pct}")
print(f"  Null values   : {df.isnull().sum().sum()}")
print(f"  Has policy_id : {'policy_id' in df.columns}")
print(f"  First 3 rows  :")
print(df.head(3).to_string())

print("\n" + "="*60)
print("  2. graph_features.csv")
print("="*60)
gp = f"{DATA}/graph_features.csv"
if os.path.exists(gp):
    gf = pd.read_csv(gp)
    print(f"  Rows × Cols   : {gf.shape}")
    print(f"  All columns   : {gf.columns.tolist()}")
    print(f"  Null values   : {gf.isnull().sum().sum()}")
else:
    print("  NOT FOUND")

print("\n" + "="*60)
print("  3. temporal_sequences/")
print("="*60)
ts_dir = f"{DATA}/temporal_sequences"
if os.path.exists(ts_dir):
    files = os.listdir(ts_dir)
    print(f"  Total files   : {len(files)}")
    print(f"  File types    : {set(f.split('.')[-1] for f in files)}")
    print(f"  Sample names  : {files[:5]}")
    for fname in files[:3]:
        fpath = f"{ts_dir}/{fname}"
        ext   = fname.split('.')[-1]
        try:
            if ext == 'npy':
                arr = np.load(fpath)
                print(f"\n  {fname}")
                print(f"    Shape        : {arr.shape}")
                print(f"    Dtype        : {arr.dtype}")
                print(f"    Min/Max      : {arr.min():.4f} / {arr.max():.4f}")
            elif ext == 'json':
                with open(fpath) as f:
                    d = json.load(f)
                print(f"\n  {fname}")
                print(f"    Keys         : {list(d.keys())[:8]}")
            elif ext == 'csv':
                tmp = pd.read_csv(fpath)
                print(f"\n  {fname}")
                print(f"    Shape        : {tmp.shape}")
        except Exception as e:
            print(f"  ERROR: {fname} → {e}")
else:
    print("  NOT FOUND")

print("\n" + "="*60)
print("  4. Policy Folders")
print("="*60)
for folder in ['realworld_policies', 'guideline_policies',
               'synthetic_policies', 'cloudgoat',
               'raw_policies', 'aws-iam-managed-policies']:
    fdir = f"{DATA}/{folder}"
    if not os.path.exists(fdir):
        print(f"  {folder:<35} NOT FOUND")
        continue
    files = os.listdir(fdir)
    types = set(f.split('.')[-1] for f in files)
    print(f"  {folder:<35} {len(files):>5} files  types={types}")
    for fname in files[:1]:
        try:
            if fname.endswith('.json'):
                with open(f"{fdir}/{fname}") as fp:
                    d = json.load(fp)
                print(f"    Sample: {fname}")
                print(f"    Keys  : {list(d.keys())[:6]}")
        except:
            pass

print("\n" + "="*60)
print("  5. Metadata Files")
print("="*60)
for fname in ['synthetic_metadata.json', 'guideline_metadata.json',
              'temporal_metadata.json',   'realworld_metadata.json',
              'validation_report.json']:
    fpath = f"{DATA}/{fname}"
    if not os.path.exists(fpath):
        print(f"  {fname}: NOT FOUND")
        continue
    with open(fpath) as f:
        d = json.load(f)
    print(f"\n  {fname}")
    print(json.dumps(d, indent=4))

print("\n" + "="*60)
print("  6. Compatibility with YOUR project")
print("="*60)
try:
    with open(r"E:\iam-graph-xai\models\feature_names_v2.pkl", 'rb') as f:
        your_features = pickle.load(f)
    print(f"  Your features       : {len(your_features)}")
    print(f"  Your feature list   : {your_features}")

    df_new  = pd.read_csv(f"{DATA}/labeled_features.csv")
    missing = [c for c in your_features if c not in df_new.columns]
    extra   = [c for c in df_new.columns
               if c not in your_features
               and c not in ('policy_id', 'risk_label', 'source')]
    overlap = [c for c in your_features if c in df_new.columns]

    print(f"\n  Overlapping         : {len(overlap)}/{len(your_features)}")
    print(f"  Missing from new    : {missing}")
    print(f"  Extra new columns   : {extra}")

    if not missing:
        print("\n  ✅ FULLY COMPATIBLE — can use directly")
    elif len(missing) < 5:
        print(f"\n  ⚠️  MOSTLY COMPATIBLE — {len(missing)} cols to fix")
    else:
        print(f"\n  ❌ NOT COMPATIBLE — {len(missing)} features missing")
except Exception as e:
    print(f"  Could not check: {e}")

print("\n" + "="*60)
print("  DONE — paste all output above back to AI")
print("="*60)

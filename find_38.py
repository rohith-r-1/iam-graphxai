import pandas as pd, pickle
from itertools import combinations

df    = pd.read_csv('data/labeled_features_v2.csv')
model = pickle.load(open('models/rf_v2.pkl', 'rb'))
drop  = ['policy_id','risk_label','prob_low','prob_medium','prob_high']
feats = [c for c in df.columns if c not in drop]
row   = df[feats].fillna(0).iloc[:1]

print("Finding which 2 columns to drop...")
for skip in combinations(range(len(feats)), 2):
    cols = [f for i,f in enumerate(feats) if i not in skip]
    try:
        model.predict(row[cols])
        print(f"\nWorks! Drop: {feats[skip[0]]} + {feats[skip[1]]}")
        print(f"Final 38 features:")
        for i,c in enumerate(cols): print(f"  {i+1:2d}. {c}")
        break
    except: pass

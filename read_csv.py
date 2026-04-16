import pandas as pd
for f in ['data/labeled_features_v2.csv', 'data/labeled_features_merged.csv',
          'data/labeled_features_with_cloudgoat.csv', 'data/labeled_features.csv']:
    try:
        df = pd.read_csv(f, nrows=2)
        print(f"\nFile: {f}  cols={len(df.columns)}")
        for i,c in enumerate(df.columns): print(f"  {i+1:2d}. {c}")
        break
    except: pass

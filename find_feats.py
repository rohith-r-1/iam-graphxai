import pandas as pd, os

# Try to find the training dataframe saved anywhere
for f in ['output/features.csv', 'output/features_v2.csv', 'data/features.csv',
          'data/processed/features.csv', 'output/policy_features.csv']:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f"Found: {f}  shape={df.shape}")
        print("Columns:")
        for i,c in enumerate(df.columns): print(f"  {i+1:2d}. {c}")
        break
else:
    # check all csvs in output/
    import glob
    csvs = glob.glob('output/*.csv') + glob.glob('data/**/*.csv', recursive=True)
    print("CSV files found:")
    for c in csvs: print(f"  {c}")

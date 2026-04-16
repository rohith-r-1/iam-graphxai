import pickle
model = pickle.load(open('models/rf_v2.pkl', 'rb'))
if hasattr(model, 'feature_names_in_'):
    feats = list(model.feature_names_in_)
    print(f"Model expects {len(feats)} features:")
    for i,f in enumerate(feats): print(f"  {i+1:2d}. {f}")
else:
    print(f"No feature_names_in_. n_features_in_ = {model.n_features_in_}")
    print("Model was trained without feature names.")

# src/model_training_v2.py
"""
CloudShield — RF + XGBoost + LightGBM Classifiers
===================================================
Fixes vs original:
  1. NumpyEncoder — json.dump no longer crashes on int64 keys
  2. Leaky features removed — max_historical_risk encodes label directly
  3. LightGBM added as drop-in XGBoost upgrade
  4. sample_weight passed to XGBoost (proper multiclass imbalance handling)
  5. Cross-validation added for honest F1 estimate

Run: python src/model_training_v2.py
"""

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
warnings.filterwarnings('ignore')

# Optional LightGBM — install if missing: pip install lightgbm
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("  ⚠️  LightGBM not installed. Run: pip install lightgbm")

DATA_PATH    = r"E:/iam-graph-xai/data/labeled_features_merged.csv"
MODEL_DIR    = r"E:/iam-graph-xai/models"
FEAT_PKL     = r"E:/iam-graph-xai/models/feature_names_v2.pkl"
RESULTS_JSON = r"E:/iam-graph-xai/models/rf_results.json"

LABEL_NAMES = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}

# ── Features that leak the label — must be excluded from training ──────
# max_historical_risk  = risk_label / 2.0  → directly encodes the answer
# rollback_risk_score  = f(createpolicyversion, privilege_escalation)
#                        which was itself derived from heuristic labels
LEAKY_FEATURES = [
    'max_historical_risk',
    'rollback_risk_score',
]


# ─────────────────────────────────────────────────────────────────────
# JSON serialisation helper
# ─────────────────────────────────────────────────────────────────────
def sanitize_for_json(obj):
    """Recursively convert numpy types → native Python for json.dump."""
    if isinstance(obj, dict):
        return {
            (int(k)   if isinstance(k, np.integer)
             else float(k) if isinstance(k, np.floating)
             else str(k)): sanitize_for_json(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj


# ─────────────────────────────────────────────────────────────────────
# Rule-based baseline
# ─────────────────────────────────────────────────────────────────────
def rule_based_predict(X_df):
    preds = []
    for _, row in X_df.iterrows():
        if (row.get('passrole_chain_exists',      0) or
            row.get('createpolicyversion_exists', 0) or
            row.get('attachuserpolicy_exists',    0) or
            row.get('has_wildcard_action',        0)):
            preds.append(2)
        elif (row.get('iam_write_permission_count', 0) >= 1 or
              row.get('dangerous_action_count',     0) >= 1):
            preds.append(1)
        else:
            preds.append(0)
    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred, target_names=None):
    target_names = target_names or ['LOW', 'MEDIUM', 'HIGH']
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc      = accuracy_score(y_true, y_pred)
    report   = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{name} Results:")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"\n  Classification Report:\n{report}")
    print(f"  Confusion Matrix:\n{cm}")

    return macro_f1, acc, report, cm.tolist()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def train_models_v2():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Loading merged dataset")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df = df[df['risk_label'].isin([0, 1, 2])].reset_index(drop=True)
    print(f"  Total rows    : {len(df)}")
    print(f"  Label dist    : {df['risk_label'].value_counts().sort_index().to_dict()}")

    # ── Feature matrix ────────────────────────────────────────────
    with open(FEAT_PKL, 'rb') as f:
        all_features = pickle.load(f)

    # Remove leaky features
    feature_cols = [f for f in all_features
                    if f in df.columns and f not in LEAKY_FEATURES]
    removed = [f for f in all_features if f in LEAKY_FEATURES]
    print(f"\n  Features used     : {len(feature_cols)}")
    print(f"  Leaky (removed)   : {removed}")

    X = df[feature_cols].values.astype(np.float32)
    y = df['risk_label'].values.astype(np.int32)

    # ── Train / test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=0.2, random_state=42, stratify=y
    )
    X_train_df = df[feature_cols].iloc[idx_train]
    X_test_df  = df[feature_cols].iloc[idx_test]

    print(f"\n  Train : {len(X_train)}  Test : {len(X_test)}")
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"  Test label dist : {dict(zip(unique.tolist(), counts.tolist()))}")

    sample_weights_train = compute_sample_weight('balanced', y_train)

    # ── Baseline ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Baseline (Rule-based)")
    print("=" * 60)
    y_base = rule_based_predict(X_test_df)
    base_f1, _, _, _ = evaluate("Baseline", y_test, y_base)

    results = {
        'baseline_macro_f1': float(base_f1),
        'features_used'    : feature_cols,
        'leaky_removed'    : removed,
        'train_size'       : int(len(X_train)),
        'test_size'        : int(len(X_test)),
        'label_distribution': {
            int(k): int(v)
            for k, v in df['risk_label'].value_counts().sort_index().items()
        },
    }

    # ── Random Forest ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Random Forest")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_f1, rf_acc, rf_report, rf_cm = evaluate(
        "Random Forest", y_test, y_pred_rf
    )

    # 5-fold CV for honest estimate
    cv_rf = cross_val_score(
        RandomForestClassifier(n_estimators=100, class_weight='balanced',
                               n_jobs=-1, random_state=42),
        X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='f1_macro'
    )
    print(f"  5-Fold CV Macro F1: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

    # Top feature importances
    importances = rf.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:10]
    print(f"\n  Top 10 Feature Importances:")
    for i in top_idx:
        bar = '█' * int(importances[i] * 100)
        print(f"    {feature_cols[i]:<45} {importances[i]:.4f} {bar}")

    results['random_forest'] = {
        'macro_f1'          : float(rf_f1),
        'accuracy'          : float(rf_acc),
        'cv_macro_f1_mean'  : float(cv_rf.mean()),
        'cv_macro_f1_std'   : float(cv_rf.std()),
        'confusion_matrix'  : rf_cm,
        'top_features'      : [
            {'feature': feature_cols[i], 'importance': float(importances[i])}
            for i in top_idx
        ],
    }

    pkl_path = os.path.join(MODEL_DIR, 'rf_v2.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"\n  Saved: {pkl_path}")

    # ── XGBoost ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  XGBoost")
    print("=" * 60)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    y_pred_xgb  = xgb_model.predict(X_test)
    xgb_f1, xgb_acc, xgb_report, xgb_cm = evaluate(
        "XGBoost", y_test, y_pred_xgb
    )

    results['xgboost'] = {
        'macro_f1'        : float(xgb_f1),
        'accuracy'        : float(xgb_acc),
        'confusion_matrix': xgb_cm,
    }

    xgb_path = os.path.join(MODEL_DIR, 'xgb_v2.json')
    xgb_model.save_model(xgb_path)
    print(f"\n  Saved: {xgb_path}")

    # ── LightGBM ──────────────────────────────────────────────────
    if HAS_LGBM:
        print("\n" + "=" * 60)
        print("  LightGBM")
        print("=" * 60)

        lgbm_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            class_weight='balanced',
            objective='multiclass',
            num_class=3,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )
        y_pred_lgbm = lgbm_model.predict(X_test)
        lgbm_f1, lgbm_acc, lgbm_report, lgbm_cm = evaluate(
            "LightGBM", y_test, y_pred_lgbm
        )

        # 5-fold CV
        cv_lgbm = cross_val_score(
            lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=63,
                class_weight='balanced', objective='multiclass',
                num_class=3, random_state=42, n_jobs=-1, verbose=-1
            ),
            X, y,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='f1_macro'
        )
        print(f"  5-Fold CV Macro F1: {cv_lgbm.mean():.4f} ± {cv_lgbm.std():.4f}")

        # Top features
        lgbm_imp = lgbm_model.feature_importances_
        top_lgbm = np.argsort(lgbm_imp)[::-1][:10]
        print(f"\n  Top 10 Feature Importances:")
        for i in top_lgbm:
            bar = '█' * int(lgbm_imp[i] / max(lgbm_imp) * 20)
            print(f"    {feature_cols[i]:<45} {lgbm_imp[i]:>6}  {bar}")

        results['lightgbm'] = {
            'macro_f1'        : float(lgbm_f1),
            'accuracy'        : float(lgbm_acc),
            'cv_macro_f1_mean': float(cv_lgbm.mean()),
            'cv_macro_f1_std' : float(cv_lgbm.std()),
            'confusion_matrix': lgbm_cm,
            'top_features'    : [
                {'feature': feature_cols[i], 'importance': int(lgbm_imp[i])}
                for i in top_lgbm
            ],
        }

        lgbm_path = os.path.join(MODEL_DIR, 'lgbm_v2.pkl')
        with open(lgbm_path, 'wb') as f:
            pickle.dump(lgbm_model, f)
        print(f"\n  Saved: {lgbm_path}")

    # ── Save feature names (without leaky features) ───────────────
    feat_path = os.path.join(MODEL_DIR, 'feature_names_v2.pkl')
    with open(feat_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"\n  Saved: {feat_path}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY (For Paper Section IV)")
    print("=" * 60)
    print(f"  Baseline (Rule-based) : {base_f1:.4f} macro F1")
    print(f"  Random Forest         : {rf_f1:.4f} macro F1"
          f"  (CV: {cv_rf.mean():.4f} ± {cv_rf.std():.4f})")
    print(f"  XGBoost               : {xgb_f1:.4f} macro F1")
    if HAS_LGBM:
        print(f"  LightGBM              : {lgbm_f1:.4f} macro F1"
              f"  (CV: {cv_lgbm.mean():.4f} ± {cv_lgbm.std():.4f})")

    print(f"\n  ⚠️  Note: If F1 > 0.98, inspect these features for residual leakage:")
    print(f"       compliance_violation_count — derived from label-based heuristics")
    print(f"       privilege_escalation_risk_score — uses escalation flags")

    # ── Save results JSON ─────────────────────────────────────────
    results_path = RESULTS_JSON
    with open(results_path, 'w') as f:
        json.dump(sanitize_for_json(results), f, indent=2)
    print(f"\n  Saved: {results_path}")
    print("\n  ✅  model_training_v2.py complete")


if __name__ == '__main__':
    train_models_v2()

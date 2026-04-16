# src/weak_supervision_v2.py
"""
Updated weak supervision with 12 LFs (was 8).
Now includes CloudGoat patterns + condition-aware LFs.
Trains on 40-feature dataset.
"""

import pandas as pd
import numpy as np
import pickle

try:
    from snorkel.labeling import labeling_function, LabelingFunction
    from snorkel.labeling import PandasLFApplier
    from snorkel.labeling.model import LabelModel
    SNORKEL_AVAILABLE = True
except ImportError:
    print("Snorkel not found. Install: pip install snorkel")
    SNORKEL_AVAILABLE = False

# Labels
LOW = 0
MEDIUM = 1
HIGH = 2
ABSTAIN = -1


# ── ORIGINAL 8 LABELING FUNCTIONS ───────────────────────────

@labeling_function()
def lf_escalation_path(x):
    """Escalation path found → HIGH"""
    return HIGH if x.escalation_path_count > 0 else ABSTAIN

@labeling_function()
def lf_wildcard_dangerous(x):
    """Wildcard + dangerous actions → HIGH"""
    if x.has_wildcard_resource and x.dangerous_action_count > 3:
        return HIGH
    return ABSTAIN

@labeling_function()
def lf_wildcard_resource(x):
    """Has wildcard resource → MEDIUM"""
    return MEDIUM if x.has_wildcard_resource else ABSTAIN

@labeling_function()
def lf_high_specificity(x):
    """Very specific policy → LOW"""
    return LOW if x.specificity_score > 0.8 else ABSTAIN

@labeling_function()
def lf_many_services(x):
    """Touches many services → MEDIUM"""
    return MEDIUM if x.service_count > 10 else ABSTAIN

@labeling_function()
def lf_many_dangerous_actions(x):
    """Many dangerous actions → HIGH"""
    return HIGH if x.dangerous_action_count >= 5 else ABSTAIN

@labeling_function()
def lf_high_attachment(x):
    """Used by many entities → MEDIUM"""
    return MEDIUM if x.attachment_count > 5 else ABSTAIN

@labeling_function()
def lf_high_out_degree(x):
    """Very broad permissions → MEDIUM"""
    return MEDIUM if x.out_degree > 50 else ABSTAIN


# ── 4 NEW LABELING FUNCTIONS ─────────────────────────────────

@labeling_function()
def lf_passrole_chain(x):
    """
    NEW LF 9: PassRole permission exists → HIGH
    PassRole is the #1 privilege escalation technique in AWS
    """
    if x.passrole_chain_exists == 1:
        return HIGH
    return ABSTAIN

@labeling_function()
def lf_condition_protection(x):
    """
    NEW LF 10: Strong conditions reduce risk → downgrade
    If policy has MFA + IP restriction → drop one risk tier
    """
    if x.has_mfa_condition == 1 and x.has_ip_restriction == 1:
        return LOW  # Well-controlled policy
    elif x.has_mfa_condition == 1:
        return LOW if x.dangerous_action_count < 3 else ABSTAIN
    return ABSTAIN

@labeling_function()
def lf_rollback_risk(x):
    """
    NEW LF 11: Rollback risk detected → HIGH
    Policy has dangerous old versions + entity can rollback
    """
    if x.rollback_risk_score > 0.5:
        return HIGH
    return ABSTAIN

@labeling_function()
def lf_compliance_violations(x):
    """
    NEW LF 12: Multiple compliance violations → MEDIUM/HIGH
    3+ violations across PCI-DSS, NIST, SOC2, ISO = HIGH
    """
    if x.compliance_violation_count >= 3:
        return HIGH
    elif x.compliance_violation_count >= 2:
        return MEDIUM
    return ABSTAIN


# ── LABEL MODEL TRAINING ─────────────────────────────────────

def run_weak_supervision_v2():
    
    print("Loading 40-feature dataset with CloudGoat data...")
    
    # Try CloudGoat-merged dataset first
    try:
        df = pd.read_csv('data/labeled_features_with_cloudgoat.csv')
        print(f"Using CloudGoat-merged dataset: {len(df)} policies")
    except:
        df = pd.read_csv('data/labeled_features_merged.csv')
        print(f"Using v2 features dataset: {len(df)} policies")
    
    # Add missing columns with defaults if needed
    defaults = {
        'has_mfa_condition': 0, 'has_ip_restriction': 0,
        'has_time_restriction': 0, 'condition_protection_score': 0.0,
        'is_bounded': 0, 'passrole_chain_exists': 0,
        'rollback_risk_score': 0.0, 'compliance_violation_count': 0,
        'policy_version_count': 1, 'max_historical_risk': 0
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    
    print(f"\nDataset shape: {df.shape}")
    
    # Separate pre-labeled HIGH-risk (CloudGoat) from unlabeled
    if 'risk_label' in df.columns:
        cloudgoat_mask = df['risk_label'] == 2
        n_cloudgoat = cloudgoat_mask.sum()
        print(f"Pre-labeled HIGH risk (CloudGoat): {n_cloudgoat}")
    
    # Apply all 12 labeling functions
    lfs = [
        lf_escalation_path, lf_wildcard_dangerous, lf_wildcard_resource,
        lf_high_specificity, lf_many_services, lf_many_dangerous_actions,
        lf_high_attachment, lf_high_out_degree,
        # New LFs:
        lf_passrole_chain, lf_condition_protection, 
        lf_rollback_risk, lf_compliance_violations
    ]
    
    print(f"\nApplying {len(lfs)} labeling functions...")
    
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df)
    
    print(f"Label matrix shape: {L_train.shape}")
    
    # Coverage per LF
    print("\nLabeling function coverage:")
    lf_names = [lf.name for lf in lfs]
    for i, name in enumerate(lf_names):
        coverage = np.mean(L_train[:, i] != ABSTAIN)
        print(f"  {name}: {coverage:.1%}")
    
    # Train LabelModel
    print("\nTraining Snorkel LabelModel...")
    label_model = LabelModel(cardinality=3, verbose=True)
    
    # Use CloudGoat labels as priors if available
    if 'risk_label' in df.columns:
        known_labels = df['risk_label'].values
        # Override CloudGoat rows with known HIGH label
        for i in df.index[cloudgoat_mask]:
            L_train[i] = [HIGH] * len(lfs)
    
    label_model.fit(L_train=L_train, n_epochs=200, lr=0.001, seed=42)
    
    # Generate probabilistic labels
    probs = label_model.predict_proba(L=L_train)
    preds = label_model.predict(L=L_train, tie_break_policy='abstain')
    
    # Build labeled dataset
    df['risk_label'] = preds
    df['prob_low'] = probs[:, 0]
    df['prob_medium'] = probs[:, 1]
    df['prob_high'] = probs[:, 2]
    
    # Force CloudGoat labels (override any abstentions)
    if 'risk_label' in df.columns:
        df.loc[cloudgoat_mask, 'risk_label'] = 2
        df.loc[cloudgoat_mask, 'prob_high'] = 0.95
        df.loc[cloudgoat_mask, 'prob_medium'] = 0.04
        df.loc[cloudgoat_mask, 'prob_low'] = 0.01
    
    print(f"\nLabel distribution after LabelModel:")
    print(df['risk_label'].value_counts())
    
    coverage = np.mean(preds != ABSTAIN)
    print(f"\nCoverage: {coverage:.1%}")
    
    # Save labeled dataset
    df.to_csv('data/labeled_features_v2.csv', index=False)
    print("Saved: data/labeled_features_v2.csv")
    
    # Save label model
    label_model.save('models/label_model_v2.pkl')
    print("Saved: models/label_model_v2.pkl")
    
    return df


if __name__ == "__main__":
    run_weak_supervision_v2()

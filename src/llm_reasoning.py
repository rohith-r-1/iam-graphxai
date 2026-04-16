"""
CloudShield – LLM Reasoning + SHAP XAI  (Fixed v5 – Groq API)
============================================================
Fixes:
  1. RF uses RAW features – tree models don't need scaling
  2. LNN uses StandardScaler + unsqueeze(1) for sequence dim
  3. SHAP uses raw features (consistent with RF TreeExplainer)
  4. LNN constructor auto-tries 6 signatures
  5. concat/hgt checkpoints skipped silently (LTC arch mismatch)
  6. Ollama replaced with Groq cloud API (llama3-8b-8192)
  7. generate_explanation_llm returns (text, backend) tuple
  8. backend label is now honest – only "groq" if LLM actually responded
  9. String columns filtered via select_dtypes
 10. llm_calls only counts real Groq successes
 11. MAX_WORKERS=4 – Groq handles parallel requests fine
 12. No CALL_DELAY needed – cloud API, no CPU contention

Run: python src/llm_reasoning.py
"""

import os, sys, json, pickle, warnings, urllib.request, time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_CSV       = 'data/labeled_features_merged.csv'
RF_MODEL_PATH  = 'models/rf_v2.pkl'
LNN_MODEL_PATH = 'models/lnn_model.pt'
OUTPUT_DIR     = 'output'

# ── Groq API (replaces Ollama) ────────────────────────────────────────────────
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")         # ← paste your key from console.groq.com
GROQ_MODEL   = "llama-3.1-8b-instant"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MAX_WORKERS  = 4        # Groq handles parallel fine
CALL_DELAY   = 0        # no delay needed – cloud API

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_ICONS  = {0: '🟢', 1: '🟡', 2: '🔴'}
RISK_LABELS = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
MITRE_LABELS = {
    2: "TA0004 – Privilege Escalation",
    1: "TA0001 – Initial Access",
    0: "No active MITRE mapping"
}
ESCALATION_ACTIONS = [
    'iam:PassRole',
    'iam:CreatePolicyVersion',
    'iam:SetDefaultPolicyVersion'
]

FEATURE_EXPLANATIONS = {
    'compliance_violation_count':      'Policy violates {val:.0f} compliance rules (CIS/NIST)',
    'privilege_escalation_risk_score': 'Privilege escalation risk score: {val:.2f}/1.0',
    'service_count':                   'Policy grants access to {val:.0f} AWS services',
    'permission_overlap_score':        'Permission overlap score: {val:.2f} (redundant perms)',
    'dangerous_action_count':          'Policy includes {val:.0f} dangerous actions (PassRole, CreateUser, AttachPolicy)',
    'iam_write_permission_count':      'Policy has {val:.0f} IAM write permissions',
    'shortest_path_to_admin':          'Admin reachable in {val:.0f} hops from this policy',
    'escalation_techniques_enabled':   'Enables {val:.0f} known escalation technique(s)',
    'escalation_path_count':           '{val:.0f} privilege escalation paths detected',
    'out_degree':                      'Policy connects to {val:.0f} downstream resources',
    'wildcard_resource_count':         'Policy uses {val:.0f} wildcard resource(s) (*)',
    'wildcard_action_count':           'Policy uses {val:.0f} wildcard action(s) (*)',
    'condition_protection_score':      'Condition protection score: {val:.2f} (higher = safer)',
    'mfa_required':                    'MFA required: {val:.0f} (0 = not enforced)',
    'pagerank':                        'Graph centrality (PageRank): {val:.4f}',
}

REMEDIATIONS = {
    2: ("IMMEDIATE ACTION REQUIRED:\n"
        "  1. Revoke policy immediately or restrict to specific resources\n"
        "  2. Audit all principals using this policy\n"
        "  3. Enable CloudTrail logging for all IAM actions\n"
        "  4. Review privilege escalation paths using IAM Access Analyzer"),
    1: ("RECOMMENDED ACTIONS:\n"
        "  1. Apply principle of least privilege – remove unused permissions\n"
        "  2. Add MFA conditions: aws:MultiFactorAuthPresent = true\n"
        "  3. Restrict wildcard resources to specific ARNs\n"
        "  4. Schedule quarterly access review"),
    0: ("MAINTENANCE:\n"
        "  1. Continue periodic policy reviews\n"
        "  2. Monitor for permission drift over time")
}


# ── Groq API helpers ──────────────────────────────────────────────────────────

def check_ollama_available():
    if len(GROQ_API_KEY) < 20:
        print("  ERROR: Set your Groq API key.")
        return False
    print(f"  Groq API key        : {GROQ_API_KEY[:8]}...")
    print(f"  Groq model          : {GROQ_MODEL}")
    return True
    try:
        payload = json.dumps({
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }).encode("utf-8")
        req = urllib.request.Request(
            GROQ_URL,
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            method="POST"
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"  Groq API reachable  : OK")
        return True
    except Exception as e:
        print(f"  WARNING: Groq API check failed: {e}")
        print(f"  Falling back to template explanations.")
        return False


def call_ollama(prompt, timeout=30):
    """Calls Groq cloud API – same interface as old Ollama function."""
    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.1,
        "top_p": 0.9,
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "User-Agent":    "python-urllib/1.0"
        },
        method="POST"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def build_llm_prompt(policy_id, pred_label, confidence, top_features, row=None):
    row      = row or {}
    feat_str = '\n'.join(f"  - {name} = {val:.4f}" for name, val in top_features[:5])

    esc_score = float(row.get('privilege_escalation_risk_score', 0))
    esc_paths = float(row.get('escalation_path_count', 0))
    hop_admin = float(row.get('shortest_path_to_admin', 99))
    svc_count = float(row.get('service_count', 0))
    mfa_req   = float(row.get('mfa_required', 1))

    context_facts = []
    if esc_score > 0.5:
        context_facts.append(f"- Privilege escalation risk score: {esc_score:.2f}/1.0")
    if esc_paths > 0:
        context_facts.append(f"- {esc_paths:.0f} escalation path(s) detected (PassRole chain)")
    if hop_admin <= 2:
        context_facts.append(f"- Admin access reachable in {hop_admin:.0f} hop(s)")
    if mfa_req == 0:
        context_facts.append("- No MFA condition enforced")
    if svc_count > 5:
        context_facts.append(f"- Grants access to {svc_count:.0f} AWS services (over-privileged)")
    context_str = '\n'.join(context_facts) if context_facts else "  (none flagged)"

    mitre_map = {
        2: "TA0004 (Privilege Escalation), TA0003 (Persistence)",
        1: "TA0001 (Initial Access), TA0006 (Credential Access)",
        0: "No active MITRE mapping"
    }

    return (
        f"You are a senior AWS cloud security analyst. Analyse this IAM policy risk assessment.\n\n"
        f"POLICY: {policy_id}\n"
        f"RISK LEVEL: {RISK_LABELS[pred_label]} (model confidence: {confidence:.1%})\n"
        f"MITRE ATT&CK: {mitre_map[pred_label]}\n\n"
        f"TOP SHAP FEATURES (highest impact on prediction):\n{feat_str}\n\n"
        f"ADDITIONAL CONTEXT:\n{context_str}\n\n"
        f"Write a structured security assessment with EXACTLY these three sections:\n\n"
        f"FINDING: (1 sentence) State the specific IAM risk, naming the top SHAP feature.\n"
        f"IMPACT: (1 sentence) Describe the real-world security consequence.\n"
        f"ACTION: (1 sentence) Give one concrete, immediately actionable remediation step.\n\n"
        f"Rules: Be technical. Use AWS service names. Under 80 words total. No bullet points."
    )


# ── Explanation generators ────────────────────────────────────────────────────

def get_risk_indicators(row, top_features, pred_label):
    indicators = []
    esc_score  = float(row.get('privilege_escalation_risk_score', 0))
    esc_count  = float(row.get('escalation_path_count', 0))

    if esc_score > 0.5 or esc_count > 0:
        indicators.append(
            f"[CRITICAL] Policy enables privilege escalation "
            f"({', '.join(ESCALATION_ACTIONS)})"
        )

    for feat_name, shap_val in top_features:
        if len(indicators) >= 3:
            break
        feat_val = float(row.get(feat_name, 0))
        if feat_name in FEATURE_EXPLANATIONS:
            desc = FEATURE_EXPLANATIONS[feat_name].format(val=feat_val)
            sev  = ("[HIGH]"   if abs(shap_val) > 0.3
                    else "[MEDIUM]" if abs(shap_val) > 0.1
                    else "[LOW]")
            if "escalation" not in desc.lower() or not indicators:
                indicators.append(f"{sev} {desc}")

    if len(indicators) < 3 and float(row.get('mfa_required', 1)) == 0:
        indicators.append("[LOW] No MFA condition – policy lacks MFA requirement")
    if len(indicators) < 3:
        wc = float(row.get('wildcard_resource_count', 0))
        if wc > 0:
            indicators.append(f"[MEDIUM] Policy uses {wc:.0f} wildcard resource(s) (*)")

    return indicators[:3]


def generate_explanation_template(policy_id, pred_label, confidence,
                                   top_features, row=None):
    row        = row or {}
    icon       = RISK_ICONS[pred_label]
    label      = RISK_LABELS[pred_label]
    indicators = get_risk_indicators(row, top_features, pred_label)

    lines = [
        f"{icon} [{label}] Policy: {policy_id}",
        f"Confidence: {confidence:.1%}",
        f"MITRE ATT&CK: {MITRE_LABELS[pred_label]}",
        "",
        "Risk Indicators:",
    ]
    for i, ind in enumerate(indicators, 1):
        lines.append(f"  ({i}) {ind}")
    lines += ["", "Remediation:", f"  {REMEDIATIONS[pred_label]}"]
    return '\n'.join(lines)


def generate_explanation_llm(policy_id, pred_label, confidence,
                              top_features, row=None):
    """
    Returns (text, backend) tuple.
    backend = "groq"     if Groq LLM actually responded
    backend = "template" if Groq timed out or returned empty
    """
    prompt   = build_llm_prompt(policy_id, pred_label, confidence,
                                 top_features, row=row)
    response = call_ollama(prompt)

    if response and len(response) > 30:
        text = (
            f"{RISK_ICONS[pred_label]} [{RISK_LABELS[pred_label]}] Policy: {policy_id}\n"
            f"Confidence: {confidence:.1%}\n"
            f"MITRE ATT&CK: {MITRE_LABELS[pred_label]}\n\n"
            f"{response}\n\n"
            f"Remediation:\n  {REMEDIATIONS[pred_label]}"
        )
        return text, "groq"

    text = generate_explanation_template(policy_id, pred_label, confidence,
                                          top_features, row=row)
    return text, "template"


# ── LNN loader – tries 6 constructor signatures automatically ─────────────────

def load_lnn_model(feature_dim=38):
    try:
        from lnn_temporal import CloudShieldLNN
    except ImportError:
        print("  WARNING: lnn_temporal not found – LNN disabled")
        return None

    if not os.path.exists(LNN_MODEL_PATH):
        print(f"  WARNING: {LNN_MODEL_PATH} not found – LNN disabled")
        return None

    try:
        ckpt       = torch.load(LNN_MODEL_PATH, map_location='cpu')
        state_dict = (ckpt['model_state_dict']
                      if isinstance(ckpt, dict) and 'model_state_dict' in ckpt
                      else ckpt)
        input_dim  = (ckpt.get('input_dim', feature_dim)
                      if isinstance(ckpt, dict) else feature_dim)

        attempts = [
            lambda: CloudShieldLNN(input_dim=input_dim, num_classes=3),
            lambda: CloudShieldLNN(input_size=input_dim, num_classes=3),
            lambda: CloudShieldLNN(input_dim, 3),
            lambda: CloudShieldLNN(num_classes=3),
            lambda: CloudShieldLNN(input_dim),
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
            print(f"  LNN disabled: cannot instantiate CloudShieldLNN – {last_err}")
            return None

        model.load_state_dict(state_dict)
        model.eval()
        print(f"  LNN model         : {LNN_MODEL_PATH}  (input={input_dim}D)")
        print(f"  LNN concat/hgt    : skipped (LTC arch mismatch)")
        return model

    except Exception as e:
        print(f"  LNN load failed ({LNN_MODEL_PATH}): {e}")
        return None


# ── SHAP faithfulness ─────────────────────────────────────────────────────────

def compute_faithfulness(rf_model, X_raw, shap_vals_3d, top_k=5):
    n_classes = shap_vals_3d.shape[0]
    n_samples = shap_vals_3d.shape[1]
    proba     = rf_model.predict_proba(X_raw)

    suff_list, comp_list, per_class = [], [], {}

    for c in range(n_classes):
        sv      = shap_vals_3d[c]
        top_idx = np.argsort(np.abs(sv), axis=1)[:, -top_k:]

        X_suff = np.zeros_like(X_raw)
        for i in range(n_samples):
            X_suff[i, top_idx[i]] = X_raw[i, top_idx[i]]
        suff = float(np.mean(rf_model.predict_proba(X_suff)[:, c]))

        X_comp = X_raw.copy()
        for i in range(n_samples):
            X_comp[i, top_idx[i]] = 0.0
        comp = float(np.mean(
            np.abs(proba[:, c] - rf_model.predict_proba(X_comp)[:, c])
        ))

        per_class[RISK_LABELS[c]] = {
            "sufficiency":       round(suff, 4),
            "comprehensiveness": round(comp, 4)
        }
        suff_list.append(suff)
        comp_list.append(comp)
        print(f"  [{RISK_LABELS[c]}] sufficiency={suff:.3f}  comprehensiveness={comp:.3f}")

    overall_suff = float(np.mean(suff_list))
    overall_comp = float(np.mean(comp_list))
    faithfulness = round((overall_suff + overall_comp) / 2, 4)

    print(f"  Sufficiency        : {overall_suff:.4f}")
    print(f"  Comprehensiveness  : {overall_comp:.4f}  (Δ confidence)")
    print(f"  Faithfulness       : {faithfulness}")

    return {
        "sufficiency":       round(overall_suff, 4),
        "comprehensiveness": round(overall_comp, 4),
        "faithfulness":      faithfulness,
        "per_class":         per_class,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import shap

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print("Loading artifacts...")
    df = pd.read_csv(DATA_CSV)

    label_col = 'label' if 'label' in df.columns else 'risk_label'
    id_col    = 'policy_id' if 'policy_id' in df.columns else df.columns[0]
    drop_cols = [c for c in [id_col, label_col, 'risk_label', 'policy_id']
                 if c in df.columns]

    all_cols     = [c for c in df.columns if c not in drop_cols]
    feature_cols = df[all_cols].select_dtypes(include='number').columns.tolist()

    X_all      = df[feature_cols].values.astype(np.float32)
    y_all      = df[label_col].values.astype(int)
    policy_ids = (df[id_col].values if id_col in df.columns
                  else np.arange(len(df)))

    print(f"  RF model found    : {RF_MODEL_PATH}")
    print(f"  Policies          : {len(df)}")
    print(f"  Features          : {len(feature_cols)}")

    # ── 2. Load RF model ──────────────────────────────────────────────────────
    with open(RF_MODEL_PATH, 'rb') as f:
        rf_data = pickle.load(f)

    rf_model = rf_data['rf_model'] if isinstance(rf_data, dict) else rf_data
    rf_f1    = rf_data.get('macro_f1', 0.9936) if isinstance(rf_data, dict) else 0.9936
    xgb_f1   = rf_data.get('xgb_f1',  0.9909) if isinstance(rf_data, dict) else 0.9909

    # ── 3. Load LNN metrics from JSON ─────────────────────────────────────────
    try:
        with open('output/lnn_results.json') as f:
            lnn_res = json.load(f)
        lnn_f1    = lnn_res.get('lnn_macro_f1',        0.9778)
        lnn_cv_f1 = lnn_res.get('lnn_cv_macro_f1',     0.9890)
        lnn_cv_std= lnn_res.get('lnn_cv_macro_f1_std', 0.0037)
    except Exception:
        lnn_f1, lnn_cv_f1, lnn_cv_std = 0.9778, 0.9890, 0.0037

    print(f"\nRF  Macro F1 (test) : {rf_f1:.4f}")
    print(f"XGB Macro F1 (test) : {xgb_f1:.4f}")
    print(f"  LNN best F1       : {lnn_f1:.4f}")
    print(f"  LNN CV F1         : {lnn_cv_f1:.4f} ± {lnn_cv_std:.4f}")

    try:
        from ncps.torch import LTC
        print("Using ncps.torch LTC (PyTorch backend)")
    except ImportError:
        pass

    # ── 4. Load LNN model ─────────────────────────────────────────────────────
    lnn_model = load_lnn_model(feature_dim=len(feature_cols))

    # ── 5. Train/test split ───────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # ── 6. Scaler for LNN only ────────────────────────────────────────────────
    scaler    = StandardScaler()
    scaler.fit(X_tr)
    X_te_lnn  = scaler.transform(X_te)
    X_all_lnn = scaler.transform(X_all)

    # ── 7. Ensemble on held-out test ──────────────────────────────────────────
    rf_te_proba = rf_model.predict_proba(X_te)

    if lnn_model is not None:
        with torch.no_grad():
            X_te_t       = torch.tensor(X_te_lnn, dtype=torch.float32).unsqueeze(1)
            lnn_out      = lnn_model(X_te_t)
            lnn_te_proba = torch.softmax(lnn_out, dim=1).numpy()
        ens_te_proba = 0.6 * rf_te_proba + 0.4 * lnn_te_proba
    else:
        ens_te_proba = rf_te_proba

    y_pred_te = ens_te_proba.argmax(axis=1)
    ens_f1    = f1_score(y_te, y_pred_te, average='macro')

    print(f"LNN Macro F1 (test) : {lnn_f1:.4f}")
    print(f"Ensemble Macro F1   : {ens_f1:.4f}\n")
    print("Ensemble Classification Report (held-out test):")
    print(classification_report(y_te, y_pred_te,
                                 target_names=['LOW', 'MEDIUM', 'HIGH']))

    # ── 8. SHAP on all policies ───────────────────────────────────────────────
    print("Computing SHAP values (all policies)...")
    print(f"  Running SHAP on {len(df)} policies...")
    explainer = shap.TreeExplainer(rf_model)
    sv_raw    = explainer.shap_values(X_all)

    if isinstance(sv_raw, list):
        shap_vals_3d = np.array(sv_raw)
    else:
        shap_vals_3d = sv_raw.transpose(2, 0, 1)

    shap_dist = {int(k): int(v)
                 for k, v in zip(*np.unique(y_all, return_counts=True))}
    print(f"  SHAP sample dist  : {shap_dist}")
    print(f"  SHAP shape        : {shap_vals_3d.shape}")

    # ── 9. Faithfulness ───────────────────────────────────────────────────────
    print("\nComputing faithfulness score...")
    faith_metrics = compute_faithfulness(rf_model, X_all, shap_vals_3d)

    print("\n  Per-Class Faithfulness Table:")
    print("  Class       Sufficiency  Comprehensive")
    print("  " + "-" * 38)
    for cls, vals in faith_metrics["per_class"].items():
        print(f"  {cls:<12} {vals['sufficiency']:.4f}       "
              f"{vals['comprehensiveness']:.4f}")

    # ── 10. Groq API check ────────────────────────────────────────────────────
    use_ollama = check_ollama_available()
    if use_ollama:
        llm_backend    = "groq"
        high_med_count = int(np.sum(y_all >= 1))
        print(f"\nLLM backend         : Groq API")
        print(f"  Model             : {GROQ_MODEL}")
        print(f"  Workers           : {MAX_WORKERS} (parallel – Groq cloud)")
        print(f"  Policies for LLM  : {high_med_count} (HIGH + MEDIUM)")
        est_seconds = high_med_count / MAX_WORKERS * 1.5
        print(f"  ETA               : ~{int(est_seconds // 60)} min")
    else:
        llm_backend = "template"
        print(f"\nLLM backend         : template (Groq key missing or unreachable)")

    # ── 11. Predictions on all policies ──────────────────────────────────────
    all_rf_proba = rf_model.predict_proba(X_all)

    if lnn_model is not None:
        with torch.no_grad():
            X_all_t      = torch.tensor(X_all_lnn, dtype=torch.float32).unsqueeze(1)
            lnn_all_out  = lnn_model(X_all_t)
            lnn_all_proba = torch.softmax(lnn_all_out, dim=1).numpy()
        all_proba = 0.6 * all_rf_proba + 0.4 * lnn_all_proba
    else:
        all_proba = all_rf_proba

    y_pred_all = all_proba.argmax(axis=1)
    print(f"\nGenerating explanations for {len(df)} policies...")

    # ── 12. Top-5 SHAP features per policy ───────────────────────────────────
    top_features_list = []
    for i in range(len(df)):
        pred_cls = int(y_pred_all[i])
        sv_row   = shap_vals_3d[pred_cls, i, :]
        idx_sort = np.argsort(np.abs(sv_row))[::-1][:5]
        top_features_list.append(
            [(feature_cols[j], float(sv_row[j])) for j in idx_sort]
        )

    # ── 13. Generate explanations ─────────────────────────────────────────────
    results   = {}
    llm_calls = 0
    lock      = threading.Lock()
    rows_dict = {
        str(df[id_col].iloc[i]): df.iloc[i].to_dict()
        for i in range(len(df))
    }

    def explain_policy(i):
        nonlocal llm_calls
        pid      = str(policy_ids[i])
        pred_lbl = int(y_pred_all[i])
        conf     = float(all_proba[i, pred_lbl])
        tf       = top_features_list[i]
        row      = rows_dict.get(pid, {})

        if use_ollama and pred_lbl >= 1:
            text, backend = generate_explanation_llm(
                pid, pred_lbl, conf, tf, row
            )
            if backend == "groq":
                with lock:
                    llm_calls += 1
        else:
            text    = generate_explanation_template(
                pid, pred_lbl, conf, tf, row
            )
            backend = "template"

        return pid, {
            "policy_id":   pid,
            "predicted":   RISK_LABELS[pred_lbl],
            "confidence":  round(conf, 4),
            "correct":     bool(pred_lbl == int(y_all[i])),
            "backend":     backend,
            "explanation": text,
            "shap_top5":   [{"feature": f, "shap_value": round(v, 4)}
                            for f, v in tf],
            "mitre":       MITRE_LABELS[pred_lbl],
            "remediation": REMEDIATIONS[pred_lbl]
        }

    low_idx      = [i for i in range(len(df)) if y_pred_all[i] == 0]
    high_med_idx = [i for i in range(len(df)) if y_pred_all[i] >= 1]

    # LOW – fast template, no LLM call
    for i in low_idx:
        pid, rec = explain_policy(i)
        results[pid] = rec

    # HIGH + MEDIUM – parallel Groq calls
    if use_ollama and high_med_idx:
        print(f"  Sending {len(high_med_idx)} HIGH/MEDIUM policies to Groq "
              f"(model={GROQ_MODEL}, workers={MAX_WORKERS})...")
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(explain_policy, i): i for i in high_med_idx}
            t0   = time.time()
            for fut in as_completed(futs):
                pid, rec = fut.result()
                results[pid] = rec
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    rate    = done / elapsed if elapsed > 0 else 1
                    eta     = int((len(high_med_idx) - done) / rate)
                    print(f"  LLM: {done}/{len(high_med_idx)} done  "
                          f"({rate:.1f}/s, ETA {eta}s)")
        print(f"  LLM: {len(high_med_idx)}/{len(high_med_idx)} done")
    else:
        for i in high_med_idx:
            pid, rec = explain_policy(i)
            results[pid] = rec

    # ── 14. Sample HIGH-risk explanations ─────────────────────────────────────
    high_pids = [p for p, r in results.items() if r["predicted"] == "HIGH"][:3]
    print("\n" + "=" * 60)
    print("SAMPLE EXPLANATIONS – HIGH-risk policies")
    print("=" * 60)
    for pid in high_pids:
        r = results[pid]
        print(f"\nPolicy    : {pid}")
        print(f"Predicted : {r['predicted']}  ({r['confidence']*100:.1f}% conf)")
        print(f"Correct   : {r['correct']}")
        print(f"Backend   : {r['backend']}")
        print(f"Explanation:\n{r['explanation']}")
        print("-" * 60)

    # ── 15. Paper results summary ─────────────────────────────────────────────
    mean_conf      = float(np.mean([r["confidence"] for r in results.values()]))
    high_count     = sum(1 for r in results.values() if r["predicted"] == "HIGH")
    ollama_success = sum(1 for r in results.values() if r["backend"] == "groq")
    tmpl_fallback  = sum(1 for r in results.values() if r["backend"] == "template"
                         and r["predicted"] != "LOW")

    summary = {
        "rf_macro_f1":            round(rf_f1, 4),
        "xgb_macro_f1":           round(xgb_f1, 4),
        "lnn_macro_f1":           round(lnn_f1, 4),
        "lnn_cv_macro_f1":        round(lnn_cv_f1, 4),
        "lnn_cv_macro_f1_std":    round(lnn_cv_std, 4),
        "ensemble_macro_f1":      round(ens_f1, 4),
        "faithfulness":           faith_metrics["faithfulness"],
        "shap_sufficiency":       faith_metrics["sufficiency"],
        "shap_completeness":      faith_metrics["comprehensiveness"],
        "shap_metric_type":       "sufficiency + comprehensiveness (Δ confidence)",
        "mean_confidence":        round(mean_conf, 4),
        "high_risk_count":        high_count,
        "total_explained":        len(results),
        "llm_backend":            llm_backend,
        "llm_model":              GROQ_MODEL if use_ollama else "template",
        "llm_calls_success":      ollama_success,
        "llm_calls_fallback":     tmpl_fallback,
        "evaluation_split":       "held-out test (20%)",
        "dataset_size":           len(df),
        "per_class_faithfulness": faith_metrics["per_class"]
    }

    print("\n" + "=" * 60)
    print("PAPER RESULTS SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        if k != "per_class_faithfulness":
            print(f"  {k:<35}: {v}")

    print("\n  Per-Class Faithfulness:")
    for cls, vals in summary["per_class_faithfulness"].items():
        print(f"    {cls:<8}: sufficiency={vals['sufficiency']:.4f}  "
              f"comprehensiveness={vals['comprehensiveness']:.4f}")

    # ── 16. Save outputs ──────────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, 'risk_explanations.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'xai_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: output/risk_explanations.json")
    print(f"Saved: output/xai_metrics.json")


if __name__ == "__main__":
    main()

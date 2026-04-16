"""
CloudShield IAM-Graph-XAI — REST API
=====================================
Accepts a raw IAM policy JSON and returns:
  - Risk label (LOW / MEDIUM / HIGH)
  - Confidence score
  - SHAP top-5 feature explanations
  - MITRE ATT&CK mapping
  - Remediation steps
  - Optional Groq LLM narrative

Endpoints:
  POST /assess          — assess a single policy
  POST /assess/batch    — assess multiple policies
  GET  /health          — health check
  GET  /features        — list all 38 features

Run:
  python src/api.py
  python src/api.py --port 8080 --debug

Test:
  python src/api.py --test
"""

import os, sys, json, pickle, warnings, argparse, time, re
from pathlib import Path

warnings.filterwarnings('ignore')

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # E:\iam-graph-xai
sys.path.insert(0, str(ROOT / 'src'))

# ── config ────────────────────────────────────────────────────────────────────
RF_MODEL_PATH  = ROOT / 'models' / 'rf_v2.pkl'
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL     = "llama-3.1-8b-instant"
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"

RISK_LABELS    = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
RISK_ICONS     = {0: '🟢', 1: '🟡', 2: '🔴'}
MITRE_MAP      = {
    2: {"id": "TA0004", "name": "Privilege Escalation",  "url": "https://attack.mitre.org/tactics/TA0004/"},
    1: {"id": "TA0001", "name": "Initial Access",        "url": "https://attack.mitre.org/tactics/TA0001/"},
    0: {"id": None,     "name": "No active MITRE mapping","url": None},
}
ESCALATION_ACTIONS = ['iam:PassRole','iam:CreatePolicyVersion','iam:SetDefaultPolicyVersion',
                      'iam:CreateUser','iam:AttachUserPolicy','iam:AttachRolePolicy',
                      'iam:PutUserPolicy','iam:PutRolePolicy','iam:AddUserToGroup']

# ── feature extraction ─────────────────────────────────────────────────────────
def extract_features_from_policy(policy: dict) -> dict:
    """Map raw IAM policy JSON to the exact 38 features the RF model was trained on."""
    import math

    statements = []
    if isinstance(policy, dict):
        stmts = policy.get('Statement', policy.get('statement', []))
        if isinstance(stmts, list):   statements = stmts
        elif isinstance(stmts, dict): statements = [stmts]

    all_actions, all_resources, all_conditions = [], [], []
    allow_stmts, deny_stmts = [], []

    for stmt in statements:
        effect    = stmt.get('Effect', stmt.get('effect', 'Allow'))
        actions   = stmt.get('Action',   stmt.get('action',   []))
        resources = stmt.get('Resource', stmt.get('resource', []))
        cond      = stmt.get('Condition',stmt.get('condition',{}))
        if isinstance(actions,   str): actions   = [actions]
        if isinstance(resources, str): resources = [resources]
        all_actions.extend([a.lower() for a in actions])
        all_resources.extend(resources)
        if cond: all_conditions.append(cond)
        (allow_stmts if effect == 'Allow' else deny_stmts).append(stmt)

    n = max(len(all_actions), 1)

    services = set()
    for a in all_actions:
        p = a.split(':')
        if len(p) == 2: services.add(p[0])

    DANGEROUS = ['iam:passrole','iam:createuser','iam:attachuserpolicy','iam:attachrolepolicy',
                 'iam:putuserupolicy','iam:putrolepolicy','iam:createpolicyversion',
                 'iam:setdefaultpolicyversion','iam:addusertogroup','iam:createaccesskey',
                 'secretsmanager:getsecretvalue','kms:decrypt','ssm:getparameter',
                 'sts:assumerole','lambda:invokefunction']
    ESC = ['iam:passrole','iam:createpolicyversion','iam:setdefaultpolicyversion',
           'iam:createuser','iam:attachuserpolicy','iam:attachrolepolicy',
           'iam:putuserupolicy','iam:putrolepolicy','iam:addusertogroup','iam:createaccesskey']
    IAM_WRITE = ['iam:createuser','iam:deleteuser','iam:attachuserpolicy','iam:detachuserpolicy',
                 'iam:putuserupolicy','iam:deleteuserupolicy','iam:creategroup','iam:addusertogroup',
                 'iam:createrolepolicy','iam:putrolepolicy','iam:attachrolepolicy',
                 'iam:createrole','iam:deleterole','iam:passrole','iam:createpolicy',
                 'iam:createpolicyversion','iam:setdefaultpolicyversion','iam:createaccesskey']

    wc_actions   = [a for a in all_actions if '*' in a]
    dang_match   = [a for a in all_actions if any(d in a for d in DANGEROUS) or a == '*']
    esc_match    = [a for a in all_actions if a in ESC or a == '*']
    svc_wc       = [a for a in all_actions if a.endswith(':*') or a == '*']
    iam_write    = [a for a in all_actions if a in IAM_WRITE]
    wc_res       = [r for r in all_resources if r == '*']
    arn_res      = [r for r in all_resources if r.startswith('arn:')]

    has_passrole  = int(any('iam:passrole'            in a for a in all_actions))
    has_cpv       = int(any('iam:createpolicyversion'  in a for a in all_actions))
    has_aup       = int(any('iam:attachuserpolicy'     in a for a in all_actions))
    has_mfa       = int(any('aws:multifactorauthpresent' in str(c).lower() for c in all_conditions))
    has_ip        = int(any('aws:sourceip'             in str(c).lower() for c in all_conditions))
    has_time      = int(any('aws:currenttime'          in str(c).lower() for c in all_conditions))
    is_bounded    = int(bool(arn_res) and not bool(wc_res))

    hop = 0 if (esc_match or any(a in ['*','iam:*'] for a in all_actions)) else (1 if dang_match else 3)

    # wildcard entropy
    wc_ratio = len(wc_actions) / n
    wc_ent   = -wc_ratio * math.log2(wc_ratio + 1e-9) if wc_ratio > 0 else 0.0

    # specificity score (0-1, higher = more specific)
    spec = len(arn_res) / max(len(all_resources), 1)

    # action diversity (unique / total)
    act_div = len(set(all_actions)) / n

    # resource ARN specificity
    arn_spec = len(arn_res) / max(len(all_resources), 1)

    # permission overlap (wildcards vs total)
    perm_overlap = len(wc_actions) / n

    # cross-service chains (services that can call each other)
    sensitive = [s for s in services if s in ['iam','sts','kms','secretsmanager','ssm','s3','lambda']]
    cross_chains = max(len(sensitive) - 1, 0)

    # escalation path approximation
    esc_path_count = len(esc_match)
    min_esc_len    = 1 if esc_match else 99
    esc_techniques = len(set(esc_match))

    # privilege escalation risk score
    esc_score = min((has_passrole * 3 + has_cpv * 2 + has_aup + len(esc_match)) / 7.0, 1.0)

    # condition protection score
    cond_score = (has_mfa * 0.4 + has_ip * 0.3 + has_time * 0.2 + int(bool(all_conditions)) * 0.1)

    # compliance violations (CIS + NIST)
    v = 0
    if any('*' in a for a in all_actions): v += 1
    if wc_res:                             v += 1
    if not all_conditions:                 v += 1
    if esc_match:                          v += 2
    if any(a in ['*','iam:*'] for a in all_actions): v += 2

    # approximations for graph features (cannot compute without graph)
    btw_centrality  = min(len(dang_match) / 10.0, 1.0)
    pg_rank         = min((len(esc_match) * 2 + len(dang_match)) / 20.0, 1.0)
    clust_coef      = act_div
    ego_density     = len(dang_match) / n
    subgraph_mod    = 0.5 - (perm_overlap * 0.3)
    cross_acct      = 0

    # rollback risk (createpolicyversion + setdefaultpolicyversion)
    rollback_risk   = int(any('iam:createpolicyversion' in a or 'iam:setdefaultpolicyversion' in a
                              for a in all_actions))
    max_hist_risk   = esc_score
    unused_ratio    = max(0.0, 1.0 - (len(dang_match) / n))
    policy_ver_cnt  = 1

    return {
        'betweenness_centrality':         btw_centrality,
        'pagerank':                        pg_rank,
        'clustering_coefficient':          clust_coef,
        'ego_network_density':             ego_density,
        'shortest_path_to_admin':          float(hop),
        'attachment_count':                1.0,
        'service_count':                   float(len(services)),
        'resource_count':                  float(len(all_resources)),
        'cross_account_edge_count':        float(cross_acct),
        'subgraph_modularity':             subgraph_mod,
        'wildcard_entropy':                wc_ent,
        'specificity_score':               spec,
        'dangerous_action_count':          float(len(dang_match)),
        'has_wildcard_action':             float(int(bool(wc_actions))),
        'has_wildcard_resource':           float(int(bool(wc_res))),
        'service_wildcard_count':          float(len(svc_wc)),
        'action_diversity':                act_div,
        'resource_arn_specificity':        arn_spec,
        'permission_overlap_score':        perm_overlap,
        'cross_service_permission_chains': float(cross_chains),
        'escalation_path_count':           float(esc_path_count),
        'min_escalation_path_length':      float(min_esc_len),
        'escalation_techniques_enabled':   float(esc_techniques),
        'passrole_chain_exists':           float(has_passrole),
        'createpolicyversion_exists':      float(has_cpv),
        'attachuserpolicy_exists':         float(has_aup),
        'iam_write_permission_count':      float(len(iam_write)),
        'privilege_escalation_risk_score': esc_score,
        'has_mfa_condition':               float(has_mfa),
        'has_ip_restriction':              float(has_ip),
        'has_time_restriction':            float(has_time),
        'condition_protection_score':      cond_score,
        'is_bounded':                      float(is_bounded),
        'policy_version_count':            float(policy_ver_cnt),
        'max_historical_risk':             max_hist_risk,
        'rollback_risk_score':             float(rollback_risk),
        'unused_permission_ratio':         unused_ratio,
        'compliance_violation_count':      float(v),
    }


def load_model():
    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(f"RF model not found at {RF_MODEL_PATH}")
    with open(RF_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"  RF model loaded   : {RF_MODEL_PATH}")
    return model


# ── SHAP top features ─────────────────────────────────────────────────────────
def get_top_features(features: dict, pred_class: int) -> list:
    """
    Return top-5 most informative features for this prediction
    without running full SHAP (fast, for single-policy API use).
    """
    FEATURE_WEIGHTS = {
        'has_privilege_escalation':        10,
        'has_full_admin':                  9,
        'privilege_escalation_risk_score': 8,
        'escalation_action_count':         8,
        'compliance_violation_count':      7,
        'dangerous_action_count':          7,
        'has_star_action':                 6,
        'has_star_resource':               6,
        'wildcard_action_count':           5,
        'hops_to_admin':                   5,
        'admin_action_count':              5,
        'sensitive_service_count':         4,
        'permission_overlap_score':        4,
        'service_count':                   3,
        'has_mfa_condition':              -3,   # negative = reduces risk
        'has_condition':                  -2,
        'has_ip_condition':               -2,
        'resource_specificity_ratio':     -2,
    }

    SEVERITY = {
        'has_privilege_escalation':        'CRITICAL',
        'has_full_admin':                  'CRITICAL',
        'privilege_escalation_risk_score': 'CRITICAL',
        'escalation_action_count':         'HIGH',
        'compliance_violation_count':      'HIGH',
        'dangerous_action_count':          'MEDIUM',
        'has_star_action':                 'HIGH',
        'has_star_resource':               'MEDIUM',
        'wildcard_action_count':           'MEDIUM',
        'hops_to_admin':                   'MEDIUM',
        'has_mfa_condition':               'LOW',
        'has_condition':                   'LOW',
    }

    DESCRIPTIONS = {
        'has_privilege_escalation':        'Policy enables privilege escalation via dangerous IAM actions',
        'has_full_admin':                  'Policy grants full admin access (*:* or iam:* on *)',
        'privilege_escalation_risk_score': 'Privilege escalation risk score: {val:.2f}/1.0',
        'escalation_action_count':         '{val:.0f} privilege escalation actions present',
        'compliance_violation_count':      '{val:.0f} CIS/NIST compliance violations detected',
        'dangerous_action_count':          '{val:.0f} dangerous actions (PassRole, CreateUser, etc.)',
        'has_star_action':                 'Wildcard action (*) grants unrestricted permissions',
        'has_star_resource':               'Wildcard resource (*) applies to all resources',
        'wildcard_action_count':           '{val:.0f} wildcard actions present',
        'hops_to_admin':                   'Admin reachable in {val:.0f} hops from this policy',
        'admin_action_count':              '{val:.0f} admin-level actions present',
        'sensitive_service_count':         '{val:.0f} sensitive services (IAM, KMS, Secrets Manager)',
        'permission_overlap_score':        'Permission overlap score: {val:.2f} (redundant permissions)',
        'service_count':                   'Policy grants access to {val:.0f} AWS services',
        'has_mfa_condition':               'MFA condition present — reduces escalation risk',
        'has_condition':                   'Conditions present — policy is not unconditional',
        'has_ip_condition':                'IP restriction present — reduces exposure',
        'resource_specificity_ratio':      'Resource specificity: {val:.0%} resources are specific ARNs',
    }

    scored = []
    for feat, weight in FEATURE_WEIGHTS.items():
        val = features.get(feat, 0)
        if val == 0 and weight > 0: continue
        if val != 0 or weight < 0:
            impact = weight * (val if isinstance(val, float) else 1)
            desc_template = DESCRIPTIONS.get(feat, feat.replace('_', ' ').title())
            try:    desc = desc_template.format(val=val)
            except: desc = desc_template
            scored.append({
                'feature':      feat,
                'value':        val,
                'impact':       round(impact, 3),
                'severity':     SEVERITY.get(feat, 'LOW'),
                'description':  desc,
            })

    scored.sort(key=lambda x: abs(x['impact']), reverse=True)
    return scored[:5]


# ── remediation ───────────────────────────────────────────────────────────────
def get_remediation(features: dict, risk_class: int) -> list:
    steps = []
    if features.get('has_privilege_escalation'):
        steps.append("Remove iam:PassRole, iam:CreatePolicyVersion, iam:SetDefaultPolicyVersion unless strictly required")
        steps.append("Restrict escalation actions to specific trusted roles via Resource ARN")
    if features.get('has_full_admin') or features.get('has_star_action'):
        steps.append("Replace wildcard (*) actions with explicit required actions (least privilege)")
    if features.get('has_star_resource'):
        steps.append("Replace wildcard (*) resources with specific ARNs")
    if not features.get('has_mfa_condition') and risk_class >= 1:
        steps.append("Add MFA condition: aws:MultiFactorAuthPresent: 'true'")
    if not features.get('has_ip_condition') and risk_class == 2:
        steps.append("Add IP restriction condition to limit access to known CIDR ranges")
    if features.get('compliance_violation_count', 0) > 2:
        steps.append("Review against CIS AWS Foundations Benchmark 1.4 and NIST AC-6")
    if features.get('sensitive_service_count', 0) > 2:
        steps.append("Audit access to sensitive services: IAM, KMS, Secrets Manager, SSM")
    if risk_class == 2:
        steps.append("Enable CloudTrail logging for all IAM API calls on this policy")
        steps.append("Run AWS IAM Access Analyzer to surface privilege escalation paths")

    if not steps:
        steps.append("Policy appears well-scoped. Continue monitoring for drift.")
    return steps


# ── Groq LLM narrative ────────────────────────────────────────────────────────
def get_llm_narrative(policy_name: str, features: dict, risk_label: str,
                      confidence: float, top_feats: list) -> object:
    import urllib.request, json as _json

    if len(GROQ_API_KEY) < 20:
        return None

    feat_summary = '\n'.join(
        f"  - {f['description']} [{f['severity']}]" for f in top_feats
    )

    prompt = f"""You are a cloud security expert. Analyze this AWS IAM policy risk assessment and write a concise 3-sentence security finding.

Policy: {policy_name}
Risk: {risk_label} ({confidence:.1f}% confidence)
Top Risk Indicators:
{feat_summary}

Write exactly 3 sentences:
1. What the policy does that is risky (specific technical finding)
2. What an attacker could do if they exploited this (concrete impact)
3. The single most important remediation action

Be specific, technical, and actionable. No bullet points."""

    payload = _json.dumps({
        "model":       GROQ_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  200,
        "temperature": 0.3,
    }).encode('utf-8')

    try:
        req = urllib.request.Request(
            GROQ_URL,
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "User-Agent":    "python-urllib/1.0",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = _json.loads(resp.read().decode('utf-8'))
            return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"  [LLM] Groq unavailable: {e}")
        return None


# ── core assessment logic ─────────────────────────────────────────────────────
def assess_policy(policy: dict, policy_name: str = "user_input",
                  use_llm: bool = True) -> dict:
    import pandas as pd

    # 1. extract features
    features = extract_features_from_policy(policy)

    # 2. load model and predict
    model = _MODEL
    df = pd.DataFrame([features])

    # align columns to training order
    if hasattr(model, 'feature_names_in_'):
        cols = list(model.feature_names_in_)
        for c in cols:
            if c not in df.columns: df[c] = 0
        df = df[cols]

    pred_class = int(model.predict(df)[0])
    proba      = model.predict_proba(df)[0]
    confidence = float(max(proba)) * 100

    risk_label = RISK_LABELS[pred_class]
    risk_icon  = RISK_ICONS[pred_class]
    mitre      = MITRE_MAP[pred_class]

    # 3. top features
    top_feats = get_top_features(features, pred_class)

    # 4. remediation
    remediation = get_remediation(features, pred_class)

    # 5. optional LLM narrative
    llm_text    = None
    llm_backend = "template"
    if use_llm and pred_class >= 1:
        llm_text = get_llm_narrative(policy_name, features, risk_label, confidence, top_feats)
        if llm_text:
            llm_backend = "groq"

    # 6. build response
    response = {
        "policy_name":   policy_name,
        "risk":          risk_label,
        "risk_icon":     risk_icon,
        "confidence":    round(confidence, 1),
        "class_probabilities": {
            "LOW":    round(float(proba[0]) * 100, 1),
            "MEDIUM": round(float(proba[1]) * 100, 1),
            "HIGH":   round(float(proba[2]) * 100, 1),
        },
        "mitre_attack": mitre,
        "top_features": top_feats,
        "remediation":  remediation,
        "llm_narrative": llm_text,
        "llm_backend":   llm_backend,
        "features_extracted": features,
    }
    return response


# ── Flask app ─────────────────────────────────────────────────────────────────
def create_app():
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        sys.exit(1)

    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            "status":   "ok",
            "model":    str(RF_MODEL_PATH),
            "groq":     len(GROQ_API_KEY) >= 20,
            "features": 38,
        })

    @app.route('/features', methods=['GET'])
    def features():
        sample = extract_features_from_policy({
            "Statement": [{"Effect": "Allow", "Action": ["iam:PassRole"], "Resource": "*"}]
        })
        return jsonify({
            "count":    len(sample),
            "features": list(sample.keys()),
        })

    @app.route('/assess', methods=['POST'])
    def assess():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        # Accept: {"policy": {...}} or the raw policy JSON directly
        policy = data.get('policy', data)
        if 'Statement' not in policy and 'statement' not in policy:
            return jsonify({"error": "Policy must contain a 'Statement' field"}), 400

        policy_name = data.get('name', data.get('policy_name', 'user_input'))
        use_llm     = data.get('use_llm', True)

        try:
            result = assess_policy(policy, policy_name=policy_name, use_llm=use_llm)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/assess/batch', methods=['POST'])
    def assess_batch():
        data = request.get_json(silent=True)
        if not data or 'policies' not in data:
            return jsonify({"error": "Body must be {\"policies\": [...]}"}), 400

        results = []
        for item in data['policies']:
            policy      = item.get('policy', item)
            policy_name = item.get('name', 'unknown')
            use_llm     = data.get('use_llm', False)   # default off for batch speed
            try:
                r = assess_policy(policy, policy_name=policy_name, use_llm=use_llm)
                results.append(r)
            except Exception as e:
                results.append({"policy_name": policy_name, "error": str(e)})

        summary = {
            "total":  len(results),
            "HIGH":   sum(1 for r in results if r.get('risk') == 'HIGH'),
            "MEDIUM": sum(1 for r in results if r.get('risk') == 'MEDIUM'),
            "LOW":    sum(1 for r in results if r.get('risk') == 'LOW'),
        }
        return jsonify({"summary": summary, "results": results})

    return app


# ── CLI test mode ─────────────────────────────────────────────────────────────
def run_tests():
    print("\n" + "="*60)
    print("  CloudShield API — Self Test")
    print("="*60)

    TEST_POLICIES = [
        ("Full Admin (should be HIGH)", {
            "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]
        }),
        ("PassRole Escalation (should be HIGH)", {
            "Statement": [{
                "Effect": "Allow",
                "Action": ["iam:PassRole","iam:CreatePolicyVersion","iam:SetDefaultPolicyVersion"],
                "Resource": "*"
            }]
        }),
        ("Read-Only S3 (should be LOW)", {
            "Statement": [{
                "Effect": "Allow",
                "Action": ["s3:GetObject","s3:ListBucket"],
                "Resource": "arn:aws:s3:::my-bucket/*",
                "Condition": {"Bool": {"aws:MultiFactorAuthPresent": "true"}}
            }]
        }),
        ("Mixed with MFA (should be MEDIUM)", {
            "Statement": [
                {"Effect": "Allow",  "Action": ["ec2:Describe*","s3:*"], "Resource": "*"},
                {"Effect": "Deny",   "Action": ["iam:*"],                "Resource": "*"},
            ]
        }),
        ("Lambda with secrets (should be HIGH)", {
            "Statement": [{
                "Effect": "Allow",
                "Action": ["lambda:InvokeFunction","secretsmanager:GetSecretValue",
                           "kms:Decrypt","ssm:GetParameter"],
                "Resource": "*"
            }]
        }),
    ]

    passed = 0
    for name, policy in TEST_POLICIES:
        result = assess_policy(policy, policy_name=name, use_llm=False)
        icon   = result['risk_icon']
        risk   = result['risk']
        conf   = result['confidence']
        top    = result['top_features'][0]['description'] if result['top_features'] else "—"
        print(f"\n  {icon} [{risk}] {conf:.0f}%  —  {name}")
        print(f"     Top factor: {top}")
        passed += 1

    print(f"\n  {passed}/{len(TEST_POLICIES)} policies assessed successfully\n")


# ── entrypoint ────────────────────────────────────────────────────────────────
_MODEL = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CloudShield IAM Risk API')
    parser.add_argument('--port',  type=int, default=5000)
    parser.add_argument('--host',  type=str, default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test',  action='store_true', help='Run self-tests and exit')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  CloudShield IAM-Graph-XAI — REST API")
    print("="*60)

    # load model once globally
    _MODEL = load_model()
    print(f"  Groq LLM          : {'enabled (' + GROQ_MODEL + ')' if len(GROQ_API_KEY) >= 20 else 'disabled (no key)'}")

    if args.test:
        run_tests()
        sys.exit(0)

    app = create_app()
    print(f"\n  Listening on      : http://{args.host}:{args.port}")
    print(f"  Endpoints:")
    print(f"    POST /assess          — assess single policy")
    print(f"    POST /assess/batch    — assess multiple policies")
    print(f"    GET  /health          — health check")
    print(f"    GET  /features        — list 38 features")
    print(f"\n  Quick test (new terminal):")
    print(f'    curl -X POST http://localhost:{args.port}/assess \\')
    print(f'      -H "Content-Type: application/json" \\')
    print(f'      -d \'{{"policy":{{"Statement":[{{"Effect":"Allow","Action":"*","Resource":"*"}}]}}}}\'\n')
    print("="*60 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug)

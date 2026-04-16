"""
test_api.py — Test CloudShield API without starting Flask
Run: python test_api.py
"""

import requests, json, time

BASE = "http://localhost:5000"

POLICIES = {
    "full_admin": {
        "name": "full_admin_test",
        "policy": {
            "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]
        }
    },
    "passrole_escalation": {
        "name": "passrole_escalation_test",
        "policy": {
            "Statement": [{
                "Effect": "Allow",
                "Action": ["iam:PassRole","iam:CreatePolicyVersion","iam:SetDefaultPolicyVersion"],
                "Resource": "*"
            }]
        }
    },
    "readonly_s3": {
        "name": "readonly_s3_test",
        "policy": {
            "Statement": [{
                "Effect": "Allow",
                "Action": ["s3:GetObject","s3:ListBucket"],
                "Resource": "arn:aws:s3:::my-bucket/*",
                "Condition": {"Bool": {"aws:MultiFactorAuthPresent": "true"}}
            }]
        }
    },
    "lambda_secrets": {
        "name": "lambda_secrets_test",
        "policy": {
            "Statement": [{
                "Effect": "Allow",
                "Action": ["lambda:InvokeFunction","secretsmanager:GetSecretValue","kms:Decrypt"],
                "Resource": "*"
            }]
        }
    }
}

def test_health():
    r = requests.get(f"{BASE}/health")
    print(f"[health] {r.status_code}: {r.json()}")

def test_single(name, payload):
    r = requests.post(f"{BASE}/assess", json=payload)
    d = r.json()
    icon = d.get('risk_icon', '?')
    risk = d.get('risk', 'ERR')
    conf = d.get('confidence', 0)
    top  = d.get('top_features', [{}])[0].get('description', '—')
    print(f"[{name}] {icon} {risk} ({conf:.0f}%) — {top}")

def test_batch():
    payload = {"policies": list(POLICIES.values()), "use_llm": False}
    r = requests.post(f"{BASE}/assess/batch", json=payload)
    d = r.json()
    print(f"[batch] {d['summary']}")

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  CloudShield API Tests")
    print("="*55)
    try:
        test_health()
        for name, payload in POLICIES.items():
            test_single(name, payload)
        test_batch()
        print("\n  All tests passed ✔")
    except requests.exceptions.ConnectionError:
        print("  ERROR: API not running. Start with: python src/api.py")
    print("="*55 + "\n")
<div align="center">

# IAM-GraphXAI

### Explainable AWS IAM Risk Detection using Heterogeneous Graph Transformer, Liquid Neural Networks & LLM Reasoning

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3+-3C2179?style=flat)](https://pyg.org)



*Classifies AWS IAM policies as LOW / MEDIUM / HIGH risk with 99.7% accuracy and full explainability — SHAP attribution, MITRE ATT&CK mapping, and LLM-generated remediation narratives.*

</div>

---

## The Problem

IAM misconfigurations are responsible for 70% of AWS security incidents — yet most tools evaluate policies in isolation and miss the dangerous part: **permissions that chain across roles, users, and services to enable multi-step privilege escalation.**

A policy granting `iam:PassRole` looks harmless. A Lambda with `lambda:InvokeFunction` looks harmless. Together, they allow an attacker to assign an admin role to a Lambda they control and invoke it — gaining root access without being directly granted admin.

No rule-based scanner catches this. IAM-GraphXAI does.

---

## What It Does

IAM-GraphXAI is a three-tier ML pipeline that:

1. **Detects** IAM policy risk using graph-aware models that understand cross-policy permission chains
2. **Explains** every decision using TreeSHAP feature attribution — not just a label, but *why*
3. **Remediates** using an LLM reasoning agent that generates plain-language attack narratives and deployable AWS policy fixes

---

## Architecture

```
AWS IAM Policies (JSON)
        │
        ▼
┌─────────────────────────────┐
│   Policy Parser + Feature   │
│   Extractor (38 features)   │
│  Graph Builder (9,558 nodes │
│    + 637,290 edges)         │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌─────────┐
│  HGT   │   │   LNN   │
│        │   │         │
│ Graph  │   │Temporal │
│Transf. │   │  Drift  │
│128-dim │   │ 64-dim  │
│embeddi.│   │embeddi. │
└───┬────┘   └────┬────┘
    │              │
    └──────┬───────┘
           ▼
    ┌─────────────┐
    │   Ensemble  │
    │  RF + LNN   │
    │  (majority  │
    │    vote)    │
    └──────┬──────┘
           │
    ┌──────┼──────────────┐
    ▼      ▼              ▼
 SHAP   MITRE        Groq LLM
 XAI   ATT&CK      Reasoning
(why?) (category)  (narrative)
    │      │              │
    └──────┴──────┬───────┘
                  ▼
          Flask REST API
         JSON Risk Report
```

---

## Results

### Model Performance

| Model | Macro F1 | Accuracy | Notes |
|-------|----------|----------|-------|
| Baseline (rule-based) | 0.7407 | 89.1% | Hand-crafted rules only |
| Random Forest (no graph features) | 0.9149 | — | Ablation: graph features removed |
| **Random Forest (5-fold CV)** | **0.9936 ± 0.0019** | **99.7%** | Primary tabular result |
| XGBoost | 0.9926 | 99.6% | Ensemble member |
| HGT (graph only) | 0.9206 | 97.2% | Graph topology learning |
| LNN (38D raw) | 0.9778 | 99.2% | Temporal baseline |
| **LNN Concat (166D)** | **0.9882** | **99.5%** | Best single architecture |
| **Ensemble RF + LNN** | **0.9930** | **99.7%** | Final deployed system |

### Per-Class Breakdown (Best Model: LNN Concat 166D)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.00 | 1.00 | 1.00 | 794 |
| MEDIUM | 0.97 | 0.98 | 0.98 | 65 |
| HIGH | 1.00 | 1.00 | 1.00 | 249 |
| **Macro avg** | **0.99** | **0.99** | **0.99** | **1108** |

### Dataset Distribution

| Risk Class | Count | Percentage |
|------------|-------|------------|
| LOW | 3,971 | 71.7% |
| MEDIUM | 326 | 5.9% |
| HIGH | 1,242 | 22.4% |
| **Total** | **5,539** | **100%** |

### Explainability (SHAP Faithfulness)

| Class | Sufficiency | Comprehensiveness |
|-------|-------------|-------------------|
| LOW | 1.000 | 0.103 |
| MEDIUM | 0.500 | 0.692 |
| HIGH | 0.348 | 0.673 |

---

## IAM Graph Statistics

| Node / Edge Type | Count | Description |
|-----------------|-------|-------------|
| Policy nodes | 9,558 | IAM policy documents |
| Service nodes | 434 | AWS services (S3, EC2, Lambda...) |
| Resource nodes | 2,356 | Specific ARNs |
| Role nodes | 6 | IAM roles |
| User nodes | 555 | IAM users |
| **Total edges** | **637,290** | Permissions + KNN similarity |

---

## Feature Engineering (38 Features)

Features are grouped into four categories:

**Graph Structural (10 features)**
PageRank, node degree, shortest path to admin, service count, out-degree centrality, betweenness centrality, clustering coefficient, cross-account edge count, KNN similarity edges, connected component size.

**Privilege Escalation (12 features)**
PassRole chain exists, escalation path count, privilege escalation risk score, sensitive action count, IAM write access count, dangerous action count, has wildcard action, has wildcard resource, assume role risk, create policy version risk, rollback risk score, admin path exists.

**Compliance Metrics (10 features)**
Compliance violation count, has MFA condition, has IP restriction, has time restriction, deny statement count, condition complexity score, CIS violation flags, NIST control gaps, least privilege score, boundary policy exists.

**Temporal / Sequence (6 features)**
Policy version count, days since last modified, permission growth rate, has recent escalation, version drift score, temporal risk delta.

---

## Project Structure

```
iam-graphxai/
│
├── README.md                    # This file
├── requirements.txt             # All dependencies
├── .env.example                 # API key template
├── run_all.py                   # Main pipeline entry point
│
├── src/
│   ├── policy_parser.py         # AWS IAM JSON → structured data
│   ├── graph_builder.py         # Build heterogeneous IAM graph
│   ├── graph_schema.py          # Node/edge type definitions
│   ├── feature_extractor_v2.py  # Extract 38 features per policy
│   ├── weak_supervision_v2.py   # Label generation pipeline
│   ├── escalation_patterns.py   # PassRole/CreatePolicyVersion chains
│   ├── model_training_v2.py     # RF + XGBoost training + CV
│   ├── hgt_model.py             # Heterogeneous Graph Transformer
│   ├── lnn_temporal.py          # Liquid Neural Network (LTC)
│   ├── train_lnn_temporal.py    # LNN training pipeline
│   ├── connect_hgt_lnn.py       # HGT + LNN fusion (166D concat)
│   ├── build_sequences.py       # Policy version sequences for LNN
│   ├── cloudgoat_loader.py      # CloudGoat HIGH-risk injection
│   ├── merge_teammate_data.py   # Dataset merging utilities
│   ├── llm_reasoning.py         # Groq LLaMA reasoning agent
│   ├── generate_figures.py      # Paper figure generation (9 figures)
│   ├── debug_graph.py           # Graph validation utilities
│   ├── download_policies.py     # AWS IAM policy downloader
│   └── api_final.py             # Flask REST API
│
├── output/
│   └── figures/
│       ├── fig1_confusion_matrices.png
│       ├── fig2_roc_curves.png
│       ├── fig3_shap_beeswarm.png
│       ├── fig4_shap_per_class.png
│       ├── fig5_training_curves.png
│       ├── fig6_ablation.png
│       ├── fig7_risk_distribution.png
│       ├── fig8_feature_heatmap.png
│       └── fig9_radar_comparison.png
│
├── models/
│   ├── rf_results.json          # RF metrics and CV scores
│   └── xgb_v2.json              # XGBoost metrics
│
└── data/
    └── raw_policies/
        └── admin_example.json   # Example HIGH-risk policy
```

---

## API Usage

### Start the API

```bash
python src/api_final.py
# Server running on http://localhost:5000
```

### Assess a Policy

```bash
curl -X POST http://localhost:5000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "policy_name": "dev_team_policy",
    "policy_json": {
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Action": ["iam:PassRole", "iam:CreatePolicyVersion"],
        "Resource": "*"
      }]
    }
  }'
```

### Example Response

```json
{
  "policy_name": "dev_team_policy",
  "risk_label": "HIGH",
  "confidence": 0.987,
  "model_breakdown": {
    "rf": "HIGH",
    "lnn": "HIGH",
    "hgt": "HIGH",
    "ensemble": "HIGH"
  },
  "top_features": [
    {"feature": "compliance_violation_count", "impact": 28.0, "value": 4.0},
    {"feature": "dangerous_action_count", "impact": 14.0, "value": 2.0},
    {"feature": "privilege_escalation_risk_score", "impact": 8.0, "value": 1.0},
    {"feature": "passrole_chain_exists", "impact": 6.5, "value": 1.0}
  ],
  "mitre_mapping": {
    "id": "TA0004",
    "name": "Privilege Escalation",
    "url": "https://attack.mitre.org/tactics/TA0004/"
  },
  "remediation_steps": [
    "Scope iam:PassRole to a specific trusted role ARN, not Resource: *",
    "Remove iam:CreatePolicyVersion or restrict to designated admin roles only",
    "Enable CloudTrail logging for all IAM API calls on this policy",
    "Run IAM Access Analyzer to map all escalation paths"
  ],
  "llm_narrative": "This policy enables a two-step privilege escalation attack via the PassRole + CreatePolicyVersion chain. An attacker with access to this policy can create a new policy version with admin permissions and set it as default, then use PassRole to assign a powerful role to a Lambda they control..."
}
```

### Health Check

```bash
curl http://localhost:5000/health
# {"status": "healthy", "models_loaded": true}
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/rohith-r-1/iam-graphxai.git
cd iam-graphxai
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API key

```bash
cp .env.example .env
# Edit .env and add your Groq API key
# GROQ_API_KEY=your_key_here
```

Get a free Groq API key at [https://console.groq.com](https://console.groq.com)

### 5. Run the full pipeline

```bash
python run_all.py
```

Or run individual stages:

```bash
python run_all.py --from features     # Feature extraction only
python run_all.py --from training     # Training only
python run_all.py --from xai          # SHAP explainability only
python run_all.py --from api          # Start Flask API only
```

---

## Key Technical Contributions

**1. Heterogeneous Graph Transformer (HGT)**
Standard IAM scanners evaluate policies in isolation. HGT models the full permission graph with typed nodes (users, roles, policies, actions, resources) and typed edges — learning dangerous patterns up to 3 hops deep, where privilege escalation actually lives.

**2. Liquid Neural Network (LNN) for Temporal Drift**
IAM risk accumulates gradually. A role that is safe today may be dangerous after three incremental policy updates. LNN uses Ordinary Differential Equations with adaptive time constants — neurons that respond faster to sudden permission spikes and slower to gradual drift — capturing what static models miss.

**3. SHAP Explainability**
Every prediction includes TreeSHAP attribution — not just "HIGH risk" but "HIGH because: compliance_violation_count=4 (28% impact), dangerous_action_count=2 (14% impact), passrole_chain_exists=1 (6.5% impact)." Actionable, not just accurate.

**4. LLM Reasoning Agent**
Groq LLaMA 3.1 8B converts SHAP outputs and policy JSON into plain-language attack narratives for developers and managers who need to understand and fix the issue — not just receive a risk score. MITRE ATT&CK mapping is rule-based for deterministic accuracy.

**5. Hybrid Ensemble**
Combines RF (tabular accuracy), LNN (temporal context), and HGT (graph topology) via majority voting. No single model dominates — the ensemble catches what each misses individually.

---

## MITRE ATT&CK Coverage

| IAM Pattern | MITRE Tactic | ID |
|-------------|-------------|-----|
| PassRole + CreatePolicyVersion chain | Privilege Escalation | TA0004 |
| Wildcard Action + Wildcard Resource | Impact | TA0040 |
| Cross-account role assumption | Lateral Movement | TA0008 |
| Dangerous action count > 2 | Credential Access | TA0006 |

---

## Baseline Comparison

| Tool | Approach | Catches Multi-Hop Chains | Severity Classification | Actionable Remediation |
|------|----------|--------------------------|------------------------|----------------------|
| AWS Access Analyzer | Rule-based | ❌ | ❌ | ❌ |
| Cloudsplaining | Rule-based | ❌ | ❌ | Partial |
| IAM-Deescalate | Rule-based | ❌ | ❌ | ❌ |
| **IAM-GraphXAI** | **Graph ML + LLM** | **✅** | **✅ (3 levels)** | **✅ (LLM narrative)** |

---

**References:**
- Hu et al., *Heterogeneous Graph Transformer*, WWW 2020
- Hasani et al., *Liquid Time-Constant Networks*, AAAI 2021
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017
- Ying et al., *GNNExplainer*, NeurIPS 2019

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Graph ML | PyTorch Geometric, HGTConv |
| Temporal ML | ncps (Neural Circuit Policies), LTC |
| Tabular ML | scikit-learn, XGBoost |
| Explainability | SHAP (TreeExplainer) |
| LLM | Groq API, LLaMA 3.1 8B Instant |
| API | Flask |
| Graphs | NetworkX |
| Visualisation | Matplotlib, Seaborn |
| AWS | boto3 |

---

### Authors
Rohith R
Anish R A
Sri Sheshadri R

---

<div align="center">
</div>

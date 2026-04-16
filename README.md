# IAM-GraphXAI

**Explainable AWS IAM Risk Detection using Heterogeneous Graph Transformer, Liquid Neural Network, and LLM Reasoning**

> IEEE Paper | VIT University, Chennai | 2026

## Overview
CloudShield AI classifies AWS IAM policies as LOW / MEDIUM / HIGH risk using a three-tier pipeline:
1. **HGT** — models relationships across 9,558 nodes and 637K edges in the IAM permission graph
2. **LNN** — detects temporal permission drift using Liquid Time Constant neurons
3. **LLM (Groq LLaMA)** — generates plain-language attack narratives and MITRE ATT&CK mappings

## Results
| Model | Macro F1 | Accuracy |
|---|---|---|
| Random Forest (baseline) | 0.9936 | 99.7% |
| LNN Concat (166D) | 0.9882 | 99.5% |
| HGT Graph Only | 0.9206 | 97.2% |
| Ensemble RF + LNN | 0.9930 | 99.7% |

## Setup
\\\ash
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY
python run_all.py
\\\

## Project Structure
\\\
src/           Source code (HGT, LNN, SHAP, API, LLM)
output/        Figures and results
models/        Model outputs and metrics
data/          IAM policy data
\\\


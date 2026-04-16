# run_all.py
"""
CloudShield IAM-Graph-XAI — Full Pipeline Runner
=================================================
Chains every step in order, times each stage,
handles failures gracefully, and prints a final
paper-ready results table.

Usage:
  python run_all.py              # full pipeline
  python run_all.py --from hgt   # resume from a step
  python run_all.py --only lnn   # run one step only
  python run_all.py --skip fig   # skip a step
  python run_all.py --status     # check cached outputs
  python run_all.py --list       # list all steps

Steps:
  features  → Feature extraction (40D)
  rf        → Random Forest + XGBoost
  hgt       → Heterogeneous Graph Transformer
  lnn       → Liquid Neural Network (raw features)
  connect   → HGT → LNN embedding bridge
  xai       → LLM reasoning + SHAP faithfulness
  figures   → Paper figures (9 PNG + PDF)
"""

import os
import sys
import time
import json
import argparse
import subprocess
import traceback
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours (Windows compatible via colorama fallback)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import colorama
    colorama.init()
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    CYAN   = '\033[96m'
    BOLD   = '\033[1m'
    RESET  = '\033[0m'
except ImportError:
    GREEN = YELLOW = RED = CYAN = BOLD = RESET = ''

TICK  = f"{GREEN}✔{RESET}"
CROSS = f"{RED}✘{RESET}"
SKIP  = f"{YELLOW}⊘{RESET}"
RUN   = f"{CYAN}▶{RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline step definitions
# ─────────────────────────────────────────────────────────────────────────────
STEPS = [
    {
        'key'    : 'merge',
        'label'  : 'Merge Teammate Data (22 missing features)',
        'script' : 'src/merge_teammate_data.py',
        'outputs': ['data/labeled_features_merged.csv'],
        'check'  : 'data/labeled_features_merged.csv',
    },
    {
        'key'    : 'features',
        'label'  : 'Feature Extraction (40D)',
        'script' : 'src/feature_extractor_v2.py',
        'outputs': ['data/labeled_features_v2.csv',
                    'models/feature_names_v2.pkl'],
        'check'  : 'data/labeled_features_v2.csv',
    },
    {
        'key'    : 'rf',
        'label'  : 'RF + XGBoost Classifiers',
        'script' : 'src/model_training_v2.py',       
        'outputs': ['models/rf_v2.pkl', 'models/xgb_v2.json'],
        'check'  : 'models/rf_v2.pkl',
    },
    {
        'key'    : 'hgt',
        'label'  : 'HGT — Heterogeneous Graph Transformer',
        'script' : 'src/hgt_model.py',
        'outputs': ['models/hgt_model.pt',
                    'models/hgt_node_index.pkl',
                    'output/hgt_results.json'],
        'check'  : 'models/hgt_model.pt',
    },
    {
        'key'    : 'lnn',
        'label'  : 'LNN — Liquid Neural Network (raw features)',
        'script' : 'src/lnn_temporal.py',
        'outputs': ['models/lnn_model.pt',
                    'models/lnn_scaler.pkl',
                    'output/lnn_results.json'],
        'check'  : 'models/lnn_model.pt',
    },
    {
        'key'    : 'connect',
        'label'  : 'HGT→LNN Architecture Bridge',
        'script' : 'src/connect_hgt_lnn.py',
        'outputs': ['data/hgt_embeddings.npy',
                    'models/lnn_hgt_model.pt',
                    'models/lnn_concat_model.pt',
                    'output/connection_results.json'],
        'check'  : 'output/connection_results.json',
    },
    {
        'key'    : 'xai',
        'label'  : 'LLM Reasoning + SHAP XAI',
        'script' : 'src/llm_reasoning.py',
        'outputs': ['output/risk_explanations.json',
                    'output/xai_metrics.json'],
        'check'  : 'output/xai_metrics.json',
    },
    {
        'key'    : 'figures',
        'label'  : 'Paper Figures (9 PNG + PDF)',
        'script' : 'src/generate_figures.py',
        'outputs': ['output/figures/fig1_confusion_matrices.png',
                    'output/figures/fig6_ablation.png',
                    'output/figures/fig9_radar_comparison.png'],
        'check'  : 'output/figures/fig9_radar_comparison.png',
    },
]

STEP_KEYS = [s['key'] for s in STEPS]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def fmt_size(path):
    """Human-readable file size — bytes for small files, KB for large."""
    raw = os.path.getsize(path)
    if raw < 1024:
        return f"{raw} B"
    elif raw < 1024 * 1024:
        return f"{raw // 1024} KB"
    else:
        return f"{raw / (1024*1024):.1f} MB"


def already_done(step):
    return os.path.exists(step['check'])


def load_metric(path, key, default='—'):
    """Read a single numeric metric from a JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        val = data.get(key, default)
        if isinstance(val, float):
            return f"{val:.4f}"
        if isinstance(val, int):
            return str(val)
        return str(val) if val is not None else default
    except Exception:
        return default


def load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def print_banner():
    print()
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  CloudShield IAM-Graph-XAI — Full Pipeline Runner{RESET}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  CWD     : {os.getcwd()}")
    print(f"{BOLD}{'='*65}{RESET}")
    print()


def print_step_header(step_num, total, step):
    print(f"\n{BOLD}{CYAN}[{step_num}/{total}] {step['label']}{RESET}")
    print(f"  Script  : {step['script']}")
    print(f"  Outputs : {', '.join(os.path.basename(o) for o in step['outputs'])}")
    print()


def run_script(script_path):
    """Run script as subprocess, stream output live to terminal."""
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.getcwd(),
        text=True,
        capture_output=False
    )
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(start_key=None, only_key=None, skip_keys=None):
    print_banner()
    skip_keys  = skip_keys or []
    results    = {}
    start_wall = time.time()

    # Determine active steps
    if only_key:
        active_steps = [s for s in STEPS if s['key'] == only_key]
    elif start_key:
        idx          = STEP_KEYS.index(start_key)
        active_steps = STEPS[idx:]
    else:
        active_steps = STEPS

    print(f"  Steps to run : {', '.join(s['key'] for s in active_steps)}")
    if skip_keys:
        print(f"  Skipping     : {', '.join(skip_keys)}")

    for step in STEPS:
        key = step['key']

        # ── Not in active set ─────────────────────────────────────────
        if step not in active_steps or key in skip_keys:
            results[key] = {'status': 'skipped', 'duration': 0}
            print(f"\n{SKIP} [{key}] {step['label']} — skipped")
            continue

        # ── Cached ────────────────────────────────────────────────────
        if already_done(step) and step not in active_steps[:1]:
            results[key] = {'status': 'cached', 'duration': 0}
            print(f"\n{TICK} [{key}] {step['label']} — cached (outputs exist)")
            continue

        print_step_header(STEP_KEYS.index(key) + 1, len(STEPS), step)

        # ── Script missing ────────────────────────────────────────────
        if not os.path.exists(step['script']):
            msg = f"Script not found: {step['script']}"
            print(f"  {CROSS} {msg}")
            results[key] = {'status': 'missing_script', 'duration': 0, 'error': msg}
            print(f"  {YELLOW}Skipping — create {step['script']} to enable this step{RESET}")
            continue

        # ── Run ───────────────────────────────────────────────────────
        print(f"  {RUN} Running {step['script']}...")
        t0 = time.time()
        try:
            rc       = run_script(step['script'])
            duration = time.time() - t0

            if rc == 0:
                results[key] = {'status': 'ok', 'duration': duration}
                print(f"\n  {TICK} [{key}] completed in {fmt_time(duration)}")
            else:
                results[key] = {
                    'status'  : 'failed',
                    'duration': duration,
                    'error'   : f"Exit code {rc}"
                }
                print(f"\n  {CROSS} [{key}] FAILED (exit code {rc}) "
                      f"after {fmt_time(duration)}")
                print(f"  {RED}Pipeline continues — downstream steps may fail.{RESET}")

        except KeyboardInterrupt:
            print(f"\n  {YELLOW}Interrupted — stopping pipeline.{RESET}")
            results[key] = {'status': 'interrupted', 'duration': time.time() - t0}
            break

        except Exception as e:
            duration = time.time() - t0
            results[key] = {'status': 'error', 'duration': duration, 'error': str(e)}
            print(f"\n  {CROSS} [{key}] ERROR: {e}")
            traceback.print_exc()

    total_time = time.time() - start_wall
    print_final_report(results, total_time)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────────────
def print_final_report(results, total_time):
    print()
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  PIPELINE COMPLETE — {fmt_time(total_time)}{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")

    # ── Step status table ─────────────────────────────────────────────
    print(f"\n  {'Step':<10} {'Status':<16} {'Time':>8}  Label")
    print("  " + "-"*58)
    for step in STEPS:
        key  = step['key']
        r    = results.get(key, {'status': 'not_run', 'duration': 0})
        stat = r['status']
        dur  = r.get('duration', 0)

        icon  = (TICK  if stat in ('ok', 'cached')              else
                 SKIP  if stat in ('skipped', 'missing_script') else
                 CROSS)
        color = (GREEN  if stat in ('ok', 'cached')              else
                 YELLOW if stat in ('skipped', 'missing_script') else
                 RED)

        print(f"  {key:<10} {color}{stat:<16}{RESET}"
              f"{fmt_time(dur):>8}  {step['label']}")

    # ── Model performance table ───────────────────────────────────────
    print(f"\n{BOLD}  MODEL PERFORMANCE SUMMARY{RESET}")
    print(f"  {'Model':<38} {'Macro F1':>10}")
    print("  " + "-"*50)

    # Each row: (display name, json path, json key, hardcoded fallback)
    model_rows = [
        ("Baseline (rule-based)",
            None,                              None,
            "0.0989"),
        ("Random Forest (40D)",
            "output/xai_metrics.json",         "rf_macro_f1",
            None),
        ("XGBoost (40D)",
            "output/xai_metrics.json",         "xgb_macro_f1",
            None),
        ("HGT (graph transformer)",
            "output/hgt_results.json",         "best_val_f1",   # ← fixed key
            None),
        ("LNN (raw 40D features)",
            "output/lnn_results.json",         "test_f1",
            None),
        ("LNN (128D HGT embeddings)",
            "output/connection_results.json",  "lnn_hgt_embeddings_f1",
            None),
        ("LNN (168D concat)",
            "output/connection_results.json",  "lnn_concat_f1",
            None),
        ("Ensemble RF + LNN",
            "output/xai_metrics.json",         "ensemble_macro_f1",
            None),
    ]

    for name, path, key, fallback in model_rows:
        if path and key:
            val = load_metric(path, key, fallback or '—')
        else:
            val = fallback or '—'

        is_final = "Ensemble" in name
        marker   = f"  {GREEN}◄ FINAL{RESET}" if is_final else ""
        bold_s   = BOLD  if is_final else ''
        bold_e   = RESET if is_final else ''
        print(f"  {bold_s}{name:<38} {val:>10}{bold_e}{marker}")

    # ── XAI metrics ───────────────────────────────────────────────────
    xai_path = "output/xai_metrics.json"
    if os.path.exists(xai_path):
        print(f"\n{BOLD}  XAI METRICS{RESET}")
        print(f"  {'Metric':<40} {'Value':>10}")
        print("  " + "-"*52)
        xai_rows = [
            ("SHAP Faithfulness (balanced)",    "faithfulness"),
            ("SHAP Sufficiency",                "shap_sufficiency"),
            ("SHAP Completeness",               "shap_completeness"),
            ("Mean Prediction Confidence",      "mean_confidence"),
            ("HIGH-risk Policies Detected",     "high_risk_count"),
            ("Total Policies Explained",        "total_explained"),
            ("LLM Backend Used",                "llm_backend"),
        ]
        for label, key in xai_rows:
            val = load_metric(xai_path, key)
            print(f"  {label:<40} {val:>10}")

    # ── Output files ──────────────────────────────────────────────────
    print(f"\n{BOLD}  OUTPUT FILES{RESET}")
    output_files = [
        # JSON results
        "output/hgt_results.json",
        "output/lnn_results.json",
        "output/connection_results.json",
        "output/xai_metrics.json",
        "output/risk_explanations.json",
        # Figures
        "output/figures/fig1_confusion_matrices.png",
        "output/figures/fig2_roc_curves.png",
        "output/figures/fig3_shap_beeswarm.png",
        "output/figures/fig4_shap_per_class.png",
        "output/figures/fig5_training_curves.png",
        "output/figures/fig6_ablation.png",
        "output/figures/fig7_risk_distribution.png",
        "output/figures/fig8_feature_heatmap.png",
        "output/figures/fig9_radar_comparison.png",
        # Models
        "models/rf_v2.pkl",
        "models/hgt_model.pt",
        "models/lnn_model.pt",
        "models/lnn_hgt_model.pt",
        "models/lnn_concat_model.pt",
    ]

    categories = {
        "output/hgt_results.json"   : "Results JSON",
        "output/figures/fig1_confusion_matrices.png": "Figures",
        "models/rf_v2.pkl"          : "Models",
    }
    current_cat = None

    for fpath in output_files:
        # Print category header
        cat = ("Results JSON" if fpath.startswith("output/") and fpath.endswith(".json")
               else "Figures"  if "figures" in fpath
               else "Models"   if fpath.startswith("models/")
               else "Other")
        if cat != current_cat:
            print(f"\n  {BOLD}{cat}{RESET}")
            current_cat = cat

        if os.path.exists(fpath):
            size = fmt_size(fpath)
            print(f"    {TICK} {os.path.basename(fpath):<45} {size:>10}")
        else:
            print(f"    {CROSS} {os.path.basename(fpath):<45}"
                  f" {RED}missing{RESET}")

    # ── Summary line ──────────────────────────────────────────────────
    n_ok   = sum(1 for r in results.values() if r['status'] in ('ok', 'cached'))
    n_fail = sum(1 for r in results.values() if r['status'] == 'failed')
    n_skip = sum(1 for r in results.values()
                 if r['status'] in ('skipped', 'missing_script'))

    print(f"\n  Steps  : {GREEN}{n_ok} passed{RESET}  "
          f"{RED}{n_fail} failed{RESET}  "
          f"{YELLOW}{n_skip} skipped{RESET}")
    print(f"  Time   : {fmt_time(total_time)}")
    print(f"  Done at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{BOLD}{'='*65}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────
def cmd_list():
    print(f"\n{BOLD}CloudShield Pipeline Steps{RESET}")
    print("-"*58)
    for i, s in enumerate(STEPS):
        done = already_done(s)
        icon = f"{GREEN}✔{RESET}" if done else "○"
        print(f"  {i+1}. {icon}  {s['key']:<12} {s['label']}")
        print(f"       Script  : {s['script']}")
        print(f"       Check   : {s['check']}")
        print()


def cmd_status():
    print(f"\n{BOLD}Pipeline Output Status{RESET}")
    print("-"*58)
    all_done = True
    for s in STEPS:
        done = already_done(s)
        if not done:
            all_done = False
        icon  = f"{GREEN}DONE  {RESET}" if done else f"{YELLOW}NEEDED{RESET}"
        age   = ""
        if done:
            mtime = os.path.getmtime(s['check'])
            age   = f"  (saved {datetime.fromtimestamp(mtime).strftime('%b %d %H:%M')})"
        print(f"  {icon}  {s['key']:<12} {s['label']}{age}")

    print()
    if all_done:
        print(f"  {GREEN}{BOLD}All outputs present — pipeline is fully cached.{RESET}")
        print(f"  Run: python run_all.py --from features  to force re-run\n")
    else:
        missing = [s['key'] for s in STEPS if not already_done(s)]
        print(f"  {YELLOW}Missing: {', '.join(missing)}{RESET}")
        print(f"  Run: python run_all.py  to generate missing outputs\n")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='CloudShield IAM-Graph-XAI pipeline runner',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_all.py                    # full pipeline\n"
            "  python run_all.py --status           # check cached outputs\n"
            "  python run_all.py --list             # list all steps\n"
            "  python run_all.py --from hgt         # resume from HGT\n"
            "  python run_all.py --only xai         # run only XAI step\n"
            "  python run_all.py --skip connect fig # skip bridge + figures\n"
        )
    )
    parser.add_argument(
        '--from', dest='from_step', metavar='STEP',
        choices=STEP_KEYS, default=None,
        help=f'Resume from step  [{", ".join(STEP_KEYS)}]'
    )
    parser.add_argument(
        '--only', dest='only_step', metavar='STEP',
        choices=STEP_KEYS, default=None,
        help='Run only one step'
    )
    parser.add_argument(
        '--skip', dest='skip_steps', metavar='STEP',
        nargs='+', choices=STEP_KEYS, default=[],
        help='Skip one or more steps'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all steps with their status and exit'
    )
    parser.add_argument(
        '--status', action='store_true',
        help='Show which steps have cached outputs and exit'
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Must be run from project root E:\iam-graph-xai
    if not os.path.exists('src') or not os.path.exists('data'):
        print(f"{RED}ERROR: Run from project root (E:\\iam-graph-xai){RESET}")
        print(f"  cd E:\\iam-graph-xai && python run_all.py")
        sys.exit(1)

    args = parse_args()

    if args.list:
        cmd_list()
        sys.exit(0)

    if args.status:
        cmd_status()
        sys.exit(0)

    if args.from_step and args.only_step:
        print(f"{RED}ERROR: --from and --only are mutually exclusive{RESET}")
        sys.exit(1)

    results = run_pipeline(
        start_key  = args.from_step,
        only_key   = args.only_step,
        skip_keys  = args.skip_steps
    )

    # Non-zero exit if any step failed (useful for CI/CD)
    failed = [k for k, r in results.items() if r['status'] == 'failed']
    if failed:
        print(f"{RED}Failed steps: {', '.join(failed)}{RESET}")
        sys.exit(1)

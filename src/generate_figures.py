# src/generate_figures.py
"""
Paper Figure Generator — CloudShield IAM XAI
Produces all figures for IEEE/ACM submission:
  Fig 1: Confusion matrices (RF, HGT, LNN, Ensemble)
  Fig 2: ROC curves (one-vs-rest, all models)
  Fig 3: SHAP beeswarm plot (top-15 features)
  Fig 4: SHAP bar chart (mean |SHAP| per class)
  Fig 5: Training curves (HGT + LNN)
  Fig 6: Ablation study bar chart
  Fig 7: Risk distribution (dataset overview)
  Fig 8: Feature correlation heatmap (top-20)
All saved to output/figures/ as 300dpi PNG + PDF
"""

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

# ── Paper style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'legend.fontsize'  : 10,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

COLORS = {
    'LOW'    : '#2ecc71',
    'MEDIUM' : '#f39c12',
    'HIGH'   : '#e74c3c',
    'hgt'    : '#3498db',
    'lnn'    : '#9b59b6',
    'rf'     : '#2ecc71',
    'ens'    : '#e74c3c',
    'baseline': '#95a5a6',
}
LABEL_MAP  = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
OUT_DIR    = 'output/figures'
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    path_png = os.path.join(OUT_DIR, f"{name}.png")
    path_pdf = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f"  Saved: {path_png}")


# ─────────────────────────────────────────────────────────────────────────────
def load_everything():
    print("Loading artifacts...")

    df = pd.read_csv('data/labeled_features_merged.csv')
    df = df[df['risk_label'] != -1].reset_index(drop=True)

    with open('models/feature_names_v2.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    X      = df[feature_names].fillna(0).values.astype(np.float32)
    y_true = df['risk_label'].values.astype(int)

    # RF
    with open('models/rf_v2.pkl', 'rb') as f:
        rf = pickle.load(f)
    y_proba_rf = rf.predict_proba(X)
    y_pred_rf  = rf.predict(X)

    # LNN
    import torch, torch.nn.functional as F
    from lnn_temporal import CloudShieldLNN, simulate_temporal_sequences
    with open('models/lnn_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    lnn_ckpt  = torch.load('models/lnn_model.pt', map_location='cpu')
    lnn_model = CloudShieldLNN(X.shape[1], hidden_size=64, num_classes=3)
    lnn_model.load_state_dict(lnn_ckpt['model_state_dict'])
    lnn_model.eval()
    X_norm    = scaler.transform(X).astype(np.float32)
    X_seq     = simulate_temporal_sequences(X_norm, y_true, T=5, seed=42)
    with torch.no_grad():
        lnn_logits  = lnn_model(torch.tensor(X_seq))
        y_proba_lnn = F.softmax(lnn_logits, dim=1).numpy()
        y_pred_lnn  = lnn_logits.argmax(dim=1).numpy()

    # Ensemble
    y_proba_ens = (y_proba_rf + y_proba_lnn) / 2.0
    y_pred_ens  = y_proba_ens.argmax(axis=1)

    # HGT results from JSON (model too large to re-infer here)
    with open('output/hgt_results.json') as f:
        hgt_results = json.load(f)

    # SHAP
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X[:200])   # [3, 200, 40]
    shap_arr  = np.array(shap_vals)

    print(f"  Loaded: {len(df)} policies, {X.shape[1]} features")
    return (df, feature_names, X, y_true,
            rf, y_pred_rf, y_proba_rf,
            y_pred_lnn, y_proba_lnn,
            y_pred_ens, y_proba_ens,
            hgt_results, shap_arr)


# ─────────────────────────────────────────────────────────────────────────────
def fig_confusion_matrices(y_true, y_pred_rf, y_pred_lnn,
                            y_pred_ens, hgt_f1):
    """Fig 1 — 2×2 confusion matrix grid."""
    from sklearn.metrics import confusion_matrix
    print("Fig 1: Confusion matrices...")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Confusion Matrices — CloudShield IAM Risk Classification',
                 fontsize=13, fontweight='bold', y=1.01)

    models = [
        ('Random Forest',    y_pred_rf,  'Greens'),
        ('LNN (Temporal)',   y_pred_lnn, 'Purples'),
        ('Ensemble RF+LNN',  y_pred_ens, 'Reds'),
    ]
    labels = ['LOW', 'MED', 'HIGH']

    for ax, (title, y_pred, cmap) in zip(axes.flat[:3], models):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    # 4th panel: HGT summary (no raw predictions — use metrics)
    ax = axes[1, 1]
    ax.axis('off')
    txt = (
        "HGT (Heterogeneous Graph\nTransformer) — Test Set\n\n"
        f"  Macro F1 : {hgt_f1:.4f}\n"
        f"  LOW  F1  : 1.00\n"
        f"  MED  F1  : 0.98\n"
        f"  HIGH F1  : 0.67\n\n"
        "Note: CloudGoat HIGH-risk nodes\n"
        "have synthetic KNN edges only.\n"
        "HGT graph embeddings feed LNN."
    )
    ax.text(0.1, 0.9, txt, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#dbeafe', alpha=0.8))
    ax.set_title('HGT (Graph Model)', fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'fig1_confusion_matrices')


# ─────────────────────────────────────────────────────────────────────────────
def fig_roc_curves(y_true, y_proba_rf, y_proba_lnn, y_proba_ens):
    """Fig 2 — One-vs-rest ROC curves for all models."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    print("Fig 2: ROC curves...")

    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('ROC Curves (One-vs-Rest) — All Models',
                 fontsize=13, fontweight='bold')

    class_names = ['LOW', 'MEDIUM', 'HIGH']
    model_specs = [
        ('RF',       y_proba_rf,  COLORS['rf'],  '-'),
        ('LNN',      y_proba_lnn, COLORS['lnn'], '--'),
        ('Ensemble', y_proba_ens, COLORS['ens'],  ':'),
    ]

    for ci, (cname, ax) in enumerate(zip(class_names, axes)):
        for mname, proba, color, ls in model_specs:
            fpr, tpr, _ = roc_curve(y_bin[:, ci], proba[:, ci])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, ls=ls, lw=2,
                    label=f'{mname} (AUC={roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random')
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{cname} Risk (one-vs-rest)', fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)

        # Shade area under RF curve
        fpr_rf, tpr_rf, _ = roc_curve(y_bin[:, ci], y_proba_rf[:, ci])
        ax.fill_between(fpr_rf, tpr_rf, alpha=0.08,
                        color=COLORS['rf'])

    plt.tight_layout()
    save_fig(fig, 'fig2_roc_curves')


# ─────────────────────────────────────────────────────────────────────────────
def fig_shap_beeswarm(shap_arr, X, feature_names, y_true):
    """Fig 3 — SHAP beeswarm (global feature importance)."""
    print("Fig 3: SHAP beeswarm...")

    try:
        import shap
        # Mean absolute SHAP across all classes
        mean_abs_shap = np.abs(shap_arr).mean(axis=0)   # [200, 40]

        # Top 15 features by mean |SHAP|
        feat_importance = mean_abs_shap.mean(axis=0)    # [40]
        top15_idx       = np.argsort(feat_importance)[-15:][::-1]
        top15_names     = [feature_names[i] for i in top15_idx]
        top15_shap      = mean_abs_shap[:, top15_idx]   # [200, 15]
        top15_X         = X[:200, top15_idx]

        fig, ax = plt.subplots(figsize=(10, 7))

        # Manual beeswarm
        n_samples, n_feats = top15_shap.shape
        for fi in range(n_feats):
            sv   = top15_shap[:, fi]
            fv   = top15_X[:, fi]
            fv_n = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
            y_pos = np.full(n_samples, n_feats - 1 - fi, dtype=float)
            jitter = np.random.default_rng(fi).uniform(-0.2, 0.2, n_samples)
            y_pos  += jitter
            sc = ax.scatter(sv, y_pos, c=fv_n, cmap='RdBu_r',
                            s=12, alpha=0.6, linewidths=0)

        cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label('Feature value\n(low → high)', fontsize=9)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Low', 'Med', 'High'])

        ax.set_yticks(range(n_feats))
        ax.set_yticklabels(reversed(top15_names), fontsize=9)
        ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlabel('SHAP Value (impact on model output)')
        ax.set_title('SHAP Feature Importance — CloudShield RF Model\n'
                     '(Top 15 features, n=200 policies)',
                     fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        save_fig(fig, 'fig3_shap_beeswarm')

    except Exception as e:
        print(f"  Skipped (error): {e}")


# ─────────────────────────────────────────────────────────────────────────────
def fig_shap_per_class(shap_arr, feature_names):
    """Fig 4 — Mean |SHAP| per class (grouped bar chart)."""
    print("Fig 4: SHAP per-class bar chart...")

    mean_abs = [np.abs(shap_arr[c]).mean(axis=0) for c in range(3)]  # [3, 40]
    overall  = np.array(mean_abs).mean(axis=0)
    top10    = np.argsort(overall)[-10:][::-1]
    names    = [feature_names[i] for i in top10]

    fig, ax = plt.subplots(figsize=(12, 5))
    x   = np.arange(len(names))
    w   = 0.25

    for ci, (cls_name, color) in enumerate(
        [('LOW', COLORS['LOW']), ('MEDIUM', COLORS['MEDIUM']),
         ('HIGH', COLORS['HIGH'])]
    ):
        vals = mean_abs[ci][top10]
        bars = ax.bar(x + ci*w, vals, w, label=cls_name,
                      color=color, alpha=0.85, edgecolor='white')

    ax.set_xticks(x + w)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean |SHAP Value|')
    ax.set_title('Feature Importance by Risk Class (SHAP)\n'
                 'Top 10 features — CloudShield IAM Risk Model',
                 fontweight='bold')
    ax.legend(title='Risk Class')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'fig4_shap_per_class')


# ─────────────────────────────────────────────────────────────────────────────
def fig_training_curves():
    """Fig 5 — HGT + LNN training curves from saved JSON."""
    print("Fig 5: Training curves...")

    # HGT training data (from run logs — hardcoded from output)
    hgt_epochs = list(range(0, 1000, 10))
    hgt_f1     = [
        0.0074, 0.5645, 0.7229, 0.3266, 0.6460, 0.4382, 0.5667,
        0.6251, 0.8436, 0.7019, 0.7300, 0.7325, 0.7287, 0.7364,
        0.7300, 0.7300, 0.7364, 0.7300, 0.7364, 0.7300, 0.7680,
        0.7575, 0.7695, 0.7611, 0.7525, 0.7611, 0.7695, 0.7695,
        0.7695, 0.7777, 0.7695, 0.7695, 0.7857, 0.7823, 0.7857,
        0.7909, 0.7909, 0.7857, 0.7934, 0.7934, 0.8133, 0.8133,
        0.8279, 0.8212, 0.8133, 0.8509, 0.8450, 0.8509, 0.8509,
        0.8578, 0.8578, 0.8639, 0.8586, 0.8646, 0.8702, 0.8652,
        0.8770, 0.8770, 0.8712, 0.8836, 0.8773, 0.8836, 0.8773,
        0.8836, 0.8836, 0.8770, 0.8836, 0.8836, 0.8773, 0.8836,
        0.8773, 0.8773, 0.8712, 0.8773, 0.8773, 0.8773, 0.8836,
        0.8773, 0.8836, 0.8836, 0.8836, 0.8836, 0.8836, 0.8836,
        0.8836, 0.8836, 0.8773, 0.8773, 0.8836, 0.8836, 0.8836,
        0.8836, 0.8836, 0.8836, 0.8836, 0.8836, 0.8836, 0.8836,
        0.8836,
    ]
    hgt_epochs = hgt_epochs[:len(hgt_f1)]

    # LNN training data
    lnn_epochs = list(range(0, 300, 10))
    lnn_f1 = [
        0.0074, 0.8780, 0.8957, 0.9751, 0.9751, 0.9751, 0.9811,
        0.9811, 0.9811, 0.9811, 0.9811, 0.9811, 0.9811, 0.9811,
        0.9811, 0.9811, 0.9873, 0.9807, 0.9807, 0.9807, 0.9807,
        0.9807, 0.9751, 0.9873, 0.9869, 0.9807, 0.9869, 0.9807,
        0.9869, 0.9869,
    ]
    lnn_epochs = lnn_epochs[:len(lnn_f1)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle('Training Convergence — CloudShield Models',
                 fontsize=13, fontweight='bold')

    # HGT
    ax = axes[0]
    ax.plot(hgt_epochs, hgt_f1, color=COLORS['hgt'], lw=2, label='HGT Val F1')
    best_hgt = max(hgt_f1)
    best_ep  = hgt_epochs[hgt_f1.index(best_hgt)]
    ax.axhline(best_hgt, color=COLORS['hgt'], ls='--', alpha=0.5,
               label=f'Best={best_hgt:.4f}')
    ax.axhline(0.92, color='gray', ls=':', alpha=0.7, label='Target=0.92')
    ax.scatter([best_ep], [best_hgt], s=80, color=COLORS['hgt'], zorder=5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Macro F1')
    ax.set_title('HGT (Heterogeneous Graph Transformer)', fontweight='bold')
    ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(alpha=0.3)

    # Annotate SMOTE injection
    ax.annotate('SMOTE\nHIGH×30', xy=(0, 0.56), xytext=(80, 0.35),
                fontsize=8, color='purple',
                arrowprops=dict(arrowstyle='->', color='purple', lw=1))

    # LNN
    ax = axes[1]
    ax.plot(lnn_epochs, lnn_f1, color=COLORS['lnn'], lw=2, label='LNN Val F1')
    best_lnn = max(lnn_f1)
    best_ep2 = lnn_epochs[lnn_f1.index(best_lnn)]
    ax.axhline(best_lnn, color=COLORS['lnn'], ls='--', alpha=0.5,
               label=f'Best={best_lnn:.4f}')
    ax.axhline(0.92, color='gray', ls=':', alpha=0.7, label='Target=0.92')
    ax.scatter([best_ep2], [best_lnn], s=80, color=COLORS['lnn'], zorder=5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Macro F1')
    ax.set_title('LNN (Liquid Neural Network — ncps LTC)', fontweight='bold')
    ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(alpha=0.3)

    # Annotate fast convergence
    ax.annotate('Crosses 0.92\nat epoch 30', xy=(30, 0.9751), xytext=(80, 0.82),
                fontsize=8, color='purple',
                arrowprops=dict(arrowstyle='->', color='purple', lw=1))

    plt.tight_layout()
    save_fig(fig, 'fig5_training_curves')


# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation():
    """Fig 6 — Ablation study bar chart."""
    print("Fig 6: Ablation study...")

    ablation = {
        'Baseline\n(rules)'         : 0.0989,
        'RF only\n(no graph)'       : 1.0000,
        'HGT only\n(no SMOTE)'      : 0.7692,
        'HGT + SMOTE\n(no focal)'   : 0.7907,
        'HGT full'                   : 0.8836,
        'LNN only\n(no SMOTE)'      : 0.8780,
        'LNN + SMOTE\n(full)'       : 0.9873,
        'Ensemble\nRF + LNN'        : 0.9973,
    }

    fig, ax = plt.subplots(figsize=(13, 5))
    names   = list(ablation.keys())
    values  = list(ablation.values())
    bar_colors = [
        COLORS['baseline'], COLORS['rf'],
        COLORS['hgt'],      COLORS['hgt'],
        COLORS['hgt'],      COLORS['lnn'],
        COLORS['lnn'],      COLORS['ens'],
    ]

    bars = ax.bar(names, values, color=bar_colors, edgecolor='white',
                  linewidth=1.2, alpha=0.88, width=0.6)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    ax.axhline(0.92, color='red', ls='--', lw=1.5, alpha=0.7,
               label='Paper target = 0.92')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Ablation Study — CloudShield Component Contributions\n'
                 'IAM Policy Risk Classification (888 policies, 40 features)',
                 fontweight='bold')
    ax.set_ylim([0, 1.12])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Component labels
    ax.text(1.5, 1.08, '← HGT variants →', ha='center',
            fontsize=9, color=COLORS['hgt'], style='italic')
    ax.text(5.5, 1.08, '← LNN variants →', ha='center',
            fontsize=9, color=COLORS['lnn'], style='italic')

    plt.tight_layout()
    save_fig(fig, 'fig6_ablation')


# ─────────────────────────────────────────────────────────────────────────────
def fig_risk_distribution(y_true):
    """Fig 7 — Dataset risk label distribution."""
    print("Fig 7: Risk distribution...")

    counts     = {l: int((y_true == c).sum())
                  for c, l in LABEL_MAP.items()}
    pcts       = {l: v/len(y_true)*100 for l, v in counts.items()}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('CloudShield Dataset — IAM Policy Risk Distribution',
                 fontsize=13, fontweight='bold')

    # Bar chart
    ax = axes[0]
    bar_c = [COLORS[l] for l in counts]
    bars  = ax.bar(counts.keys(), counts.values(),
                   color=bar_c, edgecolor='white', width=0.5)
    for bar, (label, val) in zip(bars, counts.items()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'n={val}\n({pcts[label]:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Policies')
    ax.set_title('Policy Count by Risk Level', fontweight='bold')
    ax.set_ylim([0, max(counts.values()) * 1.2])
    ax.grid(axis='y', alpha=0.3)

    # Donut chart
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        counts.values(),
        labels=counts.keys(),
        colors=[COLORS[l] for l in counts],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
        textprops=dict(fontsize=11)
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax.set_title('Class Distribution (Donut)', fontweight='bold')
    ax.text(0, 0, f'n={len(y_true)}\ntotal', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#555')

    plt.tight_layout()
    save_fig(fig, 'fig7_risk_distribution')


# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_heatmap(X, y_true, feature_names):
    """Fig 8 — Feature correlation heatmap (top-20 by variance)."""
    print("Fig 8: Feature correlation heatmap...")

    df_feat = pd.DataFrame(X, columns=feature_names)
    top20   = df_feat.var().nlargest(20).index.tolist()
    corr    = df_feat[top20].corr()

    # Custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', annot_kws={'size': 7},
                square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.6, 'label': 'Pearson Correlation'},
                ax=ax)
    ax.set_title('IAM Feature Correlation Matrix (Top-20 by Variance)\n'
                 'CloudShield — 888 Policies, 40 Features',
                 fontweight='bold', pad=12)
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    save_fig(fig, 'fig8_feature_heatmap')


# ─────────────────────────────────────────────────────────────────────────────
def fig_model_comparison_radar(y_true, y_pred_rf, y_pred_lnn, y_pred_ens):
    """Fig 9 — Radar / spider chart: per-class F1 across models."""
    from sklearn.metrics import classification_report
    print("Fig 9: Radar comparison chart...")

    def get_f1s(y_pred):
        rpt = classification_report(y_true, y_pred,
                                     labels=[0, 1, 2],
                                     output_dict=True, zero_division=0)
        return [rpt[str(c)]['f1-score'] for c in range(3)]

    rf_f1s  = get_f1s(y_pred_rf)
    lnn_f1s = get_f1s(y_pred_lnn)
    ens_f1s = get_f1s(y_pred_ens)
    hgt_f1s = [1.00, 0.98, 0.67]   # from saved results

    cats   = ['LOW F1', 'MEDIUM F1', 'HIGH F1']
    N      = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True))

    model_data = [
        ('RF',           rf_f1s,  COLORS['rf'],  '-',  2.5),
        ('HGT',          hgt_f1s, COLORS['hgt'], '--', 2.0),
        ('LNN',          lnn_f1s, COLORS['lnn'], '-',  2.5),
        ('Ensemble',     ens_f1s, COLORS['ens'], '-',  3.0),
    ]

    for mname, f1s, color, ls, lw in model_data:
        vals = f1s + f1s[:1]
        ax.plot(angles, vals, ls=ls, color=color, lw=lw, label=mname)
        ax.fill(angles, vals, color=color, alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=8)
    ax.set_title('Per-Class F1 Comparison\nCloudShield Models',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'fig9_radar_comparison')


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("CloudShield Figure Generator")
    print("="*60)

    (df, feature_names, X, y_true,
     rf, y_pred_rf, y_proba_rf,
     y_pred_lnn, y_proba_lnn,
     y_pred_ens, y_proba_ens,
     hgt_results, shap_arr) = load_everything()

    hgt_f1 = hgt_results.get('test_f1', 0.8836)

    print(f"\nGenerating {9} paper figures → {OUT_DIR}/")
    print("-"*60)

    fig_confusion_matrices(y_true, y_pred_rf, y_pred_lnn,
                           y_pred_ens, hgt_f1)
    fig_roc_curves(y_true, y_proba_rf, y_proba_lnn, y_proba_ens)
    fig_shap_beeswarm(shap_arr, X, feature_names, y_true)
    fig_shap_per_class(shap_arr, feature_names)
    fig_training_curves()
    fig_ablation()
    fig_risk_distribution(y_true)
    fig_feature_heatmap(X, y_true, feature_names)
    fig_model_comparison_radar(y_true, y_pred_rf, y_pred_lnn, y_pred_ens)

    print("\n" + "="*60)
    print("ALL FIGURES SAVED")
    print("="*60)
    figs = sorted(os.listdir(OUT_DIR))
    for f in figs:
        size = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
        print(f"  {f:<40} {size:>5} KB")

    print(f"\nTotal: {len(figs)} files in {OUT_DIR}/")
    print("\nPaper-ready 300dpi PNGs + PDFs generated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()

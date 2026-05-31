"""
Visualisations for the 20260515 new-prompt medical bias run.

Plots written to rescoring_out/plots/:
  01_type_metrics.png        – kappa / bal-acc / macro-F1 grouped bar
  02_per_class_recall.png    – recall per gold class, heatmap + grouped bar
  03_marginal_rates.png      – stacked bar of predicted-class fractions (class collapse)
  04_per_type_accuracy.png   – Explicit / Implicit / None accuracy per model
  05_confusion_matrices.png  – 7-panel normalised confusion matrices
  06_category_heatmap.png    – per-category accuracy heatmap
  07_category_summary.png    – absolute/conditional category acc + macro F1 bar
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
RUN = os.path.join(BASE, 'results',
                   '20260515-004108_tasks_phi_2.7b-llama3.2_3b-plus5_s42_lall')
RESC = os.path.join(BASE, 'rescoring_out')
OUT  = os.path.join(RESC, 'plots')
os.makedirs(OUT, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
type_sum  = pd.read_csv(f'{RESC}/type_summary.csv')
marg      = pd.read_csv(f'{RESC}/marginal_rates.csv')
cat_sum   = pd.read_csv(f'{RESC}/category_summary.csv')
per_type  = pd.read_csv(f'{RUN}/medical_bias_per_type.csv', keep_default_na=False)
per_cat   = pd.read_csv(f'{RUN}/medical_bias_per_category.csv')
items     = pd.read_csv(f'{RUN}/medical_bias_items.csv', keep_default_na=False)

MODELS  = type_sum['model'].tolist()
LABELS  = ['Explicit', 'Implicit', 'None']
CATS    = ['Age', 'Ethnicity', 'Gender', 'Lifestyle', 'Region', 'Socioeconomic']
SHORT   = {m: m.split(':')[0] for m in MODELS}   # display labels

PALETTE = sns.color_palette('tab10', len(MODELS))
MODEL_COLOR = dict(zip(MODELS, PALETTE))

# ── helpers ───────────────────────────────────────────────────────────────────
def save(fig, name):
    path = f'{OUT}/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def bar_group(ax, data, models, short, colors, ylabel, ylim=(0, 1)):
    x = np.arange(len(data.columns))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = data.loc[m].values
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, width=w,
                      color=colors[m], label=short[m], alpha=0.85, edgecolor='white')
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(data.columns, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axhline(0, color='black', linewidth=0.6)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc='upper right')


# ── 01  Type-level metrics ────────────────────────────────────────────────────
print('01 type metrics …')
metrics = ['cohen_kappa', 'balanced_accuracy', 'macro_f1', 'weighted_f1']
labels  = ['Cohen κ', 'Balanced Acc', 'Macro F1', 'Weighted F1']

data01 = type_sum.set_index('model')[metrics].rename(columns=dict(zip(metrics, labels)))

fig, ax = plt.subplots(figsize=(13, 5))
bar_group(ax, data01, MODELS, SHORT, MODEL_COLOR, 'Score', ylim=(0, 1.05))
ax.set_title('Type-level metrics per model (new-prompt run, May 2026)', fontsize=13)
save(fig, '01_type_metrics.png')

# ── 02  Per-class recall ──────────────────────────────────────────────────────
print('02 per-class recall …')
recall_cols = ['recall_Explicit', 'recall_Implicit', 'recall_None']
recall_df = type_sum.set_index('model')[recall_cols]
recall_df.columns = LABELS

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# heatmap
sns.heatmap(recall_df, ax=axes[0], annot=True, fmt='.2f',
            cmap='RdYlGn', vmin=0, vmax=1,
            linewidths=0.4, linecolor='white',
            yticklabels=[SHORT[m] for m in MODELS])
axes[0].set_title('Recall per gold class (heatmap)', fontsize=12)
axes[0].set_xlabel('Predicted as correct (recall)', fontsize=10)

# grouped bar
bar_group(axes[1], recall_df, MODELS, SHORT, MODEL_COLOR,
          'Recall', ylim=(0, 1.08))
axes[1].set_title('Recall per gold class (bar)', fontsize=12)

fig.suptitle('Per-class recall – gold Explicit / Implicit / None', fontsize=13, y=1.01)
save(fig, '02_per_class_recall.png')

# ── 03  Marginal prediction rates (class collapse) ────────────────────────────
print('03 marginal rates …')
marg_vals = marg.set_index('model')[['pred_Explicit_rate',
                                     'pred_Implicit_rate',
                                     'pred_None_rate']]
marg_vals.columns = LABELS
marg_vals.index   = [SHORT[m] for m in marg_vals.index]

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(marg_vals))
colors_class = {'Explicit': '#d62728', 'Implicit': '#1f77b4', 'None': '#2ca02c'}

for lab in LABELS:
    vals = marg_vals[lab].values
    bars = ax.bar(marg_vals.index, vals, bottom=bottom,
                  label=lab, color=colors_class[lab], alpha=0.85, edgecolor='white')
    for b, v, bot in zip(bars, vals, bottom):
        if v > 0.04:
            ax.text(b.get_x() + b.get_width() / 2, bot + v / 2,
                    f'{v:.1%}', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
    bottom += vals

ax.axhline(1 / 3, color='black', linewidth=1, linestyle='--', label='Uniform baseline')
ax.set_ylabel('Fraction of predictions', fontsize=11)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_title('Marginal prediction rates – class collapse diagnostic', fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
save(fig, '03_marginal_rates.png')

# ── 04  Per-type accuracy (Explicit / Implicit / None) ───────────────────────
print('04 per-type accuracy …')
pt = per_type.pivot(index='model', columns='bias_type', values='type_accuracy')
pt = pt[LABELS]                         # fix column order

fig, ax = plt.subplots(figsize=(13, 5))
bar_group(ax, pt, MODELS, SHORT, MODEL_COLOR, 'Recall / Accuracy', ylim=(0, 1.12))
ax.set_title('Per-type recall (Explicit / Implicit / None) per model', fontsize=13)
save(fig, '04_per_type_accuracy.png')

# ── 05  Confusion matrices ────────────────────────────────────────────────────
print('05 confusion matrices …')

LABEL_SHORT = ['Explicit', 'Implicit', 'None']

# Row 1: 4 models   Row 2: 3 models centred (pad with invisible axes)
TOP_N, BOT_N = 4, 3
fig = plt.figure(figsize=(28, 22))
fig.suptitle(
    'Confusion Matrices — row-normalised\n'
    'Each cell: count  (row %)',
    fontsize=17, fontweight='bold', y=0.99,
)

gs_top = fig.add_gridspec(1, TOP_N, top=0.93, bottom=0.52,
                           left=0.04, right=0.93, wspace=0.30)
gs_bot = fig.add_gridspec(1, BOT_N, top=0.46, bottom=0.05,
                           left=0.18, right=0.79, wspace=0.30)

axes_list = (
    [fig.add_subplot(gs_top[0, c]) for c in range(TOP_N)] +
    [fig.add_subplot(gs_bot[0, c]) for c in range(BOT_N)]
)

for ax, m in zip(axes_list, MODELS):
    safe = m.replace(':', '_').replace('.', '_')
    cm_df = pd.read_csv(f'{RESC}/confusion_{safe}.csv', index_col=0)
    cm = cm_df.values.astype(float)

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    annot = np.empty_like(cm, dtype=object)
    for r in range(3):
        for c in range(3):
            annot[r, c] = f'{int(cm[r, c])}\n({cm_norm[r, c]:.0%})'

    sns.heatmap(
        cm_norm, ax=ax,
        annot=annot, fmt='',
        cmap='Blues', vmin=0, vmax=1,
        linewidths=2, linecolor='white',
        xticklabels=LABEL_SHORT, yticklabels=LABEL_SHORT,
        square=True, cbar=False,
        annot_kws={'fontsize': 13, 'va': 'center', 'linespacing': 1.4},
    )

    # Highlight diagonal with a thick border
    for i in range(3):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1, fill=False, edgecolor='#1a252f', lw=3,
        ))

    ax.set_title(m, fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel('Predicted label', fontsize=13, labelpad=8)
    ax.set_ylabel('Gold label', fontsize=13, labelpad=8)
    ax.tick_params(axis='both', labelsize=13)
    ax.xaxis.set_tick_params(rotation=15)
    ax.yaxis.set_tick_params(rotation=0)

# Shared colourbar
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.95, 0.08, 0.013, 0.84])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Row-normalised rate', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=11)

save(fig, '05_confusion_matrices.png')

# ── 06  Category accuracy heatmap ─────────────────────────────────────────────
print('06 category heatmap …')
cat_pivot = per_cat.pivot(index='model', columns='category', values='category_accuracy')
cat_pivot = cat_pivot[CATS]
cat_pivot.index = [SHORT[m] for m in cat_pivot.index]

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(cat_pivot, ax=ax, annot=True, fmt='.2f',
            cmap='RdYlGn', vmin=0, vmax=1,
            linewidths=0.4, linecolor='white')
ax.set_title('Category accuracy per model (Implicit items only, n=943 per model)',
             fontsize=12)
ax.set_xlabel('Bias category', fontsize=11)
ax.set_ylabel('')
save(fig, '06_category_heatmap.png')

# ── 07  Category summary bar ──────────────────────────────────────────────────
print('07 category summary …')
cs = cat_sum.set_index('model')[['absolute_category_acc',
                                  'conditional_category_acc',
                                  'category_macro_f1']]
cs.columns = ['Abs Cat Acc', 'Cond Cat Acc', 'Cat Macro F1']

fig, ax = plt.subplots(figsize=(13, 5))
bar_group(ax, cs, MODELS, SHORT, MODEL_COLOR, 'Score', ylim=(0, 1.08))
ax.set_title('Category metrics (Implicit-only items)\n'
             'Abs = all items; Cond = given model predicted Implicit; '
             'Macro F1 over 6 categories',
             fontsize=11)
save(fig, '07_category_summary.png')

print(f'\nAll 7 plots written to {OUT}/')

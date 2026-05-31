"""
Rescoring of the new-prompt medical bias run.

Outputs:
  - rescoring_out/type_summary.csv       : per-model type metrics (kappa, bal_acc, macro_F1, etc.)
  - rescoring_out/marginal_rates.csv     : per-model fraction of predictions in each class
  - rescoring_out/confusion_<model>.csv  : per-model 3x3 confusion matrix
  - rescoring_out/category_summary.csv   : per-model category metrics (Implicit-only, n=943)
  - rescoring_out/per_class_recall.csv   : per-model recall on each gold class
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, cohen_kappa_score, balanced_accuracy_score,
    f1_score,
)

ITEMS_PATH = os.path.join(
    os.path.dirname(__file__),
    'results',
    '20260515-004108_tasks_phi_2.7b-llama3.2_3b-plus5_s42_lall',
    'medical_bias_items.csv',
)
OUT = os.path.join(os.path.dirname(__file__), 'rescoring_out')
os.makedirs(OUT, exist_ok=True)

# Keep "None" as a literal label (not NaN).
df = pd.read_csv(ITEMS_PATH, keep_default_na=False)

LABELS = ['Explicit', 'Implicit', 'None']
MODELS = ['phi:2.7b', 'llama3.2:3b', 'qwen3:4b', 'gemma3:4b',
          'mistral:7b', 'deepseek-r1:8b', 'gpt-oss:20b']

# ----- 1. Type-level metrics -----------------------------------------------
rows_type = []
rows_marg = []
rows_recall = []

for m in MODELS:
    sub = df[df['model'] == m].copy()
    n_total = len(sub)
    # Anything outside the closed set is a parse failure.
    sub_valid = sub[sub['pred_type'].isin(LABELS)].copy()
    n_valid = len(sub_valid)
    valid_rate = n_valid / n_total

    y_true = sub_valid['gold_type'].tolist()
    y_pred = sub_valid['pred_type'].tolist()

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f'gold_{l}' for l in LABELS],
        columns=[f'pred_{l}' for l in LABELS],
    )
    safe_name = m.replace(':', '_').replace('.', '_')
    cm_df.to_csv(f'{OUT}/confusion_{safe_name}.csv')

    strict_acc = (np.array(y_true) == np.array(y_pred)).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=LABELS)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average='weighted', zero_division=0)

    row_totals = cm.sum(axis=1)
    recall_per_class = np.divide(
        np.diag(cm), row_totals,
        out=np.zeros_like(np.diag(cm), dtype=float),
        where=row_totals != 0,
    )

    col_totals = cm.sum(axis=0)
    precision_per_class = np.divide(
        np.diag(cm), col_totals,
        out=np.zeros_like(np.diag(cm), dtype=float),
        where=col_totals != 0,
    )

    rows_type.append({
        'model': m,
        'n_total': n_total,
        'n_valid': n_valid,
        'valid_rate': round(valid_rate, 4),
        'strict_accuracy': round(strict_acc, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'cohen_kappa': round(kappa, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'recall_Explicit': round(recall_per_class[0], 4),
        'recall_Implicit': round(recall_per_class[1], 4),
        'recall_None': round(recall_per_class[2], 4),
        'precision_Explicit': round(precision_per_class[0], 4),
        'precision_Implicit': round(precision_per_class[1], 4),
        'precision_None': round(precision_per_class[2], 4),
    })

    y_pred_arr = np.array(y_pred)
    rows_marg.append({
        'model': m,
        'pred_Explicit_rate': round((y_pred_arr == 'Explicit').mean(), 4),
        'pred_Implicit_rate': round((y_pred_arr == 'Implicit').mean(), 4),
        'pred_None_rate': round((y_pred_arr == 'None').mean(), 4),
        'max_marginal': round(max(
            (y_pred_arr == 'Explicit').mean(),
            (y_pred_arr == 'Implicit').mean(),
            (y_pred_arr == 'None').mean(),
        ), 4),
    })

    for lab, r in zip(LABELS, recall_per_class):
        rows_recall.append({'model': m, 'gold_class': lab, 'recall': round(r, 4)})

pd.DataFrame(rows_type).to_csv(f'{OUT}/type_summary.csv', index=False)
pd.DataFrame(rows_marg).to_csv(f'{OUT}/marginal_rates.csv', index=False)
pd.DataFrame(rows_recall).to_csv(f'{OUT}/per_class_recall.csv', index=False)

# ----- 2. Category metrics (Implicit only, n=943) --------------------------
CATEGORIES = ['Age', 'Ethnicity', 'Gender', 'Lifestyle', 'Region', 'Socioeconomic']
rows_cat = []

for m in MODELS:
    sub = df[(df['model'] == m) & (df['category_scored'] == 1)].copy()
    n_imp = len(sub)
    conditional = sub[sub['pred_type'] == 'Implicit']
    n_cond = len(conditional)

    abs_cat_acc = sub['correct_category'].mean() if n_imp else float('nan')
    cond_cat_acc = conditional['correct_category'].mean() if n_cond else float('nan')

    y_true_cat = sub['gold_category'].tolist()
    y_pred_cat = (
        sub['pred_category']
        .apply(lambda x: x if x in CATEGORIES else 'OTHER')
        .tolist()
    )
    cat_macro_f1 = f1_score(
        y_true_cat, y_pred_cat,
        labels=CATEGORIES, average='macro', zero_division=0,
    )

    rows_cat.append({
        'model': m,
        'n_implicit_items': n_imp,
        'n_predicted_implicit': n_cond,
        'absolute_category_acc': round(abs_cat_acc, 4),
        'conditional_category_acc': (
            round(cond_cat_acc, 4) if not np.isnan(cond_cat_acc) else float('nan')
        ),
        'category_macro_f1': round(cat_macro_f1, 4),
    })

pd.DataFrame(rows_cat).to_csv(f'{OUT}/category_summary.csv', index=False)

# ----- 3. Print summary ----------------------------------------------------
print('\n===== TYPE SUMMARY =====')
print(pd.DataFrame(rows_type).to_string(index=False))
print('\n===== MARGINAL PREDICTION RATES =====')
print(pd.DataFrame(rows_marg).to_string(index=False))
print('\n===== CATEGORY SUMMARY (Implicit only, n=943) =====')
print(pd.DataFrame(rows_cat).to_string(index=False))
print(f'\nAll outputs written to {OUT}/')

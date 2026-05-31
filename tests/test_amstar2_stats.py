from pathlib import Path

import pandas as pd

from analysis.amstar2_stats import run_analysis


def test_run_analysis_writes_expected_outputs(tmp_path: Path):
    item_rows = [
        {
            "model": "model_a",
            "article_id": "article_1",
            "run": 0,
            "item": 1,
            "is_critical": False,
            "predicted": "yes",
            "gold": "yes",
            "exact_match": True,
            "lenient_match": True,
        },
        {
            "model": "model_b",
            "article_id": "article_1",
            "run": 0,
            "item": 1,
            "is_critical": False,
            "predicted": "no",
            "gold": "yes",
            "exact_match": False,
            "lenient_match": False,
        },
        {
            "model": "model_a",
            "article_id": "article_1",
            "run": 0,
            "item": 2,
            "is_critical": True,
            "predicted": "no",
            "gold": "no",
            "exact_match": True,
            "lenient_match": True,
        },
        {
            "model": "model_b",
            "article_id": "article_1",
            "run": 0,
            "item": 2,
            "is_critical": True,
            "predicted": "yes",
            "gold": "no",
            "exact_match": False,
            "lenient_match": False,
        },
    ]
    article_rows = [
        {
            "model": "model_a",
            "article_id": "article_1",
            "run": 0,
            "item_accuracy": 1.0,
            "item_lenient_accuracy": 1.0,
            "overall_exact_match": True,
            "parse_success": True,
            "raw_response": '{"item_1":"yes","item_2":"no"}',
        },
        {
            "model": "model_b",
            "article_id": "article_1",
            "run": 0,
            "item_accuracy": 0.0,
            "item_lenient_accuracy": 0.0,
            "overall_exact_match": False,
            "parse_success": True,
            "raw_response": '{"item_1":"no","item_2":"yes"}',
        },
    ]

    pd.DataFrame(item_rows).to_csv(tmp_path / "amstar2_item_details.csv", index=False)
    pd.DataFrame(article_rows).to_csv(tmp_path / "amstar2_per_article.csv", index=False)

    written = run_analysis(tmp_path, n_bootstrap=20, n_permutations=20, seed=1)

    assert set(written) == {
        "amstar2_model_ci.csv",
        "amstar2_pairwise_tests.csv",
        "amstar2_confusion_matrix.csv",
        "amstar2_variance_check.csv",
    }
    for path in written.values():
        assert path.exists()
        assert path.stat().st_size > 0

    ci = pd.read_csv(tmp_path / "amstar2_model_ci.csv")
    assert {"model", "metric", "estimate", "ci_low", "ci_high"}.issubset(ci.columns)

    tests = pd.read_csv(tmp_path / "amstar2_pairwise_tests.csv")
    assert {"paired_permutation_sign_flip", "mcnemar_exact"}.issubset(set(tests["test"]))

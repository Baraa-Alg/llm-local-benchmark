from metrics.bertscore_metric import BertScoreMetric


def test_bertscore_identical_strings_above_threshold():
    metric = BertScoreMetric(strict=True)
    text = "The patient was diagnosed with hypertension and started treatment."
    score = metric.compute(text, text)
    assert score > 0.1

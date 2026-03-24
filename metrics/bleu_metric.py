from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import traceback


def _handle_metric_exception(metric_name: str, exc: Exception, strict: bool):
    print(f"[{metric_name}] metric computation failed: {type(exc).__name__}: {exc}")
    print(traceback.format_exc())
    if strict:
        raise RuntimeError(f"{metric_name} metric computation failed") from exc

class BLEUMetric:
    """Compute BLEU score between reference and generated text."""

    def __init__(self, strict: bool = False):
        self.smooth_fn = SmoothingFunction().method1
        self.name = "BLEU"
        self.strict = strict

    def compute(self, reference: str, candidate: str) -> float:
        if not reference or not candidate:
            return 0.0
        try:
            score_val = sentence_bleu(
                [reference.split()],
                candidate.split(),
                smoothing_function=self.smooth_fn
            )
            return round(score_val, 4)
        except Exception as exc:
            _handle_metric_exception(self.name, exc, self.strict)
            return 0.0

class RougeMetric:
    name = "ROUGE_L"

    def __init__(self, strict: bool = False):
        self.strict = strict

    def compute(self, ref, hyp):
        if not ref or not hyp:
            return 0.0
        rouge = Rouge()
        try:
            scores = rouge.get_scores(hyp, ref)
            return scores[0]["rouge-l"]["f"]
        except Exception as exc:
            _handle_metric_exception(self.name, exc, self.strict)
            return 0.0

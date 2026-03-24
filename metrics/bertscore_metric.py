import traceback


class BertScoreMetric:
    name = "BERTScore"

    def __init__(self, strict: bool = False):
        self.strict = strict

    def compute(self, ref, hyp):
        if not ref or not hyp:
            return 0.0

        try:
            from bert_score import score
        except ImportError as e:
            if self.strict:
                raise RuntimeError("BERTScore unavailable") from e
            else:
                print("[WARN] BERTScore unavailable; returning NaN")
                return float("nan")

        try:
            _, _, f1 = score([hyp], [ref], lang="en", verbose=False)
            return f1.mean().item()
        except Exception as exc:
            print(f"[{self.name}] metric computation failed: {type(exc).__name__}: {exc}")
            print(traceback.format_exc())
            if self.strict:
                raise RuntimeError(f"{self.name} metric computation failed") from exc
            return 0.0

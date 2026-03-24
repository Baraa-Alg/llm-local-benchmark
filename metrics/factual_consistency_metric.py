import json
import re
import traceback


class FactualConsistencyMetric:
    """
    Sentence-level factual support metric for summarization.

    Steps:
    - Split summary into sentences.
    - Split source document into sentences.
    - For each summary sentence, compute max semantic similarity vs all source sentences.
    - Mark unsupported if max similarity < threshold.
    - Final score = supported / total summary sentences.
    """

    name = "FactualConsistency"

    _sent_re = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        threshold: float = 0.75,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        strict: bool = False,
    ):
        self.threshold = threshold
        self.model_name = model_name
        self.strict = strict

    def _split_sentences(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        return [s.strip() for s in self._sent_re.split(text) if s.strip()]

    def compute(self, source_document: str, generated_summary: str):
        return self._compute_core(source_document, generated_summary)

    def compute_with_context(
        self,
        source_document: str,
        generated_summary: str,
        reference_summary: str | None = None,
        example: dict | None = None,
    ):
        return self._compute_core(source_document, generated_summary)

    def _compute_core(self, source_document: str, generated_summary: str):
        summary_sents = self._split_sentences(generated_summary)
        source_sents = self._split_sentences(source_document)
        if not summary_sents:
            return {
                "score": 0.0,
                "supported_flags": [],
                "max_similarities": [],
                "threshold": self.threshold,
            }
        if not source_sents:
            flags = [False] * len(summary_sents)
            sims = [0.0] * len(summary_sents)
            return {
                "score": 0.0,
                "supported_flags": flags,
                "max_similarities": sims,
                "threshold": self.threshold,
            }

        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError as e:
            print("[WARN] sentence-transformers unavailable for FactualConsistency; returning NaN")
            if self.strict:
                raise RuntimeError("FactualConsistency unavailable") from e
            return {
                "score": float("nan"),
                "supported_flags": [],
                "max_similarities": [],
                "threshold": self.threshold,
            }

        try:
            model = SentenceTransformer(self.model_name)
            src_emb = model.encode(source_sents, convert_to_tensor=True)
            sum_emb = model.encode(summary_sents, convert_to_tensor=True)
            sim_matrix = util.cos_sim(sum_emb, src_emb)

            max_sims = [float(sim_matrix[i].max().item()) for i in range(len(summary_sents))]
            supported_flags = [s >= self.threshold for s in max_sims]
            supported_count = sum(1 for x in supported_flags if x)
            score = supported_count / max(1, len(summary_sents))

            return {
                "score": round(score, 4),
                "supported_flags": supported_flags,
                "max_similarities": [round(s, 4) for s in max_sims],
                "threshold": self.threshold,
            }
        except Exception as exc:
            print(f"[{self.name}] metric computation failed: {type(exc).__name__}: {exc}")
            print(traceback.format_exc())
            if self.strict:
                raise RuntimeError(f"{self.name} metric computation failed") from exc
            return {
                "score": 0.0,
                "supported_flags": [],
                "max_similarities": [],
                "threshold": self.threshold,
            }

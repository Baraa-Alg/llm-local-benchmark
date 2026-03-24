import re
from typing import List


class FaithfulnessJaccard:
    """
    Lightweight faithfulness proxy using Jaccard overlap between
    content-word sets in reference and candidate sentences.

    compute(reference, candidate) -> float in [0,1]
    - Splits both texts into sentences.
    - For each candidate sentence, finds the max Jaccard similarity to any
      reference sentence using lowercase alphabetic tokens minus stopwords.
    - Returns the average of these maxima.

    If reference or candidate is empty, returns 0.0.
    """

    name = "FaithfulnessJaccard"

    _sent_re = re.compile(r"(?<=[.!?])\s+")
    _word_re = re.compile(r"[A-Za-z']+")
    _stop = {
        "a","an","the","and","or","but","if","then","is","are","was","were","be","been","being",
        "of","to","in","on","for","with","as","by","at","from","that","this","these","those",
        "it","its","their","they","we","our","you","your","he","she","his","her","them"
    }

    def _sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        parts = self._sent_re.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _tokens(self, text: str) -> set:
        toks = [t.lower() for t in self._word_re.findall(text)]
        return {t for t in toks if t not in self._stop}

    def _jaccard(self, a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b) or 1
        return inter / union

    def compute(self, reference: str, candidate: str) -> float:
        if not reference or not candidate:
            return 0.0
        ref_sents = self._sentences(reference)
        hyp_sents = self._sentences(candidate)
        if not ref_sents or not hyp_sents:
            return 0.0
        ref_sets = [self._tokens(s) for s in ref_sents]
        scores = []
        for h in hyp_sents:
            hset = self._tokens(h)
            best = 0.0
            for rset in ref_sets:
                best = max(best, self._jaccard(hset, rset))
            scores.append(best)
        return round(sum(scores) / len(scores), 4) if scores else 0.0


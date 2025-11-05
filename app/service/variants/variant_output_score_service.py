from typing import List, Dict, Any, Optional
import re
import math

class VariantOutputScoreService:
    """
    Normalized scoring on a 0..5 scale derived from a 10-point rubric:

    Rubric (0..10):
    1) Citations coverage (0..2):
       - +1 if at least one bracketed citation like [doc-id]
       - +1 if two or more citations
    2) Distinct parent ids (0..1):
       - +1 if at least two distinct parent ids appear
    3) Numeric completeness (0..2):
       - +1 if any number is present
       - +1 if a pair-year pattern is present (both 2019 and 2018 figures)
    4) Delta expression (0..1):
       - +1 if the answer contains delta language (increase/decrease/∆ or 'delta') or a computed difference phrase
    5) Risk quotes (0..2):
       - +1 if at least one quoted risk sentence with a citation
       - +1 if at least two quoted risk sentences each with a citation
    6) Length and fluency (0..1):
       - +0.5 if length in [200..1400]
       - +0.5 if sentence-like punctuation appears ('.' present >= 3)
    7) Question overlap (0..1):
       - +1 if at least one of the key terms from question appears

    Final score = round((rubric_total / 10.0) * 5.0, 3)
    """

    # ---------- Helpers ----------
    @staticmethod
    def _extract_ids(text: str) -> List[str]:
        if not text:
            return []
        return list(dict.fromkeys(re.findall(r"\[([^\[\]]+?)\]", text)))

    @staticmethod
    def _numbers(text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\d[\d,\.]*", text)

    @staticmethod
    def _has_year_pair(text: str) -> bool:
        if not text:
            return False
        # crude signals that both 2019 and 2018 numbers likely appear
        has_2019 = "2019" in text
        has_2018 = "2018" in text
        return has_2019 and has_2018 and bool(re.search(r"\d", text))

    @staticmethod
    def _has_delta_language(text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(kw in t for kw in ["increase", "decrease", "delta", "change", "rose", "declined", "up ", "down " , "grew", "reduction", "∆"])

    @staticmethod
    def _quoted_segments_with_cites(text: str) -> int:
        if not text:
            return 0
        # Count quoted segments that also have a citation nearby
        # e.g., "...." [doc-id]
        matches = re.findall(r"\"([^\"]+?)\"\s*\[[^\[\]]+?\]", text)
        return len(matches)

    @staticmethod
    def _sentence_like(text: str) -> bool:
        if not text:
            return False
        # at least 3 periods as crude fluency proxy
        return text.count(".") >= 3

    @staticmethod
    def _question_terms(question: Optional[str]) -> List[str]:
        if not question:
            return []
        terms = re.findall(r"[A-Za-z]{4,}", question.lower())
        ignore = {"with","from","that","this","those","these","which","about","into","over","under","between","among","total","year","years"}
        return [t for t in terms if t not in ignore][:8]

    # ---------- Public API ----------
    @staticmethod
    def score_scalar(
            answer: str,
            citations: List[str],
            scoring_model: str = "heuristic_v1",
            allowed_ids: Optional[List[str]] = None,
            question: Optional[str] = None
    ) -> float:
        if not answer:
            return 0.0

        total = 0.0

        # (1) Citations coverage (0..2)
        ids_in_answer = VariantOutputScoreService._extract_ids(answer)
        if len(ids_in_answer) >= 1:
            total += 1.0
        if len(ids_in_answer) >= 2:
            total += 1.0

        # (2) Distinct parent ids (0..1)
        if len(set(ids_in_answer)) >= 2:
            total += 1.0

        # (3) Numeric completeness (0..2)
        if VariantOutputScoreService._numbers(answer):
            total += 1.0
        if VariantOutputScoreService._has_year_pair(answer):
            total += 1.0

        # (4) Delta expression (0..1)
        if VariantOutputScoreService._has_delta_language(answer):
            total += 1.0

        # (5) Risk quotes (0..2)
        qcount = VariantOutputScoreService._quoted_segments_with_cites(answer)
        if qcount >= 1:
            total += 1.0
        if qcount >= 2:
            total += 2.0 - 1.0  # add another +1 (keep additive explicit)

        # (6) Length and fluency (0..1)
        L = len(answer or "")
        if 200 <= L <= 1400:
            total += 0.5
        if VariantOutputScoreService._sentence_like(answer):
            total += 0.5

        # (7) Question overlap (0..1)
        q_terms = VariantOutputScoreService._question_terms(question)
        if q_terms and any(t in (answer.lower()) for t in q_terms):
            total += 1.0

        # Normalize to 0..5
        score = round((total / 10.0) * 5.0, 3)
        # guardrails
        score = max(0.0, min(5.0, score))
        return score

    @staticmethod
    def score_breakdown(
            answer: str,
            citations: List[str],
            scoring_model: str = "heuristic_v1",
            allowed_ids: Optional[List[str]] = None,
            question: Optional[str] = None
    ) -> Dict[str, Any]:
        ids_in_answer = VariantOutputScoreService._extract_ids(answer)
        breakdown: Dict[str, Any] = {
            "ids_in_answer": ids_in_answer,
            "distinct_parent_ids": len(set(ids_in_answer)),
            "has_numbers": bool(VariantOutputScoreService._numbers(answer)),
            "has_year_pair": VariantOutputScoreService._has_year_pair(answer),
            "has_delta_language": VariantOutputScoreService._has_delta_language(answer),
            "quoted_risk_with_cite_count": VariantOutputScoreService._quoted_segments_with_cites(answer),
            "length": len(answer or ""),
            "sentence_like": VariantOutputScoreService._sentence_like(answer),
            "question_terms_overlap": VariantOutputScoreService._question_terms(question),
        }
        breakdown["score"] = VariantOutputScoreService.score_scalar(
            answer=answer,
            citations=citations,
            scoring_model=scoring_model,
            allowed_ids=allowed_ids,
            question=question
        )
        return breakdown

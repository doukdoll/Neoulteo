"""검색 평가 지표. 정의는 rag-architect 스킬의 references/rag_evaluation_framework.md 를 따른다."""
from __future__ import annotations

import math


def hit_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0


def precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return sum(1 for r in ranked[:k] if r in relevant) / k


def mrr_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    for i, r in enumerate(ranked[:k]):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


# 질의셋별로 의미 있는 지표만 집계한다.
#  - semantic / name_lookup: 정답이 1개(동명 문서 포함 소수) → Hit@k, MRR, nDCG
#  - region_category: 정답이 수십 개 → Precision@k (상위 결과가 얼마나 조건에 맞는가)
SET_METRICS = {
    "semantic": ["hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10"],
    "name_lookup": ["hit@1", "hit@5", "mrr@10"],
    "region_category": ["p@5", "p@10", "hit@5"],
    "rc_holdout": ["p@5", "p@10", "hit@5"],  # region_category와 같은 방식, 다른 표현/지역 (scripts/build_holdout_rc.py)
}


def score(metric: str, ranked: list[int], relevant: set[int]) -> float:
    name, k = metric.split("@")
    k = int(k)
    return {"hit": hit_at_k, "p": precision_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k}[name](ranked, relevant, k)

# 평가 결과: baseline (2026-11-25)

코퍼스 50,915건. 지연시간은 Python 재현 기준(단일 프로세스, GPU 임베딩).

## semantic (n=118)

| retriever | hit@1 | hit@5 | hit@10 | mrr@10 | ndcg@10 | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|---|---|
| spring-lexical | 0.7712 | 0.9492 | 0.9576 | 0.8375 | 0.8607 | 215.8 | 379.0 |
| spring-hybrid | 0.8051 | 0.9407 | 0.9661 | 0.8681 | 0.8858 | 269.2 | 434.3 |
| fastapi-dense | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 59.4 | 84.4 |

## name_lookup (n=60)

| retriever | hit@1 | hit@5 | mrr@10 | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|
| spring-lexical | 0.5167 | 0.6 | 0.5589 | 78.8 | 229.8 |
| spring-hybrid | 0.5333 | 0.6333 | 0.5736 | 130.2 | 255.7 |
| fastapi-dense | 0.0 | 0.0 | 0.0021 | 50.3 | 78.7 |

## region_category (n=60)

| retriever | p@5 | p@10 | hit@5 | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|
| spring-lexical | 0.02 | 0.0583 | 0.05 | 156.3 | 243.8 |
| spring-hybrid | 0.2033 | 0.21 | 0.4667 | 202.9 | 301.0 |
| fastapi-dense | 0.0 | 0.0 | 0.0 | 53.4 | 81.9 |

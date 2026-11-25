# Neoulteo RAG 검색 평가 (Phase 1: 베이스라인)

Neoulteo 여행 챗봇이 쓰는 검색(Retrieval) 단계를 **같은 코퍼스, 같은 평가셋**으로 측정하기 위한 하네스입니다.
"RAG가 잘 동작한다"는 느낌 대신, 개선할 때마다 수치로 비교할 수 있게 하는 것이 목적입니다.

## 구성

```
rag-eval/
├─ scripts/
│  ├─ extract_corpus.py      # MySQL 덤프 → corpus.jsonl (관광지 50,915건)
│  ├─ build_auto_sets.py     # 규칙 기반 질의셋: name_lookup, region_category
│  ├─ merge_semantic_set.py  # LLM이 작성한 의미 검색 질의 검증(이름 누설/본문 복사 제거) 후 병합
│  └─ breakdown.py           # 난이도/분류별 세부 지표
├─ neoulteo_rag/
│  ├─ retrievers.py          # 현재 시스템의 검색 로직을 Python으로 재현
│  └─ metrics.py             # Hit@k, Precision@k, MRR, nDCG
├─ data/eval/*.jsonl         # 평가셋 (버전 관리 대상)
├─ results/                  # 실행 결과 (json + md)
└─ run_eval.py
```

## 실행

```bash
cd rag-eval
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU
.venv/Scripts/python -m pip install -r requirements.txt
python scripts/extract_corpus.py ../docs/database/SSAFY_TRIP_Dump.sql data/corpus.jsonl
python scripts/build_auto_sets.py data/corpus.jsonl data/eval
python scripts/merge_semantic_set.py data/corpus.jsonl data/eval_build data/eval/semantic.jsonl
.venv/Scripts/python run_eval.py --tag baseline
```

## 평가셋 (총 238문항)

| 세트 | n | 만드는 방법 | 정답 | 주 지표 |
|---|---|---|---|---|
| semantic | 118 | overview가 있는 문서를 보고 LLM이 이름 없이 묘사한 질의. 이름 3글자 누설 또는 본문 3어절 복사는 자동으로 제외(128 → 118) | 해당 장소(동명 레코드 포함) | Hit@5, MRR |
| name_lookup | 60 | 장소명 + 템플릿. 절반은 띄어쓰기를 바꿈("가회동 성당") | 같은 이름의 문서 | Hit@5 |
| region_category | 60 | "{시군구} {분류} 추천" (숙박, 음식점, 쇼핑, 레포츠, 문화시설, 축제 각 10개) | 해당 시군구 + 분류의 문서 전체 | Precision@5 |

## 재현한 대상 시스템

| 이름 | 원본 | 방식 |
|---|---|---|
| spring-lexical | `RagSearchService.java` (기본 설정 `NEOULTEO_AI_ENABLED=false`일 때의 실제 경로) | 부분 문자열 일치 + 수작업 동의어 그룹 |
| spring-hybrid | 같은 파일의 임베딩 경로 | lexical 후보 80개 안에서 `lexical + 2×cosine`. 원본은 OpenAI 임베딩이지만 API 키 없이 재현하려고 로컬 MiniLM으로 대체 |
| fastapi-dense | `ai-server/main.py` `/search` | MiniLM 임베딩 + 코사인 전수 비교. 문서는 `tour_data.txt`의 문장 템플릿. 실제 서버는 500줄만 색인하지만 여기서는 같은 방식을 5만 건 전체에 적용 |

## 베이스라인 결과 (2026-11-25)

| 세트 | 지표 | spring-lexical | spring-hybrid | fastapi-dense |
|---|---|---|---|---|
| semantic | Hit@1 / Hit@5 / MRR@10 | 0.771 / 0.949 / 0.838 | 0.805 / 0.941 / 0.868 | 0.000 / 0.000 / 0.000 |
| name_lookup | Hit@1 / Hit@5 | 0.517 / 0.600 | 0.533 / 0.633 | 0.000 / 0.000 |
| region_category | P@5 / Hit@5 | **0.020** / 0.050 | 0.203 / 0.467 | 0.000 / 0.000 |
| (참고) 지연 p50 | semantic | 216ms | 269ms | 59ms |

세부 결과: [results/baseline.md](results/baseline.md). 지연시간은 Python으로 재현한 기준이라 Java 원본과 절대값은 다릅니다. 상대 비교용입니다.

## 발견한 문제 (원인까지 확인)

1. **데이터 커버리지**: 50,915건 중 overview(설명)가 있는 문서는 **1,087건(2.1%)**뿐이고, 서울, 인천, 광주, 경기에만 있습니다. 나머지 지역은 "분위기 좋은 곳" 같은 묘사형 질문에 답할 근거가 데이터에 없습니다.
2. **분류 매핑 버그**: `contenttypes` 테이블에서 39(Tour API 정의상 **음식점**, 15,826건, 전체의 31%)가 '관광지'로 들어가 있습니다. 그래서 "맛집" 질의의 lexical P@5가 0.00입니다.
3. **IDF 없음**: lexical 점수에서 "위치", "알려줘" 같은 흔한 단어가 장소명과 같은 가중치(+0.6)를 받습니다. 동점이면 overview가 있는 문서가 먼저 나오도록 정렬되어 있어서, **장소명을 정확히 입력해도** 찾지 못합니다(exact Hit@5 0.64). 예: "백무동야영장 위치 알려줘" → 상위 결과는 4·19학생혁명기념탑, 홍대카페 등.
4. **띄어쓰기 민감**: 부분 문자열 일치라서 "마곡사천연송림욕장"으로는 "마곡사 천연송림욕장"을 찾지 못합니다(spacing Hit@5 0.56).
5. **지역+분류 질의에 약함**: "숙소"와 "숙박"이 연결되지 않고, 지역과 분류를 메타데이터 필터로 쓰지 않습니다(lexical P@5 0.02, hybrid 0.20).
6. **임베딩이 키워드 검색에 갇힘**: hybrid는 lexical 후보 안에서만 순위를 바꾸므로 semantic Hit@1이 3%p만 오릅니다. 후보가 0개이면 결과도 0개입니다.
7. **템플릿 때문에 임베딩이 붕괴**: FastAPI의 문장 템플릿이 임베딩을 지배해서 문서 간 평균 코사인 유사도가 **0.80**이 됩니다. 그 결과 특정 문서(예: "다원맛집")가 서로 무관한 질의에서 반복해서 1위가 되는 hubness가 생깁니다. 확인한 사례에서 정답 순위는 4,642~32,830위였습니다.

## 해석 시 주의

- semantic 세트의 정답은 모두 overview가 있는 문서이고, spring-lexical은 동점일 때 overview 문서를 먼저 내도록 정렬합니다. 그래서 **semantic 점수는 lexical에 유리하게 나옵니다.** 실제로 개선 여지가 큰 곳은 name_lookup과 region_category입니다.
- 의미 검색 질의는 LLM(Claude)이 작성했습니다. 이름 누설과 본문 복사는 자동으로 걸렀지만, 사람이 직접 검수한 세트는 아닙니다.

## Phase 2 목표 (다음 단계)

| 세트 | 지표 | 베이스라인 최고 | 목표 |
|---|---|---|---|
| name_lookup | Hit@5 | 0.633 | ≥ 0.95 |
| region_category | P@5 | 0.203 | ≥ 0.80 |
| semantic | Hit@1 / Hit@5 | 0.805 / 0.949 | ≥ 0.85 / 유지 |

개선 후보(한 번에 한 가지만 바꾸고 재측정): 분류 매핑 수정 → BM25(Kiwi 형태소) → 질의에서 지역/분류를 파싱해 메타데이터 필터 → 템플릿 없는 dense 검색(한국어 모델 비교) → RRF 융합 → cross-encoder 리랭커.

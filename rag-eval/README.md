# Neoulteo RAG 검색 평가 (Phase 1: 베이스라인, Phase 2: 개선 실험)

Neoulteo 여행 챗봇이 쓰는 검색(Retrieval) 단계를 **같은 코퍼스, 같은 평가셋**으로 측정하기 위한 하네스입니다.
"RAG가 잘 동작한다"는 느낌 대신, 개선할 때마다 수치로 비교할 수 있게 하는 것이 목적입니다.

## 구성

```
rag-eval/
├─ scripts/
│  ├─ extract_corpus.py      # MySQL 덤프 → corpus.jsonl (관광지 50,915건)
│  ├─ build_auto_sets.py     # 규칙 기반 질의셋: name_lookup, region_category
│  ├─ merge_semantic_set.py  # LLM이 작성한 의미 검색 질의 검증(이름 누설/본문 복사 제거) 후 병합
│  ├─ build_holdout_rc.py    # (Phase 2) region_category hold-out 세트: 다른 표현, 다른 지역
│  └─ breakdown.py           # 난이도/분류별 세부 지표
├─ neoulteo_rag/
│  ├─ retrievers.py          # 현재 시스템의 검색 로직을 Python으로 재현
│  ├─ phase2.py              # (Phase 2) BM25, 제목 매칭, 지역/분류 파서, dense, RRF, 리랭커
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
python scripts/build_holdout_rc.py data/corpus.jsonl data/eval
.venv/Scripts/python run_eval.py --retrievers spring-lexical spring-hybrid fastapi-dense --tag baseline
.venv/Scripts/python run_eval.py --retrievers s1-lexical+catfix s2-bm25 s3-bm25+title s4-bm25+title+meta     dense-bge-m3 s5-hybrid-rrf s5y-dense+meta s6-hybrid+rerank s6y-dense+meta+rerank --tag phase2
.venv/Scripts/python scripts/breakdown.py s4-bm25+title+meta s6-hybrid+rerank
```

Phase 2는 Hugging Face에서 `BAAI/bge-m3`(임베딩)와 `BAAI/bge-reranker-v2-m3`(리랭커)를 받습니다(각 약 2.2GB).
bge-m3 코퍼스 임베딩은 RTX 4050 Laptop 기준 약 6분 걸리고, 이후에는 `data/cache/`에서 읽습니다.

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

## Phase 2: 개선 실험 (2026-11-26)

### 목표와 달성 여부

| 세트 | 지표 | 베이스라인 최고 | 목표 | s6-hybrid+rerank | s6y-dense+meta+rerank |
|---|---|---|---|---|---|
| name_lookup | Hit@5 | 0.633 | ≥ 0.95 | **1.000** | **1.000** |
| region_category | P@5 | 0.203 | ≥ 0.80 | **1.000** | **1.000** |
| semantic | Hit@1 / Hit@5 | 0.805 / 0.949 | ≥ 0.85 / 유지 | **0.983 / 1.000** | **0.966 / 0.983** |
| rc_holdout (신규) | P@5 | 0.172 | (검증용) | 0.911 | **0.978** |

### 단계별 결과 (한 번에 한 가지씩 추가)

`x`/`y` 접미사는 같은 단계에서 다른 선택지를 비교한 행입니다. 전체 수치는 [results/phase2.md](results/phase2.md)에 있습니다.

| 단계 | 바꾼 것 | sem H@1 | sem H@5 | name H@5 | rc P@5 | holdout P@5 |
|---|---|---|---|---|---|---|
| baseline | spring-hybrid (베이스라인 최고) | 0.805 | 0.941 | 0.633 | 0.203 | 0.172 |
| s1 | spring-lexical + 분류 매핑 수정(39→음식점) | 0.771 | 0.949 | 0.650 | 0.023 | 0.139 |
| s2 | Kiwi 형태소 BM25 (k1=1.2, b=0.75) | 0.703 | 0.915 | 0.483 | 0.323 | 0.228 |
| s3 | + 띄어쓰기 무시 제목 매칭 (RRF 목록 추가) | 0.466 | 0.898 | **1.000** | 0.323 | 0.233 |
| s4x | + 지역/분류 파싱, 분류를 항상 조건으로 | 0.466 | 0.771 | 0.983 | 0.990 | 0.950 |
| s4 | + 지역/분류 파싱, 분류는 목록 요청일 때만 | 0.568 | 0.932 | 1.000 | **0.990** | 0.778 |
| dense | MiniLM, 템플릿 제거 | 0.068 | 0.093 | 0.000 | 0.023 | 0.000 |
| dense | ko-sroberta-multitask | 0.737 | 0.932 | 0.767 | 0.763 | 0.739 |
| dense | **bge-m3** | 0.822 | 0.975 | 1.000 | 0.787 | 0.717 |
| s5 | s4 + bge-m3, RRF 융합 | 0.737 | 0.983 | 1.000 | 0.997 | 0.856 |
| s5x | s5에서 제목 매칭 제거 | 0.848 | 0.975 | 0.817 | 1.000 | 0.883 |
| s5y | bge-m3 + 지역/분류만 (BM25, 제목 없음) | 0.822 | 0.975 | 1.000 | 1.000 | 0.956 |
| **s6** | s5 + bge-reranker-v2-m3 (상위 15개) | **0.983** | **1.000** | 1.000 | 1.000 | 0.911 |
| s6x | s6, 리랭크 상위 30개 | 0.983 | 1.000 | 1.000 | 1.000 | 0.950 |
| **s6y** | s5y + 리랭커 (상위 15개) | 0.966 | 0.983 | 1.000 | 1.000 | **0.978** |

지연시간(Python, RTX 4050 Laptop, p50): BM25 계열 약 1ms, bge-m3 dense 약 20~90ms, 리랭커 추가 시 semantic 약 350ms(상위 15개)와 약 730ms(상위 30개). 노트북 GPU라 같은 구성도 실행마다 20ms와 90ms 사이에서 흔들렸습니다. 절대값보다 자릿수 차이로 보세요.

### 단계별로 알게 된 것

1. **분류 매핑 수정(s1)만으로는 거의 변하지 않습니다.** "맛집"과 "음식점"이 연결되지 않는 lexical 구조에서는 분류명만 고쳐도 효과가 없습니다. 다만 이후 모든 단계에서 문서 텍스트와 필터의 전제가 되므로 반드시 고쳐야 합니다.
2. **BM25에 제목 문자 bigram을 섞으면 semantic이 무너집니다**(Hit@1 0.25). 긴 묘사형 질의의 bigram이 무관한 제목들에 높은 IDF로 걸리기 때문입니다. 그래서 본문 BM25는 형태소만 쓰고, 제목은 정확 포함 매칭으로 분리했습니다.
3. **b=0.3으로 낮추면 BM25 semantic Hit@1이 0.70에서 0.90으로 오르지만 채택하지 않았습니다.** 긴 문서(overview 보유)를 우대하는 설정인데, semantic 정답이 전부 overview 문서이므로 평가셋에 맞춘 튜닝이 됩니다. Lucene 기본값을 유지했습니다.
4. **제목 매칭은 name_lookup을 0.48에서 1.00으로 올리지만 semantic Hit@1을 깎습니다.** 묘사형 질의에도 다른 장소 이름("서울로7017과 이어진...")이 들어 있기 때문입니다. 이 충돌은 리랭커가 해소합니다(s6).
5. **분류를 항상 조건으로 걸면(s4x) semantic Hit@5가 0.93에서 0.77로 떨어집니다.** "한강변 언덕의 **기념관**", "인디밴드 **공연** 보며"처럼 분류어가 묘사로 쓰인 질의에서 정답(분류상 관광지)을 밀어냅니다. 그래서 지역/분류/일반 요청어를 빼고 남는 내용어가 1개 이하일 때만 "목록 요청"으로 보고 분류를 조건으로 거는 게이트를 넣었습니다(s4).
6. **그 게이트는 hold-out에서 약점이 드러났습니다.** "아산 가성비 숙소", "비 오는 날 아이랑 갈 박물관"처럼 수식어가 붙으면 목록 요청으로 보지 않습니다. 이런 질의(modified)에서 s4의 P@5는 0.48입니다. dense 임베딩은 분류 의미를 스스로 반영해서 같은 질의에서 0.87(s5y), 리랭커를 더하면 0.93(s6y)입니다. hold-out 결과를 보고 게이트를 다시 고치지는 않았습니다. 고치면 hold-out이 더 이상 hold-out이 아니기 때문입니다.
7. **MiniLM은 템플릿을 없애도 회복되지 않습니다**(semantic Hit@1 0.000 → 0.068). Phase 1에서는 붕괴 원인을 템플릿으로 봤지만, 5만 건 규모의 한국어 검색에서는 모델 자체가 주원인입니다. bge-m3로 바꾸면 dense 단독으로도 베이스라인 최고를 모든 세트에서 넘습니다.
8. **지역 파서 모호성**: "광주"(광주광역시와 경기 광주시)는 둘 다 후보로, "제주"는 제주시가 아니라 제주도 전체로 해석하도록 수정했습니다. "화성 행궁"처럼 지역명이 장소명의 일부이면 지역 조건을 끕니다.

### 남은 실패 사례

- s6: semantic과 name_lookup에서 Top-5 실패는 0건입니다. Hit@1에서만 semantic 2건이 1위가 아닙니다. 지역 파서를 고치기 전에는 "광주 무등산 ... 사당"이 경기 광주 문서에 밀려 5위 밖이었습니다.
- s6y: semantic Top-5 실패 2건. "땅의 신과 곡식의 신에게 제사..."(정답 사직공원)과 "전주 한옥마을에서 출발해 ... 전통 의상 대여점"(정답 한복남)입니다. BM25가 잡아 주던 고유 표현을 dense만으로는 놓칩니다.

### 권장 구성 (Phase 3: 서비스 반영 후보)

| 경로 | 권장 | 근거 |
|---|---|---|
| 기본 경로 `NEOULTEO_AI_ENABLED=false` (임베딩 없음) | **s4: BM25 + 제목 매칭 + 지역/분류** | 약 1ms. name 1.00, rc 0.99, semantic Hit@5 0.93. 현재 lexical 대비 모든 세트에서 크게 개선됩니다. Java에서는 Lucene Nori 형태소 분석기로 옮길 수 있습니다 |
| AI 경로 (임베딩 사용) | **s6y: bge-m3 + 지역/분류 + 리랭커** | 구성 요소가 가장 적고 hold-out이 가장 좋습니다(0.978). semantic에서 s6와의 차이는 2문항(118문항 중)으로, 이 표본 크기에서는 의미 있는 차이로 보기 어렵습니다 |
| 리랭커를 못 쓰는 경우 | s5y: bge-m3 + 지역/분류 | semantic Hit@1 0.82(목표 0.85에 약간 못 미침), 나머지 목표는 충족합니다 |

서비스 반영 전에 확인할 것:
- **실제 임베딩 모델로 재측정**: Spring AI 경로는 GMS `text-embedding-3-small`을 씁니다. 같은 하네스로 측정해 bge-m3와 비교하거나, bge-m3/리랭커를 `ai-server`(FastAPI)에서 서빙하는 방안을 정해야 합니다.
- **분류를 LLM에서 받기**: 규칙 게이트는 수식어에 약합니다. 질의 이해(지역/분류 추출)를 LLM에 맡기는 구성(예: 구조화 출력, `AttractionTool`의 인자)과 비교할 가치가 있습니다.
- **더 어려운 평가셋**: semantic이 0.98로 천장에 가깝습니다. 사람이 쓴 질의, 그리고 overview가 없는 지역(서울/인천/광주/경기 외) 질의가 필요합니다.

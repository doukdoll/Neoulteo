"""Neoulteo의 현재 검색 로직을 Python으로 재현한 Retriever 모음.

모든 Retriever는 search(query, k) -> list[content_id] 인터페이스를 가진다.
재현 원본:
  - SpringLexicalRetriever : backend/.../ai/service/RagSearchService.java (NEOULTEO_AI_ENABLED=false 기본 경로)
  - SpringHybridRetriever  : 같은 파일의 임베딩 결합 경로 (lexical 후보 80개 → lexical + 2*cosine)
  - FastApiDenseRetriever  : ai-server/main.py /search (MiniLM 임베딩 + 코사인 전수 비교)
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

# RagSearchService.RELATED_TERM_GROUPS 그대로
RELATED_TERM_GROUPS = [
    {"물놀이", "수영", "해수욕", "해수욕장", "워터파크", "계곡", "바다", "해변", "해안"},
    {"바다", "해변", "해수욕장", "해안", "오션", "항구", "섬"},
    {"산", "등산", "트레킹", "숲", "자연", "계곡", "둘레길", "산책"},
    {"아이", "어린이", "가족", "키즈", "체험", "놀이터", "교육"},
    {"실내", "박물관", "미술관", "전시", "문화", "공연", "체험관"},
    {"맛집", "음식", "식당", "카페", "먹거리", "시장", "디저트"},
    {"야경", "밤", "전망", "전망대", "일몰", "노을", "뷰"},
    {"역사", "유적", "유물", "궁", "고궁", "문화재", "사찰", "절"},
    {"쇼핑", "시장", "거리", "상점", "몰", "상가"},
    {"힐링", "휴식", "조용", "한적", "공원", "정원", "산책"},
    {"액티비티", "레포츠", "스포츠", "자전거", "캠핑", "낚시", "체험"},
]

# DB contenttypes 테이블 값(39가 '관광지'로 잘못 들어가 있음)을 그대로 재현
DB_CONTENT_TYPE_NAMES = {12: "관광지", 14: "문화시설", 15: "축제공연행사", 25: "여행코스",
                         28: "레포츠", 32: "숙박", 38: "쇼핑", 39: "관광지"}

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def spring_search_text(d: dict) -> str:
    """RagAttractionDocument.toSearchText() 재현."""
    parts = []
    for label, value in (("관광지명", d["title"]), ("주소", d["address"]),
                         ("분류", DB_CONTENT_TYPE_NAMES.get(d["content_type_id"], "")),
                         ("설명", d["overview"])):
        if value and value.strip():
            parts.append(f"{label}: {value}")
    return "\n".join(parts).strip()


def fastapi_text(d: dict) -> str:
    """ai-server/data/tour_data.txt 한 줄 포맷 재현 (overview 미포함)."""
    return f"관광지명 [{d['title']}]은(는) {d['address']}에 위치해 있습니다. 관련 문의를 위한 연락처는 {d.get('tel', '')} 입니다."


def spring_index_order(docs: list[dict]) -> list[dict]:
    """findRagAttractionDocuments: title/주소 or overview 있는 것, overview 있는 문서 먼저 → title 순."""
    eligible = [d for d in docs if d["title"] and (d["overview"] or d["address"])]
    return sorted(eligible, key=lambda d: (0 if d["overview"] else 1, d["title"]))


def _normalize(v: str) -> str:
    return re.sub(r"\s+", " ", (v or "").lower()).strip()


class Embedder:
    """sentence-transformers 래퍼 + 코퍼스 임베딩 디스크 캐시."""

    def __init__(self, model_name: str, device: str | None = None, query_prefix: str = "", doc_prefix: str = ""):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def encode_docs(self, texts: list[str], tag: str) -> np.ndarray:
        key = hashlib.sha1(("\x00".join(texts) + self.model_name + self.doc_prefix).encode("utf-8")).hexdigest()[:12]
        path = CACHE_DIR / f"{tag}-{self.model_name.replace('/', '_')}-{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.model.encode([self.doc_prefix + t for t in texts], batch_size=128,
                                 normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs.astype(np.float32))
        return vecs

    def encode_query(self, q: str) -> np.ndarray:
        return self.model.encode(self.query_prefix + q, normalize_embeddings=True, convert_to_numpy=True)


class SpringLexicalRetriever:
    name = "spring-lexical"

    def __init__(self, docs: list[dict]):
        ordered = spring_index_order(docs)
        self.ids = [d["content_id"] for d in ordered]
        self.texts = [spring_search_text(d) for d in ordered]
        self.norm = [_normalize(t) for t in self.texts]

    @staticmethod
    def expanded_terms(nq: str) -> set[str]:
        terms = set(nq.split(" "))
        for group in RELATED_TERM_GROUPS:
            if any(t in nq for t in group):
                terms |= group
        return terms

    def lexical_candidates(self, query: str, limit: int) -> list[tuple[int, float]]:
        nq = _normalize(query)
        terms = [t for t in self.expanded_terms(nq) if t]
        scored = []
        for i, content in enumerate(self.norm):
            s = 2.0 if nq in content else 0.0
            for t in terms:
                if t in content:
                    s += 0.6 if t in nq else 0.35
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: -x[1])  # Java stream sorted는 안정 정렬 → 동점은 인덱스 순서 유지
        return scored[:limit]

    def search(self, query: str, k: int) -> list[int]:
        return [self.ids[i] for i, _ in self.lexical_candidates(query, k)]


class SpringHybridRetriever(SpringLexicalRetriever):
    """lexical 후보(최대 80) 안에서만 임베딩 점수를 더한다. 후보가 0이면 빈 결과."""
    name = "spring-hybrid"

    def __init__(self, docs: list[dict], embedder: Embedder, candidate_limit: int = 80):
        super().__init__(docs)
        self.embedder = embedder
        self.candidate_limit = candidate_limit
        self.emb = embedder.encode_docs(self.texts, "spring")

    def search(self, query: str, k: int) -> list[int]:
        cands = self.lexical_candidates(query, max(self.candidate_limit, k))
        if not cands:
            return []
        qv = self.embedder.encode_query(query)
        scored = [(i, s + max(float(self.emb[i] @ qv), 0.0) * 2.0) for i, s in cands]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: -x[1])
        return [self.ids[i] for i, _ in scored[:k]]


class FastApiDenseRetriever:
    name = "fastapi-dense"

    def __init__(self, docs: list[dict], embedder: Embedder, text_fn=fastapi_text, tag: str = "fastapi"):
        self.ids = [d["content_id"] for d in docs]
        self.embedder = embedder
        self.emb = embedder.encode_docs([text_fn(d) for d in docs], tag)

    def search(self, query: str, k: int) -> list[int]:
        qv = self.embedder.encode_query(query)
        scores = self.emb @ qv
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [self.ids[i] for i in top]

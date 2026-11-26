"""Phase 2 개선 검색기.

한 번에 한 가지씩 켜고 끄면서 측정할 수 있도록 구성 요소를 플래그로 분리했다.

  bm25      : Kiwi 형태소 BM25 (IDF가 있어서 흔한 단어 가중치가 낮다). Lucene 기본값 k1=1.2, b=0.75
  title     : 띄어쓰기를 무시한 제목 포함 매칭. 별도 순위 목록으로 RRF에 넣는다
  meta      : 질의에서 지역(시도/시군구)과 분류를 파싱해 해당 문서에 큰 가산점 (soft filter)
  dense     : 템플릿 없는 문서 텍스트 임베딩
  rerank    : 융합 상위 N개를 cross-encoder로 재정렬

여러 순위 목록은 RRF(k=60)로 합친다. 점수 스케일이 다른 신호를 가중치 튜닝 없이 합치기 위해서다.
meta 가산점(1.0)은 RRF 최대값(목록 3개 × 1/61 ≈ 0.05)보다 훨씬 커서 사실상 필터처럼 동작하지만,
조건에 맞는 문서가 없으면 일반 순위로 자연스럽게 되돌아간다.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

from .retrievers import CACHE_DIR, Embedder

# Tour API 정의. DB contenttypes 테이블은 39를 '관광지'로 잘못 갖고 있다.
CONTENT_TYPE_NAMES = {12: "관광지", 14: "문화시설", 15: "축제공연행사", 25: "여행코스",
                      28: "레포츠", 32: "숙박", 38: "쇼핑", 39: "음식점"}

# 분류 어휘. 평가셋 템플릿이 아니라 Tour API 분류 체계와 일반적인 여행 질의 표현에서 정했다.
CATEGORY_LEXICON = {
    14: ["문화시설", "박물관", "미술관", "전시관", "전시회", "기념관", "과학관", "갤러리", "공연장", "도서관"],
    15: ["축제", "공연", "행사", "페스티벌"],
    25: ["여행코스", "여행 코스", "코스 추천"],
    28: ["레포츠", "액티비티", "레저", "스포츠", "서핑", "래프팅", "패러글라이딩", "스키", "골프", "짚라인"],
    32: ["숙박", "숙소", "호텔", "모텔", "펜션", "게스트하우스", "리조트", "민박", "콘도", "묵을", "잘 곳", "잘곳"],
    38: ["쇼핑", "기념품", "아울렛", "백화점", "면세점", "특산품", "전통시장", "시장"],
    39: ["맛집", "식당", "음식점", "레스토랑", "카페", "먹거리", "밥 먹", "밥집", "먹을 곳", "먹을만한"],
}

SIDO_ALIASES = {
    "서울": ["서울", "서울시", "서울특별시"], "부산": ["부산", "부산시", "부산광역시"],
    "대구": ["대구", "대구시", "대구광역시"], "인천": ["인천", "인천시", "인천광역시"],
    "광주": ["광주", "광주광역시"],  # '광주'는 경기도 광주시와도 겹친다 → parse에서 둘 다 후보로
    "대전": ["대전", "대전시", "대전광역시"], "울산": ["울산", "울산시", "울산광역시"],
    "세종특별자치시": ["세종", "세종시", "세종특별자치시"],
    "경기도": ["경기", "경기도"], "강원특별자치도": ["강원", "강원도", "강원특별자치도"],
    "충청북도": ["충북", "충청북도"], "충청남도": ["충남", "충청남도"],
    "전북특별자치도": ["전북", "전라북도", "전북특별자치도"], "전라남도": ["전남", "전라남도"],
    "경상북도": ["경북", "경상북도"], "경상남도": ["경남", "경상남도"],
    "제주도": ["제주", "제주도", "제주특별자치도"],
}

# 목록 요청("~ 추천해줘", "~ 어디 있어?")에서 내용 없이 쓰이는 일반 요청어. 형태소 기준.
REQUEST_WORDS = {"곳", "장소", "추천", "알리", "어디", "있", "가", "보", "싶", "정보", "하", "만하", "좋", "주"}

JOSA = ("에서", "에", "의", "쪽", "근처", "주변", "로", "은", "는", "도")
RRF_K = 60
META_BONUS = 1.0


def doc_text(d: dict) -> str:
    """템플릿 없는 문서 텍스트. Spring toSearchText()와 같은 구조에 분류만 바로잡았다."""
    parts = []
    for label, value in (("관광지명", d["title"]), ("주소", d["address"]),
                         ("분류", CONTENT_TYPE_NAMES.get(d["content_type_id"], "")),
                         ("설명", d["overview"])):
        if value and value.strip():
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def squash(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower())


# ---------------------------------------------------------------- tokenizer
class KiwiTokenizer:
    KEEP = ("NNG", "NNP", "NR", "SL", "SH", "SN", "XR", "VV", "VA")

    def __init__(self):
        from kiwipiepy import Kiwi
        self.kiwi = Kiwi()

    def morphs(self, text: str) -> list[str]:
        if not text:
            return []
        return [t.form.lower() for t in self.kiwi.tokenize(text) if t.tag.startswith(self.KEEP)]

    def morphs_batch(self, texts: list[str]) -> list[list[str]]:
        out = []
        for toks in self.kiwi.tokenize(texts):
            out.append([t.form.lower() for t in toks if t.tag.startswith(self.KEEP)])
        return out


# ---------------------------------------------------------------- BM25
class SparseBM25:
    """scipy 희소행렬 BM25. rank_bm25보다 5만 건에서 수십 배 빠르다."""

    def __init__(self, tokenized: list[list[str]], k1: float = 1.2, b: float = 0.75):
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(tokenized), dtype=np.float32)
        for i, toks in enumerate(tokenized):
            lengths[i] = len(toks)
            tf = defaultdict(int)
            for t in toks:
                tf[vocab.setdefault(t, len(vocab))] += 1
            for j, c in tf.items():
                rows.append(i); cols.append(j); vals.append(c)
        n = len(tokenized)
        tf_m = sparse.csc_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(n, len(vocab)))
        df = np.diff(tf_m.indptr).astype(np.float32)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        avg = lengths.mean() or 1.0
        norm = k1 * (1 - b + b * lengths / avg)
        # 문서별 정규화를 미리 적용: tf*(k1+1)/(tf+norm)
        tf_m = tf_m.tocoo()
        w = tf_m.data * (k1 + 1) / (tf_m.data + norm[tf_m.row])
        self.mat = sparse.csc_matrix((w, (tf_m.row, tf_m.col)), shape=tf_m.shape)
        self.vocab = vocab
        self.n = n

    def scores(self, q_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for t in set(q_tokens):
            j = self.vocab.get(t)
            if j is None:
                continue
            s, e = self.mat.indptr[j], self.mat.indptr[j + 1]
            out[self.mat.indices[s:e]] += self.idf[j] * self.mat.data[s:e]
        return out


# ---------------------------------------------------------------- query parsing
class QueryParser:
    """질의에서 지역(시도/시군구)과 분류를 뽑는다. 단어 경계(+조사)에서만 지역으로 인정한다.

    browse: 지역/분류/일반 요청어를 빼고 남는 내용 형태소가 1개 이하인 질의(예: "화성 숙박할 곳 알려줘").
    이런 '목록 요청'에서만 분류를 조건으로 쓴다. "한강변 언덕의 기념관"처럼 분류어가 묘사로 쓰인
    질의에서 분류를 조건으로 걸면 정답(분류상 관광지)을 밀어내기 때문이다.
    """

    def __init__(self, docs: list[dict], tokenizer: "KiwiTokenizer | None" = None):
        self.tok = tokenizer
        self.cat_morphs = {}
        if tokenizer is not None:
            for ct, words_ in CATEGORY_LEXICON.items():
                self.cat_morphs[ct] = {m for w in words_ for m in tokenizer.morphs(w)}
        self.sido_alias = {}
        for sido, names in SIDO_ALIASES.items():
            for a in names:
                self.sido_alias[a] = sido
        # 시군구: 전체 이름과 '시/군'을 뗀 짧은 이름. 구는 짧게 쓰지 않는다(중구→중 X).
        self.gugun_alias = defaultdict(set)  # alias -> {(sido, gugun)}
        for d in docs:
            g = d["gugun"]
            if not g:
                continue
            self.gugun_alias[g].add((d["sido"], g))
            if len(g) > 2 and g[-1] in "시군":
                self.gugun_alias[g[:-1]].add((d["sido"], g))

    @staticmethod
    def _strip_josa(w: str) -> list[str]:
        cands = [w]
        for j in JOSA:
            if w.endswith(j) and len(w) > len(j) + 1:
                cands.append(w[: -len(j)])
        return cands

    def parse(self, query: str) -> dict:
        words = re.findall(r"[0-9A-Za-z가-힣]+", query)
        sidos, ambiguous, guguns, used = set(), set(), set(), []
        for idx, w in enumerate(words):
            for c in self._strip_josa(w):
                in_s, in_g = c in self.sido_alias, c in self.gugun_alias
                if in_s and in_g and all(g[0] == self.sido_alias[c] for g in self.gugun_alias[c]):
                    in_g = False  # '제주' = 제주도 전체(제주시로 좁히지 않음)
                if in_s and in_g:  # '광주' = 광주광역시 or 경기 광주시
                    ambiguous.add(self.sido_alias[c]); guguns |= self.gugun_alias[c]
                elif in_s:
                    sidos.add(self.sido_alias[c])
                elif in_g:
                    guguns |= self.gugun_alias[c]
                if in_s or in_g:
                    used.append(idx); break
        # regions: (시도, 시군구 또는 None) 후보 집합. 문서는 하나라도 맞으면 지역 조건 충족
        narrowing = sidos | ambiguous
        narrowed = {g for g in guguns if g[0] in narrowing}
        if narrowed:  # '부산 중구' → 부산 중구만, '광주 동구' → 광주광역시 동구만
            regions = narrowed
        elif guguns:
            regions = guguns | {(sd, None) for sd in sidos | ambiguous}
        else:
            regions = {(sd, None) for sd in sidos}
        cats = set()
        for ct, words_ in CATEGORY_LEXICON.items():
            if any(x in query for x in words_):
                cats.add(ct)
        region_words = [words[i] for i in used]
        browse = None
        if self.tok is not None:
            drop = set(REQUEST_WORDS)
            for w in region_words:
                drop |= set(self.tok.morphs(w)) | {w.lower()}
            for ct in cats:
                drop |= self.cat_morphs[ct]
            residual = [m for m in self.tok.morphs(query) if m not in drop]
            browse = len(residual) <= 1
        return {"regions": regions, "categories": cats, "region_words": region_words,
                "browse": browse}


# ---------------------------------------------------------------- retriever
class Phase2Retriever:
    def __init__(self, docs: list[dict], name: str, *, bm25: bool = True, title: bool = False,
                 meta: bool = False, embedder: Embedder | None = None, reranker=None,
                 rerank_top: int = 30, dense_only: bool = False, cat_mode: str = "gated"):
        self.name = name
        self.docs = docs
        self.ids = np.array([d["content_id"] for d in docs])
        self.use_bm25, self.use_title, self.use_meta = bm25, title, meta
        self.dense_only = dense_only
        self.cat_mode = cat_mode  # "hard": 분류어가 보이면 항상 조건 / "gated": 목록 요청일 때만
        self.reranker, self.rerank_top = reranker, rerank_top
        self.texts = [doc_text(d) for d in docs]

        if (bm25 or meta) and not dense_only:
            self.tok = KiwiTokenizer()
            toks = self._cached_tokens()
            self.bm25 = SparseBM25(toks)
        if title:
            self.titles_sq = [squash(d["title"]) for d in docs]
            # 제목 길이별로 묶어 두면 질의 부분 문자열만 해시 조회하면 된다(5만 건 순회 불필요)
            self.title_index = defaultdict(list)
            for i, t in enumerate(self.titles_sq):
                if len(t) >= 2:
                    self.title_index[t].append(i)
            self.title_lengths = sorted({len(t) for t in self.title_index}, reverse=True)
        if meta:
            self.parser = QueryParser(docs, self.tok)
            self.sido_arr = np.array([d["sido"] for d in docs])
            self.gugun_arr = np.array([f"{d['sido']}|{d['gugun']}" for d in docs])
            self.ctype_arr = np.array([d["content_type_id"] for d in docs])
        self.embedder = embedder
        if embedder is not None:
            self.emb = embedder.encode_docs(self.texts, "p2doc")

    # 토큰화는 수십 초 걸리므로 캐시
    def _cached_tokens(self) -> list[list[str]]:
        import pickle
        key = hashlib.sha1("\x00".join(self.texts).encode("utf-8")).hexdigest()[:12]
        path = CACHE_DIR / f"morphs-{key}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        toks = self.tok.morphs_batch(self.texts)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(toks))
        return toks

    def query_tokens(self, q: str) -> list[str]:
        return self.tok.morphs(q)

    def title_matches(self, q: str) -> list[int]:
        """질의(공백 제거)에 통째로 들어 있는 제목. 긴 제목부터 반환."""
        qs = squash(q)
        found, seen = [], set()
        for L in self.title_lengths:
            if L > len(qs):
                continue
            for s in range(len(qs) - L + 1):
                sub = qs[s:s + L]
                if sub in self.title_index and sub not in seen:
                    seen.add(sub)
                    found.extend(self.title_index[sub])
        return found

    @staticmethod
    def _rrf_add(fused: np.ndarray, order: np.ndarray):
        fused[order] += 1.0 / (RRF_K + np.arange(1, len(order) + 1))

    @staticmethod
    def _top(scores: np.ndarray, n: int) -> np.ndarray:
        pos = np.flatnonzero(scores > 0)
        if len(pos) == 0:
            return pos
        n = min(n, len(pos))
        part = pos[np.argpartition(-scores[pos], n - 1)[:n]]
        return part[np.argsort(-scores[part], kind="stable")]

    def search(self, query: str, k: int) -> list[int]:
        n = len(self.docs)
        fused = np.zeros(n, dtype=np.float64)
        dense = None
        if self.embedder is not None:
            dense = self.emb @ self.embedder.encode_query(query)
            if self.dense_only:
                top = np.argpartition(-dense, k)[:k]
                return self.ids[top[np.argsort(-dense[top])]].tolist()
            self._rrf_add(fused, self._top(dense - dense.min() + 1e-6, 200))
        if self.use_bm25:
            self._rrf_add(fused, self._top(self.bm25.scores(self.query_tokens(query)), 200))

        title_hits = []
        if self.use_title:
            title_hits = self.title_matches(query)
            if title_hits:
                # 긴 제목 순서 그대로 하나의 순위 목록으로 취급
                self._rrf_add(fused, np.array(list(dict.fromkeys(title_hits))[:200]))

        meta_mask = None
        if self.use_meta:
            p = self.parser.parse(query)
            # 지역 단어가 실제로는 장소명의 일부인 경우(예: '화성 행궁')는 지역 조건을 쓰지 않는다
            if title_hits and p["region_words"]:
                long_titles = [self.titles_sq[i] for i in title_hits[:5] if len(self.titles_sq[i]) >= 3]
                if any(squash(w) in t for t in long_titles for w in p["region_words"]):
                    p["regions"] = set()
            bonus = np.zeros(n)
            if p["regions"]:
                keys = [f"{s}|{g}" for s, g in p["regions"] if g]
                whole = [s for s, g in p["regions"] if not g]
                hit = np.isin(self.gugun_arr, keys) | np.isin(self.sido_arr, whole)
                bonus += META_BONUS * hit
            if p["categories"] and (self.cat_mode == "hard" or p["browse"]):
                bonus += META_BONUS * np.isin(self.ctype_arr, list(p["categories"]))
            if bonus.any():
                meta_mask = bonus
                fused += bonus
                if dense is not None:  # 조건만 맞고 다른 신호가 없는 문서끼리는 임베딩 유사도로 정렬
                    fused += 1e-4 * dense

        top_n = max(k, self.rerank_top if self.reranker else k)
        cand = self._top(fused, top_n)
        if self.reranker is not None and len(cand):
            ce = np.asarray(self.reranker.predict([(query, self.texts[i]) for i in cand],
                                                  batch_size=32, show_progress_bar=False), dtype=np.float64)
            ce = 1 / (1 + np.exp(-ce))  # 0~1. meta 가산점(1.0) 단위와 맞춘다
            final = ce + (meta_mask[cand] if meta_mask is not None else 0)
            cand = cand[np.argsort(-final, kind="stable")]
        return self.ids[cand[:k]].tolist()


class FixedSpringLexical:
    """step1: Spring lexical 그대로 두고 분류 매핑(39→음식점)만 고친 버전."""

    def __new__(cls, docs):
        from . import retrievers as R
        orig = R.DB_CONTENT_TYPE_NAMES
        R.DB_CONTENT_TYPE_NAMES = CONTENT_TYPE_NAMES
        try:
            r = R.SpringLexicalRetriever(docs)
        finally:
            R.DB_CONTENT_TYPE_NAMES = orig
        r.name = "s1-lexical+catfix"
        return r

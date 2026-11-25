"""규칙 기반으로 자동 생성 가능한 평가 질의셋 2종을 만든다.

1) name_lookup: 장소 이름(띄어쓰기 변형 포함)으로 찾는 질의. 정답 = 같은 이름의 문서 전체.
2) region_category: "{지역} {분류} 추천" 형태 질의. 정답 = 해당 시군구 + 해당 content_type 문서 전체.
   (content_type 39 = 음식점. DB contenttypes 테이블에는 '관광지'로 잘못 매핑되어 있으나
    Tour API 원래 정의는 음식점이므로 정답 판정은 원래 정의를 따른다.)

usage: python scripts/build_auto_sets.py data/corpus.jsonl data/eval
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEED = 7

CATEGORY_PHRASES = {
    32: ["숙소 추천해줘", "묵을 만한 호텔이나 펜션", "숙박할 곳 알려줘"],
    39: ["맛집 추천해줘", "밥 먹을 식당 어디 있어?", "현지 맛집 알려줘"],
    38: ["쇼핑할 만한 곳", "기념품 살 곳 추천", "시장이나 쇼핑 장소"],
    28: ["레포츠 즐길 곳", "액티비티 할 수 있는 곳 추천", "체험 레저 스포츠"],
    14: ["박물관이나 미술관 추천", "실내 문화시설 가볼 곳", "전시관 어디 있어?"],
    15: ["축제나 공연 정보", "열리는 행사 알려줘"],
}

SIDO_SHORT = {
    "경기도": "경기", "강원특별자치도": "강원", "경상남도": "경남", "경상북도": "경북",
    "전라남도": "전남", "전북특별자치도": "전북", "충청남도": "충남", "충청북도": "충북",
    "제주도": "제주", "세종특별자치시": "세종",
}


def gugun_short(g: str) -> str:
    if len(g) > 2 and g[-1] in "시군" and " " not in g:
        return g[:-1]
    return g


def space_variant(title: str, rng: random.Random) -> str:
    if " " in title:
        return title.replace(" ", "")
    if len(title) >= 4:
        cut = len(title) // 2
        return title[:cut] + " " + title[cut:]
    return title


def main(corpus_path: str, out_dir: str):
    rng = random.Random(SEED)
    docs = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- name_lookup ----------
    by_title = defaultdict(list)
    for d in docs:
        by_title[d["title"]].append(d["content_id"])
    pool = [d for d in docs if 2 <= len(d["title"]) <= 20]
    rng.shuffle(pool)
    name_rows = []
    templates = ["{t} 어디야?", "{t} 위치 알려줘", "{s} {t}", "{t} 주소", "{t} 가보고 싶은데 어디 있어?"]
    for d in pool[:60]:
        variant = rng.random() < 0.5
        t = space_variant(d["title"], rng) if variant else d["title"]
        tpl = rng.choice(templates)
        q = tpl.format(t=t, s=SIDO_SHORT.get(d["sido"], d["sido"]))
        gold = by_title[d["title"]]
        if "{s}" in tpl:  # 지역을 명시했으면 같은 이름 중 해당 시도만 정답
            gold = [x["content_id"] for x in docs if x["title"] == d["title"] and x["sido"] == d["sido"]]
        name_rows.append({
            "id": f"name-{len(name_rows)+1:03d}", "set": "name_lookup", "query": q,
            "relevant": gold, "difficulty": "spacing" if variant and t != d["title"] else "exact",
        })

    # ---------- region_category ----------
    gugun_names = Counter(d["gugun"] for d in docs)  # 같은 구 이름이 여러 시도에 있는지 판단용
    gugun_sidos = defaultdict(set)
    for d in docs:
        gugun_sidos[d["gugun"]].add(d["sido"])
    groups = defaultdict(list)
    for d in docs:
        if d["gugun"] and d["content_type_id"] in CATEGORY_PHRASES:
            groups[(d["sido"], d["gugun"], d["content_type_id"])].append(d["content_id"])
    eligible = [k for k, v in groups.items() if len(v) >= 5]
    rng.shuffle(eligible)
    per_type = Counter()
    rc_rows = []
    for sido, gugun, ctype in eligible:
        if per_type[ctype] >= 10:
            continue
        per_type[ctype] += 1
        g = gugun_short(gugun)
        place = g if len(gugun_sidos[gugun]) == 1 else f"{SIDO_SHORT.get(sido, sido)} {g}"
        q = f"{place} {rng.choice(CATEGORY_PHRASES[ctype])}"
        rc_rows.append({
            "id": f"rc-{len(rc_rows)+1:03d}", "set": "region_category", "query": q,
            "relevant": groups[(sido, gugun, ctype)], "content_type_id": ctype,
        })

    for name, rows in [("name_lookup", name_rows), ("region_category", rc_rows)]:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} queries")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

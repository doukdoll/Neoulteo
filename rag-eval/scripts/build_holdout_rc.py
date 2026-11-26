"""region_category hold-out 세트.

Phase 2의 분류 어휘(CATEGORY_LEXICON)와 목록 요청 게이트(REQUEST_WORDS)는 region_category 세트를 보면서
만들었기 때문에, 그 세트 점수는 과대평가될 수 있다. 이를 확인하기 위해
  - 원래 세트에 없는 표현(수식어 포함: "가성비", "비 오는 날", "아이랑" 등)
  - 원래 세트에서 쓰지 않은 (시도, 시군구, 분류) 조합
으로 별도 세트를 만든다. 정답 판정 방식은 원래 세트와 같다(해당 시군구 + 분류의 문서 전체).

usage: python scripts/build_holdout_rc.py data/corpus.jsonl data/eval
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_auto_sets import SIDO_SHORT, gugun_short  # noqa: E402

SEED = 11

# plain: 내용어 없이 지역 + 분류만, modified: 추가 조건(수식어)이 붙은 표현
PHRASES = {
    32: [("plain", "{p}에서 하룻밤 묵을 호텔"), ("plain", "{p} 펜션 어디 있어?"), ("modified", "{p} 가성비 숙소")],
    39: [("plain", "{p} 음식점 좀 알려줘"), ("plain", "{p} 가서 들를 식당"), ("modified", "{p} 점심 먹을 맛집")],
    38: [("plain", "{p} 쇼핑 어디서 해?"), ("plain", "{p} 전통시장 알려줘"), ("modified", "{p} 선물 사기 좋은 기념품 가게")],
    28: [("plain", "{p} 레저 스포츠"), ("plain", "{p}에서 할 수 있는 액티비티"), ("modified", "{p} 주말에 할 만한 레포츠")],
    14: [("plain", "{p} 미술관 있어?"), ("plain", "{p} 박물관 가볼 곳"), ("modified", "{p} 비 오는 날 아이랑 갈 박물관")],
    15: [("plain", "{p} 축제 알려줘"), ("plain", "{p} 공연이나 행사"), ("modified", "{p} 이번 달 열리는 축제")],
}


def main(corpus_path: str, out_dir: str):
    rng = random.Random(SEED)
    docs = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]
    out = Path(out_dir)
    used = set()
    for r in map(json.loads, open(out / "region_category.jsonl", encoding="utf-8")):
        used.add((frozenset(r["relevant"])))

    gugun_sidos = defaultdict(set)
    groups = defaultdict(list)
    for d in docs:
        gugun_sidos[d["gugun"]].add(d["sido"])
        if d["gugun"] and d["content_type_id"] in PHRASES:
            groups[(d["sido"], d["gugun"], d["content_type_id"])].append(d["content_id"])
    eligible = [k for k, v in groups.items() if len(v) >= 5 and frozenset(v) not in used]
    rng.shuffle(eligible)

    per_type, rows = Counter(), []
    for sido, gugun, ctype in eligible:
        if per_type[ctype] >= 6:
            continue
        kind, tpl = PHRASES[ctype][per_type[ctype] % 3]
        per_type[ctype] += 1
        g = gugun_short(gugun)
        place = g if len(gugun_sidos[gugun]) == 1 else f"{SIDO_SHORT.get(sido, sido)} {g}"
        rows.append({"id": f"rch-{len(rows)+1:03d}", "set": "rc_holdout", "query": tpl.format(p=place),
                     "relevant": groups[(sido, gugun, ctype)], "content_type_id": ctype, "difficulty": kind})

    with open(out / "rc_holdout.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"rc_holdout: {len(rows)} queries", Counter(r["difficulty"] for r in rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

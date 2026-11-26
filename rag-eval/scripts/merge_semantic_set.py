"""서브에이전트가 작성한 의미 검색 질의(queries_*.jsonl)를 검증·병합해 data/eval/semantic.jsonl 생성.

검증:
  - content_id 가 코퍼스에 존재하는지
  - 정답 장소 이름의 3글자 이상 조각이 질의에 그대로 들어갔는지(이름 누설) → 제외
  - overview 와 어절 3개 이상이 연속으로 겹치는지(복사) → 제외
정답(relevant)은 같은 이름의 문서 전체(동명 중복 레코드 대응).

usage: python scripts/merge_semantic_set.py data/corpus.jsonl data/eval_build data/eval/semantic.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def title_leak(title: str, query: str, n: int = 3) -> str | None:
    t, q = title.replace(" ", ""), query.replace(" ", "")
    for i in range(len(t) - n + 1):
        if t[i:i + n] in q:
            return t[i:i + n]
    return None


def copied_span(overview: str, query: str, n: int = 3) -> str | None:
    ow, qw = overview.split(), query.split()
    grams = {" ".join(ow[i:i + n]) for i in range(len(ow) - n + 1)}
    for i in range(len(qw) - n + 1):
        g = " ".join(qw[i:i + n])
        if g in grams:
            return g
    return None


def main(corpus_path, build_dir, out_path):
    docs = {json.loads(l)["content_id"]: json.loads(l) for l in open(corpus_path, encoding="utf-8")}
    by_title = defaultdict(list)
    for d in docs.values():
        by_title[d["title"]].append(d["content_id"])

    kept, dropped = [], []
    for p in sorted(Path(build_dir).glob("queries_*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            d = docs.get(r["content_id"])
            if d is None:
                dropped.append((r, "unknown content_id"))
                continue
            leak = title_leak(d["title"], r["query"])
            copy = copied_span(d["overview"], r["query"])
            if leak or copy:
                dropped.append((r, f"leak={leak} copy={copy}"))
                continue
            kept.append({"id": f"sem-{len(kept)+1:03d}", "set": "semantic", "query": r["query"],
                         "relevant": by_title[d["title"]], "difficulty": r["difficulty"],
                         "target_title": d["title"]})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)}, dropped {len(dropped)}")
    for r, why in dropped:
        print("  DROP", r["content_id"], why, "|", r["query"])


if __name__ == "__main__":
    main(*sys.argv[1:4])

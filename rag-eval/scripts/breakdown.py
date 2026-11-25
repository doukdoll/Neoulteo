"""난이도/분류별 세부 지표: semantic(medium/hard), name_lookup(exact/spacing), region_category(content_type)."""
import json, sys
from collections import defaultdict
sys.path.insert(0, ".")
from run_eval import build_retrievers, load_jsonl, ROOT
from neoulteo_rag.metrics import score

docs = load_jsonl(ROOT / "data/corpus.jsonl")
rets = build_retrievers(sys.argv[1:] or ["spring-lexical", "spring-hybrid"], docs)
plan = [("semantic", "difficulty", "hit@5"), ("name_lookup", "difficulty", "hit@5"),
        ("region_category", "content_type_id", "p@5")]
for r in rets:
    for s, key, m in plan:
        g = defaultdict(list)
        for q in load_jsonl(ROOT / f"data/eval/{s}.jsonl"):
            g[q[key]].append(score(m, r.search(q["query"], 10), set(q["relevant"])))
        print(r.name, s, m, {k: f"{sum(v)/len(v):.2f} (n={len(v)})" for k, v in sorted(g.items(), key=lambda x: str(x[0]))})

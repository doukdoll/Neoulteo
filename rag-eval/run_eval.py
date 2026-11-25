"""평가 실행기: 질의셋 × Retriever 조합으로 지표와 지연시간을 측정하고 결과를 저장한다.

usage:
  .venv/Scripts/python run_eval.py --retrievers spring-lexical spring-hybrid fastapi-dense --tag baseline
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from neoulteo_rag.metrics import SET_METRICS, score

ROOT = Path(__file__).resolve().parent
SETS = ["semantic", "name_lookup", "region_category"]
BASELINE_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # ai-server/main.py 와 동일


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def build_retrievers(names: list[str], docs: list[dict]):
    from neoulteo_rag import retrievers as R

    embedders: dict[str, R.Embedder] = {}

    def emb(model: str) -> R.Embedder:
        if model not in embedders:
            embedders[model] = R.Embedder(model)
        return embedders[model]

    factory = {
        "spring-lexical": lambda: R.SpringLexicalRetriever(docs),
        # 원본은 OpenAI text-embedding-3-small. API 키 없이 재현하기 위해 FastAPI와 같은 로컬 모델로 대체.
        "spring-hybrid": lambda: R.SpringHybridRetriever(docs, emb(BASELINE_EMBED_MODEL)),
        "fastapi-dense": lambda: R.FastApiDenseRetriever(docs, emb(BASELINE_EMBED_MODEL)),
    }
    out = []
    for n in names:
        t0 = time.perf_counter()
        r = factory[n]()
        print(f"[build] {n}: {time.perf_counter() - t0:.1f}s")
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievers", nargs="+", default=["spring-lexical", "spring-hybrid", "fastapi-dense"])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    docs = load_jsonl(ROOT / "data" / "corpus.jsonl")
    title_of = {d["content_id"]: d["title"] for d in docs}
    sets = {s: load_jsonl(ROOT / "data" / "eval" / f"{s}.jsonl") for s in SETS}
    retrievers = build_retrievers(args.retrievers, docs)

    summary, failures = {}, defaultdict(list)
    for r in retrievers:
        summary[r.name] = {}
        for set_name, queries in sets.items():
            agg, lat = defaultdict(list), []
            for q in queries:
                rel = set(q["relevant"])
                t0 = time.perf_counter()
                ranked = r.search(q["query"], args.k)
                lat.append((time.perf_counter() - t0) * 1000)
                for m in SET_METRICS[set_name]:
                    agg[m].append(score(m, ranked, rel))
                if set_name != "region_category" and not any(x in rel for x in ranked[:5]):
                    failures[r.name].append({
                        "set": set_name, "id": q["id"], "query": q["query"],
                        "gold": [title_of.get(x) for x in q["relevant"]][:3],
                        "top3": [title_of.get(x) for x in ranked[:3]],
                    })
            row = {m: round(statistics.mean(v), 4) for m, v in agg.items()}
            row["n"] = len(queries)
            row["latency_ms_p50"] = round(statistics.median(lat), 1)
            row["latency_ms_p95"] = round(sorted(lat)[int(len(lat) * 0.95) - 1], 1)
            summary[r.name][set_name] = row
            print(r.name, set_name, row)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    payload = {"tag": args.tag, "date": str(date.today()), "corpus_size": len(docs),
               "embed_model_baseline": BASELINE_EMBED_MODEL, "summary": summary, "failures_top5": failures}
    (out_dir / f"{args.tag}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_dir / f"{args.tag}.md", payload)
    print(f"saved results/{args.tag}.json, results/{args.tag}.md")


def write_markdown(path: Path, payload: dict):
    lines = [f"# 평가 결과: {payload['tag']} ({payload['date']})", "",
             f"코퍼스 {payload['corpus_size']:,}건. 지연시간은 Python 재현 기준(단일 프로세스, GPU 임베딩).", ""]
    for set_name, metrics in SET_METRICS.items():
        cols = metrics + ["latency_ms_p50", "latency_ms_p95"]
        n = next(iter(payload["summary"].values()))[set_name]["n"]
        lines += [f"## {set_name} (n={n})", "", "| retriever | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for rname, sets in payload["summary"].items():
            lines.append(f"| {rname} | " + " | ".join(str(sets[set_name][c]) for c in cols) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

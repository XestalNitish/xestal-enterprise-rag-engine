"""
eval_rag.py - RAG system ka accuracy / retrieval / hallucination test.

Setup:  eval_dataset.json, eval_rag.py aur retrival.py (jisme ask() hai) ek hi folder me rakho.

Run:
    python eval_rag.py                              # sab 53 questions
    python eval_rag.py --workspace ws1 --module retrival
    python eval_rag.py --limit 10                   # sirf pehle 10
    python eval_rag.py --ids Q01 Q28 Q49            # chune hue questions
    python eval_rag.py --types table formula        # sirf ek type
    python eval_rag.py -v                           # har answer print karo

Output:
    eval_results.csv   - har question ka answer, PASS/FAIL, retrieval hit, latency, failure cause
    eval_summary.json  - overall numbers

Metrics:
    Answerable accuracy   : answer sahi hai (keyword match) aur refusal nahi hai
    Retrieval hit rate    : retrieved chunks me expected evidence mila
    False refusal rate    : answer doc me tha par bot ne "nahi hai" bol diya
    Out-of-scope handled  : doc se bahar ke sawaal par bot ne sahi refuse kiya (nahi to hallucination)
    Failure cause         : retrieval (chunk hi nahi mila) / generation (chunk mila par answer galat) / error
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REFUSAL_MARKERS = [
    "context mein ye information nahi hai",
    "information nahi hai",
    "not in the context",
]


# ───────────────────────── matching helpers ─────────────────────────
def norm(s: Any) -> str:
    s = str(s).lower()
    return s.replace("−", "-").replace("–", "-").replace("—", "-")


def compact(s: Any) -> str:
    """Whitespace, underscore, braces, backslash, $ hata do taaki '1 / χ_f' == '1/χf'."""
    return re.sub(r"[\s_{}\\$]", "", norm(s))


def hit(text: str, kw: str) -> bool:
    if kw.startswith("re:"):
        return re.search(kw[3:], norm(text)) is not None
    return compact(kw) in compact(text)


def answer_correct(text: str, must: List[List[str]]) -> bool:
    return all(any(hit(text, k) for k in group) for group in must)


def is_refusal(answer: str) -> bool:
    n = norm(answer)
    return any(m in n for m in REFUSAL_MARKERS)


def pct(n: int, d: int) -> Optional[float]:
    return round(100.0 * n / d, 1) if d else None


def fmt(n: int, d: int) -> str:
    p = pct(n, d)
    return f"{n}/{d} ({p}%)" if p is not None else "n/a"


# ───────────────────────── one question ─────────────────────────
def evaluate(item: Dict[str, Any], ask, workspace: str) -> Dict[str, Any]:
    q = item["question"]
    err = ""
    t0 = time.perf_counter()
    try:
        res = ask(q, workspace) or {}
    except Exception as e:  # noqa: BLE001
        res, err = {}, f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - t0

    answer = str(res.get("answer", "") or "")
    if not err and res.get("status") == "error":
        err = str(res.get("error") or "status=error")

    chunks = res.get("chunks") or []
    chunk_text = " ".join(
        str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in chunks
    )

    refused = is_refusal(answer)
    should = item.get("should_answer", True)
    evidence = item.get("evidence") or []

    if should:
        correct = (not err) and (not refused) and answer_correct(answer, item.get("must", []))
        retr_hit: Optional[bool] = any(hit(chunk_text, e) for e in evidence) if evidence else None
        if correct:
            cause = ""
        elif err:
            cause = "error"
        elif retr_hit is False:
            cause = "retrieval"
        else:
            cause = "generation"
    else:
        correct = (not err) and refused
        retr_hit = None
        cause = "" if correct else ("error" if err else "hallucination")

    return {
        "id": item["id"],
        "type": item.get("type", ""),
        "difficulty": item.get("difficulty", ""),
        "page": item.get("page", ""),
        "should_answer": should,
        "question": q,
        "expected_answer": item.get("expected_answer", ""),
        "bot_answer": answer,
        "result": "PASS" if correct else "FAIL",
        "refused": refused,
        "retrieval_hit": "" if retr_hit is None else retr_hit,
        "n_chunks": len(chunks),
        "latency_s": round(latency, 2),
        "failure_cause": cause,
        "error": err,
    }


# ───────────────────────── summary ─────────────────────────
def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ans = [r for r in rows if r["should_answer"]]
    oos = [r for r in rows if not r["should_answer"]]

    ans_ok = sum(r["result"] == "PASS" for r in ans)
    retr_rows = [r for r in ans if r["retrieval_hit"] != ""]
    retr_ok = sum(r["retrieval_hit"] is True for r in retr_rows)
    false_ref = sum(r["refused"] for r in ans)
    oos_ok = sum(r["result"] == "PASS" for r in oos)
    overall_ok = sum(r["result"] == "PASS" for r in rows)

    lats = sorted(r["latency_s"] for r in rows)
    p95 = lats[max(0, math.ceil(0.95 * len(lats)) - 1)] if lats else None

    def group(key: str) -> Dict[str, str]:
        g: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for r in ans:
            g[r[key]][1] += 1
            g[r[key]][0] += r["result"] == "PASS"
        return {k: fmt(v[0], v[1]) for k, v in sorted(g.items())}

    causes: Dict[str, int] = defaultdict(int)
    for r in rows:
        if r["failure_cause"]:
            causes[r["failure_cause"]] += 1

    return {
        "total": len(rows),
        "overall_pass": fmt(overall_ok, len(rows)),
        "answerable_accuracy": fmt(ans_ok, len(ans)),
        "retrieval_hit_rate": fmt(retr_ok, len(retr_rows)),
        "false_refusal_rate": fmt(false_ref, len(ans)),
        "out_of_scope_handled": fmt(oos_ok, len(oos)),
        "hallucination_rate_oos": fmt(len(oos) - oos_ok, len(oos)),
        "latency_avg_s": round(statistics.mean(lats), 2) if lats else None,
        "latency_median_s": round(statistics.median(lats), 2) if lats else None,
        "latency_p95_s": p95,
        "by_type": group("type"),
        "by_difficulty": group("difficulty"),
        "failure_causes": dict(causes),
        "failed_ids": [r["id"] for r in rows if r["result"] == "FAIL"],
    }


def print_summary(s: Dict[str, Any]) -> None:
    print("\n" + "=" * 54)
    print(" RAG EVAL SUMMARY")
    print("=" * 54)
    print(f" Overall pass            : {s['overall_pass']}")
    print(f" Answerable accuracy     : {s['answerable_accuracy']}")
    print(f" Retrieval hit rate      : {s['retrieval_hit_rate']}")
    print(f" False refusal rate      : {s['false_refusal_rate']}")
    print(f" Out-of-scope handled    : {s['out_of_scope_handled']}")
    print(f" Hallucination (oos)     : {s['hallucination_rate_oos']}")
    print(f" Latency avg/median/p95  : {s['latency_avg_s']}s / {s['latency_median_s']}s / {s['latency_p95_s']}s")
    print("\n By type (answerable):")
    for k, v in s["by_type"].items():
        print(f"   {k:<14} {v}")
    print("\n By difficulty (answerable):")
    for k, v in s["by_difficulty"].items():
        print(f"   {k:<14} {v}")
    print("\n Failure causes:", s["failure_causes"] or "none")
    print(" Failed IDs    :", ", ".join(s["failed_ids"]) or "none")
    print("=" * 54)
    print(" retrieval  = chunk hi nahi mila  -> chunking/embedding/retriever check karo")
    print(" generation = chunk mila par answer galat -> prompt/LLM check karo")
    print("=" * 54)


# ───────────────────────── main ─────────────────────────
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console fix
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="RAG evaluation")
    ap.add_argument("--module", default="retrival", help="python module jisme ask(question, workspace_id) ho")
    ap.add_argument("--workspace", default="ws1")
    ap.add_argument("--dataset", default=str(Path(__file__).with_name("eval_dataset.json")))
    ap.add_argument("--out", default="eval_results.csv")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--types", nargs="*")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay between questions in seconds to avoid API rate limits")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ask = getattr(importlib.import_module(args.module), "ask")

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    items = data["questions"]
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
    if args.types:
        items = [i for i in items if i.get("type") in set(args.types)]
    if args.limit:
        items = items[: args.limit]

    rows: List[Dict[str, Any]] = []
    for n, item in enumerate(items, 1):
        row = evaluate(item, ask, args.workspace)
        rows.append(row)
        extra = f" cause={row['failure_cause']}" if row["failure_cause"] else ""
        print(f"[{n}/{len(items)}] {row['id']} {row['result']} {row['latency_s']}s{extra}", flush=True)
        if args.verbose:
            print(f"    Q: {row['question']}\n    A: {row['bot_answer'][:300]}\n    exp: {row['expected_answer'][:200]}", flush=True)
        if args.delay > 0 and n < len(items):
            time.sleep(args.delay)

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    out.with_name("eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_summary(summary)
    print(f"\nSaved: {out}  and  {out.with_name('eval_summary.json')}")


if __name__ == "__main__":
    main()

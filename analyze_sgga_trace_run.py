#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from statistics import mean
from pathlib import Path


def infer_shard_slices_from_merged_results(records):
    slices = defaultdict(list)
    current_shard = 0
    prev_idx = None
    for line_no, obj in enumerate(records):
        idx = obj.get("index")
        if prev_idx is not None and isinstance(idx, int) and isinstance(prev_idx, int) and idx < prev_idx:
            current_shard += 1
        prev_idx = idx
        slices[current_shard].append((line_no, obj))
    return slices


def load_results(results_path: Path):
    records = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # If shard_id is present, use it directly. Otherwise infer from index reset.
    has_shard = any("shard_id" in r and r["shard_id"] is not None for r in records)
    correct_by_key = {}
    question_by_key = {}

    if has_shard:
        for r in records:
            key = (int(r.get("shard_id") or 0), int(r.get("index") or 0))
            correct_by_key[key] = bool(r.get("correct"))
            question_by_key[key] = r.get("question", "")
        return correct_by_key, question_by_key

    slices = infer_shard_slices_from_merged_results(records)
    for shard_id, entries in slices.items():
        for _, r in entries:
            key = (int(shard_id), int(r.get("index") or 0))
            correct_by_key[key] = bool(r.get("correct"))
            question_by_key[key] = r.get("question", "")
    return correct_by_key, question_by_key


def iter_trace_records(trace_path: Path):
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Trace files can rarely contain a truncated final line if a job terminates mid-write.
                # Skip malformed records to keep analysis robust.
                continue


def analyze_run(run_dir: Path):
    results_path = run_dir / "gsm8k_graph_aware_sg_ga.jsonl"
    if not results_path.exists():
        raise SystemExit(f"Missing results file: {results_path}")

    correct_by, question_by = load_results(results_path)

    trace_paths = sorted(run_dir.glob("trace_sag_ga_shard*.jsonl"))
    if not trace_paths:
        # fallback
        trace_paths = sorted(run_dir.glob("trace_sag_ga*.jsonl"))
    if not trace_paths:
        raise SystemExit(f"No trace files found under: {run_dir}")

    re_digit = re.compile(r"\d")

    agg = defaultdict(lambda: {
        "steps": 0,
        "remasked": 0,
        "digit_remasked": 0,
        "priorities": [],
    })

    for tp in trace_paths:
        for obj in iter_trace_records(tp):
            if obj.get("type") != "step":
                continue
            shard_id = obj.get("shard_id")
            sample_id = obj.get("sample_id")
            if shard_id is None or sample_id is None:
                continue
            key = (int(shard_id), int(sample_id))

            agg[key]["steps"] = max(agg[key]["steps"], int(obj.get("step") or 0))
            rem = obj.get("remasked", [])
            agg[key]["remasked"] += len(rem)
            for r in rem:
                pr = r.get("priority")
                if pr is not None:
                    agg[key]["priorities"].append(float(pr))
                ts = r.get("token_str_before", "")
                if ts and re_digit.search(ts):
                    agg[key]["digit_remasked"] += 1

    keys = sorted(correct_by.keys())
    if not keys:
        raise SystemExit("No results keys found (unexpected).")

    # Summary tables
    items = []
    for k in keys:
        corr = bool(correct_by[k])
        a = agg.get(k, {})
        rem = int(a.get("remasked", 0))
        steps = int(a.get("steps", 0))
        digit = int(a.get("digit_remasked", 0))
        frac = (digit / rem) if rem else 0.0
        items.append((corr, rem, steps, frac, k))

    acc = sum(1 for corr, *_ in items if corr) / len(items)
    print(f"Run: {run_dir}")
    print(f"Samples: {len(items)}  Accuracy: {acc:.4f}")
    print(f"Trace files: {', '.join(p.name for p in trace_paths)}")
    print("")

    # Accuracy by remask bins
    bins = [(0, 50), (50, 80), (80, 120), (120, 200), (200, 10**9)]
    print("Accuracy vs total remasks per sample")
    for lo, hi in bins:
        subset = [corr for corr, rem, *_ in items if lo <= rem < hi]
        if not subset:
            continue
        print(f"- remasks[{lo},{hi}): n={len(subset)} acc={sum(subset)/len(subset):.3f}")
    print("")

    # Accuracy by digit remask fraction (filter low-remask samples)
    bins = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.01)]
    print("Accuracy vs digit-remask fraction (only remasks>=30)")
    for lo, hi in bins:
        subset = [corr for corr, rem, _, frac, _ in items if rem >= 30 and lo <= frac < hi]
        if not subset:
            continue
        print(f"- digit_frac[{lo},{hi}): n={len(subset)} acc={sum(subset)/len(subset):.3f}")
    print("")

    # Most unstable wrong samples
    wrong = [(rem, steps, frac, k) for corr, rem, steps, frac, k in items if not corr]
    wrong.sort(reverse=True)
    print("Top wrong samples by remask count")
    for rem, steps, frac, k in wrong[:10]:
        q = question_by.get(k, "")
        q_short = (q[:100] + "...") if len(q) > 100 else q
        print(f"- key={k} remasks={rem} steps={steps} digit_frac={frac:.3f} question={q_short!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    args = ap.parse_args()
    analyze_run(Path(args.run_dir))


if __name__ == "__main__":
    main()

"""Reproduce post-hoc candidate coverage, selection, score-tail and deadline diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
SEED = 20261118
BOOTSTRAPS = 2000
THRESHOLD = 0.90
KS = (1, 2, 4, 8)
DEADLINES = (6, 12, 18, 24, 30)
CHECKS = (
    "eval_terminology", "eval_numbers", "eval_negation", "eval_conditions",
    "eval_modality", "eval_omission", "eval_addition",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().casefold() in {"true", "1", "yes"}


def load_data() -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, int], dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for raw in read_csv(ROOT / "data" / "candidates.csv"):
        row: dict[str, Any] = dict(raw)
        for field in ("rule_pass", "selector_pass", "selector_usable", "eval_usable"):
            row[field] = as_bool(row[field])
        for field in ("index", "selector_adequacy", "selector_fluency", "eval_adequacy", "eval_fluency"):
            row[field] = int(row[field])
        candidates.append(row)

    timing: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for raw in read_csv(ROOT / "data" / "generation_times.csv"):
        key = (raw["case_id"], raw["run"], raw["strategy"], int(raw["index"]))
        if key in timing:
            raise ValueError(f"Duplicate generation timing key: {key}")
        row = dict(raw)
        row["index"] = int(raw["index"])
        row["latency_s"] = float(raw["latency_s"])
        timing[key] = row

    cand_keys = {(r["case_id"], r["run"], r["strategy"], r["index"]) for r in candidates}
    if cand_keys != set(timing):
        missing_times = sorted(cand_keys - set(timing))[:5]
        missing_candidates = sorted(set(timing) - cand_keys)[:5]
        raise ValueError(f"Candidate/timing keys differ; missing times={missing_times}, extra times={missing_candidates}")
    if len(candidates) != 816 or len(timing) != 816:
        raise ValueError(f"Expected 816 test candidates and timings; got {len(candidates)} and {len(timing)}")
    if any(r["strategy"] not in {"baseline", "ordinary", "diverse"} for r in candidates):
        raise ValueError("Unexpected candidate strategy")
    return candidates, timing


def usable(row: dict[str, Any]) -> bool:
    return bool(row["rule_pass"] and row["eval_usable"])


def selected_by_filter(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [r for r in rows if r["rule_pass"] and r["selector_pass"]]
    if not eligible:
        return None
    return min(eligible, key=lambda r: (-r["selector_adequacy"], -r["selector_fluency"], r["index"]))


def selected_by_quality(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda r: (-r["selector_adequacy"], -r["selector_fluency"], r["index"]))


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in normalized if unicodedata.category(ch)[0] in {"L", "N"})


def similarity(left: str, right: str) -> float:
    a, b = normalize(left), normalize(right)
    if a == b:
        return 1.0
    ca = Counter(a[i:i + 2] for i in range(max(0, len(a) - 1)))
    cb = Counter(b[i:i + 2] for i in range(max(0, len(b) - 1)))
    denominator = sum(ca.values()) + sum(cb.values())
    if denominator == 0:
        return 0.0
    overlap = sum((ca & cb).values())
    return 2.0 * overlap / denominator


def medoid(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    scores = [0.0] * len(rows)
    for i, j in combinations(range(len(rows)), 2):
        sim = similarity(rows[i]["text"], rows[j]["text"])
        scores[i] += sim
        scores[j] += sim
    best = max(range(len(rows)), key=lambda i: (scores[i], -rows[i]["index"]))
    return rows[best]


def deduplicated_medoid(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(normalize(row["text"]), row)
    return medoid(list(unique.values()))


def majority_cluster_medoid(rows: list[dict[str, Any]], threshold: float = THRESHOLD) -> dict[str, Any] | None:
    if not rows:
        return None
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i, j in combinations(range(len(rows)), 2):
        if similarity(rows[i]["text"], rows[j]["text"]) >= threshold:
            union(i, j)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    largest = min(groups.values(), key=lambda group: (-len(group), min(r["index"] for r in group)))
    return medoid(largest)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def cluster_bootstrap_ci(rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float],
                         cluster_field: str = "cluster_id", seed: int = SEED) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row[cluster_field])].append(row)
    keys = sorted(clusters)
    if len(keys) < 2:
        return None, None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAPS):
        draw = [rng.choice(keys) for _ in keys]
        values = [value(row) for key in draw for row in clusters[key]]
        if values:
            estimates.append(statistics.mean(values))
    estimates.sort()
    return estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]


def mean_ci(rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
    estimate = mean([value(r) for r in rows])
    lo, hi = cluster_bootstrap_ci(rows, value)
    return {"mean": estimate, "ci95_low": lo, "ci95_high": hi, "n_rows": len(rows),
            "n_clusters": len({r["cluster_id"] for r in rows})}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pool_groups(candidates: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["strategy"] in {"ordinary", "diverse"}:
            groups[(row["case_id"], row["run"], row["strategy"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: r["index"])
    return groups


def candidate_trajectory(groups: dict[tuple[str, str, str], list[dict[str, Any]]],
                         timing: dict[tuple[str, str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (case, run, strategy), full_pool in sorted(groups.items()):
        for k in KS:
            rows = full_pool[:k]
            best_filtered = selected_by_filter(rows)
            best_quality = selected_by_quality(rows)
            text_medoid = medoid(rows)
            unique_valid = []
            for row in rows:
                if usable(row) and all(similarity(row["text"], old["text"]) < THRESHOLD for old in unique_valid):
                    unique_valid.append(row)
            elapsed = sum(timing[(case, run, strategy, r["index"])]["latency_s"] for r in rows)
            output.append({
                "case_id": case, "cluster_id": rows[0]["cluster_id"], "domain": rows[0]["domain"],
                "run": run, "strategy": strategy, "k": k, "candidate_count": len(rows),
                "pool_coverage": int(any(usable(r) for r in rows)), "valid_candidate_count": sum(usable(r) for r in rows),
                "distinct_valid_count_0_90": len(unique_valid),
                "mean_pairwise_similarity": mean([similarity(a["text"], b["text"]) for a, b in combinations(rows, 2)]) or 0.0,
                "serial_generation_seconds": round(elapsed, 6),
                "medoid_candidate_id": text_medoid["candidate_id"] if text_medoid else "",
                "medoid_success": int(usable(text_medoid)) if text_medoid else 0,
                "filtered_candidate_id": best_filtered["candidate_id"] if best_filtered else "",
                "filtered_success": int(usable(best_filtered)) if best_filtered else 0,
                "quality_only_candidate_id": best_quality["candidate_id"] if best_quality else "",
                "quality_only_success": int(usable(best_quality)) if best_quality else 0,
                "oracle_success": int(any(usable(r) for r in rows)),
            })
    return output


def merge_replay(groups: dict[tuple[str, str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    methods = {
        "raw_medoid": medoid,
        "normalized_dedup_medoid": deduplicated_medoid,
        "near_duplicate_majority_medoid": majority_cluster_medoid,
    }
    output = []
    for (case, run, strategy), full_pool in sorted(groups.items()):
        for k in KS:
            rows = full_pool[:k]
            coverage = int(any(usable(r) for r in rows))
            for name, select in methods.items():
                chosen = select(rows)
                output.append({
                    "case_id": case, "cluster_id": rows[0]["cluster_id"], "domain": rows[0]["domain"],
                    "run": run, "strategy": strategy, "k": k, "method": name,
                    "selected_candidate_id": chosen["candidate_id"] if chosen else "",
                    "selected_success": int(usable(chosen)) if chosen else 0,
                    "pool_coverage": coverage, "valid_candidate_count": sum(usable(r) for r in rows),
                    "normalized_unique_count": len({normalize(r["text"]) for r in rows}),
                    "near_cluster_count_0_90": len(near_components(rows)),
                    "fuzzy_threshold": THRESHOLD,
                })
    return output


def near_components(rows: list[dict[str, Any]], threshold: float = THRESHOLD) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in combinations(range(len(rows)), 2):
        if similarity(rows[i]["text"], rows[j]["text"]) >= threshold:
            a, b = find(i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return sorted(groups.values(), key=lambda group: min(r["index"] for r in group))


def summarize_merge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    subset = [r for r in rows if r["k"] == 8]
    for strategy in ("ordinary", "diverse", "all"):
        group = [r for r in subset if strategy == "all" or r["strategy"] == strategy]
        output[strategy] = {}
        raw = [r for r in group if r["method"] == "raw_medoid"]
        for method in ("raw_medoid", "normalized_dedup_medoid", "near_duplicate_majority_medoid"):
            current = [r for r in group if r["method"] == method]
            diffs = []
            if method != "raw_medoid":
                lookup = {(r["case_id"], r["run"], r["strategy"]): r for r in raw}
                diffs = [{**r, "success_difference": r["selected_success"] - lookup[(r["case_id"], r["run"], r["strategy"])]["selected_success"]}
                         for r in current]
            output[strategy][method] = {
                "selection_success": mean([r["selected_success"] for r in current]),
                "coverage": mean([r["pool_coverage"] for r in current]),
                "changed_selection_vs_raw": (sum(r["selected_candidate_id"] != next(x["selected_candidate_id"] for x in raw if (x["case_id"], x["run"], x["strategy"]) == (r["case_id"], r["run"], r["strategy"])) for r in current) / len(current)) if method != "raw_medoid" and current else 0.0,
                "paired_success_delta_vs_raw": mean([r["success_difference"] for r in diffs]) if diffs else 0.0,
                "paired_success_delta_ci95": cluster_bootstrap_ci(diffs, lambda r: r["success_difference"]) if diffs else (None, None),
                "n_pools": len(current),
            }
    return output


def sampling_rows(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["case_id"], r["run"], r["strategy"]): r for r in trajectory if r["k"] == 8}
    output = []
    for case, run in sorted({(r["case_id"], r["run"]) for r in trajectory}):
        ordinary, diverse = lookup[(case, run, "ordinary")], lookup[(case, run, "diverse")]
        row = {"case_id": case, "cluster_id": ordinary["cluster_id"], "domain": ordinary["domain"], "run": run}
        for metric in ("pool_coverage", "valid_candidate_count", "distinct_valid_count_0_90", "mean_pairwise_similarity", "filtered_success", "medoid_success"):
            row[f"ordinary_{metric}"] = ordinary[metric]
            row[f"diverse_{metric}"] = diverse[metric]
            row[f"delta_{metric}"] = diverse[metric] - ordinary[metric]
        output.append(row)
    return output


def summarize_sampling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for domain in ("all", "standards", "product_docs", "paired_synthetic"):
        subset = [r for r in rows if domain == "all" or r["domain"] == domain]
        metrics = {}
        for metric in ("pool_coverage", "valid_candidate_count", "distinct_valid_count_0_90", "mean_pairwise_similarity", "filtered_success", "medoid_success"):
            base = [{**r, "value": r[f"ordinary_{metric}"]} for r in subset]
            diff = [{**r, "value": r[f"delta_{metric}"]} for r in subset]
            metrics[metric] = {
                "ordinary": mean_ci(base, lambda r: r["value"]),
                "diverse": mean_ci([{**r, "value": r[f"diverse_{metric}"]} for r in subset], lambda r: r["value"]),
                "paired_delta_diverse_minus_ordinary": mean([r["delta_" + metric] for r in subset]),
                "paired_delta_ci95": cluster_bootstrap_ci(diff, lambda r: r["value"]),
            }
        output[domain] = {"n_pairs": len(subset), "n_clusters": len({r["cluster_id"] for r in subset}), "metrics": metrics}
    return output


def high_score_tail(groups: dict[tuple[str, str, str], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    detail, grouped = [], defaultdict(list)
    for (case, run, strategy), rows in sorted(groups.items()):
        ranked = sorted(rows, key=lambda r: (-r["selector_adequacy"], -r["selector_fluency"], r["index"]))
        for rank, row in enumerate(ranked, 1):
            rank_bin = "top_1" if rank == 1 else "rank_2_4" if rank <= 4 else "rank_5_8"
            record = {"case_id": case, "cluster_id": row["cluster_id"], "domain": row["domain"], "run": run,
                      "strategy": strategy, "candidate_id": row["candidate_id"], "rank": rank, "rank_bin": rank_bin,
                      "selector_adequacy": row["selector_adequacy"], "selector_fluency": row["selector_fluency"],
                      "selector_usable": int(row["selector_usable"]), "candidate_success": int(usable(row)),
                      "high_score_false_positive": int(row["selector_usable"] and not usable(row)),
                      "eval_adequacy": row["eval_adequacy"], "eval_fluency": row["eval_fluency"],
                      "failed_check_count": sum(row[k] == "fail" for k in CHECKS)}
            detail.append(record)
            grouped[(row["domain"], strategy, rank_bin)].append(record)
    summary = []
    for (domain, strategy, rank_bin), rows in sorted(grouped.items()):
        approved = [r for r in rows if r["selector_usable"]]
        summary.append({"domain": domain, "strategy": strategy, "rank_bin": rank_bin,
                        "n_candidates": len(rows), "selector_usable_n": len(approved),
                        "candidate_success_rate": mean([r["candidate_success"] for r in rows]),
                        "candidate_success_ci95": cluster_bootstrap_ci(rows, lambda r: float(r["candidate_success"])),
                        "selector_usable_rate": mean([r["selector_usable"] for r in rows]),
                        "false_positive_rate_among_selector_usable": mean([r["high_score_false_positive"] for r in rows]) if approved else None,
                        "false_positive_rate_ci95": cluster_bootstrap_ci(approved, lambda r: float(r["high_score_false_positive"])) if approved else (None, None),
                        "mean_selector_adequacy": mean([r["selector_adequacy"] for r in rows]),
                        "mean_selector_fluency": mean([r["selector_fluency"] for r in rows])})
    top_comparisons = []
    pool_map = defaultdict(dict)
    for r in detail:
        pool_map[(r["case_id"], r["run"], r["strategy"])][r["rank"]] = r
    for (case, run, strategy), ranks in sorted(pool_map.items()):
        rest = [ranks[i]["candidate_success"] for i in range(2, 9)]
        top_comparisons.append({"case_id": case, "cluster_id": ranks[1]["cluster_id"], "domain": ranks[1]["domain"],
                                "run": run, "strategy": strategy,
                                "top1_success": ranks[1]["candidate_success"],
                                "rank_2_8_mean_success": statistics.mean(rest),
                                "top1_minus_rest": ranks[1]["candidate_success"] - statistics.mean(rest),
                                "top1_high_score_false_positive": ranks[1]["high_score_false_positive"]})
    result = {}
    for label, subset in [("overall", top_comparisons)] + [(d, [r for r in top_comparisons if r["domain"] == d]) for d in ("standards", "product_docs", "paired_synthetic")]:
        result[label] = {"n_pools": len(subset), "top1_success_rate": mean([r["top1_success"] for r in subset]),
                         "rank_2_8_success_rate": mean([r["rank_2_8_mean_success"] for r in subset]),
                         "paired_top1_minus_rank_2_8": mean([r["top1_minus_rest"] for r in subset]),
                         "paired_delta_ci95": cluster_bootstrap_ci(subset, lambda r: r["top1_minus_rest"]),
                         "top1_selector_usable_false_positive_n": sum(r["top1_high_score_false_positive"] for r in subset)}
    return detail, summary, result


def completion_times(rows: list[dict[str, Any]], timing: dict[tuple[str, str, str, int], dict[str, Any]], workers: int) -> dict[int, float]:
    """Schedule index-ordered candidate jobs onto identical workers; jobs are launched as slots free."""
    free = [0.0] * workers
    completed: dict[int, float] = {}
    for row in sorted(rows, key=lambda r: r["index"]):
        worker = min(range(workers), key=lambda i: (free[i], i))
        start = free[worker]
        finish = start + timing[(row["case_id"], row["run"], row["strategy"], row["index"])]["latency_s"]
        free[worker] = finish
        completed[row["index"]] = finish
    return completed


def deadline_replay(groups: dict[tuple[str, str, str], list[dict[str, Any]]], timing: dict[tuple[str, str, str, int], dict[str, Any]],
                    all_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    baseline = defaultdict(list)
    for r in all_candidates:
        if r["strategy"] == "baseline":
            baseline[(r["case_id"], r["run"])].append(r)
    for (case, run, strategy), rows in sorted(groups.items()):
        for workers in (1, 6):
            finishes = completion_times(rows, timing, workers)
            for deadline in DEADLINES:
                available = [r for r in rows if finishes[r["index"]] <= deadline]
                for method, select in (("medoid", medoid), ("selector_filtered", selected_by_filter), ("selector_quality_only", selected_by_quality)):
                    selected = select(available)
                    output.append({"case_id": case, "cluster_id": rows[0]["cluster_id"], "domain": rows[0]["domain"],
                                   "run": run, "strategy": strategy, "workers_per_pool": workers,
                                   "deadline_s": deadline, "method": method, "completed_candidates": len(available),
                                   "coverage": int(any(usable(r) for r in available)),
                                   "output": int(selected is not None), "selected_success": int(usable(selected)) if selected else 0,
                                   "selected_candidate_id": selected["candidate_id"] if selected else ""})
    for (case, run), rows in sorted(baseline.items()):
        candidate = rows[0]
        seconds = timing[(case, run, "baseline", candidate["index"])]["latency_s"]
        for deadline in DEADLINES:
            available = seconds <= deadline
            output.append({"case_id": case, "cluster_id": candidate["cluster_id"], "domain": candidate["domain"],
                           "run": run, "strategy": "baseline", "workers_per_pool": 1, "deadline_s": deadline,
                           "method": "single_candidate", "completed_candidates": int(available),
                           "coverage": int(available and usable(candidate)), "output": int(available),
                           "selected_success": int(available and usable(candidate)),
                           "selected_candidate_id": candidate["candidate_id"] if available else ""})
    return output


def summarize_deadline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("strategy", "workers_per_pool", "deadline_s", "method")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, subset in sorted(groups.items()):
        output.append({**dict(zip(keys, key)), "n_pools": len(subset),
                       "mean_completed_candidates": mean([r["completed_candidates"] for r in subset]),
                       "mean_completed_candidates_ci95": cluster_bootstrap_ci(subset, lambda r: float(r["completed_candidates"])),
                       "coverage_rate": mean([r["coverage"] for r in subset]),
                       "coverage_ci95": cluster_bootstrap_ci(subset, lambda r: float(r["coverage"])),
                       "output_rate": mean([r["output"] for r in subset]),
                       "output_rate_ci95": cluster_bootstrap_ci(subset, lambda r: float(r["output"])),
                       "success_per_expected_task": mean([r["selected_success"] for r in subset]),
                       "success_per_expected_task_ci95": cluster_bootstrap_ci(subset, lambda r: float(r["selected_success"])),
                       "success_among_outputs": mean([r["selected_success"] for r in subset if r["output"]]),
                       "success_among_outputs_ci95": cluster_bootstrap_ci([r for r in subset if r["output"]], lambda r: float(r["selected_success"]))})
    return output


def validation_report(candidates: list[dict[str, Any]], timing: dict[tuple[str, str, str, int], dict[str, Any]]) -> dict[str, Any]:
    review = [json.loads(line) for line in (ROOT / "data" / "review_batch.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    astra = json.loads((ROOT / "data" / "astra_review.json").read_text(encoding="utf-8"))
    review_index = json.loads((ROOT / "data" / "review_index.json").read_text(encoding="utf-8"))
    ids = {r["case_id"] for r in candidates}
    assert len(ids) == 24 and len({r["cluster_id"] for r in candidates}) == 20
    assert all(r["text"] and len(r["text"]) <= 5000 for r in candidates)
    assert all(0 < r["latency_s"] < 180 for r in timing.values())
    assert all(set(r) == {"review_id", "domain", "original", "outputs"} for r in review)
    assert all(all(set(o) == {"output_id", "translation"} for o in r["outputs"]) for r in review)
    assert all(r["original"] and r["outputs"] for r in review)
    review_ids = {r["review_id"] for r in review}
    assert {r["id"] for r in astra["items"]} == review_ids
    assert {r["review_id"] for r in review_index["items"]} == review_ids
    assert all(r["A"] in {"通过", "应修订", "通过但偏窄"} and r["B"] in {"通过", "应修订", "通过但偏窄"} and r["note"] for r in astra["items"])
    candidate_by_id = {r["candidate_id"]: r for r in candidates}
    assert all(len(r["outputs"]) == 2 and len({o["output_id"] for o in r["outputs"]}) == 2 for r in review_index["items"])
    assert all(o["matching_candidate_ids"] and all(cid in candidate_by_id and candidate_by_id[cid]["case_id"] == r["case_id"] for cid in o["matching_candidate_ids"])
               for r in review_index["items"] for o in r["outputs"])
    return {"status": "passed", "checks": {
        "candidate_rows": len(candidates), "timing_rows": len(timing), "case_count": len(ids),
        "cluster_count": len({r["cluster_id"] for r in candidates}),
        "candidate_timing_keys_match": True, "candidate_texts_present": True,
        "timing_values_finite_positive_and_below_timeout": True,
        "blind_review_examples": len(review), "blind_review_outputs": sum(len(r["outputs"]) for r in review),
        "review_packet_contains_no_candidate_judgment_fields": True,
        "astra_review_items": len(astra["items"]), "review_index_matches_packet": True,
    }}


def plot_results(trajectory: list[dict[str, Any]], merges: list[dict[str, Any]], tail: list[dict[str, Any]],
                 deadline_summary: list[dict[str, Any]], outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plot generation requires matplotlib; install requirements.txt in the active environment.") from exc
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    palette = {"ordinary": "#376996", "diverse": "#e07a5f", "standards": "#3d5a80",
               "product_docs": "#e07a5f", "paired_synthetic": "#81b29a"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    for strategy in ("ordinary", "diverse"):
        subset = [r for r in trajectory if r["strategy"] == strategy]
        for ax, metric, title in ((axes[0], "pool_coverage", "Candidate-pool coverage"),
                                  (axes[1], "filtered_success", "Filtered selection success")):
            means, lows, highs = [], [], []
            for k in KS:
                vals = [r for r in subset if r["k"] == k]
                ci = mean_ci(vals, lambda r: float(r[metric]))
                means.append(100 * ci["mean"]); lows.append(100 * (ci["mean"] - ci["ci95_low"])); highs.append(100 * (ci["ci95_high"] - ci["mean"]))
            ax.errorbar(KS, means, yerr=[lows, highs], marker="o", capsize=3, label=strategy, color=palette[strategy])
            ax.set_title(title); ax.set_xlabel("Candidate budget K"); ax.set_xticks(KS); ax.set_ylim(0, 105); ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Rate (%)"); axes[1].legend(frameon=False)
    fig.suptitle("Coverage and selected quality across the observed candidate budget", y=1.02)
    fig.tight_layout(); fig.savefig(outdir / "fig1_budget_coverage_selection.png", dpi=180, bbox_inches="tight"); fig.savefig(outdir / "fig1_budget_coverage_selection.svg", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    methods = ("raw_medoid", "normalized_dedup_medoid", "near_duplicate_majority_medoid")
    labels = ("Raw medoid", "Normalized dedup", "Near-duplicate majority")
    for i, method in enumerate(methods):
        vals = [r for r in merges if r["k"] == 8 and r["method"] == method]
        ci = mean_ci(vals, lambda r: float(r["selected_success"]))
        ax.errorbar(i, 100 * ci["mean"], yerr=[[100 * (ci["mean"] - ci["ci95_low"])], [100 * (ci["ci95_high"] - ci["mean"])]],
                    fmt="o", markersize=8, capsize=4, color="#376996")
    ax.set_xticks(range(len(labels)), labels, rotation=12, ha="right"); ax.set_ylabel("Selection success (%)"); ax.set_ylim(0, 105)
    ax.set_title("Fixed K=8 pools: agreement-based merge choices"); ax.grid(axis="y", alpha=.22)
    fig.tight_layout(); fig.savefig(outdir / "fig2_merge_replay.png", dpi=180, bbox_inches="tight"); fig.savefig(outdir / "fig2_merge_replay.svg", bbox_inches="tight"); plt.close(fig)

    order = ("top_1", "rank_2_4", "rank_5_8")
    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    for domain in ("standards", "product_docs", "paired_synthetic"):
        rates = []
        for rank_bin in order:
            vals = [r for r in tail if r["domain"] == domain and r["rank_bin"] == rank_bin]
            rates.append(100 * mean([r["candidate_success"] for r in vals]) if vals else float("nan"))
        ax.plot(range(3), rates, marker="o", label=domain.replace("_", " "), color=palette[domain])
    ax.set_xticks(range(3), ("Rank 1", "Ranks 2–4", "Ranks 5–8")); ax.set_ylabel("Independent evaluator success (%)"); ax.set_ylim(0, 105)
    ax.set_title("Observed score rank versus evaluator usability"); ax.grid(axis="y", alpha=.22); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir / "fig3_verifier_tail.png", dpi=180, bbox_inches="tight"); fig.savefig(outdir / "fig3_verifier_tail.svg", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    styles = {1: "-", 6: "--"}
    markers = {"ordinary": "o", "diverse": "s"}
    for strategy in ("ordinary", "diverse"):
        for workers in (1, 6):
            vals = [r for r in deadline_summary if r["strategy"] == strategy and r["method"] == "selector_filtered" and r["workers_per_pool"] == workers]
            by_deadline = {r["deadline_s"]: r for r in vals}
            label = f"{strategy}, {workers} worker(s)"
            axes[0].plot(DEADLINES, [by_deadline[d]["mean_completed_candidates"] for d in DEADLINES],
                         marker=markers[strategy], linestyle=styles[workers], label=label, color=palette[strategy])
            axes[1].plot(DEADLINES, [100 * by_deadline[d]["success_per_expected_task"] for d in DEADLINES],
                         marker=markers[strategy], linestyle=styles[workers], label=label, color=palette[strategy])
    axes[0].set_title("Generated candidates before deadline"); axes[0].set_ylabel("Mean count per pool")
    axes[1].set_title("Selected success per expected task"); axes[1].set_ylabel("Rate (%)"); axes[1].set_ylim(0, 105)
    for ax in axes:
        ax.set_xlabel("Simulated generation deadline (seconds)"); ax.set_xticks(DEADLINES); ax.grid(axis="y", alpha=.22)
    axes[1].legend(frameon=False, fontsize=8); fig.suptitle("Historical request durations replayed under explicit concurrency assumptions", y=1.02)
    fig.tight_layout(); fig.savefig(outdir / "fig4_deadline_replay.png", dpi=180, bbox_inches="tight"); fig.savefig(outdir / "fig4_deadline_replay.svg", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="Write tabular results without rendering figures.")
    args = parser.parse_args()
    results = ROOT / "results"
    figures = ROOT / "figures"
    results.mkdir(parents=True, exist_ok=True)
    candidates, timing = load_data()
    validation = validation_report(candidates, timing)
    (results / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    groups = pool_groups(candidates)
    trajectory = candidate_trajectory(groups, timing)
    merges = merge_replay(groups)
    paired = sampling_rows(trajectory)
    tail_detail, tail_summary, tail_comparison = high_score_tail(groups)
    deadline_detail = deadline_replay(groups, timing, candidates)
    deadline_summary = summarize_deadline(deadline_detail)
    write_csv(results / "candidate_trajectories.csv", trajectory)
    write_csv(results / "merge_replay.csv", merges)
    write_csv(results / "sampling_paired.csv", paired)
    write_csv(results / "high_score_tail_candidates.csv", tail_detail)
    write_csv(results / "high_score_tail_summary.csv", tail_summary)
    write_csv(results / "deadline_replay.csv", deadline_detail)
    write_csv(results / "deadline_summary.csv", deadline_summary)
    merge_summary = summarize_merge(merges)
    summary = {
        "scope": {"status": "post_hoc_replay", "independent_evaluation": False, "cases": 24, "clusters": 20,
                  "candidate_rows": len(candidates), "generation_timing_rows": len(timing), "runs": 2,
                  "candidate_budgets": list(KS), "strategies": ["ordinary", "diverse"],
                  "seed_scope_note": "Seeds were remapped to run labels; generator randomness depends on the historical API implementation."},
        "definitions": {"usable_candidate": "rule_pass AND independent evaluator eval_usable",
                        "filtered_selection": "rule_pass and selector_pass candidates ranked by selector adequacy, fluency, then index",
                        "score_rank": "selector adequacy descending, fluency descending, then candidate index; tied scores are not separated probabilistically",
                        "merge": "character-bigram multiset F1 after NFKC/casefold and punctuation removal; near-duplicate families are connected components at 0.90",
                        "bootstrap": f"{BOOTSTRAPS} deterministic cluster bootstrap resamples over the 20 de-identified source/pair clusters"},
        "sampling_comparison": summarize_sampling(paired),
        "merge_replay_k8": merge_summary,
        "high_score_tail": tail_comparison,
        "deadline_scope": {"deadlines_seconds": list(DEADLINES), "worker_scenarios": [1, 6],
                           "time_basis": "observed generation-request service times only; no queueing, selector/evaluator, or user-facing latency",
                           "scheduler": "candidate index order, launch a new request whenever one of the per-pool worker slots frees"},
        "validation": validation,
    }
    (results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_plots:
        plot_results(trajectory, merges, tail_detail, deadline_summary, figures)
    print(json.dumps({"status": "ok", "candidates": len(candidates), "pools": len(groups),
                      "trajectory_rows": len(trajectory), "merge_rows": len(merges), "deadline_rows": len(deadline_detail),
                      "plots_written": not args.no_plots, "summary": str(results / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

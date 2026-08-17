#!/usr/bin/env python3
"""Strict parser and task-clustered statistics for the atomic_seen matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any


CONTRASTS = [
    ("cscka_adjusted", "ckaonly"),
    ("cscka_adjusted", "csonly"),
    ("cka_atmohb", "ckaonly"),
    ("cka_atmohb", "w4a8_atmohb"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def task_rates(rows: dict[tuple[str, int], dict], tasks: list[str], seeds: list[int]) -> dict[str, float]:
    return {
        task: sum(bool(rows[(task, seed)]["success"]) for seed in seeds) / len(seeds)
        for task in tasks
    }


def cluster_ci(rates: dict[str, float], n_boot: int, rng: random.Random) -> list[float]:
    tasks = list(rates)
    draws = []
    for _ in range(n_boot):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        draws.append(sum(rates[t] for t in sample) / len(sample))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def paired_delta_ci(
    rates_a: dict[str, float], rates_b: dict[str, float], n_boot: int, rng: random.Random
) -> tuple[float, list[float], list[float]]:
    tasks = list(rates_a)
    diffs = [rates_a[t] - rates_b[t] for t in tasks]
    draws = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        draws.append(sum(sample) / len(sample))
    return sum(diffs) / len(diffs), [percentile(draws, 0.025), percentile(draws, 0.975)], diffs


def exact_sign_flip_p(diffs: list[float]) -> float:
    observed = abs(sum(diffs) / len(diffs))
    extreme = 0
    total = 1 << len(diffs)
    for mask in range(total):
        value = sum((d if (mask >> i) & 1 else -d) for i, d in enumerate(diffs)) / len(diffs)
        if abs(value) >= observed - 1e-15:
            extreme += 1
    return extreme / total


def mcnemar(rows_a: dict, rows_b: dict, tasks: list[str], seeds: list[int]) -> dict[str, Any]:
    b = c = 0
    for task, seed in itertools.product(tasks, seeds):
        a = bool(rows_a[(task, seed)]["success"])
        z = bool(rows_b[(task, seed)]["success"])
        if a and not z:
            b += 1
        elif z and not a:
            c += 1
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
        p = min(1.0, 2 * tail)
    return {"a_wins": b, "b_wins": c, "two_sided_p": p}


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for index, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - index) * p))
        adjusted[name] = running
    return adjusted


def load_rows(run_dir: Path, manifest: dict, manifest_sha: str, allow_incomplete: bool) -> dict:
    all_rows = {}
    errors = []
    tasks = manifest["tasks"]
    seeds = manifest["seeds"]
    for config in manifest["configs"]:
        cid = config["id"]
        rows = {}
        for shard_index, filename in enumerate(config["result_files"]):
            path = Path(filename)
            if not path.exists():
                errors.append(f"{cid}: missing result file {path}")
                continue
            allowed_tasks = set(manifest["shards"][shard_index])
            for line_no, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_no}: malformed JSON: {exc}")
                    continue
                if row.get("config") != cid:
                    errors.append(f"{path}:{line_no}: wrong config {row.get('config')}")
                if row.get("manifest_sha256") != manifest_sha:
                    errors.append(f"{path}:{line_no}: wrong manifest hash")
                if row.get("config_sha256") != config["config_sha256"]:
                    errors.append(f"{path}:{line_no}: wrong config hash")
                task, seed = row.get("task"), row.get("seed")
                if task not in allowed_tasks or seed not in seeds:
                    errors.append(f"{path}:{line_no}: unexpected task/seed {task}/{seed}")
                    continue
                key = (task, seed)
                if key in rows:
                    errors.append(f"{cid}: duplicate task/seed {key}")
                    continue
                if row.get("crashed") or row.get("success") is None:
                    errors.append(f"{cid}: crashed/incomplete task/seed {key}")
                    continue
                if row.get("paired_action_noise") is not True:
                    errors.append(f"{cid}: unpaired action noise at {key}")
                if row.get("action_noise_scheme") != manifest["protocol"]["action_noise_scheme"]:
                    errors.append(f"{cid}: action-noise scheme mismatch at {key}")
                rows[key] = row
        expected = set(itertools.product(tasks, seeds))
        missing = sorted(expected - set(rows))
        if missing:
            errors.append(f"{cid}: missing {len(missing)} task/seed rows: {missing[:5]}")
        all_rows[cid] = rows
    if errors and not allow_incomplete:
        raise SystemExit("matrix validation failed:\n- " + "\n- ".join(errors))
    return {"rows": all_rows, "validation_errors": errors}


def summarize(run_dir: Path, n_boot: int, allow_incomplete: bool) -> dict:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = file_sha(manifest_path)
    loaded = load_rows(run_dir, manifest, manifest_sha, allow_incomplete)
    rows_by_config = loaded["rows"]
    tasks = manifest["tasks"]
    seeds = manifest["seeds"]
    primary_tasks = manifest["heldout_tasks"] or tasks
    rng = random.Random(20260817)
    configs = {}
    for config in manifest["configs"]:
        cid = config["id"]
        rows = rows_by_config[cid]
        if any((task, seed) not in rows for task, seed in itertools.product(tasks, seeds)):
            continue
        rates_all = task_rates(rows, tasks, seeds)
        rates_primary = {t: rates_all[t] for t in primary_tasks}
        per_task_details = {}
        for task in tasks:
            task_rows = [rows[(task, seed)] for seed in seeds]
            ok_steps = [r["steps"] for r in task_rows if r["success"]]
            fail_steps = [r["steps"] for r in task_rows if not r["success"]]
            per_task_details[task] = {
                "sr": rates_all[task],
                "successes": len(ok_steps),
                "episodes": len(task_rows),
                "mean_success_steps": sum(ok_steps) / len(ok_steps) if ok_steps else None,
                "mean_failure_steps": sum(fail_steps) / len(fail_steps) if fail_steps else None,
            }
        success_steps = [r["steps"] for r in rows.values() if r["success"]]
        failure_steps = [r["steps"] for r in rows.values() if not r["success"]]
        configs[cid] = {
            "per_task": rates_all,
            "per_task_details": per_task_details,
            "primary_task_macro_sr": sum(rates_primary.values()) / len(rates_primary),
            "primary_task_cluster_ci95": cluster_ci(rates_primary, n_boot, rng),
            "all18_task_macro_sr": sum(rates_all.values()) / len(rates_all),
            "episode_sr": sum(bool(r["success"]) for r in rows.values()) / len(rows),
            "successes": sum(bool(r["success"]) for r in rows.values()),
            "episodes": len(rows),
            "mean_success_steps": sum(success_steps) / len(success_steps) if success_steps else None,
            "mean_failure_steps": sum(failure_steps) / len(failure_steps) if failure_steps else None,
        }

    comparisons = {}
    raw_p = {}
    for a, b in CONTRASTS:
        if a not in configs or b not in configs:
            continue
        rates_a = {t: configs[a]["per_task"][t] for t in primary_tasks}
        rates_b = {t: configs[b]["per_task"][t] for t in primary_tasks}
        delta, ci, diffs = paired_delta_ci(rates_a, rates_b, n_boot, rng)
        name = f"{a}_vs_{b}"
        p = exact_sign_flip_p(diffs)
        raw_p[name] = p
        comparisons[name] = {
            "a": a, "b": b, "primary_task_macro_delta": delta,
            "task_cluster_ci95": ci, "paired_permutation_p": p,
            "episode_mcnemar": mcnemar(rows_by_config[a], rows_by_config[b], primary_tasks, seeds),
        }
    adjusted = holm_adjust(raw_p)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value

    cs_key = "cscka_adjusted_vs_ckaonly"
    cs_cmp = comparisons.get(cs_key)
    cs_enabled = bool(
        cs_cmp and cs_cmp["primary_task_macro_delta"] > 0
        and cs_cmp["task_cluster_ci95"][0] > 0
    )
    return {
        "manifest_sha256": manifest_sha,
        "primary_scope": {"tasks": primary_tasks, "n_tasks": len(primary_tasks)},
        "secondary_scope": {"tasks": tasks, "n_tasks": len(tasks)},
        "bootstrap_draws": n_boot,
        "validation_errors": loaded["validation_errors"],
        "configs": configs,
        "comparisons": comparisons,
        "acceptance": {
            "enable_cs": cs_enabled,
            "default": "cscka_adjusted" if cs_enabled else "ckaonly",
            "rule": "held-out delta > 0 and task-cluster bootstrap CI95 lower bound > 0",
        },
        "libero_context": {
            "v1.4_macro_avg": 0.892,
            "uniform_w6_macro_avg": 0.882,
            "v1.3_macro_avg": 0.852,
            "note": "cross-benchmark context only; not pooled with RoboCasa365",
        },
    }


def write_markdown(path: Path, summary: dict) -> None:
    lines = ["# RoboCasa365 atomic_seen expanded ablation", ""]
    lines += [
        f"Primary held-out scope: {summary['primary_scope']['n_tasks']} tasks.  ",
        f"Acceptance default: **{summary['acceptance']['default']}**.", "",
        "| Config | Held-out macro SR | 95% task-cluster CI | All-18 macro SR | Episodes |",
        "|---|---:|---:|---:|---:|",
    ]
    for cid, row in summary["configs"].items():
        ci = row["primary_task_cluster_ci95"]
        lines.append(
            f"| {cid} | {row['primary_task_macro_sr']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {row['all18_task_macro_sr']:.3f} | "
            f"{row['successes']}/{row['episodes']} |"
        )
    if summary["configs"]:
        config_ids = list(summary["configs"])
        tasks = summary["secondary_scope"]["tasks"]
        lines += ["", "## Per-task success rate", "",
                  "| Task | " + " | ".join(config_ids) + " |",
                  "|---|" + "---:|" * len(config_ids)]
        for task in tasks:
            values = [summary["configs"][cid]["per_task_details"][task]["sr"]
                      for cid in config_ids]
            lines.append("| " + task + " | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "## Prespecified paired comparisons", "",
              "| Contrast | Delta | 95% task-cluster CI | Permutation p | Holm p |",
              "|---|---:|---:|---:|---:|"]
    for name, row in summary["comparisons"].items():
        ci = row["task_cluster_ci95"]
        lines.append(
            f"| {name} | {row['primary_task_macro_delta']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {row['paired_permutation_p']:.4f} | "
            f"{row['holm_adjusted_p']:.4f} |"
        )
    lines += ["", "## LIBERO context", "",
              "v1.4 macro Avg 89.2%; uniform W6 88.2%; v1.3 85.2%. "
              "These values are contextual and are not pooled with RoboCasa365.", ""]
    path.write_text("\n".join(lines))


def selftest() -> None:
    import tempfile

    rates_a = {f"t{i}": v for i, v in enumerate([1.0, 0.8, 0.6, 0.4])}
    rates_b = {f"t{i}": v for i, v in enumerate([0.8, 0.6, 0.6, 0.2])}
    delta, ci, diffs = paired_delta_ci(rates_a, rates_b, 1000, random.Random(0))
    assert abs(delta - 0.15) < 1e-12 and ci[0] <= delta <= ci[1]
    assert 0 <= exact_sign_flip_p(diffs) <= 1
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = {
            "tasks": ["t0", "t1"], "seeds": [0, 1],
            "shards": [["t0"], ["t1"]],
            "protocol": {"action_noise_scheme": "scheme"},
            "configs": [{"id": "a", "config_sha256": "cfg",
                         "result_files": [str(root / "a0.jsonl"), str(root / "a1.jsonl")]}],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        msha = file_sha(manifest_path)
        for shard, task in enumerate(["t0", "t1"]):
            rows = [{"config": "a", "manifest_sha256": msha,
                     "config_sha256": "cfg", "task": task, "seed": seed,
                     "success": True, "steps": 1, "paired_action_noise": True,
                     "action_noise_scheme": "scheme"} for seed in [0, 1]]
            Path(manifest["configs"][0]["result_files"][shard]).write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )
        assert not load_rows(root, manifest, msha, False)["validation_errors"]
        # One duplicate, one bad hash, one crash and one missing row must all
        # be visible to the strict completeness validator.
        bad_path = Path(manifest["configs"][0]["result_files"][0])
        bad_rows = [json.loads(x) for x in bad_path.read_text().splitlines()]
        duplicate = dict(bad_rows[0])
        bad_rows.append(duplicate)
        bad_rows[1]["manifest_sha256"] = "wrong"
        bad_rows[1]["success"] = None
        bad_rows[1]["crashed"] = True
        bad_path.write_text("\n".join(json.dumps(row) for row in bad_rows) + "\n")
        errors = load_rows(root, manifest, msha, True)["validation_errors"]
        assert any("duplicate" in e for e in errors)
        assert any("wrong manifest hash" in e for e in errors)
        assert any("crashed/incomplete" in e for e in errors)
        assert any("missing" in e for e in errors)
    print("[parse-matrix] selftest OK (stats + strict matrix validation)")


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    run_dir = Path(args.run_dir).resolve()
    summary = summarize(run_dir, args.bootstrap, args.allow_incomplete)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(run_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

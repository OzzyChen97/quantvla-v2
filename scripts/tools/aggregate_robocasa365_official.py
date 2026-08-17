#!/usr/bin/env python3
"""Strict 50-task aggregation for the three official RoboCasa365 matrices."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import tempfile
from pathlib import Path
from typing import Any

from parse_robocasa_atomic_matrix import (
    cluster_ci,
    exact_sign_flip_p,
    file_sha,
    holm_adjust,
    load_rows,
    manifest_contrasts,
    mcnemar,
    paired_delta_ci,
    percentile,
)
from robocasa_paper_memory import calculate_manifest as calculate_paper_memory


OFFICIAL_TASK_COUNTS = {
    "atomic_seen": 18,
    "composite_seen": 16,
    "composite_unseen": 16,
}
PROTOCOL_KEYS = (
    "split",
    "official_task_horizons",
    "n_action_steps",
    "denoising_steps",
    "paired_action_noise",
    "action_noise_scheme",
    "action_noise_mode",
    "render",
    "scenarios_per_task",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--out-dir")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_gpu_rows(run_dirs: list[Path], config_ids: set[str]) -> dict[str, list[dict]]:
    grouped = {config: [] for config in config_ids}
    for run_dir in run_dirs:
        path = run_dir / "gpu_efficiency.jsonl"
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: malformed GPU sample: {exc}") from exc
            config = str(row.get("config"))
            if config not in grouped:
                raise ValueError(f"{path}:{line_no}: unknown config {config}")
            grouped[config].append(row)
    return grouped


def summarize_gpu(rows: list[dict]) -> dict[str, Any] | None:
    if not rows:
        return None
    memory = [float(row["memory_used_mib"]) for row in rows]
    utilization = [float(row["utilization_gpu_pct"]) for row in rows]
    power = [float(row["power_draw_w"]) for row in rows]
    return {
        "samples": len(rows),
        "peak_memory_mib": max(memory),
        "mean_memory_mib": mean(memory),
        "mean_gpu_utilization_pct": mean(utilization),
        "p95_gpu_utilization_pct": percentile(utilization, 0.95),
        "mean_power_w": mean(power),
        "p95_power_w": percentile(power, 0.95),
    }


def summarize_efficiency(
    rows: dict[tuple[str, int], dict],
    gpu_rows: list[dict],
    *,
    protocol: dict[str, Any] | None = None,
    rollout_shards: int | None = None,
) -> dict[str, Any]:
    driver_seconds = [float(row["driver_wall_seconds"]) for row in rows.values()
                      if row.get("driver_wall_seconds") is not None]
    episode_seconds = [float(row["episode_wall_seconds"]) for row in rows.values()
                       if row.get("episode_wall_seconds") is not None]
    construct_seconds = [float(row["env_construct_seconds"]) for row in rows.values()
                         if row.get("env_construct_seconds") is not None]
    replans = sum(int(row.get("replans", 0)) for row in rows.values())
    steps = sum(int(row["steps"]) for row in rows.values())
    inference = sum(float(row.get("inference_seconds", 0)) for row in rows.values())
    env_steps = sum(float(row.get("env_step_seconds", 0)) for row in rows.values())
    driver_total = sum(driver_seconds)
    protocol = protocol or {}
    return {
        "rollout_shards": rollout_shards,
        "egl_device_pool": protocol.get("egl_device_pool"),
        "gpu_measurement_scope": protocol.get(
            "gpu_efficiency_scope",
            ("mixed across task-set schedules" if rollout_shards is None
             else "dedicated model-GPU device total"),
        ),
        "mean_driver_wall_seconds": mean(driver_seconds),
        "p50_driver_wall_seconds": percentile(driver_seconds, 0.50),
        "p90_driver_wall_seconds": percentile(driver_seconds, 0.90),
        "mean_episode_wall_seconds": mean(episode_seconds),
        "mean_env_construct_seconds": mean(construct_seconds),
        "mean_inference_seconds_per_replan": inference / replans if replans else None,
        "mean_env_step_seconds": env_steps / steps if steps else None,
        "steps_per_driver_second": steps / driver_total if driver_total else None,
        "episodes_per_aggregate_driver_hour": (
            3600 * len(driver_seconds) / driver_total if driver_total else None
        ),
        "gpu": summarize_gpu(gpu_rows),
    }


def aggregate(
    run_dirs: list[Path], n_boot: int, require_official: bool = True
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one --run-dir is required")
    manifests = []
    matrices = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        manifest_sha = file_sha(manifest_path)
        matrices.append(load_rows(run_dir, manifest, manifest_sha, False)["rows"])
        manifests.append((run_dir, manifest, manifest_sha))

    task_sets = [str(manifest["task_set"]) for _, manifest, _ in manifests]
    if len(task_sets) != len(set(task_sets)):
        raise ValueError(f"duplicate task sets: {task_sets}")
    if require_official and set(task_sets) != set(OFFICIAL_TASK_COUNTS):
        raise ValueError(
            f"expected task sets {sorted(OFFICIAL_TASK_COUNTS)}, got {sorted(task_sets)}"
        )

    all_tasks: list[str] = []
    split_tasks: dict[str, list[str]] = {}
    for _, manifest, _ in manifests:
        task_set = str(manifest["task_set"])
        tasks = list(manifest["tasks"])
        if require_official and len(tasks) != OFFICIAL_TASK_COUNTS[task_set]:
            raise ValueError(
                f"{task_set}: expected {OFFICIAL_TASK_COUNTS[task_set]} tasks, got {len(tasks)}"
            )
        overlap = sorted(set(all_tasks) & set(tasks))
        if overlap:
            raise ValueError(f"tasks occur in multiple task sets: {overlap}")
        all_tasks.extend(tasks)
        split_tasks[task_set] = tasks
    if require_official and len(all_tasks) != 50:
        raise ValueError(f"official aggregate must contain 50 tasks, got {len(all_tasks)}")

    first = manifests[0][1]
    seeds = list(first["seeds"])
    config_ids = [str(config["id"]) for config in first["configs"]]
    protocol = first["protocol"]
    contrasts = manifest_contrasts(first)
    final_decision = {
        key: first.get("decision", {}).get(key)
        for key in ("final_config", "cka_to_cs_ratio")
    }
    for _, manifest, _ in manifests[1:]:
        ids = [str(config["id"]) for config in manifest["configs"]]
        if ids != config_ids:
            raise ValueError(f"config order mismatch: {ids} != {config_ids}")
        if list(manifest["seeds"]) != seeds:
            raise ValueError("seed matrix mismatch across task sets")
        for key in PROTOCOL_KEYS:
            if manifest["protocol"].get(key) != protocol.get(key):
                raise ValueError(f"protocol mismatch on {key}")
        if manifest_contrasts(manifest) != contrasts:
            raise ValueError("comparison list mismatch across task sets")
        for key, value in final_decision.items():
            if manifest.get("decision", {}).get(key) != value:
                raise ValueError(f"final decision mismatch on {key}")

    paper_by_task_set: dict[str, dict[str, Any]] = {}
    manifests_with_checkpoint = [bool(manifest.get("checkpoint")) for _, manifest, _ in manifests]
    if any(manifests_with_checkpoint):
        if not all(manifests_with_checkpoint):
            raise ValueError("checkpoint metadata is missing from one or more matrices")
        for run_dir, manifest, _ in manifests:
            task_set = str(manifest["task_set"])
            metric = calculate_paper_memory(run_dir / "manifest.json")
            if set(metric["configs"]) != set(config_ids):
                raise ValueError(f"{task_set}: paper-memory config set mismatch")
            paper_by_task_set[task_set] = metric
        scope_counts = {metric["scope_linear_layers"] for metric in paper_by_task_set.values()}
        if len(scope_counts) != 1:
            raise ValueError(f"paper-memory scope differs across checkpoints: {scope_counts}")

    rows_by_config: dict[str, dict[tuple[str, int], dict]] = {
        config: {} for config in config_ids
    }
    for matrix in matrices:
        for config in config_ids:
            overlap = set(rows_by_config[config]) & set(matrix[config])
            if overlap:
                raise ValueError(f"{config}: duplicate rows: {sorted(overlap)[:5]}")
            rows_by_config[config].update(matrix[config])
    expected = set(itertools.product(all_tasks, seeds))
    for config, rows in rows_by_config.items():
        if set(rows) != expected:
            missing = sorted(expected - set(rows))
            extra = sorted(set(rows) - expected)
            raise ValueError(
                f"{config}: incomplete aggregate; missing={missing[:5]} extra={extra[:5]}"
            )

    gpu_rows = load_gpu_rows(run_dirs, set(config_ids))
    matrix_by_task_set = {
        str(manifest["task_set"]): matrix
        for matrix, (_, manifest, _) in zip(matrices, manifests)
    }
    manifest_by_task_set = {
        str(manifest["task_set"]): (run_dir, manifest)
        for run_dir, manifest, _ in manifests
    }
    gpu_rows_by_task_set = {
        task_set: load_gpu_rows([run_dir], set(config_ids))
        for task_set, (run_dir, _) in manifest_by_task_set.items()
    }
    rng = random.Random(20260817)
    configs: dict[str, dict[str, Any]] = {}
    for config in config_ids:
        rows = rows_by_config[config]
        per_task = {}
        per_task_details = {}
        for task in all_tasks:
            task_rows = [rows[(task, seed)] for seed in seeds]
            successes = [row for row in task_rows if row["success"]]
            failures = [row for row in task_rows if not row["success"]]
            rate = len(successes) / len(task_rows)
            per_task[task] = rate
            per_task_details[task] = {
                "sr": rate,
                "successes": len(successes),
                "episodes": len(task_rows),
                "mean_success_steps": mean([float(row["steps"]) for row in successes]),
                "mean_failure_steps": mean([float(row["steps"]) for row in failures]),
            }
        per_task_set_efficiency = {}
        for task_set, matrix in matrix_by_task_set.items():
            _, split_manifest = manifest_by_task_set[task_set]
            per_task_set_efficiency[task_set] = summarize_efficiency(
                matrix[config], gpu_rows_by_task_set[task_set][config],
                protocol=split_manifest["protocol"],
                rollout_shards=len(split_manifest["shards"]),
            )
        paper_config = None
        if paper_by_task_set:
            by_task_set = {
                task_set: paper_by_task_set[task_set]["configs"][config]
                for task_set in split_tasks
            }
            weighted_component_bytes = sum(
                by_task_set[task_set]["component_bytes"] * len(tasks)
                for task_set, tasks in split_tasks.items()
            ) / len(all_tasks)
            weighted_fp16_bytes = sum(
                paper_by_task_set[task_set]["fp16_component_bytes"] * len(tasks)
                for task_set, tasks in split_tasks.items()
            ) / len(all_tasks)
            paper_config = {
                "by_task_set": by_task_set,
                "task_weighted_mean_component_bytes": weighted_component_bytes,
                "task_weighted_mean_component_gib": weighted_component_bytes / 2**30,
                "task_weighted_relative_savings": (
                    1.0 - weighted_component_bytes / weighted_fp16_bytes
                ),
                "task_weighted_compression_ratio": (
                    weighted_fp16_bytes / weighted_component_bytes
                ),
            }
        configs[config] = {
            "per_task": per_task,
            "per_task_details": per_task_details,
            "task_macro_sr": mean(list(per_task.values())),
            "task_cluster_ci95": cluster_ci(per_task, n_boot, rng),
            "episode_sr": sum(bool(row["success"]) for row in rows.values()) / len(rows),
            "successes": sum(bool(row["success"]) for row in rows.values()),
            "episodes": len(rows),
            "per_task_set_macro_sr": {
                task_set: mean([per_task[task] for task in tasks])
                for task_set, tasks in split_tasks.items()
            },
            "paper_style_memory": paper_config,
            "efficiency": summarize_efficiency(rows, gpu_rows[config]),
            "per_task_set_efficiency": per_task_set_efficiency,
        }

    comparisons = {}
    raw_p = {}
    for a, b in contrasts:
        delta, ci, diffs = paired_delta_ci(
            configs[a]["per_task"], configs[b]["per_task"], n_boot, rng
        )
        name = f"{a}_vs_{b}"
        p_value = exact_sign_flip_p(diffs)
        raw_p[name] = p_value
        comparisons[name] = {
            "a": a,
            "b": b,
            "task_macro_delta": delta,
            "task_cluster_ci95": ci,
            "paired_permutation_p": p_value,
            "episode_mcnemar": mcnemar(
                rows_by_config[a], rows_by_config[b], all_tasks, seeds
            ),
        }
    for name, adjusted in holm_adjust(raw_p).items():
        comparisons[name]["holm_adjusted_p"] = adjusted

    return {
        "schema_version": 1,
        "evaluation": "paired",
        "task_sets": split_tasks,
        "n_tasks": len(all_tasks),
        "seeds": seeds,
        "scenarios_per_task": len(seeds),
        "episodes_per_config": len(all_tasks) * len(seeds),
        "bootstrap_draws": n_boot,
        "protocol": {key: protocol.get(key) for key in PROTOCOL_KEYS},
        "paper_style_memory": ({
            "scope": next(iter(paper_by_task_set.values()))["scope"],
            "unit_note": next(iter(paper_by_task_set.values()))["unit_note"],
            "estimate_kind": next(iter(paper_by_task_set.values()))["estimate_kind"],
            "by_task_set": paper_by_task_set,
        } if paper_by_task_set else None),
        "final_decision": final_decision,
        "sources": [
            {
                "task_set": manifest["task_set"],
                "run_dir": str(run_dir.resolve()),
                "manifest_sha256": manifest_sha,
            }
            for run_dir, manifest, manifest_sha in manifests
        ],
        "configs": configs,
        "comparisons": comparisons,
        "libero_context": {
            "v1.4_macro_avg": 0.892,
            "uniform_w6_macro_avg": 0.882,
            "v1.3_macro_avg": 0.852,
            "note": "cross-benchmark context only; not pooled with RoboCasa365",
        },
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    config_ids = list(summary["configs"])
    lines = [
        "# Official RoboCasa365 50-task paired evaluation",
        "",
        f"{summary['n_tasks']} tasks × {summary['scenarios_per_task']} scenarios; "
        f"{summary['episodes_per_config']} episodes/config.",
        "",
        "| Config | 50-task macro SR | 95% task-cluster CI | Episode SR |",
        "|---|---:|---:|---:|",
    ]
    for config, row in summary["configs"].items():
        ci = row["task_cluster_ci95"]
        lines.append(
            f"| {config} | {row['task_macro_sr']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{row['successes']}/{row['episodes']} ({row['episode_sr']:.3f}) |"
        )
    lines += [
        "", "## Task-set macro SR", "",
        "| Task set | " + " | ".join(config_ids) + " |",
        "|---|" + "---:|" * len(config_ids),
    ]
    for task_set in summary["task_sets"]:
        values = [
            summary["configs"][config]["per_task_set_macro_sr"][task_set]
            for config in config_ids
        ]
        lines.append(
            "| " + task_set + " | " + " | ".join(f"{value:.3f}" for value in values) + " |"
        )
    lines += [
        "", "## Prespecified paired comparisons", "",
        "| Contrast | Delta | 95% task-cluster CI | Permutation p | Holm p |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in summary["comparisons"].items():
        ci = row["task_cluster_ci95"]
        lines.append(
            f"| {name} | {row['task_macro_delta']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {row['paired_permutation_p']:.4f} | "
            f"{row['holm_adjusted_p']:.4f} |"
        )
    lines += [
        "", "## Per-task SR", "",
        "| Task | " + " | ".join(config_ids) + " |",
        "|---|" + "---:|" * len(config_ids),
    ]
    for tasks in summary["task_sets"].values():
        for task in tasks:
            values = [summary["configs"][config]["per_task"][task] for config in config_ids]
            lines.append(
                "| " + task + " | " + " | ".join(f"{value:.3f}" for value in values) + " |"
            )
    lines += [
        "", "## Efficiency", "",
        "| Config | Episode wall (s) | Inference/replan (s) | Env step (s) | Peak memory (MiB) | Mean util | Mean power (W) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config, row in summary["configs"].items():
        efficiency = row["efficiency"]
        gpu = efficiency.get("gpu") or {}
        lines.append(
            f"| {config} | {efficiency['mean_episode_wall_seconds']:.1f} | "
            f"{efficiency['mean_inference_seconds_per_replan']:.3f} | "
            f"{efficiency['mean_env_step_seconds']:.3f} | "
            f"{gpu.get('peak_memory_mib', float('nan')):.0f} | "
            f"{gpu.get('mean_gpu_utilization_pct', float('nan')):.1f}% | "
            f"{gpu.get('mean_power_w', float('nan')):.1f} |"
        )
    lines += [
        "",
        "Aggregate efficiency above mixes task horizons and execution concurrency; "
        "use the task-set table below for scheduling-aware comparisons.",
        "", "## Efficiency by task set", "",
        "| Task set | Config | Shards | Episode wall (s) | Inference/replan (s) | Env step (s) | Peak model-GPU device memory (MiB) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task_set in summary["task_sets"]:
        for config, row in summary["configs"].items():
            efficiency = row["per_task_set_efficiency"][task_set]
            gpu = efficiency.get("gpu") or {}
            lines.append(
                f"| {task_set} | {config} | {efficiency['rollout_shards']} | "
                f"{efficiency['mean_episode_wall_seconds']:.1f} | "
                f"{efficiency['mean_inference_seconds_per_replan']:.3f} | "
                f"{efficiency['mean_env_step_seconds']:.3f} | "
                f"{gpu.get('peak_memory_mib', float('nan')):.0f} |"
            )
    lines += ["", "Scheduling/GPU sampling notes:", ""]
    for task_set in summary["task_sets"]:
        sample = next(iter(summary["configs"].values()))["per_task_set_efficiency"][task_set]
        lines.append(
            f"- {task_set}: {sample['rollout_shards']} shards/config; "
            f"EGL pool={sample.get('egl_device_pool')}; "
            f"GPU scope={sample['gpu_measurement_scope']}."
        )
    paper_memory = summary.get("paper_style_memory")
    if paper_memory:
        task_sets = list(summary["task_sets"])
        lines += [
            "", "## Paper-style LLM+DiT component memory", "",
            "Theoretical tightly-packed deployment storage (QuantVLA Tables 1/2 scope); "
            "this is distinct from live CUDA memory.", "",
            "| Config | " + " | ".join(f"{task_set} (GiB)" for task_set in task_sets)
            + " | 50-task weighted mean (GiB) | Savings vs FP16 |",
            "|---|" + "---:|" * (len(task_sets) + 2),
        ]
        for config, row in summary["configs"].items():
            metric = row["paper_style_memory"]
            values = [metric["by_task_set"][task_set]["component_gib"] for task_set in task_sets]
            lines.append(
                "| " + config + " | " + " | ".join(f"{value:.3f}" for value in values)
                + f" | {metric['task_weighted_mean_component_gib']:.3f} | "
                + f"{100 * metric['task_weighted_relative_savings']:.1f}% |"
            )
    lines += [
        "",
        "LIBERO context: v1.4 macro Avg 89.2%, uniform W6 88.2%, v1.3 85.2%. "
        "These values are not pooled with RoboCasa365.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_synthetic_matrix(root: Path, task_set: str, task: str) -> Path:
    run_dir = root / task_set
    run_dir.mkdir()
    configs = [
        {
            "id": config,
            "config_sha256": f"hash-{config}-{task_set}",
            "result_files": [str(run_dir / f"{config}.jsonl")],
        }
        for config in ("a", "b")
    ]
    manifest = {
        "task_set": task_set,
        "tasks": [task],
        "seeds": [0, 1],
        "shards": [[task]],
        "comparisons": [["a", "b"]],
        "decision": {"final_config": "a", "cka_to_cs_ratio": 16},
        "protocol": {
            "split": "target",
            "official_task_horizons": True,
            "n_action_steps": 16,
            "denoising_steps": 4,
            "paired_action_noise": True,
            "action_noise_scheme": "scheme",
            "action_noise_mode": "paired",
            "render": True,
            "scenarios_per_task": 2,
        },
        "configs": configs,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = file_sha(manifest_path)
    for config in configs:
        rows = []
        for seed in [0, 1]:
            success = config["id"] == "a" or seed == 0
            rows.append({
                "config": config["id"],
                "manifest_sha256": manifest_sha,
                "config_sha256": config["config_sha256"],
                "task": task,
                "seed": seed,
                "success": success,
                "crashed": False,
                "steps": 10,
                "paired_action_noise": True,
                "action_noise_scheme": "scheme",
                "driver_wall_seconds": 1.0,
                "episode_wall_seconds": 0.8,
                "env_construct_seconds": 0.1,
                "replans": 1,
                "inference_seconds": 0.2,
                "env_step_seconds": 0.5,
            })
        Path(config["result_files"][0]).write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    return run_dir


def selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dirs = [
            write_synthetic_matrix(root, "s0", "t0"),
            write_synthetic_matrix(root, "s1", "t1"),
            write_synthetic_matrix(root, "s2", "t2"),
        ]
        summary = aggregate(run_dirs, 1000, require_official=False)
        assert summary["n_tasks"] == 3
        assert summary["episodes_per_config"] == 6
        assert summary["configs"]["a"]["task_macro_sr"] == 1.0
        assert summary["configs"]["b"]["task_macro_sr"] == 0.5
        assert summary["comparisons"]["a_vs_b"]["task_macro_delta"] == 0.5
        out = root / "summary.md"
        write_markdown(out, summary)
        assert "Official RoboCasa365" in out.read_text()
    print("[aggregate-robocasa365] selftest OK (strict merge + paired stats + markdown)")


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if len(args.run_dir) != 3 or not args.out_dir:
        raise SystemExit("official aggregation requires three --run-dir values and --out-dir")
    run_dirs = [Path(value).resolve() for value in args.run_dir]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(run_dirs, args.bootstrap, require_official=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(out_dir / "summary.md", summary)
    print(json.dumps({
        "out_dir": str(out_dir),
        "n_tasks": summary["n_tasks"],
        "episodes_per_config": summary["episodes_per_config"],
        "configs": {
            config: row["task_macro_sr"] for config, row in summary["configs"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()

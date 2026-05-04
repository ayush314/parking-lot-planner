#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import statistics
import subprocess
from pathlib import Path


def scenario_files(path):
    if path.is_file():
        return [path]
    return sorted(path.glob("*.yaml"))


def load_existing_output(output_path):
    with output_path.open("r", encoding="utf-8") as handle:
        solution = json.load(handle)
    exit_code = 0 if bool(solution.get("success", False)) else 2
    return exit_code, "[reused existing output]", solution, True


def run_one(planner_exe, scenario_path, output_path, force=False):
    if output_path.exists() and not force:
        return load_existing_output(output_path)

    command = [str(planner_exe), str(scenario_path), str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode not in (0, 2):
        raise RuntimeError(
            f"planner crashed on {scenario_path.name}: exit={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    with output_path.open("r", encoding="utf-8") as handle:
        solution = json.load(handle)
    return result.returncode, result.stdout, solution, False


def summarize(rows):
    success_rows = [row for row in rows if row["success"]]
    num_agents = 0 if not rows else int(rows[0]["num_agents"])
    return {
        "num_agents": num_agents,
        "num_scenarios": len(rows),
        "num_success": sum(1 for row in rows if row["success"]),
        "num_failure": sum(1 for row in rows if not row["success"]),
        "success_rate": 0.0 if not rows else sum(1 for row in rows if row["success"]) / len(rows),
        "avg_runtime_sec": 0.0 if not rows else statistics.mean(row["runtime_sec"] for row in rows),
        "avg_expanded_states": 0.0
        if not rows
        else statistics.mean(row["expanded_states"] for row in rows),
        "avg_makespan_success": 0.0
        if not success_rows
        else statistics.mean(row["makespan"] for row in success_rows),
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the planner over a directory of parking scenarios and aggregate metrics."
    )
    parser.add_argument(
        "--planner",
        type=Path,
        default=repo_root / "build" / "parking_lot_planner",
        help="Path to the planner executable",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=repo_root / "batch" / "scenarios" / "instances",
        help="Scenario YAML file or directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "batch" / "results",
        help="Directory to write per-scenario JSON outputs and summaries",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of planner subprocesses to run in parallel",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun scenarios even if an output JSON already exists",
    )
    args = parser.parse_args()

    planner_exe = args.planner.resolve()
    inputs = scenario_files(args.scenarios)
    if not inputs:
        raise SystemExit(f"no scenario YAML files found under {args.scenarios}")

    json_dir = args.output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")

    rows = []

    def solve_scenario(scenario_path):
        output_path = json_dir / f"{scenario_path.stem}.json"
        _, _, solution, _ = run_one(
            planner_exe,
            scenario_path,
            output_path,
            force=args.force,
        )
        metrics = solution.get("metrics", {})
        return {
            "scenario_id": scenario_path.stem,
            "num_agents": int(metrics.get("num_agents", 0)),
            "success": bool(solution.get("success", False)),
            "failure_reason": solution.get("failure_reason", ""),
            "runtime_sec": float(solution.get("runtime_sec", 0.0)),
            "makespan": int(metrics.get("makespan", 0)),
            "expanded_states": int(metrics.get("total_low_level_expanded_states", 0)),
        }

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(solve_scenario, scenario_path): scenario_path
            for scenario_path in inputs
        }
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: row["scenario_id"])

    summary = summarize(rows)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2)

    print(
        f"ran {summary['num_scenarios']} scenarios | "
        f"success_rate={summary['success_rate']:.3f} | "
        f"avg_runtime_sec={summary['avg_runtime_sec']:.3f} | "
        f"avg_expanded_states={summary['avg_expanded_states']:.1f} | "
        f"avg_makespan_success={summary['avg_makespan_success']:.1f}"
    )
    print(f"json={summary_path}")


if __name__ == "__main__":
    main()

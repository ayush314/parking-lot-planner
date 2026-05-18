#!/usr/bin/env python3
"""
compare_planners.py — Run Nayesha's SIPP planner vs. the Hybrid A* baseline
across a sweep of agent counts and collect stats for both.

Usage
-----
  python compare_planners.py [OPTIONS]

Quick start (assumes you already have a build):

  python compare_planners.py \\
    --planner   build/parking_lot_planner \\
    --layout    examples/parking_layout.yaml \\
    --config    examples/parking_benchmark_config.yaml \\
    --output    comparison_results.json

The script will:
  1. Optionally run `cmake --build` for you (--build flag).
  2. Generate one scenario YAML per agent-count using the existing
     `scripts/build_parking_scenario.py` + a sampler that draws random
     start/goal pairs from the full benchmark config.
  3. Run each scenario with `sipp` AND `hybrid` low-level planners.
  4. Write a single JSON file with per-run stats and aggregated summaries
     ready for the companion dashboard (compare_dashboard.html).

Stats collected per run
-----------------------
  - success           (bool)
  - runtime_sec       (wall-clock for the whole multi-agent solve)
  - makespan          (time steps to last agent arrival)
  - total_expanded    (sum of low-level A* / SIPP expansions)
  - sum_plan_distance (sum of Euclidean arc-lengths over all trajectories)
  - sum_plan_time     (sum of final arrival time-steps over all agents)
  - failure_reason    (string, empty on success)
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare SIPP vs Hybrid A* across agent-count sweeps."
    )
    p.add_argument(
        "--planner",
        default="build/parking_lot_planner",
        help="Path to the parking_lot_planner binary (default: build/parking_lot_planner)",
    )
    p.add_argument(
        "--layout",
        default="examples/parking_layout.yaml",
        help="Parking layout YAML (default: examples/parking_layout.yaml)",
    )
    p.add_argument(
        "--config",
        default="examples/parking_benchmark_config.yaml",
        help="Benchmark config YAML with agent pool (default: examples/parking_benchmark_config.yaml)",
    )
    p.add_argument(
        "--output",
        default="comparison_results.json",
        help="Output JSON file (default: comparison_results.json)",
    )
    p.add_argument(
        "--agent-counts",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Agent counts to sweep. Defaults to 1..N where N is the full pool size. "
            "Counts exceeding the pool are silently skipped."
        ),
    )
    p.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Trials per (planner, agent-count) combination (default: 3)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for agent sampling (default: 42)",
    )
    p.add_argument(
        "--build",
        action="store_true",
        help="Run `cmake --build build` before sweeping",
    )
    p.add_argument(
        "--build-dir",
        default="build",
        help="CMake build directory used when --build is set (default: build)",
    )
    p.add_argument(
        "--scenario-script",
        default="scripts/build_parking_scenario.py",
        help="Path to build_parking_scenario.py (default: scripts/build_parking_scenario.py)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-run timeout in seconds (default: 120)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print each run result as it completes",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# YAML helpers (stdlib only — no PyYAML required at runtime)
# ---------------------------------------------------------------------------

def _load_yaml_naive(path: str) -> dict:
    """
    Minimal YAML loader that handles the simple key-value and sequence
    structure used in parking benchmark configs.  We avoid a hard PyYAML
    dependency so the script works out-of-the-box; fall back to PyYAML
    if available for robustness.
    """
    try:
        import yaml  # type: ignore
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        pass

    # Very small hand-rolled parser for the specific YAML shape we need.
    with open(path) as f:
        lines = f.readlines()

    root: dict = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    list_key: str | None = None

    for raw in lines:
        stripped = raw.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        content = stripped.strip()

        # Pop stack to current indent level
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
            list_key = None

        parent = stack[-1][1]

        if content.startswith("- "):
            # Sequence item (flat)
            val = content[2:].strip()
            if isinstance(parent, list):
                parent.append(val)
            elif list_key and isinstance(root.get(list_key), list):
                root[list_key].append(val)
        elif ":" in content:
            key, _, val = content.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                # Mapping or sequence ahead
                child: Any = {}
                if isinstance(parent, dict):
                    parent[key] = child
                stack.append((indent, child))
                list_key = key
            else:
                if isinstance(parent, dict):
                    # Try numeric conversion
                    try:
                        parent[key] = int(val)
                    except ValueError:
                        try:
                            parent[key] = float(val)
                        except ValueError:
                            parent[key] = val.strip('"').strip("'")

    return root


def _dump_yaml_agents(agents: list[dict]) -> str:
    """Serialize the agents list to YAML text."""
    lines = ["agents:"]
    for ag in agents:
        s = ag["start"]
        g = ag["goal"]
        lines.append(f'  - id: "{ag["id"]}"')
        lines.append(f'    start: [{s[0]}, {s[1]}, {s[2]}]')
        lines.append(f'    goal: [{g[0]}, {g[1]}, {g[2]}]')
    return "\n".join(lines) + "\n"


def _parse_pose(raw) -> list[float]:
    """Convert '[x, y, yaw]' string or list to [float, float, float]."""
    if isinstance(raw, list):
        return [float(v) for v in raw]
    # String form: '[1.5, 2.0, 0.0]'
    inner = raw.strip().strip("[]")
    return [float(v) for v in inner.split(",")]


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def build_scenario(
    scenario_script: str,
    layout_path: str,
    config_path: str,
    output_path: str,
) -> bool:
    """Call build_parking_scenario.py to generate a flat runtime YAML."""
    cmd = [
        sys.executable,
        scenario_script,
        "--layout", layout_path,
        "--config", config_path,
        "--output", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def _find_pose_key(agent_dict: dict, candidates: list[str]) -> str | None:
    """Return the first key from candidates that exists in agent_dict."""
    for k in candidates:
        if k in agent_dict:
            return k
    return None


def load_agent_pool_from_instance(instance_path: str) -> list[dict]:
    """
    Load the agent pool from a flat instance YAML generated by
    build_parking_scenario.py.  This file is guaranteed by io.cpp to use
    'start' and 'goal' as keys, so we never have to guess.
    """
    data = _load_yaml_naive(instance_path)
    pool = data.get("agents", [])
    if not pool:
        raise ValueError(
            f"Flat instance YAML at {instance_path!r} has no 'agents' section."
        )
    result = []
    for i, ag in enumerate(pool):
        # io.cpp always writes 'start' and 'goal'
        start_key = _find_pose_key(ag, ["start", "start_pose", "from"])
        goal_key = _find_pose_key(ag, ["goal", "goal_pose", "to"])
        if start_key is None or goal_key is None:
            found_keys = list(ag.keys())
            raise ValueError(
                f"Agent {i} in {instance_path!r} has unrecognised keys: {found_keys}. "
                "Expected 'start' and 'goal'. The flat instance may not have been "
                "generated correctly by build_parking_scenario.py."
            )
        agent_id = ag.get("id", f"agent_{i}")
        result.append({
            "id": str(agent_id),
            "start": _parse_pose(ag[start_key]),
            "goal": _parse_pose(ag[goal_key]),
        })
    return result


def sample_agents(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Draw n agents without replacement from a pre-loaded agent pool."""
    if n > len(pool):
        raise ValueError(
            f"Requested {n} agents but pool only has {len(pool)}. "
            "Lower --agent-counts or add more agents to the config."
        )
    return rng.sample(pool, n)


def write_instance_yaml(
    base_instance_path: str,
    agents: list[dict],
    output_path: str,
) -> None:
    """
    Read the base flat instance YAML (produced by build_parking_scenario.py),
    strip its existing 'agents:' block, and replace it with the sampled agents.
    """
    with open(base_instance_path) as f:
        base_text = f.read()

    # Remove existing agents block (from 'agents:' to end of file or next top-level key)
    import re
    cleaned = re.sub(r"\nagents:.*", "", base_text, flags=re.DOTALL)
    cleaned = cleaned.rstrip() + "\n\n"
    cleaned += _dump_yaml_agents(agents)

    with open(output_path, "w") as f:
        f.write(cleaned)


# ---------------------------------------------------------------------------
# Running the planner
# ---------------------------------------------------------------------------

def run_planner(
    binary: str,
    instance_path: str,
    output_path: str,
    low_level: str,
    timeout: float,
) -> dict:
    """
    Run the planner binary and return a dict of parsed stats.
    Returns a failure dict on timeout or non-zero exit.
    """
    cmd = [binary, instance_path, output_path, low_level]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "failure_reason": f"timeout after {timeout}s",
            "runtime_sec": timeout,
            "makespan": 0,
            "total_expanded": 0,
            "sum_plan_distance": 0.0,
            "sum_plan_time": 0,
        }
    wall_time = time.perf_counter() - t0

    if not os.path.exists(output_path):
        return {
            "success": False,
            "failure_reason": f"binary exited {proc.returncode}, no output written",
            "runtime_sec": wall_time,
            "makespan": 0,
            "total_expanded": 0,
            "sum_plan_distance": 0.0,
            "sum_plan_time": 0,
        }

    with open(output_path) as f:
        sol = json.load(f)

    metrics = sol.get("metrics", {})
    trajectories = sol.get("trajectories", {})

    # Compute sum of Euclidean arc-lengths across all agent trajectories
    sum_dist = 0.0
    sum_time = 0
    for _agent_id, traj in trajectories.items():
        if not traj:
            continue
        for i in range(1, len(traj)):
            dx = traj[i]["x"] - traj[i - 1]["x"]
            dy = traj[i]["y"] - traj[i - 1]["y"]
            sum_dist += math.sqrt(dx * dx + dy * dy)
        sum_time += traj[-1]["t"]  # final arrival time for this agent

    return {
        "success": sol.get("success", False),
        "failure_reason": sol.get("failure_reason", ""),
        # runtime_sec from the JSON is the planner's internal measurement;
        # we also expose wall_time for reference
        "runtime_sec": sol.get("runtime_sec", wall_time),
        "wall_time_sec": wall_time,
        "makespan": metrics.get("makespan", 0),
        "total_expanded": metrics.get("total_low_level_expanded_states", 0),
        "sum_plan_distance": round(sum_dist, 4),
        "sum_plan_time": sum_time,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(runs: list[dict]) -> dict:
    """Compute mean ± std over a list of run result dicts."""
    if not runs:
        return {}

    numeric_keys = [
        "runtime_sec", "wall_time_sec", "makespan",
        "total_expanded", "sum_plan_distance", "sum_plan_time",
    ]
    agg: dict = {
        "n_trials": len(runs),
        "success_rate": sum(1 for r in runs if r["success"]) / len(runs),
    }
    for k in numeric_keys:
        vals = [r[k] for r in runs if k in r]
        if vals:
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
            agg[f"{k}_mean"] = round(mean, 5)
            agg[f"{k}_std"] = round(math.sqrt(variance), 5)
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # -- Validate paths -------------------------------------------------------
    for label, path in [
        ("--planner", args.planner),
        ("--layout", args.layout),
        ("--config", args.config),
        ("--scenario-script", args.scenario_script),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {label} path not found: {path}", file=sys.stderr)
            print(
                "\nHint: run from the repo root, e.g.:\n"
                "  python scripts/compare_planners.py --build\n"
                "or pass explicit paths with --planner, --layout, --config.",
                file=sys.stderr,
            )
            return 1

    # -- Optional build -------------------------------------------------------
    if args.build:
        print(f"Building in {args.build_dir} ...")
        result = subprocess.run(
            ["cmake", "--build", args.build_dir],
            capture_output=False,
        )
        if result.returncode != 0:
            print("Build failed.", file=sys.stderr)
            return 1

    planners = ["sipp", "hybrid"]
    rng = random.Random(args.seed)

    all_runs: list[dict] = []
    summary: dict[str, dict[str, Any]] = {}  # key: "sipp" / "hybrid"
    for pl in planners:
        summary[pl] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate the base flat instance first — we read the agent pool
        # from it so we always use the schema the planner actually expects
        # (start/goal), regardless of what keys the raw config uses.
        base_instance = os.path.join(tmpdir, "base_instance.yaml")
        print("Generating base scenario YAML ...")
        ok = build_scenario(
            args.scenario_script,
            args.layout,
            args.config,
            base_instance,
        )
        if not ok:
            print(
                "ERROR: build_parking_scenario.py failed. "
                "Make sure --layout and --config are correct.",
                file=sys.stderr,
            )
            return 1

        # Load agent pool from the flat instance (guaranteed start/goal keys)
        agent_pool = load_agent_pool_from_instance(base_instance)
        agent_pool_size = len(agent_pool)
        print(f"Agent pool size (from compiled instance): {agent_pool_size}")

        requested_counts = args.agent_counts if args.agent_counts else list(range(1, agent_pool_size + 1))
        valid_counts = [n for n in requested_counts if n <= agent_pool_size]
        skipped = [n for n in requested_counts if n > agent_pool_size]
        if skipped:
            print(
                f"WARNING: skipping agent counts {skipped} "
                f"(pool only has {agent_pool_size} agents)"
            )
        if not valid_counts:
            print("ERROR: no valid agent counts to sweep.", file=sys.stderr)
            return 1

        total_runs = len(valid_counts) * args.trials * len(planners)
        run_num = 0

        for n_agents in sorted(valid_counts):
            print(f"\n── {n_agents} agent(s) ──────────────────────────────")
            runs_by_planner: dict[str, list[dict]] = {pl: [] for pl in planners}

            for trial in range(args.trials):
                # Sample agents once per trial (shared across planners for fairness)
                agents = sample_agents(agent_pool, n_agents, rng)
                instance_path = os.path.join(tmpdir, f"instance_{n_agents}_{trial}.yaml")
                write_instance_yaml(base_instance, agents, instance_path)

                for pl in planners:
                    run_num += 1
                    output_path = os.path.join(
                        tmpdir, f"output_{n_agents}_{trial}_{pl}.json"
                    )
                    result = run_planner(
                        args.planner,
                        instance_path,
                        output_path,
                        pl,
                        args.timeout,
                    )
                    result["planner"] = pl
                    result["n_agents"] = n_agents
                    result["trial"] = trial
                    all_runs.append(result)
                    runs_by_planner[pl].append(result)

                    status = "✓" if result["success"] else "✗"
                    if args.verbose:
                        print(
                            f"  [{run_num:3d}/{total_runs}] "
                            f"{pl:6s} | agents={n_agents} trial={trial} | "
                            f"{status} rt={result['runtime_sec']:.3f}s "
                            f"expanded={result['total_expanded']} "
                            f"dist={result['sum_plan_distance']:.2f}"
                        )
                    else:
                        print(
                            f"  {pl:6s} trial {trial}: {status} "
                            f"rt={result['runtime_sec']:.2f}s",
                            flush=True,
                        )

            for pl in planners:
                summary[pl][str(n_agents)] = aggregate(runs_by_planner[pl])

    # -- Write output ---------------------------------------------------------
    output = {
        "meta": {
            "agent_counts": sorted(valid_counts),
            "trials_per_config": args.trials,
            "seed": args.seed,
            "planners": planners,
            "planner_binary": args.planner,
        },
        "summary": summary,
        "runs": all_runs,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to: {out_path}")

    # -- Quick console summary ------------------------------------------------
    print("\n── Summary ─────────────────────────────────────────────────────────")
    print(f"{'agents':>6}  {'planner':>7}  {'success%':>9}  {'rt_mean':>9}  {'expanded_mean':>14}  {'dist_mean':>10}")
    for n in sorted(valid_counts):
        for pl in planners:
            agg = summary[pl].get(str(n), {})
            print(
                f"{n:>6}  {pl:>7}  "
                f"{agg.get('success_rate', 0)*100:>8.0f}%  "
                f"{agg.get('runtime_sec_mean', 0):>9.3f}  "
                f"{agg.get('total_expanded_mean', 0):>14.0f}  "
                f"{agg.get('sum_plan_distance_mean', 0):>10.2f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
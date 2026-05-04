#!/usr/bin/env python3

import argparse
import json
import math

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def yaw_difference(lhs, rhs):
    delta = lhs - rhs
    while delta >= math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return abs(delta)


def expect(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    parser = argparse.ArgumentParser(description="Validate smoke-test planner output.")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--solution", required=True)
    args = parser.parse_args()

    instance = load_yaml(args.instance)
    solution = load_json(args.solution)

    expect(solution.get("success") is True, "expected smoke solution success=true")
    expect("failure_reason" not in solution, "did not expect failure_reason on success")

    agents = instance["agents"]
    trajectories = solution["trajectories"]
    metrics = solution["metrics"]

    goal_position_tolerance = instance["planner"]["goal_position_tolerance"]
    goal_yaw_tolerance = instance["planner"]["goal_yaw_tolerance"]

    expected_ids = [str(agent["id"]) for agent in agents]
    actual_ids = sorted(trajectories.keys())

    expect(
        metrics["num_agents"] == len(agents),
        f"expected metrics.num_agents={len(agents)}",
    )
    expect(
        actual_ids == sorted(expected_ids),
        f"trajectory ids {actual_ids} did not match agent ids {sorted(expected_ids)}",
    )
    expect(
        metrics["total_low_level_expanded_states"] > 0,
        "expected positive expanded-state count",
    )

    makespan = 0
    for agent in agents:
        agent_id = str(agent["id"])
        trajectory = trajectories[agent_id]
        expect(trajectory, f"trajectory for agent {agent_id} was empty")

        last_t = -1
        for point in trajectory:
            expect(point["t"] >= last_t, f"trajectory for agent {agent_id} was not time-sorted")
            last_t = point["t"]

        final_state = trajectory[-1]
        goal = agent["goal"]
        dx = final_state["x"] - goal[0]
        dy = final_state["y"] - goal[1]
        position_error = math.hypot(dx, dy)
        yaw_error = yaw_difference(final_state["yaw"], goal[2])

        expect(
            position_error <= goal_position_tolerance,
            (
                f"agent {agent_id} final position error {position_error:.3f} "
                f"exceeded tolerance {goal_position_tolerance:.3f}"
            ),
        )
        expect(
            yaw_error <= goal_yaw_tolerance,
            (
                f"agent {agent_id} final yaw error {yaw_error:.3f} "
                f"exceeded tolerance {goal_yaw_tolerance:.3f}"
            ),
        )
        makespan = max(makespan, final_state["t"])

    expect(metrics["makespan"] == makespan, "metrics.makespan did not match trajectory makespan")


if __name__ == "__main__":
    main()

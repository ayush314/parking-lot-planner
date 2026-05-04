#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import yaml

import build_parking_scenario as benchmark


def two_lane_centers(low, high):
    span = high - low
    return [
        round(low + 0.25 * span, 3),
        round(low + 0.75 * span, 3),
    ]


def gap_intervals(module_bands, total_extent):
    gaps = []
    previous_top = 0.0
    for band_bottom, band_top in module_bands:
        if band_bottom - previous_top > benchmark.POSE_EPSILON:
            gaps.append((previous_top, band_bottom))
        previous_top = band_top
    if total_extent - previous_top > benchmark.POSE_EPSILON:
        gaps.append((previous_top, total_extent))
    return gaps


def overlaps_any_interval(center, half_extent, intervals):
    lower = center - half_extent
    upper = center + half_extent
    for interval_low, interval_high in intervals:
        if lower < interval_high - benchmark.POSE_EPSILON and upper > interval_low + benchmark.POSE_EPSILON:
            return True
    return False


def lane_start_candidates(layout):
    vehicle = benchmark.VEHICLE
    spacing = max(vehicle["length"] + 0.6, 4.8)
    margin = 0.5 * vehicle["length"] + 0.4
    longitudinal_half_extent = 0.5 * vehicle["length"] + 0.2

    xs = []
    x = margin
    while x <= layout["map_width"] - margin + benchmark.POSE_EPSILON:
        xs.append(round(x, 3))
        x += spacing

    ys = []
    y = margin
    while y <= layout["map_height"] - margin + benchmark.POSE_EPSILON:
        ys.append(round(y, 3))
        y += spacing

    module_bands_y = []
    for _, _, _, center_y in benchmark.zone_origins(layout):
        module_bands_y.append(
            (
                center_y - layout["stall_depth"],
                center_y + layout["stall_depth"],
            )
        )
    module_bands_y = sorted(set((round(lo, 6), round(hi, 6)) for lo, hi in module_bands_y))
    horizontal_gap_intervals = gap_intervals(module_bands_y, layout["map_height"])

    horizontal_rows = []
    for gap_low, gap_high in horizontal_gap_intervals:
        horizontal_rows.append(two_lane_centers(gap_low, gap_high))

    module_bands_x = []
    for _, _, x_start, _ in benchmark.zone_origins(layout):
        module_bands_x.append((x_start, x_start + layout["row_width"]))
    module_bands_x = sorted(set((round(lo, 6), round(hi, 6)) for lo, hi in module_bands_x))
    vertical_gap_intervals = gap_intervals(module_bands_x, layout["map_width"])

    filtered_horizontal_xs = [
        x_value
        for x_value in xs
        if not overlaps_any_interval(x_value, longitudinal_half_extent, vertical_gap_intervals)
    ]

    candidate_poses = []
    for lower_lane_y, upper_lane_y in horizontal_rows:
        for x_value in filtered_horizontal_xs:
            candidate_poses.append([x_value, lower_lane_y, 0.0])
            candidate_poses.append([x_value, upper_lane_y, 3.14159])

    vertical_columns = []
    for gap_low, gap_high in vertical_gap_intervals:
        vertical_columns.append(two_lane_centers(gap_low, gap_high))

    filtered_vertical_ys = [
        y_value
        for y_value in ys
        if not overlaps_any_interval(y_value, longitudinal_half_extent, horizontal_gap_intervals)
    ]

    for left_lane_x, right_lane_x in vertical_columns:
        for y_value in filtered_vertical_ys:
            candidate_poses.append([left_lane_x, y_value, 1.5708])
            candidate_poses.append([right_lane_x, y_value, -1.5708])

    unique = []
    seen = set()
    for pose in candidate_poses:
        key = tuple(pose)
        if key in seen:
            continue
        seen.add(key)
        if benchmark.pose_inside_map(pose, layout):
            unique.append(pose)
    return unique


def random_start_poses(layout, blocked_poses, num_agents, rng):
    candidates = lane_start_candidates(layout)
    rng.shuffle(candidates)
    starts = []
    occupied = [{"pose": pose} for pose in blocked_poses]

    for pose in candidates:
        if any(
            benchmark.rectangles_overlap(
                pose,
                other["pose"],
                benchmark.VEHICLE["length"],
                benchmark.VEHICLE["width"],
            )
            for other in occupied
        ):
            continue
        starts.append(pose)
        occupied.append({"pose": pose})
        if len(starts) == num_agents:
            return starts

    raise ValueError(
        f"could not place {num_agents} non-overlapping agent starts in the drive aisles"
    )


def build_random_config(layout, num_agents, num_static_cars, rng):
    stalls = benchmark.all_stalls(layout)
    max_static = len(stalls) - num_agents
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    if num_static_cars < 0 or num_static_cars > max_static:
        raise ValueError(
            f"num_static_cars must be between 0 and {max_static} for the chosen number of agents"
        )

    occupied = rng.sample(stalls, num_static_cars)
    occupied_keys = {benchmark.stall_key(stall) for stall in occupied}
    free_stalls = [stall for stall in stalls if benchmark.stall_key(stall) not in occupied_keys]
    goal_stalls = rng.sample(free_stalls, num_agents)
    blocked_poses = [stall["pose"] for stall in occupied] + [stall["pose"] for stall in goal_stalls]
    starts = random_start_poses(layout, blocked_poses, num_agents, rng)

    return {
        "static_cars": [{"stall": [stall["zone"], stall["side"], stall["stall_index"]]} for stall in occupied],
        "agents": [
            {
                "id": str(agent_index),
                "start": starts[agent_index],
                "goal_stall": [
                    goal_stalls[agent_index]["zone"],
                    goal_stalls[agent_index]["side"],
                    goal_stalls[agent_index]["stall_index"],
                ],
            }
            for agent_index in range(num_agents)
        ],
    }


def resolve_num_static_cars(layout, num_agents, num_static_cars, static_fill_ratio):
    stalls = benchmark.all_stalls(layout)
    max_static = len(stalls) - num_agents
    if max_static < 0:
        raise ValueError(
            f"num_agents={num_agents} exceeds available stalls={len(stalls)}"
        )
    if static_fill_ratio is None:
        return num_static_cars

    if not 0.0 <= static_fill_ratio <= 1.0:
        raise ValueError("static_fill_ratio must be between 0.0 and 1.0")
    return int(max_static * static_fill_ratio)


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a batch of random parking-field scenarios on the shared benchmark layout."
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=repo_root / "examples" / "parking_layout.yaml",
        help="Reusable parking-field layout YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "batch" / "scenarios",
        help="Directory to write configs and runtime scenarios into",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=20,
        help="Number of scenarios to generate",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=3,
        help="Number of active agents per scenario",
    )
    parser.add_argument(
        "--num-static-cars",
        type=int,
        default=8,
        help="Number of parked obstacle cars per scenario",
    )
    parser.add_argument(
        "--static-fill-ratio",
        type=float,
        default=None,
        help="If set, use this fraction of the remaining stalls as parked obstacle cars",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed",
    )
    args = parser.parse_args()

    if args.num_scenarios <= 0:
        raise SystemExit("num_scenarios must be positive")

    layout = benchmark.layout_dict(benchmark.load_layout_config(args.layout))
    num_static_cars = resolve_num_static_cars(
        layout,
        args.num_agents,
        args.num_static_cars,
        args.static_fill_ratio,
    )
    configs_dir = args.output_dir / "configs"
    instances_dir = args.output_dir / "instances"

    for scenario_index in range(args.num_scenarios):
        scenario_seed = args.seed + scenario_index
        rng = random.Random(scenario_seed)
        config = build_random_config(
            layout,
            args.num_agents,
            num_static_cars,
            rng,
        )
        instance = benchmark.build_instance_from_config(layout, config)
        stem = f"scenario_{scenario_index:03d}"
        config_path = configs_dir / f"{stem}_config.yaml"
        instance_path = instances_dir / f"{stem}.yaml"
        write_yaml(config_path, config)
        write_yaml(instance_path, instance)
    print(f"generated {args.num_scenarios} scenarios in {args.output_dir}")


if __name__ == "__main__":
    main()

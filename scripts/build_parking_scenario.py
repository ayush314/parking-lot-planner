#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import yaml


STALL_WIDTH = 3.35
STALL_DEPTH = 6.3
ZONE_GAP = 8.375
PERIMETER_GAP = 8.375
GRID_COLS = 2
GRID_ROWS = 2
STALLS_PER_SIDE = 6
POSE_EPSILON = 1e-6

VEHICLE = {
    "length": 4.77,
    "width": 1.86,
    "min_turning_radius": 5.5,
    "step_size": 0.5,
    "heading_bins": 72,
    "max_time_steps": 260,
}

PLANNER = {
    "goal_position_tolerance": 0.25,
    "goal_yaw_tolerance": 0.12,
    "max_expansions": 750000,
}


def load_layout_config(layout_path):
    raw = {}
    if layout_path is not None:
        with layout_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    return {
        "grid_cols": raw.get("grid_cols", GRID_COLS),
        "grid_rows": raw.get("grid_rows", GRID_ROWS),
        "stalls_per_side": raw.get("stalls_per_side", STALLS_PER_SIDE),
        "stall_width": raw.get("stall_width", STALL_WIDTH),
        "stall_depth": raw.get("stall_depth", STALL_DEPTH),
        "zone_gap": raw.get("zone_gap", ZONE_GAP),
        "perimeter_gap": raw.get("perimeter_gap", PERIMETER_GAP),
    }


def layout_dict(layout_config):
    row_width = layout_config["stalls_per_side"] * layout_config["stall_width"]
    module_height = 2.0 * layout_config["stall_depth"]
    parking_width = (
        2.0 * layout_config["perimeter_gap"]
        + layout_config["grid_cols"] * row_width
        + (layout_config["grid_cols"] - 1) * layout_config["zone_gap"]
    )
    parking_height = (
        2.0 * layout_config["perimeter_gap"]
        + layout_config["grid_rows"] * module_height
        + (layout_config["grid_rows"] - 1) * layout_config["zone_gap"]
    )
    return {
        **layout_config,
        "row_width": row_width,
        "module_height": module_height,
        "map_width": parking_width,
        "map_height": parking_height,
    }


def zone_origins(layout):
    origins = []
    for row_index in range(layout["grid_rows"]):
        for col_index in range(layout["grid_cols"]):
            x_start = layout["perimeter_gap"] + col_index * (
                layout["row_width"] + layout["zone_gap"]
            )
            center_y = (
                layout["perimeter_gap"]
                + layout["stall_depth"]
                + row_index * (layout["module_height"] + layout["zone_gap"])
            )
            origins.append((col_index, row_index, x_start, center_y))
    return origins


def zone_name(col_index, row_index):
    vertical = "top" if row_index == 1 else "bottom"
    horizontal = "left" if col_index == 0 else "right"
    return f"{vertical}_{horizontal}"


def pose_for_stall(layout, x_start, center_y, side, stall_index):
    x = x_start + stall_index * layout["stall_width"] + layout["stall_width"] / 2.0
    if side == "bottom":
        return [round(x, 3), round(center_y - layout["stall_depth"] / 2.0, 3), 1.5708]
    return [round(x, 3), round(center_y + layout["stall_depth"] / 2.0, 3), -1.5708]


def all_stalls(layout):
    stalls = []
    for col_index, row_index, x_start, center_y in zone_origins(layout):
        zone = zone_name(col_index, row_index)
        for side in ("bottom", "top"):
            for stall_index in range(layout["stalls_per_side"]):
                stalls.append(
                    {
                        "zone": zone,
                        "side": side,
                        "stall_index": stall_index,
                        "pose": pose_for_stall(layout, x_start, center_y, side, stall_index),
                    }
                )
    return stalls


def stall_key(stall):
    return (stall["zone"], stall["side"], stall["stall_index"])


def stall_map(layout):
    return {stall_key(stall): stall for stall in all_stalls(layout)}


def parse_stall_ref(stall_ref, field_name):
    if not isinstance(stall_ref, list) or len(stall_ref) != 3:
        raise ValueError(f"{field_name} must be [zone, side, stall_index]")
    return (str(stall_ref[0]), str(stall_ref[1]), int(stall_ref[2]))


def parse_pose(node, field_name):
    if not isinstance(node, list) or len(node) != 3:
        raise ValueError(f"{field_name} must be [x, y, yaw]")
    return [float(node[0]), float(node[1]), float(node[2])]


def build_instance_dict(layout):
    parking_layout = {
        "grid_cols": layout["grid_cols"],
        "grid_rows": layout["grid_rows"],
        "stalls_per_side": layout["stalls_per_side"],
        "stall_width": layout["stall_width"],
        "stall_depth": layout["stall_depth"],
        "zone_gap": layout["zone_gap"],
        "perimeter_gap": layout["perimeter_gap"],
    }
    return {
        "map": {
            "width": round(layout["map_width"], 3),
            "height": round(layout["map_height"], 3),
            "resolution": 0.5,
        },
        "vehicle": VEHICLE,
        "planner": PLANNER,
        "parking_layout": parking_layout,
        "static_cars": [],
        "agents": [],
    }


def rectangle_corners(pose, length, width):
    x, y, yaw = pose
    half_length = 0.5 * length
    half_width = 0.5 * width
    local_corners = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return [
        (
            x + local_x * cos_yaw - local_y * sin_yaw,
            y + local_x * sin_yaw + local_y * cos_yaw,
        )
        for local_x, local_y in local_corners
    ]


def projection_interval(points, axis):
    projections = [point[0] * axis[0] + point[1] * axis[1] for point in points]
    return min(projections), max(projections)


def rectangle_axes(pose):
    yaw = pose[2]
    return [
        (math.cos(yaw), math.sin(yaw)),
        (-math.sin(yaw), math.cos(yaw)),
    ]


def rectangles_overlap(pose_a, pose_b, length, width):
    corners_a = rectangle_corners(pose_a, length, width)
    corners_b = rectangle_corners(pose_b, length, width)
    for axis in rectangle_axes(pose_a) + rectangle_axes(pose_b):
        min_a, max_a = projection_interval(corners_a, axis)
        min_b, max_b = projection_interval(corners_b, axis)
        if max_a <= min_b + POSE_EPSILON or max_b <= min_a + POSE_EPSILON:
            return False
    return True


def pose_inside_map(pose, layout):
    for x, y in rectangle_corners(pose, VEHICLE["length"], VEHICLE["width"]):
        if x < -POSE_EPSILON or x > layout["map_width"] + POSE_EPSILON:
            return False
        if y < -POSE_EPSILON or y > layout["map_height"] + POSE_EPSILON:
            return False
    return True


def validate_non_overlapping(label, items):
    for index, lhs in enumerate(items):
        for rhs in items[index + 1 :]:
            if rectangles_overlap(lhs["pose"], rhs["pose"], VEHICLE["length"], VEHICLE["width"]):
                raise ValueError(f"{label} overlap: {lhs['name']} vs {rhs['name']}")


def validate_configuration(layout, instance):
    static_items = [
        {"name": f"static_cars[{index}]", "pose": parked["pose"]}
        for index, parked in enumerate(instance["static_cars"])
    ]
    start_items = [
        {"name": f"agents[{index}].start", "pose": agent["start"]}
        for index, agent in enumerate(instance["agents"])
    ]
    goal_items = [
        {"name": f"agents[{index}].goal", "pose": agent["goal"]}
        for index, agent in enumerate(instance["agents"])
    ]

    for item in static_items + start_items + goal_items:
        if not pose_inside_map(item["pose"], layout):
            raise ValueError(f"{item['name']} lies outside the map bounds")

    validate_non_overlapping("static cars", static_items)
    validate_non_overlapping("agent starts", start_items)
    validate_non_overlapping("agent goals", goal_items)
    validate_non_overlapping("static cars and starts", static_items + start_items)
    validate_non_overlapping("static cars and goals", static_items + goal_items)


def resolve_static_car_pose(layout, stalls, spec, index):
    has_stall = "stall" in spec
    has_pose = "pose" in spec
    if has_stall == has_pose:
        raise ValueError(
            f"static_cars[{index}] must provide exactly one of 'stall' or 'pose'"
        )
    if has_stall:
        key = parse_stall_ref(spec["stall"], f"static_cars[{index}].stall")
        if key not in stalls:
            raise ValueError(f"unknown static stall reference: {spec['stall']}")
        return stalls[key]["pose"]
    return parse_pose(spec["pose"], f"static_cars[{index}].pose")


def resolve_goal_pose(stalls, agent, index):
    has_goal_stall = "goal_stall" in agent
    has_goal_pose = "goal" in agent
    if has_goal_stall == has_goal_pose:
        raise ValueError(
            f"agents[{index}] must provide exactly one of 'goal_stall' or 'goal'"
        )
    if has_goal_stall:
        key = parse_stall_ref(agent["goal_stall"], f"agents[{index}].goal_stall")
        if key not in stalls:
            raise ValueError(f"unknown goal stall reference: {agent['goal_stall']}")
        return stalls[key]["pose"]
    return parse_pose(agent["goal"], f"agents[{index}].goal")


def build_instance_from_config(layout, config):
    stalls = stall_map(layout)
    instance = build_instance_dict(layout)

    static_specs = config.get("static_cars", [])
    for index, spec in enumerate(static_specs):
        pose = resolve_static_car_pose(layout, stalls, spec, index)
        instance["static_cars"].append(
            {
                "id": str(spec.get("id", f"s{index}")),
                "pose": pose,
            }
        )

    agents = config.get("agents", [])
    if not agents:
        raise ValueError("config must contain a non-empty agents list")

    agent_ids = set()
    for index, agent in enumerate(agents):
        agent_id = str(agent.get("id", index))
        if agent_id in agent_ids:
            raise ValueError(f"duplicate agent id: {agent_id}")
        agent_ids.add(agent_id)

        start_pose = parse_pose(agent.get("start"), f"agents[{index}].start")
        goal_pose = resolve_goal_pose(stalls, agent, index)
        instance["agents"].append(
            {
                "id": agent_id,
                "start": start_pose,
                "goal": goal_pose,
            }
        )

    validate_configuration(layout, instance)
    return instance


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compile a fixed benchmark parking-field layout plus explicit car poses into planner YAML."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "build" / "parking_benchmark.yaml",
        help="Output YAML path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "examples" / "parking_benchmark_config.yaml",
        help="Benchmark setup YAML with explicit static-car poses/stall refs and explicit agent starts",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=repo_root / "examples" / "parking_layout.yaml",
        help="Parking-field layout YAML, kept separate from car placement",
    )
    args = parser.parse_args()

    layout = layout_dict(load_layout_config(args.layout))
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    instance = build_instance_from_config(layout, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(instance, handle, sort_keys=False)


if __name__ == "__main__":
    main()

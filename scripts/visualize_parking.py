#!/usr/bin/env python3

import argparse
import json
import math
import statistics
from pathlib import Path

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import yaml


PALETTE = [
    "#12d8a0",
    "#1f7cf2",
    "#242549",
    "#c89228",
    "#8b60d3",
    "#4f8e59",
    "#27b8c9",
    "#9cc23b",
    "#5164c7",
    "#a67246",
    "#57c78f",
    "#6a46b2",
    "#3e938f",
    "#7aaa2a",
    "#2f5db8",
    "#b28b5e",
    "#9d71c8",
    "#2f9a67",
    "#4a6f9a",
    "#8b6a3f",
]

REFERENCE_STYLE = {
    "length_scale": 1.01,
    "width_scale": 1.02,
    "outline": [
        (0.00, 0.10),
        (0.08, 0.31),
        (0.20, 0.47),
        (0.39, 0.50),
        (0.78, 0.50),
        (0.91, 0.45),
        (0.97, 0.31),
        (1.00, 0.10),
    ],
    "roof_w": 0.57,
    "roof_h": 0.28,
    "roof_y": 0.36,
    "windshield": {"y": 0.61, "h": 0.15, "base": 0.52, "top": 0.35},
    "rear_window": {"y": 0.24, "h": 0.11, "base": 0.38, "top": 0.52},
    "hood": {"y": 0.76, "h": 0.11, "base": 0.58, "top": 0.33},
    "deck": {"y": 0.11, "h": 0.08, "base": 0.28, "top": 0.53},
}

ANIMATION_END_HOLD_SEC = 5.0
COLLISION_BUFFER = 0.2


def rotate_translate(points, pose):
    x, y, yaw = pose
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    transformed = []
    for local_x, local_y in points:
        transformed.append(
            (
                x + local_x * cos_yaw - local_y * sin_yaw,
                y + local_x * sin_yaw + local_y * cos_yaw,
            )
        )
    return transformed


def blend(color, target, amount):
    base_rgb = to_rgb(color)
    target_rgb = to_rgb(target)
    return tuple((1.0 - amount) * lhs + amount * rhs for lhs, rhs in zip(base_rgb, target_rgb))


def build_boxy_outline(style):
    max_half_width = max(half_width for _, half_width in style["outline"])
    utility_bias = max(
        0.0,
        min(1.0, ((style["roof_h"] - 0.27) / 0.05) + ((style["width_scale"] - 0.99) / 0.08)),
    )
    rear_cap = max_half_width * (0.54 + 0.08 * utility_bias)
    rear_corner = max_half_width * (0.74 + 0.05 * utility_bias)
    rear_shoulder = max_half_width * (0.90 + 0.06 * utility_bias)
    side_half = max_half_width
    front_shoulder = max_half_width * (0.91 + 0.05 * utility_bias)
    front_corner = max_half_width * (0.76 + 0.05 * utility_bias)
    front_cap = max_half_width * (0.58 + 0.08 * utility_bias)
    return [
        (0.00, rear_cap),
        (0.02, rear_corner),
        (0.08, rear_shoulder),
        (0.18, side_half),
        (0.82, side_half),
        (0.92, front_shoulder),
        (0.98, front_corner),
        (1.00, front_cap),
    ]


def build_body_local_points(style, car_width, car_length):
    outline = build_boxy_outline(style)
    right_side = [((y_frac - 0.5) * car_length, half_width * car_width) for y_frac, half_width in outline]
    left_side = [((y_frac - 0.5) * car_length, -half_width * car_width) for y_frac, half_width in reversed(outline)]
    return right_side + left_side


def interpolate_half_width(outline, y_frac):
    if y_frac <= outline[0][0]:
        return outline[0][1]
    if y_frac >= outline[-1][0]:
        return outline[-1][1]
    for (y0, width0), (y1, width1) in zip(outline, outline[1:]):
        if y0 <= y_frac <= y1:
            span = y1 - y0
            if span == 0.0:
                return width1
            ratio = (y_frac - y0) / span
            return width0 + ratio * (width1 - width0)
    return outline[-1][1]


def build_tapered_local_points(spec, taper_to_front, car_width, car_length):
    y0 = (spec["y"] - 0.5) * car_length
    y1 = (spec["y"] + spec["h"] - 0.5) * car_length
    base_half = 0.5 * spec["base"] * car_width
    top_half = 0.5 * spec["top"] * car_width
    if taper_to_front:
        return [(y0, -base_half), (y0, base_half), (y1, top_half), (y1, -top_half)]
    return [(y0, -top_half), (y0, top_half), (y1, base_half), (y1, -base_half)]


def make_footprint(pose, length, width, length_pad=0.0, width_pad=0.0):
    x, y, yaw = pose
    half_length = (length + length_pad) / 2.0
    half_width = (width + width_pad) / 2.0
    corners = [
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
        for local_x, local_y in corners
    ]


def normalize_half_turn(yaw):
    wrapped = math.fmod(yaw, math.pi)
    if wrapped < 0.0:
        wrapped += math.pi
    return wrapped


def angle_distance_mod_pi(lhs, rhs):
    delta = abs(normalize_half_turn(lhs) - normalize_half_turn(rhs))
    return min(delta, math.pi - delta)


def collect_plot_extent(instance, solution):
    points = []

    def add_pose(pose):
        points.append((pose[0], pose[1]))

    for parked in instance.get("static_cars", []):
        add_pose(parked["pose"])
    for agent in instance["agents"]:
        add_pose(agent["start"])
        add_pose(agent["goal"])
        for point in solution.get("trajectories", {}).get(str(agent["id"]), []):
            points.append((point["x"], point["y"]))

    if not points:
        return (0.0, instance["map"]["width"], 0.0, instance["map"]["height"])

    xs, ys = zip(*points)
    padding = max(instance["vehicle"]["length"], instance["vehicle"]["width"]) * 1.2
    return (
        max(0.0, min(xs) - padding),
        min(instance["map"]["width"], max(xs) + padding),
        max(0.0, min(ys) - padding),
        min(instance["map"]["height"], max(ys) + padding),
    )


def visualization_layout(instance):
    raw = instance.get("parking_layout")
    if raw is None:
        return None

    layout = {
        "grid_cols": raw.get("grid_cols", 2),
        "grid_rows": raw.get("grid_rows", 2),
        "stalls_per_side": raw.get("stalls_per_side", 6),
        "stall_width": raw.get("stall_width", 3.35),
        "stall_depth": raw.get("stall_depth", 6.3),
        "zone_gap": raw.get("zone_gap", 8.375),
        "perimeter_gap": raw.get("perimeter_gap", 8.375),
    }
    layout["row_width"] = layout["stalls_per_side"] * layout["stall_width"]
    layout["module_height"] = 2.0 * layout["stall_depth"]
    layout["parking_width"] = (
        2.0 * layout["perimeter_gap"]
        + layout["grid_cols"] * layout["row_width"]
        + (layout["grid_cols"] - 1) * layout["zone_gap"]
    )
    layout["parking_height"] = (
        2.0 * layout["perimeter_gap"]
        + layout["grid_rows"] * layout["module_height"]
        + (layout["grid_rows"] - 1) * layout["zone_gap"]
    )
    layout["total_height"] = layout["parking_height"]
    return layout


def layout_zone_origins(layout):
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


def draw_explicit_parking_layout(ax, layout, map_width, map_height):
    ax.add_patch(
        Polygon(
            [(0.0, 0.0), (map_width, 0.0), (map_width, map_height), (0.0, map_height)],
            closed=True,
            facecolor="#a8b0b8",
            edgecolor="#374151",
            linewidth=2.2,
            zorder=0,
        )
    )

    for _, row_index, x_start, center_y in layout_zone_origins(layout):
        center_y_world = center_y
        x_end = x_start + layout["row_width"]
        for stall_index in range(layout["stalls_per_side"] + 1):
            x = x_start + stall_index * layout["stall_width"]
            ax.add_line(
                Line2D(
                    [x, x],
                    [center_y_world - layout["stall_depth"], center_y_world + layout["stall_depth"]],
                    color="#ffffff",
                    linewidth=3.0,
                    alpha=0.98,
                    solid_capstyle="round",
                    zorder=1.7,
                )
            )
        ax.add_line(
            Line2D(
                [x_start, x_end],
                [center_y_world, center_y_world],
                color="#ffffff",
                linewidth=3.0,
                alpha=0.98,
                solid_capstyle="round",
                zorder=1.7,
            )
        )


def draw_vehicle(
    ax,
    pose,
    vehicle,
    color,
    alpha,
    linestyle="-",
    label=None,
    zorder=4,
    linewidth=1.6,
):
    style = REFERENCE_STYLE
    car_length = vehicle["length"]
    car_width = vehicle["width"]
    body_outline = build_boxy_outline(style)
    body_points = build_body_local_points(style, car_width, car_length)
    edge_color = blend(color, "#0f172a", 0.38)
    roof_color = blend(color, "#f8fafc", 0.15)
    glass_color = "#95a6b5"

    ax.add_patch(
        Polygon(
            rotate_translate(body_points, pose),
            closed=True,
            facecolor=color,
            edgecolor=edge_color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            zorder=zorder,
            joinstyle="miter",
        )
    )

    rear_window = style["rear_window"]
    windshield = style["windshield"]
    hood = style["hood"]
    deck = style["deck"]

    greenhouse_y0_frac = max(deck["y"] + 0.02, rear_window["y"] - 0.08)
    greenhouse_y1_frac = min(windshield["y"] + windshield["h"] - 0.08, hood["y"] - 0.08)
    greenhouse_length = (greenhouse_y1_frac - greenhouse_y0_frac) * car_length
    greenhouse_width = max(style["roof_w"] * car_width * 0.92, 0.45 * car_width)
    greenhouse_center_frac = 0.5 * (greenhouse_y0_frac + greenhouse_y1_frac)
    greenhouse_pose = (
        pose[0] + (greenhouse_center_frac - 0.5) * car_length * math.cos(pose[2]),
        pose[1] + (greenhouse_center_frac - 0.5) * car_length * math.sin(pose[2]),
        pose[2],
    )
    ax.add_patch(
        Polygon(
            make_footprint(greenhouse_pose, greenhouse_length, greenhouse_width),
            closed=True,
            facecolor=glass_color,
            edgecolor="none",
            alpha=alpha * 0.96,
            zorder=zorder + 0.8,
        )
    )

    roof_start_frac = max(greenhouse_y0_frac + 0.11, style["roof_y"] - 0.05)
    roof_span_frac = style["roof_h"] * 0.68
    roof_center_frac = roof_start_frac + 0.5 * roof_span_frac
    roof_pose = (
        pose[0] + (roof_center_frac - 0.5) * car_length * math.cos(pose[2]),
        pose[1] + (roof_center_frac - 0.5) * car_length * math.sin(pose[2]),
        pose[2],
    )
    ax.add_patch(
        Polygon(
            make_footprint(roof_pose, roof_span_frac * car_length, greenhouse_width),
            closed=True,
            facecolor=roof_color,
            edgecolor="none",
            alpha=alpha,
            zorder=zorder + 1.1,
        )
    )

    for spec, taper_to_front in (
        (deck, False),
        (rear_window, False),
        (windshield, True),
        (hood, True),
    ):
        patch_color = roof_color if spec in (deck, hood) else glass_color
        patch_alpha = alpha if spec in (deck, hood) else alpha * 0.98
        ax.add_patch(
            Polygon(
                rotate_translate(build_tapered_local_points(spec, taper_to_front, car_width, car_length), pose),
                closed=True,
                facecolor=patch_color,
                edgecolor="none",
                alpha=patch_alpha,
                zorder=zorder + (1.2 if spec in (deck, hood) else 1.0),
            )
        )

    front_light_y = 0.95
    rear_light_y = 0.08
    front_half_width = interpolate_half_width(body_outline, front_light_y) * car_width
    rear_half_width = interpolate_half_width(body_outline, rear_light_y) * car_width
    front_light_width = 0.19 * car_width
    rear_light_width = 0.20 * car_width
    light_height = 0.036 * car_length

    for sign in (-1.0, 1.0):
        front_center = ((front_light_y - 0.5) * car_length, sign * (front_half_width - 0.42 * front_light_width))
        rear_center = ((rear_light_y - 0.5) * car_length, sign * (rear_half_width - 0.44 * rear_light_width))
        for local_center, light_width, light_color in (
            (front_center, front_light_width, "#ffffff"),
            (rear_center, rear_light_width, "#ff4d4d"),
        ):
            cx, cy = local_center
            ax.add_patch(
                Polygon(
                    rotate_translate(
                        [
                            (cx - light_height / 2.0, cy - light_width / 2.0),
                            (cx - light_height / 2.0, cy + light_width / 2.0),
                            (cx + light_height / 2.0, cy + light_width / 2.0),
                            (cx + light_height / 2.0, cy - light_width / 2.0),
                        ],
                        pose,
                    ),
                    closed=True,
                    facecolor=light_color,
                    edgecolor="none",
                    alpha=alpha * 0.98,
                    zorder=zorder + 1.4,
                )
            )


def draw_unresolved_overlay(ax, pose, vehicle, zorder=11):
    ax.add_patch(
        Polygon(
            make_footprint(
                pose,
                vehicle["length"],
                vehicle["width"],
                length_pad=2.0 * COLLISION_BUFFER,
                width_pad=2.0 * COLLISION_BUFFER,
            ),
            closed=True,
            facecolor=(1.0, 0.2, 0.2, 0.10),
            edgecolor="#c1121f",
            linewidth=1.3,
            hatch="xx",
            zorder=zorder,
        )
    )


def trajectory_pose_at(points, frame_index):
    if not points:
        return None
    point = points[min(frame_index, len(points) - 1)]
    return (point["x"], point["y"], point["yaw"])


def draw_stall(
    ax,
    pose,
    vehicle,
    edgecolor,
    facecolor="none",
    alpha=1.0,
    linestyle="-",
    label=None,
    zorder=2,
    linewidth=2.0,
):
    polygon = Polygon(
        make_footprint(
            pose,
            vehicle["length"],
            vehicle["width"],
            length_pad=0.8,
            width_pad=0.8,
        ),
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        zorder=zorder,
    )
    ax.add_patch(polygon)


def project_pose(pose, yaw):
    heading_x = math.cos(yaw)
    heading_y = math.sin(yaw)
    lateral_x = -math.sin(yaw)
    lateral_y = math.cos(yaw)
    row_position = pose[0] * heading_x + pose[1] * heading_y
    lateral_position = pose[0] * lateral_x + pose[1] * lateral_y
    return row_position, lateral_position


def reconstruct_pose(row_position, lateral_position, yaw):
    heading_x = math.cos(yaw)
    heading_y = math.sin(yaw)
    lateral_x = -math.sin(yaw)
    lateral_y = math.cos(yaw)
    return (
        row_position * heading_x + lateral_position * lateral_x,
        row_position * heading_y + lateral_position * lateral_y,
        yaw,
    )


def infer_stall_rows(instance):
    references = []
    for parked in instance.get("static_cars", []):
        references.append(tuple(parked["pose"]))
    for agent in instance.get("agents", []):
        references.append(tuple(agent["goal"]))

    if not references:
        return []

    orientation_tolerance = 0.2
    row_tolerance = max(instance["vehicle"]["width"], instance["vehicle"]["length"]) * 0.55
    default_spacing = instance["vehicle"]["length"] + 0.8

    orientation_groups = []
    for pose in references:
        matched_group = None
        for group in orientation_groups:
            if angle_distance_mod_pi(pose[2], group["yaw"]) <= orientation_tolerance:
                matched_group = group
                break
        if matched_group is None:
            matched_group = {"yaw": normalize_half_turn(pose[2]), "poses": []}
            orientation_groups.append(matched_group)
        matched_group["poses"].append(pose)

    inferred_rows = []
    for group in orientation_groups:
        row_clusters = []
        for pose in group["poses"]:
            row_position, lateral_position = project_pose(pose, group["yaw"])
            matched_cluster = None
            for cluster in row_clusters:
                if abs(row_position - cluster["row_position"]) <= row_tolerance:
                    matched_cluster = cluster
                    break
            if matched_cluster is None:
                matched_cluster = {"row_position": row_position, "laterals": []}
                row_clusters.append(matched_cluster)
            matched_cluster["laterals"].append(lateral_position)

        for cluster in row_clusters:
            lateral_positions = sorted(cluster["laterals"])
            spacing_candidates = [
                current - previous
                for previous, current in zip(lateral_positions, lateral_positions[1:])
                if current - previous > 0.5
            ]
            spacing = min(spacing_candidates) if spacing_candidates else default_spacing
            stall_count = max(1, int(round((lateral_positions[-1] - lateral_positions[0]) / spacing)) + 1)
            inferred_rows.append(
                [
                    reconstruct_pose(
                        cluster["row_position"],
                        lateral_positions[0] + index * spacing,
                        group["yaw"],
                    )
                    for index in range(stall_count)
                ]
            )

    return inferred_rows


def row_coordinates(row):
    yaw = normalize_half_turn(row[0][2])
    projected = [project_pose(pose, yaw) for pose in row]
    row_position = statistics.mean(position[0] for position in projected)
    lateral_positions = sorted(position[1] for position in projected)
    spacing_candidates = [
        current - previous
        for previous, current in zip(lateral_positions, lateral_positions[1:])
        if current - previous > 0.5
    ]
    spacing = min(spacing_candidates) if spacing_candidates else 4.0
    return yaw, row_position, lateral_positions, spacing


def draw_row_markings(ax, row, vehicle, label=None):
    yaw, row_position, lateral_positions, spacing = row_coordinates(row)
    half_depth = (vehicle["length"] + 0.8) / 2.0
    front_edge = row_position + half_depth
    back_edge = row_position - half_depth
    boundaries = [lateral_positions[0] - spacing / 2.0]
    boundaries.extend(
        (previous + current) / 2.0
        for previous, current in zip(lateral_positions, lateral_positions[1:])
    )
    boundaries.append(lateral_positions[-1] + spacing / 2.0)

    for index, lateral in enumerate(boundaries):
        start = reconstruct_pose(back_edge, lateral, yaw)
        end = reconstruct_pose(front_edge, lateral, yaw)
        ax.add_line(
            Line2D(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#ffffff",
                linewidth=2.6,
                alpha=0.98,
                solid_capstyle="round",
                zorder=1.7,
                label=label if index == 0 else None,
            )
        )

    head_left = reconstruct_pose(front_edge, boundaries[0], yaw)
    head_right = reconstruct_pose(front_edge, boundaries[-1], yaw)
    ax.add_line(
        Line2D(
            [head_left[0], head_right[0]],
            [head_left[1], head_right[1]],
            color="#ffffff",
            linewidth=2.6,
            alpha=0.98,
            solid_capstyle="round",
            zorder=1.7,
        )
    )

    return yaw, row_position, boundaries, spacing


def draw_individual_spot(ax, pose, vehicle, label=None):
    yaw = normalize_half_turn(pose[2])
    row_position, lateral_position = project_pose(pose, yaw)
    half_depth = (vehicle["length"] + 0.8) / 2.0
    half_width = (vehicle["width"] + 0.8) / 2.0
    front_edge = row_position + half_depth
    back_edge = row_position - half_depth
    boundaries = [lateral_position - half_width, lateral_position + half_width]

    for index, lateral in enumerate(boundaries):
        start = reconstruct_pose(back_edge, lateral, yaw)
        end = reconstruct_pose(front_edge, lateral, yaw)
        ax.add_line(
            Line2D(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#ffffff",
                linewidth=2.6,
                alpha=0.98,
                solid_capstyle="round",
                zorder=1.7,
                label=label if index == 0 else None,
            )
        )

    head_left = reconstruct_pose(front_edge, boundaries[0], yaw)
    head_right = reconstruct_pose(front_edge, boundaries[-1], yaw)
    ax.add_line(
        Line2D(
            [head_left[0], head_right[0]],
            [head_left[1], head_right[1]],
            color="#ffffff",
            linewidth=2.6,
            alpha=0.98,
            solid_capstyle="round",
            zorder=1.7,
        )
    )


def draw_goal_spot(ax, pose, row_info, vehicle, color, label=None):
    yaw = normalize_half_turn(pose[2])
    if row_info is not None:
        row_yaw, row_position, boundaries, spacing = row_info
        if angle_distance_mod_pi(yaw, row_yaw) <= 0.2:
            _, lateral_position = project_pose(pose, row_yaw)
            boundary_index = min(
                range(len(boundaries) - 1),
                key=lambda idx: abs((boundaries[idx] + boundaries[idx + 1]) / 2.0 - lateral_position),
            )
            half_depth = (vehicle["length"] + 0.8) / 2.0
            top_left = reconstruct_pose(row_position - half_depth, boundaries[boundary_index], row_yaw)
            top_right = reconstruct_pose(row_position - half_depth, boundaries[boundary_index + 1], row_yaw)
            bottom_right = reconstruct_pose(row_position + half_depth, boundaries[boundary_index + 1], row_yaw)
            bottom_left = reconstruct_pose(row_position + half_depth, boundaries[boundary_index], row_yaw)
            ax.add_patch(
                Polygon(
                    [top_left[:2], top_right[:2], bottom_right[:2], bottom_left[:2]],
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1.5,
                    linestyle="-",
                    alpha=0.12,
                    zorder=2.2,
                    label=label,
                )
            )
            return

    draw_stall(
        ax,
        pose,
        vehicle,
        edgecolor=color,
        facecolor=color,
        alpha=0.08,
        label=label,
        zorder=2.2,
        linewidth=1.6,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot a parking lot planning result.")
    parser.add_argument("instance", type=Path, help="YAML instance file")
    parser.add_argument(
        "solution",
        type=Path,
        nargs="?",
        default=None,
        help="Optional JSON solution file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure instead of opening a window",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate the time-indexed trajectories",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Animation frames per second when using --animate",
    )
    args = parser.parse_args()

    with args.instance.open("r", encoding="utf-8") as handle:
        instance = yaml.safe_load(handle)

    if args.animate and args.solution is None:
        raise SystemExit("--animate requires a solution JSON")

    if args.solution is None:
        solution = {"success": False, "metrics": {}, "trajectories": {}}
        base_title = "Parking Lot Scenario"
    else:
        with args.solution.open("r", encoding="utf-8") as handle:
            solution = json.load(handle)
        metrics = solution.get("metrics", {})
        base_title = (
            "Parking Lot Plan"
            f" | success={bool(solution.get('success', False))}"
            f" | makespan={metrics.get('makespan', 0)}"
        )

    fig, ax = plt.subplots(figsize=(11.5, 9.0))
    metrics = solution.get("metrics", {})
    ax.set_title(base_title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#a8b0b8")

    vehicle = instance["vehicle"]
    layout = visualization_layout(instance)

    if layout is not None:
        map_width = instance["map"].get("width", layout["parking_width"])
        map_height = instance["map"].get("height", layout["total_height"])
        ax.set_xlim(0.0, map_width)
        ax.set_ylim(0.0, map_height)
        draw_explicit_parking_layout(ax, layout, map_width, map_height)
        row_infos = []
    else:
        xmin, xmax, ymin, ymax = collect_plot_extent(instance, solution)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        boundary = [
            (0.0, 0.0),
            (instance["map"]["width"], 0.0),
            (instance["map"]["width"], instance["map"]["height"]),
            (0.0, instance["map"]["height"]),
        ]
        ax.add_patch(
            Polygon(
                boundary,
                closed=True,
                facecolor="#a8b0b8",
                edgecolor="#374151",
                linewidth=2.2,
                zorder=0,
            )
        )

        stall_rows = infer_stall_rows(instance)
        row_infos = []
        for row in stall_rows:
            if len(row) >= 3:
                row_infos.append(draw_row_markings(ax, row, vehicle))
                continue
            for pose in row:
                draw_individual_spot(ax, pose, vehicle)

    static_cars = instance.get("static_cars", [])
    for index, parked in enumerate(static_cars):
        draw_vehicle(
            ax,
            parked["pose"],
            vehicle,
            color=PALETTE[index % len(PALETTE)],
            alpha=0.98,
            zorder=3,
        )

    for index, agent in enumerate(instance["agents"]):
        color = PALETTE[(index + len(static_cars)) % len(PALETTE)]
        goal_row_info = None
        goal_match_distance = float("inf")
        for row_info in row_infos:
            row_yaw, row_position, boundaries, spacing = row_info
            if angle_distance_mod_pi(agent["goal"][2], row_yaw) > 0.2:
                continue
            projected_row, projected_lateral = project_pose(agent["goal"], row_yaw)
            distance_to_row = abs(projected_row - row_position)
            if distance_to_row < goal_match_distance:
                goal_match_distance = distance_to_row
                goal_row_info = row_info
        draw_goal_spot(
            ax,
            agent["goal"],
            goal_row_info,
            vehicle,
            color,
        )
        draw_vehicle(
            ax,
            agent["start"],
            vehicle,
            color=color,
            alpha=0.28,
            linestyle="-",
            zorder=5,
        )
        draw_vehicle(
            ax,
            agent["goal"],
            vehicle,
            color=color,
            alpha=0.14,
            linestyle="-",
            zorder=4,
        )

        points = solution["trajectories"].get(str(agent["id"]), [])
        if not points:
            if args.solution is not None:
                draw_vehicle(
                    ax,
                    agent["start"],
                    vehicle,
                    color=color,
                    alpha=0.98,
                    zorder=8,
                )
                draw_unresolved_overlay(ax, agent["start"], vehicle, zorder=20)
            continue
        xs = [point["x"] for point in points]
        ys = [point["y"] for point in points]
        ax.plot(xs, ys, color="white", linewidth=4.4, alpha=0.22 if args.animate else 0.9, zorder=6)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.6,
            alpha=0.28 if args.animate else 1.0,
            zorder=7,
        )
        if not args.animate:
            ax.scatter(xs[0], ys[0], color=color, s=55, zorder=9)
            ax.scatter(xs[-1], ys[-1], color=color, s=80, marker="x", linewidths=2.2, zorder=9)

    fig.tight_layout()

    if args.animate:
        animated_agents = [
            {
                "agent": agent,
                "points": solution["trajectories"].get(str(agent["id"]), []),
                "color": PALETTE[(index + len(static_cars)) % len(PALETTE)],
            }
            for index, agent in enumerate(instance["agents"])
            if solution["trajectories"].get(str(agent["id"]), [])
        ]
        stationary_agents = [
            {
                "agent": agent,
                "color": PALETTE[(index + len(static_cars)) % len(PALETTE)],
            }
            for index, agent in enumerate(instance["agents"])
            if not solution["trajectories"].get(str(agent["id"]), [])
        ]
        frame_count = int(metrics.get("makespan", 0)) + 1
        if frame_count <= 0:
            frame_count = max(
                (
                    max((point.get("t", idx) for idx, point in enumerate(agent_info["points"])), default=0)
                    + 1
                    for agent_info in animated_agents
                ),
                default=1,
            )
        hold_frames = max(1, int(round(ANIMATION_END_HOLD_SEC * max(1, args.fps))))
        frame_indices = list(range(frame_count)) + [max(0, frame_count - 1)] * hold_frames

        dynamic_artists = []

        def clear_dynamic():
            while dynamic_artists:
                dynamic_artists.pop().remove()

        def update(frame_index):
            clear_dynamic()
            ax.set_title(f"{base_title} | t={frame_index}")
            for agent_info in animated_agents:
                points = agent_info["points"]
                color = agent_info["color"]
                trail = [point for point in points if point.get("t", 0) <= frame_index]
                if trail:
                    xs = [point["x"] for point in trail]
                    ys = [point["y"] for point in trail]
                    white_line = ax.plot(xs, ys, color="white", linewidth=4.4, alpha=0.95, zorder=8)[0]
                    color_line = ax.plot(xs, ys, color=color, linewidth=2.7, zorder=9)[0]
                    dynamic_artists.extend([white_line, color_line])
                pose = trajectory_pose_at(points, frame_index)
                if pose is not None:
                    before = len(ax.patches)
                    draw_vehicle(
                        ax,
                        pose,
                        vehicle,
                        color=color,
                        alpha=1.0,
                        zorder=10,
                    )
                    dynamic_artists.extend(ax.patches[before:])
            for agent_info in stationary_agents:
                before = len(ax.patches)
                draw_vehicle(
                    ax,
                    agent_info["agent"]["start"],
                    vehicle,
                    color=agent_info["color"],
                    alpha=0.98,
                    zorder=10,
                )
                draw_unresolved_overlay(
                    ax,
                    agent_info["agent"]["start"],
                    vehicle,
                    zorder=20,
                )
                dynamic_artists.extend(ax.patches[before:])
            return dynamic_artists

        animation = FuncAnimation(
            fig,
            update,
            frames=frame_indices,
            interval=max(1, int(round(1000 / max(1, args.fps)))),
            blit=False,
            repeat=True,
        )
        update(0)
        if args.output:
            animation.save(args.output, writer=PillowWriter(fps=max(1, args.fps)), dpi=160)
        else:
            plt.show()
        return

    if args.output:
        fig.savefig(args.output, dpi=160)
    else:
        plt.show()


if __name__ == "__main__":
    main()

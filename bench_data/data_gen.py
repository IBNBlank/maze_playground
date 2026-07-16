#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-16
################################################################

import json
import math
import time
from typing import Sequence

import cv2
import numpy as np
from skimage.graph import route_through_array

from utils.arg import args_init, config_init, GenCfg
from utils.common import save_preview

RC = tuple[int, int]


def _u8_view(mask: np.ndarray) -> np.ndarray:
    """Writable uint8 view of a bool/uint8 mask for in-place OpenCV drawing."""
    if mask.dtype == np.uint8:
        return mask
    if mask.dtype == np.bool_:
        return mask.view(np.uint8)
    raise TypeError(f"Expected bool or uint8 mask, got {mask.dtype}")


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


# -----------------------------------------------------------------------------
# Raster geometry (OpenCV; masks use row/col, drawing uses x/y = col/row)
# -----------------------------------------------------------------------------
def carve_disk(mask: np.ndarray, center: RC, radius: int) -> None:
    if radius < 0:
        return
    canvas = _u8_view(mask)
    cv2.circle(
        canvas,
        (int(center[1]), int(center[0])),
        int(radius),
        1,
        thickness=-1,
        lineType=cv2.LINE_8,
    )


def carve_polyline(mask: np.ndarray, points: np.ndarray, radius: int) -> None:
    if radius < 0 or len(points) == 0:
        return
    canvas = _u8_view(mask)
    if len(points) == 1:
        carve_disk(
            mask,
            (int(round(float(points[0, 0]))), int(round(float(points[0, 1])))),
            radius,
        )
        return
    # points are [row, col]; OpenCV expects [x, y] = [col, row].
    pts = np.round(points[:, [1, 0]]).astype(np.int32).reshape(-1, 1, 2)
    thickness = max(1, 2 * int(radius) + 1)
    cv2.polylines(
        canvas,
        [pts],
        isClosed=False,
        color=1,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )
    # Round caps so corridor ends match filled disks of the same radius.
    carve_disk(
        mask,
        (int(round(float(points[0, 0]))), int(round(float(points[0, 1])))),
        radius,
    )
    carve_disk(
        mask,
        (int(round(float(points[-1, 0]))), int(round(float(points[-1, 1])))),
        radius,
    )


def dense_segment_cells(a: RC, b: RC, oversample: int = 3) -> np.ndarray:
    dr, dc = int(b[0] - a[0]), int(b[1] - a[1])
    count = max(abs(dr), abs(dc)) * oversample + 1
    rows = np.rint(np.linspace(a[0], b[0], count)).astype(np.int32)
    cols = np.rint(np.linspace(a[1], b[1], count)).astype(np.int32)
    cells = np.stack([rows, cols], axis=1)
    if len(cells) <= 1:
        return cells
    keep = np.ones(len(cells), dtype=bool)
    keep[1:] = np.any(cells[1:] != cells[:-1], axis=1)
    return cells[keep]


def line_is_free(a: RC, b: RC, occupancy: np.ndarray) -> bool:
    cells = dense_segment_cells(a, b, oversample=4)
    rows, cols = cells[:, 0], cells[:, 1]
    if (np.any(rows < 0) or np.any(cols < 0)
            or np.any(rows >= occupancy.shape[0])
            or np.any(cols >= occupancy.shape[1])):
        return False
    return not bool(np.any(occupancy[rows, cols]))


def polyline_point(polyline: np.ndarray, alpha: float) -> np.ndarray:
    """Approximate a point on a control polyline by normalized segment index."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    scaled = alpha * (len(polyline) - 1)
    index = min(int(math.floor(scaled)), len(polyline) - 2)
    local = scaled - index
    return (1.0 - local) * polyline[index] + local * polyline[index + 1]


# -----------------------------------------------------------------------------
# Guaranteed multimodal free-space construction
# -----------------------------------------------------------------------------


def sample_start_goal(
        cfg: GenCfg,
        rng: np.random.Generator) -> tuple[RC, RC, np.ndarray, np.ndarray]:
    """Sample opposite-side endpoints and return tangent/normal unit vectors."""
    size = cfg.size
    margin = cfg.endpoint_margin
    center = (size - 1) * 0.5
    jitter = cfg.endpoint_center_jitter_ratio * size
    horizontal = bool(rng.integers(0, 2))

    if horizontal:
        start = (
            int(
                np.clip(round(center + rng.uniform(-jitter, jitter)), margin,
                        size - margin - 1)),
            margin,
        )
        goal = (
            int(
                np.clip(round(center + rng.uniform(-jitter, jitter)), margin,
                        size - margin - 1)),
            size - margin - 1,
        )
    else:
        start = (
            margin,
            int(
                np.clip(round(center + rng.uniform(-jitter, jitter)), margin,
                        size - margin - 1)),
        )
        goal = (
            size - margin - 1,
            int(
                np.clip(round(center + rng.uniform(-jitter, jitter)), margin,
                        size - margin - 1)),
        )

    if rng.random() < 0.5:
        start, goal = goal, start

    delta = np.asarray([goal[0] - start[0], goal[1] - start[1]],
                       dtype=np.float32)
    distance = float(np.linalg.norm(delta))
    if distance < cfg.min_start_goal_distance_ratio * size:
        raise RuntimeError(
            "Endpoint sampler produced a pair that is too close")
    tangent = delta / distance
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    return start, goal, tangent, normal


def build_route_polylines(
    start: RC,
    goal: RC,
    tangent: np.ndarray,
    normal: np.ndarray,
    cfg: GenCfg,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    start_f = np.asarray(start, dtype=np.float32)
    goal_f = np.asarray(goal, dtype=np.float32)
    route_distance = float(np.linalg.norm(goal_f - start_f))

    max_offset = min(
        cfg.route_separation_ratio * cfg.size,
        0.5 * route_distance,
    )
    offsets = np.linspace(-max_offset,
                          max_offset,
                          cfg.num_routes,
                          dtype=np.float32)
    offsets += float(rng.uniform(-0.06, 0.06) * cfg.size)

    num_points = 7
    idx_ratios = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    offset_ratios = np.sin(idx_ratios * math.pi)
    offset_ratios += float(rng.uniform(-0.06, 0.06))
    offset_ratios = offset_ratios / offset_ratios.max()

    margin = cfg.border_width + cfg.robot_radius + 2
    polylines: list[np.ndarray] = []
    for route_offset in offsets:
        points: list[np.ndarray] = []
        for control_index, (idx_ratio, offset_ratio) in enumerate(
                zip(idx_ratios, offset_ratios)):
            point = (1.0 - idx_ratio) * start_f + idx_ratio * goal_f
            if control_index != 0 and control_index != num_points - 1:
                normal_offset = offset_ratio * route_offset
                normal_offset += float(rng.normal(-0.02, 0.02) * cfg.size)
                tangent_offset = float(rng.normal(-0.02, 0.02) * cfg.size)
                point += normal * normal_offset + tangent * tangent_offset
                point = np.clip(point, margin, cfg.size - margin - 1)
            points.append(point.astype(np.float32))
        polylines.append(np.stack(points, axis=0))
    return polylines


def build_free_space(
    start: RC,
    goal: RC,
    polylines: Sequence[np.ndarray],
    cfg: GenCfg,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Carve corridors / rooms / links / branches; build per-route guide masks."""
    free_mask = np.zeros((cfg.size, cfg.size), dtype=bool)
    guide_masks: list[np.ndarray] = []

    for polyline in polylines:
        guide = np.zeros_like(free_mask)
        corridor_radius = int(
            rng.integers(cfg.robot_radius * 2, cfg.robot_radius * 3))
        carve_polyline(free_mask, polyline, corridor_radius + 2)
        carve_polyline(guide, polyline, max(1, corridor_radius))
        guide_masks.append(guide)

        room_count = int(
            rng.integers(cfg.rooms_per_route_min, cfg.rooms_per_route_max + 1))
        for _ in range(room_count):
            alpha = float(rng.uniform(0.18, 0.82))
            center = polyline_point(polyline, alpha)
            center_rc = (int(round(float(center[0]))),
                         int(round(float(center[1]))))
            radius = int(
                rng.integers(cfg.room_radius_min, cfg.room_radius_max + 1))
            carve_disk(free_mask, center_rc, radius)

    carve_disk(free_mask, start, cfg.endpoint_room_radius)
    carve_disk(free_mask, goal, cfg.endpoint_room_radius)

    for index in range(len(polylines) - 1):
        if rng.random() > cfg.crosslink_probability:
            continue
        alpha = float(rng.uniform(0.28, 0.72))
        a = polyline_point(polylines[index], alpha)
        b = polyline_point(polylines[index + 1],
                           alpha + float(rng.uniform(-0.08, 0.08)))
        link = np.stack([a, b], axis=0)
        carve_polyline(free_mask, link, cfg.robot_radius + 1)

    branch_count = int(
        rng.integers(cfg.side_branches_min, cfg.side_branches_max + 1))
    for _ in range(branch_count):
        route = polylines[int(rng.integers(0, len(polylines)))]
        alpha = float(rng.uniform(0.12, 0.88))
        anchor = polyline_point(route, alpha)
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        length = float(rng.uniform(0.07, 0.18) * cfg.size)
        endpoint = anchor + length * np.asarray(
            [math.sin(angle), math.cos(angle)], dtype=np.float32)
        margin = cfg.border_width + cfg.robot_radius + 3
        endpoint = np.clip(endpoint, margin, cfg.size - margin - 1)
        branch = np.stack([anchor, endpoint], axis=0)
        carve_polyline(free_mask, branch, cfg.robot_radius + 1)
        carve_disk(
            free_mask,
            (int(round(float(endpoint[0]))), int(round(float(endpoint[1])))),
            int(rng.integers(cfg.room_radius_min, cfg.room_radius_max + 1)),
        )

    return free_mask, guide_masks


def seal_occupancy(
    free_mask: np.ndarray,
    start: RC,
    goal: RC,
    cfg: GenCfg,
) -> np.ndarray:
    """Invert free space, force a border, and keep start/goal free."""
    occupancy = ~free_mask
    b = cfg.border_width
    occupancy[:b, :] = True
    occupancy[-b:, :] = True
    occupancy[:, :b] = True
    occupancy[:, -b:] = True
    occupancy[start] = False
    occupancy[goal] = False
    return occupancy


def inflate_occupancy(occupancy: np.ndarray, radius: int) -> np.ndarray:
    return dilate_mask(occupancy, radius)


# -----------------------------------------------------------------------------
# Search and diverse expert routes
# -----------------------------------------------------------------------------


def minimum_cost_path(
    occupancy: np.ndarray,
    start: RC,
    goal: RC,
    traversal_cost: np.ndarray | None = None,
) -> np.ndarray | None:
    """4-connected minimum-cost path via skimage.graph.MCP_Geometric."""
    if occupancy[start] or occupancy[goal]:
        return None
    if traversal_cost is None:
        traversal_cost = np.ones(occupancy.shape, dtype=np.float32)
    costs = np.asarray(traversal_cost, dtype=np.float64).copy()
    costs[occupancy] = np.inf
    costs[start] = min(float(costs[start]), 1.0)
    costs[goal] = min(float(costs[goal]), 1.0)
    try:
        path, _ = route_through_array(
            costs,
            start=start,
            end=goal,
            fully_connected=False,
            geometric=True,
        )
    except (ValueError, RuntimeError, OverflowError):
        return None
    if not path:
        return None
    result = np.asarray(path, dtype=np.int32)
    if np.any(occupancy[result[:, 0], result[:, 1]]):
        return None
    return result


def grid_path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(
        np.linalg.norm(np.diff(path.astype(np.float32), axis=0), axis=1).sum())


def buffered_path_mask(path: np.ndarray, shape: tuple[int, int],
                       radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[path[:, 0], path[:, 1]] = True
    return dilate_mask(mask, radius)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def smooth_random_cost(shape: tuple[int, int], cfg: GenCfg,
                       rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(shape).astype(np.float32)
    noise = cv2.GaussianBlur(
        noise,
        ksize=(0, 0),
        sigmaX=float(cfg.random_cost_sigma),
        borderType=cv2.BORDER_REFLECT,
    )
    noise -= float(noise.min())
    maximum = float(noise.max())
    if maximum > 1e-8:
        noise /= maximum
    return 1.0 + cfg.random_cost_strength * noise


def generate_expert_routes(
    planning_map: np.ndarray,
    guide_masks: Sequence[np.ndarray],
    start: RC,
    goal: RC,
    cfg: GenCfg,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], float] | None:
    optimal = minimum_cost_path(planning_map, start, goal)
    if optimal is None:
        return None
    optimal_length = grid_path_length(optimal)
    if not (cfg.min_optimal_path_length_ratio * cfg.size <= optimal_length <=
            cfg.max_optimal_path_length_ratio * cfg.size):
        return None

    candidates: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []

    # Search once per constructed guide corridor. The planner, rather than the
    # geometric centerline, remains the source of the expert label.
    for guide in guide_masks:
        cost = smooth_random_cost(planning_map.shape, cfg, rng)
        cost += cfg.guide_penalty * (~guide).astype(np.float32)
        route = minimum_cost_path(planning_map, start, goal, cost)
        if route is None or len(route) > cfg.max_raw_path_points:
            return None
        length = grid_path_length(route)
        if length > cfg.max_route_length_ratio * optimal_length:
            return None

        route_mask = buffered_path_mask(route, planning_map.shape,
                                        cfg.route_iou_radius)
        if candidate_masks:
            overlaps = [mask_iou(route_mask, old) for old in candidate_masks]
            if max(overlaps) > cfg.max_buffered_iou:
                return None
        candidates.append(route)
        candidate_masks.append(route_mask)

    if len(candidates) != cfg.num_routes:
        return None
    return candidates, optimal_length


# -----------------------------------------------------------------------------
# Continuous representation
# -----------------------------------------------------------------------------


def shortcut_path(
    path: np.ndarray,
    planning_map: np.ndarray,
    guide_mask: np.ndarray,
    max_lookahead: int = 56,
) -> np.ndarray:
    """Locally shortcut a route while preserving its guide-corridor mode.

    An unconstrained farthest-visible shortcut can collapse all expert routes to
    the same straight line whenever the union of corridors creates a large open
    room. Requiring every shortcut segment to remain inside a dilated,
    route-specific guide mask keeps the generated action chunks multimodal.
    """
    if len(path) <= 2:
        return path.astype(np.float32)
    allowed = dilate_mask(guide_mask, 2)
    output = [path[0]]
    current = 0
    while current < len(path) - 1:
        next_index = min(len(path) - 1, current + max_lookahead)
        while next_index > current + 1:
            a = (int(path[current, 0]), int(path[current, 1]))
            b = (int(path[next_index, 0]), int(path[next_index, 1]))
            cells = dense_segment_cells(a, b, oversample=4)
            rows, cols = cells[:, 0], cells[:, 1]
            valid = (np.all(rows >= 0) and np.all(cols >= 0)
                     and np.all(rows < planning_map.shape[0])
                     and np.all(cols < planning_map.shape[1])
                     and not np.any(planning_map[rows, cols])
                     and np.all(allowed[rows, cols]))
            if valid:
                break
            next_index -= 1
        output.append(path[next_index])
        current = next_index
    return np.asarray(output, dtype=np.float32)


def resample_polyline(path_xy: np.ndarray, num_points: int) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cumulative = np.concatenate([
        np.zeros(1, dtype=np.float32),
        np.cumsum(segment_lengths, dtype=np.float32)
    ])
    total = float(cumulative[-1])
    if total <= 1e-8:
        raise ValueError("Zero-length path")
    query = np.linspace(0.0, total, num_points, dtype=np.float32)
    result = np.empty((num_points, 2), dtype=np.float32)
    result[:, 0] = np.interp(query, cumulative, path_xy[:, 0])
    result[:, 1] = np.interp(query, cumulative, path_xy[:, 1])
    return result


def route_to_chunk(
    route_rc: np.ndarray,
    planning_map: np.ndarray,
    guide_mask: np.ndarray,
    cfg: GenCfg,
) -> tuple[np.ndarray, np.ndarray]:
    simplified_rc = shortcut_path(route_rc, planning_map, guide_mask)
    simplified_xy = simplified_rc[:, [1, 0]]  # [row,col] -> [x,y]
    pixel_waypoints = resample_polyline(simplified_xy, cfg.action_horizon + 1)
    waypoints_xy = pixel_waypoints / float(cfg.size - 1)
    actions = np.diff(waypoints_xy, axis=0) * float(cfg.action_horizon)
    return waypoints_xy.astype(np.float32), actions.astype(np.float32)


def validate_waypoints(waypoints_xy: np.ndarray, planning_map: np.ndarray,
                       cfg: GenCfg) -> bool:
    pixels_xy = waypoints_xy * float(cfg.size - 1)
    pixels_rc = pixels_xy[:, [1, 0]]
    for a, b in zip(pixels_rc[:-1], pixels_rc[1:]):
        a_rc = (int(round(float(a[0]))), int(round(float(a[1]))))
        b_rc = (int(round(float(b[0]))), int(round(float(b[1]))))
        if not line_is_free(a_rc, b_rc, planning_map):
            return False
    return True


# -----------------------------------------------------------------------------
# Sample generation and storage
# -----------------------------------------------------------------------------
def generate_single_sample(
        cfg: GenCfg, rng: np.random.Generator) -> dict[str, np.ndarray] | None:
    try:
        start, goal, tangent, normal = sample_start_goal(cfg, rng)
    except RuntimeError:
        return None

    polylines = build_route_polylines(start, goal, tangent, normal, cfg, rng)
    free_mask, guide_masks = build_free_space(start, goal, polylines, cfg, rng)
    occupancy = seal_occupancy(free_mask, start, goal, cfg)

    # Search with extra clearance so expert paths stay away from walls;
    # shortcut / validate use the nominal robot inflation.
    search_map = inflate_occupancy(
        occupancy, cfg.robot_radius + cfg.search_inflate_extra)
    planning_map = inflate_occupancy(occupancy, cfg.robot_radius)
    if planning_map[start] or planning_map[goal]:
        return None
    # Extra inflation may cover endpoint cells; keep them traversable for search.
    search_map[start] = False
    search_map[goal] = False

    route_result = generate_expert_routes(search_map, guide_masks, start, goal,
                                          cfg, rng)
    if route_result is None:
        return None
    routes, optimal_length = route_result

    waypoints = np.empty((cfg.num_routes, cfg.action_horizon + 1, 2),
                         dtype=np.float32)
    actions = np.empty((cfg.num_routes, cfg.action_horizon, 2),
                       dtype=np.float32)
    raw_paths = np.full((cfg.num_routes, cfg.max_raw_path_points, 2),
                        -1,
                        dtype=np.int16)
    raw_lengths = np.empty(cfg.num_routes, dtype=np.int16)
    route_lengths = np.empty(cfg.num_routes, dtype=np.float32)

    for index, route in enumerate(routes):
        wp, action = route_to_chunk(route, planning_map, guide_masks[index],
                                    cfg)
        if not validate_waypoints(wp, planning_map, cfg):
            return None
        waypoints[index] = wp
        actions[index] = action
        raw_paths[index, :len(route)] = route.astype(np.int16)
        raw_lengths[index] = len(route)
        route_lengths[index] = grid_path_length(route)

    return {
        "maps": occupancy.astype(np.uint8),
        "planning_maps": planning_map.astype(np.uint8),
        "starts_rc": np.asarray(start, dtype=np.int16),
        "goals_rc": np.asarray(goal, dtype=np.int16),
        "waypoints_xy": waypoints,
        "action_chunks": actions,
        "raw_paths_rc": raw_paths,
        "raw_path_lengths": raw_lengths,
        "route_lengths": route_lengths,
        "optimal_lengths": np.asarray(optimal_length, dtype=np.float32),
    }


def stack_samples(
        samples: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([sample[key] for sample in samples], axis=0)
        for key in samples[0]
    }


def main() -> None:
    args = args_init()
    gen_cfg = config_init(args)

    # prepare for generation
    rng = np.random.default_rng(gen_cfg.seed)
    shard_samples: list[dict[str, np.ndarray]] = []
    shard_names: list[str] = []
    num_valid, num_attempts = 0, 0
    preview_samples: list[dict[str, np.ndarray]] = []

    # generation
    begin = time.perf_counter()
    while num_valid < args.num_maps:
        # generate sample
        sample = None
        for _ in range(gen_cfg.max_map_attempts):
            num_attempts += 1
            sample = generate_single_sample(gen_cfg, rng)
            if sample is not None:
                break

        # if failed to generate sample, raise error
        if sample is None:
            raise RuntimeError(
                "Failed to generate a valid multimodal map. Increase "
                "--max-map-attempts or relax route separation/IoU settings.")
        num_valid += 1

        # append sample to shard
        shard_samples.append(sample)
        if len(preview_samples) < args.preview_count:
            preview_samples.append(sample)

        # save shard
        if len(shard_samples
               ) >= gen_cfg.shard_size or num_valid == args.num_maps:
            shard_name = f"shard_{len(shard_names):05d}.npz"
            np.savez_compressed(
                args.output_dir / shard_name,
                **stack_samples(shard_samples),
            )
            shard_names.append(shard_name)
            shard_samples.clear()

        # print progress
        if num_valid == 1 or num_valid % max(1, min(20,
                                                    args.num_maps // 10)) == 0:
            elapsed = time.perf_counter() - begin
            print(
                f"accepted={num_valid}/{args.num_maps} attempts={num_attempts} "
                f"acceptance={num_valid / num_attempts:.3f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    elapsed = time.perf_counter() - begin

    # finish generation
    preview_name = "preview.png"
    save_preview(preview_samples, args.output_dir / preview_name)
    manifest = {
        "format": "genplan-multimodal-grid-v2",
        "num_maps": args.num_maps,
        "num_shards": len(shard_names),
        "shards": shard_names,
        "preview": preview_name,
        "config": "config.json",
        "action_encoding": {
            "waypoints_xy": "absolute x,y normalized by size-1",
            "action_chunks": "action_horizon * diff(waypoints_xy)",
            "decode": "q[i+1] = q[i] + action[i] / action_horizon",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Done: {args.num_maps} maps in {elapsed:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()

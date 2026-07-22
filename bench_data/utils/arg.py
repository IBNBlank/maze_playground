#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-16
################################################################

import json, tyro
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Args:
    """CLI options for multimodal planning-map generation."""

    num_maps: int = 1000
    size: int = 256
    num_routes: int = 4
    # Fixed chunk length; trailing zeros after goal. Empirically ~50–70 steps
    # at size=256 / max_abs_delta=5 after shortcut (see route_to_chunk).
    action_horizon: int = 72
    max_abs_delta: float = 5.0
    output_dir: Path = Path("../dataset/genplan256")
    shard_size: int = 100
    seed: int = 7
    robot_radius: int = 5
    preview_count: int = 16
    max_map_attempts: int = 80


@dataclass(frozen=True)
class GenCfg:
    size: int = 256
    num_routes: int = 4
    action_horizon: int = 72
    max_abs_delta: float = 5.0
    max_raw_path_points: int = 2048
    robot_radius: int = 5

    # Start/goal and route skeleton.
    border_width: int = 3
    endpoint_margin: int = 18
    endpoint_center_jitter_ratio: float = 0.09
    min_start_goal_distance_ratio: float = 0.70
    route_separation_ratio: float = 0.5

    # Free-space geometry.
    protected_core_radius: int = 5
    endpoint_room_radius: int = 18
    room_radius_min: int = 9
    room_radius_max: int = 20
    rooms_per_route_min: int = 1
    rooms_per_route_max: int = 3
    side_branches_min: int = 3
    side_branches_max: int = 8
    crosslink_probability: float = 0.45

    # Search and route acceptance.
    # Expert search uses robot_radius + search_inflate_extra; shortcut/validate
    # use robot_radius only so paths stay centered while chords stay feasible.
    search_inflate_extra: int = 3
    guide_penalty: float = 8.0
    random_cost_strength: float = 0.35
    random_cost_sigma: float = 14.0
    max_route_length_ratio: float = 1.85
    max_buffered_iou: float = 0.72
    route_iou_radius: int = 3
    min_optimal_path_length_ratio: float = 0.75
    max_optimal_path_length_ratio: float = 2.8

    # Dataset.
    seed: int = 7
    shard_size: int = 100
    max_map_attempts: int = 80


@dataclass
class CheckArgs:
    """Visualize the same leading maps as preview, with per-step robots."""

    dataset_dir: Path = Path("../dataset/genplan256_r2")
    preview_count: int = 16
    output: Path | None = None


@dataclass
class SetArgs:
    """Flatten multimodal demons into map / state / action_chunk / class shards."""

    # Subdir names under demons_root (like env ids in the ManiSkill pipeline).
    demons_ids: tuple[str, ...] = ("genplan256_r2",)
    demons_root: Path = Path("../demons")
    dataset_name: str = "genplan256_r2"
    # Default: ../datasets/{dataset_name}
    out_dir: Path | None = None
    shard_size: int = 2048
    num_idx_perms: int = 300
    idx_perm_seed: int = 0
    # Validate action_chunks length; 0 skips the check.
    action_horizon: int = 72


def args_init() -> Args:
    args = tyro.cli(Args)
    if args.num_maps <= 0:
        raise ValueError("--num-maps must be positive")
    if args.size < 64:
        raise ValueError("--size must be at least 64")
    if args.num_routes < 2:
        raise ValueError(
            "--num-routes should be at least 2 for multimodal data")
    if args.action_horizon < 2:
        raise ValueError("--action-horizon must be at least 2")
    if args.max_abs_delta <= 0.0:
        raise ValueError("--max-abs-delta must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def config_init(args: Args) -> GenCfg:
    scale = args.size / 256.0
    cfg = GenCfg(
        size=args.size,
        num_routes=args.num_routes,
        action_horizon=args.action_horizon,
        max_abs_delta=float(args.max_abs_delta),
        shard_size=args.shard_size,
        seed=args.seed,
        robot_radius=args.robot_radius,
        max_map_attempts=args.max_map_attempts,
        border_width=max(2, round(3 * scale)),
        endpoint_margin=max(10, round(18 * scale)),
        protected_core_radius=max(2, round(5 * scale)),
        endpoint_room_radius=max(8, round(18 * scale)),
        room_radius_min=max(5, round(9 * scale)),
        room_radius_max=max(8, round(20 * scale)),
        search_inflate_extra=max(1, round(3 * scale)),
        random_cost_sigma=max(4.0, 14.0 * scale),
        route_iou_radius=max(1, round(3 * scale)),
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return cfg

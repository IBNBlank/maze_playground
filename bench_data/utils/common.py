#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-16
################################################################

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Preview rendering
# -----------------------------------------------------------------------------

COLORS = (
    (225, 35, 35),
    (35, 115, 235),
    (25, 175, 85),
    (225, 135, 25),
    (155, 65, 205),
    (25, 180, 180),
)


def _occupancy_rgb(occupancy: np.ndarray) -> np.ndarray:
    rgb = np.where(occupancy[..., None] > 0, 35, 242).astype(np.uint8)
    return np.repeat(rgb, 3, axis=2)


def _draw_endpoints(rgb: np.ndarray, sample: dict[str, np.ndarray]) -> None:
    sr, sc = int(sample["starts_rc"][0]), int(sample["starts_rc"][1])
    gr, gc = int(sample["goals_rc"][0]), int(sample["goals_rc"][1])
    radius = 4
    cv2.circle(rgb, (sc, sr),
               radius, (20, 215, 65),
               thickness=-1,
               lineType=cv2.LINE_AA)
    cv2.circle(rgb, (sc, sr),
               radius, (0, 65, 0),
               thickness=1,
               lineType=cv2.LINE_AA)
    cv2.circle(rgb, (gc, gr),
               radius, (245, 45, 45),
               thickness=-1,
               lineType=cv2.LINE_AA)
    cv2.circle(rgb, (gc, gr),
               radius, (85, 0, 0),
               thickness=1,
               lineType=cv2.LINE_AA)


def render_sample(sample: dict[str, np.ndarray]) -> np.ndarray:
    occupancy = sample["maps"]
    rgb = _occupancy_rgb(occupancy)
    h, w = occupancy.shape
    for route_index, route in enumerate(sample["waypoints_xy"]):
        pts = np.empty((len(route), 1, 2), dtype=np.int32)
        pts[:, 0, 0] = np.rint(route[:, 0] * (w - 1)).astype(np.int32)
        pts[:, 0, 1] = np.rint(route[:, 1] * (h - 1)).astype(np.int32)
        cv2.polylines(
            rgb,
            [pts],
            isClosed=False,
            color=COLORS[route_index % len(COLORS)],
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    _draw_endpoints(rgb, sample)
    return rgb


def render_sample_robots(
    sample: dict[str, np.ndarray],
    robot_radius: int = 2,
) -> np.ndarray:
    """Draw each waypoint as a robot disk; routes use COLORS."""
    occupancy = sample["maps"]
    rgb = _occupancy_rgb(occupancy)
    h, w = occupancy.shape
    radius = max(1, int(robot_radius))

    for route_index, route in enumerate(sample["waypoints_xy"]):
        color = COLORS[route_index % len(COLORS)]
        pts = np.empty((len(route), 1, 2), dtype=np.int32)
        pts[:, 0, 0] = np.rint(route[:, 0] * (w - 1)).astype(np.int32)
        pts[:, 0, 1] = np.rint(route[:, 1] * (h - 1)).astype(np.int32)
        cv2.polylines(
            rgb,
            [pts],
            isClosed=False,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        for step, (x, y) in enumerate(pts[:, 0]):
            # Fade early steps so later poses stay readable.
            alpha = 0.25 + 0.75 * (step / max(1, len(pts) - 1))
            overlay = rgb.copy()
            cv2.circle(
                overlay,
                (int(x), int(y)),
                radius,
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                overlay,
                (int(x), int(y)),
                radius,
                (0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.addWeighted(overlay, alpha, rgb, 1.0 - alpha, 0, dst=rgb)

    _draw_endpoints(rgb, sample)
    return rgb


def save_collage(tiles: Sequence[np.ndarray], path: Path) -> None:
    if not tiles:
        return
    columns = int(math.ceil(math.sqrt(len(tiles))))
    rows = int(math.ceil(len(tiles) / columns))
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.full((rows * tile_h, columns * tile_w, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        r0 = (index // columns) * tile_h
        c0 = (index % columns) * tile_w
        canvas[r0:r0 + tile_h, c0:c0 + tile_w] = tile
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_preview(samples: Sequence[dict[str, np.ndarray]], path: Path) -> None:
    if not samples:
        return
    save_collage([render_sample(sample) for sample in samples], path)

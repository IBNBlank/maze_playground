#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-16
################################################################

import json
from pathlib import Path

import numpy as np
import tyro

from utils.arg import CheckArgs
from utils.common import render_sample_robots, save_collage


def _load_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(dataset_dir: Path, manifest: dict) -> dict:
    path = dataset_dir / manifest.get("config", "config.json")
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_preview_samples(
    dataset_dir: Path,
    shard_names: list[str],
    preview_count: int,
) -> list[dict[str, np.ndarray]]:
    if preview_count <= 0:
        raise ValueError("--preview-count must be positive")

    samples: list[dict[str, np.ndarray]] = []
    for shard_name in shard_names:
        if len(samples) >= preview_count:
            break
        with np.load(dataset_dir / shard_name) as shard:
            batch = int(shard["maps"].shape[0])
            take = min(batch, preview_count - len(samples))
            for local_i in range(take):
                samples.append({
                    key: np.asarray(shard[key][local_i])
                    for key in shard.files if key != "split"
                })
    return samples


def main() -> None:
    args = tyro.cli(CheckArgs)
    dataset_dir = args.dataset_dir.resolve()
    manifest = _load_manifest(dataset_dir)
    config = _load_config(dataset_dir, manifest)
    robot_radius = int(config.get("robot_radius", 2))

    samples = _load_preview_samples(
        dataset_dir,
        list(manifest["shards"]),
        args.preview_count,
    )
    if not samples:
        raise RuntimeError(f"no samples found in {dataset_dir}")

    tiles = [
        render_sample_robots(sample, robot_radius=robot_radius)
        for sample in samples
    ]
    output = (args.output or (dataset_dir / "check_robots.png")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_collage(tiles, output)

    print(
        f"checked {len(samples)} maps "
        f"(preview_count={args.preview_count}) -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

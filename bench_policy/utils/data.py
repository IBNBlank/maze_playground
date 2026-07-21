#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

"""Dataset loader aligned with ``PolicyBase`` batch layout.

Train / update batch (per sample, then collated by DataLoader):
  map   : (map_size, map_size)
  state : (obs_horizon, state_dim)
  action: (pred_horizon, action_dim)

Infer obs (built in closed-loop eval):
  map   : (map_size, map_size)
  state : (obs_horizon, state_dim)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.common import ACTION_DIM, STATE_DIM, load_dataset_meta


class MazeWindowDataset(Dataset):
    """Flattened (map, route, t) windows over genplan shards."""

    def __init__(
        self,
        dataset_dir: Path | str,
        obs_horizon: int = 1,
        pred_horizon: int = 8,
        sample_stride: int = 1,
        max_samples: int = 0,
        seed: int = 0,
    ):
        self.dataset_dir = Path(dataset_dir)
        meta = load_dataset_meta(self.dataset_dir)
        self.manifest = meta["manifest"]
        self.config = meta["config"]
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.sample_stride = max(1, int(sample_stride))
        self.map_size = int(self.config["size"])
        self.action_horizon = int(self.config["action_horizon"])
        self.max_abs_delta = float(self.config["max_abs_delta"])
        self.robot_radius = int(self.config.get("robot_radius", 5))
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM

        if self.pred_horizon > self.action_horizon:
            raise ValueError(
                f"pred_horizon={self.pred_horizon} > "
                f"action_horizon={self.action_horizon}")

        self._shards: list[dict[str, np.ndarray]] = []
        self._index: list[tuple[int, int, int, int]] = []
        # (shard_i, map_i, route_i, t)

        for shard_i, shard_name in enumerate(self.manifest["shards"]):
            path = self.dataset_dir / shard_name
            data = np.load(path)
            shard = {k: data[k] for k in data.files}
            data.close()
            self._shards.append(shard)

            n_maps = int(shard["planning_maps"].shape[0])
            n_routes = int(shard["action_chunks"].shape[1])
            horizon = int(shard["action_chunks"].shape[2])
            last_t = horizon - self.pred_horizon
            for map_i in range(n_maps):
                for route_i in range(n_routes):
                    for t in range(0, last_t + 1, self.sample_stride):
                        self._index.append((shard_i, map_i, route_i, t))

        if max_samples and max_samples > 0 and max_samples < len(self._index):
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(self._index), size=max_samples, replace=False)
            pick.sort()
            self._index = [self._index[i] for i in pick.tolist()]

        self.num_maps = int(self.manifest.get("num_maps", 0))
        self.num_windows = len(self._index)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shard_i, map_i, route_i, t = self._index[idx]
        shard = self._shards[shard_i]

        planning = shard["planning_maps"][map_i].astype(np.float32)
        waypoints = shard["waypoints_xy"][map_i, route_i]  # (H+1, 2)
        actions = shard["action_chunks"][map_i, route_i]  # (H, 2)
        goal_xy = waypoints[-1].astype(np.float32)

        states = []
        for k in range(self.obs_horizon):
            tt = t - (self.obs_horizon - 1 - k)
            tt = max(0, tt)
            xy = waypoints[tt].astype(np.float32)
            states.append(
                np.array([xy[0], xy[1], goal_xy[0], goal_xy[1]],
                         dtype=np.float32))
        state = np.stack(states, axis=0)
        action = actions[t:t + self.pred_horizon].astype(np.float32)

        # Matches PolicyBase.update_batch layout (before batching):
        #   map:    (map_size, map_size)
        #   state:  (obs_horizon, state_dim)
        #   action: (pred_horizon, action_dim)
        return {
            "map": torch.from_numpy(planning),
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
        }

    def iter_eval_episodes(
        self,
        max_episodes: int = 0,
        seed: int = 0,
    ) -> list[dict]:
        """One episode per (map, route); optionally subsample."""
        episodes: list[dict] = []
        for shard_i, shard in enumerate(self._shards):
            n_maps = int(shard["planning_maps"].shape[0])
            n_routes = int(shard["action_chunks"].shape[1])
            for map_i in range(n_maps):
                for route_i in range(n_routes):
                    episodes.append({
                        "planning_map": shard["planning_maps"][map_i],
                        "start_rc": shard["starts_rc"][map_i],
                        "goal_rc": shard["goals_rc"][map_i],
                        "waypoints_xy": shard["waypoints_xy"][map_i, route_i],
                        "action_chunks": shard["action_chunks"][map_i, route_i],
                        "shard_i": shard_i,
                        "map_i": map_i,
                        "route_i": route_i,
                    })
        if max_episodes and max_episodes > 0 and max_episodes < len(episodes):
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(episodes), size=max_episodes, replace=False)
            pick.sort()
            episodes = [episodes[i] for i in pick.tolist()]
        return episodes


def make_dataloader(
    dataset: MazeWindowDataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _worker_init(worker_id: int):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=generator,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def list_dataset_names(
    repo_dir: Path | str,
    names: Optional[Sequence[str]] = None,
) -> list[str]:
    root = Path(repo_dir) / "dataset"
    if names:
        return list(names)
    found = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "manifest.json").is_file())
    return found

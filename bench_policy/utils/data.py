#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################
"""Sharded maze dataset.

Usage:
  ds.set_epoch(epoch_id, batch_size=64)   # load all shards, order by epoch idx
  while (batch := ds.get_batch()) is not None:  # last batch may be shorter
      ...

Batch fields: map (H,W), state (1,4) in [0,1], action (T,2) in [-1,1].
state_dim: int = 4: state dimension: [x, y, goal_x, goal_y] in normalized coords
action_dim: int = 2: action dimension: [dx, dy] in pixels
"""

import json
import torch
import numpy as np
from collections import OrderedDict
from pathlib import Path


class MazeWindowDataset:

    def __init__(self, dataset_dir: Path | str, cache_shards: int = 2):
        self.dataset_dir = Path(dataset_dir).resolve()
        with open(self.dataset_dir / "dataset.json", encoding="utf-8") as f:
            summary = json.load(f)
        cfg_path = self.dataset_dir / "config.json"
        config = (json.loads(cfg_path.read_text(
            encoding="utf-8")) if cfg_path.is_file() else {})

        self.pred_horizon = int(summary["action_horizon"])
        self.state_dim = int(summary["state_dim"])
        self.action_dim = int(summary["action_dim"])

        self.shards = list(summary["shards"])
        self.num_samples = int(summary["num_samples"])
        self.num_idx_perms = int(summary.get("num_idx_perms", 0))
        self.shard_size = int(
            summary.get("shard_size", self.shards[0]["num_samples"]))
        self.max_abs_delta = float(config.get("max_abs_delta", 5.0))
        self._state_scale = float(summary["map_shape"][0] - 1)

        # small LRU for episode_at / incidental shard reads
        self._cache_shards = max(1, int(cache_shards))
        self._cache: OrderedDict[int, dict] = OrderedDict()

        # filled by set_epoch
        self._epoch: dict[str, np.ndarray] | None = None
        self._batch_size = 0
        self._batch_i = 0
        self._num_batches = 0

    def __len__(self) -> int:
        return self.num_samples

    @property
    def num_batches(self) -> int:
        return self._num_batches

    def set_epoch(self, epoch_id: int, batch_size: int):
        """Load all shards, reorder by epoch idx, reset batch cursor."""
        indices = np.load(self.dataset_dir / "idx" /
                          f"epoch_{int(epoch_id):03d}.npy")
        if len(indices) != self.num_samples:
            raise ValueError(
                f"epoch_{int(epoch_id):03d}.npy length {len(indices)} != "
                f"num_samples {self.num_samples}")

        order = np.asarray(indices, dtype=np.int64)

        # load every shard once, concatenate, then permute
        maps_l, states_l, actions_l = [], [], []
        for sid in range(len(self.shards)):
            sh = self._load_shard(sid)
            maps_l.append(sh["map"])
            states_l.append(sh["state"])
            actions_l.append(sh["action"])
        maps = np.concatenate(maps_l, axis=0)[order]
        states = np.concatenate(states_l, axis=0)[order]
        actions = np.concatenate(actions_l, axis=0)[order]
        if len(maps) != self.num_samples:
            raise ValueError(f"concatenated samples {len(maps)} != "
                             f"num_samples {self.num_samples}")

        self._epoch = {"map": maps, "state": states, "action": actions}
        self._batch_size = int(batch_size)
        self._batch_i = 0
        self._num_batches = ((self.num_samples + self._batch_size - 1) //
                             self._batch_size)

    def get_batch(self) -> dict[str, torch.Tensor] | None:
        """Next training batch from the epoch cache, or None when exhausted."""
        if self._epoch is None or self._batch_i >= self._num_batches:
            return None
        s = self._batch_i * self._batch_size
        e = min(s + self._batch_size, self.num_samples)
        self._batch_i += 1
        ep = self._epoch
        return {
            "map": torch.from_numpy(ep["map"][s:e]),
            "state": torch.from_numpy(ep["state"][s:e, None, :]),
            "action": torch.from_numpy(ep["action"][s:e]),
        }

    def _load_shard(self, shard_idx: int) -> dict:
        with np.load(self.dataset_dir / self.shards[shard_idx]["path"]) as z:
            rc = np.asarray(z["state"], dtype=np.float32)
            state = np.stack(
                [rc[:, 1], rc[:, 0], rc[:, 3], rc[:, 2]],
                axis=1,
            ) / self._state_scale
            return {
                "map":
                np.asarray(z["map"], dtype=np.float32),
                "state":
                state.astype(np.float32),
                "state_rc":
                rc,
                "action":
                np.asarray(z["action_chunk"], dtype=np.float32) /
                self.max_abs_delta,
            }

    def _ensure_shard(self, shard_idx: int) -> dict:
        if shard_idx in self._cache:
            self._cache.move_to_end(shard_idx)
            return self._cache[shard_idx]
        arrays = self._load_shard(shard_idx)
        self._cache[shard_idx] = arrays
        self._cache.move_to_end(shard_idx)
        while len(self._cache) > self._cache_shards:
            self._cache.popitem(last=False)
        return arrays

    def episode_at(self, idx: int) -> dict:
        sid = min(int(idx) // self.shard_size, len(self.shards) - 1)
        local = int(idx) - sid * self.shard_size
        c = self._ensure_shard(sid)
        rc = c["state_rc"][local]
        return {
            "planning_map": c["map"][local],
            "start_rc": rc[:2],
            "goal_rc": rc[2:],
        }

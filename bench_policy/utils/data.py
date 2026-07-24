#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################
"""Sharded maze dataset.

Usage:
  ds.set_epoch(epoch_id, batch_size=64)   # load epoch index only
  while (batch := ds.get_batch()) is not None:  # last batch may be shorter
      ...

Batch fields: map (H,W), state (1, state_dim) in [0,1], action (T,2) in [-1,1].
state_dim=4: [x, y, goal_x, goal_y]; with ``use_class`` appends class -> 5.
action_dim: int = 2: action dimension: [dx, dy] in pixels

Training never materializes the full float32 epoch. Shards stay in an LRU
(maps as uint8); each batch gathers rows by global index and casts maps
to float32 only for the batch tensor.

``prefetch`` > 0 starts a background thread that assembles the next batch while
the train step runs (overlaps CPU gather with GPU compute). Batches are
``pin_memory``'d so ``H2dTrainPipeline`` can async-copy the next batch during
the previous step's compute.
"""

import json
import queue
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

DEFAULT_PREFETCH = 1


def _as_torch(array: np.ndarray, pin_memory: bool) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if pin_memory:
        tensor = tensor.pin_memory()
    return tensor


class MazeWindowDataset:

    def __init__(
        self,
        dataset_dir: Path | str,
        cache_shards: int | None = None,
        use_class: bool = False,
        prefetch: int = DEFAULT_PREFETCH,
        pin_memory: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir).resolve()
        with open(self.dataset_dir / "dataset.json", encoding="utf-8") as f:
            summary = json.load(f)
        cfg_path = self.dataset_dir / "config.json"
        config = (json.loads(cfg_path.read_text(
            encoding="utf-8")) if cfg_path.is_file() else {})

        self.pred_horizon = int(summary["action_horizon"])
        self.use_class = bool(use_class)
        self.state_dim = int(summary["state_dim"]) + (1 if self.use_class else 0)
        self.action_dim = int(summary["action_dim"])
        self.map_shape = tuple(int(x) for x in summary["map_shape"])

        self.shards = list(summary["shards"])
        self.num_samples = int(summary["num_samples"])
        self.num_idx_perms = int(summary.get("num_idx_perms", 0))
        self.shard_size = int(
            summary.get("shard_size", self.shards[0]["num_samples"]))
        self.max_abs_delta = float(config.get("max_abs_delta", 5.0))
        self.robot_radius = int(config.get("robot_radius", 5))
        self._state_scale = float(summary["map_shape"][0] - 1)

        # Default: cache every shard (uint8 maps). Under a full shuffle each
        # batch touches most shards; a tiny LRU would thrash compressed NPZ.
        if cache_shards is None:
            self._cache_shards = max(1, len(self.shards))
        else:
            self._cache_shards = max(1, int(cache_shards))
        self.prefetch = max(0, int(prefetch))
        self.pin_memory = bool(pin_memory)
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cache_lock = threading.Lock()

        # filled by set_epoch — only the index order, not sample arrays
        self._order: np.ndarray | None = None
        self._batch_size = 0
        self._batch_i = 0
        self._num_batches = 0

        self._prefetch_q: queue.Queue | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_stop = threading.Event()
        self._prefetch_exc: BaseException | None = None

    def __len__(self) -> int:
        return self.num_samples

    @property
    def num_batches(self) -> int:
        return self._num_batches

    def set_epoch(self, epoch_id: int, batch_size: int):
        """Load epoch index permutation and reset batch cursor.

        Does not load sample arrays; shards are fetched on demand in get_batch.
        """
        self._stop_prefetch()
        indices = np.load(self.dataset_dir / "idx" /
                          f"epoch_{int(epoch_id):03d}.npy")
        if len(indices) != self.num_samples:
            raise ValueError(
                f"epoch_{int(epoch_id):03d}.npy length {len(indices)} != "
                f"num_samples {self.num_samples}")

        self._order = np.asarray(indices, dtype=np.int64)
        self._batch_size = int(batch_size)
        self._batch_i = 0
        self._num_batches = ((self.num_samples + self._batch_size - 1) //
                             self._batch_size)
        self._start_prefetch()

    def get_batch(self) -> dict[str, torch.Tensor] | None:
        """Next training batch gathered from cached shards, or None when done."""
        if self.prefetch <= 0 or self._prefetch_q is None:
            return self._assemble_batch()
        if self._prefetch_exc is not None:
            raise RuntimeError("prefetch failed") from self._prefetch_exc
        thread = self._prefetch_thread
        if thread is not None and not thread.is_alive():
            try:
                batch = self._prefetch_q.get_nowait()
            except queue.Empty:
                return None
        else:
            batch = self._prefetch_q.get()
        if self._prefetch_exc is not None:
            raise RuntimeError("prefetch failed") from self._prefetch_exc
        return batch

    def _assemble_batch(self) -> dict[str, torch.Tensor] | None:
        if self._order is None or self._batch_i >= self._num_batches:
            return None
        s = self._batch_i * self._batch_size
        e = min(s + self._batch_size, self.num_samples)
        self._batch_i += 1
        idxs = self._order[s:e]
        n = int(idxs.shape[0])
        h, w = self.map_shape

        maps = np.empty((n, h, w), dtype=np.uint8)
        states = np.empty((n, self.state_dim), dtype=np.float32)
        actions = np.empty((n, self.pred_horizon, self.action_dim),
                           dtype=np.float32)

        # Group by shard so each NPZ is touched at most once per batch.
        by_shard: dict[int, list[tuple[int, int]]] = {}
        n_shards = len(self.shards)
        for pos, gidx in enumerate(idxs.tolist()):
            sid = min(int(gidx) // self.shard_size, n_shards - 1)
            local = int(gidx) - sid * self.shard_size
            by_shard.setdefault(sid, []).append((pos, local))

        for sid, pairs in by_shard.items():
            sh = self._ensure_shard(sid)
            positions = np.fromiter((p for p, _ in pairs),
                                    dtype=np.int64,
                                    count=len(pairs))
            locals_ = np.fromiter((loc for _, loc in pairs),
                                  dtype=np.int64,
                                  count=len(pairs))
            maps[positions] = sh["map"][locals_]
            states[positions] = sh["state"][locals_]
            actions[positions] = sh["action"][locals_]

        pin = self.pin_memory
        return {
            "map": _as_torch(maps.astype(np.float32), pin),
            "state": _as_torch(states[:, None, :], pin),
            "action": _as_torch(actions, pin),
        }

    def _stop_prefetch(self):
        thread = self._prefetch_thread
        if thread is None:
            return
        self._prefetch_stop.set()
        q = self._prefetch_q
        while thread.is_alive():
            if q is not None:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            thread.join(timeout=0.05)
        self._prefetch_thread = None
        self._prefetch_q = None
        self._prefetch_stop.clear()
        self._prefetch_exc = None

    def _prefetch_loop(self):
        q = self._prefetch_q
        assert q is not None
        try:
            while not self._prefetch_stop.is_set():
                batch = self._assemble_batch()
                if self._prefetch_stop.is_set():
                    break
                while not self._prefetch_stop.is_set():
                    try:
                        q.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                else:
                    break
                if batch is None:
                    break
        except BaseException as exc:
            self._prefetch_exc = exc
            try:
                q.put(None, timeout=1.0)
            except Exception:
                pass

    def _start_prefetch(self):
        if self.prefetch <= 0:
            return
        self._prefetch_stop.clear()
        self._prefetch_exc = None
        self._prefetch_q = queue.Queue(maxsize=self.prefetch)
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            name="MazeWindowDataset-prefetch",
            daemon=True,
        )
        self._prefetch_thread.start()

    def _load_shard(self, shard_idx: int) -> dict:
        with np.load(self.dataset_dir / self.shards[shard_idx]["path"]) as z:
            rc = np.asarray(z["state"], dtype=np.float32)
            state = np.stack(
                [rc[:, 1], rc[:, 0], rc[:, 3], rc[:, 2]],
                axis=1,
            ) / self._state_scale
            if self.use_class:
                state = np.concatenate(
                    [state, np.asarray(z["class"], dtype=np.float32).reshape(-1, 1)],
                    axis=1,
                )
            return {
                # Keep occupancy as uint8 in RAM; cast only when building batches.
                "map":
                np.asarray(z["map"], dtype=np.uint8),
                "state":
                state.astype(np.float32),
                "state_rc":
                rc,
                "action":
                np.asarray(z["action_chunk"], dtype=np.float32) /
                self.max_abs_delta,
            }

    def _ensure_shard(self, shard_idx: int) -> dict:
        with self._cache_lock:
            if shard_idx in self._cache:
                self._cache.move_to_end(shard_idx)
                return self._cache[shard_idx]
        arrays = self._load_shard(shard_idx)
        with self._cache_lock:
            if shard_idx in self._cache:
                self._cache.move_to_end(shard_idx)
                return self._cache[shard_idx]
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
        ep = {
            "planning_map": c["map"][local],
            "start_rc": rc[:2],
            "goal_rc": rc[2:],
        }
        if self.use_class:
            ep["class"] = float(c["state"][local, -1])
        return ep

    def close(self):
        self._stop_prefetch()
        with self._cache_lock:
            self._cache.clear()

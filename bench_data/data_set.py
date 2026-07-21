#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

"""Flatten multimodal demons into training shards.

Each expert route becomes one sample:
  map          : (H, W) uint8 occupancy
  state        : (4,) float32 = [start_r, start_c, goal_r, goal_c]
  action_chunk : (72, 2) float32 pixel-space (dx, dy)

One or more demons/{id} trees may be merged into a single named dataset.
Shards of SHARD_SIZE samples are written under
  {out_dir}/dataset/data_XXXXXX.npz
plus dataset.json and NUM_IDX_PERMS shuffled full-index permutations as
  {out_dir}/idx/epoch_XXX.npy
so training can pick epochs by seed without DataLoader shuffle state.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import tyro
from tqdm import tqdm

from utils.arg import SetArgs

SHARD_SIZE = 2048
NUM_IDX_PERMS = 300
IDX_PERM_SEED = 0


def _resolve_out_dir(args: SetArgs) -> Path:
    if args.out_dir is not None:
        return args.out_dir.resolve()
    repo = Path(__file__).resolve().parent.parent
    return (repo / "datasets" / args.dataset_name).resolve()


def _load_manifest(demons_dir: Path) -> dict:
    path = demons_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(demons_dir: Path, manifest: dict) -> dict:
    path = demons_dir / manifest.get("config", "config.json")
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _clear_output(out_dir: Path, data_dir: Path, idx_dir: Path) -> None:
    """Remove previous shards / epoch idx / summary for a clean re-run."""
    if data_dir.is_dir():
        for name in os.listdir(data_dir):
            if name.startswith("data_") and name.endswith(".npz"):
                (data_dir / name).unlink()
    if idx_dir.is_dir():
        for name in os.listdir(idx_dir):
            if name.startswith("epoch_") and name.endswith(".npy"):
                (idx_dir / name).unlink()
    summary_path = out_dir / "dataset.json"
    if summary_path.is_file():
        summary_path.unlink()


def _flush_shard(
    data_dir: Path,
    shard_idx: int,
    buf: Dict[str, List[np.ndarray]],
) -> dict:
    n = len(buf["map"])
    assert n > 0
    name = f"data_{shard_idx:06d}.npz"
    path = data_dir / name
    np.savez_compressed(
        path,
        map=np.stack(buf["map"], axis=0),
        state=np.stack(buf["state"], axis=0),
        action_chunk=np.stack(buf["action_chunk"], axis=0),
    )
    return {"path": name, "num_samples": n}


def _clear_buf(buf: Dict[str, List[np.ndarray]]) -> None:
    for k in buf:
        buf[k].clear()


def _write_idx_perms(
    idx_dir: Path,
    num_samples: int,
    num_perms: int = NUM_IDX_PERMS,
    seed: int = IDX_PERM_SEED,
) -> List[str]:
    """Write num_perms shuffled full permutations of [0, num_samples)."""
    if num_samples <= 0:
        return []
    if num_perms < 1:
        raise ValueError("num_perms must be >= 1")

    idx_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    base = np.arange(num_samples, dtype=np.int64)
    for i in tqdm(range(num_perms), desc="epoch idx"):
        rng = np.random.default_rng(seed + i)
        perm = rng.permutation(base)
        name = f"epoch_{i:03d}.npy"
        np.save(idx_dir / name, perm)
        paths.append(os.path.join("idx", name))
    return paths


def _iter_route_samples(
    demons_dir: Path,
    shard_names: List[str],
    action_horizon: int,
    demons_id: str,
):
    """Yield (map, state, action_chunk) for every route in every map."""
    map_id = 0
    for shard_name in shard_names:
        shard_path = demons_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"shard missing: {shard_path}")
        with np.load(shard_path) as shard:
            maps = shard["maps"]
            starts = shard["starts_rc"]
            goals = shard["goals_rc"]
            actions = shard["action_chunks"]
            batch, num_routes = int(actions.shape[0]), int(actions.shape[1])
            horizon = int(actions.shape[2])
            if action_horizon > 0 and horizon != action_horizon:
                raise ValueError(
                    f"{demons_id}/{shard_name}: action horizon {horizon} != "
                    f"expected {action_horizon}")
            for i in range(batch):
                state = np.concatenate(
                    [
                        starts[i].astype(np.float32),
                        goals[i].astype(np.float32),
                    ],
                    axis=0,
                )
                occ = np.asarray(maps[i], dtype=np.uint8)
                for r in range(num_routes):
                    yield {
                        "map": occ,
                        "state": state,
                        "action_chunk": np.asarray(actions[i, r],
                                                   dtype=np.float32),
                        "map_id": map_id,
                        "route_id": r,
                        "demons_id": demons_id,
                        "routes_per_map": num_routes,
                    }
                map_id += 1


def gen_dataset(args: SetArgs) -> dict:
    if args.shard_size < 1:
        raise ValueError("--shard-size must be >= 1")
    if args.num_idx_perms < 1:
        raise ValueError("--num-idx-perms must be >= 1")
    if not args.demons_ids:
        raise ValueError("--demons-ids must be non-empty")
    if not args.dataset_name:
        raise ValueError("--dataset-name must be non-empty")

    demons_root = args.demons_root.resolve()
    out_dir = _resolve_out_dir(args)
    data_dir = out_dir / "dataset"
    idx_dir = out_dir / "idx"
    dataset_name = args.dataset_name

    sources: List[dict] = []
    for demons_id in args.demons_ids:
        demons_dir = demons_root / demons_id
        if not demons_dir.is_dir():
            print(f"[dataset] skip {demons_id}: missing dir {demons_dir}")
            continue
        manifest = _load_manifest(demons_dir)
        config = _load_config(demons_dir, manifest)
        shard_names = list(manifest["shards"])
        if not shard_names:
            print(f"[dataset] skip {demons_id}: empty shards list")
            continue
        sources.append({
            "demons_id": demons_id,
            "demons_dir": demons_dir,
            "manifest": manifest,
            "config": config,
            "shard_names": shard_names,
        })

    if not sources:
        raise RuntimeError(
            f"no usable demons under {demons_root} for ids={list(args.demons_ids)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)
    _clear_output(out_dir, data_dir, idx_dir)

    # Prefer the first source config as a representative copy.
    if sources[0]["config"]:
        (out_dir / "config.json").write_text(
            json.dumps(sources[0]["config"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    buf: Dict[str, List[np.ndarray]] = {
        "map": [],
        "state": [],
        "action_chunk": [],
    }
    shards: List[dict] = []
    shard_idx = 0
    total_samples = 0
    total_maps = 0
    per_source_counts: Dict[str, int] = {}
    per_source_maps: Dict[str, int] = {}
    map_shape = None
    action_dim = None
    action_encoding = sources[0]["manifest"].get("action_encoding", {})

    for src in sources:
        demons_id = src["demons_id"]
        per_source_counts[demons_id] = 0
        per_source_maps[demons_id] = 0
        samples = _iter_route_samples(
            src["demons_dir"],
            src["shard_names"],
            action_horizon=args.action_horizon,
            demons_id=demons_id,
        )
        for sample in tqdm(samples, desc=f"dataset {dataset_name}/{demons_id}"):
            if map_shape is None:
                map_shape = tuple(int(x) for x in sample["map"].shape)
                action_dim = int(sample["action_chunk"].shape[-1])
            if sample["route_id"] == 0:
                total_maps += 1
                per_source_maps[demons_id] += 1

            buf["map"].append(sample["map"])
            buf["state"].append(sample["state"])
            buf["action_chunk"].append(sample["action_chunk"])
            total_samples += 1
            per_source_counts[demons_id] += 1

            if len(buf["map"]) >= args.shard_size:
                shards.append(_flush_shard(data_dir, shard_idx, buf))
                _clear_buf(buf)
                shard_idx += 1

    if buf["map"]:
        shards.append(_flush_shard(data_dir, shard_idx, buf))
        _clear_buf(buf)

    if total_samples == 0:
        raise RuntimeError(f"no samples produced for dataset {dataset_name}")

    idx_paths = _write_idx_perms(
        idx_dir,
        num_samples=total_samples,
        num_perms=args.num_idx_perms,
        seed=args.idx_perm_seed,
    )

    for shard in shards:
        shard["path"] = os.path.join("dataset", shard["path"])

    used_ids = [s["demons_id"] for s in sources]
    action_horizon = int(
        action_encoding.get(
            "action_horizon",
            sources[0]["config"].get("action_horizon", args.action_horizon),
        ))
    summary = {
        "format": "genplan-flat-route-v1",
        "dataset_name": dataset_name,
        "demons_ids": used_ids,
        "demons_root": str(demons_root),
        "source_format": sources[0]["manifest"].get("format"),
        "num_maps": total_maps,
        "num_samples": total_samples,
        "per_source_samples": per_source_counts,
        "per_source_maps": per_source_maps,
        "shard_size": args.shard_size,
        "num_shards": len(shards),
        "map_shape": list(map_shape) if map_shape else None,
        "state_dim": 4,
        "state_layout": ["start_r", "start_c", "goal_r", "goal_c"],
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "shards": shards,
        "num_idx_perms": len(idx_paths),
        "idx_perm_seed": args.idx_perm_seed,
        "idx_perms": idx_paths,
        "action_encoding": action_encoding,
        "output_dir": str(out_dir),
        "data_dir": str(data_dir),
        "idx_dir": str(idx_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = out_dir / "dataset.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[dataset] wrote {total_samples} samples from {len(used_ids)} source(s) "
        f"({total_maps} maps) in {len(shards)} shards "
        f"(shard_size={args.shard_size}) -> {data_dir}",
        flush=True,
    )
    print(
        f"[dataset] wrote {len(idx_paths)} epoch idx perms "
        f"(seed={args.idx_perm_seed}) -> {idx_dir}",
        flush=True,
    )
    print(f"[dataset] index: {summary_path}", flush=True)
    return summary


def main() -> None:
    args = tyro.cli(SetArgs)
    gen_dataset(args)


if __name__ == "__main__":
    main()

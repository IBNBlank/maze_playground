#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

import gc, json, os, random, shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional, Sequence
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch


@dataclass
class Metrics:
    cur_success_rate: float = float("-inf")
    cur_success_average_steps: float = float("inf")
    best_success_rate: float = float("-inf")
    best_success_average_steps: float = float("inf")

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Metrics":
        data = data or {}
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


def device_init(
    seed: int,
    torch_deterministic: bool = True,
    cuda: bool = True,
) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = not torch_deterministic
    return torch.device(
        "cuda" if torch.cuda.is_available() and cuda else "cpu")


def load_dataset_meta(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    config_path = dataset_dir / "config.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return {"manifest": manifest, "config": config, "dataset_dir": dataset_dir}


def _to_cpu_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_tree(v) for v in obj)
    return obj


def copy_best_artifacts(run_name: str, ckpt_name: str):
    src = f"runs/{run_name}/{ckpt_name}"
    if not src.endswith(".pt"):
        src += ".pt"
    dst = f"runs/{run_name}/best_success_ckpt.pt"
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)
    with open(f"runs/{run_name}/best_success.json", "w",
              encoding="utf-8") as f:
        json.dump({"ckpt_name": os.path.basename(dst)}, f, indent=2)


def peek_latest_iteration(run_name: str) -> Optional[int]:
    path = f"runs/{run_name}/latest.json"
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)
    return int(record.get("iteration", 0))


# ---------------------------------------------------------------------------
# TensorBoard init
# ---------------------------------------------------------------------------


def tensorboard_init(
    run_name: str,
    mode: str = "train",
    hparams: Optional[dict] = None,
):
    """Create a TensorBoard writer under ``runs/{run_name}/``.

    Args:
      run_name: experiment directory name.
      mode: ``\"train\"`` writes to ``runs/{run_name}``;
            ``\"eval\"`` writes to ``runs/{run_name}/eval`` so eval logs
            do not overwrite training events.
      hparams: optional dict logged once as a markdown hyperparameter table.
    """
    mode = mode.lower()
    if mode not in ("train", "eval"):
        raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")

    if mode == "train":
        log_dir = f"runs/{run_name}"
    else:
        log_dir = f"runs/{run_name}/eval"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    if hparams:
        table = "|param|value|\n|-|-|\n" + "\n".join(
            f"|{k}|{v}|" for k, v in sorted(hparams.items()))
        writer.add_text("hyperparameters", table)
    print(f"[tensorboard:{mode}] logging to {log_dir}")
    return writer


# ---------------------------------------------------------------------------
# 1) save  2) load
# ---------------------------------------------------------------------------


def save(
    policy,
    message: dict,
    *,
    run_name: str,
    algo: str,
    metrics: Metrics,
    global_step: int = 0,
    extra: Optional[dict] = None,
) -> str:
    """Save policy weights and related training state under ``runs/{run_name}/``.

    message:
      {
        "epoch": int,
        "success_rate": float,
        "success_average_steps": float,
        "is_final": bool,  # optional
      }
    """
    model = getattr(policy, "model", None)
    optimizer = getattr(policy, "optimizer", None)
    if model is None:
        raise RuntimeError("policy.model is required for save()")

    epoch = int(message["epoch"])
    success_rate = float(message["success_rate"])
    success_average_steps = float(message["success_average_steps"])
    is_final = bool(message.get("is_final", False))

    metrics.cur_success_rate = success_rate
    metrics.cur_success_average_steps = success_average_steps
    is_best = success_rate >= metrics.best_success_rate
    if is_best:
        metrics.best_success_rate = success_rate
        metrics.best_success_average_steps = success_average_steps

    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    if is_final:
        iteration = -1
        ckpt_name = "final_ckpt.pt"
    else:
        iteration = epoch
        ckpt_name = f"ckpt_{epoch}.pt"
    ckpt_path = f"{run_dir}/{ckpt_name}"

    horizons = {
        "obs_horizon": int(policy.obs_horizon),
        "pred_horizon": int(policy.pred_horizon),
        "act_horizon": int(act_horizon),
    }
    metrics_dict = asdict(metrics)
    checkpoint = {
        "algo": algo,
        "policy": _to_cpu_tree(model.state_dict()),
        "iteration": iteration,
        "global_step": int(global_step),
        "metrics": metrics_dict,
        "horizons": horizons,
        "extra": {
            "map_size": int(policy.map_size),
            "state_dim": int(policy.state_dim),
            "action_dim": int(policy.action_dim),
            **(extra or {}),
        },
    }
    if optimizer is not None:
        checkpoint["optimizer"] = _to_cpu_tree(optimizer.state_dict())
    torch.save(checkpoint, ckpt_path)

    latest_pt = f"{run_dir}/latest.pt"
    if os.path.abspath(ckpt_path) != os.path.abspath(latest_pt):
        shutil.copy(ckpt_path, latest_pt)
    record = {
        "ckpt_name": os.path.basename(ckpt_path),
        "iteration": iteration,
        "global_step": int(global_step),
        "metrics": metrics_dict,
        "algo": algo,
        "horizons": horizons,
    }
    with open(f"{run_dir}/latest.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    if is_best:
        copy_best_artifacts(run_name, ckpt_name)

    del checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"ckpt saved to {ckpt_path}")
    return ckpt_path


def load(policy, path: str) -> tuple[int, Metrics, dict]:
    """Load policy weights and related training state from ``path``."""
    model = getattr(policy, "model", None)
    if model is None:
        raise RuntimeError("policy.model is required for load()")
    if not path.endswith(".pt"):
        path = path + ".pt"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")

    device = getattr(policy, "device", torch.device("cpu"))
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["policy"])

    optimizer = getattr(policy, "optimizer", None)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    metrics = Metrics.from_dict(ckpt.get("metrics"))
    iteration = int(ckpt.get("iteration", 0))
    print(f"ckpt loaded from {path} (iteration={iteration})")
    return iteration, metrics, ckpt


# ---------------------------------------------------------------------------
# Maze rollout helpers
# ---------------------------------------------------------------------------


def pixel_to_xy(pixel_xy: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(pixel_xy, dtype=np.float64) / float(size - 1)


def is_occupied(planning_map: np.ndarray, pixel_xy: np.ndarray) -> bool:
    """True if rounded pixel lands on an occupied planning cell."""
    h, w = planning_map.shape
    x, y = float(pixel_xy[0]), float(pixel_xy[1])
    c = int(np.rint(x))
    r = int(np.rint(y))
    if r < 0 or r >= h or c < 0 or c >= w:
        return True
    return bool(planning_map[r, c] > 0)


def goal_error(pixel_xy: np.ndarray, goal_pixel_xy: np.ndarray) -> float:
    """Pixel-space L2 distance to goal."""
    d = np.asarray(pixel_xy, dtype=np.float64) - np.asarray(goal_pixel_xy,
                                                            dtype=np.float64)
    return float(np.linalg.norm(d))


def rollout_episode(
    policy,
    planning_map: np.ndarray,
    start_rc: np.ndarray,
    goal_rc: np.ndarray,
    device: torch.device,
    obs_horizon: int,
    act_horizon: int,
    max_steps: int,
    goal_tol: float = 1.0,
    max_abs_delta: float = 5.0,
) -> dict:
    """Closed-loop pixel rollout of one episode via ``policy.infer_batch``."""
    size = int(planning_map.shape[0])
    start_xy = np.array(
        [float(start_rc[1]), float(start_rc[0])], dtype=np.float64)
    goal_xy = np.array(
        [float(goal_rc[1]), float(goal_rc[0])], dtype=np.float64)

    cur = start_xy.copy()
    hist: list[np.ndarray] = []
    path = [pixel_to_xy(cur, size)]
    collided = False
    reached = False
    steps = 0
    final_error = goal_error(cur, goal_xy)

    map_t = torch.from_numpy(planning_map.astype(np.float32)[None]).to(device)

    while steps < max_steps:
        final_error = goal_error(cur, goal_xy)
        if final_error < goal_tol:
            reached = True
            break

        cur_xy_n = pixel_to_xy(cur, size)
        goal_xy_n = pixel_to_xy(goal_xy, size)
        state_vec = np.array(
            [cur_xy_n[0], cur_xy_n[1], goal_xy_n[0], goal_xy_n[1]],
            dtype=np.float32,
        )
        hist.append(state_vec)
        while len(hist) < obs_horizon:
            hist.insert(0, hist[0].copy())
        state = np.stack(hist[-obs_horizon:], axis=0)

        obs = {
            "map": map_t,
            "state": torch.from_numpy(state[None]).to(device),
        }
        with torch.no_grad():
            action_seq = policy.infer_batch(obs)
        actions = action_seq[0].detach().cpu().numpy()

        for i in range(min(act_horizon, actions.shape[0])):
            if steps >= max_steps:
                break
            dx = float(np.clip(actions[i, 0], -max_abs_delta, max_abs_delta))
            dy = float(np.clip(actions[i, 1], -max_abs_delta, max_abs_delta))
            nxt = cur + np.array([dx, dy], dtype=np.float64)
            nxt[0] = float(np.clip(nxt[0], 0.0, size - 1))
            nxt[1] = float(np.clip(nxt[1], 0.0, size - 1))
            steps += 1
            cur = nxt
            path.append(pixel_to_xy(cur, size))
            if is_occupied(planning_map, cur):
                collided = True
                final_error = goal_error(cur, goal_xy)
                return {
                    "collided": collided,
                    "reached_goal": False,
                    "steps": steps,
                    "final_error": final_error,
                    "path_xy": np.asarray(path, dtype=np.float32),
                }
            final_error = goal_error(cur, goal_xy)
            if final_error < goal_tol:
                reached = True
                return {
                    "collided": False,
                    "reached_goal": True,
                    "steps": steps,
                    "final_error": final_error,
                    "path_xy": np.asarray(path, dtype=np.float32),
                }

    final_error = goal_error(cur, goal_xy)
    reached = (not collided) and (final_error < goal_tol)
    return {
        "collided": collided,
        "reached_goal": reached,
        "steps": steps,
        "final_error": final_error,
        "path_xy": np.asarray(path, dtype=np.float32),
    }


def summarize_rollouts(results: Sequence[dict]) -> dict:
    collided = np.asarray([r["collided"] for r in results], dtype=np.float64)
    reached = np.asarray([r["reached_goal"] for r in results],
                         dtype=np.float64)
    steps = np.asarray([r["steps"] for r in results], dtype=np.float64)
    errors = np.asarray([r["final_error"] for r in results], dtype=np.float64)
    succ_steps = steps[reached > 0.5]
    mean_steps_success = (float(succ_steps.mean())
                          if len(succ_steps) else float("nan"))
    return {
        "num_episodes":
        len(results),
        "collision_rate":
        float(collided.mean()) if results else 0.0,
        "success_rate":
        float(reached.mean()) if results else 0.0,
        "mean_steps":
        float(steps.mean()) if results else 0.0,
        "mean_steps_success":
        mean_steps_success,
        "success_average_steps": (mean_steps_success if mean_steps_success
                                  == mean_steps_success else float("inf")),
        "mean_final_error":
        float(errors.mean()) if results else 0.0,
    }


# ---------------------------------------------------------------------------
# 3) eval common  4) train common
# ---------------------------------------------------------------------------
def evaluate(
    policy,
    episodes: Sequence[dict],
    *,
    device: torch.device,
    obs_horizon: int,
    act_horizon: int,
    max_steps: Optional[int],
    goal_tol: float = 1.0,
    max_abs_delta: float = 5.0,
) -> dict:
    """Closed-loop evaluation shared by ``train.py`` and ``eval.py``."""
    results = []
    for ep in episodes:
        ep_max_steps = max_steps
        if ep_max_steps is None:
            actions = ep.get("action_chunks")
            if actions is not None:
                ep_max_steps = int(np.asarray(actions).shape[0])
            else:
                ep_max_steps = 72
        results.append(
            rollout_episode(
                policy,
                planning_map=np.asarray(ep["planning_map"]),
                start_rc=np.asarray(ep["start_rc"]),
                goal_rc=np.asarray(ep["goal_rc"]),
                device=device,
                obs_horizon=obs_horizon,
                act_horizon=act_horizon,
                max_steps=int(ep_max_steps),
                goal_tol=goal_tol,
                max_abs_delta=max_abs_delta,
            ))
    return summarize_rollouts(results)


def train_epoch(
    policy,
    loader,
    *,
    global_step: int,
    log_freq: int = 0,
    writer=None,
    epoch_idx: int = 0,
    num_epochs: int = 0,
    progress_cls=None,
) -> tuple[float, int]:
    """One training epoch via ``policy.update_batch``.

    Returns ``(last_loss, global_step)``.
    """
    loss_log = defaultdict(float)
    steps_logged = 0
    last_loss = float("nan")

    iterable = loader
    if progress_cls is not None:
        iterable = progress_cls(
            loader,
            desc=f"batch ep{epoch_idx + 1}/{num_epochs}",
            leave=False,
            dynamic_ncols=True,
        )

    for batch in iterable:
        loss_val = float(policy.update_batch(batch))
        last_loss = loss_val
        loss_log["loss"] += loss_val
        steps_logged += 1
        global_step += 1
        if hasattr(iterable, "set_postfix"):
            iterable.set_postfix(loss=f"{loss_val:.4f}")

        if log_freq > 0 and global_step % log_freq == 0 and writer is not None:
            avg = loss_log["loss"] / max(steps_logged, 1)
            writer.add_scalar("train/loss", avg, global_step)
            loss_log.clear()
            steps_logged = 0

    if steps_logged > 0 and writer is not None:
        avg = loss_log["loss"] / steps_logged
        writer.add_scalar("train/loss", avg, global_step)
    return last_loss, global_step

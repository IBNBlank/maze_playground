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
from typing import Any, Optional, Sequence
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch

#--------------------------------#
# init
#--------------------------------#


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


def tensorboard_init(
    run_name: str,
    mode: str = "train",
    hparams: Optional[dict] = None,
):
    """Create a TensorBoard writer under ``runs/{run_name}/``."""
    mode = mode.lower()
    if mode not in ("train", "eval"):
        raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
    log_dir = (f"runs/{run_name}"
               if mode == "train" else f"runs/{run_name}/eval")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    if hparams:
        table = "|param|value|\n|-|-|\n" + "\n".join(
            f"|{k}|{v}|" for k, v in sorted(hparams.items()))
        writer.add_text("hyperparameters", table)
    print(f"[tensorboard:{mode}] logging to {log_dir}")
    return writer


#--------------------------------#
# train and eval
#--------------------------------#


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
    scale = float(size - 1)
    h, w = planning_map.shape
    cur = np.array([float(start_rc[1]), float(start_rc[0])], dtype=np.float64)
    goal = np.array([float(goal_rc[1]), float(goal_rc[0])], dtype=np.float64)
    hist: list[np.ndarray] = []
    steps = 0
    map_t = torch.from_numpy(planning_map.astype(np.float32)[None]).to(device)

    while steps < max_steps:
        if float(np.linalg.norm(cur - goal)) < goal_tol:
            return {"collided": False, "reached_goal": True, "steps": steps}

        state_vec = np.array(
            [cur[0] / scale, cur[1] / scale, goal[0] / scale, goal[1] / scale],
            dtype=np.float32,
        )
        hist.append(state_vec)
        while len(hist) < obs_horizon:
            hist.insert(0, hist[0].copy())
        obs = {
            "map":
            map_t,
            "state":
            torch.from_numpy(np.stack(hist[-obs_horizon:],
                                      axis=0)[None]).to(device),
        }
        with torch.no_grad():
            actions = policy.infer_batch(obs)[0].detach().cpu().numpy()

        for i in range(min(act_horizon, actions.shape[0])):
            if steps >= max_steps:
                break
            dx = float(np.clip(actions[i, 0], -1.0, 1.0)) * max_abs_delta
            dy = float(np.clip(actions[i, 1], -1.0, 1.0)) * max_abs_delta
            nxt = cur + np.array([dx, dy], dtype=np.float64)
            nxt[0] = float(np.clip(nxt[0], 0.0, size - 1))
            nxt[1] = float(np.clip(nxt[1], 0.0, size - 1))
            steps += 1
            cur = nxt

            c, r = int(np.rint(cur[0])), int(np.rint(cur[1]))
            if r < 0 or r >= h or c < 0 or c >= w or planning_map[r, c] > 0:
                return {
                    "collided": True,
                    "reached_goal": False,
                    "steps": steps
                }
            if float(np.linalg.norm(cur - goal)) < goal_tol:
                return {
                    "collided": False,
                    "reached_goal": True,
                    "steps": steps
                }

    return {
        "collided": False,
        "reached_goal": float(np.linalg.norm(cur - goal)) < goal_tol,
        "steps": steps,
    }


_RATE_KEYS = frozenset({"collision_rate", "success_rate"})


def format_eval_metric(key: str, value) -> str:
    """Human-readable eval metric; rates as percent, else plain float."""
    if isinstance(value, (int, np.integer)) and key == "num_episodes":
        return str(int(value))
    if not isinstance(value, (float, int, np.floating, np.integer)):
        return str(value)
    v = float(value)
    if key in _RATE_KEYS:
        return f"{v * 100:.2f}%"
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    return f"{v:.4f}"


def log_eval_summary(summary: dict, *, writer=None, global_step: int = 0):
    """Print eval summary; optionally write scalars (rates stay in [0, 1])."""
    for k, v in summary.items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            print(f"eval/{k}: {format_eval_metric(k, v)}")
            if writer is not None and isinstance(v, (float, np.floating)):
                writer.add_scalar(f"eval/{k}", float(v), global_step)
        else:
            print(f"eval/{k}: {v}")


def build_eval_episodes(
    dataset,
    max_episodes: int = 0,
    seed: int = 0,
) -> list[dict]:
    """Sample closed-loop eval episodes via ``dataset.episode_at``."""
    n = len(dataset)
    order = np.arange(n, dtype=np.int64)
    if 0 < max_episodes < n:
        rng = np.random.default_rng(seed)
        order = np.sort(rng.choice(n, size=int(max_episodes), replace=False))
    return [dataset.episode_at(int(i)) for i in order.tolist()]


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
            ep_max_steps = (int(np.asarray(actions).shape[0])
                            if actions is not None else 72)
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

    collided = np.asarray([r["collided"] for r in results], dtype=np.float64)
    reached = np.asarray([r["reached_goal"] for r in results],
                         dtype=np.float64)
    steps = np.asarray([r["steps"] for r in results], dtype=np.float64)
    succ_steps = steps[reached > 0.5]
    return {
        "num_episodes":
        len(results),
        "collision_rate":
        float(collided.mean()) if results else 0.0,
        "success_rate":
        float(reached.mean()) if results else 0.0,
        "success_average_steps":
        (float(succ_steps.mean()) if len(succ_steps) else float("inf")),
    }


def train_epoch(
    policy,
    dataset,
    *,
    epoch_id: int,
    batch_size: int,
    device,
    global_step: int,
    log_freq: int = 0,
    writer=None,
    epoch_idx: int = 0,
    num_epochs: int = 0,
    progress_cls=None,
) -> tuple[float, int]:
    """One training epoch via ``dataset.set_epoch`` / ``get_batch``."""
    dataset.set_epoch(int(epoch_id), batch_size=int(batch_size))

    loss_log = defaultdict(float)
    steps_logged = 0
    last_loss = float("nan")

    batch_pbar = None
    if progress_cls is not None:
        batch_pbar = progress_cls(
            total=dataset.num_batches,
            desc=(f"batch ep{epoch_idx + 1}/{num_epochs}"
                  f" idx={epoch_id}"),
            leave=False,
            dynamic_ncols=True,
        )

    while True:
        batch = dataset.get_batch()
        if batch is None:
            break
        batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        loss_val = float(policy.update_batch(batch))
        last_loss = loss_val
        loss_log["loss"] += loss_val
        steps_logged += 1
        global_step += 1
        if batch_pbar is not None:
            batch_pbar.update(1)
            batch_pbar.set_postfix(loss=f"{loss_val:.4f}")

        if log_freq > 0 and global_step % log_freq == 0 and writer is not None:
            avg = loss_log["loss"] / max(steps_logged, 1)
            writer.add_scalar("train/loss", avg, global_step)
            loss_log.clear()
            steps_logged = 0

    if batch_pbar is not None:
        batch_pbar.close()
    if steps_logged > 0 and writer is not None:
        avg = loss_log["loss"] / steps_logged
        writer.add_scalar("train/loss", avg, global_step)
    return last_loss, global_step


#--------------------------------#
# save and load
#--------------------------------#


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


def save(
    policy,
    message: dict,
    run_name: str,
    metrics: Metrics,
    global_step: int = 0,
) -> str:
    """Save resume-ready weights; sync ``latest`` / ``best_success`` artifacts.

    ``.pt`` keeps only what ``load`` needs:
      policy, optimizer?, metrics, iteration, global_step
    ``.json`` keeps:
      ckpt_name, iteration, global_step, metrics
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

    metrics_dict = asdict(metrics)
    checkpoint = {
        "policy": _to_cpu_tree(model.state_dict()),
        "iteration": iteration,
        "global_step": int(global_step),
        "metrics": metrics_dict,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = _to_cpu_tree(optimizer.state_dict())
    torch.save(checkpoint, ckpt_path)

    latest_pt = f"{run_dir}/latest.pt"
    latest_json = f"{run_dir}/latest.json"
    if os.path.abspath(ckpt_path) != os.path.abspath(latest_pt):
        shutil.copy(ckpt_path, latest_pt)
    record = {
        "ckpt_name": os.path.basename(ckpt_path),
        "iteration": iteration,
        "global_step": int(global_step),
        "metrics": metrics_dict,
    }
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    if is_best:
        shutil.copy(latest_pt, f"{run_dir}/best_success_ckpt.pt")
        shutil.copy(latest_json, f"{run_dir}/best_success.json")

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

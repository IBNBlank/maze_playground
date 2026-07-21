#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

import json, os, random, shutil
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


def log_eval_summary(
    summary: dict,
    *,
    writer=None,
    global_step: int = 0,
    best_rate: float | None = None,
    best_steps: float | None = None,
):
    """Print one compact eval line; optionally write TB scalars."""
    n = summary.get("num_episodes", "-")
    coll = float(summary["collision_rate"]) * 100
    succ = float(summary["success_rate"]) * 100
    steps = float(summary["success_average_steps"])
    steps_s = "inf" if steps != steps or steps == float(
        "inf") else f"{steps:.1f}"
    line = (f"[eval] collision={coll:.1f}%  success={succ:.1f}%  "
            f"succ_steps={steps_s}")
    if best_rate is not None:
        bs = best_steps if best_steps is not None else float("inf")
        bs_s = "inf" if bs != bs or bs == float("inf") else f"{bs:.1f}"
        line += f"  best={best_rate * 100:.1f}%/{bs_s}"
    print(line)
    if writer is not None:
        for k, v in summary.items():
            if isinstance(v, (float, np.floating)):
                writer.add_scalar(f"eval/{k}", float(v), global_step)


def build_eval_episodes(
    dataset,
    max_episodes: int = 0,
    seed: int = 0,
) -> list[dict]:
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
    max_steps: int,
    goal_tol: float = 1.0,
    max_abs_delta: float = 5.0,
) -> dict:
    """Closed-loop eval: infer action chunk, step until hit / goal / budget."""
    act_horizon = int(policy.pred_horizon)
    results = []
    for ep in episodes:
        planning_map = np.asarray(ep["planning_map"])
        size = int(planning_map.shape[0])
        scale = float(size - 1)
        h, w = planning_map.shape
        cur = np.array(
            [float(ep["start_rc"][1]),
             float(ep["start_rc"][0])],
            dtype=np.float64,
        )
        goal = np.array(
            [float(ep["goal_rc"][1]),
             float(ep["goal_rc"][0])],
            dtype=np.float64,
        )
        map_t = torch.from_numpy(planning_map.astype(
            np.float32)[None]).to(device)
        steps = 0
        collided = False
        reached = False

        while steps < max_steps and not reached and not collided:
            if float(np.linalg.norm(cur - goal)) < goal_tol:
                reached = True
                break
            state = np.array(
                [[
                    cur[0] / scale, cur[1] / scale, goal[0] / scale,
                    goal[1] / scale
                ]],
                dtype=np.float32,
            )
            with torch.no_grad():
                actions = policy.infer_batch({
                    "map":
                    map_t,
                    "state":
                    torch.from_numpy(state[None]).to(device),
                })[0].detach().cpu().numpy()

            for a in actions[:min(act_horizon, max_steps - steps)]:
                cur = np.clip(
                    cur +
                    np.clip(a.astype(np.float64), -1.0, 1.0) * max_abs_delta,
                    0.0,
                    size - 1,
                )
                steps += 1
                c, r = int(np.rint(cur[0])), int(np.rint(cur[1]))
                if r < 0 or r >= h or c < 0 or c >= w or planning_map[r,
                                                                      c] > 0:
                    collided = True
                    break
                if float(np.linalg.norm(cur - goal)) < goal_tol:
                    reached = True
                    break

        if not collided and not reached:
            reached = float(np.linalg.norm(cur - goal)) < goal_tol
        results.append((collided, reached, steps))

    collided = np.asarray([r[0] for r in results], dtype=np.float64)
    reached = np.asarray([r[1] for r in results], dtype=np.float64)
    steps = np.asarray([r[2] for r in results], dtype=np.float64)
    succ = steps[reached > 0.5]
    return {
        "num_episodes": len(results),
        "collision_rate": float(collided.mean()) if results else 0.0,
        "success_rate": float(reached.mean()) if results else 0.0,
        "success_average_steps":
        float(succ.mean()) if len(succ) else float("inf"),
    }


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
    *,
    run_name: str,
    metrics: Metrics,
    iteration: int,
    global_step: int,
    is_best: bool = False,
) -> str:
    """Write ``ckpt_*.pt`` / ``latest``; if ``is_best``, also snapshot best."""
    model = getattr(policy, "model", None)
    optimizer = getattr(policy, "optimizer", None)
    if model is None:
        raise RuntimeError("policy.model is required for save()")

    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    ckpt_name = ("final_ckpt.pt"
                 if int(iteration) < 0 else f"ckpt_{int(iteration)}.pt")
    ckpt_path = f"{run_dir}/{ckpt_name}"

    metrics_dict = asdict(metrics)
    checkpoint = {
        "policy": _to_cpu_tree(model.state_dict()),
        "iteration": int(iteration),
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
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt_name": ckpt_name,
                "iteration": int(iteration),
                "global_step": int(global_step),
                "metrics": metrics_dict,
            },
            f,
            indent=2,
        )

    if is_best:
        shutil.copy(latest_pt, f"{run_dir}/best_success_ckpt.pt")
        shutil.copy(latest_json, f"{run_dir}/best_success.json")

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

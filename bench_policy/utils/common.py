#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

import json, math, os, random, shutil
import cv2
import torch
import tqdm
import numpy as np
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional, Sequence
from torch.utils.tensorboard import SummaryWriter

#--------------------------------#
# init
#--------------------------------#


def make_run_name(
    seed: int,
    dataset_name: str,
    algo: str,
    use_class: bool = False,
) -> str:
    """Canonical run dir name under ``runs/``."""
    name = f"seed{seed}_{dataset_name}_{algo}"
    return f"priv_{name}" if use_class else name


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
    writer=None,
    step: int = 0,
    best_rate: float | None = None,
    best_steps: float | None = None,
):
    """Print one compact eval line; optionally write TB scalars."""
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
    tqdm.tqdm.write(line)
    if writer is not None:
        for k, v in summary.items():
            if isinstance(v, (float, np.floating)):
                writer.add_scalar(f"eval/{k}", float(v), step)


def build_eval_episodes(
    dataset,
    max_episodes: int = 0,
    seed: int = 0,
) -> list[dict]:
    n = len(dataset)
    order = np.arange(n, dtype=np.int64)
    if 0 < max_episodes < n:
        rng = np.random.default_rng(seed)
        order = rng.choice(n, size=int(max_episodes), replace=False)
    return [dataset.episode_at(int(i)) for i in order.tolist()]


_PREVIEW_COLORS = {
    "success": (25, 175, 85),
    "collision": (225, 35, 35),
    "fail": (225, 135, 25),
}


def inflate_occupancy(occupancy: np.ndarray, radius: int) -> np.ndarray:
    """Dilate obstacles by ``radius`` (same ellipse kernel as data_gen)."""
    occ = (np.asarray(occupancy) > 0).astype(np.uint8)
    radius = int(radius)
    if radius <= 0:
        return occ.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(occ, kernel).astype(bool)


def _render_rollout_tile(
    planning_map: np.ndarray,
    path_xy: np.ndarray,
    start_rc: np.ndarray,
    goal_rc: np.ndarray,
    *,
    collided: bool,
    reached: bool,
) -> np.ndarray:
    """RGB overlay: occupancy + rollout path + start/goal."""
    rgb = np.where(planning_map[..., None] > 0, 35, 242).astype(np.uint8)
    rgb = np.repeat(rgb, 3, axis=2)
    if collided:
        color = _PREVIEW_COLORS["collision"]
    elif reached:
        color = _PREVIEW_COLORS["success"]
    else:
        color = _PREVIEW_COLORS["fail"]

    pts = np.asarray(path_xy, dtype=np.float64)
    if len(pts) >= 2:
        poly = np.empty((len(pts), 1, 2), dtype=np.int32)
        poly[:, 0, 0] = np.rint(pts[:, 0]).astype(np.int32)
        poly[:, 0, 1] = np.rint(pts[:, 1]).astype(np.int32)
        cv2.polylines(
            rgb,
            [poly],
            isClosed=False,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    sr, sc = int(start_rc[0]), int(start_rc[1])
    gr, gc = int(goal_rc[0]), int(goal_rc[1])
    cv2.circle(rgb, (sc, sr), 4, (20, 215, 65), -1, lineType=cv2.LINE_AA)
    cv2.circle(rgb, (sc, sr), 4, (0, 65, 0), 1, lineType=cv2.LINE_AA)
    cv2.circle(rgb, (gc, gr), 4, (245, 45, 45), -1, lineType=cv2.LINE_AA)
    cv2.circle(rgb, (gc, gr), 4, (85, 0, 0), 1, lineType=cv2.LINE_AA)
    return rgb


def save_eval_preview(
    tiles: Sequence[np.ndarray],
    path: str | Path,
) -> None:
    """Write a square-ish collage of rollout overlays to ``path``."""
    if not tiles:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = int(math.ceil(math.sqrt(len(tiles))))
    rows = int(math.ceil(len(tiles) / columns))
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.full((rows * tile_h, columns * tile_w, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        r0 = (index // columns) * tile_h
        c0 = (index % columns) * tile_w
        canvas[r0:r0 + tile_h, c0:c0 + tile_w] = tile
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    tqdm.tqdm.write(f"[eval] preview saved to {path}")


def evaluate(
    policy,
    episodes: Sequence[dict],
    device: torch.device,
    goal_tol: float = 1.0,
    max_abs_delta: float = 5.0,
    robot_radius: int = 5,
    preview_path: Optional[str] = None,
    preview_count: int = 16,
) -> dict:
    """Open-loop eval: one action-chunk inference from start, then step.

    Collision uses occupancy dilated by ``max(0, robot_radius - 1)`` (relaxed
    vs data_gen ``planning_map`` which uses full ``robot_radius``).

    If ``preview_path`` is set, also write a collage of the first
    ``preview_count`` rollout overlays.
    """
    horizon = int(policy.pred_horizon)
    want_preview = preview_path is not None and int(preview_count) > 0
    results: list[tuple[bool, bool, int]] = []
    preview_tiles: list[np.ndarray] = []
    # Relaxed eval collision: one pixel less inflate than training label radius.
    collision_radius = max(0, int(robot_radius) - 1)
    n_coll = 0
    n_succ = 0

    pbar = tqdm.tqdm(
        episodes,
        desc="eval",
        leave=False,
        dynamic_ncols=True,
    )
    for ep in pbar:
        # Dataset ``map`` is raw occupancy; inflate for robot footprint.
        occupancy = np.asarray(ep["planning_map"])
        collision_map = inflate_occupancy(occupancy, collision_radius)
        size = int(occupancy.shape[0])
        scale = float(size - 1)
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
        record = want_preview and len(preview_tiles) < int(preview_count)
        path = [cur.copy()] if record else None

        steps = 0
        collided = False
        reached = float(np.linalg.norm(cur - goal)) < goal_tol
        if not reached:
            state_list = [
                cur[0] / scale,
                cur[1] / scale,
                goal[0] / scale,
                goal[1] / scale,
            ]
            # Privileged class: append as one extra state dim (4 -> 5).
            if "class" in ep:
                state_list.append(float(ep["class"]))
            state = np.asarray([state_list], dtype=np.float32)
            with torch.no_grad():
                chunk = policy.infer_batch({
                    "map":
                    torch.from_numpy(occupancy.astype(
                        np.float32)[None]).to(device),
                    "state":
                    torch.from_numpy(state[None]).to(device),
                })[0].detach().cpu().numpy()[:horizon]

            for ai, a in enumerate(chunk):
                cur = np.clip(
                    cur +
                    np.clip(a.astype(np.float64), -1.0, 1.0) * max_abs_delta,
                    0.0,
                    size - 1,
                )
                steps += 1
                if path is not None:
                    path.append(cur.copy())
                c, r = int(np.rint(cur[0])), int(np.rint(cur[1]))
                if collision_map[r, c]:
                    collided = True
                    # finish drawing the rest of this chunk into / through walls
                    if path is not None:
                        for a2 in chunk[ai + 1:]:
                            cur = np.clip(
                                cur +
                                np.clip(a2.astype(np.float64), -1.0, 1.0) *
                                max_abs_delta,
                                0.0,
                                size - 1,
                            )
                            path.append(cur.copy())
                    break
                if float(np.linalg.norm(cur - goal)) < goal_tol:
                    reached = True
                    break

        results.append((collided, reached, steps))
        n_coll += int(collided)
        n_succ += int(reached)
        if path is not None:
            preview_tiles.append(
                _render_rollout_tile(
                    occupancy,
                    np.asarray(path, dtype=np.float32),
                    ep["start_rc"],
                    ep["goal_rc"],
                    collided=collided,
                    reached=reached,
                ))
        n = len(results)
        pbar.set_postfix(
            coll=f"{n_coll / n:.0%}",
            succ=f"{n_succ / n:.0%}",
            refresh=False,
        )

    if want_preview:
        save_eval_preview(preview_tiles, preview_path)

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


def _prune_regular_ckpts(run_dir: str, keep: int = 5) -> None:
    """Keep only the newest ``keep`` regular ``ckpt_*.pt`` files under ``run_dir``.

    Does not touch ``final_ckpt.pt`` / ``best_success_ckpt.pt``.
    ``keep`` must be >= 1 so the just-written numbered ckpt is never pruned away.
    """
    keep = max(1, int(keep))
    numbered: list[tuple[int, str]] = []
    for name in os.listdir(run_dir):
        if not (name.startswith("ckpt_") and name.endswith(".pt")):
            continue
        mid = name[len("ckpt_"):-len(".pt")]
        if not mid.isdigit():
            continue
        numbered.append((int(mid), name))
    numbered.sort(key=lambda x: x[0], reverse=True)
    for _, name in numbered[keep:]:
        path = os.path.join(run_dir, name)
        try:
            os.remove(path)
            tqdm.tqdm.write(f"ckpt pruned: {path}")
        except OSError as e:
            tqdm.tqdm.write(f"ckpt prune failed: {path} ({e})")


def save(
    policy,
    run_name: str,
    metrics: Metrics,
    iteration: int,
    is_best: bool = False,
    keep_ckpts: int = 5,
) -> str:
    """Write ``ckpt_*.pt`` + ``latest.json``; if ``is_best``, also snapshot best.

    Regular numbered checkpoints are pruned to the newest ``keep_ckpts``
    (clamped to >= 1). ``final_ckpt.pt`` / ``best_success_ckpt.pt`` are always
    retained.
    """
    model = getattr(policy, "model", None)
    optimizer = getattr(policy, "optimizer", None)
    if model is None:
        raise RuntimeError("policy.model is required for save()")

    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    ckpt_name = ("final_ckpt.pt"
                 if int(iteration) < 0 else f"ckpt_{int(iteration)}.pt")
    ckpt_path = f"{run_dir}/{ckpt_name}"

    checkpoint = {"policy": _to_cpu_tree(model.state_dict())}
    if optimizer is not None:
        checkpoint["optimizer"] = _to_cpu_tree(optimizer.state_dict())
    ema = getattr(policy, "ema", None)
    if ema is not None:
        checkpoint["ema"] = _to_cpu_tree(ema.state_dict())
    torch.save(checkpoint, ckpt_path)

    latest_json = f"{run_dir}/latest.json"
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt_name": ckpt_name,
                "iteration": int(iteration),
                "metrics": asdict(metrics),
            },
            f,
            indent=2,
        )

    if is_best:
        shutil.copy(ckpt_path, f"{run_dir}/best_success_ckpt.pt")
        shutil.copy(latest_json, f"{run_dir}/best_success.json")

    if int(iteration) >= 0:
        _prune_regular_ckpts(run_dir, keep=keep_ckpts)

    tqdm.tqdm.write(f"ckpt saved to {ckpt_path}")
    return ckpt_path


def load(policy, path: str) -> None:
    """Load policy weights (and optimizer if present) from ``path``."""
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
    ema = getattr(policy, "ema", None)
    if ema is not None:
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        else:
            # Old ckpts without EMA: seed shadow from training weights.
            ema.load_state_dict(ckpt["policy"])
    print(f"ckpt loaded from {path}")

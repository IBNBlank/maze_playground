#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################
"""Evaluation entry: ``EvalArgs`` -> dataset + policy -> closed-loop eval."""

import json, os, sys, time, tyro
from pathlib import Path

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from utils.arg import EvalArgs
from utils.common import (
    build_eval_episodes,
    evaluate,
    tensorboard_init,
    load,
    device_init,
    log_eval_summary,
)
from utils.data import MazeWindowDataset
from utils.policy import build_policy
from utils.feishu import send_feishu_eval_notification


class EvalMazeIL:
    """Build dataset + policy from ``EvalArgs`` and run evaluation."""

    def __init__(self):
        self.args: EvalArgs = tyro.cli(EvalArgs)
        self.run_name = f"Seed{self.args.seed}_{self.args.dataset_name}_{self.args.algo}"

        self.device = device_init(
            0,
            torch_deterministic=self.args.torch_deterministic,
            cuda=self.args.cuda,
        )
        self.writer = tensorboard_init(
            self.run_name,
            mode="eval",
            hparams=vars(self.args),
        )

        dataset_dir = Path(REPO_DIR) / "datasets" / self.args.dataset_name
        self.dataset = MazeWindowDataset(dataset_dir)
        self.max_steps = self.dataset.pred_horizon
        self.episodes = build_eval_episodes(
            self.dataset,
            max_episodes=self.args.num_eval,
            seed=self.args.seed,
        )

        self.policy = build_policy(
            self.args.algo,
            obs_horizon=1,
            pred_horizon=self.dataset.pred_horizon,
            state_dim=self.dataset.state_dim,
            action_dim=self.dataset.action_dim,
            device=self.device,
            lr=3e-4,
        )
        load(self.policy, f"runs/{self.run_name}/{self.args.ckpt_name}")

    def run(self) -> dict:
        print(f"Starting evaluation on {self.args.dataset_name} "
              f"(run={self.run_name}, episodes={len(self.episodes)})...")
        stime = time.perf_counter()
        summary = evaluate(
            self.policy,
            self.episodes,
            device=self.device,
            max_steps=self.max_steps,
            goal_tol=self.args.goal_tol,
            max_abs_delta=self.dataset.max_abs_delta,
            preview_path=(f"runs/{self.run_name}/eval_preview.png"
                          if self.args.capture_preview else None),
        )
        eval_time = time.perf_counter() - stime

        log_eval_summary(summary, writer=self.writer, global_step=0)

        result = {
            "algo": self.args.algo,
            "dataset_name": self.args.dataset_name,
            "ckpt_name": self.args.ckpt_name,
            "train_seed": self.args.seed,
            "run_name": self.run_name,
            "num_episodes": summary.get("num_episodes", len(self.episodes)),
            "eval_time": eval_time,
            "goal_tol": self.args.goal_tol,
            "max_episode_steps": self.max_steps,
            "success_rate": summary["success_rate"],
            "success_average_steps": summary["success_average_steps"],
            "collision_rate": summary.get("collision_rate"),
            "metrics": summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        os.makedirs(f"runs/{self.run_name}", exist_ok=True)
        result_path = f"runs/{self.run_name}/eval_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[eval] result saved to {result_path}")

        send_feishu_eval_notification(
            REPO_DIR,
            algo=self.args.algo,
            dataset_name=self.args.dataset_name,
            seed=self.args.seed,
            summary=summary,
            run_name=self.run_name,
        )
        self.writer.close()
        return result


if __name__ == "__main__":
    EvalMazeIL().run()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

"""Evaluation entry: ``EvalArgs`` -> dataset + policy -> closed-loop eval."""

from __future__ import annotations

import json
import os
import sys
import time

import tyro

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from utils.arg import EvalArgs
from utils.common import (
    ACTION_DIM,
    STATE_DIM,
    default_dataset_dir,
    evaluate,
    init_tensorboard,
    load,
    seed_all,
)
from utils.data import MazeWindowDataset
from utils.policy import build_policy
from feishu import send_feishu_eval_notification


class EvalMazeIL:
    """Build dataset + policy from ``EvalArgs`` and run evaluation."""

    def __init__(self, args: EvalArgs):
        self.args = args.finalize()
        self.run_name = self.args.run_name
        assert self.run_name is not None

        # Fixed seed for deterministic episode subsample construction.
        self.device = seed_all(
            0,
            torch_deterministic=self.args.torch_deterministic,
            cuda=self.args.cuda,
        )
        self.writer = init_tensorboard(
            self.run_name,
            mode="eval",
            hparams=vars(self.args),
            enabled=self.args.tensorboard,
        )

        dataset_dir = default_dataset_dir(REPO_DIR, self.args.dataset_name)
        self.dataset = MazeWindowDataset(
            dataset_dir,
            obs_horizon=self.args.obs_horizon,
            pred_horizon=self.args.pred_horizon,
            sample_stride=self.args.sample_stride,
            max_samples=1,
            seed=0,
        )
        self.max_steps = (self.args.max_episode_steps
                          or self.dataset.action_horizon)
        self.episodes = self.dataset.iter_eval_episodes(
            max_episodes=self.args.num_eval,
            seed=self.args.seed,
        )

        self.policy = build_policy(
            self.args.algo,
            self.args.obs_horizon,
            self.args.pred_horizon,
            self.dataset.map_size,
            STATE_DIM,
            ACTION_DIM,
            device=self.device,
            lr=self.args.lr,
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
            obs_horizon=self.args.obs_horizon,
            act_horizon=self.args.act_horizon,
            max_steps=self.max_steps,
            goal_tol=self.args.goal_tol,
            max_abs_delta=self.dataset.max_abs_delta,
        )
        eval_time = time.perf_counter() - stime

        for k, v in summary.items():
            if isinstance(v, float):
                print(f"eval_{k}: {v:.4f}")
                self.writer.add_scalar(f"eval/{k}", v, 0)
            else:
                print(f"eval_{k}: {v}")

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
            "mean_steps": summary.get("mean_steps"),
            "mean_final_error": summary.get("mean_final_error"),
            "metrics": summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if self.args.save_result:
            os.makedirs(f"runs/{self.run_name}", exist_ok=True)
            result_path = f"runs/{self.run_name}/eval_result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[eval] result saved to {result_path}")
        else:
            print("[eval] skip saving eval_result.json")

        send_feishu_eval_notification(
            REPO_DIR,
            algo=self.args.algo,
            dataset_name=self.args.dataset_name,
            seed=self.args.seed,
            summary=summary,
            run_name=self.run_name,
            enabled=self.args.feishu_notification,
        )
        self.writer.close()
        return result


if __name__ == "__main__":
    EvalMazeIL().run()

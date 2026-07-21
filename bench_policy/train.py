#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

import json, os, sys, tyro
from pathlib import Path

import numpy as np
import torch
import tqdm

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from utils.arg import TrainArgs
from utils.common import (
    Metrics,
    build_eval_episodes,
    evaluate,
    load,
    save,
    device_init,
    tensorboard_init,
    log_eval_summary,
)
from utils.data import MazeWindowDataset
from utils.policy import build_policy
from utils.feishu import send_feishu_train_notification


class TrainMazeIL:
    """Build dataset + policy from ``TrainArgs`` and run training."""

    def __init__(self):
        self.args: TrainArgs = tyro.cli(TrainArgs)
        self.run_name = f"seed{self.args.seed}_{self.args.dataset_name}_{self.args.algo}"
        self.early_exit = False

        latest = self._read_latest()
        if latest is not None and int(latest.get("iteration", 0)) < 0:
            self.early_exit = True
            print(f"[train] recorded iteration={latest['iteration']}; "
                  "already finished.")
            return

        os.makedirs(f"runs/{self.run_name}", exist_ok=True)
        self.device = device_init(
            self.args.seed,
            torch_deterministic=self.args.torch_deterministic,
            cuda=self.args.cuda,
        )
        self.writer = tensorboard_init(
            self.run_name,
            mode="train",
            hparams=vars(self.args),
        )

        self.dataset_dir = Path(REPO_DIR) / "datasets" / self.args.dataset_name
        self.dataset = MazeWindowDataset(self.dataset_dir)
        self.eval_episodes = build_eval_episodes(
            self.dataset,
            max_episodes=self.args.num_eval,
            seed=self.args.seed + 1,
        )
        self.policy = build_policy(
            self.args.algo,
            obs_horizon=1,
            pred_horizon=self.dataset.pred_horizon,
            state_dim=self.dataset.state_dim,
            action_dim=self.dataset.action_dim,
            device=self.device,
            lr=self.args.lr,
        )

        rng = np.random.default_rng(int(self.args.seed))
        self.epoch_ids = rng.choice(
            self.dataset.num_idx_perms,
            size=self.args.epochs,
            replace=False,
        ).astype(np.int64)

        self.metrics = Metrics()
        self.start_epoch = 0
        self._resume_if_needed(latest)

        print(f"[train] dataset={self.dataset_dir}")
        print(f"[train] samples={len(self.dataset)} "
              f"shards={len(self.dataset.shards)}")
        print(f"[train] run_name={self.run_name} algo={self.args.algo}")
        print(f"[train] pred_horizon={self.dataset.pred_horizon}")

    def _read_latest(self) -> dict | None:
        path = f"runs/{self.run_name}/latest.json"
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resume_if_needed(self, record: dict | None):
        if record is None:
            return
        self.start_epoch = int(record.get("iteration", 0))
        self.metrics = Metrics.from_dict(record.get("metrics"))
        load(self.policy, f"runs/{self.run_name}/{record.get('ckpt_name')}")
        print(f"[train] resume from epoch index {self.start_epoch}")

    def _epoch_postfix(self, epoch_loss: float, epoch_id: int) -> dict:
        best = self.metrics.best_success_rate
        best_steps = self.metrics.best_success_average_steps
        return {
            "loss":
            f"{epoch_loss:.4f}" if epoch_loss == epoch_loss else "-",
            "best":
            (f"{best * 100:.1f}%/{best_steps:.1f}" if best == best else "-"),
            "idx":
            epoch_id,
        }

    def _train_epoch(self, epoch_idx: int, epoch_id: int) -> float:
        self.dataset.set_epoch(int(epoch_id), batch_size=self.args.batch_size)
        loss_sum = 0.0
        n_steps = 0
        epoch_1based = epoch_idx + 1

        batch_pbar = tqdm.tqdm(
            total=self.dataset.num_batches,
            desc=(f"batch ep{epoch_1based}/{self.args.epochs}"
                  f" idx={epoch_id}"),
            leave=False,
            dynamic_ncols=True,
        )
        while True:
            batch = self.dataset.get_batch()
            if batch is None:
                break
            batch = {
                k:
                v.to(self.device, non_blocking=True)
                if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            loss_val = float(self.policy.update_batch(batch))
            loss_sum += loss_val
            n_steps += 1
            batch_pbar.update(1)
            batch_pbar.set_postfix(loss=f"{loss_val:.4f}")
        batch_pbar.close()

        epoch_loss = loss_sum / n_steps if n_steps > 0 else float("nan")
        if n_steps > 0:
            self.writer.add_scalar("train/loss", epoch_loss, epoch_1based)
        return epoch_loss

    def _eval_and_save(self, epoch_1based: int, is_final: bool = False):
        print(f"[train] evaluating at epoch={epoch_1based}"
              f"{' (final)' if is_final else ''}")
        summary = evaluate(
            self.policy,
            self.eval_episodes,
            device=self.device,
            goal_tol=self.args.goal_tol,
            max_abs_delta=self.dataset.max_abs_delta,
        )

        success = float(summary["success_rate"])
        succ_steps = float(summary["success_average_steps"])
        self.metrics.cur_success_rate = success
        self.metrics.cur_success_average_steps = succ_steps

        is_best = (success > self.metrics.best_success_rate) or (
            success == self.metrics.best_success_rate
            and succ_steps < self.metrics.best_success_average_steps)
        if is_best:
            self.metrics.best_success_rate = success
            self.metrics.best_success_average_steps = succ_steps

        log_eval_summary(
            summary,
            writer=self.writer,
            step=epoch_1based,
            best_rate=self.metrics.best_success_rate,
            best_steps=self.metrics.best_success_average_steps,
        )
        save(
            self.policy,
            run_name=self.run_name,
            metrics=self.metrics,
            iteration=-1 if is_final else epoch_1based,
            is_best=is_best,
        )

    def run(self):
        if self.early_exit:
            return

        epoch_pbar = tqdm.tqdm(
            range(self.start_epoch, self.args.epochs),
            desc="epoch",
            leave=True,
            dynamic_ncols=True,
            initial=self.start_epoch,
            total=self.args.epochs,
        )
        for epoch_idx in epoch_pbar:
            epoch_id = int(self.epoch_ids[epoch_idx])
            epoch_1based = epoch_idx + 1
            epoch_loss = self._train_epoch(epoch_idx, epoch_id)

            if (self.args.eval_freq > 0
                    and epoch_1based % self.args.eval_freq == 0
                    and epoch_1based < self.args.epochs):
                self._eval_and_save(epoch_1based, is_final=False)

            epoch_pbar.set_postfix(self._epoch_postfix(epoch_loss, epoch_id))

        epoch_pbar.close()
        self._eval_and_save(self.args.epochs, is_final=True)

        send_feishu_train_notification(
            REPO_DIR,
            algo=self.args.algo,
            dataset_name=self.args.dataset_name,
            seed=self.args.seed,
            epochs=self.args.epochs,
            metrics=self.metrics,
            run_name=self.run_name,
        )
        self.writer.close()
        print(f"Training done. run_name={self.run_name}")


if __name__ == "__main__":
    TrainMazeIL().run()

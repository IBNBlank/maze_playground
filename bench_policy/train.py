#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

import json, os, sys, tyro
from collections import defaultdict

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
from utils.data import MazeWindowDataset, default_dataset_dir
from utils.policy import build_policy
from utils.feishu import send_feishu_train_notification


def _roll_epoch_ids(seed: int, epochs: int, num_idx_perms: int):
    rng = np.random.default_rng(int(seed))
    return rng.choice(num_idx_perms, size=epochs, replace=False).astype(np.int64)


class TrainMazeIL:
    """Build dataset + policy from ``TrainArgs`` and run training."""

    def __init__(self):
        self.args = tyro.cli(TrainArgs)
        self.run_name = f"Seed{self.args.seed}_{self.args.dataset_name}_{self.args.algo}"

        latest_json = f"runs/{self.run_name}/latest.json"
        recorded = None
        if os.path.isfile(latest_json):
            with open(latest_json, "r", encoding="utf-8") as f:
                recorded = int(json.load(f).get("iteration", 0))
        self.early_exit = recorded is not None and recorded < 0
        if self.early_exit:
            print(f"[train] recorded iteration={recorded}; already finished.")
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

        self.dataset_dir = default_dataset_dir(REPO_DIR, self.args.dataset_name)
        self.dataset = MazeWindowDataset(self.dataset_dir)
        self.eval_episodes = build_eval_episodes(
            self.dataset,
            max_episodes=self.args.num_eval,
            seed=self.args.seed + 1,
        )

        self.policy = build_policy(
            self.args.algo,
            obs_horizon=1,
            pred_horizon=self.dataset.action_horizon,
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            device=self.device,
            lr=self.args.lr,
        )

        self.epoch_ids = _roll_epoch_ids(
            self.args.seed,
            self.args.epochs,
            self.dataset.num_idx_perms,
        )

        self.metrics = Metrics()
        self.global_step = 0
        self.start_epoch = 0
        self._resume_if_needed()

        print(f"[train] dataset={self.dataset_dir}")
        print(
            f"[train] samples={len(self.dataset)} "
            f"shards={len(self.dataset.shards)}"
        )
        print(f"[train] run_name={self.run_name} algo={self.args.algo}")
        print(f"[train] action_horizon={self.dataset.action_horizon}")

    def _resume_if_needed(self):
        latest_json = f"runs/{self.run_name}/latest.json"
        latest_pt = f"runs/{self.run_name}/latest.pt"
        if not os.path.isfile(latest_json) or not os.path.isfile(latest_pt):
            return
        with open(latest_json, "r", encoding="utf-8") as f:
            record = json.load(f)
        iteration = int(record.get("iteration", 0))
        if iteration < 0:
            self.early_exit = True
            return
        start_iter, metrics, ckpt = load(self.policy, latest_pt)
        self.metrics = metrics
        self.global_step = int(ckpt.get("global_step", 0))
        self.start_epoch = max(0, int(start_iter))
        print(f"[train] resume from epoch index {self.start_epoch} "
              f"(global_step={self.global_step})")

    def _train_epoch(self, epoch_idx: int, epoch_id: int) -> float:
        self.dataset.set_epoch(int(epoch_id), batch_size=self.args.batch_size)
        loss_log = defaultdict(float)
        steps_logged = 0
        last_loss = float("nan")
        log_freq = int(getattr(self.args, "log_freq", 0) or 0)

        batch_pbar = tqdm.tqdm(
            total=self.dataset.num_batches,
            desc=(f"batch ep{epoch_idx + 1}/{self.args.epochs}"
                  f" idx={epoch_id}"),
            leave=False,
            dynamic_ncols=True,
        )
        while True:
            batch = self.dataset.get_batch()
            if batch is None:
                break
            batch = {
                k: v.to(self.device, non_blocking=True)
                if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            loss_val = float(self.policy.update_batch(batch))
            last_loss = loss_val
            loss_log["loss"] += loss_val
            steps_logged += 1
            self.global_step += 1
            batch_pbar.update(1)
            batch_pbar.set_postfix(loss=f"{loss_val:.4f}")
            if log_freq > 0 and self.global_step % log_freq == 0:
                avg = loss_log["loss"] / max(steps_logged, 1)
                self.writer.add_scalar("train/loss", avg, self.global_step)
                loss_log.clear()
                steps_logged = 0
        batch_pbar.close()
        if steps_logged > 0:
            avg = loss_log["loss"] / steps_logged
            self.writer.add_scalar("train/loss", avg, self.global_step)
        return last_loss

    def _eval_and_save(self, epoch_1based: int, is_final: bool = False):
        print(f"[train] evaluating at epoch={epoch_1based}"
              f"{' (final)' if is_final else ''}")
        summary = evaluate(
            self.policy,
            self.eval_episodes,
            device=self.device,
            max_steps=self.dataset.action_horizon,
            goal_tol=self.args.goal_tol,
            max_abs_delta=self.dataset.max_abs_delta,
        )

        success = float(summary["success_rate"])
        succ_steps = float(summary["success_average_steps"])
        self.metrics.cur_success_rate = success
        self.metrics.cur_success_average_steps = succ_steps
        is_best = success >= self.metrics.best_success_rate
        if is_best:
            self.metrics.best_success_rate = success
            self.metrics.best_success_average_steps = succ_steps

        log_eval_summary(
            summary,
            writer=self.writer,
            global_step=self.global_step,
            best_rate=self.metrics.best_success_rate,
            best_steps=self.metrics.best_success_average_steps,
        )
        save(
            self.policy,
            run_name=self.run_name,
            metrics=self.metrics,
            iteration=-1 if is_final else epoch_1based,
            global_step=self.global_step,
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
            last_loss = self._train_epoch(epoch_idx, epoch_id)
            best = self.metrics.best_success_rate
            best_steps = self.metrics.best_success_average_steps
            epoch_pbar.set_postfix(
                loss=f"{last_loss:.4f}" if last_loss == last_loss else "-",
                best=(f"{best * 100:.1f}%/{best_steps:.1f}"
                      if best == best else "-"),
                idx=epoch_id,
            )

            if (self.args.eval_freq > 0
                    and epoch_1based % self.args.eval_freq == 0
                    and epoch_1based < self.args.epochs):
                self._eval_and_save(epoch_1based, is_final=False)
                best = self.metrics.best_success_rate
                best_steps = self.metrics.best_success_average_steps
                epoch_pbar.set_postfix(
                    loss=f"{last_loss:.4f}" if last_loss == last_loss else "-",
                    best=(f"{best * 100:.1f}%/{best_steps:.1f}"
                          if best == best else "-"),
                    idx=epoch_id,
                )

        epoch_pbar.close()
        self._eval_and_save(self.args.epochs, is_final=True)

        with open(f"runs/{self.run_name}/latest.json", "r",
                  encoding="utf-8") as f:
            record = json.load(f)
        record["iteration"] = -1
        with open(f"runs/{self.run_name}/latest.json", "w",
                  encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        send_feishu_train_notification(
            REPO_DIR,
            algo=self.args.algo,
            dataset_name=self.args.dataset_name,
            seed=self.args.seed,
            epochs=self.args.epochs,
            metrics=self.metrics,
            run_name=self.run_name,
        )

        self.close()
        print(f"Training done. run_name={self.run_name}")

    def close(self):
        if hasattr(self, "writer"):
            try:
                self.writer.close()
            except Exception:
                pass


if __name__ == "__main__":
    TrainMazeIL().run()

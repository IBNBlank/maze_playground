#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

import json, os, sys, tyro, tqdm

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from utils.arg import TrainArgs
from utils.common import (
    ACTION_DIM,
    STATE_DIM,
    Metrics,
    default_dataset_dir,
    evaluate,
    init_tensorboard,
    load,
    peek_latest_iteration,
    save,
    seed_all,
    train_epoch,
)
from utils.data import MazeWindowDataset, make_dataloader
from utils.policy import build_policy
from feishu import send_feishu_train_notification


class TrainMazeIL:
    """Build dataset + policy from ``TrainArgs`` and run training."""

    def __init__(self):
        self.args = tyro.cli(TrainArgs)
        self.run_name = f"Seed{self.args.seed}_{self.args.dataset_name}_{self.args.algo}"

        recorded = peek_latest_iteration(self.run_name)
        self.early_exit = recorded is not None and recorded < 0
        if self.early_exit:
            print(f"[train] recorded iteration={recorded}; already finished.")
            return

        os.makedirs(f"runs/{self.run_name}", exist_ok=True)
        self.device = seed_all(
            self.args.seed,
            torch_deterministic=self.args.torch_deterministic,
            cuda=self.args.cuda,
        )

        self.writer = init_tensorboard(
            self.run_name,
            mode="train",
            hparams=vars(self.args),
        )

        dataset_dir = default_dataset_dir(REPO_DIR, self.args.dataset_name)
        self.dataset = MazeWindowDataset(
            dataset_dir,
            obs_horizon=self.args.obs_horizon,
            pred_horizon=self.args.pred_horizon,
            sample_stride=self.args.sample_stride,
            max_samples=self.args.max_train_samples,
            seed=self.args.seed,
        )
        self.loader = make_dataloader(
            self.dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_dataload_workers,
            shuffle=True,
            seed=self.args.seed,
        )
        self.eval_episodes = self.dataset.iter_eval_episodes(
            max_episodes=self.args.max_eval_episodes,
            seed=self.args.seed + 1,
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

        self.metrics = Metrics()
        self.global_step = 0
        self.start_epoch = 0
        self._resume_if_needed()

        print(f"[train] dataset={dataset_dir}")
        print(
            f"[train] windows={len(self.dataset)} maps≈{self.dataset.num_maps}"
        )
        print(f"[train] run_name={self.run_name} algo={self.args.algo}")
        print(f"[train] obs_horizon={self.args.obs_horizon} "
              f"pred_horizon={self.args.pred_horizon} "
              f"act_horizon={self.args.act_horizon}")

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

    def _eval_and_save(self, epoch_1based: int, is_final: bool = False):
        print(f"[train] evaluating at epoch={epoch_1based}"
              f"{' (final)' if is_final else ''}")
        summary = evaluate(
            self.policy,
            self.eval_episodes,
            device=self.device,
            obs_horizon=self.args.obs_horizon,
            act_horizon=self.args.act_horizon,
            max_steps=self.dataset.action_horizon,
            goal_tol=self.args.goal_tol,
            max_abs_delta=self.dataset.max_abs_delta,
        )
        for k, v in summary.items():
            if isinstance(v, float):
                self.writer.add_scalar(f"eval/{k}", v, self.global_step)
                print(f"eval/{k}: {v:.4f}")

        success = float(summary["success_rate"])
        succ_steps = float(summary["success_average_steps"])
        if self.args.save_model:
            save(
                self.policy,
                {
                    "epoch":
                    epoch_1based if not is_final else self.args.epochs,
                    "success_rate": success,
                    "success_average_steps": succ_steps,
                    "is_final": is_final,
                },
                run_name=self.run_name,
                algo=self.args.algo,
                metrics=self.metrics,
                global_step=self.global_step,
                act_horizon=self.args.act_horizon,
                extra={
                    "goal_tol": self.args.goal_tol,
                    "max_abs_delta": self.dataset.max_abs_delta
                },
            )
        print(f"eval_success={success:.4f} "
              f"best_success={self.metrics.best_success_rate:.4f}")

    def run(self):
        if self.early_exit:
            return

        epoch_pbar = tqdm(
            range(self.start_epoch, self.args.epochs),
            desc="epoch",
            leave=True,
            dynamic_ncols=True,
            initial=self.start_epoch,
            total=self.args.epochs,
        )
        for epoch_idx in epoch_pbar:
            epoch_1based = epoch_idx + 1
            last_loss, self.global_step = train_epoch(
                self.policy,
                self.loader,
                global_step=self.global_step,
                log_freq=self.args.log_freq,
                writer=self.writer,
                epoch_idx=epoch_idx,
                num_epochs=self.args.epochs,
                progress_cls=tqdm,
            )
            best = self.metrics.best_success_rate
            epoch_pbar.set_postfix(
                loss=f"{last_loss:.4f}" if last_loss == last_loss else "-",
                best_succ=f"{best:.3f}" if best == best else "-",
            )

            if (self.args.eval_freq > 0
                    and epoch_1based % self.args.eval_freq == 0
                    and epoch_1based < self.args.epochs):
                self._eval_and_save(epoch_1based, is_final=False)
                best = self.metrics.best_success_rate
                epoch_pbar.set_postfix(
                    loss=f"{last_loss:.4f}" if last_loss == last_loss else "-",
                    best_succ=f"{best:.3f}" if best == best else "-",
                )

        epoch_pbar.close()
        self._eval_and_save(self.args.epochs, is_final=True)

        if self.args.save_model:
            with open(f"runs/{self.run_name}/latest.json",
                      "r",
                      encoding="utf-8") as f:
                record = json.load(f)
            record["iteration"] = -1
            with open(f"runs/{self.run_name}/latest.json",
                      "w",
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
            enabled=self.args.feishu_notification,
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

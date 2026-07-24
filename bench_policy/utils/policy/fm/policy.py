#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################

import torch

from utils.policy.fm.loss import fm_velocity_mse_loss
from utils.policy.fm.model import FmModel, FmModelConfig
from utils.policy.fm.optim import build_fm_optimizer
from utils.policy.helper.ema import EMAModel
from utils.policy.helper.h2d import batch_to_device
from utils.policy.policy import PolicyBase


class FmPolicy(PolicyBase):
    """Conditional flow-matching action-chunking policy."""

    lr: float = 2e-4
    ema_decay: float = 0.995

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        device: torch.device | str,
        model_cfg: FmModelConfig | None = None,
    ):
        super().__init__(obs_horizon, pred_horizon, state_dim, action_dim)
        self.device = torch.device(device)

        self.model: FmModel = FmModel(
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            cfg=model_cfg,
        ).to(self.device)
        self.ema = EMAModel(self.model, decay=self.ema_decay)
        self.optimizer: torch.optim.Optimizer = build_fm_optimizer(
            self.model, self.lr)
        self.loss_fn = fm_velocity_mse_loss

    def _to_device(self, batch: dict, non_blocking: bool = False) -> dict:
        return batch_to_device(batch, self.device, non_blocking=non_blocking)

    def infer_batch(self, obs: dict) -> torch.Tensor:
        obs = self._to_device(obs)
        self.ema.shadow.eval()
        return self.ema.shadow.sample(obs["map"], obs["state"])

    def update_batch(self, batch: dict) -> float:
        return self._update_on_device(self._to_device(batch))

    def _update_on_device(self, batch: dict) -> float:
        maps = batch["map"]
        state = batch["state"]
        action = batch["action"]

        self.model.train()
        batch_size = action.shape[0]
        noise = torch.randn_like(action)
        t = self.model.fm_scheduler.sample_timesteps(
            batch_size, device=self.device, dtype=action.dtype)
        xt = self.model.fm_scheduler.interpolate(action, noise, t)
        v_target = self.model.fm_scheduler.velocity_target(action, noise)
        v_pred = self.model.predict_velocity(xt, t, maps, state)
        loss = self.loss_fn(v_pred, v_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.ema.update(self.model)
        return float(loss.detach().item())

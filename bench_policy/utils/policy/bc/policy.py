#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

import torch

from utils.policy.bc.loss import bc_mse_loss
from utils.policy.bc.model import BcModel
from utils.policy.bc.optim import build_bc_optimizer
from utils.policy.helper.ema import EMAModel
from utils.policy.helper.h2d import batch_to_device
from utils.policy.policy import PolicyBase


class BcPolicy(PolicyBase):
    """MSE behavior cloning with ConditionalUnet1D (learnable action query)."""

    lr: float = 3e-4
    ema_decay: float = 0.995

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        device: torch.device | str,
    ):
        super().__init__(obs_horizon, pred_horizon, state_dim, action_dim)
        self.device = torch.device(device)

        self.model: BcModel = BcModel(
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        ).to(self.device)
        self.ema = EMAModel(self.model, decay=self.ema_decay)
        self.optimizer: torch.optim.Optimizer = build_bc_optimizer(
            self.model, self.lr)
        self.loss_fn = bc_mse_loss

    def _to_device(self, batch: dict, non_blocking: bool = False) -> dict:
        return batch_to_device(batch, self.device, non_blocking=non_blocking)

    def infer_batch(self, obs: dict) -> torch.Tensor:
        obs = self._to_device(obs)
        self.ema.shadow.eval()
        return self.ema.shadow(obs["map"], obs["state"])

    def update_batch(self, batch: dict) -> float:
        return self._update_on_device(self._to_device(batch))

    def _update_on_device(self, batch: dict) -> float:
        maps = batch["map"]
        state = batch["state"]
        action = batch["action"]

        self.model.train()
        pred = self.model(maps, state)
        loss = self.loss_fn(pred, action)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.ema.update(self.model)
        return float(loss.detach().item())

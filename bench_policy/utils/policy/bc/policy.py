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
from utils.policy.policy import PolicyBase


class BcPolicy(PolicyBase):
    """MSE behavior cloning."""

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        map_size: int,
        state_dim: int,
        action_dim: int,
        device: torch.device | str,
        lr: float,
    ):
        super().__init__(obs_horizon, pred_horizon, map_size, state_dim,
                         action_dim)
        self.device = torch.device(device)
        self.lr = float(lr)

        self.model: BcModel = BcModel(
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        ).to(self.device)
        self.optimizer: torch.optim.Optimizer = build_bc_optimizer(
            self.model, self.lr)
        self.loss_fn = bc_mse_loss

    def infer_batch(self, obs: dict) -> torch.Tensor:
        maps = obs["map"].to(self.device)
        state = obs["state"].to(self.device)
        self.model.eval()
        return self.model(maps, state)

    def update_batch(self, batch: dict) -> float:
        maps = batch["map"].to(self.device)
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)

        self.model.train()
        pred = self.model(maps, state)
        loss = self.loss_fn(pred, action)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################

import torch

from utils.policy.act.loss import act_loss
from utils.policy.act.model import ActModel, ActModelConfig
from utils.policy.act.optim import build_act_optimizer
from utils.policy.policy import PolicyBase


class ActPolicy(PolicyBase):
    """CVAE action-chunking transformer (ACT)."""

    lr: float = 1e-4
    kl_weight: float = 10.0

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        device: torch.device | str,
        model_cfg: ActModelConfig | None = None,
    ):
        super().__init__(obs_horizon, pred_horizon, state_dim, action_dim)
        self.device = torch.device(device)

        self.model: ActModel = ActModel(
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            cfg=model_cfg,
        ).to(self.device)
        self.optimizer: torch.optim.Optimizer = build_act_optimizer(
            self.model, self.lr)
        self.loss_fn = act_loss

    def infer_batch(self, obs: dict) -> torch.Tensor:
        maps = obs["map"].to(self.device)
        state = obs["state"].to(self.device)
        self.model.eval()
        with torch.no_grad():
            a_hat, _ = self.model(maps, state, actions=None)
        return a_hat

    def update_batch(self, batch: dict) -> float:
        maps = batch["map"].to(self.device)
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)

        self.model.train()
        a_hat, (mu, logvar) = self.model(maps, state, actions=action)
        loss = self.loss_fn(
            a_hat, action, mu, logvar, kl_weight=self.kl_weight)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

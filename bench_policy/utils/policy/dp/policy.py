#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################

import torch

from utils.policy.dp.loss import dp_noise_mse_loss
from utils.policy.dp.model import DpModel, DpModelConfig
from utils.policy.dp.optim import build_dp_optimizer
from utils.policy.policy import PolicyBase


class DpPolicy(PolicyBase):
    """Conditional DDPM action-chunking policy (Diffusion Policy)."""

    lr: float = 1e-4

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        device: torch.device | str,
        model_cfg: DpModelConfig | None = None,
    ):
        super().__init__(obs_horizon, pred_horizon, state_dim, action_dim)
        self.device = torch.device(device)

        self.model: DpModel = DpModel(
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            cfg=model_cfg,
        ).to(self.device)
        self.optimizer: torch.optim.Optimizer = build_dp_optimizer(
            self.model, self.lr)
        self.loss_fn = dp_noise_mse_loss

    def infer_batch(self, obs: dict) -> torch.Tensor:
        maps = obs["map"].to(self.device)
        state = obs["state"].to(self.device)
        self.model.eval()
        return self.model.sample(maps, state)

    def update_batch(self, batch: dict) -> float:
        maps = batch["map"].to(self.device)
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)

        self.model.train()
        batch_size = action.shape[0]
        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0,
            self.model.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        noisy_actions = self.model.noise_scheduler.add_noise(
            action, noise, timesteps)
        noise_pred = self.model.predict_noise(
            noisy_actions, timesteps, maps, state)
        loss = self.loss_fn(noise_pred, noise)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

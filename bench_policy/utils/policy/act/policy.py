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
from utils.policy.helper.ema import EMAModel
from utils.policy.helper.h2d import batch_to_device
from utils.policy.policy import PolicyBase


class ActPolicy(PolicyBase):
    """MLP-CVAE ``z→step_embed`` + ConditionalUnet1D action chunker (ACT)."""

    lr: float = 2e-4
    kl_weight: float = 5.0
    ema_decay: float = 0.995

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
        self.ema = EMAModel(self.model, decay=self.ema_decay)
        self.optimizer: torch.optim.Optimizer = build_act_optimizer(
            self.model, self.lr)
        self.loss_fn = act_loss

    def _to_device(self, batch: dict, non_blocking: bool = False) -> dict:
        return batch_to_device(batch, self.device, non_blocking=non_blocking)

    def infer_batch(self, obs: dict) -> torch.Tensor:
        obs = self._to_device(obs)
        self.ema.shadow.eval()
        with torch.no_grad():
            a_hat, _ = self.ema.shadow(obs["map"], obs["state"], actions=None)
        return a_hat

    def update_batch(self, batch: dict) -> float:
        return self._update_on_device(self._to_device(batch))

    def _update_on_device(self, batch: dict) -> float:
        maps = batch["map"]
        state = batch["state"]
        action = batch["action"]

        self.model.train()
        a_hat, (mu, logvar) = self.model(maps, state, actions=action)
        loss = self.loss_fn(a_hat,
                            action,
                            mu,
                            logvar,
                            kl_weight=self.kl_weight)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.ema.update(self.model)
        return float(loss.detach().item())

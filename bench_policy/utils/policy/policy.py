#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

import torch
from abc import ABC, abstractmethod


class PolicyBase(ABC):
    """Unified IL policy interface for maze BC / ACT / DP / FM."""

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
    ):
        # state
        self.obs_horizon = obs_horizon
        self.state_dim = state_dim

        # pred
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

    @abstractmethod
    def infer_batch(self, obs: dict) -> torch.Tensor:
        """
        batch:
        {
            "map": (batch_size, map_size, map_size),
            "state": (batch_size, obs_horizon, state_dim),
        }
        return: (batch_size, pred_horizon, action_dim)
        """
        pass

    @abstractmethod
    def update_batch(self, batch: dict) -> float:
        """
        batch:
        {
            "map": (batch_size, map_size, map_size),
            "state": (batch_size, obs_horizon, state_dim),
            "action": (batch_size, pred_horizon, action_dim),
        }
        return: float
        """
        pass


def build_policy(
    algo: str,
    obs_horizon: int,
    pred_horizon: int,
    state_dim: int,
    action_dim: int,
    device,
) -> PolicyBase:
    """Construct a policy for ``algo`` (each algo picks its own lr)."""
    algo = algo.lower()
    if algo == "bc":
        from .bc.policy import BcPolicy
        return BcPolicy(
            obs_horizon,
            pred_horizon,
            state_dim,
            action_dim,
            device=device,
        )
    if algo == "act":
        from .act.policy import ActPolicy
        return ActPolicy(
            obs_horizon,
            pred_horizon,
            state_dim,
            action_dim,
            device=device,
        )
    if algo == "dp":
        from .dp.policy import DpPolicy
        return DpPolicy(
            obs_horizon,
            pred_horizon,
            state_dim,
            action_dim,
            device=device,
        )
    # elif algo == "fm":
    #     from .fm.policy import FmPolicy
    #     return FmPolicy(obs_horizon, pred_horizon, state_dim,
    #                    action_dim, device=device)
    raise ValueError(f"Unknown algo={algo}")

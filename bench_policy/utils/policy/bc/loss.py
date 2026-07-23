#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################
"""Behavior-cloning losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bc_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over ``(B, pred_horizon, action_dim)`` action chunks."""
    return F.mse_loss(pred, target)

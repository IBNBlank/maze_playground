#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Flow Matching losses: velocity-prediction MSE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fm_velocity_mse_loss(
    velocity_pred: torch.Tensor,
    velocity_target: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted and target velocity over the action chunk."""
    return F.mse_loss(velocity_pred, velocity_target)

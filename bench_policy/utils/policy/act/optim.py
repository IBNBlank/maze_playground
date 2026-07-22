#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Optimizers for ACT policies."""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim

from utils.policy.helper.optim import build_adamw_optimizer


def build_act_optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
    return build_adamw_optimizer(model, lr)

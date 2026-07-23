#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-23
################################################################
"""Exponential moving average of model weights (eval / deploy)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMAModel:
    """Shadow copy of ``model`` updated as ``θ ← τθ + (1-τ)θ_train``."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model)
        self.shadow.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend training weights into the EMA shadow (call after optimizer.step)."""
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
        for s_buf, buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(buf)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)

    def to(self, device: torch.device | str) -> EMAModel:
        self.shadow.to(device)
        return self

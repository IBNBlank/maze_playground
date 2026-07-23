#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Conditional flow-matching path + Euler ODE (OT / rectified-flow style)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass
class FlowMatchingSchedulerOutput:
    prev_sample: torch.Tensor


class FlowMatchingScheduler:
    """Linear OT path: ``x_t = (1-t) x_0 + t x_1``, ``v = x_1 - x_0``.

    ``x_0`` is Gaussian noise, ``x_1`` is the action chunk. Training samples
    ``t ~ U(0,1)``; inference integrates the ODE from ``t=0`` to ``t=1``.
    """

    def __init__(
        self,
        num_inference_steps: int = 10,
        time_embed_scale: float = 1000.0,
    ):
        self.time_embed_scale = float(time_embed_scale)
        # Keep a ``config`` attribute so FM mirrors DP's scheduler access pattern.
        self.config = SimpleNamespace(num_inference_steps=int(num_inference_steps))
        self.timesteps: torch.Tensor = torch.empty(0, dtype=torch.float32)
        self.dt: float = 0.0
        self.set_timesteps(int(num_inference_steps))

    def set_timesteps(self, num_inference_steps: int) -> None:
        steps = int(num_inference_steps)
        if steps <= 0:
            raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
        self.num_inference_steps = steps
        self.config.num_inference_steps = steps
        self.dt = 1.0 / steps
        # Left endpoints of uniform intervals on [0, 1].
        self.timesteps = torch.linspace(
            0.0, 1.0 - self.dt, steps, dtype=torch.float32)

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Train-time ``t ~ Uniform(0, 1)`` shaped ``(B,)``."""
        return torch.rand(batch_size, device=device, dtype=dtype)

    def time_for_net(self, t: torch.Tensor | float) -> torch.Tensor:
        """Scale continuous ``t`` for sinusoidal step embedding (SD-style)."""
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)
        return t.float() * self.time_embed_scale

    @staticmethod
    def _broadcast_t(t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = t.float()
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out

    def interpolate(
        self,
        x1: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """``x_t = (1-t) x_0 + t x_1``."""
        t_b = self._broadcast_t(t, x1.shape).to(device=x1.device, dtype=x1.dtype)
        return (1.0 - t_b) * x0 + t_b * x1

    def velocity_target(self, x1: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Conditional velocity for the linear path."""
        return x1 - x0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> FlowMatchingSchedulerOutput:
        """Euler: ``x ← x + dt * v`` (``timestep`` unused except for API parity)."""
        del timestep  # Uniform grid; dt fixed by ``set_timesteps``.
        return FlowMatchingSchedulerOutput(
            prev_sample=sample + self.dt * model_output)


def build_fm_scheduler(num_inference_steps: int = 10) -> FlowMatchingScheduler:
    """Build OT flow-matching scheduler with Euler inference."""
    return FlowMatchingScheduler(num_inference_steps=int(num_inference_steps))

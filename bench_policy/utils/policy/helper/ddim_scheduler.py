#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Minimal DDIM scheduler (no diffusers): same cosine schedule as DDPM."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from utils.policy.helper.ddpm_scheduler import _betas_squaredcos_cap_v2


@dataclass
class DDIMSchedulerOutput:
    prev_sample: torch.Tensor


class DDIMScheduler:
    """Epsilon-prediction DDIM with cosine beta schedule and clipped x0.

    Shares ``num_train_timesteps`` / alphas with the training DDPM schedule;
    inference uses a shorter subsampled reverse chain (``eta=0`` by default).
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
        eta: float = 0.0,
    ):
        if prediction_type != "epsilon":
            raise ValueError(
                f"Only prediction_type='epsilon' is supported, got {prediction_type!r}"
            )
        self.num_train_timesteps = int(num_train_timesteps)
        self.clip_sample = bool(clip_sample)
        self.clip_sample_range = float(clip_sample_range)
        self.prediction_type = prediction_type
        self.eta = float(eta)
        self.config = SimpleNamespace(
            num_train_timesteps=self.num_train_timesteps)

        self.betas = _betas_squaredcos_cap_v2(self.num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.timesteps: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.num_inference_steps: int = 0

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Subsample the reverse chain: evenly spaced indices ending at 0."""
        steps = int(num_inference_steps)
        if steps <= 0 or steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps must be in [1, {self.num_train_timesteps}], "
                f"got {steps}"
            )
        step_ratio = self.num_train_timesteps // steps
        timesteps = (
            torch.arange(0, steps, dtype=torch.long) * step_ratio
        ).flip(0)
        self.timesteps = timesteps
        self.num_inference_steps = steps

    def _gather_acp(
        self,
        timesteps: torch.Tensor,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        acp = self.alphas_cumprod.to(device=device, dtype=dtype)
        out = acp[timesteps.long()]
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out

    def _previous_timestep(self, timestep: int) -> int:
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        return int(timestep) - step_ratio

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        eta: float | None = None,
    ) -> DDIMSchedulerOutput:
        """One DDIM reverse step → ``x_{t-1}`` from ε̂ (Song et al.)."""
        t = int(timestep) if not torch.is_tensor(timestep) else int(
            timestep.item() if timestep.ndim == 0 else timestep.reshape(-1)[0].item()
        )
        prev_t = self._previous_timestep(t)
        eta_val = self.eta if eta is None else float(eta)

        acp = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = acp[t]
        alpha_prod_t_prev = acp[prev_t] if prev_t >= 0 else torch.tensor(
            1.0, device=sample.device, dtype=sample.dtype)
        beta_prod_t = 1.0 - alpha_prod_t

        # x0 from epsilon prediction.
        pred_original = (
            sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        if self.clip_sample:
            pred_original = pred_original.clamp(
                -self.clip_sample_range, self.clip_sample_range)

        # Direction pointing to x_t, plus optional DDIM noise (eta).
        variance = (
            (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)
        ) * (1.0 - alpha_prod_t / alpha_prod_t_prev)
        variance = torch.clamp(variance, min=0.0)
        std_dev_t = eta_val * variance.sqrt()
        pred_sample_direction = (
            1.0 - alpha_prod_t_prev - std_dev_t ** 2
        ).clamp(min=0.0).sqrt() * model_output

        prev_sample = (
            alpha_prod_t_prev.sqrt() * pred_original + pred_sample_direction
        )
        if eta_val > 0.0 and t > 0:
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            prev_sample = prev_sample + std_dev_t * noise

        return DDIMSchedulerOutput(prev_sample=prev_sample)


def build_ddim_scheduler(
    num_train_timesteps: int = 100,
    num_inference_steps: int = 10,
    eta: float = 0.0,
) -> DDIMScheduler:
    """Build epsilon-prediction DDIM with squared-cosine beta schedule."""
    scheduler = DDIMScheduler(
        num_train_timesteps=int(num_train_timesteps),
        clip_sample=True,
        prediction_type="epsilon",
        eta=float(eta),
    )
    scheduler.set_timesteps(int(num_inference_steps))
    return scheduler

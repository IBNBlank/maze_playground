#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Minimal DDPM scheduler (no diffusers): ManiSkill / Chi Diffusion Policy defaults."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch


def _betas_squaredcos_cap_v2(
    num_train_timesteps: int,
    max_beta: float = 0.999,
) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule (diffusers ``squaredcos_cap_v2``)."""

    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_train_timesteps):
        t1 = i / num_train_timesteps
        t2 = (i + 1) / num_train_timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


@dataclass
class DDPMSchedulerOutput:
    prev_sample: torch.Tensor


class DDPMScheduler:
    """Epsilon-prediction DDPM with cosine beta schedule and clipped x0."""

    def __init__(
        self,
        num_train_timesteps: int = 100,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
    ):
        if prediction_type != "epsilon":
            raise ValueError(
                f"Only prediction_type='epsilon' is supported, got {prediction_type!r}"
            )
        self.num_train_timesteps = int(num_train_timesteps)
        self.clip_sample = bool(clip_sample)
        self.clip_sample_range = float(clip_sample_range)
        self.prediction_type = prediction_type
        # Match diffusers attribute used by DpPolicy.update_batch.
        self.config = SimpleNamespace(
            num_train_timesteps=self.num_train_timesteps)

        self.betas = _betas_squaredcos_cap_v2(self.num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.timesteps: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.set_timesteps(self.num_train_timesteps)

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Full reverse chain: ``T-1, ..., 0`` (same as DDPM with equal train steps)."""
        steps = int(num_inference_steps)
        if steps <= 0 or steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps must be in [1, {self.num_train_timesteps}], "
                f"got {steps}"
            )
        # Evenly spaced indices ending at 0, matching diffusers DDPM default.
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

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward process: ``x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε``."""
        acp = self._gather_acp(
            timesteps,
            original_samples.shape,
            original_samples.device,
            original_samples.dtype,
        )
        return acp.sqrt() * original_samples + (1.0 - acp).sqrt() * noise

    def _previous_timestep(self, timestep: int) -> int:
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        return int(timestep) - step_ratio

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> DDPMSchedulerOutput:
        """One reverse step → ``x_{t-1}`` from ε̂ (Ho et al. Eq. 7)."""
        t = int(timestep) if not torch.is_tensor(timestep) else int(
            timestep.item() if timestep.ndim == 0 else timestep.reshape(-1)[0].item()
        )
        prev_t = self._previous_timestep(t)

        acp = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = acp[t]
        alpha_prod_t_prev = acp[prev_t] if prev_t >= 0 else torch.tensor(
            1.0, device=sample.device, dtype=sample.dtype)
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        # x0 from epsilon prediction.
        pred_original = (
            sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        if self.clip_sample:
            pred_original = pred_original.clamp(
                -self.clip_sample_range, self.clip_sample_range)

        # Posterior mean of q(x_{t-1} | x_t, x_0).
        pred_prev = (
            alpha_prod_t_prev.sqrt() * current_beta_t / beta_prod_t
        ) * pred_original + (
            current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
        ) * sample

        if t > 0:
            variance = (beta_prod_t_prev / beta_prod_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            pred_prev = pred_prev + variance.sqrt() * noise

        return DDPMSchedulerOutput(prev_sample=pred_prev)


def build_ddpm_scheduler(num_train_timesteps: int = 100) -> DDPMScheduler:
    """Build epsilon-prediction DDPM with squared-cosine beta schedule."""
    return DDPMScheduler(
        num_train_timesteps=int(num_train_timesteps),
        clip_sample=True,
        prediction_type="epsilon",
    )

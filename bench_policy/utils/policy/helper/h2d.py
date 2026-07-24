#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-24
################################################################

"""Pinned-host → GPU async H2D overlapped with the previous train step."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

Batch = Dict[str, torch.Tensor]
ComputeFn = Callable[[Batch], float]


def batch_to_device(
    batch: Batch,
    device: torch.device,
    non_blocking: bool = False,
) -> Batch:
    return {
        k: (v.to(device, non_blocking=non_blocking)
            if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


class H2dTrainPipeline:
    """Double-buffer: H2D(batch N+1) on a side stream while computing batch N.

    ``push`` returns the loss of the *previous* batch (or ``None`` on the first
    call). ``flush`` runs the last pending batch.
    """

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self.enabled = self.device.type == "cuda"
        self._copy_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if self.enabled else None)
        self._pending: Optional[tuple[Batch, torch.cuda.Event]] = None

    def _copy_async(self, cpu_batch: Batch) -> tuple[Batch, torch.cuda.Event]:
        assert self._copy_stream is not None
        with torch.cuda.stream(self._copy_stream):
            gpu_batch = batch_to_device(
                cpu_batch, self.device, non_blocking=True)
        event = torch.cuda.Event()
        event.record(self._copy_stream)
        return gpu_batch, event

    def _compute_pending(self, compute_fn: ComputeFn) -> float:
        assert self._pending is not None
        gpu_batch, event = self._pending
        self._pending = None
        if self.enabled:
            torch.cuda.current_stream(self.device).wait_event(event)
        return compute_fn(gpu_batch)

    def push(self, cpu_batch: Batch, compute_fn: ComputeFn) -> Optional[float]:
        if not self.enabled:
            # CPU / no overlap: compute immediately.
            return compute_fn(batch_to_device(cpu_batch, self.device))

        # Kick H2D of the new batch first so it overlaps the previous compute.
        new_pending = self._copy_async(cpu_batch)
        loss: Optional[float] = None
        if self._pending is not None:
            loss = self._compute_pending(compute_fn)
        self._pending = new_pending
        return loss

    def flush(self, compute_fn: ComputeFn) -> Optional[float]:
        if self._pending is None:
            return None
        if not self.enabled:
            gpu_batch, _ = self._pending
            self._pending = None
            return compute_fn(gpu_batch)
        return self._compute_pending(compute_fn)

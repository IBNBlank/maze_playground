#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################
"""Feishu (Lark) webhook notifications for maze IL train / eval.

Reads ``{repo_dir}/feishu.json``:

```json
{
  "train_webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/...",
  "eval_webhook_url":  "https://open.feishu.cn/open-apis/bot/v2/hook/..."
}
```
"""

import json
import os
import urllib.request
from typing import Any, Optional, Sequence

from .common import Metrics


def _load_feishu_config(repo_dir: str) -> Optional[dict]:
    config_path = os.path.join(repo_dir, "feishu.json")
    if not os.path.isfile(config_path):
        print(f"[feishu] config not found at {config_path}, skip")
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[feishu] failed to read {config_path}: {e}")
        return None


def _feishu_template_from_success(success_rate: float) -> str:
    if success_rate != success_rate:  # NaN
        return "grey"
    if success_rate <= 0.5:
        return "red"
    if success_rate <= 0.8:
        return "yellow"
    return "green"


def _post_feishu_card(
    webhook_url: str,
    *,
    title: str,
    markdown: str,
    template: str = "green",
) -> bool:
    payload = json.dumps(
        {
            "msg_type": "interactive",
            "card": {
                "schema": "2.0",
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": title,
                    },
                    "template": template,
                },
                "body": {
                    "elements": [{
                        "tag": "markdown",
                        "content": markdown,
                    }],
                },
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        result = json.loads(body)
        if result.get("code") == 0:
            print("[feishu] notification sent")
            return True
        print(f"[feishu] notification rejected: {body}")
        return False
    except Exception as e:
        print(f"[feishu] notification error: {e}")
        return False


def send_feishu_notification(
    repo_dir: str,
    *,
    mode: str = "train",
    title: Optional[str] = None,
    markdown: str,
    success_rate: Optional[float] = None,
    template: Optional[str] = None,
    enabled: bool = True,
) -> bool:
    """Push an interactive Feishu card via ``feishu.json`` webhooks.

    Args:
      repo_dir: repository root containing ``feishu.json``.
      mode: ``\"train\"`` -> ``train_webhook_url``;
            ``\"eval\"``  -> ``eval_webhook_url``.
      title: card title (defaults by mode).
      markdown: card body markdown.
      success_rate: optional; picks red/yellow/green template when set.
      template: explicit Feishu header template override.
      enabled: if False, no-op.
    """
    if not enabled:
        return False
    mode = mode.lower()
    if mode not in ("train", "eval"):
        raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")

    config = _load_feishu_config(repo_dir)
    if config is None:
        return False
    key = "train_webhook_url" if mode == "train" else "eval_webhook_url"
    webhook_url = config.get(key)
    if not webhook_url:
        print(f"[feishu] {key} missing in feishu.json, skip")
        return False

    if title is None:
        title = ("IL Training Finished"
                 if mode == "train" else "IL Evaluation Finished")
    if template is None:
        if success_rate is not None:
            template = _feishu_template_from_success(float(success_rate))
        else:
            template = "green"

    return _post_feishu_card(
        webhook_url,
        title=title,
        markdown=markdown,
        template=template,
    )


def send_feishu_train_notification(
    repo_dir: str,
    *,
    algo: str,
    dataset_name: str,
    seed: int,
    epochs: int,
    metrics: Metrics,
    run_name: Optional[str] = None,
    enabled: bool = True,
) -> bool:
    """Notify after one training run finishes."""
    md_lines = [
        f"- **algo:** {algo}",
        f"- **dataset:** {dataset_name}",
        f"- **seed:** {seed}",
        f"- **epochs:** {epochs}",
        f"- **best_success:** {metrics.best_success_rate * 100:.2f}%",
        f"- **best_success_average_steps:** {metrics.best_success_average_steps:.2f}",
    ]
    if run_name:
        md_lines.insert(0, f"- **run:** {run_name}")
    return send_feishu_notification(
        repo_dir,
        mode="train",
        markdown="\n".join(md_lines),
        success_rate=metrics.best_success_rate,
        enabled=enabled,
    )


def send_feishu_eval_notification(
    repo_dir: str,
    *,
    algo: str,
    dataset_name: str,
    seed: int,
    summary: dict,
    run_name: Optional[str] = None,
    enabled: bool = True,
) -> bool:
    """Notify after one evaluation run finishes."""
    success = float(summary.get("success_rate", 0.0))
    succ_steps = float(summary.get("success_average_steps", float("inf")))
    collision = float(summary.get("collision_rate", 0.0))
    if succ_steps != succ_steps or succ_steps == float("inf"):
        steps_text = "n/a"
    else:
        steps_text = f"{succ_steps:.2f}"

    md_lines = [
        f"- **algo:** {algo}",
        f"- **dataset:** {dataset_name}",
        f"- **seed:** {seed}",
        f"- **success_rate:** {success * 100:.2f}%",
        f"- **success_average_steps:** {steps_text}",
        f"- **collision_rate:** {collision * 100:.2f}%",
        f"- **num_episodes:** {summary.get('num_episodes', '-')}",
    ]
    if run_name:
        md_lines.insert(0, f"- **run:** {run_name}")
    return send_feishu_notification(
        repo_dir,
        mode="eval",
        markdown="\n".join(md_lines),
        success_rate=success,
        enabled=enabled,
    )


def send_feishu_train_sweep_notification(
    repo_dir: str,
    *,
    seeds: Sequence[Any],
    algos: Sequence[str],
    dataset_names: Sequence[str],
    enabled: bool = True,
) -> bool:
    """Notify after a full ``run_train.sh`` sweep finishes."""
    if not seeds or not algos or not dataset_names:
        print("[feishu] empty sweep; nothing to notify")
        return False
    md = (f"- **datasets:** {' '.join(str(d) for d in dataset_names)}\n"
          f"- **seeds:** {' '.join(str(s) for s in seeds)}\n"
          f"- **algos:** {' '.join(str(a) for a in algos)}")
    return send_feishu_notification(
        repo_dir,
        mode="train",
        title="IL Training Sweep Finished",
        markdown=md,
        template="green",
        enabled=enabled,
    )


def send_feishu_eval_sweep_notification(
    repo_dir: str,
    *,
    seeds: Sequence[Any],
    algos: Sequence[str],
    dataset_names: Sequence[str],
    mean_success_rate: Optional[float] = None,
    enabled: bool = True,
) -> bool:
    """Notify after a full ``run_eval.sh`` sweep finishes."""
    if not seeds or not algos or not dataset_names:
        print("[feishu] empty sweep; nothing to notify")
        return False
    md_lines = [
        f"- **datasets:** {' '.join(str(d) for d in dataset_names)}",
        f"- **seeds:** {' '.join(str(s) for s in seeds)}",
        f"- **algos:** {' '.join(str(a) for a in algos)}",
    ]
    if mean_success_rate is not None and mean_success_rate == mean_success_rate:
        md_lines.append(
            f"- **mean_success_rate:** {mean_success_rate * 100:.2f}%")
    return send_feishu_notification(
        repo_dir,
        mode="eval",
        title="IL Evaluation Sweep Finished",
        markdown="\n".join(md_lines),
        success_rate=mean_success_rate,
        enabled=enabled,
    )

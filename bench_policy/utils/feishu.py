#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

import json
import os
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from .common import make_run_name

if TYPE_CHECKING:
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
    except Exception as e:
        print(f"[feishu] notification error: {e}")
        return False
    if result.get("code") == 0:
        print("[feishu] notification sent")
        return True
    print(f"[feishu] notification rejected: {body}")
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
    """Push an interactive Feishu card via ``feishu.json`` webhooks."""
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
        title = ("Maze Training Finished"
                 if mode == "train" else "Maze Evaluation Finished")
    if template is None:
        template = (_feishu_template_from_success(float(success_rate))
                    if success_rate is not None else "green")

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
    metrics: "Metrics",
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
        f"- **best_success_average_steps:** "
        f"{metrics.best_success_average_steps:.2f}",
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
        print("[feishu] empty train sweep; nothing to notify")
        return False
    md = (f"- **datasets:** {' '.join(str(d) for d in dataset_names)}\n"
          f"- **seeds:** {' '.join(str(s) for s in seeds)}\n"
          f"- **algos:** {' '.join(str(a) for a in algos)}\n"
          f"- **use_class:** 0 1")
    return send_feishu_notification(
        repo_dir,
        mode="train",
        title="Maze Training Finished",
        markdown=md,
        template="green",
        enabled=enabled,
    )


def _fmt_steps(value: Any) -> str:
    try:
        steps = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if steps != steps or steps == float("inf"):
        return "n/a"
    return f"{steps:.2f}"


def collect_eval_results(
    runs_dir: str | Path,
    seeds: Sequence[Any],
    dataset_names: Sequence[str],
    algos: Sequence[str],
) -> list[dict]:
    """Load existing ``eval_result.json`` files for a sweep grid.

    Covers both ``use_class=0/1`` run dirs; missing / invalid results are skipped.
    """
    results: list[dict] = []
    root = Path(runs_dir)
    for seed in seeds:
        for dataset in dataset_names:
            for algo in algos:
                for use_class in (False, True):
                    run = make_run_name(seed,
                                        dataset,
                                        algo,
                                        use_class=use_class)
                    path = root / run / "eval" / "eval_result.json"
                    if not path.is_file():
                        continue
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        results.append({
                            "run_name": run,
                            "algo": str(data.get("algo", algo)),
                            "dataset_name": str(
                                data.get("dataset_name", dataset)),
                            "use_class": bool(
                                data.get("use_class", use_class)),
                            "seed": data.get("train_seed", seed),
                            "success_rate": float(data["success_rate"]),
                            "collision_rate": float(
                                data.get("collision_rate", float("nan"))),
                            "success_average_steps": data.get(
                                "success_average_steps"),
                            "num_episodes": data.get("num_episodes"),
                        })
                    except (OSError, json.JSONDecodeError, KeyError, TypeError,
                            ValueError):
                        continue
    return results


def send_feishu_eval_sweep_notification(
    repo_dir: str,
    *,
    seeds: Sequence[Any],
    algos: Sequence[str],
    dataset_names: Sequence[str],
    results: Optional[Sequence[dict]] = None,
    enabled: bool = True,
) -> bool:
    """Notify after a full ``run_eval.sh`` sweep finishes (aggregate summary)."""
    if not seeds or not algos or not dataset_names:
        print("[feishu] empty eval sweep; nothing to notify")
        return False

    expected = len(seeds) * len(dataset_names) * len(algos) * 2
    rows = list(results) if results is not None else []
    success_rates = [float(r["success_rate"]) for r in rows]
    collision_rates = [
        float(r["collision_rate"]) for r in rows
        if float(r["collision_rate"]) == float(r["collision_rate"])
    ]
    mean_success = (float(sum(success_rates) / len(success_rates))
                    if success_rates else None)
    mean_collision = (float(sum(collision_rates) / len(collision_rates))
                      if collision_rates else None)

    md_lines = [
        f"- **datasets:** {' '.join(str(d) for d in dataset_names)}",
        f"- **seeds:** {' '.join(str(s) for s in seeds)}",
        f"- **algos:** {' '.join(str(a) for a in algos)}",
        f"- **use_class:** 0 1",
        f"- **completed:** {len(rows)}/{expected}",
    ]
    if mean_success is not None:
        md_lines.append(f"- **mean_success_rate:** {mean_success * 100:.2f}%")
    if mean_collision is not None:
        md_lines.append(
            f"- **mean_collision_rate:** {mean_collision * 100:.2f}%")

    if rows:
        md_lines.append("")
        md_lines.append("**per-run**")
        for r in rows:
            md_lines.append(
                f"- `{r['run_name']}`: "
                f"succ={float(r['success_rate']) * 100:.2f}% "
                f"coll={float(r['collision_rate']) * 100:.2f}% "
                f"steps={_fmt_steps(r.get('success_average_steps'))}"
            )

    return send_feishu_notification(
        repo_dir,
        mode="eval",
        title="Maze Evaluation Finished",
        markdown="\n".join(md_lines),
        success_rate=mean_success,
        enabled=enabled,
    )

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REWARD_METRICS = [
    "reward",
    "reward_std",
    "rewards/reward_wrapper/mean",
    "rewards/reward_wrapper/std",
    "frac_reward_zero_std",
    "self_reward/total",
    "self_reward/exact_match",
    "self_reward/formatting",
    "self_reward/verifier",
]

METRIC_LABELS = {
    "reward": "Reward Mean",
    "reward_std": "Reward Std",
    "rewards/reward_wrapper/mean": "Wrapper Reward Mean",
    "rewards/reward_wrapper/std": "Wrapper Reward Std",
    "frac_reward_zero_std": "Frac Reward Zero Std",
    "self_reward/total": "Self Reward Total",
    "self_reward/exact_match": "Self Reward Exact Match",
    "self_reward/formatting": "Self Reward Formatting",
    "self_reward/verifier": "Self Reward Verifier",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Plot train and eval reward trajectories from the latest trainer state."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=repo_root / "outputs",
        help="Directory containing training run outputs.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=repo_root / "plots",
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint directory to read instead of auto-detecting the latest one.",
    )
    return parser.parse_args()


def checkpoint_step(checkpoint_dir: Path) -> int:
    return int(checkpoint_dir.name.split("-")[-1])


def find_latest_checkpoint(outputs_dir: Path) -> Path:
    candidates = sorted(outputs_dir.glob("**/checkpoint-*/trainer_state.json"))
    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json found under {outputs_dir}")

    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime, checkpoint_step(path.parent)),
    ).parent


def load_trainer_state(checkpoint_dir: Path) -> dict:
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    with trainer_state_path.open() as f:
        return json.load(f)


def split_log_history(log_history: list[dict]) -> tuple[list[dict], list[dict]]:
    train_entries = [
        entry
        for entry in log_history
        if "reward" in entry and "eval_reward" not in entry
    ]
    eval_entries = [entry for entry in log_history if "eval_reward" in entry]
    return train_entries, eval_entries


def available_metrics(entries: list[dict], eval_mode: bool) -> list[str]:
    prefix = "eval_" if eval_mode else ""
    metrics = []
    for metric in REWARD_METRICS:
        key = f"{prefix}{metric}"
        if any(key in entry for entry in entries):
            metrics.append(key)
    return metrics


def metric_label(metric: str) -> str:
    base_metric = metric.removeprefix("eval_")
    return METRIC_LABELS.get(base_metric, base_metric)


def plot_metric_grid(
    entries: list[dict],
    metrics: list[str],
    output_path: Path,
    title: str,
) -> None:
    if not metrics:
        raise ValueError(f"No metrics available for {title}")

    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.8 * nrows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, metric in zip(axes_list, metrics, strict=False):
        points = [
            (entry["step"], entry[metric]) for entry in entries if metric in entry
        ]
        steps = [step for step, _ in points]
        values = [value for _, value in points]

        ax.plot(steps, values, marker="o", linewidth=1.8, markersize=3)
        ax.set_title(metric_label(metric))
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)

    for ax in axes_list[len(metrics) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_train_eval_overlay(
    train_entries: list[dict],
    eval_entries: list[dict],
    output_path: Path,
) -> None:
    overlay_metrics = [
        ("reward", "eval_reward"),
        ("reward_std", "eval_reward_std"),
        ("self_reward/total", "eval_self_reward/total"),
        ("self_reward/exact_match", "eval_self_reward/exact_match"),
        ("self_reward/formatting", "eval_self_reward/formatting"),
    ]
    overlay_metrics = [
        pair
        for pair in overlay_metrics
        if any(pair[0] in entry for entry in train_entries)
        and any(pair[1] in entry for entry in eval_entries)
    ]
    if not overlay_metrics:
        return

    ncols = 2
    nrows = math.ceil(len(overlay_metrics) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.8 * nrows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, (train_metric, eval_metric) in zip(
        axes_list, overlay_metrics, strict=False
    ):
        train_points = [
            (entry["step"], entry[train_metric])
            for entry in train_entries
            if train_metric in entry
        ]
        eval_points = [
            (entry["step"], entry[eval_metric])
            for entry in eval_entries
            if eval_metric in entry
        ]
        ax.plot(
            [step for step, _ in train_points],
            [value for _, value in train_points],
            marker="o",
            linewidth=1.8,
            markersize=3,
            label="train",
        )
        ax.plot(
            [step for step, _ in eval_points],
            [value for _, value in eval_points],
            marker="s",
            linewidth=1.8,
            markersize=4,
            label="eval",
        )
        ax.set_title(metric_label(train_metric))
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend()

    for ax in axes_list[len(overlay_metrics) :]:
        ax.axis("off")

    fig.suptitle("Train vs Eval Reward Trajectories")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_dir = (
        Path(args.checkpoint)
        if args.checkpoint
        else find_latest_checkpoint(Path(args.outputs_dir))
    )
    plots_dir = Path(args.plots_dir)

    # Get output directory name
    output_dir_name = checkpoint_dir.parent.name
    plots_dir = plots_dir / output_dir_name

    trainer_state = load_trainer_state(checkpoint_dir)
    train_entries, eval_entries = split_log_history(trainer_state["log_history"])

    train_metrics = available_metrics(train_entries, eval_mode=False)
    eval_metrics = available_metrics(eval_entries, eval_mode=True)

    plot_metric_grid(
        train_entries,
        train_metrics,
        plots_dir / "train_reward_trajectories.png",
        f"Train Reward Trajectories ({checkpoint_dir.name})",
    )
    plot_metric_grid(
        eval_entries,
        eval_metrics,
        plots_dir / "eval_reward_trajectories.png",
        f"Eval Reward Trajectories ({checkpoint_dir.name})",
    )
    plot_train_eval_overlay(
        train_entries,
        eval_entries,
        plots_dir / "train_eval_reward_overlays.png",
    )

    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Saved: {plots_dir / 'train_reward_trajectories.png'}")
    print(f"Saved: {plots_dir / 'eval_reward_trajectories.png'}")
    overlay_path = plots_dir / "train_eval_reward_overlays.png"
    if overlay_path.exists():
        print(f"Saved: {overlay_path}")


if __name__ == "__main__":
    main()

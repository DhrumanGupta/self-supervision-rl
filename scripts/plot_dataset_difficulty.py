from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot the exact-value difficulty distribution for a dataset."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="zwhe99/DeepMath-103K",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional Hugging Face dataset config.",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default="train",
        help="Source split to load before making the seeded 95/5 split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the deterministic 95/5 train/test split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.05,
        help="Fraction of examples assigned to the test split.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=repo_root / "plots",
        help="Directory where plots will be written.",
    )
    return parser.parse_args()


def difficulty_counts(values: list[float]) -> tuple[list[float], list[int]]:
    counts = Counter(float(value) for value in values)
    difficulties = sorted(counts)
    frequencies = [counts[difficulty] for difficulty in difficulties]
    return difficulties, frequencies


def plot_split(ax, values: list[float], title: str) -> None:
    difficulties, frequencies = difficulty_counts(values)
    ax.bar(difficulties, frequencies, width=0.35, align="center")
    ax.plot(difficulties, frequencies, marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Count")
    ax.set_xticks(difficulties)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    args = parse_args()

    dataset = load_dataset(
        args.dataset_name, args.dataset_config, split=args.source_split
    )
    split_datasets = dataset.train_test_split(
        test_size=args.test_fraction,
        seed=args.seed,
    )

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 14), sharex=True)
    plot_split(axes[0], dataset["difficulty"], "Full Dataset Difficulty Distribution")
    plot_split(
        axes[1], split_datasets["train"]["difficulty"], "Train Difficulty Distribution"
    )
    plot_split(
        axes[2], split_datasets["test"]["difficulty"], "Test Difficulty Distribution"
    )
    fig.suptitle(
        f"{args.dataset_name} Difficulty Distribution (seed={args.seed}, test_fraction={args.test_fraction})"
    )
    fig.tight_layout()

    output_dir = args.plots_dir / args.dataset_name.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "difficulty_distribution.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

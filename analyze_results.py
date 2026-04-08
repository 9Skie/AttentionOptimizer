#!/usr/bin/env python3
"""Analyze and visualize training results."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path("logs")
OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_metrics(run_id):
    path = LOG_DIR / run_id / "metrics.jsonl"
    if not path.exists():
        return None
    metrics = []
    with open(path) as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def analyze_run(run_id):
    metrics = load_metrics(run_id)
    if not metrics:
        return None
    losses = [m["loss"] for m in metrics]
    steps = [m["step"] for m in metrics]
    return {
        "run_id": run_id,
        "steps": steps,
        "losses": losses,
        "final_loss": losses[-1],
        "best_loss": min(losses),
    }


def get_optimizer_label(run_id):
    """Group runs by optimizer type."""
    if run_id == "SGD":
        return "SGD"
    elif run_id == "ADAMW":
        return "AdamW"
    elif run_id == "MUON":
        return "Muon"
    elif run_id.startswith("ATTNRAW-V1"):
        return "AttnRaw-v1"
    elif run_id.startswith("ATTNRAW-V2"):
        return "AttnRaw-v2"
    elif run_id.startswith("ATTNRAW-V3"):
        return "AttnRaw-v3"
    elif run_id.startswith("AVG-V1"):
        return "SimpleAvg-v1"
    elif run_id.startswith("AVG-V2"):
        return "SimpleAvg-v2"
    elif run_id.startswith("AVG-V3"):
        return "SimpleAvg-v3"
    return run_id


def get_config_label(run_id):
    """Get the L and T config from run ID."""
    parts = run_id.split("-")
    if len(parts) == 2:
        return run_id
    if len(parts) == 3:
        return parts[1] + "-" + parts[2]
    if len(parts) == 4:
        return parts[2] + "-" + parts[3]
    return run_id


def plot_loss_curves(results, output_path):
    """Plot loss curves for all runs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training Loss Curves by Optimizer", fontsize=16, fontweight="bold")

    optimizer_groups = {
        "Baselines": ["SGD", "ADAMW", "MUON"],
        "v1 variants (keep m+v)": ["AttnRaw-v1", "SimpleAvg-v1"],
        "v2 variants (keep v)": ["AttnRaw-v2", "SimpleAvg-v2"],
        "v3 variants (keep neither)": ["AttnRaw-v3", "SimpleAvg-v3"],
    }

    color_map = {
        "SGD": "#e74c3c",
        "ADAMW": "#3498db",
        "MUON": "#9b59b6",
        "AttnRaw-v1": "#27ae60",
        "AttnRaw-v2": "#2980b9",
        "AttnRaw-v3": "#8e44ad",
        "SimpleAvg-v1": "#2ecc71",
        "SimpleAvg-v2": "#1abc9c",
        "SimpleAvg-v3": "#16a085",
    }

    for ax, (group_name, opt_names) in zip(axes.flat, optimizer_groups.items()):
        for r in results:
            opt = get_optimizer_label(r["run_id"])
            if opt not in opt_names:
                continue
            label = get_config_label(r["run_id"])
            color = color_map.get(opt, "#333333")
            linestyle = (
                "-" if "AttnRaw" in opt or opt in ["SGD", "ADAMW", "MUON"] else "--"
            )
            ax.plot(
                r["steps"],
                r["losses"],
                label=f"{opt} ({label})",
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                alpha=0.8,
            )

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(group_name)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)
        ax.set_ylim(3.5, 6.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_losses(results, output_path):
    """Plot bar chart of final losses by optimizer group."""
    optimizer_groups = {}
    for r in results:
        opt = get_optimizer_label(r["run_id"])
        if opt not in optimizer_groups:
            optimizer_groups[opt] = []
        optimizer_groups[opt].append(r["final_loss"])

    opt_names = list(optimizer_groups.keys())
    means = [np.mean(optimizer_groups[o]) for o in opt_names]
    stds = [np.std(optimizer_groups[o]) for o in opt_names]

    colors = {
        "AttnRaw-v1": "#27ae60",
        "SimpleAvg-v1": "#2ecc71",
        "AdamW": "#3498db",
        "SimpleAvg-v2": "#1abc9c",
        "AttnRaw-v2": "#2980b9",
        "SimpleAvg-v3": "#16a085",
        "AttnRaw-v3": "#8e44ad",
        "Muon": "#9b59b6",
        "SGD": "#e74c3c",
    }

    bar_colors = [colors.get(o, "#333333") for o in opt_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        range(len(opt_names)),
        means,
        yerr=stds,
        capsize=4,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.set_xticks(range(len(opt_names)))
    ax.set_xticklabels(opt_names, rotation=45, ha="right")
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Loss by Optimizer Type (lower is better)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_curves_overlay(results, output_path):
    """Plot all runs overlaid on one graph."""
    fig, ax = plt.subplots(figsize=(14, 8))

    optimizer_colors = {
        "SGD": "#e74c3c",
        "ADAMW": "#3498db",
        "MUON": "#9b59b6",
        "AttnRaw-v1": "#27ae60",
        "AttnRaw-v2": "#2980b9",
        "AttnRaw-v3": "#8e44ad",
        "SimpleAvg-v1": "#2ecc71",
        "SimpleAvg-v2": "#1abc9c",
        "SimpleAvg-v3": "#16a085",
    }

    for r in results:
        opt = get_optimizer_label(r["run_id"])
        color = optimizer_colors.get(opt, "#333333")
        label = f"{opt}"

        if "AttnRaw" in opt or opt in ["SGD", "ADAMW", "MUON"]:
            linestyle = "-"
            linewidth = 1.5
        else:
            linestyle = "--"
            linewidth = 1.2

        ax.plot(
            r["steps"],
            r["losses"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.7,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("All Optimizers - Training Loss Curves", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_table(results, output_path):
    """Generate a markdown table of results."""
    results_sorted = sorted(results, key=lambda x: x["final_loss"])

    lines = [
        "# Training Results Summary\n",
        f"| Rank | Run ID | Final Loss | Best Loss | Optimizer |",
        f"|------:|--------|----------:|----------:|-----------|",
    ]

    for i, r in enumerate(results_sorted, 1):
        opt = get_optimizer_label(r["run_id"])
        lines.append(
            f"| {i} | {r['run_id']} | {r['final_loss']:.6f} | {r['best_loss']:.6f} | {opt} |"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def generate_summary_by_group(results, output_path):
    """Generate summary table by optimizer group."""
    optimizer_groups = {}
    for r in results:
        opt = get_optimizer_label(r["run_id"])
        if opt not in optimizer_groups:
            optimizer_groups[opt] = []
        optimizer_groups[opt].append(r)

    lines = [
        "# Results by Optimizer Type\n",
        "| Optimizer | Runs | Best Final | Worst Final | Mean Final |",
        "|----------:|-----:|----------:|----------:|-----------:|",
    ]

    for opt in sorted(
        optimizer_groups.keys(),
        key=lambda o: min(optimizer_groups[o], key=lambda x: x["final_loss"])[
            "final_loss"
        ],
    ):
        runs = optimizer_groups[opt]
        losses = [r["final_loss"] for r in runs]
        best = min(runs, key=lambda x: x["final_loss"])
        lines.append(
            f"| {opt} | {len(runs)} | {min(losses):.6f} ({best['run_id']}) | {max(losses):.6f} | {np.mean(losses):.6f} |"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def main():
    run_dirs = sorted([d.name for d in LOG_DIR.iterdir() if d.is_dir()])
    results = [analyze_run(rid) for rid in run_dirs]
    results = [r for r in results if r is not None]

    print(f"Loaded {len(results)} runs")

    print("\nGenerating visualizations...")
    plot_loss_curves(results, OUTPUT_DIR / "loss_curves.png")
    plot_final_losses(results, OUTPUT_DIR / "final_losses.png")
    plot_all_curves_overlay(results, OUTPUT_DIR / "all_curves.png")

    print("\nGenerating tables...")
    generate_table(results, OUTPUT_DIR / "results_table.md")
    generate_summary_by_group(results, OUTPUT_DIR / "summary_by_group.md")

    print("\nDone!")


if __name__ == "__main__":
    main()

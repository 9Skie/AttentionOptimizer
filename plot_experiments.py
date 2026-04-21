import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


EXPERIMENT_1_GROUPS = [
    "Baselines",
    "SimpleAvg",
    "AttnRaw V1",
    "AttnRaw V2",
    "AttnRaw V3",
]

EXPERIMENT_2_GROUPS = [
    "V1-G Sweep",
    "AttnRaw Mix Sweep",
    "SimpleAvg Mix Sweep",
]

GROUP_ACCENTS = {
    "Baselines": "#4b5563",
    "Baseline": "#4b5563",
    "SimpleAvg": "#2563eb",
    "AttnRaw": "#ea580c",
    "SimpleAvg V1": "#2563eb",
    "SimpleAvg V2": "#0f766e",
    "SimpleAvg V3": "#0891b2",
    "AttnRaw V1": "#ea580c",
    "AttnRaw V2": "#dc2626",
    "AttnRaw V3": "#7c3aed",
    "V1-G Sweep": "#7c3aed",
    "AttnRaw Mix Sweep": "#ea580c",
    "SimpleAvg Mix Sweep": "#2563eb",
}

GROUP_COLORMAPS = {
    "Baselines": "Greys",
    "Baseline": "Greys",
    "SimpleAvg": "Blues",
    "AttnRaw": "Oranges",
    "SimpleAvg V1": "Blues",
    "SimpleAvg V2": "BuGn",
    "SimpleAvg V3": "PuBu",
    "AttnRaw V1": "Oranges",
    "AttnRaw V2": "Reds",
    "AttnRaw V3": "Purples",
    "V1-G Sweep": "Purples",
    "AttnRaw Mix Sweep": "Oranges",
    "SimpleAvg Mix Sweep": "Blues",
}


def read_loss_curve(metrics_path):
    steps = []
    losses = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))

    if not steps:
        raise ValueError(f"No metrics rows found in {metrics_path}")

    return {
        "steps": steps,
        "losses": losses,
        "final_loss": losses[-1],
        "best_loss": min(losses),
    }


def classify_experiment_1_run(run_id):
    if run_id in {"ADAMW", "MUON", "SGD"}:
        return "Baselines"
    if run_id.startswith("SIMPLEAVG-"):
        return "SimpleAvg"
    if run_id.startswith("ATTNRAW-V1-"):
        return "AttnRaw V1"
    if run_id.startswith("ATTNRAW-V2-"):
        return "AttnRaw V2"
    if run_id.startswith("ATTNRAW-V3-"):
        return "AttnRaw V3"
    return None


def classify_experiment_2_run(run_id):
    if run_id == "SIMPLEAVG-G-V1-L4" or run_id.startswith("ATTNRAW-V1-G-"):
        return "V1-G Sweep"
    if run_id.startswith("ATTNRAW-MIX"):
        return "AttnRaw Mix Sweep"
    if run_id.startswith("SIMPLEAVG-MIX"):
        return "SimpleAvg Mix Sweep"
    return None


def palette_for_group(group_name, count):
    cmap = plt.get_cmap(GROUP_COLORMAPS[group_name])
    if count == 1:
        return [cmap(0.7)]
    positions = np.linspace(0.45, 0.9, count)
    return [cmap(position) for position in positions]


def group_runs(runs, classifier, ordered_groups):
    grouped = {group_name: {} for group_name in ordered_groups}
    for run_id, curve in sorted(runs.items()):
        group_name = classifier(run_id)
        if group_name is None or group_name not in grouped:
            continue
        grouped[group_name][run_id] = curve
    return {group_name: grouped[group_name] for group_name in ordered_groups if grouped[group_name]}


def add_reference_run_to_groups(grouped_runs, run_id, curve, target_groups=None):
    if curve is None:
        return grouped_runs

    with_reference = {}
    for group_name, runs in grouped_runs.items():
        updated_runs = dict(runs)
        if target_groups is None or group_name in target_groups:
            updated_runs[run_id] = curve
        with_reference[group_name] = updated_runs
    return with_reference


def load_experiment_1_runs(experiment_dir):
    runs = {}
    for metrics_path in sorted(Path(experiment_dir).glob("*/metrics.jsonl")):
        runs[metrics_path.parent.name] = read_loss_curve(metrics_path)
    return runs


def align_curves(curves):
    steps = sorted({step for curve in curves for step in curve["steps"]})
    if not steps:
        raise ValueError("Curves do not contain any logged steps")

    aligned_losses = np.full((len(curves), len(steps)), np.nan, dtype=float)
    for curve_index, curve in enumerate(curves):
        by_step = dict(zip(curve["steps"], curve["losses"]))
        for step_index, step in enumerate(steps):
            if step in by_step:
                aligned_losses[curve_index, step_index] = by_step[step]
    return steps, aligned_losses


def aggregate_experiment_2_runs(experiment_dir):
    raw_runs = {}
    for metrics_path in sorted(Path(experiment_dir).glob("seed_*/*/metrics.jsonl")):
        run_id = metrics_path.parent.name
        raw_runs.setdefault(run_id, []).append(read_loss_curve(metrics_path))

    aggregated = {}
    for run_id, curves in raw_runs.items():
        steps, aligned_losses = align_curves(curves)
        mean_loss = np.nanmean(aligned_losses, axis=0)
        std_loss = np.nanstd(aligned_losses, axis=0)
        aggregated[run_id] = {
            "steps": steps,
            "mean_loss": mean_loss.tolist(),
            "std_loss": std_loss.tolist(),
            "final_loss": float(mean_loss[-1]),
            "best_loss": float(mean_loss.min()),
            "seed_count": len(curves),
        }
    return aggregated


def panel_layout(count, horizontal=False):
    if count <= 0:
        raise ValueError("No plot groups were available")
    if horizontal:
        return {"rows": 1, "cols": count}
    cols = 2
    rows = int(np.ceil(count / cols))
    return {"rows": rows, "cols": cols}


def build_subplot_grid(count, horizontal=False):
    if count <= 0:
        raise ValueError("No plot groups were available")
    layout = panel_layout(count, horizontal=horizontal)
    rows = layout["rows"]
    cols = layout["cols"]
    width_per_panel = 5.2 if horizontal else 7.5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * width_per_panel, rows * 4.8), squeeze=False)
    flat_axes = axes.flatten()
    for axis in flat_axes[count:]:
        axis.set_visible(False)
    return fig, flat_axes


def plot_group_panels(grouped_runs, output_path, title, with_variance=False, horizontal=False):
    if not grouped_runs:
        raise ValueError("No recognized runs were found for plotting")

    fig, axes = build_subplot_grid(len(grouped_runs), horizontal=horizontal)
    for axis, (group_name, curves) in zip(axes, grouped_runs.items()):
        colors = palette_for_group(group_name, len(curves))
        for color, (run_id, curve) in zip(colors, sorted(curves.items())):
            y_key = "mean_loss" if with_variance else "losses"
            x_values = curve["steps"]
            y_values = np.asarray(curve[y_key], dtype=float)
            axis.plot(x_values, y_values, label=run_id, color=color, linewidth=2)
            if with_variance:
                std_values = np.asarray(curve["std_loss"], dtype=float)
                axis.fill_between(x_values, y_values - std_values, y_values + std_values, color=color, alpha=0.18)

        axis.set_title(group_name)
        axis.set_xlabel("Step")
        axis.set_ylabel("Training loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def select_best_runs(grouped_runs):
    best = {}
    for group_name, curves in grouped_runs.items():
        best[group_name] = min(curves.items(), key=lambda item: item[1]["final_loss"])
    return best


def plot_best_runs(best_runs, output_path, title, with_variance=False):
    if not best_runs:
        raise ValueError("No best runs were available for plotting")

    fig, axis = plt.subplots(figsize=(10.5, 6))
    for group_name, (run_id, curve) in best_runs.items():
        y_key = "mean_loss" if with_variance else "losses"
        x_values = curve["steps"]
        y_values = np.asarray(curve[y_key], dtype=float)
        color = GROUP_ACCENTS[group_name]
        axis.plot(x_values, y_values, label=f"{group_name}: {run_id}", color=color, linewidth=2.5)
        if with_variance:
            std_values = np.asarray(curve["std_loss"], dtype=float)
            axis.fill_between(x_values, y_values - std_values, y_values + std_values, color=color, alpha=0.18)

    axis.set_title(title)
    axis.set_xlabel("Step")
    axis.set_ylabel("Training loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_experiment_1_plots(experiment_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = load_experiment_1_runs(experiment_dir)
    grouped_runs = group_runs(all_runs, classify_experiment_1_run, EXPERIMENT_1_GROUPS)
    if not grouped_runs:
        raise ValueError(f"No recognized Experiment 1 runs found in {experiment_dir}")

    panel_runs = add_reference_run_to_groups(
        grouped_runs,
        "ADAMW",
        all_runs.get("ADAMW"),
        {"SimpleAvg", "AttnRaw V1", "AttnRaw V2", "AttnRaw V3"},
    )
    layout = panel_layout(len(panel_runs), horizontal=True)

    panel_path = output_dir / "experiment_1_training_loss_by_optimizer.png"
    best_path = output_dir / "experiment_1_best_by_group.png"
    plot_group_panels(panel_runs, panel_path, "Experiment 1: Training Loss by Optimizer Family", horizontal=True)

    best_runs = select_best_runs(grouped_runs)
    plot_best_runs(best_runs, best_path, "Experiment 1: Best Curve from Each Optimizer Family")

    return {
        "panel_path": panel_path,
        "best_path": best_path,
        "panel_layout": layout,
        "panel_run_ids": {group_name: list(sorted(runs)) for group_name, runs in panel_runs.items()},
        "best_run_ids": {group_name: run_id for group_name, (run_id, _) in best_runs.items()},
    }


def generate_experiment_2_plots(experiment_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated_runs = aggregate_experiment_2_runs(experiment_dir)
    sweep_runs = group_runs(aggregated_runs, classify_experiment_2_run, EXPERIMENT_2_GROUPS)
    if not sweep_runs:
        raise ValueError(f"No recognized Experiment 2 runs found in {experiment_dir}")

    panel_runs = add_reference_run_to_groups(sweep_runs, "ADAMW", aggregated_runs.get("ADAMW"))
    layout = panel_layout(len(panel_runs), horizontal=True)

    panel_path = output_dir / "experiment_2_training_loss_by_optimizer.png"
    best_path = output_dir / "experiment_2_best_by_group.png"
    plot_group_panels(
        panel_runs,
        panel_path,
        "Experiment 2: Training Loss by Optimizer Family (mean +/- SD across seeds)",
        with_variance=True,
        horizontal=True,
    )

    best_runs = select_best_runs(sweep_runs)
    plot_best_runs(
        best_runs,
        best_path,
        "Experiment 2: Best Mean Curve from Each Optimizer Family (mean +/- SD across seeds)",
        with_variance=True,
    )

    return {
        "panel_path": panel_path,
        "best_path": best_path,
        "panel_layout": layout,
        "panel_run_ids": {group_name: list(sorted(runs)) for group_name, runs in panel_runs.items()},
        "best_run_ids": {group_name: run_id for group_name, (run_id, _) in best_runs.items()},
    }


def generate_all_plots(logs_root, output_dir):
    logs_root = Path(logs_root)
    output_dir = Path(output_dir)
    return {
        "experiment_1": generate_experiment_1_plots(logs_root / "experiment_1", output_dir),
        "experiment_2": generate_experiment_2_plots(logs_root / "experiment_2", output_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate experiment training-loss plots from metrics.jsonl logs.")
    parser.add_argument("--logs-root", default="logs", help="Directory containing experiment_1/ and experiment_2/")
    parser.add_argument("--output-dir", default="assets", help="Directory for generated plot files")
    args = parser.parse_args()

    results = generate_all_plots(args.logs_root, args.output_dir)
    for experiment_name, experiment_result in results.items():
        print(f"{experiment_name}: {experiment_result['panel_path']}")
        print(f"{experiment_name}: {experiment_result['best_path']}")


if __name__ == "__main__":
    main()

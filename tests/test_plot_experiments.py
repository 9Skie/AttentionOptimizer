import json
import tempfile
import unittest
from pathlib import Path

from plot_experiments import (
    aggregate_experiment_2_runs,
    generate_experiment_1_plots,
    generate_experiment_2_plots,
)


def write_metrics(run_dir: Path, losses):
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for index, loss in enumerate(losses, start=1):
            row = {"step": index * 25, "loss": loss, "lr": 1e-4, "tokens_per_sec": 1000}
            handle.write(json.dumps(row) + "\n")


def write_metric_rows(run_dir: Path, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for step, loss in rows:
            row = {"step": step, "loss": loss, "lr": 1e-4, "tokens_per_sec": 1000}
            handle.write(json.dumps(row) + "\n")


class PlotExperimentsTest(unittest.TestCase):
    def test_generate_experiment_1_plots_creates_group_and_best_figures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exp1_dir = root / "logs" / "experiment_1"
            output_dir = root / "assets"

            write_metrics(exp1_dir / "ADAMW", [5.0, 4.1, 4.0])
            write_metrics(exp1_dir / "MUON", [5.1, 4.5, 4.3])
            write_metrics(exp1_dir / "SIMPLEAVG-V1-L4", [5.0, 4.0, 3.9])
            write_metrics(exp1_dir / "SIMPLEAVG-V1-L8", [5.0, 3.9, 3.8])
            write_metrics(exp1_dir / "ATTNRAW-V1-L4-T1.0", [5.0, 3.8, 3.7])
            write_metrics(exp1_dir / "ATTNRAW-V1-L8-T1.0", [5.0, 3.7, 3.6])
            write_metrics(exp1_dir / "ATTNRAW-V2-L4-T1.0", [5.1, 4.2, 4.0])
            write_metrics(exp1_dir / "ATTNRAW-V3-L4-T1.0", [5.2, 4.4, 4.2])

            result = generate_experiment_1_plots(exp1_dir, output_dir)

            self.assertTrue((output_dir / "experiment_1_training_loss_by_optimizer.png").exists())
            self.assertTrue((output_dir / "experiment_1_best_by_group.png").exists())
            self.assertEqual(result["panel_layout"], {"rows": 1, "cols": 5})
            self.assertEqual(
                set(result["panel_run_ids"]),
                {"Baselines", "SimpleAvg", "AttnRaw V1", "AttnRaw V2", "AttnRaw V3"},
            )
            self.assertIn("ADAMW", result["panel_run_ids"]["SimpleAvg"])
            self.assertIn("ADAMW", result["panel_run_ids"]["AttnRaw V1"])
            self.assertIn("ADAMW", result["panel_run_ids"]["AttnRaw V2"])
            self.assertIn("ADAMW", result["panel_run_ids"]["AttnRaw V3"])
            self.assertEqual(result["best_run_ids"]["Baselines"], "ADAMW")
            self.assertEqual(result["best_run_ids"]["SimpleAvg"], "SIMPLEAVG-V1-L8")
            self.assertEqual(result["best_run_ids"]["AttnRaw V1"], "ATTNRAW-V1-L8-T1.0")
            self.assertEqual(result["best_run_ids"]["AttnRaw V2"], "ATTNRAW-V2-L4-T1.0")
            self.assertEqual(result["best_run_ids"]["AttnRaw V3"], "ATTNRAW-V3-L4-T1.0")

    def test_aggregate_experiment_2_runs_computes_mean_and_std_without_dropping_late_steps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exp2_dir = root / "logs" / "experiment_2"

            write_metric_rows(exp2_dir / "seed_1" / "ADAMW", [(25, 5.0), (50, 4.0), (75, 3.0)])
            write_metric_rows(exp2_dir / "seed_2" / "ADAMW", [(25, 5.4), (50, 4.2)])

            aggregated = aggregate_experiment_2_runs(exp2_dir)

            adamw = aggregated["ADAMW"]
            self.assertEqual(adamw["steps"], [25, 50, 75])
            self.assertEqual(adamw["seed_count"], 2)
            self.assertAlmostEqual(adamw["mean_loss"][0], 5.2)
            self.assertAlmostEqual(adamw["mean_loss"][1], 4.1)
            self.assertAlmostEqual(adamw["mean_loss"][2], 3.0)
            self.assertAlmostEqual(adamw["std_loss"][0], 0.2)
            self.assertAlmostEqual(adamw["std_loss"][1], 0.1)
            self.assertAlmostEqual(adamw["std_loss"][2], 0.0)
            self.assertNotIn("variance_loss", adamw)

    def test_generate_experiment_2_plots_creates_mean_variance_and_best_figures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exp2_dir = root / "logs" / "experiment_2"
            output_dir = root / "assets"

            for seed_name, offset in (("seed_1", 0.0), ("seed_2", 0.2)):
                seed_dir = exp2_dir / seed_name
                write_metrics(seed_dir / "ADAMW", [5.0 + offset, 4.2 + offset])
                write_metrics(seed_dir / "SIMPLEAVG-G-V1-L4", [4.8 + offset, 4.0 + offset])
                write_metrics(seed_dir / "ATTNRAW-V1-G-L4-T1.0", [4.7 + offset, 3.8 + offset])
                write_metrics(seed_dir / "ATTNRAW-MIX50-L4-T1.0", [4.9 + offset, 4.1 + offset])
                write_metrics(seed_dir / "ATTNRAW-MIX90-L4-T1.0", [4.8 + offset, 3.9 + offset])
                write_metrics(seed_dir / "SIMPLEAVG-MIX25-L4-T1.0", [5.0 + offset, 4.3 + offset])
                write_metrics(seed_dir / "SIMPLEAVG-MIX90-L4-T1.0", [4.9 + offset, 4.0 + offset])

            result = generate_experiment_2_plots(exp2_dir, output_dir)

            self.assertTrue((output_dir / "experiment_2_training_loss_by_optimizer.png").exists())
            self.assertTrue((output_dir / "experiment_2_best_by_group.png").exists())
            self.assertEqual(result["panel_layout"], {"rows": 1, "cols": 3})
            self.assertEqual(set(result["panel_run_ids"]), {"V1-G Sweep", "AttnRaw Mix Sweep", "SimpleAvg Mix Sweep"})
            self.assertIn("ADAMW", result["panel_run_ids"]["V1-G Sweep"])
            self.assertIn("ADAMW", result["panel_run_ids"]["AttnRaw Mix Sweep"])
            self.assertIn("ADAMW", result["panel_run_ids"]["SimpleAvg Mix Sweep"])
            self.assertEqual(result["best_run_ids"]["V1-G Sweep"], "ATTNRAW-V1-G-L4-T1.0")
            self.assertEqual(result["best_run_ids"]["AttnRaw Mix Sweep"], "ATTNRAW-MIX90-L4-T1.0")
            self.assertEqual(result["best_run_ids"]["SimpleAvg Mix Sweep"], "SIMPLEAVG-MIX90-L4-T1.0")

    def test_generate_experiment_1_plots_rejects_empty_experiment_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exp1_dir = root / "logs" / "experiment_1"
            output_dir = root / "assets"
            exp1_dir.mkdir(parents=True, exist_ok=True)

            with self.assertRaises(ValueError):
                generate_experiment_1_plots(exp1_dir, output_dir)

    def test_generate_experiment_2_plots_rejects_empty_experiment_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exp2_dir = root / "logs" / "experiment_2"
            output_dir = root / "assets"
            exp2_dir.mkdir(parents=True, exist_ok=True)

            with self.assertRaises(ValueError):
                generate_experiment_2_plots(exp2_dir, output_dir)


if __name__ == "__main__":
    unittest.main()

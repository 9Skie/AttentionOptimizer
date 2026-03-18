import argparse

import torch

from configs.runs import MODEL_CONFIG, RUNS, TRAIN_CONFIG
from train import build_model, build_optimizer


def _tiny_model_override():
    return {
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "vocab_size": 1024,
        "block_size": 128,
    }


def run_preflight(run_id: str, device: str, tiny: bool):
    run_cfg = RUNS[run_id]
    model_override = _tiny_model_override() if tiny else None
    active_model_cfg = model_override or MODEL_CONFIG
    seq_len = min(TRAIN_CONFIG["seq_len"], active_model_cfg["block_size"])

    torch.manual_seed(TRAIN_CONFIG["seed"])
    if device == "cuda":
        torch.cuda.manual_seed(TRAIN_CONFIG["seed"])

    model = build_model(run_cfg, model_config_override=model_override).to(device)
    optimizer = build_optimizer(model, run_cfg)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    x = torch.randint(
        low=0,
        high=active_model_cfg["vocab_size"],
        size=(2, seq_len),
        device=device,
    )
    y = torch.randint(
        low=0,
        high=active_model_cfg["vocab_size"],
        size=(2, seq_len),
        device=device,
    )

    autocast_enabled = device == "cuda"
    autocast_ctx = torch.autocast(
        device_type=device,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    )
    with autocast_ctx:
        _, loss = model(x, y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
    optimizer.step()

    return float(loss.detach().cpu())


def main():
    parser = argparse.ArgumentParser(description="Cheap one-step synthetic preflight for all runs.")
    parser.add_argument("--run_id", action="append", help="Optional run_id to test. Repeat for multiple runs.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tiny", action="store_true", help="Use a much smaller model for faster CPU smoke tests.")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")

    run_ids = args.run_id or list(RUNS.keys())
    failures = []

    print(f"Running synthetic preflight on {device} for {len(run_ids)} run(s)...")
    for run_id in run_ids:
        try:
            loss = run_preflight(run_id, device=device, tiny=args.tiny)
            print(f"[ok] {run_id:7s} loss={loss:.4f}")
        except Exception as exc:
            failures.append((run_id, exc))
            print(f"[fail] {run_id:7s} {type(exc).__name__}: {exc}")

    if failures:
        raise SystemExit(
            "Preflight failed for: " + ", ".join(run_id for run_id, _ in failures)
        )

    print("Preflight completed without errors.")


if __name__ == "__main__":
    main()

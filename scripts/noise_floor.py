"""
Measure the empirical noise floor of the canonical baseline.

Runs the unchanged baseline (empty patch) N times with different seeds,
captures val_bpb under hidden eval each time, and reports mean, stddev,
and a suggested "decisively beats the king" margin (default: 2 * stddev,
the §5.7 noise-floor margin).

In Phase 0 this calibrates the validator's accept-threshold on the launch
track. In production every track runs its own version of this script at
commissioning.

Usage:
    python scripts/noise_floor.py --runs 10 --base-seed 1000
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import run_hidden_eval
from model import KarpathianBase, KarpathianConfig
from proof.runner import run_proof_test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--karpathian-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--config", default="configs/proxy_cpu_smoke.json")
    p.add_argument("--out-dir", type=Path, default=Path("runs/noise_floor"))
    p.add_argument("--keep-bundles", action="store_true", help="don't delete intermediate run dirs")
    args = p.parse_args()

    karpathian_root = args.karpathian_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    val_bpbs: list[float] = []
    benchmarks: list[float] = []
    wall_clocks: list[float] = []

    results = []
    for i in range(args.runs):
        seed = args.base_seed + i
        run_id = f"baseline_seed{seed}"
        sub_dir = karpathian_root / "submissions" / f"noise_{run_id}"
        proof_dir = args.out_dir / f"proof_{run_id}"
        if proof_dir.exists():
            shutil.rmtree(proof_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / "patch.diff").write_text("")
        nonce = "0x" + secrets.token_hex(32)
        proof_request = {
            "handshake_nonce": nonce,
            "seed": seed,
            "config_path": args.config,
            "miner_hotkey": "5NoiseFloorCalib",
        }
        (sub_dir / "proof_request.json").write_text(json.dumps(proof_request))

        print(f"\n[noise_floor] run {i + 1}/{args.runs} (seed={seed})")
        t0 = time.time()
        bundle = run_proof_test(
            karpathian_root=karpathian_root,
            submission_dir=sub_dir,
            out_dir=proof_dir,
        )
        wall = time.time() - t0

        # Hidden eval the produced checkpoint.
        ckpt = torch.load(bundle.checkpoint_path, weights_only=False, map_location="cpu")
        saved = ckpt["config"]
        cfg = KarpathianConfig(
            vocab_size=saved["vocab_size"],
            dim=saved["dim"],
            n_layers=saved["n_layers"],
            n_heads=saved["n_heads"],
            head_dim=saved["head_dim"],
            ffn_mult=saved["ffn_mult"],
            max_seq_len=saved["max_seq_len"],
        )
        model = KarpathianBase(cfg)
        model.load_state_dict(ckpt["model"])
        if torch.cuda.is_available():
            model = model.cuda()
        eval_result = run_hidden_eval(model, karpathian_root / "eval" / "private", seq_len=cfg.max_seq_len // 2)

        val_bpbs.append(eval_result.val_bpb)
        benchmarks.append(eval_result.benchmark_accuracy)
        wall_clocks.append(wall)
        results.append({
            "seed": seed,
            "val_bpb": eval_result.val_bpb,
            "benchmark_accuracy": eval_result.benchmark_accuracy,
            "wall_clock_s": wall,
            "bundle_hash": bundle.bundle_hash,
        })

        if not args.keep_bundles:
            shutil.rmtree(proof_dir)

    val_mean = statistics.fmean(val_bpbs)
    val_std = statistics.pstdev(val_bpbs) if len(val_bpbs) > 1 else 0.0
    bench_mean = statistics.fmean(benchmarks)
    bench_std = statistics.pstdev(benchmarks) if len(benchmarks) > 1 else 0.0
    suggested_margin = 2 * val_std

    summary = {
        "runs": args.runs,
        "config": args.config,
        "val_bpb": {
            "mean": val_mean,
            "std": val_std,
            "min": min(val_bpbs),
            "max": max(val_bpbs),
            "values": val_bpbs,
        },
        "benchmark_accuracy": {
            "mean": bench_mean,
            "std": bench_std,
            "values": benchmarks,
        },
        "wall_clock_s": {
            "mean": statistics.fmean(wall_clocks),
            "total": sum(wall_clocks),
        },
        "suggested_noise_floor_margin": suggested_margin,
        "results": results,
    }
    summary_path = args.out_dir / "noise_floor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"NOISE FLOOR CALIBRATION — {args.runs} baseline runs")
    print("=" * 60)
    print(f"val_bpb:    mean={val_mean:.4f}  std={val_std:.4f}  min={min(val_bpbs):.4f}  max={max(val_bpbs):.4f}")
    print(f"benchmark:  mean={bench_mean:.4f}  std={bench_std:.4f}")
    print(f"wall-clock: mean={statistics.fmean(wall_clocks):.2f}s  total={sum(wall_clocks):.2f}s")
    print(f"\nsuggested noise-floor margin (2σ): {suggested_margin:.4f} val_bpb")
    print(f"written to: {summary_path}")


if __name__ == "__main__":
    main()

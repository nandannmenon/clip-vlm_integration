"""Orchestration script for Stages 24-28: CLIP + CoMA continual learning experiments.

Stages:
  Stage 24: Zero-Shot CLIP Baseline (ZERO_SHOT_CLIP)
  Stage 25: Ordinary Sequential CLIP Baseline (SEQUENTIAL_CLIP_ALPHA_1.0)
  Stage 26: CLIP + CoMA Core Experiment (CLIP_COMA_ALPHA_0.7)
  Stage 27: Core Method Comparison (consolidated table & matrix analysis)
  Stage 28: CoMA Alpha Sensitivity Sweep (alphas 0.25, 0.50, 0.70, 0.90, 1.00)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import certifi
import numpy as np
import torch

os.environ["SSL_CERT_FILE"] = certifi.where()

from clip_models import CLIPClassifier
from clip_tasks import get_clip_cifar100_tasks
from clip_coma import run_clip_coma, run_zero_shot_clip
from metrics import calculate_metrics


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def run_stage_24_zero_shot(device, task_count=10, batch_size=8, seed=42, output_dir="results/zero_shot_clip"):
    print("\n" + "=" * 60)
    print("STAGE 24: ZERO-SHOT CLIP BASELINE")
    print("=" * 60)
    set_seed(seed)

    config = {
        "method": "ZERO_SHOT_CLIP",
        "alpha": None,
        "task_count": task_count,
        "batch_size": batch_size,
        "epochs": 0,
        "learning_rate": 0.0,
        "seed": seed,
        "device": str(device),
    }

    # Always initialize fresh pretrained CLIP
    clip = CLIPClassifier(device=device)
    tasks = get_clip_cifar100_tasks(clip.preprocess, batch_size=batch_size,
                                    task_count=task_count, seed=seed,
                                    num_workers=0)

    start_time = time.time()
    matrix, metrics = run_zero_shot_clip(tasks, clip, device=device,
                                         checkpoint_dir=output_dir, config=config)
    elapsed = time.time() - start_time
    print(f"\nStage 24 finished in {elapsed:.1f}s.")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if device.type == "mps":
        torch.mps.empty_cache()

    return matrix, metrics


def run_single_experiment(method_name, alpha, device, task_count=10, batch_size=8,
                           epochs=1, lr=1e-5, seed=42, output_dir=None):
    print("\n" + "=" * 60)
    print(f"RUNNING EXPERIMENT: {method_name} (alpha={alpha})")
    print("=" * 60)
    set_seed(seed)

    config = {
        "method": method_name,
        "alpha": alpha,
        "task_count": task_count,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "seed": seed,
        "device": str(device),
    }

    # Always initialize fresh pretrained CLIP
    clip = CLIPClassifier(device=device)
    tasks = get_clip_cifar100_tasks(clip.preprocess, batch_size=batch_size,
                                    task_count=task_count, seed=seed,
                                    num_workers=0)

    start_time = time.time()
    accuracy_matrix = run_clip_coma(tasks, clip, epochs=epochs, alpha=alpha,
                                    device=device, lr=lr,
                                    checkpoint_dir=output_dir, resume=True,
                                    config=config)
    elapsed = time.time() - start_time
    metrics = calculate_metrics(accuracy_matrix)

    print(f"\n{method_name} completed in {elapsed:.1f}s.")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if device.type == "mps":
        torch.mps.empty_cache()

    return accuracy_matrix, metrics


def compare_core_methods(results_dict, output_path="results/core_comparison.json"):
    print("\n" + "=" * 60)
    print("STAGE 27: CORE METHOD COMPARISON")
    print("=" * 60)

    # Core methods: Zero-Shot, Sequential (alpha=1.0), CoMA (alpha=0.7)
    comparison_data = {}
    print(f"\n{'Method':<28} | {'Alpha':<6} | {'Avg Acc':<10} | {'Forgetting':<10} | {'Plasticity':<10}")
    print("-" * 75)

    table_rows = []
    for key, data in results_dict.items():
        alpha_str = f"{data['alpha']:.2f}" if data['alpha'] is not None else "N/A"
        metrics = data["metrics"]
        avg_acc = metrics["Average Accuracy"]
        forgetting = metrics["Average Forgetting"]
        plasticity = metrics["Plasticity"]

        row = {
            "method": data["method"],
            "alpha": data["alpha"],
            "average_accuracy": avg_acc,
            "forgetting": forgetting,
            "plasticity": plasticity,
            "final_task_accuracies": [data["accuracy_matrix"][i][-1] for i in range(len(data["accuracy_matrix"]))],
        }
        comparison_data[key] = row
        table_rows.append(row)
        print(f"{data['method']:<28} | {alpha_str:<6} | {avg_acc:<10.4f} | {forgetting:<10.4f} | {plasticity:<10.4f}")

    # Comparative analysis observations
    seq_acc = comparison_data.get("sequential", {}).get("average_accuracy", 0.0)
    coma_acc = comparison_data.get("coma_0.7", {}).get("average_accuracy", 0.0)
    seq_fgt = comparison_data.get("sequential", {}).get("forgetting", 0.0)
    coma_fgt = comparison_data.get("coma_0.7", {}).get("forgetting", 0.0)
    seq_pla = comparison_data.get("sequential", {}).get("plasticity", 0.0)
    coma_pla = comparison_data.get("coma_0.7", {}).get("plasticity", 0.0)

    analysis = {
        "improves_average_accuracy": bool(coma_acc > seq_acc),
        "accuracy_delta": float(coma_acc - seq_acc),
        "reduces_forgetting": bool(coma_fgt < seq_fgt),
        "forgetting_delta": float(coma_fgt - seq_fgt),
        "plasticity_delta": float(coma_pla - seq_pla),
    }
    comparison_data["analysis"] = analysis

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nSaved core comparison results to {output_path}")

    return comparison_data


def run_alpha_sweep(device, baseline_results, sweep_alphas=[0.25, 0.50, 0.70, 0.90, 1.00],
                    output_path="results/alpha_sensitivity_sweep.json"):
    print("\n" + "=" * 60)
    print("STAGE 28: CoMA ALPHA SENSITIVITY SWEEP")
    print("=" * 60)

    sweep_results = {}
    for alpha in sweep_alphas:
        # Check if already completed in earlier stages
        if abs(alpha - 1.0) < 1e-6 and "sequential" in baseline_results:
            print(f"\nReusing completed Sequential CLIP baseline for alpha = 1.00")
            data = baseline_results["sequential"]
            sweep_results[alpha] = {
                "alpha": 1.00,
                "method": "SEQUENTIAL_CLIP_ALPHA_1.0",
                "metrics": data["metrics"],
                "accuracy_matrix": data["accuracy_matrix"],
                "checkpoint_dir": "results/sequential_clip_alpha_1.0",
            }
        elif abs(alpha - 0.70) < 1e-6 and "coma_0.7" in baseline_results:
            print(f"\nReusing completed Core CoMA experiment for alpha = 0.70")
            data = baseline_results["coma_0.7"]
            sweep_results[alpha] = {
                "alpha": 0.70,
                "method": "CLIP_COMA_ALPHA_0.7",
                "metrics": data["metrics"],
                "accuracy_matrix": data["accuracy_matrix"],
                "checkpoint_dir": "results/clip_coma_alpha_0.7",
            }
        else:
            method_name = f"CLIP_COMA_ALPHA_{alpha:.2f}"
            ckpt_dir = f"results/clip_coma_alpha_{alpha:.2f}"
            matrix, metrics = run_single_experiment(
                method_name=method_name,
                alpha=alpha,
                device=device,
                task_count=10,
                batch_size=8,
                epochs=1,
                lr=1e-5,
                seed=42,
                output_dir=ckpt_dir,
            )
            sweep_results[alpha] = {
                "alpha": alpha,
                "method": method_name,
                "metrics": metrics,
                "accuracy_matrix": matrix,
                "checkpoint_dir": ckpt_dir,
            }

    print("\n" + "-" * 60)
    print("ALPHA SENSITIVITY TABLE:")
    print(f"{'Alpha':<8} | {'Average Accuracy':<18} | {'Forgetting':<14} | {'Plasticity':<12}")
    print("-" * 60)

    best_acc_alpha = None
    best_acc_val = -1.0
    best_fgt_alpha = None
    best_fgt_val = 999.0

    summary_rows = []
    for alpha in sorted(sweep_results.keys()):
        m = sweep_results[alpha]["metrics"]
        acc = m["Average Accuracy"]
        fgt = m["Average Forgetting"]
        pla = m["Plasticity"]
        summary_rows.append({
            "alpha": alpha,
            "average_accuracy": acc,
            "forgetting": fgt,
            "plasticity": pla,
        })
        print(f"{alpha:<8.2f} | {acc:<18.4f} | {fgt:<14.4f} | {pla:<12.4f}")

        if acc > best_acc_val:
            best_acc_val = acc
            best_acc_alpha = alpha
        if fgt < best_fgt_val:
            best_fgt_val = fgt
            best_fgt_alpha = alpha

    print("-" * 60)
    print(f"Best alpha for Average Accuracy: {best_acc_alpha:.2f} ({best_acc_val:.4f})")
    print(f"Best alpha for Forgetting:        {best_fgt_alpha:.2f} ({best_fgt_val:.4f})")

    final_payload = {
        "sweep_rows": summary_rows,
        "best_alpha_accuracy": {"alpha": best_acc_alpha, "value": best_acc_val},
        "best_alpha_forgetting": {"alpha": best_fgt_alpha, "value": best_fgt_val},
        "full_results": {str(k): v for k, v in sweep_results.items()},
    }

    with open(output_path, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"Saved alpha sensitivity sweep results to {output_path}")

    return final_payload


def main():
    parser = argparse.ArgumentParser(description="Run Stages 24-28 CLIP + CoMA Experiments")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "24", "25", "26", "27", "28"],
                        help="Stage to execute (or 'all' for complete pipeline)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device selected: {device}")
    if device.type != "mps" and not torch.cuda.is_available():
        print("WARNING: Neither MPS nor CUDA is active. Fine-tuning on CPU will be extremely slow.")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Dictionary to collect results across stages
    stage_results = {}

    # Helper to load existing results if skipping or resuming
    def load_if_exists(json_path):
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                return {
                    "method": data.get("method"),
                    "alpha": data.get("alpha"),
                    "metrics": data.get("metrics"),
                    "accuracy_matrix": {int(k): v for k, v in data.get("accuracy_matrix", {}).items()},
                }
        return None

    # Stage 24
    if args.stage in ["all", "24"]:
        matrix24, metrics24 = run_stage_24_zero_shot(device, task_count=10, batch_size=8, seed=42)
        stage_results["zero_shot"] = {
            "method": "ZERO_SHOT_CLIP",
            "alpha": None,
            "metrics": metrics24,
            "accuracy_matrix": matrix24,
        }
    else:
        existing = load_if_exists("results/zero_shot_clip/results.json")
        if existing:
            stage_results["zero_shot"] = existing

    # Stage 25
    if args.stage in ["all", "25"]:
        matrix25, metrics25 = run_single_experiment(
            method_name="SEQUENTIAL_CLIP_ALPHA_1.0",
            alpha=1.0,
            device=device,
            task_count=10,
            batch_size=8,
            epochs=1,
            lr=1e-5,
            seed=42,
            output_dir="results/sequential_clip_alpha_1.0",
        )
        stage_results["sequential"] = {
            "method": "SEQUENTIAL_CLIP_ALPHA_1.0",
            "alpha": 1.0,
            "metrics": metrics25,
            "accuracy_matrix": matrix25,
        }
    else:
        existing = load_if_exists("results/sequential_clip_alpha_1.0/results.json")
        if existing:
            stage_results["sequential"] = existing

    # Stage 26
    if args.stage in ["all", "26"]:
        matrix26, metrics26 = run_single_experiment(
            method_name="CLIP_COMA_ALPHA_0.7",
            alpha=0.7,
            device=device,
            task_count=10,
            batch_size=8,
            epochs=1,
            lr=1e-5,
            seed=42,
            output_dir="results/clip_coma_alpha_0.7",
        )
        stage_results["coma_0.7"] = {
            "method": "CLIP_COMA_ALPHA_0.7",
            "alpha": 0.7,
            "metrics": metrics26,
            "accuracy_matrix": matrix26,
        }
    else:
        existing = load_if_exists("results/clip_coma_alpha_0.7/results.json")
        if existing:
            stage_results["coma_0.7"] = existing

    # Stage 27
    if args.stage in ["all", "27"]:
        compare_core_methods(stage_results, output_path="results/core_comparison.json")

    # Stage 28
    if args.stage in ["all", "28"]:
        run_alpha_sweep(device, stage_results, sweep_alphas=[0.25, 0.50, 0.70, 0.90, 1.00],
                        output_path="results/alpha_sensitivity_sweep.json")

    print("\n" + "=" * 60)
    print("ALL REQUESTED EXPERIMENT STAGES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

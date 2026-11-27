"""Task-incremental CLIP fine-tuning with CoMA parameter averaging."""
from __future__ import annotations

import copy
import glob
import json
import os
import torch
import torch.nn.functional as F

from metrics import calculate_metrics


def average_visual_parameters(previous_average, current_model, alpha: float):
    """Return previous full state with visual parameters updated by CoMA.

    theta_bar_t = alpha * theta_t + (1-alpha) * theta_bar_(t-1)
    Only named parameters in the CLIP visual tower are averaged; text tower,
    logit scale, and non-parameter buffers are retained from the current model.
    """
    averaged = copy.deepcopy(current_model.state_dict())
    current_parameters = dict(current_model.named_parameters())
    for name, current_value in current_parameters.items():
        if not name.startswith("visual."):
            continue
        old_value = previous_average[name].to(current_value.device)
        averaged[name] = alpha * current_value.detach() + (1.0 - alpha) * old_value
    return averaged


def train_one_task(clip, train_loader, class_names, epochs, device, lr=1e-5,
                   weight_decay=0.1, text_features=None):
    clip.train_mode()
    if text_features is None:
        text_features = clip.make_text_features(class_names)

    optimizer = torch.optim.AdamW(clip.trainable_parameters(), lr=lr,
                                  weight_decay=weight_decay)
    for epoch in range(epochs):
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = clip.image_text_logits(images, text_features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        print(f"    epoch {epoch + 1}/{epochs} | loss={running_loss / max(seen, 1):.4f}")
    return text_features


@torch.no_grad()
def evaluate_task(clip, test_loader, class_names, device, text_features=None):
    clip.model.eval()
    if text_features is None:
        text_features = clip.make_text_features(class_names)
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = clip.image_text_logits(images, text_features)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def run_zero_shot_clip(tasks, clip, device="cuda", checkpoint_dir=None, config=None):
    """Evaluate pretrained Zero-Shot CLIP model on all 10 tasks without training."""
    device = torch.device(device)
    task_count = len(tasks)
    clip.model.eval()

    results_file = os.path.join(checkpoint_dir, "results.json") if checkpoint_dir else None
    if results_file and os.path.exists(results_file):
        print(f"Loaded existing Zero-Shot results from {results_file}")
        with open(results_file, "r") as f:
            data = json.load(f)
            matrix = {int(k): v for k, v in data["accuracy_matrix"].items()}
            return matrix, data["metrics"]

    text_features = {
        task_id: clip.make_text_features(task["class_names"])
        for task_id, task in tasks.items()
    }

    task_accuracies = []
    print("\nEvaluating Zero-Shot CLIP across all tasks:")
    for task_id in range(task_count):
        acc = evaluate_task(clip, tasks[task_id]["test"], tasks[task_id]["class_names"],
                            device, text_features[task_id])
        task_accuracies.append(acc)
        print(f"    task {task_id} zero-shot accuracy: {acc:.4f}")

    avg_accuracy = sum(task_accuracies) / max(task_count, 1)
    accuracy_matrix = {task_id: [acc] for task_id, acc in enumerate(task_accuracies)}
    metrics = {
        "Average Accuracy": avg_accuracy,
        "Average Forgetting": 0.0,
        "Plasticity": avg_accuracy,
    }

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        results_data = {
            "method": "ZERO_SHOT_CLIP",
            "alpha": None,
            "config": config or {},
            "task_accuracies": task_accuracies,
            "accuracy_matrix": accuracy_matrix,
            "metrics": metrics,
        }
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        torch.save({
            "model": clip.model.state_dict(),
            "accuracy_matrix": accuracy_matrix,
            "metrics": metrics,
            "config": config,
        }, os.path.join(checkpoint_dir, "final_model.pt"))
        print(f"Saved Zero-Shot results to {checkpoint_dir}")

    return accuracy_matrix, metrics


def run_clip_coma(tasks, clip, epochs=1, alpha=0.7, device="cuda", lr=1e-5,
                  weight_decay=0.1, save_path=None, checkpoint_dir=None,
                  resume=True, config=None):
    """Train sequentially and evaluate every learned task after each stage."""
    device = torch.device(device)
    task_count = len(tasks)
    accuracy_matrix = {task_id: [] for task_id in range(task_count)}
    text_features = {
        task_id: clip.make_text_features(task["class_names"])
        for task_id, task in tasks.items()
    }

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        results_file = os.path.join(checkpoint_dir, "results.json")
        if resume and os.path.exists(results_file):
            print(f"Experiment already completed in {checkpoint_dir}. Loading results.")
            with open(results_file, "r") as f:
                data = json.load(f)
                matrix = {int(k): v for k, v in data["accuracy_matrix"].items()}
                return matrix

    start_task_id = 0
    previous_average = copy.deepcopy(clip.model.state_dict())

    # Check for latest completed checkpoint if resume is enabled
    if checkpoint_dir and resume:
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_task_*.pt"))
        if ckpt_files:
            completed_ids = []
            for f in ckpt_files:
                base = os.path.basename(f)
                num_str = base.replace("checkpoint_task_", "").replace(".pt", "")
                if num_str.isdigit():
                    completed_ids.append(int(num_str))
            if completed_ids:
                latest_id = max(completed_ids)
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_task_{latest_id}.pt")
                print(f"Resuming from checkpoint {ckpt_path} (completed task {latest_id + 1}/{task_count})")
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                clip.model.load_state_dict(ckpt["model_state"])
                previous_average = ckpt["previous_average"]
                accuracy_matrix = ckpt["accuracy_matrix"]
                start_task_id = latest_id + 1

    for task_id in range(start_task_id, task_count):
        task = tasks[task_id]
        print(f"\nTraining task {task_id + 1}/{task_count}: {task['class_names']}")
        train_one_task(clip, task["train"], task["class_names"], epochs,
                       device, lr, weight_decay, text_features[task_id])

        averaged_state = average_visual_parameters(previous_average, clip.model, alpha)
        clip.model.load_state_dict(averaged_state)
        previous_average = copy.deepcopy(averaged_state)

        for eval_id in range(task_count):
            if eval_id > task_id:
                accuracy = 0.0
            else:
                accuracy = evaluate_task(clip, tasks[eval_id]["test"],
                                         tasks[eval_id]["class_names"], device,
                                         text_features[eval_id])
            accuracy_matrix[eval_id].append(accuracy)
            if eval_id <= task_id:
                print(f"    task {eval_id} accuracy: {accuracy:.4f}")

        # Memory cleanup for Apple Silicon MPS
        if device.type == "mps":
            torch.mps.empty_cache()

        # Save per-task checkpoint
        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_task_{task_id}.pt")
            torch.save({
                "task_id": task_id,
                "model_state": clip.model.state_dict(),
                "previous_average": previous_average,
                "accuracy_matrix": accuracy_matrix,
                "alpha": alpha,
                "config": config or {},
            }, ckpt_path)

    metrics = calculate_metrics(accuracy_matrix)

    if checkpoint_dir:
        # Save final model and results metadata
        final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
        torch.save({
            "model": clip.model.state_dict(),
            "alpha": alpha,
            "accuracy_matrix": accuracy_matrix,
            "metrics": metrics,
            "config": config or {},
        }, final_model_path)

        results_data = {
            "method": (config or {}).get("method", f"CLIP_COMA_ALPHA_{alpha}"),
            "alpha": alpha,
            "config": config or {},
            "accuracy_matrix": accuracy_matrix,
            "metrics": metrics,
        }
        results_file = os.path.join(checkpoint_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Saved experiment results to {results_file}")

    if save_path:
        torch.save({"model": clip.model.state_dict(), "alpha": alpha,
                    "accuracy_matrix": accuracy_matrix, "metrics": metrics,
                    "config": config or {}}, save_path)

    return accuracy_matrix

"""Entry point for the CLIP + CoMA CIFAR-100 experiment."""
import random
import numpy as np
import torch

from clip_models import CLIPClassifier
from clip_tasks import get_clip_cifar100_tasks
from clip_coma import run_clip_coma
from metrics import calculate_metrics


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    if device.type == "cpu":
        print("Note: CLIP fine-tuning will be slow on CPU; Colab GPU is recommended.")

    task_count = 10
    batch_size = 8
    epochs = 1  # smoke-test setting; increase to the report's planned 5-10 range
    alpha = 0.7
    learning_rate = 1e-5

    clip = CLIPClassifier(device=device)
    tasks = get_clip_cifar100_tasks(clip.preprocess, batch_size=batch_size,
                                    task_count=task_count, seed=seed)
    results = run_clip_coma(tasks, clip, epochs=epochs, alpha=alpha,
                            device=device, lr=learning_rate,
                            save_path="clip_coma_cifar100.pt")

    print("\nTask-wise accuracy matrix:")
    for task_id, values in results.items():
        print(f"Task {task_id}: {values}")
    print("\nMetrics:")
    for name, value in calculate_metrics(results).items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

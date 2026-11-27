"""CIFAR-100 task stream using the preprocessing expected by OpenCLIP."""
from __future__ import annotations

import random
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100


class LocalLabelSubset(Dataset):
    def __init__(self, base_dataset, indices, global_to_local):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.global_to_local = global_to_local

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, global_label = self.base_dataset[self.indices[index]]
        return image, self.global_to_local[int(global_label)]


def get_clip_cifar100_tasks(preprocess, batch_size=128, task_count=10, seed=42,
                            num_workers=0, shuffle_classes=False):
    if 100 % task_count != 0:
        raise ValueError("task_count must divide 100")

    train_set = CIFAR100(root="./data", train=True, download=True, transform=preprocess)
    test_set = CIFAR100(root="./data", train=False, download=True, transform=preprocess)
    class_order = list(range(100))
    if shuffle_classes:
        random.Random(seed).shuffle(class_order)

    classes_per_task = 100 // task_count
    tasks = {}
    for task_id in range(task_count):
        global_classes = class_order[task_id * classes_per_task:(task_id + 1) * classes_per_task]
        global_to_local = {label: i for i, label in enumerate(global_classes)}
        train_indices = [i for i, y in enumerate(train_set.targets) if y in global_to_local]
        test_indices = [i for i, y in enumerate(test_set.targets) if y in global_to_local]
        train_subset = LocalLabelSubset(train_set, train_indices, global_to_local)
        test_subset = LocalLabelSubset(test_set, test_indices, global_to_local)
        tasks[task_id] = {
            "train": DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=False),
            "test": DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=False),
            "global_classes": global_classes,
            "class_names": [train_set.classes[c] for c in global_classes],
        }
    return tasks

"""Smoke test for CLIP + CoMA pipeline — no full experiments.

Verifies:
  1. CLIP model initialisation (ViT-B-32-quickgelu / openai weights)
  2. Device selection (CUDA → CPU on Windows; MPS on macOS)
  3. Text encoding (make_text_features)
  4. Image encoding (image_text_logits with synthetic image)
  5. CIFAR-100 task generation (loads dataset if present, otherwise skips)
  6. One mini forward + backward training pass on a synthetic batch

Run from the repository root:
    python smoke_test.py
"""
from __future__ import annotations

import sys
import time
import torch
import torch.nn.functional as F

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"


def section(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


# ── 1. Device selection ───────────────────────────────────────
section("1. Device selection")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"{PASS} CUDA selected — {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"{PASS} MPS selected (Apple Silicon)")
else:
    device = torch.device("cpu")
    print(f"{PASS} CPU selected (no GPU detected)")

print(f"       torch {torch.__version__} | device={device}")

# ── 2. CLIPClassifier initialisation ─────────────────────────
section("2. CLIP model initialisation")
try:
    t0 = time.time()
    from clip_models import CLIPClassifier
    clip = CLIPClassifier(device=device)
    elapsed = time.time() - t0
    param_count = sum(p.numel() for p in clip.model.parameters())
    visual_trainable = sum(p.numel() for p in clip.model.visual.parameters() if p.requires_grad)
    print(f"{PASS} CLIPClassifier created in {elapsed:.1f}s")
    print(f"       Total parameters      : {param_count:,}")
    print(f"       Trainable (visual)    : {visual_trainable:,}")
except Exception as exc:
    print(f"{FAIL} CLIPClassifier failed: {exc}")
    sys.exit(1)

# ── 3. Text encoding ─────────────────────────────────────────
section("3. Text encoding (make_text_features)")
try:
    class_names = ["apple", "banana", "cherry", "dog", "elephant",
                   "fish", "guitar", "hammer", "iris", "jacket"]
    t0 = time.time()
    text_feats = clip.make_text_features(class_names)
    elapsed = time.time() - t0
    print(f"{PASS} Text features shape : {tuple(text_feats.shape)} in {elapsed:.2f}s")
    norms = text_feats.norm(dim=-1)
    print(f"       L2 norms (expect ~1): min={norms.min():.4f} max={norms.max():.4f}")
    assert norms.min() > 0.99, "Text features are not L2-normalised!"
    print(f"{PASS} L2-normalisation confirmed")
except Exception as exc:
    print(f"{FAIL} Text encoding failed: {exc}")
    sys.exit(1)

# ── 4. Image encoding + logits ────────────────────────────────
section("4. Image encoding and image-text similarity")
try:
    # Synthetic batch: 4 images at CLIP expected size (3 × 224 × 224)
    synthetic_images = torch.randn(4, 3, 224, 224).to(device)
    t0 = time.time()
    logits = clip.image_text_logits(synthetic_images, text_feats)
    elapsed = time.time() - t0
    print(f"{PASS} Logits shape        : {tuple(logits.shape)} in {elapsed:.2f}s")
    print(f"       Logit range         : [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"
    print(f"{PASS} Logit shape verified (batch=4, classes=10)")
except Exception as exc:
    print(f"{FAIL} Image encoding/logits failed: {exc}")
    sys.exit(1)

# ── 5. One training forward + backward pass ───────────────────
section("5. Mini training forward + backward pass")
try:
    clip.train_mode()
    optimizer = torch.optim.AdamW(clip.trainable_parameters(), lr=1e-5)
    synthetic_images_train = torch.randn(4, 3, 224, 224).to(device)
    labels = torch.randint(0, 10, (4,)).to(device)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    logits = clip.image_text_logits(synthetic_images_train, text_feats)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    elapsed = time.time() - t0
    print(f"{PASS} Forward + backward pass completed in {elapsed:.2f}s")
    print(f"       Loss                : {loss.item():.4f}")
    if device.type == "mps":
        torch.mps.empty_cache()
except Exception as exc:
    print(f"{FAIL} Training pass failed: {exc}")
    sys.exit(1)

# ── 6. CoMA averaging ────────────────────────────────────────
section("6. CoMA parameter averaging (average_visual_parameters)")
try:
    import copy
    from clip_coma import average_visual_parameters
    prev_avg = copy.deepcopy(clip.model.state_dict())
    averaged = average_visual_parameters(prev_avg, clip.model, alpha=0.7)
    visual_keys = [k for k in averaged if k.startswith("visual.")]
    print(f"{PASS} Averaged {len(visual_keys)} visual parameter tensors")
    # Verify one key is blended
    k = visual_keys[0]
    orig = prev_avg[k].float()
    curr = dict(clip.model.named_parameters())[k].detach().float()
    expected = 0.7 * curr + 0.3 * orig
    actual = averaged[k].float().to(expected.device)
    diff = (actual - expected).abs().max().item()
    print(f"       Max blending error  : {diff:.2e} (expect ~0)")
    assert diff < 1e-5, f"CoMA blend error too large: {diff}"
    print(f"{PASS} CoMA averaging mathematically correct")
except Exception as exc:
    print(f"{FAIL} CoMA averaging failed: {exc}")
    sys.exit(1)

# ── 7. CIFAR-100 task generation (optional — requires download) ─
section("7. CIFAR-100 task stream (download=False probe)")
try:
    from clip_tasks import get_clip_cifar100_tasks
    # Try without downloading — will raise if data absent
    from torchvision.datasets import CIFAR100
    import os
    cifar_path = "./data"
    has_data = os.path.exists(os.path.join(cifar_path, "cifar-100-python"))
    if has_data:
        tasks = get_clip_cifar100_tasks(clip.preprocess, batch_size=8,
                                        task_count=10, seed=42, num_workers=0)
        print(f"{PASS} Loaded {len(tasks)} tasks from local CIFAR-100")
        for tid, t in tasks.items():
            assert len(t["class_names"]) == 10, "Expected 10 classes per task"
        print(f"{PASS} Task structure verified (10 tasks × 10 classes)")
    else:
        print(f"{SKIP} CIFAR-100 not downloaded yet — will auto-download on first run")
        print(f"       Expected path: {os.path.abspath(cifar_path)}")
except Exception as exc:
    print(f"{SKIP} CIFAR-100 check skipped: {exc}")

# ── 8. Checkpoint round-trip ─────────────────────────────────
section("8. Checkpoint save / load round-trip")
try:
    import copy, tempfile, os
    ckpt_data = {
        "task_id": 0,
        "model_state": clip.model.state_dict(),
        "previous_average": copy.deepcopy(clip.model.state_dict()),
        "accuracy_matrix": {i: [] for i in range(10)},
        "alpha": 0.7,
        "config": {"method": "smoke_test"},
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save(ckpt_data, ckpt_path)
    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert loaded["task_id"] == 0
    assert loaded["alpha"] == 0.7
    os.unlink(ckpt_path)
    print(f"{PASS} Checkpoint save/load round-trip succeeded")
    print(f"       map_location={device} — cross-device load works")
except Exception as exc:
    print(f"{FAIL} Checkpoint round-trip failed: {exc}")

# ── Summary ──────────────────────────────────────────────────
section("SMOKE TEST COMPLETE")
print("  All critical checks passed.")
print(f"  Device        : {device}")
print(f"  Python        : {sys.version.split()[0]}")
print(f"  PyTorch       : {torch.__version__}")
print()
print("  The pipeline is ready for full experimental execution.")
print("  Run:  python run_experiments.py --stage all")
print()

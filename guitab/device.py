"""Compute device for training and inference: --device auto | cuda | mps | cpu (auto = CUDA, else Apple MPS, else CPU),
plus device-neutral memory helpers for the accelerators."""

import os

import torch

DEVICES = ("auto", "cuda", "mps", "cpu")


def pick_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def _accelerator(device):
    """torch.cuda or torch.mps for an accelerator device, None for the CPU."""
    return getattr(torch, device.type) if device.type in ("cuda", "mps") else None


def limit_memory(device, fraction):
    """Cap this process's share of the accelerator memory (no cap when fraction is None or on the CPU). Call it before the
    first tensor goes to the device. On Apple GPUs the cap is the allocator's high watermark, and a low watermark at 70%
    of it makes the allocator release cached memory early (a cap alone runs out of memory at the same size)."""
    if fraction is None or not _accelerator(device):
        return
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(fraction))
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(round(0.7 * fraction, 3)))
    else:
        torch.cuda.set_per_process_memory_fraction(fraction, device)


def empty_cache(device):
    if _accelerator(device):
        _accelerator(device).empty_cache()


def allocated_gb(device):
    """Memory held by live tensors on the accelerator, in GB (0 on the CPU)."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 2**30
    return torch.mps.current_allocated_memory() / 2**30 if device.type == "mps" else 0.0

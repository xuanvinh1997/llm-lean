"""
GPU VRAM monitoring and environment validation helpers.
Designed for 96GB GPUs (RTX 6000 Ada / Ampere Pro, H100, A100-80G).
"""

import os
import torch
from typing import Dict, Any

def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Returns current GPU memory allocation and reserved statistics in GB.
    """
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_name": "CPU",
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_gb": 0.0
        }

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    total = props.total_memory / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    return {
        "cuda_available": True,
        "device_name": props.name,
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "peak_gb": round(peak, 2),
        "free_gb": round(total - reserved, 2)
    }

def print_gpu_vram_summary(tag: str = "Status"):
    info = get_gpu_memory_info()
    if not info["cuda_available"]:
        print(f"[{tag}] CUDA is not available. Running on CPU.")
        return

    print(f"[{tag}] GPU: {info['device_name']} | Total: {info['total_gb']} GB")
    print(
        f"       Allocated: {info['allocated_gb']} GB | "
        f"Reserved: {info['reserved_gb']} GB | "
        f"Peak: {info['peak_gb']} GB | "
        f"Free: {info['free_gb']} GB"
    )

def clear_gpu_cache():
    """
    Manually release cached PyTorch memory and collect garbage.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

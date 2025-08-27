"""
Memory management utilities for GPU operations.
"""

import torch
from typing import Dict, Any

class MemoryUtils:
    """Utility class for GPU memory management (identical to existing implementations)."""
    
    @staticmethod
    def reset_memory_stats():
        """Reset CUDA memory statistics."""
        torch.cuda.reset_peak_memory_stats(device=0)
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get peak memory usage in GB."""
        return round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    
    @staticmethod
    def clear_cache():
        """Clear CUDA cache."""
        torch.cuda.empty_cache()
    
    @staticmethod
    def synchronize():
        """Synchronize CUDA operations."""
        torch.cuda.synchronize()
    
    @staticmethod
    def get_model_size(model) -> tuple:
        """Get model size in parameters and MB."""
        params = sum(p.numel() for p in model.parameters())
        size_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        return params, size_mb
    
    @staticmethod
    def print_memory_info():
        """Print current GPU memory information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        else:
            print("CUDA not available")
    
    @staticmethod
    def get_memory_summary() -> Dict[str, float]:
        """Get memory usage summary."""
        if not torch.cuda.is_available():
            return {"allocated_gb": 0, "reserved_gb": 0, "peak_reserved_gb": 0}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3
        }

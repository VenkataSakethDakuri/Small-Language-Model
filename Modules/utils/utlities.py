"""
Utility functions for memory management, data loading, and model operations.
"""

import torch
import pandas as pd
import os
from typing import Dict, Any, Optional
from pathlib import Path

class MemoryUtils:
    """Memory management utilities for CUDA operations."""
    
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


class DataUtils:
    """Data loading utilities."""
    
    @staticmethod
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV file with error handling."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            df = pd.read_csv(file_path, **kwargs)
            print(f"Successfully loaded CSV: {file_path} with shape {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error loading CSV file {file_path}: {str(e)}")
            raise


class ModelUtils:
    """Model saving utilities for LoRA adapters."""
    
    @staticmethod
    def save_model_for_lora(model, tokenizer, output_dir: str):
        """Save model and tokenizer for LoRA adapter usage."""
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save model
            model.save_pretrained(output_dir)
            print(f"Model saved to: {output_dir}")
            
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)
            print(f"Tokenizer saved to: {output_dir}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise


class FileUtils:
    """General file utilities."""
    
    @staticmethod
    def ensure_directory(dir_path: str) -> str:
        """Ensure directory exists, create if not."""
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return dir_path

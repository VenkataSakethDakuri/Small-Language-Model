"""
Base inference engine for model inference.
"""

from abc import ABC, abstractmethod
import torch
import time
from typing import Dict, Any, List, Optional

class BaseInferenceEngine(ABC):
    """Abstract base class for all inference engine implementations."""

    def __init__(self, lora_path: str, config: Dict[str, Any]):
        self.lora_path = lora_path
        self.config = config

    @abstractmethod
    def load_model(self) -> None:
        """Load the model for inference."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
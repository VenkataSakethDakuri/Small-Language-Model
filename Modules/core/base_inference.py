"""
Base inference engine for model inference.
"""

from abc import ABC, abstractmethod
import torch
import time
from typing import Dict, Any, List, Optional
from transformers import TextStreamer

class BaseInferenceEngine(ABC):
    """Abstract base class for all inference engine implementations."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.model = None
        self.tokenizer = None
        self.inference_time = 0
        self.first_token_time = 0
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model for inference."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
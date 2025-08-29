"""
Base inference engine for model inference.
"""

from abc import ABC, abstractmethod
import torch
import time
from typing import Dict, Any, List, Optional
from transformers import TextStreamer

class CustomTextStreamer(TextStreamer):
    """Custom text streamer to measure time to first token."""
    
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.first_token_time = 0
        self.flag = False
    
    def put(self, value):
        if not self.flag:
            self.first_token_time = time.time()
            self.flag = True
        
        super().put(value)

class BaseInferenceEngine(ABC):
    """Abstract base class for all inference engine implementations."""

    def __init__(self, lora_path: str, config: Dict[str, Any]):
        self.lora_path = lora_path
        self.config = config
        self.inference_time = 0
        self.first_token_time = 0
        self.peak_memory = 0

    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """Load the model for inference."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
"""
Base trainer class for all training approaches.
"""

from abc import ABC, abstractmethod
import torch
import time
from typing import Dict, Any, Optional

class BaseTrainer(ABC):
    """Abstract base class for all trainer implementations."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        self.training_time = 0
        self.peak_memory = 0
    
    @abstractmethod
    def setup_model(self) -> None:
        """Setup the model and tokenizer."""
        pass
    
    @abstractmethod
    def prepare_data(self, dataset) -> Any:
        """Prepare and preprocess the training data."""
        pass
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Execute the training process."""
        pass

      
    def setup_tokenizer_padding(self):
        """Setup padding token if not exists."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
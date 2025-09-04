"""
Unsloth trainer using Unsloth library (identical to Unsloth.py implementation).
"""

import torch
import time
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from typing import Dict, Any
import pandas as pd

from Modules.core.base_trainer import BaseTrainer
from Modules.data.data_processor import DataProcessor, DatasetFactory
from Modules.utils.utlities import MemoryUtils

class UnslothTrainer(BaseTrainer):
    """Unsloth trainer implementation (identical to Unsloth.py)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.data_processor = None
    
    def setup_model(self, **kwargs) -> None:
        """Setup model and tokenizer using Unsloth (identical to Unsloth.py implementation)."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=kwargs.get("model_name", self.config.get("model_name", "Qwen/Qwen2-0.5B")),
            dtype=kwargs.get("dtype", self.config.get("dtype", None)),
            device_map=kwargs.get("device_map", self.config.get("device_map", "auto")),
        )
        
        self.setup_tokenizer_padding()
        
        # Apply LoRA adapters (identical to original)
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=kwargs.get("lora_r", self.config.get("lora_r", 16)),
            target_modules=kwargs.get("target_modules", self.config.get("target_modules", 
                ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])),
            lora_alpha=kwargs.get("lora_alpha", self.config.get("lora_alpha", 16)),
            lora_dropout=kwargs.get("lora_dropout", self.config.get("lora_dropout", 0)),
            bias="none",
            use_gradient_checkpointing=kwargs.get("use_gradient_checkpointing", self.config.get("use_gradient_checkpointing", "unsloth")),
            use_rslora=kwargs.get("use_rslora", self.config.get("use_rslora", False)),
            loftq_config=kwargs.get("loftq_config", self.config.get("loftq_config", None)),
            random_state=kwargs.get("seed", self.config.get("seed", 108))
        )
        
        self.data_processor = DataProcessor(self.tokenizer)
    
    def prepare_data(self, data, dataset_type: str = "math"):
        """Prepare training data based on dataset type.
        
        Args:
            data: Input data (DataFrame for math, dataset_split string for alpaca)
            dataset_type: Type of dataset to prepare ("math" or "alpaca")
        
        Returns:
            Prepared dataset object
        """
        if dataset_type.lower() == "math":
            return self.prepare_math_data(data)
        elif dataset_type.lower() == "alpaca":
            # For alpaca, data should be a dataset_split string
            dataset = DatasetFactory.create_alpaca_dataset(data if isinstance(data, str) else "train[:10000]")
            return self.prepare_alpaca_data(dataset)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. Accepted values are math, alpaca.")

    def train(self, data, dataset_type: str = "math", **kwargs) -> Dict[str, Any]:
        """Execute training process based on dataset type.
        
        Args:
            data: Input data (DataFrame for math, dataset_split string for alpaca)
            dataset_type: Type of dataset to train on ("math" or "alpaca")
            **kwargs: Additional training parameters
        
        Returns:
            Dictionary containing training results and metrics
        """
        if dataset_type.lower() == "math":
            return self.train_math(data, **kwargs)
        elif dataset_type.lower() == "alpaca":
            # For alpaca, data should be a dataset_split string
            dataset_split = data if isinstance(data, str) else "train[:10000]"
            return self.train_alpaca(dataset_split, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. Accepted values are math, alpaca.")

    def prepare_alpaca_data(self, dataset) -> Any:
        """Prepare alpaca training data."""
        return dataset.map(self.data_processor.alpaca_data_processing, batched=True)
    
    def prepare_math_data(self, data: pd.DataFrame) -> Any:
        """Prepare math training data."""
        # Convert DataFrame to dataset format that Unsloth expects
        processed_data = self.data_processor.math_data_processing(data)
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": processed_data})
        return dataset
    
    def train_alpaca(self, dataset_split: str = "train[:10000]", **kwargs) -> Dict[str, Any]:
        """Execute alpaca training process using SFTTrainer."""
        # Load and prepare dataset
        dataset = DatasetFactory.create_alpaca_dataset(dataset_split)
        processed_dataset = self.prepare_alpaca_data(dataset)
        
        # SFTTrainer handles tokenization and collation internally
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=processed_dataset,
            dataset_text_field="text",
            max_seq_length=kwargs.get("max_seq_length", self.config.get("max_seq_length", 512)),
            packing=False,
            args=SFTConfig(
                per_device_train_batch_size=kwargs.get("batch_size", self.config.get("batch_size", 2)),
                gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", self.config.get("gradient_accumulation_steps", 4)),
                warmup_steps=kwargs.get("warmup_steps", self.config.get("warmup_steps", 5)),
                num_train_epochs=kwargs.get("num_epochs", self.config.get("num_epochs", 1)),
                learning_rate=kwargs.get("learning_rate", self.config.get("learning_rate", 2e-4)),
                logging_steps=kwargs.get("logging_steps", self.config.get("logging_steps", 1)),
                optim=kwargs.get("optimizer", self.config.get("optimizer", "adamw_8bit")),
                weight_decay=kwargs.get("weight_decay", self.config.get("weight_decay", 0.01)),
                lr_scheduler_type=kwargs.get("lr_scheduler_type", self.config.get("lr_scheduler_type", "linear")),
                seed=kwargs.get("seed", self.config.get("seed", 108)),
                output_dir=kwargs.get("output_dir", self.config.get("output_dir", "TrainingAlpaca_Unsloth")),
                report_to="none",
            ),
        )
        
        # Training with metrics tracking
        MemoryUtils.reset_memory_stats()
        train_time_start = time.time()
        
        trainer_result = trainer.train()
        
        MemoryUtils.synchronize()
        train_time_end = time.time()

        output_dir = kwargs.get("output_dir", self.config.get("output_dir", "TrainingAlpaca_Unsloth"))
        self.save_model(output_dir)

        MemoryUtils.clear_cache()
        
        # Store metrics
        self.training_time = train_time_end - train_time_start
        self.peak_memory = MemoryUtils.get_memory_usage()
        
        return {
            "trainer_result": trainer_result,
            "training_time": self.training_time,
            "peak_memory": self.peak_memory,
            "output_dir": output_dir
        }
    
    def train_math(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute math training process using SFTTrainer."""
        # Prepare math dataset
        processed_dataset = self.prepare_math_data(data)
        
        # SFTTrainer handles tokenization and collation internally
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=processed_dataset,
            dataset_text_field="text",
            max_seq_length=kwargs.get("max_seq_length", self.config.get("max_seq_length", 512)),
            packing=False,
            args=SFTConfig(
                per_device_train_batch_size=kwargs.get("batch_size", self.config.get("batch_size", 2)),
                gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", self.config.get("gradient_accumulation_steps", 4)),
                warmup_steps=kwargs.get("warmup_steps", self.config.get("warmup_steps", 5)),
                num_train_epochs=kwargs.get("num_epochs", self.config.get("num_epochs", 1)),
                learning_rate=kwargs.get("learning_rate", self.config.get("learning_rate", 2e-4)),
                logging_steps=kwargs.get("logging_steps", self.config.get("logging_steps", 1)),
                optim=kwargs.get("optimizer", self.config.get("optimizer", "adamw_8bit")),
                weight_decay=kwargs.get("weight_decay", self.config.get("weight_decay", 0.01)),
                lr_scheduler_type=kwargs.get("lr_scheduler_type", self.config.get("lr_scheduler_type", "linear")),
                seed=kwargs.get("seed", self.config.get("seed", 108)),
                output_dir=kwargs.get("output_dir", self.config.get("output_dir", "TrainingMath_Unsloth")),
                report_to="none",
            ),
        )
        
        # Training with metrics tracking
        MemoryUtils.reset_memory_stats()
        train_time_start = time.time()
        
        trainer_result = trainer.train()
        
        MemoryUtils.synchronize()
        train_time_end = time.time()

        output_dir = kwargs.get("output_dir", self.config.get("output_dir", "TrainingMath_Unsloth"))
        self.save_model(output_dir)

        MemoryUtils.clear_cache()
        
        # Store metrics
        self.training_time = train_time_end - train_time_start
        self.peak_memory = MemoryUtils.get_memory_usage()
        
        return {
            "trainer_result": trainer_result,
            "training_time": self.training_time,
            "peak_memory": self.peak_memory,
            "output_dir": output_dir
        }
    
    def print_training_metrics(self):
        """Print training metrics separately."""
        print("Training Metrics:")
        print(f"{self.training_time} seconds used for training.")
        print(f"Peak reserved memory = {self.peak_memory} GB.")
    
    def save_model(self, output_dir: str) -> None:
        """Save the trained model (identical to Unsloth.py implementation)."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

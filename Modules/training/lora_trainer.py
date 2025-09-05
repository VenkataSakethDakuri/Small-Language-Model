"""
LoRA trainer for GPT-OSS style models (identical to GPT_oss.py implementation).
"""

import torch
import time
import pandas as pd
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
from typing import Dict, Any, List

from Modules.core.base_trainer import BaseTrainer
from Modules.data.data_processor import DataProcessor
from Modules.utils.utlities import MemoryUtils

class LoRATrainer(BaseTrainer):
    """LoRA trainer for GPT-OSS style training (identical to GPT_oss.py)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.data_processor = None
    
    def setup_model(self, **kwargs) -> None:
        """Setup model and tokenizer (identical to GPT_oss.py implementation)."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("model_name", "Qwen/Qwen2-0.5B"))
        self.model = AutoModelForCausalLM.from_pretrained(self.config.get("model_name", "Qwen/Qwen2-0.5B"), device_map="auto")
        self.setup_tokenizer_padding()
        
        # LoRA configuration (identical to original)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=kwargs.get("lora_r", self.config.get("lora_r", 8)),
            target_modules=kwargs.get("target_modules", self.config.get("target_modules", 
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])),
            lora_alpha=kwargs.get("lora_alpha", self.config.get("lora_alpha", 16)),
            lora_dropout=kwargs.get("lora_dropout", self.config.get("lora_dropout", 0)),
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.data_processor = DataProcessor(self.tokenizer)
    
    def train(self, data: pd.DataFrame, dataset_type: str = "math", **kwargs) -> Dict[str, Any]:
        """Execute training process based on dataset type.
        
        Args:
            data: Input DataFrame containing the training data
            dataset_type: Type of dataset to train on ("math" or "alpaca")
            **kwargs: Additional training parameters
        
        Returns:
            Dictionary containing training results and metrics
        """
        if dataset_type.lower() == "math":
            return self.train_math(data, **kwargs)
        elif dataset_type.lower() == "alpaca":
            return self.train_alpaca(data, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. Use 'math' or 'alpaca'.")
    
    def prepare_data(self, data: pd.DataFrame, dataset_type: str = "math") -> Dataset:
        """Prepare training data based on dataset type.
        
        Args:
            data: Input DataFrame containing the training data
            dataset_type: Type of dataset to prepare ("math" or "alpaca")
        
        Returns:
            Prepared Dataset object
        """
        if dataset_type.lower() == "math":
            return self.prepare_math_data(data)
        elif dataset_type.lower() == "alpaca":
            return self.prepare_alpaca_data(data)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. Use 'math' or 'alpaca'.")

    def save_model(self, output_dir: str) -> None:
        """Save the trained model (identical to GPT_oss.py implementation)."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")
    
    def prepare_math_data(self, data: pd.DataFrame) -> Dataset:
        """Prepare math training data."""
        processed_data = self.data_processor.math_data_processing(data)
        
        dataset = Dataset.from_dict({"text": processed_data})
        tokenized_dataset = dataset.map(self.data_processor.tokenize_for_math, batched=True)
        
        return tokenized_dataset
    
    def prepare_alpaca_data(self, data: pd.DataFrame) -> Dataset:
        """Prepare alpaca training data."""
        processed_data = self.data_processor.alpaca_data_processing(data)
        
        dataset = Dataset.from_dict({"text": processed_data})
        tokenized_dataset = dataset.map(self.data_processor.tokenize_for_alpaca, batched=True)
        
        return tokenized_dataset
    
    def train_math(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute math training process."""
        # Prepare data
        dataset = self.prepare_math_data(data)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get("output_dir", self.config.get("output_dir", "TrainingMath_LoRA")),
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
            report_to="none",
            save_strategy="epoch",
            dataloader_drop_last=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Training with metrics tracking
        MemoryUtils.reset_memory_stats()
        train_time_start = time.time()
        
        trainer_result = trainer.train()
        
        MemoryUtils.synchronize()
        train_time_end = time.time()
        
        # Save the trained model
        output_dir = kwargs.get("output_dir", self.config.get("output_dir", "TrainingMath_LoRA"))
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
    
    def train_alpaca(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute alpaca training process."""
        # Prepare data
        dataset = self.prepare_alpaca_data(data)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get("output_dir", self.config.get("output_dir", "TrainingAlpaca_LoRA")),
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
            report_to="none",
            save_strategy="epoch",
            dataloader_drop_last=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Training with metrics tracking
        MemoryUtils.reset_memory_stats()
        train_time_start = time.time()
        
        trainer_result = trainer.train()
        
        MemoryUtils.synchronize()
        train_time_end = time.time()
        
        # Save the trained model
        output_dir = kwargs.get("output_dir", self.config.get("output_dir", "TrainingAlpaca_LoRA"))
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
    

    


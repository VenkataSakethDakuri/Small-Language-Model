"""
LLM Compressor trainer with quantization (identical to LLMCompressor.py implementation).
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from llmcompressor.transformers import oneshot
from llmcompressor.transformers import SparseAutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any
import pandas as pd
from datasets import Dataset

from Modules.core.base_trainer import BaseTrainer
from Modules.data.data_processor import DataProcessor, DatasetFactory
from Modules.utils.utlities import MemoryUtils

class CompressedTrainer(BaseTrainer):
    """LLM Compressor trainer with quantization (identical to LLMCompressor.py)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.data_processor = None
        self.quantized_model_path = None
    
    def setup_model(self, **kwargs) -> None:
        """Setup model and tokenizer (identical to LLMCompressor.py implementation)."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            kwargs.get("model_name", self.config.get("model_name", "Qwen/Qwen2-0.5B"))
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            kwargs.get("model_name", self.config.get("model_name", "Qwen/Qwen2-0.5B"))
        )
        
        self.setup_tokenizer_padding()
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
            dataset = DatasetFactory.load_hf_dataset("yahma/alpaca-cleaned", 
                                                    data if isinstance(data, str) else "train[:10000]")
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
        processed_dataset = dataset.map(self.data_processor.alpaca_data_processing, batched=True)
        tokenized_dataset = processed_dataset.map(
            self.data_processor.tokenize_for_alpaca, 
            batched=True, 
            remove_columns=processed_dataset.column_names
        )
        return tokenized_dataset
    
    def prepare_math_data(self, data: pd.DataFrame) -> Any:
        """Prepare math training data."""
        processed_data = self.data_processor.math_data_processing(data)
        dataset = Dataset.from_dict({"text": processed_data})
        tokenized_dataset = dataset.map(
            self.data_processor.tokenize_for_math, 
            batched=True
        )
        return tokenized_dataset
    
    def train_alpaca(self, dataset_split: str = "train[:10000]", **kwargs) -> Dict[str, Any]:
        """Execute alpaca training process."""
        # Load dataset
        dataset = DatasetFactory.load_hf_dataset("yahma/alpaca-cleaned", dataset_split)
        tokenized_dataset = self.prepare_alpaca_data(dataset)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get("output_dir", self.config.get("output_dir", "TrainingAlpaca_LLMCompressor")),
            per_device_train_batch_size=kwargs.get("batch_size", self.config.get("batch_size", 2)),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", self.config.get("gradient_accumulation_steps", 4)),
            warmup_steps=kwargs.get("warmup_steps", self.config.get("warmup_steps", 5)),
            num_train_epochs=kwargs.get("num_epochs", self.config.get("num_epochs", 1)),
            learning_rate=kwargs.get("learning_rate", self.config.get("learning_rate", 2e-4)),
            logging_steps=kwargs.get("logging_steps", self.config.get("logging_steps", 1)),
            weight_decay=kwargs.get("weight_decay", self.config.get("weight_decay", 0.01)),
            lr_scheduler_type=kwargs.get("lr_scheduler_type", self.config.get("lr_scheduler_type", "linear")),
            seed=kwargs.get("seed", self.config.get("seed", 108)),
            report_to="none",
            save_strategy="no"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Training with metrics tracking
        MemoryUtils.reset_memory_stats()
        train_time_start = time.time()
        
        trainer_result = trainer.train()
        
        MemoryUtils.synchronize()
        train_time_end = time.time()
        MemoryUtils.clear_cache()
        
        # Store metrics
        self.training_time = train_time_end - train_time_start
        self.peak_memory = MemoryUtils.get_memory_usage()
        
        return {
            "trainer_result": trainer_result,
            "training_time": self.training_time,
            "peak_memory": self.peak_memory
        }
    
    def train_math(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute math training process."""
        # Prepare data
        tokenized_dataset = self.prepare_math_data(data)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get("output_dir", self.config.get("output_dir", "TrainingMath_LLMCompressor")),
            per_device_train_batch_size=kwargs.get("batch_size", self.config.get("batch_size", 2)),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", self.config.get("gradient_accumulation_steps", 4)),
            warmup_steps=kwargs.get("warmup_steps", self.config.get("warmup_steps", 5)),
            num_train_epochs=kwargs.get("num_epochs", self.config.get("num_epochs", 1)),
            learning_rate=kwargs.get("learning_rate", self.config.get("learning_rate", 2e-4)),
            logging_steps=kwargs.get("logging_steps", self.config.get("logging_steps", 1)),
            weight_decay=kwargs.get("weight_decay", self.config.get("weight_decay", 0.01)),
            lr_scheduler_type=kwargs.get("lr_scheduler_type", self.config.get("lr_scheduler_type", "linear")),
            seed=kwargs.get("seed", self.config.get("seed", 108)),
            report_to="none",
            save_strategy="no"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Training with metrics tracking
        MemoryUtils.reset_memory_stats()
        train_time_start = time.time()
        
        trainer_result = trainer.train()
        
        MemoryUtils.synchronize()
        train_time_end = time.time()
        MemoryUtils.clear_cache()
        
        # Store metrics
        self.training_time = train_time_end - train_time_start
        self.peak_memory = MemoryUtils.get_memory_usage()
        
        return {
            "trainer_result": trainer_result,
            "training_time": self.training_time,
            "peak_memory": self.peak_memory
        }
    
    def print_training_metrics(self):
        """Print training metrics separately."""
        print(f"\nTraining Metrics:")
        print(f"{self.training_time} seconds used for training.")
        print(f"Peak reserved memory = {self.peak_memory} GB.")
    
    def quantize_model(self, adapter_path: str, quantized_output_dir: str) -> None:
        """Quantize model using LLMCompressor (identical to LLMCompressor.py)."""
        # Save adapter first
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        MemoryUtils.clear_cache()
        
        # Create calibration dataset (identical to original)
        calibration_dataset = DatasetFactory.create_calibration_dataset(
            "train[:1000]", self.data_processor
        )
        
        # Quantization recipe (identical to original)
        gtpq_recipe = [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(
                targets="Linear",
                scheme=self.config.get("quantization_scheme", "W4A16"),
                ignore=["lm_head"]
            )
        ]
        
        # Apply quantization (identical to original)
        oneshot(
            model=self.model,
            dataset=calibration_dataset,
            recipe=gtpq_recipe,
            output_dir=quantized_output_dir
        )
        
        self.quantized_model_path = quantized_output_dir
    
    def load_quantized_model_with_lora(self) -> None:
        """Load quantized model and apply LoRA (identical to LLMCompressor.py)."""
        if not self.quantized_model_path:
            raise ValueError("Model must be quantized first")
        
        # Load quantized model (identical to original)
        self.model = SparseAutoModelForCausalLM.from_pretrained(
            self.quantized_model_path,
            load_in_4bit=False
        )
        
        # Apply LoRA configuration (identical to original)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get("lora_r", 16),
            target_modules=self.config.get("target_modules", 
                ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]),
            lora_alpha=self.config.get("lora_alpha", 16),
            lora_dropout=self.config.get("lora_dropout", 0),
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model = self.model.to("cuda")
        self.model.eval()
    
    def save_model(self, output_dir: str) -> None:
        """Save the trained model (identical to LLMCompressor.py implementation)."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

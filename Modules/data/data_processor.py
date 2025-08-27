"""
Data processing utilities for training datasets.
"""

import pandas as pd
from datasets import Dataset, load_dataset
from typing import Dict, Any, List

class DataProcessor:
    """Data processing utilities for different data formats."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV data."""
        return pd.read_csv(file_path)
    
    def alpaca_data_processing(self, training_data: Dict[str, List]) -> Dict[str, List]:
        """Process Alpaca format data (identical to existing implementation)."""
        instructions = training_data["instruction"]
        inputs = training_data["input"]
        outputs = training_data["output"]
        texts = []

        for instruction, input_text, output_text in zip(instructions, inputs, outputs):
            text = f"### Instruction:\n{instruction}\n"
            if input_text:
                text += f"### Input:\n{input_text}\n"
            text += f"### Response:\n{output_text}\n<|endoftext|>"
            texts.append(text)
        
        return {"text": texts}
    
    def math_data_processing(self, training_data: pd.DataFrame) -> List[str]:
        """Process math problem data for GPT-OSS style training."""
        processed_data = []
        for _, row in training_data.iterrows():
            Problem = row['Problem']
            Solution = row['Solution']
            
            messages = [
                {
                    "role": "system",
                    "content": "Solve mathematical problems with logically complete and formally justified solutions:"
                },
                {
                    "role": "user",
                    "content": f"### Problem:\n{Problem}"
                },
                {
                    "role": "assistant", 
                    "content": f"### Solution:\n{Solution}"
                }
            ]
            
            formatted_text = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=False, 
                tokenize=False, 
                add_special_tokens=False
            )
            processed_data.append(formatted_text)
        
        return processed_data
    
    def tokenize_for_alpaca(self, examples: Dict[str, List], max_length: int = 512, 
                         padding: str = "max_length") -> Dict[str, Any]:
        """Tokenize examples (identical to existing implementation)."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=padding,
            max_length=max_length
        )
    
    def tokenize_for_math(self, examples: Dict[str, List],  max_length: int = 512) -> Dict[str, Any]:
        """Tokenize for math problems (GPT-OSS style)."""
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

class DatasetFactory:
    """Factory class for creating datasets."""
    
    @staticmethod
    def create_alpaca_dataset(split: str = "train[:10000]") -> Dataset:
        """Create Alpaca dataset (identical to existing implementation)."""
        return load_dataset("yahma/alpaca-cleaned", split=split)
    
    @staticmethod
    def create_training_dataset(data, processor: DataProcessor, 
                              data_type: str = "alpaca") -> Dataset:
        """Create training dataset from processed data."""
        if data_type == "alpaca":
            processed_data = data.map(processor.alpaca_data_processing, batched=True)
        elif data_type == "math":
            processed_data = {"text": processor.math_data_processing(data)}
            processed_data = Dataset.from_dict(processed_data)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        return processed_data
    
    @staticmethod
    def create_tokenized_dataset(dataset: Dataset, processor: DataProcessor, 
                               tokenize_type: str = "standard") -> Dataset:
        """Create tokenized dataset."""
        if tokenize_type == "standard":
            return dataset.map(processor.tokenize_for_alpaca, batched=True, 
                             remove_columns=dataset.column_names)
        elif tokenize_type == "math":
            return dataset.map(processor.tokenize_for_math, batched=True)
        else:
            raise ValueError(f"Unsupported tokenize type: {tokenize_type}")
    
    @staticmethod
    def create_calibration_dataset(split: str = "train[:1000]", 
                                 processor: DataProcessor = None) -> Dataset:
        """Create calibration dataset for quantization."""
        dataset = load_dataset("yahma/alpaca-cleaned", split=split)
        if processor:
            dataset = dataset.map(processor.alpaca_data_processing, batched=True)
            
            def tokenize(sample):
                return processor.tokenizer(
                    sample["text"], 
                    padding=False, 
                    max_length=512, 
                    truncation=True, 
                    add_special_tokens=True
                )
            
            dataset = dataset.map(tokenize, batched=True, 
                                remove_columns=["instruction", "input", "output"])
        
        return dataset

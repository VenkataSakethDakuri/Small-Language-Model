"""
Standard inference engine using transformers (identical to existing implementations).
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Dict, Any, List
import pandas as pd
import json

from Modules.core.base_inference import BaseInferenceEngine, CustomTextStreamer
from Modules.utils.utlities import MemoryUtils

class StandardInferenceEngine(BaseInferenceEngine):
    """Standard inference engine using transformers (identical to existing implementations)."""
    
    def __init__(self, lora_path: str, config: Dict[str, Any]):
        super().__init__(lora_path, config)
        self.model = None
        self.tokenizer = None
    
    def load_model(self, **kwargs) -> None:
        """Load model for inference (identical to Qwen.py implementation)."""
        MemoryUtils.clear_cache()
        
        base_model = AutoModelForCausalLM.from_pretrained(
            kwargs.get("base_model", self.config.get("base_model", "Qwen/Qwen2-0.5B")),
            device_map=kwargs.get("device_map", self.config.get("device_map", "auto"))
        )
        
        self.model = PeftModel.from_pretrained(
            base_model, 
            kwargs.get("lora_path", self.lora_path), 
            device_map=kwargs.get("device_map", self.config.get("device_map", "auto"))
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            kwargs.get("tokenizer_path", kwargs.get("lora_path", self.lora_path))
        )
        
        self.model.eval()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt (identical to existing implementations)."""
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        text_streamer = CustomTextStreamer(self.tokenizer)
        MemoryUtils.reset_memory_stats()
        start_time = time.time()
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                streamer=text_streamer, 
                max_new_tokens=kwargs.get("max_new_tokens", self.config.get("max_new_tokens", 128)),
                do_sample=kwargs.get("do_sample", self.config.get("do_sample", False)),
                temperature=kwargs.get("temperature", self.config.get("temperature", 0.5))
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Store metrics
        self.inference_time = end_time - start_time
        self.first_token_time = text_streamer.first_token_time - start_time
        
        return generated_text
    
    def alpaca_inference(self, instruction: str, input_text: str = "", **kwargs) -> str:
        """Generate response for alpaca-style prompt."""
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
        prompt = alpaca_prompt.format(instruction, input_text, "")
        return self.generate(prompt, **kwargs)
    
    def math_inference(self, problem: str, system_message: str = None, **kwargs) -> str:
        """Generate response for math problem solving."""
        if system_message is None:
            system_message = "Solve the following problem with logically complete and formally justified solutions:"
        
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user", 
                "content": f"### Problem:\n{problem}\n\n### Solution:"
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return self.generate(prompt, **kwargs)
    
    def batch_inference_on_math_dataset(self, data_path: str, output_path: str, system_message: str = None, **kwargs) -> List[Dict[str, Any]]:
        """Run batch inference on test dataset."""
        testing_dataset = pd.read_csv(data_path)
        generated_outputs = []
        
        if system_message is None:
            system_message = "Solve the following problem with logically complete and formally justified solutions:"

        MemoryUtils.reset_memory_stats()
        start_time = time.time()

        for i in range(len(testing_dataset)):
            row = testing_dataset.iloc[i]
            
            generated_text = self.math_inference(
                problem=row['Problem'], 
                system_message=system_message,
                **kwargs
            )
            
            generated_outputs.append({
                'problem': row['Problem'],
                'original_solution': row['Solution'],
                'generated_solution': generated_text,
            })
        
        self.peak_reserved_memory = MemoryUtils.get_memory_summary()["peak_reserved_gb"]
        torch.cuda.synchronize()
        end_time = time.time()
        self.inference_time = end_time - start_time
        
        # Save results
        with open(output_path, "w") as f:
            json.dump(generated_outputs, f)
        
        return generated_outputs
    
    def print_inference_metrics(self):
        """Print inference metrics (identical to existing implementations)."""
        param_count, model_size_mb = self.get_model_size()
        print(f"\nInference Metrics:")
        print(f"Model size: {model_size_mb:.2f} MB with {param_count} parameters")
        print(f"Peak reserved memory: {self.get_memory_usage()} GB")
        print(f"Time to first token: {self.first_token_time:.5f} seconds")
        print(f"Inference time: {self.inference_time:.2f} seconds")

    def metrics(self):
        return {
            "inference_time": self.inference_time,
            "peak_reserved_memory": self.peak_reserved_memory
        }
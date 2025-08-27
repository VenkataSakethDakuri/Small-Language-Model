"""
VLLM inference engine with LoRA support (identical to existing implementations).
"""

import torch
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import Dict, Any, List
import pandas as pd
import json

from Modules.core.base_inference import BaseInferenceEngine

class VLLMInferenceEngine(BaseInferenceEngine):
    """VLLM inference engine with LoRA support (identical to existing implementations)."""
    
    def __init__(self, lora_path: str, config: Dict[str, Any]):
        super().__init__(lora_path, config)
        self.vllm_model = None
        self.sampling_params = None
    
    def load_model(self) -> None:
        """Load VLLM model (identical to existing implementations)."""
        self.clear_cache()
        
        self.vllm_model = LLM(
            model=self.config.get("base_model", self.lora_path),
            gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.3),
            enable_lora=self.config.get("enable_lora", True)
        )
        
        # Setup sampling parameters (identical to original)
        self.sampling_params = SamplingParams(
            temperature=self.config.get("temperature", 0.5),
            max_tokens=self.config.get("max_tokens", 512),
            stop=self.config.get("stop_tokens", ["<|end_of_text|>"])
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using VLLM (identical to existing implementations)."""
        if not self.vllm_model:
            self.load_model()
        
        self.clear_cache()
        self.reset_memory_stats()
        
        start_time = time.time()
        
        # Create LoRA request if adapter specified
        lora_request = None
        if self.config.get("adapter_path"):
            lora_request = LoRARequest(
                self.config.get("adapter_name", "adapter"), 
                self.config.get("adapter_id", 1), 
                self.config["adapter_path"]
            )
        
        # Override sampling params with kwargs if provided
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.sampling_params.temperature),
            max_tokens=kwargs.get("max_new_tokens", self.sampling_params.max_tokens),
            stop=kwargs.get("stop_tokens", self.sampling_params.stop)
        )
        
        response = self.vllm_model.generate(
            [prompt], 
            sampling_params, 
            lora_request=lora_request
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        self.inference_time = end_time - start_time
        
        return response[0].outputs[0].text
    
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
        
        # For VLLM, we'll create a simple prompt format since it doesn't use tokenizer.apply_chat_template
        prompt = f"{system_message}\n\n### Problem:\n{problem}\n\n### Solution:"
        
        return self.generate(prompt, **kwargs)
    
    def batch_inference_on_dataset(self, data_path: str, output_path: str, system_message: str = None, **kwargs) -> List[Dict[str, Any]]:
        """Run batch inference on test dataset."""
        testing_dataset = pd.read_csv(data_path)
        generated_outputs = []
        
        if system_message is None:
            system_message = "Solve the following problem with logically complete and formally justified solutions:"
        
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
        
        # Save results
        with open(output_path, "w") as f:
            json.dump(generated_outputs, f)
        
        return generated_outputs
    
    def print_inference_metrics(self):
        """Print VLLM inference metrics (identical to existing implementations)."""
        print(f"VLLM Inference time: {self.inference_time:.5f} seconds")
        print(f"VLLM Peak reserved memory: {self.get_memory_usage()} GB")
    
    def cleanup(self):
        """Cleanup VLLM model (identical to existing implementations)."""
        if self.vllm_model:
            del self.vllm_model
            self.clear_cache()

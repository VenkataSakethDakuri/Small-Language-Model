# Small Language Model

## Prerequisites
- Install all dependencies from requirements.txt using:
```bash
pip install -r requirements.txt
```
- Thirst to explore 😊

## Getting Started
A small language model is a compact AI language model with far fewer parameters (typically millions to a few billion) designed to perform targeted NLP tasks efficiently on limited hardware, trading broad capability for lower latency, cost, and easier deployment.

This repository has codes to experiment with different libraries and different techniques to make a high performing small language model. 

## Folders
- **Experiment 1** has code with using Unsloth and LLMCompressor libraries. It also uses vLLM for faster inference. `Qwen.py` is made without any optimizations to set the baseline. All files use Alpaca-cleaned dataset.

- **Experiment 2** has code with using gpt-oss 20B model for Lora finetuning. Math dataset is used for evaluation from https://github.com/ziye2chen/DEMI-MathAnalysis. Gemini 2.5 pro is used as a LLM as a judge. 


## Modules Documentation

This directory contains modularized components for Small Language Model (SLM) training, inference, and evaluation. The architecture follows a clean separation of concerns with abstract base classes and specialized implementations.

### 📁 Directory Structure

```
Modules/
├── core/               # Base classes and interfaces
├── data/               # Data processing utilities
├── evaluation/         # Model evaluation tools
├── inference/          # Inference engines
├── training/           # Training implementations
└── utils/              # Utility functions
```

### 🔧 Core Components

#### `core/`
Contains abstract base classes that define common interfaces for all training and inference implementations. Provides the foundation that other modules inherit from to ensure consistency across different approaches.

### 📊 Data Processing

#### `data/`
Handles data loading, preprocessing, and formatting for different dataset types. Supports multiple formats including Alpaca instruction datasets, mathematical problem datasets, and generic CSV data. Includes tokenization utilities for model training.

### 🧪 Evaluation Tools

#### `evaluation/`
Provides model evaluation capabilities using LLM-as-a-Judge methodology with Gemini API integration. Compares model outputs and provides structured scoring for mathematical problem-solving tasks.

### 🚀 Inference Engines

#### `inference/`
Multiple inference implementations optimized for different use cases:
- **Standard**: Compatible transformers-based inference with PEFT support
- **Unsloth**: Optimized for 2x faster inference and memory efficiency  
- **vLLM**: High-throughput serving with GPU optimization and LoRA support

### 🎓 Training Implementations

#### `training/`
Specialized trainers for different optimization approaches:
- **Unsloth**: Fastest training with lowest memory usage
- **LoRA**: Standard PEFT implementation with broad compatibility
- **LLMCompressor**: Quantized training with model compression support

### 🛠️ Utilities

#### `utils/`
Common utility functions for memory management, CUDA operations, and model diagnostics. Includes tools for monitoring GPU usage, clearing cache, and getting model statistics.

### 🎯 Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Extensibility**: Easy to add new trainers/inference engines  
3. **Consistency**: Common interfaces across implementations
4. **Performance**: Optimized implementations for different use cases
5. **Flexibility**: Configuration-driven behavior

### 📋 Usage Guidelines

- **Memory Limited**: Use Unsloth components for training and inference
- **Speed Critical**: Use vLLM for inference, Unsloth for training  
- **Model Compression**: Use LLMCompressor trainer for quantized models
- **Standard Compatibility**: Use standard implementations for broad compatibility

### ⚙️ Example Configurations

#### Training Configuration
```python
config = {
    "model_name": "Qwen/Qwen2-0.5B",
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "max_new_tokens": 128,
    "device_map": "auto",
    "seed": 108
}
```

#### Inference Configuration
```python
config = {
    "base_model": "Qwen/Qwen2-0.5B",
    "dtype": None,
    "device_map": "auto",
    "max_new_tokens": 128,
    "temperature": 0.5,
    "gpu_memory_utilization": 0.3,
    "do_sample": False
}
```

---

This modular architecture enables flexible experimentation with different training techniques and inference optimizations while maintaining code reusability and clean interfaces.


## References
- **[Unsloth](https://unsloth.ai/)** - Fast and memory-efficient fine-tuning
- **[LLMCompressor](https://developers.redhat.com/articles/2024/08/14/llm-compressor-here-faster-inference-vllm#enabling_activation_quantization_in_vllm)** - Model compression for faster inference
- **[vLLM](https://docs.vllm.ai/en/latest/)** - High-performance LLM serving framework
- **[My Notebook](https://notebooklm.google.com/notebook/eee2c93a-12a8-4dba-9311-a76b464c58ac)** - Learnings and experiment results

## Contributing
Feel free to raise an issue or submit a pull request if you find any mistakes or have suggestions for improvement. Your contributions are welcome and appreciated!

---

Happy Coding!
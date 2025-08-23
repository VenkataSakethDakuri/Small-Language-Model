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

## References
- **[Unsloth](https://unsloth.ai/)** - Fast and memory-efficient fine-tuning
- **[LLMCompressor](https://developers.redhat.com/articles/2024/08/14/llm-compressor-here-faster-inference-vllm#enabling_activation_quantization_in_vllm)** - Model compression for faster inference
- **[vLLM](https://docs.vllm.ai/en/latest/)** - High-performance LLM serving framework
- **[My Notebook](https://notebooklm.google.com/notebook/eee2c93a-12a8-4dba-9311-a76b464c58ac)** - Learnings and experiment results

## Contributing
Feel free to raise an issue or submit a pull request if you find any mistakes or have suggestions for improvement. Your contributions are welcome and appreciated!

---

Happy Coding!
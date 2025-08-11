"""
This code is an example of a pipeline using the Unsloth library with LoRA adapters. It stores the adapter separately
and later is used for inference. Adapted for Qwen2-0.5B (smallest compatible Qwen model).
"""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from datasets import load_dataset
import time
import psutil
import os
import gc
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

if __name__ == "__main__":

    # Load the smallest Qwen model compatible with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2-0.5B",  # Smallest Qwen2 variant (0.5B params)
       # max_seq_length=512,
        dtype=None,
     #   load_in_4bit=True,  # Use 4-bit during training to reduce memory
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Apply LoRA adapters (target modules adjusted for Qwen2 architecture)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        loftq_config=None,
        random_state=108
    )

    # def data_processing(training_data):
    #     instructions = training_data["instruction"]
    #     inputs = training_data["input"]
    #     outputs = training_data["output"]
    #     texts = []

    #     for instruction, input_text, output_text in zip(instructions, inputs, outputs):
    #         # Structure as a chat with system, user, and assistant
    #         messages = [
    #             {"role": "system", "content": "You are a helpful assistant."},
    #             {"role": "user", "content": f"{instruction}\n{input_text}" if input_text else instruction},
    #             {"role": "assistant", "content": output_text}
    #         ]
    #         # Apply Qwen2's chat template (assumes tokenizer has apply_chat_template)
    #         text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    #         texts.append(text)
        
    #     return {"text": texts}

    def data_processing(training_data):
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



    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:10000]")

    dataset = dataset.map(data_processing, batched=True)

    # SFTTrainer handles tokenization and collation internally
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,  # Can make training 5x faster for short sequences
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,  # Set this for 1 full training run
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=108,
            output_dir="TrainingUnsloth",
            report_to="none",
        ),
    )

    torch.cuda.reset_peak_memory_stats(device=0)

    train_time_start = time.time()

    trainer_result = trainer.train()

    torch.cuda.synchronize()  
    train_time_end = time.time()

    torch.cuda.empty_cache()

    reserved_memory = torch.cuda.max_memory_reserved(device=0)

    print("Training Metrics:")
    print(f"{train_time_end - train_time_start} seconds used for training.")
    print(f"Peak reserved memory = {round(reserved_memory / 1024 / 1024 / 1024, 3)} GB.")

    # Save the adapter and tokenizer separately (Unsloth handles base model loading)
    model.save_pretrained("adapterUnsloth")
    tokenizer.save_pretrained("adapterUnsloth")

    torch.cuda.empty_cache()
    #gc.collect()  # Clean up memory after training

    # Reload for inference (simplified for small model; no heavy offloading needed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="adapterUnsloth",
       # max_seq_length=512,
        dtype=None,
       # load_in_4bit=True,
        device_map="auto"  # Auto device placement
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    model.eval()

    def get_model_size(model):
        params = sum(p.numel() for p in model.parameters())
        size_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        return params, size_mb

    torch.cuda.reset_peak_memory_stats(device=0)

    # Measure model size
    param_count, model_size_mb = get_model_size(model)

    # Custom text streamer to measure time to first token
    class CustomTextStreamer(TextStreamer):
        def __init__(self, tokenizer, **kwargs):
            super().__init__(tokenizer, **kwargs)
            self.first_token_time = 0
            self.flag = False
        
        def put(self, value):
            if not self.flag:
                self.first_token_time = time.time()
                self.flag = True
            
            super().put(value)

    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Explain what is photosynthesis in simple terms.",  # instruction
            "",  # input
            "",  # output - leave this blank for generation!
        )
    ], return_tensors="pt").to("cuda")

    text_streamer = CustomTextStreamer(tokenizer)

    torch.cuda.reset_peak_memory_stats(device=0)

    start_time = time.time()

    with torch.no_grad():
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

    torch.cuda.synchronize()
    end_time = time.time()

    print(f"\nInference Metrics:")
    print(f"Model size: {model_size_mb:.2f} MB with {param_count} parameters")
    print(f"Time to first token: {text_streamer.first_token_time - start_time:.5f} seconds")
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    print(f"Peak reserved memory: {round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)} GB")

    torch.cuda.empty_cache()

    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained("mergedUnsloth")
    # tokenizer.save_pretrained("mergedUnsloth")


    # vllm_model = LLM(
    #         model="mergedUnsloth",
    #         gpu_memory_utilization=0.3, #reduce for avoiding OOM errors, increase for faster inference if you have enough GPU memory
    #         #enforce_eager=True,  # Disable advanced optimizations like CUDA Graphs to reduce temp allocations
    #     )

    #vLLM can be used without merging the adapter

    vllm_model = LLM(model="Qwen/Qwen2-0.5B", enable_lora=True, gpu_memory_utilization=0.3)

    sampling_params = SamplingParams(
        temperature=0.5,
        max_tokens=512,
        stop=["<|end_of_text|>"]
    )

    prompt = alpaca_prompt.format(
        "Explain what is photosynthesis in simple terms.",
        "",
        ""
    )

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device=0)

    vllm_start_time = time.time()

    response = vllm_model.generate([prompt], sampling_params=sampling_params, lora_request=LoRARequest("adapter", 1, "adapterUnsloth")) #

    torch.cuda.synchronize()  
    vllm_end_time = time.time()

    print(f"VLLM Inference time: {vllm_end_time - vllm_start_time:.5f} seconds")
    print(f"VLLM Response: {response[0].outputs[0].text}")
    print(f"VLLM Peak reserved memory: {round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)} GB")

    del vllm_model  
    torch.cuda.empty_cache()
"""
This code is a pipeline for making small language models using the LLM Compressor library with LoRA adapters. It stores the adapter separately
and later is used for inference. VLLM is used for inference. Adapted for Qwen/Qwen2-0.5B (smallest compatible Qwen model).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TextStreamer
from datasets import load_dataset
import time
import psutil
import os
from llmcompressor.transformers import oneshot
from llmcompressor.transformers import SparseAutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from vllm import LLM, SamplingParams
from peft import LoraConfig, get_peft_model, TaskType
import gc



#__main__ guard is needed so that the child process spawned by vLLM dont execute the entire script again
if __name__ == "__main__":

    model_name = "Qwen/Qwen2-0.5B"  # Smallest Qwen2 variant (0.5B params)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
      #  load_in_4bit=True  # Use 4-bit during training to reduce memory 
    )

    #loadin_4bit quantizes the weights only not activations. torch_dtype=torch.float16 reduces precision of both weights and activations.

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Apply LoRA adapters (target modules adjusted for Qwen2 architecture)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none"
    )

    model = get_peft_model(model, lora_config)

    def data_processing(training_data):
        instructions = training_data["instruction"]
        inputs = training_data["input"]
        outputs = training_data["output"]
        texts = []

        for instruction, input_text, output_text in zip(instructions, inputs, outputs):
            # Structure as a chat with system, user, and assistant
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{instruction}\n{input_text}" if input_text else instruction},
                {"role": "assistant", "content": output_text}
            ]
            # Apply Qwen2's chat template (assumes tokenizer has apply_chat_template)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        
        return {"text": texts}


    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1]")
    dataset = dataset.map(data_processing, batched=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Standard Trainer configuration
    training_args = TrainingArguments(
        output_dir="TrainingLLMCompressor",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=108,
        report_to="none",
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    torch.cuda.reset_peak_memory_stats(device=0)

    train_time_start = time.time()

    trainer_result = trainer.train()

    torch.cuda.synchronize()  
    train_time_end = time.time()

    torch.cuda.empty_cache()

    # Track final GPU memory and training time usage
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"\nTraining Metrics:")
    print(f"{train_time_end - train_time_start} seconds used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    
 
    model.save_pretrained("adapterLLMCompressor")
    tokenizer.save_pretrained("adapterLLMCompressor")

    # Add cache clearing here before loading new model
    torch.cuda.empty_cache()

    # Load the base model and apply the saved adapter
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=None, device_map="auto")
    model = get_peft_model(model, lora_config)
    model.load_adapter("adapterLLMCompressor", adapter_name="default")

    # Merge the adapter into the base model
    merged_model = model.merge_and_unload()

    # Add cache clearing here after merging and before quantization
    torch.cuda.empty_cache()

    calibration_dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1]")

    calibration_dataset = calibration_dataset.map(data_processing, batched=True)


    # Create calibration dataset for quantization
    # def create_calibration_dataset():
    #     calibration_samples = []
    #     for i in range(min(256, len(dataset))):
    #         sample = dataset[i]["text"]
    #         calibration_samples.append({"text": sample})
        
    #     return calibration_samples

    # calibration_dataset = create_calibration_dataset()

    def tokenize(sample):
        return tokenizer(sample["text"], padding=False, max_length=512, truncation=True, add_special_tokens=True)

    calibration_dataset = calibration_dataset.map(tokenize, batched=True, remove_columns=["instruction", "input", "output"])


    # Quantization recipe (experiment with schemes like W4A4, W8A8, etc.)
    gtpq_recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(
            targets="Linear",
            scheme="W4A16",  # Experiment with other schemes like W8A8 etc.
            ignore=["lm_head"]
        )
    ]

    # Apply quantization
    oneshot(
        model=merged_model,
        dataset=calibration_dataset,
        recipe=gtpq_recipe,
        output_dir="LLMCompressorInferenceQuantized"
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    # Load the quantized model
    model = SparseAutoModelForCausalLM.from_pretrained(
        "LLMCompressorInferenceQuantized"
    )

    #Better to explicity move to GPU as inputs are in GPU. Otherwise mismatch might occur where inputs are on GPU and model is on CPU.
    model = model.to("cuda")

    model.eval()

    def get_model_size(model):
        params = sum(p.numel() for p in model.parameters())
        size_mb = params * 4 / (1024 * 1024)  # Assume 4 bytes per param
        return params, size_mb

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
            "Continue the fibonnaci sequence.",  # instruction
            "1, 1, 2, 3, 5, 8",  # input
            "",  # output - leave this blank for generation!
        )
    ], return_tensors="pt").to("cuda")

    text_streamer = CustomTextStreamer(tokenizer)

    torch.cuda.reset_peak_memory_stats(device=0)

    start_time = time.time()

    with torch.no_grad():
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128, eos_token=None)

    end_time = time.time()

    print(f"\nInference Metrics:")
    print(f"Model size: {model_size_mb:.2f} MB with {param_count} parameters")
    print(f"Time to first token: {text_streamer.first_token_time - start_time:.2f} seconds")
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    print(f"Peak reserved memory: {round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)} GB")

    # Free up unused GPU memory
    torch.cuda.empty_cache()

   # vLLM inference
    vllm_model = LLM(
        model="LLMCompressorInferenceQuantized",
        gpu_memory_utilization=0.3, #reduce for avoiding OOM errors, increase for faster inference if you have enough GPU memory
        enforce_eager=True,  # Disable advanced optimizations like CUDA Graphs to reduce temp allocations
    )

    sampling_params = SamplingParams(
        temperature=0.5,
        max_tokens=512,
        stop=["<|end_of_text|>"]
    )

    prompt = alpaca_prompt.format(
        "Continue the fibonnaci sequence.",
        "1, 1, 2, 3, 5, 8",
        ""
    )

    torch.cuda.empty_cache()
    gc.collect()  

    torch.cuda.reset_peak_memory_stats(device=0)

    vllm_start_time = time.time()

    response = vllm_model.generate([prompt], sampling_params=sampling_params)

    torch.cuda.synchronize()  
    vllm_end_time = time.time()

    print(f"VLLM Inference time: {vllm_end_time - vllm_start_time:.2f} seconds")
    print(f"VLLM Response: {response[0].outputs[0].text}")
    print(f"VLLM Peak reserved memory: {round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)} GB")

    del vllm_model  
    torch.cuda.empty_cache()
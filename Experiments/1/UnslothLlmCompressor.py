"""
This code is an example of pipeline using the Unsloth and LLM Compressor library with LoRA adapters. It stores the adapter separately
and later is used for inference. Uses VLLM for inference.
"""

#Currently getting File "Qwen2ForCausalLM_8758051722619_autowrapped", line 4, in <module>
# the file from the modular. If any change should be done, please apply the change to the
#NameError: name 'BlockDiagonalCausalMask' is not defined

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from datasets import load_dataset
import time
import psutil
import os
from llmcompressor.transformers import oneshot
from llmcompressor.transformers import SparseAutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from vllm import LLM, SamplingParams




if __name__ == "__main__":

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        max_seq_length=512,
        #load_in_4bit=True,  # Use 4-bit during training, reduces training memory not final model size
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_alpha=16,
        lora_dropout=0, #increase to reduce overfitting
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        loftq_config=None,
        random_state=108
    )

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


    dataset = load_dataset("yahma/alpaca-cleaned", split = "train[:1]")

    dataset = dataset.map(data_processing, batched=True)


    #SFTTrainer does tokenzing and data collating internally, so no need to do it separately.
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 512,
        packing = False , # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1, # Set this for 1 full training run.
            #max_steps = 100, keep either num_train_epochs or max_steps, not both. Better to use max_steps for short training runs.
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 108,
            output_dir = "TrainingUnslothLLMCompressor",
            report_to = "none", 
        ),
    )

    torch.cuda.reset_peak_memory_stats(device=0)

    train_time_start = time.time()

    #start training
    trainer_result = trainer.train()

    torch.cuda.synchronize()  
    train_time_end = time.time()

    torch.cuda.empty_cache()

    print(f"\nTraining Metrics:")
    print(f"Training time: {train_time_end - train_time_start} seconds")
    reserved_memory = torch.cuda.max_memory_reserved(device=0)
    print(f"Peak reserved memory: {round(reserved_memory / 1024 / 1024 / 1024, 3)} GB.")

    # Saving the adapter and base model seperately without merging, use gguf for merging
    model.save_pretrained("adapterUnslothLLMCompressor")
    tokenizer.save_pretrained("adapterUnslothLLMCompressor")

    torch.cuda.empty_cache()

    #LoRA adapters modify internal layer computations, not the vocabulary or tokenization logic. So we need to keep tokenizer of base model only.
    # base_model = model.get_base_model()
    # base_model.save_pretrained("Experiments/1/base_model")
    # tokenizer.save_pretrained("Experiments/1/base_model")
    # print("Adapter and base model saved successfully.")

    #unsloth automatically loads the base model, need not save it seperately.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "adapterUnslothLLMCompressor", 
        max_seq_length = 512,
        dtype = None,  
        load_in_4bit = False, #Dont quantize using unsloth, use LLM Compressor for that.
       # low_cpu_mem_usage=False, 
        device_map="auto"  # Auto device placement
    )


    merged_model = model.merge_and_unload() 

    torch.cuda.empty_cache()  

    calibration_dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1]")

    calibration_dataset = calibration_dataset.map(data_processing, batched=True)

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

    #need to play around with different recipes for quantization.
    gtpq_recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(
            targets="Linear",
            scheme="W4A16",        #need to experiment with other schemes like W8A8, W4A8, etc.
            ignore=["lm_head"]
        )
    ]

    torch.cuda.empty_cache()

    oneshot(
        model=merged_model,
        dataset=calibration_dataset,
        recipe=gtpq_recipe,
        output_dir="UnslothLLMCompressorInferenceQuantized"
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""


    #Use the quantized model using LLMCompressor
    model = SparseAutoModelForCausalLM.from_pretrained(
        "UnslothLLMCompressorInferenceQuantized"
    )

    model = model.to("cuda")

    model.eval()

    def get_model_size(model):
        params = sum(p.numel() for p in model.parameters())
        size_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        return params, size_mb

    torch.cuda.reset_peak_memory_stats(device=0)

    # Measure model size
    param_count, model_size_mb = get_model_size(model)

    #custom text streamer to measure time to first token
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
            "Continue the fibonnaci sequence.", # instruction
            "1, 1, 2, 3, 5, 8", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    text_streamer = CustomTextStreamer(tokenizer)

    torch.cuda.reset_peak_memory_stats(device=0)

    start_time = time.time()

    with torch.no_grad():
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

    end_time = time.time()

    print(f"\nInference Metrics:")
    print(f"Model size: {model_size_mb:.2f} MB with {param_count} parameters")
    print(f"Time to first token: {text_streamer.first_token_time - start_time:.2f} seconds")
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    print(f"Peak reserved memory: {round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)} GB")


    #frees up unused GPU memory by emptying the CUDA memory cache
    torch.cuda.empty_cache()


    vllm_model = LLM(
        model="UnslothLLMCompressorInferenceQuantized",
        gpu_memory_utilization=0.3
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

    torch.cuda.reset_peak_memory_stats(device=0)

    vllm_start_time = time.time()

    response = vllm_model.generate([prompt], sampling_params=sampling_params
                                )

    vllm_end_time = time.time()


    print(f"VLLM Inference time: {vllm_end_time - vllm_start_time:.2f} seconds")
    print(f"VLLM Response: {response.outputs.text}") #check with response.outputs[0].text 
    print(f"VLLM Peak reserved memory: {round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)} GB")

   
    del vllm_model  
    torch.cuda.empty_cache()





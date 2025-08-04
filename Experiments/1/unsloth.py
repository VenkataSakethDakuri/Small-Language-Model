"""
This code is an example of training distilgpt2 using the Unsloth library with LoRA adapters. It stores the adapter separately
and later is used for inference. The model is trained on the Alpaca dataset. Base model is Quantized to 4 bit during inference
and training.
"""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from datasets import load_dataset
import time
import psutil
import os



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="distilgpt2",
    max_seq_length=512,
    dtype=None,  
    load_in_4bit=True,  # Use 4-bit during training, reduces training memory not final model size
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["c_attn", "c_proj", "c_fc"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    use_rslora = False,  
    loftq_config = None, 
    random_state = 108
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
    
    return {f"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split = "train[:1000]")

dataset = dataset.map(data_processing, batched=True)


#SFTTrainer does tokenzing and data collating internally, so no need to do it separately.
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        #max_steps = 100, keep either num_train_epochs or max_steps, not both.
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 108,
        output_dir = "outputsUnsloth",
        report_to = "none", 
    ),
)

train_time_start = time.time()

#start training
trainer.train()

train_time_end = time.time()

# Saving the adapter and base model seperately without merging, use gguf for merging
model.save_pretrained("Experiments/1/adapterUnsloth")
tokenizer.save_pretrained("Experiments/1/adapterUnsloth")

#LoRA adapters modify internal layer computations, not the vocabulary or tokenization logic. So we need to keep tokenizer of base model only.
# base_model = model.get_base_model()
# base_model.save_pretrained("Experiments/1/base_model")
# tokenizer.save_pretrained("Experiments/1/base_model")
# print("Adapter and base model saved successfully.")

#unsloth automatically loads the base model, need not save it seperately.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Experiments/1/adapterUnsloth", 
    max_seq_length = 512,
    dtype = None,  
    load_in_4bit = True,  # Uses 4-bit for base model not adapter.  LoRA adapters are tiny (~2-5MB), so quantizing them saves minimal memory
)



alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_memory_mb():
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024
    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    return cpu_mem + gpu_mem

def get_model_size(model):
    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4 / (1024 * 1024)  # Assume 4 bytes per param
    return params, size_mb

# Measure model size
param_count, model_size_mb = get_model_size(model)

start_time = time.time()


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

with torch.no_grad():
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

end_time = time.time()


print(f"Model size: {model_size_mb:.2f} MB with {param_count} parameters")
print(f"Time to first token: {text_streamer.first_token_time - start_time:.2f} seconds")
print(f"Inference time: {end_time - start_time:.2f} seconds")
print(f"Training time: {train_time_end - train_time_start:.2f} seconds")







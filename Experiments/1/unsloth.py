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



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="distilgpt2",
    max_seq_length=512,
    dtype=None,  
    load_in_4bit=True,  # Use 4-bit during training, reduces training memory not final model size
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["c_attn", "c_proj", "c_fc"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
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
    
    return texts

dataset = load_dataset("yahma/alpaca-cleaned", split = "train[:1000]")

dataset = dataset.map(data_processing, batched=True)

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
        output_dir = "outputs",
        report_to = "none", 
    ),
)


#start training
trainer.train()

# Saving the adapter and base model seperately without merging, use gguf for merging
model.save_pretrained("Experiments/1/adapter")
tokenizer.save_pretrained("Experiments/1/adapter")

#LoRA adapters modify internal layer computations, not the vocabulary or tokenization logic. So we need to keep tokenizer of base model only.
# base_model = model.get_base_model()
# base_model.save_pretrained("Experiments/1/base_model")
# tokenizer.save_pretrained("Experiments/1/base_model")
# print("Adapter and base model saved successfully.")

#unsloth automatically loads the base model, need not save it seperately.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Experiments/1/adapter", 
    max_seq_length = 512,
    dtype = None,  
    load_in_4bit = True,  # Uses 4-bit for base model not adapter.  LoRA adapters are tiny (~2-5MB), so quantizing them saves minimal memory
)
FastLanguageModel.for_inference(model) 


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)





import torch
import json
from transformers import AutoTokenizer
from datasets import load_dataset
import os

# Load model name from config.json
with open("config.json", "r") as f:
    config = json.load(f)
model_name = config["model_name"]

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
EOS_TOKEN = tokenizer.eos_token

# Define prompt format
prompt_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

# Function to format prompts
def formatting_prompts_func(examples):
    instructions = examples["prompt"]
    outputs = examples["response"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = prompt_format.format(instruction, output) + EOS_TOKEN  # Add EOS token at the end
        texts.append(text)
    return {"formatted_text": texts}

# Load and format dataset
dataset = load_dataset("amphora/krx-sample-instructions", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
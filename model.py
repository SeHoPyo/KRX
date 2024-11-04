import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import os
import numpy as np
import random

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# Authenticate with Hugging Face hub
login(config["hf_token"])

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Model parameters
model_name = config["model_name"]
torch_dtype = getattr(torch, config["torch_dtype"])
device_map = config["device_map"]
load_in_8bit = config["load_in_8bit"]

# Configure bitsandbytes if using 8-bit
quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)

# Load tokenizer and model with configurations
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype=torch_dtype,
    quantization_config=quantization_config if load_in_8bit else None,
    use_cache=config["use_cache"]
)

print("Model and tokenizer loaded successfully.")
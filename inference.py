import torch
from transformers import AutoTokenizer
from model import model
from dataset import prompt_format

# Initialize the tokenizer (already set up in model.py but re-importing to ensure context)
tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)

# Prepare model for inference
model.eval()  # Set model to evaluation mode
model = model.half()  # Use half-precision for faster inference if supported by the hardware

# Set up prompt for inference
prompt = prompt_format.format(
    "수요와 공급의 원리에 대해서 설명해줘.",  # instruction
    ""  # Leave output blank for generation
)

# Tokenize the input prompt
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

# Generate model output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)

# Decode and display the generated response
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("Generated Response:", response[0])
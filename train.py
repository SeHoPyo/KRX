import torch
import json
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
import os

# Import model, tokenizer, and dataset
from model import model, tokenizer
from dataset import dataset

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# Set training parameters
max_seq_length = config.get("max_length", 400)

# Initialize trainer with training arguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="formatted_text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Set to True if you want faster training on shorter texts
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
    ),
)

# Start training and log training statistics
trainer_stats = trainer.train()

# Optionally save the trainer stats
with open(os.path.join("outputs", "trainer_stats.json"), "w") as f:
    json.dump(trainer_stats.metrics, f, indent=4)

# Save the model and tokenizer after training
model.save_pretrained("outputs/trained_model")
tokenizer.save_pretrained("outputs/trained_model")

print("Training completed and model weights saved.")
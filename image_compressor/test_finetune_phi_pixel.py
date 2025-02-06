import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk
from torchvision import transforms
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
import random
import wandb
import os
from PIL import Image
import requests
from io import BytesIO
from functools import partial

from mem_forward import PhiCompressor
from dataset_utils import CustomEvaluationCallback, collate_fn


# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
    task_type="CAUSAL_LM",
)
# Configure training arguments
training_args = SFTConfig(
    output_dir="pixel_phi",  # Directory to save the model
    num_train_epochs=10,  # Number of training epochs
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    gradient_accumulation_steps=1,  # Steps to accumulate gradients
    # gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=1e-5,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=20,  # Steps interval for logging
    eval_steps=100,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    # save_strategy="steps",  # Strategy for saving the model
    save_steps=50000,  # Steps interval for saving
    # metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    # greater_is_better=False,  # Whether higher metric values are better
    # load_best_model_at_end=True,  # Load the best model after training
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.1,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    report_to="wandb",  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    # gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    #max_seq_length=1024  # Maximum sequence length for input
)
training_args.remove_unused_columns = False  # Keep unused columns in dataset



# Initialize wandb
wandb.init(
    project="phi35_mem_vision",  # change this
    name="phi35_mem_vision_pixel20k",  # change this
    config=training_args,
)
# Load the dataset
dataset = load_from_disk("pixelprose_with_images_20k_p")
dataset_length = len(dataset)
# Load the model
model = PhiCompressor(num_mem=1, device="cuda", lora_config=None)
# Configure collate function
custom_collate_fn = partial(collate_fn, processor=model.processor)
# Create the trainer
trainer = CustomEvaluationCallback(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset= dataset.select(range(dataset_length - 50, dataset_length)),
    data_collator=custom_collate_fn,
    tokenizer=model.processor.tokenizer,
)
# Train the model
trainer.train()

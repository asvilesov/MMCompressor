import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
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

class CustomEvaluationCallback(SFTTrainer):
    def evaluate(self, *args, **kwargs):
        # Run regular evaluation
        metrics = super().evaluate(*args, **kwargs)

        # Add sample answer generation
        num_samples = 5
        sample_inputs = self.gen_eval_data(num_samples=num_samples)
        sample_outputs = self.generate_sample_answers(sample_inputs)

        #Save Generations to text files and images
        global_step = self.state.global_step
        os.makedirs(f"eval_gen/step_{global_step}", exist_ok=True)
        folder_path = f"eval_gen/step_{global_step}"
        with open(f"{folder_path}/sample_outputs_{global_step}.txt", "w") as f:
            for i in range(num_samples):
                f.write(f"Step: {i}\n")
                f.write(f"GT: {sample_inputs[i]['A_text']}\n")
                f.write(f"Output: {sample_outputs[i]}\n")
                f.write("-------------\n")
                # images[i].save(f"{folder_path}/sample_image_{global_step}_{i}.png")
        return metrics

    def generate_sample_answers(self, sample_inputs):
        output_list = []
        for sample_input in tqdm(sample_inputs, total=len(sample_inputs)):
            output = self.model.generate(**sample_input)
            output_list.append(output)
        return output_list
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def gen_eval_data(self, num_samples=5):
        outputs = []
        
        dataset_size = len(self.train_dataset)
        # Select random indices
        random_indices = random.sample(range(dataset_size), num_samples)
        test_dataset = self.train_dataset.select(random_indices)
        for example in test_dataset:
            # Add each item into a list 
            for key in example:
                example[key] = [example[key]]
            outputs.append(collate_fn(example, self.tokenizer, 4, batchify=False))
        return outputs


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
    output_dir="mem_dict",  # Directory to save the model
    num_train_epochs=10,  # Number of training epochs
    per_device_train_batch_size=32,  # Batch size for training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    gradient_accumulation_steps=1,  # Steps to accumulate gradients
    # gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=1e-5,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=20,  # Steps interval for logging
    eval_steps=50,  # Steps interval for evaluation
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

# Initialize the tokenizer
model_id = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

new_tokens = ["[MEMORY]"]
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

def collate_fn(batch, tokenizer, num_memory_slots, batchify=True):
    # batchify the data
    if batchify:
        batch_dict = {}
        batch_dict['instruction'] = [item['instruction'] for item in batch]
        batch_dict['timesteps'] = [item['timesteps'] for item in batch]
        batch_dict['Q'] = [item['Q'] for item in batch]
        batch_dict['A'] = [item['A'] for item in batch]
        batch = batch_dict
    # Extract the relevant fields from the batch
    instructions = [item for item in batch['instruction']]
    time_steps = [item for item in batch['timesteps']]
    questions = [item for item in batch['Q']]
    answers = [item for item in batch['A']]

    messages = [[{"role": "user", "content": instruction + time_step[1]},
                {"role": "assistant", "content": "[MEMORY]"*num_memory_slots}] for instruction, time_step in zip(instructions, time_steps)]
    messages = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]

    qa = [[{"role": "user", "content": question}, {"role": "assistant", "content": answer}] 
                for question, answer in zip(questions, answers)]
    qa = [tokenizer.apply_chat_template(qa_pair, tokenize=False, add_generation_prompt=False) for qa_pair in qa]


    mem_tokenized = tokenizer(messages, padding=True, truncation=True, max_length=512)
    qa_tokenized = tokenizer(qa, padding=True, truncation=True, max_length=512)
    labels = qa_tokenized["input_ids"].copy()
    for i, label in enumerate(labels):
        assistant_idx = label.index(32001)
        labels[i] = torch.tensor([-100 if idx <= assistant_idx else label[idx] for idx in range(len(label))])
    labels = torch.stack(labels)

    # Move tokenized output to CUDA
    mem_tokenized = {key: torch.tensor(value) for key, value in mem_tokenized.items()}
    qa_tokenized = {key: torch.tensor(value) for key, value in qa_tokenized.items()}


    # Return the tokenized inputs and labels
    return {
        'memory_save': mem_tokenized,
        'QA': qa_tokenized,
        'labels': labels,
        'Q_text': questions,
        'A_text': answers,
    }

# Initialize wandb
wandb.init(
    project="mem_dict",  # change this
    name="mem_dict_initial",  # change this
    config=training_args,
)
# Load the dataset
dataset = load_from_disk("mem_dict_false")
dataset_length = len(dataset)
# Load the model
model = PhiCompressor(num_mem=4, device="cuda", lora_config=None, tokenizer=tokenizer)
# Configure collate function
custom_collate_fn = partial(collate_fn, tokenizer=tokenizer, num_memory_slots=4)
# Create the trainer
trainer = CustomEvaluationCallback(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset= dataset.select(range(dataset_length - 50, dataset_length)),
    data_collator=custom_collate_fn,
    tokenizer=model.tokenizer,
)
trainer.set_tokenizer(model.tokenizer)
# Train the model
trainer.train()

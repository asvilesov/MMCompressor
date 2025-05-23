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
from transformers.utils import logging
import random
# logging.set_verbosity_error() # suppress logging
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
from transformers.optimization import get_cosine_schedule_with_warmup

from mem_forward import VideoCompressor

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
                f.write(f"Example: {i}\n")
                num_timesteps = len(list(sample_outputs[i].keys()))
                for t in range(num_timesteps):
                    t_str = f"T{t}"
                    f.write(f"Time step {t}\n")
                    f.write(f"MP4 Path    : {sample_inputs[i][t_str]['mp4_path']}\n")
                    f.write(f"Question    : {sample_inputs[i][t_str]['q_str']}\n")
                    f.write(f"Ground Truth: {sample_inputs[i][t_str]['a_str']}\n")
                    f.write(f"Inference   : {sample_outputs[i][t_str]}\n")
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
            outputs.append(collate_fn([example], self.tokenizer, NUM_MEMORY_TOKENS, max_chunks=NUM_CHUNKS, batchify=False, cuda=True))
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
lr_scheduler_type = "cosine_with_min_lr"
lr_scheduler_kwargs = {"min_lr": 1e-6}
# Configure training arguments
training_args = SFTConfig(
    output_dir="mem_video",  # Directory to save the model
    num_train_epochs=6,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size for training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    gradient_accumulation_steps=4,  # Steps to accumulate gradients
    # gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=1e-4,  # Learning rate for training
    lr_scheduler_type=lr_scheduler_type,  # Type of learning rate scheduler
    lr_scheduler_kwargs=lr_scheduler_kwargs,  # Arguments for learning rate scheduler
    # warmup_steps=200,
    # Logging and evaluation
    logging_steps=20,  # Steps interval for logging
    eval_steps=50,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    # include_inputs_for_metrics=True,  # Include input data in metrics
    # save_strategy="steps",  # Strategy for saving the model
    save_steps=50000,  # Steps interval for saving
    # metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    # greater_is_better=False,  # Whether higher metric values are better
    # load_best_model_at_end=True,  # Load the best model after training
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.1,  # Maximum norm for gradient clipping
    warmup_ratio=0.01,  # Ratio of total steps for warmup
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
model_id = "DAMO-NLP-SG/VideoLLaMA3-2B"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
tokenizer = processor.tokenizer
NUM_MEMORY_TOKENS = 64
new_tokens = [f"[MEMORY{i}]" for i in range(NUM_MEMORY_TOKENS)]
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

def collect_and_pad(batch, tokenizer):
    pad_token = 41234 #tokenizer.pad_token_id
    max_len = max([len(b_item["input_ids"][0]) for b_item in batch])
    batch_dict = {"input_ids": [], "attention_mask": []}
    # input_ids
    for b_item in batch:
        # Pad the input_ids to the max length
        input_ids = b_item["input_ids"]
        padding_length = max_len - len(input_ids[0])
        input_ids = torch.cat([input_ids[0], pad_token*torch.ones(padding_length, dtype=torch.long)])
        batch_dict["input_ids"].append(input_ids)
    batch_dict["input_ids"] = torch.stack(batch_dict["input_ids"])
    # attention_mask
    for b_item in batch:
        # Pad the attention_mask to the max length
        attention_mask = b_item["attention_mask"]
        padding_length = max_len - len(attention_mask[0])
        attention_mask = torch.cat([attention_mask[0], torch.ones(padding_length, dtype=torch.long)])
        batch_dict["attention_mask"].append(attention_mask)
    batch_dict["attention_mask"] = torch.stack(batch_dict["attention_mask"])

    # assum no padding required for image related items
    for key in ["pixel_values", "grid_sizes", "merge_sizes", "modals"]:
        if key in batch[0]:
            batch_dict[key] = batch[0][key]
        #     if(key != "modals"):
        #         batch_dict[key] = torch.stack([b_item[key] for b_item in batch])
        #     else:
        #         batch_dict[key] = [b_item[key] for b_item in batch]
    return batch_dict


def collate_fn(batch, tokenizer, num_memory_slots, max_chunks=3, chunk_time=10, batchify=False, cuda=False):
    # batchify the data
    if batchify:
        batch_dict = {}
        batch_dict['mp4'] = [item['mp4'] for item in batch]
        batch_dict['chunks'] = [item['chunks'] for item in batch]
        batch = batch_dict
    # Extract the relevant fields from the batch
    chunks = batch[0]
    mp4_path = chunks['vid_path']
    q_types = ["description", "timestep", "qa"]



    full_batch_dict = {}
    num_chunks = len(chunks) if len(chunks) < max_chunks else max_chunks
    #hotfix for now
    for t in range(max_chunks):
        chunks[f"T{t}_timestep"]["Q"] = chunks[f"T{t}_timestep"]["Q:"]
    
    b_size = 1 #required for video models
    past_sample = False
    mem_token_string = ""
    for i in range(num_memory_slots):
        mem_token_string += f"[MEMORY{i}]"
    for t in range(num_chunks):
        start_time = chunk_time*t
        end_time = chunk_time*(t+1)
        # Tokenize the instructions, time steps, questions, and answers
        messages = [[{"role": "user", "content": 
                        [
                            {"type": "video", "video": {"video_path": mp4_path, "fps": 1, "max_frames": chunk_time, 
                                                        "start_time":start_time, "end_time":end_time}},
                                                        # "start_time":chunks[t]['interval'][0], "end_time":chunks[t]['interval'][1]}},
                            {"type": "text", "text": " "},
                        ]},
                    {"role": "assistant", "content": mem_token_string}] for i in range(b_size)]
        messages = [processor(conversation=message, return_tensors="pt") for message in messages]
        messages = collect_and_pad(messages, tokenizer)

        if past_sample:
            sample_t = random.choice(list(range(t+1)))
            sample_q_type = random.choice(q_types)
            q_str = chunks[f'T{sample_t}_{sample_q_type}']['Q']
            a_str = chunks[f'T{sample_t}_{sample_q_type}']['A']
        else:
            sample_q_type = random.choice(q_types)
            q_str = chunks[f'T{t}_{sample_q_type}']['Q']
            a_str = chunks[f'T{t}_{sample_q_type}']['A']
        qa = [[{"role": "user", "content": q_str},
                {"role": "assistant", "content": a_str}]]
        qa = [processor(conversation=message, return_tensors="pt") for message in qa]
        qa = collect_and_pad(qa, tokenizer)
        labels = []
        for i in range(qa["input_ids"].shape[0]):
            assistant_token_idx = torch.where(qa["input_ids"][i] == tokenizer.convert_tokens_to_ids("assistant"))[0][0]
            labels_item = qa["input_ids"][i].clone()
            labels_item[0:assistant_token_idx+1] = -100
            labels.append(labels_item)
        labels = torch.stack(labels)
        qa["labels"] = labels

        # # Move tokenized output to CUDA
        # mem_tokenized = {key: torch.tensor(value) for key, value in mem_tokenized.items()}
        # qa_tokenized = {key: torch.tensor(value) for key, value in qa_tokenized.items()}
        # labels = labels
        if(cuda):
            messages = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in messages.items()}
            if "pixel_values" in messages:
                messages["pixel_values"] = messages["pixel_values"].to(torch.bfloat16)
            qa = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in qa.items()}
        
        full_batch_dict[f"T{t}"] = {
            'memory_save': messages,
            'QA': qa,
            'q_str': q_str,
            'a_str': a_str,
            'mp4_path': mp4_path
        }
    return full_batch_dict

# Initialize wandb
wandb.init(
    project="mem_dict",  # change this
    name=f"lora_8x",  # change this
    config=training_args,
)
# Load the dataset
dataset = load_from_disk("/home/ubuntu/temp/10k_vid_dataset/finevideo_dataset")
dataset_length = len(dataset)
# Load the model
NUM_CHUNKS = 2
NUM_MEMORY_TOKENS = 64
model = VideoCompressor(num_mem=NUM_MEMORY_TOKENS, 
                        device="cuda", tokenizer=tokenizer, 
                        lora_config=peft_config, freeze_vision_encoder=False)
# Configure collate function
custom_collate_fn = partial(collate_fn, tokenizer=tokenizer, num_memory_slots=NUM_MEMORY_TOKENS, max_chunks=NUM_CHUNKS)
# Create the trainer
trainer = CustomEvaluationCallback(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset= dataset.select(range(dataset_length - 10, dataset_length)),
    data_collator=custom_collate_fn,
    tokenizer=model.tokenizer,
)
trainer.set_tokenizer(model.tokenizer)
trainer.can_return_loss = True
# Train the model
trainer.train()

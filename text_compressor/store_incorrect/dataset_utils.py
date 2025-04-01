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

class CustomEvaluationCallback(SFTTrainer):
    def evaluate(self, *args, **kwargs):
        # Run regular evaluation
        metrics = super().evaluate(*args, **kwargs)

        # Add sample answer generation
        num_samples = 5
        sample_inputs, sample_inputs_text, images = self.gen_eval_data(num_samples=num_samples)
        sample_outputs = self.generate_sample_answers(sample_inputs)

        #Save Generations to text files and images
        global_step = self.state.global_step
        os.makedirs(f"eval_gen/step_{global_step}", exist_ok=True)
        folder_path = f"eval_gen/step_{global_step}"
        with open(f"{folder_path}/sample_outputs_{global_step}.txt", "w") as f:
            for i in range(num_samples):
                f.write(f"Step: {i}\n")
                f.write(f"GT: {sample_inputs_text[i]}\n")
                f.write(f"Output: {sample_outputs[i]}\n")
                f.write("-------------\n")
                images[i].save(f"{folder_path}/sample_image_{global_step}_{i}.png")
        return metrics

    def generate_sample_answers(self, sample_inputs):
        output_list = []
        for sample_input in tqdm(sample_inputs, total=len(sample_inputs)):
            output = self.model.generate(sample_input)
            output_list.append(output)
        return output_list
    
    def set_processor(self, processor):
        self.processor = processor

    def gen_eval_data(self, num_samples=5):
        outputs = []
        outputs_text = []
        images = []
        
        dataset_size = len(self.train_dataset)
        # Select random indices
        random_indices = random.sample(range(dataset_size), num_samples)
        test_dataset = self.train_dataset.select(random_indices)

        for example in test_dataset:
            text = f"<|user|>\n<|image_1|>\nDescribe the image.<|end|>\n<|assistant|>"
            image = example['image']
            input_data = self.processor(text=text, images=image, return_tensors="pt", padding=True)
            outputs.append(input_data)
            outputs_text.append(example['vlm_caption'])
            images.append(image)
        return outputs, outputs_text, images

def get_image(url):
    try:
        response = requests.get(url, timeout=0.3)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return False

# Create a data collator to encode text and image pairs
MAX_LENGTH = 1100
def collate_fn(examples, processor):
    # Get the texts and images, and apply the chat template
    text_originals = [example['vlm_caption'] for example in examples]  # Prepare texts for processing
    images = [example['image'] for example in examples]  # Prepare images for processing
    texts = [f"<|user|>\n<|image_1|>\nDescribe the image.<|end|>\n<|assistant|>\n{example}<|end|><|endoftext|>\n" for example in text_originals]
    # Find assistant token and mask everything before it
    assistant_token = processor.tokenizer.convert_tokens_to_ids("<|assistant|>")
    pad_token = processor.tokenizer.pad_token_id
    # Tokenize the texts and process the images
    batch = [processor(text=text, images=image, return_tensors="pt", padding=True) for text, image in zip(texts, images)]  # Encode texts and images into tensors
    # Pad the input IDs and attention masks to the maximum sequence length in the batch and merge
    max_size = max([example["input_ids"].shape[1] for example in batch])
    for item_ in batch:
        labels = item_["input_ids"].clone()
        pad_labels = -100*torch.ones(max_size-labels.shape[1], dtype=labels.dtype)
        item_["labels"] = torch.concatenate((labels[0], pad_labels)).reshape((1, -1))
        pad_tokens = 32007*torch.ones(max_size-len(item_["input_ids"][0]), dtype=item_["input_ids"][0].dtype) # avoid left-padding KV-cache issue
        item_["input_ids"] = torch.concatenate((item_["input_ids"][0], pad_tokens)).reshape((1, -1))
        item_["attention_mask"] = torch.concatenate((item_["attention_mask"][0], torch.ones(max_size-len(item_["attention_mask"][0])))).reshape((1, -1))
    merged_batch = {}
    for key in batch[0]:
        merged_batch[key] = torch.concatenate([example[key] for example in batch])
    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    # Mask image token IDs in the labels
    image_tokens = [-1]  # Convert image token to ID
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels
    # Mask the question token IDs in the labels <system>___<end><user>___<end><assistant>___<end>
    assistant_token_index = (merged_batch["labels"] == assistant_token).nonzero()
    for i in range(len(merged_batch["labels"])):
        merged_batch["labels"][i, :assistant_token_index[i,1]] = -100
    # Truncate Batch 
    if merged_batch["input_ids"].shape[1] > MAX_LENGTH:
        merged_batch["input_ids"] = merged_batch["input_ids"][:, :MAX_LENGTH]
        merged_batch["attention_mask"] = merged_batch["attention_mask"][:, :MAX_LENGTH]
        merged_batch["labels"] = merged_batch["labels"][:, :MAX_LENGTH]
    return merged_batch  # Return the prepared batch
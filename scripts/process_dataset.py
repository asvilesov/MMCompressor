from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from tqdm import tqdm
import torch

# Step 1: Load the dataset
dataset = load_dataset("tomg-group-umd/pixelprose", num_proc=8)

def remove_transparency(img, bg_color=(255, 255, 255)):
    """Convert transparent images to RGB with a solid background."""
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, bg_color)
        background.paste(img, mask=img.split()[-1])  # Paste with alpha channel as mask
        return background
    else:
        return img.convert('RGB')

# Step 2: Define the image downloading function
def download_and_process_image(example):
    image_url = example["url"]  # Adjust this key if your dataset uses a different field name
    try:
        response = requests.get(image_url, timeout=1)
        response.raise_for_status()
        img = remove_transparency(Image.open(BytesIO(response.content)).convert("RGB"))
        clean_img = Image.new("RGB", img.size)
        clean_img.paste(img)
        example["image"] = clean_img

    except Exception as e:
        # print(f"Failed to download {image_url}: {e}")
        example["image"] = None  # Use None if the image can't be downloaded
    return example

# Step 3: Apply the function to the dataset
subset = dataset["train"].select(range(20000))
dataset_with_images = subset.map(download_and_process_image, num_proc=64)

# Step 4: Filter out examples where the image failed to download
dataset_with_images = dataset_with_images.filter(lambda example: example["image"] is not None)

# Now you can use dataset_with_images for training or further processing
print(dataset_with_images)
print(len(dataset_with_images))

model_id = "microsoft/Phi-3.5-vision-instruct" 
# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, 
  trust_remote_code=True, 
  num_crops=4
) 

def filter_dataset(dataset, processor):
    def is_valid_image(example):
        text = "<|user|>\n<|image_1|>\nDescribe the image."
        processed_image_shape = processor(text, example['image'], return_tensors="pt")['image_sizes']
        return torch.equal(processed_image_shape[0], torch.tensor([672, 672]))
    
    filtered_dataset = dataset.filter(is_valid_image)
    return filtered_dataset

print(f"Original dataset size: {len(dataset_with_images)}")
filtered_dataset = filter_dataset(dataset_with_images, processor)
print(f"Filtered dataset size: {len(filtered_dataset)}")

filtered_dataset.save_to_disk("pixelprose_with_images_20k_p")

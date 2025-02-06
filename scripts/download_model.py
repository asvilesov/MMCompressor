from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import torch


model_id = "microsoft/Phi-3.5-vision-instruct"

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )

from huggingface_hub import snapshot_download

snapshot_download(repo_id=model_id)
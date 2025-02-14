from datasets import load_dataset

from datasets import load_dataset

# Load dataset in streaming mode to avoid full download
# dataset = load_dataset("zhangyifei55/LLaVA-Instruct-150K", split="train")

# dataset2 = load_dataset("HuggingFaceM4/ChartQA")
# dataset3= load_dataset("databricks/databricks-dolly-15k")

dataset = load_dataset("tomg-group-umd/pixelprose", num_proc=8)
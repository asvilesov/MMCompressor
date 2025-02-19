from datasets import load_dataset
import os

#full dataset (600GB of data)
dataset = load_dataset("HuggingFaceFV/finevideo", split="train", num_proc=12)
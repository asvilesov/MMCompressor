from huggingface_hub import snapshot_download

# Specify the model ID (e.g., "bert-base-uncased")
model_id = "microsoft/Phi-3.5-mini-instruct"

# Download the model
snapshot_download(repo_id=model_id)


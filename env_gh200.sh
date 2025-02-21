# Run from root
pip install datasets transformers peft pillow==11 jinja2==3.1.0 tf-keras trl wandb ipywidgets jupyter tqdm

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 # Currently, only the PyTorch nightly has wheels for aarch64 with CUDA.
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py # remove all vllm dependency specification of pytorch
pip install -r requirements-build.txt # install the rest build time dependency
pip install -vvv -e . --no-build-isolation # use --no-build-isolation to build with the current pytorch

# Install Triton otherwise throws Triton Module Not Found
git clone https://github.com/triton-lang/triton.git
cd triton
pip install ninja cmake wheel pybind11 # build-time dependencies
pip install -e python
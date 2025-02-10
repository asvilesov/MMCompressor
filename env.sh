conda remove -n mmcompressor --all -y
conda create -n mmcompressor python=3.10 -y

source activate mmcompressor

which python
which pip

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
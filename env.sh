conda remove -n mmcompressor --all -y
conda create -n mmcompressor python=3.10 -y

source activate mmcompressor

which python
which pip

pip install -r requirements.txt
# Quick Start

Install miniconda or anaconda, then prepare an environment:
```bash
conda create python=3.7 --name nnpf
conda activate nnpf

conda install numpy scipy matplotlib jupyter

conda config --add channels pytorch
conda install pytorch torchvision cpuonly
```


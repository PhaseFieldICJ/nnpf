# Quick Start

**Download** or clone this repository:
```bash
git clone https://plmlab.math.cnrs.fr/phase-field-icj/nn-phase-field.git
cd nn-phase-field
```

**Install** miniconda or anaconda, then prepare an environment:
```bash
conda create python=3.7 --name nnpf
conda activate nnpf

conda config --add channels pytorch
conda install numpy matplotlib jupyter pytorch torchvision cpuonly pytorch-lightning=0.8.5
```

**Test** the installation with:
```bash
python3 self_test.py
```

**Launch** the learning of the reaction term of the Allen-Cahn equation, with default parameters:
```bash
python3 reaction_model.py --batch_size 10
```
and/or with custom hidden layers:
```bash
python3 reaction_model.py --batch_size 10 --layer_dims 8 8 3 --activation ReLU
```

**Visualize** the loss evolution and compare hyper-parameters using TensorBoard:
```bash
tensorboard --logdir logs
```
and open your browser at http://localhost:6006/


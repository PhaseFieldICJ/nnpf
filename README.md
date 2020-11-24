# Quick Start

**Download** or clone this repository:
```bash
git clone https://plmlab.math.cnrs.fr/phase-field-icj/nn-phase-field.git
cd nn-phase-field
```

**Install** miniconda or anaconda, then prepare an environment:
- for CPU only:
```bash
conda env create -f environment_cpu.yml
conda activate nnpf_cpu
```
- for CPU and GPU:
```bash
conda env create -f environment_gpu.yml
conda activate nnpf_gpu
```

To **update** instead an already created environement:
```bash
conda env update -f environment_cpu.yml
```
and/or
```bash
conda env update -f environment_gpu.yml
```
depending on which environment you already have.

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

If you have an CUDA compatible GPU, you can speedup the learning by simply adding the `--gpu` option:
```bash
python3 reaction_model.py --batch_size 10 --layer_dims 8 8 3 --activation ReLU --gpu 1
```


**Visualize** the loss evolution and compare hyper-parameters using TensorBoard:
```bash
tensorboard --logdir logs
```
and open your browser at http://localhost:6006/


# Quick Start

**Download** or clone this repository:
```bash
git clone https://github.com/PhaseFieldICJ/nnpf
cd nnpf
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

**Install** the nnpf module:
```bash
pip install .
```

You can now move in your **working directory**.

**Test** the installation with:
```bash
nnpf selftest
```

**Launch** the learning of the reaction term of the Allen-Cahn equation, with default parameters:
```bash
nnpf train Reaction --batch_size 10
```
and/or with custom hidden layers:
```bash
nnpf train Reaction --batch_size 10 --layer_dims 8 8 3 --activation ReLU
```

If you have an CUDA compatible GPU, you can speedup the learning by simply adding the `--gpu` option:
```bash
nnpf train --batch_size 10 --layer_dims 8 8 3 --activation ReLU --gpu 1
```

**Check** informations of one trained model:
```bash
nnpf infos logs/Reaction/version_0
```

**Visualize** the loss evolution and compare hyper-parameters using TensorBoard:
```bash
tensorboard --logdir logs
```
and open your browser at http://localhost:6006/


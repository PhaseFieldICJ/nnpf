# Quick Start

## Python

This package requires a Python **version between 3.8 and 3.10** (versions above have not been tested).

You can use a currently installed Python or, for example, **create a miniconda/anaconda environment**:
```bash
conda create --name nnpf python=3.10
conda activate nnpf
```

If you need a specific computation platform context, like an older CUDA version or the support of ROCm, you should install PyTorch manually using the instructions available on the [official website](https://pytorch.org/get-started/locally/).

## Install from Pypi

```bash
pip install nnpf
```

## Install from source

**Download** or clone this repository:
```bash
git clone https://github.com/PhaseFieldICJ/nnpf
cd nnpf
```

**Install** the nnpf module:
```bash
pip install .
```

You can now move in your **working directory**.

## Self test

You can check the installation with:
```bash
nnpf selftest
```

## Basic training

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
nnpf train Reaction --batch_size 10 --layer_dims 8 8 3 --activation ReLU --gpu 1
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


# Custom model

You can also create a custom model in a file and make it derives from the problem you want to solve.

For example, create a file `model.py` with the following content:
```Python
from torch.nn import Sequential

from nnpf.problems import AllenCahnProblem
from nnpf.models import Reaction, HeatArray
from nnpf.utils import get_default_args

class ModelDR(AllenCahnProblem):
    def __init__(self,
                 kernel_size=17, kernel_init='zeros',
                 layers=[8, 3], activation='GaussActivation',
                 **kwargs):
        super().__init__(**kwargs)

        # Fix kernel size to match domain dimension
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        else:
            kernel_size = list(kernel_size)
        if len(kernel_size) == 1:
            kernel_size = kernel_size * self.domain.dim

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters(
          'kernel_size', 'kernel_init',
          'layers', 'activation',
        )

        self.model = Sequential(
            HeatArray(
                kernel_size=kernel_size, init=kernel_init,
                bounds=self.hparams.bounds, N=self.hparams.N
            ),
            Reaction(layers, activation),
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = AllenCahnProblem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("Allen-Cahn DR", "Options specific to this model")
        group.add_argument('--kernel_size', type=int, nargs='+', help='Size of the kernel (nD)')
        group.add_argument('--kernel_init', choices=['zeros', 'random'], help="Initialization of the convolution kernel")
        group.add_argument('--layers', type=int, nargs='+', help='Sizes of the hidden layers')
        group.add_argument('--activation', type=str, help='Name of the activation function')
        group.set_defaults(**{**get_default_args(ModelDR), **defaults})
        return parser
```
with some boilerplate to handle command-line arguments and save hyper-parameters (see Lightning documentation).
`ModelDR` is declared as a model of the Allen-Cahn problem and thus inherits from the associated training and validation datasets.

You can then display the command-line interface with the `--help` option after specifying the file and the model name:
```bash
nnpf train model.py:ModelDR --help
```

You can start the training for the 2D case with some custom arguments:
```bash
nnpf train model.py:ModelDR --kernel_size 33 --max_epochs 2000 --check_val_every_n_epoch 100
```

For the 3D case:
```bash
nnpf train model.py:ModelDR --bounds [0,1]x[0,1]x[0,1] --N 64 --max_epochs 2000 --check_val_every_n_epoch 100
```

Using a GPU:
```bash
nnpf train model.py:ModelDR --bounds [0,1]x[0,1]x[0,1] --N 64 --max_epochs 2000 --check_val_every_n_epoch 100 --gpus 1
```

Using a configuration file in YAML format:
```bash
nnpf train model.py:ModelDR --config config.yml
```


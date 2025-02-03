<div align="center">
	<img src="https://github.com/ML-KULeuven/congrads/blob/main/docs/_static/congrads_export.png?raw=true" height="200">
	<p>
	<b>Incorporate constraints into neural network training for more reliable and robust models.</b>
	</p>
	<br/>

[![PyPi](https://img.shields.io/pypi/v/congrads.svg)](https://pypi.org/project/congrads)
[![Read the Docs](https://img.shields.io/readthedocs/congrads/latest.svg?label=Read%20the%20Docs)](https://congrads.readthedocs.io)
[![Python Version: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://pypi.org/project/congrads)
[![Downloads](https://img.shields.io/pypi/dm/congrads.svg)](https://pypistats.org/packages/congrads)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<br/>
<br/>
</div>

**Congrads** is a Python toolbox that brings **constraint-guided gradient descent** capabilities to your machine learning projects. Built with seamless integration into PyTorch, Congrads empowers you to enhance the training and optimization process by incorporating constraints into your training pipeline.

Whether you're working with simple inequality constraints, combinations of input-output relations, or custom constraint formulations, Congrads provides the tools and flexibility needed to build more robust and generalized models.

## Key Features

- **Constraint-Guided Training**: Add constraints to guide the optimization process, ensuring that your model generalizes better by trying to satisfy the constraints.
- **Flexible Constraint Definition**: Define constraints on inputs, outputs, or combinations thereof, using an intuitive and extendable interface. Make use of pre-programmed constraint classes or write your own.
- **Seamless PyTorch Integration**: Use Congrads within your existing PyTorch workflows with minimal setup.
- **Flexible and extendible**: Write your own custom networks, constraints and dataset classes to easily extend the functionality of the toolbox.

## Getting Started

### 1. **Installation**

First, make sure to install PyTorch since Congrads heavily relies on its deep learning framework. Please refer to the [PyTorch's getting started guide](https://pytorch.org/get-started/locally/). Make sure to install with CUDA support for GPU training.

Next, install the Congrads toolbox. The recommended way to install it is to use pip:

```bash
pip install congrads
```

This should automatically install all required dependencies for you. If you would like to install dependencies manually, Congrads depends on the following:

- Python 3.9 - 3.12
- **PyTorch** (install with CUDA support for GPU training, refer to [PyTorch's getting started guide](https://pytorch.org/get-started/locally/))
- **NumPy** (install with `pip install numpy`, or refer to [NumPy's install guide](https://numpy.org/install/).)
- **Pandas** (install with `pip install pandas`, or refer to [Panda's install guide](https://pandas.pydata.org/docs/getting_started/install.html).)
- **Tqdm** (install with `pip install tqdm`)
- **Torchvision** (install with `pip install torchvision`)
- **Tensorboard** (install with `pip install tensorboard`)

### 2. **Core concepts**

Before diving into the toolbox, it is recommended to familiarize yourself with Congrads's core concept and topics.
Please read the documentation at https://congrads.readthedocs.io/en/latest/ to get up-to-date.

### 3. **Basic Usage**

Below, a basic example can be found that illustrates how to work with the Congrads toolbox.
For additional examples, refer to the [examples](https://github.com/ML-KULeuven/congrads/tree/main/examples) and [notebooks](https://github.com/ML-KULeuven/congrads/tree/main/notebooks) folders in the repository.

#### 1. First, select the device to run your code on with.

```python
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
```

#### 2. Next, load your data and split it into training, validation and testing subsets.

```python
data = BiasCorrection(
    "./datasets", preprocess_BiasCorrection, download=True
)
loaders = split_data_loaders(
    data,
    loader_args={"batch_size": 100, "shuffle": True},
    valid_loader_args={"shuffle": False},
    test_loader_args={"shuffle": False},
)
```

#### 3. Instantiate your neural network, make sure the dimensions match up with your data.

```python
network = MLPNetwork(25, 2, n_hidden_layers=3, hidden_dim=35)
network = network.to(device)
```

#### 4. Choose your loss function and optimizer.

```python
criterion = MSELoss()
optimizer = Adam(network.parameters(), lr=0.001)
```

#### 5. Then, setup the descriptor, that will attach names to specific parts of your network.

```python
descriptor = Descriptor()
descriptor.add("output", 0, "Tmax")
descriptor.add("output", 1, "Tmin")
```

#### 6. Define your constraints on the network.

```python
Constraint.descriptor = descriptor
Constraint.device = device
constraints = [
    ScalarConstraint("Tmin", ge, 0),
    ScalarConstraint("Tmin", le, 1),
    ScalarConstraint("Tmax", ge, 0),
    ScalarConstraint("Tmax", le, 1),
    BinaryConstraint("Tmax", gt, "Tmin"),
]
```

#### 7. Instantiate metric manager and core, and start the training.

```python
metric_manager = MetricManager()
core = CongradsCore(
    descriptor,
    constraints,
    loaders,
    network,
    criterion,
    optimizer,
    metric_manager,
    device,
    checkpoint_manager,
)

core.fit(max_epochs=50)
```

## Example Use Cases

- **Optimization with Domain Knowledge**: Ensure outputs meet real-world restrictions or safety standards.
- **Improve Training Process**: Inject domain knowledge in the training stage, increasing learning efficiency.
- **Physics-Informed Neural Networks (PINNs)**: Coming soon, Enforce physical laws as constraints in your models.

## Roadmap

- [ ] Add ODE/PDE constraints to support PINNs
- [ ] Add support for constraint parser that can interpret equations
- [ ] Determine if it is feasible to add unit and or functional tests

## Research

If you make use of this package or it's concepts in your research, please consider citing the following papers.

- Van Baelen, Q., & Karsmakers, P. (2023). **Constraint guided gradient descent: Training with inequality constraints with applications in regression and semantic segmentation.**
  Neurocomputing, 556, 126636. doi:10.1016/j.neucom.2023.126636 <br/>[ [pdf](https://www.sciencedirect.com/science/article/abs/pii/S0925231223007592) | [bibtex](https://raw.githubusercontent.com/ML-KULeuven/congrads/main/docs/_static/VanBaelen2023.bib) ]

## Contributing

We welcome contributions to Congrads! Whether you want to report issues, suggest features, or contribute code via issues and pull requests.

## License

Congrads is licensed under the [The 3-Clause BSD License](LICENSE). We encourage companies that are interested in a collaboration for a specific topic to contact the authors for more information or to set up joint research projects.

---

Elevate your neural networks with Congrads! 🚀

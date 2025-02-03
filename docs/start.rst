.. _start:

Quick Start Guide
=================

To get started using the Congrads toolbox, we provide some basic instructions and informative explanations below to get you up-to-date.
We will first elaborate on the installation procedure to then explain some core concepts in more detail.
Finally, we continue with a step-by-step tutorial to build your first constraint-guided neural network.

Installation
------------

Since Congrads is a Python toolbox, it requires Python to be installed to function. Congrads currently supports Python version 3.9 - 3.12.
Get the latest version of Python at https://www.python.org/downloads/.

Congrads heavily relies on PyTorch as its main deep learning framework, leveraging its automatic differentiation and tensor computation capabilities to implement constraint-guided gradient descent.
Please refer to the `PyTorch's getting started guide <https://pytorch.org/get-started/locally/>`_ and follow the installation procedure.
To enable GPU training, install PyTorch with CUDA support.

Next, install the Congrads package. The recommended way to install the toolbox is to use pip::

  pip install congrads

This should automatically install all required dependencies for you.
If you want to manually install all dependencies, Congrads depends on the following:

* **PyTorch** (install with CUDA support for GPU training, refer to `PyTorch's getting started guide <https://pytorch.org/get-started/locally/>`_)
* **NumPy** (install with ```pip install numpy```, or refer to `NumPy's install guide <https://numpy.org/install/>`_)
* **Pandas** (install with ```pip install pandas```, or refer to `Panda's install guide <https://pandas.pydata.org/docs/getting_started/install.html>`_)
* **Tqdm** (install with ```pip install tqdm```)
* **Torchvision** (install with ```pip install torchvision```)
* **Tensorboard** (install with ```pip install tensorboard```)

Requirements and concepts
-------------------------

Before diving into the toolbox, it is probably a good idea to familiarize yourself with its concepts.
The :ref:`core concepts page <concepts>` provides a good starting point that explains some crucial components in the toolbox, along with examples and things to keep in mind.

Congrads requires minimal conditions to integrate your code with the toolbox.
You have to structure your code to comply with the following:

* Your data needs to be provided as a PyTorch Dataset class
* Your network (model) needs to extend torch.nn.Module
* Your network (model) needs to output a dictionary having key "output"

Example
-------

1. First, select the device to run your code on with.

.. code-block:: python

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

2. Next, load your data and split it into training, validation and testing subsets.

.. code-block:: python

    data = BiasCorrection(
        "./datasets", preprocess_BiasCorrection, download=True
    )
    loaders = split_data_loaders(
        data,
        loader_args={"batch_size": 100, "shuffle": True},
        valid_loader_args={"shuffle": False},
        test_loader_args={"shuffle": False},
    )

3. Instantiate your neural network, make sure the dimensions match up with your data.

.. code-block:: python

    network = MLPNetwork(25, 2, n_hidden_layers=3, hidden_dim=35)
    network = network.to(device)

4. Choose your loss function and optimizer.

.. code-block:: python

    criterion = MSELoss()
    optimizer = Adam(network.parameters(), lr=0.001)

5. Then, setup the descriptor, that will attach names to specific parts of your network.

.. code-block:: python

    descriptor = Descriptor()
    descriptor.add("output", 0, "Tmax")
    descriptor.add("output", 1, "Tmin")

6. Define your constraints on the network.

.. code-block:: python

    Constraint.descriptor = descriptor
    Constraint.device = device
    constraints = [
        ScalarConstraint("Tmin", ge, 0),
        ScalarConstraint("Tmin", le, 1),
        ScalarConstraint("Tmax", ge, 0),
        ScalarConstraint("Tmax", le, 1),
        BinaryConstraint("Tmax", gt, "Tmin"),
    ]

7. Instantiate metric manager and core, and start the training.

.. code-block:: python

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

For more examples, refer to the GitHub repository's `example folder <https://github.com/ML-KULeuven/congrads/tree/main/examples>`_ or the `notebooks folder <https://github.com/ML-KULeuven/congrads/tree/main/notebooks>`_ for more examples.
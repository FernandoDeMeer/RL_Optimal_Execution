# RL Optimal Trade Execution
This is the code repository of "A Modular Framework for RL Optimal Execution".

## Installation

We recommend running the experiments via the `Dockerfile`.

Alternatively, one can manually install all the necessary libraries in a virtual environment via

``pip install -r requirements.txt``

## Entrypoints

The `train_{algo}.py` files  are the entrypoints of the experiments. Training and evaluations can be carried out modifying their `main()` functions. Adding new agents and training on different periods (if the data is provided) can be done via modifying the `config` dict.

## Framework Modules
`src/core/data/historical:datafeed.py` contains the implementation of the `DataFeed` class. 

`src/core/environment/limit_orders_setup` contains the implementations of the `Execution Algo`, `Broker` classes as well as the `gym` environment. 

## Miscellaneous 

`src/tests` contains the implementation of the multiple UnitTests applied to the environment.

The `plot_schedule` method of the `Execution Algo` class can reproduce Figures 5 & 6 of the paper. 

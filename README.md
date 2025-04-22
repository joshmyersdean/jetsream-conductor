# Jetstream Conductor
Helper tools to orchestrate pytorch multi-node training in the Jetsream2 ecosystem

## A Note on Development
This repo was developed for the IVC group at CU Boulder to allow for orchestrating multi-node GPU trainings in the Jetsream2 ecosystem. However, I hope it will be useful for all!


## Tools

### Orchestrator
We provide a script, `orchtesrate.sh` that launches a distributed PyTorch training job on multiple nodes, as defined by `config.yaml`. You need to have `yq` installed (`pip install yq`). You will need the private and public IP addresses for the instances you wish to use for training.

To stress test, we provide a `ddp_test` directory that you can use by default to make sure your system is working. To utilize this example, simply run

```bash
bash orchestrate.sh
```

and it will use the default config file. 

By default, orchestrate looks for a `.venv` file in your project directory, and if one is not found then we create a [uv](https://github.com/astral-sh/uv) environment and install PyTorch and Torchvision with CUDA 12.1 drivers. We recommend using `uv` as your base environment, but future iterations will allow for more flexibility (e.g., using Conda).


## TODOs
- [x] Basic Orchestrator script 
- [ ] Add in flexible logic for different package managers (e.g., conda)
- [ ] Script to launch instances from CLI
- [ ] Script to generate config files dynamically based on running instances
# AISTATS 2024
### Near-Optimal Pure Exploration in Matrix Games: A Generalization of Stochastic Bandits & Dueling Bandits
##### Instructions
1. `conda env create -f env.yml`
2. `conda activate midsearch-3.11`
3. `cd src`
4. `./run_experiments.py` and/or `./make_plots.py`

##### Notes
1. Change `n_workers` in `psne/__init__.py` to make some algorithms parallelize across the CPU. The default is 8 (4 cores x 2 threads); you may need to decrease if running on e.g. a dual core CPU.
2. We recommend first adding a `n_trials=10` kwarg in `run_experiments.py` to make sure the script is running correctly (i.e. not running with the default `n_trials=300` used for the plots in the paper).   
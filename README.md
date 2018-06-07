# SeCTAr
Code for [Self-Consistent Trajectory Autoencoder: Hierarchical Reinforcement Learning with Trajectory Embeddings](https://sites.google.com/view/sectar/)

## Running
Experiment scripts are in `exps/`, environments are located under `traj2vec/envs`, main algorithms are in `traj2vec/algos/vaepdentropy.py` and `traj2vec/algos/vae_bc.py`

## Installation
Create the conda env traj2vecv3 with
```
conda env create -f environment.yml
```

This repo also requires rllab so clone the [repo](https://github.com/rll/rllab) and point your python path to it.


## Logging
The log dir for the scripts is set to `data`. You can plot recorded results by giving the exp log dir to ```traj2vec/viskit/frontend.py```.
# SeCTAr
Code for [Self-Consistent Trajectory Autoencoder: Hierarchical Reinforcement Learning with Trajectory Embeddings](https://sites.google.com/view/sectar/)

## Running
Experiment scripts are in `exps/`, environments are located under `traj2vec/envs`, main algorithms are in `traj2vec/algos/vaepdentropy.py` and `traj2vec/algos/vae_bc.py`

## Installation
*Download
** [sectar](https://github.com/wyndwarrior/Sectar)
** [rllab](https://github.com/rll/rllab)
** [doodad](https://github.com/justinjfu/doodad)
** [baselines](https://github.com/openai/baselines)
*Add these repos to your python path and follow instructions for setting them up.
*Install Mujoco [instructions](https://github.com/openai/mujoco-py)
*Modify `traj2vec/launchers/config.py` to point to the appropiate paths.

*Create the conda env sectar with
```
conda env create -f environment.yml
```

## Logging
The log dir for the scripts is set to `data`. You can plot recorded results by giving the exp log dir to ```traj2vec/viskit/frontend.py```.

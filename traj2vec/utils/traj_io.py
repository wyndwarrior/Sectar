import os

import numpy as np


def load_traj_file(file, skip_first=-1):
    # Assumes each trajectory is on a single line.
    paths = []
    with open(file, 'r') as f:
        i = 0
        for line in f:
            i += 1
            if i < skip_first:
                continue
            path = np.fromstring(line.rstrip(), sep=" ").astype(np.float32)
            paths.append(path)

    return paths

def load_trajs(traj_dir, file_names, env_dim, keep_last=200, upto=-1,
               keep_env_dim=-1):

    traj_files = [os.path.join(traj_dir, f_name, 'traj.txt') for f_name in file_names]

    combined_traj = [load_traj_file(f)[-keep_last:upto] for f in traj_files]
    min_path_length = min([path.size for trajs in combined_traj for path in trajs])
    combined_traj = [np.reshape(np.concatenate([path[:min_path_length] for path in paths]), (-1, env_dim),) for paths in combined_traj]
    combined_traj = [trajs[:, :keep_env_dim] for trajs in combined_traj]
    return min_path_length // env_dim, combined_traj
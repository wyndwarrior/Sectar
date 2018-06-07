import os
import traj2vec.utils.logger as logger
import numpy as np



def record_tabular(stats, csv_file):
    file = os.path.join(logger.get_snapshot_dir(), csv_file)
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write(','.join(stats.keys()) + '\n')
    with open(file, 'ab') as f:
        np.savetxt(f, np.expand_dims(np.array([x for x in stats.values()]), 0), delimiter=',')
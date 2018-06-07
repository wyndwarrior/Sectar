
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from traj2vec.utils.traj_io import load_trajs


def plot_trajs(trajs, seq_len, pred=True):
    labels = ['pred', 'target', 'clone']
    markers = ['+', '^', 'o']
    num_trajs = trajs[0].shape[0] // seq_len
    assert trajs[0].shape[0] == trajs[1].shape[0]
    assert trajs[1].shape[0] == trajs[2].shape[0]
    colors = cm.rainbow(np.linspace(0, 1, num_trajs))
    for j in range(num_trajs):
        for i, (traj_set, label, marker) in enumerate(zip(trajs, labels, markers)):
            if not pred and i == 0:
                continue
            traj = traj_set[j*seq_len:(j+1)*seq_len, ...]
            plt.scatter(traj[:, 0], traj[:, 1], label=label, marker=marker, color=colors[j])

    if not pred:
        labels = labels[1:]
    plt.legend(labels)

    plt.savefig('traj1.png', bbox_inches='tight')

if __name__ == '__main__':
    traj_dir = sys.argv[1]
    seq_len, trajs = load_trajs(traj_dir,
                                ['/traj.txt'],
                                4, keep_last=30, upto=-20)
    plot_trajs(trajs, seq_len, pred=False)
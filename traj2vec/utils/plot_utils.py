import errno
import os

import matplotlib.pyplot as plt
import numpy as np
import traj2vec.utils.logger as logger
from matplotlib.pyplot import cm

_snapshot_dir = None

IMG_DIR = 'img'
VIDEO_DIR = 'video'
_snapshot_dir = None

def get_time_stamp():
    import datetime
    import dateutil.tz
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S_%f')
    return timestamp

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def set_snapshot_dir(dirname):
    global _snapshot_dir
    _snapshot_dir = dirname


def get_snapshot_dir():
    return logger.get_snapshot_dir() or _snapshot_dir or None


def logger_active():
    return get_snapshot_dir() is not None


def get_img_dir(dir_name):
    if not logger_active():
        raise NotImplementedError()
    dirname = os.path.join(get_snapshot_dir(), dir_name)
    mkdir_p(dirname)
    return dirname

def record_fig(name, dir_name, itr=None):
    if not logger_active():
        return
    if itr is not None:
        name = ('itr%d_' % itr) + name
    filename = os.path.join(get_img_dir(dir_name), name)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_traj(traj, name, marker=None, color=None, fillstyle=None):
    return plt.plot(traj[:, 0], traj[:, 1], label=name,
             marker=marker, color=color, fillstyle=fillstyle)

def preprocess_trajs(trajs, keep, env_name):
    if 'Reacher' in env_name:
        states = trajs[:keep, :, 8:10]
        states += trajs[:keep, :, 4:6]
    else:
        states = trajs[:keep, :, :]
    return states

def plot_traj_sets(traj_sets, traj_names, itr, env_id='TwoDMaze',
                   dir_name='trajs', figname='SD_PD', fillstyle='full'):
    if env_id == 'TwoDMaze':
        axes_lim = ([-.25, .25], [-.25, .25])
    elif env_id =='Button-v0':
        axes_lim = ([0, 10], [0, 64])
    else:
        axes_lim = ([-3, 3], [-3, 3])

    fig = plt.figure()
    fig.add_subplot(1, 1, 1, axisbg='#AAAAAA')
    default_markers = ['o', '*', 'v']
    num_trajs = traj_sets[0].shape[0]
    colors = cm.rainbow(np.linspace(0, 1, num_trajs))
    # set alpha
    colors[:, -1] = 0.5
    handles = []
    for i, (traj_set, traj_name) in enumerate(zip(traj_sets, traj_names)):
        handles.append([])
        for j, traj in enumerate(traj_set):
            handle = plot_traj(traj, traj_name, marker=default_markers[i], color=colors[j],
                               fillstyle=fillstyle)
            handles[i].append(handle[0])
    axes = plt.gca()
    axes.set_xlim(axes_lim[0])
    axes.set_ylim(axes_lim[1])
    plt.legend(handles=[x[0] for x in handles])
    record_fig(figname, dir_name, itr)
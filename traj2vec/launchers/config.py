CODE_DIRS_TO_MOUNT = [
    '/home/jcoreyes/embed/rllab',
    '/home/jcoreyes/embed/traj2vecv3',
    '/home/jcoreyes/embed/baselines'
]
DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/home/jcoreyes/.mujoco/',
        mount_point='/root/.mujoco',
    ),
]
LOCAL_LOG_DIR = '/home/jcoreyes/embed/traj2vecv3/data/local/'
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/home/jcoreyes/embed/traj2vecv3/scripts/run_experiment_from_doodad.py'
)
DOODAD_DOCKER_IMAGE = 'jcoreyes/traj2vecv3:latest'

# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'
# OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/dir/from/railrl-config/'

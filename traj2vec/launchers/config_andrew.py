CODE_DIRS_TO_MOUNT = [
    '/home/andrewliu/research/rllab',
    '/home/andrewliu/research/lang/traj2vecv3_jd',
    '/home/andrewliu/research/lang/baselines2'
]
DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/home/andrewliu/.mujoco/',
        mount_point='/root/.mujoco',
    ),
]
LOCAL_LOG_DIR = '/home/andrewliu/research/lang/traj2vecv3_jd/data/local/'
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/home/andrewliu/research/lang/traj2vecv3_jd/scripts/run_experiment_from_doodad.py'
)
DOODAD_DOCKER_IMAGE = 'wynd07/traj2vecv3-pytorch-0.4.0:latest'
# DOODAD_DOCKER_IMAGE = 'jcoreyes/traj2vecv3:latest'


# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'
# OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/dir/from/railrl-config/'

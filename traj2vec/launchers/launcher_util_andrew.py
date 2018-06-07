import base64
import json
import os
import os.path as osp
import pickle
import random
import sys
import time
import uuid

import __main__ as main
import cloudpickle
# import datetime
import dateutil.tz
# import git
import joblib
import numpy as np
#import tensorflow as tf
from datetime import datetime
import traj2vec.pythonplusplus as ppp
from traj2vec.launchers import config_andrew as config
from traj2vec.utils.torch_utils import set_gpu_mode
import traj2vec.utils.logger as logger
from rllab.misc.instrument import run_experiment_lite, query_yes_no

ec2_okayed = False
gpu_ec2_okayed = False


def run_experiment(
        method_call,
        mode='local',
        exp_prefix='default',
        seed=None,
        variant=None,
        exp_id=0,
        unique_id=None,
        prepend_date_to_exp_prefix=True,
        use_gpu=False,
        snapshot_mode='last',
        snapshot_gap=1,
        n_parallel=0,
        base_log_dir=None,
        instance_type='p2.xlarge',
        sync_interval=180,
        docker_image=config.DOODAD_DOCKER_IMAGE,
        local_input_dir_to_mount_point_dict=None,  # TODO(vitchyr): test this
):
    """
    Usage:

    ```
    def foo(variant):
        x = variant['x']
        y = variant['y']
        logger.log("sum", x+y)

    variant = {
        'x': 4,
        'y': 3,
    }
    run_experiment(foo, variant, exp_prefix="my-experiment")
    ```

    Results are saved to
    `base_log_dir/<date>-my-experiment/<date>-my-experiment-<unique-id>`

    By default, the base_log_dir is determined by
    `config.LOCAL_LOG_DIR/`

    :param method_call: a function that takes in a dictionary as argument
    :param mode: 'local', 'local_docker', or 'ec2'
    :param exp_prefix: name of experiment
    :param seed: Seed for this specific trial.
    :param variant: Dictionary
    :param exp_id: One experiment = one variant setting + multiple seeds
    :param unique_id: If not set, the unique id is generated.
    :param prepend_date_to_exp_prefix: If False, do not prepend the date to
    the experiment directory.
    :param use_gpu:
    :param snapshot_mode: See rllab.logger
    :param snapshot_gap: See rllab.logger
    :param n_parallel:
    :param base_log_dir: Will over
    :param sync_interval: How often to sync s3 data (in seconds).
    :param local_input_dir_to_mount_point_dict: Dictionary for doodad.
    :return:
    """
    try:
        import doodad
        import doodad.mode
        import doodad.mount as mount
        from doodad.utils import REPO_DIR
    except ImportError:
        return run_experiment_old(
            method_call,
            exp_prefix=exp_prefix,
            seed=seed,
            variant=variant,
            time_it=True,
            mode=mode,
            exp_id=exp_id,
            unique_id=unique_id,
            prepend_date_to_exp_prefix=prepend_date_to_exp_prefix,
            use_gpu=use_gpu,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            n_parallel=n_parallel,
            base_log_dir=base_log_dir,
            periodic_sync_interval=sync_interval,
        )
    global ec2_okayed
    global gpu_ec2_okayed
    if local_input_dir_to_mount_point_dict is None:
        local_input_dir_to_mount_point_dict = {}
    else:
        raise NotImplementedError("TODO(vitchyr): Implement this")
    # Modify some of the inputs
    if seed is None:
        seed = random.randint(0, 100000)
    if variant is None:
        variant = {}
    for key, value in ppp.recursive_items(variant):
        # This check isn't really necessary, but it's to prevent myself from
        # forgetting to pass a variant through dot_map_dict_to_nested_dict.
        if "." in key:
            raise Exception(
                "Variants should not have periods in keys. Did you mean to "
                "convert {} into a nested dictionary?".format(key)
            )
    if unique_id is None:
        unique_id = str(uuid.uuid4())
    if prepend_date_to_exp_prefix:
        exp_prefix = time.strftime("%m-%d") + "-" + exp_prefix
    variant['seed'] = str(seed)
    variant['exp_id'] = str(exp_id)
    variant['unique_id'] = str(unique_id)
    # logger.log("Variant:")
    # logger.log(json.dumps(ppp.dict_to_safe_json(variant), indent=2))

    mode_str_to_doodad_mode = {
        'local': doodad.mode.Local(),
        'local_docker': doodad.mode.LocalDocker(
            image=docker_image,
            gpu=use_gpu,
        ),
        'ec2': doodad.mode.EC2AutoconfigDocker(
            image=docker_image,
            region='us-east-1',
            instance_type=instance_type,
            # instance_type='c4.large',
            spot_price=0.5,
            s3_log_prefix=exp_prefix,
            s3_log_name="{}-{}-id{}-s{}".format(exp_prefix, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), exp_id, seed),
            gpu=use_gpu,
            network_interfaces=[dict(
        SubnetId='subnet-954d40be',
        Groups=['sg-26d7b854'],
        DeviceIndex=0,
        AssociatePublicIpAddress=True,
    )],
        ),
    }

    if base_log_dir is None:
        base_log_dir = config.LOCAL_LOG_DIR
    output_mount_point = config.OUTPUT_DIR_FOR_DOODAD_TARGET
    mounts = [
        mount.MountLocal(local_dir=REPO_DIR, pythonpath=True),
    ]
    for code_dir in config.CODE_DIRS_TO_MOUNT:
        mounts.append(mount.MountLocal(local_dir=code_dir, pythonpath=True))
    for dir, mount_point in local_input_dir_to_mount_point_dict.items():
        mounts.append(mount.MountLocal(
            local_dir=dir,
            mount_point=mount_point,
            pythonpath=False,
        ))

    if mode != 'local':
        for non_code_mapping in config.DIR_AND_MOUNT_POINT_MAPPINGS:
            mounts.append(mount.MountLocal(**non_code_mapping))

    if mode == 'ec2':
        if not ec2_okayed and not query_yes_no(
                "EC2 costs money. Are you sure you want to run?"
        ):
            sys.exit(1)
        if not gpu_ec2_okayed and use_gpu:
            if not query_yes_no(
                    "EC2 is more expensive with GPUs. Confirm?"
            ):
                sys.exit(1)
            gpu_ec2_okayed = True
        ec2_okayed = True
        output_mount = mount.MountS3(
            s3_path='',
            mount_point=output_mount_point,
            output=True,
            sync_interval=sync_interval,
        )
        # This will be over-written by the snapshot dir, but I'm setting it for
        # good measure.
        base_log_dir_for_script = output_mount_point
        # The snapshot dir needs to be specified for S3 because S3 will
        # automatically create the experiment director and sub-directory.
        snapshot_dir_for_script = output_mount_point
    elif mode == 'local':
        output_mount = mount.MountLocal(
            local_dir=base_log_dir,
            mount_point=None,  # For purely local mode, skip mounting.
            output=True,
        )
        base_log_dir_for_script = base_log_dir
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
    else:
        output_mount = mount.MountLocal(
            local_dir=base_log_dir,
            mount_point=output_mount_point,
            output=True,
        )
        base_log_dir_for_script = output_mount_point
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
    mounts.append(output_mount)

    # repo = git.Repo(os.getcwd())
    # code_diff = repo.git.diff(None)
    # if len(code_diff) > 5000:
    #     logger.log("Git diff %d greater than 5000. Not saving diff." % len(code_diff))
    code_diff = None
    run_experiment_kwargs = dict(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        use_gpu=use_gpu,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        code_diff=code_diff,
        # commit_hash=repo.head.commit.hexsha,
        script_name=main.__file__,
        n_parallel=n_parallel,
        base_log_dir=base_log_dir_for_script,
    )
    doodad.launch_python(
        target=config.RUN_DOODAD_EXPERIMENT_SCRIPT_PATH,
        mode=mode_str_to_doodad_mode[mode],
        mount_points=mounts,
        args={
            'method_call': method_call,
            'output_dir': snapshot_dir_for_script,
            'run_experiment_kwargs': run_experiment_kwargs,
        },
        use_cloudpickle=True,
        fake_display=True if mode != 'local' else False,
        verbose=True,
    )


def run_experiment_old(
        task,
        exp_prefix='default',
        seed=None,
        variant=None,
        time_it=True,
        save_profile=False,
        profile_file='time_log.prof',
        mode='here',
        exp_id=0,
        unique_id=None,
        prepend_date_to_exp_prefix=True,
        use_gpu=False,
        snapshot_mode='last',
        snapshot_gap=1,
        n_parallel=0,
        base_log_dir=None,
        **run_experiment_lite_kwargs
):
    """
    Run a task via the rllab interface, i.e. serialize it and then run it via
    the run_experiment_lite script.

    This will soon be deprecated.

    :param task:
    :param exp_prefix:
    :param seed:
    :param variant:
    :param time_it: Add a "time" command to the python command?
    :param save_profile: Create a cProfile log?
    :param profile_file: Where to save the cProfile log.
    :param mode: 'here' will run the code in line, without any serialization
    Other options include 'local', 'local_docker', and 'ec2'. See
    run_experiment_lite documentation to learn what those modes do.
    :param exp_id: Experiment ID. Should be unique across all
    experiments. Note that one experiment may correspond to multiple seeds.
    :param unique_id: Unique ID should be unique across all runs--even different
    seeds!
    :param prepend_date_to_exp_prefix: If True, prefix "month-day_" to
    exp_prefix
    :param run_experiment_lite_kwargs: kwargs to be passed to
    `run_experiment_lite`
    :return:
    """
    if seed is None:
        seed = random.randint(0, 100000)
    if variant is None:
        variant = {}
    if unique_id is None:
        unique_id = str(uuid.uuid4())
    if prepend_date_to_exp_prefix:
        exp_prefix = time.strftime("%m-%d") + "_" + exp_prefix
    variant['seed'] = str(seed)
    variant['exp_id'] = str(exp_id)
    variant['unique_id'] = str(unique_id)
    logger.log("Variant:")
    logger.log(json.dumps(ppp.dict_to_safe_json(variant), indent=2))
    command_words = []
    if time_it:
        command_words.append('time')
    command_words.append('python')
    if save_profile:
        command_words += ['-m cProfile -o', profile_file]
    repo = git.Repo(os.getcwd())
    diff_string = repo.git.diff(None)
    commit_hash = repo.head.commit.hexsha
    script_name = "tmp"
    if mode == 'here':
        log_dir, exp_name = create_log_dir(exp_prefix, exp_id, seed,
                                           base_log_dir)
        data = dict(
            log_dir=log_dir,
            exp_name=exp_name,
            mode=mode,
            variant=variant,
            exp_id=exp_id,
            exp_prefix=exp_prefix,
            seed=seed,
            use_gpu=use_gpu,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            diff_string=diff_string,
            commit_hash=commit_hash,
            n_parallel=n_parallel,
            base_log_dir=base_log_dir,
            script_name=script_name,
        )
        save_experiment_data(data, log_dir)
    if mode == 'here':
        run_experiment_here(
            task,
            exp_prefix=exp_prefix,
            variant=variant,
            exp_id=exp_id,
            seed=seed,
            use_gpu=use_gpu,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            code_diff=diff_string,
            commit_hash=commit_hash,
            script_name=script_name,
            n_parallel=n_parallel,
            base_log_dir=base_log_dir,
        )
    else:
        if mode == "ec2" and use_gpu:
            if not query_yes_no(
                    "EC2 is more expensive with GPUs. Confirm?"
            ):
                sys.exit(1)
        code_diff = (
            base64.b64encode(cloudpickle.dumps(diff_string)).decode("utf-8")
        )
        run_experiment_lite(
            task,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            exp_prefix=exp_prefix,
            variant=variant,
            seed=seed,
            use_cloudpickle=True,
            python_command=' '.join(command_words),
            mode=mode,
            use_gpu=use_gpu,
            script="railrl/scripts/run_experiment_lite.py",
            code_diff=code_diff,
            commit_hash=commit_hash,
            script_name=script_name,
            n_parallel=n_parallel,
            **run_experiment_lite_kwargs
        )


def save_experiment_data(dictionary, log_dir):
    with open(log_dir + '/experiment.pkl', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def resume_torch_algorithm(variant):
    from railrl.torch import pytorch_util as ptu
    load_file = variant.get('params_file', None)
    if load_file is not None and osp.exists(load_file):
        data = joblib.load(load_file)
        algorithm = data['algorithm']
        epoch = data['epoch']+1
        use_gpu = variant['use_gpu']
        if use_gpu and ptu.gpu_enabled():
            algorithm.cuda()
        algorithm.train(start_epoch=epoch + 1)


def continue_experiment(load_experiment_dir, resume_function):
    path = os.path.join(load_experiment_dir, 'experiment.pkl')
    if osp.exists(path):
        data = joblib.load(path)
        mode = data['mode']
        exp_prefix = data['exp_prefix']
        variant = data['variant']
        variant['params_file'] = load_experiment_dir + '/extra_data.pkl' # load from snapshot directory
        exp_id = data['exp_id']
        seed = data['seed']
        use_gpu = data['use_gpu']
        snapshot_mode = data['snapshot_mode']
        snapshot_gap = data['snapshot_gap']
        diff_string = data['diff_string']
        commit_hash = data['commit_hash']
        n_parallel = data['n_parallel']
        base_log_dir = data['base_log_dir']
        log_dir = data['log_dir']
        exp_name = data['exp_name']
        if mode == 'here':
            run_experiment_here(
                resume_function,
                variant=variant,
                exp_prefix=exp_prefix,
                exp_id=exp_id,
                seed=seed,
                use_gpu=use_gpu,
                snapshot_mode=snapshot_mode,
                snapshot_gap=snapshot_gap,
                code_diff=diff_string,
                commit_hash=commit_hash,
                n_parallel=n_parallel,
                base_log_dir=base_log_dir,
                log_dir=log_dir,
                exp_name=exp_name,
            )
    else:
        raise Exception('invalid experiment_file')


def run_experiment_here(
        experiment_function,
        exp_prefix="default",
        variant=None,
        exp_id=0,
        seed=0,
        use_gpu=True,
        snapshot_mode='last',
        snapshot_gap=1,
        code_diff=None,
        commit_hash=None,
        script_name=None,
        n_parallel=0,
        base_log_dir=None,
        log_dir=None,
        exp_name=None,
):
    """
    Run an experiment locally without any serialization.

    :param experiment_function: Function. `variant` will be passed in as its
    only argument.
    :param exp_prefix: Experiment prefix for the save file.
    :param variant: Dictionary passed in to `experiment_function`.
    :param exp_id: Experiment ID. Should be unique across all
    experiments. Note that one experiment may correspond to multiple seeds,.
    :param seed: Seed used for this experiment.
    :param use_gpu: Run with GPU. By default False.
    :param script_name: Name of the running script
    :param log_dir: If set, set the log directory to this. Otherwise,
    the directory will be auto-generated based on the exp_prefix.
    :return:
    """
    if variant is None:
        variant = {}
    if seed is None and 'seed' not in variant:
        seed = random.randint(0, 100000)
        variant['seed'] = str(seed)
    if n_parallel > 0:
        from rllab.sampler import parallel_sampler
        parallel_sampler.initialize(n_parallel=n_parallel)
        parallel_sampler.set_seed(seed)
    variant['exp_id'] = str(exp_id)
    reset_execution_environment()
    set_seed(seed)
    setup_logger(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        base_log_dir=base_log_dir,
        log_dir=log_dir,
        exp_name=exp_name,
    )
    log_dir = logger.get_snapshot_dir()
    if code_diff is not None:
        with open(osp.join(log_dir, "code.diff"), "w") as f:
            f.write(code_diff)
    if commit_hash is not None:
        with open(osp.join(log_dir, "commit_hash.txt"), "w") as f:
            f.write(commit_hash)
    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    set_gpu_mode(use_gpu)
    return experiment_function(variant)


def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)


def create_log_dir(exp_prefix, exp_id=0, seed=0, base_log_dir=None):
    """
    Creates and returns a unique log directory.

    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: Different exp_ids will be in different directories.
    :return:
    """
    exp_name = create_exp_name(exp_prefix, exp_id=exp_id,
                               seed=seed)
    if base_log_dir is None:
        base_log_dir = config.LOCAL_LOG_DIR
    log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, exp_name


def setup_logger(
        exp_prefix="default",
        exp_id=0,
        seed=0,
        variant=None,
        base_log_dir=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        exp_name=None,
):
    """
    Set up logger to have some reasonable default settings.

    Will save log output to

    based_log_dir/exp_prefix/exp_name.

    :param exp_prefix: The sub-directory for this specific experiment.
    :param exp_id: The number of the specific experiment run within this
    experiment.
    :param variant:
    :param base_log_dir: The directory where all log should be saved.
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :return:
    """
    first_time = log_dir is None and exp_name is None
    if first_time:
        log_dir, exp_name = create_log_dir(exp_prefix, exp_id=exp_id, seed=seed,
                                           base_log_dir=base_log_dir)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    if variant is not None:
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    logger.add_text_output(text_log_path)
    if first_time:
        logger.add_tabular_output(tabular_log_path)
    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs,
                           logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    logger.push_prefix("[%s] " % exp_name)


def set_seed(seed):
    """
    Set the seed for all the possible random number generators.

    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    #tf.set_random_seed(seed)


def reset_execution_environment():
    """
    Call this between calls to separate experiments.
    :return:
    """
    #tf.reset_default_graph()
    import importlib
    importlib.reload(logger)


def create_run_experiment_multiple_seeds(n_seeds, experiment, **kwargs):
    """
    Run a function multiple times over different seeds and return the average
    score.
    :param n_seeds: Number of times to run an experiment.
    :param experiment: A function that returns a score.
    :param kwargs: keyword arguements to pass to experiment.
    :return: Average score across `n_seeds`.
    """

    def run_experiment_with_multiple_seeds(variant):
        seed = int(variant['seed'])
        scores = []
        for i in range(n_seeds):
            variant['seed'] = str(seed + i)
            scores.append(run_experiment(
                experiment,
                variant=variant,
                exp_id=i,
                mode='here',
                **kwargs
            ))
        return np.mean(scores)

    return run_experiment_with_multiple_seeds

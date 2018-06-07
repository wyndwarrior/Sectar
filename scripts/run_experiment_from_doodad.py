import doodad as dd
from traj2vec.launchers.launcher_util_andrew import run_experiment_here

args_dict = dd.get_args()
method_call = args_dict['method_call']
run_experiment_kwargs = args_dict['run_experiment_kwargs']
output_dir = args_dict['output_dir']
print("START from run_experiment_from_doodad:")
print("output_dir", output_dir)
print("END from run_experiment_from_doodad:")
run_experiment_here(
    method_call,
    log_dir=output_dir,
    **run_experiment_kwargs
)

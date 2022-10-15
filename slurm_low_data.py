import os
from datetime import datetime
import argparse
import time
import socket
import itertools
import shutil
import pandas as pd
import glob

qos_dict = {
    "sailon": {"nhrs": 72, "cores": 16, "mem": 128},
    "scav": {"nhrs": 72, "cores": 16, "mem": 128},
    "high": {"gpu": 4, "cores": 16, "mem": 128, "nhrs": 36},
    "medium": {"gpu": 2, "cores": 8, "mem": 64, "nhrs": 72},
    "default": {"gpu": 1, "cores": 4, "mem": 32, "nhrs": 168},
}


def check_qos(args):

    for key, max_value in qos_dict[args.qos].items():
        val_from_args = getattr(args, key)
        if val_from_args != None:
            if val_from_args > max_value:
                raise ValueError("Invalid paramter for {} for {}".format(key, args.qos))
        else:
            setattr(args, key, max_value)
    return args


# TODO: Add day funtionality too
parser = argparse.ArgumentParser()
parser.add_argument("--nhrs", type=int, default=None)
parser.add_argument("--base-dir", default=f"{os.getcwd()}")
parser.add_argument("--output-dirname", default="output")
parser.add_argument("--dryrun", action="store_true")
parser.add_argument("--qos", default="sailon", type=str, help="Qos to run")
parser.add_argument(
    "--env", default="motif", type=str, help="Set the name of the dir you want to dump"
)
parser.add_argument("--gpu", default=1, type=int, help="Number of gpus")
parser.add_argument("--cores", default=4, type=int, help="Number of cpu cores")
parser.add_argument("--mem", default=32, type=int, help="RAM in G")
parser.add_argument("--log-dir", default="./logs")


# parser.add_argument('--path', default='/fs/vulcan-projects/actionbytes/vis/ab_training_run3_rerun_32_0.0001_4334_new_dl_nocasl_checkpoint_best_dmap_ab_info.hkl')
# parser.add_argument('--num_ab', default= 100000, type=int, help='number of actionbytes')

args = parser.parse_args()


args = parser.parse_args()
args.env += str(int(time.time()))


output_dir = os.path.join(args.base_dir, args.output_dirname, args.env)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Output Directory: %s" % output_dir)

weights_dir = (
    "/fs/vulcan-projects/sailon_root/Sonaal/fsl/CV_LTH_Pre-training/imgnet_152"
)

datasets = ["cifar10", "cifar100", "svhn"]
few_shot_ratios = [0.8, 0.6, 0.4, 0.2]
sample_numbers = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
params = sample_numbers
weights_dir = glob.glob(weights_dir + "/*model*")

with open(f"{args.base_dir}/output/{args.env}/now.txt", "w") as nowfile, open(
    f"{args.base_dir}/output/{args.env}/log.txt", "w"
) as output_namefile, open(
    f"{args.base_dir}/output/{args.env}/err.txt", "w"
) as error_namefile, open(
    f"{args.base_dir}/output/{args.env}/name.txt", "w"
) as namefile:

    count = 0
    for dataset in datasets:
        for j, weights in enumerate(weights_dir):
            num = weights.split("/")[-1].split("model")[0]
            for i, (n_samples) in enumerate(params):
                now = datetime.now()
                datetimestr = now.strftime("%m%d_%H%M:%S.%f")
                name = f"{dataset}_{num}_{(n_samples)}"
                cmd = f"python -u main_eval_downstream.py --dataset {dataset} --arch resnet152 --save_dir weights/{dataset}_{num}  --dict_key state_dict  --mask_dir {weights} --save_model --number_of_samples {(n_samples)}"

                print(count, name)
                count += 1

                nowfile.write(f"{cmd}\n")
                namefile.write(f"{(os.path.join(output_dir, name))}.log\n")
                output_namefile.write(f"{(os.path.join(output_dir, name))}_log.txt\n")
                error_namefile.write(f"{(os.path.join(output_dir, name))}_error.txt\n")
###########################################################################
# Make a {name}.slurm file in the {output_dir} which defines this job.
# slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
start = 1
slurm_script_path = os.path.join(output_dir, f"bbn.slurm")
slurm_command = "sbatch %s" % slurm_script_path

# Make the .slurm file
with open(slurm_script_path, "w") as slurmfile:
    slurmfile.write("#!/bin/bash\n")
    slurmfile.write(
        f"#SBATCH --array=1-{len(params) * len(weights_dir) * len(datasets) }\n"
    )
    slurmfile.write("#SBATCH --output=/dev/null\n")
    slurmfile.write("#SBATCH --error=/dev/null\n")
    slurmfile.write("#SBATCH --requeue\n")
    slurmfile.write("#SBATCH --exclude=vulcan30,vulcan31,vulcan32,vulcan24\n")
    # slurmfile.write("#SBATCH --exclude=vulcan24\n")
    # slurmfile.write("#SBATCH --exclude=vulcan[00-23]\n")

    args = check_qos(args)

    if args.qos == "scav":
        # slurmfile.write("#SBATCH --account=abhinav\n")
        slurmfile.write("#SBATCH --partition scavenger\n")
        slurmfile.write("#SBATCH --qos scavenger\n")
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)
        slurmfile.write("#SBATCH --qos=scavenger\n")

        if not args.gpu is None:
            slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
        else:
            raise ValueError("Specify the gpus for scavenger")
    else:
        slurmfile.write("#SBATCH --account=abhinav\n")
        slurmfile.write("#SBATCH --qos=%s\n" % args.qos)
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)
        # slurmfile.write("#SBATCH --exclude=vulcan[30-32]\n")
        # slurmfile.write("#SBATCH --exclude=vulcan24\n")

    slurmfile.write("\n")
    slurmfile.write("cd " + args.base_dir + "\n")
    slurmfile.write('eval "$(conda shell.bash hook)"' "\n")
    slurmfile.write(
        "conda activate /fs/vulcan-projects/sailon_root/miniconda3/envs/OSU\n"
    )
    # slurmfile.write("module load cuda/11.3.1")
    slurmfile.write("export MKL_SERVICE_FORCE_INTEL=1\n")
    slurmfile.write(
        f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/log.txt | tail -n 1) --error=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/err.txt | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/now.txt | tail -n 1)\n"
    )
    slurmfile.write("\n")
print(slurm_command)
print(
    "Running on {}, with {} gpus, {} cores, {} mem for {} hour".format(
        args.qos, args.gpu, args.cores, args.mem, args.nhrs
    )
)
import torch

print(torch.__version__, torch.cuda.get_arch_list())
if not args.dryrun:
    os.system("%s &" % slurm_command)

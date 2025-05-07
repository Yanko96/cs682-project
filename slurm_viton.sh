#!/bin/bash -l

#Submit this script with: sbatch thefilename

#SBATCH --job-name VITON   ##name that will show up in the queue
#SBATCH --output SUBSTITE_WITH_YOUR_OUTPUT_FILE_PATH   ##filename of the output; the %j will append the jobID to the end of the name making the output files unique despite the sane job name; default is slurm-[jobID].out
#SBATCH --error SUBSTITE_WITH_YOUR_OUTPUT_FILE_PATH   ##filename of the error output; the %j will append the jobID to the end of the name making the output files unique despite the sane job name; default is slurm-[jobID].out
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes 1  ##number of nodes to use
#SBATCH --ntasks 8  ##number of tasks (analyses) to run
#SBATCH --time 3-0:0:00  ##time for analysis (day-hour:min:sec)
#SBATCH --cpus-per-task 8  ##the number of threads the code will use
#SBATCH --partition contrib-gpuq,gpuq  ##the partition to run in [options: normal, gpu, debug]
#SBATCH --gres gpu:A100.80gb:4 ##GPU(s) allocated
#SBATCH --mail-user ykong7@gmu.edu  ##your email address
#SBATCH --mail-type BEGIN  ##slurm will email you when your job starts
#SBATCH --mail-type END  ##slurm will email you when your job ends
#SBATCH --mail-type FAIL  ##slurm will email you when your job fails

# nvidia-smi

# pip list

set -x
umask 0022 
nvidia-smi 
env|grep -i slurm
nproc --all
grep MemTotal /proc/meminfo

source /projects/RobotiXX/yangzhe/dress/bin/activate

module load gcc
module load cuda

cd /projects/RobotiXX/yangzhe/VITON/scripts
/projects/RobotiXX/yangzhe/dress/bin/python train.py --gpu_ids "0,1,2,3" --num_epochs 1000 --lr 1e-4
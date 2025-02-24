#!/bin/bash

# SLURM job options
#SBATCH --partition=gpu2
#SBATCH --job-name=alexnet
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=32000
#SBATCH --time=00:10:00
#SBATCH --output=%x.o%j.%N
#SBATCH --error=%x.e%j.%N

# Print job details
NOW=`date +%H:%M-%a-%d/%b/%Y`
echo '------------------------------------------------------'
echo 'This job is allocated on '$SLURM_JOB_CPUS_PER_NODE' cpu(s)'
echo 'Job is running on node(s): '
echo  $SLURM_JOB_NODELIST
echo '------------------------------------------------------'
#
# ==== End of Info part (say things) ===== #
#

cd $SLURM_SUBMIT_DIR            # here we go into the submission directory
export SLURM_NTASKS_PER_NODE=2  # need to export this, not for all clusters but Ulysses has a bug :/

module load cuda/12.1           # Loading cuda
conda activate devtools_scicomp # activate the environment

# Run the script
srun python scripts/run.py fit --config=experiments/config.yaml --strategy=ddp --devices=2
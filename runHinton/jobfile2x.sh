#!/bin/bash
#SBATCH --gres=shard:8      # Number of (V100) GPUs
#SBATCH --job-name=TrainVarNet
#SBATCH --mem=8G               # max memory per node
#SBATCH --cpus-per-task=10      # max CPU cores per MPI process
#SBATCH --time=04-00:00         # time limit (DD-HH:MM)
#SBATCH --nice=10             # allow other priority jobs to go first
#SBATCH --output=slurm_output_%A.out

# Initialize conda functions and activate your environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/dmvandenberg/.conda/envs/mridc/

# Actual job to be performed
python -m mridc.launch --config-path /home/dmvandenberg/PycharmProjects/mridc/projects/reconstruction/model_zoo/conf/ --config-name base_cirim_train_2x.yaml

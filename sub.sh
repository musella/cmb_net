#!/bin/bash -l
#SBATCH --job-name=jlr_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joosep.pata@cern.ch
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --mem=20G
#SBATCH -A d78

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

source $HOME/env.sh

cd $HOME/jlr
srun $@

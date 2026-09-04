#!/bin/bash
#SBATCH --account=c2016159
#SBATCH --constraint=MI250
#SBATCH --job-name=gcg_single_run
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --time=1:00:00

srun --ntasks-per-node=1 --cpus-per-task=8  --threads-per-core=1 -- python run_attacks.py \
	    model=meta-llama/Meta-Llama-3.1-8B-Instruct \
	    dataset=adv_behaviors \
	    datasets.adv_behaviors.idx=2 \
	    attack=gcg \

#!/bin/bash
#SBATCH --account=c2016159
#SBATCH --constraint=MI250
#SBATCH --job-name=hello_world
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=slurm_outputs/hello_world.out
#SBATCH --error=slurm_outputs/hello_world.out
#SBATCH --time=1:00:00

# # With    HyperThreading (SMT), 192 cores and 384 hardware threads.
# srun --ntasks-per-node=24 --cpus-per-task=16 --threads-per-core=2 -- ./hello_world
# Without HyperThreading (SMT), 192 cores and 192 hardware threads.
srun --ntasks-per-node=4 --cpus-per-task=16  --threads-per-core=1 -- python hello_world.py

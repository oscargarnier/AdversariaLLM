#!/bin/bash
#SBATCH --account=ogarnier
#SBATCH --job-name="hello_world"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:00:00

# # With    HyperThreading (SMT), 192 cores and 384 hardware threads.
# srun --ntasks-per-node=24 --cpus-per-task=16 --threads-per-core=2 -- ./hello_world
# Without HyperThreading (SMT), 192 cores and 192 hardware threads.
srun --ntasks-per-node=24 --cpus-per-task=8  --threads-per-core=1 -- python hello_world.py
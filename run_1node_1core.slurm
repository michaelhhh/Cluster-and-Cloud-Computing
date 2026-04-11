#!/bin/bash
#SBATCH --job-name=ccc_a1_1n1c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=output_1n1c.txt

module load GCC/11.3.0
module load OpenMPI/4.1.4
module load mpi4py/3.1.4

mpirun -np 1 python mpi_language_counter.py mastodon-large.ndjson bluesky-large.ndjson

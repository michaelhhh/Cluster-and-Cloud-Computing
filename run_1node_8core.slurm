#!/bin/bash
#SBATCH --job-name=ccc_a1_1n8c
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=output_1n8c.txt

module load GCC/11.3.0
module load OpenMPI/4.1.4
module load mpi4py/3.1.4

mpirun -np 8 python mpi_language_counter.py mastodon-large.ndjson bluesky-large.ndjson

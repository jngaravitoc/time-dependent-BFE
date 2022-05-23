#!/bin/bash 

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ngaravito@flatironinstitute.org
#SBATCH --time=100:00:00
#SBATCH --job-name MWLMC_bfe_100_150
#SBATCH -N1 --ntasks-per-node=64 
#SBATCH -e stderr.txt
#SBATCH -o stdout.txt


module purge
module load slurm
module load python
module load openmpi/4.0.7
module load gsl

echo
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo


pipeline=/mnt/home/nico/ceph/codes/bfe-py/bfe/pipeline/pipeline.py
#param=MWLMC5_MO3_beta0_50_100.yaml
param=MWLMC5_MO3_beta0_100_150.yaml
python3 $pipeline --param=$param --ncores=64

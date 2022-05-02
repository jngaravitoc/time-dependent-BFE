#!/bin/bash 

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ngaravito@flatironinstitute.org
#SBATCH --time=36:00:00
#SBATCH --job-name MWLMC_bfe_200_400_1e6
#SBATCH -N1 --ntasks-per-node=32 
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
param=MWLMC5_MO3_beta0.yaml
python3 $pipeline --param=$param --ncores=32 

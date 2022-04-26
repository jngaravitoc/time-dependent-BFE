# Don't forget to update the virtual env source /bfe_nbody/bin/activate


pipeline=/mnt/home/nico/codes/bfe-py/bfe/pipeline/pipeline.py
param=MW5_MO5_beta0.yaml
python3 $pipeline --param=$param --ncores=32 

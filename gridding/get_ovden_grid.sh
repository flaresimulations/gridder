#!/bin/bash -l
#SBATCH --ntasks 256
#SBATCH -N 4
#SBATCH --array=1-20%4
#SBATCH -J FLARES2-OVDEN-GRID-L5600N5040
#SBATCH -o logs/L5600N5040.%J.out
#SBATCH -e logs/L5600N5040.%J.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 5:00:00

module purge
#load the modules used to build your program.
module load python/3.9.1-C7 gnu_comp/11.1.0 openmpi/4.1.1 ucx/1.10.1

cd /cosma8/data/dp004/dc-rope1/FLARES-2/flares2-parent-weighting

source ../flares2-env/bin/activate

i=$(($SLURM_ARRAY_TASK_ID - 1))

#mpirun -np 256 python grid_parent_distributed.py $i L2800N5040 HYDRO
#mpirun -np 256 python grid_parent_distributed.py $i L2800N5040 DMO
mpirun -np 256 python grid_parent_distributed.py $i L5600N5040 DMO

echo "Job done, info follows..."
sstat --jobs=${SLURM_JOBID}.batch --format=JobID,MaxRSS,AveCPU,AvePages,AveRSS,AveVMSize
exit




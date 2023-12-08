#!/bin/bash -l
#SBATCH --ntasks 512
#SBATCH --array=1-20%10
#SBATCH --cpus-per-task=2
#SBATCH -J FLARES2-REGION-GRID-L2800N5040
#SBATCH -o logs/L2800N5040_regions.%J.out
#SBATCH -e logs/L2800N5040_regions.%J.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 24:00:00

module purge
#load the modules used to build your program.
module load python/3.9.1-C7 gnu_comp/11.1.0 openmpi/4.1.1 ucx/1.10.1

cd /cosma8/data/dp004/dc-rope1/FLARES-2/flares2-parent-weighting

source ../flares2-env/bin/activate

i=$(($SLURM_ARRAY_TASK_ID - 1))

#mpirun -np 1024 python smoothed_grid.py $i L2800N5040 HYDRO
#mpirun -np 1024 python smoothed_grid.py $i L2800N5040 DMO
mpirun -np 512 python smoothed_grid.py $i L5600N5040 DMO

echo "Job done, info follows..."
sstat --jobs=${SLURM_JOBID}.batch --format=JobID,MaxRSS,AveCPU,AvePages,AveRSS,AveVMSize
exit




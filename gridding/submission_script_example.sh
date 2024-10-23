#!/bin/bash
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH -J FLARES2-GRID-L5600N5040-DMO_FIDUCIAL
#SBATCH -o logs/L5600N5040_DMO_FIDUCIAL.%A.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH --array=0-24%4

module purge
module load python/3.10.7 gnu_comp/11.1.0 hdf5/1.12.0 openmpi/4.1.1

cd /cosma8/data/dp104/dc-love2/codes/zoom_region_selection/gridding/

source /cosma8/data/dp104/dc-love2/envs/regions/bin/activate

# Set your base directory
base_dir="/cosma8/data/dp004/flamingo/Runs/L5600N5040/DMO_FIDUCIAL/snapshots"
output_dir="/snap8/scratch/dp004/dc-love2/flamingo_grids/"

# Get the padded SLURM_ARRAY_TASK_ID
padded_task_id=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

# Construct input and output paths
input_file="${base_dir}/flamingo_${padded_task_id}/flamingo_${padded_task_id}.hdf5"
output_file="${output_dir}/grid_${padded_task_id}.hdf5"

# Check if the output directory exists, create it if not
if [ ! -d "$(dirname "$output_dir")" ]; then
    mkdir -p "$(dirname "$output_dir")"
fi

# Your mpirun command with variable filepaths
mpirun -np $SLURM_NTASKS python3 generate_grid.py \
    --input "$input_file" \
    --output "$output_file" \
    --nthreads=$SLURM_CPUS_PER_TASK \
    --kernel_diameters 4 8 16 32 \
    --delete_distributed=1 \
    --grid_width="4" \

echo "Job done, info follows..."
sstat --jobs=${SLURM_JOBID}.batch --format=JobID,MaxRSS,AveCPU,AvePages,AveRSS,AveVMSize
exit

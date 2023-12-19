#!/bin/bash -l
#SBATCH --ntasks 32
#SBATCH --cpus-per-task 64
#SBATCH --array=0-19%4
#SBATCH -J FLARES2-OVDEN-GRID-L5600N5040
#SBATCH -o logs/L5600N5040.%J.out
#SBATCH -e logs/L5600N5040.%J.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 5:00:00

module purge
module load python/3.10.12 gnu_comp/10.2.0 openmpi/4.1.1

cd /cosma8/data/dp004/dc-rope1/FLARES-2/zoom_region_selection

source ../flares2-env/bin/activate

# Set simulation parameters
simulation_name="L5600N5040"
simulation_type="DMO_FIDUCIAL"

# Set your base directory
base_dir="/cosma8/data/dp004/flamingo/Runs/${simulation_name}/${simulation_type}/snapshots"
output_dir="../gridded_data/FLAMINGO/${simulation_name}/${simulation_type}"

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
mpirun -np $SLURM_NTASKS python3 generate_regions.py \
    --input "$input_file" \
    --output "$output_file" \
    --nthreads=$SLURM_CPUS_PER_TASK \
    --kernel_diameters 2.5 5 15 30 \
    --delete_distributed=1 \
    --grid_width=1.0

echo "Job done, info follows..."
sstat --jobs=${SLURM_JOBID}.batch --format=JobID,MaxRSS,AveCPU,AvePages,AveRSS,AveVMSize
exit




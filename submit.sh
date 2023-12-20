#!/bin/bash

# Example usage
#   sbatch submit.sh -n 32 -c 8 -s L5600N5040 -t DMO_FIDUCIAL -w 1.0 -d "2.5 5 15 30" -b 100

# Parse named arguments
while getopts ":n:c:s:t:w:d:b:" opt; do
  case $opt in
    n) ntasks="$OPTARG";;
    c) cpus_per_task="$OPTARG";;
    s) simulation_name="$OPTARG";;
    t) simulation_type="$OPTARG";;
    w) width="$OPTARG";;
    d) diameters="$OPTARG";;
    b) batch_size="$OPTARG";;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

# Create a temporary script with correct SBATCH directives
temp_script=$(mktemp "temp_script_XXXXXX.sh")

cat <<EOL > "$temp_script"
#!/bin/bash
#SBATCH --ntasks=$ntasks
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --array=0-24%4
#SBATCH -J FLARES2-GRID-$simulation_name-$simulation_type
#SBATCH -o logs/$simulation_name_$simulation_type.%A.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 8:00:00

module purge
module load python/3.10.12 gnu_comp/10.2.0 openmpi/4.1.1

cd /cosma8/data/dp004/dc-rope1/FLARES-2/zoom_region_selection

source /cosma8/data/dp004/dc-rope1/envs/flares-env/bin/activate

# Set your base directory
base_dir="/cosma8/data/dp004/flamingo/Runs/${simulation_name}/${simulation_type}/snapshots"
output_dir="../gridded_data/FLAMINGO/${simulation_name}/${simulation_type}"

# Get the padded SLURM_ARRAY_TASK_ID
padded_task_id=\$(printf "%04d" \$SLURM_ARRAY_TASK_ID)

# Construct input and output paths
input_file="\${base_dir}/flamingo_\${padded_task_id}/flamingo_\${padded_task_id}.hdf5"
output_file="\${output_dir}/grid_\${padded_task_id}.hdf5"

# Check if the output directory exists, create it if not
if [ ! -d "\$(dirname "\$output_dir")" ]; then
    mkdir -p "\$(dirname "\$output_dir")"
fi

# Your mpirun command with variable filepaths
mpirun -np \$SLURM_NTASKS python3 generate_regions.py \
    --input "\$input_file" \
    --output "\$output_file" \
    --nthreads=\$SLURM_CPUS_PER_TASK \
    --kernel_diameters "$diameters" \
    --delete_distributed=1 \
    --grid_width="$width" \
    --batch_size=$batch_size

echo "Job done, info follows..."
sstat --jobs=\${SLURM_JOBID}.batch --format=JobID,MaxRSS,AveCPU,AvePages,AveRSS,AveVMSize
exit
EOL

# Submit the temporary script to SLURM
sbatch "$temp_script"

# Remove the temporary script
rm "$temp_script"

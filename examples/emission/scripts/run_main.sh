#!/bin/bash
#SBATCH --ntasks=15
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=15:00:00
#SBATCH --qos=bbdefault
#SBATCH --mail-type=NONE
#SBATCH --account=piettaaa-exo-mapping
#SBATCH --output=/rds/projects/p/piettaaa-exo-mapping/code/plastar/examples/emission/scripts/slurm_outputs/slurm-%A.out
set -e

module purge; module load bluebear
module load bear-apps/2024a
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA

export VENV_DIR="/rds/projects/p/piettaaa-exo-mapping/code/virtual-environments"
# export VENV_PATH="${VENV_DIR}/hydra-jax-${BB_CPU}"
export VENV_PATH="${VENV_DIR}/plastar-emerald"
# Create a master venv directory if necessary
mkdir -p ${VENV_DIR}

# Activate the virtual environment
source ${VENV_PATH}/bin/activate
module load OpenMPI
module load OpenBLAS
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:"/rds/projects/p/piettaaa-exo-mapping/code/MultiNest/lib"

# Change to the script directory 
cd "/rds/projects/p/piettaaa-exo-mapping/code/plastar/examples/emission/scripts/"
python main.py

# --gres=gpu:a100_80:1
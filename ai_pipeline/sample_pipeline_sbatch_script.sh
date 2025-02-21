#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-training25
#SBATCH --time 24:00:00
#SBATCH --ntasks 16
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 36
#SBATCH --job-name get_predictions_2017  # Title for the job

module purge; module load baskerville

module load bask-apps/live
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)" 

source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

CONDA_ENV_PATH="/bask/projects/v/vjgo8416-demoland/spatial_sigs"

CONDA_PKGS_DIRS=/tmp

# Activate the environment
mamba activate "${CONDA_ENV_PATH}"

python /bask/homes/f/fedu7800/vjgo8416-demoland/spatial_signatures/eo/ai_pipeline/pipeline_spatialsigs_2017.py
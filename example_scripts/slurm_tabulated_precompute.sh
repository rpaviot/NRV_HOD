#!/bin/bash
#SBATCH --job-name=tab_cache_DMO
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=tab_cache_DMO_%j.log

# One-time: DMO halo-center lensing cache WITH xi_gm tabulation
# (consumed by run_tabulated_chains.py via TabulatedDeltaSigma).

source ~/.bashrc
conda activate NRV_ENV
cd ~/NRV_HOD
export PYTHONPATH=$PWD:$PYTHONPATH

FLAM=/sps/euclid/Users/rpaviot/flamingo
PARTICLES=${PARTICLES:-$FLAM/snapshots_DMO/particle_catalogue_DMO.parquet}

python example_scripts/precompute_halo_center_cache.py \
    --halo_path $FLAM/snapshots_DMO/host_catalogue_ab.parquet \
    --particle_path $PARTICLES \
    --output $FLAM/tabulated_cache_DMO.h5 \
    --particle_fraction 0.25 \
    --n_workers 30 \
    --tabulate --ab_column fs_norm \
    --n_logM_bins 40 --n_fI_bins 8

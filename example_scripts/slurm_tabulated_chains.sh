#!/bin/bash
#SBATCH --job-name=tab_chains
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=tab_chains_%j.log

# Nautilus DeltaSigma fits with the tabulated forward model (DMO).
# Single-process vectorized (jit/vmap) likelihood — no multiprocessing pool;
# the 20 CPUs are used by XLA threading. Nautilus checkpoints to
# checkpoint_<case>_rmin<rp>.h5, so resubmitting resumes a timed-out job.
# Usage: sbatch slurm_tabulated_chains.sh [CASE] [RP_MIN]
#   CASE   default: all cases;  RP_MIN default: all of {0.2, 0.5, 1.0, 2.0}
#   (submit one job per (CASE, RP_MIN) to parallelize across nodes)

source ~/.bashrc
conda activate NRV_ENV
cd ~/NRV_HOD
export PYTHONPATH=$PWD:$PYTHONPATH

FLAM=/sps/euclid/Users/rpaviot/flamingo

python example_scripts/run_tabulated_chains.py ${1:-} \
    --cache_path $FLAM/tabulated_cache_DMO.h5 \
    --output_dir $FLAM/chains_TABULATED_DMO \
    ${2:+--rp_min_values $2}

#!/bin/bash
# =============================================================================
# SLURM job-array script for DeltaSigma emulator grid generation.
#
# Three phases, submitted with job dependencies:
#
#   JID_GEN=$(sbatch --parsable --export=PHASE=gen   run_emulator_grid.slurm)
#   JID_ARR=$(sbatch --parsable --export=PHASE=array --dependency=afterok:$JID_GEN run_emulator_grid.slurm)
#   JID_MRG=$(sbatch --parsable --export=PHASE=merge --dependency=afterok:$JID_ARR run_emulator_grid.slurm)
#
# Edit the SLURM directives and CONFIG section below before submitting.
# =============================================================================

#SBATCH --job-name=emulator_grid
#SBATCH --partition=compute          # <-- edit: your cluster partition
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/emulator_%A_%a.out
#SBATCH --error=logs/emulator_%A_%a.err
#SBATCH --array=0-4                  # <-- edit: 0-(N_JOBS-1)

# =============================================================================
# CONFIG — edit these to match your setup
# =============================================================================
N_JOBS=5                             # must match --array upper bound + 1
FIT_CASE=STANDARD_NFW
N_SAMPLES=100000
OUTPUT_DIR=emulator_grid
GRID_PATH=${OUTPUT_DIR}/param_grid_full.parquet
CACHE_PATH=halo_center_lensing_cache.h5
N_REALIZATIONS=5
GALAXY_FRACTION=0.10
PARTICLE_FRACTION=0.05
CHECKPOINT_EVERY=200
BASE_SEED=42
# =============================================================================

mkdir -p logs

# Default phase is 'array' (the main evaluation step)
case "${PHASE:-array}" in

  gen)
    # Phase 1: generate LHS parameter grid and save to disk (runs once, no array)
    echo "[Phase 1] Generating parameter grid..."
    python run_emulator_grid.py \
        --generate_grid_only \
        --fit_case        ${FIT_CASE} \
        --n_samples       ${N_SAMPLES} \
        --output_dir      ${OUTPUT_DIR} \
        --base_seed       ${BASE_SEED}
    ;;

  array)
    # Phase 2: each array job evaluates its slice of the parameter grid
    echo "[Phase 2] Job ${SLURM_ARRAY_TASK_ID}/${N_JOBS} evaluating grid slice..."
    python run_emulator_grid.py \
        --job_array \
        --n_jobs          ${N_JOBS} \
        --grid_path       ${GRID_PATH} \
        --fit_case        ${FIT_CASE} \
        --cache_path      ${CACHE_PATH} \
        --output_dir      ${OUTPUT_DIR} \
        --n_realizations  ${N_REALIZATIONS} \
        --galaxy_fraction ${GALAXY_FRACTION} \
        --particle_fraction ${PARTICLE_FRACTION} \
        --checkpoint_every ${CHECKPOINT_EVERY} \
        --base_seed       ${BASE_SEED}
    ;;

  merge)
    # Phase 3: merge per-job chunks into grid_merged.npz (runs once, no array)
    echo "[Phase 3] Merging ${N_JOBS} chunks..."
    python run_emulator_grid.py \
        --merge_only \
        --n_jobs          ${N_JOBS} \
        --output_dir      ${OUTPUT_DIR} \
        --fit_case        ${FIT_CASE}
    ;;

  *)
    echo "Unknown PHASE='${PHASE}'. Use gen, array, or merge." >&2
    exit 1
    ;;

esac

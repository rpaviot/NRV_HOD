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
#SBATCH --partition=htc          # <-- edit: your cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2500MB
#SBATCH --time=72:00:00
#SBATCH --output=logs/emulator_%A_%a.out
#SBATCH --error=logs/emulator_%A_%a.err
#SBATCH --array=0-24                 # <-- edit: 0-(N_JOBS-1)


source ~/miniconda3/etc/profile.d/conda.sh
conda activate NRV_ENV

# =============================================================================
# CONFIG — edit these to match your setup
# =============================================================================
N_JOBS=25                             # must match --array upper bound + 1
FIT_CASE=CONFORMITY
N_SAMPLES=500_000
OUTPUT_DIR=/sps/euclid/Users/rpaviot/flamingo/emulator_grid_FULL_BB
GRID_PATH=${OUTPUT_DIR}/param_grid_full.parquet
CACHE_PATH=/sps/euclid/Users/rpaviot/flamingo/snapshots_DMO/halo_center_lensing_cache_corrected.h5
N_REALIZATIONS=6
GALAXY_FRACTION=0.10
PARTICLE_FRACTION=0.025
CHECKPOINT_EVERY=200
BASE_SEED=42
MASS_FUNCTION=Despali16
ELG_SAT=1            # 1 = ELG satellite cutoff (Mcut/Mmax instead of kappa), 0 = standard kappa
POPULATION_BACKEND=numba

# Satellite placement: nfw (default) or subhalo (requires SUBHALO_PATH)
SATELLITE_PLACEMENT=nfw
SUBHALO_PATH=        # path to subhalo_catalogue.npz; leave empty for nfw

# Assembly bias: set ASSEMBLY_BIAS=1 to enable; provide at least one of AB_FI / AB_FE
ASSEMBLY_BIAS=1
AB_FI=               # intrinsic AB column name (e.g. concentration); leave empty to omit
AB_FE=fs_norm              # external AB column name (e.g. fs_norm); leave empty to omit

# Extra fixed parameters beyond Ac (space-separated KEY=VALUE pairs)
# Example: FIXED_PARAMS="M1=13 Mmax=15 A_cent=0 A_sat=0"
FIXED_PARAMS="M1=13 Mmax=15 A_cent=0 A_sat=0"
# =============================================================================

mkdir -p logs

# Build optional flag lists from config variables
_ab_flags=""
if [ "${ASSEMBLY_BIAS}" = "1" ]; then
    _ab_flags="--assembly_bias"
    [ -n "${AB_FI}" ] && _ab_flags="${_ab_flags} --ab_fI ${AB_FI}"
    [ -n "${AB_FE}" ] && _ab_flags="${_ab_flags} --ab_fE ${AB_FE}"
fi

_subhalo_flags=""
[ -n "${SUBHALO_PATH}" ] && _subhalo_flags="--subhalo_path ${SUBHALO_PATH}"

_fixed_flags=""
[ -n "${FIXED_PARAMS}" ] && _fixed_flags="--fixed_params ${FIXED_PARAMS}"

# Default phase is 'array' (the main evaluation step)
case "${PHASE:-array}" in

  gen)
    # Phase 1: generate LHS parameter grid and save to disk (runs once, no array)
    echo "[Phase 1] Generating parameter grid..."
    python run_emulator_grid.py \
        --generate_grid_only \
        --fit_case            ${FIT_CASE} \
        --n_samples           ${N_SAMPLES} \
        --output_dir          ${OUTPUT_DIR} \
        --base_seed           ${BASE_SEED} \
        --mass_function       ${MASS_FUNCTION} \
        --elg_satellite       ${ELG_SAT} \
        --satellite_placement ${SATELLITE_PLACEMENT} \
        --population_backend  ${POPULATION_BACKEND} \
        ${_ab_flags} ${_subhalo_flags} ${_fixed_flags}
    ;;

  array)
    # Phase 2: each array job evaluates its slice of the parameter grid
    echo "[Phase 2] Job ${SLURM_ARRAY_TASK_ID}/${N_JOBS} evaluating grid slice..."
    python run_emulator_grid.py \
        --job_array \
        --n_jobs              ${N_JOBS} \
        --grid_path           ${GRID_PATH} \
        --fit_case            ${FIT_CASE} \
        --cache_path          ${CACHE_PATH} \
        --output_dir          ${OUTPUT_DIR} \
        --n_realizations      ${N_REALIZATIONS} \
        --galaxy_fraction     ${GALAXY_FRACTION} \
        --particle_fraction   ${PARTICLE_FRACTION} \
        --checkpoint_every    ${CHECKPOINT_EVERY} \
        --base_seed           ${BASE_SEED} \
        --mass_function       ${MASS_FUNCTION} \
        --elg_satellite       ${ELG_SAT} \
        --satellite_placement ${SATELLITE_PLACEMENT} \
        --population_backend  ${POPULATION_BACKEND} \
        ${_ab_flags} ${_subhalo_flags} ${_fixed_flags}
    ;;

  merge)
    # Phase 3: merge per-job chunks into grid_merged.npz (runs once, no array)
    echo "[Phase 3] Merging ${N_JOBS} chunks..."
    python run_emulator_grid.py \
        --merge_only \
        --n_jobs     ${N_JOBS} \
        --output_dir ${OUTPUT_DIR} \
        --fit_case   ${FIT_CASE}
    ;;
  *)
    echo "Unknown PHASE='${PHASE}'. Use gen, array, or merge." >&2
    exit 1
    ;;

esac

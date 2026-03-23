"""
SLURM job-array grid evaluation for w_gg emulator training.

Three-phase workflow (no MPI required):

Phase 1 — generate the parameter grid (once)::

    python run_emulator_grid_wgg.py \\
        --generate_grid_only \\
        --fit_case STANDARD_NFW \\
        --n_samples 100000 \\
        --output_dir emulator_grid_wgg/

Phase 2 — evaluate grid slices in parallel (SLURM job array)::

    SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=5 \\
        python run_emulator_grid_wgg.py \\
        --job_array --n_jobs 5 \\
        --grid_path emulator_grid_wgg/param_grid_full.parquet \\
        --fit_case STANDARD_NFW \\
        --output_dir emulator_grid_wgg/

Phase 3 — merge per-job chunks::

    python run_emulator_grid_wgg.py \\
        --merge_only --n_jobs 5 \\
        --output_dir emulator_grid_wgg/

Single-process smoke test (no SLURM)::

    python run_emulator_grid_wgg.py --n_samples 50 --no_mpi

Notes
-----
* No particle catalog is needed: w_gg depends only on galaxy positions.
* No precomputed cache is needed: pair counting is fast without DM particles.
* Each job writes its own checkpoint file; resume by re-running the same command.
* After all jobs finish and merge, train the emulator with::

      python -c "
      import numpy as np
      from HOD_NRV.utilsf.emulator_nn import train_emulator
      d = np.load('emulator_grid_wgg/wgg_merged.npz', allow_pickle=True)
      train_emulator(d['params_array'], d['wgg_array'], d['rp_centers'],
                     save_path='emulator_wgg_STANDARD_NFW.pt')
      "
"""

import argparse
import os
import warnings
import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SLURM job-array w_gg grid evaluation for emulator training"
    )
    parser.add_argument(
        "--fit_case",
        choices=["STANDARD_NFW", "EXTENDED_PROFILE", "CONFORMITY"],
        default="STANDARD_NFW",
        help="HOD model complexity (determines free parameters)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100_000,
        help="Total number of LHS grid points (split across all jobs)",
    )
    parser.add_argument(
        "--n_realizations",
        type=int,
        default=10,
        help="HOD realizations per grid point",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="emulator_grid_wgg",
        help="Directory for per-job checkpoint files and merged output",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=200,
        help="Save checkpoint every N completed grid points",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base random seed for LHS + HOD realizations",
    )
    parser.add_argument(
        "--no_mpi",
        action="store_true",
        help="Run without MPI (single process, for local testing)",
    )
    parser.add_argument(
        "--job_array",
        action="store_true",
        help="Activate job array mode: reads rank/size from SLURM_ARRAY_TASK_ID / "
             "SLURM_ARRAY_TASK_COUNT and loads the parameter grid from --grid_path",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of array jobs (fallback if SLURM_ARRAY_TASK_COUNT is not set)",
    )
    parser.add_argument(
        "--grid_path",
        type=str,
        default=None,
        help="Path to pre-saved param_grid_full.parquet (required with --job_array)",
    )
    parser.add_argument(
        "--generate_grid_only",
        action="store_true",
        help="Phase 1: generate and save the LHS parameter grid, then exit",
    )
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="Phase 3: merge per-job grid_rank{i}.npz chunks, then exit",
    )
    parser.add_argument(
        "--population_backend",
        choices=["jax", "numba"],
        default="numba",
        help="Galaxy population backend: 'numba' (default, faster, float64) or 'jax'",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Flamingo data paths and cosmology (edit for your cluster)
# ---------------------------------------------------------------------------

HALO_PATH = "/Users/ler13nrv/Documents/flamingo_data/parquet_halo_catalogue_L1000N1800.parquet"

LBOX = 681.0          # Mpc/h
ZEFF = 1.0
MASS_DEFINITION = "MassDef200m"
TARGET_NGAL = 2e-4    # (Mpc/h)^-3
M1_FIXED = 13.0       # log10 M_sun/h

COLUMN_MAPPING = {
    "x": "x", "y": "y", "z": "z",
    "vx": "vx", "vy": "vy", "vz": "vz",
    "mass": "mass", "radius": "rvir", "c": "c", "vrms": "vrms",
}

COSMO_PARAMS = {
    "h": 0.681,
    "Omc": 0.306 - 0.0486 - 1.39e-3,
    "Omb": 0.0486,
    "A_s": 2.099e-9,
    "n_s": 0.967,
    "Omnu": 1.39e-3,
}

RP_BINS = np.geomspace(0.1, 50.0, 16)   # 15 rp bins [Mpc/h]
PI_BINS = np.linspace(0, 60.0, 61)       # 60 pi bins, 1 Mpc/h wide

# ---------------------------------------------------------------------------
# Parameter ranges per fit case
# ---------------------------------------------------------------------------

_BASE_RANGES = {
    "As":         (0.002, 0.05),
    "Mmin":       (11.5, 13.5),
    "sig_M":      (0.1, 2.0),
    "gamma":      (0.0, 10.0),
    "alpha":      (0.1, 2.0),
    "kappa":      (0.1, 2.0),
    "lambda_NFW": (0.1, 2.0),
}

_EXTENDED_RANGES = {
    "f_exp": (0.1, 0.9),
    "tau":   (1.0, 10.0),
}

_CONFORMITY_RANGES = {
    "kappa_EE": (0.5, 2.0),
}

FIXED_PARAMS = {"M1": M1_FIXED}


def get_param_ranges(fit_case_str: str) -> dict:
    ranges = dict(_BASE_RANGES)
    if fit_case_str in ("EXTENDED_PROFILE", "CONFORMITY"):
        ranges.update(_EXTENDED_RANGES)
    if fit_case_str == "CONFORMITY":
        ranges.update(_CONFORMITY_RANGES)
    return ranges


# ---------------------------------------------------------------------------
# w_gg grid evaluation loop (analogous to run_hod_grid in emulator_utils.py)
# ---------------------------------------------------------------------------

def run_wgg_grid(
    param_grid,
    halo,
    rp_bins: np.ndarray,
    pi_bins: np.ndarray,
    n_realizations: int = 10,
    save_path=None,
    checkpoint_every: int = 200,
    base_seed: int = 42,
    mpi_rank: int = 0,
    target_ngal=None,
) -> tuple:
    """
    Evaluate w_gg on a pre-generated HOD parameter grid.

    Calls halo.compute_avg_clustering(mode='rppi', output='wp') for each row.

    Parameters
    ----------
    param_grid : pd.DataFrame
    halo : HaloOccupation
    rp_bins : array, shape (n_rp+1,)
    pi_bins : array, shape (n_pi+1,)   -- positive-side pi edges [Mpc/h]
    n_realizations : int
    save_path : str or None
    checkpoint_every : int
    base_seed : int
    mpi_rank : int
    target_ngal : float or None

    Returns
    -------
    params_array : np.ndarray, shape (n_points, n_params)
    rp_centers : np.ndarray, shape (n_rp,)
    wgg_array : np.ndarray, shape (n_points, n_rp)
    """
    from HOD_NRV.utilsf.emulator_utils import rescale_Ac_to_target_ngal

    n_points = len(param_grid)
    param_names = list(param_grid.columns)

    rp_centers = np.sqrt(rp_bins[:-1] * rp_bins[1:])
    n_rp = len(rp_centers)

    params_array = param_grid.values.copy()
    wgg_array = np.full((n_points, n_rp), np.nan)

    # Resume from checkpoint if available
    start_idx = 0
    if save_path is not None and os.path.exists(save_path):
        try:
            ckpt = np.load(save_path, allow_pickle=True)
            n_done = int(ckpt.get("n_completed", 0))
            if n_done > 0 and n_done <= n_points:
                wgg_array[:n_done] = ckpt["wgg_array"][:n_done]
                params_array[:n_done] = ckpt["params_array"][:n_done]
                start_idx = n_done
                print(f"[rank {mpi_rank}] Resuming from checkpoint: {n_done}/{n_points} done")
        except Exception as e:
            warnings.warn(f"[rank {mpi_rank}] Could not load checkpoint ({e}); starting fresh")

    print(f"[rank {mpi_rank}] Evaluating {n_points - start_idx} grid points "
          f"({start_idx} already done)")

    for i in range(start_idx, n_points):
        row_params = param_grid.iloc[i].to_dict()
        point_seed = base_seed + i * n_realizations

        if target_ngal is not None:
            Ac_r, As_r = rescale_Ac_to_target_ngal(
                halo.HOD, row_params, target_ngal, Ac_fiducial=0.01
            )
            row_params_for_hod = {**row_params, 'Ac': Ac_r, 'As': As_r}
        else:
            row_params_for_hod = row_params

        try:
            _, wgg_mean, _ = halo.compute_avg_clustering(
                row_params_for_hod,
                n_realizations=n_realizations,
                mode='rppi',
                bins1=rp_bins,
                bins2=pi_bins,
                output='wp',
                base_seed=point_seed,
            )
            wgg_array[i] = wgg_mean
        except Exception as e:
            warnings.warn(f"[rank {mpi_rank}] Grid point {i} failed: {e}")
            wgg_array[i] = np.nan

        if save_path is not None and ((i + 1) % checkpoint_every == 0 or i == n_points - 1):
            np.savez_compressed(
                save_path,
                params_array=params_array,
                rp_centers=rp_centers,
                wgg_array=wgg_array,
                param_names=np.array(param_names, dtype=object),
                n_completed=i + 1,
            )
            print(f"[rank {mpi_rank}] Checkpoint saved: {i + 1}/{n_points}")

    print(f"[rank {mpi_rank}] Grid evaluation complete.")
    return params_array, rp_centers, wgg_array


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def merge_wgg_chunks(directory: str, n_ranks: int, output: str) -> tuple:
    """Merge per-job wgg_rank{i}.npz files into a single wgg_merged.npz."""
    all_params, all_wgg = [], []
    rp_centers = None
    param_names = None

    for rank in range(n_ranks):
        path = os.path.join(directory, f"wgg_rank{rank}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing chunk: {path}")
        d = np.load(path, allow_pickle=True)
        all_params.append(d["params_array"])
        all_wgg.append(d["wgg_array"])
        if rp_centers is None:
            rp_centers = d["rp_centers"]
        if param_names is None:
            param_names = d["param_names"]

    params_merged = np.concatenate(all_params, axis=0)
    wgg_merged = np.concatenate(all_wgg, axis=0)

    # Drop failed rows (all-NaN wgg)
    valid = ~np.all(np.isnan(wgg_merged), axis=1)
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"[merge] Dropping {n_dropped} failed rows")
    params_merged = params_merged[valid]
    wgg_merged = wgg_merged[valid]

    np.savez_compressed(
        output,
        params_array=params_merged,
        rp_centers=rp_centers,
        wgg_array=wgg_merged,
        param_names=param_names,
    )
    return params_merged, rp_centers, wgg_merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    import pandas as pd
    from HOD_NRV.HOD_numerical.HOD import HaloOccupation
    from HOD_NRV.utilsf.emulator_utils import generate_hod_parameter_grid

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1: generate grid and exit
    # -----------------------------------------------------------------------
    if args.generate_grid_only:
        print(f"[Phase 1] Loading halo catalog from {HALO_PATH}")
        halo_gen = HaloOccupation(
            cosmology=COSMO_PARAMS,
            zeff=ZEFF,
            Lbox=LBOX,
            column_mapping=COLUMN_MAPPING,
            mass_definition=MASS_DEFINITION,
            halo_path=HALO_PATH,
            DataFrame_part=None,
            apply_rsd=False,
            do_test=False,
            population_backend=args.population_backend,
        )
        hod_type = "ELG_mHMQ"
        use_conformity = (args.fit_case == "CONFORMITY")
        halo_gen.set_halo_model(hod_type, conformity=use_conformity)

        param_ranges = get_param_ranges(args.fit_case)
        print(f"[Phase 1] Generating {args.n_samples} LHS grid points "
              f"for fit_case={args.fit_case}")
        param_grid = generate_hod_parameter_grid(
            halo=halo_gen,
            hod_type=hod_type,
            param_ranges=param_ranges,
            n_samples=args.n_samples,
            target_ngal=TARGET_NGAL,
            fixed_params=FIXED_PARAMS,
            random_seed=args.base_seed,
            conformity=use_conformity,
            verbose=True,
        )
        grid_meta_path = os.path.join(args.output_dir, "param_grid_full.parquet")
        param_grid.to_parquet(grid_meta_path, index=False)
        print(f"[Phase 1] Full parameter grid saved to {grid_meta_path} "
              f"({len(param_grid)} rows)")
        print("[Phase 1] Done. Submit Phase 2 (job array) next.")
        return

    # -----------------------------------------------------------------------
    # Phase 3: merge chunks and exit
    # -----------------------------------------------------------------------
    if args.merge_only:
        merged_path = os.path.join(args.output_dir, "wgg_merged.npz")
        print(f"[Phase 3] Merging {args.n_jobs} per-job chunks in {args.output_dir}...")
        params_merged, rp_centers, wgg_merged = merge_wgg_chunks(
            directory=args.output_dir,
            n_ranks=args.n_jobs,
            output=merged_path,
        )
        print(f"[Phase 3] Merged grid: {params_merged.shape[0]} points, "
              f"{wgg_merged.shape[1]} rp bins")
        print(f"[Phase 3] Merged file: {merged_path}")
        print()
        print("Next step — train the emulator:")
        fit_case = args.fit_case
        print(f"  python -c \"")
        print(f"  import numpy as np")
        print(f"  from HOD_NRV.utilsf.emulator_nn import train_emulator")
        print(f"  d = np.load('{merged_path}', allow_pickle=True)")
        print(f"  train_emulator(d['params_array'], d['wgg_array'], d['rp_centers'],")
        print(f"                 save_path='emulator_wgg_{fit_case}.pt')\"")
        return

    # -----------------------------------------------------------------------
    # Phase 2: evaluate grid slice
    # -----------------------------------------------------------------------

    if args.no_mpi:
        rank, size = 0, 1
    elif args.job_array:
        rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        size = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", args.n_jobs))
    else:
        rank, size = 0, 1

    # Load or generate parameter grid
    if args.job_array:
        if args.grid_path is None:
            raise ValueError("--grid_path is required when using --job_array")
        print(f"[job {rank}/{size}] Loading parameter grid from {args.grid_path}")
        param_grid = pd.read_parquet(args.grid_path)
    else:
        print(f"[rank {rank}] Loading halo catalog from {HALO_PATH}")
        halo_grid = HaloOccupation(
            cosmology=COSMO_PARAMS,
            zeff=ZEFF,
            Lbox=LBOX,
            column_mapping=COLUMN_MAPPING,
            mass_definition=MASS_DEFINITION,
            halo_path=HALO_PATH,
            DataFrame_part=None,
            apply_rsd=False,
            do_test=False,
            population_backend=args.population_backend,
        )
        hod_type_tmp = "ELG_mHMQ"
        use_conformity_tmp = (args.fit_case == "CONFORMITY")
        halo_grid.set_halo_model(hod_type_tmp, conformity=use_conformity_tmp)

        param_ranges = get_param_ranges(args.fit_case)
        print(f"[rank {rank}] Generating {args.n_samples} LHS grid points "
              f"for fit_case={args.fit_case}")
        param_grid = generate_hod_parameter_grid(
            halo=halo_grid,
            hod_type=hod_type_tmp,
            param_ranges=param_ranges,
            n_samples=args.n_samples,
            target_ngal=TARGET_NGAL,
            fixed_params=FIXED_PARAMS,
            random_seed=args.base_seed,
            conformity=use_conformity_tmp,
            verbose=True,
        )
        grid_meta_path = os.path.join(args.output_dir, "param_grid_full.parquet")
        param_grid.to_parquet(grid_meta_path, index=False)
        print(f"[rank {rank}] Full parameter grid saved to {grid_meta_path}")

    # Slice grid for this job
    n_total = len(param_grid)
    chunk = n_total // size
    start = rank * chunk
    end = start + chunk if rank < size - 1 else n_total
    my_grid = param_grid.iloc[start:end].reset_index(drop=True)

    print(f"[job {rank}/{size}] Processing rows {start}:{end} ({len(my_grid)} points)")

    # Load halo catalog (no particles needed for w_gg)
    print(f"[job {rank}/{size}] Loading halo catalog...")
    halo = HaloOccupation(
        cosmology=COSMO_PARAMS,
        zeff=ZEFF,
        Lbox=LBOX,
        column_mapping=COLUMN_MAPPING,
        mass_definition=MASS_DEFINITION,
        halo_path=HALO_PATH,
        DataFrame_part=None,
        apply_rsd=True,
        do_test=False,
        population_backend=args.population_backend,
    )
    hod_type = "ELG_mHMQ"
    use_conformity = (args.fit_case == "CONFORMITY")
    halo.set_halo_model(hod_type, conformity=use_conformity)
    print(f"[job {rank}/{size}] Population backend: {args.population_backend}")
    print(f"[job {rank}/{size}] Catalog loaded.")

    # Evaluate grid
    save_path = os.path.join(args.output_dir, f"wgg_rank{rank}.npz")
    _, rp_centers, _ = run_wgg_grid(
        my_grid,
        halo,
        RP_BINS,
        PI_BINS,
        n_realizations=args.n_realizations,
        save_path=save_path,
        checkpoint_every=args.checkpoint_every,
        base_seed=args.base_seed + start,
        mpi_rank=rank,
        target_ngal=TARGET_NGAL,
    )

    print(f"[job {rank}/{size}] Done. Results saved to {save_path}")

    if not args.job_array:
        merged_path = os.path.join(args.output_dir, "wgg_merged.npz")
        print("[rank 0] Merging per-rank chunks...")
        params_merged, rp_centers, wgg_merged = merge_wgg_chunks(
            directory=args.output_dir,
            n_ranks=size,
            output=merged_path,
        )
        print(f"[rank 0] Merged grid: {params_merged.shape[0]} points, "
              f"{wgg_merged.shape[1]} rp bins")
        print(f"[rank 0] Merged file: {merged_path}")
        print()
        print("Next step — train the emulator:")
        print(f"  python -c \"")
        print(f"  import numpy as np")
        print(f"  from HOD_NRV.utilsf.emulator_nn import train_emulator")
        print(f"  d = np.load('{merged_path}', allow_pickle=True)")
        print(f"  train_emulator(d['params_array'], d['wgg_array'], d['rp_centers'],")
        print(f"                 save_path='emulator_wgg_{args.fit_case}.pt')\"")
    else:
        print(f"[job {rank}/{size}] Run Phase 3 (--merge_only) after all jobs complete.")


if __name__ == "__main__":
    main()

"""
Subhalo catalogue builder for FLAMINGO SOAP HDF5 files.

Builds both a host halo catalogue (column-compatible with HaloOccupation's
parquet format) and a memory-efficient CSR subhalo catalogue from the same
SOAP file in a single pass.

Typical usage
-------------
    from internal.subhalo_catalogue import (
        build_halo_and_subhalo_catalogues, save_catalogues
    )
    host_cat, sub_cat, csr = build_halo_and_subhalo_catalogues(
        filepath, h=0.681
    )
    save_catalogues(host_cat, sub_cat, csr, output_dir="/data/flamingo/")

Then pass the saved files to HaloOccupation::

    halo = HaloOccupation(
        ...,
        halo_path="/data/flamingo/host_catalogue.parquet",
        subhalo_path="/data/flamingo/subhalo_catalogue.npz",
    )
"""

import gc
import os
import numpy as np
import pandas as pd
import swiftsimio as sw
import unyt as u


def get_vrms_halo(SOAP_catalogue):
    """Return 1D velocity dispersion for all halos from the DM dispersion matrix."""
    vxx, vyy, vzz, _, _, _ = (
        SOAP_catalogue.bound_subhalo
            .dark_matter_velocity_dispersion_matrix
            .to(u.km**2 / u.s**2).T
    )
    vrms = np.sqrt((vxx + vyy + vzz) / 3)
    del vxx, vyy, vzz
    gc.collect()
    return vrms


def build_nisp_catalogue(
    filepath,
    h,
    Lbox=681.0,
    stellar_mass_min=10**10.1,
    stellar_mass_max=10**12.1,
    sfr_min=5.0,
    disc_frac_min=0.7,
):
    """Build the NISP-like ELG galaxy catalogue from a FLAMINGO SOAP file.

    The selection is the one used to make `NISP_catalogue_flamingo.parquet`:
    a stellar-mass window, a star-formation-rate floor and a disc-to-total
    stellar mass fraction floor, all on `exclusive_sphere` apertures.

    Both mass definitions are returned. The catalogue on disk stored
    M200*c* in its `mass` column while every consumer downstream (the host
    catalogue, `MASS_DEFINITION = "MassDef200m"`, the tabulated caches) works
    in M200*m*, so the two have to be carried side by side to see the
    difference rather than inherit it silently. A central takes its own halo's
    mass, a satellite its host's.

    Returns
    -------
    dict with x, y, z [Mpc/h], vx, vy, vz [km/s], mass_200m, mass_200c
    [Msun/h], type (0 central / 1 satellite), stellar_mass [Msun],
    host_row (the SOAP row of the galaxy's host, -1 where unresolved), and
    the host geometry every galaxy needs for a radial-profile measurement:
    r_host [Mpc/h] (periodic distance from the galaxy to its host's centre of
    potential), rvir_host [kpc/h] and c_host.

    r_host is the *exact* offset: it uses the SOAP host index, not a
    nearest-halo match. That distinction is the whole measurement for
    satellites beyond the virial radius -- a nearest-halo link reassigns them
    to whichever small halo they happen to sit next to and truncates the
    measured profile at ~Rvir by construction.
    """
    full = sw.load(filepath)

    sfr = np.array(full.exclusive_sphere_300kpc.star_formation_rate.to(
        u.Msun / u.yr))
    mstar = np.array(full.exclusive_sphere_300kpc.stellar_mass.to(u.Msun))
    disc_frac = np.array(
        full.exclusive_sphere_100kpc.disc_to_total_stellar_mass_fraction)

    blue = ((mstar > stellar_mass_min) & (mstar < stellar_mass_max)
            & (sfr > sfr_min) & (disc_frac > disc_frac_min))
    print(f"  selection: {blue.sum():,} of {len(blue):,} subhalos "
          f"(Mstar {stellar_mass_min:.3e}-{stellar_mass_max:.3e} Msun, "
          f"SFR > {sfr_min}, disc frac > {disc_frac_min})")

    host_idx = full.soap.host_halo_index.value.astype(np.int64)
    is_cen_flag = np.array(full.input_halos.is_central).astype(bool)
    # The host catalogue keys "is a host" off host_halo_index == -1; the NISP
    # builder keyed it off input_halos.is_central. Report any disagreement
    # rather than assume, because the two would silently split the sample.
    n_disagree = int((is_cen_flag != (host_idx == -1)).sum())
    if n_disagree:
        print(f"  WARNING: is_central and host_halo_index==-1 disagree for "
              f"{n_disagree:,} subhalos")

    is_sat = ~is_cen_flag
    mask_cen = blue & is_cen_flag
    mask_sat = blue & is_sat
    # A satellite with no host index would silently index the LAST halo (-1).
    orphan = mask_sat & (host_idx < 0)
    if orphan.sum():
        print(f"  WARNING: {orphan.sum():,} selected satellites have "
              f"host_halo_index < 0 -- dropped (indexing with -1 would have "
              f"assigned them the last halo in the file)")
    mask_sat = mask_sat & (host_idx >= 0)

    m200m = h * np.array(
        full.spherical_overdensity_200_mean.total_mass.to(u.Msun))
    m200c = h * np.array(
        full.spherical_overdensity_200_crit.total_mass.to(u.Msun))

    pos = h * np.array(
        full.exclusive_sphere_300kpc.stellar_centre_of_mass.to(u.Mpc))
    vel = np.array(
        full.exclusive_sphere_300kpc.stellar_centre_of_mass_velocity
        .to_physical().to(u.km / u.s))

    idx_cen = np.where(mask_cen)[0]
    idx_sat = np.where(mask_sat)[0]
    host_of_sat = host_idx[idx_sat]

    # Host geometry, for the satellite radial profile. The centre is the host's
    # centre of potential -- the same point the mock places a central on and
    # measures satellite offsets from -- while `pos` above is the galaxy's
    # stellar centre of mass, so a central lands a few kpc/h off zero.
    cop = h * np.array(full.input_halos.halo_centre.to(u.Mpc))
    rvir = 1e3 * h * np.array(
        full.spherical_overdensity_200_mean.soradius.to(u.Mpc))     # kpc/h
    conc = np.array(full.spherical_overdensity_200_mean.concentration)

    host_row = np.concatenate([idx_cen, host_of_sat])
    delta = pos[np.concatenate([idx_cen, idx_sat])] - cop[host_row]
    delta -= Lbox * np.round(delta / Lbox)                          # min image
    r_host = np.linalg.norm(delta, axis=1)                          # Mpc/h

    out = {
        "x": np.concatenate([pos[idx_cen, 0], pos[idx_sat, 0]]) % Lbox,
        "y": np.concatenate([pos[idx_cen, 1], pos[idx_sat, 1]]) % Lbox,
        "z": np.concatenate([pos[idx_cen, 2], pos[idx_sat, 2]]) % Lbox,
        "vx": np.concatenate([vel[idx_cen, 0], vel[idx_sat, 0]]),
        "vy": np.concatenate([vel[idx_cen, 1], vel[idx_sat, 1]]),
        "vz": np.concatenate([vel[idx_cen, 2], vel[idx_sat, 2]]),
        "mass_200m": np.concatenate([m200m[idx_cen], m200m[host_of_sat]]),
        "mass_200c": np.concatenate([m200c[idx_cen], m200c[host_of_sat]]),
        "type": np.concatenate([np.zeros(len(idx_cen), dtype=np.int32),
                                np.ones(len(idx_sat), dtype=np.int32)]),
        "stellar_mass": np.concatenate([mstar[idx_cen], mstar[idx_sat]]),
        "host_row": host_row,
        "r_host": r_host,                       # Mpc/h, to the host's CoP
        "rvir_host": rvir[host_row],            # kpc/h
        "c_host": conc[host_row],
    }

    # Host mass function, both definitions, for the HOD denominator.
    is_host = host_idx == -1
    out["_host_m200m"] = m200m[is_host]
    out["_host_m200c"] = m200c[is_host]
    out["_host_rows"] = np.where(is_host)[0]
    out["_counts_per_host"] = np.bincount(host_of_sat,
                                          minlength=len(host_idx))[is_host]
    print(f"  centrals {len(idx_cen):,}  satellites {len(idx_sat):,}  "
          f"fsat = {len(idx_sat) / (len(idx_cen) + len(idx_sat)):.4f}")
    x_sat = r_host[len(idx_cen):] / (1e-3 * rvir[host_of_sat])
    print(f"  central offset from host CoP: median "
          f"{1e3 * np.median(r_host[:len(idx_cen)]):.2f} kpc/h")
    print(f"  satellite r/Rvir: median {np.median(x_sat):.3f}, "
          f"{100 * (x_sat > 1).mean():.2f}% beyond Rvir, "
          f"{100 * (x_sat > 3).mean():.2f}% beyond 3 Rvir")
    return out


def build_halo_and_subhalo_catalogues(
    filepath,
    h,
    Lbox=681.0,
    mass_threshold=1e11,
    ranking='r_descending',
    hybrid_alpha=0.5,
):
    """
    Build a host halo catalogue and a CSR subhalo catalogue from a SOAP HDF5 file.

    Both catalogues share the same compressed host index (0..N_host-1) so the
    CSR host_ids directly address rows in the returned host_cat arrays.

    Parameters
    ----------
    filepath       : str   — path to SOAP HDF5 file (swiftsimio-compatible)
    h              : float — dimensionless Hubble parameter (e.g. 0.681)
    Lbox           : float — box size in Mpc/h (default 681)
    mass_threshold : float — minimum host M200m in Msun/h (default 1e11)

    Returns
    -------
    host_cat : dict
        N_host entries; keys: cop, com, v, rvir, mass, concentration, vrms, n_subs
    sub_cat  : dict
        N_sub entries; keys: cop, v, vpeak, host_id, r_to_cop
    csr      : dict
        CSR structure; keys: offsets (N_host+1,), subhalo_indices (N_sub,), max_subs
    """
    # ------------------------------------------------------------------ #
    # 1.  Load the full catalogue                                          #
    # ------------------------------------------------------------------ #
    full = sw.load(filepath)

    host_halo_index = full.soap.host_halo_index.value   # -1 = central/host
    is_host = (host_halo_index == -1)
    is_sub  = (host_halo_index >= 0)

    # ------------------------------------------------------------------ #
    # 2.  Mass cut on hosts                                                #
    # ------------------------------------------------------------------ #
    # .to(Msun) is essential: swiftsimio returns the file's internal mass unit
    # (1e10 Msun for FLAMINGO SOAP), so h * raw is 1e10x too small and the
    # mass_threshold cut silently selects zero hosts. Lengths happen to come
    # back in Mpc because U_L is Mpc, but do not rely on that either.
    so_mass   = h * np.array(
        full.spherical_overdensity_200_mean.total_mass.to(u.Msun))
    host_mask = is_host & (so_mass > mass_threshold)
    host_idx_full = np.where(host_mask)[0]

    # Remap: full SOAP index → compressed host index (-1 if not selected)
    full_to_host = np.full(len(host_halo_index), -1, dtype=np.int64)
    full_to_host[host_idx_full] = np.arange(len(host_idx_full), dtype=np.int64)
    del host_mask

    # ------------------------------------------------------------------ #
    # 3.  Select subhalos whose host passed the mass cut                   #
    # ------------------------------------------------------------------ #
    sub_idx_full_all = np.where(is_sub)[0]
    del is_host, is_sub
    host_of_sub_full = host_halo_index[sub_idx_full_all]
    del host_halo_index

    host_compressed  = full_to_host[host_of_sub_full]
    del full_to_host, host_of_sub_full
    valid_sub        = (host_compressed >= 0)

    sub_idx_full = sub_idx_full_all[valid_sub]
    sub_host_id  = host_compressed[valid_sub]
    del sub_idx_full_all, host_compressed, valid_sub
    gc.collect()

    # ------------------------------------------------------------------ #
    # 4.  Host halo properties                                             #
    # ------------------------------------------------------------------ #
    cop_host = h * np.array(
        full.input_halos.halo_centre[host_idx_full].to(u.Mpc)
    )                                                          # (N_host, 3) [Mpc/h]

    com_host = h * np.array(
        full.spherical_overdensity_200_mean.centre_of_mass[host_idx_full].to(u.Mpc)
    )                                                          # (N_host, 3) [Mpc/h]

    vel_host = np.array(
        full.spherical_overdensity_200_mean
            .centre_of_mass_velocity[host_idx_full]
            .to(u.km / u.s).to_physical()
    )                                                          # (N_host, 3) [km/s]

    rvir_host = h * np.array(
        full.spherical_overdensity_200_mean.soradius[host_idx_full].to(u.kpc)
    )                                                          # (N_host,)   [kpc/h]

    mass_host = so_mass[host_idx_full]                         # (N_host,)   [Msun/h]
    c_host    = np.array(
        full.spherical_overdensity_200_mean.concentration[host_idx_full]
    )                                                          # (N_host,)

    # Velocity dispersion (1D) from DM dispersion matrix diagonal
    vrms_host = get_vrms_halo(full)[host_idx_full]

    host_cat = {
        "cop":           cop_host,    # (N_host, 3) [Mpc/h] — centre of potential
        "com":           com_host,    # (N_host, 3) [Mpc/h] — centre of mass
        "v":             vel_host,    # (N_host, 3) [km/s]
        "rvir":          rvir_host,   # (N_host,)   [kpc/h]
        "mass":          mass_host,   # (N_host,)   [Msun/h]
        "concentration": c_host,      # (N_host,)
        "vrms":          vrms_host,   # (N_host,)   [km/s]
    }

    # ------------------------------------------------------------------ #
    # 5.  Subhalo properties                                               #
    # ------------------------------------------------------------------ #
    cop_sub = h * np.array(
        full.input_halos.halo_centre[sub_idx_full].to(u.Mpc)
    )                                                          # (N_sub, 3) [Mpc/h]

    vel_sub = np.array(
        full.bound_subhalo.centre_of_mass_velocity[sub_idx_full]
            .to(u.km / u.s).to_physical()
    )                                                          # (N_sub, 3) [km/s]

    vpeak_sub = np.array(
        full.input_halos_hbtplus.last_max_vmax_physical[sub_idx_full]
            .to(u.km / u.s).to_physical()
    )                                                          # (N_sub,)   [km/s]

    del full, host_idx_full, sub_idx_full
    gc.collect()

    # Distance to host CoP with periodic boundary conditions
    delta    = cop_sub - host_cat["cop"][sub_host_id]          # (N_sub, 3)
    delta   -= Lbox * np.round(delta / Lbox)                   # minimum image
    r_to_cop = np.linalg.norm(delta, axis=1)                   # (N_sub,)   [Mpc/h]
    del delta
    gc.collect()

    sub_cat = {
        "cop":      cop_sub,       # (N_sub, 3) [Mpc/h]
        "v":        vel_sub,       # (N_sub, 3) [km/s]
        "vpeak":    vpeak_sub,     # (N_sub,)   [km/s]
        "host_id":  sub_host_id,   # (N_sub,)   compressed host index
        "r_to_cop": r_to_cop,      # (N_sub,)   [Mpc/h]
    }

    # ------------------------------------------------------------------ #
    # 6.  CSR structure — no padded matrix                                 #
    # ------------------------------------------------------------------ #
    _VALID_RANKINGS = {'r_ascending', 'r_descending', 'vpeak_descending', 'hybrid'}
    if ranking not in _VALID_RANKINGS:
        raise ValueError(f"ranking must be one of {_VALID_RANKINGS}, got {ranking!r}")

    if ranking == 'r_ascending':
        sort_order = np.lexsort((r_to_cop, sub_host_id))
    elif ranking == 'r_descending':
        sort_order = np.lexsort((-r_to_cop, sub_host_id))
    elif ranking == 'vpeak_descending':
        sort_order = np.lexsort((-vpeak_sub, sub_host_id))
    else:  # hybrid
        rank_df = pd.DataFrame({
            'host_id': sub_host_id,
            'r':       r_to_cop,
            'vpeak':   vpeak_sub,
        })
        # ascending=False, pct=True → highest value gets rank≈0; score≈0 means best
        rank_df['rank_r'] = rank_df.groupby('host_id')['r'].rank(ascending=False, pct=True)
        rank_df['rank_v'] = rank_df.groupby('host_id')['vpeak'].rank(ascending=False, pct=True)
        score = (hybrid_alpha * rank_df['rank_r'].values
                 + (1 - hybrid_alpha) * rank_df['rank_v'].values)
        sort_order = np.lexsort((score, sub_host_id))  # ascending → best (score≈0) first
        del rank_df, score
        gc.collect()
    sorted_host_ids = sub_host_id[sort_order]
    sorted_sub_ids  = np.arange(len(sub_host_id), dtype=np.int64)[sort_order]
    del sort_order

    n_hosts = len(host_cat["cop"])
    counts  = np.bincount(sorted_host_ids, minlength=n_hosts).astype(np.int64)
    del sorted_host_ids
    offsets = np.zeros(n_hosts + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    host_cat["n_subs"] = counts

    # CSR: subhalos for host i → sub_cat[csr_sorted_indices[offsets[i]:offsets[i+1]]]
    csr = {
        "offsets":         offsets,          # (N_host+1,)
        "subhalo_indices": sorted_sub_ids,   # (N_sub,) sorted by r_to_cop within host
        "max_subs":        int(counts.max()) if len(counts) > 0 else 0,
    }

    return host_cat, sub_cat, csr


def save_catalogues(host_cat, sub_cat, csr, output_dir,
                    ranking='r_descending', hybrid_alpha=0.5):
    """
    Save host parquet and subhalo npz to disk.

    Output files
    ------------
    host_catalogue.parquet
        Column names match the default COLUMN_MAPPING in run_emulator_grid.py
        (x, y, z, vx, vy, vz, mass, rvir, c, vrms). Pass its path as
        ``halo_path`` to HaloOccupation.

    subhalo_catalogue.npz
        Keys: sub_positions (N_sub,3) [Mpc/h], sub_velocities (N_sub,3) [km/s],
        csr_offsets (N_host+1,), csr_sorted_indices (N_sub,),
        ranking (bytes scalar), hybrid_alpha (float64). Pass its path as
        ``subhalo_path`` to HaloOccupation.

    Parameters
    ----------
    host_cat     : dict  — output of build_halo_and_subhalo_catalogues()
    sub_cat      : dict  — output of build_halo_and_subhalo_catalogues()
    csr          : dict  — output of build_halo_and_subhalo_catalogues()
    output_dir   : str   — directory to write into (created if absent)
    ranking      : str   — ranking mode used to build the CSR (metadata only)
    hybrid_alpha : float — hybrid weight used (metadata only)
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Host parquet ---
    cop = host_cat["cop"]
    v   = host_cat["v"]
    host_df = pd.DataFrame({
        "x":    cop[:, 0],
        "y":    cop[:, 1],
        "z":    cop[:, 2],
        "vx":   v[:, 0],
        "vy":   v[:, 1],
        "vz":   v[:, 2],
        "mass": host_cat["mass"],
        "rvir": host_cat["rvir"],
        "c":    host_cat["concentration"],
        "vrms": host_cat["vrms"],
    })
    host_path = os.path.join(output_dir, "host_catalogue.parquet")
    host_df.to_parquet(host_path, index=False)
    print(f"Host catalogue: {host_path}  ({len(host_df):,} halos)")

    # --- Subhalo npz ---
    sub_path = os.path.join(output_dir, "subhalo_catalogue.npz")
    np.savez_compressed(
        sub_path,
        sub_positions=sub_cat["cop"].astype(np.float64),
        sub_velocities=sub_cat["v"].astype(np.float64),
        csr_offsets=csr["offsets"].astype(np.int64),
        csr_sorted_indices=csr["subhalo_indices"].astype(np.int64),
        ranking=np.bytes_(ranking),
        hybrid_alpha=np.float64(hybrid_alpha),
    )
    print(f"Subhalo catalogue: {sub_path}  "
          f"({len(sub_cat['cop']):,} subhalos, max {csr['max_subs']:,} per host, "
          f"ranking={ranking})")

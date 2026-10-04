"""Regression: effective_halo_mass must return the mass of a spike occupation,
in the user's mass units, for both units_per_h settings (it once multiplied an
already-h-unit M_eff by h, putting every M_eff 0.17 dex low).

Run with the NRV venv:
    /home/rpaviot/NRV_HOD/.venv_hod/bin/python -m pytest HOD_NRV/HOD_analytical/tests/test_effective_halo_mass.py
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from HOD_NRV.HOD_analytical.halo_model import HaloModel
from HOD_NRV.HOD_analytical.hod_analytical import N_GL, gl_nodes_scaled


class _Params:
    params = {"dummy": 0.0}


def _stub_model(units_per_h, node):
    """HaloModel with only the attributes effective_halo_mass reads (no CCL)."""
    m = object.__new__(HaloModel)
    m.units_per_h = units_per_h
    m.h = 0.6766
    m.n_z = 1
    m.is_single_z = True
    m.log10M_min, m.log10M_max = 10.0, 16.0
    log10M = gl_nodes_scaled(m.log10M_min, m.log10M_max)
    m._M_jax = 10.0 ** log10M
    m._n_M_jax = jnp.ones((1, N_GL))
    m.hod = _Params()
    N_c = jnp.zeros(N_GL).at[node].set(1.0)
    m._get_occupation = lambda iz: (N_c, jnp.zeros(N_GL))
    return m


@pytest.mark.parametrize("units_per_h", [True, False])
@pytest.mark.parametrize("node", [40, 120])
def test_spike_occupation_returns_its_mass(units_per_h, node):
    m = _stub_model(units_per_h, node)
    np.testing.assert_allclose(m.effective_halo_mass(), float(m._M_jax[node]), rtol=1e-12)

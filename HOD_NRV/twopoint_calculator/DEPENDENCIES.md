# Dependencies for twopoint_calculator Module

## External Dependencies

### Required Python Packages

1. **numpy** - Core numerical operations
   - Used in: all modules
   - Arrays, numerical operations

2. **scipy** - Scientific computing
   - `scipy.spatial.cKDTree` - KD-tree for fast neighbor queries
   - `scipy.interpolate.interp1d` - 1D interpolation
   - `scipy.interpolate.RBFInterpolator` - Radial basis function interpolation
   - Used in: `precompute_deltasigma.py`, `fast_two_point.py`, `two_point_legacy.py`

3. **h5py** - HDF5 file I/O
   - Used in: `precompute_deltasigma.py`
   - For saving/loading pre-computed lensing data

4. **pycorr** - Correlation function calculations
   - Used in: `two_point_legacy.py`
   - For legacy correlation function computations

5. **jax/jaxlib** (optional for type hints)
   - Used in: `two_point_legacy.py`
   - Only for `jax.numpy` type hints (can be removed if needed)

### Standard Library

- `typing` - Type hints
- `multiprocessing` - CPU count detection (legacy module)

## Internal Dependencies

### From Parent Module (`HOD_NRV`)

1. **`HOD_NRV.utils.gauss_legendre_integration`**
   - Used in: `precompute_deltasigma.py`, `two_point_legacy.py`
   - For numerical integration via Gauss-Legendre quadrature

## Dependency Tree

```
twopoint_calculator/
├── __init__.py
│   ├── imports: precompute_deltasigma, fast_two_point, two_point_legacy
│   └── exports: all public functions
│
├── precompute_deltasigma.py
│   ├── numpy
│   ├── h5py
│   ├── scipy.spatial.cKDTree
│   ├── scipy.interpolate.interp1d
│   └── ..utils.gauss_legendre_integration
│
├── fast_two_point.py
│   ├── numpy
│   ├── scipy.spatial.cKDTree
│   ├── scipy.interpolate.RBFInterpolator
│   └── .precompute_deltasigma.load_precomputed_lensing
│
└── two_point_legacy.py
    ├── pycorr.TwoPointCorrelationFunction
    ├── numpy
    ├── scipy.interpolate.interp1d
    ├── jax.numpy (type hints only)
    └── ..utils.gauss_legendre_integration
```

## Installation

### Minimal Installation (fast calculator only)

```bash
pip install numpy scipy h5py
```

### Full Installation (including legacy)

```bash
pip install numpy scipy h5py pycorr jax jaxlib
```

### Conda Installation

```bash
conda install numpy scipy h5py
pip install pycorr  # Not available via conda
```

## Import Verification

Test that all dependencies are available:

```python
# Test fast calculator dependencies
try:
    import numpy
    import scipy.spatial
    import scipy.interpolate
    import h5py
    from HOD_NRV.utils import gauss_legendre_integration
    print("✓ Fast calculator dependencies OK")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")

# Test legacy calculator dependencies
try:
    import pycorr
    print("✓ Legacy calculator dependencies OK")
except ImportError:
    print("⚠ pycorr not installed (legacy calculator unavailable)")
```

## Notes

- **pycorr** is only required if using legacy `compute_corr` function
- **jax** is only used for type hints in legacy module (can be removed)
- **h5py** can be replaced with numpy `.npz` format if needed (requires code changes)
- All relative imports use `..` to go up one level from `twopoint_calculator/` to `HOD_NRV/`

## Compatibility

- **Python**: 3.7+
- **NumPy**: 1.18+
- **SciPy**: 1.5+
- **h5py**: 2.10+
- **pycorr**: any recent version
- **JAX**: 0.2+ (optional)

# Extended NFW Profile Updates

## Summary of Changes

Modified `HOD_NRV/HOD_numerical/satellites/NFW_jax.py` to implement proper continuity and ellipticity behavior for extended NFW profiles, following Rocher et al. methodology.

## Key Modifications

### 1. **Continuity at r = Rvir** (NFW_jax.py:179-241)

Added new functions to ensure density continuity at the virial radius:

- `exponential_profile_CDF_continuous(r, tau, Rs, Rvir)`: CDF for exponential profile starting at Rvir (line 179)
- `single_exponential_inverse_CDF_continuous(u, tau, Rs, Rvir, Rmax)`: Inverse CDF sampling ensuring continuity (line 212)

**Optimization**: Uses `vmap(..., in_axes=(0, None, 0, 0, 0))` to efficiently broadcast scalar `tau` parameter without creating arrays.

**Physics**: The exponential component now satisfies:
```
ρ_exp(r) = ρ_NFW(Rvir) × exp(-(r - Rvir)/(τ×Rs))  for r ≥ Rvir
```

This ensures **dN/dr_p continuity** at r = Rvir, matching Rocher et al. approach.

### 2. **Ellipticity Only Inside Rvir** (NFW_jax.py:316-404)

Modified `extended_elliptical_NFW_satellites_positions()` to:

- Apply elliptical transformation **only** for satellites with r < Rvir
- Use **isotropic** directions for satellites with r > Rvir
- Properly handle the transition at the virial radius

**Implementation** (lines 382-402):
```python
# Determine which satellites are inside Rvir
is_inside_rvir = radii_m <= sat_Rvir

# Apply elliptical transformation only for inner satellites
final_directions = jnp.where(
    is_inside_rvir[:, None],
    rotated_elliptical,  # Elliptical for r < Rvir
    directions           # Isotropic for r > Rvir
)
```

### 3. **Consistent Spherical Profile** (NFW_jax.py:244-313)

Updated `extended_NFW_satellites_positions()` to use the same continuity approach, ensuring consistency across both spherical and elliptical implementations.

## Test Results

### Original Tests (HOD_NRV/test.py)
All existing tests pass successfully:
- Spherical NFW: Recovers Rvir and c to within 0.1%
- Triaxial NFW: Recovers axis ratios and orientations correctly

### New Extended Profile Test (test_extended_profiles.py)

Demonstrates the new features:

```
Total satellites: 100000
Inner (r <= Rvir): 70011 (70.0%)
Outer (r > Rvir): 29989 (30.0%)
Expected outer fraction: 30.0%

Inner satellites (r < Rvir) - Expected ellipticity:
  Input axis ratios: [1.00, 0.70, 0.50]
  Measured axis ratios: [1.00, 0.69, 0.50]  ✓

Outer satellites (r > Rvir) - Expected isotropy:
  Input axis ratios: [1.00, 1.00, 1.00] (isotropic)
  Measured axis ratios: [1.00, 0.99, 0.99]
  Isotropy test: PASSED  ✓
```

## Physical Interpretation

### Before Changes:
- ❌ No continuity enforcement at r = Rvir
- ❌ Exponential profile applied ellipticity everywhere
- ❌ Density profiles could have discontinuous jumps

### After Changes:
- ✅ Density continuous at r = Rvir (following Rocher et al.)
- ✅ Inner region (r < Rvir): Elliptical NFW
- ✅ Outer region (r > Rvir): Isotropic exponential
- ✅ Smooth transition between profiles

## Usage

```python
from HOD_NRV.HOD_numerical.HOD import HaloOccupation

# Extended elliptical NFW with proper continuity
halo = HaloOccupation(
    cosmology=cosmo_params,
    zeff=1.0,
    Lbox=1000,
    DataFrame=df,
    triaxial_NFW=True,        # Enable ellipticity for r < Rvir
    f_exp=0.3,                # 30% satellites in outer exponential
    tau=6.0,                  # Exponential decay scale
    lambda_NFW=1.0            # NFW rescaling factor
)
```

## References

This implementation follows the methodology described in:
- Rocher et al. (assumed reference for dN/dr_p continuity)
- Standard NFW profile: Navarro, Frenk & White (1997)

## Files Modified

- `HOD_NRV/HOD_numerical/satellites/NFW_jax.py`: Core implementation
- `test_extended_profiles.py`: New test demonstrating features (not tracked)

## Backward Compatibility

✅ All changes are backward compatible:
- Default parameters maintain original behavior
- Existing code continues to work unchanged
- New features are opt-in via parameters

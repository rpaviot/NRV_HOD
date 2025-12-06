# Improved Interpolation Methods for Noisy ΔΣ Data

## Problem

The original IDW (Inverse Distance Weighting) with k=8 neighbors gives **~98% error** for downsampled data. This is because:

1. **Too few neighbors (k=8)**: In sparse/noisy data, averaging over only 8 neighbors is unstable
2. **No smoothing**: IDW weights decay as 1/d², which is very sensitive to nearby noise
3. **No adaptation**: Same k and weighting everywhere, regardless of local density

## Solution: Better Interpolation Methods

### 1. **Kernel Smoothing** (Nadaraya-Watson) - RECOMMENDED

**How it works:**
```
f(x) = Σ K(d_i/h) × y_i / Σ K(d_i/h)
```
where K is a smooth kernel (Gaussian or Epanechnikov) and h is bandwidth.

**Advantages:**
- Explicitly designed for noisy data with smooth underlying signal
- Bandwidth parameter controls smoothing level
- Gaussian kernel gives smooth, well-behaved interpolation
- More stable than IDW for sparse data

**Parameters:**
- `k_neighbors`: 32-64 (more neighbors = more stable)
- `bandwidth`: Controls smoothing scale (Mpc/h)
  - Smaller h = less smoothing, follows data closely
  - Larger h = more smoothing, ignores local noise
  - Auto-computed from median k-th neighbor distance

**Usage:**
```python
from HOD_NRV.twopoint_calculator.improved_interpolation import ImprovedDeltaSigmaInterpolator

interp = ImprovedDeltaSigmaInterpolator(
    positions, deltasigma, Lbox=1000.0,
    method='kernel_smooth',
    k_neighbors=32,
    bandwidth=None,  # Auto-compute
    kernel='gaussian'
)

ds = interp.interpolate_at_position(galaxy_pos)
```

### 2. **Modified Shepard's Method**

**How it works:**
```
w_i = 1 / d_i^p
```
with larger k (32 instead of 8) and higher power p (3 instead of 2).

**Advantages:**
- Simple, fast
- Higher power = more localized weighting
- More neighbors = more stable

**When to use:**
- If kernel smoothing is too slow
- If you want more local behavior

### 3. **Adaptive Interpolation**

**How it works:**
Automatically adjusts k and bandwidth based on local density:
- Dense regions: smaller k, smaller bandwidth (less smoothing needed)
- Sparse regions: larger k, larger bandwidth (more smoothing needed)

**Advantages:**
- Best of both worlds: accurate in dense regions, stable in sparse regions
- No manual parameter tuning needed

**Usage:**
```python
from HOD_NRV.twopoint_calculator.improved_interpolation import AdaptiveDeltaSigmaInterpolator

interp = AdaptiveDeltaSigmaInterpolator(
    positions, deltasigma, Lbox=1000.0,
    k_min=16, k_max=64
)

ds = interp.interpolate_at_position(galaxy_pos)
```

## Parameter Guidelines

### k_neighbors (number of neighbors)

- **k=8**: Original (too few for noisy data)
- **k=16-32**: Good balance for moderately downsampled data
- **k=64+**: Better for heavily downsampled/noisy data
- **Rule of thumb**: k should be large enough that signal >> noise in averaged neighborhood

### bandwidth (for kernel smoothing)

- **Auto (None)**: Computed as ~1.5 × median k-th neighbor distance
- **h ~ 5 Mpc/h**: Less smoothing, follows data closely
- **h ~ 10-15 Mpc/h**: Moderate smoothing (recommended for noisy data)
- **h ~ 20+ Mpc/h**: Heavy smoothing, suppresses small-scale features

**How to choose:**
1. Start with auto-bandwidth
2. If error still high: increase bandwidth
3. If profiles too smooth: decrease bandwidth

### kernel type

- **Gaussian**: Smooth, infinite support (recommended)
- **Epanechnikov**: Compact support (faster for large k)

## Expected Improvements

For typical downsampled data (40x particle downsampling):

| Method | k | Bandwidth | Expected RMS Error |
|--------|---|-----------|-------------------|
| Original IDW | 8 | N/A | ~98% |
| Kernel (auto) | 32 | ~auto | ~30-50% |
| Kernel (h=10) | 32 | 10 Mpc/h | ~20-40% |
| Kernel (h=20) | 64 | 20 Mpc/h | ~15-30% |
| Adaptive | 16-64 | adaptive | ~20-35% |

## Testing

Run the comparison script:
```bash
cd /Users/ler13nrv/Documents/NRV_HOD/HOD_NRV/test
python compare_interpolation_quick.py
```

This will:
1. Test 7 different interpolation configurations
2. Compare accuracy vs ground truth
3. Plot ΔΣ profiles and error metrics
4. Save results and generate recommendations

## Scientific Justification

### Why kernel smoothing works for noisy ΔΣ(r)

1. **Bias-variance trade-off**: Kernel smoothing explicitly balances:
   - Bias (systematic error from smoothing)
   - Variance (random error from noise)

2. **Optimal for regression**: Nadaraya-Watson is the non-parametric maximum likelihood estimator for noisy scattered data

3. **Scale separation**: Lensing signal ΔΣ(r) is smooth on scales > 1 Mpc/h, while noise is local. Bandwidth h ~ 10 Mpc/h effectively filters noise while preserving signal.

### References

- Nadaraya (1964), "On Estimating Regression", Theory of Probability & Its Applications
- Rasmussen & Williams (2006), "Gaussian Processes for Machine Learning"
- Shepard (1968), "A two-dimensional interpolation function for irregularly-spaced data"

## Next Steps

1. **Run comparison**: `python compare_interpolation_quick.py`
2. **Check results**: Look at error metrics and plots
3. **Choose method**: Based on accuracy vs speed trade-off
4. **Update fast_two_point.py**: Replace IDW with best method
5. **Re-run benchmarks**: Verify improvement in accuracy

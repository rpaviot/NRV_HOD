# Memory-Mapped Arrays for Zero Downsampling

## Problem

When using `joblib` with the `loky` backend for parallel processing with **zero downsampling** (processing millions of particles), each worker process creates a **copy** of the particle position array. This leads to:

- **Memory explosion**: With N workers, you need N × array_size memory
- **Example**: 10M particles × 24 bytes × 16 workers = **3.6 GB** just for particle positions!
- **Slow startup**: Copying large arrays to workers takes significant time

## Solution: Memory-Mapped Arrays (memmap)

Memory-mapped arrays allow **zero-copy sharing** of read-only data across processes:

```
WITHOUT MEMMAP:                    WITH MEMMAP:
┌─────────────────┐               ┌─────────────────┐
│  Main Process   │               │  Main Process   │
│  particles (1GB)│               │  particles.mmap │
└─────────────────┘               └─────────────────┘
        │                                  │
        ├─────┬─────┬─────┬─────          │
        ▼     ▼     ▼     ▼     ▼          ▼ (shared read-only)
    Worker Worker Worker Worker      ┌─────┬─────┬─────┐
    copy   copy   copy   copy        │  W  │  W  │  W  │
    (1GB)  (1GB)  (1GB)  (1GB)       └─────┴─────┴─────┘

Total: 5 GB                          Total: 1 GB
```

## Implementation

### Key Changes

1. **Added `_create_memmap_array()` helper**:
   - Creates temporary memory-mapped file
   - Copies data once to disk
   - Returns read-only memmap view

2. **Updated `precompute_lensing_grid_parallel()`**:
   - New parameter: `use_memmap=True` (default enabled)
   - Creates memmaps for particle arrays when using `loky` backend
   - Automatic cleanup after computation

3. **Updated test script**:
   - Enabled `use_memmap=True` by default
   - Changed backend to `'loky'` (required for memmap)
   - Added memory usage warnings

### Usage

```python
from HOD_NRV.twopoint_calculator.precompute_deltasigma_parallel import (
    precompute_lensing_grid_parallel
)

# Zero downsampling with memory-mapping (LOW MEMORY!)
positions, deltasigma = precompute_lensing_grid_parallel(
    selected_particle_positions=selected_positions,
    particle_positions_full=all_particles,  # Can be 10M+ particles!
    RHO_M=RHO_M,
    rp_bins=rp_bins,
    Lbox=Lbox,
    backend='loky',        # Required for memmap
    use_memmap=True,       # Enable memory-mapping (default)
    n_jobs=-1,
    verbose=True
)
```

### When to Use Memmap

| Scenario | use_memmap | backend | Memory Usage |
|----------|------------|---------|--------------|
| Small data (< 1M particles) | False | any | Low (copies OK) |
| Medium data (1-5M particles) | True | loky | Medium |
| **Large data (> 5M particles)** | **True** | **loky** | **Low** |
| **Zero downsampling** | **True** | **loky** | **Essential!** |

### Performance Impact

**Memory savings example** (10M particles, 16 workers):

```
WITHOUT memmap:
- Main process: 240 MB (particles)
- 16 workers:   240 MB × 16 = 3,840 MB
- Total:        4,080 MB (~4 GB)

WITH memmap:
- Main process: 240 MB (particles)
- Memmap file:  240 MB (on disk, shared)
- 16 workers:   ~0 MB (read-only mapping)
- Total:        ~480 MB (~0.5 GB)

MEMORY SAVED: ~3.6 GB (88% reduction!)
```

**Speed**: Memmap is **slightly slower** initially (disk write), but **faster overall** because:
1. No data copying to workers (saves seconds per worker)
2. Workers start immediately (no waiting for data transfer)
3. Less memory pressure (no swapping)

### Backend Comparison

| Backend | Memory Copy | Memmap Support | Speed | Stability |
|---------|-------------|----------------|-------|-----------|
| **loky** | ✓ (without memmap) | ✓ **Best** | Fast | Excellent |
| multiprocessing | ✓ | Limited | Fast | macOS issues |
| threading | ✗ (shared) | N/A | Slow (GIL) | Excellent |
| numba | ✗ (shared) | N/A | **Fastest** | Excellent |

**Recommendation**:
- **For joblib**: Use `backend='loky'` with `use_memmap=True`
- **For maximum speed**: Use Numba implementation (`precompute_lensing_grid_numba_nokdtree()`)

## Testing

Run the test with zero downsampling:

```python
# In test_fast_dsigma_parallel.py
downsample_factor = 1  # Zero downsampling!
backend = 'loky'
use_memmap = True

# Monitor memory usage during execution
# Expected: ~1-2 GB total instead of 10+ GB
```

Expected output:
```
Creating memory-mapped arrays (backend=loky)...
  This enables zero-copy shared memory across workers
  Particle array size: 240.0 MB
  ✓ Memory-mapped arrays created successfully
  Expected memory saving: 3.60 GB

Processing 50,000 selected particles with -1 parallel jobs...
Backend: loky
Memory-mapping: enabled
```

## Troubleshooting

### Problem: "MemoryError: Unable to allocate array"

**Solution**: Enable memmap!
```python
use_memmap=True  # This should be the default
```

### Problem: "Temporary disk space full"

Memmap writes temporary files to `/tmp` (or `$TMPDIR`). If your `/tmp` is small:

```bash
# Set temporary directory to a larger disk
export TMPDIR=/path/to/large/disk
python your_script.py
```

### Problem: Memmap files not cleaned up

Memmap files are automatically removed after computation. If you see leftover files:

```bash
# Manual cleanup
rm /tmp/joblib_memmap_*.mmap
```

## Technical Details

### Memmap File Format

```
File: /tmp/joblib_memmap_particles_<PID>.mmap
Format: Raw binary (NumPy dtype)
Access: Read-only (mode='r')
Lifetime: Deleted after computation
```

### Memory Layout

```
Main process:
├── particle_positions_full (NumPy array, ~240 MB)
├── particle_positions_mmap (memmap view, ~0 MB extra)
└── Memmap file on disk (~240 MB)

Worker processes (×16):
└── particle_positions_mmap (memmap view, ~0 MB each)
    └── Maps to same disk file (read-only)
```

## Summary

**Use memory-mapped arrays with `loky` backend for zero downsampling!**

This enables processing of **full particle catalogs** without memory explosion, making it feasible to run on typical workstations with 16-32 GB RAM.

**Key parameters**:
```python
backend='loky'
use_memmap=True
downsample_factor=1  # Zero downsampling now possible!
```

**Memory savings**: 80-90% reduction with N workers
**Speed impact**: Minimal (disk I/O amortized across computation)
**Stability**: Excellent (automatic cleanup)

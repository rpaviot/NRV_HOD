import jax.random as jrandom
from .NFW_jax import create_point_on_unit_sphere,spherical_NFW_satellites_positions,elliptical_NFW_satellites_positions
from scipy.optimize import minimize
import numpy as np
import jax.numpy as jnp

def random_orthonormal_basis(seed=None):
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)           # QR gives orthonormal Q
    # Ensure right-handed (det = +1). If det < 0, flip first column.
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q

def spherical_nfw_null_test():
    import numpy as np
    from scipy.optimize import minimize
    import jax.random as jrandom

    # Halo parameters
    Rvir = 200.0
    c_true = 10.0
    Rs = Rvir / c_true
    center = np.array([0.0, 0.0, 0.0])
    N_s = 100_000

    # Sample satellites
    key = jrandom.PRNGKey(0)
    SpherePoints = create_point_on_unit_sphere(key)
    
    key_pos = jrandom.PRNGKey(1)
    cent = jnp.array([center])
    rvir = jnp.array([Rvir])
    c_arr = jnp.array([c_true])
    N_s_arr = jnp.array([N_s])
    N_s_tot = int(jnp.sum(N_s_arr))

    sat_pos = 1000*spherical_NFW_satellites_positions(key_pos, SpherePoints, cent, rvir, c_arr, N_s_arr, N_s_tot)
    sat_pos_np = np.array(sat_pos)

    # Radial distances
    radii = np.linalg.norm(sat_pos_np - center, axis=1)
    radii = np.sort(radii)

    # Empirical CDF
    ecdf = np.arange(1, len(radii)+1) / len(radii)

    # NFW CDF model
    def NFW_cdf(r, Rs, c):
        return (np.log(1 + r / Rs) - (r / Rs) / (1 + r / Rs)) / (np.log(1 + c) - c / (1 + c))

    # Fit c (and optionally Rvir) to ECDF
    def loss_fn(params):
        Rvir_fit, c_fit = params
        Rs_fit = Rvir_fit / c_fit
        model_cdf = NFW_cdf(radii, Rs_fit, c_fit)
        return np.sum((model_cdf - ecdf) ** 2)

    res = minimize(loss_fn, [Rvir, c_true], bounds=[(50, 400), (1, 30)])
    Rvir_fit, c_fit = res.x

    print(f"[SPHERICAL TEST]")
    print(f"  Input    Rvir = {Rvir:.2f}, c = {c_true}")
    print(f"  Recovered Rvir = {Rvir_fit:.2f}, c = {c_fit:.2f}")



def triaxial_nfw_null_test(Q):
    # --- Halo parameters ---
    Rvir = 200.0
    c_true = 10.0
    center = np.array([0.0, 0.0, 0.0])
    N_s = 300_000

    # --- Axis ratios (a = 1) ---
    b_a_true = 0.8
    c_a_true = 0.6
    ratios_for_sat = jnp.array([[b_a_true, c_a_true]])  # shape (1,2)

    # --- Non-trivial basis (columns are basis vectors in world coords) ---
    raw_basis = Q
    # # Orthonormalize using QR and ensure right-handed rotation (determinant +1)
    # Q, R_qr = np.linalg.qr(raw_basis)
    # if np.linalg.det(Q) < 0:
    #     Q[:, 0] *= -1.0
    eigenvectors = jnp.array(raw_basis)   # shape (3,3), columns are orthonormal basis

    shape_matrices = jnp.stack([raw_basis])  # shape (1,3,3)

    # --- JAX setup ---
    key = jrandom.PRNGKey(0)
    SpherePoints = create_point_on_unit_sphere(key)
    key_s_pos = jrandom.PRNGKey(1)
    halo_centers = jnp.array([center])
    Rvir_arr = jnp.array([Rvir])
    c_arr = jnp.array([c_true])
    N_s_arr = jnp.array([N_s])
    N_s_tot = int(jnp.sum(N_s_arr))

    # --- Generate satellites using the ellipsoidal sampler ---
    sat_positions = 1000*elliptical_NFW_satellites_positions(
        key_s_pos,
        SpherePoints,
        halo_centers,
        Rvir_arr,
        c_arr,
        shape_matrices,
        ratios_for_sat,
        N_s_arr,
        N_s_tot
    )
    sat_pos_np = np.array(sat_positions)

    # ------------------------------------------------------------
    # CDF FIT — using ellipsoidal radius m (accounting for rotation)
    # ------------------------------------------------------------
    centered = sat_pos_np - center

    # Correct inverse transform: principal coords = centered @ R
    # (since generation was pos = R @ (D @ x) * r + center)
    centered_principal = centered @ np.array(eigenvectors)   # (N,3)

    # invD consistent with D=[1, b/a, c/a]
    invD = np.diag([1.0, 1.0 / b_a_true, 1.0 / c_a_true])
    m_vals = np.linalg.norm(centered_principal @ invD, axis=1)
    m_vals = np.sort(m_vals)
    ecdf = np.arange(1, len(m_vals) + 1) / len(m_vals)

    def NFW_cdf(r, Rs, c):
        return (np.log(1 + r / Rs) - (r / Rs) / (1 + r / Rs)) / (np.log(1 + c) - c / (1 + c))

    def loss_fn(params):
        Rvir_fit, c_fit = params
        Rs_fit = Rvir_fit / c_fit
        model_cdf = NFW_cdf(m_vals, Rs_fit, c_fit)
        return np.sum((model_cdf - ecdf) ** 2)

    res = minimize(loss_fn, [Rvir, c_true], bounds=[(50, 400), (1, 30)])
    Rvir_fit, c_fit = res.x

    print(f"[TRIAXIAL TEST — ELLIPSOIDAL CDF FIT]")
    print(f" Input Rvir = {Rvir:.2f}, c = {c_true}")
    print(f" Recovered Rvir = {Rvir_fit:.2f}, c = {c_fit:.2f}")

    # ------------------------------------------------------------
    # SHAPE RECOVERY — inertia tensor in Cartesian space
    # ------------------------------------------------------------
    inertia = centered.T @ centered
    evals, evecs = np.linalg.eigh(inertia)  # ascending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    axis_lengths = np.sqrt(evals)
    axis_ratios_fit = axis_lengths / axis_lengths[0]  # normalize so a=1

    print(f"\n[TRIAXIAL TEST — SHAPE RECOVERY]")
    print(f" Input axis ratios (a=1): [1.00, {b_a_true}, {c_a_true}]")
    print(f" Fitted axis ratios (a=1): [{axis_ratios_fit[0]:.2f}, {axis_ratios_fit[1]:.2f}, {axis_ratios_fit[2]:.2f}]")

    print("\n Input eigenvectors (columns):\n", np.array(eigenvectors))
    print(" Fitted eigenvectors (columns):\n", evecs)

def run_all_tests():
    """Run all NFW tests"""
    print("Running spherical test...")
    spherical_nfw_null_test()
    print("\nRunning triaxial test (ellipsoidal)...")
    Q = random_orthonormal_basis(seed=10)
    triaxial_nfw_null_test(Q)

    # Only run if called directly (not imported)
if __name__ == "__main__":
    print("Running NFW null tests...")
    run_all_tests()
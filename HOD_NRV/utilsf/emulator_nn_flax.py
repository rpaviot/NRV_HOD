"""
Pure JAX/Flax re-implementation of the CosmoPower cosmopower_NN emulator.

Trains a neural network mapping HOD parameters → log10 ΔΣ(rp).
Identical public API to emulator_nn.py but requires no TensorFlow.

Architecture mirrors CosmoPower exactly:
  [n_inputs] → [Dense → CustomActivation] × L → [Dense]

Custom activation per layer (trainable α, β):
  f(x) = (β + (1 − β) · sigmoid(α · x)) · x

Usage
-----
Training::

    from HOD_NRV.utilsf.emulator_nn_flax import train_emulator, load_emulator

    emulator = train_emulator(
        params, dsigma, param_names, rp_centers,
        save_path="emulator_STANDARD_NFW"
    )

Inference::

    model, norm_stats = load_emulator("emulator_STANDARD_NFW")
    ds_pred = predict_dsigma(model, norm_stats, theta)
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax.traverse_util import flatten_dict, unflatten_dict


# ---------------------------------------------------------------------------
# Flax modules
# ---------------------------------------------------------------------------

class _CustomActivation(nn.Module):
    """Trainable activation: f(x) = (β + (1 − β) · sigmoid(α · x)) · x"""
    features: int

    @nn.compact
    def __call__(self, x):
        alpha = self.param('alpha', nn.initializers.ones, (self.features,))
        beta  = self.param('beta',  nn.initializers.ones, (self.features,))
        return (beta + (1.0 - beta) * nn.sigmoid(alpha * x)) * x


class _CosmoPowerFlax(nn.Module):
    """MLP mirroring the cosmopower_NN architecture."""
    n_hidden: tuple
    n_outputs: int

    @nn.compact
    def __call__(self, x):
        for h in self.n_hidden:
            x = nn.Dense(h)(x)
            x = _CustomActivation(h)(x)
        return nn.Dense(self.n_outputs)(x)


# ---------------------------------------------------------------------------
# FlaxEmulator wrapper
# ---------------------------------------------------------------------------

class FlaxEmulator:
    """
    Wrapper around a trained Flax model that holds normalization statistics
    and provides a fast JIT-compiled predict() method.

    Parameters
    ----------
    params : dict
        Flax parameter pytree (from model.init or loaded from file).
    apply_fn : callable
        model.apply — the Flax forward function.
    params_mean, params_std : np.ndarray, shape (n_params,)
        Input normalization statistics.
    features_mean, features_std : np.ndarray, shape (n_rp,)
        Output normalization statistics (in log10 ΔΣ space).
    n_hidden : tuple of int
        Hidden layer sizes.
    n_outputs : int
        Number of output bins (= n_rp).
    """

    def __init__(self, params, apply_fn, params_mean, params_std,
                 features_mean, features_std, n_hidden, n_outputs):
        self.flax_params   = params
        self.apply_fn      = apply_fn
        self.params_mean   = np.asarray(params_mean,   dtype=np.float64)
        self.params_std    = np.asarray(params_std,    dtype=np.float64)
        self.features_mean = np.asarray(features_mean, dtype=np.float64)
        self.features_std  = np.asarray(features_std,  dtype=np.float64)
        self.n_hidden      = tuple(n_hidden)
        self.n_outputs     = int(n_outputs)

        # JIT-compile the forward pass once
        self._predict_jit = jax.jit(
            lambda x: apply_fn({'params': params}, x)
        )

    def predict(self, raw_params_batch: np.ndarray) -> np.ndarray:
        """
        Run normalized forward pass and denormalize.

        Parameters
        ----------
        raw_params_batch : np.ndarray, shape (N, n_params)
            Raw (un-normalized) parameter vectors.

        Returns
        -------
        np.ndarray, shape (N, n_rp)
            log10 ΔΣ predictions.
        """
        x_norm = (raw_params_batch - self.params_mean) / self.params_std
        y_norm = np.array(self._predict_jit(jnp.asarray(x_norm, dtype=jnp.float32)))
        return y_norm * self.features_std + self.features_mean


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_train_step():
    """Return a JIT-compiled training step function."""

    @jax.jit
    def train_step(state, x_batch, y_batch):
        def loss_fn(params):
            pred = state.apply_fn({'params': params}, x_batch)
            return jnp.sqrt(jnp.mean((pred - y_batch) ** 2))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step


def _val_loss(state, x_val, y_val):
    pred = state.apply_fn({'params': state.params}, x_val)
    return float(jnp.sqrt(jnp.mean((pred - y_val) ** 2)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_emulator(
    params: np.ndarray,
    dsigma: np.ndarray,
    param_names: list,
    rp_centers: np.ndarray,
    save_path: str,
    n_hidden: list = None,
    validation_split: float = 0.1,
    learning_rates: list = None,
    batch_sizes: list = None,
    patience_values: list = None,
    max_epochs: list = None,
    verbose: bool = True,
) -> FlaxEmulator:
    """
    Train a Flax/JAX neural emulator on a pre-computed HOD grid.

    Parameters
    ----------
    params : np.ndarray, shape (N, n_params)
        Raw HOD parameter values from the grid.
    dsigma : np.ndarray, shape (N, n_rp)
        Corresponding DeltaSigma values. Rows with any non-positive or
        non-finite value are discarded.
    param_names : list of str
        Parameter name for each column of params.
    rp_centers : np.ndarray, shape (n_rp,)
        Projected separation bin centers [Mpc/h]. Stored in the metadata file.
    save_path : str
        Base path for saving. Weights go to ``save_path.flax.npz``;
        metadata is saved to ``save_path.meta.npz``.
    n_hidden : list of int, default=[1024, 1024, 1024, 1024]
        Number of units in each hidden layer.
    validation_split : float, default=0.1
        Fraction of data held out for validation.
    learning_rates : list of float, default=[1e-2, 1e-3, 1e-4, 1e-5]
        Learning rate for each training stage.
    batch_sizes : list of int, default=[512, 512, 512, 512]
        Batch size for each training stage.
    patience_values : list of int, default=[100, 100, 100, 100]
        Early stopping patience for each training stage.
    max_epochs : list of int, default=[1000, 1000, 1000, 1000]
        Maximum epochs for each training stage.
    verbose : bool, default=True
        Print training progress.

    Returns
    -------
    FlaxEmulator
        Trained emulator wrapper.

    Notes
    -----
    * Training is performed in log10(ΔΣ) space.
    * Rows where any ΔΣ value is NaN, non-positive, or params are non-finite
      are dropped before training.
    * The metadata file stores rp_centers and param_names for later loading.
    """
    # Defaults
    if n_hidden is None:
        n_hidden = [1024, 1024, 1024, 1024]
    if learning_rates is None:
        learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    if batch_sizes is None:
        batch_sizes = [512, 512, 512, 512]
    if patience_values is None:
        patience_values = [100, 100, 100, 100]
    if max_epochs is None:
        max_epochs = [1000, 1000, 1000, 1000]

    # --- Drop invalid rows ---
    valid_ds     = np.all(np.isfinite(dsigma) & (dsigma > 0), axis=1)
    valid_params = np.all(np.isfinite(params), axis=1)
    valid        = valid_ds & valid_params
    params       = params[valid]
    dsigma       = dsigma[valid]

    n_total = len(params)
    if n_total < 10:
        raise ValueError(f"Only {n_total} valid grid points — not enough to train.")

    if verbose:
        print(f"Training on {n_total} valid grid points, {dsigma.shape[1]} rp bins.")

    # --- Train/val split ---
    n_val  = max(1, int(n_total * validation_split))
    n_train = n_total - n_val
    rng_np = np.random.default_rng(42)
    idx    = rng_np.permutation(n_total)
    idx_tr, idx_va = idx[:n_train], idx[n_train:]

    params_tr, params_va = params[idx_tr], params[idx_va]
    log_ds_tr  = np.log10(dsigma[idx_tr])
    log_ds_va  = np.log10(dsigma[idx_va])

    # --- Normalization (computed on training split) ---
    params_mean  = params_tr.mean(axis=0)
    params_std   = params_tr.std(axis=0) + 1e-10
    features_mean = log_ds_tr.mean(axis=0)
    features_std  = log_ds_tr.std(axis=0) + 1e-10

    x_tr = ((params_tr   - params_mean)  / params_std ).astype(np.float32)
    y_tr = ((log_ds_tr   - features_mean) / features_std).astype(np.float32)
    x_va = ((params_va   - params_mean)  / params_std ).astype(np.float32)
    y_va = ((log_ds_va   - features_mean) / features_std).astype(np.float32)

    n_inputs  = params.shape[1]
    n_outputs = dsigma.shape[1]

    # --- Build Flax model ---
    model = _CosmoPowerFlax(
        n_hidden=tuple(n_hidden),
        n_outputs=n_outputs,
    )

    # Initialise parameters with a dummy forward pass
    key = jax.random.PRNGKey(0)
    dummy_x = jnp.ones((1, n_inputs), dtype=jnp.float32)
    init_vars = model.init(key, dummy_x)
    best_params = init_vars['params']

    train_step = _make_train_step()

    # --- Multi-stage training ---
    n_stages = len(learning_rates)
    for stage, (lr, bs, pat, me) in enumerate(
            zip(learning_rates, batch_sizes, patience_values, max_epochs)):

        if verbose:
            print(f"  Stage {stage+1}/{n_stages}: lr={lr}, bs={bs}, "
                  f"patience={pat}, max_epochs={me}")

        optimizer = optax.adam(lr)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=best_params,     # carry best params into each new stage
            tx=optimizer,
        )

        best_val  = np.inf
        no_improve = 0
        rng_shuf  = np.random.default_rng(stage)

        for epoch in range(me):
            # Shuffle training data
            perm  = rng_shuf.permutation(n_train)
            x_shuf = x_tr[perm]
            y_shuf = y_tr[perm]

            # Mini-batch loop
            train_losses = []
            for start in range(0, n_train, bs):
                xb = jnp.asarray(x_shuf[start:start + bs])
                yb = jnp.asarray(y_shuf[start:start + bs])
                state, loss = train_step(state, xb, yb)
                train_losses.append(float(loss))

            val_loss = _val_loss(state, jnp.asarray(x_va), jnp.asarray(y_va))

            if val_loss < best_val:
                best_val    = val_loss
                best_params = state.params
                no_improve  = 0
            else:
                no_improve += 1

            if verbose and (epoch % 100 == 0 or no_improve == 0):
                print(f"    epoch {epoch:4d}  train={np.mean(train_losses):.5f}"
                      f"  val={val_loss:.5f}  best={best_val:.5f}")

            if no_improve >= pat:
                if verbose:
                    print(f"    Early stop at epoch {epoch} (no improvement for {pat} epochs)")
                break

        if verbose:
            print(f"  Stage {stage+1} done. Best val RMSE = {best_val:.5f}")

    # --- Wrap in FlaxEmulator ---
    emulator = FlaxEmulator(
        params=best_params,
        apply_fn=model.apply,
        params_mean=params_mean,
        params_std=params_std,
        features_mean=features_mean,
        features_std=features_std,
        n_hidden=n_hidden,
        n_outputs=n_outputs,
    )

    # --- Post-training validation report ---
    if verbose:
        log_ds_va_pred = emulator.predict(params_va)          # (n_val, n_rp), log10 ΔΣ
        ds_va_pred = 10.0 ** log_ds_va_pred
        ds_va_true = 10.0 ** log_ds_va
        frac_err = np.abs(ds_va_pred - ds_va_true) / ds_va_true
        print(f"Validation set ({n_val} points):")
        print(f"  Median |ΔΣ_pred - ΔΣ_true| / ΔΣ_true = {np.median(frac_err)*100:.2f}%")
        print(f"  95th pct                               = {np.percentile(frac_err, 95)*100:.2f}%")
        print(f"  Max                                    = {frac_err.max()*100:.2f}%")

    # --- Save weights ---
    weights_path = save_path + ".flax.npz"
    flat_params  = flatten_dict({'params': best_params}, sep='/')
    save_dict    = {
        'arch_n_hidden':      np.array(n_hidden, dtype=np.int32),
        'arch_n_outputs':     np.array([n_outputs], dtype=np.int32),
        'norm_params_mean':   params_mean.astype(np.float64),
        'norm_params_std':    params_std.astype(np.float64),
        'norm_features_mean': features_mean.astype(np.float64),
        'norm_features_std':  features_std.astype(np.float64),
    }
    for k, v in flat_params.items():
        save_dict[f'w_{k}'] = np.array(v)
    np.savez(weights_path, **save_dict)

    # --- Save metadata (same format as emulator_nn.py) ---
    meta_path = save_path + ".meta.npz"
    np.savez(
        meta_path,
        rp_centers=rp_centers.astype(np.float64),
        param_names=np.array(param_names),
    )

    if verbose:
        print(f"Saved weights to: {weights_path}")
        print(f"Saved metadata to: {meta_path}")

    return emulator


def load_emulator(path: str) -> tuple:
    """
    Load a trained Flax emulator and its metadata.

    Parameters
    ----------
    path : str
        Base path used when saving (written by train_emulator()).
        The companion ``<path>.meta.npz`` must exist.

    Returns
    -------
    emulator : FlaxEmulator
        Restored emulator wrapper.
    norm_stats : dict
        Metadata dict with keys:
        ``rp_centers`` — projected separation bin centers (float64 array),
        ``param_names`` — list of parameter names.
    """
    weights_path = path + ".flax.npz"
    meta_path    = path + ".meta.npz"

    # --- Load weights file ---
    data = np.load(weights_path, allow_pickle=False)

    n_hidden  = list(data['arch_n_hidden'])
    n_outputs = int(data['arch_n_outputs'][0])

    params_mean   = data['norm_params_mean']
    params_std    = data['norm_params_std']
    features_mean = data['norm_features_mean']
    features_std  = data['norm_features_std']

    # Reconstruct flat param dict and unflatten
    flat_params = {}
    prefix = 'w_params/'
    for key in data.files:
        if key.startswith('w_params/'):
            inner_key = key[len('w_'):]   # strip leading 'w_'
            flat_params[tuple(inner_key.split('/'))] = data[key]

    nested_params = unflatten_dict(flat_params)
    # unflatten_dict returns the full nested dict; extract 'params' subtree
    flax_params = nested_params.get('params', nested_params)

    # Rebuild model and wrap
    model = _CosmoPowerFlax(
        n_hidden=tuple(n_hidden),
        n_outputs=n_outputs,
    )

    emulator = FlaxEmulator(
        params=flax_params,
        apply_fn=model.apply,
        params_mean=params_mean,
        params_std=params_std,
        features_mean=features_mean,
        features_std=features_std,
        n_hidden=n_hidden,
        n_outputs=n_outputs,
    )

    # --- Load metadata ---
    meta      = np.load(meta_path, allow_pickle=True)
    rp_centers = meta["rp_centers"]
    param_names = list(meta["param_names"])

    norm_stats = {
        "rp_centers":  rp_centers,
        "param_names": param_names,
    }

    return emulator, norm_stats


def predict_dsigma(
    model: FlaxEmulator,
    norm_stats: dict,
    params: np.ndarray,
) -> np.ndarray:
    """
    Run emulator inference on a batch of parameter vectors.

    Parameters
    ----------
    model : FlaxEmulator
        Trained model (from load_emulator or train_emulator).
    norm_stats : dict
        Metadata dict (from load_emulator). Unused internally but kept for
        API compatibility with emulator_nn.py.
    params : np.ndarray, shape (N, n_params) or (n_params,)
        Raw HOD parameter values.

    Returns
    -------
    dsigma_pred : np.ndarray, shape (N, n_rp) or (n_rp,)
        Predicted DeltaSigma values in original units.
    """
    scalar_input = (params.ndim == 1)
    if scalar_input:
        params = params[np.newaxis, :]

    params = np.asarray(params, dtype=np.float32)
    log10_ds = model.predict(params)      # (N, n_rp) in log10 ΔΣ space
    ds = 10.0 ** log10_ds

    return ds[0] if scalar_input else ds

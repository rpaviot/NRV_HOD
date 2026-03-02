"""
CosmoPower neural emulator for DeltaSigma(rp) predictions.

Trains a cosmopower_NN mapping HOD parameters → log10 ΔΣ(rp).
Designed to replace the slow numerical forward model (~3.2s/call) with a fast
emulator (~μs/call) for Nautilus nested sampling.

Usage
-----
Training::

    from HOD_NRV.utilsf.emulator_nn import train_emulator, load_emulator

    emulator = train_emulator(
        params, dsigma, param_names, rp_centers,
        save_path="emulator_STANDARD_NFW"
    )

Inference::

    model, norm_stats = load_emulator("emulator_STANDARD_NFW")
    ds_pred = predict_dsigma(model, norm_stats, theta)
"""

import numpy as np
from cosmopower_NN import cosmopower_NN


def train_emulator(
    params: np.ndarray,
    dsigma: np.ndarray,
    param_names: list,
    rp_centers: np.ndarray,
    save_path: str,
    n_hidden: list = [1024, 1024, 1024, 1024],
    validation_split: float = 0.1,
    learning_rates: list = [1e-2, 1e-3, 1e-4, 1e-5],
    batch_sizes: list = [512, 512, 512, 512],
    patience_values: list = [100, 100, 100, 100],
    max_epochs: list = [1000, 1000, 1000, 1000],
    verbose: bool = True,
) -> "cosmopower_NN":
    """
    Train a cosmopower_NN emulator on a pre-computed HOD grid.

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
        Base path for saving. cosmopower writes the model weights here;
        metadata is saved to ``save_path + ".meta.npz"``.
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
    cp_nn : cosmopower_NN
        Trained cosmopower model.

    Notes
    -----
    * Training is performed in log10(ΔΣ) space — critical because ΔΣ varies
      by orders of magnitude across rp bins and HOD parameters.
    * Rows where any ΔΣ value is NaN, non-positive, or params are non-finite
      are dropped before training.
    * The metadata file stores rp_centers and param_names for later loading.
    """
    # Drop rows with invalid DeltaSigma or non-finite params
    valid_ds = np.all(np.isfinite(dsigma) & (dsigma > 0), axis=1)
    valid_params = np.all(np.isfinite(params), axis=1)
    valid = valid_ds & valid_params

    params = params[valid]
    dsigma = dsigma[valid]

    n_total = len(params)
    if n_total < 10:
        raise ValueError(f"Only {n_total} valid grid points — not enough to train.")

    if verbose:
        print(f"Training on {n_total} valid grid points, {dsigma.shape[1]} rp bins.")

    # Train in log10 space
    features = np.log10(dsigma)

    # Build cosmopower training dict
    training_parameters = {name: list(params[:, i]) for i, name in enumerate(param_names)}

    n_rp = dsigma.shape[1]
    n_stages = len(learning_rates)

    cp_nn = cosmopower_NN(
        parameters=param_names,
        modes=np.linspace(-1, 1, n_rp),
        n_hidden=n_hidden,
        verbose=verbose,
    )

    cp_nn.train(
        training_parameters=training_parameters,
        training_features=features,
        filename_saved_model=save_path,
        validation_split=validation_split,
        learning_rates=learning_rates,
        batch_sizes=batch_sizes,
        gradient_accumulation_steps=[1] * n_stages,
        patience_values=patience_values,
        max_epochs=max_epochs,
    )

    # Save metadata
    meta_path = save_path + ".meta.npz"
    np.savez(
        meta_path,
        rp_centers=rp_centers.astype(np.float64),
        param_names=np.array(param_names),
    )
    if verbose:
        print(f"Saved metadata to: {meta_path}")

    return cp_nn


def load_emulator(path: str) -> tuple:
    """
    Load a trained cosmopower_NN emulator and its metadata.

    Parameters
    ----------
    path : str
        Base path used when saving (written by train_emulator()).
        The companion ``<path>.meta.npz`` must exist.

    Returns
    -------
    cp_nn : cosmopower_NN
        Restored cosmopower model.
    norm_stats : dict
        Metadata dict with keys:
        ``rp_centers`` — projected separation bin centers (float64 array),
        ``param_names`` — list of parameter names.
    """
    meta_path = path + ".meta.npz"
    meta = np.load(meta_path, allow_pickle=True)
    rp_centers = meta["rp_centers"]
    param_names = list(meta["param_names"])

    n_rp = len(rp_centers)

    cp_nn = cosmopower_NN(
        parameters=param_names,
        modes=np.linspace(-1, 1, n_rp),
    )
    cp_nn.restore(path)

    norm_stats = {
        "rp_centers": rp_centers,
        "param_names": param_names,
    }

    return cp_nn, norm_stats


def predict_dsigma(
    model: "cosmopower_NN",
    norm_stats: dict,
    params: np.ndarray,
) -> np.ndarray:
    """
    Run emulator inference on a batch of parameter vectors.

    Parameters
    ----------
    model : cosmopower_NN
        Trained model (from load_emulator or train_emulator).
    norm_stats : dict
        Metadata dict (from load_emulator). Must contain ``param_names``.
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

    param_names = norm_stats["param_names"]
    params_dict = {name: params[:, i] for i, name in enumerate(param_names)}

    log10_ds = model.predictions_np(params_dict)
    ds = 10.0 ** log10_ds

    return ds[0] if scalar_input else ds

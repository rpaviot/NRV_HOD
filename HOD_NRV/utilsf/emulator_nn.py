"""
PyTorch MLP emulator for DeltaSigma(rp) predictions.

Trains a 4-layer residual MLP mapping normalized HOD parameters θ → log ΔΣ(rp).
Designed to replace the slow numerical forward model (~3.2s/call) with a fast
emulator (~μs/call) for Nautilus nested sampling.

Usage
-----
Training::

    from HOD_NRV.utilsf.emulator_nn import train_emulator, load_emulator

    emulator = train_emulator(
        params, dsigma, rp_centers,
        save_path="emulator_STANDARD_NFW.pt"
    )

Inference::

    emulator, norm_stats = load_emulator("emulator_STANDARD_NFW.pt")
    theta_norm = (theta - norm_stats["x_mean"]) / norm_stats["x_std"]
    log_ds_norm = emulator(torch.tensor(theta_norm, dtype=torch.float32))
    ds_pred = np.exp(log_ds_norm.detach().numpy() * norm_stats["y_std"] + norm_stats["y_mean"])
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple


class DeltaSigmaEmulator(nn.Module):
    """
    4-hidden-layer MLP with residual connections for DeltaSigma emulation.

    Architecture: Input(n_params) → [256→256→256→256] ReLU + skip → Output(n_rp)

    Each pair of consecutive layers forms a residual block:
      - Block 1: layers[0] → layers[1] (with a learned projection if dims differ)
      - Block 2: layers[2] → layers[3]

    Parameters
    ----------
    n_params : int
        Number of input HOD parameters.
    n_rp : int
        Number of rp bins (output dimension).
    hidden_size : int, default=256
        Width of all hidden layers.
    """

    def __init__(self, n_params: int, n_rp: int, hidden_size: int = 256):
        super().__init__()
        self.n_params = n_params
        self.n_rp = n_rp
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(n_params, hidden_size)

        # Residual blocks: each block has 2 linear layers + ReLU
        self.block1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, n_rp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (..., n_params)
            Normalized input parameters.

        Returns
        -------
        torch.Tensor, shape (..., n_rp)
            Normalized log DeltaSigma predictions.
        """
        h = self.relu(self.input_proj(x))
        h = self.relu(self.block1(h) + h)   # residual
        h = self.relu(self.block2(h) + h)   # residual
        return self.output(h)


def _compute_norm_stats(
    params: np.ndarray,
    log_dsigma: np.ndarray,
) -> dict:
    """Compute input/output normalization statistics."""
    x_mean = params.mean(axis=0)
    x_std = params.std(axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)  # avoid division by zero

    y_mean = log_dsigma.mean(axis=0)
    y_std = log_dsigma.std(axis=0)
    y_std = np.where(y_std == 0, 1.0, y_std)

    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }


def train_emulator(
    params: np.ndarray,
    dsigma: np.ndarray,
    rp_centers: np.ndarray,
    val_fraction: float = 0.2,
    n_epochs: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 100,
    hidden_size: int = 256,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> "DeltaSigmaEmulator":
    """
    Train a DeltaSigmaEmulator MLP on a pre-computed HOD grid.

    Parameters
    ----------
    params : np.ndarray, shape (N, n_params)
        Raw HOD parameter values from the grid.
    dsigma : np.ndarray, shape (N, n_rp)
        Corresponding DeltaSigma values. Rows with any NaN are discarded.
    rp_centers : np.ndarray, shape (n_rp,)
        Projected separation bin centers [Mpc/h]. Stored in the saved file.
    val_fraction : float, default=0.2
        Fraction of data held out for validation / early stopping.
    n_epochs : int, default=2000
        Maximum training epochs.
    batch_size : int, default=512
        Mini-batch size.
    lr : float, default=1e-3
        Initial learning rate (Adam optimizer).
    patience : int, default=100
        Early stopping: stop if validation loss does not improve for this
        many epochs.
    hidden_size : int, default=256
        Width of hidden layers in the MLP.
    save_path : str, optional
        If provided, save the trained model + normalization stats to this path.
        Two files are written:
          - ``save_path`` — torch model state dict (.pt)
          - ``save_path + ".norm.npz"`` — normalization stats
    verbose : bool, default=True
        Print training progress.

    Returns
    -------
    model : DeltaSigmaEmulator
        Trained model (on CPU, in eval mode).

    Notes
    -----
    * Inputs are normalized: ``(θ - μ_θ) / σ_θ``
    * Outputs are trained on ``log(ΔΣ)``, normalized: ``(log ΔΣ - μ_y) / σ_y``
    * Rows where any ΔΣ value is NaN or non-positive are dropped before training.
    """
    # Drop rows with invalid DeltaSigma
    valid = np.all(np.isfinite(dsigma) & (dsigma > 0), axis=1)
    params = params[valid].astype(np.float32)
    dsigma = dsigma[valid].astype(np.float32)

    n_total = len(params)
    if n_total < 10:
        raise ValueError(f"Only {n_total} valid grid points — not enough to train.")

    log_dsigma = np.log(dsigma)

    # Normalization
    norm_stats = _compute_norm_stats(params, log_dsigma)
    x_norm = (params - norm_stats["x_mean"]) / norm_stats["x_std"]
    y_norm = (log_dsigma - norm_stats["y_mean"]) / norm_stats["y_std"]

    # Train/val split (random shuffle)
    rng = np.random.default_rng(seed=0)
    idx = rng.permutation(n_total)
    n_val = max(1, int(val_fraction * n_total))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_train = torch.from_numpy(x_norm[train_idx])
    y_train = torch.from_numpy(y_norm[train_idx])
    x_val   = torch.from_numpy(x_norm[val_idx])
    y_val   = torch.from_numpy(y_norm[val_idx])

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    n_params_dim = params.shape[1]
    n_rp = dsigma.shape[1]
    model = DeltaSigmaEmulator(n_params_dim, n_rp, hidden_size=hidden_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(x_train)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = loss_fn(val_pred, y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 100 == 0:
            print(
                f"Epoch {epoch:5d}/{n_epochs}  "
                f"train_loss={train_loss:.4e}  val_loss={val_loss:.4e}  "
                f"best={best_val_loss:.4e}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if verbose:
        print(f"Training complete. Best val loss: {best_val_loss:.4e}  "
              f"({n_total} points, {n_rp} rp bins)")

    if save_path is not None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "n_params": n_params_dim,
                "n_rp": n_rp,
                "hidden_size": hidden_size,
            },
            save_path,
        )
        norm_path = save_path + ".norm.npz"
        np.savez(
            norm_path,
            x_mean=norm_stats["x_mean"],
            x_std=norm_stats["x_std"],
            y_mean=norm_stats["y_mean"],
            y_std=norm_stats["y_std"],
            rp_centers=rp_centers.astype(np.float32),
        )
        if verbose:
            print(f"Saved model to: {save_path}")
            print(f"Saved norm stats to: {norm_path}")

    return model


def load_emulator(path: str) -> Tuple["DeltaSigmaEmulator", dict]:
    """
    Load a trained DeltaSigmaEmulator and its normalization statistics.

    Parameters
    ----------
    path : str
        Path to the saved .pt model file (written by train_emulator()).
        The companion .norm.npz file must exist at ``path + ".norm.npz"``.

    Returns
    -------
    model : DeltaSigmaEmulator
        Loaded model in eval mode on CPU.
    norm_stats : dict
        Normalization statistics with keys:
        ``x_mean``, ``x_std``, ``y_mean``, ``y_std``, ``rp_centers``.
        All values are float32 numpy arrays.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = DeltaSigmaEmulator(
        n_params=ckpt["n_params"],
        n_rp=ckpt["n_rp"],
        hidden_size=ckpt.get("hidden_size", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm_path = path + ".norm.npz"
    norm_data = np.load(norm_path)
    norm_stats = {
        "x_mean": norm_data["x_mean"],
        "x_std": norm_data["x_std"],
        "y_mean": norm_data["y_mean"],
        "y_std": norm_data["y_std"],
        "rp_centers": norm_data["rp_centers"],
    }

    return model, norm_stats


def predict_dsigma(
    model: "DeltaSigmaEmulator",
    norm_stats: dict,
    params: np.ndarray,
) -> np.ndarray:
    """
    Run emulator inference on a batch of parameter vectors.

    Parameters
    ----------
    model : DeltaSigmaEmulator
        Trained model (from load_emulator or train_emulator).
    norm_stats : dict
        Normalization stats (from load_emulator or _compute_norm_stats).
    params : np.ndarray, shape (N, n_params) or (n_params,)
        Raw (unnormalized) HOD parameter values.

    Returns
    -------
    dsigma_pred : np.ndarray, shape (N, n_rp) or (n_rp,)
        Predicted DeltaSigma values in original units.
    """
    scalar_input = (params.ndim == 1)
    if scalar_input:
        params = params[np.newaxis, :]

    x_norm = ((params.astype(np.float32) - norm_stats["x_mean"])
              / norm_stats["x_std"])
    x_t = torch.from_numpy(x_norm)

    model.eval()
    with torch.no_grad():
        y_norm = model(x_t).numpy()

    log_ds = y_norm * norm_stats["y_std"] + norm_stats["y_mean"]
    dsigma_pred = np.exp(log_ds)

    return dsigma_pred[0] if scalar_input else dsigma_pred

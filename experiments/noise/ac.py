"""Allen--Cahn paired control-label noise experiment.

Inputs are read from data/ and generated artifacts are written under
outputs/noise/ac/. Training and evaluation use explicitly selected stages.
"""

from __future__ import print_function

import sys
sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from torch.utils.data import DataLoader, TensorDataset


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT / "outputs" / "noise" / "ac"


def release_path(path: Path) -> str:
    """Serialize provenance without an installation-specific absolute path."""
    return Path(path).resolve().relative_to(PACKAGE_ROOT.resolve()).as_posix()
STATE_DIM = 51
CONTROL_DIM = 1
SIGMA = 0.1
Q_WEIGHT = 0.02
R_WEIGHT = 0.1
R_INV = 10.0
TEST_FRACTION = 0.2
VALUE_HIDDEN_DIM = 128
VALUE_HIDDEN_LAYERS = 2
POLICY_HIDDEN_DIM = 512
POLICY_HIDDEN_LAYERS = 3

RATIO_DIRS = {
    "0.1": "0.1ratio",
    "0.25": "0.25ratio",
    "0.5": "0.5ratio",
    "0.75": "0.75ratio",
}
METHOD_DIRS = {
    "u_hjb": "example2_51d_Allen_Cahn",
    "u_only": "example2_51d_Allen_Cahn-no-HJB",
    "policy_net": "Allen-Cahn_PolicyNet_supervised",
}
METHOD_LABELS = {
    "u_hjb": "u+HJB",
    "u_only": "u-only",
    "policy_net": "PolicyNet-supervised",
}

# The same five Gaussian noise directions are used at every ratio and only
# their scale changes.  This makes comparisons across ratios paired.
NOISE_SEEDS = tuple(range(2026, 2031))
TRAIN_SEED = 42
SPLIT_SEED = 42
FOURIER_TEST_SEED = 20260726
TEST_INITIAL_COUNT = 5

PROTOCOL_DIR = ROOT / "protocol"
FIXED_CLEAN_DATA = PROTOCOL_DIR / "fixed_clean_expert_data.mat"
TEST_INITIALS_FILE = PROTOCOL_DIR / "fourier_test_initial_conditions.mat"
SDRE_CACHE_FILE = PROTOCOL_DIR / "sdre_test_baseline.json"
CANONICAL_SOURCE = PACKAGE_ROOT / "data" / "ac_noise.mat"


def normalize_ratio(value: str) -> str:
    value = str(value).strip().lower().replace("ratio", "")
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid noise ratio: %s" % value) from exc
    for key in RATIO_DIRS:
        if math.isclose(numeric, float(key), rel_tol=0.0, abs_tol=1e-12):
            return key
    raise argparse.ArgumentTypeError(
        "Noise ratio must be one of: %s" % ", ".join(RATIO_DIRS)
    )


def select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment has no CUDA-enabled "
            "PyTorch. Install a CUDA-enabled PyTorch build or pass --device cpu."
        )
    device = torch.device(requested)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        print("Device: %s (%s)" % (device, torch.cuda.get_device_name(index)))
    else:
        print("Device: CPU")
    return device


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The network contains only deterministic dense operations.  These flags
    # remove avoidable cuDNN variability without forcing unsupported kernels.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_b_numpy() -> np.ndarray:
    xi = np.linspace(0.0, 1.0, STATE_DIM)
    return ((xi >= 0.6) & (xi <= 0.9)).astype(np.float64).reshape(-1, 1)


def build_a_base_numpy() -> np.ndarray:
    h = 1.0 / (STATE_DIM - 1)
    coeff = SIGMA / (h * h)
    matrix = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
    matrix[0, 0] = -2.0 * coeff
    matrix[0, 1] = 2.0 * coeff
    for index in range(1, STATE_DIM - 1):
        matrix[index, index - 1] = coeff
        matrix[index, index] = -2.0 * coeff
        matrix[index, index + 1] = coeff
    matrix[-1, -2] = 2.0 * coeff
    matrix[-1, -1] = -2.0 * coeff
    return matrix


def _as_sample_rows(array: np.ndarray, width: int, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1 and width == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("%s must be a 2-D array, got %s" % (name, array.shape))
    if array.shape[1] == width:
        result = array
    elif array.shape[0] == width:
        result = array.T
    else:
        raise ValueError(
            "%s must have one dimension equal to %d, got %s"
            % (name, width, array.shape)
        )
    return np.ascontiguousarray(result)


def load_mat_dataset(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError("Training data not found: %s" % path)
    data = sio.loadmat(str(path))
    required = ("X", "V_all", "DV_all")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError("%s is missing variables: %s" % (path, ", ".join(missing)))

    x = _as_sample_rows(data["X"], STATE_DIM, "X").astype(np.float64)
    value = _as_sample_rows(data["V_all"], 1, "V_all").astype(np.float64)
    gradient = _as_sample_rows(data["DV_all"], STATE_DIM, "DV_all").astype(
        np.float64
    )
    control = (
        _as_sample_rows(data["U_all"], CONTROL_DIM, "U_all").astype(np.float64)
        if "U_all" in data
        else -0.5 * R_INV * (gradient @ build_b_numpy())
    )
    n_samples = x.shape[0]
    for name, array in (("V_all", value), ("DV_all", gradient), ("U_all", control)):
        if array.shape[0] != n_samples:
            raise ValueError(
                "%s has %d samples but X has %d" % (name, array.shape[0], n_samples)
            )
    for name, array in (("X", x), ("V_all", value), ("DV_all", gradient), ("U_all", control)):
        if not np.all(np.isfinite(array)):
            raise ValueError("%s contains NaN or Inf" % name)
    return {"X": x, "V": value, "DV": gradient, "U": control}


def _save_dataset(
    path: Path,
    x: np.ndarray,
    value: np.ndarray,
    gradient: np.ndarray,
    control: np.ndarray,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "X": np.asarray(x, dtype=np.float64),
        "V_all": np.asarray(value, dtype=np.float64),
        "DV_all": np.asarray(gradient, dtype=np.float64).T,
        "U_all": np.asarray(control, dtype=np.float64).T,
        "d": np.array([[STATE_DIM]], dtype=np.int64),
        "state_dim": np.array([[STATE_DIM]], dtype=np.int64),
        "control_dim": np.array([[CONTROL_DIM]], dtype=np.int64),
    }
    if extra:
        payload.update(extra)
    sio.savemat(str(path), payload, do_compression=True)


def prepare_fixed_clean_data(overwrite: bool = False) -> Dict[str, np.ndarray]:
    """Create/load the one clean expert data set used by every experiment."""
    if FIXED_CLEAN_DATA.exists() and not overwrite:
        clean = load_mat_dataset(FIXED_CLEAN_DATA)
        print("Fixed clean expert data: %s" % FIXED_CLEAN_DATA)
        return clean

    source = load_mat_dataset(CANONICAL_SOURCE)
    b_matrix = build_b_numpy()
    # The SDRE generator stores grad V = 2 P x, hence
    # u = -R^{-1} B^T P x = -0.5 R^{-1} B^T grad V.
    clean_control = -0.5 * R_INV * (source["DV"] @ b_matrix)
    if float(np.std(clean_control, ddof=1)) <= 0.0:
        raise ValueError("Reconstructed clean controls have zero variance")

    stored_difference = source["U"] - clean_control
    _save_dataset(
        FIXED_CLEAN_DATA,
        source["X"],
        source["V"],
        source["DV"],
        clean_control,
        extra={
            "source_file": release_path(CANONICAL_SOURCE),
            "control_reconstruction": "-0.5 * R_inv * B.T * DV",
            "input_control_max_abs_difference": np.array(
                [[float(np.max(np.abs(stored_difference)))]], dtype=np.float64
            ),
        },
    )
    print("Created fixed clean expert data: %s" % FIXED_CLEAN_DATA)
    print(
        "  samples=%d, clean U range=[%.6f, %.6f], std=%.6f"
        % (
            source["X"].shape[0],
            float(clean_control.min()),
            float(clean_control.max()),
            float(clean_control.std(ddof=1)),
        )
    )
    return {"X": source["X"], "V": source["V"], "DV": source["DV"], "U": clean_control}


def noisy_data_path(ratio: str, noise_seed: int) -> Path:
    return (
        ROOT
        / RATIO_DIRS[ratio]
        / "noise_repeats"
        / ("noise_seed_%d" % noise_seed)
        / "allen_cahn_training_data.mat"
    )


def model_run_dir(ratio: str, noise_seed: int, method: str) -> Path:
    return (
        ROOT
        / RATIO_DIRS[ratio]
        / METHOD_DIRS[method]
        / "noise_repeats"
        / ("noise_seed_%d" % noise_seed)
    )


def model_checkpoint_path(ratio: str, noise_seed: int, method: str) -> Path:
    filename = (
        "policynet_supervised.pth"
        if method == "policy_net"
        else "pde_controller_model_51d.pth"
    )
    return model_run_dir(ratio, noise_seed, method) / filename


def selected_noise_seeds_for_method(
    method: str, selected_noise_seeds: Sequence[int]
) -> List[int]:
    return list(selected_noise_seeds)


def test_count_for_method(method: str) -> int:
    return TEST_INITIAL_COUNT


def prepare_noisy_dataset(
    clean: Dict[str, np.ndarray],
    ratio: str,
    noise_seed: int,
    overwrite: bool = False,
) -> Path:
    path = noisy_data_path(ratio, noise_seed)
    if path.exists() and not overwrite:
        existing = load_mat_dataset(path)
        if existing["X"].shape != clean["X"].shape:
            raise ValueError("Existing noisy data has the wrong shape: %s" % path)
        print("Noisy data already exists: %s" % path)
        return path

    ratio_value = float(ratio)
    clean_control = clean["U"]
    clean_scale = float(np.std(clean_control, ddof=1))
    noise_std = ratio_value * clean_scale
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(0.0, noise_std, size=clean_control.shape)
    noisy_control = clean_control + noise
    if not np.all(np.isfinite(noisy_control)):
        raise ValueError("Generated noisy controls contain NaN or Inf")

    _save_dataset(
        path,
        clean["X"],
        clean["V"],
        clean["DV"],
        noisy_control,
        extra={
            "U_clean_all": clean_control.T,
            "U_noise_all": noise.T,
            "noise_ratio": np.array([[ratio_value]], dtype=np.float64),
            "noise_seed": np.array([[noise_seed]], dtype=np.int64),
            "clean_u_scale": np.array([[clean_scale]], dtype=np.float64),
            "target_noise_std": np.array([[noise_std]], dtype=np.float64),
            "realized_noise_std": np.array(
                [[float(np.std(noise, ddof=1))]], dtype=np.float64
            ),
        },
    )
    print(
        "Created noisy data: ratio=%s seed=%d target_std=%.6f -> %s"
        % (ratio, noise_seed, noise_std, path)
    )
    return path


class QRNet(nn.Module):
    """Two-layer value network for Allen--Cahn feedback control."""

    def __init__(
        self,
        d: int = STATE_DIM,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        if x_mean is None:
            x_mean = np.zeros(d, dtype=np.float32)
        if x_std is None:
            x_std = np.ones(d, dtype=np.float32)
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.as_tensor(x_std, dtype=torch.float32))
        self.fc1 = nn.Linear(d, VALUE_HIDDEN_DIM)
        self.fc2 = nn.Linear(VALUE_HIDDEN_DIM, VALUE_HIDDEN_DIM)
        self.out = nn.Linear(VALUE_HIDDEN_DIM, 1)
        self.activation = nn.Tanh()

    def _raw_forward(self, x_normalized: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.fc1(x_normalized))
        hidden = self.activation(self.fc2(hidden))
        return self.out(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = (x - self.x_mean) / (self.x_std + 1e-8)
        zero = torch.zeros(
            (1, self.x_mean.shape[0]), device=x.device, dtype=x.dtype
        )
        zero_normalized = (zero - self.x_mean) / (self.x_std + 1e-8)
        shifted_value = self._raw_forward(x_normalized) - self._raw_forward(
            zero_normalized
        )
        positive_value = torch.relu(shifted_value)
        # ReLU in the forward pass, straight-through derivative in backward.
        return shifted_value + (positive_value - shifted_value).detach()


class PolicyNetwork(nn.Module):
    """Direct supervised baseline mapping the physical state to scalar control."""

    def __init__(
        self,
        input_dim: int = STATE_DIM,
        control_dim: int = CONTROL_DIM,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        hidden_dim: int = POLICY_HIDDEN_DIM,
        num_hidden_layers: int = POLICY_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        if x_mean is None:
            x_mean = np.zeros(input_dim, dtype=np.float32)
        if x_std is None:
            x_std = np.ones(input_dim, dtype=np.float32)
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32).clone())
        self.register_buffer("x_std", torch.as_tensor(x_std, dtype=torch.float32).clone())

        layers: List[nn.Module] = []
        in_features = input_dim
        for _ in range(num_hidden_layers):
            layers.extend((nn.Linear(in_features, hidden_dim), nn.Tanh()))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, control_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.x_mean) / (self.x_std + 1e-8)
        return self.network(normalized)


def torch_system_matrices(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    a_base = torch.as_tensor(build_a_base_numpy(), dtype=torch.float32, device=device)
    b_matrix = torch.as_tensor(build_b_numpy(), dtype=torch.float32, device=device)
    return a_base, b_matrix


def model_quantities(
    model: QRNet,
    x: torch.Tensor,
    a_base: torch.Tensor,
    b_matrix: torch.Tensor,
    need_hjb: bool,
    create_graph: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not x.requires_grad:
        x = x.requires_grad_(True)
    value = model(x)
    gradient = torch.autograd.grad(
        outputs=value,
        inputs=x,
        grad_outputs=torch.ones_like(value),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    projected_gradient = gradient @ b_matrix
    control = -0.5 * R_INV * projected_gradient

    if not need_hjb:
        return control, None
    dynamics = x @ a_base.T + x * (1.0 - x.square())
    state_cost = Q_WEIGHT * x.square().sum(dim=1, keepdim=True)
    drift_term = (gradient * dynamics).sum(dim=1, keepdim=True)
    control_term = 0.25 * R_INV * projected_gradient.square()
    hjb_residual = state_cost + drift_term - control_term
    return control, hjb_residual


def fixed_split_indices(
    n_samples: int, test_fraction: float = TEST_FRACTION
) -> Tuple[np.ndarray, np.ndarray]:
    if n_samples < 2:
        raise ValueError("At least two samples are required")
    n_test = int(math.ceil(test_fraction * n_samples))
    permutation = np.random.RandomState(SPLIT_SEED).permutation(n_samples)
    test_indices = permutation[:n_test]
    train_indices = permutation[n_test:]
    return train_indices, test_indices


def _label_loss(
    model: QRNet,
    loader: DataLoader,
    method: str,
    device: torch.device,
    a_base: torch.Tensor,
    b_matrix: torch.Tensor,
    hjb_weight: float,
) -> Tuple[float, float, float]:
    model.eval()
    sum_u = 0.0
    sum_hjb = 0.0
    count = 0
    for batch_x, batch_u in loader:
        batch_x = batch_x.to(device, non_blocking=True).requires_grad_(True)
        batch_u = batch_u.to(device, non_blocking=True)
        control, residual = model_quantities(
            model,
            batch_x,
            a_base,
            b_matrix,
            need_hjb=(method == "u_hjb"),
            create_graph=False,
        )
        batch_count = batch_x.shape[0]
        sum_u += float(torch.mean((control - batch_u).square()).item()) * batch_count
        if residual is not None:
            sum_hjb += float(torch.mean(residual.square()).item()) * batch_count
        count += batch_count
    mean_u = sum_u / count
    mean_hjb = sum_hjb / count if method == "u_hjb" else 0.0
    return hjb_weight * mean_hjb + mean_u, mean_hjb, mean_u


def _training_preflight(
    x_train: np.ndarray,
    u_train: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    method: str,
    requested_batch_size: int,
    device: torch.device,
    hjb_weight: float,
) -> int:
    """Find a batch size that fits before starting a long GPU run."""
    if device.type != "cuda":
        return min(requested_batch_size, len(x_train))

    def try_batch(candidate_batch_size: int) -> None:
        # Keep allocations in a nested scope so that an OOM unwinds all local
        # tensor references before the next, smaller attempt.
        set_global_seed(TRAIN_SEED)
        model = QRNet(STATE_DIM, x_mean, x_std).to(device)
        a_base, b_matrix = torch_system_matrices(device)
        batch_x = torch.as_tensor(
            x_train[:candidate_batch_size], dtype=torch.float32, device=device
        ).requires_grad_(True)
        batch_u = torch.as_tensor(
            u_train[:candidate_batch_size], dtype=torch.float32, device=device
        )
        control, residual = model_quantities(
            model,
            batch_x,
            a_base,
            b_matrix,
            need_hjb=(method == "u_hjb"),
            create_graph=True,
        )
        loss = torch.mean((control - batch_u).square())
        if residual is not None:
            loss = loss + hjb_weight * torch.mean(residual.square())
        loss.backward()

    candidate = min(requested_batch_size, len(x_train))
    while candidate >= 32:
        try:
            try_batch(candidate)
            torch.cuda.empty_cache()
            return candidate
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            candidate //= 2
            print("CUDA memory preflight: reducing batch size to %d" % candidate)
    raise RuntimeError("Even batch size 32 does not fit in CUDA memory")


def train_one_model(
    ratio: str,
    noise_seed: int,
    method: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hjb_weight: float,
    log_interval: int,
    overwrite: bool,
) -> Path:
    output_dir = model_run_dir(ratio, noise_seed, method)
    checkpoint_path = model_checkpoint_path(ratio, noise_seed, method)
    history_path = output_dir / "training_history.csv"
    if checkpoint_path.exists() and not overwrite:
        print("Model already exists, skipping: %s" % checkpoint_path)
        return checkpoint_path

    dataset = load_mat_dataset(noisy_data_path(ratio, noise_seed))
    x = dataset["X"].astype(np.float32)
    control = dataset["U"].astype(np.float32)
    train_indices, test_indices = fixed_split_indices(len(x))
    x_train, u_train = x[train_indices], control[train_indices]
    x_test, u_test = x[test_indices], control[test_indices]

    # Normalization statistics use the full fixed input-state array.
    x_mean = x.mean(axis=0).astype(np.float32)
    x_std = x.std(axis=0).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0

    effective_batch_size = _training_preflight(
        x_train,
        u_train,
        x_mean,
        x_std,
        method,
        batch_size,
        device,
        hjb_weight,
    )
    set_global_seed(TRAIN_SEED)
    model = QRNet(STATE_DIM, x_mean, x_std).to(device)
    a_base, b_matrix = torch_system_matrices(device)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(u_train)
    )
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(u_test))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(TRAIN_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        generator=loader_generator,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, object]] = []
    started = time.time()
    print(
        "\nTraining %s | ratio=%s noise_seed=%d samples=%d batch=%d epochs=%d"
        % (
            METHOD_LABELS[method],
            ratio,
            noise_seed,
            len(x_train),
            effective_batch_size,
            epochs,
        )
    )

    for epoch in range(1, epochs + 1):
        model.train()
        sum_u = 0.0
        sum_hjb = 0.0
        count = 0
        for batch_x, batch_u in train_loader:
            batch_x = batch_x.to(device, non_blocking=True).requires_grad_(True)
            batch_u = batch_u.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted_u, residual = model_quantities(
                model,
                batch_x,
                a_base,
                b_matrix,
                need_hjb=(method == "u_hjb"),
                create_graph=True,
            )
            loss_u = torch.mean((predicted_u - batch_u).square())
            loss_hjb = (
                torch.mean(residual.square())
                if residual is not None
                else torch.zeros((), dtype=loss_u.dtype, device=device)
            )
            total_loss = loss_u + hjb_weight * loss_hjb
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    "Non-finite loss at epoch %d for %s, ratio=%s, seed=%d"
                    % (epoch, method, ratio, noise_seed)
                )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_count = batch_x.shape[0]
            sum_u += float(loss_u.item()) * batch_count
            sum_hjb += float(loss_hjb.item()) * batch_count
            count += batch_count
        scheduler.step()

        should_log = epoch == 1 or epoch == epochs or epoch % log_interval == 0
        if should_log:
            train_u = sum_u / count
            train_hjb = sum_hjb / count if method == "u_hjb" else 0.0
            test_total, test_hjb, test_u = _label_loss(
                model,
                test_loader,
                method,
                device,
                a_base,
                b_matrix,
                hjb_weight,
            )
            row = {
                "epoch": epoch,
                "train_total": train_u + hjb_weight * train_hjb,
                "train_hjb": train_hjb,
                "train_u": train_u,
                "test_total": test_total,
                "test_hjb": test_hjb,
                "test_u": test_u,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(
                "Epoch %4d/%d | train total %.6g (HJB %.6g, U %.6g) | "
                "test total %.6g | %.1fs"
                % (
                    epoch,
                    epochs,
                    row["train_total"],
                    row["train_hjb"],
                    row["train_u"],
                    row["test_total"],
                    row["elapsed_seconds"],
                )
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": STATE_DIM,
            "hidden_dim": VALUE_HIDDEN_DIM,
            "num_hidden_layers": VALUE_HIDDEN_LAYERS,
            "output_dim": 1,
        },
        "method": method,
        "noise_ratio": float(ratio),
        "noise_seed": noise_seed,
        "train_seed": TRAIN_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": epochs,
        "batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "hjb_weight": hjb_weight if method == "u_hjb" else 0.0,
        "test_size": TEST_FRACTION,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "source_data": release_path(noisy_data_path(ratio, noise_seed)),
    }
    torch.save(checkpoint, str(checkpoint_path))
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print("Saved model: %s" % checkpoint_path)
    return checkpoint_path


def _policy_label_loss(
    model: PolicyNetwork,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    squared_error = 0.0
    elements = 0
    with torch.no_grad():
        for batch_x, batch_u in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_u = batch_u.to(device, non_blocking=True)
            prediction = model(batch_x)
            squared_error += float(torch.sum((prediction - batch_u).square()).item())
            elements += batch_u.numel()
    return squared_error / max(elements, 1)


def train_one_policy_model(
    ratio: str,
    noise_seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    log_interval: int,
    overwrite: bool,
) -> Path:
    """Train the direct x -> u PolicyNet baseline on one noisy label set."""
    method = "policy_net"
    output_dir = model_run_dir(ratio, noise_seed, method)
    checkpoint_path = model_checkpoint_path(ratio, noise_seed, method)
    history_path = output_dir / "training_history.csv"
    if checkpoint_path.exists() and not overwrite:
        print("Model already exists, skipping: %s" % checkpoint_path)
        return checkpoint_path

    dataset = load_mat_dataset(noisy_data_path(ratio, noise_seed))
    x = dataset["X"].astype(np.float32)
    control = dataset["U"].astype(np.float32)
    train_indices, test_indices = fixed_split_indices(len(x))
    x_train, u_train = x[train_indices], control[train_indices]
    x_test, u_test = x[test_indices], control[test_indices]

    x_mean = x.mean(axis=0).astype(np.float32)
    x_std = x.std(axis=0).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0
    effective_batch_size = min(batch_size, len(x_train))

    set_global_seed(TRAIN_SEED)
    model = PolicyNetwork(
        input_dim=STATE_DIM,
        control_dim=CONTROL_DIM,
        x_mean=x_mean,
        x_std=x_std,
        hidden_dim=POLICY_HIDDEN_DIM,
        num_hidden_layers=POLICY_HIDDEN_LAYERS,
    ).to(device)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(u_train)
    )
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(u_test))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(TRAIN_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        generator=loader_generator,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, object]] = []
    started = time.time()
    print(
        "\nTraining %s | ratio=%s noise_seed=%d samples=%d batch=%d epochs=%d"
        % (
            METHOD_LABELS[method],
            ratio,
            noise_seed,
            len(x_train),
            effective_batch_size,
            epochs,
        )
    )

    for epoch in range(1, epochs + 1):
        model.train()
        squared_error = 0.0
        elements = 0
        for batch_x, batch_u in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_u = batch_u.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = torch.mean((prediction - batch_u).square())
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite PolicyNet loss at epoch %d, ratio=%s, seed=%d"
                    % (epoch, ratio, noise_seed)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            squared_error += float(
                torch.sum((prediction.detach() - batch_u).square()).item()
            )
            elements += batch_u.numel()
        scheduler.step()

        should_log = epoch == 1 or epoch == epochs or epoch % log_interval == 0
        if should_log:
            train_u = squared_error / max(elements, 1)
            test_u = _policy_label_loss(model, test_loader, device)
            row = {
                "epoch": epoch,
                "train_total": train_u,
                "train_hjb": 0.0,
                "train_u": train_u,
                "test_total": test_u,
                "test_hjb": 0.0,
                "test_u": test_u,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(
                "Epoch %4d/%d | train control MSE %.6g | test control MSE %.6g | %.1fs"
                % (epoch, epochs, train_u, test_u, row["elapsed_seconds"])
            )

    checkpoint = {
        "baseline": "PolicyNet-supervised",
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": STATE_DIM,
            "control_dim": CONTROL_DIM,
            "hidden_dim": POLICY_HIDDEN_DIM,
            "num_hidden_layers": POLICY_HIDDEN_LAYERS,
        },
        "training_config": {
            "seed": TRAIN_SEED,
            "split_seed": SPLIT_SEED,
            "epochs": epochs,
            "batch_size": effective_batch_size,
            "learning_rate": learning_rate,
            "test_size": TEST_FRACTION,
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "noise_ratio": float(ratio),
            "noise_seed": noise_seed,
            "source_data": release_path(noisy_data_path(ratio, noise_seed)),
        },
        "final_train_mse": history[-1]["train_u"],
        "final_test_mse": history[-1]["test_u"],
    }
    torch.save(checkpoint, str(checkpoint_path))
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print("Saved model: %s" % checkpoint_path)
    return checkpoint_path


def generate_fourier_initial_conditions(
    count: int = TEST_INITIAL_COUNT,
    seed: int = FOURIER_TEST_SEED,
    max_amplitude: float = 2.0,
    harmonics: int = 5,
) -> np.ndarray:
    """Use exactly the smooth Fourier family from the expert-data generator."""
    rng = np.random.default_rng(seed)
    xi = np.linspace(0.0, 1.0, STATE_DIM)
    initial_conditions = np.zeros((count, STATE_DIM), dtype=np.float64)
    for sample_index in range(count):
        state = np.zeros(STATE_DIM, dtype=np.float64)
        for harmonic in range(1, harmonics + 1):
            amplitude = rng.uniform(-1.0 / harmonic, 1.0 / harmonic)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            state += amplitude * np.cos(2.0 * np.pi * harmonic * xi + phase)
        scaling = rng.uniform(0.5, max_amplitude)
        baseline = rng.uniform(-max_amplitude / 2.0, max_amplitude / 2.0)
        initial_conditions[sample_index] = scaling * state + baseline
    return initial_conditions


def test_initial_hash(initial_conditions: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(initial_conditions, dtype=np.float64).tobytes()
    ).hexdigest()


def prepare_test_initial_conditions(overwrite: bool = False) -> np.ndarray:
    generated = generate_fourier_initial_conditions(count=TEST_INITIAL_COUNT)
    if TEST_INITIALS_FILE.exists() and not overwrite:
        data = sio.loadmat(str(TEST_INITIALS_FILE))
        cached = _as_sample_rows(data["X0_test"], STATE_DIM, "X0_test").astype(
            np.float64
        )
        if cached.shape[0] == TEST_INITIAL_COUNT:
            if not np.array_equal(cached, generated):
                raise ValueError(
                    "Cached Fourier initial conditions do not match the fixed seed"
                )
            print("Fourier test initial conditions: %s" % TEST_INITIALS_FILE)
            return cached
        if cached.shape[0] > TEST_INITIAL_COUNT:
            raise ValueError(
                "Cached test set has %d states; expected %d"
                % (cached.shape[0], TEST_INITIAL_COUNT)
            )
        if not np.array_equal(cached, generated[: cached.shape[0]]):
            raise ValueError(
                "Cannot extend the cached test set because its existing prefix "
                "does not match the fixed Fourier seed"
            )
        print(
            "Extending Fourier test initial conditions from %d to %d while "
            "preserving the existing prefix."
            % (cached.shape[0], TEST_INITIAL_COUNT)
        )

    initial_conditions = generated
    TEST_INITIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        str(TEST_INITIALS_FILE),
        {
            "X0_test": initial_conditions,
            "fourier_seed": np.array([[FOURIER_TEST_SEED]], dtype=np.int64),
            "num_harmonics": np.array([[5]], dtype=np.int64),
            "max_amplitude": np.array([[2.0]], dtype=np.float64),
            "test_initial_hash": test_initial_hash(initial_conditions),
        },
        do_compression=True,
    )
    print(
        "Created %d shared Fourier test initial conditions: %s"
        % (TEST_INITIAL_COUNT, TEST_INITIALS_FILE)
    )
    for index, state in enumerate(initial_conditions, start=1):
        print(
            "  test %d: mean=% .4f, RMS=%.4f, range=[%.4f, %.4f]"
            % (
                index,
                float(np.mean(state)),
                float(np.sqrt(np.mean(np.square(state)))),
                float(np.min(state)),
                float(np.max(state)),
            )
        )
    return initial_conditions


class AllenCahnSystem:
    def __init__(self, final_time: float = 20.0, time_step: float = 0.01) -> None:
        self.final_time = float(final_time)
        self.time_step = float(time_step)
        self.a_base = build_a_base_numpy()
        self.b_matrix = build_b_numpy()
        self.q_matrix = Q_WEIGHT * np.eye(STATE_DIM)
        self.r_matrix = np.array([[R_WEIGHT]], dtype=np.float64)

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return (
            self.a_base @ state
            + state * (1.0 - np.square(state))
            + self.b_matrix[:, 0] * float(control[0])
        )

    def sdre_control(self, state: np.ndarray) -> np.ndarray:
        a_state = self.a_base + np.diag(1.0 - np.square(state))
        riccati = solve_continuous_are(
            a_state, self.b_matrix, self.q_matrix, self.r_matrix
        )
        return -R_INV * (self.b_matrix.T @ riccati @ state)

    def simulate(
        self,
        initial_state: np.ndarray,
        policy: Callable[[np.ndarray], np.ndarray],
        failure_threshold: float = 1000.0,
    ) -> Dict[str, object]:
        n_steps = int(round(self.final_time / self.time_step)) + 1
        states = np.zeros((n_steps, STATE_DIM), dtype=np.float64)
        controls = np.zeros((n_steps, CONTROL_DIM), dtype=np.float64)
        states[0] = np.asarray(initial_state, dtype=np.float64)
        reason = ""
        completed_steps = 1

        for step in range(n_steps - 1):
            try:
                control = np.asarray(policy(states[step]), dtype=np.float64).reshape(1)
            except Exception as exc:  # Preserve the exact failing test in the CSV.
                reason = "control error at step %d: %s" % (step, exc)
                break
            if not np.all(np.isfinite(control)):
                reason = "non-finite control at step %d" % step
                break
            controls[step] = control
            solution = solve_ivp(
                lambda _time, state: self.dynamics(state, control),
                (step * self.time_step, (step + 1) * self.time_step),
                states[step],
                t_eval=[(step + 1) * self.time_step],
                method="RK45",
            )
            if not solution.success or solution.y.size == 0:
                reason = "ODE solver failed at step %d: %s" % (step, solution.message)
                break
            next_state = solution.y[:, -1]
            if not np.all(np.isfinite(next_state)):
                reason = "non-finite state at step %d" % (step + 1)
                break
            states[step + 1] = next_state
            completed_steps = step + 2
            if float(np.max(np.abs(next_state))) > failure_threshold:
                reason = "state magnitude exceeded %.1f at step %d" % (
                    failure_threshold,
                    step + 1,
                )
                break

        success = completed_steps == n_steps and not reason
        if success:
            try:
                controls[-1] = np.asarray(policy(states[-1]), dtype=np.float64).reshape(1)
            except Exception as exc:
                success = False
                reason = "final control error: %s" % exc

        used_states = states[:completed_steps]
        used_controls = controls[:completed_steps]
        if success:
            running_cost = (
                Q_WEIGHT * np.sum(np.square(used_states), axis=1)
                + R_WEIGHT * np.sum(np.square(used_controls), axis=1)
            )
            cost = float(np.trapz(running_cost, dx=self.time_step))
        else:
            cost = float("nan")
        return {
            "success": success,
            "cost": cost,
            "failure_reason": reason,
            "completed_steps": completed_steps,
            "initial_rms": float(np.sqrt(np.mean(np.square(states[0])))),
            "terminal_rms": float(np.sqrt(np.mean(np.square(used_states[-1])))),
            "max_abs_state": float(np.max(np.abs(used_states))),
        }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def get_sdre_baseline(
    initial_conditions: np.ndarray,
    final_time: float,
    time_step: float,
    overwrite: bool = False,
) -> List[Dict[str, object]]:
    expected_hash = test_initial_hash(initial_conditions)
    tests: List[Dict[str, object]] = []
    if SDRE_CACHE_FILE.exists() and not overwrite:
        with SDRE_CACHE_FILE.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        same_simulation = math.isclose(
            float(cached.get("final_time")), final_time
        ) and math.isclose(float(cached.get("time_step")), time_step)
        cached_tests = list(cached.get("tests", []))
        if same_simulation and cached.get("test_initial_hash") == expected_hash:
            if len(cached_tests) != len(initial_conditions):
                raise ValueError("SDRE cache hash matches but its test count is invalid")
            print("Using cached SDRE test baseline: %s" % SDRE_CACHE_FILE)
            return cached_tests

        # The previous protocol used the first five states generated by the
        # same RNG stream. Reuse that exact prefix and compute only new states.
        prefix_count = len(cached_tests)
        prefix_is_valid = (
            same_simulation
            and 0 < prefix_count < len(initial_conditions)
            and cached.get("test_initial_hash")
            == test_initial_hash(initial_conditions[:prefix_count])
            and [int(row["test_index"]) for row in cached_tests]
            == list(range(1, prefix_count + 1))
        )
        if prefix_is_valid:
            tests = cached_tests
            print(
                "Reusing %d cached SDRE tests and computing %d new tests."
                % (prefix_count, len(initial_conditions) - prefix_count)
            )

    system = AllenCahnSystem(final_time=final_time, time_step=time_step)
    print(
        "\nComputing the missing SDRE baselines for %d shared test initial states..."
        % len(initial_conditions)
    )
    for test_index in range(len(tests) + 1, len(initial_conditions) + 1):
        initial_state = initial_conditions[test_index - 1]
        started = time.time()
        result = system.simulate(initial_state, system.sdre_control)
        if not result["success"]:
            raise RuntimeError(
                "SDRE baseline failed for test %d: %s"
                % (test_index, result["failure_reason"])
            )
        row = {"test_index": test_index, **result}
        tests.append(row)
        print(
            "  SDRE test %d: J=%.6f, terminal RMS=%.6f (%.1fs)"
            % (
                test_index,
                result["cost"],
                result["terminal_rms"],
                time.time() - started,
            )
        )
    _write_json(
        SDRE_CACHE_FILE,
        {
            "test_initial_hash": expected_hash,
            "fourier_test_seed": FOURIER_TEST_SEED,
            "final_time": final_time,
            "time_step": time_step,
            "tests": tests,
        },
    )
    return tests


def load_trained_model(checkpoint_path: Path, device: torch.device) -> QRNet:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Missing trained model: %s. Run the train stage first." % checkpoint_path
        )
    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False
        )
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model = QRNet(STATE_DIM).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_trained_policy_model(
    checkpoint_path: Path, device: torch.device
) -> PolicyNetwork:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Missing trained PolicyNet model: %s. Run the train stage first."
            % checkpoint_path
        )
    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Invalid PolicyNet checkpoint: %s" % checkpoint_path)
    config = checkpoint.get("model_config", {})
    model = PolicyNetwork(
        input_dim=int(config.get("input_dim", STATE_DIM)),
        control_dim=int(config.get("control_dim", CONTROL_DIM)),
        hidden_dim=int(config.get("hidden_dim", POLICY_HIDDEN_DIM)),
        num_hidden_layers=int(
            config.get("num_hidden_layers", POLICY_HIDDEN_LAYERS)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_nn_policy(
    model: QRNet, device: torch.device
) -> Callable[[np.ndarray], np.ndarray]:
    b_matrix = torch.as_tensor(build_b_numpy(), dtype=torch.float32, device=device)

    def policy(state: np.ndarray) -> np.ndarray:
        with torch.enable_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            state_tensor.requires_grad_(True)
            value = model(state_tensor)
            gradient = torch.autograd.grad(value.sum(), state_tensor)[0]
            control = -0.5 * R_INV * (gradient @ b_matrix)
        return np.array([float(control.detach().cpu().item())], dtype=np.float64)

    return policy


def make_direct_policy(
    model: PolicyNetwork, device: torch.device
) -> Callable[[np.ndarray], np.ndarray]:
    def policy(state: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            control = model(state_tensor)
        return np.asarray(control.squeeze(0).cpu().numpy(), dtype=np.float64).reshape(1)

    return policy


EVALUATION_FIELDS = (
    "noise_ratio",
    "noise_seed",
    "method",
    "test_index",
    "simulation_success",
    "J_controller",
    "J_sdre",
    "normalized_cost",
    "relative_gap_percent",
    "initial_rms",
    "terminal_rms",
    "max_abs_state",
    "completed_steps",
    "failure_reason",
)


def evaluate_one_model(
    ratio: str,
    noise_seed: int,
    method: str,
    device: torch.device,
    initial_conditions: np.ndarray,
    sdre_tests: Sequence[Dict[str, object]],
    final_time: float,
    time_step: float,
    overwrite: bool,
) -> Path:
    run_dir = model_run_dir(ratio, noise_seed, method)
    result_path = run_dir / "evaluation.csv"
    if len(sdre_tests) != len(initial_conditions):
        raise ValueError("SDRE baseline count does not match the Fourier test set")

    existing_by_test: Dict[int, Dict[str, object]] = {}
    if result_path.exists() and not overwrite:
        for row in _read_csv(result_path):
            test_index = int(row["test_index"])
            if test_index < 1 or test_index > len(initial_conditions):
                continue
            if (
                not math.isclose(float(row["noise_ratio"]), float(ratio))
                or int(row["noise_seed"]) != noise_seed
                or row["method"] != METHOD_LABELS[method]
            ):
                raise ValueError("Existing evaluation metadata is inconsistent: %s" % result_path)
            existing_by_test[test_index] = row

    missing_test_indices = [
        index
        for index in range(1, len(initial_conditions) + 1)
        if index not in existing_by_test
    ]
    if not missing_test_indices:
        print(
            "All %d evaluations already exist, skipping: %s"
            % (len(initial_conditions), result_path)
        )
        return result_path

    checkpoint_path = model_checkpoint_path(ratio, noise_seed, method)
    if method == "policy_net":
        policy_model = load_trained_policy_model(checkpoint_path, device)
        policy = make_direct_policy(policy_model, device)
    else:
        value_model = load_trained_model(checkpoint_path, device)
        policy = make_nn_policy(value_model, device)
    system = AllenCahnSystem(final_time=final_time, time_step=time_step)
    rows: List[Dict[str, object]] = list(existing_by_test.values())
    print(
        "\nEvaluating %s | ratio=%s noise_seed=%d | missing tests=%s"
        % (METHOD_LABELS[method], ratio, noise_seed, missing_test_indices)
    )
    for test_index in missing_test_indices:
        initial_state = initial_conditions[test_index - 1]
        sdre_result = sdre_tests[test_index - 1]
        result = system.simulate(initial_state, policy)
        sdre_cost = float(sdre_result["cost"])
        controller_cost = float(result["cost"])
        if result["success"] and np.isfinite(controller_cost):
            normalized = controller_cost / sdre_cost
            gap = 100.0 * (controller_cost - sdre_cost) / sdre_cost
        else:
            normalized = float("nan")
            gap = float("nan")
        row = {
            "noise_ratio": float(ratio),
            "noise_seed": noise_seed,
            "method": METHOD_LABELS[method],
            "test_index": test_index,
            "simulation_success": int(bool(result["success"])),
            "J_controller": controller_cost,
            "J_sdre": sdre_cost,
            "normalized_cost": normalized,
            "relative_gap_percent": gap,
            "initial_rms": result["initial_rms"],
            "terminal_rms": result["terminal_rms"],
            "max_abs_state": result["max_abs_state"],
            "completed_steps": result["completed_steps"],
            "failure_reason": result["failure_reason"],
        }
        rows.append(row)
        print(
            "  test %d: success=%s J=%s SDRE=%.6f normalized=%s"
            % (
                test_index,
                bool(result["success"]),
                "%.6f" % controller_cost if np.isfinite(controller_cost) else "NaN",
                sdre_cost,
                "%.6f" % normalized if np.isfinite(normalized) else "NaN",
            )
        )
    rows.sort(key=lambda row: int(row["test_index"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return result_path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_values(rows: Sequence[Dict[str, str]], field: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _mean_std(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return float(np.mean(values)), standard_deviation


def summarize_ratio(ratio: str) -> Tuple[Path, Path, Path]:
    all_rows: List[Dict[str, str]] = []
    for method in METHOD_DIRS:
        method_noise_seeds = selected_noise_seeds_for_method(method, NOISE_SEEDS)
        method_test_count = test_count_for_method(method)
        for noise_seed in method_noise_seeds:
            path = model_run_dir(ratio, noise_seed, method) / "evaluation.csv"
            if path.exists():
                all_rows.extend(
                    row
                    for row in _read_csv(path)
                    if int(row["test_index"]) <= method_test_count
                )
    if not all_rows:
        raise FileNotFoundError("No evaluation CSV files found for ratio %s" % ratio)

    output_dir = ROOT / RATIO_DIRS[ratio] / "noise_repeats"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "all_evaluation_cases.csv"
    with cases_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_fields = (
        "noise_ratio",
        "method",
        "noise_realizations",
        "evaluations",
        "successful_evaluations",
        "success_rate",
        "mean_cost",
        "std_cost",
        "mean_normalized_cost",
        "std_normalized_cost",
        "median_normalized_cost",
        "mean_relative_gap_percent",
        "std_relative_gap_percent",
        "mean_seed_averaged_normalized_cost",
        "std_between_noise_realizations",
    )
    summary_rows: List[Dict[str, object]] = []
    for method_label in METHOD_LABELS.values():
        method_rows = [row for row in all_rows if row["method"] == method_label]
        if not method_rows:
            continue
        successes = [row for row in method_rows if int(row["simulation_success"]) == 1]
        costs = _finite_values(successes, "J_controller")
        normalized = _finite_values(successes, "normalized_cost")
        gaps = _finite_values(successes, "relative_gap_percent")
        mean_cost, std_cost = _mean_std(costs)
        mean_normalized, std_normalized = _mean_std(normalized)
        mean_gap, std_gap = _mean_std(gaps)
        seed_means = []
        seeds_present = sorted({int(row["noise_seed"]) for row in method_rows})
        for seed in seeds_present:
            seed_values = _finite_values(
                [row for row in successes if int(row["noise_seed"]) == seed],
                "normalized_cost",
            )
            if seed_values.size:
                seed_means.append(float(np.mean(seed_values)))
        seed_mean, seed_std = _mean_std(np.asarray(seed_means, dtype=np.float64))
        summary_rows.append(
            {
                "noise_ratio": float(ratio),
                "method": method_label,
                "noise_realizations": len(seeds_present),
                "evaluations": len(method_rows),
                "successful_evaluations": len(successes),
                "success_rate": len(successes) / len(method_rows),
                "mean_cost": mean_cost,
                "std_cost": std_cost,
                "mean_normalized_cost": mean_normalized,
                "std_normalized_cost": std_normalized,
                "median_normalized_cost": (
                    float(np.median(normalized)) if normalized.size else float("nan")
                ),
                "mean_relative_gap_percent": mean_gap,
                "std_relative_gap_percent": std_gap,
                "mean_seed_averaged_normalized_cost": seed_mean,
                "std_between_noise_realizations": seed_std,
            }
        )
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    paired_fields = (
        "noise_ratio",
        "paired_cases",
        "u_hjb_wins",
        "u_hjb_win_rate",
        "mean_normalized_cost_difference_u_hjb_minus_u_only",
        "std_normalized_cost_difference",
    )
    keyed: Dict[Tuple[int, int], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in all_rows:
        if int(row["simulation_success"]) == 1:
            keyed[(int(row["noise_seed"]), int(row["test_index"]))][row["method"]] = row
    differences = []
    for methods in keyed.values():
        if "u+HJB" in methods and "u-only" in methods:
            hjb_value = float(methods["u+HJB"]["normalized_cost"])
            only_value = float(methods["u-only"]["normalized_cost"])
            if np.isfinite(hjb_value) and np.isfinite(only_value):
                differences.append(hjb_value - only_value)
    differences_array = np.asarray(differences, dtype=np.float64)
    difference_mean, difference_std = _mean_std(differences_array)
    paired_row = {
        "noise_ratio": float(ratio),
        "paired_cases": len(differences),
        "u_hjb_wins": int(np.sum(differences_array < 0.0)),
        "u_hjb_win_rate": (
            float(np.mean(differences_array < 0.0)) if differences else float("nan")
        ),
        "mean_normalized_cost_difference_u_hjb_minus_u_only": difference_mean,
        "std_normalized_cost_difference": difference_std,
    }
    paired_path = output_dir / "paired_summary.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerow(paired_row)

    pairwise_fields = (
        "noise_ratio",
        "method_a",
        "method_b",
        "paired_cases",
        "method_a_wins",
        "method_a_win_rate",
        "mean_normalized_cost_difference_a_minus_b",
        "std_normalized_cost_difference",
    )
    pairwise_rows: List[Dict[str, object]] = []
    comparisons = (
        ("u+HJB", "u-only"),
        ("u+HJB", "PolicyNet-supervised"),
        ("u-only", "PolicyNet-supervised"),
    )
    for method_a, method_b in comparisons:
        pair_differences = []
        for methods in keyed.values():
            if method_a in methods and method_b in methods:
                value_a = float(methods[method_a]["normalized_cost"])
                value_b = float(methods[method_b]["normalized_cost"])
                if np.isfinite(value_a) and np.isfinite(value_b):
                    pair_differences.append(value_a - value_b)
        pair_array = np.asarray(pair_differences, dtype=np.float64)
        pair_mean, pair_std = _mean_std(pair_array)
        pairwise_rows.append(
            {
                "noise_ratio": float(ratio),
                "method_a": method_a,
                "method_b": method_b,
                "paired_cases": len(pair_differences),
                "method_a_wins": int(np.sum(pair_array < 0.0)),
                "method_a_win_rate": (
                    float(np.mean(pair_array < 0.0))
                    if pair_differences
                    else float("nan")
                ),
                "mean_normalized_cost_difference_a_minus_b": pair_mean,
                "std_normalized_cost_difference": pair_std,
            }
        )
    pairwise_path = output_dir / "pairwise_summary.csv"
    with pairwise_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pairwise_fields)
        writer.writeheader()
        writer.writerows(pairwise_rows)
    print("Summary: %s" % summary_path)
    print("Paired summary: %s" % paired_path)
    print("All-method pairwise summary: %s" % pairwise_path)
    return cases_path, summary_path, paired_path


def summarize_all_ratios(ratios: Sequence[str]) -> Optional[Path]:
    rows: List[Dict[str, str]] = []
    for ratio in ratios:
        path = ROOT / RATIO_DIRS[ratio] / "noise_repeats" / "summary.csv"
        if path.exists():
            rows.extend(_read_csv(path))
    if not rows:
        return None
    path = PROTOCOL_DIR / "all_ratios_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("All-ratio summary: %s" % path)
    return path


def run_self_check(device: torch.device) -> None:
    """Check all three objectives on eight in-memory states, without I/O or ODEs."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.25, 0.25, (8, STATE_DIM)).astype(np.float32)
    u = rng.normal(0.0, 0.1, (8, CONTROL_DIM)).astype(np.float32)
    mean, std = np.zeros(STATE_DIM, dtype=np.float32), np.ones(STATE_DIM, dtype=np.float32)
    a_base, b_matrix = torch_system_matrices(device)
    for method in METHOD_DIRS:
        set_global_seed(TRAIN_SEED)
        states = torch.as_tensor(x, device=device).requires_grad_(method != "policy_net")
        labels = torch.as_tensor(u, device=device)
        if method == "policy_net":
            model = PolicyNetwork(STATE_DIM, CONTROL_DIM, x_mean=mean, x_std=std).to(device)
            prediction, residual = model(states), None
        else:
            model = QRNet(STATE_DIM, mean, std).to(device)
            prediction, residual = model_quantities(model, states, a_base, b_matrix,
                                                    need_hjb=(method == "u_hjb"), create_graph=True)
        loss = (prediction - labels).square().mean()
        if residual is not None:
            loss = loss + residual.square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError(f"Nonfinite objective or gradient for {method}.")
        print(f"AC {method}: in-memory objective/gradient check passed")
    initial = generate_fourier_initial_conditions()
    if initial.shape != (5, STATE_DIM) or not np.isfinite(initial).all():
        raise AssertionError("Invalid fixed test initial conditions.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paired Allen--Cahn noisy-control-label experiment."
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=normalize_ratio,
        default=list(RATIO_DIRS.keys()),
        help="Noise ratios (default: 0.1 0.25 0.5 0.75).",
    )
    parser.add_argument(
        "--noise-indices",
        nargs="+",
        type=int,
        default=list(range(len(NOISE_SEEDS))),
        help="Indices 0..4 selecting noise seeds 2026..2030.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_DIRS.keys()),
        default=list(METHOD_DIRS.keys()),
        help="Methods to train/evaluate (default: all three).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("prepare", "train", "evaluate", "summarize", "check"),
        default=("check",),
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--hjb-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate/retrain/re-evaluate existing repeated-run artifacts.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    invalid_indices = [
        index for index in args.noise_indices if index < 0 or index >= len(NOISE_SEEDS)
    ]
    if invalid_indices:
        raise ValueError(
            "Noise indices must lie in 0..%d: %s"
            % (len(NOISE_SEEDS) - 1, invalid_indices)
        )
    if args.epochs <= 0 or args.batch_size <= 0 or args.log_interval <= 0:
        raise ValueError("epochs, batch size and log interval must be positive")
    if args.learning_rate <= 0.0 or args.hjb_weight < 0.0:
        raise ValueError("learning rate must be positive and HJB weight non-negative")
    if args.final_time <= 0.0 or args.time_step <= 0.0:
        raise ValueError("final time and time step must be positive")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    ratios = list(dict.fromkeys(args.ratios))
    noise_seeds = [NOISE_SEEDS[index] for index in dict.fromkeys(args.noise_indices)]
    methods = list(dict.fromkeys(args.methods))
    device = select_device(args.device)

    if "check" in args.stages:
        run_self_check(device)

    needs_data = any(stage in args.stages for stage in ("prepare", "train"))
    clean: Optional[Dict[str, np.ndarray]] = None
    if needs_data:
        clean = prepare_fixed_clean_data(overwrite=args.overwrite and "prepare" in args.stages)
        prepare_test_initial_conditions(overwrite=args.overwrite and "prepare" in args.stages)
        for ratio in ratios:
            for noise_seed in noise_seeds:
                prepare_noisy_dataset(
                    clean,
                    ratio,
                    noise_seed,
                    overwrite=args.overwrite and "prepare" in args.stages,
                )

    if "train" in args.stages:
        for ratio in ratios:
            for method in methods:
                method_noise_seeds = selected_noise_seeds_for_method(
                    method, noise_seeds
                )
                for noise_seed in method_noise_seeds:
                    if method == "policy_net":
                        train_one_policy_model(
                            ratio=ratio,
                            noise_seed=noise_seed,
                            device=device,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            log_interval=args.log_interval,
                            overwrite=args.overwrite,
                        )
                    else:
                        train_one_model(
                            ratio=ratio,
                            noise_seed=noise_seed,
                            method=method,
                            device=device,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            hjb_weight=args.hjb_weight,
                            log_interval=args.log_interval,
                            overwrite=args.overwrite,
                        )

    if "evaluate" in args.stages:
        initial_conditions = prepare_test_initial_conditions(overwrite=False)
        sdre_tests = get_sdre_baseline(
            initial_conditions,
            final_time=args.final_time,
            time_step=args.time_step,
            overwrite=args.overwrite,
        )
        for ratio in ratios:
            for method in methods:
                method_noise_seeds = selected_noise_seeds_for_method(
                    method, noise_seeds
                )
                method_test_count = test_count_for_method(method)
                method_initial_conditions = initial_conditions[:method_test_count]
                method_sdre_tests = sdre_tests[:method_test_count]
                for noise_seed in method_noise_seeds:
                    evaluate_one_model(
                        ratio=ratio,
                        noise_seed=noise_seed,
                        method=method,
                        device=device,
                        initial_conditions=method_initial_conditions,
                        sdre_tests=method_sdre_tests,
                        final_time=args.final_time,
                        time_step=args.time_step,
                        overwrite=args.overwrite,
                    )

    if "summarize" in args.stages:
        for ratio in ratios:
            try:
                summarize_ratio(ratio)
            except FileNotFoundError as exc:
                print("Summary skipped: %s" % exc)
        # Scan every ratio directory so that running one ratio at a time builds
        # an accumulating global table instead of overwriting it with the last ratio.
        summarize_all_ratios(list(RATIO_DIRS.keys()))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)

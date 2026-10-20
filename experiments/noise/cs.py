"""Cucker--Smale paired control-label noise experiment.

Inputs are read from data/ and generated artifacts are written under
outputs/noise/cs/. Training and evaluation use explicitly selected stages.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
import math
import random
import re
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
ROOT = PACKAGE_ROOT / "outputs" / "noise" / "cs"


def release_path(path: Path) -> str:
    """Serialize provenance without an installation-specific absolute path."""
    return Path(path).resolve().relative_to(PACKAGE_ROOT.resolve()).as_posix()

NA = 20
STATE_DIM = 2 * NA
CONTROL_DIM = NA
Q_SCALE = 1.0 / NA
R_SCALE = 1.0 / NA
R_INV_SCALE = float(NA)

RATIO_DIRS = {
    "0.1": "0.1ratio",
    "0.25": "0.25ratio",
    "0.5": "0.5ratio",
    "0.75": "0.75ratio",
}
METHOD_DIRS = {
    "hjb": "example1_multi_agent",
    "u_only": "example1_multi_agent - no-HJB",
    "policy_net": "Multi-agent_PolicyNet_supervised",
}
METHOD_LABELS = {
    "hjb": "u+HJB (Ours)",
    "u_only": "u-only value network",
    "policy_net": "PolicyNet-supervised",
}

NOISE_SEEDS = tuple(range(2026, 2031))
TRAIN_SEED = 42
SPLIT_SEED = 42
TEST_INITIAL_SEED = 20260728
TEST_INITIAL_COUNT = 5
TEST_FRACTION = 0.2
DEFAULT_EPOCHS = 4000
DEFAULT_BATCH_SIZE = 4096
DEFAULT_LEARNING_RATE = 8e-4
VALUE_HIDDEN_DIM = 128
U_ONLY_HIDDEN_DIM = 256
VALUE_NUM_HIDDEN_LAYERS = 2
POLICY_HIDDEN_DIM = 512
POLICY_NUM_HIDDEN_LAYERS = 3
SCHEDULER_STEP_SIZE = 200
SCHEDULER_GAMMA = 0.8
DEFAULT_HORIZON = 15.0
DEFAULT_DT = 0.01
CHECKPOINT_INTERVAL = 50

PROTOCOL_NAME = "paired_noise_5x5"
PROTOCOL_DIR = ROOT / "protocol"
TEST_INITIALS_FILE = PROTOCOL_DIR / "test_initial_conditions.mat"
SDRE_BASELINE_FILE = PROTOCOL_DIR / "sdre_baseline.json"
ALL_RATIOS_SUMMARY_FILE = PROTOCOL_DIR / "all_ratios_summary.csv"
EVALUATION_PROTOCOLS_DIR = ROOT / "evaluations"

CLEAN_DATA_FILE = PACKAGE_ROOT / "data" / "cs_noise.mat"


def normalize_ratio(value: str) -> str:
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid noise ratio: {value}") from exc
    for ratio in RATIO_DIRS:
        if math.isclose(numeric, float(ratio), rel_tol=0.0, abs_tol=1e-12):
            return ratio
    allowed = ", ".join(RATIO_DIRS)
    raise argparse.ArgumentTypeError(f"Noise ratio must be one of: {allowed}")


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or pass --device cpu."
        )
    device = torch.device(requested)
    if device.type == "cuda":
        torch.cuda.init()
        index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"Device: {device} ({torch.cuda.get_device_name(index)})")
    else:
        print(f"Device: {device}")
    return device


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _as_rows(array: np.ndarray, width: int, name: str, n: Optional[int] = None) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 1:
        value = value.reshape(-1, width)
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {value.shape}.")
    if value.shape[1] != width and value.shape[0] == width:
        value = value.T
    if value.shape[1] != width:
        raise ValueError(f"Expected {name} with width {width}, got {value.shape}.")
    if n is not None and value.shape[0] != n:
        raise ValueError(f"Expected {name} with {n} rows, got {value.shape}.")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return value


def load_clean_dataset(path: Path = CLEAN_DATA_FILE) -> Dict[str, np.ndarray | str]:
    if not path.exists():
        raise FileNotFoundError(f"Fixed clean expert data not found: {path}")
    data = sio.loadmat(path)
    x = _as_rows(data["X"], STATE_DIM, "X")
    n = len(x)
    dv = _as_rows(data["DV_all"], STATE_DIM, "DV_all", n)
    u = (
        _as_rows(data["U_all"], CONTROL_DIM, "U_all", n)
        if "U_all" in data
        else -0.5 * R_INV_SCALE * dv[:, NA:]
    )
    value = np.asarray(data.get("V_all", np.zeros((n, 1))), dtype=np.float64)
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    if value.shape[0] != n and value.shape[1] == n:
        value = value.T
    if value.shape[0] != n or not np.isfinite(value).all():
        raise ValueError(f"Invalid V_all shape or entries: {value.shape}")

    # For B=[0;I] and R^{-1}=20I, the clean SDRE relation is
    # u=-0.5 R^{-1} B^T DV=-10*DV_velocity.  Refuse a noisy source.
    u_from_dv = -0.5 * R_INV_SCALE * dv[:, NA:]
    relation_error = float(np.max(np.abs(u - u_from_dv)))
    tolerance = 1e-7 + 1e-6 * float(np.max(np.abs(u_from_dv)))
    if relation_error > tolerance:
        raise ValueError(
            "The configured expert source is not clean: max |U+10*DV_v|="
            f"{relation_error:.6g}. Expected <= {tolerance:.6g}."
        )

    clean_hash = array_hash(x, u)
    return {
        "X": x,
        "V_all": value,
        "DV_all": dv,
        "U_clean": u,
        "clean_hash": clean_hash,
    }


def shared_run_dir(ratio: str, noise_seed: int) -> Path:
    return (
        ROOT
        / RATIO_DIRS[ratio]
        / PROTOCOL_NAME
        / f"noise_seed_{noise_seed}"
    )


def noisy_data_path(ratio: str, noise_seed: int) -> Path:
    return shared_run_dir(ratio, noise_seed) / "multi_agent_training_data.mat"


def method_run_dir(ratio: str, noise_seed: int, method: str) -> Path:
    return (
        ROOT
        / RATIO_DIRS[ratio]
        / METHOD_DIRS[method]
        / PROTOCOL_NAME
        / f"noise_seed_{noise_seed}"
    )


def final_checkpoint_path(ratio: str, noise_seed: int, method: str) -> Path:
    filename = "policynet_supervised.pth" if method == "policy_net" else "controller_model.pth"
    return method_run_dir(ratio, noise_seed, method) / filename


def last_checkpoint_path(ratio: str, noise_seed: int, method: str) -> Path:
    return method_run_dir(ratio, noise_seed, method) / "checkpoint_last.pth"


def evaluation_protocol_dir(evaluation_tag: Optional[str]) -> Path:
    if evaluation_tag is None:
        return PROTOCOL_DIR
    return EVALUATION_PROTOCOLS_DIR / evaluation_tag


def evaluation_result_path(
    ratio: str,
    noise_seed: int,
    method: str,
    evaluation_tag: Optional[str],
) -> Path:
    if evaluation_tag is None:
        return method_run_dir(ratio, noise_seed, method) / "evaluation.csv"
    return (
        evaluation_protocol_dir(evaluation_tag)
        / RATIO_DIRS[ratio]
        / METHOD_DIRS[method]
        / f"noise_seed_{noise_seed}"
        / "evaluation.csv"
    )


def ratio_summary_dir(ratio: str, evaluation_tag: Optional[str]) -> Path:
    if evaluation_tag is None:
        return ROOT / RATIO_DIRS[ratio] / PROTOCOL_NAME
    return evaluation_protocol_dir(evaluation_tag) / RATIO_DIRS[ratio]


def prepare_noisy_dataset(
    ratio: str,
    noise_seed: int,
    clean: Dict[str, np.ndarray | str],
    overwrite: bool = False,
) -> Path:
    path = noisy_data_path(ratio, noise_seed)
    expected_hash = str(clean["clean_hash"])
    if path.exists() and not overwrite:
        existing = sio.loadmat(path)
        saved_ratio = float(np.asarray(existing["noise_ratio"]).reshape(-1)[0])
        saved_seed = int(np.asarray(existing["noise_seed"]).reshape(-1)[0])
        saved_hash_raw = np.asarray(existing["clean_hash"]).reshape(-1)[0]
        saved_hash = str(saved_hash_raw).strip()
        if (
            not math.isclose(saved_ratio, float(ratio), abs_tol=1e-12)
            or saved_seed != noise_seed
            or saved_hash != expected_hash
        ):
            raise ValueError(
                f"Existing noisy data metadata is inconsistent: {path}. "
                "Use --overwrite only if replacement is intended."
            )
        return path

    x = np.asarray(clean["X"], dtype=np.float64)
    value = np.asarray(clean["V_all"], dtype=np.float64)
    dv = np.asarray(clean["DV_all"], dtype=np.float64)
    u_clean = np.asarray(clean["U_clean"], dtype=np.float64)
    u_scale = float(np.std(u_clean, ddof=1))
    rng = np.random.default_rng(noise_seed)
    standardized_noise = rng.standard_normal(u_clean.shape)
    u_noise = float(ratio) * u_scale * standardized_noise
    u_noisy = u_clean + u_noise

    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        path,
        {
            "X": x,
            "V_all": value,
            "DV_all": dv,
            "U_all": u_noisy,
            "U_clean": u_clean,
            "U_noise": u_noise,
            "noise_ratio": np.array([[float(ratio)]], dtype=np.float64),
            "noise_seed": np.array([[noise_seed]], dtype=np.int64),
            "u_clean_std": np.array([[u_scale]], dtype=np.float64),
            "clean_hash": expected_hash,
        },
        do_compression=True,
    )
    actual_ratio = float(np.std(u_noise, ddof=1) / u_scale)
    print(
        f"Prepared ratio={ratio}, seed={noise_seed}: {path} "
        f"(realized noise/std={actual_ratio:.6f})"
    )
    return path


def load_noisy_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    data = sio.loadmat(path)
    x = _as_rows(data["X"], STATE_DIM, "X").astype(np.float32)
    u = _as_rows(data["U_all"], CONTROL_DIM, "U_all", len(x)).astype(np.float32)
    metadata: Dict[str, object] = {
        "noise_ratio": float(np.asarray(data["noise_ratio"]).reshape(-1)[0]),
        "noise_seed": int(np.asarray(data["noise_seed"]).reshape(-1)[0]),
        "clean_hash": str(np.asarray(data["clean_hash"]).reshape(-1)[0]).strip(),
    }
    return x, u, metadata


def fixed_split_indices(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    n_test = int(math.ceil(TEST_FRACTION * n_samples))
    permutation = np.random.RandomState(SPLIT_SEED).permutation(n_samples)
    test_indices = permutation[:n_test]
    train_indices = permutation[n_test:]
    return train_indices, test_indices


class ValueNetwork(nn.Module):
    def __init__(
        self,
        x_mean: Optional[np.ndarray | torch.Tensor] = None,
        x_std: Optional[np.ndarray | torch.Tensor] = None,
        input_dim: int = STATE_DIM,
        hidden_dim: int = VALUE_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if x_mean is None:
            x_mean = torch.zeros(input_dim)
        if x_std is None:
            x_std = torch.ones(input_dim)
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32).clone())
        self.register_buffer("x_std", torch.as_tensor(x_std, dtype=torch.float32).clone())
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()

    def _raw_forward(self, normalized_state: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.fc1(normalized_state))
        hidden = self.activation(self.fc2(hidden))
        return self.out(hidden)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        normalized = (state - self.x_mean) / (self.x_std + 1e-8)
        zero = torch.zeros((1, STATE_DIM), dtype=state.dtype, device=state.device)
        normalized_zero = (zero - self.x_mean) / (self.x_std + 1e-8)
        shifted = self._raw_forward(normalized) - self._raw_forward(normalized_zero)
        positive = torch.relu(shifted)
        # Forward value is nonnegative, while the straight-through gradient is
        # the gradient of the shifted smooth network for feedback and residuals.
        return shifted + (positive - shifted).detach()


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        x_mean: Optional[np.ndarray | torch.Tensor] = None,
        x_std: Optional[np.ndarray | torch.Tensor] = None,
        input_dim: int = STATE_DIM,
        control_dim: int = CONTROL_DIM,
        hidden_dim: int = POLICY_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if x_mean is None:
            x_mean = torch.zeros(input_dim)
        if x_std is None:
            x_std = torch.ones(input_dim)
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32).clone())
        self.register_buffer("x_std", torch.as_tensor(x_std, dtype=torch.float32).clone())
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, control_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        normalized = (state - self.x_mean) / (self.x_std + 1e-8)
        return self.network(normalized)


def hidden_dim_for_method(method: str) -> int:
    if method == "policy_net":
        return POLICY_HIDDEN_DIM
    if method == "u_only":
        return U_ONLY_HIDDEN_DIM
    return VALUE_HIDDEN_DIM


def hidden_layers_for_method(method: str) -> int:
    return POLICY_NUM_HIDDEN_LAYERS if method == "policy_net" else VALUE_NUM_HIDDEN_LAYERS


def build_model(
    method: str,
    x_mean: Optional[np.ndarray | torch.Tensor] = None,
    x_std: Optional[np.ndarray | torch.Tensor] = None,
    hidden_dim: Optional[int] = None,
) -> nn.Module:
    width = hidden_dim_for_method(method) if hidden_dim is None else int(hidden_dim)
    if method == "policy_net":
        return PolicyNetwork(x_mean=x_mean, x_std=x_std, hidden_dim=width)
    return ValueNetwork(x_mean=x_mean, x_std=x_std, hidden_dim=width)


def drift_torch(state: torch.Tensor) -> torch.Tensor:
    positions = state[:, :NA]
    velocities = state[:, NA:]
    differences = positions.unsqueeze(2) - positions.unsqueeze(1)
    interactions = 1.0 / (1.0 + differences.square())
    eye = torch.eye(NA, dtype=torch.bool, device=state.device).unsqueeze(0)
    interactions = interactions.masked_fill(eye, 0.0)
    aligned_velocity = (
        torch.bmm(interactions, velocities.unsqueeze(2)).squeeze(2)
        - interactions.sum(dim=2) * velocities
    ) / NA
    return torch.cat((velocities, aligned_velocity), dim=1)


def value_quantities(
    model: ValueNetwork,
    state: torch.Tensor,
    create_graph: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not state.requires_grad:
        state = state.requires_grad_(True)
    value = model(state)
    gradient = torch.autograd.grad(
        value,
        state,
        grad_outputs=torch.ones_like(value),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    control = -0.5 * R_INV_SCALE * gradient[:, NA:]
    state_cost = Q_SCALE * state.square().sum(dim=1, keepdim=True)
    drift_term = (gradient * drift_torch(state)).sum(dim=1, keepdim=True)
    control_hamiltonian = 0.25 * R_INV_SCALE * gradient[:, NA:].square().sum(
        dim=1, keepdim=True
    )
    hjb = state_cost + drift_term - control_hamiltonian
    return hjb, control, value


def mean_losses(
    model: nn.Module,
    loader: DataLoader,
    method: str,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_u_squared = 0.0
    total_u_elements = 0
    total_hjb_squared = 0.0
    total_hjb_elements = 0
    for batch_x, batch_u in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_u = batch_u.to(device, non_blocking=True)
        if method == "policy_net":
            with torch.no_grad():
                predicted_u = model(batch_x)
            hjb = None
        else:
            batch_x.requires_grad_(True)
            with torch.enable_grad():
                hjb, predicted_u, _ = value_quantities(
                    model, batch_x, create_graph=False
                )
        total_u_squared += float((predicted_u - batch_u).square().sum().item())
        total_u_elements += batch_u.numel()
        if hjb is not None:
            total_hjb_squared += float(hjb.square().sum().item())
            total_hjb_elements += hjb.numel()
    u_mse = total_u_squared / max(total_u_elements, 1)
    hjb_mse = (
        total_hjb_squared / max(total_hjb_elements, 1)
        if total_hjb_elements
        else float("nan")
    )
    alpha = 1.0 if method == "hjb" else 0.0
    total = u_mse + (alpha * hjb_mse if np.isfinite(hjb_mse) else 0.0)
    return total, hjb_mse, u_mse


def _checkpoint_metadata_matches(
    checkpoint: Dict[str, object],
    method: str,
    ratio: str,
    noise_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    clean_hash: str,
) -> bool:
    config = checkpoint.get("training_config", {})
    if not isinstance(config, dict):
        return False
    expected_hidden_dim = hidden_dim_for_method(method)
    expected_hidden_layers = hidden_layers_for_method(method)
    return (
        checkpoint.get("method") == method
        and math.isclose(float(config.get("noise_ratio", -1.0)), float(ratio), abs_tol=1e-12)
        and int(config.get("noise_seed", -1)) == noise_seed
        and int(config.get("epochs", -1)) == epochs
        and int(config.get("batch_size", -1)) == batch_size
        and math.isclose(
            float(config.get("learning_rate", -1.0)),
            learning_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and int(config.get("hidden_dim", -1)) == expected_hidden_dim
        and int(config.get("num_hidden_layers", -1)) == expected_hidden_layers
        and int(config.get("scheduler_step_size", -1)) == SCHEDULER_STEP_SIZE
        and math.isclose(
            float(config.get("scheduler_gamma", -1.0)),
            SCHEDULER_GAMMA,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and config.get("clean_hash") == clean_hash
    )


def _load_torch_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint: {path}")
    return checkpoint


def train_one_model(
    ratio: str,
    noise_seed: int,
    method: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    overwrite: bool,
) -> Path:
    data_path = noisy_data_path(ratio, noise_seed)
    x, u, data_metadata = load_noisy_dataset(data_path)
    clean_hash = str(data_metadata["clean_hash"])
    output_dir = method_run_dir(ratio, noise_seed, method)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_checkpoint_path(ratio, noise_seed, method)
    last_path = last_checkpoint_path(ratio, noise_seed, method)

    if final_path.exists() and not overwrite:
        existing = _load_torch_checkpoint(final_path, torch.device("cpu"))
        if not _checkpoint_metadata_matches(
            existing,
            method,
            ratio,
            noise_seed,
            epochs,
            batch_size,
            learning_rate,
            clean_hash,
        ):
            raise ValueError(
                f"Existing final checkpoint has different metadata: {final_path}. "
                "Use --overwrite only if replacement is intended."
            )
        print(f"Training already complete, skipping: {final_path}")
        return final_path

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    train_indices, test_indices = fixed_split_indices(len(x))
    x_train, u_train = x[train_indices], u[train_indices]
    x_test, u_test = x[test_indices], u[test_indices]

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(u_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(u_test))
    loader_generator = torch.Generator().manual_seed(TRAIN_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    set_global_seed(TRAIN_SEED)
    model = build_model(method, x_mean=x_mean, x_std=x_std).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=SCHEDULER_STEP_SIZE,
        gamma=SCHEDULER_GAMMA,
    )
    start_epoch = 0
    history: List[Dict[str, float | int]] = []

    if last_path.exists() and not overwrite:
        resume = _load_torch_checkpoint(last_path, device)
        if not _checkpoint_metadata_matches(
            resume,
            method,
            ratio,
            noise_seed,
            epochs,
            batch_size,
            learning_rate,
            clean_hash,
        ):
            raise ValueError(
                f"Existing resume checkpoint has different metadata: {last_path}."
            )
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch = int(resume["epoch"])
        history = list(resume.get("history", []))
        generator_state = resume.get("loader_generator_state")
        if generator_state is not None:
            loader_generator.set_state(generator_state)
        print(f"Resuming {METHOD_LABELS[method]} at epoch {start_epoch + 1}")

    training_config: Dict[str, object] = {
        "noise_ratio": float(ratio),
        "noise_seed": noise_seed,
        "train_seed": TRAIN_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim_for_method(method),
        "num_hidden_layers": hidden_layers_for_method(method),
        "scheduler_step_size": SCHEDULER_STEP_SIZE,
        "scheduler_gamma": SCHEDULER_GAMMA,
        "test_fraction": TEST_FRACTION,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "clean_hash": clean_hash,
        "data_file": release_path(data_path),
        "device": str(device),
    }
    alpha = 1.0 if method == "hjb" else 0.0
    criterion = nn.MSELoss(reduction="mean")

    print(
        f"\nTraining {METHOD_LABELS[method]} | ratio={ratio} | "
        f"noise_seed={noise_seed} | epochs={epochs} | device={device}"
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        sum_total = 0.0
        sum_hjb = 0.0
        sum_u = 0.0
        batches = 0
        for batch_x, batch_u in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_u = batch_u.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if method == "policy_net":
                predicted_u = model(batch_x)
                loss_u = criterion(predicted_u, batch_u)
                loss_hjb = torch.zeros((), device=device)
            else:
                batch_x.requires_grad_(True)
                hjb, predicted_u, _ = value_quantities(
                    model, batch_x, create_graph=True
                )
                loss_u = criterion(predicted_u, batch_u)
                loss_hjb = criterion(hjb, torch.zeros_like(hjb))
            total_loss = loss_u + alpha * loss_hjb
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite loss for {method}, ratio={ratio}, "
                    f"seed={noise_seed}, epoch={epoch + 1}."
                )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            sum_total += float(total_loss.item())
            sum_hjb += float(loss_hjb.item())
            sum_u += float(loss_u.item())
            batches += 1
        scheduler.step()

        train_total = sum_total / max(batches, 1)
        train_hjb = sum_hjb / max(batches, 1)
        train_u = sum_u / max(batches, 1)
        test_total, test_hjb, test_u = mean_losses(model, test_loader, method, device)
        row: Dict[str, float | int] = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_total": train_total,
            "train_hjb": train_hjb,
            "train_u": train_u,
            "test_total": test_total,
            "test_hjb": test_hjb,
            "test_u": test_u,
        }
        history.append(row)

        should_report = epoch == start_epoch or (epoch + 1) % 50 == 0 or epoch + 1 == epochs
        if should_report:
            print(
                f"Epoch {epoch + 1:04d}/{epochs}: "
                f"train={train_total:.6e}, HJB={train_hjb:.6e}, U={train_u:.6e}, "
                f"test={test_total:.6e}"
            )
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0 and epoch + 1 < epochs:
            torch.save(
                {
                    "method": method,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "history": history,
                    "loader_generator_state": loader_generator.get_state(),
                    "training_config": training_config,
                },
                last_path,
            )

    checkpoint = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": STATE_DIM,
            "control_dim": CONTROL_DIM,
            "hidden_dim": hidden_dim_for_method(method),
            "num_hidden_layers": hidden_layers_for_method(method),
        },
        "training_config": training_config,
        "history": history,
    }
    torch.save(checkpoint, final_path)
    history_path = output_dir / "training_history.csv"
    if history:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    summary = {
        "checkpoint": release_path(final_path),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "training_config": training_config,
        "final_losses": history[-1] if history else {},
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved checkpoint: {final_path}")
    return final_path


def generate_test_initial_conditions(
    count: int = TEST_INITIAL_COUNT,
    seed: int = TEST_INITIAL_SEED,
) -> np.ndarray:
    # Exact expert initial-state family: positions and velocities are sampled
    # independently and uniformly from [-1, 1].
    rng = np.random.RandomState(seed)
    initial = np.zeros((count, STATE_DIM), dtype=np.float64)
    initial[:, :NA] = rng.uniform(-1.0, 1.0, size=(count, NA))
    initial[:, NA:] = rng.uniform(-1.0, 1.0, size=(count, NA))
    return initial


def prepare_test_initial_conditions(
    test_seed: int = TEST_INITIAL_SEED,
    test_count: int = TEST_INITIAL_COUNT,
    protocol_dir: Path = PROTOCOL_DIR,
    overwrite: bool = False,
) -> np.ndarray:
    test_initials_file = protocol_dir / "test_initial_conditions.mat"
    expected = generate_test_initial_conditions(count=test_count, seed=test_seed)
    if test_initials_file.exists() and not overwrite:
        cached = sio.loadmat(test_initials_file)
        actual = _as_rows(cached["X0_test"], STATE_DIM, "X0_test")
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                f"Cached test initial conditions do not match the fixed protocol: "
                f"{test_initials_file}. Use a different --evaluation-tag rather "
                "than replacing an existing test protocol."
            )
        return actual
    protocol_dir.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        test_initials_file,
        {
            "X0_test": expected,
            "test_seed": np.array([[test_seed]], dtype=np.int64),
            "test_count": np.array([[test_count]], dtype=np.int64),
            "position_range": np.array([[-1.0, 1.0]]),
            "velocity_range": np.array([[-1.0, 1.0]]),
            "initial_hash": array_hash(expected),
        },
    )
    print(f"Created {len(expected)} shared test initial conditions: {test_initials_file}")
    return expected


class CuckerSmaleSystem:
    def __init__(self, horizon: float = DEFAULT_HORIZON, dt: float = DEFAULT_DT) -> None:
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.q = Q_SCALE * np.eye(STATE_DIM)
        self.r = R_SCALE * np.eye(CONTROL_DIM)
        self.b = np.vstack((np.zeros((NA, NA)), np.eye(NA)))

    @staticmethod
    def interaction_matrix(positions: np.ndarray) -> np.ndarray:
        differences = positions[:, None] - positions[None, :]
        weights = 1.0 / (1.0 + differences**2)
        np.fill_diagonal(weights, 0.0)
        matrix = weights / NA
        np.fill_diagonal(matrix, -weights.sum(axis=1) / NA)
        return matrix

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        positions = state[:NA]
        velocities = state[NA:]
        acceleration = self.interaction_matrix(positions) @ velocities + control
        return np.concatenate((velocities, acceleration))

    def sdre_control(self, state: np.ndarray) -> np.ndarray:
        small_a = self.interaction_matrix(state[:NA])
        system_a = np.block(
            [[np.zeros((NA, NA)), np.eye(NA)], [np.zeros((NA, NA)), small_a]]
        )
        riccati = solve_continuous_are(system_a, self.b, self.q, self.r)
        return -R_INV_SCALE * self.b.T @ riccati @ state

    def simulate(
        self,
        initial_state: np.ndarray,
        controller: Callable[[np.ndarray], np.ndarray],
    ) -> Dict[str, object]:
        times = np.arange(0.0, self.horizon + 0.5 * self.dt, self.dt)
        states = np.zeros((len(times), STATE_DIM), dtype=np.float64)
        controls = np.zeros((len(times), CONTROL_DIM), dtype=np.float64)
        states[0] = np.asarray(initial_state, dtype=np.float64)
        for step in range(len(times) - 1):
            current_state = states[step]
            current_control = np.asarray(controller(current_state), dtype=np.float64).reshape(
                CONTROL_DIM
            )
            if not np.isfinite(current_control).all():
                raise FloatingPointError(f"Non-finite control at step {step}.")
            controls[step] = current_control
            solution = solve_ivp(
                lambda _time, state: self.dynamics(state, current_control),
                (times[step], times[step + 1]),
                current_state,
                t_eval=[times[step + 1]],
                method="RK45",
            )
            if not solution.success:
                raise RuntimeError(
                    f"ODE solver failed at step {step}: {solution.message}"
                )
            next_state = solution.y[:, -1]
            if not np.isfinite(next_state).all():
                raise FloatingPointError(f"Non-finite state at step {step + 1}.")
            states[step + 1] = next_state
        controls[-1] = np.asarray(controller(states[-1]), dtype=np.float64).reshape(
            CONTROL_DIM
        )
        running_cost = Q_SCALE * np.sum(states**2, axis=1)
        running_cost += R_SCALE * np.sum(controls**2, axis=1)
        cost = float(np.trapezoid(running_cost, times))
        return {
            "cost": cost,
            "terminal_state_norm": float(np.linalg.norm(states[-1])),
            "max_state_norm": float(np.max(np.linalg.norm(states, axis=1))),
            "max_control_norm": float(np.max(np.linalg.norm(controls, axis=1))),
        }


def get_sdre_baseline(
    initial_conditions: np.ndarray,
    system: CuckerSmaleSystem,
    protocol_dir: Path = PROTOCOL_DIR,
    test_seed: int = TEST_INITIAL_SEED,
    overwrite: bool = False,
) -> List[Dict[str, object]]:
    sdre_baseline_file = protocol_dir / "sdre_baseline.json"
    expected_hash = array_hash(initial_conditions)
    if sdre_baseline_file.exists() and not overwrite:
        cached = json.loads(sdre_baseline_file.read_text(encoding="utf-8"))
        if (
            cached.get("initial_hash") == expected_hash
            and math.isclose(float(cached.get("horizon", -1.0)), system.horizon)
            and math.isclose(float(cached.get("dt", -1.0)), system.dt)
            and len(cached.get("tests", [])) == len(initial_conditions)
        ):
            return list(cached["tests"])
        raise ValueError(
            f"Cached SDRE baseline does not match this protocol: {sdre_baseline_file}."
        )

    tests: List[Dict[str, object]] = []
    print(f"\nComputing SDRE baselines for {len(initial_conditions)} shared initial states...")
    for index, initial_state in enumerate(initial_conditions, start=1):
        result = system.simulate(initial_state, system.sdre_control)
        row = {"test_index": index, **result}
        tests.append(row)
        print(f"SDRE test {index}/{len(initial_conditions)}: J={result['cost']:.8f}")
    protocol_dir.mkdir(parents=True, exist_ok=True)
    sdre_baseline_file.write_text(
        json.dumps(
            {
                "initial_hash": expected_hash,
                "test_seed": test_seed,
                "test_count": len(initial_conditions),
                "horizon": system.horizon,
                "dt": system.dt,
                "tests": tests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return tests


def load_model_for_evaluation(
    checkpoint_path: Path,
    method: str,
    device: torch.device,
) -> nn.Module:
    checkpoint = _load_torch_checkpoint(checkpoint_path, device)
    if checkpoint.get("method") != method:
        raise ValueError(f"Wrong method in checkpoint: {checkpoint_path}")
    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, dict):
        model_config = {}
    hidden_dim = int(model_config.get("hidden_dim", hidden_dim_for_method(method)))
    model = build_model(method, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_controller(
    model: nn.Module,
    method: str,
    device: torch.device,
) -> Callable[[np.ndarray], np.ndarray]:
    if method == "policy_net":
        def direct_controller(state: np.ndarray) -> np.ndarray:
            tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                control = model(tensor)
            return control.squeeze(0).detach().cpu().numpy()

        return direct_controller

    def value_controller(state: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor.requires_grad_(True)
        value = model(tensor)
        gradient = torch.autograd.grad(
            value,
            tensor,
            grad_outputs=torch.ones_like(value),
            create_graph=False,
            retain_graph=False,
        )[0]
        control = -0.5 * R_INV_SCALE * gradient[:, NA:]
        return control.squeeze(0).detach().cpu().numpy()

    return value_controller


EVALUATION_FIELDS = (
    "ratio",
    "noise_seed",
    "method",
    "method_label",
    "test_index",
    "success",
    "cost",
    "sdre_cost",
    "normalized_cost",
    "terminal_state_norm",
    "max_state_norm",
    "max_control_norm",
    "error",
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_one_model(
    ratio: str,
    noise_seed: int,
    method: str,
    device: torch.device,
    initial_conditions: np.ndarray,
    sdre_tests: Sequence[Dict[str, object]],
    system: CuckerSmaleSystem,
    evaluation_tag: Optional[str],
    overwrite: bool,
) -> Path:
    result_path = evaluation_result_path(ratio, noise_seed, method, evaluation_tag)
    output_dir = result_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [] if overwrite else _read_csv(result_path)
    completed = {int(row["test_index"]) for row in existing}
    missing = [index for index in range(1, len(initial_conditions) + 1) if index not in completed]
    if not missing:
        print(f"All evaluations already exist, skipping: {result_path}")
        return result_path

    checkpoint_path = final_checkpoint_path(ratio, noise_seed, method)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Train the model first: {checkpoint_path}")
    model = load_model_for_evaluation(checkpoint_path, method, device)
    controller = make_controller(model, method, device)
    rows: List[Dict[str, object]] = [dict(row) for row in existing]
    print(
        f"\nEvaluating {METHOD_LABELS[method]} | ratio={ratio} | "
        f"noise_seed={noise_seed} | tests={missing}"
    )
    for test_index in missing:
        sdre_cost = float(sdre_tests[test_index - 1]["cost"])
        try:
            result = system.simulate(initial_conditions[test_index - 1], controller)
            cost = float(result["cost"])
            normalized = cost / sdre_cost
            success = bool(np.isfinite(cost) and np.isfinite(normalized))
            error = ""
        except Exception as exc:  # Keep failed rollouts in the statistical record.
            result = {
                "terminal_state_norm": float("nan"),
                "max_state_norm": float("nan"),
                "max_control_norm": float("nan"),
            }
            cost = float("nan")
            normalized = float("nan")
            success = False
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "ratio": ratio,
            "noise_seed": noise_seed,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "test_index": test_index,
            "success": int(success),
            "cost": cost,
            "sdre_cost": sdre_cost,
            "normalized_cost": normalized,
            "terminal_state_norm": result["terminal_state_norm"],
            "max_state_norm": result["max_state_norm"],
            "max_control_norm": result["max_control_norm"],
            "error": error,
        }
        rows.append(row)
        print(
            f"test {test_index}/{len(initial_conditions)}: "
            f"success={success}, J={cost:.8g}, J/J_SDRE={normalized:.8g}"
        )
        with result_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda item: int(item["test_index"])))
    return result_path


def _finite(rows: Iterable[Dict[str, str]], field: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def summarize_ratio(
    ratio: str,
    evaluation_tag: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    test_initials_file = evaluation_protocol_dir(evaluation_tag) / "test_initial_conditions.mat"
    if not test_initials_file.exists():
        raise FileNotFoundError(f"Test protocol does not exist: {test_initials_file}")
    test_data = sio.loadmat(test_initials_file)
    test_count = len(_as_rows(test_data["X0_test"], STATE_DIM, "X0_test"))
    all_rows: List[Dict[str, str]] = []
    for method in METHOD_DIRS:
        for noise_seed in NOISE_SEEDS:
            all_rows.extend(
                _read_csv(evaluation_result_path(ratio, noise_seed, method, evaluation_tag))
            )
    if not all_rows:
        raise FileNotFoundError(f"No evaluation results exist for ratio {ratio}.")

    output_dir = ratio_summary_dir(ratio, evaluation_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "all_evaluation_cases.csv"
    with cases_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_fields = (
        "ratio",
        "method",
        "method_label",
        "noise_realizations",
        "evaluations",
        "successful_evaluations",
        "success_rate",
        "normalized_cost_mean",
        "normalized_cost_std_across_noise",
        "mean_cost",
    )
    summary_rows: List[Dict[str, object]] = []
    for method in METHOD_DIRS:
        method_rows = [row for row in all_rows if row["method"] == method]
        successes = [row for row in method_rows if int(row["success"]) == 1]
        per_noise_means = []
        for noise_seed in NOISE_SEEDS:
            seed_rows = [
                row
                for row in successes
                if int(row["noise_seed"]) == noise_seed
            ]
            normalized = _finite(seed_rows, "normalized_cost")
            if len(normalized) == test_count:
                per_noise_means.append(float(np.mean(normalized)))
        normalized_mean = float(np.mean(per_noise_means)) if per_noise_means else float("nan")
        normalized_std = (
            float(np.std(per_noise_means, ddof=1))
            if len(per_noise_means) > 1
            else float("nan")
        )
        costs = _finite(successes, "cost")
        summary_rows.append(
            {
                "ratio": ratio,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "noise_realizations": len(per_noise_means),
                "evaluations": len(method_rows),
                "successful_evaluations": len(successes),
                "success_rate": len(successes) / len(method_rows) if method_rows else 0.0,
                "normalized_cost_mean": normalized_mean,
                "normalized_cost_std_across_noise": normalized_std,
                "mean_cost": float(np.mean(costs)) if len(costs) else float("nan"),
            }
        )
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    paired_fields = (
        "ratio",
        "ours_method",
        "baseline_method",
        "paired_cases",
        "ours_wins",
        "ties",
        "paired_win_rate",
        "mean_normalized_cost_difference_ours_minus_baseline",
    )
    case_map: Dict[Tuple[str, int, int], Dict[str, str]] = {}
    for row in all_rows:
        if int(row["success"]) == 1:
            case_map[(row["method"], int(row["noise_seed"]), int(row["test_index"]))] = row
    paired_rows: List[Dict[str, object]] = []
    for baseline in ("u_only", "policy_net"):
        differences = []
        wins = 0
        ties = 0
        for noise_seed in NOISE_SEEDS:
            for test_index in range(1, test_count + 1):
                ours = case_map.get(("hjb", noise_seed, test_index))
                other = case_map.get((baseline, noise_seed, test_index))
                if ours is None or other is None:
                    continue
                difference = float(ours["normalized_cost"]) - float(other["normalized_cost"])
                differences.append(difference)
                if math.isclose(difference, 0.0, rel_tol=0.0, abs_tol=1e-12):
                    ties += 1
                elif difference < 0.0:
                    wins += 1
        paired_rows.append(
            {
                "ratio": ratio,
                "ours_method": METHOD_LABELS["hjb"],
                "baseline_method": METHOD_LABELS[baseline],
                "paired_cases": len(differences),
                "ours_wins": wins,
                "ties": ties,
                "paired_win_rate": wins / len(differences) if differences else float("nan"),
                "mean_normalized_cost_difference_ours_minus_baseline": (
                    float(np.mean(differences)) if differences else float("nan")
                ),
            }
        )
    paired_path = output_dir / "paired_summary.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired_rows)
    print(f"Summary: {summary_path}")
    print(f"Paired summary: {paired_path}")
    return cases_path, summary_path, paired_path


def summarize_all_ratios(
    ratios: Optional[Sequence[str]] = None,
    evaluation_tag: Optional[str] = None,
) -> Optional[Path]:
    rows: List[Dict[str, str]] = []
    selected_ratios = list(RATIO_DIRS) if ratios is None else list(ratios)
    for ratio in selected_ratios:
        summary_path = ratio_summary_dir(ratio, evaluation_tag) / "summary.csv"
        rows.extend(_read_csv(summary_path))
    if not rows:
        return None
    protocol_dir = evaluation_protocol_dir(evaluation_tag)
    all_ratios_summary_file = protocol_dir / "all_ratios_summary.csv"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    with all_ratios_summary_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"All-ratio summary: {all_ratios_summary_file}")
    return all_ratios_summary_file


def run_self_check(device: torch.device) -> None:
    """Check all three objectives on eight in-memory states, without I/O or ODEs."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.25, 0.25, (8, STATE_DIM)).astype(np.float32)
    u = rng.normal(0.0, 0.1, (8, CONTROL_DIM)).astype(np.float32)
    mean, std = np.zeros(STATE_DIM, dtype=np.float32), np.ones(STATE_DIM, dtype=np.float32)
    for method in METHOD_DIRS:
        set_global_seed(TRAIN_SEED)
        states = torch.as_tensor(x, device=device).requires_grad_(method != "policy_net")
        labels = torch.as_tensor(u, device=device)
        model = build_model(method, x_mean=mean, x_std=std).to(device)
        if method == "policy_net":
            prediction = model(states)
            loss = (prediction - labels).square().mean()
        else:
            residual, prediction, _ = value_quantities(model, states, create_graph=True)
            loss = (prediction - labels).square().mean()
            if method == "hjb":
                loss = loss + residual.square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError(f"Nonfinite objective or gradient for {method}.")
        print(f"CS {method}: in-memory objective/gradient check passed")
    initial = generate_test_initial_conditions()
    if initial.shape != (5, STATE_DIM) or not np.isfinite(initial).all():
        raise AssertionError("Invalid fixed test initial conditions.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=normalize_ratio,
        default=list(RATIO_DIRS.keys()),
    )
    parser.add_argument(
        "--noise-indices",
        nargs="+",
        type=int,
        default=list(range(len(NOISE_SEEDS))),
        help=f"Indices into fixed seeds {NOISE_SEEDS}; valid range 0-{len(NOISE_SEEDS)-1}.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_DIRS.keys()),
        default=list(METHOD_DIRS.keys()),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("prepare", "train", "evaluate", "summarize", "self-check"),
        default=("self-check",),
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--test-seed", type=int, default=TEST_INITIAL_SEED)
    parser.add_argument("--test-count", type=int, default=TEST_INITIAL_COUNT)
    parser.add_argument(
        "--evaluation-tag",
        default=None,
        help=(
            "Optional safe directory name for an evaluation-only test protocol. "
            "Named protocols reuse paired_noise_5x5 checkpoints and never overwrite "
            "the default five-initial-state evaluations."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    invalid = [index for index in args.noise_indices if index < 0 or index >= len(NOISE_SEEDS)]
    if invalid:
        raise ValueError(f"Invalid noise indices: {invalid}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive.")
    if args.horizon <= 0.0 or args.dt <= 0.0:
        raise ValueError("horizon and dt must be positive.")
    if args.test_count <= 0:
        raise ValueError("test-count must be positive.")
    if args.test_seed < 0 or args.test_seed > np.iinfo(np.uint32).max:
        raise ValueError("test-seed must be in the uint32 range.")
    if args.evaluation_tag is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.evaluation_tag):
            raise ValueError(
                "evaluation-tag must contain only letters, numbers, '.', '_' or '-', "
                "and must start with a letter or number."
            )
        if args.evaluation_tag == PROTOCOL_NAME:
            raise ValueError(f"evaluation-tag is reserved: {PROTOCOL_NAME}")
        forbidden = {"prepare", "train"}.intersection(args.stages)
        if forbidden:
            raise ValueError(
                "A named evaluation protocol is evaluation-only and cannot run "
                f"training stages: {sorted(forbidden)}. Use --stages evaluate summarize."
            )
    elif args.test_seed != TEST_INITIAL_SEED or args.test_count != TEST_INITIAL_COUNT:
        raise ValueError(
            "Custom test-seed/test-count requires --evaluation-tag so the default "
            "five-initial-state results cannot be overwritten."
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    ratios = list(dict.fromkeys(args.ratios))
    methods = list(dict.fromkeys(args.methods))
    noise_seeds = [NOISE_SEEDS[index] for index in dict.fromkeys(args.noise_indices)]
    stages = set(args.stages)
    device = select_device(args.device)
    test_protocol_dir = evaluation_protocol_dir(args.evaluation_tag)

    if "self-check" in stages:
        run_self_check(device)
        if stages == {"self-check"}:
            return 0

    clean: Optional[Dict[str, object]] = None
    if stages.intersection({"prepare", "train"}):
        clean = load_clean_dataset()
        print(f"Fixed clean data: {CLEAN_DATA_FILE}")
        print(f"Clean data hash: {clean['clean_hash']}")
        print(f"Samples: {len(np.asarray(clean['X']))}; noise seeds: {noise_seeds}")

    if "prepare" in stages:
        assert clean is not None
        prepare_test_initial_conditions(overwrite=args.overwrite)
        for ratio in ratios:
            for noise_seed in noise_seeds:
                prepare_noisy_dataset(
                    ratio, noise_seed, clean, overwrite=args.overwrite
                )

    if "train" in stages:
        assert clean is not None
        for ratio in ratios:
            for noise_seed in noise_seeds:
                if not noisy_data_path(ratio, noise_seed).exists():
                    prepare_noisy_dataset(ratio, noise_seed, clean, overwrite=False)
                for method in methods:
                    train_one_model(
                        ratio=ratio,
                        noise_seed=noise_seed,
                        method=method,
                        device=device,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        overwrite=args.overwrite,
                    )

    if "evaluate" in stages:
        initial_conditions = prepare_test_initial_conditions(
            test_seed=args.test_seed,
            test_count=args.test_count,
            protocol_dir=test_protocol_dir,
            overwrite=False,
        )
        system = CuckerSmaleSystem(horizon=args.horizon, dt=args.dt)
        sdre_tests = get_sdre_baseline(
            initial_conditions,
            system,
            protocol_dir=test_protocol_dir,
            test_seed=args.test_seed,
            overwrite=args.overwrite,
        )
        for ratio in ratios:
            for noise_seed in noise_seeds:
                for method in methods:
                    evaluate_one_model(
                        ratio=ratio,
                        noise_seed=noise_seed,
                        method=method,
                        device=device,
                        initial_conditions=initial_conditions,
                        sdre_tests=sdre_tests,
                        system=system,
                        evaluation_tag=args.evaluation_tag,
                        overwrite=args.overwrite,
                    )

    if "summarize" in stages:
        for ratio in ratios:
            try:
                summarize_ratio(ratio, evaluation_tag=args.evaluation_tag)
            except FileNotFoundError as exc:
                print(f"Summary skipped: {exc}")
        summarize_all_ratios(ratios=ratios, evaluation_tag=args.evaluation_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

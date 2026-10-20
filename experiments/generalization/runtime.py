"""Public full/half-data and generalization protocol; no datasets or weights bundled."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import math
import os
import random
import re
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "generalization"
EXPERIMENT = "neurips2026_public_generalization_v1"


def out(*parts):
    if not OUTPUT_ROOT.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("The output directory must not resolve outside the release root.")
    path = OUTPUT_ROOT.joinpath(*map(str, parts)).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Output must remain inside outputs/generalization.")
    return path


for key, sub in {"MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch",
                 "XDG_CACHE_HOME": "cache", "TORCHINDUCTOR_CACHE_DIR": "inductor",
                 "CUDA_CACHE_PATH": "cuda", "TRITON_CACHE_DIR": "triton",
                 "TEMP": "tmp", "TMP": "tmp"}.items():
    path = out(".runtime", sub)
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy
from scipy.io import loadmat
import sklearn
from sklearn.model_selection import train_test_split
import torch
import reference_ood as ref
import reference_training_cs as loss_cs
import reference_training_ac as loss_ac

ValueNetwork = ref.ValueNetwork
SPECS = {
    "cs": {"name": "Cucker--Smale", "input_dim": 40, "control_dim": 20,
           "total_count": 20000, "train_count": 16000, "held_count": 4000,
           "test_size": 0.2, "epochs": 4000, "batch_size": 128,
           "learning_rate": 0.0002, "scheduler": "step", "original_device": "cpu",
           "parameter": "interaction_strength", "nominal_parameter": 1.0,
           "parameter_values": [0.9, 1.0, 1.1]},
    "ac": {"name": "Allen--Cahn", "input_dim": 51, "control_dim": 1,
           "total_count": 10000, "train_count": 9000, "held_count": 1000,
           "test_size": 0.1, "epochs": 3000, "batch_size": 4096,
           "learning_rate": 0.002, "scheduler": "cosine", "original_device": "cuda",
           "parameter": "diffusion", "nominal_parameter": 0.1,
           "parameter_values": [0.09, 0.1, 0.11]},
}
PAPER_EPOCHS = {key: value["epochs"] for key, value in SPECS.items()}
EVALUATION = {"horizon": 50.0, "dt": 0.01, "diagnostic_dt": 0.2,
              "rtol": 1e-6, "atol": 1e-8, "blowup_norm": 1e6,
              "success_relative_radius": 0.05, "success_absolute_radius": 0.001,
              "entry_deadline_fraction": 0.8, "equilibrium_correction": False}
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(array_hash(state[name].detach().cpu().contiguous().numpy()).encode())
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_hash():
    return {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))}


def write_json(path, value):
    path = out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fieldnames=None):
    path = out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_npz(path, **arrays):
    path = out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def environment():
    return {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_cpu_threads": torch.get_num_threads()}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_training_device(system, requested="original"):
    name = SPECS[system]["original_device"] if requested == "original" else requested
    if name not in {"cpu", "cuda"}:
        raise ValueError("Device must be original, cpu or cuda.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; request --device cpu explicitly. No fallback.")
    return torch.device(name)


def dataset_path(system):
    return ROOT / "data" / f"{system}_main.mat"


@lru_cache(maxsize=4)
def data_split(system, budget="half"):
    spec = SPECS[system]
    path = dataset_path(system)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.relative_to(ROOT)}; run the release data generator first.")
    raw = loadmat(path)
    x = np.asarray(raw["X"], dtype=np.float32)
    u = np.asarray(raw["U_all"], dtype=np.float32)
    if u.shape == (spec["control_dim"], spec["total_count"]):
        u = u.T.copy()
    if x.shape != (spec["total_count"], spec["input_dim"]) or u.shape != (len(x), spec["control_dim"]):
        raise ValueError(f"Unexpected {system} dimensions: {x.shape}, {u.shape}.")
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError("Dataset has nonfinite entries; no synthetic fallback.")
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    train, held = train_test_split(np.arange(len(x)), test_size=spec["test_size"], random_state=42)
    order = train[np.random.RandomState(42).permutation(len(train))]
    if budget == "half":
        a = np.sort(np.random.default_rng(2026).choice(len(order), len(order) // 2, replace=False))
        b = np.sort(np.random.default_rng(2027).choice(len(held), len(held) // 2, replace=False))
        order, held = order[a], held[b]
        train = order.copy()
    elif budget != "full":
        raise ValueError("Budget must be full or half.")
    return x, u, {"train_indices": train, "held_indices": held, "train_order": order,
                  "mean": mean, "std": std}


def training_protocol(system, budget, device="original"):
    spec = SPECS[system]
    device = spec["original_device"] if device == "original" else str(device)
    if device not in {"cpu", "cuda"}:
        raise ValueError("Recorded training device must be cpu or cuda.")
    _, _, split = data_split(system, budget)
    ntrain, nheld = len(split["train_indices"]), len(split["held_indices"])
    return {"experiment": EXPERIMENT, "system": system, "budget": budget,
            "data_sha256": sha256(dataset_path(system)), "paired_data_count": ntrain + nheld,
            "train_count": ntrain, "held_count": nheld, "test_size": spec["test_size"],
            "epochs": spec["epochs"], "paper_epochs": PAPER_EPOCHS[system],
            "formal_training": spec["epochs"] == PAPER_EPOCHS[system],
            "batch_size": spec["batch_size"], "learning_rate": spec["learning_rate"],
            "scheduler": spec["scheduler"],
            "scheduler_parameters": {"step_size": 100, "gamma": 0.8} if system == "cs" else {"eta_min": 1e-6},
            "optimizer": "Adam", "gamma": 1.0, "lambda": 0.0, "gradient_clip_norm": 1.0,
            "hidden_layers": 3, "hidden_width": 256, "activation": "tanh",
            "initialization_seed": 42, "split_seed": 42, "train_subset_seed": 2026,
            "held_subset_seed": 2027, "normalization": "fixed full-X mean/std for both budgets",
            "split_hashes": {key: array_hash(value) for key, value in split.items()},
            "device": device, "optimizer_steps": spec["epochs"] * math.ceil(ntrain / spec["batch_size"]),
            "processed_training_examples": spec["epochs"] * ntrain,
            "checkpoint_selection": "last epoch, no validation selection",
            "availability": "X/U pairs and corresponding HJB states reduced together"}


def training_terms(system, model, states):
    module = loss_cs if system == "cs" else loss_ac
    hamiltonian, gradient, value = module.compute_HJB_H(model, states, states.device)
    return hamiltonian, module.compute_u_from_gradV(gradient, states.device), value


class InteractionStrengthSystem(ref.CuckerSmaleSystem):
    def __init__(self, strength=1.0):
        super().__init__(20)
        self.strength = float(strength)

    def _interaction_matrix(self, y):
        return self.strength * super()._interaction_matrix(y)

    def drift_torch(self, x):
        nominal = super().drift_torch(x)
        return torch.cat([nominal[:, :20], self.strength * nominal[:, 20:]], dim=1)


def make_system(system, parameter=None):
    value = SPECS[system]["nominal_parameter"] if parameter is None else float(parameter)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Plant parameter must be finite and positive.")
    return InteractionStrengthSystem(value) if system == "cs" else ref.AllenCahnSystem(51, sigma=value)


def fixed_cases(system, ood=False):
    scales = [1.0, 1.25, 1.5, 2.0] if ood else [1.0]
    if system == "cs":
        return ref.cucker_initial_conditions(scales, 5, 2026, include_fixed=False)
    return ref.allen_initial_conditions(scales, 5, 0.05, include_negative=False)


def validate_inputs(systems=("cs", "ac")):
    return {system: {"dataset": str(dataset_path(system).relative_to(ROOT)),
                     "sha256": sha256(dataset_path(system)),
                     "count": len(data_split(system, "full")[0])} for system in systems}


def prepare_run(name="main_protocol"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in {".", ".."}:
        raise ValueError("Use an ASCII run name without path separators.")
    directory = out(name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

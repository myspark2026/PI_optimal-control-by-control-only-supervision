"""Train and evaluate clean-label loss ablations and weight sensitivity.

The default action is an eight-sample interface check. Training is explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCRIPT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_ROOT.parents[1]
OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "ablation"

if __package__:
    from .models import PolicyNetwork, ValueNetwork
    from .science import SPECS, neural_control, sdre_control, simulate, value_terms, values_on_reference
else:
    sys.path.insert(0, str(SCRIPT_ROOT))
    from models import PolicyNetwork, ValueNetwork
    from science import SPECS, neural_control, sdre_control, simulate, value_terms, values_on_reference

import numpy as np
import scipy
from scipy.io import loadmat
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F

GAMMAS = ("0", "0.001", "0.01", "0.1", "1", "10", "100", "1000", "inf")
SETTINGS = GAMMAS + ("PolicyNet",)
EXPERIMENT = "code_release_clean_ablation_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(array_hash(state[name].detach().cpu().numpy()).encode())
    return digest.hexdigest()


def code_hash():
    return {p.name: sha256(p) for p in sorted(SCRIPT_ROOT.glob("*.py"))}


def output_path(*parts):
    path = OUTPUT_ROOT.joinpath(*map(str, parts)).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("All ablation outputs must stay under package/outputs/ablation.")
    return path


def write_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_npz(path, **arrays):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".tmp").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    path.with_name(path.name + ".tmp").replace(path)


def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def choose_device(system, requested, stage):
    name = requested
    if name == "original":
        name = SPECS[system]["training_device"] if stage == "train" else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose --device cpu explicitly.")
    return torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment():
    return {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def data_path(system, argument):
    if argument is None:
        return PACKAGE_ROOT / "data" / f"{system}_main.mat"
    path = Path(argument).expanduser()
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path / f"{system}_main.mat" if path.is_dir() else path


def load_data(system, path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Expert data are not distributed. Missing {path.name}; generate them with the package's "
            "data generator or pass --data PATH. No synthetic training fallback is used.")
    raw = loadmat(path)
    if "X" not in raw or "U_all" not in raw:
        raise ValueError("MAT input must contain X and U_all; value/gradient labels are not required.")
    spec = SPECS[system]
    x, u = np.asarray(raw["X"], dtype=np.float32), np.asarray(raw["U_all"], dtype=np.float32)
    if x.ndim == 2 and x.shape[1] != spec["dimension"] and x.shape[0] == spec["dimension"]:
        x = x.T.copy()
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if u.shape == (spec["controls"], len(x)):
        u = u.T.copy()
    if (x.ndim != 2 or x.shape[1] != spec["dimension"]
            or u.shape != (len(x), spec["controls"]) or len(x) < 10):
        raise ValueError(f"Unexpected {system} data shape: X={x.shape}, U_all={u.shape}.")
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError("Nonfinite expert data.")
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.
    train, held = train_test_split(np.arange(len(x)), test_size=.2, random_state=42)
    order = train[np.random.RandomState(42).permutation(len(train))]
    return x, u, {"train_indices": train, "held_indices": held, "train_order": order,
                  "mean": mean, "std": std}, sha256(path)


def make_model(system, setting, split):
    spec = SPECS[system]
    if setting == "PolicyNet":
        return PolicyNetwork(spec["dimension"], spec["controls"], split["mean"], split["std"])
    return ValueNetwork(spec["dimension"], split["mean"], split["std"])


def protocol(system, setting, x, split, fingerprint, epochs, debug):
    spec = SPECS[system]
    return {"experiment": EXPERIMENT, "system": system, "setting": setting,
            "data_sha256": fingerprint, "data_count": len(x),
            "train_count": len(split["train_indices"]), "held_count": len(split["held_indices"]),
            "test_size": .2, "initialization_seed": 42, "split_seed": 42,
            "input_dimension": spec["dimension"], "hidden_layers": 3, "hidden_width": 256,
            "activation": "tanh", "epochs": epochs, "batch_size": spec["batch"],
            "optimizer": "Adam", "initial_learning_rate": spec["lr"], "gradient_clip_norm": 1.,
            "scheduler": ({"name": "StepLR", "step_size": 100, "gamma": .8} if system == "cs"
                          else {"name": "CosineAnnealingLR", "T_max": epochs, "eta_min": 1e-6}),
            "objective": ("mean squared control error" if setting in ("inf", "PolicyNet")
                          else f"HJB_MSE + {setting} * control_MSE"),
            "lambda": 0., "normalization": "full supplied fixed X before the pointwise split",
            "split_hashes": {name: array_hash(value) for name, value in split.items()},
            "data_order": ("split order; dedicated Torch loader generator=42" if setting == "PolicyNet"
                           else "NumPy permutation seed=42; " + (
                               "global Torch loader RNG" if system == "cs" else "dedicated Torch loader generator=42")),
            "checkpoint_selection": "last epoch; no held-out model selection",
            "debug_protocol": bool(debug or len(x) != spec["samples"]),
            "numerical_scope": "Results are computed from the selected expert dataset.",
            "code_hash": code_hash()}


def epoch(model, loader, system, setting, device, optimizer=None):
    training, is_policy = optimizer is not None, setting == "PolicyNet"
    model.train(training)
    totals, count, steps = np.zeros(3), 0, 0
    # Value losses need input derivatives even on the held-out partition.
    with torch.set_grad_enabled(training or not is_policy):
        for bx, bu in loader:
            bx, bu = bx.to(device), bu.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if is_policy:
                error_sum = F.mse_loss(model(bx), bu, reduction="sum")
                control_loss = error_sum / bu.numel()
                hjb_loss = torch.zeros((), device=device)
                loss = control_loss
            else:
                bx = bx.detach().requires_grad_(True)
                _, control, hamiltonian = value_terms(system, model, bx, create_graph=training)
                if system == "cs":
                    control_loss = F.mse_loss(control, bu, reduction="mean")
                    hjb_loss = F.mse_loss(hamiltonian, torch.zeros_like(hamiltonian), reduction="mean")
                else:
                    control_loss, hjb_loss = (control - bu).square().mean(), hamiltonian.square().mean()
                loss = control_loss if setting == "inf" else hjb_loss + float(setting) * control_loss
            values = np.array([float(loss.detach().cpu()), float(hjb_loss.detach().cpu()),
                               float(control_loss.detach().cpu())])
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite training/held-out loss; no completed checkpoint will be written.")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                steps += 1
            totals += len(bx) * values
            count += len(bx)
    return totals / count, steps, count


def train_one(directory, system, setting, data, device, epochs, debug):
    x, u, split, fingerprint = data
    expected = protocol(system, setting, x, split, fingerprint, epochs, debug)
    target = directory / system / "models" / token(setting)
    path = target / "model.pth"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        verify_model(saved, expected)
        if saved.get("training_device") != str(device):
            raise ValueError("Existing model used a different training device; use a new --run-name.")
        print(f"Completed matching model unchanged: {system}/{setting}", flush=True)
        return
    if target.exists() and any(target.iterdir()):
        raise RuntimeError("Incomplete model directory exists. Keep it for inspection and use a new --run-name; no silent overwrite.")
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "protocol.json", expected)
    set_seed()
    is_policy = setting == "PolicyNet"
    order = split["train_indices"] if is_policy else split["train_order"]
    generator = (None if system == "cs" and not is_policy
                 else torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(x[order]), torch.from_numpy(u[order])),
        batch_size=SPECS[system]["batch"], shuffle=True, generator=generator)
    held_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(x[split["held_indices"]]), torch.from_numpy(u[split["held_indices"]])),
        batch_size=SPECS[system]["batch"], shuffle=False)
    model = make_model(system, setting, split).to(device)
    initial_hash = state_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=SPECS[system]["lr"])
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, 100, .8) if system == "cs"
                 else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6))
    history, steps, examples, training_seconds = [], 0, 0, 0.
    write_json(target / "status.json", {"status": "running", "debug_protocol": expected["debug_protocol"]})
    try:
        for number in range(1, epochs + 1):
            sync(device)
            start = time.perf_counter()
            lr = float(optimizer.param_groups[0]["lr"])
            train_loss, batch_steps, samples = epoch(model, train_loader, system, setting, device, optimizer)
            held_loss, _, _ = epoch(model, held_loader, system, setting, device)
            scheduler.step()
            sync(device)
            training_seconds += time.perf_counter() - start
            steps, examples = steps + batch_steps, examples + samples
            history.append({"epoch": number, "learning_rate_used": lr,
                            "train_total": train_loss[0], "train_hjb": train_loss[1],
                            "train_control": train_loss[2], "held_total": held_loss[0],
                            "held_hjb": held_loss[1], "held_control": held_loss[2],
                            "optimizer_steps": steps, "processed_examples": examples,
                            "training_seconds": training_seconds})
            if number == 1 or number % 50 == 0 or number == epochs:
                write_csv(target / "history.csv", history)
                print(f"{system}/{setting} {number}/{epochs}: train={train_loss[0]:.6g}, held={held_loss[0]:.6g}", flush=True)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        saved = {"experiment": EXPERIMENT, "protocol": expected, "model_state_dict": state,
                 "model_state_sha256": state_hash(state), "initial_state_sha256": initial_hash,
                 "epochs_completed": epochs, "optimizer_steps": steps, "processed_examples": examples,
                 "training_seconds": training_seconds, "training_device": str(device), "environment": environment(),
                 "timing_scope": "synchronized training + held-out + scheduler epoch loops; excludes I/O and setup"}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(saved, temporary)
        temporary.replace(path)
        write_json(target / "status.json", {"status": "completed", "epochs_completed": epochs,
                   "model_sha256": sha256(path), "optimizer_steps": steps, "processed_examples": examples,
                   "debug_protocol": expected["debug_protocol"], "training_seconds": training_seconds})
    except BaseException as exc:
        write_json(target / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   "epochs_completed": len(history), "error": repr(exc), "traceback": traceback.format_exc()})
        raise


def verify_model(saved, expected):
    if (saved.get("experiment") != EXPERIMENT or saved.get("protocol") != expected
            or saved.get("epochs_completed") != expected["epochs"]
            or saved.get("optimizer_steps") != expected["epochs"] * (
                (expected["train_count"] + expected["batch_size"] - 1) // expected["batch_size"])
            or saved.get("processed_examples") != expected["epochs"] * expected["train_count"]
            or saved.get("model_state_sha256") != state_hash(saved["model_state_dict"])):
        raise ValueError("Checkpoint settings, data, epoch count or state hash differ from this request.")


def token(setting):
    return "PolicyNet" if setting == "PolicyNet" else "gamma_" + setting.replace(".", "p")


def evaluate(directory, system, settings, data, device, epochs, debug):
    x, _u, split, fingerprint = data
    models, fingerprints = {}, {}
    missing = [setting for setting in settings
               if not (directory / system / "models" / token(setting) / "model.pth").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {system} model(s): {', '.join(missing)}. Explicitly run --stage train first. "
            "Train the requested setting and budget before evaluation.")
    for setting in settings:
        path = directory / system / "models" / token(setting) / "model.pth"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        verify_model(saved, protocol(system, setting, x, split, fingerprint, epochs, debug))
        model = make_model(system, setting, split)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[setting], fingerprints[setting] = model, sha256(path)
    target = directory / system / "evaluation"
    settings_record = {"system": system, "methods": list(settings), "data_sha256": fingerprint,
                       "models": fingerprints, "code_hash": code_hash(), "device": str(device),
                       "horizon": SPECS[system]["horizon"], "dt": .01, "rtol": 1e-3, "atol": 1e-6,
                       "debug_protocol": bool(debug or len(x) != SPECS[system]["samples"])}
    metadata_path = target / "metadata.json"
    if metadata_path.is_file() and json.loads(metadata_path.read_text(encoding="utf-8")) != settings_record:
        raise ValueError("Existing evaluation uses different methods/data/models/settings; use a new --run-name.")
    if (target / "results.json").is_file():
        print(f"Matching completed evaluation unchanged: {system}", flush=True)
        return json.loads((target / "results.json").read_text(encoding="utf-8"))["rows"]
    target.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, settings_record)
    print(f"{system}: computing one common SDRE reference", flush=True)
    reference = simulate(system, lambda state: sdre_control(system, state))
    save_npz(target / "sdre_reference.npz", **reference)
    rows = []
    for setting in settings:
        row = {"system": system, "setting": setting, "C": None, "E_AUC": None,
               "rollout_completed": False, "cost": None, "terminal_norm": None,
               "rollout_error": None, "value_error": None,
               "debug_protocol": settings_record["debug_protocol"]}
        model = models[setting]
        if setting != "PolicyNet":
            try:
                row["E_AUC"], values = values_on_reference(model, reference)
                save_npz(target / (token(setting) + "_value_on_sdre.npz"), times=reference["times"],
                         predicted_value=values, sdre_cost_to_go=reference["cost_to_go"])
            except Exception as exc:
                row["value_error"] = str(exc)
        try:
            rollout = simulate(system, lambda state: neural_control(system, model, state, setting == "PolicyNet"))
            row.update(C=rollout["cost"] / reference["cost"], cost=rollout["cost"],
                       terminal_norm=float(np.linalg.norm(rollout["states"][-1])), rollout_completed=True)
            save_npz(target / (token(setting) + "_rollout.npz"), **rollout)
        except Exception as exc:
            row["rollout_error"] = str(exc)
        rows.append(row)
        write_csv(target / "cases_in_progress.csv", rows)
        print(row, flush=True)
    rows.append({"system": system, "setting": "SDRE", "C": 1., "E_AUC": None,
                 "rollout_completed": True, "cost": reference["cost"],
                 "terminal_norm": float(np.linalg.norm(reference["states"][-1])),
                 "rollout_error": None, "value_error": None,
                 "debug_protocol": settings_record["debug_protocol"]})
    write_csv(target / "summary.csv", rows)
    write_json(target / "results.json", {"metadata": settings_record, "rows": rows})
    return rows


def format_number(value):
    return "--" if value is None else f"{float(value):.4f}"


def combined_table(directory, all_rows):
    indexed = {(row["system"], row["setting"]): row for row in all_rows}
    methods = [setting for setting in SETTINGS + ("SDRE",)
               if any((system, setting) in indexed for system in SPECS)]
    rows = []
    for setting in methods:
        row = {"setting": setting}
        for system in SPECS:
            value = indexed.get((system, setting), {})
            row[system.upper() + "_C"] = value.get("C")
            row[system.upper() + "_E_AUC"] = value.get("E_AUC")
        rows.append(row)
    write_csv(directory / "combined_summary.csv", rows)
    debug = any(row["debug_protocol"] for row in all_rows)
    caption = ("Diagnostic training budget. " if debug else "")
    caption += ("Clean-label ablation using the supplied data and a common nominal SDRE reference per system. "
                "Both systems use an 80/20 pointwise training/held-out split. "
                "Dashes denote unavailable metrics; incomplete rollouts are retained in the CSV.")
    lines = [r"\begin{table}[tbp]", r"\centering", "\\caption{" + caption + "}",
             r"\begin{tabular}{lrrrr}", r"\toprule", r"Setting & CS $C$ & CS $E_{\rm AUC}$ & AC $C$ & AC $E_{\rm AUC}$ \\", r"\midrule"]
    for row in rows:
        setting = row["setting"]
        label = (setting if setting in {"PolicyNet", "SDRE"} else
                 r"$\gamma=\infty$ ($u$-only)" if setting == "inf" else "$\\gamma=" + setting + "$"
                 + (" (Ours)" if setting == "1" else " (HJB-only)" if setting == "0" else ""))
        lines.append(" & ".join([label] + [format_number(row[key]) for key in ["CS_C", "CS_E_AUC", "AC_C", "AC_E_AUC"]]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output_path(directory / "combined_table.tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def check(directory, system, requested, path):
    device = choose_device(system, requested, "check")
    if path.is_file():
        x, u, split, _ = load_data(system, path)
        x, u = x[:8], u[:8]
        source = "eight samples from the supplied expert dataset"
    else:
        rng = np.random.default_rng(1234)
        x = (.05 * rng.standard_normal((8, SPECS[system]["dimension"]))).astype(np.float32)
        u = (.02 * rng.standard_normal((8, SPECS[system]["controls"]))).astype(np.float32)
        split = {"mean": np.zeros(x.shape[1], dtype=np.float32), "std": np.ones(x.shape[1], dtype=np.float32)}
        source = "synthetic interface inputs"
    results = []
    for setting in ["1", "PolicyNet"]:
        set_seed()
        model = make_model(system, setting, split).to(device)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(u)), batch_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=SPECS[system]["lr"])
        metrics, steps, samples = epoch(model, loader, system, setting, device, optimizer)
        results.append({"setting": setting, "finite_loss": bool(np.isfinite(metrics).all()),
                        "optimizer_steps": steps, "samples": samples})
    report = {"stage": "interface_check_only", "system": system, "input_scope": source,
              "device": str(device), "results": results, "formal_training_performed": False}
    write_json(directory / "checks" / f"{system}.json", report)
    print(report, flush=True)


def parse_settings(arguments):
    settings = []
    for argument in arguments:
        for item in argument.split(","):
            item = "PolicyNet" if item.lower() in {"policy", "policynet"} else item
            if item not in SETTINGS:
                raise ValueError(f"Unknown setting {item}; use {', '.join(SETTINGS)}.")
            if item not in settings:
                settings.append(item)
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("cs", "ac", "both"), default="both")
    parser.add_argument("--stage", choices=("check", "train", "evaluate"), default="check")
    parser.add_argument("--gammas", nargs="+", default=list(SETTINGS), help="Gamma values and/or PolicyNet; comma-separated values are also accepted.")
    parser.add_argument("--device", choices=("original", "cpu", "cuda"), default="original",
                        help="original: CS CPU and AC CUDA for training; CPU for checks/evaluation.")
    parser.add_argument("--epochs", type=int, help="Diagnostic epoch override; outputs use a separate diagnostic run name.")
    parser.add_argument("--data", help="MAT file for one system, or a directory with cs_main.mat/ac_main.mat. Relative paths start at package/.")
    parser.add_argument("--run-name", default="paper_protocol")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name) or args.run_name in {".", ".."}:
        parser.error("Invalid --run-name; use ASCII letters/digits, _, - and .")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive.")
    try:
        settings = parse_settings(args.gammas)
    except ValueError as exc:
        parser.error(str(exc))
    systems = tuple(SPECS) if args.system == "both" else (args.system,)
    if args.data and args.system == "both":
        supplied = Path(args.data)
        supplied = supplied if supplied.is_absolute() else PACKAGE_ROOT / supplied
        if not supplied.is_dir():
            parser.error("For --system both, --data must be a directory, not one MAT file.")
    name = args.run_name + (f"_debug_epochs_{args.epochs}" if args.epochs is not None else "")
    directory = output_path(name)
    all_rows = []
    for system in systems:
        path = data_path(system, args.data)
        if args.stage == "check":
            if args.data is not None and not path.is_file():
                raise FileNotFoundError(f"The explicitly supplied check dataset does not exist: {path}")
            check(directory, system, args.device, path)
            continue
        data = load_data(system, path)
        epochs = args.epochs if args.epochs is not None else SPECS[system]["epochs"]
        device = choose_device(system, args.device, args.stage)
        if args.stage == "train":
            for setting in settings:
                train_one(directory, system, setting, data, device, epochs, args.epochs is not None)
        else:
            all_rows.extend(evaluate(directory, system, settings, data, device, epochs, args.epochs is not None))
    if args.stage == "evaluate":
        combined_table(directory, all_rows)


if __name__ == "__main__":
    main()

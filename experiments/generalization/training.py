"""Train full/half-data models under the public release protocol.

``loop_seconds`` measures the training batches; ``train_seconds`` includes held-out evaluation
and scheduler stepping. Both are synchronized elapsed times, excluding file I/O.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import time
import traceback

from runtime import (
    ROOT, OUTPUT_ROOT, EXPERIMENT, out, SPECS, ValueNetwork, choose_training_device, code_hash, data_split,
    environment, np, prepare_run, set_seed, sha256, state_hash, torch,
    training_protocol, training_terms, validate_inputs, write_csv, write_json,
)


CHECKPOINT_INTERVAL = 100
TIME_DEFINITION = (
    "Synchronized elapsed seconds summed over completed epochs; loop_seconds "
    "covers training batches, while train_seconds also covers held-out "
    "evaluation and scheduler stepping. Loading, initialization, console/file "
    "I/O and interruption downtime are excluded."
)


def _field(spec, key):
    return spec[key] if isinstance(spec, dict) else getattr(spec, key)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_state(model):
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def _atomic_torch_save(path, value):
    path = out(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _rng_state(device):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": ([item.cpu() for item in torch.cuda.get_rng_state_all()]
                 if device.type == "cuda" else None),
    }


def _restore_rng(state, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"], numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        if state["cuda"] is None:
            raise RuntimeError("Cannot resume CUDA training without CUDA RNG states.")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _ordered_train_indices(split):
    indices = np.asarray(split["train_indices"], dtype=np.int64).reshape(-1)
    order = np.asarray(split["train_order"], dtype=np.int64).reshape(-1)
    if len(order) != len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("Training indices/order must be equally sized and unique.")
    if np.array_equal(np.sort(order), np.sort(indices)):
        return order  # Runtime supplied the ordered global data indices.
    if np.array_equal(np.sort(order), np.arange(len(indices))):
        return indices[order]  # Runtime supplied a local permutation instead.
    raise ValueError("train_order is neither a global ordering nor a local permutation.")


def _verify_checkpoint(saved, system, budget, protocol, initial_hash, hashes):
    expected = {
        "experiment": EXPERIMENT, "system": system, "budget": budget,
        "protocol": protocol, "init_state_sha256": initial_hash,
        "code_hash": hashes,
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise RuntimeError(f"Existing checkpoint has incompatible {key}; use a new run name.")
    if state_hash(saved["model_state_dict"]) != saved["model_state_sha256"]:
        raise RuntimeError("Existing checkpoint model-state hash is inconsistent.")


def _epoch(model, loader, system, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "hjb": 0.0, "control": 0.0}
    examples = 0
    steps = 0
    # HJB diagnostics require input derivatives and the original create_graph
    # convention even during held-out evaluation; do not use torch.no_grad().
    with torch.enable_grad():
        for states, labels in loader:
            states = states.to(device).detach().requires_grad_(True)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad()
            hamiltonian, control, _value = training_terms(system, model, states)
            hjb_loss = hamiltonian.square().mean()
            control_loss = (control - labels).square().mean()
            loss = hjb_loss + control_loss
            values = (float(loss.detach().cpu()), float(hjb_loss.detach().cpu()),
                      float(control_loss.detach().cpu()))
            if not all(np.isfinite(value) for value in values):
                raise FloatingPointError("A non-finite training/held-out loss was encountered.")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                               error_if_nonfinite=True)
                optimizer.step()
                steps += 1
            count = len(states)
            examples += count
            for key, value in zip(totals, values):
                totals[key] += count * value
    if not examples:
        raise RuntimeError("Empty data loader.")
    return {key: value / examples for key, value in totals.items()}, steps, examples


def train_one(run_directory, system, budget, requested_device):
    spec = SPECS[system]
    device = torch.device(choose_training_device(system, requested_device))
    protocol = training_protocol(system, budget, str(device))
    epochs = int(_field(spec, "epochs"))
    batch_size = int(_field(spec, "batch_size"))
    directory = Path(run_directory) / "training" / system / budget
    if not directory.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Training output must remain inside outputs/generalization.")

    states, labels, split = data_split(system, budget)
    train_indices = _ordered_train_indices(split)
    held_indices = np.asarray(split["held_indices"], dtype=np.int64).reshape(-1)
    if len(np.intersect1d(train_indices, held_indices)):
        raise ValueError("Training and held-out membership overlap.")
    if len(states) != int(_field(spec, "total_count")) or len(states) != len(labels):
        raise ValueError("Runtime must return the complete fixed data arrays.")
    fraction = 1.0 if budget == "full" else 0.5
    if (len(train_indices) != int(_field(spec, "train_count") * fraction)
            or len(held_indices) != int(_field(spec, "held_count") * fraction)):
        raise ValueError("Unexpected full/half train and held-out counts.")
    if not len(held_indices) or len(np.unique(held_indices)) != len(held_indices):
        raise ValueError("Held-out indices must be nonempty and unique.")
    if min(train_indices.min(), held_indices.min()) < 0 or max(
            train_indices.max(), held_indices.max()) >= len(states):
        raise ValueError("Split indices are outside the fixed dataset.")
    if np.asarray(labels).ndim == 1:
        labels = np.asarray(labels).reshape(-1, 1)

    set_seed(42)
    train_data = torch.utils.data.TensorDataset(
        torch.as_tensor(states[train_indices], dtype=torch.float32),
        torch.as_tensor(labels[train_indices], dtype=torch.float32),
    )
    held_data = torch.utils.data.TensorDataset(
        torch.as_tensor(states[held_indices], dtype=torch.float32),
        torch.as_tensor(labels[held_indices], dtype=torch.float32),
    )
    # Preserve the original global Torch RNG convention: construct loaders,
    # initialize the network, then first iterate the loaders. No generator is set.
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    held_loader = torch.utils.data.DataLoader(
        held_data, batch_size=batch_size, shuffle=False, drop_last=False)
    model = ValueNetwork(int(_field(spec, "input_dim")))
    with torch.no_grad():
        model.x_mean.copy_(torch.as_tensor(split["mean"], dtype=torch.float32))
        model.x_std.copy_(torch.as_tensor(split["std"], dtype=torch.float32))
    model.to(device)
    initial_hash = state_hash(_cpu_state(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(_field(spec, "learning_rate")))
    if _field(spec, "scheduler") == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    elif _field(spec, "scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1.0e-6)
    else:
        raise ValueError("Unknown scheduler in the fixed main-experiment protocol.")

    hashes = code_hash()
    final_path = directory / "model.pth"
    resume_path = directory / "checkpoint_last.pth"
    history = []
    completed = optimizer_steps = processed_examples = 0
    loop_seconds = held_seconds = train_seconds = 0.0
    if final_path.is_file():
        saved = torch.load(final_path, map_location="cpu", weights_only=True)
        _verify_checkpoint(saved, system, budget, protocol, initial_hash, hashes)
        if int(saved["epochs_completed"]) != epochs:
            raise RuntimeError("Final model is incomplete; refusing to overwrite it.")
        print(f"Already completed; unchanged: {final_path}", flush=True)
        return
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        _verify_checkpoint(saved, system, budget, protocol, initial_hash, hashes)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        history = saved["history"]
        completed = int(saved["epochs_completed"])
        optimizer_steps = int(saved["optimizer_steps"])
        processed_examples = int(saved["processed_examples"])
        loop_seconds = float(saved["loop_seconds"])
        held_seconds = float(saved["held_seconds"])
        train_seconds = float(saved["train_seconds"])
        if len(history) != completed or not 0 <= completed <= epochs:
            raise RuntimeError("Resume checkpoint history/epoch count is inconsistent.")
        _restore_rng(saved["rng_state"], device)
        print(f"Resuming {system}/{budget} after epoch {completed}.", flush=True)
    elif directory.exists() and any(directory.iterdir()):
        raise RuntimeError(
            f"Unfinished output has no resumable checkpoint: {directory}. "
            "Use a new run name; existing files will not be overwritten.")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "environment.json", environment())

    def checkpoint():
        state = _cpu_state(model)
        return {
            "experiment": EXPERIMENT, "system": system, "budget": budget,
            "protocol": protocol, "model_state_dict": state,
            "model_state_sha256": state_hash(state), "init_state_sha256": initial_hash,
            "epochs_completed": completed, "optimizer_steps": optimizer_steps,
            "processed_examples": processed_examples, "loop_seconds": loop_seconds,
            "held_seconds": held_seconds, "train_seconds": train_seconds,
            "time_definition": TIME_DEFINITION, "code_hash": hashes,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _rng_state(device), "history": history,
            "checkpoint_selection": "last epoch, no validation selection",
            "training_device": str(device),
        }

    # Commit the initialized model and RNG before the first batch, so even an
    # interruption before epoch 100 can safely resume instead of losing the run.
    if not resume_path.is_file():
        _atomic_torch_save(resume_path, checkpoint())

    write_json(directory / "status.json", {
        "status": "running", "epochs_completed": completed, "epochs": epochs,
        "system": system, "budget": budget, "device": str(device),
    })
    committed_epoch = completed
    try:
        for epoch in range(completed + 1, epochs + 1):
            _sync(device)
            epoch_start = time.perf_counter()
            block_start = time.perf_counter()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_loss, steps, examples = _epoch(
                model, train_loader, system, device, optimizer)
            _sync(device)
            epoch_loop_seconds = time.perf_counter() - block_start
            block_start = time.perf_counter()
            held_loss, _, _ = _epoch(model, held_loader, system, device)
            _sync(device)
            epoch_held_seconds = time.perf_counter() - block_start
            scheduler.step()
            _sync(device)
            epoch_seconds = time.perf_counter() - epoch_start
            loop_seconds += epoch_loop_seconds
            held_seconds += epoch_held_seconds
            train_seconds += epoch_seconds
            optimizer_steps += steps
            processed_examples += examples
            completed = epoch
            history.append({
                "epoch": epoch, "learning_rate_used": learning_rate,
                "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
                "train_total": train_loss["total"], "train_hjb": train_loss["hjb"],
                "train_control": train_loss["control"],
                "held_total": held_loss["total"], "held_hjb": held_loss["hjb"],
                "held_control": held_loss["control"], "optimizer_steps": optimizer_steps,
                "processed_examples": processed_examples,
                "epoch_loop_seconds": epoch_loop_seconds,
                "epoch_held_seconds": epoch_held_seconds,
                "epoch_train_seconds": epoch_seconds, "loop_seconds": loop_seconds,
                "held_seconds": held_seconds, "train_seconds": train_seconds,
            })
            if epoch % CHECKPOINT_INTERVAL == 0 or epoch == epochs:
                _atomic_torch_save(resume_path, checkpoint())
                committed_epoch = completed
                write_csv(directory / "history.csv", history)
                write_json(directory / "status.json", {
                    "status": "running", "epochs_completed": completed, "epochs": epochs,
                    "optimizer_steps": optimizer_steps, "processed_examples": processed_examples,
                    "loop_seconds": loop_seconds, "train_seconds": train_seconds,
                })
            if epoch == 1 or epoch % CHECKPOINT_INTERVAL == 0 or epoch == epochs:
                print(f"{system}/{budget} {epoch}/{epochs}: "
                      f"train={train_loss['total']:.6g} held={held_loss['total']:.6g} "
                      f"steps={optimizer_steps} seconds={train_seconds:.1f}", flush=True)

        expected_steps = epochs * len(train_loader)
        if optimizer_steps != expected_steps or processed_examples != epochs * len(train_data):
            raise RuntimeError("Completed training counters do not match the fixed budget.")
        saved = checkpoint()
        _atomic_torch_save(final_path, saved)
        write_csv(directory / "history.csv", history)
        write_json(directory / "status.json", {
            "status": "completed", "experiment": EXPERIMENT,
            "system": system, "budget": budget, "epochs_completed": completed,
            "optimizer_steps": optimizer_steps, "processed_examples": processed_examples,
            "loop_seconds": loop_seconds, "held_seconds": held_seconds,
            "train_seconds": train_seconds, "time_definition": TIME_DEFINITION,
            "model_sha256": sha256(final_path), "model_state_sha256": saved["model_state_sha256"],
            "init_state_sha256": initial_hash, "final_metrics": history[-1],
        })
    except BaseException as error:
        write_json(directory / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "system": system, "budget": budget, "epochs_completed_in_memory": completed,
            "recoverable_epochs": committed_epoch, "epochs": epochs,
            "error": repr(error), "traceback": traceback.format_exc(),
            "resume": "Automatic from checkpoint_last.pth, if present; no silent restart.",
        })
        raise


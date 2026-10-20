"""Public OOD, data-budget and frozen-dynamics evaluations; no implicit model fallback."""
from __future__ import annotations
import json
import time
from pathlib import Path
from runtime import (ROOT, EXPERIMENT, SPECS, EVALUATION,
    ValueNetwork, canonical_hash, code_hash, data_split, environment, fixed_cases,
    make_system, out, ref, save_npz, sha256, state_hash, training_protocol, write_csv, write_json, np, torch)

def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None

def complete_rollout(rollout, horizon):
    count = int(round(horizon / EVALUATION["dt"])) + 1
    return bool(rollout.complete and len(rollout.t) == count
                and np.isclose(rollout.t[-1], horizon)
                and all(np.isfinite(getattr(rollout, field)).all()
                        for field in ["t", "x", "u", "running_cost", "cumulative_cost"]))


def basic_metrics(rollout):
    if not complete_rollout(rollout, EVALUATION["horizon"]):
        return {"completed": False, "successful": False, "cost": None, "terminal_norm": None,
                "entry_time": None, "failure_reason": rollout.failure_reason or "Incomplete or nonfinite full-horizon trajectory"}
    norms = np.linalg.norm(rollout.x, axis=0)
    radius = max(EVALUATION["success_absolute_radius"], EVALUATION["success_relative_radius"] * norms[0])
    eligible = np.flatnonzero(np.maximum.accumulate(norms[::-1])[::-1] <= radius)
    entry = finite(rollout.t[eligible[0]]) if eligible.size else None
    return {"completed": True,
            "successful": bool(entry is not None and entry <= EVALUATION["entry_deadline_fraction"] * EVALUATION["horizon"]),
            "cost": finite(rollout.cumulative_cost[-1]), "terminal_norm": finite(norms[-1]),
            "entry_time": entry, "success_radius": finite(radius), "failure_reason": None}


def cached_rollout(directory, system, parameter, case, kind, model=None, model_sha=None, device="cuda"):
    plant = make_system(system, parameter)
    software = environment()
    settings = {"experiment": EXPERIMENT, "system": system, "parameter": parameter,
                "parameter_name": SPECS[system]["parameter"], "case_id": case["case_id"],
                "x0_sha256": canonical_hash(np.asarray(case["x0"], dtype=float).tolist()),
                "evaluation": EVALUATION, "controller": kind,
                "model_sha256": model_sha if kind == "neural" else None,
                "device": str(device) if kind == "neural" else "cpu",
                "software": {key: software[key] for key in ["torch", "numpy", "scipy"]},
                "code_hash": code_hash()}
    key = canonical_hash(settings)
    target = out(directory / "cache" / kind / key)
    metadata_path, arrays_path = target / "record.json", target / "rollout.npz"
    if metadata_path.is_file():
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if record["settings"] != settings or sha256(arrays_path) != record["arrays_sha256"]:
            raise ValueError(f"Cached rollout metadata/data changed: {target}")
        with np.load(arrays_path, allow_pickle=False) as saved:
            rollout = ref.Rollout(t=saved["t"], x=saved["x"], u=saved["u"], running_cost=saved["running_cost"],
                                  cumulative_cost=saved["cumulative_cost"], dt=EVALUATION["dt"],
                                  complete=record["raw_complete"], failure_reason=record["raw_failure_reason"])
        return rollout, record, target
    target.mkdir(parents=True, exist_ok=True)
    if kind == "sdre":
        controller = plant.sdre_control
    else:
        controller = ref.LearnedValueController(model, plant, torch.device(device), equilibrium_correction=False).control
    calls = 0
    intervals = int(round(EVALUATION["horizon"] / EVALUATION["dt"]))
    def progress_controller(state):
        nonlocal calls
        calls += 1
        if calls == 1 or calls % 1000 == 0:
            print(f"    {kind}: control evaluation {calls}/{intervals + 1}", flush=True)
        return controller(state)
    started = time.perf_counter()
    rollout = ref.simulate_closed_loop(plant, progress_controller, np.asarray(case["x0"], dtype=float),
                                      horizon=EVALUATION["horizon"], dt=EVALUATION["dt"],
                                      rtol=EVALUATION["rtol"], atol=EVALUATION["atol"], blowup_norm=EVALUATION["blowup_norm"])
    save_npz(arrays_path, t=rollout.t, x=rollout.x, u=rollout.u, running_cost=rollout.running_cost,
             cumulative_cost=rollout.cumulative_cost, x0=case["x0"])
    record = {"settings": settings, "arrays_sha256": sha256(arrays_path),
              "raw_complete": bool(rollout.complete), "raw_failure_reason": rollout.failure_reason,
              "metrics": basic_metrics(rollout), "evaluation_seconds": time.perf_counter() - started}
    write_json(metadata_path, record)
    return rollout, record, target


def value_on_reference(model, reference):
    with torch.no_grad():
        device = next(model.parameters()).device
        values = np.concatenate([model(torch.as_tensor(reference.x[:, start:start + 512].T,
                     dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
                     for start in range(0, len(reference.t), 512)])
    areas = np.diff(reference.t) * (reference.running_cost[:-1] + reference.running_cost[1:]) / 2.0
    target = np.concatenate([np.cumsum(areas[::-1])[::-1], [0.0]])
    denominator = float(np.trapezoid(np.abs(target), reference.t))
    if denominator <= 0 or not np.isfinite(values).all():
        raise ValueError("Invalid SDRE cost-to-go normalization or value predictions.")
    error = finite(np.trapezoid(np.abs(values - target), reference.t) / denominator)
    return error, values, target


def paired_case(directory, system, parameter, case, model, info, device, value_metric):
    print(f"\n{system} {SPECS[system]['parameter']}={parameter:g}, {case['case_id']}, {info['source']}", flush=True)
    reference, sdre_record, sdre_path = cached_rollout(directory, system, parameter, case, "sdre")
    neural, nn_record, nn_path = cached_rollout(directory, system, parameter, case, "neural", model,
                                               info["checkpoint_sha256"], device)
    sm, nm = sdre_record["metrics"], nn_record["metrics"]
    row = {"system": system, "source": info["source"], "parameter_name": SPECS[system]["parameter"],
           "parameter_value": parameter, "case_id": case["case_id"], "scale": float(case["scale"]), "checkpoint_sha256": info["checkpoint_sha256"],
           "nn_completed": nm["completed"], "nn_successful": nm["successful"], "nn_cost": nm["cost"],
           "sdre_completed": sm["completed"], "sdre_successful": sm["successful"], "sdre_cost": sm["cost"],
           "cost_ratio": None, "terminal_norm": nm["terminal_norm"], "mean_abs_hjb_residual": None,
           "E_AUC": None, "nn_failure_reason": nm["failure_reason"], "sdre_failure_reason": sm["failure_reason"],
           "diagnostic_error": None, "value_error": None,
           "nn_cache": str(nn_path.relative_to(ROOT)), "sdre_cache": str(sdre_path.relative_to(ROOT))}
    if nm["completed"] and sm["completed"] and sm["cost"] is not None and sm["cost"] > 0:
        row["cost_ratio"] = finite(nm["cost"] / sm["cost"])
    if nm["completed"]:
        try:
            stride = int(round(EVALUATION["diagnostic_dt"] / EVALUATION["dt"]))
            indices = np.arange(0, len(neural.t), stride)
            if indices[-1] != len(neural.t) - 1:
                indices = np.append(indices, len(neural.t) - 1)
            controller = ref.LearnedValueController(model, make_system(system, parameter), torch.device(device),
                                                    equilibrium_correction=False)
            diagnostics = controller.diagnostics(neural.x[:, indices].T)
            absolute = np.abs(diagnostics["hjb"])
            if not np.isfinite(absolute).all():
                raise ValueError("Nonfinite perturbed-plant HJB diagnostic.")
            row["mean_abs_hjb_residual"] = finite(absolute.mean())
            save_npz(nn_path / "hjb_diagnostics.npz", t=neural.t[indices], hjb=diagnostics["hjb"], hjb_absolute=absolute)
        except Exception as exc:
            row["diagnostic_error"] = str(exc)
    if value_metric and sm["completed"]:
        try:
            error, values, target = value_on_reference(model, reference)
            row["E_AUC"] = error
            save_npz(nn_path / "value_on_sdre.npz", t=reference.t, predicted_value=values, sdre_cost_to_go=target)
        except Exception as exc:
            row["value_error"] = str(exc)
    elif value_metric:
        row["value_error"] = "No complete SDRE reference; no full-horizon E_AUC assigned."
    print({key: row[key] for key in ["nn_completed", "nn_successful", "cost_ratio", "terminal_norm", "mean_abs_hjb_residual", "E_AUC"]}, flush=True)
    return row


def summarize(rows):
    def mean(field, selected):
        values = [r[field] for r in selected if r[field] is not None]
        return float(np.mean(values)) if values else None
    ratios = [r["cost_ratio"] for r in rows if r["cost_ratio"] is not None]
    successful = [r["cost_ratio"] for r in rows if r["nn_successful"] and r["cost_ratio"] is not None]
    completed = sum(r["nn_completed"] for r in rows)
    nsuccess = sum(r["nn_successful"] for r in rows)
    diagnostic_cases = sum(r["mean_abs_hjb_residual"] is not None for r in rows)
    value_cases = sum(r["E_AUC"] is not None for r in rows)
    result = {"system": rows[0]["system"], "source": rows[0]["source"],
              "parameter_name": rows[0]["parameter_name"], "parameter_value": rows[0]["parameter_value"],
              "scale": rows[0]["scale"], "test_cases": len(rows), "nn_completed": completed,
              "nn_successful": nsuccess, "success_rate": nsuccess / len(rows),
              "sdre_completed": sum(r["sdre_completed"] for r in rows),
              "sdre_successful": sum(r["sdre_successful"] for r in rows),
              "cost_ratio_cases": len(ratios),
              "mean_cost_ratio": float(np.mean(ratios)) if len(ratios) == len(rows) else None,
              "mean_cost_ratio_available_cases": float(np.mean(ratios)) if ratios else None,
              "median_cost_ratio_successful": float(np.median(successful)) if successful and len(successful) == nsuccess else None,
              "successful_cost_ratio_cases": len(successful),
              "mean_terminal_norm": mean("terminal_norm", rows) if completed == len(rows) else None,
              "mean_terminal_norm_available_cases": mean("terminal_norm", rows),
              "terminal_norm_cases": sum(r["terminal_norm"] is not None for r in rows),
              "mean_abs_hjb_residual": mean("mean_abs_hjb_residual", rows) if diagnostic_cases == len(rows) else None,
              "mean_abs_hjb_available_cases": mean("mean_abs_hjb_residual", rows),
              "hjb_diagnostic_cases": diagnostic_cases,
              "mean_E_AUC": mean("E_AUC", rows) if value_cases == len(rows) else None,
              "value_metric_cases": value_cases}
    return result


def _validate_checkpoint(saved, system, budget):
    """Require a completed structured checkpoint for this workflow and dataset."""
    if (not isinstance(saved, dict) or saved.get("experiment") != EXPERIMENT
            or saved.get("system") != system or saved.get("budget") != budget):
        raise ValueError("Not a checkpoint for the requested system/budget.")
    expected = training_protocol(system, budget, saved["training_device"])
    if saved.get("protocol") != expected or saved.get("code_hash") != code_hash():
        raise ValueError("Checkpoint protocol, data or implementation changed; use its matching code and dataset.")
    if (saved.get("epochs_completed") != expected["epochs"]
            or saved.get("optimizer_steps") != expected["optimizer_steps"]
            or saved.get("processed_examples") != expected["processed_training_examples"]):
        raise ValueError("Checkpoint is not a completed training budget.")
    state = saved["model_state_dict"]
    if state_hash(state) != saved.get("model_state_sha256"):
        raise ValueError("Checkpoint model-state checksum is inconsistent.")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Checkpoint contains nonfinite model parameters or buffers.")
    return expected


def load_model(directory, system, budget, device, explicit_path=None):
    """Load only structured checkpoints saved by this workflow."""
    path = (Path(explicit_path).expanduser().resolve() if explicit_path is not None
            else directory / "training" / system / budget / "model.pth")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {system}/{budget} checkpoint: {path}. Train the model first or supply its explicit path.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    artifact_hash = sha256(path)
    protocol = _validate_checkpoint(saved, system, budget)
    state = saved["model_state_dict"]
    model = ValueNetwork(SPECS[system]["input_dim"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _, _, split = data_split(system, "full")
    if not (np.array_equal(model.x_mean.cpu().numpy(), split["mean"])
            and np.array_equal(model.x_std.cpu().numpy(), split["std"])):
        raise ValueError("Model normalization does not match the selected full dataset.")
    path_label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
    info = {"source": f"trained_{budget}", "path": path_label,
            "checkpoint_sha256": artifact_hash, "model_state_sha256": state_hash(state),
            "protocol": protocol, "formal_training": protocol["formal_training"],
            "training": compute_record(system, budget, saved, protocol)}
    return model, info


def compute_record(system, budget, saved, protocol):
    spec = SPECS[system]
    ntrain = protocol["train_count"]
    epochs = protocol["epochs"]
    return {"system": system, "source": f"trained_{budget}", "budget": budget,
            "paired_data_count": ntrain + protocol["held_count"],
            "train_count": ntrain, "held_count": protocol["held_count"],
            "epochs": epochs, "batch_size": spec["batch_size"],
            "optimizer_steps": epochs * int(np.ceil(ntrain / spec["batch_size"])),
            "processed_training_examples": epochs * ntrain,
            "loop_seconds": saved.get("loop_seconds"),
            "train_seconds": saved.get("train_seconds"),
            "initial_state_sha256": saved.get("init_state_sha256"),
            "training_device": saved.get("training_device"),
            "formal_training": protocol["formal_training"],
            "time_definition": saved.get("time_definition")}


def export_tex(target, summaries, study):
    """Only summaries of the current actual evaluation; no paper numbers embedded."""
    def fmt(value):
        return "--" if value is None else f"{value:.5g}"
    if study == "labels":
        header = r"System & Budget & Success & Mean $C$ & Mean $E_{\rm AUC}$ & Updates \\"
        columns = "lllrrr"
        lines = [" & ".join([s["system"].upper(), s["budget"],
                  f"{s['nn_successful']}/{s['test_cases']}", fmt(s["mean_cost_ratio"]),
                  fmt(s["mean_E_AUC"]), str(s["optimizer_steps"])]) + r" \\" for s in summaries]
    else:
        header = r"System & Scale/parameter & Success & Median $C$ & Mean $\|x(T)\|_2$ & Mean abs. HJB \\"
        columns = "lclrrr"
        lines = [" & ".join([s["system"].upper(), fmt(s["scale"] if study == "ood" else s["parameter_value"]),
                  f"{s['nn_successful']}/{s['test_cases']}", fmt(s["median_cost_ratio_successful"]),
                  fmt(s["mean_terminal_norm"]), fmt(s["mean_abs_hjb_residual"])]) + r" \\" for s in summaries]
    text = [r"\begin{tabular}{" + columns + "}", r"\toprule", header, r"\midrule", *lines,
            r"\bottomrule", r"\end{tabular}"]
    target = out(target / f"{study}_table.tex")
    target.write_text("\n".join(text) + "\n", encoding="utf-8")


def evaluate_study(directory, systems, study, requested_device, full_checkpoint=None, half_checkpoint=None):
    from runtime import choose_training_device
    # Validate every selected model before starting any long ODE rollout.
    loaded = {}
    for system in systems:
        device = choose_training_device(system, requested_device)
        loaded[system] = {"device": device, "full": load_model(directory, system, "full", device, full_checkpoint)}
        if study == "labels":
            loaded[system]["half"] = load_model(directory, system, "half", device, half_checkpoint)
            full, half = loaded[system]["full"][1], loaded[system]["half"][1]
            a, b = full["training"], half["training"]
            if a["initial_state_sha256"] != b["initial_state_sha256"] or a["training_device"] != b["training_device"]:
                raise ValueError("Full/half models must share initialization and training device.")
    target = out(directory / study)
    settings = {"experiment": EXPERIMENT, "study": study, "systems": list(systems),
                "evaluation": EVALUATION, "code_hash": code_hash(),
                "models": {s: {b: item[1] for b, item in loaded[s].items() if b != "device"} for s in systems},
                "environment": environment(), "device_selection": requested_device,
                "reference": "SDRE recomputed for the actual plant parameters",
                "statistics": "five common initial conditions; one initialization, not five independent training seeds"}
    metadata = target / "metadata.json"
    if metadata.exists() and json.loads(metadata.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing evaluation uses different settings/models; select a new run name.")
    write_json(metadata, settings)
    all_rows, summaries, compute = [], [], []
    for system in systems:
        device = loaded[system]["device"]
        for budget in (["full", "half"] if study == "labels" else ["full"]):
            model, info = loaded[system][budget]
            compute.append(info["training"])
            parameters = SPECS[system]["parameter_values"] if study == "dynamics" else [SPECS[system]["nominal_parameter"]]
            cases = fixed_cases(system, ood=study == "ood")
            for parameter in parameters:
                for scale in sorted({case["scale"] for case in cases}):
                    rows = []
                    for case in [case for case in cases if case["scale"] == scale]:
                        row = paired_case(directory, system, parameter, case, model, info, device, study == "labels")
                        row.update(formal_training=info["formal_training"])
                        rows.append(row)
                        all_rows.append(row)
                        write_csv(target / "cases_in_progress.csv", all_rows)
                    summary = summarize(rows)
                    summary.update(formal_training=info["formal_training"])
                    if study == "labels":
                        summary.update(info["training"])
                    summaries.append(summary)
    write_csv(target / "cases.csv", all_rows)
    write_csv(target / "summary.csv", summaries)
    write_csv(target / "compute_budget.csv", compute)
    write_json(target / "results.json", {"metadata": settings, "summaries": summaries, "cases": all_rows})
    export_tex(target, summaries, study)
    print(f"Saved actual {study} evaluation: {target}", flush=True)
    return summaries


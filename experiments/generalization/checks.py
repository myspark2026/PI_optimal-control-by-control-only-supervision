"""Small in-memory scientific-interface checks, requiring no data/checkpoints."""
from runtime import (ROOT, EVALUATION, SPECS, ValueNetwork, choose_training_device,
                     data_split, dataset_path, environment, fixed_cases, make_system,
                     np, out, ref, set_seed, state_hash, torch, training_terms, write_json)


def check_setup(directory, systems, requested_device):
    report = {"formal_training": False, "formal_evaluation": False,
              "environment": environment(), "systems": {}}
    try:
        out(ROOT / "outside_generalization.json")
    except ValueError:
        report["output_escape_rejected"] = True
    else:
        raise AssertionError("Output containment check failed.")
    for system in systems:
        device = choose_training_device(system, requested_device)
        set_seed(42)
        model = ValueNetwork(SPECS[system]["input_dim"]).to(device)
        states = (0.05 * torch.randn(8, SPECS[system]["input_dim"], device=device)).requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=SPECS[system]["learning_rate"])
        hamiltonian, control, _ = training_terms(system, model, states)
        loss = hamiltonian.square().mean() + control.square().mean()
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite test loss.")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError("Missing/nonfinite parameter gradients.")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()  # one in-memory diagnostic step; no checkpoint is saved
        sample = states.detach().cpu().numpy().astype(float)
        controls, residuals, drift_errors = [], [], []
        model.eval()
        for parameter in SPECS[system]["parameter_values"]:
            plant = make_system(system, parameter)
            drift_np = np.stack([plant.drift(state) for state in sample])
            drift_pt = plant.drift_torch(torch.as_tensor(sample, dtype=torch.float32, device=device)).detach().cpu().numpy()
            if not np.allclose(drift_np, drift_pt, atol=1e-4, rtol=2e-5):
                raise AssertionError("NumPy/Torch perturbed drifts disagree.")
            drift_errors.append(float(np.max(np.abs(drift_np - drift_pt))))
            controller = ref.LearnedValueController(model, plant, device, equilibrium_correction=False)
            controls.append(controller.control(sample[0]))
            residuals.append(controller.diagnostics(sample)["hjb"])
        if not all(np.array_equal(controls[0], item) for item in controls[1:]):
            raise AssertionError("Changing the plant changed the frozen policy mapping.")
        if np.allclose(residuals[0], residuals[-1], atol=1e-9, rtol=0):
            raise AssertionError("HJB diagnostic failed to respond to perturbed drift.")
        plant = make_system(system)
        controller = ref.LearnedValueController(model, plant, device, equilibrium_correction=False)
        x0 = fixed_cases(system)[0]["x0"] * 0.01
        short = {}
        for kind, policy in [("sdre", plant.sdre_control), ("neural", controller.control)]:
            rollout = ref.simulate_closed_loop(plant, policy, x0, horizon=0.02, dt=0.01,
                                               rtol=1e-6, atol=1e-8, blowup_norm=1e6)
            if not rollout.complete or len(rollout.t) != 3 or not np.isfinite(rollout.cumulative_cost).all():
                raise AssertionError("Two-step closed-loop interface check failed.")
            short[kind] = {"steps": 2, "cost": float(rollout.cumulative_cost[-1]),
                           "terminal_norm": float(np.linalg.norm(rollout.x[:, -1]))}
        item = {"device": str(device), "loss": float(loss.detach().cpu()),
                "drift_max_errors": drift_errors, "frozen_policy_invariant": True,
                "perturbed_hjb_changes": True, "short_rollouts": short,
                "dataset_present": dataset_path(system).is_file()}
        if dataset_path(system).is_file():
            _, _, full = data_split(system, "full")
            _, _, half = data_split(system, "half")
            assert set(half["train_indices"]).issubset(full["train_indices"])
            assert set(half["held_indices"]).issubset(full["held_indices"])
            assert not np.intersect1d(half["train_indices"], half["held_indices"]).size
            assert np.array_equal(full["mean"], half["mean"]) and np.array_equal(full["std"], half["std"])
            hashes = []
            for split in [full, half]:
                set_seed(42)
                initial = ValueNetwork(SPECS[system]["input_dim"])
                initial.x_mean.copy_(torch.as_tensor(split["mean"]))
                initial.x_std.copy_(torch.as_tensor(split["std"]))
                hashes.append(state_hash(initial.state_dict()))
            assert hashes[0] == hashes[1]
            item.update(train_counts=[len(full["train_indices"]), len(half["train_indices"])],
                        held_counts=[len(full["held_indices"]), len(half["held_indices"])],
                        shared_initial_state_sha256=hashes[0])
        report["systems"][system] = item
    write_json(directory / "check_setup.json", report)
    print("Small setup checks passed; no trained model or formal rollout was produced.", flush=True)
    return report

"""Nominal dynamics, training losses, and common-reference evaluation.

This module defines the control and value-discrepancy calculations.
"""
from functools import lru_cache
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

try:
    from . import cs_loss
except ImportError:
    import cs_loss


SPECS = {
    "cs": {"dimension": 40, "controls": 20, "samples": 20000, "epochs": 4000,
           "batch": 128, "lr": 2e-4, "test_size": .2, "horizon": 15.,
           "training_device": "cpu"},
    "ac": {"dimension": 51, "controls": 1, "samples": 10000, "epochs": 3000,
           "batch": 4096, "lr": .002, "test_size": .2, "horizon": 20.,
           "training_device": "cuda"},
}


@lru_cache(maxsize=1)
def ac_matrices():
    xi = np.linspace(0., 1., 51)
    coefficient = .1 / (1. / 50.) ** 2
    a = np.diag(np.full(51, -2. * coefficient))
    a += np.diag(np.full(50, coefficient), 1) + np.diag(np.full(50, coefficient), -1)
    a[0, 1] = a[-1, -2] = 2. * coefficient
    b = ((xi >= .6) & (xi <= .9)).astype(float).reshape(51, 1)
    return xi, a, b


def ac_training_matrices(device):
    _, a, _ = ac_matrices()
    # Use the float32 Torch grid for the training control mask.
    xi = torch.linspace(0., 1., 51, device=device)
    b = ((xi >= .6) & (xi <= .9)).float().reshape(51, 1)
    return torch.as_tensor(a, dtype=torch.float32, device=device), b


def value_terms(system, model, states, create_graph=True):
    if system == "cs":
        # CS builds its derivative graph during both training and
        # held-out evaluation.
        hamiltonian, gradient, value = cs_loss.compute_HJB_H(model, states, states.device)
        control = cs_loss.compute_u_from_gradV(gradient, states.device)
        return value, control, hamiltonian
    a, b = ac_training_matrices(states.device)
    states = states.requires_grad_(True)
    value = model(states)
    gradient = torch.autograd.grad(value, states, torch.ones_like(value),
                                   create_graph=create_graph)[0]
    projected = gradient @ b
    drift = states @ a.T + states * (1. - states.square())
    hamiltonian = (.02 * states.square().sum(1, keepdim=True)
                   + (gradient * drift).sum(1, keepdim=True) - 2.5 * projected.square())
    return value, -5. * projected, hamiltonian


def initial_state(system):
    if system == "cs":
        return np.r_[np.linspace(0., .2, 20), np.linspace(.2, .4, 20)]
    xi, _, _ = ac_matrices()
    return 1. + xi * (1. - xi)


def cs_interaction(position):
    p = 1. / (1. + (position[:, None] - position[None, :]) ** 2)
    np.fill_diagonal(p, 0.)
    a = p / 20.
    np.fill_diagonal(a, -p.sum(axis=1) / 20.)
    return a


def drift(system, state):
    if system == "cs":
        return np.r_[state[20:], cs_interaction(state[:20]) @ state[20:]]
    return ac_matrices()[1] @ state + state * (1. - state ** 2)


def controlled_drift(system, state, control):
    result = drift(system, state)
    if system == "cs":
        result[20:] += control
    else:
        result += ac_matrices()[2][:, 0] * float(control[0])
    return result


def sdre_control(system, state):
    if system == "cs":
        a = np.block([[np.zeros((20, 20)), np.eye(20)],
                      [np.zeros((20, 20)), cs_interaction(state[:20])]])
        b = np.vstack([np.zeros((20, 20)), np.eye(20)])
        p = solve_continuous_are(a, b, np.eye(40) / 20., np.eye(20) / 20.)
        return -20. * b.T @ p @ state
    _, base, b = ac_matrices()
    a = base + np.diag(1. - state ** 2)
    p = solve_continuous_are(a, b, .02 * np.eye(51), np.array([[.1]]))
    return (-10. * b.T @ p @ state).reshape(1)


def neural_control(system, model, state, is_policy=False):
    device = next(model.parameters()).device
    x = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    if is_policy:
        with torch.no_grad():
            return model(x).detach().cpu().numpy().reshape(-1)
    with torch.enable_grad():
        x = x.requires_grad_(True)
        gradient = torch.autograd.grad(model(x), x)[0].detach().cpu().numpy().reshape(-1)
    # Match evaluation's float32 neural gradient followed by float64 physical B.
    if system == "cs":
        return -10. * gradient.astype(float)[20:]
    return (-5. * ac_matrices()[2].T @ gradient.astype(float)).reshape(1)


def running_cost(system, states, controls):
    if system == "cs":
        return (np.square(states).sum(axis=1) + np.square(controls).sum(axis=1)) / 20.
    return .02 * np.square(states).sum(axis=1) + .1 * np.square(controls).sum(axis=1)


def simulate(system, controller):
    horizon, dt = SPECS[system]["horizon"], .01
    times = (np.arange(0., horizon + dt, dt) if system == "cs"
             else np.linspace(0., horizon, int(round(horizon / dt)) + 1))
    if len(times) != int(round(horizon / dt)) + 1 or not np.isclose(times[-1], horizon):
        raise RuntimeError("Unexpected nominal time grid.")
    states = np.zeros((len(times), SPECS[system]["dimension"]))
    controls = np.zeros((len(times), SPECS[system]["controls"]))
    states[0] = initial_state(system)
    for k in range(len(times) - 1):
        controls[k] = controller(states[k])
        if not np.isfinite(controls[k]).all():
            raise RuntimeError(f"Nonfinite control at interval {k}.")
        held = controls[k].copy()
        solution = solve_ivp(
            lambda _t, x: controlled_drift(system, x, held),
            (times[k], times[k + 1]), states[k], t_eval=[times[k + 1]],
            method="RK45", rtol=1e-3, atol=1e-6,
        )
        if (not solution.success or solution.y.shape != (SPECS[system]["dimension"], 1)
                or not np.isfinite(solution.y).all()):
            raise RuntimeError(f"Incomplete/nonfinite rollout at interval {k}: {solution.message}")
        states[k + 1] = solution.y[:, -1]
        if (k + 1) % 500 == 0:
            print(f"  {system} rollout: {k + 1}/{len(times) - 1}", flush=True)
    controls[-1] = controller(states[-1])
    running = running_cost(system, states, controls)
    if not np.isfinite(running).all():
        raise RuntimeError("Nonfinite full-horizon running costs.")
    areas = np.diff(times) * (running[:-1] + running[1:]) / 2.
    to_go = np.r_[np.cumsum(areas[::-1])[::-1], 0.]
    return {"times": times, "states": states, "controls": controls,
            "running_cost": running, "cost_to_go": to_go, "cost": float(areas.sum())}


def values_on_reference(model, reference):
    device = next(model.parameters()).device
    with torch.no_grad():
        values = np.concatenate([
            model(torch.as_tensor(reference["states"][start:start + 512],
                                  dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
            for start in range(0, len(reference["states"]), 512)
        ])
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite value predictions on the common SDRE trajectory.")
    times, target = reference["times"], reference["cost_to_go"]
    def trap(y):
        return float(np.sum(np.diff(times) * (y[:-1] + y[1:]) / 2.))
    denominator = trap(np.abs(target))
    if denominator <= 0.:
        raise RuntimeError("Invalid SDRE value-normalization denominator.")
    return trap(np.abs(values - target)) / denominator, values

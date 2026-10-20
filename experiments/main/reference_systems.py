"""Value networks, dynamical systems, and closed-loop trajectory utilities."""
from __future__ import annotations


import argparse


import csv


import hashlib


import json


import math


import sys


import time


import warnings


from dataclasses import asdict, dataclass


from datetime import datetime


from pathlib import Path


from typing import Callable, Iterable, Sequence


import numpy as np


import torch


import torch.nn as nn


from scipy.integrate import solve_ivp


from scipy.linalg import solve_continuous_are


class ValueNetwork(nn.Module):
    """Three-layer value network for gradient-based feedback control."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("x_std", torch.ones(input_dim, dtype=torch.float32))
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)
        self.activation = nn.Tanh()

    def _raw_forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.fc1(x_norm))
        h = self.activation(self.fc2(h))
        h = self.activation(self.fc3(h))
        return self.out(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.x_mean) / (self.x_std + 1.0e-8)
        zero = torch.zeros(
            (1, self.x_mean.numel()), device=x.device, dtype=x.dtype
        )
        zero_norm = (zero - self.x_mean) / (self.x_std + 1.0e-8)
        z = self._raw_forward(x_norm) - self._raw_forward(zero_norm)

        # Preserve the exact straight-through construction used in training:
        # forward value is ReLU(z), while its input gradient is the gradient of z.
        value_relu = torch.relu(z)
        return z + (value_relu - z).detach()


def load_value_network(
    checkpoint_path: Path, input_dim: int, device: torch.device
) -> ValueNetwork:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ValueNetwork(input_dim)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class ControlSystem:
    name: str
    state_dim: int
    control_dim: int
    Q: np.ndarray
    R: np.ndarray
    B: np.ndarray

    def drift(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def drift_torch(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sdre_control(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.drift(x) + self.B @ u

    def running_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        return float(x @ self.Q @ x + u @ self.R @ u)


class AllenCahnSystem(ControlSystem):
    def __init__(self, d: int = 51, sigma: float = 0.1):
        self.name = "Allen-Cahn"
        self.state_dim = d
        self.control_dim = 1
        self.sigma = sigma
        self.xi = np.linspace(0.0, 1.0, d)
        self.A_base = self._build_a_base()
        self.B = ((self.xi >= 0.6) & (self.xi <= 0.9)).astype(float)[:, None]
        self.Q = 0.02 * np.eye(d)
        self.R = np.array([[0.1]])
        self.R_inv = np.array([[10.0]])

    def _build_a_base(self) -> np.ndarray:
        h = 1.0 / (self.state_dim - 1)
        coeff = self.sigma / (h * h)
        a = np.zeros((self.state_dim, self.state_dim))
        a[0, 0], a[0, 1] = -2.0 * coeff, 2.0 * coeff
        for i in range(1, self.state_dim - 1):
            a[i, i - 1] = coeff
            a[i, i] = -2.0 * coeff
            a[i, i + 1] = coeff
        a[-1, -2], a[-1, -1] = 2.0 * coeff, -2.0 * coeff
        return a

    def drift(self, x: np.ndarray) -> np.ndarray:
        return self.A_base @ x + x * (1.0 - x * x)

    def drift_torch(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.as_tensor(self.A_base, dtype=x.dtype, device=x.device)
        return x @ a.T + x * (1.0 - x * x)

    def sdre_control(self, x: np.ndarray) -> np.ndarray:
        a_x = self.A_base + np.diag(1.0 - x * x)
        p = solve_continuous_are(a_x, self.B, self.Q, self.R)
        u = -self.R_inv @ self.B.T @ p @ x
        if not np.all(np.isfinite(u)):
            raise FloatingPointError("Allen-Cahn SDRE returned a non-finite control")
        return np.asarray(u, dtype=float).reshape(1)


class CuckerSmaleSystem(ControlSystem):
    def __init__(self, n_agents: int = 20):
        self.name = "Cucker-Smale"
        self.n_agents = n_agents
        self.state_dim = 2 * n_agents
        self.control_dim = n_agents
        self.Q = np.eye(self.state_dim) / n_agents
        self.R = np.eye(self.control_dim) / n_agents
        self.R_inv = n_agents * np.eye(self.control_dim)
        self.B = np.vstack(
            [np.zeros((n_agents, n_agents)), np.eye(n_agents)]
        )

    def _interaction_matrix(self, y: np.ndarray) -> np.ndarray:
        differences = y[:, None] - y[None, :]
        weights = 1.0 / (1.0 + differences * differences)
        np.fill_diagonal(weights, 0.0)
        a = weights / self.n_agents
        diagonal = -np.sum(weights, axis=1) / self.n_agents
        np.fill_diagonal(a, diagonal)
        return a

    def drift(self, x: np.ndarray) -> np.ndarray:
        y = x[: self.n_agents]
        velocity = x[self.n_agents :]
        a = self._interaction_matrix(y)
        return np.concatenate([velocity, a @ velocity])

    def drift_torch(self, x: torch.Tensor) -> torch.Tensor:
        y = x[:, : self.n_agents]
        velocity = x[:, self.n_agents :]
        differences = y.unsqueeze(2) - y.unsqueeze(1)
        weights = 1.0 / (1.0 + differences * differences)
        eye = torch.eye(
            self.n_agents, dtype=torch.bool, device=x.device
        ).unsqueeze(0)
        weights = weights.masked_fill(eye, 0.0)
        a = weights / self.n_agents
        diagonal = -weights.sum(dim=2) / self.n_agents
        a = a - torch.diag_embed(torch.diagonal(a, dim1=1, dim2=2))
        a = a + torch.diag_embed(diagonal)
        acceleration = torch.bmm(a, velocity.unsqueeze(2)).squeeze(2)
        return torch.cat([velocity, acceleration], dim=1)

    def sdre_control(self, x: np.ndarray) -> np.ndarray:
        y = x[: self.n_agents]
        a_small = self._interaction_matrix(y)
        zero = np.zeros((self.n_agents, self.n_agents))
        identity = np.eye(self.n_agents)
        a = np.block([[zero, identity], [zero, a_small]])
        p = solve_continuous_are(a, self.B, self.Q, self.R)
        u = -self.R_inv @ self.B.T @ p @ x
        if not np.all(np.isfinite(u)):
            raise FloatingPointError("Cucker-Smale SDRE returned a non-finite control")
        return np.asarray(u, dtype=float).reshape(self.control_dim)


class LearnedValueController:
    def __init__(
        self,
        model: ValueNetwork,
        system: ControlSystem,
        device: torch.device,
        equilibrium_correction: bool = False,
    ):
        self.model = model
        self.system = system
        self.device = device
        self._b_torch = torch.as_tensor(
            system.B, dtype=torch.float32, device=device
        )
        self._q_torch = torch.as_tensor(
            system.Q, dtype=torch.float32, device=device
        )
        self._r_torch = torch.as_tensor(
            system.R, dtype=torch.float32, device=device
        )
        self._r_inv_torch = torch.linalg.inv(self._r_torch)
        self.equilibrium_correction = equilibrium_correction
        self.u_origin_raw = self.raw_control(np.zeros(system.state_dim, dtype=float))
        self._u_origin_torch = torch.as_tensor(
            self.u_origin_raw, dtype=torch.float32, device=device
        )

    def value_gradient(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        states_2d = np.atleast_2d(states).astype(np.float32, copy=False)
        x = torch.as_tensor(states_2d, device=self.device).requires_grad_(True)
        with torch.enable_grad():
            value = self.model(x)
            gradient = torch.autograd.grad(value.sum(), x, create_graph=False)[0]
        return (
            value.detach().cpu().numpy().reshape(-1),
            gradient.detach().cpu().numpy(),
        )

    def raw_control(self, x: np.ndarray) -> np.ndarray:
        _, gradient = self.value_gradient(x)
        gradient_tensor = torch.as_tensor(
            gradient, dtype=torch.float32, device=self.device
        )
        u = -0.5 * (gradient_tensor @ self._b_torch) @ self._r_inv_torch
        return u[0].detach().cpu().numpy().astype(float, copy=False)

    def control(self, x: np.ndarray) -> np.ndarray:
        u = self.raw_control(x)
        if self.equilibrium_correction:
            u = u - self.u_origin_raw
        return u

    def diagnostics(self, states: np.ndarray) -> dict[str, np.ndarray]:
        states_2d = np.atleast_2d(states).astype(np.float32, copy=False)
        x = torch.as_tensor(states_2d, device=self.device).requires_grad_(True)
        with torch.enable_grad():
            value = self.model(x)
            gradient = torch.autograd.grad(value.sum(), x, create_graph=False)[0]
            open_loop_drift = self.system.drift_torch(x)
            u_raw = -0.5 * (gradient @ self._b_torch) @ self._r_inv_torch
            u_actual = (
                u_raw - self._u_origin_torch
                if self.equilibrium_correction
                else u_raw
            )
            state_cost = torch.sum((x @ self._q_torch) * x, dim=1)
            raw_control_cost = torch.sum(
                (u_raw @ self._r_torch) * u_raw, dim=1
            )
            actual_control_cost = torch.sum(
                (u_actual @ self._r_torch) * u_actual, dim=1
            )
            grad_f = torch.sum(gradient * open_loop_drift, dim=1)
            grad_bu_raw = torch.sum(
                (gradient @ self._b_torch) * u_raw, dim=1
            )
            grad_bu_actual = torch.sum(
                (gradient @ self._b_torch) * u_actual, dim=1
            )
            # The HJB residual always uses the Hamiltonian-minimizing raw
            # gradient policy.  The policy Bellman residual uses the control
            # that is actually applied, which may include the origin correction.
            hjb = state_cost + raw_control_cost + grad_f + grad_bu_raw
            policy_bellman = (
                state_cost + actual_control_cost + grad_f + grad_bu_actual
            )
            v_dot_closed_loop = grad_f + grad_bu_actual
            hjb_scale = (
                torch.abs(state_cost)
                + torch.abs(raw_control_cost)
                + torch.abs(grad_f + grad_bu_raw)
                + 1.0e-12
            )
            bellman_scale = (
                torch.abs(state_cost)
                + torch.abs(actual_control_cost)
                + torch.abs(v_dot_closed_loop)
                + 1.0e-12
            )

        return {
            "value": value.detach().cpu().numpy().reshape(-1),
            "gradient": gradient.detach().cpu().numpy(),
            "u_nn": u_actual.detach().cpu().numpy(),
            "u_raw": u_raw.detach().cpu().numpy(),
            "hjb": hjb.detach().cpu().numpy(),
            "hjb_relative": (torch.abs(hjb) / hjb_scale).detach().cpu().numpy(),
            "policy_bellman": policy_bellman.detach().cpu().numpy(),
            "policy_bellman_relative": (
                torch.abs(policy_bellman) / bellman_scale
            ).detach().cpu().numpy(),
            "v_dot": v_dot_closed_loop.detach().cpu().numpy(),
        }


@dataclass
class Rollout:
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    running_cost: np.ndarray
    cumulative_cost: np.ndarray
    dt: float
    complete: bool
    failure_reason: str


def simulate_closed_loop(
    system: ControlSystem,
    controller: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    horizon: float,
    dt: float,
    rtol: float,
    atol: float,
    blowup_norm: float,
) -> Rollout:
    n_steps = int(round(horizon / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    x = np.full((system.state_dim, n_steps), np.nan, dtype=float)
    u = np.full((system.control_dim, n_steps), np.nan, dtype=float)
    x[:, 0] = np.asarray(x0, dtype=float)
    last_valid = 0
    failure_reason = ""

    for k in range(n_steps - 1):
        try:
            u_current = np.asarray(controller(x[:, k]), dtype=float).reshape(
                system.control_dim
            )
            if not np.all(np.isfinite(u_current)):
                raise FloatingPointError("controller returned non-finite values")
            u[:, k] = u_current

            solution = solve_ivp(
                lambda _time, state: system.dynamics(state, u_current),
                (t[k], t[k + 1]),
                x[:, k],
                t_eval=[t[k + 1]],
                method="RK45",
                rtol=rtol,
                atol=atol,
            )
            if not solution.success or solution.y.shape[1] != 1:
                raise RuntimeError(solution.message)
            next_state = solution.y[:, -1]
            next_norm = float(np.linalg.norm(next_state))
            if not np.all(np.isfinite(next_state)):
                raise FloatingPointError("integrator returned non-finite state")
            if next_norm > blowup_norm:
                raise FloatingPointError(
                    f"state norm {next_norm:.3e} exceeded {blowup_norm:.3e}"
                )
            x[:, k + 1] = next_state
            last_valid = k + 1
        except Exception as exc:  # failure is part of the robustness result
            failure_reason = f"{type(exc).__name__}: {exc}"
            break

    complete = last_valid == n_steps - 1
    valid_steps = last_valid + 1
    t = t[:valid_steps]
    x = x[:, :valid_steps]
    u = u[:, :valid_steps]

    try:
        u[:, -1] = np.asarray(controller(x[:, -1]), dtype=float).reshape(
            system.control_dim
        )
    except Exception as exc:
        if complete:
            complete = False
            failure_reason = f"terminal control failed: {type(exc).__name__}: {exc}"

    running_cost = np.full(valid_steps, np.nan, dtype=float)
    for k in range(valid_steps):
        if np.all(np.isfinite(u[:, k])):
            running_cost[k] = system.running_cost(x[:, k], u[:, k])

    cumulative_cost = np.zeros(valid_steps, dtype=float)
    if valid_steps > 1:
        increments = 0.5 * (running_cost[:-1] + running_cost[1:]) * dt
        cumulative_cost[1:] = np.cumsum(increments)

    return Rollout(
        t=t,
        x=x,
        u=u,
        running_cost=running_cost,
        cumulative_cost=cumulative_cost,
        dt=dt,
        complete=complete,
        failure_reason=failure_reason,
    )


def allen_initial_conditions(
    scales: Sequence[float],
    n_variants: int,
    perturbation_amplitude: float,
    include_negative: bool,
) -> list[dict[str, object]]:
    xi = np.linspace(0.0, 1.0, 51)
    base = 1.0 + (1.0 - xi) * xi
    polarities = [1.0, -1.0] if include_negative else [1.0]
    cases: list[dict[str, object]] = []

    for variant in range(n_variants):
        if variant == 0:
            perturbation = np.zeros_like(xi)
        else:
            phase = 2.0 * np.pi * (variant - 1) / max(1, n_variants - 1)
            perturbation = perturbation_amplitude * (
                0.65 * np.cos(2.0 * np.pi * xi + phase)
                + 0.35 * np.cos(4.0 * np.pi * xi - 0.5 * phase)
            )
            # Keep the OOD direction attributable to the baseline/range scaling.
            perturbation = perturbation - np.mean(perturbation)
        profile = base + perturbation

        for polarity in polarities:
            for scale in scales:
                x0 = polarity * scale * profile
                cases.append(
                    {
                        "case_family": "scaled_quadratic_fourier",
                        "case_id": (
                            f"allen_s{scale:g}_v{variant}_"
                            f"{'pos' if polarity > 0 else 'neg'}"
                        ),
                        "scale": float(scale),
                        "variant": variant,
                        "polarity": int(polarity),
                        "ood_by_design": True,
                        "x0": x0,
                    }
                )
    return cases


def cucker_initial_conditions(
    scales: Sequence[float],
    n_directions: int,
    seed: int,
    include_fixed: bool,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    cases: list[dict[str, object]] = []
    n_agents = 20

    if include_fixed:
        x0 = np.concatenate(
            [np.linspace(0.0, 0.2, n_agents), np.linspace(0.2, 0.4, n_agents)]
        )
        cases.append(
            {
                "case_family": "fixed_nominal",
                "case_id": "cucker_fixed_nominal",
                "scale": "nominal",
                "variant": -1,
                "polarity": 0,
                "ood_by_design": False,
                "x0": x0,
            }
        )

    # The same random directions are reused across scales, giving paired OOD tests.
    for direction in range(n_directions):
        position = rng.uniform(-1.0, 1.0, n_agents)
        velocity = rng.uniform(-1.0, 1.0, n_agents)
        position /= max(np.max(np.abs(position)), 1.0e-12)
        velocity /= max(np.max(np.abs(velocity)), 1.0e-12)
        unit_box_direction = np.concatenate([position, velocity])

        for scale in scales:
            x0 = float(scale) * unit_box_direction
            cases.append(
                {
                    "case_family": "radial_box_scaling",
                    "case_id": f"cucker_s{scale:g}_d{direction}",
                    "scale": float(scale),
                    "variant": direction,
                    "polarity": 0,
                    "ood_by_design": bool(scale > 1.0 + 1.0e-12),
                    "x0": x0,
                }
            )
    return cases



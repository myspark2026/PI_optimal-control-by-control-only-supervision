"""Nominal main-text control, state and value comparisons; training is explicit."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

PACKAGE = Path(__file__).resolve().parents[2]
for key in ['TEMP', 'TMP', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH', 'TORCH_HOME', 'XDG_CACHE_HOME']:
    path = PACKAGE / 'outputs' / '.runtime' / key.lower()
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
from scipy.linalg import solve_continuous_are
import reference_systems as ref


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def choose_device(value):
    if value == 'original':
        return torch.device('cpu')  # Nominal trajectory evaluation uses CPU by default.
    if value == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; no silent device fallback was used.')
    return torch.device(value)


def system_and_case(system):
    if system == 'cs':
        return ref.CuckerSmaleSystem(20), np.r_[np.linspace(0, 0.2, 20), np.linspace(0.2, 0.4, 20)], 15.0
    xi = np.linspace(0, 1, 51)
    return ref.AllenCahnSystem(51, sigma=0.1), 1 + xi * (1 - xi), 20.0


def remaining_cost(rollout):
    increments = np.diff(rollout.t) * (rollout.running_cost[:-1] + rollout.running_cost[1:]) / 2
    return np.r_[np.cumsum(increments[::-1])[::-1], 0.0]


def values(model, states, device):
    with torch.no_grad():
        return model(torch.as_tensor(states.T, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)


def sdre_values(plant, states, system):
    result = []
    for index, x in enumerate(states.T):
        if system == 'cs':
            small = plant._interaction_matrix(x[:20])
            zeros = np.zeros((20, 20))
            a = np.block([[zeros, np.eye(20)], [zeros, small]])
        else:
            a = plant.A_base + np.diag(1 - x * x)
        p = solve_continuous_are(a, plant.B, plant.Q, plant.R)
        result.append(float(x @ p @ x))
        if index % 1000 == 0:
            print(f'{system}: SDRE value proxy {index + 1}/{states.shape[1]}', flush=True)
    return np.asarray(result)


def benchmark(controller, plant, state, device, repetitions):
    for _ in range(5):
        controller.control(state)
        plant.sdre_control(state)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repetitions):
        controller.control(state)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    neural_seconds = (time.perf_counter() - started) / repetitions
    started = time.perf_counter()
    for _ in range(repetitions):
        plant.sdre_control(state)
    sdre_seconds = (time.perf_counter() - started) / repetitions
    return {'repetitions': repetitions, 'seconds_per_neural_control': neural_seconds,
            'seconds_per_sdre_control': sdre_seconds, 'measured_speedup': sdre_seconds / neural_seconds,
            'device': str(device), 'note': 'Controller timing measured in this run.'}


def figures(directory, system, neural, baseline, nn_value, sdre_value, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / (name + '.pdf'))
        fig.savefig(directory / (name + '.png'), dpi=160)
        plt.close(fig)
    count = 2 if system == 'cs' else 1
    for label, run in [('neural', neural), ('sdre', baseline)]:
        if system == 'cs':
            for name, data in [('positions', run.x[:20]), ('velocities', run.x[20:])]:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(run.t, data.T, linewidth=0.7)
                ax.set(xlabel='Time', ylabel=name.capitalize())
                save(fig, f'{label}_{name}')
        else:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(projection='3d')
            time_grid, space_grid = np.meshgrid(run.t[::5], np.linspace(0, 1, 51))
            ax.plot_surface(time_grid, space_grid, run.x[:, ::5], cmap='viridis', linewidth=0)
            ax.set(xlabel='Time', ylabel='Space', zlabel='State')
            save(fig, label + '_state')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, label, run in zip(axes, ['Neural', 'SDRE'], [neural, baseline]):
        ax.plot(run.t, run.u.T, linewidth=0.7)
        ax.set(title=label, xlabel='Time', ylabel='Control')
    save(fig, 'controls')
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(neural.t, nn_value, label='NN value on NN trajectory')
    ax.plot(baseline.t, sdre_value, label='SDRE value proxy on SDRE trajectory')
    ax.plot(baseline.t, target, '--', label='Realized SDRE cost-to-go')
    ax.set(xlabel='Time', ylabel='Value / remaining cost')
    ax.legend(fontsize=8)
    save(fig, 'value_comparison')


def check(system, device):
    torch.manual_seed(42)
    plant, x0, _ = system_and_case(system)
    model = ref.ValueNetwork(plant.state_dim).to(device)
    controller = ref.LearnedValueController(model, plant, device, equilibrium_correction=False)
    for label, feedback in [('sdre', plant.sdre_control), ('random_untrained_neural', controller.control)]:
        run = ref.simulate_closed_loop(plant, feedback, x0, horizon=0.02, dt=0.01,
                                      rtol=1e-3, atol=1e-6, blowup_norm=1e6)
        if not run.complete or len(run.t) != 3 or not np.isfinite(run.cumulative_cost).all():
            raise RuntimeError('The short nominal ODE interface check failed.')
    print(json.dumps({'system': system, 'passed': True, 'horizon': 0.02,
                      'diagnostic_only': True, 'trained_model_used': False}, indent=2))


def evaluate(args, system):
    device = choose_device(args.device)
    plant, x0, horizon = system_and_case(system)
    explicit = getattr(args, 'checkpoint_' + system)
    path = explicit if explicit is not None else PACKAGE / 'outputs' / 'generalization' / args.run_name / 'training' / system / 'full' / 'model.pth'
    if not path.is_file():
        raise FileNotFoundError(f'Missing full-data main-text model: {path}. Train the main model first or explicitly supply --checkpoint-{system}.')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(saved, dict) and (saved.get('diagnostic_only') or saved.get('debug_protocol')):
        raise ValueError('A diagnostic/debug checkpoint cannot be used for full-budget evaluation.')
    if isinstance(saved, dict) and 'epochs_completed' in saved and saved['epochs_completed'] != (4000 if system == 'cs' else 3000):
        raise ValueError('The main-text checkpoint has not completed the declared epoch budget.')
    protocol_verified = False
    if isinstance(saved, dict) and 'model_state_dict' in saved:
        recorded = saved.get('protocol', {})
        if not isinstance(recorded, dict):
            raise ValueError('The checkpoint protocol is not a valid mapping.')
        if recorded.get('formal_training') is False:
            raise ValueError('The checkpoint is marked as diagnostic training.')
        expected = {'system': system, 'budget': 'full', 'gamma': 1.0,
                    'test_size': 0.2 if system == 'cs' else 0.1,
                    'train_count': 16000 if system == 'cs' else 9000}
        for key, value in expected.items():
            actual = recorded.get(key, saved.get(key))
            if key == 'gamma' and actual is not None:
                actual = float(actual)
            if actual != value:
                raise ValueError(f'Main-text checkpoint {key} mismatch: expected {value!r}, found {actual!r}. Do not substitute half-data, AC 80/20 or other-loss models.')
        protocol_verified = True
    model = ref.load_value_network(path, plant.state_dim, device)
    controller = ref.LearnedValueController(model, plant, device, equilibrium_correction=False)
    directory = PACKAGE / 'outputs' / 'main' / args.run_name / system
    if (directory / 'metrics.json').exists():
        raise FileExistsError(f'Existing main results are preserved: {directory}; choose a new --run-name.')
    directory.mkdir(parents=True, exist_ok=True)
    settings = {'study': 'nominal_main_text', 'system': system, 'horizon': horizon, 'dt': 0.01,
                'rtol': 1e-3, 'atol': 1e-6, 'equilibrium_correction': False,
                'x0': x0.tolist(), 'checkpoint_sha256': sha(path), 'device': str(device),
                'checkpoint_training_metadata_verified': protocol_verified,
                'checkpoint_metadata_note': 'Structured full-data gamma=1 protocol verified.' if protocol_verified else 'Raw weights contain no training metadata; the supplied model training protocol cannot be independently verified here.',
                'note': 'Nominal closed-loop evaluation includes the exact endpoint T.'}
    (directory / 'protocol.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    runs = []
    for label, feedback in [('SDRE', plant.sdre_control), ('neural', controller.control)]:
        print(f'{system}: running nominal {label} trajectory, T={horizon:g}', flush=True)
        run = ref.simulate_closed_loop(plant, feedback, x0, horizon=horizon, dt=0.01,
                                      rtol=1e-3, atol=1e-6, blowup_norm=1e6)
        np.savez_compressed(directory / f'{label.lower()}_rollout.npz', t=run.t, x=run.x, u=run.u,
                            running_cost=run.running_cost, cumulative_cost=run.cumulative_cost)
        if not run.complete or not all(np.isfinite(a).all() for a in [run.x, run.u, run.running_cost, run.cumulative_cost]):
            (directory / 'failure.json').write_text(json.dumps({'controller': label, 'failure_reason': run.failure_reason,
                'completed': False, 'full_horizon_cost_assigned': False}), encoding='utf-8')
            raise RuntimeError(f'{label} did not complete the nominal test; partial trajectory was retained.')
        runs.append(run)
    baseline, neural = runs
    target = remaining_cost(baseline)
    nn_value = values(model, neural.x, device)
    common_value = values(model, baseline.x, device)
    proxy = sdre_values(plant, baseline.x, system)
    denominator = float(np.trapezoid(np.abs(target), baseline.t))
    error = float(np.trapezoid(np.abs(common_value - target), baseline.t) / denominator)
    np.savez_compressed(directory / 'value_curves.npz', t=baseline.t, nn_value_on_neural=nn_value,
                        nn_value_on_sdre=common_value, sdre_value_proxy=proxy, sdre_cost_to_go=target)
    result = {'protocol': settings, 'neural_cost': float(neural.cumulative_cost[-1]),
              'sdre_cost': float(baseline.cumulative_cost[-1]),
              'cost_ratio': float(neural.cumulative_cost[-1] / baseline.cumulative_cost[-1]),
              'E_AUC_on_common_sdre_trajectory': error,
              'terminal_norm': float(np.linalg.norm(neural.x[:, -1])),
              'completed': True, 'data_provenance': 'Training metadata are available when recorded in the selected checkpoint.'}
    if args.benchmark:
        result['fresh_local_timing'] = benchmark(controller, plant, x0, device, args.benchmark)
    (directory / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    figures(directory, system, neural, baseline, nn_value, proxy, target)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['check', 'train', 'evaluate'], default='check')
    parser.add_argument('--system', choices=['both', 'cs', 'ac'], default='both')
    parser.add_argument('--device', choices=['original', 'cpu', 'cuda'], default='original')
    parser.add_argument('--run-name', default='main_protocol')
    parser.add_argument('--checkpoint-cs', type=Path)
    parser.add_argument('--checkpoint-ac', type=Path)
    parser.add_argument('--benchmark', type=int, default=0, help='Optional number of fresh local timing repetitions.')
    parser.add_argument('--epochs', type=int, help='Training-only diagnostic epoch override; use a separate run name.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name) or args.run_name in {'.', '..'}:
        parser.error('Use a simple ASCII run name without path separators.')
    if args.benchmark < 0:
        parser.error('--benchmark must be nonnegative.')
    if args.stage == 'train':
        command = [sys.executable, '-B', str(PACKAGE / 'experiments/generalization/run.py'),
                   '--stage', 'train', '--study', 'labels', '--budget', 'full', '--system', args.system,
                   '--run-name', args.run_name, '--device', args.device]
        if args.epochs is not None:
            command.extend(['--epochs', str(args.epochs)])
        raise SystemExit(subprocess.call(command, cwd=PACKAGE))
    if args.epochs is not None:
        parser.error('--epochs is only valid for explicit training.')
    for system in (['cs', 'ac'] if args.system == 'both' else [args.system]):
        if args.stage == 'check':
            check(system, choose_device(args.device))
        else:
            evaluate(args, system)


if __name__ == '__main__':
    main()

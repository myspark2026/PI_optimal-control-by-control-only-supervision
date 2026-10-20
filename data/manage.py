"""Generate SDRE training data or import local simulation datasets."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

PACKAGE = Path(__file__).resolve().parents[1]
for key in ['TEMP', 'TMP', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH', 'TORCH_HOME', 'XDG_CACHE_HOME']:
    path = PACKAGE / 'outputs' / '.runtime' / key.lower()
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from scipy.io import loadmat, savemat
from generator_models import CuckerDataGenerator, AllenCahnDataGenerator

SPECS = {'cs': {'dimension': 40, 'controls': 20, 'trajectories': 400, 'points': 50, 'horizon': 10.0, 'dt': 0.2},
         'ac': {'dimension': 51, 'controls': 1, 'trajectories': 50, 'points': 200, 'horizon': 2.0, 'dt': 0.01}}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def rows(value, n, width, label):
    value = np.asarray(value, dtype=float)
    if value.shape == (width, n):
        value = value.T
    if value.shape != (n, width) or not np.isfinite(value).all():
        raise ValueError(f'Invalid {label}: {value.shape}; expected {(n, width)}')
    return value

def validate(path, system, require_full=True):
    spec = SPECS[system]
    raw = loadmat(path)
    x = np.asarray(raw['X'], dtype=float)
    if x.ndim != 2 or x.shape[1] != spec['dimension'] or not np.isfinite(x).all():
        raise ValueError(f'Invalid X array in {path}')
    n = len(x)
    if require_full and n != spec['trajectories'] * spec['points']:
        raise ValueError(f'Expected the full {system} protocol sample count, got {n}.')
    u = rows(raw['U_all'], n, spec['controls'], 'U_all')
    dv = rows(raw['DV_all'], n, spec['dimension'], 'DV_all')
    value = rows(raw['V_all'], n, 1, 'V_all')
    if system == 'cs':
        from_dv = -10.0 * dv[:, 20:]
    else:
        xi = np.linspace(0, 1, 51)
        b = ((xi >= 0.6) & (xi <= 0.9)).astype(float)
        from_dv = -5.0 * dv @ b[:, None]
    mismatch = float(np.max(np.abs(from_dv - u)))
    tolerance = 1e-7 + 1e-6 * float(np.max(np.abs(from_dv)))
    return {'samples': n, 'dimension': spec['dimension'], 'controls': spec['controls'],
            'stored_control_vs_sdre_gradient_max_error': mismatch,
            'stored_controls_consistent': mismatch <= tolerance}

def record(path, payload):
    payload = dict(payload, file=path.name, sha256=sha(path), bytes=path.stat().st_size)
    # Never record private absolute source paths in a public dataset manifest.
    path.with_suffix('.metadata.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')

def generate(system, seed, diagnostics=False):
    spec = SPECS[system]
    if diagnostics:
        output = PACKAGE / 'outputs' / 'checks' / 'data_generation'
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = PACKAGE / 'data'
    destination = output / f'{system}_main.mat'
    noise_destination = output / f'{system}_noise.mat'
    if diagnostics and destination.is_file() and noise_destination.is_file():
        previous = json.loads(destination.with_suffix('.metadata.json').read_text(encoding='utf-8'))
        if previous.get('diagnostic_only') and previous.get('seed') == seed:
            check = validate(destination, system, require_full=False)
            if check['samples'] == 3 and check['stored_controls_consistent']:
                print(json.dumps({'system': system, 'passed': True, 'diagnostic_only': True,
                                  'reused_existing_short_check': True, 'validation': check}, indent=2))
                return
    if destination.exists() or noise_destination.exists():
        raise FileExistsError(f'Inputs already exist: {destination}. Existing datasets are never overwritten.')
    np.random.seed(seed)
    horizon = 0.02 if diagnostics else spec['horizon']
    dt = 0.01 if diagnostics else spec['dt']
    generator = CuckerDataGenerator(Na=20, T=horizon, dt=dt) if system == 'cs' else AllenCahnDataGenerator(d=51, T=horizon, dt=dt)
    kwargs = {'n_trajectories': 1 if diagnostics else spec['trajectories'],
              'n_samples_per_traj': 3 if diagnostics else spec['points']}
    if system == 'cs':
        kwargs.update(pos_range=(-1, 1), vel_range=(-1, 1))
    else:
        kwargs.update(state_range=(-2, 2))
    x, v, dv, u = generator.generate_training_data(**kwargs)
    if not all(np.isfinite(a).all() for a in [x, v, dv, u]):
        raise ValueError('Generated data contain nonfinite values; not saved.')
    generator.save_data(str(destination), x, v, dv, u)
    check = validate(destination, system, require_full=not diagnostics)
    if not check['stored_controls_consistent']:
        raise ValueError('Generated controls do not satisfy the declared SDRE label relation.')
    shutil.copyfile(destination, noise_destination)
    info = {'origin': 'generated', 'seed': seed, 'system': system,
            'diagnostic_only': diagnostics, 'horizon': horizon, 'control_update_interval': dt,
            'trajectories': kwargs['n_trajectories'], 'points_per_trajectory': kwargs['n_samples_per_traj'],
            'license': 'CC-BY-4.0', 'validation': check}
    record(destination, info)
    record(noise_destination, dict(info, dataset_role='noise'))
    print(json.dumps({'system': system, 'diagnostic_only': diagnostics, 'validation': check,
                      'output': destination.relative_to(PACKAGE).as_posix()}, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    gen = sub.add_parser('generate', help='Generate the full paper-size synthetic datasets; may be slow.')
    gen.add_argument('--system', choices=['both', 'cs', 'ac'], default='both')
    gen.add_argument('--seed', type=int, default=42)
    check = sub.add_parser('check', help='Three points on one short trajectory per system; never writes to data/.')
    check.add_argument('--system', choices=['both', 'cs', 'ac'], default='both')
    check.add_argument('--seed', type=int, default=42)
    imp = sub.add_parser('import', help='Import datasets you already possess; no download or publication.')
    for key in ['cs-main', 'ac-main', 'cs-noise', 'ac-noise']:
        imp.add_argument('--' + key, type=Path)
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--system', choices=['both', 'cs', 'ac'], default='both')
    args = parser.parse_args()
    if args.stage in ['generate', 'check']:
        for system in (['cs', 'ac'] if args.system == 'both' else [args.system]):
            generate(system, args.seed, args.stage == 'check')
    elif args.stage == 'import':
        requested = {key: getattr(args, key) for key in ['cs_main', 'ac_main', 'cs_noise', 'ac_noise'] if getattr(args, key) is not None}
        if not requested:
            parser.error('Specify at least one source dataset.')
        for key, source in requested.items():
            target = PACKAGE / 'data' / (key + '.mat')
            if target.exists():
                raise FileExistsError(f'Not overwriting {target}.')
            validation = validate(source, key[:2])
            if key == 'cs_noise' and not validation['stored_controls_consistent']:
                raise ValueError('CS noise needs clean labels consistent with DV_all.')
            shutil.copyfile(source, target)
            record(target, {'origin': 'local_import', 'system': key[:2], 'source_sha256': sha(source),
                            'validation': validation, 'license': 'CC-BY-4.0'})
            print(f'Imported and validated: data/{target.name}')
    else:
        for system in (['cs', 'ac'] if args.system == 'both' else [args.system]):
            for kind in ['main', 'noise']:
                path = PACKAGE / 'data' / f'{system}_{kind}.mat'
                print(json.dumps({'file': path.name, 'present': path.exists(),
                                  'validation': validate(path, system) if path.exists() else None}, indent=2))

if __name__ == '__main__':
    main()

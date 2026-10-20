"""Portable dispatcher. Explicitly choose generation, training or evaluation."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def execute(relative, args):
    subprocess.run([sys.executable, '-B', str(ROOT / relative), *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='For study-specific arguments: python reproduce.py <study> --help')
    parser.add_argument('study', choices=['check', 'data', 'main', 'ablation', 'noise', 'ood', 'labels', 'dynamics'])
    # Parse only the dispatch token so `<study> --help` reaches that study.
    args = parser.parse_args(sys.argv[1:2])
    forwarded = sys.argv[2:]
    if args.study == 'check':
        child = argparse.ArgumentParser(description='Small no-data checks; no full training or full closed-loop evaluations.')
        child.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
        options = child.parse_args(forwarded)
        for script, tail in [
            ('data/manage.py', ['check']),
            ('experiments/main/run.py', ['--stage', 'check', '--device', options.device]),
            ('experiments/ablation/run.py', ['--stage', 'check', '--device', options.device]),
            ('experiments/noise/run.py', ['--stage', 'check', '--device', options.device]),
            ('experiments/generalization/run.py', ['--stage', 'check', '--study', 'ood', '--run-name', 'release_check', '--device', options.device]),
        ]:
            print(f'Checking {script}', flush=True)
            execute(script, tail)
        print('All small checks passed. No full dataset, trained model or paper-performance result was generated.')
    elif args.study in ['ood', 'labels', 'dynamics']:
        execute('experiments/generalization/run.py', ['--study', args.study, *forwarded])
    else:
        target = {'data': 'data/manage.py', 'main': 'experiments/main/run.py',
                  'ablation': 'experiments/ablation/run.py', 'noise': 'experiments/noise/run.py'}[args.study]
        execute(target, forwarded)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)

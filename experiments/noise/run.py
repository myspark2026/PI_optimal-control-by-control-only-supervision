"""Run paired control-label noise training and evaluation."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RATIOS = ("0.1", "0.25", "0.5", "0.75")


def ratio_argument(value: str) -> str:
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("A noise ratio must be numeric.") from exc
    for ratio in RATIOS:
        if abs(numeric - float(ratio)) <= 1e-12:
            return ratio
    raise argparse.ArgumentTypeError("Choose 0.1, 0.25, 0.5, or 0.75.")


def load_study(system: str):
    path = Path(__file__).resolve().with_name(f"{system}.py")
    spec = importlib.util.spec_from_file_location(f"_release_noise_{system}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the {system} noise implementation.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Reject output symlinks that redirect writes outside this study's subtree.
    relative = module.ROOT.resolve().relative_to(PACKAGE_ROOT.resolve())
    if relative.parts != ("outputs", "noise", system):
        raise ValueError("Noise outputs must remain in outputs/noise/<system>.")
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "prepare", "train", "evaluate", "summarize"),
                        required=True)
    parser.add_argument("--system", choices=("both", "cs", "ac"), default="both")
    parser.add_argument("--ratios", nargs="+", type=ratio_argument, default=list(RATIOS))
    parser.add_argument("--noise-indices", nargs="+", type=int, choices=range(5), default=list(range(5)),
                        help="Indices 0..4 select label-noise seeds 2026..2030.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    args = parser.parse_args(argv)
    systems = ("cs", "ac") if args.system == "both" else (args.system,)
    ratios = list(dict.fromkeys(args.ratios))
    indices = list(dict.fromkeys(args.noise_indices))
    for system in systems:
        study = load_study(system)
        if args.stage == "check":
            study.run_self_check(study.select_device(args.device))
            continue
        forwarded = ["--stages", args.stage, "--ratios", *ratios,
                     "--noise-indices", *(str(index) for index in indices),
                     "--device", args.device]
        status = study.main(forwarded)
        if status:
            return int(status)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)

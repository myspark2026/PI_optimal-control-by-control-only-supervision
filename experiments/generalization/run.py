"""Entry point for code-only OOD, data-size and dynamics-perturbation studies."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path

# Import runtime first so all runtime caches are scoped before scientific imports.
import runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["ood", "labels", "dynamics"], required=True)
    parser.add_argument("--stage", choices=["check", "train", "evaluate"], default="check")
    parser.add_argument("--system", choices=["both", "cs", "ac"], default="both")
    parser.add_argument("--device", choices=["original", "cpu", "cuda"], default="original",
                        help="original uses CS CPU / AC CUDA explicitly; no availability fallback")
    parser.add_argument("--budget", choices=["full", "half"], default="half")
    parser.add_argument("--run-name", default="main_protocol")
    parser.add_argument("--epochs", type=int, help="explicit diagnostic epoch override; use a separate --run-name")
    parser.add_argument("--full-checkpoint", type=Path,
                        help="single-system full-data checkpoint saved by this workflow")
    parser.add_argument("--half-checkpoint", type=Path,
                        help="single-system half-data checkpoint saved by this workflow")
    args = parser.parse_args(argv)
    systems = ("cs", "ac") if args.system == "both" else (args.system,)
    if (args.full_checkpoint or args.half_checkpoint) and args.system == "both":
        parser.error("Explicit checkpoint paths require --system cs or --system ac; otherwise use the documented per-system layout.")
    if (args.full_checkpoint or args.half_checkpoint) and args.stage != "evaluate":
        parser.error("Explicit checkpoint paths are evaluation-only.")
    if args.half_checkpoint and args.study != "labels":
        parser.error("--half-checkpoint is only used by --study labels.")
    if args.epochs is not None:
        if args.epochs <= 0:
            parser.error("--epochs must be positive")
        if args.run_name == "main_protocol":
            parser.error("An epoch override requires a separate --run-name, e.g. debug_epochs1")
        for system in systems:
            runtime.SPECS[system]["epochs"] = args.epochs
    directory = runtime.prepare_run(args.run_name)
    if args.stage == "check":
        from checks import check_setup
        check_setup(directory, systems, args.device)
    elif args.stage == "train":
        from training import train_one
        runtime.validate_inputs(systems)
        for system in systems:
            train_one(directory, system, args.budget, args.device)
    else:
        from evaluation import evaluate_study
        evaluate_study(directory, systems, args.study, args.device,
                       args.full_checkpoint, args.half_checkpoint)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run paper-matched iRF training followed by raw RIT extraction.

This orchestration script is intentionally Python 3.6 compatible. 

How to run (example): 
    # python tutorials/pbmc/02_irf_rit.py \
    # --input tutorials/pbmc/results/01_irf_inputs \
    # --output-dir tutorials/pbmc/results/02_irf \
    # --irf-module-path /path/to/iterative-Random-Forest (cloned git repository of iRF)
"""

import argparse
import re
from pathlib import Path

from _tutorial_utils import ensure_crane_grn_importable


ensure_crane_grn_importable(start=Path(__file__).resolve().parent)

from crane_grn.irf.rit import extract_rit_parameters
from crane_grn.irf.training import train_irf


def _input_paths(path):
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        if requested.suffix.lower() != ".parquet":
            raise ValueError("The iRF input file must be Parquet: {}".format(requested))
        return [requested]
    if requested.is_dir():
        matches = sorted(requested.glob("*_iRF_input.parquet"))
        if not matches:
            raise FileNotFoundError(
                "No *_iRF_input.parquet files found in {}.".format(requested)
            )
        return matches
    raise FileNotFoundError("iRF input path not found: {}".format(requested))


def _run_name(input_path):
    name = input_path.stem
    if name.endswith("_iRF_input"):
        name = name[: -len("_iRF_input")]
    if name.lower().startswith("pbmc_"):
        name = name[len("pbmc_") :]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return safe or "group"


def _parser():
    parser = argparse.ArgumentParser(
        description="Train iRF and extract RIT interactions for one file or a directory."
    )
    parser.add_argument("--input", required=True, help="Parquet file or input directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--irf-module-path", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--B", type=int, default=30)
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--rit-max-depth", type=int, default=3)
    parser.add_argument("--num-splits", type=int, default=2)
    parser.add_argument("--n-estimators-bootstrap", type=int, default=500)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    inputs = _input_paths(args.input)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    names = [_run_name(path) for path in inputs]
    if len(names) != len(set(names)):
        raise ValueError("Input filenames produce duplicate deterministic run names.")

    print("Discovered {} iRF input(s).".format(len(inputs)))
    for position, input_path in enumerate(inputs, 1):
        run_dir = output_root / _run_name(input_path)
        print(
            "[{}/{}] Training {} -> {}".format(
                position, len(inputs), input_path.name, run_dir
            )
        )
        outputs = train_irf(
            input_path=input_path,
            output_dir=run_dir,
            irf_module_path=args.irf_module_path,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            rf_max_depth=None,
            max_features="sqrt",
            n_jobs=args.n_jobs,
            class_weight="balanced_subsample",
            K=args.K,
            B=args.B,
            propn_n_samples=1.0,
            bin_class_type=1,
            M=args.M,
            rit_max_depth=args.rit_max_depth,
            noisy_split=True,
            num_splits=args.num_splits,
            n_estimators_bootstrap=args.n_estimators_bootstrap,
            write_named_stability=False,
            keep_debug_artifacts=False,
        )
        interaction_path = run_dir / "interaction_parameters.json"
        interactions = extract_rit_parameters(
            outputs["rit_bootstrap_output"],
            outputs["feature_names"],
            interaction_path,
        )
        print(
            "Completed {}: {} raw RIT interactions.".format(
                run_dir.name, len(interactions)
            )
        )


if __name__ == "__main__":
    main()

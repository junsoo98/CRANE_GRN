#!/usr/bin/env python
"""Train one paper-matched iterative Random Forest model."""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _optional_integer(value):
    if value is None or str(value).lower() in ("none", "null", ""):
        return None
    return int(value)


def _optional_class_weight(value):
    if value is None or str(value).lower() in ("none", "null", ""):
        return None
    return value


def _boolean(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "y")


def _canonical_interaction_key(key):
    def _to_group(value):
        numbers = []
        if isinstance(value, (tuple, list)):
            for token in value:
                numbers.extend(int(x) for x in re.findall(r"\d+", str(token)))
        else:
            numbers.extend(int(x) for x in re.findall(r"\d+", str(value)))
        return tuple(sorted(numbers))

    if isinstance(key, (tuple, list)):
        groups = [_to_group(group) for group in key]
    else:
        groups = [
            _to_group(part)
            for part in re.split(r"\|\|?|\s*__\s*|\s*\|\s*", str(key))
            if part
        ]
    return tuple(sorted(group for group in groups if group))


def _named_stability_scores(stability_scores, feature_names):
    output = {}
    for key, value in stability_scores.items():
        groups = _canonical_interaction_key(key)
        named_groups = []
        for group in groups:
            names = [
                feature_names[index]
                if 0 <= index < len(feature_names)
                else "feature{}".format(index)
                for index in group
            ]
            named_groups.append("+".join(sorted(names)))
        output["|".join(named_groups)] = float(value)
    return output


def _load_external_irf(irf_module_path):
    path = Path(irf_module_path).resolve()
    if not path.is_dir():
        raise NotADirectoryError(
            "The iterative-Random-Forest repository path was not found: {}".format(
                path
            )
        )
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    try:
        from irf import irf_utils
        from irf.ensemble import RandomForestClassifierWithWeights
    except ImportError as error:
        raise ImportError(
            "Could not import iRF from {}. Clone "
            "https://github.com/Yu-Group/iterative-Random-Forest and pass its "
            "repository path as irf_module_path.".format(path)
        ) from error
    return irf_utils, RandomForestClassifierWithWeights


def train_irf(
    input_path,
    output_dir,
    irf_module_path,
    test_size=0.2,
    random_state=42,
    n_estimators=500,
    min_samples_leaf=30,
    rf_max_depth=None,
    max_features="sqrt",
    n_jobs=12,
    class_weight="balanced_subsample",
    K=5,
    B=30,
    propn_n_samples=1.0,
    bin_class_type=1,
    M=500,
    rit_max_depth=3,
    noisy_split=True,
    num_splits=2,
    n_estimators_bootstrap=500,
    write_named_stability=False,
    keep_debug_artifacts=False,
):
    """Train iRF on one region-by-factor table with an ``ATAC_label`` column.

    The external iRF repository is imported from ``irf_module_path``. The
    function writes the five required paper artifacts and optionally a
    feature-name stability map and large debugging pickles.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError("iRF input was not found: {}".format(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(input_path)
    if "ATAC_label" not in table.columns:
        raise ValueError("Required column 'ATAC_label' was not found.")
    feature_names = table.columns.drop("ATAC_label").astype(str).tolist()
    if not feature_names:
        raise ValueError("The iRF input contains no factor features.")
    X = table.loc[:, feature_names].values
    y = table["ATAC_label"].values
    classes = pd.Series(y).value_counts().sort_index()
    if len(classes) < 2:
        raise ValueError("ATAC_label contains only one class.")

    input_summary = {
        "input_path": str(input_path),
        "n_regions": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "positive_rate": float((y == 1).mean()),
        "feature_names": feature_names,
    }
    with (output_dir / "input_summary.json").open("w") as handle:
        json.dump(input_summary, handle, indent=2)
    with (output_dir / "feature_names.json").open("w") as handle:
        json.dump(feature_names, handle, indent=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        stratify=y,
        random_state=int(random_state),
    )
    irf_utils, classifier_type = _load_external_irf(irf_module_path)
    classifier = classifier_type(
        n_estimators=int(n_estimators),
        max_depth=rf_max_depth,
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        n_jobs=int(n_jobs),
        class_weight=class_weight,
    )

    run_config = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "n_estimators": int(n_estimators),
        "rf_max_depth": rf_max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "max_features": max_features,
        "n_jobs": int(n_jobs),
        "class_weight": class_weight,
        "K": int(K),
        "B": int(B),
        "propn_n_samples": float(propn_n_samples),
        "bin_class_type": int(bin_class_type),
        "M": int(M),
        "rit_max_depth": int(rit_max_depth),
        "noisy_split": bool(noisy_split),
        "num_splits": int(num_splits),
        "n_estimators_bootstrap": int(n_estimators_bootstrap),
        "irf_module_path": str(Path(irf_module_path).resolve()),
    }
    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(run_config, handle, indent=2)

    (
        all_rf_weights,
        all_iterations,
        all_rf_bootstrap,
        all_rit_bootstrap,
        stability_scores,
    ) = irf_utils.run_iRF(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        rf=classifier,
        K=int(K),
        B=int(B),
        random_state_classifier=int(random_state),
        propn_n_samples=float(propn_n_samples),
        bin_class_type=int(bin_class_type),
        M=int(M),
        max_depth=int(rit_max_depth),
        noisy_split=bool(noisy_split),
        num_splits=int(num_splits),
        n_estimators_bootstrap=int(n_estimators_bootstrap),
    )

    raw_scores = {
        repr(key): float(value) if isinstance(value, np.floating) else value
        for key, value in stability_scores.items()
    }
    with (output_dir / "stability_score_raw.json").open("w") as handle:
        json.dump(raw_scores, handle, indent=2)
    if write_named_stability:
        named = _named_stability_scores(stability_scores, feature_names)
        with (output_dir / "stability_score_names.json").open("w") as handle:
            json.dump(named, handle, indent=2)
    with (output_dir / "rit_bootstrap_output.pkl").open("wb") as handle:
        pickle.dump(all_rit_bootstrap, handle)

    if keep_debug_artifacts:
        debug = {
            "rf_weights.pkl": all_rf_weights,
            "rf_all_iterations.pkl": all_iterations,
            "rf_bootstrap_output.pkl": all_rf_bootstrap,
        }
        for filename, value in debug.items():
            with (output_dir / filename).open("wb") as handle:
                pickle.dump(value, handle)

    return {
        "input_summary": output_dir / "input_summary.json",
        "run_config": output_dir / "run_config.json",
        "feature_names": output_dir / "feature_names.json",
        "stability_score_raw": output_dir / "stability_score_raw.json",
        "rit_bootstrap_output": output_dir / "rit_bootstrap_output.pkl",
    }


def _parser():
    parser = argparse.ArgumentParser(description="Train iRF on one CRANE input.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--irf-module-path", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=30)
    parser.add_argument("--rf-max-depth", default="None")
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--class-weight", default="balanced_subsample")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--B", type=int, default=30)
    parser.add_argument("--propn-n-samples", type=float, default=1.0)
    parser.add_argument("--bin-class-type", type=int, default=1)
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--rit-max-depth", type=int, default=3)
    parser.add_argument("--noisy-split", default="True")
    parser.add_argument("--num-splits", type=int, default=2)
    parser.add_argument("--n-estimators-bootstrap", type=int, default=500)
    parser.add_argument("--write-named-stability", action="store_true")
    parser.add_argument("--keep-debug-artifacts", action="store_true")
    return parser


def main(argv=None):
    """Run the command-line interface for one iRF training job."""
    args = _parser().parse_args(argv)
    train_irf(
        input_path=args.input_path,
        output_dir=args.output_dir,
        irf_module_path=args.irf_module_path,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        rf_max_depth=_optional_integer(args.rf_max_depth),
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        class_weight=_optional_class_weight(args.class_weight),
        K=args.K,
        B=args.B,
        propn_n_samples=args.propn_n_samples,
        bin_class_type=args.bin_class_type,
        M=args.M,
        rit_max_depth=args.rit_max_depth,
        noisy_split=_boolean(args.noisy_split),
        num_splits=args.num_splits,
        n_estimators_bootstrap=args.n_estimators_bootstrap,
        write_named_stability=args.write_named_stability,
        keep_debug_artifacts=args.keep_debug_artifacts,
    )


if __name__ == "__main__":
    main()

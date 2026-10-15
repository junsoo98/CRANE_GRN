#!/usr/bin/env python
"""Extract raw RIT interaction statistics from an iRF run."""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def summarize_interaction_parameters(rit_bootstrap_output):
    """Count raw interaction occurrences, supporting trees, and bootstraps."""
    occurrences = Counter()
    tree_support = defaultdict(set)
    bootstrap_support = defaultdict(set)

    for bootstrap_id, tree_data in rit_bootstrap_output.items():
        bootstrap_text = str(bootstrap_id)
        for tree_id, values in tree_data.items():
            seen_in_tree = set()
            for interaction in values.get("rit_intersected_values", []):
                array = np.asarray(interaction)
                if array.size == 0:
                    continue
                key = "_".join(map(str, np.ravel(array).astype(int).tolist()))
                occurrences[key] += 1
                bootstrap_support[key].add(bootstrap_text)
                seen_in_tree.add(key)
            tree_tag = "{}::{}".format(bootstrap_text, tree_id)
            for key in seen_in_tree:
                tree_support[key].add(tree_tag)

    keys = sorted(set(occurrences) | set(tree_support) | set(bootstrap_support))
    return {
        key: {
            "occurrences": int(occurrences.get(key, 0)),
            "tree_support": len(tree_support.get(key, set())),
            "bootstrap_support": len(bootstrap_support.get(key, set())),
        }
        for key in keys
    }


def extract_rit_parameters(
    rit_bootstrap_path,
    feature_names_path,
    output_path,
):
    """Extract ``interaction_parameters.json`` from the two required artifacts."""
    rit_bootstrap_path = Path(rit_bootstrap_path)
    feature_names_path = Path(feature_names_path)
    output_path = Path(output_path)
    for path in (rit_bootstrap_path, feature_names_path):
        if not path.exists():
            raise FileNotFoundError("Required RIT input was not found: {}".format(path))
    with feature_names_path.open("r") as handle:
        feature_names = json.load(handle)
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("feature_names.json must contain a non-empty JSON list.")
    with rit_bootstrap_path.open("rb") as handle:
        bootstrap_output = pickle.load(handle)
    parameters = summarize_interaction_parameters(bootstrap_output)

    for key in parameters:
        indices = [int(value) for value in key.split("_")]
        if indices and max(indices) >= len(feature_names):
            raise ValueError(
                "RIT interaction {} references a feature outside feature_names.json.".format(
                    key
                )
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(parameters, handle, indent=2)
    return parameters


def _parser():
    parser = argparse.ArgumentParser(description="Extract RIT interaction parameters.")
    parser.add_argument("--irf-output-dir", required=True)
    parser.add_argument("--output-path", default=None)
    return parser


def main(argv=None):
    """Run the command-line interface for RIT parameter extraction."""
    args = _parser().parse_args(argv)
    run_dir = Path(args.irf_output_dir)
    output = (
        Path(args.output_path)
        if args.output_path is not None
        else run_dir / "interaction_parameters.json"
    )
    parameters = extract_rit_parameters(
        run_dir / "rit_bootstrap_output.pkl",
        run_dir / "feature_names.json",
        output,
    )
    print("Saved {} raw RIT interactions to {}".format(len(parameters), output))


if __name__ == "__main__":
    main()

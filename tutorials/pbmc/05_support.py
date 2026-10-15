#!/usr/bin/env python
"""Compute posterior-gated support and public group aggregates for PBMC.
# python "tutorials/pbmc/05_support.py" \
#   --data-root /.../CRANE_GRN_data \
#   --work-dir "tutorials/pbmc/results" \
#   --group-column cell_type \
#   --batch-size 256 \
#   --posterior-exponent 3.0
"""

import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from _tutorial_utils import ensure_crane_grn_importable, require_path


ensure_crane_grn_importable(start=Path(__file__).resolve().parent)

from crane_grn.io import extract_cistopic_matrices, get_modality
from crane_grn.support import aggregate_support, compute_support, validate_aggregate


def _parser():
    parser = argparse.ArgumentParser(
        description="Compute CRANE-GRN support and six-field group aggregates."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--group-column", default="cell_type")
    parser.add_argument("--single-group-name", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--posterior-exponent", type=float, default=3.0)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        import mudata
    except ImportError as error:
        raise ImportError(
            "mudata is required by 05_support.py; activate the main CRANE environment."
        ) from error
    data_root = Path(args.data_root).expanduser().resolve()
    work_root = Path(args.work_dir).expanduser().resolve()
    input_root = data_root / "PBMC" / "prepared_inputs"
    reference_root = data_root / "PBMC" / "outputs"
    support_root = work_root / "05_support"
    support_root.mkdir(parents=True, exist_ok=True)

    scplus_path = require_path(
        input_root / "scplusmdata.h5mu", "PBMC scplusmdata prepared input"
    )
    cistopic_path = require_path(
        input_root / "cistopic_obj.pkl", "PBMC cisTopic prepared input"
    )
    aucell_path = require_path(
        reference_root / "AUCell_direct.h5mu", "PBMC direct AUCell output"
    )
    metadata_path = require_path(
        reference_root / "eRegulon_direct.tsv", "PBMC direct eRegulon metadata"
    )
    responsibility_path = require_path(
        work_root / "03_responsibility" / "responsibility_triplet.parquet",
        "tutorial triplet responsibility",
    )
    posterior_path = require_path(
        work_root / "04_bayesian" / "posterior.parquet", "tutorial posterior"
    )

    scplus = mudata.read(str(scplus_path))
    atac = get_modality(scplus, "scATAC_counts")
    if args.single_group_name is None:
        if args.group_column not in atac.obs:
            raise KeyError(
                "Group column {!r} was not found in scATAC_counts.obs.".format(
                    args.group_column
                )
            )
        groups = atac.obs[args.group_column].astype(str).copy()
    else:
        groups = pd.Series(
            str(args.single_group_name), index=atac.obs_names.astype(str), dtype=str
        )

    with cistopic_path.open("rb") as handle:
        cistopic = pickle.load(handle)
    region_by_topic, cell_by_topic = extract_cistopic_matrices(cistopic)
    responsibility = pd.read_parquet(responsibility_path)
    posterior = pd.read_parquet(posterior_path)
    metadata = pd.read_csv(metadata_path, sep="\t")

    aucell_container = mudata.read(str(aucell_path))
    aucell = get_modality(aucell_container, "Region_based")
    print(
        "Computing support with posterior_exponent={} and batch_size={}.".format(
            args.posterior_exponent, args.batch_size
        )
    )
    output_paths = compute_support(
        region_by_topic=region_by_topic,
        cell_by_topic=cell_by_topic,
        responsibility_triplet=responsibility,
        posterior=posterior,
        aucell=aucell,
        eregulon_metadata=metadata,
        groups=groups,
        output_dir=support_root,
        posterior_exponent=args.posterior_exponent,
        batch_size=args.batch_size,
        restrict_eregulons_to_aucell=True,
    )
    aggregates = aggregate_support(
        output_dir=support_root,
        groups=groups,
        aucell=aucell,
        batch_size=args.batch_size,
        save_npz=True,
        output_prefix="agg",
    )

    aggregate_paths = sorted(support_root.glob("agg_*.npz"))
    expected_groups = 1 if args.single_group_name is not None else 13
    if len(aggregate_paths) != expected_groups:
        raise ValueError(
            "Expected {} aggregate files, found {}.".format(
                expected_groups, len(aggregate_paths)
            )
        )
    for path in aggregate_paths:
        validate_aggregate(path)
        with np.load(str(path), allow_pickle=False) as archive:
            shape = archive["support_mean_KE"].shape
        if args.single_group_name is None and shape != (530, 113):
            raise ValueError(
                "PBMC aggregate {} has shape {}; expected (530, 113).".format(
                    path.name, shape
                )
            )

    print("Created support intermediates for {} group(s).".format(len(output_paths)))
    print("Validated {} public aggregate files in {}.".format(len(aggregates), support_root))
    print("The support memmaps are local working files, not distributable outputs.")

    del aggregates, output_paths, aucell, aucell_container
    del metadata, posterior, responsibility, region_by_topic, cell_by_topic
    del cistopic, atac, scplus
    gc.collect()


if __name__ == "__main__":
    main()

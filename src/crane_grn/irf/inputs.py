"""Construct paper-matched inputs for iterative Random Forest training."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..regions import canonicalize_regions, parse_region


def _open_fraction(accessibility):
    if sparse.issparse(accessibility):
        positive = accessibility > 0
        return np.asarray(positive.getnnz(axis=0), dtype=float) / float(
            accessibility.shape[0]
        )
    return (np.asarray(accessibility) > 0).mean(axis=0)


def _binary_entropy(open_fraction):
    probability = np.asarray(open_fraction, dtype=float)
    entropy = np.zeros_like(probability)
    positive = probability > 0
    below_one = probability < 1
    entropy[positive] -= probability[positive] * np.log2(probability[positive])
    complement = 1.0 - probability
    entropy[below_one] -= complement[below_one] * np.log2(complement[below_one])
    return entropy


def build_irf_input(
    accessibility,
    region_names,
    region_by_factor,
    open_fraction_threshold=0.5,
    entropy_minimum=0.05,
    label_column="ATAC_label",
):
    """Build one region-by-feature iRF input table.

    Parameters
    ----------
    accessibility
        Cell-by-region accessibility matrix. Sparse input remains sparse.
    region_names
        Region labels corresponding to the matrix columns.
    region_by_factor
        Binary DataFrame with regions as rows and factors as columns.

    Returns
    -------
    pandas.DataFrame
        Informative, factor-occupied regions by factors, followed by a binary
        accessibility label column.
    """
    if accessibility.ndim != 2:
        raise ValueError("Accessibility input must be two-dimensional.")
    regions = canonicalize_regions(region_names)
    if accessibility.shape[1] != len(regions):
        raise ValueError(
            "Accessibility has {} columns but {} region names were provided.".format(
                accessibility.shape[1], len(regions)
            )
        )
    if regions.has_duplicates:
        raise ValueError("Accessibility region labels are not unique.")

    factors = region_by_factor.copy()
    factors.index = canonicalize_regions(factors.index)
    if factors.index.has_duplicates:
        raise ValueError("Factor-matrix region labels are not unique.")
    missing = regions.difference(factors.index)
    if len(missing):
        raise ValueError(
            "The region-by-factor matrix is missing {} accessibility regions; "
            "examples: {}".format(len(missing), missing[:3].tolist())
        )

    open_fraction = _open_fraction(accessibility)
    entropy = _binary_entropy(open_fraction)
    informative = entropy > float(entropy_minimum)
    informative_regions = regions[informative]
    labels = (open_fraction[informative] >= float(open_fraction_threshold)).astype(
        np.int8
    )

    output = factors.loc[informative_regions].copy()
    occupied = output.sum(axis=1) > 0
    output[label_column] = labels
    return output.loc[occupied].copy()


def _safe_group_name(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "group"


def build_irf_inputs_by_group(
    accessibility,
    cell_names,
    region_names,
    groups,
    region_by_factor,
    output_dir=None,
    open_fraction_threshold=0.5,
    entropy_minimum=0.05,
    minimum_cells=100,
    minimum_class_regions=100,
    filename_prefix="crane",
    save_bed=True,
):
    """Build one iRF input per biological group using ``build_irf_input``.

    Group labels are aligned by cell name. If ``output_dir`` is provided, each
    input is written as Parquet and its retained regions can also be written as
    a headerless BED file.
    """
    cells = pd.Index(cell_names).astype(str)
    if cells.has_duplicates:
        raise ValueError("Cell labels are not unique.")
    if accessibility.shape[0] != len(cells):
        raise ValueError("Accessibility rows do not match the cell axis.")

    if isinstance(groups, pd.Series):
        group_series = groups.copy()
        group_series.index = group_series.index.astype(str)
        missing = cells.difference(group_series.index)
        if len(missing):
            raise ValueError(
                "Group labels are missing {} cells; examples: {}".format(
                    len(missing), missing[:3].tolist()
                )
            )
        group_series = group_series.loc[cells].astype(str)
    else:
        values = np.asarray(groups)
        if values.shape[0] != len(cells):
            raise ValueError("Group-label length does not match the cell axis.")
        group_series = pd.Series(values.astype(str), index=cells)

    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    outputs = {}
    summaries = []
    for group in sorted(group_series.dropna().unique().tolist()):
        mask = group_series.to_numpy() == group
        number_cells = int(mask.sum())
        if number_cells < int(minimum_cells):
            summaries.append(
                {
                    "group": group,
                    "status": "SKIP_low_cells",
                    "n_cells": number_cells,
                    "n_regions": 0,
                    "n_positive": 0,
                    "n_negative": 0,
                    "positive_rate": np.nan,
                    "path": None,
                }
            )
            continue

        subset = accessibility[mask, :]
        table = build_irf_input(
            subset,
            region_names,
            region_by_factor,
            open_fraction_threshold=open_fraction_threshold,
            entropy_minimum=entropy_minimum,
        )
        positive = int((table["ATAC_label"] == 1).sum())
        negative = int((table["ATAC_label"] == 0).sum())
        total = int(table.shape[0])
        rate = positive / float(total) if total else np.nan
        if total == 0:
            status = "FAIL_no_regions"
        elif positive < minimum_class_regions or negative < minimum_class_regions:
            status = "LOW_CONF_class_count"
        elif rate < 0.05 or rate > 0.95:
            status = "LOW_CONF_imbalance"
        else:
            status = "PASS"

        path = None
        if destination is not None:
            safe_group = _safe_group_name(group)
            path = destination / "{}_{}_iRF_input.parquet".format(
                filename_prefix, safe_group
            )
            table.to_parquet(path)
            if save_bed:
                bed_rows = [parse_region(region) for region in table.index]
                bed = pd.DataFrame(
                    bed_rows, columns=["Chromosome", "Start", "End"]
                )
                bed.to_csv(
                    destination
                    / "{}_{}_filtered_regions.bed".format(
                        filename_prefix, safe_group
                    ),
                    sep="\t",
                    header=False,
                    index=False,
                )
        outputs[group] = table
        summaries.append(
            {
                "group": group,
                "status": status,
                "n_cells": number_cells,
                "n_regions": total,
                "n_positive": positive,
                "n_negative": negative,
                "positive_rate": rate,
                "path": str(path) if path is not None else None,
            }
        )
    return outputs, pd.DataFrame(summaries)

"""Load RIT outputs, select modules, and build factor membership matrices."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .regions import canonicalize_regions


def collect_region_union(input_paths, label_column="ATAC_label"):
    """Collect the sorted union of region labels from one or more iRF inputs."""
    regions = set()
    summaries = []
    for path in input_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("iRF input was not found: {}".format(path))
        table = pd.read_parquet(path)
        if label_column not in table.columns:
            raise ValueError(
                "Required column {!r} was not found in {}.".format(label_column, path)
            )
        labels = canonicalize_regions(table.index)
        regions.update(labels.tolist())
        summaries.append(
            {
                "source": path.stem,
                "path": str(path),
                "n_regions": int(table.shape[0]),
                "n_factors": int(table.shape[1] - 1),
                "n_positive": int((table[label_column] == 1).sum()),
                "n_negative": int((table[label_column] == 0).sum()),
            }
        )
    return pd.DataFrame(summaries), pd.Index(sorted(regions))


def build_union_region_by_factor(region_by_factor, union_regions):
    """Subset a region-by-factor matrix to an explicit union region axis."""
    matrix = region_by_factor.copy()
    matrix.index = canonicalize_regions(matrix.index)
    requested = canonicalize_regions(union_regions)
    if matrix.index.has_duplicates or requested.has_duplicates:
        raise ValueError("Region labels must be unique after canonicalization.")
    missing = requested.difference(matrix.index)
    if len(missing):
        raise ValueError(
            "The factor matrix is missing {} union regions; examples: {}".format(
                len(missing), missing[:3].tolist()
            )
        )
    return matrix.loc[requested].astype(int).copy()


def compute_selection_idf(region_by_factor):
    """Compute the selection IDF ``log((N + 1) / (df + 1))`` per factor."""
    number_regions = region_by_factor.shape[0]
    if number_regions == 0:
        raise ValueError("The region-by-factor matrix has no rows.")
    counts = region_by_factor.astype(int).sum(axis=0)
    return pd.Series(
        np.log((number_regions + 1.0) / (counts.to_numpy(dtype=float) + 1.0)),
        index=region_by_factor.columns.astype(str),
        name="selection_idf",
    )


def _interaction_parameter_path(run_dir):
    candidates = [
        run_dir / "interaction_parameters.json",
        run_dir / "export" / "interaction_parameters.json",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(
            "interaction_parameters.json was not found in {} or its export "
            "subdirectory.".format(run_dir)
        )
    if len(existing) > 1:
        raise ValueError(
            "Two interaction_parameters.json files were found for {}. Pass a "
            "single unambiguous run directory.".format(run_dir)
        )
    return existing[0]


def _canonical_stability_key(key):
    text = str(key).replace("'", "").strip()
    if re.match(r"^\d+(?:_\d+)*$", text):
        return text
    numbers = re.findall(r"\d+", text)
    return "_".join(numbers) if numbers else text


def load_irf_result(run_dir, source=None, group=None):
    """Load one iRF/RIT result directory into a raw interaction table."""
    run_dir = Path(run_dir)
    feature_path = run_dir / "feature_names.json"
    stability_path = run_dir / "stability_score_raw.json"
    interaction_path = _interaction_parameter_path(run_dir)
    for path in (feature_path, stability_path):
        if not path.exists():
            raise FileNotFoundError("Required iRF result was not found: {}".format(path))

    with feature_path.open("r") as handle:
        feature_names = list(map(str, json.load(handle)))
    with stability_path.open("r") as handle:
        raw_stability = json.load(handle)
    stability = {
        _canonical_stability_key(key): value
        for key, value in raw_stability.items()
    }
    with interaction_path.open("r") as handle:
        parameters = json.load(handle)

    if parameters and not set(parameters).intersection(stability):
        raise ValueError(
            "No RIT interaction keys matched the raw stability-score keys in {}.".format(
                run_dir
            )
        )
    source = str(source) if source is not None else run_dir.name
    group = str(group) if group is not None else source
    rows = []
    for interaction_key, values in parameters.items():
        indices = [int(value) for value in str(interaction_key).split("_")]
        if indices and (min(indices) < 0 or max(indices) >= len(feature_names)):
            raise ValueError(
                "RIT interaction {} is outside the feature-name axis.".format(
                    interaction_key
                )
            )
        names = [feature_names[index] for index in indices]
        rows.append(
            {
                "source": source,
                "group": group,
                "rit_interaction_index": str(interaction_key),
                "rit_interaction": names,
                "rit_interaction_name": ";".join(names),
                "order": len(names),
                "occurrences": int(values["occurrences"]),
                "tree_support": int(values["tree_support"]),
                "bootstrap_support": int(values["bootstrap_support"]),
                "stability_score": float(stability.get(str(interaction_key), np.nan)),
            }
        )
    return pd.DataFrame(rows)


def load_irf_results(runs):
    """Load multiple iRF results from ``source -> path`` or iterable entries.

    Iterable entries may be ``(source, group, path)`` or ``(source, path)``.
    """
    entries = []
    if isinstance(runs, dict):
        entries = [(source, source, path) for source, path in runs.items()]
    else:
        for entry in runs:
            if len(entry) == 2:
                entries.append((entry[0], entry[0], entry[1]))
            elif len(entry) == 3:
                entries.append(tuple(entry))
            else:
                raise ValueError("Each iRF run entry must have two or three values.")
    if not entries:
        raise ValueError("No iRF runs were provided.")
    tables = [
        load_irf_result(path, source=source, group=group)
        for source, group, path in entries
    ]
    return pd.concat(tables, ignore_index=True)


def _rank01(values):
    return values.rank(method="average", pct=True)


def select_modules_one_source(
    interaction_table,
    selection_idf,
    max_order=8,
    top_fraction_per_order=0.10,
    selection_idf_weight=0.5,
    stability_weight=0.5,
    tree_weight=0.5,
    stability_threshold=0.5,
):
    """Select candidate modules from one raw RIT source.

    Ranking and quantile tie behavior follow the paper implementation. The
    selected table introduces the CRANE terms ``module`` and ``factors``;
    unselected rows retain raw RIT terminology in the scored table.
    """
    required = {
        "rit_interaction",
        "rit_interaction_name",
        "order",
        "tree_support",
        "stability_score",
    }
    missing = sorted(required.difference(interaction_table.columns))
    if missing:
        raise ValueError("Interaction table is missing columns: {}".format(missing))

    scored = interaction_table.copy()
    scored["stability_score"] = scored["stability_score"].fillna(0.0)
    scored = scored[scored["order"] <= int(max_order)].copy()
    if scored.empty:
        return scored.copy(), scored

    scored["r_tree"] = scored.groupby("order")["tree_support"].transform(
        lambda values: _rank01(np.log1p(values))
    )
    scored["order_score_raw"] = (
        float(stability_weight) * scored["stability_score"] + float(tree_weight) * scored["r_tree"]
    )
    scored["idf_mean"] = scored["rit_interaction"].apply(
        lambda factors: float(
            np.mean([selection_idf.get(factor, 0.0) for factor in factors])
        )
        if factors
        else 0.0
    )
    scored["r_idf"] = scored.groupby("order")["idf_mean"].transform(_rank01)
    scored["order_score"] = (
        (1.0 - float(selection_idf_weight)) * scored["order_score_raw"]
        + float(selection_idf_weight)
        * (scored["stability_score"] * scored["r_idf"])
    )

    selected_indices = []
    for _, block in scored.groupby("order"):
        threshold = block["order_score"].quantile(
            1.0 - float(top_fraction_per_order)
        )
        selected_indices.extend(
            block.index[block["order_score"] >= threshold].tolist()
        )
    selected = scored.loc[
        scored.index.isin(selected_indices)
        & (scored["stability_score"] >= float(stability_threshold))
    ].copy()
    selected["module"] = selected["rit_interaction_name"]
    selected["factors"] = selected["rit_interaction"]
    return selected, scored


def select_modules_by_source(interaction_table, selection_idf, **selection_options):
    """Run the same module-selection procedure independently for each source."""
    if "source" not in interaction_table.columns:
        raise ValueError("Interaction table has no source column.")
    selected_tables = []
    scored_tables = []
    summaries = []
    for source, block in interaction_table.groupby("source", sort=True):
        selected, scored = select_modules_one_source(
            block, selection_idf, **selection_options
        )
        selected_tables.append(selected)
        scored_tables.append(scored)
        summaries.append(
            {
                "source": source,
                "group": block["group"].iloc[0] if "group" in block else source,
                "n_input": int(block.shape[0]),
                "n_after_max_order": int(scored.shape[0]),
                "n_selected": int(selected.shape[0]),
                "selected_rate": float(selected.shape[0] / max(scored.shape[0], 1)),
            }
        )
    selected_long = (
        pd.concat(selected_tables, ignore_index=True)
        if selected_tables
        else pd.DataFrame()
    )
    scored_long = (
        pd.concat(scored_tables, ignore_index=True)
        if scored_tables
        else pd.DataFrame()
    )
    return selected_long, scored_long, pd.DataFrame(summaries)


def union_selected_modules(selected_long):
    """Combine source-level selections into one ordered module catalog."""
    if selected_long.empty:
        raise ValueError("No selected modules were provided.")

    def _join_unique(values):
        return ";".join(sorted(set(map(str, values))))

    catalog = (
        selected_long.groupby("module", as_index=False)
        .agg(
            factors=("factors", "first"),
            order=("order", "first"),
            max_stability=("stability_score", "max"),
            mean_stability=("stability_score", "mean"),
            median_stability=("stability_score", "median"),
            max_order_score=("order_score", "max"),
            mean_order_score=("order_score", "mean"),
            median_order_score=("order_score", "median"),
            max_occurrences=("occurrences", "max"),
            mean_occurrences=("occurrences", "mean"),
            max_tree_support=("tree_support", "max"),
            mean_tree_support=("tree_support", "mean"),
            max_bootstrap_support=("bootstrap_support", "max"),
            mean_bootstrap_support=("bootstrap_support", "mean"),
            n_groups=("group", "nunique"),
            groups=("group", _join_unique),
            sources=("source", _join_unique),
        )
        .sort_values(
            [
                "max_stability",
                "max_order_score",
                "max_bootstrap_support",
                "max_tree_support",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    return catalog


def build_factor_by_module(module_catalog):
    """Build a binary factor-by-module membership matrix.

    Rows are regulatory factors and columns are selected factor-combination
    modules. Module order follows the input catalog.
    """
    required = {"module", "factors"}
    missing = sorted(required.difference(module_catalog.columns))
    if missing:
        raise ValueError("Module catalog is missing columns: {}".format(missing))
    if module_catalog["module"].astype(str).duplicated().any():
        raise ValueError("Module names in the catalog are not unique.")
    factors = sorted(
        {factor for values in module_catalog["factors"] for factor in values}
    )
    modules = module_catalog["module"].astype(str).tolist()
    membership = pd.DataFrame(0, index=factors, columns=modules, dtype=np.int8)
    for module, values in zip(modules, module_catalog["factors"]):
        unknown = sorted(set(values).difference(membership.index))
        if unknown:
            raise ValueError("Unexpected factors in module {}: {}".format(module, unknown))
        membership.loc[list(values), module] = 1
    return membership

"""Small infrastructure and PBMC-specific visualization helpers for reproducability.

This module deliberately contains no CRANE-GRN scientific algorithms. 
"""

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PUBLIC_AGGREGATE_FIELDS = (
    "n_cells",
    "modules",
    "eregs",
    "reg_count",
    "support_mean_KE",
    "corr_auc_sup_KE",
)

PBMC_GROUP_ORDER = [
    "CD14 mono",
    "CD16 mono",
    "intermediate mono",
    "mDC",
    "pDC",
    "naïve B",
    "memory B",
    "CD4+ naïve T",
    "CD4+ memory T",
    "CD8+ naïve T",
    "CD8+ activated T",
    "MAIT",
    "NK",
]

_LINEAGE_FACTORS = {
    "T/NK": {"BCL11B", "GATA3", "BATF"},
    "B": {"EBF1", "PAX5"},
    "Myeloid": {"SPI1", "CEBPA", "CEBPB", "IRF8"},
}

_CROSS_LINEAGE_FACTORS = {"IRF4", "RUNX1"}

_EXPECTED_GROUPS = {
    "T/NK": {
        "CD4+ naïve T",
        "CD4+ memory T",
        "CD8+ naïve T",
        "CD8+ activated T",
        "MAIT",
        "NK",
    },
    "B": {"naïve B", "memory B"},
    "Myeloid": {"CD14 mono", "CD16 mono", "intermediate mono", "mDC"},
}

NETWORK_DIRECTION_COLORS = {
    "+/+": "#5995D1",
    "-/+": "#6DD146",
    "+/-": "#F85E80",
    "-/-": "#EC974D",
    "unknown": "#8A8A8A",
}

NETWORK_REGION_COLOR = "#009E73"
NETWORK_MODULE_COLOR = "#F0E442"
NETWORK_MODULE_EDGE_COLOR = "#C6A700"
NETWORK_GENE_COLOR = "#777777"
NETWORK_BACKGROUND_REGION_COLOR = "#BDBDBD"
NETWORK_TEXT_COLOR = "#2B2B2B"


def find_repository_root(start=None):
    """Find the nearest parent containing ``src/crane_grn``."""
    origin = Path.cwd() if start is None else Path(start)
    origin = origin.expanduser().resolve()
    candidates = [origin] + list(origin.parents)
    for candidate in candidates:
        if (candidate / "src" / "crane_grn").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find a repository containing src/crane_grn from {}.".format(
            origin
        )
    )


def ensure_crane_grn_importable(start=None):
    """Import ``crane_grn``, adding the discovered repository ``src`` if needed."""
    try:
        return importlib.import_module("crane_grn")
    except ImportError:
        repository_root = find_repository_root(start=start)
        source_root = str(repository_root / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        try:
            return importlib.import_module("crane_grn")
        except ImportError as error:
            raise ImportError(
                "Found src/crane_grn at {}, but import still failed. Verify the "
                "main CRANE environment and PYTHONPATH.".format(repository_root)
            ) from error


def require_path(path, description):
    """Return a resolved path or raise an actionable missing-input error."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError("Required {} not found: {}".format(description, resolved))
    return resolved


def safe_group_name(value):
    """Convert a biological group label to a deterministic filename token."""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "group"


def compare_dataframe_checkpoint(observed, reference, rtol=1e-5, atol=1e-8):
    """Return a concise, non-destructive comparison of two labeled matrices."""
    if not isinstance(observed, pd.DataFrame) or not isinstance(reference, pd.DataFrame):
        raise TypeError("Checkpoint comparison requires two pandas DataFrames.")
    observed_rows = pd.Index(observed.index.astype(str))
    reference_rows = pd.Index(reference.index.astype(str))
    observed_columns = pd.Index(observed.columns.astype(str))
    reference_columns = pd.Index(reference.columns.astype(str))
    common_rows = observed_rows.intersection(reference_rows)
    common_columns = observed_columns.intersection(reference_columns)
    values = observed.to_numpy(dtype=float, copy=False)
    finite = values[np.isfinite(values)]
    report = {
        "observed_shape": tuple(observed.shape),
        "reference_shape": tuple(reference.shape),
        "row_labels_exact": observed_rows.equals(reference_rows),
        "column_labels_exact": observed_columns.equals(reference_columns),
        "common_rows": int(len(common_rows)),
        "common_columns": int(len(common_columns)),
        "observed_min": float(finite.min()) if finite.size else np.nan,
        "observed_max": float(finite.max()) if finite.size else np.nan,
        "observed_mean": float(finite.mean()) if finite.size else np.nan,
        "observed_missing": int(np.isnan(values).sum()),
        "allclose": None,
        "max_absolute_difference": None,
    }
    if (
        observed.shape == reference.shape
        and set(observed_rows) == set(reference_rows)
        and set(observed_columns) == set(reference_columns)
    ):
        left = observed.copy()
        left.index = observed_rows
        left.columns = observed_columns
        right = reference.copy()
        right.index = reference_rows
        right.columns = reference_columns
        right = right.loc[left.index, left.columns]
        left_values = left.to_numpy(dtype=float, copy=False)
        right_values = right.to_numpy(dtype=float, copy=False)
        report["allclose"] = bool(
            np.allclose(left_values, right_values, rtol=rtol, atol=atol, equal_nan=True)
        )
        difference = np.abs(left_values - right_values)
        finite_difference = difference[np.isfinite(difference)]
        report["max_absolute_difference"] = (
            float(finite_difference.max()) if finite_difference.size else 0.0
        )
    return report


def load_public_aggregate(path):
    """Load and validate one six-field public aggregate archive."""
    path = require_path(path, "support aggregate")
    with np.load(str(path), allow_pickle=False) as archive:
        fields = set(archive.files)
        expected = set(PUBLIC_AGGREGATE_FIELDS)
        if fields != expected:
            raise ValueError(
                "Aggregate schema mismatch for {}: missing={}, extra={}.".format(
                    path, sorted(expected - fields), sorted(fields - expected)
                )
            )
        result = {field: archive[field] for field in PUBLIC_AGGREGATE_FIELDS}
    modules = result["modules"].astype(str)
    eregulons = result["eregs"].astype(str)
    expected_shape = (len(modules), len(eregulons))
    if result["support_mean_KE"].shape != expected_shape:
        raise ValueError("support_mean_KE does not match its labeled axes.")
    if result["corr_auc_sup_KE"].shape != expected_shape:
        raise ValueError("corr_auc_sup_KE does not match its labeled axes.")
    if result["reg_count"].shape != (len(eregulons),):
        raise ValueError("reg_count does not match the eRegulon axis.")
    return result


def cell_group_mean(score, groups):
    """Average a cell-by-feature score matrix within explicitly aligned groups."""
    matrix = score.copy()
    matrix.index = matrix.index.astype(str)
    aligned = groups.copy()
    aligned.index = aligned.index.astype(str)
    missing = matrix.index.difference(aligned.index)
    if len(missing):
        raise ValueError("Group labels are missing {} score cells.".format(len(missing)))
    joined = matrix.copy()
    joined.insert(0, "group", aligned.loc[matrix.index].astype(str).to_numpy())
    return joined.groupby("group", sort=False).mean()


def zscore_across_groups(group_by_feature):
    """Z-score every feature across biological groups with zero-safe scaling."""
    centered = group_by_feature - group_by_feature.mean(axis=0)
    scale = group_by_feature.std(axis=0, ddof=0).replace(0, np.nan)
    return centered.divide(scale, axis=1).fillna(0.0)


def build_quadrant_summary(
    prior,
    log_likelihood,
    posterior,
    groups,
    sample_points=10000,
    seed=0,
    heatmap_cells=3000,
    robust_epsilon=1e-8,
    saturation_low=0.02,
    saturation_high=0.98,
):
    """Reproduce the manuscript PBMC prior-evidence quadrant analysis.

    The analysis:

    1. aligns the prior, log-likelihood, and posterior matrices;
    2. reconstructs the standardized evidence used by the posterior model;
    3. samples cell-by-module pairs using the manuscript RNG sequence;
    4. estimates low/high thresholds from non-saturated prior values;
    5. summarizes raw posterior probabilities and posterior-minus-prior shifts.

    Parameters
    ----------
    prior
        Cell-by-module prior probability DataFrame.
    log_likelihood
        Cell-by-module ATAC log-likelihood DataFrame.
    posterior
        Cell-by-module posterior probability DataFrame. Its columns define
        the modules retained by the posterior calculation.
    groups
        Cell-group labels as a pandas Series indexed by cell name.
    sample_points
        Number of cell-by-module pairs sampled for the diagnostic.
    seed
        Random seed used by the manuscript diagnostic.
    heatmap_cells
        Number of cells consumed by the manuscript heatmap sampling step
        before quadrant sampling. This preserves the historical RNG state and
        therefore reproduces the manuscript pair sample.
    robust_epsilon
        Lower bound for the within-group likelihood IQR.
    saturation_low, saturation_high
        Prior values outside these bounds are considered saturated and are
        excluded only when estimating quadrant thresholds.

    Returns
    -------
    sampled_pairs : pandas.DataFrame
        Sampled cell-by-module values. Rows assigned ``"middle"`` do not fall
        in both the low/high prior and low/high evidence tails.
    summary : pandas.DataFrame
        Raw posterior and posterior-shift summaries for the four quadrants.
    thresholds : dict
        Prior and standardized-evidence thresholds used for classification.
    """
    if not isinstance(prior, pd.DataFrame):
        raise TypeError("prior must be a pandas DataFrame.")
    if not isinstance(log_likelihood, pd.DataFrame):
        raise TypeError("log_likelihood must be a pandas DataFrame.")
    if not isinstance(posterior, pd.DataFrame):
        raise TypeError("posterior must be a pandas DataFrame.")
    if not isinstance(groups, pd.Series):
        raise TypeError("groups must be a pandas Series indexed by cell name.")

    prior = prior.copy()
    log_likelihood = log_likelihood.copy()
    posterior = posterior.copy()
    groups = groups.copy()

    for frame in (prior, log_likelihood, posterior):
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)

    groups.index = groups.index.astype(str)

    # Preserve posterior order: posterior already contains the valid modules.
    cells = posterior.index[
        posterior.index.isin(prior.index)
        & posterior.index.isin(log_likelihood.index)
        & posterior.index.isin(groups.index)
    ]
    modules = posterior.columns[
        posterior.columns.isin(prior.columns)
        & posterior.columns.isin(log_likelihood.columns)
    ]

    if len(cells) == 0 or len(modules) == 0:
        raise ValueError("No common cells or modules were found.")

    prior_aligned = prior.loc[cells, modules].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    likelihood_aligned = log_likelihood.loc[cells, modules].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    posterior_aligned = posterior.loc[cells, modules].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    group_aligned = groups.loc[cells].astype(str)

    for name, values in (
        ("prior", prior_aligned),
        ("log_likelihood", likelihood_aligned),
        ("posterior", posterior_aligned),
    ):
        if not np.isfinite(values).all():
            raise ValueError(
                "{} contains non-finite values after alignment to posterior "
                "cells and modules.".format(name)
            )

    # Reconstruct the evidence transformation used in compute_posterior():
    #
    #   dL = log_likelihood - group/module median(log_likelihood)
    #   E  = (dL - median(dL)) / max(IQR(dL), epsilon)
    evidence = np.empty_like(likelihood_aligned, dtype=np.float64)

    for group_name in sorted(group_aligned.unique()):
        positions = np.flatnonzero(
            group_aligned.to_numpy() == group_name
        )
        group_likelihood = likelihood_aligned[positions, :]

        baseline = np.median(group_likelihood, axis=0)
        centered = group_likelihood - baseline[None, :]

        q25 = np.quantile(centered, 0.25, axis=0)
        q50 = np.quantile(centered, 0.50, axis=0)
        q75 = np.quantile(centered, 0.75, axis=0)
        scale = np.maximum(q75 - q25, robust_epsilon)

        evidence[positions, :] = (
            centered - q50[None, :]
        ) / scale[None, :]

    n_cells, n_modules = posterior_aligned.shape
    total_pairs = n_cells * n_modules
    n_sample = min(int(sample_points), total_pairs)

    rng = np.random.default_rng(seed)

    # Preserve RNG progression from the original manuscript diagnostic:
    # 1. scatter pair sample;
    # 2. heatmap cell sample;
    # 3. quadrant pair sample.
    rng.choice(total_pairs, size=n_sample, replace=False)
    rng.choice(
        n_cells,
        size=min(int(heatmap_cells), n_cells),
        replace=False,
    )
    sampled_flat = rng.choice(
        total_pairs,
        size=n_sample,
        replace=False,
    )

    cell_positions, module_positions = np.unravel_index(
        sampled_flat,
        (n_cells, n_modules),
    )

    sampled = pd.DataFrame(
        {
            "cell": cells.to_numpy()[cell_positions],
            "module": modules.to_numpy()[module_positions],
            "prior": prior_aligned[cell_positions, module_positions],
            "evidence": evidence[cell_positions, module_positions],
            "posterior": posterior_aligned[
                cell_positions,
                module_positions,
            ],
        }
    )
    sampled["posterior_shift"] = (
        sampled["posterior"] - sampled["prior"]
    )
    sampled["saturated"] = (
        (sampled["prior"] <= saturation_low)
        | (sampled["prior"] >= saturation_high)
    )

    threshold_source = sampled.loc[~sampled["saturated"]]
    if threshold_source.empty:
        raise ValueError(
            "No non-saturated sampled pairs are available for thresholding."
        )

    prior_low, prior_high = threshold_source["prior"].quantile(
        [0.25, 0.75]
    )
    evidence_low, evidence_high = threshold_source["evidence"].quantile(
        [0.25, 0.75]
    )

    thresholds = {
        "prior_low": float(prior_low),
        "prior_high": float(prior_high),
        "evidence_low": float(evidence_low),
        "evidence_high": float(evidence_high),
        "n_sampled_pairs": int(len(sampled)),
        "n_non_saturated_pairs": int(len(threshold_source)),
    }

    sampled["quadrant"] = "middle"

    quadrant_masks = [
        (
            "prior high / likelihood high",
            (sampled["prior"] >= prior_high)
            & (sampled["evidence"] >= evidence_high),
        ),
        (
            "prior high / likelihood low",
            (sampled["prior"] >= prior_high)
            & (sampled["evidence"] <= evidence_low),
        ),
        (
            "prior low / likelihood high",
            (sampled["prior"] <= prior_low)
            & (sampled["evidence"] >= evidence_high),
        ),
        (
            "prior low / likelihood low",
            (sampled["prior"] <= prior_low)
            & (sampled["evidence"] <= evidence_low),
        ),
    ]

    summary_rows = []

    for quadrant_name, mask in quadrant_masks:
        sampled.loc[mask, "quadrant"] = quadrant_name
        subset = sampled.loc[mask]

        summary_rows.append(
            {
                "quadrant": quadrant_name,
                "n": int(len(subset)),
                "posterior_mean": float(subset["posterior"].mean()),
                "posterior_median": float(
                    subset["posterior"].median()
                ),
                "delta_mean_post_minus_prior": float(
                    subset["posterior_shift"].mean()
                ),
                "delta_median_post_minus_prior": float(
                    subset["posterior_shift"].median()
                ),
                "saturated_fraction": float(
                    subset["saturated"].mean()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return sampled, summary, thresholds


def classify_module_lineage(module):
    """Classify a factor-combination module by PBMC lineage-marker content."""
    factors = set(str(module).split(";"))
    hits = [name for name, markers in _LINEAGE_FACTORS.items() if factors & markers]
    if len(hits) == 1 and not (factors & _CROSS_LINEAGE_FACTORS):
        return hits[0]
    if hits or (factors & _CROSS_LINEAGE_FACTORS):
        return "Mixed"
    return "Unassigned"


def lineage_match_summary(score, groups, score_name, top_quantile=0.75):
    """Select high-scoring group/module candidates and label lineage matches."""
    group_mean = cell_group_mean(score, groups)
    group_z = zscore_across_groups(group_mean)
    rows = []
    for group in group_z.index:
        for module in group_z.columns:
            lineage = classify_module_lineage(module)
            rows.append(
                {
                    "score": str(score_name),
                    "group": str(group),
                    "module": str(module),
                    "lineage": lineage,
                    "score_z": float(group_z.loc[group, module]),
                    "expected_match": (
                        str(group) in _EXPECTED_GROUPS[lineage]
                        if lineage in _EXPECTED_GROUPS
                        else np.nan
                    ),
                }
            )
    candidates = pd.DataFrame(rows)
    threshold = float(candidates["score_z"].quantile(float(top_quantile)))
    selected = candidates.loc[candidates["score_z"] >= threshold].copy()
    selected["selection_threshold"] = threshold
    return selected


def plot_heatmap(
    matrix,
    title,
    colorbar_label="score",
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    x_rotation=90,
):
    """Plot a compact labeled heatmap and return the Matplotlib axes."""
    import matplotlib.pyplot as plt

    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix must be a pandas DataFrame.")

    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(6, matrix.shape[1] * 0.55), 4.8)
        )

    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        matrix.columns,
        rotation=x_rotation,
        ha="right" if x_rotation else "center",
        fontsize=8,
    )

    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(title)

    ax.figure.colorbar(
        image,
        ax=ax,
        label=colorbar_label,
        fraction=0.035,
        pad=0.02,
    )

    return ax


def prepare_module_overlay_network(
    edge_table,
    module,
    responsibility_triplet,
    eregulon_metadata,
    max_responsible_regions=10,
    max_background_regions=15,
    random_seed=42,
    region_column="Region",
    gene_column="Gene",
    eregulon_column="Region_signature_name",
):
    """Prepare a two-view eRegulon union and CRANE module-overlay network.

    The union view records the SCENIC+ eRegulon-region-gene structure for
    eRegulons connected to the selected module. The overlay view uses the same
    nodes and coordinates, while adding the CRANE module and highlighting
    regions with positive module responsibility.
    """
    ensure_crane_grn_importable()

    from crane_grn.eregulon import (
        extract_eregulon_direction,
        extract_eregulon_tf,
        normalize_eregulon_name,
    )
    from crane_grn.regions import canonicalize_regions

    required_edges = {
        "module",
        "ereg_raw",
        "support_mean",
    }
    missing_edges = required_edges.difference(edge_table.columns)
    if missing_edges:
        raise KeyError(
            "edge_table is missing columns: {}".format(
                sorted(missing_edges)
            )
        )

    metadata_required = {
        region_column,
        gene_column,
        eregulon_column,
    }
    missing_metadata = metadata_required.difference(
        eregulon_metadata.columns
    )
    if missing_metadata:
        raise KeyError(
            "eRegulon metadata is missing columns: {}".format(
                sorted(missing_metadata)
            )
        )

    target_module = str(module)

    selected_edges = edge_table.loc[
        edge_table["module"].astype(str) == target_module
    ].copy()

    if selected_edges.empty:
        raise ValueError(
            "No eRegulon edges were found for module {!r}.".format(
                target_module
            )
        )

    if selected_edges["ereg_raw"].astype(str).duplicated().any():
        raise ValueError(
            "Duplicate module–eRegulon edges were found for the "
            "selected module."
        )

    responsibility = responsibility_triplet.copy()
    responsibility.index = canonicalize_regions(
        responsibility.index.astype(str)
    ).astype(str)
    responsibility.columns = responsibility.columns.astype(str)

    if (
        responsibility.index.has_duplicates
        or responsibility.columns.has_duplicates
    ):
        raise ValueError(
            "Responsibility region and module labels must be unique."
        )

    if target_module not in responsibility.columns:
        raise KeyError(
            "The selected module is absent from responsibility_triplet."
        )

    module_responsibility = pd.to_numeric(
        responsibility[target_module],
        errors="coerce",
    )
    positive_responsibility = module_responsibility.loc[
        module_responsibility > 0
    ]

    responsibility_map = positive_responsibility.astype(float).to_dict()
    responsible_universe = set(responsibility_map)

    metadata = eregulon_metadata.copy()
    metadata[eregulon_column] = metadata[
        eregulon_column
    ].astype(str)
    metadata[region_column] = canonicalize_regions(
        metadata[region_column].astype(str)
    ).astype(str)
    metadata[gene_column] = metadata[gene_column].astype(str)

    selected_edges["ereg_raw"] = selected_edges["ereg_raw"].astype(str)
    selected_edges = selected_edges.sort_values(
        ["support_mean", "ereg_raw"],
        ascending=[False, True],
    )

    regulator_names = {
        extract_eregulon_tf(
            normalize_eregulon_name(eregulon)
        )
        for eregulon in selected_edges["ereg_raw"]
    }

    rng = np.random.default_rng(int(random_seed))
    branches = {}

    for _, edge in selected_edges.iterrows():
        eregulon = str(edge["ereg_raw"])
        clean_name = normalize_eregulon_name(eregulon)

        current = metadata.loc[
            metadata[eregulon_column] == eregulon
        ].copy()

        if current.empty:
            continue

        # Avoid displaying regulator genes again as outer target-gene nodes.
        current = current.loc[
            ~current[gene_column].isin(regulator_names)
        ]

        all_regions = set(current[region_column].unique())
        responsible_all = responsible_universe & all_regions

        # An edge without an overlapping responsible region cannot contribute
        # a CRANE module overlay and is therefore not drawn.
        if not responsible_all:
            continue

        responsible_sorted = sorted(
            responsible_all,
            key=lambda region: (
                -responsibility_map.get(region, 0.0),
                region,
            ),
        )
        responsible_shown = set(
            responsible_sorted[: int(max_responsible_regions)]
        )

        non_responsible = sorted(all_regions - responsible_all)
        background_count = min(
            int(max_background_regions),
            len(non_responsible),
        )

        if background_count:
            background_regions = set(
                rng.choice(
                    np.asarray(non_responsible, dtype=object),
                    size=background_count,
                    replace=False,
                ).tolist()
            )
        else:
            background_regions = set()

        shown_regions = responsible_shown | background_regions

        shown_metadata = current.loc[
            current[region_column].isin(shown_regions)
        ]

        region_gene_pairs = list(
            dict.fromkeys(
                zip(
                    shown_metadata[region_column].astype(str),
                    shown_metadata[gene_column].astype(str),
                )
            )
        )
        genes = list(
            dict.fromkeys(
                gene for _, gene in region_gene_pairs
            )
        )

        branches[eregulon] = {
            "ereg_clean": clean_name,
            "tf_name": extract_eregulon_tf(clean_name),
            "direction": extract_eregulon_direction(clean_name),
            "support_mean": float(edge["support_mean"]),
            "responsible_regions": responsible_shown,
            "background_regions": background_regions,
            "shown_regions": shown_regions,
            "region_gene_pairs": region_gene_pairs,
            "genes": genes,
        }

    if not branches:
        raise ValueError(
            "No drawable eRegulon branch contained a responsible region."
        )

    return {
        "target_module": target_module,
        "branches": branches,
        "parameters": {
            "max_responsible_regions": int(max_responsible_regions),
            "max_background_regions": int(max_background_regions),
            "random_seed": int(random_seed),
        },
    }


def _module_overlay_radial_layout(network):
    """Assign shared radial coordinates to both network views."""
    branches = network["branches"]
    positions = {"module": np.array([0.0, 0.0])}

    branch_names = list(branches)
    branch_angles = np.linspace(
        0,
        2 * np.pi,
        len(branch_names),
        endpoint=False,
    )

    for eregulon, branch_angle in zip(
        branch_names,
        branch_angles,
    ):
        positions[(eregulon, "eregulon")] = (
            2.0
            * np.array(
                [
                    np.cos(branch_angle),
                    np.sin(branch_angle),
                ]
            )
        )

        regions = sorted(
            branches[eregulon]["shown_regions"]
        )
        spread = min(
            0.72,
            0.10 * max(len(regions) - 1, 1),
        )

        if len(regions) > 1:
            region_angles = np.linspace(
                branch_angle - spread,
                branch_angle + spread,
                len(regions),
            )
        else:
            region_angles = np.asarray([branch_angle])

        region_angle_map = {}

        for region, region_angle in zip(
            regions,
            region_angles,
        ):
            positions[(eregulon, "region", region)] = (
                3.45
                * np.array(
                    [
                        np.cos(region_angle),
                        np.sin(region_angle),
                    ]
                )
            )
            region_angle_map[region] = region_angle

        for gene in branches[eregulon]["genes"]:
            connected_angles = [
                region_angle_map[region]
                for region, paired_gene
                in branches[eregulon]["region_gene_pairs"]
                if (
                    paired_gene == gene
                    and region in region_angle_map
                )
            ]

            if connected_angles:
                # Circular mean avoids incorrect placement around -pi/pi.
                gene_angle = np.arctan2(
                    np.mean(np.sin(connected_angles)),
                    np.mean(np.cos(connected_angles)),
                )
            else:
                gene_angle = branch_angle

            positions[(eregulon, "gene", gene)] = (
                4.65
                * np.array(
                    [
                        np.cos(gene_angle),
                        np.sin(gene_angle),
                    ]
                )
            )

    return positions


def _draw_module_overlay_view(
    axis,
    network,
    positions,
    view,
):
    """Draw either the eRegulon union or the CRANE module overlay."""
    if view not in {"union", "module"}:
        raise ValueError("view must be 'union' or 'module'.")

    show_module = view == "module"

    for eregulon, branch in network["branches"].items():
        direction = branch["direction"]
        branch_color = NETWORK_DIRECTION_COLORS.get(
            direction,
            NETWORK_DIRECTION_COLORS["unknown"],
        )

        eregulon_position = positions[
            (eregulon, "eregulon")
        ]

        if show_module:
            axis.plot(
                [
                    positions["module"][0],
                    eregulon_position[0],
                ],
                [
                    positions["module"][1],
                    eregulon_position[1],
                ],
                color="#777777",
                linestyle="--",
                linewidth=1.0,
                alpha=0.40,
                zorder=1,
            )

        for region in sorted(branch["shown_regions"]):
            region_position = positions[
                (eregulon, "region", region)
            ]
            responsible = (
                region in branch["responsible_regions"]
            )

            axis.plot(
                [
                    eregulon_position[0],
                    region_position[0],
                ],
                [
                    eregulon_position[1],
                    region_position[1],
                ],
                color=branch_color,
                linewidth=(
                    1.05
                    if show_module and responsible
                    else 0.40
                ),
                alpha=(
                    0.82
                    if show_module and responsible
                    else 0.26
                ),
                zorder=1,
            )

            if show_module and responsible:
                axis.plot(
                    [
                        positions["module"][0],
                        region_position[0],
                    ],
                    [
                        positions["module"][1],
                        region_position[1],
                    ],
                    color=NETWORK_MODULE_EDGE_COLOR,
                    linewidth=0.85,
                    alpha=0.60,
                    zorder=2,
                )

            axis.scatter(
                *region_position,
                s=38 if show_module and responsible else 9,
                marker=(
                    "D"
                    if show_module and responsible
                    else "o"
                ),
                color=(
                    NETWORK_REGION_COLOR
                    if show_module and responsible
                    else NETWORK_BACKGROUND_REGION_COLOR
                ),
                edgecolor=(
                    "white"
                    if show_module and responsible
                    else "none"
                ),
                linewidth=0.6,
                alpha=0.95 if show_module and responsible else 0.55,
                zorder=4,
            )

        drawn_pairs = set()

        for region, gene in branch["region_gene_pairs"]:
            pair = (region, gene)
            if pair in drawn_pairs:
                continue
            drawn_pairs.add(pair)

            region_key = (eregulon, "region", region)
            gene_key = (eregulon, "gene", gene)

            if (
                region_key not in positions
                or gene_key not in positions
            ):
                continue

            responsible = (
                region in branch["responsible_regions"]
            )

            axis.plot(
                [
                    positions[region_key][0],
                    positions[gene_key][0],
                ],
                [
                    positions[region_key][1],
                    positions[gene_key][1],
                ],
                color=branch_color,
                linewidth=(
                    0.65
                    if show_module and responsible
                    else 0.35
                ),
                alpha=(
                    0.62
                    if show_module and responsible
                    else 0.25
                ),
                zorder=1,
            )

        for gene in branch["genes"]:
            gene_position = positions[
                (eregulon, "gene", gene)
            ]

            axis.scatter(
                *gene_position,
                s=10,
                color=NETWORK_GENE_COLOR,
                alpha=0.75,
                zorder=3,
            )

            radial = gene_position / max(
                np.linalg.norm(gene_position),
                1e-12,
            )

            axis.text(
                *(gene_position + 0.13 * radial),
                str(gene),
                fontsize=5.0,
                ha="left" if radial[0] >= 0 else "right",
                va="center",
                color=NETWORK_TEXT_COLOR,
            )

        axis.scatter(
            *eregulon_position,
            s=450,
            color=branch_color,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )

        axis.text(
            *eregulon_position,
            "{}\n{}".format(
                branch["tf_name"],
                direction,
            ),
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="semibold",
            color=NETWORK_TEXT_COLOR,
            zorder=6,
        )

    if show_module:
        axis.scatter(
            *positions["module"],
            s=1400,
            color=NETWORK_MODULE_COLOR,
            edgecolor=NETWORK_TEXT_COLOR,
            linewidth=1.5,
            zorder=7,
        )
        axis.text(
            *positions["module"],
            "M",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            zorder=8,
        )

    axis.set_aspect("equal")
    axis.set_xlim(-5.35, 5.35)
    axis.set_ylim(-5.35, 5.35)
    axis.axis("off")


def _format_module_title(module, factors_per_line=4):
    """Wrap a semicolon-delimited module label for figure titles."""
    factors = [
        part.strip()
        for part in str(module).split(";")
        if part.strip()
    ]

    return "\n".join(
        " · ".join(
            factors[
                start : start + int(factors_per_line)
            ]
        )
        for start in range(
            0,
            len(factors),
            int(factors_per_line),
        )
    )


def plot_module_overlay_network(
    network,
    output_path=None,
):
    """Draw matched eRegulon-union and CRANE-module overlay views."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    positions = _module_overlay_radial_layout(network)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 7),
        facecolor="white",
    )

    _draw_module_overlay_view(
        axes[0],
        network,
        positions,
        view="union",
    )
    axes[0].set_title(
        "SCENIC+ eRegulon union",
        fontsize=11,
        fontweight="semibold",
    )

    _draw_module_overlay_view(
        axes[1],
        network,
        positions,
        view="module",
    )
    axes[1].set_title(
        "CRANE-GRN module overlay",
        fontsize=11,
        fontweight="semibold",
    )

    figure.suptitle(
        ("Module: " + _format_module_title(network["target_module"])),
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    legend_handles = [
        mpatches.Patch(
            color=NETWORK_DIRECTION_COLORS["+/+"],
            label="eRegulon +/+",
        ),
        mpatches.Patch(
            color=NETWORK_DIRECTION_COLORS["-/+"],
            label="eRegulon -/+",
        ),
        mpatches.Patch(
            color=NETWORK_DIRECTION_COLORS["+/-"],
            label="eRegulon +/-",
        ),
        mpatches.Patch(
            color=NETWORK_DIRECTION_COLORS["-/-"],
            label="eRegulon -/-",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=NETWORK_MODULE_COLOR,
            markeredgecolor=NETWORK_TEXT_COLOR,
            markersize=9,
            label="M: factor module",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=NETWORK_REGION_COLOR,
            markeredgecolor="white",
            markersize=7,
            label="Responsible region",
        ),
    ]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8.5,
    )

    figure.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.12,
        top=0.84,
        wspace=0.02,
    )

    if output_path is not None:
        figure.savefig(
            output_path,
            bbox_inches="tight",
            dpi=200,
        )

    return figure, axes

"""External ARID1A-knockout ATAC validation for the OV application.

Purpose
-------
Record the manuscript-specific analysis that connected CRANE-GRN region-module
responsibility to independently processed WT and ARID1A-knockout ATAC tracks.
The code preserves module and region classification, eRegulon-aware comparison
sets, peak overlap, bigWig signal extraction, KO/WT log2 fold change,
distributional statistics, stability resampling, and maximum-responsibility
quartiles.

Expected upstream inputs
------------------------
- factor_by_module.parquet (factors by modules).
- responsibility_triplet.parquet (regions by modules).
- eRegulon_direct.tsv.
- The OV tumor support aggregate used to construct module-eRegulon edges.
- WT.bigWig, KO.bigWig, WT.narrowPeak.gz, and KO.narrowPeak.gz.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from crane_grn.chip_atlas import regions_to_bed
from crane_grn.eregulon import (
    build_module_eregulon_edges,
    extract_eregulon_direction,
    extract_eregulon_tf,
    normalize_eregulon_name,
)
from crane_grn.regions import canonicalize_regions
from crane_grn.support import validate_aggregate


ATAC_SIGNAL_PSEUDOCOUNT = 1e-3
STABILITY_REPETITIONS = 500
STABILITY_SEED = 1
BOOTSTRAP_REPETITIONS = 2_000
BOOTSTRAP_SEED = 42


def load_tumor_edges(
    aggregate_path,
    responsibility_triplet,
    eregulon_metadata,
    top_k=5,
):
    """Build the tumor module-eRegulon edge table from public artifacts."""
    validate_aggregate(aggregate_path)
    with np.load(aggregate_path, allow_pickle=False) as archive:
        return build_module_eregulon_edges(
            support_mean=archive["support_mean_KE"],
            corr_auc_sup=archive["corr_auc_sup_KE"],
            modules=archive["modules"].astype(str),
            eregulons=archive["eregs"].astype(str),
            responsibility_triplet=responsibility_triplet,
            eregulon_metadata=eregulon_metadata,
            top_k=top_k,
            support_minimum=0.0,
        )


def classify_arid1a_modules(factor_by_module):
    """Classify modules using the factor-by-module membership matrix.

    The matrix, rather than semicolon parsing, is the membership authority.
    A singleton contains ARID1A and no other factor. A multi-factor ARID1A
    module contains ARID1A plus at least one additional factor.
    """
    matrix = factor_by_module.copy()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError("Factor and module labels must be unique.")
    if "ARID1A" not in matrix.index:
        raise KeyError("ARID1A is absent from the factor axis.")
    membership = matrix > 0
    factor_counts = membership.sum(axis=0)
    has_arid1a = membership.loc["ARID1A"]
    singleton = set(matrix.columns[has_arid1a & (factor_counts == 1)])
    multifactor = set(matrix.columns[has_arid1a & (factor_counts > 1)])
    without_arid1a = set(matrix.columns[~has_arid1a])
    if singleton | multifactor | without_arid1a != set(matrix.columns):
        raise RuntimeError("Module classification did not cover the complete axis.")
    return {
        "arid1a_singleton_modules": singleton,
        "arid1a_multifactor_modules": multifactor,
        "non_arid1a_modules": without_arid1a,
    }


def validate_module_name_membership(factor_by_module, modules=None):
    """Check that public factor membership agrees with semicolon module labels."""
    matrix = factor_by_module.copy()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    selected = matrix.columns if modules is None else pd.Index(modules).astype(str)
    rows = []
    for module in selected:
        if module not in matrix.columns:
            rows.append({"module": module, "matches": False, "reason": "missing"})
            continue
        matrix_factors = set(matrix.index[matrix[module] > 0])
        label_factors = {token.strip() for token in module.split(";") if token.strip()}
        rows.append(
            {
                "module": module,
                "matches": matrix_factors == label_factors,
                "matrix_factors": ";".join(sorted(matrix_factors)),
                "label_factors": ";".join(sorted(label_factors)),
            }
        )
    return pd.DataFrame(rows)


def regions_with_responsibility(responsibility_triplet, modules):
    """Return regions with positive responsibility for at least one module."""
    responsibility = responsibility_triplet.copy()
    responsibility.index = canonicalize_regions(responsibility.index).astype(str)
    responsibility.columns = responsibility.columns.astype(str)
    usable = [module for module in modules if module in responsibility.columns]
    if not usable:
        return set()
    return set(responsibility.index[(responsibility[usable] > 0).any(axis=1)])


def classify_arid1a_regions(responsibility_triplet, module_classes):
    """Construct broad and dominant ARID1A responsibility sets.

    arid1a_connected_regions definition: the maximum responsibility among ARID1A-containing
    multi-factor modules must be greater than zero and strictly greater than
    the maximum among modules without ARID1A.
    """
    responsibility = responsibility_triplet.copy()
    responsibility.index = canonicalize_regions(responsibility.index).astype(str)
    responsibility.columns = responsibility.columns.astype(str)
    singleton = sorted(
        set(module_classes["arid1a_singleton_modules"])
        & set(responsibility.columns)
    )
    multifactor = sorted(
        set(module_classes["arid1a_multifactor_modules"])
        & set(responsibility.columns)
    )
    non_arid1a = sorted(
        set(module_classes["non_arid1a_modules"])
        & set(responsibility.columns)
    )
    if not multifactor or not non_arid1a:
        raise ValueError("Both ARID1A multi-factor and non-ARID1A modules are required.")
    singleton_regions = regions_with_responsibility(responsibility, singleton)
    multifactor_regions = regions_with_responsibility(responsibility, multifactor)
    non_arid1a_regions = regions_with_responsibility(responsibility, non_arid1a)
    arid1a_max = responsibility[multifactor].max(axis=1)
    non_arid1a_max = responsibility[non_arid1a].max(axis=1)
    connected_mask = (arid1a_max > 0) & (arid1a_max > non_arid1a_max)
    return {
        "arid1a_singleton_regions": singleton_regions,
        "arid1a_multifactor_regions": multifactor_regions,
        "arid1a_all_responsible_regions": singleton_regions | multifactor_regions,
        "non_arid1a_responsible_regions": non_arid1a_regions,
        "arid1a_connected_regions": set(responsibility.index[connected_mask]),
    }


def _metadata_eregulon_column(eregulon_metadata):
    for column in ("Region_signature_name", "eRegulon_name"):
        if column in eregulon_metadata.columns:
            return column
    raise KeyError("eRegulon metadata has no eRegulon identifier column.")


def eregulon_regions(eregulon_metadata, eregulon, region_universe=None):
    """Return regions for one eRegulon using the original matching hierarchy."""
    metadata = eregulon_metadata.copy()
    identifier_column = _metadata_eregulon_column(metadata)
    metadata["_clean_eregulon"] = metadata[identifier_column].map(
        normalize_eregulon_name
    )
    clean = normalize_eregulon_name(eregulon)
    matched = metadata[metadata["_clean_eregulon"] == clean]
    if matched.empty:
        factor = extract_eregulon_tf(clean)
        direction = extract_eregulon_direction(clean)
        matched = metadata[
            metadata["_clean_eregulon"].str.startswith(factor + "_")
            & metadata["_clean_eregulon"].str.endswith(direction)
        ]
    regions = set(canonicalize_regions(matched["Region"].astype(str)).astype(str))
    if region_universe is not None:
        regions &= set(canonicalize_regions(region_universe).astype(str))
    return regions


def build_arid1a_comparison_sets(
    module_eregulon_edges,
    eregulon_metadata,
    responsibility_triplet,
    module_classes,
    region_classes,
):
    """Build the eRegulon-aware region sets used in the validation.

    The final comparisons use:

    - arid1a_connected_regions: strict ARID1A multi-factor dominance.
    - non_arid1a_regions: unconnected regions from non-ARID1A edges.
    - arid1a_unconnected_regions: unconnected regions within eRegulons
      linked to ARID1A-containing multi-factor modules.

    """
    edges = module_eregulon_edges.copy()
    required = {"module", "ereg_raw"}
    missing = required.difference(edges.columns)
    if missing:
        raise KeyError("Edge table is missing: {}".format(sorted(missing)))
    universe = set(canonicalize_regions(responsibility_triplet.index).astype(str))
    singleton_edges = edges[
        edges["module"].isin(module_classes["arid1a_singleton_modules"])
    ]
    multifactor_edges = edges[
        edges["module"].isin(module_classes["arid1a_multifactor_modules"])
    ]
    non_arid1a_edges = edges[
        edges["module"].isin(module_classes["non_arid1a_modules"])
    ]

    singleton_responsible = set()
    singleton_background = set()
    for eregulon in singleton_edges["ereg_raw"].astype(str):
        current = eregulon_regions(eregulon_metadata, eregulon, universe)
        responsible = current & region_classes["arid1a_singleton_regions"]
        singleton_responsible.update(responsible)
        singleton_background.update(current - responsible)
    singleton_background -= singleton_responsible

    broad_candidates = set()
    connected = set()
    arid1a_unconnected = set()
    for eregulon in multifactor_edges["ereg_raw"].astype(str):
        current = eregulon_regions(eregulon_metadata, eregulon, universe)
        broad_candidates.update(current & region_classes["arid1a_multifactor_regions"])
        connected.update(current & region_classes["arid1a_connected_regions"])
        arid1a_unconnected.update(
            current - region_classes["arid1a_all_responsible_regions"]
        )

    non_arid1a_regions = set()
    for eregulon in non_arid1a_edges["ereg_raw"].astype(str):
        current = eregulon_regions(eregulon_metadata, eregulon, universe)
        non_arid1a_regions.update(
            current - region_classes["arid1a_all_responsible_regions"]
        )
    non_arid1a_regions -= broad_candidates
    arid1a_unconnected -= broad_candidates
    connected -= arid1a_unconnected
    return {
        "singleton_responsible_regions": singleton_responsible,
        "singleton_background_regions": singleton_background,
        "arid1a_candidate_regions": broad_candidates,
        "arid1a_connected_regions": connected,
        "non_arid1a_regions": non_arid1a_regions,
        "arid1a_unconnected_regions": arid1a_unconnected,
    }


def load_narrowpeak(path):
    """Load the genomic interval columns from a narrowPeak file."""
    compression = "gzip" if str(path).endswith(".gz") else None
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        compression=compression,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )


def union_peak_intervals(wt_peak_path, ko_peak_path):
    """Merge the union of WT and KO peak intervals with BEDTools."""
    import pybedtools

    peaks = pd.concat(
        [load_narrowpeak(wt_peak_path), load_narrowpeak(ko_peak_path)],
        ignore_index=True,
    )
    merged = pybedtools.BedTool.from_dataframe(peaks).sort().merge()
    return merged.to_dataframe(names=["chrom", "start", "end"])


def peak_overlapping_regions(region_set, union_peaks):
    """Retain regions overlapping the WT/KO peak union by at least one bp."""
    import pybedtools

    if not region_set:
        return set()
    bed = regions_to_bed(sorted(region_set))
    bed.columns = ["chrom", "start", "end"]
    bed["name"] = bed.apply(
        lambda row: "{}:{}-{}".format(row["chrom"], row["start"], row["end"]),
        axis=1,
    )
    regions = pybedtools.BedTool.from_dataframe(bed).sort()
    peaks = pybedtools.BedTool.from_dataframe(union_peaks).sort()
    overlap = regions.intersect(peaks, wa=True, u=True)
    if overlap.count() == 0:
        return set()
    frame = overlap.to_dataframe(names=["chrom", "start", "end", "name"])
    return set(frame["name"].astype(str))


def extract_atac_signals(
    region_set,
    wt_bigwig,
    ko_bigwig,
    pseudocount=ATAC_SIGNAL_PSEUDOCOUNT,
):
    """Extract mean bigWig signal and calculate KO-versus-WT log2 fold change."""
    from crane_grn.regions import parse_region

    records = []
    for region in sorted(region_set):
        chromosome, start, end = parse_region(region)
        try:
            wt_signal = wt_bigwig.stats(chromosome, start, end, type="mean")[0]
            ko_signal = ko_bigwig.stats(chromosome, start, end, type="mean")[0]
        except (RuntimeError, KeyError, ValueError):
            continue
        if wt_signal is None or ko_signal is None:
            continue
        records.append(
            {"region": region, "WT": float(wt_signal), "KO": float(ko_signal)}
        )
    if not records:
        return pd.DataFrame(columns=["WT", "KO", "atac_log2fc_ko_vs_wt"])
    result = pd.DataFrame(records).set_index("region")
    result["atac_log2fc_ko_vs_wt"] = np.log2(
        (result["KO"] + float(pseudocount))
        / (result["WT"] + float(pseudocount))
    )
    return result


def extract_all_validation_signals(
    comparison_sets,
    union_peaks,
    wt_bigwig,
    ko_bigwig,
    pseudocount=ATAC_SIGNAL_PSEUDOCOUNT,
):
    """Apply peak overlap and signal extraction to each comparison set."""
    output = {}
    for name, regions in comparison_sets.items():
        retained = peak_overlapping_regions(regions, union_peaks)
        output[name] = extract_atac_signals(
            retained, wt_bigwig, ko_bigwig, pseudocount=pseudocount
        )
    return output


def rank_biserial(u_statistic, first_size, second_size):
    return 1.0 - (2.0 * float(u_statistic)) / (first_size * second_size)


def cliffs_delta(first, second):
    """Calculate the original all-pairs Cliff's delta."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    differences = first[:, None] - second[None, :]
    return float(
        (np.sum(differences > 0) - np.sum(differences < 0))
        / differences.size
    )


def hodges_lehmann_shift(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    return float(np.median((first[:, None] - second[None, :]).ravel()))


def bootstrap_median_difference(
    first,
    second,
    n_bootstrap=BOOTSTRAP_REPETITIONS,
    confidence_level=0.95,
    random_seed=BOOTSTRAP_SEED,
):
    """Bootstrap the median difference using the executed percentile method."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    rng = np.random.default_rng(int(random_seed))
    differences = np.empty(int(n_bootstrap), dtype=float)
    for iteration in range(int(n_bootstrap)):
        first_sample = rng.choice(first, size=len(first), replace=True)
        second_sample = rng.choice(second, size=len(second), replace=True)
        differences[iteration] = np.median(first_sample) - np.median(second_sample)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "observed": float(np.median(first) - np.median(second)),
        "lower": float(np.quantile(differences, alpha)),
        "upper": float(np.quantile(differences, 1.0 - alpha)),
    }


def distribution_comparison(first, second, alternative="less"):
    """Run the original one-sided distribution and effect-size summary."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    statistic, pvalue = mannwhitneyu(first, second, alternative=alternative)
    confidence_interval = bootstrap_median_difference(first, second)
    return {
        "n_first": len(first),
        "n_second": len(second),
        "median_first": float(np.median(first)),
        "median_second": float(np.median(second)),
        "u_statistic": float(statistic),
        "pvalue": float(pvalue),
        "rank_biserial": rank_biserial(statistic, len(first), len(second)),
        "cliffs_delta": cliffs_delta(first, second),
        "hodges_lehmann": hodges_lehmann_shift(first, second),
        "bootstrap_median_difference": confidence_interval,
    }


def final_group_comparisons(signal_tables):
    """Compare ARID1A-connected regions with the two final control sets."""
    value_column = "atac_log2fc_ko_vs_wt"
    connected = signal_tables["arid1a_connected_regions"][value_column]
    non_arid1a = signal_tables["non_arid1a_regions"][value_column]
    unconnected = signal_tables["arid1a_unconnected_regions"][value_column]
    return {
        "connected_vs_non_arid1a": distribution_comparison(
            connected, non_arid1a, alternative="less"
        ),
        "connected_vs_arid1a_unconnected": distribution_comparison(
            connected, unconnected, alternative="less"
        ),
    }


def stability_resampling(
    arid1a_candidate_values,
    arid1a_connected_values,
    non_arid1a_values,
    arid1a_unconnected_values,
    repetitions=STABILITY_REPETITIONS,
    random_seed=STABILITY_SEED,
):
    broad = np.asarray(arid1a_candidate_values, dtype=float)
    connected = np.asarray(arid1a_connected_values, dtype=float)
    non_arid1a = np.asarray(non_arid1a_values, dtype=float)
    unconnected = np.asarray(arid1a_unconnected_values, dtype=float)
    sample_size = min(1_600, len(connected) - 1)
    if sample_size < 1:
        raise ValueError("Too few ARID1A-connected regions for stability analysis.")
    rng = np.random.default_rng(int(random_seed))
    rows = []
    for iteration in range(int(repetitions)):
        broad_sample = rng.choice(broad, size=sample_size, replace=False)
        connected_sample = rng.choice(connected, size=sample_size, replace=False)
        for source, values in (
            ("candidate", broad_sample),
            ("connected", connected_sample),
        ):
            for comparator, controls in (
                ("non_arid1a", non_arid1a),
                ("arid1a_unconnected", unconnected),
            ):
                _, pvalue = mannwhitneyu(values, controls, alternative="less")
                rows.append(
                    {
                        "iteration": iteration,
                        "source": source,
                        "comparator": comparator,
                        "pvalue": float(pvalue),
                    }
                )
    return pd.DataFrame(rows)


def attach_max_arid1a_responsibility(
    arid1a_connected_signal,
    responsibility_triplet,
    arid1a_multifactor_modules,
):
    """Attach maximum ARID1A multi-factor responsibility to connected regions."""
    responsibility = responsibility_triplet.copy()
    responsibility.index = canonicalize_regions(responsibility.index).astype(str)
    responsibility.columns = responsibility.columns.astype(str)
    modules = sorted(
        set(arid1a_multifactor_modules) & set(responsibility.columns)
    )
    if not modules:
        raise ValueError("No ARID1A multi-factor module occurs in responsibility.")
    output = arid1a_connected_signal.copy()
    output["max_arid1a_module_responsibility"] = responsibility[modules].max(
        axis=1
    ).reindex(output.index)
    return output


def responsibility_quartile_analysis(connected_table):
    """Stratify connected regions with Q1 strongest and Q4 weakest."""
    required = {
        "max_arid1a_module_responsibility",
        "atac_log2fc_ko_vs_wt",
    }
    missing = required.difference(connected_table.columns)
    if missing:
        raise KeyError("Connected table is missing: {}".format(sorted(missing)))
    table = connected_table.loc[:, sorted(required)].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(table) < 8:
        raise ValueError("Too few connected regions for quartile analysis.")
    descending_rank = table["max_arid1a_module_responsibility"].rank(
        method="first", ascending=False
    )
    table["responsibility_quartile"] = pd.qcut(
        descending_rank, q=4, labels=[1, 2, 3, 4]
    ).astype(int)
    q1 = table.loc[
        table["responsibility_quartile"] == 1, "atac_log2fc_ko_vs_wt"
    ]
    q4 = table.loc[
        table["responsibility_quartile"] == 4, "atac_log2fc_ko_vs_wt"
    ]
    statistic, pvalue = mannwhitneyu(q1, q4, alternative="less")
    summary = (
        table.groupby("responsibility_quartile", observed=True)[
            "atac_log2fc_ko_vs_wt"
        ]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), n="size")
        .reindex([1, 2, 3, 4])
    )
    return table, summary, {"u_statistic": float(statistic), "pvalue": float(pvalue)}


def _ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    return values, np.arange(1, len(values) + 1) / len(values)


def plot_connected_region_cdfs(signal_tables, output_path=None):
    """Draw full-range and near-zero ECDF comparisons for the final groups."""
    column = "atac_log2fc_ko_vs_wt"
    connected = signal_tables["arid1a_connected_regions"][column]
    comparisons = [
        ("Non-ARID1A", signal_tables["non_arid1a_regions"][column]),
        ("ARID1A-unconnected", signal_tables["arid1a_unconnected_regions"][column]),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), facecolor="white")
    for row, (label, control) in enumerate(comparisons):
        for column_position, x_limits in enumerate(((-1.5, 1.5), (-0.15, 0.15))):
            axis = axes[row, column_position]
            for values, name, color in (
                (connected, "ARID1A-connected", "#B2182B"),
                (control, label, "#2166AC"),
            ):
                x, y = _ecdf(values)
                axis.plot(x, y, label=name, color=color, linewidth=2)
            axis.set_xlim(x_limits)
            axis.set_xlabel("ATAC log2 fold change (KO / WT)")
            axis.set_ylabel("Cumulative proportion")
            axis.grid(alpha=0.25)
            if column_position == 0:
                axis.legend(frameon=False)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    return figure


def plot_group_effect_summary(signal_tables, output_path=None):
    """Draw median, IQR, and 1.5-IQR whiskers for the three final groups."""
    column = "atac_log2fc_ko_vs_wt"
    groups = [
        ("ARID1A-connected", signal_tables["arid1a_connected_regions"], "#B2182B"),
        ("Non-ARID1A", signal_tables["non_arid1a_regions"], "#2166AC"),
        ("ARID1A-unconnected", signal_tables["arid1a_unconnected_regions"], "#67A9CF"),
    ]
    figure, axis = plt.subplots(figsize=(7.2, 3.8), facecolor="white")
    for position, (label, frame, color) in enumerate(groups[::-1]):
        values = frame[column].dropna().to_numpy(dtype=float)
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q75 - q25
        low = max(values.min(), q25 - 1.5 * iqr)
        high = min(values.max(), q75 + 1.5 * iqr)
        axis.plot([low, high], [position, position], color=color, linewidth=1.5)
        axis.barh(position, q75 - q25, left=q25, color=color, alpha=0.35)
        axis.scatter(median, position, color=color, edgecolor="white", zorder=3)
    axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_yticks(np.arange(3), [group[0] for group in groups[::-1]])
    axis.set_xlabel("ATAC log2 fold change (KO / WT)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    return figure


def plot_responsibility_quartiles(connected_table, output_path=None):
    """Draw the corrected Q1-strongest to Q4-weakest ATAC response boxplot."""
    table, summary, test = responsibility_quartile_analysis(connected_table)
    data = [
        table.loc[
            table["responsibility_quartile"] == quartile,
            "atac_log2fc_ko_vs_wt",
        ]
        for quartile in (1, 2, 3, 4)
    ]
    figure, axis = plt.subplots(figsize=(5.8, 4.6), facecolor="white")
    boxes = axis.boxplot(data, patch_artist=True, showfliers=False)
    colors = ("#A50F15", "#DE2D26", "#FB6A4A", "#FCAE91")
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
    axis.axhline(0, color="#888888", linestyle="--", linewidth=1)
    axis.set_xticks([1, 2, 3, 4], ["Q1\n(top)", "Q2", "Q3", "Q4\n(bottom)"])
    axis.set_xlabel("Maximum ARID1A-module responsibility quartile")
    axis.set_ylabel("ATAC log2 fold change (KO / WT)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    return figure, summary, test

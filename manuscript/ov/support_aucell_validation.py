"""OV support-AUCell validation analyses used in the manuscript.

This module is a curated reference implementation, not a standalone workflow.

Purpose
-------
Preserve the OV downstream analyses that evaluated whether CRANE-GRN module
support tracks eRegulon activity. The archive covers support-rank quartiles,
within-eRegulon rank permutation, eRegulon-direction comparisons, cell-label
permutation, and region-based versus gene-based AUCell comparisons.

Reproducibility status
----------------------
- Aggregate rank and direction analyses: public-data reproducible.
- Cell-label permutation and cell-resolved region/gene comparison:
  reference-only.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, pearsonr, rankdata, spearmanr

from crane_grn.eregulon import (
    extract_eregulon_direction,
    normalize_eregulon_name,
)
from crane_grn.support import validate_aggregate


DIRECTION_ORDER = ("+/+", "-/+", "+/-", "-/-")
QUARTILE_ORDER = ("Q1 (top)", "Q2", "Q3", "Q4 (bottom)")
QUARTILE_COLORS = ("#A50F15", "#DE2D26", "#FB6A4A", "#FCAE91")
DIRECTION_COLORS = {
    "+/+": "#009E73",
    "-/+": "#56B4E9",
    "+/-": "#E69F00",
    "-/-": "#CC79A7",
}


def load_public_support_aggregate(path):
    """Load and label public six-field support aggregate.

    Returns a dictionary containing the original label arrays plus labeled
    module-by-eRegulon DataFrames for mean support and support-AUCell
    correlation.
    """
    validate_aggregate(path)
    with np.load(path, allow_pickle=False) as archive:
        modules = archive["modules"].astype(str)
        eregulons = archive["eregs"].astype(str)
        return {
            "n_cells": int(archive["n_cells"][0]),
            "modules": modules,
            "eregulons": eregulons,
            "reg_count": archive["reg_count"].astype(np.int64),
            "support_mean": pd.DataFrame(
                archive["support_mean_KE"], index=modules, columns=eregulons
            ),
            "corr_auc_sup": pd.DataFrame(
                archive["corr_auc_sup_KE"], index=modules, columns=eregulons
            ),
        }


def _validate_matching_frames(first, second, first_name, second_name):
    if not isinstance(first, pd.DataFrame) or not isinstance(second, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")
    if first.index.has_duplicates or first.columns.has_duplicates:
        raise ValueError("{} has duplicate axis labels.".format(first_name))
    if second.index.has_duplicates or second.columns.has_duplicates:
        raise ValueError("{} has duplicate axis labels.".format(second_name))
    if not first.index.astype(str).equals(second.index.astype(str)):
        raise ValueError("{} and {} row labels differ.".format(first_name, second_name))
    if not first.columns.astype(str).equals(second.columns.astype(str)):
        raise ValueError(
            "{} and {} column labels differ.".format(first_name, second_name)
        )


def build_support_rank_correlation_table(
    support_mean,
    corr_auc_sup,
    minimum_supported_modules=8,
):
    """Assign support-ranked modules to four within-eRegulon quartiles.

    For each eRegulon, only finite module pairs with support greater than zero
    are considered. Modules are sorted from strongest to weakest support and
    divided into four nearly equal groups with ``numpy.array_split``. This
    preserves the executed notebook definition: Q1 is strongest and Q4 is
    weakest. 
    """
    _validate_matching_frames(
        support_mean, corr_auc_sup, "support_mean", "corr_auc_sup"
    )
    support = support_mean.copy()
    correlation = corr_auc_sup.copy()
    support.index = support.index.astype(str)
    support.columns = support.columns.astype(str)
    correlation.index = correlation.index.astype(str)
    correlation.columns = correlation.columns.astype(str)

    rows = []
    for eregulon in support.columns:
        support_values = support[eregulon].to_numpy(dtype=float)
        correlation_values = correlation[eregulon].to_numpy(dtype=float)
        valid_positions = np.flatnonzero(
            np.isfinite(support_values)
            & (support_values > 0)
            & np.isfinite(correlation_values)
        )
        if len(valid_positions) < int(minimum_supported_modules):
            continue
        ordered_positions = valid_positions[
            np.argsort(support_values[valid_positions])[::-1]
        ]
        quartile_positions = np.array_split(
            np.arange(len(ordered_positions), dtype=int), 4
        )
        clean_name = normalize_eregulon_name(eregulon)
        for quartile_number, positions in enumerate(quartile_positions, start=1):
            for array_position in positions:
                module_position = ordered_positions[array_position]
                global_rank = int(array_position) + 1
                rows.append(
                    {
                        "ereg_raw": eregulon,
                        "ereg_clean": clean_name,
                        "direction": extract_eregulon_direction(clean_name),
                        "module": support.index[module_position],
                        "global_rank": global_rank,
                        "quartile": QUARTILE_ORDER[quartile_number - 1],
                        "q_num": quartile_number,
                        "is_top5": global_rank <= 5,
                        "support_mean": float(support_values[module_position]),
                        "corr": float(correlation_values[module_position]),
                        "n_valid_mods": len(ordered_positions),
                    }
                )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No eRegulon had enough supported modules for quartiles.")
    return table


def summarize_support_rank_coupling(quartile_table):
    """Calculate the executed rank-coupling statistics.

    The returned dictionary contains the Kruskal-Wallis test across quartiles,
    a naive global Spearman correlation, the one-sided Q1-greater-than-Q4
    Mann-Whitney test, and descriptive quartile summaries.
    """
    required = {"q_num", "global_rank", "corr", "is_top5"}
    missing = required.difference(quartile_table.columns)
    if missing:
        raise KeyError("Quartile table is missing: {}".format(sorted(missing)))
    table = quartile_table.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["q_num", "global_rank", "corr"]
    )
    groups = [
        table.loc[table["q_num"] == number, "corr"].to_numpy(dtype=float)
        for number in (1, 2, 3, 4)
    ]
    if any(len(group) == 0 for group in groups):
        raise ValueError("At least one support-rank quartile is empty.")
    h_statistic, kruskal_pvalue = kruskal(*groups)
    spearman_rho, spearman_pvalue = spearmanr(
        table["global_rank"], table["corr"]
    )
    q1_u, q1_q4_pvalue = mannwhitneyu(
        groups[0], groups[3], alternative="greater"
    )
    summary = (
        table.groupby("q_num", observed=True)["corr"]
        .agg(mean="mean", median="median", sem="sem", n="size")
        .reindex([1, 2, 3, 4])
    )
    return {
        "quartile_summary": summary,
        "kruskal_h": float(h_statistic),
        "kruskal_pvalue": float(kruskal_pvalue),
        "spearman_rho": float(spearman_rho),
        "spearman_pvalue": float(spearman_pvalue),
        "q1_q4_u": float(q1_u),
        "q1_q4_pvalue": float(q1_q4_pvalue),
        "top5_mean": float(table.loc[table["is_top5"], "corr"].mean()),
        "top5_sem": float(table.loc[table["is_top5"], "corr"].sem()),
    }


def within_eregulon_rank_permutation(
    quartile_table,
    n_permutations=10_000,
    random_seed=42,
):
    """Test rank-correlation coupling by shuffling ranks within eRegulons.

    Each permutation preserves every eRegulon's correlation distribution and
    number of modules while breaking the within-eRegulon association between
    support rank and AUCell correlation.
    """
    table = quartile_table.loc[:, ["ereg_clean", "global_rank", "corr"]].copy()
    table = table.replace([np.inf, -np.inf], np.nan).dropna()
    x_ranked = rankdata(table["global_rank"].to_numpy(dtype=float), method="average")
    y_ranked = rankdata(table["corr"].to_numpy(dtype=float), method="average")
    x_centered = x_ranked - x_ranked.mean()
    y_centered = y_ranked - y_ranked.mean()
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denominator <= 0:
        raise ValueError("Spearman denominator is zero.")
    observed_rho = float(np.dot(x_centered, y_centered) / denominator)
    group_positions = [
        np.asarray(positions, dtype=int)
        for positions in table.groupby("ereg_clean", sort=False).indices.values()
        if len(positions) > 1
    ]
    rng = np.random.default_rng(int(random_seed))
    null_rhos = np.empty(int(n_permutations), dtype=float)
    for permutation in range(int(n_permutations)):
        permuted = x_centered.copy()
        for positions in group_positions:
            permuted[positions] = rng.permutation(permuted[positions])
        null_rhos[permutation] = np.dot(permuted, y_centered) / denominator
    pvalue = (np.sum(null_rhos <= observed_rho) + 1) / (len(null_rhos) + 1)
    return {
        "observed_rho": observed_rho,
        "null_rhos": null_rhos,
        "pvalue_less": float(pvalue),
        "random_seed": int(random_seed),
    }


def plot_support_rank_coupling(quartile_table, output_path=None):
    """Draw the two retained support-rank coupling panels."""
    statistics = summarize_support_rank_coupling(quartile_table)
    summary = statistics["quartile_summary"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), facecolor="white")

    x = np.arange(4)
    axes[0].bar(
        x,
        summary["mean"],
        yerr=summary["sem"],
        color=QUARTILE_COLORS,
        edgecolor="black",
        capsize=4,
    )
    axes[0].axhline(0, color="#888888", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, ["Q1\n(top)", "Q2", "Q3", "Q4\n(bottom)"])
    axes[0].set_ylabel("Mean support-AUCell correlation")
    axes[0].set_title("Correlation by support-rank quartile")

    sample_size = min(3_000, len(quartile_table))
    sampled = quartile_table.sample(sample_size, random_state=42)
    axes[1].scatter(
        sampled["global_rank"],
        sampled["corr"],
        c=[QUARTILE_COLORS[int(number) - 1] for number in sampled["q_num"]],
        s=10,
        alpha=0.3,
        edgecolors="none",
    )
    coefficients = np.polyfit(sampled["global_rank"], sampled["corr"], 1)
    line_x = np.linspace(sampled["global_rank"].min(), sampled["global_rank"].max(), 100)
    axes[1].plot(line_x, np.polyval(coefficients, line_x), color="black")
    axes[1].axhline(0, color="#888888", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Support rank within eRegulon (1 = strongest)")
    axes[1].set_ylabel("Support-AUCell correlation")
    axes[1].set_title("Rank-correlation relationship")

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    return figure, statistics


def build_direction_correlation_table(support_mean, corr_auc_sup):
    """Collect connected module-eRegulon pairs by eRegulon direction."""
    _validate_matching_frames(
        support_mean, corr_auc_sup, "support_mean", "corr_auc_sup"
    )
    rows = []
    for eregulon in support_mean.columns.astype(str):
        direction = extract_eregulon_direction(
            normalize_eregulon_name(eregulon)
        )
        if direction not in DIRECTION_ORDER:
            continue
        support_values = support_mean[eregulon].to_numpy(dtype=float)
        correlations = corr_auc_sup[eregulon].to_numpy(dtype=float)
        valid = (
            np.isfinite(support_values)
            & np.isfinite(correlations)
            & (support_values > 0)
        )
        for module, support_value, correlation in zip(
            support_mean.index[valid], support_values[valid], correlations[valid]
        ):
            rows.append(
                {
                    "module": str(module),
                    "eregulon": eregulon,
                    "direction": direction,
                    "support_mean": float(support_value),
                    "corr_auc_sup": float(correlation),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No connected pairs were found for direction analysis.")
    return result


def pairwise_direction_tests(direction_table):
    """Run the original two-sided pairwise Mann-Whitney comparisons."""
    rows = []
    for first_position, first in enumerate(DIRECTION_ORDER):
        first_values = direction_table.loc[
            direction_table["direction"] == first, "corr_auc_sup"
        ].dropna()
        for second in DIRECTION_ORDER[first_position + 1 :]:
            second_values = direction_table.loc[
                direction_table["direction"] == second, "corr_auc_sup"
            ].dropna()
            if len(first_values) and len(second_values):
                statistic, pvalue = mannwhitneyu(
                    first_values, second_values, alternative="two-sided"
                )
                rows.append(
                    {
                        "direction_a": first,
                        "direction_b": second,
                        "u_statistic": float(statistic),
                        "pvalue": float(pvalue),
                    }
                )
    return pd.DataFrame(rows)


def plot_direction_correlations(direction_table, output_path=None):
    """Draw the retained direction-specific correlation boxplot."""
    data = [
        direction_table.loc[
            direction_table["direction"] == direction, "corr_auc_sup"
        ].dropna()
        for direction in DIRECTION_ORDER
    ]
    figure, axis = plt.subplots(figsize=(5.8, 5.2), facecolor="white")
    boxes = axis.boxplot(data, patch_artist=True, showfliers=False)
    for patch, direction in zip(boxes["boxes"], DIRECTION_ORDER):
        patch.set_facecolor(DIRECTION_COLORS[direction])
    axis.axhline(0, color="#888888", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(1, 5), DIRECTION_ORDER)
    axis.set_xlabel("eRegulon direction")
    axis.set_ylabel("Support-AUCell correlation")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    return figure


def _aucell_to_frame(aucell):
    if isinstance(aucell, pd.DataFrame):
        frame = aucell.copy()
    elif hasattr(aucell, "to_df"):
        frame = aucell.to_df()
    else:
        values = aucell.X
        if hasattr(values, "toarray"):
            values = values.toarray()
        frame = pd.DataFrame(values, index=aucell.obs_names, columns=aucell.var_names)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("AUCell cell and eRegulon labels must be unique.")
    return frame


def _normalized_column_lookup(columns):
    lookup = {}
    for column in map(str, columns):
        clean = normalize_eregulon_name(column)
        if clean in lookup:
            raise ValueError(
                "Multiple AUCell columns normalize to {!r}.".format(clean)
            )
        lookup[clean] = column
    return lookup


def cell_label_permutation_validation(
    support_memmap_path,
    support_metadata_path,
    region_aucell,
    support_mean,
    n_sample_pairs=1_000,
    n_permutations=100,
    random_seed=42,
):
    """Run the executed cell-label shuffle using the original support tensor.

    Exactly one set of connected module-eRegulon pairs is sampled, after which
    support cell labels are shuffled while AUCell values remain in the original
    order. The original settings were 1,000 sampled pairs, 100 permutations,
    and seed 42. This section is reference-only.
    """
    metadata_path = Path(support_metadata_path)
    memmap_path = Path(support_memmap_path)
    if not metadata_path.exists() or not memmap_path.exists():
        raise FileNotFoundError(
            "The original support memmap and metadata are required."
        )
    with np.load(metadata_path, allow_pickle=True) as metadata:
        cells = metadata["cells"].astype(str)
        modules = metadata["modules"].astype(str)
        eregulons = metadata["eregs"].astype(str)
    auc = _aucell_to_frame(region_aucell)
    common_cells = pd.Index(cells).intersection(auc.index, sort=False)
    auc_lookup = _normalized_column_lookup(auc.columns)
    usable_eregulons = [
        eregulon
        for eregulon in eregulons
        if normalize_eregulon_name(eregulon) in auc_lookup
    ]
    if len(common_cells) == 0 or len(usable_eregulons) == 0:
        raise ValueError("Support and AUCell inputs have no usable overlap.")
    cell_positions = pd.Index(cells).get_indexer(common_cells)
    module_positions = {name: position for position, name in enumerate(modules)}
    eregulon_positions = {
        name: position for position, name in enumerate(eregulons)
    }
    support_labeled = support_mean.copy()
    support_labeled.index = support_labeled.index.astype(str)
    support_labeled.columns = support_labeled.columns.astype(str)
    connected_pairs = [
        (module, eregulon, module_positions[module], eregulon_positions[eregulon])
        for module in modules
        if module in support_labeled.index
        for eregulon in usable_eregulons
        if eregulon in support_labeled.columns
        and float(support_labeled.at[module, eregulon]) > 0
    ]
    if not connected_pairs:
        raise ValueError("No connected module-eRegulon pairs were found.")

    random = np.random.RandomState(int(random_seed))
    if len(connected_pairs) > int(n_sample_pairs):
        chosen = random.choice(
            len(connected_pairs), size=int(n_sample_pairs), replace=False
        )
        connected_pairs = [connected_pairs[position] for position in chosen]
    auc_arrays = {
        eregulon: auc.loc[
            common_cells, auc_lookup[normalize_eregulon_name(eregulon)]
        ].to_numpy(dtype=float)
        for eregulon in usable_eregulons
    }
    support_tensor = np.memmap(
        memmap_path,
        mode="r",
        dtype=np.float32,
        shape=(len(cells), len(modules), len(eregulons)),
    )

    def correlations(current_cell_positions):
        values = []
        for _, eregulon, module_position, eregulon_position in connected_pairs:
            support_values = np.asarray(
                support_tensor[
                    current_cell_positions, module_position, eregulon_position
                ],
                dtype=float,
            )
            auc_values = auc_arrays[eregulon]
            if support_values.std() > 1e-9 and auc_values.std() > 1e-9:
                correlation, _ = pearsonr(support_values, auc_values)
                if np.isfinite(correlation):
                    values.append(float(correlation))
        return np.asarray(values, dtype=float)

    observed = correlations(cell_positions)
    null_means = np.empty(int(n_permutations), dtype=float)
    null_positive_rates = np.empty(int(n_permutations), dtype=float)
    for permutation in range(int(n_permutations)):
        shuffled = cell_positions.copy()
        random.shuffle(shuffled)
        permuted = correlations(shuffled)
        null_means[permutation] = permuted.mean()
        null_positive_rates[permutation] = 100.0 * np.mean(permuted > 0)
    del support_tensor

    observed_mean = float(observed.mean())
    observed_positive_rate = float(100.0 * np.mean(observed > 0))
    pvalue_mean = (np.sum(null_means >= observed_mean) + 1) / (
        len(null_means) + 1
    )
    pvalue_positive = (
        np.sum(null_positive_rates >= observed_positive_rate) + 1
    ) / (len(null_positive_rates) + 1)
    return {
        "observed_correlations": observed,
        "observed_mean": observed_mean,
        "observed_median": float(np.median(observed)),
        "observed_positive_rate": observed_positive_rate,
        "null_means": null_means,
        "null_positive_rates": null_positive_rates,
        "pvalue_mean": float(pvalue_mean),
        "pvalue_positive_rate": float(pvalue_positive),
        "n_sampled_pairs": len(connected_pairs),
        "n_permutations": int(n_permutations),
        "random_seed": int(random_seed),
    }


def cell_level_region_gene_correlations(
    support_memmap_path,
    support_metadata_path,
    region_aucell,
    gene_aucell,
    support_mean,
):
    """Compare cell-level support coupling to region and gene AUCell.

    This reference-only calculation reads the original support tensor and
    computes one Pearson correlation per connected module-eRegulon pair for
    each AUCell representation.
    """
    with np.load(support_metadata_path, allow_pickle=True) as metadata:
        cells = metadata["cells"].astype(str)
        modules = metadata["modules"].astype(str)
        eregulons = metadata["eregs"].astype(str)
    region_frame = _aucell_to_frame(region_aucell)
    gene_frame = _aucell_to_frame(gene_aucell)
    common_cells = (
        pd.Index(cells)
        .intersection(region_frame.index, sort=False)
        .intersection(gene_frame.index, sort=False)
    )
    region_lookup = _normalized_column_lookup(region_frame.columns)
    gene_lookup = _normalized_column_lookup(gene_frame.columns)
    common_clean = set(region_lookup).intersection(gene_lookup)
    usable = [
        eregulon
        for eregulon in eregulons
        if normalize_eregulon_name(eregulon) in common_clean
    ]
    cell_positions = pd.Index(cells).get_indexer(common_cells)
    support_labeled = support_mean.copy()
    support_labeled.index = support_labeled.index.astype(str)
    support_labeled.columns = support_labeled.columns.astype(str)
    tensor = np.memmap(
        support_memmap_path,
        mode="r",
        dtype=np.float32,
        shape=(len(cells), len(modules), len(eregulons)),
    )
    rows = []
    for module_position, module in enumerate(modules):
        if module not in support_labeled.index:
            continue
        for eregulon_position, eregulon in enumerate(eregulons):
            if eregulon not in usable or eregulon not in support_labeled.columns:
                continue
            support_average = float(support_labeled.at[module, eregulon])
            if not np.isfinite(support_average) or support_average <= 0:
                continue
            clean = normalize_eregulon_name(eregulon)
            support_values = np.asarray(
                tensor[cell_positions, module_position, eregulon_position],
                dtype=float,
            )
            region_values = region_frame.loc[
                common_cells, region_lookup[clean]
            ].to_numpy(dtype=float)
            gene_values = gene_frame.loc[
                common_cells, gene_lookup[clean]
            ].to_numpy(dtype=float)
            if (
                support_values.std() <= 1e-9
                or region_values.std() <= 1e-9
                or gene_values.std() <= 1e-9
            ):
                continue
            region_correlation, _ = pearsonr(support_values, region_values)
            gene_correlation, _ = pearsonr(support_values, gene_values)
            rows.append(
                {
                    "module": module,
                    "eregulon": eregulon,
                    "direction": extract_eregulon_direction(clean),
                    "support_mean": support_average,
                    "corr_region": float(region_correlation),
                    "corr_gene": float(gene_correlation),
                }
            )
    del tensor
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No cell-level region/gene correlations were calculated.")
    return result


def summarize_region_gene_correlations(correlation_table):
    """Return mean region- and gene-based correlations by direction."""
    table = correlation_table[
        correlation_table["direction"].isin(DIRECTION_ORDER)
    ]
    return (
        table.groupby("direction", observed=True)[["corr_region", "corr_gene"]]
        .mean()
        .reindex(DIRECTION_ORDER)
    )


def plot_region_gene_correlations(correlation_table, output_path=None):
    """Draw the retained direction-specific region/gene comparison."""
    summary = summarize_region_gene_correlations(correlation_table)
    figure, axis = plt.subplots(figsize=(5.2, 4.3), facecolor="white")
    x = np.arange(len(DIRECTION_ORDER))
    width = 0.36
    axis.bar(
        x - width / 2,
        summary["corr_region"],
        width,
        color=[DIRECTION_COLORS[value] for value in DIRECTION_ORDER],
        edgecolor="#4F4F4F",
        label="Region-based",
    )
    axis.bar(
        x + width / 2,
        summary["corr_gene"],
        width,
        color="white",
        edgecolor=[DIRECTION_COLORS[value] for value in DIRECTION_ORDER],
        hatch="//",
        label="Gene-based",
    )
    axis.axhline(0, color="#888888", linestyle="--", linewidth=1)
    axis.set_xticks(x, DIRECTION_ORDER)
    axis.set_ylabel("Mean correlation")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    return figure

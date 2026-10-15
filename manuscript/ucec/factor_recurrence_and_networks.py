"""Factor recurrence and representative UCEC module-network analyses.

Purpose
-------
Preserve two related manuscript analyses downstream of CRANE-GRN: paired
eRegulon-level recurrence of regulatory factors among the top five modules,
and the candidate filtering and network display used for selected Tumor- and
MUC5B-associated modules.

Reproducibility status
----------------------
Factor recurrence and networks for the two frozen representative modules use
public CRANE-GRN artifacts. Re-running candidate selection additionally needs
the original condition-specific mean region-AUCell values.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from crane_grn.eregulon import (
    build_module_eregulon_edges,
    extract_eregulon_direction,
    extract_eregulon_tf,
    normalize_eregulon_name,
    summarize_modules,
)
from crane_grn.regions import canonicalize_regions
from crane_grn.support import validate_aggregate


PRIMARY_TOP_K = 5
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 43
BOOTSTRAP_CHUNK_SIZE = 1000

MAX_RESPONSIBLE_REGIONS_PER_EREGULON = 10
MAX_BACKGROUND_REGIONS_PER_EREGULON = 15
NETWORK_RANDOM_SEED = 42

# These are the representative modules selected for the manuscript network
# visualization after the candidate-filtering analysis below.
REPRESENTATIVE_TUMOR_MODULE = (
    "CEBPB;FOXA2;PAX8;PBRM1;RUVBL2;SMARCA2;SMARCC1"
)
REPRESENTATIVE_MUC5B_MODULE = (
    "ARID1A;FOXA2;KLF4;PAX8;RUVBL2;SMARCA2;SMARCC1;TFAP2C"
)

TUMOR_COLOR = "#D55E00"
MUC5B_COLOR = "#0072B2"
SHARED_COLOR = "#8A8A8A"
REGION_COLOR = "#009E73"
MODULE_COLOR = "#F0E442"
MODULE_EDGE_COLOR = "#C6A700"
GENE_COLOR = "#777777"
BACKGROUND_REGION_COLOR = "#BDBDBD"
GHOST_COLOR = "#D6D6D6"
TEXT_COLOR = "#2B2B2B"


def load_public_aggregate(path):
    """Load and label one support aggregate."""
    validate_aggregate(path)
    with np.load(path, allow_pickle=False) as archive:
        modules = pd.Index(archive["modules"].astype(str), name="module")
        eregulons = pd.Index(archive["eregs"].astype(str), name="eRegulon")
        if modules.has_duplicates or eregulons.has_duplicates:
            raise ValueError("Aggregate module and eRegulon labels must be unique.")
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


def rank_support_for_factor_recurrence(support):
    """Rank positive support for the paired factor-recurrence input."""
    frame = support.copy()
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Support axes must be uniquely labeled.")
    frame = frame.apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
    return frame.rank(axis=0, ascending=False, method="min", na_option="keep")


def validate_factor_by_module(factor_by_module, modules=None):
    """Validate factor membership and compare it with public module labels."""
    matrix = factor_by_module.copy()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError("Factor and module labels must be unique.")
    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("factor_by_module must contain numeric membership values.")
    selected = matrix.columns if modules is None else pd.Index(modules).astype(str)
    rows = []
    for module in selected:
        if module not in numeric.columns:
            rows.append(
                {
                    "module": module,
                    "matches_label": False,
                    "reason": "missing from factor_by_module",
                }
            )
            continue
        matrix_factors = set(numeric.index[numeric[module] > 0])
        label_factors = {part.strip() for part in module.split(";") if part.strip()}
        rows.append(
            {
                "module": module,
                "matches_label": matrix_factors == label_factors,
                "reason": "",
                "matrix_factors": ";".join(sorted(matrix_factors)),
                "label_factors": ";".join(sorted(label_factors)),
            }
        )
    return pd.DataFrame(rows), numeric


def extract_top_ranked_modules(rank_matrix, condition, eregulons, top_k=PRIMARY_TOP_K):
    """Extract the strongest support-ranked modules for paired eRegulons.

    Lower numerical rank is stronger. Stable mergesort preserves the executed
    deterministic ordering when tied ranks occur.
    """
    ranks = rank_matrix.copy()
    ranks.index = ranks.index.astype(str)
    ranks.columns = ranks.columns.astype(str)
    rows = []
    for eregulon in map(str, eregulons):
        if eregulon not in ranks.columns:
            continue
        vector = pd.to_numeric(ranks[eregulon], errors="coerce").dropna()
        selected = vector.sort_values(ascending=True, kind="mergesort").iloc[
            : int(top_k)
        ]
        for rank_position, (module, original_rank) in enumerate(
            selected.items(), start=1
        ):
            rows.append(
                {
                    "eRegulon": eregulon,
                    "condition": str(condition),
                    "rank_pos": int(rank_position),
                    "module": str(module),
                    "original_rank": float(original_rank),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["eRegulon", "condition", "rank_pos", "module", "original_rank"],
    )


def build_factor_recurrence_table(
    selected_modules,
    paired_eregulons,
    factor_by_module,
):
    """Build a complete paired eRegulon-by-condition-by-factor table.

    Recurrence is the fraction of selected top modules containing a factor.
    Factors absent from an eRegulon-condition pair receive an explicit zero.
    """
    required = {"eRegulon", "condition", "rank_pos", "module"}
    missing = required.difference(selected_modules.columns)
    if missing:
        raise KeyError("Selected-module table is missing: {}.".format(sorted(missing)))
    module_labels = selected_modules["module"].astype(str).unique()
    label_validation, membership = validate_factor_by_module(
        factor_by_module, modules=module_labels
    )
    if not label_validation["matches_label"].all():
        failed = label_validation.loc[
            ~label_validation["matches_label"], "module"
        ].tolist()
        raise ValueError(
            "Factor membership disagrees with public module labels: {}.".format(
                failed[:5]
            )
        )

    occurrence_rows = []
    for row in selected_modules.itertuples(index=False):
        if row.module not in membership.columns:
            raise KeyError("Module is absent from factor_by_module: {}".format(row.module))
        factors = membership.index[membership[row.module] > 0].astype(str)
        for factor in factors:
            occurrence_rows.append(
                {
                    "eRegulon": str(row.eRegulon),
                    "condition": str(row.condition),
                    "rank_pos": int(row.rank_pos),
                    "module": str(row.module),
                    "factor": factor,
                }
            )
    occurrence = pd.DataFrame(occurrence_rows)
    if occurrence.empty:
        raise ValueError("No factor occurrences were extracted.")
    denominators = (
        selected_modules.groupby(["eRegulon", "condition"], observed=True)
        .size()
        .rename("n_selected_modules")
        .reset_index()
    )
    counts = (
        occurrence.groupby(["eRegulon", "condition", "factor"], observed=True)
        .size()
        .rename("topk_count")
    )
    all_factors = sorted(occurrence["factor"].unique())
    complete_index = pd.MultiIndex.from_product(
        [
            list(map(str, paired_eregulons)),
            ["tumor", "muc5b"],
            all_factors,
        ],
        names=["eRegulon", "condition", "factor"],
    )
    recurrence = (
        counts.reindex(complete_index, fill_value=0)
        .rename("topk_count")
        .reset_index()
        .merge(
            denominators,
            on=["eRegulon", "condition"],
            how="left",
            validate="many_to_one",
        )
    )
    if recurrence["n_selected_modules"].isna().any():
        raise ValueError("At least one paired eRegulon-condition has no modules.")
    recurrence["recurrence_rate"] = (
        recurrence["topk_count"] / recurrence["n_selected_modules"]
    )
    recurrence["present"] = recurrence["topk_count"] > 0
    if not recurrence["recurrence_rate"].between(0, 1).all():
        raise ValueError("Factor recurrence lies outside [0, 1].")
    return occurrence, recurrence, label_validation


def build_paired_factor_tables(factor_recurrence, top_k=PRIMARY_TOP_K):
    """Calculate paired Tumor-minus-MUC5B recurrence for every factor."""
    rate_wide = (
        factor_recurrence.pivot_table(
            index=["eRegulon", "factor"],
            columns="condition",
            values="recurrence_rate",
            aggfunc="first",
            fill_value=0.0,
        )
        .reset_index()
    )
    missing = {"tumor", "muc5b"}.difference(rate_wide.columns)
    if missing:
        raise KeyError("Paired recurrence is missing conditions: {}.".format(sorted(missing)))
    rate_wide["delta"] = rate_wide["tumor"] - rate_wide["muc5b"]
    rate_wide["slot_delta_equivalent"] = rate_wide["delta"] * int(top_k)

    count_summary = (
        factor_recurrence.groupby(["factor", "condition"], observed=True)[
            "topk_count"
        ]
        .sum()
        .unstack(fill_value=0)
        .rename(
            columns={
                "tumor": "total_count_tumor",
                "muc5b": "total_count_muc5b",
            }
        )
    )
    prevalence_summary = (
        factor_recurrence.groupby(["factor", "condition"], observed=True)[
            "present"
        ]
        .mean()
        .unstack(fill_value=0.0)
        .rename(
            columns={
                "tumor": "prevalence_tumor",
                "muc5b": "prevalence_muc5b",
            }
        )
    )
    summary = (
        rate_wide.groupby("factor", observed=True)
        .agg(
            mean_recurrence_tumor=("tumor", "mean"),
            mean_recurrence_muc5b=("muc5b", "mean"),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            sd_delta=("delta", "std"),
            n_paired_eregulons=("eRegulon", "nunique"),
        )
        .join(count_summary, how="left")
        .join(prevalence_summary, how="left")
        .reset_index()
    )
    summary["mean_slot_delta_equivalent"] = summary["mean_delta"] * int(top_k)
    summary["absolute_mean_delta"] = summary["mean_delta"].abs()
    summary["prevalence_delta"] = (
        summary["prevalence_tumor"] - summary["prevalence_muc5b"]
    )
    summary["direction_descriptive"] = np.select(
        [summary["mean_delta"] > 0, summary["mean_delta"] < 0],
        ["tumor", "muc5b"],
        default="equal",
    )
    summary = summary.sort_values(
        ["absolute_mean_delta", "factor"], ascending=[False, True]
    ).reset_index(drop=True)
    return rate_wide, summary


def paired_bootstrap_mean_ci(
    difference_matrix,
    n_bootstrap=BOOTSTRAP_REPETITIONS,
    random_seed=BOOTSTRAP_SEED,
    chunk_size=BOOTSTRAP_CHUNK_SIZE,
    confidence_level=0.95,
):
    """Bootstrap paired eRegulons together to estimate factor-level mean CIs."""
    if difference_matrix.empty:
        raise ValueError("The factor-difference matrix is empty.")
    differences = difference_matrix.to_numpy(dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Factor differences must be complete and finite.")
    n_eregulons, n_factors = differences.shape
    bootstrap_means = np.empty((int(n_bootstrap), n_factors), dtype=float)
    rng = np.random.default_rng(int(random_seed))
    completed = 0
    while completed < int(n_bootstrap):
        current_chunk = min(int(chunk_size), int(n_bootstrap) - completed)
        sampled_indices = rng.integers(
            low=0,
            high=n_eregulons,
            size=(current_chunk, n_eregulons),
        )
        bootstrap_means[completed : completed + current_chunk, :] = differences[
            sampled_indices, :
        ].mean(axis=1)
        completed += current_chunk
    alpha = (1.0 - float(confidence_level)) / 2.0
    return pd.DataFrame(
        {
            "factor": difference_matrix.columns.astype(str),
            "mean_delta_ci_lower": np.percentile(
                bootstrap_means, 100 * alpha, axis=0
            ),
            "mean_delta_ci_upper": np.percentile(
                bootstrap_means, 100 * (1.0 - alpha), axis=0
            ),
        }
    )


def paired_factor_recurrence_analysis(
    rank_tumor,
    rank_muc5b,
    highly_rewired_eregulons,
    factor_by_module,
    top_k=PRIMARY_TOP_K,
):
    """Run the retained paired factor-recurrence and bootstrap calculation."""
    tumor = rank_tumor.copy()
    muc5b = rank_muc5b.copy()
    tumor.index = tumor.index.astype(str)
    tumor.columns = tumor.columns.astype(str)
    muc5b.index = muc5b.index.astype(str)
    muc5b.columns = muc5b.columns.astype(str)
    requested = list(dict.fromkeys(map(str, highly_rewired_eregulons)))
    paired = [
        eregulon
        for eregulon in requested
        if eregulon in tumor.columns and eregulon in muc5b.columns
    ]
    if not paired:
        raise ValueError("No highly rewired eRegulons occur in both rank matrices.")
    selected_modules = pd.concat(
        [
            extract_top_ranked_modules(tumor, "tumor", paired, top_k=top_k),
            extract_top_ranked_modules(muc5b, "muc5b", paired, top_k=top_k),
        ],
        ignore_index=True,
    )
    occurrence, recurrence, label_validation = build_factor_recurrence_table(
        selected_modules, paired, factor_by_module
    )
    paired_factors, factor_summary = build_paired_factor_tables(
        recurrence, top_k=top_k
    )
    difference_matrix = paired_factors.pivot(
        index="eRegulon", columns="factor", values="delta"
    ).sort_index(axis=0).sort_index(axis=1)
    confidence_intervals = paired_bootstrap_mean_ci(difference_matrix)
    factor_results = factor_summary.merge(
        confidence_intervals, on="factor", how="left", validate="one_to_one"
    )
    return {
        "paired_eregulons": paired,
        "selected_modules": selected_modules,
        "factor_occurrences": occurrence,
        "factor_recurrence": recurrence,
        "paired_factor_effects": paired_factors,
        "factor_difference_matrix": difference_matrix,
        "factor_results": factor_results,
        "module_label_validation": label_validation,
    }


def plot_factor_recurrence(factor_results, output_path=None):
    """Plot mean paired recurrence differences with bootstrap intervals."""
    required = {
        "factor",
        "mean_delta",
        "mean_delta_ci_lower",
        "mean_delta_ci_upper",
    }
    missing = required.difference(factor_results.columns)
    if missing:
        raise KeyError("Factor results are missing: {}.".format(sorted(missing)))
    table = factor_results.loc[:, sorted(required)].copy()
    table = table.dropna().sort_values(
        ["mean_delta", "factor"], ascending=[False, True]
    )
    y_positions = np.arange(len(table))
    colors = np.where(table["mean_delta"] > 0, TUMOR_COLOR, MUC5B_COLOR)
    figure, axis = plt.subplots(
        figsize=(6.6, max(4.0, 0.28 * len(table) + 1.4)), facecolor="white"
    )
    axis.axvline(0, color="#4D4D4D", linewidth=0.9)
    for position, (_, row), color in zip(
        y_positions, table.iterrows(), colors
    ):
        axis.errorbar(
            row["mean_delta"],
            position,
            xerr=np.asarray(
                [
                    [row["mean_delta"] - row["mean_delta_ci_lower"]],
                    [row["mean_delta_ci_upper"] - row["mean_delta"]],
                ]
            ),
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=2,
        )
    axis.scatter(
        table["mean_delta"], y_positions, c=colors, s=36, edgecolor="white", zorder=3
    )
    axis.set_yticks(y_positions, table["factor"])
    axis.invert_yaxis()
    axis.set_xlabel("Mean recurrence difference (Tumor - MUC5B)")
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight")
    return figure, axis


def build_condition_edges(
    aggregate_path,
    responsibility_triplet,
    eregulon_metadata,
    region_aucell_mean=None,
    top_k=PRIMARY_TOP_K,
):
    """Build one condition's top module-eRegulon edges by public labels.

    `region_aucell_mean` is an optional Series indexed by raw eRegulon name.
    It is needed only for reproducing candidate filtering, not for rebuilding
    the frozen representative networks.
    """
    aggregate = load_public_aggregate(aggregate_path)
    edges = build_module_eregulon_edges(
        support_mean=aggregate["support_mean"],
        corr_auc_sup=aggregate["corr_auc_sup"],
        modules=aggregate["modules"],
        eregulons=aggregate["eregulons"],
        responsibility_triplet=responsibility_triplet,
        eregulon_metadata=eregulon_metadata,
        top_k=int(top_k),
        support_minimum=0.0,
    )
    if region_aucell_mean is not None:
        auc_mean = pd.Series(region_aucell_mean).copy()
        auc_mean.index = auc_mean.index.astype(str)
        if auc_mean.index.has_duplicates:
            raise ValueError("Condition AUCell means must have unique eRegulon labels.")
        edges["ereg_auc_mean"] = edges["ereg_raw"].map(auc_mean)
    return edges


def build_candidate_module_summary(
    tumor_edges,
    muc5b_edges,
    factor_by_module,
):
    """Build the condition-comparison table used for module candidacy."""
    summaries = []
    for condition, edges in (("tumor", tumor_edges), ("muc5b", muc5b_edges)):
        summary = summarize_modules(edges)
        if "ereg_auc_mean" in edges.columns:
            auc_summary = (
                edges.groupby("module", observed=True)["ereg_auc_mean"]
                .mean()
                .rename("auc_mean")
            )
            summary = summary.join(auc_summary, on="module")
        renamed = summary.rename(
            columns={
                column: "{}_{}".format(column, condition)
                for column in summary.columns
                if column != "module"
            }
        )
        summaries.append(renamed)
    combined = summaries[0].merge(
        summaries[1], on="module", how="inner", validate="one_to_one"
    )
    _, membership = validate_factor_by_module(
        factor_by_module, modules=combined["module"]
    )
    missing_modules = pd.Index(combined["module"]).difference(membership.columns)
    if len(missing_modules):
        raise ValueError(
            "Candidate summary modules are absent from factor_by_module: {}.".format(
                missing_modules[:5].tolist()
            )
        )
    combined["n_factors"] = (
        membership.loc[:, combined["module"]].gt(0).sum(axis=0).to_numpy()
    )
    combined["support_delta"] = (
        combined["support_mean_tumor"] - combined["support_mean_muc5b"]
    )
    combined["n_resp_delta"] = (
        combined["n_resp_total_tumor"] - combined["n_resp_total_muc5b"]
    )
    combined["corr_positive_fraction_average"] = (
        combined["corr_positive_fraction_tumor"]
        + combined["corr_positive_fraction_muc5b"]
    ) / 2.0
    if {"auc_mean_tumor", "auc_mean_muc5b"}.issubset(combined.columns):
        combined["auc_delta"] = (
            combined["auc_mean_tumor"] - combined["auc_mean_muc5b"]
        )
    return combined


def select_candidate_modules(module_summary):
    """Apply the executed complexity, coherence, and direction filters.

    The previously calculated support upper-quartile threshold was inactive in
    the executed selection and is intentionally absent here.
    """
    required = {
        "module",
        "n_ereg_tumor",
        "n_ereg_muc5b",
        "n_factors",
        "n_resp_total_tumor",
        "n_resp_total_muc5b",
        "support_delta",
        "auc_delta",
        "corr_positive_fraction_average",
    }
    missing = required.difference(module_summary.columns)
    if missing:
        raise KeyError(
            "Candidate selection requires columns: {}.".format(sorted(missing))
        )
    base_mask = (
        module_summary["n_ereg_tumor"].between(2, 8)
        & module_summary["n_ereg_muc5b"].between(2, 8)
        & module_summary["n_factors"].between(3, 8)
        & (module_summary["n_resp_total_tumor"] <= 50)
        & (module_summary["n_resp_total_muc5b"] <= 50)
        & (module_summary["corr_positive_fraction_average"] >= 0.2)
    )
    base_candidates = module_summary[base_mask].copy()
    tumor_candidates = base_candidates[
        (base_candidates["n_resp_total_muc5b"] >= 2)
        & (
            base_candidates["n_resp_total_tumor"]
            > base_candidates["n_resp_total_muc5b"]
        )
        & (base_candidates["auc_delta"] > 0)
        & (base_candidates["support_delta"] > 0)
    ].sort_values("corr_positive_fraction_average", ascending=False)
    muc5b_candidates = base_candidates[
        (base_candidates["n_resp_total_tumor"] >= 2)
        & (
            base_candidates["n_resp_total_muc5b"]
            > base_candidates["n_resp_total_tumor"]
        )
        & (base_candidates["auc_delta"] < 0)
        & (base_candidates["support_delta"] < 0)
    ].sort_values("corr_positive_fraction_average", ascending=False)
    return {
        "base_candidates": base_candidates,
        "tumor_candidates": tumor_candidates,
        "muc5b_candidates": muc5b_candidates,
        "counts": {
            "base": int(len(base_candidates)),
            "tumor": int(len(tumor_candidates)),
            "muc5b": int(len(muc5b_candidates)),
        },
    }


def _unique_edge_value(indexed_edges, eregulon, column):
    if eregulon not in indexed_edges.index:
        return 0.0
    value = indexed_edges.loc[eregulon, column]
    if isinstance(value, pd.Series):
        raise ValueError(
            "Duplicate module-eRegulon edges were found for {}.".format(eregulon)
        )
    return float(value)


def prepare_representative_network(
    target_module,
    tumor_edges,
    muc5b_edges,
    eregulon_metadata,
    responsibility_triplet,
    max_responsible_regions=MAX_RESPONSIBLE_REGIONS_PER_EREGULON,
    max_background_regions=MAX_BACKGROUND_REGIONS_PER_EREGULON,
    random_seed=NETWORK_RANDOM_SEED,
    region_column="Region",
    gene_column="Gene",
    eregulon_column="Region_signature_name",
):
    """Prepare one frozen representative module for a three-view network.

    For each connected eRegulon, at most ten positive-responsibility regions
    with the largest responsibility values are shown. Up to 15 non-responsible
    eRegulon regions are sampled with seed 42. 
    """
    required_edges = {"module", "ereg_raw", "support_mean"}
    for name, edges in (("tumor_edges", tumor_edges), ("muc5b_edges", muc5b_edges)):
        missing = required_edges.difference(edges.columns)
        if missing:
            raise KeyError("{} is missing {}.".format(name, sorted(missing)))
        duplicated = edges.duplicated(subset=["module", "ereg_raw"])
        if duplicated.any():
            raise ValueError("{} contains duplicate module-eRegulon edges.".format(name))
    metadata_required = {region_column, gene_column, eregulon_column}
    missing_metadata = metadata_required.difference(eregulon_metadata.columns)
    if missing_metadata:
        raise KeyError(
            "eRegulon metadata is missing: {}.".format(sorted(missing_metadata))
        )
    responsibility = responsibility_triplet.copy()
    responsibility.index = canonicalize_regions(responsibility.index).astype(str)
    responsibility.columns = responsibility.columns.astype(str)
    if responsibility.index.has_duplicates or responsibility.columns.has_duplicates:
        raise ValueError("Responsibility region and module labels must be unique.")
    target_module = str(target_module)
    if target_module not in responsibility.columns:
        raise KeyError("Representative module is absent from responsibility_triplet.")

    tumor_subset = tumor_edges[tumor_edges["module"] == target_module].copy()
    muc5b_subset = muc5b_edges[muc5b_edges["module"] == target_module].copy()
    tumor_set = set(tumor_subset["ereg_raw"].astype(str))
    muc5b_set = set(muc5b_subset["ereg_raw"].astype(str))
    union_eregulons = sorted(tumor_set | muc5b_set)
    if not union_eregulons:
        raise ValueError("No eRegulons are connected to the selected module.")
    clean_tumor = {normalize_eregulon_name(value) for value in tumor_set}
    clean_muc5b = {normalize_eregulon_name(value) for value in muc5b_set}
    edge_classes = {
        "shared": clean_tumor & clean_muc5b,
        "tumor_specific": clean_tumor - clean_muc5b,
        "muc5b_specific": clean_muc5b - clean_tumor,
    }

    module_responsibility = pd.to_numeric(
        responsibility[target_module], errors="coerce"
    )
    positive = module_responsibility[module_responsibility > 0]
    responsibility_map = positive.astype(float).to_dict()
    responsible_universe = set(responsibility_map)
    tumor_indexed = tumor_subset.set_index(tumor_subset["ereg_raw"].astype(str))
    muc5b_indexed = muc5b_subset.set_index(muc5b_subset["ereg_raw"].astype(str))
    metadata = eregulon_metadata.copy()
    metadata[eregulon_column] = metadata[eregulon_column].astype(str)
    metadata[region_column] = canonicalize_regions(
        metadata[region_column].astype(str)
    ).astype(str)
    rng = np.random.default_rng(int(random_seed))
    branches = {}
    for eregulon in union_eregulons:
        clean_name = normalize_eregulon_name(eregulon)
        if clean_name in edge_classes["tumor_specific"]:
            edge_class = "tumor_specific"
        elif clean_name in edge_classes["muc5b_specific"]:
            edge_class = "muc5b_specific"
        else:
            edge_class = "shared"
        current = metadata[metadata[eregulon_column] == eregulon].copy()
        if current.empty:
            continue
        all_regions = set(current[region_column].unique())
        responsible_all = responsible_universe & all_regions
        if not responsible_all:
            continue
        responsible_sorted = sorted(
            responsible_all,
            key=lambda region: (-responsibility_map.get(region, 0.0), region),
        )
        responsible_shown = set(
            responsible_sorted[: int(max_responsible_regions)]
        )
        non_responsible = sorted(all_regions - responsible_all)
        background_count = min(int(max_background_regions), len(non_responsible))
        if background_count:
            background = set(
                rng.choice(
                    np.asarray(non_responsible, dtype=object),
                    size=background_count,
                    replace=False,
                ).tolist()
            )
        else:
            background = set()
        shown_regions = responsible_shown | background
        shown_metadata = current[current[region_column].isin(shown_regions)]
        region_gene_pairs = list(
            dict.fromkeys(
                zip(
                    shown_metadata[region_column].astype(str),
                    shown_metadata[gene_column].astype(str),
                )
            )
        )
        genes = list(dict.fromkeys(gene for _, gene in region_gene_pairs))
        branches[eregulon] = {
            "ereg_clean": clean_name,
            "tf_name": extract_eregulon_tf(clean_name),
            "direction": extract_eregulon_direction(clean_name),
            "edge_class": edge_class,
            "in_tumor": eregulon in tumor_set,
            "in_muc5b": eregulon in muc5b_set,
            "responsible_regions": responsible_shown,
            "background_regions": background,
            "shown_regions": shown_regions,
            "region_gene_pairs": region_gene_pairs,
            "genes": genes,
            "support_tumor": _unique_edge_value(
                tumor_indexed, eregulon, "support_mean"
            ),
            "support_muc5b": _unique_edge_value(
                muc5b_indexed, eregulon, "support_mean"
            ),
        }
    if not branches:
        raise ValueError("No drawable eRegulon branch has a responsible region.")
    return {
        "target_module": target_module,
        "branches": branches,
        "edge_classes": edge_classes,
        "parameters": {
            "max_responsible_regions": int(max_responsible_regions),
            "max_background_regions": int(max_background_regions),
            "random_seed": int(random_seed),
        },
    }


def _radial_network_layout(network):
    branches = network["branches"]
    positions = {"module": np.array([0.0, 0.0])}
    branch_names = list(branches)
    branch_angles = np.linspace(0, 2 * np.pi, len(branch_names), endpoint=False)
    for branch_index, (eregulon, angle) in enumerate(zip(branch_names, branch_angles)):
        positions[(eregulon, "tf")] = 2.0 * np.array([np.cos(angle), np.sin(angle)])
        regions = sorted(branches[eregulon]["shown_regions"])
        spread = min(0.72, 0.10 * max(len(regions) - 1, 1))
        region_angles = (
            np.linspace(angle - spread, angle + spread, len(regions))
            if len(regions) > 1
            else np.asarray([angle])
        )
        region_angle_map = {}
        for region_index, (region, region_angle) in enumerate(
            zip(regions, region_angles)
        ):
            positions[(eregulon, "region", region)] = 3.45 * np.array(
                [np.cos(region_angle), np.sin(region_angle)]
            )
            region_angle_map[region] = region_angle
        for gene_index, gene in enumerate(branches[eregulon]["genes"]):
            connected_angles = [
                region_angle_map[region]
                for region, paired_gene in branches[eregulon]["region_gene_pairs"]
                if paired_gene == gene and region in region_angle_map
            ]
            gene_angle = (
                float(np.mean(connected_angles))
                if connected_angles
                else angle + 0.02 * gene_index
            )
            positions[(eregulon, "gene", gene)] = 4.65 * np.array(
                [np.cos(gene_angle), np.sin(gene_angle)]
            )
    return positions


def _draw_network_view(axis, network, positions, view):
    branches = network["branches"]
    structure_only = view == "union"
    for eregulon, branch in branches.items():
        if branch["edge_class"] == "tumor_specific":
            branch_color = TUMOR_COLOR
        elif branch["edge_class"] == "muc5b_specific":
            branch_color = MUC5B_COLOR
        else:
            branch_color = SHARED_COLOR
        in_condition = (
            True
            if structure_only
            else branch["in_tumor"]
            if view == "tumor"
            else branch["in_muc5b"]
        )
        alpha = 0.62 if structure_only else 1.0 if in_condition else 0.10
        tf_position = positions[(eregulon, "tf")]
        if not structure_only and in_condition:
            axis.plot(
                [positions["module"][0], tf_position[0]],
                [positions["module"][1], tf_position[1]],
                color=SHARED_COLOR,
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
                zorder=1,
            )
        for region in sorted(branch["shown_regions"]):
            region_position = positions[(eregulon, "region", region)]
            responsible = region in branch["responsible_regions"]
            axis.plot(
                [tf_position[0], region_position[0]],
                [tf_position[1], region_position[1]],
                color=SHARED_COLOR if structure_only else branch_color,
                linewidth=1.05 if responsible and in_condition else 0.35,
                alpha=(0.86 if responsible else 0.22) * alpha,
                zorder=1,
            )
            if not structure_only and in_condition and responsible:
                axis.plot(
                    [positions["module"][0], region_position[0]],
                    [positions["module"][1], region_position[1]],
                    color=MODULE_EDGE_COLOR,
                    linewidth=0.85,
                    alpha=0.58,
                    zorder=2,
                )
            axis.scatter(
                *region_position,
                s=38 if responsible and in_condition and not structure_only else 9,
                marker="D" if responsible and in_condition and not structure_only else "o",
                color=REGION_COLOR
                if responsible and in_condition and not structure_only
                else BACKGROUND_REGION_COLOR,
                alpha=alpha,
                edgecolor="white" if responsible else "none",
                linewidth=0.6,
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
            if region_key not in positions or gene_key not in positions:
                continue
            axis.plot(
                [positions[region_key][0], positions[gene_key][0]],
                [positions[region_key][1], positions[gene_key][1]],
                color=SHARED_COLOR if structure_only else branch_color,
                linewidth=0.55,
                alpha=0.46 * alpha,
                zorder=1,
            )
        for gene in branch["genes"]:
            position = positions[(eregulon, "gene", gene)]
            axis.scatter(*position, s=10, color=GENE_COLOR, alpha=alpha, zorder=3)
            radial = position / max(np.linalg.norm(position), 1e-12)
            axis.text(
                *(position + 0.13 * radial),
                str(gene),
                fontsize=5.0,
                ha="left" if radial[0] >= 0 else "right",
                va="center",
                color=TEXT_COLOR if in_condition else GHOST_COLOR,
                alpha=alpha,
            )
        axis.scatter(
            *tf_position,
            s=430,
            color=SHARED_COLOR if structure_only else branch_color,
            alpha=alpha,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
        axis.text(
            *tf_position,
            branch["tf_name"],
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="semibold",
            color=TEXT_COLOR,
            alpha=alpha,
            zorder=6,
        )
    if not structure_only:
        axis.scatter(
            *positions["module"],
            s=1350,
            color=MODULE_COLOR,
            edgecolor=TEXT_COLOR,
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
    factors = [part.strip() for part in str(module).split(";") if part.strip()]
    return "\n".join(
        " · ".join(factors[start : start + int(factors_per_line)])
        for start in range(0, len(factors), int(factors_per_line))
    )


def plot_representative_network(network, output_path=None):
    """Draw union, Tumor, and normal-epithelial views of one network."""
    positions = _radial_network_layout(network)
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 6.1), facecolor="white")
    views = ["union", "tumor", "muc5b"]
    titles = ["Union structure", "Tumor", "Normal epithelial"]
    for axis, view, title in zip(axes, views, titles):
        _draw_network_view(axis, network, positions, view)
        axis.set_title(title, fontsize=11, fontweight="semibold")
    figure.suptitle(
        _format_module_title(network["target_module"]),
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    legend_handles = [
        mpatches.Patch(color=SHARED_COLOR, label="Shared eRegulon"),
        mpatches.Patch(color=TUMOR_COLOR, label="Tumor-specific eRegulon"),
        mpatches.Patch(color=MUC5B_COLOR, label="MUC5B-specific eRegulon"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=MODULE_COLOR,
            markeredgecolor=TEXT_COLOR,
            markersize=9,
            label="M: factor module",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=REGION_COLOR,
            markeredgecolor="white",
            markersize=7,
            label="Responsible region",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.10, top=0.84, wspace=0.02)
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight")
    return figure, axes

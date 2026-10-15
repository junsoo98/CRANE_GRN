"""Tumor-versus-MUC5B regulatory rewiring analyses for the UCEC application.

Purpose
-------
Preserve the manuscript-specific statistic that compares the relative ordering
of CRANE-GRN modules supporting the same eRegulon in Tumor and MUC5B epithelial
cells. The file also records the within-condition split controls and the
association with regulon-specificity scores.

Reproducibility status
----------------------
The primary Tumor-versus-MUC5B rewiring calculation is public-data
reproducible. Split controls and RSS association are reference-only unless the
original split matrices and RSS table are supplied.

Important caveat
----------------
MUC5B epithelial cells are the normal epithelial comparison group in this
analysis, but ``muc5b`` is retained as the internal condition name. This file
is a method reference implementation rather than a standalone workflow.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from crane_grn.eregulon import extract_eregulon_tf, normalize_eregulon_name
from crane_grn.support import validate_aggregate


PRIMARY_TOP_MODULES = 10
MINIMUM_RANK_UNION = 4
CONDITION_COMPARISON = "tumor_vs_muc5b"
TUMOR_SPLIT_COMPARISON = "tumor_a_vs_tumor_b"
MUC5B_SPLIT_COMPARISON = "muc5b_a_vs_muc5b_b"


def load_support_mean(path):
    """Load a public aggregate as a labeled module-by-eRegulon matrix."""
    validate_aggregate(path)
    with np.load(path, allow_pickle=False) as archive:
        modules = pd.Index(archive["modules"].astype(str), name="module")
        eregulons = pd.Index(archive["eregs"].astype(str), name="eRegulon")
        if modules.has_duplicates or eregulons.has_duplicates:
            raise ValueError("Aggregate module and eRegulon labels must be unique.")
        return pd.DataFrame(
            archive["support_mean_KE"],
            index=modules,
            columns=eregulons,
        )


def support_to_rank(support):
    """Convert mean support to within-eRegulon module ranks.

    Zero support is treated as missing because the downstream statistic
    compares the relative ordering of modules with observed support. Positive
    support is ranked in descending order, with rank 1 denoting the strongest
    module and tied values receiving the minimum rank.
    """
    if not isinstance(support, pd.DataFrame):
        raise TypeError("support must be a pandas DataFrame.")
    frame = support.copy()
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Support module and eRegulon labels must be unique.")
    frame = frame.apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
    return frame.rank(axis=0, ascending=False, method="min", na_option="keep")


def compute_rewiring_scores(
    rank_tumor,
    rank_muc5b,
    eregulons=None,
    top_n=PRIMARY_TOP_MODULES,
    comparison=CONDITION_COMPARISON,
):
    """Calculate the executed module-rank rewiring score for each eRegulon.

    For one eRegulon, the calculation first retains modules with a finite rank
    in both conditions. At least ``top_n`` common modules are required. It then
    takes the union of the condition-specific top modules, re-ranks the two
    rank vectors within that union, and calculates Spearman correlation. The
    rewiring score is ``1 - rho``. A union with fewer than four modules is
    excluded, matching the original implementation.
    """
    if not isinstance(rank_tumor, pd.DataFrame) or not isinstance(
        rank_muc5b, pd.DataFrame
    ):
        raise TypeError("Both rank inputs must be pandas DataFrames.")
    tumor = rank_tumor.copy()
    muc5b = rank_muc5b.copy()
    for frame, name in ((tumor, "rank_tumor"), (muc5b, "rank_muc5b")):
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError("{} has duplicate axis labels.".format(name))

    if eregulons is None:
        selected_eregulons = tumor.columns.intersection(muc5b.columns).tolist()
    else:
        selected_eregulons = list(dict.fromkeys(map(str, eregulons)))

    rows = []
    for eregulon in selected_eregulons:
        if eregulon not in tumor.columns or eregulon not in muc5b.columns:
            continue
        tumor_ranks = pd.to_numeric(tumor[eregulon], errors="coerce").dropna()
        muc5b_ranks = pd.to_numeric(muc5b[eregulon], errors="coerce").dropna()
        common_modules = tumor_ranks.index.intersection(muc5b_ranks.index)
        if len(common_modules) < int(top_n):
            continue

        top_tumor = set(
            tumor_ranks.loc[common_modules].nsmallest(int(top_n)).index.astype(str)
        )
        top_muc5b = set(
            muc5b_ranks.loc[common_modules].nsmallest(int(top_n)).index.astype(str)
        )
        union_modules = sorted(top_tumor | top_muc5b)
        if len(union_modules) < MINIMUM_RANK_UNION:
            continue

        rho, _ = spearmanr(
            tumor_ranks.loc[union_modules].rank(),
            muc5b_ranks.loc[union_modules].rank(),
        )
        clean_name = normalize_eregulon_name(eregulon)
        rows.append(
            {
                "eRegulon": eregulon,
                "TF": extract_eregulon_tf(clean_name),
                "spearman_rho": float(rho),
                "rewiring_score": float(1.0 - rho),
                "top10_overlap": int(len(top_tumor & top_muc5b)),
                "comparison": str(comparison),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "eRegulon",
            "TF",
            "spearman_rho",
            "rewiring_score",
            "top10_overlap",
            "comparison",
        ],
    )


def calculate_primary_rewiring(tumor_aggregate_path, muc5b_aggregate_path):
    """Calculate public-data rewiring scores from the two UCEC aggregates."""
    tumor_support = load_support_mean(tumor_aggregate_path)
    muc5b_support = load_support_mean(muc5b_aggregate_path)
    common_eregulons = tumor_support.columns.intersection(muc5b_support.columns)
    return compute_rewiring_scores(
        support_to_rank(tumor_support),
        support_to_rank(muc5b_support),
        eregulons=common_eregulons,
        top_n=PRIMARY_TOP_MODULES,
        comparison=CONDITION_COMPARISON,
    )


def calculate_split_control_rewiring(
    support_tumor_a,
    support_tumor_b,
    support_muc5b_a,
    support_muc5b_b,
    eregulons,
):
    """Calculate reference-only within-condition split rewiring scores.

    The four inputs are the original split support matrices. New pseudo-splits
    must not be reconstructed from the two public full-condition aggregates.
    """
    tumor_control = compute_rewiring_scores(
        support_to_rank(support_tumor_a),
        support_to_rank(support_tumor_b),
        eregulons=eregulons,
        comparison=TUMOR_SPLIT_COMPARISON,
    )
    muc5b_control = compute_rewiring_scores(
        support_to_rank(support_muc5b_a),
        support_to_rank(support_muc5b_b),
        eregulons=eregulons,
        comparison=MUC5B_SPLIT_COMPARISON,
    )
    return tumor_control, muc5b_control


def compare_rewiring_with_split_controls(
    primary_rewiring,
    tumor_split_rewiring,
    muc5b_split_rewiring,
):
    """Compare condition rewiring with both within-condition noise baselines.

    Only eRegulons present in all three comparisons are retained. The original
    analysis reported one-sided Mann-Whitney tests on the distributions and
    one-sided paired Wilcoxon tests on matched eRegulons.
    """
    tables = {
        "primary": primary_rewiring,
        "tumor_split": tumor_split_rewiring,
        "muc5b_split": muc5b_split_rewiring,
    }
    indexed = {}
    for name, table in tables.items():
        required = {"eRegulon", "rewiring_score"}
        missing = required.difference(table.columns)
        if missing:
            raise KeyError("{} is missing {}.".format(name, sorted(missing)))
        if table["eRegulon"].astype(str).duplicated().any():
            raise ValueError("{} has duplicate eRegulons.".format(name))
        current = table.loc[:, ["eRegulon", "rewiring_score"]].copy()
        current["eRegulon"] = current["eRegulon"].astype(str)
        current["rewiring_score"] = pd.to_numeric(
            current["rewiring_score"], errors="coerce"
        )
        indexed[name] = current.dropna().set_index("eRegulon")["rewiring_score"]

    common = sorted(
        set(indexed["primary"].index)
        & set(indexed["tumor_split"].index)
        & set(indexed["muc5b_split"].index)
    )
    if not common:
        raise ValueError("No eRegulons are shared by all three comparisons.")
    paired = pd.DataFrame(
        {
            CONDITION_COMPARISON: indexed["primary"].loc[common],
            TUMOR_SPLIT_COMPARISON: indexed["tumor_split"].loc[common],
            MUC5B_SPLIT_COMPARISON: indexed["muc5b_split"].loc[common],
        },
        index=pd.Index(common, name="eRegulon"),
    )
    delta_tumor = (
        paired[CONDITION_COMPARISON] - paired[TUMOR_SPLIT_COMPARISON]
    ).to_numpy()
    delta_muc5b = (
        paired[CONDITION_COMPARISON] - paired[MUC5B_SPLIT_COMPARISON]
    ).to_numpy()

    _, mwu_tumor = mannwhitneyu(
        paired[CONDITION_COMPARISON],
        paired[TUMOR_SPLIT_COMPARISON],
        alternative="greater",
    )
    _, mwu_muc5b = mannwhitneyu(
        paired[CONDITION_COMPARISON],
        paired[MUC5B_SPLIT_COMPARISON],
        alternative="greater",
    )
    _, wilcoxon_tumor = wilcoxon(delta_tumor, alternative="greater")
    _, wilcoxon_muc5b = wilcoxon(delta_muc5b, alternative="greater")
    statistics = {
        "n_common_eregulons": len(common),
        "median_primary": float(paired[CONDITION_COMPARISON].median()),
        "median_tumor_split": float(paired[TUMOR_SPLIT_COMPARISON].median()),
        "median_muc5b_split": float(paired[MUC5B_SPLIT_COMPARISON].median()),
        "mwu_primary_greater_tumor_split_pvalue": float(mwu_tumor),
        "mwu_primary_greater_muc5b_split_pvalue": float(mwu_muc5b),
        "wilcoxon_primary_greater_tumor_split_pvalue": float(wilcoxon_tumor),
        "wilcoxon_primary_greater_muc5b_split_pvalue": float(wilcoxon_muc5b),
        "n_primary_greater_tumor_split": int(np.sum(delta_tumor > 0)),
        "n_primary_greater_muc5b_split": int(np.sum(delta_muc5b > 0)),
    }
    return paired, statistics


def attach_rss_annotations(
    rewiring,
    rss_table,
    tumor_column="rss_Tumor",
    muc5b_column="rss_Normal",
    delta_column="rss_delta",
):
    """Join the original RSS table to rewiring results by eRegulon label."""    
    required = {tumor_column, muc5b_column, delta_column}
    missing = required.difference(rss_table.columns)
    if missing:
        raise KeyError("RSS table is missing columns: {}.".format(sorted(missing)))
    if rss_table.index.astype(str).duplicated().any():
        raise ValueError("RSS eRegulon labels must be unique.")
    rss = rss_table.loc[:, [tumor_column, muc5b_column, delta_column]].copy()
    rss.index = rss.index.astype(str)
    rss.columns = ["rss_tumor", "rss_muc5b", "rss_delta"]
    result = rewiring.copy()
    result["eRegulon"] = result["eRegulon"].astype(str)
    return result.merge(rss, left_on="eRegulon", right_index=True, how="inner")


def rss_rewiring_association(rewiring_with_rss):
    """Calculate the three original Spearman RSS associations."""
    rows = []
    for rss_column in ("rss_tumor", "rss_muc5b", "rss_delta"):
        subset = rewiring_with_rss.loc[
            :, ["rewiring_score", rss_column]
        ].dropna()
        rho, pvalue = spearmanr(subset["rewiring_score"], subset[rss_column])
        rows.append(
            {
                "rss_measure": rss_column,
                "spearman_rho": float(rho),
                "pvalue": float(pvalue),
                "n": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def plot_split_control_comparison(paired_scores, output_path=None):
    """Plot primary rewiring beside the two within-condition controls."""
    required = [
        CONDITION_COMPARISON,
        TUMOR_SPLIT_COMPARISON,
        MUC5B_SPLIT_COMPARISON,
    ]
    missing = set(required).difference(paired_scores.columns)
    if missing:
        raise KeyError("Paired score table is missing: {}.".format(sorted(missing)))
    values = [paired_scores[column].dropna().to_numpy() for column in required]
    figure, axis = plt.subplots(figsize=(6.4, 4.5), facecolor="white")
    boxes = axis.boxplot(values, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], ("#7B3294", "#D55E00", "#0072B2")):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.set_xticklabels(
        ["Tumor vs MUC5B", "Tumor split", "MUC5B split"], rotation=15, ha="right"
    )
    axis.set_ylabel("Rewiring score (1 - Spearman rho)")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight")
    return figure, axis


def plot_rss_association(
    rewiring_with_rss,
    rss_column="rss_tumor",
    display_label="Tumor RSS",
    output_path=None,
):
    """Draw one RSS-versus-rewiring scatter plot with a linear guide."""
    subset = rewiring_with_rss.loc[
        :, ["rewiring_score", rss_column]
    ].dropna()
    if len(subset) < 2:
        raise ValueError("At least two complete observations are required.")
    x = subset["rewiring_score"].to_numpy(dtype=float)
    y = subset[rss_column].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    rho, pvalue = spearmanr(x, y)
    figure, axis = plt.subplots(figsize=(5.2, 4.4), facecolor="white")
    axis.scatter(x, y, s=24, alpha=0.72, color="#5E3C99", edgecolor="none")
    line_x = np.linspace(x.min(), x.max(), 100)
    axis.plot(line_x, slope * line_x + intercept, color="#2B2B2B", linewidth=1.2)
    axis.set_xlabel("Rewiring score (1 - Spearman rho)")
    axis.set_ylabel(display_label)
    axis.set_title("Spearman rho={:.3f}, p={:.3g}".format(rho, pvalue))
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight")
    return figure, axis

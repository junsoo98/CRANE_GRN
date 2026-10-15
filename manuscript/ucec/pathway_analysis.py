"""Pathway analyses downstream of UCEC regulatory rewiring.

Purpose
-------
Record the manuscript calculations that connect highly rewired eRegulons to
target genes, over-representation analysis (ORA), raw-count
pseudobulk differential expression, preranked GSEA, and ORA/GSEA overlap.

Reproducibility status
----------------------
Target extraction and KEGG ORA can be evaluated when the frozen gene-set
resource is supplied. Pseudobulk DE and GSEA are reference-only unless the
original raw counts and sample metadata are available.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


REWIRING_QUANTILE = 0.75
MINIMUM_CELLS_PER_PSEUDOBULK = 20
MINIMUM_GENE_TOTAL = 10
GSEA_PERMUTATIONS = 1000
GSEA_SEED = 42
GSEA_MINIMUM_SET_SIZE = 10
GSEA_MAXIMUM_SET_SIZE = 500
GSEA_THREADS = 4
SIGNIFICANCE_FDR = 0.05


def select_highly_rewired_targets(
    rewiring,
    eregulon_metadata,
    quantile=REWIRING_QUANTILE,
    eregulon_column="Region_signature_name",
    gene_column="Gene",
):
    """Select all target genes linked to the top rewiring-score quartile.

    The threshold is the inclusive 75th percentile by default. All metadata
    target genes linked to the selected eRegulons are retained.
    """
    required_rewiring = {"eRegulon", "rewiring_score"}
    missing_rewiring = required_rewiring.difference(rewiring.columns)
    if missing_rewiring:
        raise KeyError(
            "Rewiring table is missing columns: {}.".format(
                sorted(missing_rewiring)
            )
        )
    required_metadata = {eregulon_column, gene_column}
    missing_metadata = required_metadata.difference(eregulon_metadata.columns)
    if missing_metadata:
        raise KeyError(
            "eRegulon metadata is missing columns: {}.".format(
                sorted(missing_metadata)
            )
        )
    scores = rewiring.copy()
    scores["eRegulon"] = scores["eRegulon"].astype(str)
    scores["rewiring_score"] = pd.to_numeric(
        scores["rewiring_score"], errors="coerce"
    )
    scores = scores.dropna(subset=["rewiring_score"])
    if scores.empty:
        raise ValueError("No finite rewiring scores are available.")
    threshold = float(scores["rewiring_score"].quantile(float(quantile)))
    highly_rewired = scores[scores["rewiring_score"] >= threshold].copy()
    selected = set(highly_rewired["eRegulon"])
    metadata = eregulon_metadata.loc[:, [eregulon_column, gene_column]].copy()
    metadata[eregulon_column] = metadata[eregulon_column].astype(str)
    target_genes = sorted(
        set(
            metadata.loc[
                metadata[eregulon_column].isin(selected), gene_column
            ]
            .dropna()
            .astype(str)
        )
    )
    return {
        "threshold": threshold,
        "highly_rewired_eregulons": highly_rewired,
        "target_genes": target_genes,
    }


def validate_frozen_kegg_resource(kegg_gene_sets):
    """Validate a supplied ``KEGG_2021_Human`` pathway-to-gene mapping."""
    if not isinstance(kegg_gene_sets, dict) or not kegg_gene_sets:
        raise TypeError("kegg_gene_sets must be a non-empty dictionary.")
    validated = {}
    for pathway, genes in kegg_gene_sets.items():
        clean_genes = sorted(set(map(str, genes)))
        if clean_genes:
            validated[str(pathway)] = clean_genes
    if not validated:
        raise ValueError("The KEGG resource contains no non-empty gene sets.")
    return validated


def run_kegg_overrepresentation(target_genes, kegg_gene_sets):
    """Run the executed KEGG 2021 ORA."""
    import gseapy as gp

    genes = sorted(set(map(str, target_genes)))
    if not genes:
        raise ValueError("At least one target gene is required for ORA.")
    resource = validate_frozen_kegg_resource(kegg_gene_sets)
    enrichment = gp.enrichr(
        gene_list=genes,
        gene_sets=resource,
        organism="human",
        outdir=None,
        verbose=False,
    )
    result = enrichment.results.copy()
    result.columns = result.columns.str.strip()
    result["source"] = "KEGG_2021_Human"
    return result


def _coerce_count_matrix(raw_counts, cell_names=None, gene_names=None):
    """Return a CSR count matrix plus unique string cell and gene axes."""
    if isinstance(raw_counts, pd.DataFrame):
        cells = raw_counts.index.astype(str)
        genes = raw_counts.columns.astype(str)
        if all(isinstance(dtype, pd.SparseDtype) for dtype in raw_counts.dtypes):
            matrix = raw_counts.sparse.to_coo().tocsr()
        else:
            matrix = sparse.csr_matrix(raw_counts.to_numpy())
    elif sparse.issparse(raw_counts):
        if cell_names is None or gene_names is None:
            raise ValueError(
                "cell_names and gene_names are required for an unlabeled sparse matrix."
            )
        matrix = raw_counts.tocsr()
        cells = pd.Index(cell_names).astype(str)
        genes = pd.Index(gene_names).astype(str)
    else:
        array = np.asarray(raw_counts)
        if array.ndim != 2 or cell_names is None or gene_names is None:
            raise ValueError(
                "A dense count array requires two dimensions plus cell and gene names."
            )
        matrix = sparse.csr_matrix(array)
        cells = pd.Index(cell_names).astype(str)
        genes = pd.Index(gene_names).astype(str)
    if matrix.shape != (len(cells), len(genes)):
        raise ValueError("Count dimensions do not match the supplied labels.")
    if cells.has_duplicates or genes.has_duplicates:
        raise ValueError("Raw-count cell and gene labels must be unique.")
    if matrix.nnz and (not np.isfinite(matrix.data).all() or matrix.data.min() < 0):
        raise ValueError("Raw counts must be finite and non-negative.")
    return matrix, cells, genes


def build_pseudobulk_counts(
    raw_counts,
    cell_metadata,
    cell_names=None,
    gene_names=None,
    sample_column="sample_id",
    cell_type_column="final_cell_type",
    tumor_label="Tumor",
    muc5b_label="MUC5B",
    minimum_cells=MINIMUM_CELLS_PER_PSEUDOBULK,
    minimum_gene_total=MINIMUM_GENE_TOTAL,
):
    """Sum raw counts within sample-by-cell-type pseudobulk replicates.

    Biological samples, not cells, are the downstream replicates. Groups with
    ``n_cells < minimum_cells`` are excluded, preserving the executed inclusive
    20-cell boundary. Genes with a total pseudobulk count below 10 are removed
    by default.
    """
    required = {sample_column, cell_type_column}
    missing = required.difference(cell_metadata.columns)
    if missing:
        raise KeyError("Cell metadata is missing columns: {}.".format(sorted(missing)))
    matrix, cells, genes = _coerce_count_matrix(
        raw_counts, cell_names=cell_names, gene_names=gene_names
    )
    metadata = cell_metadata.loc[:, [sample_column, cell_type_column]].copy()
    metadata.index = metadata.index.astype(str)
    if metadata.index.has_duplicates:
        raise ValueError("Cell metadata index must be unique.")
    metadata = metadata[
        metadata[cell_type_column].isin([tumor_label, muc5b_label])
    ].dropna()
    common_cells = metadata.index.intersection(cells)
    if common_cells.empty:
        raise ValueError("Raw counts and cell metadata have no cells in common.")
    metadata = metadata.loc[common_cells]
    matrix_positions = cells.get_indexer(common_cells)
    aligned_counts = matrix[matrix_positions, :]

    count_rows = []
    metadata_rows = []
    for (sample, cell_type), group in metadata.groupby(
        [sample_column, cell_type_column], sort=True
    ):
        group_positions = metadata.index.get_indexer(group.index)
        number_cells = len(group_positions)
        if number_cells < int(minimum_cells):
            continue
        summed = np.asarray(
            aligned_counts[group_positions, :].sum(axis=0)
        ).ravel()
        pseudobulk_id = "{}__{}".format(sample, cell_type)
        count_rows.append(summed)
        metadata_rows.append(
            {
                "pb_id": pseudobulk_id,
                "sample": sample,
                "cell_type": cell_type,
                "condition": "tumor" if cell_type == tumor_label else "muc5b",
                "n_cells": number_cells,
            }
        )
    if not count_rows:
        raise ValueError("No pseudobulk group passed the cell-count threshold.")
    pseudobulk_metadata = pd.DataFrame(metadata_rows).set_index("pb_id")
    if pseudobulk_metadata.index.has_duplicates:
        raise ValueError("Pseudobulk identifiers are not unique.")
    pseudobulk_counts = pd.DataFrame(
        np.vstack(count_rows),
        index=pseudobulk_metadata.index,
        columns=genes,
    )
    if not np.allclose(pseudobulk_counts.to_numpy(), np.rint(pseudobulk_counts)):
        raise ValueError("Raw-count aggregation produced non-integer values.")
    pseudobulk_counts = pseudobulk_counts.astype(np.int64)
    keep_genes = pseudobulk_counts.columns[
        pseudobulk_counts.sum(axis=0) >= int(minimum_gene_total)
    ]
    pseudobulk_counts = pseudobulk_counts.loc[:, keep_genes].copy()
    return pseudobulk_counts, pseudobulk_metadata


def run_pseudobulk_differential_expression(
    pseudobulk_counts,
    pseudobulk_metadata,
):
    """Run the executed PyDESeq2 condition-only Tumor-versus-MUC5B model."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    if not pseudobulk_counts.index.equals(pseudobulk_metadata.index):
        raise ValueError("Pseudobulk count and metadata rows are not aligned.")
    metadata = pseudobulk_metadata.loc[:, ["condition", "n_cells"]].copy()
    metadata["condition"] = pd.Categorical(
        metadata["condition"], categories=["muc5b", "tumor"]
    )
    if metadata["condition"].isna().any():
        raise ValueError("Pseudobulk conditions must be 'tumor' or 'muc5b'.")
    dataset = DeseqDataSet(
        counts=pseudobulk_counts.astype(int),
        metadata=metadata,
        design_factors="condition",
        ref_level=["condition", "muc5b"],
    )
    dataset.deseq2()
    statistics = DeseqStats(
        dataset,
        contrast=["condition", "tumor", "muc5b"],
    )
    statistics.summary()
    result = statistics.results_df.copy().reset_index()
    result = result.rename(columns={result.columns[0]: "gene"})
    result = result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["gene", "log2FoldChange", "stat", "pvalue"]
    )
    return result


def build_gsea_ranking(differential_expression):
    """Build the two-column, signed-statistic ranking used by preranked GSEA."""
    required = {"gene", "stat"}
    missing = required.difference(differential_expression.columns)
    if missing:
        raise KeyError(
            "Differential-expression table is missing: {}.".format(sorted(missing))
        )
    ranking = differential_expression.loc[:, ["gene", "stat"]].copy()
    ranking["gene"] = ranking["gene"].astype(str)
    ranking["stat"] = pd.to_numeric(ranking["stat"], errors="coerce")
    ranking = (
        ranking.dropna()
        .drop_duplicates(subset="gene")
        .sort_values("stat", ascending=False)
    )
    ranking.columns = ["gene", "score"]
    return ranking


def run_kegg_preranked_gsea(
    ranking,
    kegg_gene_sets,
    permutation_num=GSEA_PERMUTATIONS,
    seed=GSEA_SEED,
    min_size=GSEA_MINIMUM_SET_SIZE,
    max_size=GSEA_MAXIMUM_SET_SIZE,
    threads=GSEA_THREADS,
):
    """Run the executed KEGG 2021 preranked GSEA configuration."""
    import gseapy as gp

    required = {"gene", "score"}
    missing = required.difference(ranking.columns)
    if missing:
        raise KeyError("GSEA ranking is missing: {}.".format(sorted(missing)))
    resource = validate_frozen_kegg_resource(kegg_gene_sets)
    result_object = gp.prerank(
        rnk=ranking.loc[:, ["gene", "score"]],
        gene_sets=resource,
        permutation_num=int(permutation_num),
        outdir=None,
        seed=int(seed),
        min_size=int(min_size),
        max_size=int(max_size),
        threads=int(threads),
        verbose=False,
    )
    result = result_object.res2d.copy()
    result.columns = result.columns.str.strip()
    return result


def compare_ora_and_gsea(
    ora_results,
    gsea_results,
    fdr_threshold=SIGNIFICANCE_FDR,
):
    """Classify significant KEGG pathways as shared or method-specific."""
    required_ora = {"Term", "Adjusted P-value"}
    required_gsea = {"Term", "FDR q-val"}
    if required_ora.difference(ora_results.columns):
        raise KeyError("ORA results do not contain the expected columns.")
    if required_gsea.difference(gsea_results.columns):
        raise KeyError("GSEA results do not contain the expected columns.")
    ora_set = set(
        ora_results.loc[
            pd.to_numeric(ora_results["Adjusted P-value"], errors="coerce")
            < float(fdr_threshold),
            "Term",
        ].astype(str)
    )
    gsea_set = set(
        gsea_results.loc[
            pd.to_numeric(gsea_results["FDR q-val"], errors="coerce")
            < float(fdr_threshold),
            "Term",
        ].astype(str)
    )
    return {
        "both": ora_set & gsea_set,
        "ora_only": ora_set - gsea_set,
        "gsea_only": gsea_set - ora_set,
        "ora_significant": ora_set,
        "gsea_significant": gsea_set,
    }


def plot_kegg_overrepresentation(
    ora_results,
    maximum_pathways=15,
    fdr_threshold=SIGNIFICANCE_FDR,
    output_path=None,
):
    """Plot significant KEGG ORA odds ratios ordered by adjusted p-value."""
    required = {"Term", "Adjusted P-value", "Odds Ratio"}
    missing = required.difference(ora_results.columns)
    if missing:
        raise KeyError("ORA results are missing: {}.".format(sorted(missing)))
    table = ora_results.copy()
    table["Adjusted P-value"] = pd.to_numeric(
        table["Adjusted P-value"], errors="coerce"
    )
    table["Odds Ratio"] = pd.to_numeric(table["Odds Ratio"], errors="coerce")
    table = (
        table[table["Adjusted P-value"] < float(fdr_threshold)]
        .dropna(subset=["Term", "Odds Ratio"])
        .sort_values(["Adjusted P-value", "Odds Ratio"], ascending=[True, False])
        .head(int(maximum_pathways))
        .sort_values("Odds Ratio")
    )
    figure, axis = plt.subplots(
        figsize=(7.2, max(3.2, 0.32 * len(table) + 1.4)), facecolor="white"
    )
    axis.barh(table["Term"], table["Odds Ratio"], color="#7B3294", alpha=0.82)
    axis.set_xlabel("Odds ratio")
    axis.set_ylabel("")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight")
    return figure, axis


def plot_kegg_gsea_results(
    gsea_results,
    maximum_per_direction=15,
    fdr_threshold=SIGNIFICANCE_FDR,
    output_path=None,
):
    """Plot significant positive and negative KEGG normalized scores.

    Positive NES denotes Tumor enrichment and negative NES denotes MUC5B
    epithelial enrichment. 
    """
    required = {"Term", "NES", "FDR q-val"}
    missing = required.difference(gsea_results.columns)
    if missing:
        raise KeyError("GSEA results are missing: {}.".format(sorted(missing)))
    table = gsea_results.loc[:, ["Term", "NES", "FDR q-val"]].copy()
    table["NES"] = pd.to_numeric(table["NES"], errors="coerce")
    table["FDR q-val"] = pd.to_numeric(table["FDR q-val"], errors="coerce")
    table = table.dropna()
    table = table[table["FDR q-val"] < float(fdr_threshold)]
    tumor = table[table["NES"] > 0].nlargest(int(maximum_per_direction), "NES")
    muc5b = table[table["NES"] < 0].nsmallest(int(maximum_per_direction), "NES")
    selected = pd.concat([tumor, muc5b], ignore_index=True).sort_values("NES")
    figure, axis = plt.subplots(
        figsize=(7.4, max(3.2, 0.30 * len(selected) + 1.4)), facecolor="white"
    )
    colors = np.where(selected["NES"] > 0, "#D55E00", "#0072B2")
    axis.barh(selected["Term"], selected["NES"], color=colors, alpha=0.84)
    axis.axvline(0, color="#4D4D4D", linewidth=0.8)
    axis.set_xlabel("Normalized enrichment score")
    axis.set_ylabel("")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight")
    return figure, axis


def plot_ora_gsea_overlap(overlap, output_path=None):
    """Plot the retained fixed-category ORA/GSEA pathway count summary."""
    categories = ["Both", "ORA only", "GSEA only"]
    counts = [
        len(overlap["both"]),
        len(overlap["ora_only"]),
        len(overlap["gsea_only"]),
    ]
    figure, axis = plt.subplots(figsize=(5.4, 4.2), facecolor="white")
    bars = axis.bar(
        categories,
        counts,
        color=["#E74C3C", "#F39C12", "#3498DB"],
        alpha=0.85,
        edgecolor="white",
    )
    for bar, count in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(count),
            ha="center",
        )
    axis.set_ylabel("Pathways (FDR < 0.05)")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight")
    return figure, axis

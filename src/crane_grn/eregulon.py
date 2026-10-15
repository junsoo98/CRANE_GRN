"""Connect module support to eRegulons and regulatory triplets."""

import re

import numpy as np
import pandas as pd

from .regions import canonicalize_regions


def normalize_eregulon_name(raw_name):
    """Remove the terminal SCENIC+ region/gene-count suffix from an eRegulon."""
    return re.sub(r"_\(\d+[rg]\)$", "", str(raw_name))


def extract_eregulon_direction(clean_name):
    """Extract a terminal ``+/+``, ``-/+``, ``+/-``, or ``-/-`` direction."""
    match = re.search(r"([+\-]/[+\-])$", str(clean_name))
    return match.group(1) if match else "unknown"


def extract_eregulon_tf(clean_name):
    """Extract the leading transcription-factor token from an eRegulon name."""
    return str(clean_name).split("_")[0]


def build_support_rank_matrices(
    support_by_eregulon,
    eregulon_metadata,
    region_universe,
    region_column="Region",
    tf_column="TF",
    gene_column="Gene",
    eregulon_column="Region_signature_name",
    triplet_rank_column="triplet_rank",
    regulation_column="regulation",
    rho_tf_to_gene_column="rho_TF2G",
    rho_region_to_gene_column="rho_R2G",
):
    """Build module/eRegulon and module/triplet rank matrices.

    ``support_by_eregulon`` is modules by eRegulons. Zero support is treated as
    missing. 
    """
    support = support_by_eregulon.copy()
    support.index = support.index.astype(str)
    support.columns = support.columns.astype(str)
    if support.index.has_duplicates or support.columns.has_duplicates:
        raise ValueError("Support module and eRegulon labels must be unique.")
    support = support.mask(support == 0)
    module_ranking_per_eregulon = support.rank(
        axis=0, ascending=False, method="min"
    )
    eregulon_ranking_per_module = support.rank(
        axis=1, ascending=False, method="min"
    )

    required = [
        region_column,
        tf_column,
        gene_column,
        eregulon_column,
        triplet_rank_column,
        regulation_column,
        rho_tf_to_gene_column,
        rho_region_to_gene_column,
    ]
    missing = [column for column in required if column not in eregulon_metadata]
    if missing:
        raise ValueError("eRegulon metadata is missing columns: {}".format(missing))

    region_set = set(canonicalize_regions(region_universe).astype(str))
    eregulon_set = set(support.columns)
    triplets = eregulon_metadata.copy()
    triplets[region_column] = canonicalize_regions(
        triplets[region_column].astype(str)
    ).astype(str)
    for column in (tf_column, gene_column, eregulon_column):
        triplets[column] = triplets[column].astype(str)
    triplets = triplets[
        triplets[region_column].isin(region_set)
        & triplets[eregulon_column].isin(eregulon_set)
    ].copy()
    condition = triplets[regulation_column] == 1
    condition &= triplets[rho_region_to_gene_column] > 0
    triplets = triplets[condition].copy()
    triplets = triplets.drop_duplicates(
        subset=[region_column, tf_column, gene_column, eregulon_column]
    )
    triplets["triplet_id"] = (
        triplets[region_column]
        + "|"
        + triplets[tf_column]
        + "|"
        + triplets[gene_column]
        + "|"
        + triplets[eregulon_column]
    )
    triplets[triplet_rank_column] = pd.to_numeric(
        triplets[triplet_rank_column], errors="coerce"
    ).astype("Int64")
    triplets = triplets.set_index("triplet_id", drop=False)

    triplet_ids = triplets["triplet_id"].astype(str).to_numpy()
    triplet_eregulons = triplets[eregulon_column].astype(str).to_numpy()
    module_ranking_per_triplet = module_ranking_per_eregulon.loc[
        :, triplet_eregulons
    ].copy()
    module_ranking_per_triplet.columns = triplet_ids

    base_rank = eregulon_ranking_per_module.loc[:, triplet_eregulons].copy()
    base_rank.columns = triplet_ids
    original_triplet_rank = triplets[triplet_rank_column].to_numpy(dtype="float64")
    finite_triplet_rank = original_triplet_rank[np.isfinite(original_triplet_rank)]
    missing_rank_value = (
        float(finite_triplet_rank.max() + 1e6)
        if finite_triplet_rank.size
        else 1e12
    )
    filled_triplet_rank = np.where(
        np.isnan(original_triplet_rank), missing_rank_value, original_triplet_rank
    )
    raw_rank = base_rank.to_numpy(dtype=np.float64, copy=True)
    ranked = np.full(raw_rank.shape, -1, dtype=np.int32)
    for module_position in range(raw_rank.shape[0]):
        current = raw_rank[module_position, :]
        finite = np.isfinite(current)
        if not np.any(finite):
            continue
        ordering = np.lexsort(
            (filled_triplet_rank[finite], current[finite])
        )
        positions = np.flatnonzero(finite)[ordering]
        ranked[module_position, positions] = np.arange(
            1, positions.size + 1, dtype=np.int32
        )
    triplet_ranking_per_module = (
        pd.DataFrame(
            ranked,
            index=eregulon_ranking_per_module.index.astype(str),
            columns=triplet_ids,
        )
        .replace(-1, pd.NA)
        .astype("Int32")
    )
    return {
        "module_ranking_per_eregulon": module_ranking_per_eregulon,
        "eregulon_ranking_per_module": eregulon_ranking_per_module,
        "module_ranking_per_triplet": module_ranking_per_triplet,
        "triplet_ranking_per_module": triplet_ranking_per_module,
        "triplets": triplets,
    }


def _labeled_matrix(values, rows, columns, row_name, column_name):
    rows = pd.Index(rows).astype(str)
    columns = pd.Index(columns).astype(str)
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        missing_rows = rows.difference(frame.index)
        missing_columns = columns.difference(frame.columns)
        if len(missing_rows) or len(missing_columns):
            raise ValueError(
                "Labeled matrix is missing {} or {} labels.".format(
                    row_name, column_name
                )
            )
        return frame.loc[rows, columns]
    array = np.asarray(values)
    if array.shape != (len(rows), len(columns)):
        raise ValueError(
            "Matrix shape {} does not match {} by {} axes.".format(
                array.shape, row_name, column_name
            )
        )
    return pd.DataFrame(array, index=rows, columns=columns)


def build_module_eregulon_edges(
    support_mean,
    corr_auc_sup,
    modules,
    eregulons,
    responsibility_triplet,
    eregulon_metadata,
    top_k=5,
    support_minimum=0.0,
    region_column="Region",
    eregulon_column="Region_signature_name",
):
    """Select top-support module-to-eRegulon edges and count linked regions."""
    modules = pd.Index(modules).astype(str)
    eregulons = pd.Index(eregulons).astype(str)
    support = _labeled_matrix(
        support_mean, modules, eregulons, "module", "eRegulon"
    )
    correlation = _labeled_matrix(
        corr_auc_sup, modules, eregulons, "module", "eRegulon"
    )
    responsibility = responsibility_triplet.copy()
    responsibility.index = canonicalize_regions(responsibility.index)
    responsibility.columns = responsibility.columns.astype(str)
    missing_modules = modules.difference(responsibility.columns)
    if len(missing_modules):
        raise ValueError(
            "Triplet responsibility is missing modules: {}".format(
                missing_modules.tolist()
            )
        )
    responsibility = responsibility.loc[:, modules]

    required = [region_column, eregulon_column]
    missing = [column for column in required if column not in eregulon_metadata]
    if missing:
        raise ValueError("eRegulon metadata is missing columns: {}".format(missing))
    metadata = eregulon_metadata.loc[:, required].dropna().copy()
    metadata[region_column] = canonicalize_regions(
        metadata[region_column].astype(str)
    ).astype(str)
    metadata[eregulon_column] = metadata[eregulon_column].astype(str)
    region_set = set(responsibility.index.astype(str))
    metadata = metadata[
        metadata[region_column].isin(region_set)
        & metadata[eregulon_column].isin(set(eregulons))
    ]
    eregulon_regions = {
        name: set(block[region_column].astype(str))
        for name, block in metadata.groupby(eregulon_column)
    }
    module_regions = {
        module: set(
            responsibility.index[responsibility[module] > 0].astype(str)
        )
        for module in modules
    }

    rows = []
    support_array = support.to_numpy(dtype=float)
    correlation_array = correlation.to_numpy(dtype=float)
    for eregulon_position, eregulon in enumerate(eregulons):
        column = support_array[:, eregulon_position]
        valid = np.where(
            np.isfinite(column) & (column > float(support_minimum))
        )[0]
        if valid.size == 0:
            continue
        selected = valid[np.argsort(column[valid])[::-1][: int(top_k)]]
        rank = np.argsort(np.argsort(-np.nan_to_num(column))) + 1
        clean_name = normalize_eregulon_name(eregulon)
        for module_position in selected:
            module = modules[module_position]
            rows.append(
                {
                    "module": module,
                    "ereg_raw": eregulon,
                    "ereg_clean": clean_name,
                    "tf_name": extract_eregulon_tf(clean_name),
                    "direction": extract_eregulon_direction(clean_name),
                    "support_mean": float(column[module_position]),
                    "corr_auc_sup": float(
                        correlation_array[module_position, eregulon_position]
                    )
                    if np.isfinite(
                        correlation_array[module_position, eregulon_position]
                    )
                    else np.nan,
                    "rank_in_ereg": int(rank[module_position]),
                    "n_resp": len(
                        module_regions[module]
                        & eregulon_regions.get(eregulon, set())
                    ),
                }
            )
    columns = [
        "module",
        "ereg_raw",
        "ereg_clean",
        "tf_name",
        "direction",
        "support_mean",
        "corr_auc_sup",
        "rank_in_ereg",
        "n_resp",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_modules(edge_table, group=None):
    """Aggregate a module-to-eRegulon edge table into generic module summaries."""
    if edge_table.empty:
        columns = [
            "module",
            "n_ereg",
            "n_ereg_positive_direction",
            "support_mean",
            "support_max",
            "corr_mean",
            "corr_positive_fraction",
            "n_resp_total",
        ]
        if group is not None:
            columns.append("group")
        return pd.DataFrame(columns=columns)
    required = {
        "module",
        "ereg_clean",
        "direction",
        "support_mean",
        "corr_auc_sup",
        "n_resp",
    }
    missing = sorted(required.difference(edge_table.columns))
    if missing:
        raise ValueError("Edge table is missing columns: {}".format(missing))
    summary = (
        edge_table.groupby("module")
        .agg(
            n_ereg=("ereg_clean", "count"),
            n_ereg_positive_direction=(
                "direction",
                lambda values: values.isin(["+/+", "-/+"]).sum(),
            ),
            support_mean=("support_mean", "mean"),
            support_max=("support_mean", "max"),
            corr_mean=("corr_auc_sup", "mean"),
            corr_positive_fraction=(
                "corr_auc_sup",
                lambda values: (values > 0.3).mean(),
            ),
            n_resp_total=("n_resp", "sum"),
        )
        .reset_index()
    )
    if group is not None:
        summary["group"] = str(group)
    return summary

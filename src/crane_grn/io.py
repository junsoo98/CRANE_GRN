"""Lightweight I/O and matrix-preparation helpers for CRANE-GRN."""

import numpy as np
import pandas as pd
from scipy import sparse

from .regions import canonicalize_regions


def validate_unique_labels(labels, axis_name):
    """Return string labels after checking that an axis is unique."""
    index = pd.Index(labels).astype(str)
    if index.has_duplicates:
        examples = index[index.duplicated()].unique()[:3].tolist()
        raise ValueError(
            "{} labels are not unique; examples: {}".format(
                axis_name,
                examples,
            )
        )
    return index


def validate_dataframe_axes(frame, row_name, column_name):
    """Validate unique row and column labels on a DataFrame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    validate_unique_labels(frame.index, row_name)
    validate_unique_labels(frame.columns, column_name)
    return frame


def load_table(path, index_col=0):
    """Load a Parquet, CSV, or TSV table using its filename suffix."""
    path_text = str(path)
    lower = path_text.lower()

    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path_text)

    if lower.endswith(".tsv") or lower.endswith(".txt"):
        return pd.read_csv(
            path_text,
            sep="\t",
            index_col=index_col,
        )

    if lower.endswith(".csv"):
        return pd.read_csv(
            path_text,
            index_col=index_col,
        )

    raise ValueError(
        "Unsupported table format: {}".format(path_text)
    )


def normalize_log1p_expression(
    raw_counts,
    target_sum=10000.0,
    dtype=np.float32,
):
    """Library-size normalize a full cell-by-gene count matrix, then log1p it.

    The normalization denominator is calculated over the complete gene
    library. Sparse input remains sparse. Gene subsetting must therefore occur
    only after this function has completed.
    """
    if raw_counts.ndim != 2:
        raise ValueError(
            "Expression input must be a two-dimensional matrix."
        )

    if sparse.issparse(raw_counts):
        normalized = raw_counts.tocsr(
            copy=True
        ).astype(dtype)

        library_size = np.asarray(
            normalized.sum(axis=1)
        ).ravel()

        scale = np.zeros_like(
            library_size,
            dtype=dtype,
        )
        nonzero = library_size > 0
        scale[nonzero] = (
            float(target_sum)
            / library_size[nonzero]
        )

        normalized = sparse.diags(
            scale
        ).dot(
            normalized
        ).tocsr()

        normalized.data = np.log1p(
            normalized.data
        ).astype(
            dtype,
            copy=False,
        )
        normalized.eliminate_zeros()
        return normalized

    values = np.asarray(
        raw_counts,
        dtype=dtype,
    )
    library_size = values.sum(
        axis=1,
        dtype=np.float64,
    )
    scale = np.zeros_like(
        library_size,
        dtype=np.float64,
    )
    nonzero = library_size > 0
    scale[nonzero] = (
        float(target_sum)
        / library_size[nonzero]
    )

    return np.log1p(
        values * scale[:, None]
    ).astype(
        dtype,
        copy=False,
    )


def extract_factor_expression(
    normalized_expression,
    cell_names,
    gene_names,
    factors,
    dtype=np.float32,
):
    """Extract a dense cell-by-factor DataFrame from normalized expression.

    ``normalized_expression`` must already have been normalized using the full
    transcriptome. Only the requested factor genes are materialized densely.

    This factor-only matrix is suitable for inspection and downstream tasks
    that require factor values alone. It is not a valid calibration input for
    ``compute_module_prior()``, which requires the full background
    transcriptome to fit group-specific expression tiers.
    """
    cells = validate_unique_labels(
        cell_names,
        "cell",
    )
    genes = validate_unique_labels(
        gene_names,
        "gene",
    )
    requested = validate_unique_labels(
        factors,
        "factor",
    )

    if normalized_expression.shape != (
        len(cells),
        len(genes),
    ):
        raise ValueError(
            "Expression shape {} does not match {} cells by {} genes.".format(
                normalized_expression.shape,
                len(cells),
                len(genes),
            )
        )

    missing = requested.difference(genes)
    if len(missing):
        raise ValueError(
            "Expression matrix is missing {} required factors: {}".format(
                len(missing),
                missing.tolist(),
            )
        )

    positions = genes.get_indexer(requested)
    selected = normalized_expression[:, positions]

    if sparse.issparse(selected):
        selected = selected.toarray()

    return pd.DataFrame(
        np.asarray(
            selected,
            dtype=dtype,
        ),
        index=cells,
        columns=requested,
    )


def prepare_factor_expression(
    raw_counts,
    cell_names,
    gene_names,
    factors,
    target_sum=10000.0,
    dtype=np.float32,
):
    """Normalize a full RNA library, then extract factor expression.

    The returned factor-only DataFrame is not sufficient for
    ``compute_module_prior()`` tier calibration. Pass the full matrix returned
    by ``normalize_log1p_expression()`` directly to ``compute_module_prior()``
    and provide its cell and gene labels.
    """
    normalized = normalize_log1p_expression(
        raw_counts,
        target_sum=target_sum,
        dtype=dtype,
    )

    return extract_factor_expression(
        normalized,
        cell_names,
        gene_names,
        factors,
        dtype=dtype,
    )


def extract_cistopic_matrices(
    cistopic_object,
    validate_sums=True,
    atol=5e-3,
):
    """Extract region-by-topic ``L`` and cell-by-topic ``T`` matrices.

    The selected cisTopic model is used. Region labels are canonicalized, topic
    columns are named ``Topic1`` through ``TopicJ``, and the original region and
    cell orders are preserved.
    """
    if not hasattr(cistopic_object, "selected_model"):
        raise TypeError(
            "The object does not expose a selected cisTopic model."
        )

    model = cistopic_object.selected_model
    required = (
        "region_topic",
        "cell_topic",
        "n_topic",
    )
    missing = [
        name
        for name in required
        if not hasattr(model, name)
    ]

    if missing:
        raise ValueError(
            "Selected cisTopic model is missing: {}".format(
                missing
            )
        )

    regions = validate_unique_labels(
        cistopic_object.region_names,
        "region",
    )
    cells = validate_unique_labels(
        cistopic_object.cell_names,
        "cell",
    )
    topics = [
        "Topic{}".format(j + 1)
        for j in range(int(model.n_topic))
    ]

    region_by_topic = pd.DataFrame(
        model.region_topic,
        index=canonicalize_regions(regions),
        columns=topics,
    )
    cell_by_topic = pd.DataFrame(
        model.cell_topic,
        index=topics,
        columns=cells,
    ).T

    validate_dataframe_axes(
        region_by_topic,
        "region",
        "topic",
    )
    validate_dataframe_axes(
        cell_by_topic,
        "cell",
        "topic",
    )

    if validate_sums:
        if not np.allclose(
            cell_by_topic.sum(axis=1).to_numpy(),
            1.0,
            atol=atol,
        ):
            raise ValueError(
                "Cell-by-topic rows do not sum to one within tolerance."
            )

        if not np.allclose(
            region_by_topic.sum(axis=0).to_numpy(),
            1.0,
            atol=atol,
        ):
            raise ValueError(
                "Region-by-topic columns do not sum to one within tolerance."
            )

    return region_by_topic, cell_by_topic


def get_modality(container, key):
    """Return a named modality from a MuData-like object with a clear error."""
    if not hasattr(container, "mod"):
        raise TypeError(
            "Input object does not expose a modality mapping."
        )

    if key not in container.mod:
        raise KeyError(
            "Required modality {!r} was not found; "
            "available modalities: {}".format(
                key,
                list(container.mod.keys()),
            )
        )

    return container.mod[key]
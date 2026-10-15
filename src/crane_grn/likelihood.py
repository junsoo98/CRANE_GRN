"""ATAC-informed log-likelihood calculation for CRANE-GRN modules."""

import numpy as np
import pandas as pd
from scipy import sparse

from .regions import canonicalize_regions


def _matrix_with_labels(matrix_or_object, cell_names, region_names):
    if hasattr(matrix_or_object, "X"):
        matrix = matrix_or_object.X
        cells = pd.Index(matrix_or_object.obs_names).astype(str)
        regions = canonicalize_regions(matrix_or_object.var_names)
    else:
        matrix = matrix_or_object
        if cell_names is None or region_names is None:
            raise ValueError(
                "Cell and region labels are required for a raw accessibility matrix."
            )
        cells = pd.Index(cell_names).astype(str)
        regions = canonicalize_regions(region_names)
    if matrix.shape != (len(cells), len(regions)):
        raise ValueError("Accessibility shape does not match its cell and region labels.")
    if cells.has_duplicates or regions.has_duplicates:
        raise ValueError("Accessibility cell and region labels must be unique.")
    return matrix, cells, regions


def _align_groups(groups, cells):
    cells = pd.Index(cells).astype(str)
    if isinstance(groups, pd.Series):
        aligned = groups.copy()
        aligned.index = aligned.index.astype(str)
        missing = cells.difference(aligned.index)
        if len(missing):
            raise ValueError(
                "Group labels are missing {} cells; examples: {}".format(
                    len(missing), missing[:3].tolist()
                )
            )
        return aligned.loc[cells].astype(str)
    values = np.asarray(groups)
    if values.shape[0] != len(cells):
        raise ValueError("Group-label length does not match the requested cells.")
    return pd.Series(values.astype(str), index=cells)


def compute_log_likelihood(
    responsibility,
    region_by_topic,
    cell_by_topic,
    observed_accessibility,
    groups,
    accessibility_cell_names=None,
    accessibility_region_names=None,
    responsibility_exponent=1.5,
    batch_size=256,
    epsilon=1e-6,
    dtype="float32",
    normalization_cell_fraction=0.5,
    normalization_region_fraction=0.5,
    minimum_coverage=10,
    minimum_cells_per_group=50,
):
    """Compute a cell-by-module responsibility-weighted log-likelihood.

    Parameters
    ----------
    responsibility
        Region-by-module responsibility matrix.
    region_by_topic
        Region-by-topic cisTopic loading matrix.
    cell_by_topic
        Cell-by-topic cisTopic activation matrix.
    observed_accessibility
        Cell-by-region matrix or an object exposing ``X``, ``obs_names``, and
        ``var_names``. Values are binarized during batched access.

    Notes
    -----
    Modules with fewer than ``minimum_coverage`` nonzero regions remain NaN.
    The effective region count is used internally for normalization.
    """
    responsibility = responsibility.copy()
    responsibility.index = canonicalize_regions(responsibility.index)
    responsibility.columns = responsibility.columns.astype(str)
    region_by_topic = region_by_topic.copy()
    region_by_topic.index = canonicalize_regions(region_by_topic.index)
    region_by_topic.columns = region_by_topic.columns.astype(str)
    cell_by_topic = cell_by_topic.copy()
    cell_by_topic.index = cell_by_topic.index.astype(str)
    cell_by_topic.columns = cell_by_topic.columns.astype(str)
    for frame, row_axis, column_axis in (
        (responsibility, "region", "module"),
        (region_by_topic, "region", "topic"),
        (cell_by_topic, "cell", "topic"),
    ):
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError(
                "{} or {} labels are not unique.".format(row_axis, column_axis)
            )

    missing_regions = responsibility.index.difference(region_by_topic.index)
    if len(missing_regions):
        raise ValueError(
            "Region-by-topic matrix is missing {} responsibility regions; "
            "examples: {}".format(len(missing_regions), missing_regions[:3].tolist())
        )
    region_by_topic = region_by_topic.loc[responsibility.index]
    if set(region_by_topic.columns) != set(cell_by_topic.columns):
        raise ValueError("Region-by-topic and cell-by-topic topic labels differ.")
    cell_by_topic = cell_by_topic.loc[:, region_by_topic.columns]

    accessibility, accessibility_cells, accessibility_regions = _matrix_with_labels(
        observed_accessibility,
        accessibility_cell_names,
        accessibility_region_names,
    )
    missing_accessibility_regions = responsibility.index.difference(
        accessibility_regions
    )
    if len(missing_accessibility_regions):
        raise ValueError(
            "Observed accessibility is missing {} requested regions; examples: {}".format(
                len(missing_accessibility_regions),
                missing_accessibility_regions[:3].tolist(),
            )
        )
    region_positions = accessibility_regions.get_indexer(responsibility.index)
    accessibility = accessibility[:, region_positions]

    common_cells = pd.Index(
        [cell for cell in cell_by_topic.index if cell in set(accessibility_cells)]
    )
    if len(common_cells) == 0:
        raise ValueError("No common cells were found between topic and ATAC inputs.")
    accessibility_positions = accessibility_cells.get_indexer(common_cells)
    accessibility = accessibility[accessibility_positions, :]
    cell_by_topic = cell_by_topic.loc[common_cells]
    group_series = _align_groups(groups, common_cells)

    loading = region_by_topic.to_numpy(dtype=dtype, copy=False)
    topic = cell_by_topic.to_numpy(dtype=dtype, copy=False)
    responsibility_values = responsibility.to_numpy(dtype=dtype, copy=False)
    number_regions = loading.shape[0]
    number_cells = topic.shape[0]
    number_modules = responsibility_values.shape[1]

    coverage = (responsibility_values > 0).sum(axis=0)
    full_weights = np.power(responsibility_values, float(responsibility_exponent))
    sum_weights = full_weights.sum(axis=0)
    valid_modules = (coverage >= int(minimum_coverage)) & (sum_weights > 0)
    valid_weights = full_weights[:, valid_modules]
    sum_valid = valid_weights.sum(axis=0)
    sum_squared = (valid_weights ** 2).sum(axis=0) + 1e-12
    effective_count = np.maximum((sum_valid ** 2) / sum_squared, 1e-12)

    log_likelihood = np.full(
        (number_cells, number_modules), np.nan, dtype=dtype
    )
    rng = np.random.default_rng(0)

    def _dense_binary_rows(row_positions):
        subset = accessibility[row_positions, :]
        if sparse.issparse(subset):
            subset = subset.toarray()
        return (np.asarray(subset) > 0).astype(dtype, copy=False)

    for group in sorted(group_series.unique().tolist()):
        group_positions = np.where(group_series.to_numpy() == group)[0]
        number_group_cells = len(group_positions)
        if number_group_cells < int(minimum_cells_per_group):
            continue
        group_topic = topic[group_positions, :]
        sample_cells = max(
            1, int(number_group_cells * float(normalization_cell_fraction))
        )
        sample_regions = max(
            1, int(number_regions * float(normalization_region_fraction))
        )
        cell_sample = rng.choice(
            number_group_cells, size=sample_cells, replace=False
        )
        region_sample = rng.choice(number_regions, size=sample_regions, replace=False)
        sampled_accessibility = (
            loading[region_sample, :] @ group_topic[cell_sample, :].T
        )
        lower = np.percentile(sampled_accessibility, 1)
        upper = np.percentile(sampled_accessibility, 99)
        normalization_range = (upper - lower) + 1e-12

        group_blocks = []
        for start in range(0, number_group_cells, int(batch_size)):
            end = min(start + int(batch_size), number_group_cells)
            positions = group_positions[start:end]
            reconstructed = loading @ topic[positions, :].T
            reconstructed = np.clip(
                (reconstructed - lower) / normalization_range,
                float(epsilon),
                1.0 - float(epsilon),
            )
            observed = _dense_binary_rows(positions).T
            bernoulli = observed * np.log(reconstructed) + (
                1.0 - observed
            ) * np.log1p(-reconstructed)
            block_valid = bernoulli.T @ valid_weights
            block_valid = block_valid / effective_count[None, :]
            block = np.full((len(positions), number_modules), np.nan, dtype=dtype)
            block[:, valid_modules] = block_valid
            group_blocks.append(block)
        log_likelihood[group_positions, :] = np.vstack(group_blocks)

    return pd.DataFrame(
        log_likelihood,
        index=common_cells,
        columns=responsibility.columns.astype(str),
    )

"""Posterior-gated module support and AUCell correlation aggregation."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .regions import canonicalize_regions


AGGREGATE_FIELDS = (
    "n_cells",
    "modules",
    "eregs",
    "reg_count",
    "support_mean_KE",
    "corr_auc_sup_KE",
)


def build_eregulon_region_map(
    eregulon_metadata,
    regions,
    region_column="Region",
    eregulon_column="Region_signature_name",
):
    """Map each eRegulon to positions in a canonical region universe."""
    required = [region_column, eregulon_column]
    missing = [column for column in required if column not in eregulon_metadata]
    if missing:
        raise ValueError("eRegulon metadata is missing columns: {}".format(missing))
    region_index = canonicalize_regions(regions)
    if region_index.has_duplicates:
        raise ValueError("Support region labels are not unique.")
    metadata = eregulon_metadata.loc[:, required].dropna().copy()
    metadata[region_column] = canonicalize_regions(
        metadata[region_column].astype(str)
    ).astype(str)
    metadata[eregulon_column] = metadata[eregulon_column].astype(str)
    region_set = set(region_index.astype(str))
    metadata = metadata[metadata[region_column].isin(region_set)].drop_duplicates(
        subset=[region_column, eregulon_column]
    )
    eregulons = sorted(metadata[eregulon_column].unique().tolist())
    region_to_position = {
        region: position for position, region in enumerate(region_index.astype(str))
    }
    positions = {}
    counts = np.zeros(len(eregulons), dtype=np.int32)
    for index, eregulon in enumerate(eregulons):
        labels = metadata.loc[
            metadata[eregulon_column] == eregulon, region_column
        ].tolist()
        current = np.array(
            [region_to_position[label] for label in labels], dtype=np.int32
        )
        positions[eregulon] = current
        counts[index] = current.size
    return eregulons, positions, counts


def estimate_accessibility_range(
    region_by_topic,
    cell_by_topic,
    cell_sampling=0.5,
    region_sampling=0.5,
    lower_percentile=1.0,
    upper_percentile=99.0,
    seed=0,
):
    """Estimate the global reconstructed-accessibility range by subsampling."""
    loading = np.asarray(region_by_topic)
    topic = np.asarray(cell_by_topic)
    rng = np.random.default_rng(seed)
    region_count = loading.shape[0]
    cell_count = topic.shape[0]
    cell_positions = rng.choice(
        cell_count,
        max(1, int(cell_count * float(cell_sampling))),
        replace=False,
    )
    region_positions = rng.choice(
        region_count,
        max(1, int(region_count * float(region_sampling))),
        replace=False,
    )
    sampled = loading[region_positions, :] @ topic[cell_positions, :].T
    lower = float(np.percentile(sampled, lower_percentile))
    upper = float(np.percentile(sampled, upper_percentile))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(
            "Invalid reconstructed-accessibility range: [{}, {}].".format(
                lower, upper
            )
        )
    return lower, upper


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
        aligned = aligned.loc[cells]
        if aligned.isna().any():
            raise ValueError("Group labels contain missing values.")
        return aligned.astype(str)
    values = np.asarray(groups)
    if values.shape[0] != len(cells):
        raise ValueError("Group-label length does not match the cell axis.")
    return pd.Series(values.astype(str), index=cells)


def _auc_axes(aucell):
    if isinstance(aucell, pd.DataFrame):
        return aucell.index.astype(str), aucell.columns.astype(str)
    return pd.Index(aucell.obs_names).astype(str), pd.Index(aucell.var_names).astype(str)


def _auc_values(aucell, cells, eregulons):
    if isinstance(aucell, pd.DataFrame):
        values = aucell.loc[cells, eregulons].to_numpy(dtype=np.float32)
    else:
        subset = aucell[cells, :][:, eregulons]
        values = subset.X
        if sparse.issparse(values):
            values = values.toarray()
        values = np.asarray(values, dtype=np.float32)
    return values


def _safe_group_name(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "group"


def compute_support(
    region_by_topic,
    cell_by_topic,
    responsibility_triplet,
    posterior,
    aucell,
    eregulon_metadata,
    groups,
    output_dir,
    posterior_exponent=3.0,
    batch_size=256,
    accessibility_epsilon=1e-6,
    accessibility_minimum=None,
    accessibility_maximum=None,
    restrict_eregulons_to_aucell=True,
):
    """Compute per-cell posterior-gated support into per-group memmaps.

    ``region_by_topic`` and ``responsibility_triplet`` are region-indexed.
    ``cell_by_topic`` and ``posterior`` are cell-indexed. The memmap for each
    group has shape cells by modules by eRegulons.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    responsibility = responsibility_triplet.copy()
    responsibility.index = canonicalize_regions(responsibility.index)
    responsibility.columns = responsibility.columns.astype(str)
    loading = region_by_topic.copy()
    loading.index = canonicalize_regions(loading.index)
    loading.columns = loading.columns.astype(str)
    topic = cell_by_topic.copy()
    topic.index = topic.index.astype(str)
    topic.columns = topic.columns.astype(str)
    posterior_values = posterior.copy()
    posterior_values.index = posterior_values.index.astype(str)
    posterior_values.columns = posterior_values.columns.astype(str)

    if responsibility.index.has_duplicates or responsibility.columns.has_duplicates:
        raise ValueError("Responsibility region and module labels must be unique.")
    if loading.index.has_duplicates or loading.columns.has_duplicates:
        raise ValueError("Region-by-topic labels must be unique.")
    if topic.index.has_duplicates or topic.columns.has_duplicates:
        raise ValueError("Cell-by-topic labels must be unique.")
    if posterior_values.index.has_duplicates or posterior_values.columns.has_duplicates:
        raise ValueError("Posterior cell and module labels must be unique.")

    missing_regions = responsibility.index.difference(loading.index)
    if len(missing_regions):
        raise ValueError(
            "Region-by-topic input is missing {} responsibility regions; examples: {}".format(
                len(missing_regions), missing_regions[:3].tolist()
            )
        )
    loading = loading.loc[responsibility.index]
    if set(loading.columns) != set(topic.columns):
        raise ValueError("Region-by-topic and cell-by-topic topic labels differ.")
    topic = topic.loc[:, loading.columns]

    modules = pd.Index(
        [
            module
            for module in responsibility.columns
            if module in set(posterior_values.columns)
        ]
    )
    if len(modules) == 0:
        raise ValueError(
            "No overlapping modules were found between responsibility and posterior matrices."
        )
    responsibility = responsibility.loc[:, modules].astype(np.float32)
    posterior_values = posterior_values.loc[:, modules].astype(np.float32)

    auc_cells, auc_eregulons = _auc_axes(aucell)
    cell_candidates = set(posterior_values.index) & set(auc_cells)
    cells = pd.Index(
        [cell for cell in topic.index if cell in cell_candidates]
    )
    if len(cells) == 0:
        raise ValueError("No overlapping cells were found across support inputs.")
    topic = topic.loc[cells].astype(np.float32)
    posterior_values = posterior_values.loc[cells].astype(np.float32)
    group_series = _align_groups(groups, cells)

    all_eregulons, all_positions, _ = build_eregulon_region_map(
        eregulon_metadata, responsibility.index
    )
    if restrict_eregulons_to_aucell:
        auc_set = set(auc_eregulons)
        eregulons = [name for name in all_eregulons if name in auc_set]
    else:
        eregulons = all_eregulons
    positions = {
        name: all_positions[name]
        for name in eregulons
        if all_positions[name].size > 0
    }
    eregulons = [name for name in eregulons if name in positions]
    if not eregulons:
        raise ValueError("No eRegulons overlap the support region universe.")
    region_counts = np.array(
        [positions[name].size for name in eregulons], dtype=np.int32
    )

    loading_array = loading.to_numpy(dtype=np.float32, copy=False)
    responsibility_array = responsibility.to_numpy(dtype=np.float32, copy=False)
    topic_array = topic.to_numpy(dtype=np.float32, copy=False)
    posterior_array = posterior_values.to_numpy(dtype=np.float32, copy=False)

    if accessibility_minimum is None or accessibility_maximum is None:
        range_minimum, range_maximum = estimate_accessibility_range(
            loading_array, topic_array, seed=0
        )
    else:
        range_minimum = float(accessibility_minimum)
        range_maximum = float(accessibility_maximum)
        if range_maximum <= range_minimum:
            raise ValueError("Accessibility maximum must exceed its minimum.")
    range_width = (range_maximum - range_minimum) + 1e-12

    number_regions, number_modules = responsibility_array.shape
    number_eregulons = len(eregulons)
    binary_responsibility = (responsibility_array > 0).astype(np.float32)
    membership = np.zeros((number_regions, number_eregulons), dtype=np.float32)
    for eregulon_position, eregulon in enumerate(eregulons):
        membership[positions[eregulon], eregulon_position] = 1.0
    responsible_counts = binary_responsibility.T @ membership

    output_paths = {}
    for group in sorted(group_series.unique().tolist()):
        group_mask = group_series.to_numpy() == group
        group_positions = np.where(group_mask)[0].astype(np.int32)
        group_cells = cells[group_mask]
        safe_group = _safe_group_name(group)
        support_path = output_dir / "support_{}.dat".format(safe_group)
        metadata_path = output_dir / "meta_{}.npz".format(safe_group)
        support_memmap = np.memmap(
            str(support_path),
            mode="w+",
            dtype=np.float32,
            shape=(len(group_positions), number_modules, number_eregulons),
        )

        write_position = 0
        for start in range(0, len(group_positions), int(batch_size)):
            end = min(start + int(batch_size), len(group_positions))
            current_positions = group_positions[start:end]
            current_topic = topic_array[current_positions, :]
            reconstructed = loading_array @ current_topic.T
            reconstructed = np.clip(
                (reconstructed - range_minimum) / range_width,
                float(accessibility_epsilon),
                1.0 - float(accessibility_epsilon),
            ).astype(np.float32)
            current_posterior = posterior_array[current_positions, :]
            posterior_gate = np.power(
                np.clip(current_posterior, 1e-12, 1.0),
                float(posterior_exponent),
            )
            block_size = len(current_positions)
            for module_position in range(number_modules):
                region_mask = binary_responsibility[:, module_position]
                summed = (reconstructed * region_mask[:, None]).T @ membership
                denominator = np.maximum(
                    responsible_counts[module_position, :], 1.0
                )
                mean_accessibility = summed / denominator[None, :]
                support_memmap[
                    write_position : write_position + block_size,
                    module_position,
                    :,
                ] = (
                    posterior_gate[:, module_position : module_position + 1]
                    * mean_accessibility
                ).astype(np.float32)
            write_position += block_size
        support_memmap.flush()
        del support_memmap

        np.savez(
            metadata_path,
            cells=np.asarray(group_cells.astype(str)),
            modules=np.asarray(modules.astype(str)),
            eregs=np.asarray(eregulons, dtype=object),
            reg_count=region_counts,
            posterior_exponent=np.asarray([posterior_exponent], dtype=np.float32),
            accessibility_minimum=np.asarray([range_minimum], dtype=np.float32),
            accessibility_maximum=np.asarray([range_maximum], dtype=np.float32),
        )
        output_paths[str(group)] = {
            "support_memmap": str(support_path),
            "metadata": str(metadata_path),
        }
    return output_paths


def _pearson_from_sums(
    number,
    sum_first,
    sum_first_squared,
    sum_second,
    sum_second_squared,
    sum_product,
    epsilon=1e-12,
):
    numerator = number * sum_product - sum_first * sum_second
    first_variance = number * sum_first_squared - sum_first * sum_first
    second_variance = number * sum_second_squared - sum_second * sum_second
    denominator = np.sqrt(
        np.maximum(first_variance, 0.0) * np.maximum(second_variance, 0.0)
    ) + epsilon
    return (numerator / denominator).astype(np.float32)


def aggregate_support(
    output_dir,
    groups,
    aucell,
    batch_size=256,
    save_npz=True,
    output_prefix="agg",
):
    """Aggregate support memmaps into the six-field  NPZ contract."""
    output_dir = Path(output_dir)
    metadata_files = sorted(output_dir.glob("meta_*.npz"))
    if not metadata_files:
        raise FileNotFoundError(
            "No support metadata files were found in {}.".format(output_dir)
        )
    results = {}
    for metadata_path in metadata_files:
        safe_group = metadata_path.stem.replace("meta_", "", 1)
        support_path = output_dir / "support_{}.dat".format(safe_group)
        if not support_path.exists():
            raise FileNotFoundError(
                "Support memmap is missing for {}: {}".format(
                    metadata_path.name, support_path
                )
            )
        with np.load(metadata_path, allow_pickle=True) as metadata:
            cells = metadata["cells"].astype(str)
            modules = metadata["modules"].astype(str)
            eregulons = metadata["eregs"].astype(str)
            region_counts = metadata["reg_count"].astype(np.int64)
        number_cells = len(cells)
        number_modules = len(modules)
        number_eregulons = len(eregulons)
        group_series = _align_groups(groups, cells)
        auc_values = _auc_values(aucell, cells.tolist(), eregulons.tolist())
        support_memmap = np.memmap(
            str(support_path),
            dtype=np.float32,
            mode="r",
            shape=(number_cells, number_modules, number_eregulons),
        )

        for group in sorted(group_series.unique().tolist()):
            positions = np.where(group_series.to_numpy() == group)[0]
            number = len(positions)
            if number == 0:
                continue
            sum_support = np.zeros(
                (number_modules, number_eregulons), dtype=np.float64
            )
            sum_support_squared = np.zeros_like(sum_support)
            sum_auc = np.zeros(number_eregulons, dtype=np.float64)
            sum_auc_squared = np.zeros(number_eregulons, dtype=np.float64)
            sum_product = np.zeros_like(sum_support)

            for start in range(0, number, int(batch_size)):
                end = min(start + int(batch_size), number)
                block_positions = positions[start:end]
                support_block = np.asarray(
                    support_memmap[block_positions, :, :], dtype=np.float32
                )
                auc_block = auc_values[block_positions, :]
                sum_support += support_block.sum(axis=0, dtype=np.float64)
                sum_support_squared += (support_block * support_block).sum(
                    axis=0, dtype=np.float64
                )
                sum_auc += auc_block.sum(axis=0, dtype=np.float64)
                sum_auc_squared += (auc_block * auc_block).sum(
                    axis=0, dtype=np.float64
                )
                sum_product += (
                    support_block * auc_block[:, None, :]
                ).sum(axis=0, dtype=np.float64)

            support_mean = (sum_support / number).astype(np.float32)
            correlation = _pearson_from_sums(
                number,
                sum_auc[None, :],
                sum_auc_squared[None, :],
                sum_support,
                sum_support_squared,
                sum_product,
            )
            aggregate = {
                "n_cells": np.asarray([number], dtype=np.int32),
                "modules": modules,
                "eregs": eregulons,
                "reg_count": region_counts,
                "support_mean_KE": support_mean,
                "corr_auc_sup_KE": correlation,
            }
            results[str(group)] = aggregate
            if save_npz:
                aggregate_path = output_dir / "{}_{}.npz".format(
                    output_prefix, _safe_group_name(group)
                )
                np.savez_compressed(aggregate_path, **aggregate)
        del support_memmap
    return results


def validate_aggregate(path):
    """Validate the exact six-field support aggregate schema and dimensions."""
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        expected = set(AGGREGATE_FIELDS)
        if keys != expected:
            raise ValueError(
                "Aggregate fields differ from the public contract; missing={}, extra={}.".format(
                    sorted(expected - keys), sorted(keys - expected)
                )
            )
        modules = archive["modules"].astype(str)
        eregulons = archive["eregs"].astype(str)
        region_counts = archive["reg_count"]
        support_mean = archive["support_mean_KE"]
        correlation = archive["corr_auc_sup_KE"]
        if support_mean.shape != (len(modules), len(eregulons)):
            raise ValueError("support_mean_KE dimensions do not match module/eRegulon axes.")
        if correlation.shape != support_mean.shape:
            raise ValueError("corr_auc_sup_KE dimensions do not match support_mean_KE.")
        if region_counts.shape != (len(eregulons),):
            raise ValueError("reg_count length does not match the eRegulon axis.")
        if archive["n_cells"].shape != (1,):
            raise ValueError("n_cells must be a length-one array.")
    return True

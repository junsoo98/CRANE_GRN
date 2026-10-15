"""RNA-informed prior calculation for factor-combination modules."""

import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.mixture import GaussianMixture


def _sigmoid(values):
    return expit(values)


def _safe_logit(probability, probability_clip=1e-4):
    return logit(np.clip(probability, probability_clip, 1.0 - probability_clip))


def _separation_score(means, variances):
    on_component = int(np.argmax(means))
    off_component = 1 - on_component
    numerator = float(means[on_component] - means[off_component])
    denominator = float(
        np.sqrt(0.5 * (variances[on_component] + variances[off_component]))
        + 1e-12
    )
    return numerator / denominator


def _quantile_tier(
    baseline,
    quantile_low,
    quantile_high,
    target_low,
    target_high,
    epsilon,
    probability_clip,
):
    values = baseline[np.isfinite(baseline)].astype(np.float64)
    if values.size == 0:
        return np.full_like(baseline, 0.5, dtype=np.float32)
    first = float(np.quantile(values, quantile_low))
    second = float(np.quantile(values, quantile_high))
    low_logit = float(_safe_logit(target_low, probability_clip))
    high_logit = float(_safe_logit(target_high, probability_clip))
    logit_range = high_logit - low_logit
    if abs(logit_range) < 1e-6 or abs(second - first) < 1e-12:
        scale = 1.0
        center = float(np.median(values))
    else:
        scale = float(np.clip((second - first) / logit_range, 1e-6, 1e6))
        center = first - scale * low_logit
    tier = _sigmoid((baseline.astype(np.float64) - center) / (scale + epsilon))
    return np.clip(tier, probability_clip, 1.0 - probability_clip).astype(
        np.float32
    )


def _fit_tier_gmm(
    baseline,
    regularization=1e-3,
    minimum_nonzero=100,
    fallback_quantile_low=0.25,
    fallback_quantile_high=0.75,
    fallback_target_low=0.05,
    fallback_target_high=0.95,
    epsilon=1e-8,
    probability_clip=1e-4,
):
    values = baseline.astype(np.float64)
    finite = np.isfinite(values)
    fit_values = values[finite]
    if fit_values.size == 0 or int((fit_values > 0).sum()) < int(minimum_nonzero):
        tier = _quantile_tier(
            values,
            fallback_quantile_low,
            fallback_quantile_high,
            fallback_target_low,
            fallback_target_high,
            epsilon,
            probability_clip,
        )
        return tier, {"separation": None, "forced_fallback": True}

    mixture = GaussianMixture(
        n_components=2,
        covariance_type="full",
        reg_covar=float(regularization),
        max_iter=200,
        random_state=0,
        init_params="kmeans",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixture.fit(fit_values.reshape(-1, 1))
    means = mixture.means_.reshape(-1)
    variances = mixture.covariances_.reshape(2, 1, 1)[:, 0, 0]
    separation = _separation_score(means, variances)
    on_component = int(np.argmax(means))
    responsibilities = mixture.predict_proba(fit_values.reshape(-1, 1))
    tier = np.zeros_like(values, dtype=np.float32)
    tier[finite] = responsibilities[:, on_component].astype(np.float32)
    tier = np.clip(tier, probability_clip, 1.0 - probability_clip)
    return tier, {
        "separation": float(separation),
        "means": means.tolist(),
        "forced_fallback": False,
    }


def _apply_separation_threshold(
    tier_gmm,
    gmm_information,
    baseline,
    minimum_separation,
    fallback_quantile_low=0.25,
    fallback_quantile_high=0.75,
    fallback_target_low=0.05,
    fallback_target_high=0.95,
    epsilon=1e-8,
    probability_clip=1e-4,
):
    if gmm_information.get("forced_fallback", False):
        return tier_gmm
    separation = gmm_information.get("separation")
    if (
        separation is not None
        and np.isfinite(separation)
        and separation >= float(minimum_separation)
    ):
        return tier_gmm
    return _quantile_tier(
        baseline,
        fallback_quantile_low,
        fallback_quantile_high,
        fallback_target_low,
        fallback_target_high,
        epsilon,
        probability_clip,
    )


def _aligned_groups(groups, cells):
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
        raise ValueError("Group-label length does not match the expression rows.")
    return pd.Series(values.astype(str), index=cells)


def _coerce_labeled_expression(normalized_expression, cell_names, gene_names):
    """Return a numeric matrix and validated cell/gene labels."""
    if isinstance(normalized_expression, pd.DataFrame):
        cells = pd.Index(normalized_expression.index).astype(str)
        genes = pd.Index(normalized_expression.columns).astype(str)

        if cell_names is not None:
            supplied_cells = pd.Index(cell_names).astype(str)
            if not cells.equals(supplied_cells):
                raise ValueError(
                    "cell_names does not match the normalized-expression index."
                )

        if gene_names is not None:
            supplied_genes = pd.Index(gene_names).astype(str)
            if not genes.equals(supplied_genes):
                raise ValueError(
                    "gene_names does not match the normalized-expression columns."
                )

        if all(
            isinstance(dtype, pd.SparseDtype)
            for dtype in normalized_expression.dtypes
        ):
            matrix = normalized_expression.sparse.to_coo().tocsr()
        else:
            matrix = normalized_expression.to_numpy(
                dtype=np.float32,
                copy=False,
            )
    else:
        if cell_names is None or gene_names is None:
            raise ValueError(
                "cell_names and gene_names are required when normalized_expression "
                "is not a pandas DataFrame."
            )
        cells = pd.Index(cell_names).astype(str)
        genes = pd.Index(gene_names).astype(str)
        matrix = normalized_expression

    if cells.has_duplicates:
        examples = cells[cells.duplicated()].unique()[:3].tolist()
        raise ValueError(
            "Cell labels are not unique; examples: {}".format(examples)
        )

    if genes.has_duplicates:
        examples = genes[genes.duplicated()].unique()[:3].tolist()
        raise ValueError(
            "Gene labels are not unique; examples: {}".format(examples)
        )

    if not hasattr(matrix, "ndim") or matrix.ndim != 2:
        raise ValueError("Expression input must be a two-dimensional matrix.")

    if matrix.shape != (len(cells), len(genes)):
        raise ValueError(
            "Expression shape {} does not match {} cells by {} genes.".format(
                matrix.shape,
                len(cells),
                len(genes),
            )
        )

    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=False)
        stored = matrix.data
        if stored.size and not np.isfinite(stored).all():
            raise ValueError("Expression input contains non-finite values.")
        if stored.size and float(stored.min()) < 0.0:
            raise ValueError("Expression input must be non-negative.")
    else:
        matrix = np.asarray(matrix)
        if not np.isfinite(matrix).all():
            raise ValueError("Expression input contains non-finite values.")
        if matrix.size and float(matrix.min()) < 0.0:
            raise ValueError("Expression input must be non-negative.")

    return matrix, cells, genes


def _transform_group_expression(expression, row_positions, input_is_log1p):
    """Select one group and optionally apply log1p without densifying sparse input."""
    group_expression = expression[row_positions, :]

    if sparse.issparse(group_expression):
        transformed = group_expression.tocsr(copy=True).astype(np.float32)
        if not input_is_log1p:
            transformed.data = np.log1p(transformed.data).astype(
                np.float32,
                copy=False,
            )
        transformed.eliminate_zeros()
        return transformed

    transformed = np.asarray(group_expression, dtype=np.float32)
    if input_is_log1p:
        return transformed
    return np.log1p(transformed).astype(np.float32, copy=False)


def _column_quantile_nonnegative(matrix, quantile):
    """Compute exact column quantiles, including implicit sparse zeros."""
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must lie between zero and one.")

    if matrix.shape[0] == 0:
        raise ValueError("Cannot calculate a column quantile from zero rows.")

    if not sparse.issparse(matrix):
        return np.quantile(
            np.asarray(matrix),
            float(quantile),
            axis=0,
        ).astype(np.float32)

    csc = matrix.tocsc(copy=True)
    csc.sum_duplicates()
    csc.eliminate_zeros()

    if csc.data.size and float(csc.data.min()) < 0.0:
        raise ValueError(
            "Sparse column quantiles require non-negative expression values."
        )

    number_rows, number_columns = csc.shape
    position = (number_rows - 1) * float(quantile)
    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))
    interpolation_weight = float(position - lower_rank)
    output = np.zeros(number_columns, dtype=np.float32)

    for column in range(number_columns):
        start = csc.indptr[column]
        stop = csc.indptr[column + 1]
        nonzero = np.sort(
            csc.data[start:stop].astype(np.float64, copy=False)
        )
        number_zeros = number_rows - nonzero.size

        def value_at_rank(rank):
            if rank < number_zeros:
                return 0.0
            return float(nonzero[rank - number_zeros])

        lower_value = value_at_rank(lower_rank)
        upper_value = value_at_rank(upper_rank)
        output[column] = lower_value + interpolation_weight * (
            upper_value - lower_value
        )

    return output


def _dense_columns(matrix, positions):
    """Materialize only selected columns as a dense float32 matrix."""
    selected = matrix[:, positions]
    if sparse.issparse(selected):
        selected = selected.toarray()
    return np.asarray(selected, dtype=np.float32)


def _tier_method(gmm_information, minimum_separation):
    if gmm_information.get("forced_fallback", False):
        return "fallback_minimum_nonzero"

    separation = gmm_information.get("separation")
    if (
        separation is not None
        and np.isfinite(separation)
        and separation >= float(minimum_separation)
    ):
        return "GMM"

    return "fallback_separation"


def compute_module_prior(
    normalized_expression,
    factor_by_module,
    groups,
    *,
    cell_names=None,
    gene_names=None,
    prior_clip_max=0.80,
    beta_slope=1.5,
    z_min_scale=0.10,
    z_cap=1.5,
    tier_min_separation=0.35,
    prior_clip_min=0.01,
    z_quantile_low=0.25,
    z_quantile_high=0.75,
    epsilon=1e-8,
    probability_clip=1e-4,
    minimum_cells_per_group=500,
    input_is_log1p=True,
    return_diagnostics=False,
):
    """Compute a cell-by-module RNA-informed prior matrix.

    ``normalized_expression`` is the full cell-by-gene expression matrix used
    to calibrate expression tiers within each group. It may be a labeled
    pandas DataFrame, a SciPy sparse matrix, or a NumPy array. Sparse/array
    inputs require explicit ``cell_names`` and ``gene_names``. Values must be
    non-negative and must originate from full-library normalization. Set
    ``input_is_log1p=False`` only when the supplied values still require the
    element-wise log1p transformation.

    ``factor_by_module`` has factors as rows and modules as columns. The GMM is
    fitted to the per-gene baseline distribution of the complete expression
    matrix; only the module-factor columns are materialized densely for
    cell-level scoring and module aggregation. A factor-only expression matrix
    is therefore not a valid input.

    Groups below ``minimum_cells_per_group`` retain an all-zero prior. When
    diagnostics are requested, gene- and tier-level tables contain module
    factors only; background genes are used for calibration but are not
    expanded into diagnostic rows.
    """
    expression, cells, genes = _coerce_labeled_expression(
        normalized_expression,
        cell_names,
        gene_names,
    )

    membership = factor_by_module.copy()
    membership.index = membership.index.astype(str)
    membership.columns = membership.columns.astype(str)

    if membership.index.has_duplicates or membership.columns.has_duplicates:
        raise ValueError("Membership factor and module labels must be unique.")

    factors = pd.Index(membership.index).astype(str)
    missing_factors = factors.difference(genes)
    if len(missing_factors):
        raise ValueError(
            "Normalized expression is missing required factors: {}".format(
                missing_factors.tolist()
            )
        )

    if len(genes.difference(factors)) == 0:
        raise ValueError(
            "compute_module_prior requires a background transcriptome for "
            "group-level GMM calibration; a factor-only expression matrix is "
            "not a valid input."
        )

    group_series = _aligned_groups(groups, cells)
    factor_gene_positions = genes.get_indexer(factors)
    if (factor_gene_positions < 0).any():
        raise ValueError("Factor-to-expression alignment failed.")

    modules = membership.columns.tolist()
    factor_position = {
        factor: position
        for position, factor in enumerate(factors)
    }

    module_positions = []
    module_sizes = np.zeros(len(modules), dtype=np.int32)

    for module_position, module in enumerate(modules):
        module_factors = membership.index[
            membership[module].astype(bool)
        ].tolist()
        positions = [
            factor_position[factor]
            for factor in module_factors
        ]
        module_positions.append(positions)
        module_sizes[module_position] = len(positions)

    prior_values = np.zeros(
        (len(cells), len(modules)),
        dtype=np.float32,
    )
    module_rows = []
    gene_rows = [] if return_diagnostics else None
    tier_rows = [] if return_diagnostics else None

    group_values = group_series.to_numpy()

    for group in pd.Categorical(group_series).categories.tolist():
        group_positions = np.flatnonzero(group_values == group)
        number_cells = len(group_positions)

        if number_cells < int(minimum_cells_per_group):
            continue

        transformed = _transform_group_expression(
            expression,
            group_positions,
            input_is_log1p,
        )

        # Calibrate the expression tier against the complete gene pool.
        full_baseline = _column_quantile_nonnegative(
            transformed,
            0.50,
        )

        tier_gmm, gmm_information = _fit_tier_gmm(
            full_baseline,
            probability_clip=probability_clip,
            epsilon=epsilon,
        )
        full_tier = _apply_separation_threshold(
            tier_gmm,
            gmm_information,
            full_baseline,
            minimum_separation=tier_min_separation,
            probability_clip=probability_clip,
            epsilon=epsilon,
        )

        # Only module factors are materialized densely.
        factor_expression = _dense_columns(
            transformed,
            factor_gene_positions,
        )
        factor_baseline = full_baseline[
            factor_gene_positions
        ]
        factor_tier = full_tier[
            factor_gene_positions
        ]

        lower = np.quantile(
            factor_expression,
            z_quantile_low,
            axis=0,
        ).astype(np.float32)
        upper = np.quantile(
            factor_expression,
            z_quantile_high,
            axis=0,
        ).astype(np.float32)
        raw_spread = (upper - lower).astype(np.float32)
        scale = np.maximum(
            raw_spread,
            float(z_min_scale),
        ).astype(np.float32)

        z_score = np.clip(
            (
                factor_expression
                - factor_baseline.reshape(1, -1)
            )
            / (
                scale.reshape(1, -1)
                + epsilon
            ),
            -float(z_cap),
            float(z_cap),
        ).astype(np.float32)

        clipped_tier = np.clip(
            factor_tier,
            float(prior_clip_min),
            float(prior_clip_max),
        ).astype(np.float32)

        factor_logit = (
            _safe_logit(
                clipped_tier,
                probability_clip,
            ).reshape(1, -1)
            + float(beta_slope) * z_score
        )

        factor_prior = np.clip(
            _sigmoid(factor_logit),
            0.0,
            1.0,
        ).astype(np.float32)

        group_prior = np.zeros(
            (number_cells, len(modules)),
            dtype=np.float32,
        )

        for module_position, positions in enumerate(module_positions):
            if positions:
                group_prior[:, module_position] = (
                    factor_prior[:, positions].mean(axis=1)
                )

        group_prior = np.clip(
            group_prior,
            0.0,
            1.0,
        ).astype(np.float32)

        prior_values[group_positions, :] = group_prior

        for module_position, module in enumerate(modules):
            values = group_prior[:, module_position]
            module_rows.append(
                {
                    "group": group,
                    "module": module,
                    "module_size": int(
                        module_sizes[module_position]
                    ),
                    "prior_q01": float(
                        np.quantile(values, 0.01)
                    ),
                    "prior_q50": float(
                        np.quantile(values, 0.50)
                    ),
                    "prior_q99": float(
                        np.quantile(values, 0.99)
                    ),
                    "prior_mean": float(np.mean(values)),
                    "prior_std": float(np.std(values)),
                }
            )

        if return_diagnostics:
            method = _tier_method(
                gmm_information,
                tier_min_separation,
            )
            calibration_n_positive = int(
                np.sum(
                    np.isfinite(full_baseline)
                    & (full_baseline > 0)
                )
            )

            for position, factor in enumerate(factors):
                gene_rows.append(
                    {
                        "group": group,
                        "gene": factor,
                        "baseline_median_log1p": float(
                            factor_baseline[position]
                        ),
                        "tier": float(
                            factor_tier[position]
                        ),
                        "raw_spread_q75_q25": float(
                            raw_spread[position]
                        ),
                        "scale_used": float(
                            scale[position]
                        ),
                        "n_cells": int(number_cells),
                    }
                )

                tier_rows.append(
                    {
                        "group": group,
                        "gene": factor,
                        "baseline_median_log1p": float(
                            factor_baseline[position]
                        ),
                        "tier": float(
                            factor_tier[position]
                        ),
                        "tier_method": method,
                        "tier_separation": (
                            gmm_information.get("separation")
                        ),
                        "tier_means": (
                            gmm_information.get("means")
                        ),
                        "calibration_n_genes": int(
                            len(genes)
                        ),
                        "calibration_n_positive": (
                            calibration_n_positive
                        ),
                        "n_cells": int(number_cells),
                    }
                )

    prior = pd.DataFrame(
        prior_values,
        index=cells,
        columns=modules,
        dtype=np.float32,
    )
    module_diagnostics = pd.DataFrame(module_rows)

    if return_diagnostics:
        return (
            prior,
            module_diagnostics,
            pd.DataFrame(gene_rows),
            pd.DataFrame(tier_rows),
        )

    return prior
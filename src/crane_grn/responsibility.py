"""Compute region-by-module responsibility matrices."""

import itertools

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr


DEFAULT_POPULARITY_STRENGTH = 0.2
DEFAULT_SOFTMAX_TEMPERATURE = 0.1
DEFAULT_PARAMETER_GATES = {"max_stability_jsd": 0.05, "max_order_corr": 0.20}
_EPSILON = 1e-9


def compute_responsibility_idf(region_by_factor):
    """Compute responsibility IDF ``log((1 + N) / (1 + df)) + 1``."""
    if region_by_factor.shape[0] == 0:
        raise ValueError("The region-by-factor matrix has no rows.")
    frequency = region_by_factor.astype(bool).sum(axis=0).to_numpy(dtype=float)
    values = np.log((1.0 + region_by_factor.shape[0]) / (1.0 + frequency)) + 1.0
    return pd.Series(
        values,
        index=region_by_factor.columns.astype(str),
        name="responsibility_idf",
    )


def tversky_similarity(
    region_by_factor,
    factor_by_module,
    factor_idf,
    tversky_alpha=0.3,
):
    """Calculate IDF-weighted Tversky similarity for regions by modules.

    ``region_by_factor`` has regions as rows and factors as columns.
    ``factor_by_module`` has factors as rows and modules as columns.
    """
    if not 0.0 <= float(tversky_alpha) <= 1.0:
        raise ValueError("tversky_alpha must be between zero and one.")
    factors = factor_by_module.index.astype(str)
    missing = factors.difference(region_by_factor.columns.astype(str))
    if len(missing):
        raise ValueError(
            "The region-by-factor matrix is missing module factors: {}".format(
                missing.tolist()
            )
        )
    missing_idf = factors.difference(factor_idf.index.astype(str))
    if len(missing_idf):
        raise ValueError("Factor IDF is missing factors: {}".format(missing_idf.tolist()))

    regions = region_by_factor.loc[:, factors].astype(bool)
    membership = factor_by_module.loc[factors].astype(bool)
    idf = factor_idf.loc[factors].to_numpy(dtype=float)
    weighted_regions = regions.to_numpy(dtype=float) * idf[None, :]
    module_values = membership.to_numpy(dtype=bool)
    intersection = weighted_regions @ module_values.astype(int)
    region_mass = weighted_regions.sum(axis=1)[:, None]
    module_mass = (idf[:, None] * module_values).sum(axis=0)[None, :]
    tversky_beta = 1.0 - float(tversky_alpha)
    denominator = (
        float(tversky_alpha) * region_mass + tversky_beta * module_mass
    )
    return intersection / (denominator + _EPSILON)


def popularity_penalty(region_by_factor, factor_by_module, factor_idf):
    """Return log IDF-weighted module popularity and unlogged popularity."""
    factors = factor_by_module.index.astype(str)
    regions = region_by_factor.loc[:, factors].astype(bool).to_numpy(dtype=float)
    membership = factor_by_module.loc[factors].astype(bool).to_numpy(dtype=bool)
    idf = factor_idf.loc[factors].to_numpy(dtype=float)
    weighted_regions = regions * idf[None, :]
    intersection = weighted_regions @ membership.astype(int)
    module_mass = (idf[:, None] * membership).sum(axis=0)[None, :]
    popularity = (intersection / (module_mass + _EPSILON)).mean(axis=0)
    penalty = np.log(np.clip(popularity, _EPSILON, 1.0))
    return penalty, popularity


def build_responsibility(
    similarity,
    factor_by_module,
    region_by_factor,
    factor_idf,
    order_strength,
    top_m,
    popularity_strength=DEFAULT_POPULARITY_STRENGTH,
    softmax_temperature=DEFAULT_SOFTMAX_TEMPERATURE,
):
    """Transform Tversky similarity into region-by-module responsibility.

    The paper order correction, popularity correction, softmax, and Top-M
    sparsification are applied in that order. Rows sum to one after optional
    Top-M renormalization.
    """
    similarity = np.asarray(similarity, dtype=float)
    expected_shape = (region_by_factor.shape[0], factor_by_module.shape[1])
    if similarity.shape != expected_shape:
        raise ValueError(
            "Similarity shape {} does not match regions by modules {}.".format(
                similarity.shape, expected_shape
            )
        )
    adjusted = similarity.copy()
    if float(order_strength) > 0:
        module_order = factor_by_module.sum(axis=0).to_numpy(dtype=float)
        base = np.maximum(module_order, 1.0)
        geometric_mean = np.exp(np.mean(np.log(base)))
        adjusted = adjusted / np.power(
            base / geometric_mean, float(order_strength)
        )[None, :]

    penalty, _ = popularity_penalty(
        region_by_factor, factor_by_module, factor_idf
    )
    logits = (
        adjusted / max(float(softmax_temperature), _EPSILON)
        - float(popularity_strength) * penalty[None, :]
    )
    logits -= logits.max(axis=1, keepdims=True)
    responsibility = softmax(logits, axis=1)

    number_modules = responsibility.shape[1]
    if 0 < int(top_m) < number_modules:
        indices = np.argpartition(
            -responsibility,
            kth=min(int(top_m), number_modules - 1),
            axis=1,
        )[:, : int(top_m)]
        mask = np.zeros_like(responsibility, dtype=bool)
        mask[np.arange(responsibility.shape[0])[:, None], indices] = True
        responsibility = np.where(mask, responsibility, 0.0)
        responsibility /= np.maximum(
            responsibility.sum(axis=1, keepdims=True), _EPSILON
        )
    return responsibility


def compute_responsibility(
    region_by_factor,
    factor_by_module,
    tversky_alpha,
    order_strength,
    top_m,
    popularity_strength=DEFAULT_POPULARITY_STRENGTH,
    softmax_temperature=DEFAULT_SOFTMAX_TEMPERATURE,
):
    """Compute a labeled region-by-module responsibility DataFrame."""
    if region_by_factor.index.has_duplicates:
        raise ValueError("Region labels are not unique.")
    if factor_by_module.index.has_duplicates or factor_by_module.columns.has_duplicates:
        raise ValueError("Factor or module labels are not unique.")
    factor_idf = compute_responsibility_idf(region_by_factor)
    similarity = tversky_similarity(
        region_by_factor,
        factor_by_module,
        factor_idf,
        tversky_alpha=tversky_alpha,
    )
    values = build_responsibility(
        similarity,
        factor_by_module,
        region_by_factor,
        factor_idf,
        order_strength=order_strength,
        top_m=top_m,
        popularity_strength=popularity_strength,
        softmax_temperature=softmax_temperature,
    )
    return pd.DataFrame(
        values,
        index=region_by_factor.index.astype(str),
        columns=factor_by_module.columns.astype(str),
    )


def stability_jsd(responsibility, bootstrap_count=3, fraction=0.3, seed=42):
    """Measure stability of top-module assignment using adjacent bootstrap JSD."""
    values = np.asarray(responsibility)
    number_regions, number_modules = values.shape
    sample_size = max(int(number_regions * float(fraction)), 10)
    rng = np.random.default_rng(seed)
    distributions = []
    for _ in range(int(bootstrap_count)):
        indices = rng.choice(number_regions, size=sample_size, replace=False)
        counts = np.bincount(
            values[indices].argmax(axis=1), minlength=number_modules
        ).astype(float)
        counts /= max(counts.sum(), _EPSILON)
        distributions.append(counts)
    divergences = []
    for first, second in zip(distributions[:-1], distributions[1:]):
        midpoint = 0.5 * (first + second)
        midpoint = np.clip(midpoint, _EPSILON, 1.0)
        divergence = 0.5 * (
            np.where(
                first > 0,
                first * np.log(np.clip(first, _EPSILON, 1.0) / midpoint),
                0,
            ).sum()
            + np.where(
                second > 0,
                second * np.log(np.clip(second, _EPSILON, 1.0) / midpoint),
                0,
            ).sum()
        )
        divergences.append(divergence)
    return float(np.mean(divergences))


def assignment_gini(responsibility):
    """Return the Gini coefficient of top-module assignment counts."""
    values = np.asarray(responsibility)
    counts = np.bincount(values.argmax(axis=1), minlength=values.shape[1]).astype(float)
    ordered = np.sort(counts)
    cumulative = np.cumsum(ordered)
    number = len(ordered)
    return float(
        (number + 1 - 2 * cumulative.sum() / max(cumulative[-1], _EPSILON))
        / number
    )


def median_assignment_gap(responsibility):
    """Return the median difference between the largest two row probabilities."""
    values = np.asarray(responsibility)
    if values.shape[1] < 2:
        raise ValueError("At least two modules are required for assignment-gap scoring.")
    partitioned = np.partition(values, -2, axis=1)
    return float(np.median(partitioned[:, -1] - partitioned[:, -2]))


def order_assignment_correlation(responsibility, factor_by_module):
    """Return Spearman correlation between module order and top-1 frequency."""
    values = np.asarray(responsibility)
    counts = np.bincount(values.argmax(axis=1), minlength=values.shape[1]).astype(float)
    rates = counts / max(counts.sum(), _EPSILON)
    orders = factor_by_module.sum(axis=0).to_numpy(dtype=float)
    correlation, _ = spearmanr(orders, rates)
    return float(correlation)


def parameter_grid_search(
    region_by_factor,
    factor_by_module,
    tversky_grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    top_m_grid=(0, 3, 4, 5, 6, 7, 8, 9, 10),
    order_grid=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    gates=None,
):
    """Evaluate the paper responsibility parameter grid without plotting."""
    gates = DEFAULT_PARAMETER_GATES if gates is None else gates
    factor_idf = compute_responsibility_idf(region_by_factor)
    rows = []
    for tversky_value in tversky_grid:
        similarity = tversky_similarity(
            region_by_factor,
            factor_by_module,
            factor_idf,
            tversky_alpha=tversky_value,
        )
        for order_strength, top_m in itertools.product(order_grid, top_m_grid):
            values = build_responsibility(
                similarity,
                factor_by_module,
                region_by_factor,
                factor_idf,
                order_strength=order_strength,
                top_m=top_m,
            )
            jsd = stability_jsd(values)
            if jsd > float(gates["max_stability_jsd"]) * 1.5:
                rows.append(
                    {
                        "tversky_alpha": tversky_value,
                        "top_m": top_m,
                        "order_strength": order_strength,
                        "stability_jsd": jsd,
                        "gini": np.nan,
                        "gap_median": np.nan,
                        "order_corr": np.nan,
                        "passes_absolute": False,
                    }
                )
                continue
            correlation = order_assignment_correlation(values, factor_by_module)
            rows.append(
                {
                    "tversky_alpha": tversky_value,
                    "top_m": top_m,
                    "order_strength": order_strength,
                    "stability_jsd": jsd,
                    "gini": assignment_gini(values),
                    "gap_median": median_assignment_gap(values),
                    "order_corr": correlation,
                    "passes_absolute": jsd <= gates["max_stability_jsd"]
                    and abs(correlation) <= gates["max_order_corr"],
                }
            )
    return pd.DataFrame(rows)


def select_responsibility_parameters(results, gates=None):
    """Apply the paper two-stage gates and rank aggregation to grid results."""
    gates = DEFAULT_PARAMETER_GATES if gates is None else gates
    complete = results.dropna(
        subset=["stability_jsd", "gini", "gap_median", "order_corr"]
    ).copy()
    if complete.empty:
        raise ValueError("No complete responsibility parameter results were supplied.")
    stage_one = complete[
        (complete["stability_jsd"] <= gates["max_stability_jsd"])
        & (complete["order_corr"].abs() <= gates["max_order_corr"])
    ].copy()
    if stage_one.empty:
        stage_one = complete
    maximum_gini = float(np.percentile(stage_one["gini"], 30))
    minimum_gap = float(np.percentile(stage_one["gap_median"], 70))
    stage_two = stage_one[
        (stage_one["gini"] <= maximum_gini)
        & (stage_one["gap_median"] >= minimum_gap)
    ].copy()
    if stage_two.empty:
        stage_two = stage_one.copy()
    stage_two["rank_stability"] = stage_two["stability_jsd"].rank(ascending=True)
    stage_two["rank_gini"] = stage_two["gini"].rank(ascending=True)
    stage_two["rank_gap"] = stage_two["gap_median"].rank(ascending=False)
    stage_two["rank_order"] = stage_two["order_corr"].abs().rank(ascending=True)
    stage_two["rank_sum"] = stage_two[
        ["rank_stability", "rank_gini", "rank_gap", "rank_order"]
    ].sum(axis=1)
    ranked = stage_two.sort_values("rank_sum")
    return (
        ranked.iloc[0].to_dict(),
        {"max_gini": maximum_gini, "min_gap": minimum_gap},
        ranked,
    )

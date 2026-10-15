"""Integrate RNA prior and ATAC evidence into module posterior values."""

import numpy as np
import pandas as pd


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
        raise ValueError("Group-label length does not match the aligned cell axis.")
    return pd.Series(values.astype(str), index=cells)


def compute_posterior(
    log_likelihood,
    prior,
    groups,
    prior_weight=1.0,
    evidence_weight=1.0,
    prior_clip=(1e-6, 1.0 - 1e-6),
    epsilon=1e-8,
    logit_cap=None,
):
    """Compute a cell-by-module posterior using group-wise median/IQR evidence.

    Cells and modules are aligned by name. Modules whose log-likelihood is
    entirely missing are excluded. Any partially non-finite retained module is
    rejected because median/IQR evidence would otherwise be ambiguous.
    """
    likelihood = log_likelihood.copy()
    likelihood.index = likelihood.index.astype(str)
    likelihood.columns = likelihood.columns.astype(str)
    prior_values = prior.copy()
    prior_values.index = prior_values.index.astype(str)
    prior_values.columns = prior_values.columns.astype(str)
    if likelihood.index.has_duplicates or likelihood.columns.has_duplicates:
        raise ValueError("Log-likelihood cell and module labels must be unique.")
    if prior_values.index.has_duplicates or prior_values.columns.has_duplicates:
        raise ValueError("Prior cell and module labels must be unique.")

    cells = pd.Index(
        [cell for cell in likelihood.index if cell in set(prior_values.index)]
    )
    modules = pd.Index(
        [module for module in likelihood.columns if module in set(prior_values.columns)]
    )
    if len(cells) == 0:
        raise ValueError("No overlapping cells were found between likelihood and prior.")
    if len(modules) == 0:
        raise ValueError("No overlapping modules were found between likelihood and prior.")
    likelihood = likelihood.loc[cells, modules].astype(np.float32)
    prior_values = prior_values.loc[cells, modules].astype(np.float32)

    all_missing = likelihood.isna().all(axis=0)
    valid_modules = modules[~all_missing.to_numpy()]
    if len(valid_modules) == 0:
        raise ValueError("All aligned log-likelihood modules are entirely missing.")
    likelihood = likelihood.loc[:, valid_modules]
    prior_values = prior_values.loc[:, valid_modules]
    likelihood_array = likelihood.to_numpy(dtype=np.float64)
    prior_array = prior_values.to_numpy(dtype=np.float64)
    if not np.isfinite(likelihood_array).all():
        bad = likelihood.columns[
            ~np.isfinite(likelihood_array).all(axis=0)
        ].tolist()
        raise ValueError(
            "Retained log-likelihood modules contain partially non-finite values: {}".format(
                bad[:10]
            )
        )
    if not np.isfinite(prior_array).all():
        raise ValueError("Prior contains non-finite values after label alignment.")

    group_series = _aligned_groups(groups, cells)
    lower, upper = prior_clip
    clipped_prior = np.clip(prior_array, float(lower), float(upper))
    prior_logit = np.log(clipped_prior) - np.log(1.0 - clipped_prior)
    prior_term = float(prior_weight) * prior_logit
    if logit_cap is not None:
        prior_term = np.clip(prior_term, -float(logit_cap), float(logit_cap))

    posterior = np.full_like(likelihood_array, np.nan, dtype=np.float64)
    group_values = group_series.to_numpy()
    for group in pd.Categorical(group_series).categories.tolist():
        positions = np.where(group_values == group)[0]
        if positions.size == 0:
            continue
        block = likelihood_array[positions, :]
        baseline = np.median(block, axis=0, keepdims=True)
        centered = block - baseline
        first_quartile = np.quantile(centered, 0.25, axis=0)
        median = np.quantile(centered, 0.50, axis=0)
        third_quartile = np.quantile(centered, 0.75, axis=0)
        scale = np.maximum(third_quartile - first_quartile, float(epsilon))
        evidence = (centered - median) / scale
        combined = prior_term[positions, :] + float(evidence_weight) * evidence
        if logit_cap is not None:
            combined = np.clip(combined, -float(logit_cap), float(logit_cap))
        posterior[positions, :] = 1.0 / (1.0 + np.exp(-combined))

    return pd.DataFrame(posterior, index=cells, columns=valid_modules)

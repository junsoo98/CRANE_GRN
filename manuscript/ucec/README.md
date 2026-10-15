# UCEC manuscript analyses

The UCEC archive documents downstream comparisons between Tumor cells and the
MUC5B epithelial population used as the normal epithelial counterpart. Internal
code uses the explicit labels `tumor` and `muc5b`; display text may use
`Normal epithelial` where appropriate.

These files are manuscript-specific reference implementations. They assume
that the relevant CRANE-GRN outputs and, for selected analyses, original
manuscript intermediates have already been generated.

## Files

### `rewiring_analysis.py`

Records condition-specific module ranking, the Tumor vs Normal rewiring
score, within-condition split controls, paired statistical comparisons, and RSS
association. The primary  Tumor vs Normal score can be calculated from the
public aggregates. Split controls and RSS association are reference-only unless
the original split matrices and RSS table are supplied.

### `pathway_analysis.py`

Records target-gene extraction from the top quartile of rewired eRegulons,
KEGG 2021 over-representation analysis, raw-count pseudobulk aggregation,
PyDESeq2 differential expression, preranked GSEA, and ORA/GSEA comparison.  
Pseudobulk analysis is reference-only because it requires raw RNA counts and
sample-level metadata. The executed code excluded groups with fewer than 20
cells.
### `factor_recurrence_and_networks.py`

Records paired eRegulon-level regulatory-factor recurrence, paired bootstrap
confidence intervals, candidate-module filtering, frozen representative-module
selection, and representative module network visualization.




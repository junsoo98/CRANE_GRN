# Manuscript-specific analysis archive

This directory contains curated reference implementations of the downstream
analyses used in the CRANE-GRN manuscript. These files preserve scientific
calculations, dataset-specific definitions, statistical tests, and analysis
provenance after the  CRANE-GRN framework outputs had been generated.

They are not standalone end-to-end workflows. Some sections can be evaluated
from the public data archive, while others require manuscript-specific
intermediate objects that are not redistributed. Those requirements are stated
at file and section level. No synthetic replacement or approximate
reconstruction is used when an original intermediate is unavailable.

## Reproducibility status

The terms used in this archive have the following meanings:

- **Public-data reproducible**: the section uses objects distributed in the
  CRANE-GRN public data archive in Zenodo.
- **Reference-only**: the section records the executed calculation but requires
  a non-distributed manuscript intermediate.

## Contents

- `ov/support_aucell_validation.py`: support-rank, direction, AUCell, and
  cell-label permutation analyses.
- `ov/arid1a_ko_atac_validation.py`: external ARID1A-knockout ATAC validation.
- `ucec/rewiring_analysis.py`: Tumor vs Normal module-rank rewiring and associated controls.
- `ucec/pathway_analysis.py`: target-gene enrichment, pseudobulk differential
  expression, and preranked GSEA.
- `ucec/factor_recurrence_and_networks.py`: paired factor recurrence, candidate-module summaries, and illustrative module networks.


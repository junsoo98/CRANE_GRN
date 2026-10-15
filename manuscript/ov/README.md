# OV manuscript analyses

The OV archive records downstream analyses performed after CRANE-GRN had
generated module responsibility, support, and eRegulon-linked
outputs for tumor cells. It is a scientific reference archive, not a raw-data
processing workflow.

## Files

### `support_aucell_validation.py`

This file records:

- support-rank quartile comparisons within eRegulons;
- within-eRegulon rank permutation testing;
- support-AUCell consistency by eRegulon direction;
- cell-label permutation of cell-resolved support;
- region-based versus gene-based AUCell comparisons.

### `arid1a_ko_atac_validation.py`

This file records the classification of ARID1A-containing modules and regions,
intersection with independently processed ATAC peaks, extraction of WT and KO
bigWig signal, KO-versus-WT log2 fold change, group comparisons, stability
resampling, and maximum-responsibility quartile analysis.




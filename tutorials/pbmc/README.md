# CRANE-GRN PBMC canonical tutorial

This tutorial starts from standardized post-SCENIC+/cisTopic inputs and follows
the CRANE-GRN workflow through factor-combination modules, Bayesian integration,
posterior-gated support, and connection to direct eRegulons. It does not repeat
raw 10x processing or full SCENIC+/cisTopic reconstruction. PBMC is the canonical
multi-cell-type example; the same framework also supports one biological group.

## Prerequisites

- The CRANE-GRN repository with `src/crane_grn/`.
- The prepared PBMC archive described below.
- The main CRANE environment for notebooks 00-01 and steps 03-06.
- A separate Python 3.6 environment for iterative Random Forest execution.

Until packaging metadata is added, run from the repository root and expose the
source layout directly:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export CRANE_GRN_DATA_ROOT=/path/to/CRANE_GRN_data
export CRANE_GRN_TUTORIAL_WORK="$PWD/tutorials/pbmc/results"
```

The iRF runner imports a cloned checkout of
[Yu-Group/iterative-Random-Forest](https://github.com/Yu-Group/iterative-Random-Forest)
through `--irf-module-path`; it is not treated as a standard pip package. The
historical environment used Python 3.6.15 with Cython 0.29.24, NumPy 1.17.5,
pandas 1.1.5, SciPy 1.5.3, scikit-learn 0.22.2.post1, PyArrow 6.0.1, and
packaging 21.3. 

## Data and output layout

```text
CRANE_GRN_data/PBMC/
├── prepared_inputs/
│   ├── cistopic_obj.pkl
│   ├── fileList.tab
│   ├── region_by_factor_hvr.parquet
│   ├── region_by_factor_triplet.parquet
│   └── scplusmdata.h5mu
└── outputs/
    ├── AUCell_direct.h5mu
    ├── eRegulon_direct.tsv
    ├── factor_by_module.parquet
    ├── prior.parquet
    ├── log_likelihood.parquet
    ├── posterior.parquet
    ├── responsibility_hvr.parquet
    ├── responsibility_triplet.parquet
    └── agg_<cell_type>.npz
```

`prepared_inputs` contains the public workflow boundary. `outputs` contains
frozen manuscript references. New tutorial results are written
under `CRANE_GRN_TUTORIAL_WORK` (default: `tutorials/pbmc/results`) and never
overwrite the archive.

## Execution order and environments

| Step | Environment | Role |
|---|---|---|
| `00_get_prepared_inputs.ipynb` | main CRANE | Inspect the public workflow boundary. |
| `01_factor_region_inputs.ipynb` | main CRANE | Explain factor-region construction and create 13 grouped iRF inputs. |
| `02_irf_rit.py` | Python 3.6 iRF | Train iRF and extract raw RIT interaction statistics. |
| `03_responsibility.ipynb` | main CRANE | Select modules and compute region-by-module responsibility. |
| `04_bayesian_integration.ipynb` | main CRANE | Compute RNA prior, ATAC log-likelihood, and posterior. |
| `05_support.py` | main CRANE | Compute local support tensors and six-field group aggregates. |
| `06_connect_eregulons.ipynb` | main CRANE | Build ranks, module-eRegulon edges, and compact summaries. |

Launch the notebooks from their directory so local helper imports are stable:

```bash
cd tutorials/pbmc
jupyter lab 00_get_prepared_inputs.ipynb
```

After running notebooks 00 and 01, switch to the iRF environment:

```bash
python tutorials/pbmc/02_irf_rit.py \
  --input tutorials/pbmc/results/01_irf_inputs \
  --output-dir tutorials/pbmc/results/02_irf \
  --irf-module-path /path/to/iterative-Random-Forest
```

For a one-group dataset, pass one Parquet file instead of a directory:

```bash
python tutorials/pbmc/02_irf_rit.py \
  --input /path/to/group_iRF_input.parquet \
  --output-dir /path/to/work/02_irf \
  --irf-module-path /path/to/iterative-Random-Forest
```

Return to the main CRANE environment for notebook 03, notebook 04, and support:

```bash
python tutorials/pbmc/05_support.py \
  --data-root "$CRANE_GRN_DATA_ROOT" \
  --work-dir "$CRANE_GRN_TUTORIAL_WORK" \
  --group-column cell_type \
  --batch-size 256 \
  --posterior-exponent 3.0
```

To run the prepared PBMC inputs as one group, replace `--group-column` with a
constant label. Dataset-specific reproduction wrappers use the same core source
pattern with their own prepared paths:

```bash
python tutorials/pbmc/05_support.py \
  --data-root "$CRANE_GRN_DATA_ROOT" \
  --work-dir "$CRANE_GRN_TUTORIAL_WORK" \
  --single-group-name all_cells
```






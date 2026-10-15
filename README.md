# CRANE-GRN

CRANE-GRN extends an existing single-cell multiomic GRN/eRegulon backbone by
associating candidate upstream regulatory-factor combinations with accessible
regulatory regions of existing eRegulons. The public workflow begins from
standardized post-SCENIC+/cisTopic inputs (scplusmdata.h5mu).
The method follows the conceptual path:

```text
regulatory-factor combinations -> regulatory regions -> existing eRegulons
```

## Repository overview

- [`src/crane_grn/`](src/crane_grn/) contains the reusable, dataset-independent
  CRANE-GRN source code.
- [`tutorials/pbmc/`](tutorials/pbmc/) is the canonical executable tutorial and
  manuscript-matched reference workflow.
- [`manuscript/`](manuscript/) contains curated OV and UCEC downstream analyses.
  It is a dataset-specific reference archive, not a standalone end-to-end
  workflow.

## Installation

Clone the repository and enter its root directory:

```bash
git clone <repository-url>
cd CRANE-GRN
```

This release uses a source-layout checkout rather than an installable package.
Expose the source directory before running the tutorial or manuscript scripts:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

## Environment setup

Most CRANE-GRN steps use the main environment:

```bash
conda env create -f environments/crane_grn.yml
conda activate crane_grn
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

Iterative Random Forest training and RIT extraction use a separate legacy
environment because their historical dependencies conflict with the main
stack:

```bash
conda env create -f environments/irf_env.yml
conda activate irf_env
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

The iRF step imports a separately cloned checkout of
[Yu-Group/iterative-Random-Forest](https://github.com/Yu-Group/iterative-Random-Forest)
through an `--irf-module-path` argument; it is not installed as a standard pip
package. The exact historical iRF commit remains unresolved and is not guessed
in this repository. See the [PBMC tutorial README](tutorials/pbmc/README.md) for
the environment transition and invocation details.

The environment files preserve the versions and platform-specific build pins
from the working manuscript environments. 
## How to use

For a first run, follow the numbered steps in the
[PBMC canonical tutorial](tutorials/pbmc/README.md). It documents the prepared
input boundary, environment switch, expected dimensions, frozen reference
outputs, and execution commands.

The [manuscript analysis archive](manuscript/README.md) records curated OV and
UCEC analyses performed after CRANE-GRN outputs were generated. 
## Analysis flow

The canonical workflow is organized as follows:

```text
post-SCENIC+/cisTopic inputs
  -> ChIP-informed region-by-factor occupancy matrices
  -> iterative Random Forest and RIT interactions
  -> selected factor-combination modules
  -> region-by-module responsibility matrices
  -> RNA prior + ATAC likelihood -> posterior module score
  -> posterior-gated support
  -> connections to existing direct eRegulons
```

PBMC provides the complete canonical example. OV and UCEC retain the clean
manuscript analyses that can be evaluated from their documented public inputs.

## Repository structure

```text
CRANE-GRN/
├── environments/
│   ├── crane.yml
│   └── irf.yml
├── manuscript/
│   ├── ov/
│   └── ucec/
├── src/
│   └── crane_grn/
├── tutorials/
│   └── pbmc/
├── .gitignore
├── LICENSE
└── README.md
```

## Data

Large prepared inputs and frozen manuscript reference outputs are distributed
separately from the Git repository. The PBMC archive begins at standardized
post-SCENIC+/cisTopic objects and includes the reference artifacts needed to
validate the canonical workflow. File roles, expected names, dimensions, and
checkpoints are listed in the [PBMC tutorial README](tutorials/pbmc/README.md).

Raw and large upstream data are not duplicated in this repository. Consult the
dataset documentation for provenance, accession information, and whether an
analysis is public-data reproducible or reference-only. An archive DOI can be
added after the public data release is finalized.

## License

CRANE-GRN is distributed under the
[GNU General Public License, version 3](LICENSE).

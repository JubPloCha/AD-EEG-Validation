# Third-party notices and license scope

This file records the source and licensing boundaries for the files distributed
in this repository. The root [MIT License](../LICENSE) applies only to original
contributions that Peichen Jiang has the right to license. It does not replace a
third-party license, grant rights to data or fitted models, or turn a reference
to a published method into a claim of authorship.

## Original project contributions

The workflow, input/path adapters, feature-table contract, tests, examples,
reports, documentation and release configuration were organized and written
for this project and are covered by the root MIT grant, subject to the specific
exception below. PCA, cPCA, logistic regression, Haufe patterns and the EEG
feature formulas are established methods; the project does not claim to have
invented them.

## cPCA method and implementation

The contrastive principal component analysis (cPCA) method follows Abid et al.
(2018). The authors' [contrastive repository](https://github.com/abidlabs/contrastive)
is acknowledged as the reference implementation.

In this project, `cpca_fit` in `src/cpca_ad_nc/modeling.py` computes directions
from the target-minus-background covariance objective using a QR-based subspace
eigendecomposition. The function is integrated into the project's training,
cross-validation and frozen external-validation workflow. This description
documents the numerical implementation and method attribution; it does not
establish source-code derivation or licensing.

**Reference:** Abid A, Zhang MJ, Bagaria VK, Zou J. Exploring patterns enriched
in a dataset with contrastive principal component analysis. *Nature Communications*.
2018;9:2134. [doi:10.1038/s41467-018-04608-8](https://doi.org/10.1038/s41467-018-04608-8).

### Source-code licensing and evidence

The source-code derivation of `cpca_fit` cannot be established from the public
release history or the private project history available for this audit. The
function remains excluded from Peichen Jiang's MIT grant until provenance
and rights evidence identifies whether it is an independent implementation or
a permitted adaptation. This is a provenance reservation, not a claim that
the function infringes copyright.

The related upstream repository is [abidlabs/contrastive](https://github.com/abidlabs/contrastive).
The locally inspected history provides these evidence points:

* `ab8865889f49af2acc66dfebdc83f390e8dbd0bb` (2017-08-05) contains a standard
  MIT `LICENSE`, Copyright (c) 2017 Abubakar Abid.
* `03d138451f7caefa6903267b37e7c8442aca084a` (2018-10-19) contains the
  historical `contrastive/__init__.py` implementation. Its `CPCA.cpca_alpha`
  computes `fg_cov - alpha * bg_cov` and calls an eigen-decomposition. This is
  algorithmic evidence only; it does not prove that this repository copied
  source code.
* `63a6dff4e288a21358f7928e0e7fb8a115533f75` (2023-02-06) still has the
  standard MIT text.
* `575db5a3530d6fabb828dd8ebeb5e4acde316511` (2025-10-07) changes the
  upstream license to MIT–NSFG. The verbatim copy retained at
  [`contrastive_LICENSE.md`](contrastive_LICENSE.md) has SHA256
  `8cce841e8a0d086a8d459cb7da3754e6d21d8987bea2d0b7952de9c668d67b6b` and
  remains attributed to Abubakar Abid. Its additional condition must not be
  removed or relabeled as standard MIT.

The available history does not identify which upstream revision, if any, was
actually consulted for `cpca_fit`. No upstream code is imported at runtime and
no upstream source file is distributed in this repository. Similar mathematics
and function names are insufficient to determine copying or a license grant.

## Python dependencies

These packages are imported by the released code or required by its build and
retain their own licenses. The versions below are the tested environment
versions; their installed distribution metadata and license files were checked
locally during this audit. Their bundled subcomponents retain any additional
notices shipped by those distributions.

| Component | Tested version | Use | License evidence |
|---|---:|---|---|
| NumPy | 2.4.4 | arrays and linear algebra | BSD-3-Clause plus bundled component notices in its distribution metadata |
| SciPy | 1.17.1 | signal processing, statistics and linear algebra | BSD-3-Clause plus bundled OpenBLAS/LAPACK/GCC-runtime notices in its distribution metadata |
| pandas | 3.0.2 | tabular input/output | BSD 3-Clause License in the distribution license file |
| h5py | 3.16.0 | EEGLAB v7.3 HDF5 reading | BSD-3-Clause; HDF5 and bundled component notices remain with h5py |
| scikit-learn | 1.8.0 | PCA and logistic regression | BSD-3-Clause in `COPYING` |
| statsmodels | 0.14.6 | optional cognition regression | BSD License in the distribution license file |
| threadpoolctl | 3.6.0 | deterministic thread limits | BSD-3-Clause in the distribution license file |
| setuptools | 82.0.1 | build backend only | MIT; not a runtime library and not vendored here |

The repository does not copy these packages or their license files into source
code. Installing them creates a separate dependency environment governed by
their respective terms; the project MIT grant does not relicense them.

## EEGLAB, MATLAB and REST

`matlab/verify_eeglab_reader.m` is a small project wrapper that calls the user's
separately installed EEGLAB `pop_loadset` and `eeg_getversion` functions. EEGLAB,
MATLAB and any REST implementation are not bundled or redistributed here. The
wrapper does not copy their source and does not perform preprocessing. Users
must obtain and license those tools from their official sources and record the
versions used for a new run; the private audit recorded EEGLAB 2026.0.0, while
the historical REST plugin version was not preserved.

## Methods, papers and data

The cPCA method is cited in `modeling.py` (Abid et al., *Nature
Communications*, 2018, DOI `10.1038/s41467-018-04608-8`). Haufe et al.'s
interpretation method is cited in the interpretation documentation (DOI
`10.1016/j.neuroimage.2013.10.067`). These citations acknowledge methods; they
are not code licenses.

The external dataset reference is OpenNeuro `ds004504` v1.0.8, pinned in the
project's private provenance to commit
`73b1b04c0a8e45e96c6f0682cc077a7919480fc4`, with CC0 metadata. The source is
<https://github.com/OpenNeuroDatasets/ds004504>. No EEG, clinical table,
identity mapping, subject-level feature, fitted model or private transformation
is distributed here. Dataset terms and access restrictions remain independent
of the software license.

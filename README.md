# AD–NC EEG validation

Research code for resting EEG classification of Alzheimer's disease (AD) versus normal controls (NC), with internal validation, a matched sensitivity cohort, frozen independent external testing, and interpretation. This repository contains code and **random synthetic demonstrations only**. It contains no study EEG, clinical records, subject features, identities, predictions or fitted study models.

Complete EEG, ordinary PCA and contrastive PCA (cPCA) are **parallel representation branches**, each followed by regularized, class-balanced logistic regression. Each branch is also evaluated with age and sex; a demographic-only branch provides a baseline. cPCA uses AD as target and NC as background. It is not a post-hoc explanation of the complete EEG classifier. The final study selection included **alpha=0**: this does not establish an additional NC-background contrast effect or superiority over ordinary PCA. Interpret each classifier using its own parameters.

## Install and run a quick demonstration

Tested with Python 3.12.3 on Windows. From the repository root, in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\cpca-ad-nc.exe --help
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe examples\synthetic_demo.py
```

The demo generates 96 training and 32 independent test identities beginning with `SYN_`, using all 752 schema columns. A deliberately small **demo-only** grid trains all seven branches, freezes/reloads them and verifies identical predictions and rejection of wrong feature order. It uses no real feature values. Omit `--output` to use a temporary directory, or supply `--output .\demo_output`. Synthetic performance is not a paper result.

The release check used an isolated virtual environment with the existing tested dependencies; installation, command help, eight regression tests, the demo and a synthetic run through every primary analysis stage were actually executed. A private comparison of 350 saved outer-fold models and seven frozen external prediction branches retained the original AUCs (maximum probability difference below 1e-8). Full EEG extraction and full paper nested CV were not repeated for this packaging step. See [reproducibility](docs/reproducibility.md) for the boundaries of this check.

## Use your own authorized data

```powershell
Copy-Item configs\config.example.toml configs\config.local.toml
# Edit config.local.toml before running. It is ignored by Git.
.\.venv\Scripts\cpca-ad-nc.exe prepare --config configs\config.local.toml
.\.venv\Scripts\cpca-ad-nc.exe internal --config configs\config.local.toml
.\.venv\Scripts\cpca-ad-nc.exe freeze --config configs\config.local.toml
.\.venv\Scripts\cpca-ad-nc.exe external --config configs\config.local.toml
.\.venv\Scripts\cpca-ad-nc.exe interpret --config configs\config.local.toml
.\.venv\Scripts\cpca-ad-nc.exe report --config configs\config.local.toml
```

`all` runs those six stages in order. Paths resolve relative to the configuration file; Unicode and spaces are supported. All results and fitted models go to your configured private workspace. Existing model fingerprints must match to reuse a run; changed training inputs require a new workspace. No old project or repair script is needed.

There are two supported input paths:

1. **Subject-level features:** provide metadata and a feature CSV with `subject_id` plus the exact ordered 752 columns in [feature_schema.json](configs/feature_schema.json). The [feature dictionary](configs/feature_dictionary.csv) supplies type, band, electrode, region and units. Tables join explicitly by unique ID. Required metadata: `age`, `sex` (female=0, male=1), and `group` (AD/NC) for training/evaluation. MMSE and education are optional and never classification eligibility criteria.
2. **Processed EEG:** the retained extractor reads the validated **EEGLAB v7.3 HDF5 `.set` + paired little-endian float32 `.fdt` layout**, 500 Hz, REST and verified stored microvolt units. Add `set_path` to metadata; run `features --cohort development` (or `external`). Then set that cohort's `*_features = ''` to use the generated features and QC metadata. This is not an EDF reader or a raw preprocessing/ICA/REST pipeline. See [MATLAB interoperability](docs/reproducibility.md#eeglab-and-matlab).

For **prediction only**, supply external metadata (age/sex; no diagnosis needed), features and a private `paths.frozen_bundle`:

```powershell
.\.venv\Scripts\cpca-ad-nc.exe predict --config configs\config.local.toml
```

It loads frozen parameters and performs transform/predict only. External data never fit nuisance regression, scaling, PCA/cPCA, classifiers or selection. `external` additionally requires both diagnostic groups and computes the prespecified metrics and uncertainty. The study's private frozen bundle is not distributed.

Optional `cognition` and `stability` stages provide frozen-score MMSE analyses and conditional bootstrap/leave-one-out analysis. Neither affects classifier selection. Interpretation exports backprojected weights, training-only Haufe patterns, fixed-model grouped test permutations, electrode/region tables and complete component loadings. `report` summarizes actual outputs in Markdown; the paper's custom figure rendering and private figure files are not part of this release.

## Data access and interpretation limits

The study's own clinical and EEG data are access-controlled and are not downloadable here. The public external source is OpenNeuro [ds004504 **v1.0.8**](https://doi.org/10.18112/openneuro.ds004504.v1.0.8), CC0, pinned upstream metadata commit `73b1b04c0a8e45e96c6f0682cc077a7919480fc4`. Obtain it from the original repository; it is not bundled. Public raw recordings still require compatible, documented preprocessing before using this extractor. See [third-party notices](docs/THIRD_PARTY_NOTICES.md).

Sixteen scalp electrodes support electrode-level and coarse scalp-region patterns, not cortical networks or source localization. Weights, Haufe associations, permutation dependence and patient metric increases/decreases have different meanings. Pooled AD+NC MMSE correlation can reflect diagnostic separation; it is not proof of cognitive severity differences within AD. AD-only analyses and planned covariate adjustment must be inspected separately. All synthetic runs verify software paths only; paper-number reproduction needs the corresponding restricted inputs, exact splits, configurations and environment.

## License and attribution

Original code and documentation that Peichen Jiang has the right to license are released under the standard [MIT License](LICENSE), **Copyright (c) 2026 Peichen Jiang**. This covers the project's original workflow, adapters, tests, examples and documentation; it does not claim authorship of PCA, cPCA, logistic regression, Haufe patterns or other established methods.

**Specific exception:** the derivation history of `src/cpca_ad_nc/modeling.py::cpca_fit` remains unverified, so that function is excluded from this MIT grant pending provenance/rights evidence. The upstream cPCA repository has both a verified historical standard-MIT revision and a later MIT–NSFG revision; the historical revision actually referenced by this project is unknown. The retained [MIT–NSFG license](docs/contrastive_LICENSE.md) is unchanged and is not standard MIT. See [third-party notices](docs/THIRD_PARTY_NOTICES.md) for exact file/function scope, commit evidence and the unresolved gap. The repository is therefore not represented as unconditionally MIT-only.

External dependencies and third-party materials retain their own copyrights and licenses. This software license grants no rights to private EEG, clinical records, external datasets or fitted study models; their access and license terms remain separate.

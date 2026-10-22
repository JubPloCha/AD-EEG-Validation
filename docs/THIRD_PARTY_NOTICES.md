# Third-party sources

Numerical implementation uses NumPy, SciPy, scikit-learn, pandas, h5py,
statsmodels and matplotlib under their respective licenses. EEGLAB is used
locally for read verification; it is not bundled. No raw/private EEG is bundled.

cPCA reference implementation and license: https://github.com/abidlabs/contrastive
(the retained upstream LICENSE.md is MIT–NSFG v1, with an additional restriction;
see the verbatim docs/contrastive_LICENSE.md, not an unqualified MIT claim). The current
thin-QR covariance computation preserves the audited study implementation and
does not import that repository at runtime.

Haufe et al. (2014), On the interpretation of weight vectors of linear models
in multivariate neuroimaging, NeuroImage 87:96–110,
https://doi.org/10.1016/j.neuroimage.2013.10.067 .

ds004504 source: https://github.com/OpenNeuroDatasets/ds004504 .
Pinned dataset metadata: commit 73b1b04c0a8e45e96c6f0682cc077a7919480fc4,
dataset DOI https://doi.org/10.18112/openneuro.ds004504.v1.0.8 , CC0.
Miltiadous et al. (2023), A Dataset of Scalp EEG Recordings of Alzheimer's
Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG,
Data 8(6):95, https://doi.org/10.3390/data8060095 .
First associated analysis: https://doi.org/10.1109/ACCESS.2023.3294618 .
Official source data/license does not imply that private local transformations,
identities or study models may automatically be redistributed.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from .features import BANDS, CHANNELS

ALPHA_GRID = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DIM_GRID = (1, 2, 3, 5, 10, 20)
C_GRID = (0.01, 0.1, 1.0, 10.0)
MODEL_NAMES = (
    "age_sex", "age_adjusted_eeg", "age_adjusted_pca", "age_adjusted_cpca",
    "age_sex_age_adjusted_eeg", "age_sex_age_adjusted_pca",
    "age_sex_age_adjusted_cpca",
)


@dataclass(frozen=True)
class ModelConfig:
    alpha_grid: tuple[float, ...] = ALPHA_GRID
    dim_grid: tuple[int, ...] = DIM_GRID
    c_grid: tuple[float, ...] = C_GRID
    inner_splits: int = 3
    outer_splits: int = 5
    outer_repeats: int = 5
    random_seed: int = 20261018
    robust_clip: bool = False


def feature_columns(columns) -> tuple[list[str], dict[str, list[str]]]:
    spectral, connectivity, complexity = [], [], []
    for name in columns:
        name = str(name)
        if name.startswith("hjorth_complexity_broad_"):
            band = "broad"
        else:
            import re
            m = re.search(r"(?:power_log10|relative_log10|spectral_entropy|dwpli2|hjorth_complexity)_([^_]+)", name)
            band = m.group(1) if m else "broad"
        if band not in set(BANDS) | {"broad"}:
            continue
        if name.startswith(("power_log10_", "relative_log10_", "spectral_entropy_")):
            spectral.append(name)
        elif name.startswith("dwpli2_"):
            connectivity.append(name)
        elif name.startswith("hjorth_complexity_"):
            complexity.append(name)
    blocks = {k: v for k, v in {"spectral": spectral, "connectivity": connectivity, "complexity": complexity}.items() if v}
    return [*spectral, *connectivity, *complexity], blocks


def _fit_age_reference(X: np.ndarray, groups: np.ndarray, age: np.ndarray, sex: np.ndarray):
    if not np.isfinite(age).all() or not np.isfinite(sex).all():
        raise ValueError("Age/sex nuisance fitting requires complete covariates")
    nc = np.flatnonzero(groups == "NC")
    if len(nc) < 8:
        raise ValueError(f"At least 8 NC training subjects required for nuisance regression, got {len(nc)}")
    A = np.column_stack([np.ones(len(nc)), age[nc], sex[nc]])
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("NC age/sex design is rank deficient")
    beta, *_ = np.linalg.lstsq(A, X[nc], rcond=None)
    return {"beta": beta, "n_nc": len(nc), "age_min": float(age[nc].min()), "age_max": float(age[nc].max())}


def _apply_age_reference(X: np.ndarray, age: np.ndarray, sex: np.ndarray, reference):
    if not np.isfinite(age).all() or not np.isfinite(sex).all():
        raise ValueError("Age-adjusted prediction requires complete age and sex")
    A = np.column_stack([np.ones(len(age)), age, sex])
    return np.asarray(X, float) - A @ reference["beta"]


def _fit_transform(X: np.ndarray, names: list[str], groups: np.ndarray, age: np.ndarray,
                   sex: np.ndarray, cfg: ModelConfig):
    X = np.asarray(X, float)
    median = np.nanmedian(X, axis=0)
    median[~np.isfinite(median)] = 0.0
    Xi = np.where(np.isfinite(X), X, median)
    qlo = np.full(X.shape[1], -np.inf); qhi = np.full(X.shape[1], np.inf)
    if cfg.robust_clip:
        qlo = np.quantile(Xi, 0.01, axis=0); qhi = np.quantile(Xi, 0.99, axis=0)
        Xi = np.clip(Xi, qlo, qhi)
    sd = np.nanstd(Xi, axis=0, ddof=1)
    keep = np.isfinite(sd) & (sd > 1e-12)
    if not keep.any():
        raise ValueError("All EEG features are constant or non-finite")
    kept = [name for name, flag in zip(names, keep) if flag]
    Xk = Xi[:, keep]
    age_ref = _fit_age_reference(Xk, groups, age, sex)
    R = _apply_age_reference(Xk, age, sex, age_ref)
    center = R.mean(axis=0)
    scale = R.std(axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    _, blocks = feature_columns(kept)
    weights = np.ones(len(kept))
    for block_names in blocks.values():
        idx = np.isin(kept, block_names)
        weights[idx] = 1.0 / np.sqrt(int(idx.sum()))
    return {
        "feature_names": list(names), "kept_names": kept, "keep": keep,
        "median": median, "qlo": qlo, "qhi": qhi, "center": center,
        "scale": scale, "block_weights": weights, "age_beta": age_ref["beta"],
        "age_support_min": age_ref["age_min"], "age_support_max": age_ref["age_max"],
        "n_nc_reference": age_ref["n_nc"], "robust_clip": bool(cfg.robust_clip),
    }


def _transform(X: np.ndarray, age: np.ndarray, sex: np.ndarray, tr: dict[str, Any]):
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != len(tr["feature_names"]):
        raise ValueError("EEG feature matrix does not match the frozen feature schema")
    Xi = np.where(np.isfinite(X), X, tr["median"])
    if tr["robust_clip"]:
        Xi = np.clip(Xi, tr["qlo"], tr["qhi"])
    R = Xi[:, tr["keep"]]
    if not np.isfinite(age).all() or not np.isfinite(sex).all():
        raise ValueError("Frozen age-adjusted models require observed age and sex")
    A = np.column_stack([np.ones(len(X)), age, sex])
    R = R - A @ tr["age_beta"]
    Z = (R - tr["center"]) / tr["scale"]
    return Z * tr["block_weights"]


def _fit_demo(A: np.ndarray):
    if not np.isfinite(A).all():
        raise ValueError("Age/sex model requires observed covariates")
    center = A.mean(axis=0); scale = A.std(axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return {"center": center, "scale": scale}


def _demo_transform(A: np.ndarray, tr):
    if not np.isfinite(A).all():
        raise ValueError("Age/sex model requires observed covariates")
    return (A - tr["center"]) / tr["scale"]


# Method: Abid et al. (2018), doi:10.1038/s41467-018-04608-8.
# Reference implementation: https://github.com/abidlabs/contrastive.
# This implementation uses a QR-based subspace eigendecomposition of the
# target-minus-background covariance objective. Method attribution and
# source-code licensing/provenance are documented separately in
# docs/THIRD_PARTY_NOTICES.md; its cpca_fit licensing exception remains in force.
def cpca_fit(Z_target: np.ndarray, Z_background: np.ndarray, alpha: float):
    """Fit target covariance minus alpha times background covariance."""
    T = np.asarray(Z_target, float); B = np.asarray(Z_background, float)
    if len(T) < 3 or len(B) < 3 or not np.isfinite(T).all() or not np.isfinite(B).all():
        raise ValueError("cPCA requires finite target and background training matrices")
    Tc = T - T.mean(axis=0, keepdims=True); Bc = B - B.mean(axis=0, keepdims=True)
    dt, db = len(T) - 1, len(B) - 1
    factor = np.concatenate([Tc.T / np.sqrt(dt), np.sqrt(max(alpha, 0.0)) * Bc.T / np.sqrt(db)], axis=1)
    Q, _ = np.linalg.qr(factor, mode="reduced")
    Ct = (Q.T @ Tc.T) @ (Tc @ Q) / dt
    Cb = (Q.T @ Bc.T) @ (Bc @ Q) / db
    vals, vecs = np.linalg.eigh((Ct - alpha * Cb + (Ct - alpha * Cb).T) / 2)
    order = np.argsort(vals)[::-1]
    return {"components": Q @ vecs[:, order], "eigenvalues": vals[order], "alpha": float(alpha),
            "target_mean": T.mean(axis=0), "background_mean": B.mean(axis=0)}


def fit_logistic(A: np.ndarray, y: np.ndarray, C: float, seed: int):
    clf = LogisticRegression(C=float(C), max_iter=2000, class_weight="balanced", random_state=int(seed))
    clf.fit(A, y)
    if clf.classes_.tolist() != [0, 1]:
        raise AssertionError(f"Expected NC=0, AD=1 classes, got {clf.classes_.tolist()}")
    return clf


def _inner_splits(y, n_splits, seed, pair_ids=None):
    y = np.asarray(y, int)
    if pair_ids is not None:
        pair_ids = np.asarray(pair_ids)
        if len(pair_ids) != len(y) or pd.isna(pair_ids).any():
            raise ValueError("Matched analysis needs one observed pair_id per subject")
        # Preserve v3.1 first-occurrence pair order and its seeded allocation.
        # Each pair has both classes; the dummy pair stratum has no label role.
        pairs = pd.unique(pair_ids)
        for pid in pairs:
            idx = np.flatnonzero(pair_ids == pid)
            if len(idx) != 2 or sorted(y[idx].tolist()) != [0, 1]:
                raise ValueError(f"Pair {pid} must contain exactly one NC and one AD")
        n = min(int(n_splits), len(pairs))
        if n < 2:
            raise ValueError("Insufficient matched pairs for inner CV")
        cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=int(seed))
        return [(np.flatnonzero(np.isin(pair_ids, pairs[tr])), np.flatnonzero(np.isin(pair_ids, pairs[va])))
                for tr, va in cv.split(pairs, np.zeros(len(pairs), dtype=int))]
    n = min(int(n_splits), int(np.bincount(y).min()))
    if n < 2:
        raise ValueError("Insufficient class counts for inner CV")
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=int(seed))
    return list(cv.split(np.zeros(len(y)), y))


def _candidate_rows(X, y, groups, age, sex, names, cfg, seed, pair_ids=None):
    rows = []
    for inner_fold, (tr_i, va_i) in enumerate(_inner_splits(y, cfg.inner_splits, seed, pair_ids)):
        gtr = groups[tr_i]
        Ztr_tr = _fit_transform(X[tr_i], names, gtr, age[tr_i], sex[tr_i], cfg)
        Ztr = _transform(X[tr_i], age[tr_i], sex[tr_i], Ztr_tr)
        Zva = _transform(X[va_i], age[va_i], sex[va_i], Ztr_tr)
        Dtr_raw = np.column_stack([age[tr_i], sex[tr_i]])
        Dva_raw = np.column_stack([age[va_i], sex[va_i]])
        Dtr_tr = _fit_demo(Dtr_raw)
        Dtr, Dva = _demo_transform(Dtr_raw, Dtr_tr), _demo_transform(Dva_raw, Dtr_tr)
        ncomp = min(max(cfg.dim_grid), Ztr.shape[0] - 1, Ztr.shape[1])
        pca = PCA(n_components=ncomp, random_state=seed + inner_fold).fit(Ztr)
        Ptr = pca.transform(Ztr); Pva = pca.transform(Zva)
        target, bg = gtr == "AD", gtr == "NC"
        branches = {
            "age_sex": [(Dtr, Dva, 0.0, 0)],
            "age_adjusted_eeg": [(Ztr, Zva, 0.0, 0)],
            "age_sex_age_adjusted_eeg": [(np.column_stack([Dtr, Ztr]), np.column_stack([Dva, Zva]), 0.0, 0)],
            "age_adjusted_pca": [], "age_sex_age_adjusted_pca": [],
        }
        for dim in cfg.dim_grid:
            d = min(dim, ncomp)
            if d >= 1:
                branches["age_adjusted_pca"].append((Ptr[:, :d], Pva[:, :d], 0.0, d))
                branches["age_sex_age_adjusted_pca"].append((np.column_stack([Dtr, Ptr[:, :d]]), np.column_stack([Dva, Pva[:, :d]]), 0.0, d))
        for model, model_branches in branches.items():
            for Atr, Ava, alpha, dim in model_branches:
                for C in cfg.c_grid:
                    try:
                        clf = fit_logistic(Atr, y[tr_i], C, seed + inner_fold)
                        auc = roc_auc_score(y[va_i], clf.predict_proba(Ava)[:, 1])
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    rows.append({"inner_fold": inner_fold, "model": model, "alpha": alpha, "dimension": dim,
                                 "C": C, "inner_auc": float(auc), "n_train": len(tr_i), "n_valid": len(va_i)})
        for alpha in cfg.alpha_grid:
            cp = cpca_fit(Ztr[target], Ztr[bg], alpha)
            dmax = min(max(cfg.dim_grid), cp["components"].shape[1], int(target.sum()) - 1, int(bg.sum()) - 1)
            Str = Ztr @ cp["components"][:, :dmax]
            Sva = Zva @ cp["components"][:, :dmax]
            for dim in cfg.dim_grid:
                dcp = min(dim, dmax)
                if dcp < 1:
                    continue
                cbranches = {
                    "age_adjusted_cpca": (Str[:, :dcp], Sva[:, :dcp]),
                    "age_sex_age_adjusted_cpca": (np.column_stack([Dtr, Str[:, :dcp]]), np.column_stack([Dva, Sva[:, :dcp]])),
                }
                for model, (Atr, Ava) in cbranches.items():
                    for C in cfg.c_grid:
                        try:
                            clf = fit_logistic(Atr, y[tr_i], C, seed + inner_fold)
                            auc = roc_auc_score(y[va_i], clf.predict_proba(Ava)[:, 1])
                        except (ValueError, np.linalg.LinAlgError):
                            continue
                        rows.append({"inner_fold": inner_fold, "model": model, "alpha": float(alpha), "dimension": int(dcp),
                                     "C": C, "inner_auc": float(auc), "n_train": len(tr_i), "n_valid": len(va_i)})
    return pd.DataFrame(rows)


def select_hyperparameters(X, y, groups, age, sex, names, cfg: ModelConfig, seed: int, pair_ids=None):
    table = _candidate_rows(np.asarray(X, float), np.asarray(y, int), np.asarray(groups, str),
                            np.asarray(age, float), np.asarray(sex, float), list(names), cfg, seed, pair_ids)
    if table.empty:
        raise RuntimeError("No valid inner-CV candidate completed")
    means = table.groupby(["model", "alpha", "dimension", "C"], as_index=False).inner_auc.mean()
    selected = {}
    for model in MODEL_NAMES:
        q = means[means.model == model]
        if q.empty:
            raise RuntimeError(f"No valid candidates for {model}")
        q = q.sort_values(["inner_auc", "dimension", "alpha", "C"], ascending=[False, True, True, True], kind="mergesort")
        best = q.iloc[0].to_dict()
        selected[model] = {"model": model, "alpha": float(best["alpha"]),
                           "dimension": int(best["dimension"]), "C": float(best["C"]),
                           "inner_auc": float(best["inner_auc"]), "inner_n": int(cfg.inner_splits)}
    return selected, table


def fit_frozen_models(X, y, groups, age, sex, names, selected, cfg: ModelConfig, seed: int):
    X = np.asarray(X, float); y = np.asarray(y, int); groups = np.asarray(groups, str)
    age = np.asarray(age, float); sex = np.asarray(sex, float)
    tr = _fit_transform(X, list(names), groups, age, sex, cfg)
    Z = _transform(X, age, sex, tr)
    demo = _fit_demo(np.column_stack([age, sex]))
    D = _demo_transform(np.column_stack([age, sex]), demo)
    target, bg = groups == "AD", groups == "NC"
    pca = PCA(n_components=min(max(cfg.dim_grid), Z.shape[0] - 1, Z.shape[1]), random_state=seed).fit(Z)
    models = {}
    for model in MODEL_NAMES:
        choice = selected[model]
        pack = {"choice": choice, "transform": tr, "demo_transform": demo,
                "class_positive": "AD", "class_negative": "NC", "threshold": 0.5,
                "feature_names": list(names)}
        if model == "age_sex":
            A = D
        elif model == "age_adjusted_eeg":
            A = Z
        elif model == "age_sex_age_adjusted_eeg":
            A = np.column_stack([D, Z])
        elif model.endswith("_pca"):
            d = choice["dimension"]
            pack["pca_mean"] = pca.mean_.copy(); pack["pca_components"] = pca.components_[:d].T.copy()
            A = np.column_stack([D, pca.transform(Z)[:, :d]]) if model.startswith("age_sex") else pca.transform(Z)[:, :d]
        else:
            cp = cpca_fit(Z[target], Z[bg], choice["alpha"])
            d = choice["dimension"]
            V = cp["components"][:, :d].copy()
            for k in range(d):
                j = int(np.argmax(np.abs(V[:, k])))
                if V[j, k] < 0: V[:, k] *= -1
            pack.update({"cpca_components": V, "cpca_eigenvalues": cp["eigenvalues"][:d],
                         "cpca_alpha": choice["alpha"], "cpca_target_mean": cp["target_mean"],
                         "cpca_background_mean": cp["background_mean"]})
            S = Z @ V
            A = np.column_stack([D, S]) if model.startswith("age_sex") else S
        clf = fit_logistic(A, y, choice["C"], seed)
        pack["clf_coef"] = clf.coef_.ravel().copy(); pack["clf_intercept"] = float(clf.intercept_[0])
        pack["clf_classes"] = clf.classes_.copy()
        pack["train_decision"] = clf.decision_function(A)
        models[model] = pack
    return models


def predict_frozen(models: dict[str, Any], X, age, sex):
    out = {}
    for model, pack in models.items():
        tr = pack["transform"]
        if model == "age_sex":
            A = _demo_transform(np.column_stack([age, sex]), pack["demo_transform"])
            eeg_score = np.zeros(len(age))
        else:
            Z = _transform(X, age, sex, tr)
            if model == "age_adjusted_eeg":
                A = Z; eeg_score = Z @ pack["clf_coef"]
            elif model == "age_sex_age_adjusted_eeg":
                eeg_beta = pack["clf_coef"][2:]
                A = np.column_stack([_demo_transform(np.column_stack([age, sex]), pack["demo_transform"]), Z])
                eeg_score = Z @ eeg_beta
            elif model.endswith("_pca"):
                P = (Z - pack["pca_mean"]) @ pack["pca_components"]
                if model.startswith("age_sex"):
                    A = np.column_stack([_demo_transform(np.column_stack([age, sex]), pack["demo_transform"]), P])
                    eeg_beta = pack["clf_coef"][2:]
                else:
                    A = P; eeg_beta = pack["clf_coef"]
                eeg_score = P @ eeg_beta
            else:
                S = Z @ pack["cpca_components"]
                if model.startswith("age_sex"):
                    A = np.column_stack([_demo_transform(np.column_stack([age, sex]), pack["demo_transform"]), S])
                    eeg_beta = pack["clf_coef"][2:]
                else:
                    A = S; eeg_beta = pack["clf_coef"]
                eeg_score = S @ eeg_beta
        decision = A @ pack["clf_coef"] + pack["clf_intercept"]
        probability = 1 / (1 + np.exp(-np.clip(decision, -60, 60)))
        out[model] = {"probability_ad": probability, "decision": decision, "eeg_score": eeg_score}
    return out


def save_frozen_bundle(bundle_dir: Path, models: dict[str, Any], selected: dict[str, Any],
                       dev_ids: list[str], source_hashes: dict[str, str], cfg: ModelConfig,
                       versions: dict[str, str]):
    """Write a self-contained frozen bundle. Manifest creation is create-only."""
    bundle_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "freeze_id": "ADNC_" + hashlib.sha256(("|".join(sorted(dev_ids)) + json.dumps(selected, sort_keys=True)).encode()).hexdigest()[:12],
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "task": "AD=1 versus NC=0",
        "threshold": 0.5, "model_ids": list(models), "selected_parameters": selected,
        "selection_used_mmse": False, "development_subject_count": len(dev_ids),
        "development_subject_id_sha256": hashlib.sha256("\n".join(sorted(dev_ids)).encode()).hexdigest(),
        "source_sha256": source_hashes, "versions": versions,
        "configuration": dict(cfg.__dict__),
        "estimator": "LogisticRegression(class_weight=balanced, max_iter=2000); internal 3-fold nested selection",
        "feature_order": models["age_adjusted_eeg"]["feature_names"],
        "frequency_bands_hz": BANDS, "channels": CHANNELS,
        "cognitive_analysis_plan": "Primary cPC1; all other selected cPCs exploratory; AD-only and AD+NC Spearman; AD-only HC3 MMSE~cPC1+age+sex; pooled MMSE~cPC1+age+sex+diagnosis; BH within all component/model association tests.",
        "external_plan": "Use all eligible AD and NC; report missingness separately; never tune on external results; 2000 stratified subject bootstrap samples; paired AUC differences on the same bootstrap draws; 0.5 threshold.",
        "status": "frozen before external outcome evaluation",
    }
    for name, pack in models.items():
        metadata = {k: v for k, v in pack.items() if not isinstance(v, np.ndarray) and k not in {"transform", "demo_transform"}}
        tr = pack["transform"]; demo = pack["demo_transform"]
        metadata.update({"age_support_min": tr["age_support_min"], "age_support_max": tr["age_support_max"],
                         "robust_clip": tr["robust_clip"], "age_reference_n_nc": tr["n_nc_reference"]})
        arrays = {
            "feature_names": np.asarray(pack["feature_names"], dtype=str),
            "kept_names": np.asarray(tr["kept_names"], dtype=str),
            "keep": tr["keep"], "median": tr["median"], "qlo": tr["qlo"], "qhi": tr["qhi"],
            "center": tr["center"], "scale": tr["scale"], "block_weights": tr["block_weights"],
            "age_beta": tr["age_beta"], "demo_center": demo["center"], "demo_scale": demo["scale"],
            "clf_coef": pack["clf_coef"], "clf_classes": pack["clf_classes"],
            "pca_mean": pack.get("pca_mean", np.empty(0)),
            "pca_components": pack.get("pca_components", np.empty((0, 0))),
            "cpca_components": pack.get("cpca_components", np.empty((0, 0))),
            "cpca_eigenvalues": pack.get("cpca_eigenvalues", np.empty(0)),
        }
        arrays["clf_intercept"] = np.asarray([pack["clf_intercept"]], float)
        arrays["cpca_alpha"] = np.asarray([pack.get("cpca_alpha", np.nan)], float)
        arrays["cpca_target_mean"] = pack.get("cpca_target_mean", np.empty(0))
        arrays["cpca_background_mean"] = pack.get("cpca_background_mean", np.empty(0))
        np.savez_compressed(bundle_dir / f"{name}.npz", **arrays)
        (bundle_dir / f"{name}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(bundle_dir.iterdir()) if p.is_file() and p.name != "freeze_manifest.json"}
    manifest["bundle_file_sha256"] = hashes
    with (bundle_dir / "freeze_manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def load_frozen_bundle(bundle_dir: Path):
    manifest = json.loads((bundle_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["bundle_file_sha256"].items():
        if hashlib.sha256((bundle_dir / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen file fingerprint mismatch: {filename}")
    models = {}
    for name in manifest["model_ids"]:
        meta = json.loads((bundle_dir / f"{name}.json").read_text(encoding="utf-8"))
        with np.load(bundle_dir / f"{name}.npz", allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        tr = {"feature_names": arrays["feature_names"].astype(str).tolist(),
              "kept_names": arrays["kept_names"].astype(str).tolist(), "keep": arrays["keep"],
              "median": arrays["median"], "qlo": arrays["qlo"], "qhi": arrays["qhi"],
              "center": arrays["center"], "scale": arrays["scale"],
              "block_weights": arrays["block_weights"], "age_beta": arrays["age_beta"],
              "age_support_min": meta.get("age_support_min", np.nan), "age_support_max": meta.get("age_support_max", np.nan),
              "robust_clip": bool(meta.get("robust_clip", False))}
        pack = {**meta, "transform": tr, "feature_names": tr["feature_names"],
                "demo_transform": {"center": arrays["demo_center"], "scale": arrays["demo_scale"]},
                "clf_coef": arrays["clf_coef"], "clf_intercept": float(arrays["clf_intercept"][0]),
                "clf_classes": arrays["clf_classes"], "pca_mean": arrays["pca_mean"],
                "pca_components": arrays["pca_components"], "cpca_components": arrays["cpca_components"],
                "cpca_eigenvalues": arrays["cpca_eigenvalues"], "cpca_alpha": float(arrays["cpca_alpha"][0]),
                "cpca_target_mean": arrays["cpca_target_mean"], "cpca_background_mean": arrays["cpca_background_mean"]}
        models[name] = pack
    return manifest, models

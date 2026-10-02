"""DAY 2 regression: select on Batch 1, freeze, then evaluate Batch 2.

The public functions support the same stages in the modeling notebook and CLI.
No Batch 2 labels enter fitting or hyperparameter tuning. Final family selection uses Batch 2 MAPE by user request. Batch 3 is never
used in this experiment. Run from the project root: ``python -m src.train``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

SEED = 42
MODEL_THREADS = 1
OUTER_FOLDS = 5
INNER_FOLDS = 3
CORE_FEATURES = ["delta_logvar", "policy_C1", "policy_switch_SOC_pct", "policy_C2"]
PAPER_HEADLINE_MAPE = 9.1
PAPER_ABSTRACT_URL = "https://www.nature.com/articles/s41560-019-0356-8"
PAPER_PDF_URL = "https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainingData:
    """Only training targets are exposed before selection is frozen."""

    input_dir: Path
    X_train: pd.DataFrame
    y_train: pd.Series
    meta_train: pd.DataFrame
    X_train_sens: pd.DataFrame
    y_train_sens: pd.Series
    meta_train_sens: pd.DataFrame
    test_ids: pd.Index
    development_ids: pd.Index
    holdout_ids: pd.Index


@dataclass
class EvaluationResults:
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_test: pd.DataFrame
    test_scores: pd.DataFrame
    test_predictions: pd.DataFrame
    sensitivity_test_scores: pd.DataFrame
    sensitivity_test_predictions: pd.DataFrame
    test_table: pd.DataFrame
    selected_predictions: pd.DataFrame
    selected_score: pd.Series
    prediction_bias: float
    overpredicted_n: int


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="sample_id")
    _check(frame.index.is_unique, f"Duplicate sample_id in {path}")
    return frame


def load_training_data(input_dir: str | Path | None = None) -> TrainingData:
    """Load aligned FE CSVs and expose Batch 1 targets, leaving test evaluation later."""
    folder = Path(input_dir).resolve() if input_dir else PROJECT_ROOT / "data" / "processed"
    X = _read_indexed(folder / "X.csv")
    meta = _read_indexed(folder / "metadata.csv")
    _check(X.index.equals(meta.index), "X.csv and metadata.csv sample order differs")
    _check(X.columns.tolist() == CORE_FEATURES, "Expected the four frozen CORE_FEATURES in order")
    _check(bool(np.isfinite(X.to_numpy()).all()), "Nonfinite main features")
    train_ids = meta.index[meta.batch_id.eq(1)]
    test_ids = meta.index[meta.batch_id.eq(2)]
    _check(len(train_ids) == 46 and len(test_ids) == 39, "Expected Batch 1=46 / Batch 2=39 FE samples")
    _check(not set(train_ids) & set(test_ids), "Train/test IDs overlap")
    X_train, meta_train = X.loc[train_ids].copy(), meta.loc[train_ids].copy()
    target = _read_indexed(folder / "y.csv")
    _check(X.index.equals(target.index), "X.csv and y.csv sample order differs")
    y_train = target.loc[train_ids, "cycle_life"].copy()
    _check(bool(np.isfinite(y_train).all() and (y_train > 100).all()), "Invalid Batch 1 target")
    _check(not bool(meta_train.author_quality_flag.any()), "Training cohort contains excluded quality cells")
    sens_ids = train_ids[~meta_train.endpoint_review_candidate.to_numpy(dtype=bool)]
    _check(len(sens_ids) == 36, "Expected 36 training rows after endpoint sensitivity exclusions")
    groups = protocol_groups(meta_train)
    di, hi = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(X_train, groups=groups))
    development_ids, holdout_ids = train_ids[di], train_ids[hi]
    _check(not set(groups.iloc[di]) & set(groups.iloc[hi]), "Hold-out policy leakage")
    _check(not set(development_ids) & set(holdout_ids), "Hold-out sample leakage")
    return TrainingData(folder, X_train, y_train, meta_train, X_train.loc[sens_ids].copy(),
                        y_train.loc[sens_ids].copy(), meta_train.loc[sens_ids].copy(), test_ids.copy(), development_ids, holdout_ids)


def protocol_groups(metadata: pd.DataFrame) -> pd.Series:
    """Group identical nominal C1/SOC/C2 policies, including newstructure suffixes."""
    return metadata.charging_policy.str.replace(r"-newstructure$", "", regex=True)


def model_specs() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Fresh estimators and the established bounded hyperparameter grids."""
    ridge = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler()), ("model", Ridge(solver="svd"))])
    forest = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("model", RandomForestRegressor(n_estimators=200, random_state=SEED,
                                                       n_jobs=MODEL_THREADS, max_features=1.0))])
    boost = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("model", XGBRegressor(objective="reg:squarederror", n_estimators=250,
                                              learning_rate=.04, reg_lambda=5, reg_alpha=0,
                                              subsample=1, colsample_bytree=1, tree_method="hist",
                                              random_state=SEED, n_jobs=MODEL_THREADS, verbosity=0))])
    return {
        "Ridge": (TransformedTargetRegressor(regressor=ridge, func=np.log, inverse_func=np.exp),
                  {"regressor__model__alpha": [.01, .1, 1, 10, 100]}),
        "Random Forest": (TransformedTargetRegressor(regressor=forest, func=np.log, inverse_func=np.exp),
                          {"regressor__model__max_depth": [3, 5, None],
                           "regressor__model__min_samples_leaf": [2, 5]}),
        "XGBoost": (TransformedTargetRegressor(regressor=boost, func=np.log, inverse_func=np.exp),
                    {"regressor__model__max_depth": [1, 2, 3],
                     "regressor__model__min_child_weight": [3, 6]}),
    }


def candidate_grid_table() -> pd.DataFrame:
    return pd.DataFrame([{"model": name, "configurations": len(ParameterGrid(grid)),
                          "inner_folds": INNER_FOLDS, "search_grid": json.dumps(grid)}
                         for name, (_, grid) in model_specs().items()])


def metrics(actual: Any, predicted: Any) -> dict[str, float]:
    """Metrics in original cycle units after ln/exp inverse transformation."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    _check(a.ndim == p.ndim == 1 and len(a) == len(p) and len(a) > 0, "Invalid metric array shape")
    _check(bool(np.isfinite(a).all() and (a > 0).all()), "Targets must be finite and positive")
    _check(bool(np.isfinite(p).all() and (p > 0).all()), "Predictions must be finite and positive")
    return {"MAE": float(mean_absolute_error(a, p)), "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "R2": float(r2_score(a, p)), "MAPE_pct": float(100 * np.mean(np.abs(a - p) / a))}


def grouped_splits(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                   n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    _check(groups.nunique() >= n_splits, "Insufficient distinct policies for GroupKFold")
    result = list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(X, y, groups))
    for train_idx, valid_idx in result:
        _check(not set(X.index[train_idx]) & set(X.index[valid_idx]), "CV sample leakage")
        _check(not set(groups.iloc[train_idx]) & set(groups.iloc[valid_idx]), "CV policy leakage")
    return result


def _audit_row(experiment: str, stage: str, outer: int, inner: int,
               train_ids: pd.Index, valid_ids: pd.Index) -> dict[str, Any]:
    return dict(experiment=experiment, stage=stage, outer_fold=outer, inner_fold=inner,
                train_ids=json.dumps(train_ids.tolist()), validation_ids=json.dumps(valid_ids.tolist()),
                train_batch=1, validation_batch=1, sample_overlap=0, policy_overlap=0)


def _search(estimator: Any, grid: dict, splits: list, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    search = GridSearchCV(estimator, grid, scoring="neg_mean_absolute_error", cv=splits,
                          refit=True, n_jobs=1, error_score="raise", return_train_score=False)
    with threadpool_limits(limits=MODEL_THREADS):
        search.fit(X, y)
    return search


def _tuning_rows(search: GridSearchCV, experiment: str, stage: str, fold: int, name: str) -> list[dict]:
    return [dict(experiment=experiment, stage=stage, fold=fold, model=name, params=json.dumps(row["params"]),
                 CV_MAE=-row["mean_test_score"], CV_MAE_std=row["std_test_score"],
                 rank=int(row["rank_test_score"]))
            for row in pd.DataFrame(search.cv_results_).to_dict("records")]


def select_on_batch1(X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame,
                     experiment: str = "main_46") -> dict[str, Any]:
    """Select family via nested Policy GroupKFold outer5/inner3, without test data.

    Development tuning, Hold-out evaluation and fixed-parameter full-B1 refit happen separately.
    """
    _check(bool(metadata.batch_id.eq(1).all()), "Only Batch 1 may enter selection")
    _check(X.index.equals(y.index) and X.index.equals(metadata.index), "Training inputs misaligned")
    _check(X.columns.tolist() == CORE_FEATURES, "Training feature order differs")
    groups = protocol_groups(metadata)
    folds, predictions, tuning, audit = [], [], [], []
    outer = grouped_splits(X, y, groups, OUTER_FOLDS)
    for fold, (ti, vi) in enumerate(outer, 1):
        inner = grouped_splits(X.iloc[ti], y.iloc[ti], groups.iloc[ti], INNER_FOLDS)
        audit.append(_audit_row(experiment, "outer", fold, 0, X.index[ti], X.index[vi]))
        for inner_fold, (it, iv) in enumerate(inner, 1):
            audit.append(_audit_row(experiment, "inner", fold, inner_fold, X.index[ti[it]], X.index[ti[iv]]))
        for name, (estimator, grid) in model_specs().items():
            search = _search(estimator, grid, inner, X.iloc[ti], y.iloc[ti])
            with threadpool_limits(limits=MODEL_THREADS):
                pred = search.predict(X.iloc[vi])
            folds.append(dict(experiment=experiment, fold=fold, model=name, train_n=len(ti), validation_n=len(vi),
                              inner_MAE=-search.best_score_, best_params=json.dumps(search.best_params_),
                              **metrics(y.iloc[vi], pred)))
            predictions.extend(dict(experiment=experiment, fold=fold, model=name, sample_id=sid,
                                    batch_id=1, actual=actual, predicted=prediction)
                               for sid, actual, prediction in zip(X.index[vi], y.iloc[vi], pred))
            tuning.extend(_tuning_rows(search, experiment, "outer_inner", fold, name))
    fold_frame, pred_frame = pd.DataFrame(folds), pd.DataFrame(predictions)
    summary_rows = []
    for simplicity, name in enumerate(model_specs()):
        scores, oof = fold_frame.loc[fold_frame.model.eq(name)], pred_frame.loc[pred_frame.model.eq(name)]
        _check(len(oof) == len(X) and oof.sample_id.nunique() == len(X), "OOF coverage incomplete")
        summary_rows.append(dict(model=name, CV_MAE_mean=scores.MAE.mean(), CV_MAE_std=scores.MAE.std(ddof=1),
                                 CV_RMSE_mean=scores.RMSE.mean(), CV_R2_mean=scores.R2.mean(),
                                 CV_MAPE_mean=scores.MAPE_pct.mean(), simplicity=simplicity,
                                 **{"OOF_" + key: value for key, value in metrics(oof.actual, oof.predicted).items()}))
    summary = pd.DataFrame(summary_rows).sort_values(["CV_MAE_mean", "CV_RMSE_mean", "simplicity"]).reset_index(drop=True)
    return dict(selected=summary.iloc[0].model, summary=summary, folds=fold_frame, predictions=pred_frame,
                tuning=pd.DataFrame(tuning), audit=pd.DataFrame(audit), experiment=experiment,
                training_sample_ids=X.index.tolist(), models={})


def _fit_final_candidates(result: dict, X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame) -> None:
    _check(result["training_sample_ids"] == X.index.tolist(), "Selection and final fit cohorts differ")
    _check(bool(metadata.batch_id.eq(1).all()), "Only Batch 1 may enter final fit")
    splits = grouped_splits(X, y, protocol_groups(metadata), OUTER_FOLDS)
    experiment = result["experiment"]
    audit = [_audit_row(experiment, "final_tuning", fold, 0, X.index[ti], X.index[vi])
             for fold, (ti, vi) in enumerate(splits, 1)]
    final, tuning, fitted = [], [], {}
    for name, (estimator, grid) in model_specs().items():
        search = _search(estimator, grid, splits, X, y)
        with threadpool_limits(limits=MODEL_THREADS):
            train_pred = search.predict(X)
        fitted[name] = search.best_estimator_
        final.append(dict(model=name, tuning_CV_MAE=-search.best_score_,
                          train_MAE=mean_absolute_error(y, train_pred), best_params=json.dumps(search.best_params_)))
        tuning.extend(_tuning_rows(search, experiment, "final_tuning", 0, name))
    result["models"], result["final_tuning"] = fitted, pd.DataFrame(final)
    result["tuning"] = pd.concat([result["tuning"], pd.DataFrame(tuning)], ignore_index=True)
    result["audit"] = pd.concat([result["audit"], pd.DataFrame(audit)], ignore_index=True)


def fit_and_lock_selection(data: TrainingData, main: dict, sensitivity: dict,
                           output_dir: str | Path | None = None) -> dict:
    """Freeze development-only selection, evaluate Hold-out, then refit fixed params on B1."""
    out = Path(output_dir).resolve() if output_dir else PROJECT_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    dev = data.development_ids
    sens_dev = data.X_train_sens.index.intersection(dev, sort=False)
    for result, ids in [(main, dev), (sensitivity, sens_dev)]:
        _check(result["training_sample_ids"] == ids.tolist(), "Development selection cohort differs")
        _check(not set(result["training_sample_ids"]) & set(data.holdout_ids), "Hold-out entered selection")
        _fit_final_candidates(result, data.X_train.loc[ids], data.y_train.loc[ids], data.meta_train.loc[ids])
    lock = dict(selected_model=main["selected"], training_batch=1, training_rows=len(data.X_train),
                training_sample_ids=data.X_train.index.tolist(), feature_columns=CORE_FEATURES.copy(),
                development_rows=len(dev), development_sample_ids=dev.tolist(),
                holdout_rows=len(data.holdout_ids), holdout_sample_ids=data.holdout_ids.tolist(),
                holdout_split="GroupShuffleSplit(test_size=0.2, random_state=42), nominal policy",
                selection_rule="Minimum development-only nested GroupKFold mean MAE; ties RMSE then simplicity",
                final_hyperparameters=json.loads(main["final_tuning"].set_index("model").loc[main["selected"], "best_params"]),
                holdout_y_used_for_selection=False, holdout_y_used_for_tuning=False,
                test_y_used_for_selection=False, test_y_used_for_fit=False, batch3_used=False,
                sensitivity_selected_model=sensitivity["selected"],
                sensitivity_development_sample_ids=sens_dev.tolist(),
                sensitivity_training_sample_ids=data.X_train_sens.index.tolist(),
                final_refit="Same family and hyperparameters refit on full Batch 1 before Batch 2 evaluation")
    # Persist the family/parameter decision BEFORE reading Hold-out scores.
    (out / "selection_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result, full_ids in [(main, data.X_train.index), (sensitivity, data.X_train_sens.index)]:
        hi = data.holdout_ids.intersection(full_ids, sort=False)
        scores, predictions = evaluate_fitted(result, data.X_train.loc[hi], data.y_train.loc[hi],
                                               result["experiment"] + "_holdout", batch_id=1)
        result["holdout_scores"], result["holdout_predictions"] = scores, predictions
        result["final_fit_sample_ids"] = full_ids.tolist()
        frozen = result["final_tuning"].set_index("model").best_params
        for name, fitted in list(result["models"].items()):
            joblib.dump(fitted, out / f"{result['experiment']}_{name.lower().replace(' ', '_')}_development.joblib")
            model = clone(fitted)
            params = json.loads(frozen.loc[name])
            _check(all(model.get_params()[k] == v for k, v in params.items()), "Frozen params changed")
            with threadpool_limits(limits=MODEL_THREADS):
                model.fit(data.X_train.loc[full_ids], data.y_train.loc[full_ids])
            result["models"][name] = model
    joblib.dump(main["models"][main["selected"]], out / "final_model.joblib")
    return lock


def evaluate_fitted(result: dict, X_test: pd.DataFrame, y_test: pd.Series,
                     experiment: str, batch_id: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, predictions = [], []
    for name, model in result["models"].items():
        with threadpool_limits(limits=MODEL_THREADS):
            pred = model.predict(X_test)
        rows.append(dict(experiment=experiment, model=name, selected_before_test=name == result["selected"],
                         train_batch=1, test_batch=batch_id, test_n=len(y_test), **metrics(y_test, pred)))
        predictions.extend(dict(experiment=experiment, model=name, sample_id=sid, batch_id=batch_id,
                                actual=actual, predicted=prediction, residual=actual - prediction,
                                abs_error=abs(actual - prediction), APE_pct=100 * abs(actual - prediction) / actual)
                           for sid, actual, prediction in zip(X_test.index, y_test, pred))
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def evaluate_batch2(data: TrainingData, main: dict, sensitivity: dict,
                    output_dir: str | Path | None = None) -> EvaluationResults:
    """Load test targets only after validating the persisted selection lock."""
    out = Path(output_dir).resolve() if output_dir else PROJECT_ROOT / "results"
    lock = json.loads((out / "selection_lock.json").read_text(encoding="utf-8"))
    _check(lock["selected_model"] == main["selected"] and lock["training_sample_ids"] == data.X_train.index.tolist(),
           "Pre-test main selection lock differs")
    _check(lock["sensitivity_selected_model"] == sensitivity["selected"], "Sensitivity selection lock differs")
    X_test = _read_indexed(data.input_dir / "X.csv").loc[data.test_ids, CORE_FEATURES].copy()
    meta_test = _read_indexed(data.input_dir / "metadata.csv").loc[data.test_ids].copy()
    y_test = _read_indexed(data.input_dir / "y.csv").loc[data.test_ids, "cycle_life"].copy()
    _check(y_test.index.equals(X_test.index), "Test inputs misaligned")
    _check(bool(np.isfinite(y_test).all() and (y_test > 100).all()), "Invalid Batch 2 target")
    _check(bool(meta_test.batch_id.eq(2).all()), "Test data must contain Batch 2 only")
    _check(not bool(meta_test.author_quality_flag.any()), "Test cohort contains excluded quality cells")
    _check(not set(data.X_train.index) & set(X_test.index), "Train/test IDs overlap")
    test_scores, predictions = evaluate_fitted(main, X_test, y_test, "main_46")
    sens_scores, sens_predictions = evaluate_fitted(sensitivity, X_test, y_test, "sensitivity_36")
    test_table = test_scores.copy()
    selected_pred = predictions.loc[predictions.model.eq(main["selected"])].set_index("sample_id")
    _check(selected_pred.index.equals(X_test.index), "Selected prediction order differs")
    return EvaluationResults(X_test, y_test, meta_test, test_scores, predictions, sens_scores, sens_predictions,
                             test_table, selected_pred, test_scores.set_index("model").loc[main["selected"]],
                             float((selected_pred.predicted - selected_pred.actual).mean()),
                             int((selected_pred.predicted > selected_pred.actual).sum()))


def select_by_test_mape(main: dict, evaluation: EvaluationResults,
                        output_dir: str | Path | None = None) -> dict:
    """Requested post-test family selection; Batch 2 is now selection data, not independent testing."""
    out = Path(output_dir).resolve() if output_dir else PROJECT_ROOT / "results"
    lock = json.loads((out / "selection_lock.json").read_text(encoding="utf-8"))
    name = evaluation.test_scores.sort_values(["MAPE_pct", "MAE", "model"]).iloc[0].model
    lock["development_CV_selected_model"] = lock["selected_model"]
    lock["selected_model"] = main["selected"] = name
    lock["selection_rule"] = "Minimum Batch 2 test MAPE; ties MAE then model name"
    lock["test_y_used_for_selection"] = True
    lock["batch2_independent_final_test"] = False
    lock["final_hyperparameters"] = json.loads(main["final_tuning"].set_index("model").loc[name, "best_params"])
    for table in [evaluation.test_scores, evaluation.test_table]:
        table["selected_final"] = table.model.eq(name)
    evaluation.selected_predictions = evaluation.test_predictions.loc[
        evaluation.test_predictions.model.eq(name)].set_index("sample_id")
    evaluation.selected_score = evaluation.test_scores.set_index("model").loc[name]
    pred = evaluation.selected_predictions
    evaluation.prediction_bias = float((pred.predicted-pred.actual).mean())
    evaluation.overpredicted_n = int((pred.predicted>pred.actual).sum())
    (out / "selection_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    joblib.dump(main["models"][name], out / "final_model.joblib")
    return lock


def diagnose_domain_shift(data: TrainingData, evaluation: EvaluationResults) -> dict[str, pd.DataFrame]:
    """Post-test descriptive diagnostics; they never change the selected model."""
    Xt, Xv = data.X_train, evaluation.X_test
    train_policies = set(protocol_groups(data.meta_train))
    test_policies = protocol_groups(evaluation.meta_test)
    seen = test_policies.isin(train_policies)
    outside = pd.DataFrame({f: (Xv[f] < Xt[f].min()) | (Xv[f] > Xt[f].max()) for f in CORE_FEATURES}, index=Xv.index)
    range_audit = pd.DataFrame([dict(feature=f, train_min=Xt[f].min(), train_max=Xt[f].max(),
                                    test_min=Xv[f].min(), test_max=Xv[f].max(),
                                    test_outside_train_range=int(outside[f].sum())) for f in CORE_FEATURES])
    shift = pd.DataFrame([
        dict(batch=1, role="train", n=len(Xt), nominal_policies=len(train_policies), life_mean=data.y_train.mean(),
             life_min=data.y_train.min(), life_max=data.y_train.max()),
        dict(batch=2, role="test", n=len(Xv), nominal_policies=test_policies.nunique(), life_mean=evaluation.y_test.mean(),
             life_min=evaluation.y_test.min(), life_max=evaluation.y_test.max())])
    strata = []
    for label, mask in [("Seen policy", seen), ("Unseen policy", ~seen),
                        ("Inside all training feature ranges", ~outside.any(axis=1)),
                        ("Outside >=1 training feature range", outside.any(axis=1))]:
        if mask.sum() >= 2:
            frame = evaluation.selected_predictions.loc[mask]
            strata.append(dict(group=label, n=len(frame), **metrics(frame.actual, frame.predicted)))
    return {"feature_range_audit": range_audit, "batch_shift": shift,
            "stratified_test_scores": pd.DataFrame(strata), "outside_training_ranges": outside,
            "seen_policy": seen.to_frame("seen_policy")}


def paper_comparison(evaluation: EvaluationResults) -> pd.DataFrame:
    """Published reference values, with different cohorts/splits clearly identified."""
    rows = [("Abstract", "대표 test error", "Headline", np.nan, 9.1),
            ("Table 1", "Primary (이상 셀 포함)", "Variance", 138, 14.7),
            ("Table 1", "Primary (이상 셀 포함)", "Discharge", 91, 13.0),
            ("Table 1", "Primary (이상 셀 포함)", "Full", 118, 14.1),
            ("Table 1", "Secondary", "Variance", 196, 11.4),
            ("Table 1", "Secondary", "Discharge", 173, 8.6),
            ("Table 1", "Secondary", "Full", 214, 10.7)]
    frame = pd.DataFrame(rows, columns=["source", "split", "model", "RMSE", "MAPE_pct"])
    frame["our_MAPE_minus_reference_pp"] = evaluation.selected_score.MAPE_pct - frame.MAPE_pct
    frame["source_url"] = np.where(frame.source.eq("Abstract"), PAPER_ABSTRACT_URL, PAPER_PDF_URL)
    return frame


def sensitivity_comparison(main: dict, sensitivity: dict, evaluation: EvaluationResults) -> pd.DataFrame:
    main_row = evaluation.test_scores.loc[evaluation.test_scores.model.eq(main["selected"])].copy()
    main_row["train_n"] = len(main["final_fit_sample_ids"])
    sens_row = evaluation.sensitivity_test_scores.loc[
        evaluation.sensitivity_test_scores.model.eq(sensitivity["selected"])].copy()
    sens_row["train_n"] = len(sensitivity["final_fit_sample_ids"])
    return pd.concat([main_row, sens_row], ignore_index=True)


def validation_test_summary(main: dict, evaluation: EvaluationResults) -> pd.DataFrame:
    cv = float(main["summary"].set_index("model").loc[main["selected"], "CV_MAPE_mean"])
    valid = float(main["holdout_scores"].set_index("model").loc[main["selected"], "MAPE_pct"])
    test = float(evaluation.selected_score.MAPE_pct)
    return pd.DataFrame([
        ("Train (Batch 1 CV)", cv, "%"), ("Valid (Batch 1 Hold-out)", valid, "%"),
        ("Test (Batch 2)", test, "%"), ("Gap (Train-Valid)", valid-cv, "%p"),
        ("Gap (Valid-Test)", test-valid, "%p"), ("Gap (Target-Test)", test-PAPER_HEADLINE_MAPE, "%p")
    ], columns=["stage", "MAPE_or_gap", "unit"])


def build_report(data: TrainingData, main: dict, sensitivity: dict, evaluation: EvaluationResults,
                 diagnostics: dict, lock: dict) -> str:
    score, winner = evaluation.selected_score, main["summary"].set_index("model").loc[main["selected"]]
    sens = sensitivity_comparison(main, sensitivity, evaluation).iloc[1]
    unseen = int((~diagnostics["seen_policy"].seen_policy).sum())
    delta_out = int(diagnostics["outside_training_ranges"].delta_logvar.sum())
    gap = score.MAPE_pct - PAPER_HEADLINE_MAPE
    status = "수치상 달성" if gap <= 0 else "미달"
    return f"""# DAY 2 모델 개발 및 평가

1. **문제:** 초기 100사이클 → 제공 cycle_life 회귀. X: {', '.join(CORE_FEATURES)}.
2. **분할:** Batch 1 학습 {len(data.X_train)}개 / Batch 2 테스트 {len(evaluation.X_test)}개. Batch 3 미사용.
3. **최종 선정:** 세 후보의 Batch 2 MAPE 최소 → **{main['selected']}**. 개발 CV 우승 모델은 {lock["development_CV_selected_model"]}.
   Batch 2를 선정에 사용했으므로 별도의 독립 최종 테스트는 아직 없음.
   CV MAE {winner.CV_MAE_mean:.2f} ± {winner.CV_MAE_std:.2f} cycles (fold 표준편차).
   최종 파라미터는 Hold-out 제외 개발 구간 그룹5-fold에서 결정: {lock['final_hyperparameters']}.
   Pipeline 안에서 fold별 전처리, ln/exp 타깃 변환. 파라미터 튜닝·fit은 Batch 1만 사용; 최종 모델 종류는 Batch 2 MAPE로 선정.
4. **Batch 2:** MAE **{score.MAE:.2f} cycles**, RMSE **{score.RMSE:.2f} cycles**,
   R² **{score.R2:.3f}**, MAPE **{score.MAPE_pct:.2f}%**.
   {len(evaluation.X_test)}개 중 {evaluation.overpredicted_n}개 과대 예측;
   평균 편향(예측−실제) {evaluation.prediction_bias:+.2f} cycles.
5. **배치 차이:** 신규 명목 Policy {unseen}/{len(evaluation.X_test)}개;
   delta_logvar 학습 범위 밖 {delta_out}개. 평균 수명 Batch 1 {data.y_train.mean():.1f} →
   Batch 2 {evaluation.y_test.mean():.1f} cycles. 특정 조건이 오차의 주원인이라고 단정하지 않음.
6. **라벨 민감도:** 점검 후보10개 제외, Hold-out을 뺀 개발 {len(sensitivity["training_sample_ids"])}개에서 선정하고
   전체 Batch 1 36개로 같은 파라미터를 재학습한
   {sensitivity['selected']}, 같은 Batch 2 MAPE {sens.MAPE_pct:.2f}%. 주 실험 선정 유지.
7. **논문 참고 목표:** 초록 9.1% 대비 **{gap:+.2f}%p**, {status}.
   MAPE = 100 × mean(abs(actual−predicted)/actual).
   [논문 초록]({PAPER_ABSTRACT_URL}), [Table 1·Methods]({PAPER_PDF_URL}).
   논문 pooled train41/primary43/secondary40과 본 과제 B1→B2는 다르고 B2 파일 날짜도 다름.
   동일 조건 재현 또는 논문 모델 대비 우열의 증거로 해석하지 않음.
   Table 1 primary 값은 이상 셀 포함; 괄호의 한 셀 제외값과 구분함.
   Full 모델은 온도 센서 문제로 별도 제외한 셀도 있음. 평가 오차를 보고 셀을 제외하지 않음.
   논문 EOL 정의는 정격1.1Ah의80%(0.88Ah); 본 실험은 제공 cycle_life 라벨 사용.
8. **최종 모델:** final_model.joblib은 Batch 1 46개로만 fit. 평가 후 Batch 2/3 재학습 없음.
   이전 EDA·검증의 Batch 2 노출 이력이 있어 신규 블라인드 테스트는 아님.

## 검증·테스트 요약

{validation_test_summary(main, evaluation).to_string(index=False)}

Hold-out은 개발 모델로 평가하고, Batch 2는 같은 파라미터로 전체 Batch 1 재학습한 모델로 평가한다.

## Batch 2 점수

{evaluation.test_table.to_string(index=False)}

## 논문 참고 성능

{paper_comparison(evaluation).drop(columns='source_url').to_string(index=False)}

## 다음 실험

모델 개선은 Batch 1 내부에서 사전에 계획한 방식으로 검증하고 새로운 평가 배치에서 확인한다.
제공 수명 라벨의 종단 완결성도 점검할 필요가 있다.
"""


def save_results(data: TrainingData, main: dict, sensitivity: dict, evaluation: EvaluationResults,
                  diagnostics: dict | None = None, output_dir: str | Path | None = None) -> dict:
    """Save scores, CV evidence, frozen model and provenance without refitting."""
    out = Path(output_dir).resolve() if output_dir else PROJECT_ROOT / "results"
    lock = json.loads((out / "selection_lock.json").read_text(encoding="utf-8"))
    _check(lock["selected_model"] == main["selected"], "Selected model changed after test")
    diagnostics = diagnostics if diagnostics is not None else diagnose_domain_shift(data, evaluation)
    for prefix, result in [("main", main), ("sensitivity", sensitivity)]:
        for key in ["summary", "folds", "predictions", "final_tuning", "tuning", "audit", "holdout_scores", "holdout_predictions"]:
            result[key].to_csv(out / f"{prefix}_{key}.csv", index=False)
        for name, model in result["models"].items():
            joblib.dump(model, out / f"{prefix}_{name.lower().replace(' ', '_')}.joblib")
    frames = {"test_scores": evaluation.test_table, "test_predictions": evaluation.test_predictions,
              "sensitivity_test_scores": evaluation.sensitivity_test_scores,
              "sensitivity_test_predictions": evaluation.sensitivity_test_predictions,
              "sensitivity_comparison": sensitivity_comparison(main, sensitivity, evaluation),
              "paper_reference": paper_comparison(evaluation),
              **{key: diagnostics[key] for key in ["feature_range_audit", "batch_shift", "stratified_test_scores"]}}
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    validation_summary = validation_test_summary(main, evaluation)
    validation_summary.to_csv(out / "validation_test_summary.csv", index=False)
    pd.DataFrame({"sample_id": data.X_train.index,
                  "role": ["development" if sid in set(data.development_ids) else "holdout" for sid in data.X_train.index],
                  "nominal_policy": protocol_groups(data.meta_train).to_numpy()}).to_csv(out / "batch1_holdout_split.csv", index=False)
    performance = evaluation.test_table.copy()
    performance.insert(4, "train_n", len(data.X_train))
    performance["selected"] = performance.model.eq(main["selected"])
    performance.to_csv(out / "model_performance.csv", index=False)
    evaluation.selected_predictions.to_csv(out / "selected_test_predictions.csv")
    pd.DataFrame({"sample_id": list(data.X_train.index) + list(data.test_ids),
                  "batch_id": [1]*len(data.X_train) + [2]*len(data.test_ids),
                  "role": ["train"]*len(data.X_train) + ["test"]*len(data.test_ids)}).to_csv(out / "train_test_split.csv", index=False)
    source_manifest_path = data.input_dir / "manifest.json"
    sources = json.loads(source_manifest_path.read_text(encoding="utf-8")).get("sources", {}) if source_manifest_path.is_file() else {}
    hashes = {}
    for path in [data.input_dir / name for name in ["X.csv", "y.csv", "metadata.csv", "manifest.json"]]:
        if path.is_file():
            label = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
            hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    score = evaluation.selected_score
    manifest = dict(**lock, target="cycle_life", internal_target_transform="ln/exp", feature_window=[10, 100],
                    test_batch=2, test_rows=len(data.test_ids), test_sample_ids=data.test_ids.tolist(),
                    selected_holdout_metrics={key: float(main["holdout_scores"].set_index("model").loc[main["selected"], key]) for key in ["MAE", "RMSE", "R2", "MAPE_pct"]},
                    selected_test_metrics={key: float(score[key]) for key in ["MAE", "RMSE", "R2", "MAPE_pct"]},
                    final_model_refit_after_test=False, training_rows_by_batch={"1": len(data.X_train)},
                    nested_CV={"outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, "group": "nominal charging policy"},
                    final_tuning_folds=OUTER_FOLDS, random_seed=SEED,
                    sensitivity={"train_rows": len(data.X_train_sens), "train_ids": data.X_train_sens.index.tolist(),
                                 "selected_model": sensitivity["selected"], "test_rows": len(data.test_ids)},
                    paper_headline_MAPE_pct=PAPER_HEADLINE_MAPE,
                    difference_to_paper_headline_pp=float(score.MAPE_pct - PAPER_HEADLINE_MAPE),
                    paper_comparison="Reference only; different split, source files, features and label cleanup",
                    paper_sources={"abstract": PAPER_ABSTRACT_URL, "table1_methods": PAPER_PDF_URL},
                    prior_EDA_and_validation_exposure_to_test=True,
                    source_files={key: value for key, value in sources.items() if key in {"1", "2"}},
                    prediction_bias_cycles=evaluation.prediction_bias, overpredicted_test_rows=evaluation.overpredicted_n,
                    packages={"python": platform.python_version(), "sklearn": sklearn.__version__,
                              "xgboost": xgboost.__version__, "numpy": np.__version__,
                              "pandas": pd.__version__, "joblib": joblib.__version__}, input_sha256=hashes)
    _check(manifest["training_rows_by_batch"] == {"1": 46}, "Final main training cohort changed")
    _check(not set(manifest["training_sample_ids"]) & set(manifest["test_sample_ids"]), "Saved split overlap")
    (out / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(build_report(data, main, sensitivity, evaluation, diagnostics, lock) + "\n", encoding="utf-8")
    with threadpool_limits(limits=MODEL_THREADS):
        np.testing.assert_allclose(joblib.load(out / "final_model.joblib").predict(evaluation.X_test),
                                   evaluation.selected_predictions.predicted.to_numpy())
    return manifest


def run_training(input_dir: str | Path | None = None, output_dir: str | Path | None = None) -> dict:
    """End-to-end CLI/notebook workflow using genuine reusable Python functions."""
    data = load_training_data(input_dir)
    ids = data.development_ids
    main = select_on_batch1(data.X_train.loc[ids], data.y_train.loc[ids], data.meta_train.loc[ids], "main_development")
    ids = data.X_train_sens.index.intersection(data.development_ids, sort=False)
    sensitivity = select_on_batch1(data.X_train_sens.loc[ids], data.y_train_sens.loc[ids], data.meta_train_sens.loc[ids], "sensitivity_development")
    lock = fit_and_lock_selection(data, main, sensitivity, output_dir)
    evaluation = evaluate_batch2(data, main, sensitivity, output_dir)
    lock = select_by_test_mape(main, evaluation, output_dir)
    diagnostics = diagnose_domain_shift(data, evaluation)
    manifest = save_results(data, main, sensitivity, evaluation, diagnostics, output_dir)
    return {"data": data, "main": main, "sensitivity": sensitivity, "selection_lock": lock,
            "evaluation": evaluation, "diagnostics": diagnostics, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch 1 train / Batch 2 test regression")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    result = run_training(args.input_dir, args.output_dir)
    print(result["evaluation"].test_table.to_string(index=False))
    print(f"Selected by Batch 2 MAPE: {result['main']['selected']}")
    print(f"Saved: {args.output_dir.resolve() / 'model_performance.csv'}")


if __name__ == "__main__":
    main()

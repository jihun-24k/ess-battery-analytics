"""Leakage-aware early-cycle features and reproducible feature tables.

The default inputs are ΔQ log variance and three parsed charging conditions.
Lifetime, endpoint diagnostics, cell IDs, and post-cycle-100 measurements are
kept out of every X variant. No imputer, scaler, or encoder is fitted here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import json
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd

from .preprocess import (
    BATCH3_QUALITY_IDS, EARLY_END, EARLY_START, SUMMARY_COLUMNS,
    clean_early_summary, decode_policy, delta_q_features, discover_project_root,
    endpoint_values, get_batch_paths, read_cell_field, read_early_current_stats,
    summarize_early_summary, validate_cycle_labels, vif_table,
)

POLICY_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)C\((\d+(?:\.\d+)?)%\)-(\d+(?:\.\d+)?)C(-newstructure)?$")
POLICY_NUMERIC = ["policy_C1", "policy_switch_SOC_pct", "policy_C2"]
CORE = ["delta_logvar"] + POLICY_NUMERIC
ADDITIONAL = ["early_fade_rate", "mean_QD", "mean_Tavg", "mean_charge_current", "peak_charge_current"]
OPTIONAL = ["delta_min", "mean_chargetime", "mean_IR"]
METADATA = ["batch_id", "cell_id", "charging_policy", "structure_group", "known_collection_issue",
            "author_quality_flag", "endpoint_review_candidate", "last_QD", "last_cycle",
            "last_to_initial_ratio", "record_minus_life"]
DELTA_FEATURES = ["delta_mean", "delta_var", "delta_logvar", "delta_min", "delta_max", "delta_abs_area"]
CURRENT_FEATURES = ["mean_charge_current", "peak_charge_current", "high_current_fraction"]
BASE_FEATURES = ["mean_QD", "std_QD", "mean_IR", "mean_Tavg", "mean_Tmax", "mean_chargetime"]
FEATURES = BASE_FEATURES + ["early_fade_rate"] + CURRENT_FEATURES + DELTA_FEATURES
FORBIDDEN_X = {"cycle_life", "batch_id", "cell_id", "sample_id", "last_QD", "last_cycle",
               "last_to_initial_ratio", "record_minus_life", "endpoint_review_candidate",
               "author_quality_flag", "known_collection_issue", "knee_candidate", "late_slope", "group"}


@dataclass
class FeatureEngineeringResult:
    """Tables used by the feature notebook and training entry point."""
    project_root: Path
    batch_paths: dict[int, Path]
    all_features: pd.DataFrame
    excluded_features: pd.DataFrame
    feature_quality: pd.DataFrame
    examples: dict
    source_scope: pd.DataFrame
    raw_delta_correlations: pd.DataFrame
    policy_preview: pd.DataFrame
    endpoint_audit: pd.DataFrame
    excluded_quality: pd.DataFrame
    selected: pd.DataFrame
    conservative: pd.DataFrame
    sample_scope: pd.DataFrame
    X: pd.DataFrame
    X_structured: pd.DataFrame
    X_baseline: pd.DataFrame
    X_categorical: pd.DataFrame
    X_structured_with_label: pd.DataFrame
    X_extended: pd.DataFrame
    X_optional: pd.DataFrame
    y: pd.DataFrame
    metadata: pd.DataFrame
    X_sensitivity: pd.DataFrame
    y_sensitivity: pd.DataFrame
    metadata_sensitivity: pd.DataFrame
    X_variants: dict[str, pd.DataFrame]
    feature_definitions: pd.DataFrame
    missing_summary: pd.DataFrame
    target_correlations: pd.DataFrame
    delta_selected: pd.DataFrame
    candidate_corr: pd.DataFrame
    candidate_vif: pd.DataFrame
    policy_generalization: pd.DataFrame
    nominal_policy_collisions: pd.DataFrame
    manifest: dict

    def as_notebook_namespace(self) -> dict:
        """Expose named tables for display; this performs no code execution."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


def parse_policy(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with C1/SOC/C2 and the original suffix indicator."""
    result = frame.copy()
    parsed = result.charging_policy.str.extract(POLICY_PATTERN)
    if parsed.iloc[:, :3].isna().any().any():
        unknown = result.loc[parsed.iloc[:, :3].isna().any(axis=1), "charging_policy"].unique()
        raise ValueError(f"Unparsed charging Policy: {list(unknown)}")
    result["policy_C1"] = parsed[0].astype(float)
    result["policy_switch_SOC_pct"] = parsed[1].astype(float)
    result["policy_C2"] = parsed[2].astype(float)
    result["policy_newstructure_label"] = parsed[3].notna().astype(int)
    result["structure_group"] = np.where(result.policy_newstructure_label.eq(1),
                                          "newstructure label", "Other labelled protocols")
    if not result.policy_C1.gt(0).all() or not result.policy_C2.gt(0).all() or not result.policy_switch_SOC_pct.between(0, 100).all():
        raise ValueError("Charging Policy has invalid C-rate/SOC values")
    return result


def extract_batch_features(path: str | Path, batch_id: int) -> tuple:
    """Read actual cycle 10–100 summary/current data and Q10/Q100 only.

    Returns (features, exclusions, quality, first_example, raw_cell_count).
    Endpoint diagnostics are deliberately computed in a separate function.
    """
    feature_rows, excluded_rows, quality_rows = [], [], []
    example = None
    with h5py.File(path, "r") as file:
        batch = file["batch"]
        raw_count = batch["cycle_life"].size
        for cid in range(raw_count):
            lifetime = float(read_cell_field(file, batch, cid, "cycle_life")[0])
            if not np.isfinite(lifetime):
                excluded_rows.append(dict(batch_id=batch_id, cell_id=cid, reason="Missing cycle_life"))
                continue
            if lifetime <= EARLY_END or not lifetime.is_integer():
                excluded_rows.append(dict(batch_id=batch_id, cell_id=cid, reason="Invalid/too-short lifetime label"))
                continue
            summary = file[batch["summary"][cid, 0]]
            labels = validate_cycle_labels(summary["cycle"][()].ravel(), f"Batch {batch_id}, cell {cid}")
            detailed = file[batch["cycles"][cid, 0]]
            if len(labels) != detailed["Qdlin"].size:
                raise ValueError(f"Batch {batch_id}, cell {cid}: summary/detailed cycle mismatch")
            if not {EARLY_START, EARLY_END}.issubset(set(labels)):
                excluded_rows.append(dict(batch_id=batch_id, cell_id=cid, reason="Missing actual cycle 10/100"))
                continue
            positions = np.flatnonzero((labels >= EARLY_START) & (labels <= EARLY_END))
            early_raw = pd.DataFrame({name: summary[name][()].ravel()[positions] for name in SUMMARY_COLUMNS})
            early, cleaning = clean_early_summary(early_raw)
            try:
                early_stats = summarize_early_summary(early, cleaning)
            except ValueError:
                excluded_rows.append(dict(batch_id=batch_id, cell_id=cid, reason="Insufficient valid early Qd"))
                continue
            currents, current_quality = read_early_current_stats(file, detailed, labels)
            i10, i100 = (int(np.flatnonzero(labels == label)[0]) for label in (EARLY_START, EARLY_END))
            q10, q100 = (file[detailed["Qdlin"][index, 0]][()].ravel() for index in (i10, i100))
            try:
                delta, curves = delta_q_features(read_cell_field(file, batch, cid, "Vdlin"), q10, q100)
            except ValueError as error:
                if str(error) == "voltage-grid length mismatch":
                    raise ValueError(f"Batch {batch_id}, cell {cid}: {error}") from error
                excluded_rows.append(dict(batch_id=batch_id, cell_id=cid, reason=str(error)))
                continue
            sample_id = f"b{batch_id}_c{cid}"
            feature_rows.append(dict(
                sample_id=sample_id, batch_id=batch_id, cell_id=cid,
                charging_policy=decode_policy(read_cell_field(file, batch, cid, "policy_readable")),
                cycle_life=int(lifetime), known_collection_issue=(batch_id == 3 and cid == 37),
                author_quality_flag=(batch_id == 3 and cid in BATCH3_QUALITY_IDS),
                **early_stats, mean_charge_current=currents[0], peak_charge_current=currents[1],
                high_current_fraction=currents[2], **delta))
            quality_rows.append(dict(
                sample_id=sample_id, batch_id=batch_id, cell_id=cid,
                feature_cycle_min=int(early.cycle.min()), feature_cycle_max=int(early.cycle.max()),
                early_summary_rows=len(early), valid_QD=int(early.QDischarge.notna().sum()),
                valid_IR=int(early.IR.notna().sum()), valid_charge_time=int(early.chargetime.notna().sum()),
                removed_time_gaps=current_quality["removed_time_gaps"],
                removed_charge_spikes=cleaning["removed_charge_spikes"],
                measured_charge_cycles=current_quality["measured_charge_cycles"]))
            if example is None:
                example = dict(sample_id=sample_id, **curves)
    return (pd.DataFrame(feature_rows), pd.DataFrame(excluded_rows, columns=["batch_id", "cell_id", "reason"]),
            pd.DataFrame(quality_rows), example, raw_count)


def audit_endpoints(paths: dict[int, Path], features: pd.DataFrame) -> pd.DataFrame:
    """Return EOL review metadata without altering provided lifetime labels."""
    rows = []
    for batch_id, path in paths.items():
        with h5py.File(path, "r") as file:
            batch = file["batch"]
            for sample_id, row in features.loc[features.batch_id.eq(batch_id)].iterrows():
                summary = file[batch["summary"][int(row.cell_id), 0]]
                audit = endpoint_values(summary["cycle"][()].ravel(), summary["QDischarge"][()].ravel(), row.cycle_life)
                rows.append(dict(sample_id=sample_id, last_QD=audit["last_QD"], last_cycle=audit["last_cycle"],
                                 last_to_initial_ratio=audit["last_to_initial_ratio"], record_minus_life=audit["record_minus_life"],
                                 endpoint_review_candidate=audit["endpoint_review_candidate"]))
    return pd.DataFrame(rows).set_index("sample_id")


def feature_definitions() -> pd.DataFrame:
    rows = [
        ["delta_logvar", "log10(var(Q100-Q10)), ddof=0", "초기 방전곡선 변화", "Core"],
        ["policy_C1", "원본 Policy 1차 C-rate", "1차 충전 조건", "Core"],
        ["policy_switch_SOC_pct", "원본 Policy 전환 SOC(%)", "전환 조건", "Core"],
        ["policy_C2", "원본 Policy 2차 C-rate", "2차 충전 조건", "Core"],
        ["charging_policy", "원본 Policy 문자열", "범주형 Policy 전체", "Categorical alternative"],
        ["policy_newstructure_label", "원본 꼬리말 존재 여부(0/1)", "라벨 그룹 정보; 물리적 의미 단정 없음", "Label alternative"],
        ["early_fade_rate", "10~100 Qd 직선 기울기의 음수(Ah/cycle)", "초기 감소/증가 속도", "Extended"],
        ["mean_QD", "10~100 평균 Qd(Ah)", "초기 용량 수준", "Extended"],
        ["mean_Tavg", "10~100 평균 Tavg(°C)", "초기 열 조건", "Extended"],
        ["mean_charge_current", "사이클별 시간 가중 충전 전류 평균의 평균(A)", "지속 충전 강도", "Extended"],
        ["peak_charge_current", "사이클별 최대 충전 전류의 평균(A)", "피크 조건; 전체 단일 최댓값과 다름", "Extended"],
        ["delta_min", "min(Q100-Q10)(Ah)", "ΔQ 대체 표현", "Optional"],
        ["mean_chargetime", "이상값 제외 초기 평균 충전시간(min)", "초기 충전시간", "Optional"],
        ["mean_IR", "비양수 값 제외 초기 평균 IR(ohm)", "초기 내부저항; 일부 결측", "Optional"],
    ]
    return pd.DataFrame(rows, columns=["feature", "formula", "meaning", "role"])


def candidate_vif_table(frame: pd.DataFrame) -> pd.DataFrame:
    """VIF convention used by the existing early-feature analysis."""
    complete, rows = frame.dropna(), []
    for feature in frame.columns:
        target = complete[feature].to_numpy()
        tss = np.sum((target - target.mean()) ** 2)
        others = complete.drop(columns=feature)
        z = ((others - others.mean()) / others.std().replace(0, 1)).to_numpy()
        design = np.column_stack([np.ones(len(complete)), z])
        pred = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        rss = np.sum((target - pred) ** 2)
        vif = (np.inf if rss <= 1e-12 * tss else tss / rss) if tss > 0 else np.nan
        rows.append(dict(feature=feature, VIF=vif, n=len(complete)))
    return pd.DataFrame(rows)


def build_feature_tables(
    batch_paths: dict[int, str | Path] | None = None,
    project_root: str | Path | None = None,
) -> FeatureEngineeringResult:
    """Extract the main and endpoint-sensitivity feature sets from raw MAT files."""
    root = discover_project_root() if project_root is None else Path(project_root).resolve()
    paths = get_batch_paths(root) if batch_paths is None else {b: Path(path).resolve() for b, path in batch_paths.items()}
    frames, exclusions, qualities, examples, scopes = [], [], [], {}, []
    for batch_id, path in paths.items():
        features, excluded, quality, example, raw_count = extract_batch_features(path, batch_id)
        if features.empty:
            raise ValueError(f"Batch {batch_id}: no eligible early-cycle features")
        frames.append(features); exclusions.append(excluded); qualities.append(quality)
        examples[batch_id] = example
        scopes.append(dict(batch_id=batch_id, raw_cells=raw_count, extracted_labelled_cells=len(features), excluded=len(excluded)))
    all_features = pd.concat(frames, ignore_index=True).set_index("sample_id")
    excluded_features = pd.concat(exclusions, ignore_index=True)
    feature_quality = pd.concat(qualities, ignore_index=True).set_index("sample_id")
    raw_delta_correlations = pd.DataFrame([
        dict(batch_id=b, n=len(part), pearson=part.delta_logvar.corr(part.cycle_life),
             spearman=part.delta_logvar.corr(part.cycle_life, method="spearman"))
        for b, part in all_features.groupby("batch_id")])
    all_features = parse_policy(all_features)
    policy_preview = all_features[["charging_policy"] + POLICY_NUMERIC + ["policy_newstructure_label"]].drop_duplicates()
    endpoint_audit = audit_endpoints(paths, all_features)
    all_features = all_features.join(endpoint_audit, validate="one_to_one")
    excluded_quality = all_features.loc[all_features.author_quality_flag].copy()
    selected = all_features.loc[~all_features.author_quality_flag].copy()
    conservative = selected.loc[~selected.endpoint_review_candidate].copy()
    sample_scope = pd.DataFrame([
        {"scope": "Labelled feature extraction", "n": len(all_features)},
        {"scope": "Default: exclude Batch3 quality flags", "n": len(selected)},
        {"scope": "Sensitivity: also exclude endpoint review candidates", "n": len(conservative)},
    ])
    X = selected[CORE].copy()
    X_variants = {
        "X": X, "X_baseline": selected[["delta_logvar"]].copy(),
        "X_categorical": selected[["delta_logvar", "charging_policy"]].copy(),
        "X_structured_with_label": selected[CORE + ["policy_newstructure_label"]].copy(),
        "X_extended": selected[CORE + ADDITIONAL].copy(), "X_optional": selected[OPTIONAL].copy(),
    }
    y, metadata = selected[["cycle_life"]].astype(int), selected[METADATA].copy()
    for name, table in X_variants.items():
        if not table.index.equals(y.index) or not table.index.is_unique or not table.columns.is_unique:
            raise ValueError(f"Unaligned/duplicate rows or columns in {name}")
        if set(table.columns) & FORBIDDEN_X:
            raise ValueError(f"Target/metadata leakage in {name}")
    if not X.index.equals(metadata.index) or not y.cycle_life.gt(EARLY_END).all():
        raise ValueError("Invalid feature/target alignment")
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(X_variants["X_extended"].to_numpy()).all():
        raise ValueError("Nonfinite core/extended feature values")
    if not feature_quality.feature_cycle_min.ge(EARLY_START).all() or not feature_quality.feature_cycle_max.le(EARLY_END).all():
        raise ValueError("Feature window extends beyond initial cycles")
    numeric_review = CORE + ADDITIONAL + OPTIONAL + ["std_QD", "mean_Tmax"]
    missing_summary = pd.DataFrame({"missing_cells": selected[numeric_review].isna().sum(),
                                    "missing_pct": 100 * selected[numeric_review].isna().mean(),
                                    "unique_values": selected[numeric_review].nunique()})
    correlation_rows = []
    for batch_id, part in selected.groupby("batch_id"):
        for feature in numeric_review:
            pair = part[[feature, "cycle_life"]].dropna()
            correlation_rows.append(dict(batch_id=batch_id, feature=feature, n=len(pair),
                                         pearson=pair[feature].corr(pair.cycle_life),
                                         spearman=pair[feature].corr(pair.cycle_life, method="spearman")))
    target_correlations = pd.DataFrame(correlation_rows)
    candidate_numeric = ["delta_logvar"] + ADDITIONAL
    generalization_rows = []
    for batch_id in sorted(metadata.batch_id.unique()):
        train_meta = metadata.loc[metadata.batch_id.ne(batch_id)]
        test_meta = metadata.loc[metadata.batch_id.eq(batch_id)]
        unknown = ~test_meta.charging_policy.isin(set(train_meta.charging_policy))
        generalization_rows.append(dict(held_out_batch=int(batch_id), train_rows=len(train_meta),
                                        test_rows=len(test_meta), unseen_policy_cells=int(unknown.sum()),
                                        unseen_policy_pct=100 * unknown.mean()))
    sources = {}
    for batch_id, path in paths.items():
        try:
            sources[str(batch_id)] = str(path.relative_to(root))
        except ValueError:
            sources[str(batch_id)] = str(path)
    manifest = dict(
        target="cycle_life", feature_window=[EARLY_START, EARLY_END], delta_cycles=[10, 100], sources=sources,
        sample_index="sample_id", raw_labelled_feature_rows=len(all_features), selected_rows=len(selected),
        rows_by_batch={str(b): int(n) for b, n in selected.groupby("batch_id").size().items()},
        core_feature_columns=CORE, variants={name: list(table.columns) for name, table in X_variants.items()},
        excluded_quality_ids=excluded_quality[["batch_id", "cell_id"]].to_dict("records"),
        sensitivity_rows=len(conservative), sensitivity_rule="Additional exclusion of endpoint_review_candidate; descriptive only",
        label_status="Provided cycle_life; endpoint completeness requires verification",
        fitted_preprocessing=False, model_training=False,
        selection_status="EDA-informed candidates; no cross-validation selection")
    return FeatureEngineeringResult(
        project_root=root, batch_paths=paths, all_features=all_features, excluded_features=excluded_features,
        feature_quality=feature_quality, examples=examples, source_scope=pd.DataFrame(scopes),
        raw_delta_correlations=raw_delta_correlations, policy_preview=policy_preview, endpoint_audit=endpoint_audit,
        excluded_quality=excluded_quality, selected=selected, conservative=conservative, sample_scope=sample_scope,
        X=X, X_structured=X.copy(), X_baseline=X_variants["X_baseline"], X_categorical=X_variants["X_categorical"],
        X_structured_with_label=X_variants["X_structured_with_label"], X_extended=X_variants["X_extended"],
        X_optional=X_variants["X_optional"], y=y, metadata=metadata, X_sensitivity=conservative[CORE].copy(),
        y_sensitivity=conservative[["cycle_life"]].astype(int), metadata_sensitivity=conservative[METADATA].copy(),
        X_variants=X_variants, feature_definitions=feature_definitions(), missing_summary=missing_summary,
        target_correlations=target_correlations, delta_selected=target_correlations.loc[target_correlations.feature.eq("delta_logvar")],
        candidate_corr=selected[candidate_numeric].corr(), candidate_vif=candidate_vif_table(selected[candidate_numeric]),
        policy_generalization=pd.DataFrame(generalization_rows),
        nominal_policy_collisions=selected.groupby(POLICY_NUMERIC).charging_policy.nunique().rename("original_labels").reset_index(),
        manifest=manifest)


def save_feature_tables(result: FeatureEngineeringResult, output_dir: str | Path) -> Path:
    """Write model-ready tables and descriptive audits; no model is trained."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    indexed = {
        **result.X_variants, "X_structured": result.X_structured, "y": result.y, "metadata": result.metadata,
        "all_extracted_features": result.all_features, "selected_feature_table": result.selected,
        "feature_quality": result.feature_quality, "endpoint_audit": result.endpoint_audit,
        "excluded_quality_cells": result.excluded_quality, "candidate_feature_correlations": result.candidate_corr,
    }
    flat = {
        "excluded_feature_cells": result.excluded_features, "feature_definitions": result.feature_definitions,
        "target_correlations_by_batch": result.target_correlations, "candidate_vif": result.candidate_vif,
        "raw_delta_correlations": result.raw_delta_correlations, "policy_generalization": result.policy_generalization,
        "sample_scope": result.sample_scope, "source_scope": result.source_scope,
    }
    for name, table in indexed.items():
        table.to_csv(output / f"{name}.csv")
    for name, table in flat.items():
        table.to_csv(output / f"{name}.csv", index=False)
    result.missing_summary.to_csv(output / "missing_summary.csv", index_label="feature")
    sensitivity = output / "sensitivity"
    sensitivity.mkdir(exist_ok=True)
    for name, table in result.X_variants.items():
        table.loc[result.conservative.index].to_csv(sensitivity / f"{name}.csv")
    result.y_sensitivity.to_csv(sensitivity / "y.csv")
    result.metadata_sensitivity.to_csv(sensitivity / "metadata.csv")
    (output / "manifest.json").write_text(json.dumps(result.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "feature_engineering_summary.md").write_text(build_feature_report(result) + "\n", encoding="utf-8")
    return output


def build_feature_report(result: FeatureEngineeringResult) -> str:
    """Describe the exported feature definitions and sample scope."""
    batch3 = result.delta_selected.loc[result.delta_selected.batch_id.eq(3), "pearson"]
    lines = [
        "# Feature Engineering 결과", "",
        f"- 원본 수명 유효 피처 {len(result.all_features)}개, 품질 선별 기본 {len(result.selected)}개, 종단 점검 민감도 {len(result.conservative)}개.",
        "- 기본 X: " + ", ".join(CORE) + ".",
        "- 기본 숫자형 Policy는 C1/SOC/C2이며 원본 Policy 범주형과 꼬리말 보존 대안도 제공한다.",
        "- Target: 제공된 cycle_life. 확정된 EOL 관측 표본만 있다고 가정하지 않는다.",
        "- 피처 계산 구간: 실제 10~100사이클. ΔQ = Q100−Q10, log10(var), ddof=0.",
        "- 확장 피처는 초기 열화율·용량·온도·전류이며 EDA 기반 후보다.",
    ]
    if len(batch3):
        lines.append(f"- Batch 3 기본{int(result.selected.batch_id.eq(3).sum())}개에서 ΔQ 상관은 {float(batch3.iloc[0]):.3f}. 원본44개 EDA와 구분한다.")
    lines += [
        "- 일부 IR 결측을 유지하고, 보간/표준화/OneHotEncoder는 학습 fold 안에서 fit해야 한다.",
        "- Knee·후기 기울기·전체 기록 길이·수명 그룹·배치/셀 ID·품질/종단 플래그는 기본 X에서 제외한다.",
        "- 정책 미관측 표는 기술적 진단이다. DAY 2 모델링은 Batch 1만 학습·튜닝하고 Batch 2를 테스트하며 Batch 3은 사용하지 않는다.",
        "- 이 모듈은 모델을 학습하거나 검증 성능을 산출하지 않는다.",
        "- 실행 노트북: notebooks/02_feature_engineering.ipynb. 기본 출력: data/processed.",
    ]
    return "\n".join(lines)


def run_feature_engineering(
    project_root: str | Path | None = None, output_dir: str | Path | None = None,
) -> FeatureEngineeringResult:
    """Extract and save features, by default under data/processed."""
    result = build_feature_tables(project_root=project_root)
    save_feature_tables(result, output_dir or result.project_root / "data" / "processed")
    return result


def prepare_delta_features(batch_result: dict) -> dict:
    """Attach EDA ΔQ tables/curves/groups using the same shared definition."""
    rows, curves, exclusions = [], {}, []
    for cell in batch_result["cells"]:
        if not {10, 100}.issubset(cell["q"]):
            exclusions.append(dict(cell_id=cell["cell_id"], reason="Missing actual cycle 10/100"))
            continue
        try:
            stats, example = delta_q_features(cell["voltage"], cell["q"][10], cell["q"][100])
        except ValueError as error:
            if str(error) == "voltage-grid length mismatch":
                raise ValueError(f"Batch {batch_result['batch']}, cell {cell['cell_id']}: {error}") from error
            reason = "Nonfinite/duplicate voltage or delta Q" if str(error) == "Invalid voltage/delta Q" else str(error)
            exclusions.append(dict(cell_id=cell["cell_id"], reason=reason))
            continue
        curves[cell["cell_id"]] = (example["voltage"], example["delta"])
        rows.append(dict(cell_id=cell["cell_id"], **stats))
    delta = pd.DataFrame(rows).merge(batch_result["life"], on="cell_id", validate="one_to_one")
    q1, q3 = batch_result.get("quartiles", batch_result["life"].cycle_life.quantile([.25, .75]))
    delta["quartile_group"] = np.select([delta.cycle_life <= q1, delta.cycle_life >= q3],
                                         ["Lower quartile", "Upper quartile"], default="Middle")
    delta["threshold_group"] = np.select([delta.cycle_life < 500, delta.cycle_life > 1000],
                                          ["Short <500", "Long >1000"], default="Middle")
    batch_result.update(delta=delta, delta_curves=curves,
                        delta_exclusions=pd.DataFrame(exclusions, columns=["cell_id", "reason"]))
    return batch_result


def add_eda_features(batch_result: dict) -> dict:
    """Attach early/Policy tables to the existing EDA batch-result structure."""
    if "delta" not in batch_result:
        prepare_delta_features(batch_result)
    rows = []
    quality = batch_result["quality"].set_index("cell_id").copy()
    for cell in batch_result["cells"]:
        early, cleaning = clean_early_summary(cell["summary"])
        stats = summarize_early_summary(early, cleaning)
        quality.loc[cell["cell_id"], "charge_time_spikes"] = cleaning["removed_charge_spikes"]
        quality.loc[cell["cell_id"], "valid_IR"] = int(early.IR.notna().sum())
        rows.append(dict(cell_id=cell["cell_id"], **stats,
                         mean_charge_current=cell["current"][0], peak_charge_current=cell["current"][1],
                         high_current_fraction=cell["current"][2]))
    early = pd.DataFrame(rows).merge(batch_result["life"], on="cell_id", validate="one_to_one")
    early = early.merge(batch_result["delta"][["cell_id"] + DELTA_FEATURES], on="cell_id", how="left", validate="one_to_one")
    # Preserve the original EDA column order and its three numeric Policy fields.
    early["structure_group"] = np.where(early.charging_policy.str.endswith("-newstructure"),
                                          "newstructure label", "Other labelled protocols")
    parsed = parse_policy(early)
    early[POLICY_NUMERIC] = parsed[POLICY_NUMERIC]
    policy = early.groupby("charging_policy").cycle_life.agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    batch_result.update(early=early, policy=policy, quality=quality.reset_index())
    return batch_result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract early-cycle ESSHealth features without training a model")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    result = run_feature_engineering(args.project_root, args.output_dir)
    print(json.dumps({"X_shape": list(result.X.shape), "y_shape": list(result.y.shape),
                      "rows_by_batch": result.manifest["rows_by_batch"],
                      "sensitivity_rows": len(result.conservative)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

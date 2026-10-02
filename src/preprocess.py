"""Selective MATLAB loading and shared battery-data preprocessing.

Only requested HDF5 datasets are read. MATLAB string metadata is never decoded
by a general-purpose MAT loader, and detailed current traces after cycle 100
are never read. Full summary curves are retained only by the EDA loader.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

BATCH_FILES = {
    1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
EARLY_START, EARLY_END = 10, 100
SUMMARY_COLUMNS = ["cycle", "QDischarge", "IR", "Tavg", "Tmax", "chargetime"]
BATCH3_QUALITY_IDS = frozenset({2, 37, 42, 43})


def discover_project_root(start: str | Path | None = None) -> Path:
    """Find the repository from its root, notebooks directory, or src directory.

    Discovery does not require downloading raw files: the data directory and
    source module identify a checkout. An explicit project root can always be
    passed to downstream functions.
    """
    initial = Path(start or Path.cwd()).expanduser().resolve()
    if initial.is_file():
        initial = initial.parent
    for candidate in (initial, *initial.parents):
        if (candidate / "data").is_dir() and (
            (candidate / "src" / "preprocess.py").is_file()
            or (candidate / "data" / "archive").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        f"Project root not found above {initial}. Pass project_root explicitly."
    )


def get_batch_paths(project_root: str | Path | None = None) -> dict[int, Path]:
    """Return the three required raw MAT paths, checking all files exist."""
    root = discover_project_root() if project_root is None else Path(project_root).resolve()
    paths = {number: root / "data" / "archive" / name for number, name in BATCH_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required raw data missing:\n" + "\n".join(missing))
    return paths


def read_cell_field(file: h5py.File, batch: h5py.Group, cell_id: int, name: str) -> np.ndarray:
    """Dereference one MATLAB struct field without loading unrelated fields."""
    return file[batch[name][cell_id, 0]][()].ravel()


def validate_cycle_labels(values: np.ndarray, context: str = "Cell") -> np.ndarray:
    values = np.asarray(values).ravel()
    if not len(values) or not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{context}: invalid cycle labels")
    labels = values.astype(int)
    if len(np.unique(labels)) != len(labels) or np.any(np.diff(labels) <= 0):
        raise ValueError(f"{context}: duplicate/unsorted cycles")
    return labels


def decode_policy(values: np.ndarray) -> str:
    return "".join(chr(int(value)) for value in values).strip()


def clean_early_summary(
    summary: pd.DataFrame, start: int = EARLY_START, end: int = EARLY_END
) -> tuple[pd.DataFrame, dict]:
    """Clean the actual early-cycle labels without fitting a learned transform.

    Charging-time spikes are identified using this battery's own early window.
    The raw mean and median are returned for descriptive sensitivity checks.
    """
    early = summary.loc[summary.cycle.between(start, end)].copy()
    early.loc[~early.QDischarge.between(.05, 1.5), "QDischarge"] = np.nan
    for column in ("IR", "Tavg", "Tmax", "chargetime"):
        early.loc[(early[column] <= 0) | ~np.isfinite(early[column]), column] = np.nan
    raw_charge_mean = early.chargetime.mean()
    charge_median = early.chargetime.median()
    spikes = early.chargetime > 5 * charge_median
    early.loc[spikes, "chargetime"] = np.nan
    return early, {
        "raw_mean_chargetime": raw_charge_mean,
        "median_chargetime": charge_median,
        "removed_charge_spikes": int(spikes.sum()),
    }


def summarize_early_summary(early: pd.DataFrame, cleaning: Mapping) -> dict:
    """Summarize a cleaned early window; negative fade rates remain negative."""
    good = early.QDischarge.notna()
    if good.sum() < 2:
        raise ValueError("Insufficient valid early Qd")
    slope = np.polyfit(early.loc[good, "cycle"], early.loc[good, "QDischarge"], 1)[0]
    return {
        "mean_QD": early.QDischarge.mean(), "std_QD": early.QDischarge.std(),
        "mean_IR": early.IR.mean(), "mean_Tavg": early.Tavg.mean(),
        "mean_Tmax": early.Tmax.mean(), "mean_chargetime": early.chargetime.mean(),
        "raw_mean_chargetime": cleaning["raw_mean_chargetime"],
        "median_chargetime": cleaning["median_chargetime"], "early_fade_rate": -slope,
    }


def charging_current_stats(traces: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, dict]:
    """Average cycle-wise time-weighted current, peak, and high-current fraction.

    Input traces use current in A and recorded time in minutes. Only positive
    charging current and finite positive intervals below the gap limit count.
    The peak is averaged across cycles, rather than taking one global maximum.
    """
    rows, gap_count = [], 0
    for amps, time in traces:
        amps, time = np.asarray(amps).ravel(), np.asarray(time).ravel()
        if len(amps) != len(time):
            raise ValueError("I/t mismatch")
        dt = np.diff(time)
        positive_dt = dt[np.isfinite(dt) & (dt > 0)]
        limit = max(1.0, 10 * np.median(positive_dt)) if len(positive_dt) else 1.0
        gap_count += int((dt > limit).sum())
        valid = (amps[:-1] > .1) & np.isfinite(amps[:-1]) & np.isfinite(dt) & (dt > 0) & (dt <= limit)
        if valid.any():
            a, weights = amps[:-1][valid], dt[valid]
            peak = a.max()
            rows.append([np.average(a, weights=weights), peak,
                         weights[a >= .8 * peak].sum() / weights.sum()])
    stats = np.mean(rows, axis=0) if rows else np.full(3, np.nan)
    return stats, {"removed_time_gaps": gap_count, "measured_charge_cycles": len(rows)}


def read_early_current_stats(
    file: h5py.File, detailed: h5py.Group, labels: np.ndarray,
    start: int = EARLY_START, end: int = EARLY_END,
) -> tuple[np.ndarray, dict]:
    positions = np.flatnonzero((labels >= start) & (labels <= end))
    traces = ((file[detailed["I"][j, 0]][()].ravel(), file[detailed["t"][j, 0]][()].ravel())
              for j in positions)
    return charging_current_stats(traces)


def delta_q_features(
    voltage: np.ndarray, q10: np.ndarray, q100: np.ndarray
) -> tuple[dict, dict]:
    """Calculate Q100−Q10 on the shared voltage grid, population variance."""
    voltage, q10, q100 = (np.asarray(value).ravel() for value in (voltage, q10, q100))
    if not (len(voltage) == len(q10) == len(q100)):
        raise ValueError("voltage-grid length mismatch")
    order = np.argsort(voltage)
    v, q10, q100 = voltage[order], q10[order], q100[order]
    delta = q100 - q10
    if not np.isfinite(v).all() or not np.isfinite(delta).all() or np.any(np.diff(v) <= 0):
        raise ValueError("Invalid voltage/delta Q")
    variance = np.var(delta)
    if variance <= 0:
        raise ValueError("Nonpositive delta variance")
    stats = dict(delta_mean=delta.mean(), delta_var=variance, delta_logvar=np.log10(variance),
                 delta_min=delta.min(), delta_max=delta.max(), delta_abs_area=np.trapezoid(np.abs(delta), v))
    curves = dict(voltage=v, q10=q10, q100=q100, delta=delta, delta_logvar=np.log10(variance))
    return stats, curves


def endpoint_values(cycles: np.ndarray, qd: np.ndarray, lifetime: int) -> dict:
    """Audit label completeness; these values must never enter model X."""
    good = np.isfinite(qd) & (qd >= .05) & (qd <= 1.5)
    last_qd = float(qd[good][-1]) if good.any() else np.nan
    initial = np.median(qd[good & (cycles >= 10) & (cycles <= 30)])
    crossings = cycles[good & (cycles >= 10) & (qd <= .88)]
    return dict(last_QD=last_qd, last_cycle=int(cycles[-1]), last_to_initial_ratio=last_qd / initial,
                record_minus_life=int(cycles[-1] - lifetime), endpoint_review_candidate=bool(last_qd > .885),
                first_cycle_QD_le_0_88=float(crossings[0]) if len(crossings) else np.nan)


def load_batch(path: str | Path, batch_number: int) -> dict:
    """Read one batch for EDA, retaining complete summary capacity curves.

    Missing lifetime labels are reported separately and not imputed. Original
    zero-based cell IDs are retained even when some cells are excluded.
    """
    path = Path(path)
    cells, excluded, quality, endpoints = [], [], [], []
    with h5py.File(path, "r") as file:
        batch = file["batch"]
        raw_count = batch["cycle_life"].size
        for cid in range(raw_count):
            summary_group = file[batch["summary"][cid, 0]]
            summary = pd.DataFrame({name: summary_group[name][()].ravel() for name in summary_group})
            labels = validate_cycle_labels(summary.cycle.to_numpy(), f"Batch {batch_number}, cell {cid}")
            life = float(read_cell_field(file, batch, cid, "cycle_life")[0])
            if not np.isfinite(life):
                excluded.append(dict(cell_id=cid, reason="Missing cycle_life", recorded_cycles=len(summary), last_cycle=int(labels[-1])))
                continue
            if life <= 0 or not life.is_integer():
                raise ValueError(f"Batch {batch_number}, cell {cid}: invalid lifetime")
            detailed = file[batch["cycles"][cid, 0]]
            if len(labels) != detailed["Qdlin"].size:
                raise ValueError(f"Batch {batch_number}, cell {cid}: summary/detailed cycle mismatch")
            q = {int(label): file[detailed["Qdlin"][j, 0]][()].ravel()
                 for j, label in enumerate(labels) if label in (EARLY_START, EARLY_END)}
            currents, current_quality = read_early_current_stats(file, detailed, labels)
            policy = decode_policy(read_cell_field(file, batch, cid, "policy_readable"))
            cells.append(dict(cell_id=cid, cycle_life=int(life), charging_policy=policy, summary=summary,
                              voltage=read_cell_field(file, batch, cid, "Vdlin"), q=q, current=currents))
            quality.append(dict(cell_id=cid, time_gap_intervals=current_quality["removed_time_gaps"],
                                measured_early_cycles=current_quality["measured_charge_cycles"]))
            audit = endpoint_values(labels, summary.QDischarge.to_numpy(), int(life))
            endpoints.append(dict(batch=batch_number, cell_id=cid, provided_life=int(life),
                                  last_cycle=audit["last_cycle"], recorded_minus_life=audit["record_minus_life"],
                                  last_QD=audit["last_QD"], last_to_initial_ratio=audit["last_to_initial_ratio"],
                                  first_cycle_QD_le_0_88=audit["first_cycle_QD_le_0_88"],
                                  endpoint_review_candidate=audit["endpoint_review_candidate"]))
    life = pd.DataFrame([{key: cell[key] for key in ("cell_id", "cycle_life", "charging_policy")} for cell in cells],
                        columns=["cell_id", "cycle_life", "charging_policy"])
    life["known_collection_issue"] = life.cell_id.eq(37) if batch_number == 3 else False
    life["author_quality_flag"] = life.cell_id.isin(BATCH3_QUALITY_IDS) if batch_number == 3 else False
    return dict(batch=batch_number, path=path, raw_count=raw_count, cells=cells, life=life,
                excluded=pd.DataFrame(excluded, columns=["cell_id", "reason", "recorded_cycles", "last_cycle"]),
                quality=pd.DataFrame(quality), endpoint=pd.DataFrame(endpoints))


def analyze_curves(cells: Iterable[dict]) -> tuple[pd.DataFrame, dict]:
    """EDA-only capacity slopes and exploratory continuous two-segment knees.

    This uses the complete observed curve and must not be used for initial
    100-cycle model inputs. Knee candidates depend on smoothing and search.
    """
    rows, smoothed = [], {}
    for cell in cells:
        summary = cell["summary"]
        good = np.isfinite(summary.QDischarge) & summary.QDischarge.between(.05, 1.5) & (summary.cycle >= 10)
        x = summary.loc[good, "cycle"].to_numpy()
        y = summary.loc[good, "QDischarge"].rolling(21, center=True, min_periods=1).median().to_numpy()
        if len(x) < 10 or np.ptp(x) <= 0:
            raise ValueError(f"Cell {cell['cell_id']}: insufficient capacity curve")
        smoothed[cell["cell_id"]] = (x, y)
        relative = (x - x.min()) / np.ptp(x)
        early_slope = np.polyfit(x[relative <= .3], y[relative <= .3], 1)[0]
        late_slope = np.polyfit(x[relative >= .7], y[relative >= .7], 1)[0]
        design0 = np.column_stack([np.ones(len(x)), x])
        linear = np.linalg.lstsq(design0, y, rcond=None)[0]
        rss0 = np.sum((y - design0 @ linear) ** 2)
        lower, upper = x.min() + .2 * np.ptp(x), x.min() + .8 * np.ptp(x)
        best = None
        for knee in np.linspace(lower, upper, 100):
            design = np.column_stack([design0, np.maximum(x - knee, 0)])
            coef = np.linalg.lstsq(design, y, rcond=None)[0]
            rss = np.sum((y - design @ coef) ** 2)
            if best is None or rss < best[0]:
                best = (rss, knee, coef)
        rss, knee, coef = best
        gain = len(x) * np.log(max(rss0, 1e-20) / max(rss, 1e-20)) - 2 * np.log(len(x))
        candidate = gain > 10 and coef[1] < 0 and coef[2] < 0 and coef[1] + coef[2] < 2 * coef[1]
        initial = summary.loc[summary.cycle.between(10, 30), "QDischarge"].median()
        rows.append(dict(cell_id=cell["cell_id"], early_slope=early_slope, late_slope=late_slope,
                         knee_candidate=knee if candidate else np.nan, before_slope=coef[1],
                         after_slope=coef[1] + coef[2], bic_gain=gain, search_upper=upper,
                         knee_at_upper_boundary=bool(candidate and np.isclose(knee, upper, atol=1e-6, rtol=0)),
                         end_capacity_ratio=y[-1] / initial))
    return pd.DataFrame(rows), smoothed


def vif_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive VIF on complete cases, including constant guards."""
    complete = frame.replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    for name in complete.columns:
        target = complete[name].to_numpy()
        total = np.sum((target - target.mean()) ** 2)
        if len(complete) <= len(complete.columns) or total <= 1e-20:
            rows.append(dict(feature=name, VIF=np.nan, n=len(complete), note="Insufficient rows/constant feature"))
            continue
        others = complete.drop(columns=name)
        z = ((others - others.mean()) / others.std().replace(0, 1)).to_numpy()
        design = np.column_stack([np.ones(len(complete)), z])
        pred = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        ratio = np.sum((target - pred) ** 2) / total
        rows.append(dict(feature=name, VIF=np.inf if ratio <= 1e-12 else 1 / ratio, n=len(complete), note=""))
    return pd.DataFrame(rows)

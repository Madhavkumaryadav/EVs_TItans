import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_CSV = REPO_ROOT / "results.csv"


@dataclass
class BatteryParams:
    q_ah: float = 2.0
    r0_ohm: float = 0.035
    r1_ohm: float = 0.020
    c1_f: float = 2200.0
    eta_charge: float = 0.995
    eta_discharge: float = 1.0
    v_min: float = 2.8
    v_max: float = 4.2


@dataclass
class SimConfig:
    t_end_s: float = 9000.0
    dt_s: float = 1.0
    soc0: float = 0.95
    vp0_v: float = 0.0


def current_profile_a(t_s: np.ndarray) -> np.ndarray:
    i = np.zeros_like(t_s)
    i[(t_s >= 600.0) & (t_s < 4200.0)] = 1.0
    i[(t_s >= 4800.0) & (t_s < 7800.0)] = -0.8
    return i


def ocv_from_soc(soc: np.ndarray) -> np.ndarray:
    eps = 1e-6
    s = np.clip(soc, eps, 1.0 - eps)

    a0 = 3.10
    a1 = 1.00
    a2 = -0.22
    a3 = 0.05
    a4 = -0.04

    return a0 + a1 * s + a2 * s * s + a3 * np.log(s) + a4 * np.log(1.0 - s)


def run_simulation(params: BatteryParams, cfg: SimConfig) -> dict:
    t_s = np.arange(0.0, cfg.t_end_s + cfg.dt_s, cfg.dt_s)
    n = t_s.size

    i_a = current_profile_a(t_s)

    soc = np.zeros(n)
    vp_v = np.zeros(n)

    soc[0] = np.clip(cfg.soc0, 0.0, 1.0)
    vp_v[0] = cfg.vp0_v

    q_as = params.q_ah * 3600.0

    for k in range(n - 1):
        ik = i_a[k]
        eta = params.eta_discharge if ik >= 0.0 else params.eta_charge

        dsoc_dt = -(eta * ik) / q_as
        dvp_dt = -(vp_v[k] / (params.r1_ohm * params.c1_f)) + (ik / params.c1_f)

        soc[k + 1] = np.clip(soc[k] + cfg.dt_s * dsoc_dt, 0.0, 1.0)
        vp_v[k + 1] = vp_v[k] + cfg.dt_s * dvp_dt

    ocv_v = ocv_from_soc(soc)
    vt_v = ocv_v - i_a * params.r0_ohm - vp_v

    return {
        "t_s": t_s,
        "i_a": i_a,
        "soc": soc,
        "vp_v": vp_v,
        "ocv_v": ocv_v,
        "vt_v": vt_v,
    }


def load_reference_csv(path: str | Path) -> dict:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Reference CSV is empty: {path}")

    keys = rows[0].keys()
    return {k: np.array([float(row[k]) for row in rows], dtype=float) for k in keys}


def _median_positive(values: np.ndarray, fallback: float) -> float:
    filtered = values[np.isfinite(values) & (values > 0.0)]
    if filtered.size == 0:
        return fallback
    return float(np.median(filtered))


def fit_params_from_reference(reference: dict, base_params: BatteryParams) -> BatteryParams:
    t = reference["t_s"]
    i_a = reference["i_a"]
    soc = reference["soc"]
    vp_v = reference["vp_v"]
    vt_v = reference["vt_v"]
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0

    dsoc_dt = np.diff(soc) / dt
    i_step = i_a[:-1]
    q_candidates = []
    for current, slope in zip(i_step, dsoc_dt):
        if abs(current) < 1e-8 or abs(slope) < 1e-12:
            continue
        eta = base_params.eta_discharge if current >= 0.0 else base_params.eta_charge
        q_as = -(eta * current) / slope
        if np.isfinite(q_as) and q_as > 0.0:
            q_candidates.append(q_as)

    q_ah = base_params.q_ah
    if q_candidates:
        q_ah = float(np.median(q_candidates) / 3600.0)

    ocv_v = ocv_from_soc(soc)
    active = np.abs(i_a) > 1e-8
    r0_candidates = (ocv_v[active] - vp_v[active] - vt_v[active]) / i_a[active]
    r0_ohm = _median_positive(np.asarray(r0_candidates, dtype=float), base_params.r0_ohm)

    dvp_dt = np.diff(vp_v) / dt
    a_matrix = np.column_stack([vp_v[:-1], i_a[:-1]])
    active_vp = np.isfinite(dvp_dt) & (np.abs(i_a[:-1]) > 1e-8)
    if int(np.sum(active_vp)) >= 2:
        coeffs, *_ = np.linalg.lstsq(a_matrix[active_vp], dvp_dt[active_vp], rcond=None)
        a_coeff, b_coeff = coeffs
        if np.isfinite(a_coeff) and np.isfinite(b_coeff) and b_coeff > 0.0 and a_coeff < 0.0:
            c1_f = float(1.0 / b_coeff)
            r1_ohm = float(-b_coeff / a_coeff)
        else:
            c1_f = base_params.c1_f
            r1_ohm = base_params.r1_ohm
    else:
        c1_f = base_params.c1_f
        r1_ohm = base_params.r1_ohm

    return BatteryParams(
        q_ah=q_ah,
        r0_ohm=r0_ohm,
        r1_ohm=r1_ohm,
        c1_f=c1_f,
        eta_charge=base_params.eta_charge,
        eta_discharge=base_params.eta_discharge,
        v_min=base_params.v_min,
        v_max=base_params.v_max,
    )


def compute_validation_metrics(reference: dict, simulated: dict) -> dict:
    metrics = {}
    for key in ["soc", "vp_v", "ocv_v", "vt_v"]:
        error = simulated[key] - reference[key]
        metrics[f"{key}_rmse"] = float(np.sqrt(np.mean(error * error)))
        metrics[f"{key}_mae"] = float(np.mean(np.abs(error)))
        metrics[f"{key}_max_abs"] = float(np.max(np.abs(error)))

    vt_error = simulated["vt_v"] - reference["vt_v"]
    metrics["vt_bias"] = float(np.mean(vt_error))
    metrics["final_soc_error"] = float(simulated["soc"][-1] - reference["soc"][-1])
    metrics["final_vt_error"] = float(simulated["vt_v"][-1] - reference["vt_v"][-1])
    metrics["energy_rmse_proxy"] = float(np.sqrt(np.mean((simulated["vt_v"] * simulated["i_a"] - reference["vt_v"] * reference["i_a"]) ** 2)))
    return metrics


def save_validation_plot(reference: dict, simulated: dict, path: str | Path, title: str) -> None:
    t = reference["t_s"]
    vt_error = simulated["vt_v"] - reference["vt_v"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(t, reference["i_a"], label="Reference", linewidth=1.6)
    axes[0].set_ylabel("Current (A)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, reference["vt_v"], label="Reference", linewidth=1.6)
    axes[1].plot(t, simulated["vt_v"], linestyle="--", label="Simulated", linewidth=1.4)
    axes[1].set_ylabel("V_t (V)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(t, reference["soc"], label="Reference", linewidth=1.6)
    axes[2].plot(t, simulated["soc"], linestyle="--", label="Simulated", linewidth=1.4)
    axes[2].set_ylabel("SOC")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(t, vt_error, color="#B23A48", linewidth=1.5)
    axes[3].axhline(0.0, linestyle="--", linewidth=1.0, color="gray")
    axes[3].set_ylabel("Error (V)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_validation_report(
    path: str | Path,
    reference_csv: str | Path,
    reference: dict,
    simulated: dict,
    fitted_params: BatteryParams,
    source_params: BatteryParams,
) -> None:
    report_path = Path(path)
    plot_path = report_path.with_suffix(".png")
    metrics = compute_validation_metrics(reference, simulated)

    report = {
        "reference_csv": str(reference_csv),
        "report_type": "battery_ecm_validation",
        "source_parameters": {
            "q_ah": source_params.q_ah,
            "r0_ohm": source_params.r0_ohm,
            "r1_ohm": source_params.r1_ohm,
            "c1_f": source_params.c1_f,
            "eta_charge": source_params.eta_charge,
            "eta_discharge": source_params.eta_discharge,
        },
        "fitted_parameters": {
            "q_ah": fitted_params.q_ah,
            "r0_ohm": fitted_params.r0_ohm,
            "r1_ohm": fitted_params.r1_ohm,
            "c1_f": fitted_params.c1_f,
            "eta_charge": fitted_params.eta_charge,
            "eta_discharge": fitted_params.eta_discharge,
        },
        "metrics": metrics,
        "plot_path": str(plot_path),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    save_validation_plot(reference, simulated, plot_path, "Silver-Ion ECM validation report")


def write_csv(path: str, data: dict) -> None:
    keys = ["t_s", "i_a", "soc", "vp_v", "ocv_v", "vt_v"]
    rows = zip(*(data[k] for k in keys))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def plot_results(data: dict, params: BatteryParams) -> None:
    t = data["t_s"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t, data["i_a"], linewidth=1.6)
    axes[0].set_ylabel("Current (A)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, data["soc"], linewidth=1.6)
    axes[1].set_ylabel("SOC")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, data["vp_v"], linewidth=1.6)
    axes[2].set_ylabel("V_p (V)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, data["vt_v"], label="Terminal voltage", linewidth=1.8)
    axes[3].axhline(params.v_min, linestyle="--", linewidth=1.0, label="V_min")
    axes[3].axhline(params.v_max, linestyle="--", linewidth=1.0, label="V_max")
    axes[3].set_ylabel("Voltage (V)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    fig.suptitle("Silver-Ion Battery ECM Simulation")
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Silver-ion battery ECM simulation")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--reference-csv", type=str, default=str(DEFAULT_REFERENCE_CSV), help="Reference CSV used for calibration and validation")
    parser.add_argument("--fit-reference", action="store_true", help="Fit ECM parameters against the reference CSV before simulation")
    parser.add_argument("--validation-report", type=str, default="", help="Write a JSON validation report and companion PNG plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_params = BatteryParams()
    params = BatteryParams()
    cfg = SimConfig()
    reference = None
    fitted_params = base_params

    if args.fit_reference or args.validation_report:
        reference = load_reference_csv(args.reference_csv)
        reference_dt = float(np.median(np.diff(reference["t_s"])) if reference["t_s"].size > 1 else 1.0)
        cfg = SimConfig(
            t_end_s=float(reference["t_s"][-1]),
            dt_s=reference_dt,
            soc0=float(reference["soc"][0]),
            vp0_v=float(reference["vp_v"][0]),
        )
        if args.fit_reference:
            fitted_params = fit_params_from_reference(reference, base_params)
            params = fitted_params
            print("Fitted ECM parameters from reference data")
            print(f"  q_ah: {params.q_ah:.6f}")
            print(f"  r0_ohm: {params.r0_ohm:.6f}")
            print(f"  r1_ohm: {params.r1_ohm:.6f}")
            print(f"  c1_f: {params.c1_f:.6f}")
        else:
            fitted_params = base_params

    data = run_simulation(params, cfg)

    if args.csv:
        write_csv(args.csv, data)
        print(f"Saved simulation data to: {args.csv}")

    print("Simulation complete")
    print(f"Final SOC: {data['soc'][-1]:.4f}")
    print(f"Final terminal voltage: {data['vt_v'][-1]:.4f} V")

    if args.validation_report:
        if reference is None:
            reference = load_reference_csv(args.reference_csv)
        validation_report = Path(args.validation_report)
        write_validation_report(validation_report, args.reference_csv, reference, data, fitted_params, base_params)
        print(f"Saved validation report: {validation_report}")
        print(f"Saved validation plot: {validation_report.with_suffix('.png')}")

    if not args.no_plot:
        plot_results(data, params)


if __name__ == "__main__":
    main()

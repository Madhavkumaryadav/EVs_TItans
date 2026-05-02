import argparse
import csv
import json
from dataclasses import dataclass

import numpy as np

from drive_cycles import get_cycle


@dataclass
class BatteryCellParams:
    q_ah: float = 2.0
    r0_ohm: float = 0.035
    r1_ohm: float = 0.020
    c1_f: float = 2200.0
    eta_charge: float = 0.995
    eta_discharge: float = 1.0
    v_min: float = 2.8
    v_max: float = 4.2


@dataclass
class PackConfig:
    n_series: int = 96
    n_parallel: int = 80
    soc0: float = 0.90
    vp0_v: float = 0.0


@dataclass
class VehicleParams:
    mass_kg: float = 1750.0
    crr: float = 0.012
    cd: float = 0.28
    frontal_area_m2: float = 2.2
    drivetrain_eff: float = 0.90
    regen_eff: float = 0.65
    accessory_kw: float = 0.7


@dataclass
class ThermalParams:
    t_ambient_c: float = 28.0
    t0_c: float = 30.0
    c_th_j_per_k: float = 180000.0
    r_th_k_per_w: float = 0.30
    warm_c: float = 45.0
    hot_c: float = 55.0
    critical_c: float = 65.0


@dataclass
class EVSimConfig:
    dt_s: float = 1.0
    duration_s: float = 3600.0
    cycle_name: str = "mixed"
    operation_mode: str = "drive"
    include_charging: bool = True
    charge_power_kw: float = 50.0
    cv_start_soc: float = 0.80
    charge_target_soc: float = 1.0


def ocv_from_soc(soc: np.ndarray) -> np.ndarray:
    eps = 1e-6
    s = np.clip(soc, eps, 1.0 - eps)
    a0, a1, a2, a3, a4 = 3.10, 1.00, -0.22, 0.05, -0.04
    return a0 + a1 * s + a2 * s * s + a3 * np.log(s) + a4 * np.log(1.0 - s)


def estimate_time_to_target_soc_hr(
    soc: float,
    target_soc: float,
    p_charge_kw: float,
    pack_energy_kwh: float,
    cv_start_soc: float,
) -> float:
    if soc >= target_soc or p_charge_kw <= 1e-6:
        return 0.0

    soc = float(np.clip(soc, 0.0, 1.0))
    target_soc = float(np.clip(target_soc, 0.0, 1.0))

    if target_soc <= cv_start_soc:
        e_needed = (target_soc - soc) * pack_energy_kwh
        return max(0.0, e_needed / p_charge_kw)

    if soc < cv_start_soc:
        e_cc = (cv_start_soc - soc) * pack_energy_kwh
        e_cv = (target_soc - cv_start_soc) * pack_energy_kwh
        return max(0.0, e_cc / p_charge_kw + e_cv / (0.5 * p_charge_kw))

    e_cv = (target_soc - soc) * pack_energy_kwh
    return max(0.0, e_cv / (0.5 * p_charge_kw))


def solve_current_from_power(power_w: float, u_v: float, r0_ohm: float) -> float:
    i = power_w / max(u_v, 1e-3)
    for _ in range(10):
        f = (u_v - r0_ohm * i) * i - power_w
        df = u_v - 2.0 * r0_ohm * i
        if abs(df) < 1e-9:
            break
        i_new = i - f / df
        if abs(i_new - i) < 1e-8:
            i = i_new
            break
        i = i_new
    return i


def simulate_ev(
    cell: BatteryCellParams,
    pack: PackConfig,
    vehicle: VehicleParams,
    thermal: ThermalParams,
    cfg: EVSimConfig,
) -> dict:
    mode = cfg.operation_mode.lower().strip()
    if mode == "charge":
        t = np.arange(0.0, cfg.duration_s + cfg.dt_s, cfg.dt_s)
        v = np.zeros_like(t)
        grade = np.zeros_like(t)
    else:
        cycle = get_cycle(cfg.cycle_name, cfg.dt_s, cfg.duration_s)
        t = cycle.t_s
        v = cycle.speed_mps
        grade = cycle.grade_rad
    n = t.size

    g = 9.81
    rho_air = 1.225

    q_pack_ah = cell.q_ah * pack.n_parallel
    q_pack_as = q_pack_ah * 3600.0
    r0_pack = (pack.n_series / pack.n_parallel) * cell.r0_ohm
    r1_pack = (pack.n_series / pack.n_parallel) * cell.r1_ohm
    c1_pack = (pack.n_parallel / pack.n_series) * cell.c1_f

    soc = np.zeros(n)
    vp = np.zeros(n)
    ocv = np.zeros(n)
    vt = np.zeros(n)
    i_batt = np.zeros(n)
    p_batt_kw = np.zeros(n)
    p_trac_kw = np.zeros(n)
    regen_kw = np.zeros(n)
    temp_c = np.zeros(n)
    range_km = np.zeros(n)
    t80_hr = np.zeros(n)
    t100_hr = np.zeros(n)
    charge_pct = np.zeros(n)
    efficiency_km_per_kwh = np.zeros(n)
    soh_pct = np.zeros(n)
    temp_status_code = np.zeros(n)
    thermal_alert = np.zeros(n)

    soc[0] = np.clip(pack.soc0, 0.0, 1.0)
    vp[0] = pack.vp0_v
    temp_c[0] = thermal.t0_c
    soh_pct[0] = 100.0

    distance_km = 0.0
    cum_energy_out_kwh = 0.0
    throughput_ah = 0.0
    thermal_stress_c_min = 0.0

    speed_prev = v[0]

    for k in range(n):
        vk = v[k]
        ak = (vk - speed_prev) / cfg.dt_s if k > 0 else 0.0
        speed_prev = vk

        if mode == "charge":
            f_roll = 0.0
            f_drag = 0.0
            f_grade = 0.0
            f_acc = 0.0
            p_mech_w = 0.0
            taper = 1.0
            if soc[k - 1] > cfg.cv_start_soc:
                taper = max(0.15, 1.0 - (soc[k - 1] - cfg.cv_start_soc) / max(1.0 - cfg.cv_start_soc, 1e-6))
            p_batt_w = -cfg.charge_power_kw * 1000.0 * taper
        else:
            f_roll = vehicle.mass_kg * g * vehicle.crr
            f_drag = 0.5 * rho_air * vehicle.cd * vehicle.frontal_area_m2 * vk * vk
            f_grade = vehicle.mass_kg * g * np.sin(grade[k])
            f_acc = vehicle.mass_kg * ak

            p_mech_w = vk * (f_roll + f_drag + f_grade + f_acc)
            if p_mech_w >= 0.0:
                p_batt_w = p_mech_w / max(vehicle.drivetrain_eff, 1e-3)
            else:
                p_batt_w = p_mech_w * vehicle.regen_eff

            p_batt_w += vehicle.accessory_kw * 1000.0

            if cfg.include_charging and k > int(0.9 * n):
                taper = 1.0
                if soc[k - 1] > cfg.cv_start_soc:
                    taper = max(0.20, 1.0 - (soc[k - 1] - cfg.cv_start_soc) / (1.0 - cfg.cv_start_soc))
                p_batt_w = -cfg.charge_power_kw * 1000.0 * taper

        soc_now = soc[k - 1] if k > 0 else soc[0]
        vp_now = vp[k - 1] if k > 0 else vp[0]

        ocv_pack = pack.n_series * float(ocv_from_soc(np.array([soc_now]))[0])
        u = ocv_pack - vp_now
        ib = solve_current_from_power(p_batt_w, u, r0_pack)

        eta = cell.eta_discharge if ib >= 0.0 else cell.eta_charge
        dsoc_dt = -(eta * ib) / q_pack_as
        dvp_dt = -(vp_now / (r1_pack * c1_pack)) + (ib / c1_pack)

        soc_next = float(np.clip(soc_now + cfg.dt_s * dsoc_dt, 0.0, 1.0))
        vp_next = vp_now + cfg.dt_s * dvp_dt

        vt_now = ocv_pack - ib * r0_pack - vp_now
        p_loss_w = (ib * ib) * (r0_pack + 0.25 * r1_pack)
        dtemp_dt = (p_loss_w - (temp_c[k - 1] - thermal.t_ambient_c) / thermal.r_th_k_per_w) / thermal.c_th_j_per_k if k > 0 else 0.0
        temp_next = (temp_c[k - 1] + cfg.dt_s * dtemp_dt) if k > 0 else temp_c[0]

        throughput_ah += abs(ib) * cfg.dt_s / 3600.0
        equivalent_full_cycles = throughput_ah / max(2.0 * q_pack_ah, 1e-6)
        thermal_stress_c_min += max(0.0, temp_next - thermal.t_ambient_c) * cfg.dt_s / 60.0
        soh_now = max(70.0, 100.0 - 0.18 * equivalent_full_cycles - 0.015 * thermal_stress_c_min)

        if temp_next >= thermal.critical_c:
            status = 3.0
            alert = 1.0
        elif temp_next >= thermal.hot_c:
            status = 2.0
            alert = 1.0
        elif temp_next >= thermal.warm_c:
            status = 1.0
            alert = 0.0
        else:
            status = 0.0
            alert = 0.0

        if k < n - 1:
            soc[k] = soc_now
            vp[k] = vp_now
            temp_c[k] = temp_c[k - 1] if k > 0 else temp_c[0]
            soc[k + 1] = soc_next
            vp[k + 1] = vp_next
            temp_c[k + 1] = temp_next
            soh_pct[k] = soh_now
            soh_pct[k + 1] = soh_now
            temp_status_code[k] = status
            temp_status_code[k + 1] = status
            thermal_alert[k] = alert
            thermal_alert[k + 1] = alert

        ocv[k] = ocv_pack
        vt[k] = vt_now
        i_batt[k] = ib
        p_batt_kw[k] = p_batt_w / 1000.0
        p_trac_kw[k] = p_mech_w / 1000.0
        regen_kw[k] = max(0.0, -p_mech_w / 1000.0)
        charge_pct[k] = soc_now * 100.0

        if vt_now > 1e-3 and ib > 0.0:
            cum_energy_out_kwh += (vt_now * ib) * cfg.dt_s / 3_600_000.0

        distance_km += vk * cfg.dt_s / 1000.0
        if cum_energy_out_kwh > 1e-5:
            efficiency_km_per_kwh[k] = distance_km / cum_energy_out_kwh
        elif k > 0:
            efficiency_km_per_kwh[k] = efficiency_km_per_kwh[k - 1]

        if distance_km > 0.1 and cum_energy_out_kwh > 1e-5:
            consumption_kwh_per_km = cum_energy_out_kwh / distance_km
            usable_energy_kwh = soc_now * q_pack_ah * ocv_pack / 1000.0
            range_km[k] = max(0.0, usable_energy_kwh / max(consumption_kwh_per_km, 1e-6))
        elif k > 0:
            range_km[k] = range_km[k - 1]

        p_charge_kw = max(0.0, -p_batt_w / 1000.0)
        pack_energy_kwh = q_pack_ah * ocv_pack / 1000.0
        t80_hr[k] = estimate_time_to_target_soc_hr(soc_now, 0.80, p_charge_kw, pack_energy_kwh, cfg.cv_start_soc)
        t100_hr[k] = estimate_time_to_target_soc_hr(soc_now, 1.00, p_charge_kw, pack_energy_kwh, cfg.cv_start_soc)

    summary = {
        "cycle": cfg.cycle_name,
        "operation_mode": mode,
        "car_status": "Charging" if mode == "charge" else "Driving",
        "plugged_in": bool(mode == "charge" or cfg.include_charging),
        "duration_s": float(t[-1]),
        "distance_km": float(distance_km),
        "final_soc_pct": float(charge_pct[-1]),
        "final_range_km": float(range_km[-1]),
        "final_temp_c": float(temp_c[-1]),
        "final_soh_pct": float(soh_pct[-1]),
        "final_efficiency_km_per_kwh": float(efficiency_km_per_kwh[-1]),
        "final_pack_voltage_v": float(vt[-1]),
        "initial_time_to_full_hr": float(t100_hr[0]) if t100_hr.size else 0.0,
        "final_time_to_full_hr": float(t100_hr[-1]) if t100_hr.size else 0.0,
        "max_temperature_c": float(np.max(temp_c)),
        "thermal_alert_count": int(np.sum(thermal_alert)),
        "max_battery_power_kw": float(np.max(p_batt_kw)),
        "min_battery_power_kw": float(np.min(p_batt_kw)),
        "max_regen_power_kw": float(np.max(regen_kw)),
        "energy_out_kwh": float(cum_energy_out_kwh),
    }

    return {
        "t_s": t,
        "speed_mps": v,
        "traction_power_kw": p_trac_kw,
        "battery_current_a": i_batt,
        "battery_voltage_v": vt,
        "battery_power_kw": p_batt_kw,
        "regen_power_kw": regen_kw,
        "soc": np.clip(soc, 0.0, 1.0),
        "charge_pct": charge_pct,
        "ocv_v": ocv,
        "vp_v": vp,
        "temperature_c": temp_c,
        "thermal_status_code": temp_status_code,
        "thermal_alert": thermal_alert,
        "efficiency_km_per_kwh": efficiency_km_per_kwh,
        "soh_pct": soh_pct,
        "range_km": range_km,
        "time_to_80_hr": t80_hr,
        "time_to_100_hr": t100_hr,
        "summary": summary,
    }


def write_timeseries_csv(path: str, out: dict) -> None:
    keys = [
        "t_s",
        "speed_mps",
        "traction_power_kw",
        "battery_current_a",
        "battery_voltage_v",
        "battery_power_kw",
        "regen_power_kw",
        "soc",
        "charge_pct",
        "ocv_v",
        "vp_v",
        "temperature_c",
        "thermal_status_code",
        "thermal_alert",
        "efficiency_km_per_kwh",
        "soh_pct",
        "range_km",
        "time_to_80_hr",
        "time_to_100_hr",
    ]

    rows = zip(*(out[k] for k in keys))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def write_summary_json(path: str, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EV simulation with silver-ion ECM pack model")
    parser.add_argument("--cycle", type=str, default="mixed", choices=["urban", "highway", "mixed"])
    parser.add_argument("--mode", type=str, default="drive", choices=["drive", "charge", "mixed"])
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--soc0", type=float, default=0.90)
    parser.add_argument("--no-charging", action="store_true")
    parser.add_argument("--charge-power-kw", type=float, default=50.0)
    parser.add_argument("--csv", type=str, default="results_ev.csv")
    parser.add_argument("--summary", type=str, default="summary_ev.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cell = BatteryCellParams()
    pack = PackConfig(soc0=args.soc0)
    veh = VehicleParams()
    thermal = ThermalParams()
    cfg = EVSimConfig(
        dt_s=args.dt_s,
        duration_s=args.duration_s,
        cycle_name=args.cycle,
        operation_mode=args.mode,
        include_charging=not args.no_charging,
        charge_power_kw=args.charge_power_kw,
    )

    out = simulate_ev(cell, pack, veh, thermal, cfg)
    write_timeseries_csv(args.csv, out)
    write_summary_json(args.summary, out["summary"])

    print("EV simulation complete")
    print(f"Mode: {args.mode}")
    print(f"Cycle: {args.cycle}")
    print(f"Saved time series: {args.csv}")
    print(f"Saved summary: {args.summary}")
    print(f"Final charge: {out['charge_pct'][-1]:.2f}%")
    print(f"Final range: {out['range_km'][-1]:.2f} km")
    print(f"Final battery temp: {out['temperature_c'][-1]:.2f} C")
    print(f"Final efficiency: {out['efficiency_km_per_kwh'][-1]:.2f} km/kWh")
    print(f"Initial time to full charge: {out['summary'].get('initial_time_to_full_hr', 0.0):.2f} hr")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
from pathlib import Path

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.figure import Figure
import numpy as np


LAST_ANIMATION = None
REPO_ROOT = Path(__file__).resolve().parents[1]


def save_animation(anim: mpl_animation.FuncAnimation, path: str, fps: int = 25) -> None:
    target = _resolve_path(path)
    suffix = target.suffix.lower()

    if suffix == ".gif":
        writer = mpl_animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        writer = mpl_animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError("Unsupported animation format. Use .gif or .mp4")

    anim.save(str(target), writer=writer)


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return candidate.resolve()


def read_csv(path: str) -> dict:
    resolved_path = _resolve_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    keys = rows[0].keys()
    out = {k: np.array([float(r[k]) for r in rows]) for k in keys}
    return out


def read_summary(path: str) -> dict:
    resolved_path = _resolve_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _thermal_label(code: int) -> str:
    return {0: "Normal", 1: "Warm", 2: "Hot", 3: "Critical"}.get(code, "Unknown")


def _bar_color(value: float, maximum: float) -> str:
    ratio = 0.0 if maximum <= 0.0 else float(np.clip(value / maximum, 0.0, 1.0))
    if ratio < 0.45:
        return "#2A9D8F"
    if ratio < 0.75:
        return "#E9C46A"
    return "#E76F51"


def _draw_progress_bar(ax, y: float, label: str, value: float, maximum: float, unit: str, color: str) -> None:
    ax.text(0.04, y + 0.045, f"{label}: {value:.2f} {unit}", fontsize=10.5, weight="bold", va="bottom")
    ax.add_patch(Rectangle((0.04, y), 0.86, 0.03, facecolor="#DDE5E8", edgecolor="none"))
    frac = 0.0 if maximum <= 0.0 else float(np.clip(value / maximum, 0.0, 1.0))
    ax.add_patch(Rectangle((0.04, y), 0.86 * frac, 0.03, facecolor=color, edgecolor="none"))


def _draw_status_chip(ax, x: float, y: float, text: str, color: str) -> None:
    ax.add_patch(Rectangle((x, y), 0.22, 0.07, facecolor=color, edgecolor="none", alpha=0.94))
    ax.text(x + 0.11, y + 0.035, text, color="white", fontsize=9.5, weight="bold", ha="center", va="center")


def _draw_metric_card(ax, x: float, y: float, w: float, h: float, title: str, value: str, subtitle: str, accent: str) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#101828", edgecolor="#233044", linewidth=1.0, alpha=0.98))
    ax.add_patch(Rectangle((x, y + h - 0.04), w, 0.04, facecolor=accent, edgecolor="none"))
    ax.text(x + 0.05, y + h - 0.08, title, color="#CBD5E1", fontsize=9.5, weight="bold", va="top")
    ax.text(x + 0.05, y + 0.095, value, color="white", fontsize=13.8, weight="bold", va="bottom")
    ax.text(x + 0.05, y + 0.03, subtitle, color="#94A3B8", fontsize=8.2, va="bottom")


def build_investor_dashboard(data: dict, summary: dict) -> Figure:
    t_min = data["t_s"] / 60.0
    mode_label = summary.get("operation_mode", "drive").capitalize()
    car_status = summary.get("car_status", "Driving")
    plugged_in = "Yes" if summary.get("plugged_in", False) else "No"
    final_soc = float(summary.get("final_soc_pct", float(data["charge_pct"][-1])))
    final_range = float(summary.get("final_range_km", float(data["range_km"][-1])))
    range_title = "Range"
    range_display = final_range
    range_subtitle = "Distance available per run"
    if final_range <= 0.01:
        range_title = "Projected range"
        range_display = max(1.0, final_soc * 3.8)
        range_subtitle = "At current SOC"
    final_temp = float(summary.get("final_temp_c", float(data["temperature_c"][-1])))
    final_soh = float(summary.get("final_soh_pct", float(data["soh_pct"][-1])))
    energy_out = float(summary.get("energy_out_kwh", 0.0))
    charge_eta = float(summary.get("initial_time_to_full_hr", 0.0))
    charge_eta_display = f"{charge_eta:.2f} hr" if charge_eta > 0.01 else "N/A"
    charge_eta_subtitle = "Estimated full-charge time" if charge_eta > 0.01 else "Drive mode demo"
    max_temp = float(summary.get("max_temperature_c", final_temp))
    alerts = int(summary.get("thermal_alert_count", 0))
    max_regen = float(summary.get("max_regen_power_kw", 0.0))
    final_power = float(data["battery_power_kw"][-1])
    final_speed = float(data["speed_mps"][-1] * 3.6)

    fig = plt.figure(figsize=(17, 10), constrained_layout=True, facecolor="#0B1020")
    gs = fig.add_gridspec(3, 4, height_ratios=[0.78, 1.1, 0.92], width_ratios=[1.15, 1.15, 1.0, 1.0])

    ax_header = fig.add_subplot(gs[0, :])
    ax_scene = fig.add_subplot(gs[1:, :2])
    ax_kpi = fig.add_subplot(gs[1, 2:])
    ax_chart = fig.add_subplot(gs[2, 2:])

    for axis in [ax_header, ax_scene, ax_kpi, ax_chart]:
        axis.set_facecolor("#0B1020")

    ax_header.axis("off")
    ax_header.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax_header.transAxes, facecolor="#0B1020", edgecolor="none"))
    ax_header.add_patch(Rectangle((0.0, 0.0), 1.0, 0.12, transform=ax_header.transAxes, facecolor="#D4AF37", edgecolor="none", alpha=0.95))
    ax_header.text(0.03, 0.72, "Silver-Ion EV Platform", color="white", fontsize=24, weight="bold", transform=ax_header.transAxes)
    ax_header.text(0.03, 0.48, "Investor visualization: lower downtime, predictable range, and fast-charge ready operation.", color="#CBD5E1", fontsize=12.2, transform=ax_header.transAxes)
    ax_header.text(0.03, 0.18, f"Mode: {mode_label}   |   Status: {car_status}   |   Plugged in: {plugged_in}", color="#94A3B8", fontsize=10.8, transform=ax_header.transAxes)
    ax_header.text(0.74, 0.62, f"Current power {final_power:+.1f} kW", color="#F8FAFC", fontsize=13, weight="bold", ha="right", transform=ax_header.transAxes)
    ax_header.text(0.74, 0.36, f"Speed {final_speed:.1f} km/h", color="#F8FAFC", fontsize=13, weight="bold", ha="right", transform=ax_header.transAxes)
    ax_header.text(0.74, 0.10, f"Energy out {energy_out:.2f} kWh", color="#F8FAFC", fontsize=13, weight="bold", ha="right", transform=ax_header.transAxes)

    ax_scene.set_xlim(0, 12)
    ax_scene.set_ylim(0, 7)
    ax_scene.axis("off")
    ax_scene.add_patch(Rectangle((0, 0), 12, 7, facecolor="#111827", edgecolor="none"))
    ax_scene.add_patch(Rectangle((0, 0), 12, 1.0, facecolor="#1F2937", edgecolor="none"))
    ax_scene.add_patch(Rectangle((0, 1.0), 12, 0.14, facecolor="#6B7280", edgecolor="none"))
    ax_scene.add_patch(Rectangle((0, 1.14), 12, 0.08, facecolor="#D4AF37", edgecolor="none", alpha=0.75))
    ax_scene.add_patch(Rectangle((0.7, 1.4), 10.6, 0.28, facecolor="#3F3F46", edgecolor="none"))

    ax_scene.text(0.55, 6.45, "Investable story", color="#F8FAFC", fontsize=15, weight="bold")
    ax_scene.text(0.55, 6.15, "The pack powers an EV while keeping range, temperature, and charging time visible.", color="#CBD5E1", fontsize=10.8)

    vehicle_x = 3.1
    vehicle_y = 2.0
    ax_scene.add_patch(Rectangle((vehicle_x, vehicle_y), 4.0, 1.25, facecolor="#1D4ED8", edgecolor="#93C5FD", linewidth=1.5))
    ax_scene.add_patch(Rectangle((vehicle_x + 0.5, vehicle_y + 0.68), 1.3, 0.35, facecolor="#BFDBFE", edgecolor="none", alpha=0.9))
    ax_scene.add_patch(Rectangle((vehicle_x + 2.0, vehicle_y + 0.68), 1.2, 0.35, facecolor="#BFDBFE", edgecolor="none", alpha=0.9))
    ax_scene.add_patch(Circle((vehicle_x + 0.8, vehicle_y - 0.08), 0.35, facecolor="#0F172A", edgecolor="white", linewidth=1.1))
    ax_scene.add_patch(Circle((vehicle_x + 3.2, vehicle_y - 0.08), 0.35, facecolor="#0F172A", edgecolor="white", linewidth=1.1))
    ax_scene.add_patch(Circle((vehicle_x + 0.8, vehicle_y - 0.08), 0.13, facecolor="#D1D5DB", edgecolor="none"))
    ax_scene.add_patch(Circle((vehicle_x + 3.2, vehicle_y - 0.08), 0.13, facecolor="#D1D5DB", edgecolor="none"))
    ax_scene.text(vehicle_x + 2.0, vehicle_y + 1.52, "EV platform", color="white", fontsize=14, weight="bold", ha="center")
    ax_scene.text(vehicle_x + 2.0, vehicle_y + 1.27, f"{range_title} {range_display:.1f} km", color="#E0F2FE", fontsize=12, ha="center")

    pack_x = 4.05
    pack_y = 0.55
    pack_width = 2.2
    ax_scene.add_patch(Rectangle((pack_x, pack_y), pack_width, 0.42, facecolor="#E5E7EB", edgecolor="#CBD5E1", linewidth=1.0))
    ax_scene.add_patch(Rectangle((pack_x + 0.04, pack_y + 0.05), pack_width * np.clip(final_soc / 100.0, 0.0, 1.0) - 0.08, 0.32, facecolor="#22C55E", edgecolor="none"))
    ax_scene.text(pack_x + pack_width / 2.0, pack_y + 0.21, f"SOC {final_soc:.1f}%", color="#0F172A", fontsize=10.5, weight="bold", ha="center", va="center")
    ax_scene.add_patch(Rectangle((pack_x + pack_width + 0.05, pack_y + 0.07), 0.12, 0.22, facecolor="#CBD5E1", edgecolor="none"))
    ax_scene.text(pack_x + pack_width / 2.0, pack_y - 0.12, "Silver-ion battery pack", color="#E2E8F0", fontsize=10.3, ha="center")

    charger_x = 0.95
    charger_y = 2.35
    ax_scene.add_patch(Rectangle((charger_x, charger_y), 1.0, 2.0, facecolor="#0EA5E9", edgecolor="#7DD3FC", linewidth=1.2))
    ax_scene.add_patch(Rectangle((charger_x + 0.18, charger_y + 1.35), 0.64, 0.3, facecolor="#F8FAFC", edgecolor="none"))
    ax_scene.add_patch(Rectangle((charger_x + 0.32, charger_y + 0.45), 0.34, 0.7, facecolor="#F8FAFC", edgecolor="none"))
    ax_scene.text(charger_x + 0.5, charger_y + 2.18, "Fast charge", color="#E0F2FE", fontsize=11.5, weight="bold", ha="center")
    ax_scene.text(charger_x + 0.5, charger_y - 0.2, f"ETA to full {charge_eta:.2f} hr", color="#E0F2FE", fontsize=9.8, ha="center")
    ax_scene.add_patch(FancyArrowPatch((2.0, 3.5), (3.05, 3.0), arrowstyle="-|>", mutation_scale=20, linewidth=3.0, color="#22C55E", alpha=0.95))
    ax_scene.add_patch(FancyArrowPatch((3.25, 1.0), (4.1, 0.78), arrowstyle="-|>", mutation_scale=18, linewidth=3.0, color="#F59E0B", alpha=0.95))

    ax_scene.add_patch(Rectangle((8.2, 4.1), 2.0, 1.3, facecolor="#0F172A", edgecolor="#334155", linewidth=1.0))
    ax_scene.text(9.2, 5.08, "Thermal control", color="#F8FAFC", fontsize=11.5, weight="bold", ha="center")
    ax_scene.text(9.2, 4.64, f"Peak temp {max_temp:.1f} C", color="#FDE68A", fontsize=11, ha="center")
    ax_scene.text(9.2, 4.28, f"Alerts {alerts}", color="#F8FAFC", fontsize=11, ha="center")
    ax_scene.add_patch(Rectangle((8.3, 3.15), 1.8, 0.3, facecolor="#E5E7EB", edgecolor="none"))
    ax_scene.add_patch(Rectangle((8.3, 3.15), 1.8 * np.clip(final_temp / 70.0, 0.0, 1.0), 0.3, facecolor="#F97316", edgecolor="none"))
    ax_scene.text(9.2, 3.62, f"Current temp {final_temp:.1f} C", color="#CBD5E1", fontsize=10.2, ha="center")

    ax_kpi.axis("off")
    ax_kpi.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax_kpi.transAxes, facecolor="#0B1020", edgecolor="#263244", linewidth=1.0))
    ax_kpi.text(0.04, 0.94, "Executive metrics", color="white", fontsize=14, weight="bold", transform=ax_kpi.transAxes)
    ax_kpi.text(0.04, 0.88, "What investors usually want to know first", color="#94A3B8", fontsize=9.8, transform=ax_kpi.transAxes)
    _draw_metric_card(ax_kpi, 0.04, 0.56, 0.28, 0.24, range_title, f"{range_display:.1f} km", range_subtitle, "#22C55E")
    _draw_metric_card(ax_kpi, 0.36, 0.56, 0.28, 0.24, "Charge time", charge_eta_display, charge_eta_subtitle, "#0EA5E9")
    _draw_metric_card(ax_kpi, 0.68, 0.56, 0.28, 0.24, "Battery health", f"{final_soh:.1f}%", "Health proxy retained by model", "#D4AF37")
    _draw_metric_card(ax_kpi, 0.04, 0.22, 0.28, 0.24, "Temperature", f"{final_temp:.1f} C", "Thermal operating point", "#F97316")
    _draw_metric_card(ax_kpi, 0.36, 0.22, 0.28, 0.24, "Energy used", f"{energy_out:.2f} kWh", "Trip consumption in this run", "#8B5CF6")
    _draw_metric_card(ax_kpi, 0.68, 0.22, 0.28, 0.24, "Thermal alerts", f"{alerts}", "Safety status during the demo", "#EF4444")

    ax_kpi.text(0.04, 0.08, f"Final SOH {final_soh:.1f}%   |   Max regen {max_regen:.1f} kW   |   Pack power {final_power:+.1f} kW", color="#CBD5E1", fontsize=10.2, transform=ax_kpi.transAxes)

    ax_chart.plot(t_min, data["charge_pct"], color="#22C55E", linewidth=2.0, label="SOC (%)")
    ax_chart.plot(t_min, data["temperature_c"], color="#F97316", linewidth=1.8, label="Temp (C)")
    ax_chart.plot(t_min, data["battery_power_kw"], color="#60A5FA", linewidth=1.8, label="Power (kW)")
    ax_chart.set_title("Run timeline", color="white", fontsize=12.5)
    ax_chart.set_xlabel("Time (min)", color="#CBD5E1")
    ax_chart.tick_params(colors="#CBD5E1")
    for spine in ax_chart.spines.values():
        spine.set_color("#334155")
    ax_chart.grid(alpha=0.18, color="#475569")
    ax_chart.legend(loc="upper right", frameon=False, labelcolor="#E2E8F0")

    return fig


def build_learning_dashboard(data: dict, summary: dict) -> Figure:
    t_min = data["t_s"] / 60.0
    current = -1
    mode_label = summary.get("operation_mode", "drive").capitalize()
    car_status = summary.get("car_status", "Driving")
    plugged_in = "Yes" if summary.get("plugged_in", False) else "No"
    status_code = int(round(float(data["thermal_status_code"][current])))
    status_name = _thermal_label(status_code)
    status_color = {0: "#2A9D8F", 1: "#E9C46A", 2: "#F4A261", 3: "#D62828"}.get(status_code, "#264653")

    soc_pct = float(data["charge_pct"][current])
    power_kw = float(data["battery_power_kw"][current])
    range_km = float(data["range_km"][current])
    temp_c = float(data["temperature_c"][current])
    soh_pct = float(data["soh_pct"][current])
    speed_kmh = float(data["speed_mps"][current] * 3.6)
    battery_current_a = float(data["battery_current_a"][current])
    battery_voltage_v = float(data["battery_voltage_v"][current])
    regen_kw = max(0.0, -float(data["traction_power_kw"][current]))

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[2.25, 1.05, 1.0], height_ratios=[1.2, 1.0, 0.85])

    ax_scene = fig.add_subplot(gs[0:2, 0:2])
    ax_controls = fig.add_subplot(gs[0:2, 2])
    ax_history = fig.add_subplot(gs[2, :])

    ax_scene.set_xlim(0, 10)
    ax_scene.set_ylim(0, 6)
    ax_scene.axis("off")
    ax_scene.set_facecolor("#F6F1E8")
    ax_scene.add_patch(Rectangle((0, 0), 10, 6, facecolor="#EAF5FF", edgecolor="none"))
    ax_scene.add_patch(Rectangle((0, 0), 10, 0.85, facecolor="#C9B79D", edgecolor="none"))
    ax_scene.add_patch(Rectangle((0, 0.85), 10, 0.18, facecolor="#555555", edgecolor="none"))
    ax_scene.add_patch(Rectangle((0.9, 1.35), 1.25, 0.25, facecolor="#64748B", edgecolor="none"))

    vehicle_x = 3.2
    vehicle_y = 1.3
    ax_scene.add_patch(Rectangle((vehicle_x, vehicle_y), 3.0, 0.95, facecolor="#1D3557", edgecolor="#0B1A2A", linewidth=1.5))
    ax_scene.add_patch(Rectangle((vehicle_x + 0.48, vehicle_y + 0.52), 1.1, 0.28, facecolor="#A8DADC", edgecolor="none", alpha=0.8))
    ax_scene.add_patch(Rectangle((vehicle_x + 1.72, vehicle_y + 0.52), 1.0, 0.28, facecolor="#A8DADC", edgecolor="none", alpha=0.8))
    ax_scene.add_patch(Circle((vehicle_x + 0.6, vehicle_y - 0.03), 0.25, facecolor="#111827", edgecolor="white", linewidth=1.1))
    ax_scene.add_patch(Circle((vehicle_x + 2.45, vehicle_y - 0.03), 0.25, facecolor="#111827", edgecolor="white", linewidth=1.1))
    ax_scene.add_patch(Circle((vehicle_x + 0.6, vehicle_y - 0.03), 0.10, facecolor="#D1D5DB", edgecolor="none"))
    ax_scene.add_patch(Circle((vehicle_x + 2.45, vehicle_y - 0.03), 0.10, facecolor="#D1D5DB", edgecolor="none"))
    ax_scene.add_patch(Rectangle((vehicle_x + 0.2, 2.35), 2.45, 0.3, facecolor="#0F172A", edgecolor="none", alpha=0.9))
    ax_scene.text(vehicle_x + 1.45, 2.51, f"EV mode: {mode_label}", color="white", fontsize=12, weight="bold", ha="center", va="center")

    battery_x = 3.45
    battery_y = 0.38
    ax_scene.add_patch(Rectangle((battery_x, battery_y), 2.55, 0.30, facecolor="#E5E7EB", edgecolor="#374151", linewidth=1.1))
    ax_scene.add_patch(Rectangle((battery_x + 2.56, battery_y + 0.05), 0.12, 0.20, facecolor="#374151", edgecolor="none"))
    ax_scene.add_patch(Rectangle((battery_x + 0.03, battery_y + 0.03), 2.49 * float(np.clip(soc_pct / 100.0, 0.0, 1.0)), 0.24, facecolor=_bar_color(soc_pct, 100.0), edgecolor="none"))
    ax_scene.text(battery_x + 1.27, battery_y + 0.15, f"SOC {soc_pct:.1f}%", color="black", fontsize=10.5, weight="bold", ha="center", va="center")

    charger_x = 0.8
    charger_y = 1.45
    ax_scene.add_patch(Rectangle((charger_x, charger_y), 0.85, 1.65, facecolor="#0EA5E9", edgecolor="#155E75", linewidth=1.2))
    ax_scene.add_patch(Rectangle((charger_x + 0.18, charger_y + 1.2), 0.49, 0.25, facecolor="#F8FAFC", edgecolor="none"))
    ax_scene.add_patch(Rectangle((charger_x + 0.28, charger_y + 0.35), 0.29, 0.55, facecolor="#F8FAFC", edgecolor="none"))
    ax_scene.text(charger_x + 0.42, charger_y + 1.93, "Silver-ion\nfast charge", ha="center", va="bottom", fontsize=10, color="#0F172A", weight="bold")
    ax_scene.text(charger_x + 0.42, charger_y + 0.07, f"{summary.get('initial_time_to_full_hr', 0.0):.2f} hr to full", ha="center", va="bottom", fontsize=9.5, color="#0F172A")

    cable_color = "#22C55E" if mode_label == "Charge" or plugged_in == "Yes" else "#64748B"
    ax_scene.add_patch(FancyArrowPatch((1.67, 2.2), (3.3, 1.9), arrowstyle="-|>", mutation_scale=18, linewidth=3.0, color=cable_color, alpha=0.95))

    ax_scene.add_patch(Circle((8.35, 4.2), 0.68, facecolor="#FFF7ED", edgecolor="#FB923C", linewidth=1.5))
    temp_ratio = float(np.clip((temp_c - 20.0) / 50.0, 0.0, 1.0))
    ax_scene.add_patch(Rectangle((8.15, 3.38), 0.4, 1.3, facecolor="#E5E7EB", edgecolor="#374151", linewidth=1.0))
    ax_scene.add_patch(Rectangle((8.19, 3.40), 0.32, 1.28 * temp_ratio, facecolor=_bar_color(temp_c, 65.0), edgecolor="none"))
    ax_scene.text(8.35, 4.2, f"{temp_c:.1f} C", fontsize=12.5, weight="bold", ha="center", va="center")
    ax_scene.text(8.35, 5.07, "Thermal\nwatch", fontsize=10.5, weight="bold", ha="center", va="center")

    power_arrow = 2.2 if abs(power_kw) < 20 else min(2.8, 1.5 + abs(power_kw) / 35.0)
    power_color = "#0EA5E9" if power_kw < 0 else "#EF4444"
    if power_kw < 0:
        ax_scene.add_patch(FancyArrowPatch((3.25, 2.05), (3.25 + power_arrow, 2.05), arrowstyle="-|>", mutation_scale=20, linewidth=3.1, color=power_color, alpha=0.92))
    else:
        ax_scene.add_patch(FancyArrowPatch((5.85, 2.05), (5.85 - power_arrow, 2.05), arrowstyle="-|>", mutation_scale=20, linewidth=3.1, color=power_color, alpha=0.92))

    ax_scene.text(5.0, 4.82, f"{power_kw:+.1f} kW pack power", fontsize=13, weight="bold", ha="center", va="center", color="#0F172A")
    ax_scene.text(5.0, 4.45, f"{speed_kmh:.1f} km/h | {range_km:.1f} km range", fontsize=11, ha="center", va="center", color="#0F172A")
    ax_scene.text(5.0, 4.14, f"{battery_voltage_v:.1f} V | {battery_current_a:+.1f} A", fontsize=10.5, ha="center", va="center", color="#334155")

    _draw_status_chip(ax_scene, 0.9, 5.0, f"{car_status.upper()}", "#0F766E" if car_status == "Driving" else "#2563EB")
    _draw_status_chip(ax_scene, 3.1, 5.0, f"PLUGGED IN: {plugged_in}", "#2563EB" if plugged_in == "Yes" else "#64748B")
    _draw_status_chip(ax_scene, 5.4, 5.0, status_name.upper(), status_color)

    ax_controls.set_xlim(0, 1)
    ax_controls.set_ylim(0, 1)
    ax_controls.axis("off")
    ax_controls.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.0, alpha=0.96))
    ax_controls.text(0.08, 0.94, "Control Area", color="white", fontsize=15, weight="bold", va="top")
    ax_controls.text(0.08, 0.89, "Educational overview", color="#CBD5E1", fontsize=10.2, va="top")

    _draw_progress_bar(ax_controls, 0.75, "Charge", soc_pct, 100.0, "%", _bar_color(soc_pct, 100.0))
    _draw_progress_bar(ax_controls, 0.67, "Range", range_km, max(range_km, 1.0), "km", "#2A9D8F")
    _draw_progress_bar(ax_controls, 0.59, "Temperature", temp_c, 65.0, "C", _bar_color(temp_c, 65.0))
    _draw_progress_bar(ax_controls, 0.51, "SOH", soh_pct, 100.0, "%", "#264653")

    ax_controls.text(0.08, 0.43, f"Mode: {mode_label}", color="white", fontsize=11, weight="bold")
    ax_controls.text(0.08, 0.38, f"Cycle: {summary.get('cycle', 'n/a')}", color="#E2E8F0", fontsize=10.2)
    ax_controls.text(0.08, 0.33, f"Status: {status_name}", color="#E2E8F0", fontsize=10.2)
    ax_controls.text(0.08, 0.28, f"Distance: {summary.get('distance_km', 0.0):.1f} km", color="#E2E8F0", fontsize=10.2)
    ax_controls.text(0.08, 0.23, f"Energy out: {summary.get('energy_out_kwh', 0.0):.2f} kWh", color="#E2E8F0", fontsize=10.2)
    ax_controls.text(0.08, 0.18, f"Max regen: {summary.get('max_regen_power_kw', 0.0):.1f} kW", color="#E2E8F0", fontsize=10.2)
    ax_controls.text(0.08, 0.13, f"Initial ETA to full: {summary.get('initial_time_to_full_hr', 0.0):.2f} hr", color="#E2E8F0", fontsize=10.2)
    ax_controls.text(0.08, 0.08, f"Current power: {power_kw:+.1f} kW", color="#E2E8F0", fontsize=10.2)
    ax_controls.text(0.08, 0.04, f"Current current: {battery_current_a:+.1f} A", color="#E2E8F0", fontsize=10.2)

    ax_history.plot(t_min, data["charge_pct"], label="SOC (%)", color="#0E7C7B", linewidth=2)
    ax_history.plot(t_min, data["battery_power_kw"], label="Pack power (kW)", color="#1D3557", linewidth=2)
    ax_history.plot(t_min, data["temperature_c"], label="Temp (C)", color="#E76F51", linewidth=2)
    ax_history.set_title("Learning timeline")
    ax_history.set_xlabel("Time (min)")
    ax_history.grid(alpha=0.28)
    ax_history.legend(loc="upper right", ncol=3, frameon=False)

    fig.suptitle("EV Learning Scene (Silver-Ion Pack Simulation)", fontsize=16, weight="bold")
    return fig


def build_dashboard(data: dict, summary: dict) -> Figure:
    t_min = data["t_s"] / 60.0
    status_labels = {0: "Normal", 1: "Warm", 2: "Hot", 3: "Critical"}
    mode_label = summary.get("operation_mode", "drive").capitalize()
    car_status = summary.get("car_status", "Driving")
    plugged_in = "Yes" if summary.get("plugged_in", False) else "No"

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 3)

    ax_soc = fig.add_subplot(gs[0, 0])
    ax_power = fig.add_subplot(gs[0, 1])
    ax_range = fig.add_subplot(gs[0, 2])
    ax_temp = fig.add_subplot(gs[1, 0])
    ax_charge = fig.add_subplot(gs[1, 1])
    ax_speed = fig.add_subplot(gs[1, 2])
    ax_health = fig.add_subplot(gs[2, 0])
    ax_alerts = fig.add_subplot(gs[2, 1])
    ax_text = fig.add_subplot(gs[2, 2])
    ax_footer = fig.add_subplot(gs[3, :])

    ax_soc.plot(t_min, data["charge_pct"], color="#0E7C7B", linewidth=2)
    ax_soc.set_title("Charge Percentage")
    ax_soc.set_ylabel("SOC (%)")
    ax_soc.set_xlabel("Time (min)")
    ax_soc.grid(alpha=0.3)

    ax_power.plot(t_min, data["battery_power_kw"], color="#1D3557", linewidth=2)
    ax_power.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    ax_power.set_title("Battery Power")
    ax_power.set_ylabel("kW")
    ax_power.set_xlabel("Time (min)")
    ax_power.grid(alpha=0.3)

    ax_range.plot(t_min, data["range_km"], color="#2A9D8F", linewidth=2)
    ax_range.set_title("Available Range")
    ax_range.set_ylabel("km")
    ax_range.set_xlabel("Time (min)")
    ax_range.grid(alpha=0.3)

    ax_temp.plot(t_min, data["temperature_c"], color="#E76F51", linewidth=2)
    ax_temp.axhline(45.0, linestyle="--", linewidth=1, color="orange", label="Warm")
    ax_temp.axhline(55.0, linestyle="--", linewidth=1, color="red", label="Hot")
    ax_temp.set_title("Battery Temperature")
    ax_temp.set_ylabel("C")
    ax_temp.set_xlabel("Time (min)")
    ax_temp.legend(loc="best")
    ax_temp.grid(alpha=0.3)

    ax_charge.plot(t_min, data["time_to_80_hr"] * 60.0, label="Time to 80%", linewidth=2)
    ax_charge.plot(t_min, data["time_to_100_hr"] * 60.0, label="Time to 100%", linewidth=2)
    ax_charge.set_title("Charging Time Remaining")
    ax_charge.set_ylabel("minutes")
    ax_charge.set_xlabel("Time (min)")
    ax_charge.legend(loc="best")
    ax_charge.grid(alpha=0.3)

    ax_speed.plot(t_min, data["speed_mps"] * 3.6, color="#6A4C93", linewidth=2)
    ax_speed.set_title("Vehicle Speed")
    ax_speed.set_ylabel("km/h")
    ax_speed.set_xlabel("Time (min)")
    ax_speed.grid(alpha=0.3)

    ax_health.plot(t_min, data["soh_pct"], color="#264653", linewidth=2)
    ax_health.axhline(90.0, linestyle="--", linewidth=1, color="orange", label="Health watch")
    ax_health.set_title("Battery SOH")
    ax_health.set_ylabel("SOH (%)")
    ax_health.set_xlabel("Time (min)")
    ax_health.set_ylim(68, 101)
    ax_health.legend(loc="best")
    ax_health.grid(alpha=0.3)

    ax_alerts.step(t_min, data["thermal_status_code"], where="post", color="#D62828", linewidth=2)
    ax_alerts.set_title("Thermal Status")
    ax_alerts.set_ylabel("Status")
    ax_alerts.set_xlabel("Time (min)")
    ax_alerts.set_yticks([0, 1, 2, 3])
    ax_alerts.set_yticklabels(["Normal", "Warm", "Hot", "Critical"])
    ax_alerts.grid(alpha=0.3)

    ax_text.axis("off")
    metric_text = (
        f"Cycle: {summary.get('cycle', 'n/a')}\n"
        f"Mode: {mode_label}\n"
        f"Car Status: {car_status}\n"
        f"Plugged In: {plugged_in}\n"
        f"Distance: {summary.get('distance_km', 0.0):.2f} km\n"
        f"Final Charge: {summary.get('final_soc_pct', 0.0):.2f}%\n"
        f"Final Range: {summary.get('final_range_km', 0.0):.2f} km\n"
        f"Final Battery Temp: {summary.get('final_temp_c', 0.0):.2f} C\n"
        f"Final SOH: {summary.get('final_soh_pct', 0.0):.2f}%\n"
        f"Full Charge ETA: {summary.get('initial_time_to_full_hr', 0.0):.2f} hr\n"
        f"Thermal Alerts: {summary.get('thermal_alert_count', 0)}\n"
        f"Max Temp: {summary.get('max_temperature_c', 0.0):.2f} C\n"
        f"Max Battery Power: {summary.get('max_battery_power_kw', 0.0):.2f} kW\n"
        f"Min Battery Power: {summary.get('min_battery_power_kw', 0.0):.2f} kW\n"
        f"Energy Out: {summary.get('energy_out_kwh', 0.0):.3f} kWh"
    )
    ax_text.text(0.01, 0.98, metric_text, va="top", fontsize=12)

    ax_footer.axis("off")
    final_status = int(round(float(data["thermal_status_code"][-1])))
    status_name = status_labels.get(final_status, "Unknown")
    footer_text = f"Final thermal status: {status_name}"
    ax_footer.text(0.01, 0.5, footer_text, va="center", fontsize=12, fontweight="bold")

    fig.suptitle("EV Dashboard (Silver-Ion Pack Simulation)", fontsize=16)
    return fig


def build_animated_dashboard(data: dict, summary: dict, interval_ms: int = 40) -> tuple[Figure, mpl_animation.FuncAnimation]:
    t_min = data["t_s"] / 60.0
    status_labels = {0: "Normal", 1: "Warm", 2: "Hot", 3: "Critical"}
    mode_label = summary.get("operation_mode", "drive").capitalize()
    car_status = summary.get("car_status", "Driving")
    plugged_in = "Yes" if summary.get("plugged_in", False) else "No"

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 3)

    ax_soc = fig.add_subplot(gs[0, 0])
    ax_power = fig.add_subplot(gs[0, 1])
    ax_range = fig.add_subplot(gs[0, 2])
    ax_temp = fig.add_subplot(gs[1, 0])
    ax_charge = fig.add_subplot(gs[1, 1])
    ax_speed = fig.add_subplot(gs[1, 2])
    ax_health = fig.add_subplot(gs[2, 0])
    ax_alerts = fig.add_subplot(gs[2, 1])
    ax_text = fig.add_subplot(gs[2, 2])
    ax_footer = fig.add_subplot(gs[3, :])

    for axis in [ax_soc, ax_power, ax_range, ax_temp, ax_charge, ax_speed, ax_health, ax_alerts]:
        axis.grid(alpha=0.3)

    ax_soc.set_title("Charge Percentage")
    ax_soc.set_ylabel("SOC (%)")
    ax_soc.set_xlabel("Time (min)")
    ax_soc.set_xlim(t_min[0], t_min[-1])
    ax_soc.set_ylim(0, 100)
    line_soc, = ax_soc.plot([], [], color="#0E7C7B", linewidth=2)

    ax_power.set_title("Battery Power")
    ax_power.set_ylabel("kW")
    ax_power.set_xlabel("Time (min)")
    ax_power.set_xlim(t_min[0], t_min[-1])
    power_min = min(float(np.min(data["battery_power_kw"])), -1.0)
    power_max = max(float(np.max(data["battery_power_kw"])), 1.0)
    ax_power.set_ylim(power_min * 1.15, power_max * 1.15)
    ax_power.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    line_power, = ax_power.plot([], [], color="#1D3557", linewidth=2)

    ax_range.set_title("Available Range")
    ax_range.set_ylabel("km")
    ax_range.set_xlabel("Time (min)")
    ax_range.set_xlim(t_min[0], t_min[-1])
    ax_range.set_ylim(0, max(1.0, float(np.max(data["range_km"])) * 1.1))
    line_range, = ax_range.plot([], [], color="#2A9D8F", linewidth=2)

    ax_temp.set_title("Battery Temperature")
    ax_temp.set_ylabel("C")
    ax_temp.set_xlabel("Time (min)")
    ax_temp.set_xlim(t_min[0], t_min[-1])
    ax_temp.set_ylim(min(20.0, float(np.min(data["temperature_c"])) - 2.0), max(60.0, float(np.max(data["temperature_c"])) + 2.0))
    ax_temp.axhline(45.0, linestyle="--", linewidth=1, color="orange", label="Warm")
    ax_temp.axhline(55.0, linestyle="--", linewidth=1, color="red", label="Hot")
    ax_temp.legend(loc="best")
    line_temp, = ax_temp.plot([], [], color="#E76F51", linewidth=2)

    ax_charge.set_title("Charging Time Remaining")
    ax_charge.set_ylabel("minutes")
    ax_charge.set_xlabel("Time (min)")
    ax_charge.set_xlim(t_min[0], t_min[-1])
    max_charge_min = max(float(np.max(data["time_to_100_hr"])) * 60.0, float(np.max(data["time_to_80_hr"])) * 60.0, 1.0)
    ax_charge.set_ylim(0, max_charge_min * 1.1)
    line_t80, = ax_charge.plot([], [], label="Time to 80%", linewidth=2)
    line_t100, = ax_charge.plot([], [], label="Time to 100%", linewidth=2)
    ax_charge.legend(loc="best")

    ax_speed.set_title("Vehicle Speed")
    ax_speed.set_ylabel("km/h")
    ax_speed.set_xlabel("Time (min)")
    ax_speed.set_xlim(t_min[0], t_min[-1])
    ax_speed.set_ylim(0, max(40.0, float(np.max(data["speed_mps"]) * 3.6) * 1.1))
    line_speed, = ax_speed.plot([], [], color="#6A4C93", linewidth=2)

    ax_health.set_title("Battery SOH")
    ax_health.set_ylabel("SOH (%)")
    ax_health.set_xlabel("Time (min)")
    ax_health.set_xlim(t_min[0], t_min[-1])
    ax_health.set_ylim(68, 101)
    ax_health.axhline(90.0, linestyle="--", linewidth=1, color="orange", label="Health watch")
    ax_health.legend(loc="best")
    line_soh, = ax_health.plot([], [], color="#264653", linewidth=2)

    ax_alerts.set_title("Thermal Status")
    ax_alerts.set_ylabel("Status")
    ax_alerts.set_xlabel("Time (min)")
    ax_alerts.set_xlim(t_min[0], t_min[-1])
    ax_alerts.set_yticks([0, 1, 2, 3])
    ax_alerts.set_yticklabels(["Normal", "Warm", "Hot", "Critical"])
    line_alerts, = ax_alerts.step([], [], where="post", color="#D62828", linewidth=2)

    ax_text.axis("off")
    metric_text = ax_text.text(0.01, 0.98, "", va="top", fontsize=12)

    ax_footer.axis("off")
    footer_text = ax_footer.text(0.01, 0.5, "", va="center", fontsize=12, fontweight="bold")

    fig.suptitle("EV Dashboard (Silver-Ion Pack Simulation)", fontsize=16)

    def update(frame: int):
        end = frame + 1
        x = t_min[:end]
        line_soc.set_data(x, data["charge_pct"][:end])
        line_power.set_data(x, data["battery_power_kw"][:end])
        line_range.set_data(x, data["range_km"][:end])
        line_temp.set_data(x, data["temperature_c"][:end])
        line_t80.set_data(x, data["time_to_80_hr"][:end] * 60.0)
        line_t100.set_data(x, data["time_to_100_hr"][:end] * 60.0)
        line_speed.set_data(x, data["speed_mps"][:end] * 3.6)
        line_soh.set_data(x, data["soh_pct"][:end])
        line_alerts.set_data(x, data["thermal_status_code"][:end])

        metric_text.set_text(
            f"Cycle: {summary.get('cycle', 'n/a')}\n"
            f"Mode: {mode_label}\n"
            f"Car Status: {car_status}\n"
            f"Plugged In: {plugged_in}\n"
            f"Distance: {summary.get('distance_km', 0.0):.2f} km\n"
            f"Charge: {float(data['charge_pct'][frame]):.2f}%\n"
            f"Range: {float(data['range_km'][frame]):.2f} km\n"
            f"Temp: {float(data['temperature_c'][frame]):.2f} C\n"
            f"SOH: {float(data['soh_pct'][frame]):.2f}%\n"
            f"Full Charge ETA: {summary.get('initial_time_to_full_hr', 0.0):.2f} hr\n"
            f"Thermal Alerts: {int(np.sum(data['thermal_alert'][:end]))}\n"
            f"Power: {float(data['battery_power_kw'][frame]):.2f} kW"
        )

        status_code = int(round(float(data["thermal_status_code"][frame])))
        footer_text.set_text(f"Final thermal status: {status_labels.get(status_code, 'Unknown')}")
        return (
            line_soc,
            line_power,
            line_range,
            line_temp,
            line_t80,
            line_t100,
            line_speed,
            line_soh,
            line_alerts,
            metric_text,
            footer_text,
        )

    anim = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=len(t_min),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    return fig, anim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashboard viewer for EV simulation outputs")
    parser.add_argument("--csv", type=str, default=str(REPO_ROOT / "results_ev.csv"))
    parser.add_argument("--summary", type=str, default=str(REPO_ROOT / "summary_ev.json"))
    parser.add_argument("--save", type=str, default="", help="Optional image path to save the dashboard")
    parser.add_argument("--animate", action="store_true", help="Play back the dashboard as an animation")
    parser.add_argument("--save-animation", type=str, default="", help="Optional animation output path (.gif or .mp4)")
    parser.add_argument("--fps", type=int, default=25, help="Animation export frame rate")
    parser.add_argument("--interval-ms", type=int, default=40, help="Animation frame interval in milliseconds")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_csv(args.csv)
    summary = read_summary(args.summary)
    global LAST_ANIMATION
    anim = None
    backend = plt.get_backend().lower()
    use_animation = args.animate or bool(args.save_animation)

    if use_animation:
        fig, LAST_ANIMATION = build_animated_dashboard(data, summary, interval_ms=args.interval_ms)
        anim = LAST_ANIMATION
    else:
        fig = build_dashboard(data, summary)

    if args.save:
        fig.savefig(args.save, dpi=160, bbox_inches="tight")
        print(f"Saved dashboard image: {args.save}")

    if args.save_animation and anim is not None:
        save_animation(anim, args.save_animation, fps=args.fps)
        print(f"Saved dashboard animation: {args.save_animation}")

    if not args.no_show and "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

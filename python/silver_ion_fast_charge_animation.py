import argparse
import csv
import json
from pathlib import Path

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_fast_charge_profile(
    total_minutes: float,
    dt_s: float,
    start_soc_pct: float,
    soc_80_min: float,
    soc_100_min: float,
    max_range_miles: float,
) -> dict:
    t_s = np.arange(0.0, total_minutes * 60.0 + dt_s, dt_s)
    t_min = t_s / 60.0

    start_soc = float(np.clip(start_soc_pct, 0.0, 100.0))
    soc = np.zeros_like(t_min)

    for idx, tm in enumerate(t_min):
        if tm <= soc_80_min:
            frac = tm / max(soc_80_min, 1e-9)
            soc[idx] = start_soc + frac * (80.0 - start_soc)
        elif tm <= soc_100_min:
            frac = (tm - soc_80_min) / max(soc_100_min - soc_80_min, 1e-9)
            soc[idx] = 80.0 + frac * 20.0
        else:
            soc[idx] = 100.0

    soc = np.clip(soc, 0.0, 100.0)

    dsoc_dt = np.gradient(soc, t_s, edge_order=1)
    power_kw = np.clip(420.0 * dsoc_dt, 20.0, None)
    range_miles = (soc / 100.0) * max_range_miles

    time_to_80_min = np.maximum(soc_80_min - t_min, 0.0)
    time_to_100_min = np.maximum(soc_100_min - t_min, 0.0)

    return {
        "t_s": t_s,
        "t_min": t_min,
        "soc_pct": soc,
        "power_kw": power_kw,
        "range_miles": range_miles,
        "time_to_80_min": time_to_80_min,
        "time_to_100_min": time_to_100_min,
    }


def save_csv(path: Path, data: dict) -> None:
    keys = [
        "t_s",
        "t_min",
        "soc_pct",
        "power_kw",
        "range_miles",
        "time_to_80_min",
        "time_to_100_min",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*(data[k] for k in keys)))


def save_summary(path: Path, data: dict, soc_80_min: float, soc_100_min: float, max_range_miles: float) -> None:
    summary = {
        "technology": "Silver-ion / silver-carbon solid-state (emerging)",
        "charge_to_80_min": float(soc_80_min),
        "charge_to_100_min": float(soc_100_min),
        "max_range_miles": float(max_range_miles),
        "final_soc_pct": float(data["soc_pct"][-1]),
        "final_range_miles": float(data["range_miles"][-1]),
        "availability_note": "Pilot-stage technology with broader adoption expected from 2026 onward.",
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def build_animation(data: dict, title: str, interval_ms: int) -> tuple[Figure, mpl_animation.FuncAnimation]:
    t_min = data["t_min"]

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_battery = fig.add_subplot(gs[:, 0])
    ax_soc = fig.add_subplot(gs[0, 1])
    ax_power = fig.add_subplot(gs[0, 2])
    ax_range = fig.add_subplot(gs[1, 1])
    ax_eta = fig.add_subplot(gs[1, 2])

    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax_battery.set_xlim(0, 1)
    ax_battery.set_ylim(0, 1)
    ax_battery.axis("off")

    shell = Rectangle((0.22, 0.12), 0.56, 0.76, linewidth=2.5, edgecolor="#1F2937", facecolor="none")
    cap = Rectangle((0.42, 0.89), 0.16, 0.06, linewidth=2.0, edgecolor="#1F2937", facecolor="#D1D5DB")
    fill_rect = Rectangle((0.24, 0.14), 0.52, 0.0, linewidth=0, facecolor="#22C55E")

    ax_battery.add_patch(shell)
    ax_battery.add_patch(cap)
    ax_battery.add_patch(fill_rect)

    battery_text = ax_battery.text(0.5, 0.05, "", ha="center", va="center", fontsize=12, fontweight="bold")
    status_text = ax_battery.text(0.5, 0.97, "", ha="center", va="top", fontsize=10)

    for ax in [ax_soc, ax_power, ax_range, ax_eta]:
        ax.grid(alpha=0.3)
        ax.set_xlim(t_min[0], t_min[-1])

    ax_soc.set_title("Charge Level")
    ax_soc.set_ylabel("SOC (%)")
    ax_soc.set_ylim(0, 100)
    line_soc, = ax_soc.plot([], [], color="#0EA5E9", linewidth=2.2)

    ax_power.set_title("Charging Power")
    ax_power.set_ylabel("Power (kW)")
    ax_power.set_ylim(0, max(50.0, float(np.max(data["power_kw"])) * 1.15))
    line_power, = ax_power.plot([], [], color="#F97316", linewidth=2.2)

    ax_range.set_title("Estimated EV Range")
    ax_range.set_xlabel("Time (min)")
    ax_range.set_ylabel("Range (miles)")
    ax_range.set_ylim(0, max(100.0, float(np.max(data["range_miles"])) * 1.08))
    line_range, = ax_range.plot([], [], color="#22C55E", linewidth=2.2)

    ax_eta.set_title("Charge ETA")
    ax_eta.set_xlabel("Time (min)")
    ax_eta.set_ylabel("Minutes Remaining")
    ax_eta.set_ylim(0, max(1.0, float(np.max(data["time_to_100_min"])) * 1.15))
    line_eta_80, = ax_eta.plot([], [], color="#2563EB", linewidth=2.0, label="To 80%")
    line_eta_100, = ax_eta.plot([], [], color="#7C3AED", linewidth=2.0, label="To 100%")
    ax_eta.legend(loc="upper right")

    def _status(soc_pct: float) -> str:
        if soc_pct < 80.0:
            return "Fast-charge phase"
        if soc_pct < 100.0:
            return "Top-up phase"
        return "Charge complete"

    def update(frame: int):
        end = frame + 1
        x = t_min[:end]

        line_soc.set_data(x, data["soc_pct"][:end])
        line_power.set_data(x, data["power_kw"][:end])
        line_range.set_data(x, data["range_miles"][:end])
        line_eta_80.set_data(x, data["time_to_80_min"][:end])
        line_eta_100.set_data(x, data["time_to_100_min"][:end])

        soc_now = float(data["soc_pct"][frame])
        fill_height = 0.74 * (soc_now / 100.0)
        fill_rect.set_height(fill_height)
        fill_color = "#22C55E" if soc_now < 80.0 else ("#84CC16" if soc_now < 100.0 else "#16A34A")
        fill_rect.set_facecolor(fill_color)

        battery_text.set_text(
            f"{soc_now:.1f}%  |  {float(data['range_miles'][frame]):.0f} mi"
        )
        status_text.set_text(
            f"{_status(soc_now)} | t={float(data['t_min'][frame]):.2f} min"
        )

        return (
            line_soc,
            line_power,
            line_range,
            line_eta_80,
            line_eta_100,
            fill_rect,
            battery_text,
            status_text,
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


def save_animation(anim: mpl_animation.FuncAnimation, path: Path, fps: int) -> None:
    suffix = path.suffix.lower()
    if suffix == ".gif":
        writer = mpl_animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        writer = mpl_animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError("Unsupported animation format. Use .gif or .mp4")

    anim.save(str(path), writer=writer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate silver-ion fast charging in a 9-10 minute scenario")
    parser.add_argument("--start-soc", type=float, default=20.0, help="Initial SOC percentage")
    parser.add_argument("--total-minutes", type=float, default=10.0, help="Animation length in minutes")
    parser.add_argument("--soc-80-min", type=float, default=9.0, help="Minutes to reach 80%% SOC")
    parser.add_argument("--soc-100-min", type=float, default=10.0, help="Minutes to reach 100%% SOC")
    parser.add_argument("--max-range-miles", type=float, default=600.0, help="Full-charge EV range assumption")
    parser.add_argument("--dt-s", type=float, default=1.0, help="Simulation timestep (s)")
    parser.add_argument("--interval-ms", type=int, default=35, help="Animation interval (ms)")
    parser.add_argument("--fps", type=int, default=30, help="Output animation frame rate")
    parser.add_argument("--csv", type=str, default=str(REPO_ROOT / "results_silver_fastcharge_anim.csv"))
    parser.add_argument("--summary", type=str, default=str(REPO_ROOT / "summary_silver_fastcharge_anim.json"))
    parser.add_argument("--save-animation", type=str, default="", help="Optional .gif or .mp4 output path")
    parser.add_argument("--no-show", action="store_true", help="Do not open animation window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.soc_80_min <= 0.0 or args.soc_100_min <= args.soc_80_min:
        raise ValueError("Expected 0 < --soc-80-min < --soc-100-min")

    data = build_fast_charge_profile(
        total_minutes=args.total_minutes,
        dt_s=args.dt_s,
        start_soc_pct=args.start_soc,
        soc_80_min=args.soc_80_min,
        soc_100_min=args.soc_100_min,
        max_range_miles=args.max_range_miles,
    )

    csv_path = Path(args.csv)
    summary_path = Path(args.summary)
    save_csv(csv_path, data)
    save_summary(summary_path, data, args.soc_80_min, args.soc_100_min, args.max_range_miles)

    title = "Silver-Ion Fast Charge Simulation (9-10 Minute Generation)"
    fig, anim = build_animation(data, title=title, interval_ms=args.interval_ms)

    print(f"Saved simulation CSV: {csv_path}")
    print(f"Saved summary JSON: {summary_path}")

    if args.save_animation:
        output_path = Path(args.save_animation)
        save_animation(anim, output_path, fps=args.fps)
        print(f"Saved animation: {output_path}")

    backend = plt.get_backend().lower()
    if not args.no_show and "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

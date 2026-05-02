import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ev_dashboard import build_animated_dashboard, build_dashboard, build_investor_dashboard, build_learning_dashboard, read_csv, read_summary, save_animation
from ev_simulation import EVSimConfig, BatteryCellParams, PackConfig, ThermalParams, VehicleParams, simulate_ev, write_summary_json, write_timeseries_csv


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EV simulation and open dashboard in one command")
    parser.add_argument("--cycle", type=str, default="mixed", choices=["urban", "highway", "mixed"])
    parser.add_argument("--mode", type=str, default="drive", choices=["drive", "charge", "mixed"])
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--soc0", type=float, default=0.90)
    parser.add_argument("--no-charging", action="store_true")
    parser.add_argument("--charge-power-kw", type=float, default=50.0)
    parser.add_argument("--csv", type=str, default=str(REPO_ROOT / "results_ev.csv"))
    parser.add_argument("--summary", type=str, default=str(REPO_ROOT / "summary_ev.json"))
    parser.add_argument("--layout", type=str, default="dashboard", choices=["dashboard", "scene", "investor"], help="Choose the standard dashboard, learning scene, or investor layout")
    parser.add_argument("--save", type=str, default="", help="Optional image path to save the dashboard")
    parser.add_argument("--animate", action="store_true", help="Play back the dashboard as an animation")
    parser.add_argument("--save-animation", type=str, default="", help="Optional animation output path (.gif or .mp4)")
    parser.add_argument("--fps", type=int, default=25, help="Animation export frame rate")
    parser.add_argument("--interval-ms", type=int, default=40, help="Animation frame interval in milliseconds")
    parser.add_argument("--no-show", action="store_true", help="Do not open the dashboard window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cell = BatteryCellParams()
    pack = PackConfig(soc0=args.soc0)
    vehicle = VehicleParams()
    thermal = ThermalParams()
    config = EVSimConfig(
        dt_s=args.dt_s,
        duration_s=args.duration_s,
        cycle_name=args.cycle,
        operation_mode=args.mode,
        include_charging=not args.no_charging,
        charge_power_kw=args.charge_power_kw,
    )

    out = simulate_ev(cell, pack, vehicle, thermal, config)
    write_timeseries_csv(args.csv, out)
    write_summary_json(args.summary, out["summary"])

    print("EV runner complete")
    print(f"Mode: {args.mode}")
    print(f"Saved time series: {args.csv}")
    print(f"Saved summary: {args.summary}")

    if args.no_show and not args.save and not args.save_animation:
        return

    data = read_csv(args.csv)
    summary = read_summary(args.summary)
    backend = plt.get_backend().lower()
    anim = None
    if args.layout == "scene":
        fig = build_learning_dashboard(data, summary)
    elif args.layout == "investor":
        fig = build_investor_dashboard(data, summary)
    elif args.animate or args.save_animation:
        fig, anim = build_animated_dashboard(data, summary, interval_ms=args.interval_ms)
    else:
        fig = build_dashboard(data, summary)

    if args.save:
        fig.savefig(args.save, dpi=160, bbox_inches="tight")
        print(f"Saved dashboard image: {args.save}")

    if args.save_animation and anim is not None:
        save_animation(anim, args.save_animation, fps=args.fps)
        print(f"Saved dashboard animation: {args.save_animation}")

    if not args.no_show:
        if "agg" not in backend:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()

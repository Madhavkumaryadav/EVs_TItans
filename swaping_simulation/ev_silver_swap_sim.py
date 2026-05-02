"""
Discrete-event simulation for:
"A Zero-Charging EV Infrastructure Model using Silver-Ion Battery Swapping Technology"

This script models:
1) Ag-Zn battery swapping station (BaaS model)
2) Conventional fast-charging station
3) Stockout behavior and buffer-sizing
4) Basic monthly cost comparison

Outputs:
- Console summary metrics
- PNG plots in ./outputs
- GIF animation of queue/inventory dynamics in ./outputs
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, TypedDict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import simpy
from matplotlib import animation


# -------------------------------
# Configuration Data Structures
# -------------------------------


@dataclass
class SwapConfig:
    sim_hours: float = 12.0
    arrival_rate_per_hour: float = 15.0
    swap_time_min: float = 3.0
    charge_time_min: float = 120.0
    bays: int = 2
    buffer_charged: int = 10
    charger_slots: int = 10
    random_seed: int = 42
    monitor_dt_min: float = 1.0


@dataclass
class FastChargeConfig:
    sim_hours: float = 12.0
    arrival_rate_per_hour: float = 15.0
    charge_time_min: float = 45.0
    chargers: int = 2
    random_seed: int = 42
    monitor_dt_min: float = 1.0


@dataclass
class CostConfig:
    battery_capacity_kwh: float = 20.0
    battery_unit_cost_usd: float = 12000.0
    cycle_life_swaps: int = 250
    monthly_swaps_per_user: float = 25.0
    station_service_fee_per_swap_usd: float = 1.8
    baas_margin: float = 0.15


class SwapRunResult(TypedDict):
    avg_wait_min: float
    avg_system_min: float
    avg_queue_len: float
    station_utilization: float
    battery_utilization: float
    stockout_probability: float
    throughput_swaps: int
    arrivals: int
    time_series: Dict[str, List[float]]
    wait_times: np.ndarray
    system_times: np.ndarray


class FastChargeRunResult(TypedDict):
    avg_wait_min: float
    avg_system_min: float
    avg_queue_len: float
    utilization: float
    arrivals: int
    time_series: Dict[str, List[float]]
    wait_times: np.ndarray
    system_times: np.ndarray


# -------------------------------
# Swap Station DES
# -------------------------------


class SwapStationDES:
    """SimPy model for a battery swapping station with charging in background.

    Assumption:
    - A vehicle acquires a swap bay first, then requests a charged battery.
    - If charged inventory is empty, bay is blocked until recharge completes.
    """

    def __init__(self, cfg: SwapConfig):
        self.cfg = cfg
        self.env = simpy.Environment()
        self.rng = np.random.default_rng(cfg.random_seed)

        self.bays = simpy.Resource(self.env, capacity=cfg.bays)
        self.chargers = simpy.Resource(self.env, capacity=cfg.charger_slots)
        self.charged_inventory = simpy.Container(
            self.env, init=cfg.buffer_charged, capacity=cfg.buffer_charged
        )
        self.empty_buffer = simpy.Store(self.env)

        self.wait_times_min: List[float] = []
        self.system_times_min: List[float] = []

        self.total_arrivals = 0
        self.stockout_arrivals = 0
        self.total_swaps = 0

        self.time_series: Dict[str, List[float]] = {
            "time_min": [],
            "queue_len": [],
            "busy_bays": [],
            "charged_inventory": [],
            "charging_busy": [],
            "empty_buffer": [],
        }

        self.bay_busy_time = 0.0

    @property
    def sim_time_min(self) -> float:
        return self.cfg.sim_hours * 60.0

    def vehicle_process(self, vehicle_id: int):
        arrival_t = self.env.now
        self.total_arrivals += 1

        if self.charged_inventory.level < 1:
            self.stockout_arrivals += 1

        with self.bays.request() as bay_req:
            yield bay_req
            yield self.charged_inventory.get(1)

            swap_start_t = self.env.now
            self.wait_times_min.append(swap_start_t - arrival_t)

            yield self.env.timeout(self.cfg.swap_time_min)

            self.bay_busy_time += self.cfg.swap_time_min
            self.total_swaps += 1
            self.system_times_min.append(self.env.now - arrival_t)

            yield self.empty_buffer.put(1)

    def arrival_process(self):
        vehicle_id = 0
        while self.env.now < self.sim_time_min:
            mean_inter_arrival_min = 60.0 / self.cfg.arrival_rate_per_hour
            inter_arrival = self.rng.exponential(mean_inter_arrival_min)
            yield self.env.timeout(inter_arrival)
            vehicle_id += 1
            self.env.process(self.vehicle_process(vehicle_id))

    def charger_worker(self, worker_id: int):
        while True:
            _ = yield self.empty_buffer.get()
            with self.chargers.request() as req:
                yield req
                yield self.env.timeout(self.cfg.charge_time_min)
                yield self.charged_inventory.put(1)

    def monitor_process(self):
        while self.env.now <= self.sim_time_min:
            self.time_series["time_min"].append(float(self.env.now))
            self.time_series["queue_len"].append(float(len(self.bays.queue)))
            self.time_series["busy_bays"].append(float(self.bays.count))
            self.time_series["charged_inventory"].append(float(self.charged_inventory.level))
            self.time_series["charging_busy"].append(float(self.chargers.count))
            self.time_series["empty_buffer"].append(float(len(self.empty_buffer.items)))
            yield self.env.timeout(self.cfg.monitor_dt_min)

    def run(self) -> SwapRunResult:
        for i in range(self.cfg.charger_slots):
            self.env.process(self.charger_worker(i + 1))
        self.env.process(self.arrival_process())
        self.env.process(self.monitor_process())
        self.env.run(until=self.sim_time_min)

        queue_arr = np.array(self.time_series["queue_len"], dtype=float)
        charging_busy_arr = np.array(self.time_series["charging_busy"], dtype=float)

        avg_wait = float(np.mean(self.wait_times_min)) if self.wait_times_min else 0.0
        avg_system = float(np.mean(self.system_times_min)) if self.system_times_min else 0.0
        avg_queue = float(np.mean(queue_arr)) if queue_arr.size else 0.0

        station_util = self.bay_busy_time / (self.cfg.bays * self.sim_time_min)
        battery_util = (
            float(np.mean(charging_busy_arr)) / self.cfg.buffer_charged
            if self.cfg.buffer_charged > 0 and charging_busy_arr.size
            else 0.0
        )

        return {
            "avg_wait_min": avg_wait,
            "avg_system_min": avg_system,
            "avg_queue_len": avg_queue,
            "station_utilization": station_util,
            "battery_utilization": battery_util,
            "stockout_probability": (
                self.stockout_arrivals / self.total_arrivals if self.total_arrivals else 0.0
            ),
            "throughput_swaps": self.total_swaps,
            "arrivals": self.total_arrivals,
            "time_series": self.time_series,
            "wait_times": np.array(self.wait_times_min, dtype=float),
            "system_times": np.array(self.system_times_min, dtype=float),
        }


# -------------------------------
# Fast Charging DES
# -------------------------------


class FastChargeDES:
    """Conventional fast charging where EVs queue for chargers."""

    def __init__(self, cfg: FastChargeConfig):
        self.cfg = cfg
        self.env = simpy.Environment()
        self.rng = np.random.default_rng(cfg.random_seed)

        self.chargers = simpy.Resource(self.env, capacity=cfg.chargers)

        self.wait_times_min: List[float] = []
        self.system_times_min: List[float] = []
        self.total_arrivals = 0

        self.time_series: Dict[str, List[float]] = {
            "time_min": [],
            "queue_len": [],
            "busy_chargers": [],
        }

        self.charger_busy_time = 0.0

    @property
    def sim_time_min(self) -> float:
        return self.cfg.sim_hours * 60.0

    def vehicle_process(self, vehicle_id: int):
        arrival_t = self.env.now
        self.total_arrivals += 1

        with self.chargers.request() as req:
            yield req
            service_start_t = self.env.now
            self.wait_times_min.append(service_start_t - arrival_t)

            yield self.env.timeout(self.cfg.charge_time_min)

            self.charger_busy_time += self.cfg.charge_time_min
            self.system_times_min.append(self.env.now - arrival_t)

    def arrival_process(self):
        vehicle_id = 0
        while self.env.now < self.sim_time_min:
            mean_inter_arrival_min = 60.0 / self.cfg.arrival_rate_per_hour
            inter_arrival = self.rng.exponential(mean_inter_arrival_min)
            yield self.env.timeout(inter_arrival)
            vehicle_id += 1
            self.env.process(self.vehicle_process(vehicle_id))

    def monitor_process(self):
        while self.env.now <= self.sim_time_min:
            self.time_series["time_min"].append(float(self.env.now))
            self.time_series["queue_len"].append(float(len(self.chargers.queue)))
            self.time_series["busy_chargers"].append(float(self.chargers.count))
            yield self.env.timeout(self.cfg.monitor_dt_min)

    def run(self) -> FastChargeRunResult:
        self.env.process(self.arrival_process())
        self.env.process(self.monitor_process())
        self.env.run(until=self.sim_time_min)

        queue_arr = np.array(self.time_series["queue_len"], dtype=float)

        avg_wait = float(np.mean(self.wait_times_min)) if self.wait_times_min else 0.0
        avg_system = float(np.mean(self.system_times_min)) if self.system_times_min else 0.0
        avg_queue = float(np.mean(queue_arr)) if queue_arr.size else 0.0

        utilization = self.charger_busy_time / (self.cfg.chargers * self.sim_time_min)

        return {
            "avg_wait_min": avg_wait,
            "avg_system_min": avg_system,
            "avg_queue_len": avg_queue,
            "utilization": utilization,
            "arrivals": self.total_arrivals,
            "time_series": self.time_series,
            "wait_times": np.array(self.wait_times_min, dtype=float),
            "system_times": np.array(self.system_times_min, dtype=float),
        }


# -------------------------------
# Analysis Helpers
# -------------------------------


def run_replications_swap(cfg: SwapConfig, n_rep: int) -> Dict[str, float]:
    metrics: List[SwapRunResult] = []
    for i in range(n_rep):
        local_cfg = SwapConfig(**{**cfg.__dict__, "random_seed": cfg.random_seed + i})
        out = SwapStationDES(local_cfg).run()
        metrics.append(out)

    keys = [
        "avg_wait_min",
        "avg_system_min",
        "avg_queue_len",
        "station_utilization",
        "battery_utilization",
        "stockout_probability",
    ]
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys}


def find_min_buffer_for_stockout(
    base_cfg: SwapConfig,
    target_stockout: float = 0.05,
    b_min: int = 2,
    b_max: int = 40,
    n_rep: int = 15,
) -> Tuple[int | None, List[Tuple[int, float]]]:
    trajectory: List[Tuple[int, float]] = []
    for b in range(b_min, b_max + 1):
        cfg = SwapConfig(
            **{
                **base_cfg.__dict__,
                "buffer_charged": b,
                "charger_slots": max(base_cfg.charger_slots, b),
            }
        )
        avg = run_replications_swap(cfg, n_rep=n_rep)
        p_stockout = avg["stockout_probability"]
        trajectory.append((b, p_stockout))
        if p_stockout < target_stockout:
            return b, trajectory
    return None, trajectory


def stockout_vs_arrival_rate(
    base_cfg: SwapConfig,
    arrival_rates: np.ndarray,
    buffers: List[int],
    n_rep: int = 10,
) -> Dict[int, List[float]]:
    result: Dict[int, List[float]] = {b: [] for b in buffers}
    for b in buffers:
        for lam in arrival_rates:
            cfg = SwapConfig(
                **{
                    **base_cfg.__dict__,
                    "arrival_rate_per_hour": float(lam),
                    "buffer_charged": b,
                    "charger_slots": max(base_cfg.charger_slots, b),
                }
            )
            avg = run_replications_swap(cfg, n_rep=n_rep)
            result[b].append(avg["stockout_probability"])
    return result


def compute_costs(cost_cfg: CostConfig) -> Dict[str, float]:
    depreciation_per_swap = cost_cfg.battery_unit_cost_usd / cost_cfg.cycle_life_swaps
    monthly_depreciation = depreciation_per_swap * cost_cfg.monthly_swaps_per_user

    monthly_baas_fee = (
        monthly_depreciation + cost_cfg.station_service_fee_per_swap_usd * cost_cfg.monthly_swaps_per_user
    ) * (1.0 + cost_cfg.baas_margin)

    monthly_purchase_equivalent = monthly_depreciation

    months_to_replace = cost_cfg.cycle_life_swaps / max(cost_cfg.monthly_swaps_per_user, 1e-6)

    return {
        "depreciation_per_swap_usd": depreciation_per_swap,
        "monthly_baas_fee_usd": monthly_baas_fee,
        "monthly_purchase_equivalent_usd": monthly_purchase_equivalent,
        "upfront_purchase_usd": cost_cfg.battery_unit_cost_usd,
        "estimated_replacement_interval_months": months_to_replace,
    }


# -------------------------------
# Plotting / Animation
# -------------------------------


def plot_wait_distribution(
    swap_wait: np.ndarray, fast_wait: np.ndarray, out_dir: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(np.max(swap_wait, initial=1), np.max(fast_wait, initial=1)), 35)
    ax.hist(swap_wait, bins=bins, alpha=0.6, label="Swapping", density=True)
    ax.hist(fast_wait, bins=bins, alpha=0.6, label="Fast charging", density=True)
    ax.set_xlabel("Waiting time (min)")
    ax.set_ylabel("Density")
    ax.set_title("Waiting Time Distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    out_file = out_dir / "waiting_time_distribution.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    return out_file


def plot_queue_over_time(
    swap_series: Dict[str, List[float]],
    fast_series: Dict[str, List[float]],
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(swap_series["time_min"], swap_series["queue_len"], label="Swap queue", linewidth=2)
    ax.plot(fast_series["time_min"], fast_series["queue_len"], label="Fast-charge queue", linewidth=2)
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("Queue length (vehicles)")
    ax.set_title("Queue Length Over Time")
    ax.legend()
    ax.grid(alpha=0.25)
    out_file = out_dir / "queue_length_over_time.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    return out_file


def plot_stockout_vs_arrival(
    arrival_rates: np.ndarray,
    stockout_curves: Dict[int, List[float]],
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for b, probs in stockout_curves.items():
        ax.plot(arrival_rates, probs, marker="o", label=f"Buffer B={b}")

    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.4, label="5% threshold")
    ax.set_xlabel("Arrival rate lambda (vehicles/hour)")
    ax.set_ylabel("Stockout probability")
    ax.set_title("Stockout Probability vs Arrival Rate")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    out_file = out_dir / "stockout_probability_vs_arrival_rate.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    return out_file


def make_station_animation(swap_series: Dict[str, List[float]], out_dir: Path) -> Path | None:
    times = np.array(swap_series["time_min"], dtype=float)
    q = np.array(swap_series["queue_len"], dtype=float)
    c = np.array(swap_series["charged_inventory"], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Station Dynamics Animation")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Count")
    ax.set_xlim(float(np.min(times, initial=0)), float(np.max(times, initial=1)))
    ymax = max(float(np.max(q, initial=1)), float(np.max(c, initial=1))) + 2
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.25)

    queue_line, = ax.plot([], [], linewidth=2, label="Queue length")
    inv_line, = ax.plot([], [], linewidth=2, label="Charged inventory")
    cursor = ax.axvline(0, color="black", alpha=0.3, linestyle="--")
    ax.legend()

    def init():
        queue_line.set_data([], [])
        inv_line.set_data([], [])
        cursor.set_xdata([0, 0])
        return queue_line, inv_line, cursor

    def update(frame: int):
        queue_line.set_data(times[: frame + 1], q[: frame + 1])
        inv_line.set_data(times[: frame + 1], c[: frame + 1])
        t = times[frame]
        cursor.set_xdata([t, t])
        return queue_line, inv_line, cursor

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(times),
        interval=80,
        blit=True,
        repeat=False,
    )

    out_file = out_dir / "station_dynamics.gif"
    try:
        ani.save(out_file, writer="pillow", fps=15)
    except Exception:
        plt.close(fig)
        return None

    plt.close(fig)
    return out_file


# -------------------------------
# Main
# -------------------------------


def format_minutes(x: float) -> str:
    return f"{x:.2f} min"


def main():
    parser = argparse.ArgumentParser(description="EV silver-ion battery swap DES model")
    parser.add_argument("--sim-hours", type=float, default=12.0)
    parser.add_argument("--arrival-rate", type=float, default=15.0)
    parser.add_argument("--swap-bays", type=int, default=2)
    parser.add_argument("--buffer", type=int, default=10)
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    swap_cfg = SwapConfig(
        sim_hours=args.sim_hours,
        arrival_rate_per_hour=args.arrival_rate,
        bays=args.swap_bays,
        buffer_charged=args.buffer,
        charger_slots=max(args.buffer, 4),
        random_seed=101,
    )
    fast_cfg = FastChargeConfig(
        sim_hours=args.sim_hours,
        arrival_rate_per_hour=args.arrival_rate,
        chargers=args.swap_bays,
        random_seed=202,
    )

    swap_model = SwapStationDES(swap_cfg)
    fast_model = FastChargeDES(fast_cfg)

    swap_res = swap_model.run()
    fast_res = fast_model.run()

    min_b, trajectory = find_min_buffer_for_stockout(
        swap_cfg,
        target_stockout=0.05,
        b_min=2,
        b_max=30,
        n_rep=max(8, args.replications),
    )

    arrival_sweep = np.arange(6.0, 26.0, 2.0)
    stockout_curves = stockout_vs_arrival_rate(
        swap_cfg,
        arrival_rates=arrival_sweep,
        buffers=[6, 8, 10, 12, 14],
        n_rep=max(6, args.replications // 2),
    )

    cost_res = compute_costs(CostConfig())

    fig1 = plot_wait_distribution(
        swap_wait=swap_res["wait_times"],
        fast_wait=fast_res["wait_times"],
        out_dir=out_dir,
    )
    fig2 = plot_queue_over_time(
        swap_series=swap_res["time_series"],
        fast_series=fast_res["time_series"],
        out_dir=out_dir,
    )
    fig3 = plot_stockout_vs_arrival(
        arrival_rates=arrival_sweep,
        stockout_curves=stockout_curves,
        out_dir=out_dir,
    )
    gif_file = make_station_animation(swap_series=swap_res["time_series"], out_dir=out_dir)

    print("\n=== Silver-Ion Swapping Scenario ===")
    print(f"Arrivals: {int(swap_res['arrivals'])}")
    print(f"Throughput swaps: {int(swap_res['throughput_swaps'])}")
    print(f"Average waiting time: {format_minutes(swap_res['avg_wait_min'])}")
    print(f"Average system time (wait+swap): {format_minutes(swap_res['avg_system_min'])}")
    print(f"Average queue length: {swap_res['avg_queue_len']:.2f}")
    print(f"Station utilization: {100.0 * swap_res['station_utilization']:.1f}%")
    print(f"Battery utilization: {100.0 * swap_res['battery_utilization']:.1f}%")
    print(f"Stockout probability: {100.0 * swap_res['stockout_probability']:.2f}%")

    print("\n=== Conventional Fast-Charging Scenario ===")
    print(f"Arrivals: {int(fast_res['arrivals'])}")
    print(f"Average waiting time: {format_minutes(fast_res['avg_wait_min'])}")
    print(f"Average system time (wait+charge): {format_minutes(fast_res['avg_system_min'])}")
    print(f"Average queue length: {fast_res['avg_queue_len']:.2f}")
    print(f"Charger utilization: {100.0 * fast_res['utilization']:.1f}%")

    print("\n=== Buffer Sizing Result ===")
    if min_b is None:
        print("No buffer in searched range satisfied stockout < 5%.")
    else:
        print(f"Minimum buffer B for stockout < 5%: {min_b}")

    print("\n=== Cost Analysis (BaaS vs Purchase) ===")
    print(f"Battery upfront purchase: ${cost_res['upfront_purchase_usd']:.0f}")
    print(f"Depreciation per swap: ${cost_res['depreciation_per_swap_usd']:.2f}")
    print(f"Estimated monthly BaaS fee: ${cost_res['monthly_baas_fee_usd']:.2f}")
    print(f"Monthly equivalent purchase burden: ${cost_res['monthly_purchase_equivalent_usd']:.2f}")
    print(
        "Estimated replacement interval at 25 swaps/month: "
        f"{cost_res['estimated_replacement_interval_months']:.2f} months"
    )

    print("\n=== Figures ===")
    print(f"Waiting distribution: {fig1}")
    print(f"Queue over time: {fig2}")
    print(f"Stockout vs arrival rate: {fig3}")
    if gif_file is not None:
        print(f"Animation: {gif_file}")
    else:
        print("Animation: could not save GIF (install pillow and retry).")

    # Save buffer trajectory as plain text for paper appendix.
    traj_file = out_dir / "buffer_stockout_trajectory.csv"
    with traj_file.open("w", encoding="utf-8") as f:
        f.write("buffer_B,stockout_probability\n")
        for b, p in trajectory:
            f.write(f"{b},{p:.6f}\n")
    print(f"Buffer trajectory table: {traj_file}")


if __name__ == "__main__":
    main()

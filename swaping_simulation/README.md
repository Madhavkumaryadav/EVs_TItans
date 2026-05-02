# Zero-Charging EV Infrastructure Simulation (Silver-Ion Battery Swapping)

This repository contains a run-ready discrete event simulation (DES) for:

- Silver-zinc battery swapping station (Battery-as-a-Service)
- Conventional fast charging comparison
- Buffer sizing against stockout target
- Basic cost model for BaaS vs battery ownership

## Model Assumptions

- Battery chemistry: silver-zinc (Ag-Zn)
- Battery pack size: 20 kWh
- Swap time: 3 minutes
- Station charging time (background): 2 hours
- Fast charging benchmark: 45 minutes
- Arrivals follow a Poisson process (exponential inter-arrivals)

## Files

- `ev_silver_swap_sim.py`: Main simulation code (SimPy + matplotlib)
- `requirements.txt`: Python dependencies
- `outputs/`: Generated figures and animation after running

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python ev_silver_swap_sim.py
```

Optional parameters:

```bash
python ev_silver_swap_sim.py --sim-hours 12 --arrival-rate 15 --swap-bays 2 --buffer 10 --replications 10
```

## Output Metrics

The script prints and/or saves:

- Average waiting time and average system time (wait + service)
- Queue length and utilization
- Stockout probability
- Minimum buffer size to keep stockout < 5%
- Monthly cost estimates (BaaS vs ownership)
- Figures:
  - waiting_time_distribution.png
  - queue_length_over_time.png
  - stockout_probability_vs_arrival_rate.png
  - station_dynamics.gif
  - buffer_stockout_trajectory.csv

## Notes

- The current model assumes a vehicle acquires a swap bay, then receives a charged battery.
- If inventory is empty, bay blocking can occur. This is conservative for congestion analysis.
- You can extend this to network-level multi-station modeling for future work.

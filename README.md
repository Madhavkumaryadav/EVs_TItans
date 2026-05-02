# Silver-Ion Battery Simulation (MATLAB and Python)

## 1. Overview

This project provides a practical, physics-inspired silver-ion battery simulation in both:

- Python: `python/silver_ion_battery_sim.py`
- MATLAB: `matlab/silver_ion_battery_sim.m`

The model is useful for:

- State of charge (SOC) tracking
- Terminal voltage prediction under charge/discharge current
- Dynamic polarization behavior

## 2. Model Concept

The battery is represented by an Equivalent Circuit Model (ECM) with:

- Open-circuit voltage: `OCV(SOC)`
- Series resistance: `R0`
- Polarization branch: `R1 || C1`

States:

- `soc` in `[0, 1]`
- `vp` polarization voltage

Sign convention:

- `I > 0`: discharge
- `I < 0`: charge

## 3. Governing Equations

SOC dynamics:

`dsoc/dt = -(eta * I) / Q_As`, where `Q_As = Q_Ah * 3600`

Polarization dynamics:

`dvp/dt = -(vp / (R1 * C1)) + (I / C1)`

Terminal voltage:

`Vt = OCV(soc) - I * R0 - vp`

## 4. Default Parameters

- `Q_Ah = 2.0`
- `R0 = 0.035 Ohm`
- `R1 = 0.020 Ohm`
- `C1 = 2200.0 F`
- `eta_charge = 0.995`
- `eta_discharge = 1.0`
- `V_min = 2.8 V`
- `V_max = 4.2 V`

These are engineering starter values for workflow demonstration.

## 5. Current Profile

- `0-600 s`: rest (`I = 0.0 A`)
- `600-4200 s`: discharge (`I = +1.0 A`)
- `4200-4800 s`: rest (`I = 0.0 A`)
- `4800-7800 s`: charge (`I = -0.8 A`)
- `7800-9000 s`: rest (`I = 0.0 A`)

## 6. Project Structure

```text
silver_ion_battery/
	README.md
	requirements.txt
	results.csv
	docs/
		silver-ion-battery-making-guide.md
	python/
		silver_ion_battery_sim.py
	matlab/
		silver_ion_battery_sim.m
```

## 7. Python Usage

From project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python python/silver_ion_battery_sim.py
```

Optional CSV output:

```bash
python python/silver_ion_battery_sim.py --csv results.csv
```

Headless run:

```bash
python python/silver_ion_battery_sim.py --no-plot --csv results.csv
```

## 8. MATLAB Usage

From MATLAB:

```matlab
run('matlab/silver_ion_battery_sim.m')
```

## 9. Documentation for Making Process

The making documentation is now separated into individual files in:

- `docs/README.md`

Quick links:

- `docs/01-purpose-safety.md`
- `docs/02-concepts-chemicals.md`
- `docs/03-quantity-guidance.md`
- `docs/04-step-by-step-process.md`
- `docs/05-architecture-and-template.md`
- `docs/06-detailed-phases.md`
- `docs/07-failure-modes-and-update-layout.md`
- `docs/08-validation-checklist.md`
- `docs/sim_docs.md`

## 10. Disclaimer

This project is for educational and engineering prototyping use. For physical battery development, use verified chemistry data, proper safety controls, and expert supervision.

## 11. Quick Start (Python and MATLAB)

### Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python python/silver_ion_battery_sim.py --csv results.csv
```

### MATLAB

```matlab
run('matlab/silver_ion_battery_sim.m')
```

Expected simulation outputs include SOC, OCV, polarization voltage, and terminal voltage traces.

## 12. EV Simulation and Dashboard Quick Start

Run EV simulation with silver-ion pack model:

```bash
python python/ev_runner.py --mode drive --cycle mixed --duration-s 3600 --csv results_ev.csv --summary summary_ev.json
```

Open dashboard with key metrics (charge percentage, charging time, power, range, temperature):

```bash
python python/ev_dashboard.py --csv results_ev.csv --summary summary_ev.json
```

Open the PhET-style learning scene:

```bash
python show_ev_output.py --layout scene --mode drive --cycle mixed --duration-s 3600 --csv results_ev.csv --summary summary_ev.json
```

Open the investor presentation view:

```bash
python show_ev_output.py --layout investor --mode drive --cycle mixed --duration-s 3600 --csv results_ev.csv --summary summary_ev.json --save investor_demo.png
```

Play the dashboard as a live animation:

```bash
python python/ev_dashboard.py --csv results_ev.csv --summary summary_ev.json --animate
```

Run the full EV flow in one command:

```bash
python python/ev_runner.py --mode drive --cycle mixed --duration-s 3600 --csv results_ev.csv --summary summary_ev.json
```

From the repository root, you can also launch the same flow with:

```bash
python show_ev_output.py --mode drive --cycle mixed --duration-s 3600 --csv results_ev.csv --summary summary_ev.json
```

Fast-charge example:

```bash
python show_ev_output.py --mode charge --soc0 0.20 --duration-s 1800 --charge-power-kw 50 --csv results_ev.csv --summary summary_ev.json
```

Dedicated silver-ion 9-10 minute fast-charge animation:

```bash
python python/silver_ion_fast_charge_animation.py --start-soc 20 --soc-80-min 9 --soc-100-min 10 --save-animation silver_fast_charge.gif
```

## 13. Calibration and Validation

Fit the simple ECM model against the reference dataset and generate a validation report:

```bash
python python/silver_ion_battery_sim.py --fit-reference --reference-csv results.csv --validation-report validation_report.json --csv fitted_results.csv
```

The report writes `validation_report.json` and the companion plot `validation_report.png`.
# EVs_TItans
# EVs_TItans

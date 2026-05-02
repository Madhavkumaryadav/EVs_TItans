# Validation Checklist (One-Page)

Use this checklist after each trial run before updating model parameters.

## Printable Quick Form

Run ID: ____________________

Date: ____________________

Operator: ____________________

Cell Format: ____________________

### Quick Checks

- [ ] Data package complete (run info, files, lot numbers)
- [ ] Formation run completed without abnormal voltage behavior
- [ ] Pulse and rest segments available for fitting
- [ ] Capacity estimate validated on stabilized cycle data
- [ ] IR drop used for `r0_ohm` from consistent pulse condition
- [ ] Relaxation fit used for `r1_ohm` and `c1_f`
- [ ] Safe voltage window used for `v_min` and `v_max`
- [ ] Simulated voltage trend matches measured trend
- [ ] Error pattern is stable (no systematic drift)
- [ ] Output artifacts archived (raw data, fit output, parameter card, comparison plot)

### Decision

- [ ] Accept parameter set
- [ ] Reject and repeat run

Repeat focus area if rejected: _______________________________________

### Sign-Off

Reviewer Name: ____________________

Reviewer Signature: ____________________

Approval Date: ____________________

## A. Data Readiness

- [ ] Run ID, date, and operator name are recorded.
- [ ] Cell format and geometry are recorded.
- [ ] Material lot numbers are recorded.
- [ ] Test files are archived with clear filenames.
- [ ] Sampling interval is consistent across test segments.

## B. Formation and Baseline Review

- [ ] Formation protocol was executed as planned.
- [ ] No abnormal voltage excursion was observed.
- [ ] Rest segments are present for OCV and relaxation checks.
- [ ] First-cycle trend is physically plausible.

## C. Parameter Extraction Checks

- [ ] Capacity estimate is computed from stabilized cycle data.
- [ ] Instantaneous IR drop value is extracted from consistent pulse edges.
- [ ] Relaxation fitting window is clean (outliers removed).
- [ ] Voltage limits are derived from observed safe operation region.

## D. Simulation Mapping Checks

- [ ] q_ah updated from measured capacity.
- [ ] r0_ohm updated from IR drop estimate.
- [ ] r1_ohm and c1_f updated from transient relaxation fitting.
- [ ] v_min and v_max updated from validated test envelope.

## E. Model-Measurement Comparison

- [ ] Simulated voltage trend follows measured direction under same current profile.
- [ ] Pulse transient shape is reasonably matched.
- [ ] Rest-region OCV checkpoints are reasonably matched.
- [ ] Error pattern is stable and not drifting systematically.

## F. Decision Gate

- [ ] Accept parameters for current baseline.
- [ ] OR reject and repeat selected phase(s):
  - [ ] Slurry formulation
  - [ ] Coating and drying
  - [ ] Assembly and wetting
  - [ ] Formation protocol
  - [ ] Pulse test and fitting

## G. Required Output Artifacts

- [ ] Raw test data file
- [ ] Processed fit output (notebook/script result)
- [ ] Final parameter card
- [ ] One comparison plot (measured vs simulated voltage)
- [ ] Short conclusion note for next iteration

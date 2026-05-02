# Detailed Process by Phases

## Phase 1: Requirement and Design Definition

Objective:
- Define what cell you are trying to build and what success means.

Inputs:
- Target format (coin or pouch)
- Target voltage window
- Trial current range
- Preliminary chemistry system

Actions:
- Define acceptance metrics (capacity retention, first-cycle efficiency, voltage profile shape)
- Freeze one baseline design before starting experiments
- Create a run sheet template for all future batches

Checkpoints:
- Clear pass or fail criteria are written
- All units are fixed (g, mg/cm^2, mL, V, A)

Outputs:
- Design spec sheet for batch B0

## Phase 2: Material Preparation and Preconditioning

Objective:
- Prepare stable and contamination-controlled starting materials.

Inputs:
- Active powder, conductive additive, binder, separator, collector foils

Actions:
- Verify material identity and lot number
- Pre-dry moisture-sensitive materials per supplier guidance
- Store prepared materials in controlled environment before mixing

Checkpoints:
- No visible contamination
- Lot traceability recorded
- Moisture handling condition recorded (ambient, dry room, glovebox)

Outputs:
- Preconditioned materials ready for slurry

## Phase 3: Binder and Solvent Preparation

Objective:
- Build a stable binder phase with repeatable viscosity behavior.

Inputs:
- Binder system and compatible solvent package

Actions:
- Add binder gradually while stirring to avoid clumping
- Continue mixing until visually homogeneous
- Rest or degas mixture if bubbles are present

Checkpoints:
- No large gel lumps
- Viscosity trend is consistent with previous successful batches

Outputs:
- Binder phase ready for final slurry mixing

## Phase 4: Slurry Formulation and Homogenization

Objective:
- Produce a coatable slurry with uniform particle distribution.

Inputs:
- Active-conductive-binder ratio target (for example 85:8:7)
- Solids loading target (typical trial range 25-45 wt%)

Actions:
- Dry-blend active and conductive components first
- Add binder phase incrementally
- Tune viscosity using solvent in small increments
- Continue mixing until texture is uniform and stable

Checkpoints:
- No visible agglomerates
- Slurry does not phase-separate during short standing interval
- Coating trial strip is continuous and defect-minimized

Outputs:
- Controlled slurry batch with recorded composition and mixing time

## Phase 5: Coating and Thickness Control

Objective:
- Transfer slurry to collector with consistent area loading.

Inputs:
- Prepared slurry
- Clean current collector foil

Actions:
- Set coating gap and speed
- Apply coating in one controlled pass where possible
- Label every coated strip with batch and timestamp

Checkpoints:
- Wet film continuity without streaks or pinholes
- Edge defects and peel zones identified and excluded

Outputs:
- Coated electrode sheets prepared for drying

## Phase 6: Drying, Densification, and Electrode Conditioning

Objective:
- Remove solvent and set electrode microstructure.

Inputs:
- Coated sheets

Actions:
- Dry with controlled temperature-time profile
- Apply optional calendaring to hit target density or porosity window
- Rest electrodes before cutting

Checkpoints:
- Mass change stabilizes after drying
- No severe cracking or delamination
- Thickness uniformity within internal tolerance band

Outputs:
- Mechanically stable dried electrodes

## Phase 7: Cutting, Mass Loading, and Dimensional QA

Objective:
- Produce geometry-controlled electrodes with traceable loading values.

Inputs:
- Dried electrodes

Actions:
- Punch or cut to target geometry
- Measure areal loading (mg/cm^2)
- Measure thickness at multiple points

Checkpoints:
- Loading variation is within selected tolerance band
- Burrs and edge defects minimized

Outputs:
- Qualified electrode set for assembly

## Phase 8: Cell Assembly and Electrolyte Wetting

Objective:
- Build a short-free cell stack with complete ionic wetting.

Inputs:
- Cathode, separator, counter electrode, electrolyte

Actions:
- Assemble in correct layer order and polarity
- Add electrolyte in measured increments
- Allow soak time for uniform wetting before sealing

Checkpoints:
- No misalignment that can cause edge short
- No dry separator region remains
- Seal integrity visually confirmed

Outputs:
- Sealed trial cell ready for formation

## Phase 9: Formation Cycling

Objective:
- Stabilize interfaces and establish reproducible early-cycle behavior.

Inputs:
- Assembled and sealed cell

Actions:
- Run low-rate charge and discharge cycles
- Include rest periods to capture relaxation behavior
- Record voltage and current with consistent sampling interval

Checkpoints:
- No abnormal voltage excursion
- Coulombic efficiency trend improves over initial cycles

Outputs:
- Formed cell and baseline cycle dataset

## Phase 10: Baseline Characterization and Parameter Extraction

Objective:
- Convert test data to simulation-ready ECM parameters.

Inputs:
- Formation and pulse-test datasets

Actions:
- Estimate capacity from cycle data -> q_ah
- Estimate instantaneous IR drop -> r0_ohm
- Fit relaxation transient -> r1_ohm, c1_f
- Determine safe operating bounds -> v_min, v_max

Checkpoints:
- Parameter values remain physically plausible
- Fitted model reproduces baseline voltage trend with acceptable error

Outputs:
- Parameter card for simulation model update

## Validation Mapping to Simulation

After extracting parameters, run a validation pass before committing values to the model.

### What to Compare

- Compare measured voltage curve vs simulated terminal voltage trend
- Compare rest-region OCV checkpoints
- Compare pulse transient shape and relaxation timing

### Parameter-Level Checks

- `q_ah`:
	Check that simulated SOC progression matches measured charge throughput trend.

- `r0_ohm`:
	Check immediate voltage step at pulse edges.

- `r1_ohm`, `c1_f`:
	Check relaxation curve slope and settling behavior.

- `v_min`, `v_max`:
	Check simulation limits are within observed safe test window.

### Acceptance Guidance

- Simulated trend should match measured direction and shape across the same current profile.
- Residual error should be stable and not drift systematically over time.
- If mismatch is systematic, revisit pulse protocol consistency and repeat extraction.

### Output Artifact

Store one validation package per run:

- Raw test data file
- Processed fitting notebook or script output
- Final parameter card
- One comparison plot (measured vs simulated voltage)

Use the print-ready acceptance form for final review:

- [08-validation-checklist.md](08-validation-checklist.md)

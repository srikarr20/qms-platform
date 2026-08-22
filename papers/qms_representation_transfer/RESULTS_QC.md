# QMS Representation Transfer Manuscript — Numerical QC

## Purpose

Cross-check every quantitative statement in `MANUSCRIPT.md` against the original experiment evidence before figure generation or publication.

---

## QMS-REP-001

Verify:

- noise 0.0:
  - effective dimension ≈ 1.6728
  - explained-variance fraction (first 2) ≈ 0.99997
  - observable/control correlation ≈ 0.8793

- noise 0.2:
  - effective dimension ≈ 2.4922
  - explained-variance fraction (first 2) ≈ 0.83265
  - observable/control correlation ≈ 0.4389

Status: [x]

---

## QMS-REP-002

Verify:

- effective dimension remains approximately 18–19 across count scales
- similarity degrades approximately:
  - 0.9995
  - to 0.9516

Status: [x]

---

## QMS-REP-003

Verify:

- records: 1500
- similarity vs Bell fidelity correlation ≈ -0.00735
- similarity vs minimum eigenvalue correlation ≈ 0.87389
- mean similarity ≈ 0.98551
- mean fidelity ≈ 1.0
- mean minimum eigenvalue ≈ -0.06521

Status: [x]

---

## QMS-REP-004

Verify:

- records: 10800
- overall similarity/minimum-eigenvalue correlation ≈ 0.60684
- per-state correlation range ≈ 0.529–0.705
- nonphysical fraction ≈ 0.8415

Status: [x]

---

## QMS-REAL-001 — ISTA IQ

Verify operating conditions:

- 0 Hz
- 10 Hz
- 50 Hz
- 250 Hz
- 500 Hz
- 1000 Hz

Verify reported QND fidelity endpoints:

- 0 Hz ≈ 0.91445
- 1000 Hz ≈ 0.69235

Verify effective dimensions:

- 1.19464
- 1.19852
- 1.21313
- 1.24197
- 1.26342
- 1.29776

Verify Pearson correlations:

- centroid separation ≈ 0.99785
- spread ≈ -0.97867
- normalized separation ≈ 0.97070
- Mahalanobis separation ≈ 0.96431
- effective dimension ≈ -0.98810

Status: [x]

---

## QMS-REAL-003 — Label-free IQ

Verify:

- effective dimension vs QND ≈ -0.98810
- first-PC fraction vs QND ≈ 0.98694
- anisotropy vs QND ≈ 0.98694
- covariance determinant vs QND ≈ -0.97424
- radial std vs QND ≈ -0.99072
- radial p99 vs QND ≈ -0.98465

Verify covariance trace and entropy are weaker metrics.

Status: [x]

---

## QMS-REAL-005 — Provisional MDI

Verify MDI values:

- 0 Hz: -0.00897
- 10 Hz: 5.5789
- 50 Hz: 15.7655
- 250 Hz: 39.1971
- 500 Hz: 57.6005
- 1000 Hz: 85.2832

Verify MDI/QND correlation:

- ≈ -0.98869

Status: [x]

---

## QMS-REAL-006 — Glasgow Representation Geometry

Verify Heralded Diffraction:

- mean effective dimension ≈ 1.95084
- effective-dimension std ≈ 0.028995
- mean first-PC fraction ≈ 0.576667
- radial std ≈ 78.0113
- radial p99 ≈ 315.282

Status: [x]

---

## QMS-REAL-007 — Heralded Diffraction Convergence

Verify:

- first cosine ≈ 0.77010
- first JS ≈ 0.23258
- maximum cosine ≈ 0.9975005
- minimum JS ≈ 0.00264973
- late cosine ≈ 0.986999
- late JS ≈ 0.0065014

Verify correlations:

- window / cosine ≈ 0.6413
- window / JS ≈ -0.7178
- ED / cosine ≈ 0.8528
- firstPC / cosine ≈ -0.8184
- ED / JS ≈ -0.9136
- firstPC / JS ≈ 0.8847

Status: [x]

---

## QMS-REAL-008 — Heralded State Classifier

Verify:

- initial: 1
- converging: 25
- stable_or_mixed: 4
- drifting: 11
- first drift: frames 3001–3100

Status: [x]

---

## QMS-REAL-009 — Robustness

Verify:

- configurations: 9
- all 9 detect late drift
- first drift frame mean ≈ 3034.33
- std ≈ 164.99
- min ≈ 2851
- max ≈ 3401

Status: [x]

---

## QMS-REAL-017 — Ghost Diffraction

Verify:

- 4070 frames
- effective dimension start ≈ 1.8704
- effective dimension end ≈ 1.9874
- firstPC start ≈ 0.63162
- firstPC end ≈ 0.53977

State classifier:

- initial: 1
- converging: 27
- stable_or_mixed: 4
- drifting: 9
- first drift: 3201–3300

Status: [x]

---

## QMS-REAL-018 — Diffraction Transfer

Verify cross-condition correlations:

- effective dimension ≈ 0.32049
- firstPC ≈ 0.26217
- cosine-to-final ≈ 0.9968958
- JS-divergence-to-final ≈ 0.9969648

Verify normalized RMSE:

- cosine ≈ 0.01936
- JS ≈ 0.02437

Verify best-region differences:

- maximum cosine: 2 windows
- minimum JS: 1 window

Status: [x]

---

## QMS-REAL-020 — Heralded Imaging

Verify:

- frames: 1000
- windows: 10
- initial: 1
- converging: 7
- drifting: 2
- first drift: 801–900
- maximum cosine window: 6
- minimum JS window: 6

Status: [x]

---

## QMS-REAL-022 — Ghost Imaging

Verify:

- frames: 1000
- windows: 10
- initial: 1
- converging: 7
- drifting: 2
- first drift: 801–900
- maximum cosine window: 7
- minimum JS window: 7

Status: [x]

---

## QMS-REAL-023 — Aggregate Transfer

Verify normalized first-drift positions:

- Heralded Diffraction ≈ 0.7495086
- Ghost Diffraction ≈ 0.7986486
- Heralded Imaging ≈ 0.8505
- Ghost Imaging ≈ 0.8505

Verify aggregate:

- mean ≈ 0.8122893
- std ≈ 0.041975
- min ≈ 0.7495
- max ≈ 0.8505

Verify family means:

- diffraction ≈ 0.774079
- imaging ≈ 0.8505

Status: [x]

---

## QMS-REAL-011–016 — Causal Experiments

Verify:

### REAL-012
- rolling reference:
  - warnings: 0
  - alerts: 0

### REAL-013
- frozen early reference:
  - warnings: 3
  - alerts: 31
  - first warning: 701–800
  - first alert: 1001–1100

### REAL-014
- baseline lock ≈ frame 2200
- first alert: 2301–2400

### REAL-015
- persistent warning: 2401–2500
- persistent alert: 2501–2600

### REAL-016
- warnings: 0
- alerts: 0

Status: [x]

---

## QMS-REAL-024 — Evidence Registry

Verify:

- validated findings: 6
- transfer findings: 6
- negative results: 6
- provisional heuristics: 2
- unsupported/not established: 6

Status: [x]

---

## QMS-REAL-025 — Reproducibility Manifest

Verify:

- experiments indexed before manifest: 24
- missing scripts: none
- missing evidence: none
- all declared scripts present
- all declared evidence present

Status: [x]

---

# Manuscript Claim Boundary

Confirm manuscript does NOT claim:

- universal detector health
- universal 80% drift threshold
- confirmed Glasgow hardware degradation
- validated prospective causal early warning
- universal representation geometry
- cross-laboratory transfer
- universal quantum-hardware validation

Status: [x]

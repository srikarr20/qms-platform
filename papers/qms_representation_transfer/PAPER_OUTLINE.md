# QMS Platform: Representation Diagnostics, Measurement Convergence, and Cross-Condition Transfer in Real Quantum Measurements

## Working paper status

Second QMS paper.

This manuscript builds on the initial QMS observability and reconstruction-assurance paper and focuses specifically on:

- representation diagnostics,
- real-measurement behavior,
- convergence structure,
- cross-condition transfer,
- operational measurement-state classification,
- negative causal-drift results.

The paper should not claim universal detector health, universal drift timing, validated causal early warning, or cross-laboratory generality.

---

# 1. Abstract

## Core problem

Quantum measurement pipelines often move directly from raw detector data to reconstruction or downstream inference without explicitly characterizing:

1. the structure of the measurement representation,
2. whether that representation is stable or converging,
3. whether diagnostic behavior transfers across measurement conditions,
4. whether apparent measurement-state change can be detected causally.

## Approach

QMS is extended with representation diagnostics including:

- covariance spectra,
- participation-ratio effective dimension,
- explained-variance fraction,
- observable sensitivity.

These diagnostics are evaluated on:

- synthetic representation experiments,
- simulated quantum-state tomography ensembles,
- real superconducting-qubit IQ measurements,
- raw spatial single-photon detector measurements.

Cross-condition transfer is then tested across four Glasgow acquisitions:

- Heralded Diffraction SM,
- Ghost Diffraction SM,
- Heralded Imaging MM set 1,
- Ghost Imaging MM set 1.

## Main findings

The abstract should state only the following supported results:

- Representation complexity and measurement usefulness are not equivalent.
- Selected IQ representation diagnostics track QND-fidelity degradation across six ISTA operating conditions.
- Distribution-level convergence trajectories are highly reproducible across the two Glasgow diffraction acquisitions.
- Cosine-to-final trajectory correlation is approximately 0.9969.
- JS-divergence-to-final trajectory correlation is approximately 0.9970.
- Fine effective-dimension and first-PC trajectories are much less reproducible across conditions.
- An unchanged measurement-state classifier transfers across four Glasgow acquisitions.
- First drift occurs at normalized acquisition positions between approximately 0.750 and 0.851.
- Causal early-warning capability is not established by the tested methods.

## Abstract conclusion

The key conclusion should be:

> Measurement representation, convergence, observability, and health interpretation should be treated as distinct layers. Distribution-level convergence appears more transferable across tested acquisition conditions than fine representation geometry.

---

# 2. Introduction

## 2.1 Motivation

Introduce the measurement-assurance problem:

- raw measurements can be high dimensional,
- high dimensionality does not imply high information content,
- noisy measurements can appear more complex,
- reconstruction quality depends on what information is actually present,
- detector state and measurement convergence can change during acquisition.

## 2.2 Gap

Current measurement pipelines often lack a layer that asks:

> What structure is present in the measurement before reconstruction, and how stable is that structure across operating conditions?

## 2.3 QMS extension

Position this work as an extension of the original QMS framework.

Original emphasis:

- observability,
- reconstruction assurance,
- null-space structure,
- state-dependent recoverability,
- error decomposition.

New emphasis:

- representation diagnostics,
- convergence diagnostics,
- operational measurement-state classification,
- real-measurement transfer.

## 2.4 Contributions

Suggested contribution list:

1. A scale-invariant representation-diagnostics layer for QMS.
2. Synthetic and tomography-based tests showing representation complexity is not equivalent to measurement usefulness.
3. Real superconducting-qubit IQ validation across six operating conditions.
4. Raw single-photon detector validation across multiple acquisition conditions.
5. Quantification of cross-condition convergence-trajectory reproducibility.
6. Transfer of an unchanged operational state classifier across four Glasgow acquisitions.
7. Explicit negative results showing that current causal-drift formulations do not establish prospective early warning.
8. A machine-readable evidence registry and reproducibility manifest.

---

# 3. QMS Measurement-Representation Framework

## 3.1 Layered architecture

Use the architecture:

raw measurement
    ->
representation diagnostics
    ->
convergence/reference diagnostics
    ->
operational state classification
    ->
observability diagnostics
    ->
reconstruction assurance
    ->
error decomposition
    ->
validated health interpretation

Explain why each layer is distinct.

## 3.2 Representation diagnostics

Define:

### Covariance spectrum

Given measurement samples X, compute the covariance matrix and its eigenvalue spectrum.

### Participation-ratio effective dimension

For nonnegative spectral values lambda_i:

d_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2

State explicitly:

- the implementation is scale invariant,
- d_eff is a representation-complexity measure,
- it is not inherently a health metric.

### Explained-variance fraction

Fraction of covariance-spectrum mass captured by the first k components.

### Observable sensitivity

Association between a candidate observable and a known control variable.

## 3.3 Representation versus observability

This section is conceptually important.

Explain:

- representation dimension describes variation structure,
- observability asks whether the measurement operator permits recovery of the target,
- they answer different questions.

A noisy dataset can exhibit greater apparent representation complexity while becoming less informative.

---

# 4. Experimental Design

## 4.1 QMS-REP synthetic experiments

### QMS-REP-001

Synthetic four-channel representation under increasing Gaussian noise.

Reported behavior:

- effective dimension increased,
- explained-variance concentration decreased,
- energy-control correlation weakened.

Interpretation:

> Noise can increase apparent representation complexity while reducing measurement usefulness.

### QMS-REP-002

Bell-state tomography measurement ensemble under Poisson count degradation.

Key result:

- effective dimension remained broadly similar while measurement similarity degraded.

Boundary:

Effective dimension here reflects noisy ensemble variation, not ideal quantum-state manifold dimension.

### QMS-REP-003

1500 noisy tomography records.

Key reported correlations:

- similarity versus Bell fidelity: approximately -0.007
- similarity versus minimum eigenvalue: approximately 0.874

Interpretation:

Measurement consistency tracked physicality failure more strongly than reported Bell fidelity under the linear estimator.

### QMS-REP-004

Bell plus random pure states under Poisson/background degradation.

Overall similarity versus minimum-eigenvalue correlation:

approximately 0.607

Per-state range:

approximately 0.529 to 0.705

Interpretation:

The relationship persists across states and degradation mechanisms, but is not a universal fidelity predictor.

---

# 5. Real Superconducting-Qubit IQ Validation

## 5.1 Dataset

ISTA all-optical superconducting-qubit readout.

Use six operating conditions:

- 0 Hz
- 10 Hz
- 50 Hz
- 250 Hz
- 500 Hz
- 1000 Hz

Reported QND fidelities:

- 0.91445
- 0.90829
- 0.89846
- 0.84371
- 0.78324
- 0.69235

## 5.2 Labeled IQ diagnostics

Metrics include:

- centroid separation,
- spread,
- normalized separation,
- Mahalanobis separation,
- effective dimension.

Reported Pearson correlations with QND fidelity:

- centroid separation: approximately 0.9979
- spread: approximately -0.9787
- normalized separation: approximately 0.9707
- Mahalanobis separation: approximately 0.9643
- effective dimension: approximately -0.9881

Spearman relationships are monotonic across all six operating conditions.

## 5.3 Bootstrap robustness

QMS-REAL-002:

- 500 bootstrap samples,
- normalized separation and Mahalanobis intervals remain cleanly ordered,
- effective dimension is less separated at the closest operating conditions.

## 5.4 Label-free IQ metrics

QMS-REAL-003 and REAL-004.

Strong metrics:

- effective dimension,
- first-PC fraction,
- anisotropy,
- covariance determinant,
- radial standard deviation,
- radial 99th percentile.

Weak metrics:

- covariance trace,
- entropy.

Important point:

> Not every available statistic should be promoted into a measurement-health metric.

## 5.5 Measurement Degradation Index

Present MDI as provisional.

Do not present the health-score transform as calibrated.

State:

- MDI is reference specific,
- strong same-dataset correlation does not establish universal validity.

---

# 6. Raw Glasgow Single-Photon Measurements

## 6.1 Raw data

The Glasgow experiments contain sparse 512 x 512 detector frames stored as ASC matrices.

Example characteristics:

- many frames are extremely sparse,
- nonzero pixels are interpreted as detector-event coordinates.

## 6.2 QMS-REAL-006

Apply the same four label-free geometry metrics originally frozen from the ISTA work:

- effective dimension,
- first-PC fraction,
- radial standard deviation,
- radial 99th percentile.

Key conclusion:

> The representation-diagnostics architecture operates across at least two real measurement modalities.

Boundary:

Numerical calibration does not transfer directly between IQ and spatial single-photon measurements.

---

# 7. Convergence Versus Representation Drift

## 7.1 QMS-REAL-007

For each 100-frame window:

- construct a normalized 32 x 32 spatial histogram,
- compare with the mature full-acquisition distribution.

Metrics:

- cosine similarity,
- Jensen-Shannon divergence.

Reported behavior in Heralded Diffraction SM:

- early cosine approximately 0.770,
- late cosine approximately 0.987,
- early JS approximately 0.233,
- late JS approximately 0.0065.

Best-convergence region occurs before the end of acquisition.

## 7.2 Representation interpretation

Effective dimension increases during the major convergence phase.

Contrast with ISTA:

- ISTA: increasing effective dimension accompanies lower QND fidelity.
- Glasgow: increasing effective dimension accompanies convergence toward the mature spatial distribution.

Critical conclusion:

> Representation complexity is not intrinsically good or bad. Its meaning depends on the measurement objective and reference.

---

# 8. Operational Measurement-State Classification

## 8.1 Classifier

Use three-window smoothed convergence indicators.

Rule:

converging:
    delta cosine > 0
    delta JS < 0

drifting:
    delta cosine < 0
    delta JS > 0

stable_or_mixed:
    otherwise

State clearly:

- this is an operational heuristic,
- it is not a calibrated hardware-failure classifier.

## 8.2 Heralded Diffraction SM

Reported state counts:

- initial: 1
- converging: 25
- stable_or_mixed: 4
- drifting: 11

First drift:

frames 3001-3100

## 8.3 Robustness

QMS-REAL-009 tests:

- window sizes 50, 100, 200,
- smoothing widths 2, 3, 5.

All 9 configurations detect a late-acquisition transition in the same broad region.

Do not claim exact-frame calibration.

---

# 9. Cross-Condition Transfer

## 9.1 Ghost Diffraction SM

Same raw-frame count:

4070

No metric changes.

Key summary:

- effective dimension start: approximately 1.870
- effective dimension end: approximately 1.987
- first-PC fraction start: approximately 0.632
- first-PC fraction end: approximately 0.540

## 9.2 Cross-condition trajectory consistency

Compare Heralded Diffraction SM and Ghost Diffraction SM.

Reported correlations:

### Distribution-level metrics

- cosine-to-final: 0.9968958
- JS-divergence-to-final: 0.9969648

### Fine representation metrics

- effective dimension: 0.32049
- first-PC fraction: 0.26217

This is a central result.

Interpretation:

> Convergence trajectories are highly reproducible while fine geometry is condition dependent.

## 9.3 State-classifier transfer

Ghost Diffraction SM:

- initial: 1
- converging: 27
- stable_or_mixed: 4
- drifting: 9

First drift:

frames 3201-3300

Transition differs from Heralded Diffraction SM by approximately 200 frames.

---

# 10. Cross-Objective Transfer: Imaging

## 10.1 Heralded Imaging MM set 1

1000 frames.

10 fixed 100-frame windows.

State counts:

- initial: 1
- converging: 7
- drifting: 2

First drift:

frames 801-900

## 10.2 Ghost Imaging MM set 1

1000 frames.

State counts:

- initial: 1
- converging: 7
- drifting: 2

First drift:

frames 801-900

## 10.3 Four-dataset aggregate

Normalized first-drift positions:

- Heralded Diffraction SM: 0.7495
- Ghost Diffraction SM: 0.7986
- Heralded Imaging MM: 0.8505
- Ghost Imaging MM: 0.8505

Aggregate:

- mean: 0.8123
- standard deviation: 0.0420
- minimum: 0.7495
- maximum: 0.8505

Family means:

- diffraction: approximately 0.7741
- imaging: 0.8505

Important interpretation:

> Late-state timing is reproducible within tested families but differs across acquisition families.

Do not call 0.8 a universal threshold.

---

# 11. Causal Drift Detection: Negative Results

This section should be retained prominently.

## 11.1 QMS-REAL-011

Global retrospective change-point analysis primarily detects early convergence/saturation.

Conclusion:

Does not validate an impending-drift precursor.

## 11.2 QMS-REAL-012

Rolling causal reference.

Result:

- zero warnings,
- zero alerts.

Interpretation:

The reference adapts with the gradual redistribution.

## 11.3 QMS-REAL-013

Frozen early reference.

Result:

- many alerts,
- first alert far too early.

Interpretation:

The early acquisition is not a valid stationary baseline.

## 11.4 QMS-REAL-014

Automatic stability-based baseline lock.

Baseline lock:

frame 2200

First alert:

frames 2301-2400

Interpretation:

Baseline departure is not equivalent to regime-change drift.

## 11.5 QMS-REAL-015

Persistent thresholds.

First persistent warning:

2401-2500

First persistent alert:

2501-2600

Still earlier than retrospective late transition.

## 11.6 QMS-REAL-016

Causal divergence-slope acceleration.

Result:

- no warning,
- no alert.

## 11.7 Scientific conclusion

> Current causal experiments do not establish prospective early-warning capability.

This negative conclusion should appear in the abstract, discussion, and conclusion.

---

# 12. Discussion

## 12.1 Representation complexity is contextual

Discuss the contrast between ISTA and Glasgow.

Effective dimension alone cannot be interpreted as measurement health.

## 12.2 Distribution-level convergence is more transferable

The strongest empirical cross-condition result is the approximately 0.997 trajectory correlation for distribution-level metrics.

## 12.3 Operational state transfer

The unchanged classifier reproduces late-acquisition departure across four conditions.

This supports operational reuse within the tested Glasgow platform.

## 12.4 Why causal detection is harder

Retrospective mature-distribution comparison has information unavailable to a live runtime.

Rolling baselines adapt.

Frozen baselines can become stale.

This explains why retrospective state structure does not automatically imply prospective prediction.

## 12.5 Relationship to observability

Representation/convergence diagnostics should precede rather than replace observability.

A converged measurement can still be informationally incomplete.

An observable measurement can still be noisy or drifting.

---

# 13. Limitations

Explicitly list:

1. Glasgow conditions originate from one experimental platform.
2. Cross-laboratory transfer is not tested.
3. Imaging datasets contain only 1000 frames and 10 windows each.
4. The mature full-acquisition distribution is retrospective.
5. State labels are operational, not hardware-failure ground truth.
6. QND fidelity in ISTA is used as convergent validation, not independent blinded ground truth.
7. Measurement Degradation Index is reference specific.
8. No prospective early-warning method has been validated.
9. Fine representation geometry is not universal.
10. Real phase-aware DPI upstream reconstruction remains unavailable for the Glasgow datasets.

---

# 14. Reproducibility and Evidence Governance

## QMS-REAL-024

Machine-readable evidence registry separates:

- validated findings,
- transfer findings,
- negative results,
- provisional heuristics,
- unsupported claims.

## QMS-REAL-025

Reproducibility manifest:

- 24 real-measurement experiments indexed,
- missing scripts: none,
- missing evidence: none,
- all declared scripts present,
- all declared evidence present.

Mention:

- GitHub repository,
- release tag,
- Zenodo software DOI.

Current software DOI:

10.5281/zenodo.22060410

---

# 15. Conclusion

Suggested conclusion structure:

1. QMS representation diagnostics operate across two distinct real measurement modalities.
2. Representation complexity cannot be interpreted as health independently of context.
3. Distribution-level convergence is highly reproducible across two real diffraction conditions.
4. An unchanged operational state classifier transfers across four Glasgow acquisition conditions.
5. Fine representation geometry remains condition dependent.
6. Current evidence supports cross-condition transfer within one platform.
7. Prospective causal early warning and cross-laboratory transfer remain open validation targets.

Final sentence:

> These results motivate a measurement-assurance architecture in which representation, convergence, observability, reconstruction, and health interpretation remain explicitly separated and independently validated.

---

# 16. Proposed Figures

## Figure 1 — QMS layered measurement-assurance architecture

raw measurement
→ representation
→ convergence/reference
→ operational state
→ observability
→ reconstruction
→ error decomposition
→ interpretation

## Figure 2 — ISTA IQ degradation

Multi-panel:

- IQ distributions by operating condition,
- effective dimension versus QND fidelity,
- normalized separation versus QND fidelity,
- Mahalanobis separation versus QND fidelity.

## Figure 3 — Representation complexity versus usefulness

Use QMS-REP-001/002.

Show that complexity can increase while measurement quality degrades.

## Figure 4 — Glasgow Heralded Diffraction convergence

Window index versus:

- cosine-to-final,
- JS divergence,
- effective dimension,
- first-PC fraction.

## Figure 5 — Cross-condition diffraction trajectory comparison

Overlay Heralded and Ghost normalized:

- cosine trajectories,
- JS trajectories.

Annotate r ≈ 0.997.

## Figure 6 — Fine geometry cross-condition comparison

Overlay:

- effective dimension,
- first-PC fraction.

Show lower correlation.

## Figure 7 — Four-dataset state timelines

Horizontal state bars:

- Heralded Diffraction,
- Ghost Diffraction,
- Heralded Imaging,
- Ghost Imaging.

## Figure 8 — Causal detection outcomes

Summarize REAL-011 through REAL-016:

- retrospective transition,
- rolling reference,
- frozen baseline,
- locked baseline,
- persistence,
- slope acceleration.

Emphasize negative results.

---

# 17. Proposed Tables

## Table 1 — Dataset inventory

Columns:

- dataset,
- measurement modality,
- experiment family,
- raw frames / shots,
- dimensionality,
- label availability,
- validation role.

## Table 2 — ISTA metric correlations

Metrics versus reported QND fidelity.

## Table 3 — Glasgow cross-condition correlations

Include:

- effective dimension,
- first-PC fraction,
- cosine-to-final,
- JS-divergence-to-final.

## Table 4 — State-transfer summary

Columns:

- dataset,
- family,
- frame count,
- state counts,
- first drift,
- normalized drift position.

## Table 5 — Causal detector results

Columns:

- experiment,
- detector formulation,
- first warning,
- first alert,
- outcome,
- interpretation.

## Table 6 — Supported and unsupported claims

Separate:

- established,
- provisional,
- not established.

---

# 18. Candidate Paper Claims

## Strong claims

- QMS representation diagnostics operate across superconducting-qubit IQ and single-photon spatial measurements.
- Distribution-level convergence trajectories are highly reproducible across two tested Glasgow diffraction conditions.
- An unchanged operational state classifier transfers across four tested Glasgow acquisition conditions.
- Fine representation geometry is substantially less transferable than distribution-level convergence.

## Moderate claims

- Operational state classification may provide a reusable measurement-state abstraction within related acquisition families.
- Representation and convergence diagnostics can complement observability and reconstruction assurance.

## Claims to avoid

- universal detector health,
- universal 80% drift timing,
- causal prediction of failure,
- confirmed hardware degradation,
- universal representation dimension,
- cross-laboratory generality,
- general quantum-hardware validation.

---

# 19. Relationship to Prior QMS Paper

Prior paper:

QMS Platform: A Quantum Measurement Observability and Reconstruction-Assurance Framework with Initial Validation

DOI:

10.5281/zenodo.22057334

This second paper should clearly state that it extends the prior work from:

observability + reconstruction assurance

to:

representation diagnostics + real-measurement convergence + cross-condition transfer.

Avoid duplicating the first paper's tomography derivations except where required for context.

---

# 20. Next Work After This Paper

Highest-value validation target:

Apply the frozen representation/convergence/state framework without retuning to:

1. another independent acquisition family,
2. preferably a different laboratory or detector platform.

The decisive next evidence level is:

cross-condition
    ->
cross-platform
    ->
cross-laboratory

## Representation and Measurement Assurance

QMS now includes a representation-diagnostics layer in addition to observability and reconstruction assurance.

Current stack:

```text
raw measurement
    ->
representation diagnostics
    ->
observability diagnostics
    ->
reconstruction assurance
    ->
error decomposition
    ->
measurement-state interpretation
```

Representation diagnostics include covariance spectrum, participation-ratio effective dimension, explained-variance fraction, and observable sensitivity.

Current validation:

```text
9 representation tests passed
```

QMS-REP-001 through QMS-REP-004 show that representation complexity and measurement usefulness are not equivalent. Fine representation geometry remains measurement-condition dependent.

### Expanded real-measurement validation

Real-measurement validation now spans:

- superconducting-qubit IQ measurements,
- spatial single-photon detector measurements.

Across four Glasgow acquisition conditions — Heralded Diffraction SM, Ghost Diffraction SM, Heralded Imaging MM set 1, and Ghost Imaging MM set 1 — the same distribution-level convergence and state-classification logic was transferred without dataset-specific retuning.

Across the two 4070-frame diffraction acquisitions:

```text
cosine-to-final correlation:         ~0.9969
JS-divergence-to-final correlation:  ~0.9970
```

The normalized first-drift positions across the four acquisitions were:

```text
Heralded Diffraction SM      0.750
Ghost Diffraction SM         0.799
Heralded Imaging MM set 1    0.851
Ghost Imaging MM set 1       0.851
```

These are descriptive acquisition coordinates, not a universal drift threshold.

Current evidence does not establish prospective causal early-warning capability, universal detector-health calibration, universal representation geometry, or cross-laboratory classifier transfer.

A consolidated scientific report is included at:

`docs/QMS_Representation_and_Real_Measurement_Validation_Report.docx`

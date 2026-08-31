# QMS + Glasgow Validation Manifest

Contact: r.srikar.30@gmail.com

## Validation Target

This package separates reusable QMS measurement-twin software from the Glasgow-specific experimental-data adapter.

## QMS Software

qms_core/measurement_twin/

Provides:

- observable measurement-state construction
- cosine similarity
- Jensen-Shannon divergence
- causal one-step prediction
- innovation calculation
- frozen robust threshold calibration
- future-only alert evaluation
- connected alert-episode consolidation

## Glasgow Integration

integrations/glasgow/adapter.py

Converts cumulative detector frames into incremental detector measurements.

## Executable Regression

tests/integration/test_glasgow_causal_regression.py

Validated causal-replay result:

- reference end: 399
- calibration end: 699
- calibration samples: 199
- median innovation: approximately 0.00448429
- MAD: approximately 0.00229512
- frozen threshold: approximately 0.02490077
- future samples: 3369
- raw future flags: 64
- connected alert episodes: 6

Episode peaks:

951, 1598, 2316, 2761, 3458, 3875

## Reproduction

export GLASGOW_ZIP=/path/to/Heralded_Diffraction_SM.zip

python3 -m pytest tests/integration/test_glasgow_causal_regression.py -q

## Interpretation

This is a causal-replay analysis of real experimental measurements.

It demonstrates prediction and alerting in observable measurement-distribution space.

It does not identify a detector mechanism, establish detector degradation, or infer quantum-field dynamics.

## Stronger Independent Validation

A useful next validation would use:

- an independently selected Glasgow experimental archive or run
- acquisition metadata where available
- no target windows supplied to QMS in advance

QMS could then apply the frozen-reference workflow without prior knowledge of where strong measurement-state excursions occur.

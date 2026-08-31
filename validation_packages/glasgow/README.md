# QMS + Glasgow External Validation Package

Contact: r.srikar.30@gmail.com

## Purpose

This package provides a focused external-validation snapshot of QMS and its Glasgow experimental measurement integration.

The Glasgow pathway operates on real cumulative single-photon detector measurements and does not assume a calibrated detector forward model.

## Package Structure

qms_core/measurement_twin/
    Provider-independent causal measurement-state twin.

integrations/glasgow/
    Glasgow cumulative-archive adapter.

tests/integration/
    Executable causal-replay regression.

## Measurement Interface

For cumulative detector frames F(t-1) and F(t), the adapter reconstructs the incremental measurement:

D(t) = F(t) - F(t-1)

## Validated Causal Replay

The reusable QMS implementation reproduces:

- causal reference frozen on increments 0-399
- measurement window size: 100
- innovation calibration frozen through window 699
- calibration samples: 199
- median innovation: approximately 0.00448429
- MAD: approximately 0.00229512
- frozen threshold: approximately 0.02490077
- future samples: 3369
- raw future flags: 64
- connected alert episodes: 6

Episode peak windows:

951, 1598, 2316, 2761, 3458, 3875

Previously examined windows 701 and 801 are also flagged in the causal replay using only past observable states.

## Running the Validation

Set the external archive location:

export GLASGOW_ZIP=/path/to/Heralded_Diffraction_SM.zip

Run:

python3 -m pytest tests/integration/test_glasgow_causal_regression.py -q

Expected result:

1 passed

## Scientific Boundary

This integration supports observable-state tracking, causal short-horizon prediction, innovation measurement, frozen pre-test calibration, future-only alerting, and temporal consolidation of overlapping alerts.

It does not establish:

- detector failure
- detector degradation
- hardware health
- a physical causal mechanism
- CTI or readout-noise parameters
- quantum-field dynamics
- statistically independent physical events

The connected alert episodes are algorithmic clusters of temporally dependent alerts produced from overlapping measurement windows.

The appropriate description is:

causal-replay experimental measurement-state twin

## External Validation Question

Can QMS provide a useful and scientifically well-bounded way to characterize unexpected detector-measurement evolution when evaluated on an independently supplied experimental archive without preselected target windows?

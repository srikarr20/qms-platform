**Quantum Measurement Stack (QMS)**

**A Three-Twin Architecture for Measurement Observability and Adaptive
Inference**

**Srikar R.**

Independent Research

ORCID: 0009-0002-9398-0847

Contact: r.srikar.30@gmail.com

QMS Platform v0.4.0

Software: https://github.com/srikarr20/qms-platform

License: Apache-2.0

# Abstract

The Quantum Measurement Stack (QMS) is an open-source research and
software framework for studying what can be inferred about a physical or
quantum system from what its detector and measurement architecture make
observable.

QMS has evolved from earlier work on detector-plane imaging and
measurement observability into a provider-independent three-twin
architecture comprising a Quantum Twin, Detector Twin, and Measurement
Twin.

The Quantum Twin implements finite two-mode Gaussian dynamics,
observability analysis, state reconstruction, Kalman estimation,
physicality checks, parameter identification, model-family comparison,
adaptive estimation, and measurement recommendation. The Detector Twin
provides provider-independent residual analysis, stage-wise divergence
assessment, inverse calibration, mechanism ranking, prediction, and
detector-state comparison. The Measurement Twin operates directly on
evolving observable measurements and provides causal short-horizon
prediction, innovation analysis, frozen calibration, future-only
alerting, and connected alert-episode consolidation.

Current validation includes controlled computational experiments, a
Pyxel-based charge-transfer inefficiency parameter-recovery test, and
causal-replay analysis of archival single-photon experimental
measurements from the University of Glasgow.

The present evidence demonstrates internal computational consistency,
finite-dimensional observability and estimation, detector-model
parameter recovery under matched forward-model assumptions, and causal
measurement-state tracking on real experimental data. It does not
establish quantum-field reconstruction, universal quantum tomography,
independently validated real-detector calibration, physical fault
attribution, or prospective experimental validation.

The next validation stage is blind evaluation using independently
configured detector simulations and held-out experimental measurements.

# 1. Introduction

Measurement systems do not provide direct access to an underlying
physical state. They produce representations shaped by detector
response, acquisition architecture, noise, sampling, processing, and
model assumptions.

> ***What can be inferred about an underlying physical or quantum system
> from what its measurement architecture actually makes observable?***

The Quantum Measurement Stack was developed to investigate this question
computationally.

The research lineage leading to QMS is:

Imprint Hypothesis -\> Detector-Plane Imaging -\> Measurement
Observability -\> QMS / QMVA -\> AURORA -\> Adaptive Measurement Twin

The current QMS implementation consolidates these developments into a
reusable architecture with three complementary twin layers.

# 2. Three-Twin Architecture

The current QMS architecture is organized as:

QMS\
\|\
\|-- Quantum Twin\
\|-- Detector Twin\
\`-- Measurement Twin

These layers address different inference questions while sharing a
provider-independent measurement and inference philosophy.

## 2.1 Quantum Twin

> ***What hidden finite-dimensional state could have produced the
> observed measurements, and is that state observable under the current
> measurement architecture?***

The current implementation uses a finite two-mode Gaussian state
representation:

x = \[x1, p1, x2, p2\]

Current capabilities include:

- linear Gaussian dynamics

- observability-matrix construction

- rank and null-space analysis

- conditioning analysis

- state reconstruction

- Kalman prediction and update

- Gaussian physicality checks

- residual and divergence analysis

- parameter identification

- model-family comparison

- adaptive parameter estimation

- measurement-configuration recommendation

The implementation is computational and finite-dimensional. It does not
constitute general quantum-state tomography, quantum-field
reconstruction, particle-trajectory inference, or experimentally
validated quantum dynamics.

# 3. Detector Twin

> ***What detector or measurement-chain behavior could explain the
> observed detector output?***

The provider-independent detector layer includes:

- residual analysis

- stage-wise divergence detection

- mechanism ranking

- monotonic inverse calibration

- detector prediction

- innovation analysis

- detector-state assessment

Provider-specific detector physics are deliberately kept outside the QMS
core. This separation allows QMS to interact with detector simulators or
real detector systems without embedding a particular provider's
implementation into the core architecture.

# 4. Measurement Twin

> ***Is the observable measurement state evolving differently from what
> the current predictive model expects?***

This layer does not require a fully specified detector-physics model.
Current capabilities include:

- measurement-state construction

- cumulative-to-incremental reconstruction

- normalized measurement distributions

- cosine-similarity analysis

- Jensen-Shannon divergence

- constant-velocity state prediction

- innovation calculation

- frozen robust calibration

- future-only alerting

- connected alert-episode consolidation

- causal replay

This layer is intended to characterize unexpected measurement evolution
without prematurely assigning a physical mechanism.

# 5. Pyxel Detector-Twin Integration

QMS includes a separate integration with the ESA-developed Pyxel
detector-simulation framework. Pyxel provides the detector forward-model
environment, while QMS provides observability, residual-analysis,
inference, and adaptive-twin logic around the detector outputs.

A controlled charge-transfer inefficiency experiment was performed using
Pyxel's charge-transfer model.

| Quantity                  | Result              |
|---------------------------|---------------------|
| Known trap density        | 5.0e9 cm^-3         |
| Recovered trap density    | ~4.99999948e9 cm^-3 |
| Relative error            | ~1.05e-7            |
| Pixel-stage residual RMSE | ~1.53e-7            |

This result demonstrates numerical and inferential consistency within
the tested Pyxel model family. It is a same-forward-model computational
recovery test. It should not be interpreted as independent detector
validation, real-hardware calibration, or evidence of general
detector-parameter inference.

## 5.1 Pyxel Citation

Arko, M., Prod'homme, T., Lemmel, F., Serra, B., George, E. M., Kelman,
B., Pichon, T., Biancalani, E., & Gilbert, J. (2022). Pyxel 1.0: an open
source Python framework for detector and end-to-end instrument
simulation. Journal of Astronomical Telescopes, Instruments, and
Systems, 8(4), 048002. DOI: https://doi.org/10.1117/1.JATIS.8.4.048002

Project: https://esa.gitlab.io/pyxel/

Source: https://gitlab.com/esa/pyxel

# 6. Glasgow Experimental Measurement-State Twin

QMS also includes a separate integration for archival single-photon
experimental measurements from the University of Glasgow.

The current causal-replay analysis uses the archive:

Heralded Diffraction SM.zip

The workflow reconstructs incremental measurements from cumulative
detector frames and then applies frozen-reference causal prediction and
innovation analysis.

| Configuration / outcome  | Value       |
|--------------------------|-------------|
| Reference end            | 399         |
| Calibration end          | 699         |
| Calibration samples      | 199         |
| Median innovation        | ~0.00448429 |
| MAD                      | ~0.00229512 |
| Frozen threshold         | ~0.02490077 |
| Future samples           | 3369        |
| Raw future flags         | 64          |
| Connected alert episodes | 6           |

Connected episode peak windows:

951, 1598, 2316, 2761, 3458, 3875

These are algorithmic connected clusters of temporally dependent alerts.
They are not interpreted as statistically independent physical events,
detector failures, or identified microscopic mechanisms.

The appropriate description of the present implementation is:
causal-replay experimental measurement-state twin.

## 6.1 Glasgow Dataset Citation

Aspden, R. S., Padgett, M., & Spalding, G. (2016). Video recording true
single-photon double-slit interference \[Data collection\]. University
of Glasgow. DOI: https://doi.org/10.5525/gla.researchdata.281

Associated publication: Aspden, R. S., Padgett, M. J., & Spalding, G. C.
(2016). Video recording true single-photon double-slit interference.
American Journal of Physics, 84(9), 671-677. DOI:
https://doi.org/10.1119/1.4955173

# 7. Software Architecture and Reproducibility

The current architecture separates reusable QMS logic from
provider-specific integrations:

qms_core/\
\|-- quantum_twin/\
\|-- detector_twin/\
\`-- measurement_twin/\
\
integrations/\
\|-- pyxel/\
\`-- glasgow/

Focused external-review packages are included under:

validation_packages/\
\|-- pyxel/\
\`-- glasgow/

The current test suite includes unit tests for the three QMS twin
layers, the Pyxel CTI integration regression, and the Glasgow
causal-replay integration regression.

| Release-check test          | Result               |
|-----------------------------|----------------------|
| Main test suite             | 10 passed, 1 skipped |
| Pyxel external regression   | 1 passed             |
| Glasgow external regression | 1 passed             |

The externally dependent tests require access to their corresponding
external data or simulation environments.

# 8. Scientific Boundaries

QMS v0.4.0 does not establish:

- quantum-field reconstruction

- general quantum-state tomography

- experimentally validated quantum dynamics

- universal detector-parameter inference

- independently validated real-detector calibration

- detector-health diagnosis

- physical fault attribution

- statistically independent physical-event discovery

- universal cross-instrument transfer

- prospective experimental validation

These boundaries are explicit because the present QMS objective is to
build and validate the measurement and inference architecture before
making stronger physical claims.

# 9. External Validation Roadmap

The strongest next experiments are:

1.  independently configured Pyxel simulations whose parameters are
    hidden from QMS

2.  model-family mismatch experiments

3.  multiple competing detector mechanisms

4.  held-out experimental detector sequences

5.  frozen analysis parameters before test-data inspection

6.  real detector calibration metadata where available

7.  prospective streaming measurement evaluation

These experiments are intended to test where the QMS architecture fails
as well as where it succeeds.

# 10. Conclusion

QMS v0.4.0 represents the transition from a collection of
measurement-observability experiments into a provider-independent
three-twin architecture.

The Quantum Twin addresses finite-dimensional hidden-state observability
and estimation. The Detector Twin addresses model-based detector
inference and residual behavior. The Measurement Twin addresses causal
evolution of observable experimental measurement states.

Current evidence supports computational observability and estimation,
same-model detector-parameter recovery, and causal-replay analysis on
real experimental measurements.

The next major milestone is blind external validation.

# Software Availability

Quantum Measurement Stack: https://github.com/srikarr20/qms-platform

Version: QMS Platform v0.4.0

License: Apache-2.0

# Author

Srikar R.\
Independent Research\
ORCID: 0009-0002-9398-0847\
r.srikar.30@gmail.com

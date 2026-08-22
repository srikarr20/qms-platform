# QMS-REP-002 — Representation Diagnostics Under Tomography Shot Noise

## Objective

Test whether covariance-spectrum representation diagnostics track measurement
quality degradation in the existing QMS two-qubit tomography architecture.

## Data

The experiment uses the 36-measurement KwiatLab Bell-state reference dataset
already used by QMS-QST-001 and QMS-QST-002.

Poisson count noise was simulated at count scales:

- 10.0
- 3.0
- 1.0
- 0.3
- 0.1

Each condition used 500 trials.

## Diagnostics

For each count scale QMS calculated:

- covariance-spectrum effective dimension
- variance explained by the first two principal components
- variance explained by the first five principal components
- cosine similarity to the ideal normalized measurement vector

## Results

As count scale decreased from 10.0 to 0.1:

- mean similarity to the ideal measurement decreased from approximately
  0.9995 to 0.9516
- similarity variability increased substantially
- effective dimension remained approximately 18–19
- variance explained by the first two components remained approximately
  17–19%
- variance explained by the first five components remained approximately
  35–38%

## Interpretation

Poisson degradation is clearly detected by distance from the expected
measurement representation, but is not strongly reflected in raw covariance
participation ratio.

Therefore effective dimension should not be interpreted as a standalone
measurement-quality or information-richness score.

QMS should treat representation geometry, measurement consistency,
observability, and reconstruction quality as distinct diagnostics.

## Evidence level

Controlled computational validation using the existing QMS tomography
measurement architecture.

No claim is made that participation ratio alone measures quantum information
content or measurement quality.

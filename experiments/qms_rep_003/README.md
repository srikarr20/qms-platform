# QMS-REP-003 — Pre-Reconstruction Consistency vs Reconstruction Quality

## Objective

Determine whether diagnostics computed directly from the measurement
representation can indicate degradation in downstream quantum-state
reconstruction.

## Method

The experiment uses the QMS two-qubit Bell-state tomography architecture.

Poisson count noise was applied at count scales:

- 10.0
- 3.0
- 1.0
- 0.3
- 0.1

Each condition used 300 trials, producing 1500 valid reconstructions.

For every trial QMS measured:

- cosine similarity between the noisy normalized measurement vector and the
  ideal reference measurement
- Bell-state fidelity after linear reconstruction
- minimum eigenvalue of the reconstructed density matrix

## Results

Across 1500 trials:

- similarity–fidelity correlation: approximately -0.007
- similarity–minimum-eigenvalue correlation: approximately 0.874
- mean fidelity: 1.0
- mean minimum eigenvalue: approximately -0.065

## Interpretation

Similarity to the expected measurement representation did not predict the
reported Bell-state fidelity.

However, measurement similarity was strongly associated with the minimum
eigenvalue of the reconstructed density matrix.

This exposes an important distinction: a conventional target-state fidelity
metric can remain apparently ideal while linear reconstruction becomes
increasingly nonphysical.

The result motivates using pre-reconstruction measurement-consistency
diagnostics alongside physicality and observability checks.

## Evidence level

Controlled computational validation using one Bell-state tomography
architecture, Poisson count noise, and linear inversion.

The observed relationship between measurement similarity and reconstruction
physicality is a hypothesis requiring broader validation.

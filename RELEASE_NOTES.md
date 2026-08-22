# QMS Platform v0.2.0-paper1

## Scope
Paper-linked research software release consolidating the current QMS tomography, reconstruction-assurance, and observability experiments.

## Demonstrated capabilities
- Two-qubit Bell-state tomography reference reconstruction
- Context normalization and Born-rule consistency
- Linear inversion diagnostics
- Poisson and background-count degradation sweeps
- Physical-state projection and exact PSD trace-one projection
- Maximum-likelihood reconstruction
- Measurement-context removal and operator-rank analysis
- State-dependent context criticality
- 500-random-state / 4,500-reconstruction observability validation
- Noisy pseudoinverse error decomposition

## Key validated result
For the linear noisy model b = A x + epsilon with pseudoinverse reconstruction:
x_hat - x = -P_N(A)x + A^+epsilon
was numerically verified with residuals near floating-point precision.

## Limitations
This is an initial validation release. It does not claim universal detector coverage, universal quantum-platform validation, or production readiness. External real-data and broader detector-family validation are ongoing.

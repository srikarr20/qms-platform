# QMS-REP-001 — Representation Complexity vs Observable Sensitivity

## Objective

Test whether QMS representation diagnostics can distinguish increasing
measurement complexity from useful sensitivity to an underlying control
variable.

## Method

A controlled four-channel synthetic measurement representation was generated
from a scalar control variable.

Gaussian noise was added at levels:

- 0.00
- 0.01
- 0.05
- 0.10
- 0.20

For each condition QMS measured:

- effective dimension using covariance-spectrum participation ratio
- variance explained by the first two principal components
- energy sensitivity to the control variable
- variance sensitivity to the control variable

## Results

As noise increased:

- effective dimension increased from approximately 1.67 to 2.49
- variance explained by the first two components decreased from approximately
  1.00 to 0.83
- energy-control correlation decreased from approximately 0.88 to 0.44
- the energy regression slope remained approximately stable near 0.20
- variance showed weak control sensitivity throughout

## Interpretation

The experiment demonstrates that increased representation dimensionality does
not necessarily indicate increased useful measurement information.

Noise can increase apparent representation complexity while reducing the
ability of a detector-derived observable to track the underlying experimental
control.

This motivates treating representation richness and observable sensitivity as
separate QMS diagnostics.

## Evidence level

Controlled synthetic validation only.

This experiment does not establish the behavior for real quantum measurement
data.

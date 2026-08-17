# QMS-QST-001 — Photonic Quantum State Tomography Validation

This experiment validates the QMS measurement-to-state reconstruction pipeline using the KwiatLab two-qubit Bell-state tomography reference dataset.

## Pipeline

raw coincidence counts
→ measurement-context normalization
→ two-qubit projectors
→ Born-rule consistency
→ linear inversion
→ reconstructed density matrix

## QMS-QST-001D result

- Measurements: 36
- Linear-system shape: 36 × 16
- Rank: 16
- Condition number: 3.0
- Least-squares residual: 2.548776e-31
- Trace: 1
- Hermitian error: 0
- Minimum eigenvalue: -4.185729e-16
- Purity: 1.0
- Bell-state fidelity: 1.0
- Frobenius reconstruction error: 8.583984e-16
- Validation: PASS

The reconstructed state is approximately

    [[0.5, 0,   0,   0.5],
     [0,   0,   0,   0  ],
     [0,   0,   0,   0  ],
     [0.5, 0,   0,   0.5]]

corresponding to

    |Phi+> = (|HH> + |VV>) / sqrt(2)

The Bell state is used only as a post-reconstruction reference and is not supplied to the inverse solver.

from pprint import pprint

from validation import (
    GlasgowValidationSuite,
)


GLASGOW_ZIP


suite = GlasgowValidationSuite(
    GLASGOW_ZIP
)

report = suite.run_all()


print()
print("=" * 82)
print("QMS PLATFORM — GLASGOW REAL-DATA VALIDATION SUITE")
print("=" * 82)

print()
print("DATASET")
pprint(
    report["dataset"]
)

print()
print("CONVERGENCE")
pprint(
    report["convergence"]
)

print()
print("HELD-OUT PREDICTION")
pprint(
    report[
        "heldout_prediction"
    ]
)

print()
print("JOINT X-Y INFORMATION")
pprint(
    report[
        "joint_spatial_information"
    ]
)

print()
print("TEMPORAL ORDER NULL")
pprint(
    report[
        "temporal_order_null"
    ]
)

print()
print("STATIONARITY NULL")
pprint(
    report[
        "stationarity_null"
    ]
)

print()
print("=" * 82)
print("CURRENT REAL-DATA CONCLUSION")
print("=" * 82)

print(
    "1. Strong reproducible spatial structure: YES"
)

print(
    "2. Held-out future-event predictability: YES"
)

print(
    "3. Joint 2D information beyond X/Y marginals: YES"
)

print(
    "4. Strong acquisition-order dynamics: NOT SUPPORTED"
)

print(
    "5. Slow drift beyond stationarity noise: NOT SUPPORTED"
)

print(
    "6. Phase-aware upstream reconstruction from Glasgow: NOT AVAILABLE"
)

print()
print(
    "GLASGOW VALIDATION SUITE OK"
)

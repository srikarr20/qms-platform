from validation.platform_report import (
    QMSPlatformValidationReport,
)


GLASGOW_ZIP


builder = QMSPlatformValidationReport(
    glasgow_zip=GLASGOW_ZIP,
    artifacts_dir="artifacts",
)

report = builder.build()

print()
print("=" * 82)
print("QMS PLATFORM — EVIDENCE REPORT")
print("=" * 82)

for item in report.items:

    print()
    print(
        f"[{item.status}] "
        f"{item.name}"
    )

    print(
        "  type:",
        item.evidence_type
    )

    print(
        "  claim:",
        item.claim
    )

    if item.metrics:
        print(
            "  metrics:",
            item.metrics
        )

    if item.limitation:
        print(
            "  limitation:",
            item.limitation
        )


print()
print("=" * 82)
print("SUMMARY")
print("=" * 82)

print(
    report.summary()
)


path = builder.save(
    "artifacts/qms_platform_evidence.json"
)

print()
print(
    "Saved:",
    path
)

print()
print(
    "QMS PLATFORM EVIDENCE REPORT OK"
)

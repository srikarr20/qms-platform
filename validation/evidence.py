from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class EvidenceItem:
    name: str
    status: str
    evidence_type: str
    basis: str
    metrics: Dict[str, Any]
    claim: str
    limitation: str


class PlatformEvidenceReport:
    """
    Canonical evidence registry for QMS Platform.

    Status vocabulary:
        DEMONSTRATED
        SUPPORTED
        NOT_SUPPORTED
        NOT_TESTED
        NOT_AVAILABLE

    Evidence types:
        synthetic
        real_data
        architecture
    """

    def __init__(self):
        self.items: List[EvidenceItem] = []


    def add(
        self,
        *,
        name,
        status,
        evidence_type,
        basis,
        metrics=None,
        claim="",
        limitation="",
    ):
        self.items.append(
            EvidenceItem(
                name=name,
                status=status,
                evidence_type=evidence_type,
                basis=basis,
                metrics=metrics or {},
                claim=claim,
                limitation=limitation,
            )
        )


    def to_dict(self):
        return {
            "items": [
                asdict(item)
                for item in self.items
            ]
        }


    def summary(self):
        counts = {}

        for item in self.items:
            counts[item.status] = (
                counts.get(
                    item.status,
                    0
                )
                + 1
            )

        return counts

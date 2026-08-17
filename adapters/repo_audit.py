from pathlib import Path
import json
import re

DESKTOP = Path.home() / "Desktop"

CANDIDATES = {
    "quantum-measurement-stack":
        DESKTOP
        / "Quantum-Research"
        / "quantum-measurement-stack"
        / "repositories"
        / "quantum-measurement-stack",

    "quantum-measurement-stack-demo":
        DESKTOP
        / "Quantum-Research"
        / "quantum-measurement-stack"
        / "repositories"
        / "quantum-measurement-stack-demo",

    "aurora":
        DESKTOP
        / "aurora-github",
}

IGNORE = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
}

ENTRY_FILES = [
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "README.md",
    "README.rst",
]


def ignored(path):
    return any(part in IGNORE for part in path.parts)


def python_files(root):
    out = []
    for p in root.rglob("*.py"):
        if not ignored(p):
            out.append(p)
    return sorted(out)


def find_entry_files(root):
    found = {}
    for name in ENTRY_FILES:
        p = root / name
        if p.exists():
            found[name] = str(p)
    return found


def extract_symbols(pyfile):
    try:
        text = pyfile.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return {
            "classes": [],
            "functions": [],
        }

    classes = re.findall(
        r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)",
        text,
        flags=re.MULTILINE,
    )

    functions = re.findall(
        r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)",
        text,
        flags=re.MULTILINE,
    )

    return {
        "classes": classes,
        "functions": functions,
    }


def classify_path(path):
    s = str(path).lower()

    tags = []

    keyword_groups = {
        "acquisition": [
            "acquisition",
            "daq",
            "ingest",
            "capture",
            "stream",
        ],
        "detector": [
            "detector",
            "sensor",
            "camera",
        ],
        "propagation": [
            "propagation",
            "fresnel",
            "wavefront",
            "diffraction",
        ],
        "reconstruction": [
            "reconstruct",
            "inverse",
            "phase",
            "holog",
        ],
        "observability": [
            "observ",
            "coherence",
            "visibility",
            "manifold",
        ],
        "runtime": [
            "runtime",
            "engine",
            "orchestr",
        ],
        "validation": [
            "valid",
            "benchmark",
            "metric",
            "test",
        ],
        "trajectory": [
            "trajectory",
            "tracking",
            "forecast",
            "dynamics",
        ],
        "persistence": [
            "replay",
            "persist",
            "snapshot",
            "artifact",
        ],
    }

    for tag, keywords in keyword_groups.items():
        if any(k in s for k in keywords):
            tags.append(tag)

    return tags


def audit_repo(name, root):
    result = {
        "name": name,
        "root": str(root),
        "exists": root.exists(),
        "entry_files": {},
        "python_file_count": 0,
        "python_files": [],
        "tag_counts": {},
        "interesting_symbols": [],
    }

    if not root.exists():
        return result

    result["entry_files"] = find_entry_files(root)

    files = python_files(root)
    result["python_file_count"] = len(files)

    tag_counts = {}

    for p in files:
        rel = p.relative_to(root)

        tags = classify_path(rel)

        for tag in tags:
            tag_counts[tag] = (
                tag_counts.get(tag, 0) + 1
            )

        symbols = extract_symbols(p)

        record = {
            "path": str(rel),
            "tags": tags,
            "classes": symbols["classes"],
            "functions": symbols["functions"],
        }

        result["python_files"].append(record)

        if (
            tags
            or symbols["classes"]
        ):
            result["interesting_symbols"].append(
                record
            )

    result["tag_counts"] = dict(
        sorted(
            tag_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )
    )

    return result


def recommended_role(result):
    name = result["name"]

    mapping = {
        "qms-runtime":
            "runtime / acquisition / replay backbone",

        "dpi-runtime":
            "detector observability and DPI runtime logic",

        "dpi-validation-framework":
            "benchmark and validation layer",

        "quantum-measurement-stack-demo":
            "forward measurement model and reference experiments",

        "aurora":
            "dynamic observability / trajectories / forecasting",
    }

    return mapping.get(
        name,
        "unclassified",
    )


def main():
    reports = []

    for name, root in CANDIDATES.items():
        reports.append(
            audit_repo(
                name,
                root,
            )
        )

    out_json = (
        Path(__file__).resolve().parent
        / "repo_audit.json"
    )

    out_txt = (
        Path(__file__).resolve().parent
        / "repo_audit.txt"
    )

    out_json.write_text(
        json.dumps(
            reports,
            indent=2,
        )
    )

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "QMS PLATFORM — LOCAL REPOSITORY COMPATIBILITY AUDIT"
    )

    lines.append(
        "=" * 80
    )

    for result in reports:
        lines.append("")
        lines.append(
            result["name"]
        )
        lines.append(
            "-" * 80
        )

        lines.append(
            f"Path: {result['root']}"
        )

        lines.append(
            f"Exists: {result['exists']}"
        )

        lines.append(
            "Planned role: "
            + recommended_role(result)
        )

        if not result["exists"]:
            continue

        lines.append(
            f"Python files: "
            f"{result['python_file_count']}"
        )

        if result["entry_files"]:
            lines.append(
                "Package/config files:"
            )

            for key in sorted(
                result["entry_files"]
            ):
                lines.append(
                    f"  {key}"
                )

        lines.append(
            "Capability-tag counts:"
        )

        if result["tag_counts"]:
            for tag, count in (
                result["tag_counts"].items()
            ):
                lines.append(
                    f"  {tag:16s} {count}"
                )
        else:
            lines.append(
                "  none detected"
            )

        lines.append(
            "Interesting Python modules:"
        )

        shown = 0

        for item in result[
            "interesting_symbols"
        ]:
            if shown >= 20:
                break

            classes = ", ".join(
                item["classes"][:5]
            )

            funcs = ", ".join(
                item["functions"][:5]
            )

            description = []

            if item["tags"]:
                description.append(
                    "tags="
                    + ",".join(
                        item["tags"]
                    )
                )

            if classes:
                description.append(
                    "classes="
                    + classes
                )

            if funcs:
                description.append(
                    "funcs="
                    + funcs
                )

            lines.append(
                "  "
                + item["path"]
                + (
                    "  ["
                    + "; ".join(
                        description
                    )
                    + "]"
                    if description
                    else ""
                )
            )

            shown += 1

    lines.append("")
    lines.append(
        "=" * 80
    )

    lines.append(
        "AUDIT OUTPUT"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        str(out_txt)
    )

    lines.append(
        str(out_json)
    )

    out_txt.write_text(
        "\n".join(lines)
    )

    print(
        "\n".join(lines)
    )


if __name__ == "__main__":
    main()

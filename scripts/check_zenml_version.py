"""Assert the pinned ZenML version is consistent everywhere in the repo.

Canonical source: `pyproject.toml [project].dependencies` -> `zenml==X.Y.Z`.
Every other reference to a ZenML version must match. With --remote, also
asserts `ghcr.io/hotosm/zenml-postgres:VERSION` exists on the registry.
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
GHCR_REPO = "hotosm/zenml-postgres"

DEP_RE = re.compile(r"^zenml(?:\[[^\]]+\])?\s*(==|>=)\s*(\S+)$")


def canonical_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text())
    for dep in data["project"]["dependencies"]:
        if dep.startswith("zenml==") or dep == "zenml":
            return dep.removeprefix("zenml==")
    raise RuntimeError("pyproject.toml [project].dependencies has no `zenml==X.Y.Z` pin")


def check_pyproject(version: str) -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    locations: list[tuple[str, str]] = []
    locations.extend(("[project].dependencies", d) for d in data["project"].get("dependencies", []))
    for name, deps in data["project"].get("optional-dependencies", {}).items():
        locations.extend((f"[project.optional-dependencies].{name}", d) for d in deps)
    for name, deps in data.get("dependency-groups", {}).items():
        locations.extend((f"[dependency-groups].{name}", d) for d in deps)

    failures: list[str] = []
    for section, dep in locations:
        if not isinstance(dep, str) or not dep.startswith("zenml"):
            continue
        m = DEP_RE.match(dep)
        if not m:
            failures.append(f"pyproject.toml {section}: unparseable zenml spec '{dep}'")
            continue
        found = m.group(2)
        if found != version:
            failures.append(f"pyproject.toml {section}: '{dep}' pins {found}, expected {version}")
    return failures


def check_text_file(path: Path, label: str, pattern: re.Pattern[str], version: str) -> list[str]:
    rel = path.relative_to(REPO)
    if not path.exists():
        return [f"{rel}: missing"]
    matches = pattern.findall(path.read_text())
    if not matches:
        return [f"{rel}: no match for {label} (pattern: {pattern.pattern})"]
    return [f"{rel}: {label} = {m}, expected {version}" for m in matches if m != version]


FILE_CHECKS: list[tuple[Path, str, re.Pattern[str]]] = [
    (REPO / "stacks/compose.yaml", "zenml_version", re.compile(r"^zenml_version:\s*(\S+)\s*$", re.M)),
    (REPO / "stacks/dok8s.yaml", "zenml_version", re.compile(r"^zenml_version:\s*(\S+)\s*$", re.M)),
    (
        REPO / "infra/compose/docker-compose.yml",
        "zenml-postgres image tag",
        re.compile(r"ghcr\.io/hotosm/zenml-postgres:(\S+)"),
    ),
    (
        REPO / "infra/values/zenml.yaml.gotmpl",
        "tag",
        re.compile(r'^\s*tag:\s*"([^"]+)"\s*$', re.M),
    ),
]


def check_ghcr(version: str) -> str | None:
    token_url = f"https://ghcr.io/token?service=ghcr.io&scope=repository:{GHCR_REPO}:pull"
    token = httpx.get(token_url, timeout=15).raise_for_status().json()["token"]
    accept = ",".join(
        [
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        ]
    )
    resp = httpx.head(
        f"https://ghcr.io/v2/{GHCR_REPO}/manifests/{version}",
        headers={"Authorization": f"Bearer {token}", "Accept": accept},
        timeout=15,
    )
    if resp.status_code == 200:
        return None
    return f"ghcr.io/{GHCR_REPO}:{version} not reachable (HTTP {resp.status_code})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true", help="Probe GHCR for the image tag.")
    args = parser.parse_args()

    version = canonical_version()
    print(f"canonical ZenML version: {version}  (pyproject.toml [project].dependencies)")

    failures = check_pyproject(version)
    for path, label, pat in FILE_CHECKS:
        failures.extend(check_text_file(path, label, pat, version))

    if args.remote:
        ghcr_failure = check_ghcr(version)
        if ghcr_failure:
            failures.append(ghcr_failure)
        else:
            print(f"ghcr.io/{GHCR_REPO}:{version} exists")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Rebuild the SHA-256 manifest for the frozen reproducibility release."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checksums_sha256.csv"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included_files() -> list[Path]:
    """Return deterministic release files, excluding caches and this manifest."""

    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        if EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    rows = [
        {
            "relative_path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in included_files()
    ]
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "bytes"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} checksums to {MANIFEST}")


if __name__ == "__main__":
    main()

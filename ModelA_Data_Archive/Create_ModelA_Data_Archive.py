"""Create uploadable archive parts for large ModelA generated data.

This script packs V_Wave_Data and V_Wave_Data_Hor into independent .tar parts,
each targeting roughly 4 GB before compression. It also writes SHA256 checksums
and a small README for uploading the parts to Aliyun Drive.
"""

from __future__ import annotations

import hashlib
import math
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [
    PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data",
    PROJECT_ROOT / "ModelA_Virtual_Internal_Solitary_Wave_Data_Generation" / "V_Wave_Data_Hor",
]
ARCHIVE_DIR = PROJECT_ROOT / "ModelA_Data_Archive"
TARGET_PART_BYTES = int(3.8 * 1024**3)


def dir_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def chunk_children(source_dir: Path) -> list[list[Path]]:
    children = sorted([child for child in source_dir.iterdir() if child.is_dir()])
    chunks: list[list[Path]] = []
    current: list[Path] = []
    current_size = 0

    for child in children:
        size = dir_size(child)
        if current and current_size + size > TARGET_PART_BYTES:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(child)
        current_size += size

    if current:
        chunks.append(current)
    return chunks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_part(source_dir: Path, children: list[Path], part_index: int, part_count: int) -> Path:
    archive_name = f"{source_dir.name}_part{part_index:03d}_of_{part_count:03d}.tar"
    archive_path = ARCHIVE_DIR / archive_name
    if archive_path.exists():
        print(f"skip existing {archive_path}")
        return archive_path

    print(f"writing {archive_path}")
    with tarfile.open(archive_path, "w") as tar:
        for child in children:
            arcname = source_dir.relative_to(PROJECT_ROOT) / child.name
            tar.add(child, arcname=str(arcname))
    return archive_path


def write_readme(archive_paths: list[Path], checksums: list[tuple[str, Path]]) -> None:
    readme = ARCHIVE_DIR / "DataArchive.md"
    total_gb = sum(path.stat().st_size for path in archive_paths) / 1024**3
    lines = [
        "# ModelA Data Archive",
        "",
        "These archive parts contain generated ModelA data folders only.",
        "The generation programs under `V_Wave_Data_Generate*` are not included.",
        "",
        f"- Archive parts: {len(archive_paths)}",
        f"- Total archive size: {total_gb:.2f} GB",
        "- Upload target: Aliyun Drive",
        "",
        "## Restore",
        "",
        "Extract every `.tar` part from the repository root. Each part contains",
        "`ModelA_Virtual_Internal_Solitary_Wave_Data_Generation/V_Wave_Data*`",
        "relative folder content.",
        "",
        "## Checksums",
        "",
        "Run this in PowerShell after download:",
        "",
        "```powershell",
        "Get-FileHash *.tar -Algorithm SHA256",
        "```",
        "",
        "Compare the result with `checksums.sha256`.",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")

    checksum_path = ARCHIVE_DIR / "checksums.sha256"
    checksum_lines = [f"{digest}  {path.name}" for digest, path in checksums]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def main() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_paths: list[Path] = []

    for source_dir in SOURCE_DIRS:
        chunks = chunk_children(source_dir)
        part_count = len(chunks)
        print(f"{source_dir.name}: {part_count} parts")
        for index, children in enumerate(chunks, start=1):
            archive_paths.append(write_part(source_dir, children, index, part_count))

    checksums = []
    for index, path in enumerate(archive_paths, start=1):
        print(f"hashing {index}/{len(archive_paths)} {path.name}")
        checksums.append((sha256_file(path), path))

    write_readme(archive_paths, checksums)
    print(ARCHIVE_DIR)
    print(f"parts={len(archive_paths)}")
    print(f"size_gb={sum(path.stat().st_size for path in archive_paths) / 1024**3:.2f}")


if __name__ == "__main__":
    main()

import argparse
import re
import subprocess
import zipfile
from collections.abc import Iterator
from pathlib import Path

from harness.rules import MAX_UNZIPPED_BYTES

DEFAULT_INCLUDES = ("weights",)
SKIP = {"__pycache__", ".DS_Store"}

BUILD_ID_PATTERN = r'^BUILD_ID = .*'


def members(root: Path, includes: tuple[str, ...]) -> Iterator[tuple[Path, str]]:
    named: set[str] = set()
    for path in sorted(root.glob("*.py")):
        named.add(path.name)
        yield path, path.name
    for name in includes:
        if name in named:
            continue
        source = root / name
        if source.is_file():
            named.add(name)
            yield source, name
        elif source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and not SKIP & set(path.parts):
                    yield path, str(path.relative_to(root))


def current_build_id(root: Path) -> str:
    """Short commit hash, +dirty appended if the working tree has uncommitted
    changes - stamped into agent.py's BUILD_ID so a rated game's log always
    says exactly which commit produced it (see BUILD_ID's own comment)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("+dirty" if status.strip() else "")


def build(root: Path, destination: Path, includes: tuple[str, ...]) -> list[str]:
    entries = list(members(root, includes))
    written = [name for _, name in entries]
    if "agent.py" not in written:
        raise SystemExit(f"{root / 'agent.py'} does not exist; the platform imports it by name")

    build_id = current_build_id(root)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, name in entries:
            if name == "agent.py":
                source_text = source.read_text(encoding="utf-8")
                stamped, count = re.subn(
                    BUILD_ID_PATTERN, f'BUILD_ID = "{build_id}"', source_text,
                    count=1, flags=re.M)
                if count != 1:
                    raise SystemExit(
                        "agent.py: BUILD_ID pattern matched "
                        f"{count} lines, expected 1 - packaging cannot stamp the build id")
                archive.writestr(name, stamped)
            else:
                archive.write(source, name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a submission zip.")
    parser.add_argument("--out", type=Path, default=Path("submission.zip"))
    parser.add_argument("--include", action="append", default=[])
    arguments = parser.parse_args()

    includes = DEFAULT_INCLUDES + tuple(arguments.include)
    root = Path.cwd()
    written = build(root, arguments.out, includes)
    size = arguments.out.stat().st_size
    unzipped = sum((root / name).stat().st_size for name in written)
    print(f"{arguments.out} ({size:,} bytes, {unzipped:,} unzipped)")
    for name in written:
        print(f"  {name}")
    if unzipped > MAX_UNZIPPED_BYTES:
        print(
            f"\nwarning: {unzipped:,} bytes unzipped is over the "
            f"{MAX_UNZIPPED_BYTES // 1_000_000} MB limit. The platform will reject this upload"
        )


if __name__ == "__main__":
    main()
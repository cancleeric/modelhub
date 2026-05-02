"""
Sprint 7.5 — 版本號管理（單一來源 VERSION 檔）

使用：
- `from version import VERSION, BUILD_INFO`
- CLI bump：`python -m version bump patch` → 自動遞增並寫回 VERSION
"""
import os
import sys
from pathlib import Path

_VERSION_FILE = Path(__file__).parent / "VERSION"


def _read_version() -> str:
    try:
        return _VERSION_FILE.read_text().strip() or "0.0.0"
    except FileNotFoundError:
        return "0.0.0"


VERSION = _read_version()
BUILD_INFO = {
    "version": VERSION,
    "build": os.environ.get("MODELHUB_BUILD", "dev"),
    "commit": os.environ.get("MODELHUB_COMMIT", "local"),
}


def bump(level: str = "patch") -> str:
    """bump semver 並寫回 VERSION，回傳新版本"""
    parts = _read_version().split(".")
    while len(parts) < 3:
        parts.append("0")
    major, minor, patch = (int(x) for x in parts[:3])
    if level == "major":
        major, minor, patch = major + 1, 0, 0
    elif level == "minor":
        minor, patch = minor + 1, 0
    elif level == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump level: {level}")
    new_ver = f"{major}.{minor}.{patch}"
    _VERSION_FILE.write_text(new_ver + "\n")
    return new_ver


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "bump":
        new_ver = bump(sys.argv[2])
        print(new_ver)
    else:
        print(VERSION)

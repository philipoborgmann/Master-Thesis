# scripts/create_repo_context.py

from pathlib import Path
import os
import pandas as pd

ROOT = Path(".").resolve()
OUT = ROOT / "repo_context.md"

EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "transformer_cache",
}

DATA_EXTS = {".csv", ".parquet", ".xlsx", ".json", ".pkl"}
CODE_EXTS = {".py", ".yaml", ".yml", ".toml", ".md", ".txt"}

MAX_TREE_DEPTH = 4
MAX_FILES_PER_DIR = 80


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def make_tree(path: Path, prefix="", depth=0):
    if depth > MAX_TREE_DEPTH or is_excluded(path):
        return []

    lines = []
    try:
        items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return [f"{prefix}[permission denied] {path.name}"]

    items = items[:MAX_FILES_PER_DIR]

    for item in items:
        if is_excluded(item):
            continue
        rel = item.relative_to(ROOT)
        if item.is_dir():
            lines.append(f"{prefix}📁 {rel}/")
            lines.extend(make_tree(item, prefix + "  ", depth + 1))
        else:
            size = file_size_mb(item)
            lines.append(f"{prefix}📄 {rel} ({size:.2f} MB)")
    return lines


def inspect_csv(path: Path):
    try:
        df = pd.read_csv(path, nrows=500)
        return {
            "rows_sampled": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        }
    except Exception as e:
        return {"error": str(e)}


def inspect_parquet(path: Path):
    try:
        df = pd.read_parquet(path)
        info = {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        }

        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower() or col.lower() in {"timestamp", "datetime"}:
                try:
                    s = pd.to_datetime(df[col], errors="coerce")
                    info[f"{col}_min"] = str(s.min())
                    info[f"{col}_max"] = str(s.max())
                except Exception:
                    pass

        return info
    except Exception as e:
        return {"error": str(e)}


def inspect_excel(path: Path):
    try:
        xl = pd.ExcelFile(path)
        return {"sheets": xl.sheet_names}
    except Exception as e:
        return {"error": str(e)}


def inspect_data_files():
    results = []
    for path in ROOT.rglob("*"):
        if is_excluded(path) or not path.is_file():
            continue
        if path.suffix.lower() not in DATA_EXTS:
            continue

        rel = path.relative_to(ROOT)
        entry = {
            "path": str(rel),
            "extension": path.suffix.lower(),
            "size_mb": round(file_size_mb(path), 2),
        }

        if path.suffix.lower() == ".csv":
            entry["schema"] = inspect_csv(path)
        elif path.suffix.lower() == ".parquet":
            entry["schema"] = inspect_parquet(path)
        elif path.suffix.lower() == ".xlsx":
            entry["schema"] = inspect_excel(path)
        else:
            entry["schema"] = "not inspected"

        results.append(entry)
    return results


def inspect_code_files():
    files = []
    for path in ROOT.rglob("*"):
        if is_excluded(path) or not path.is_file():
            continue
        if path.suffix.lower() in CODE_EXTS:
            files.append(str(path.relative_to(ROOT)))
    return sorted(files)


def main():
    tree_lines = make_tree(ROOT)
    data_files = inspect_data_files()
    code_files = inspect_code_files()

    lines = []
    lines.append("# Repository Context Pack\n")
    lines.append("## Project root\n")
    lines.append(f"`{ROOT}`\n")

    lines.append("## Directory tree\n")
    lines.append("```text")
    lines.extend(tree_lines)
    lines.append("```\n")

    lines.append("## Code/config/documentation files\n")
    lines.append("```text")
    lines.extend(code_files)
    lines.append("```\n")

    lines.append("## Data inventory and schemas\n")

    for item in data_files:
        lines.append(f"### `{item['path']}`")
        lines.append(f"- Extension: `{item['extension']}`")
        lines.append(f"- Size: `{item['size_mb']} MB`")
        lines.append("- Schema / metadata:")
        lines.append("```text")
        lines.append(str(item["schema"]))
        lines.append("```\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
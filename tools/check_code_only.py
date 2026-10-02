"""Fail if tracked repository files contain data artifacts or notebook outputs."""
import json
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
    data_extensions = {".npy", ".npz", ".h5", ".hdf5", ".tif", ".tiff", ".msh", ".obs",
                       ".csv", ".tsv", ".xyz", ".xlsx", ".xls", ".pdf", ".docx", ".zip",
                       ".png", ".jpg", ".jpeg", ".svg", ".log"}
    errors = []
    for name in filter(None, tracked):
        path = root / name
        if path.suffix.lower() in data_extensions:
            errors.append(f"Data/output artifact tracked: {name}")
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8-sig"))
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("outputs") or cell.get("attachments") or cell.get("execution_count") is not None:
                    errors.append(f"Notebook output/attachment/execution count: {name}, cell {index}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Code-only repository policy passed.")


if __name__ == "__main__":
    main()

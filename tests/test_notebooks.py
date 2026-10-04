"""Check portable notebook setup and keep published notebooks output-free."""
import ast
import json
from pathlib import Path

import pytest


NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize("path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.name)
@pytest.mark.parametrize("shared_workspace", [False, True])
def test_notebook_code_and_workspace_setup(path, monkeypatch, tmp_path, shared_workspace):
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    for cell in cells:
        source = "".join(cell["source"])
        compile(source, path.name, "exec")
        assert not cell.get("outputs")
        assert cell.get("execution_count") is None
        assert "Path.cwd()" not in source
        assert "sys.path.insert" not in source
    # Execute the shared setup without triggering inversion, LLM calls or plotting.
    parsed = ast.parse("".join(cells[0]["source"]))
    setup = []
    for node in parsed.body:
        if isinstance(node, ast.ImportFrom) and node.module == "geosage.paths":
            setup.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ROOT" for t in node.targets):
            setup.append(node)
            break
    assert len(setup) == 2
    # Exercise actual notebook setup in both supported layouts, independently
    # of the developer's own workspace marker or environment override.
    monkeypatch.delenv("GEOSAGE_WORKSPACE", raising=False)
    checkout = tmp_path / "GeoSAGE"
    (checkout / "notebooks").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text('[project]\nname="geosage"\n')
    if shared_workspace:
        (tmp_path / ".geosage-workspace").touch()
    monkeypatch.chdir(checkout / "notebooks")
    namespace = {}
    exec(compile(ast.Module(body=setup, type_ignores=[]), path.name, "exec"), namespace)
    assert namespace["ROOT"] == (tmp_path if shared_workspace else checkout)

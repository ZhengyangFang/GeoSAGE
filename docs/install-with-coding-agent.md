# Install with Claude Code or Codex

Claude Code and Codex can download both repositories, create the `uv`
environment, verify the installation, and open Professional Studio. They are
installers in this workflow; GeoSAGE still runs locally in PyHydroGeophysX.

## 1. Open an empty workspace

Create or choose a folder that will contain the two source repositories. Open
that folder in Claude Code or Codex. Do not open a folder containing private
survey data unless you intend to give the coding agent access to it.

The finished layout should be:

```text
GeoSAGE-Workspace/
├── PyHydroGeophysX/
├── GeoSAGE/
└── .venv-studio/
```

## 2. Paste the installation request

Copy the text from
[`install-agent-prompt.txt`](install-agent-prompt.txt) into Claude Code or
Codex. The request tells the agent to:

- use the official repositories;
- use one small `uv` environment with Python 3.12;
- reuse an existing checkout instead of making duplicates;
- install PyHydroGeophysX and GeoSAGE side by side;
- run import, quick, and Studio self-tests before reporting success; and
- avoid downloading research inputs, results, or requesting an API key.

Review the proposed directory and commands, then allow the Git clone and `uv`
installation when your coding agent asks. Installation can download Python
packages, but it should not alter the scientific code.

## 3. Check the result

On Windows, a successful source installation can be reopened from the workspace
with:

```powershell
.\.venv-studio\Scripts\geosage-studio.exe --module one_click
```

On macOS or Linux:

```bash
./.venv-studio/bin/geosage-studio --module one_click
```

Professional Studio should open with the normal PyHydroGeophysX home and AQUAH
as the default assistant. Open **Assistant** and select **GeoSAGE** when you want
the gravity–magnetic modeling workflow.

## 4. Keep research files outside Git

Place private inputs in a local `data/` directory and generated products in
`outputs/`, or point `GEOSAGE_WORKSPACE` to another local folder. These paths are
ignored by the GeoSAGE repository. Never paste an API key into a tracked file;
use the Studio session settings or your coding agent's documented secret store.

For manual commands and troubleshooting, continue with the
[Studio integration guide](studio.md).

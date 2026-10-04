"""Task-first desktop setup; the same configuration resolver drives the run."""

from html import escape
from pathlib import Path
import json
from copy import deepcopy

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QLabel, QLineEdit, QPushButton, QTextBrowser, QVBoxLayout, QWidget,
                              QDialog, QDialogButtonBox, QFormLayout, QPlainTextEdit, QHBoxLayout)

from .configuration import configure, FILE_ROLES


class WorkflowSetup(QWidget):
    """Optional host setup interface: update_inputs, request, prepare_payload."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.inputs = {}
        self._configuration = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.task = QComboBox()
        for label, key in (("Inspect existing results · local", "inspect"),
                           ("Run a new inversion · local", "invert"),
                           ("Interpret existing results · with AI", "interpret")):
            self.task.addItem(label, key)
        layout.addWidget(QLabel("What would you like to do?"))
        layout.addWidget(self.task)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.goal = QLineEdit()
        self.goal.setPlaceholderText("Optional objective or question about these results")
        self.goal.setAccessibleName("Analysis objective")
        layout.addWidget(self.goal)
        self.edit = QPushButton("Create / edit inversion configuration")
        self.edit.clicked.connect(self.edit_configuration)
        layout.addWidget(self.edit)
        self.check = QPushButton("Check inputs && preview configuration")
        checks = QHBoxLayout()
        checks.addWidget(self.check)
        layout.addLayout(checks)
        self.preview = QTextBrowser()
        self.preview.setMinimumHeight(90)
        self.preview.setMaximumHeight(140)
        self.preview.hide()
        self.preview.setAccessibleName("Configuration and input readiness")
        layout.addWidget(self.preview)
        self.check.clicked.connect(self.refresh_preview)
        self.all_parameters = QPushButton("View all resolved parameters")
        self.all_parameters.clicked.connect(self.show_parameters)
        checks.addWidget(self.all_parameters)
        self.task.currentIndexChanged.connect(self._task_changed)
        self._task_changed()

    @property
    def needs_ai(self):
        return self.task.currentData() == "interpret"

    @property
    def action_label(self):
        return {"inspect": "Open results", "invert": "Run inversion", "interpret": "Generate interpretation"}[self.task.currentData()]

    @property
    def primary_role(self):
        return 'config_file' if self.task.currentData() == 'invert' else 'source_inversion_dir'

    def request(self):
        return (self.goal.text().strip() if self.needs_ai else '') or {
            "inspect": "Inspect the existing numerical models and preserve their labels.",
            "invert": "Run the configured joint inversion and summarize its numerical evidence.",
            "interpret": "Interpret these existing results using the supplied geological evidence and review the report.",
        }[self.task.currentData()]

    def show_error(self, message):
        self.preview.show()
        self.preview.setPlainText(str(message))

    def _task_changed(self):
        self.edit.setVisible(self.task.currentData() == "invert")
        self.goal.setVisible(self.needs_ai)
        self.note.setText({
            "inspect": "Add an existing inversion folder. No AI account is needed. Recorded labels are preserved; missing labels use explicitly identified property clusters.",
            "invert": "Add a GeoSAGE JSON configuration and its survey inputs. The preview is the authority for this run; independent processing tools do not change these settings.",
            "interpret": "Add existing results and optional geological references. AI receives the goal and structured evidence/reference text. Configure the provider in Assistant settings.",
        }[self.task.currentData()])
        self.preview.setPlainText("Add the files below, then check inputs before starting.")
        self.preview.hide()
        self.changed.emit()

    def allowed_roles(self):
        common = {"config_file", "unit_defs_file", "unit_groups_file", "reference_file"}
        return common | (set(FILE_ROLES) | {"input_dir"} if self.task.currentData() == "invert" else {"source_inversion_dir"})

    def update_inputs(self, inputs):
        if inputs.get("config_file") != self.inputs.get("config_file"):
            self._configuration = None
        self.inputs = dict(inputs)
        self.preview.setPlainText("Inputs changed. Check the configuration before starting; it is validated again at launch.")

    def prepare_payload(self, payload):
        payload = dict(payload, studio_task=self.task.currentData())
        payload['inputs'] = {k: v for k, v in payload['inputs'].items() if k in self.allowed_roles()}
        if self._configuration is not None and self.task.currentData() == 'invert':
            payload["config"] = deepcopy(self._configuration)
            payload["inputs"] = {k: v for k, v in payload["inputs"].items() if k != "config_file"}
        cfg = configure(payload)
        project = cfg["project"]
        task_name = {'inspect': 'Inspect', 'invert': 'Inversion', 'interpret': 'AI interpretation'}[self.task.currentData()]
        project_label = project['name']
        if project_label == 'GeoSAGE' and project.get('source_inversion_dir'):
            project_label = Path(project['source_inversion_dir']).name
        payload['run_label'] = f'{project_label} · {task_name}'
        if cfg["run"]["execution_mode"] == "interpret_existing":
            from geosage.existing_results import REQUIRED_ARTIFACTS

            required = [Path(project["source_inversion_dir"]) / p for p in REQUIRED_ARTIFACTS.values()]
        else:
            required = [Path(payload["inputs"].get(role) or Path(project["input_dir"]) / f'{project["name"]}_{suffix}')
                        for role, suffix in FILE_ROLES.items()]
        required += [Path(cfg["geology"][k]) for k in ("unit_defs_csv", "unit_groups_csv", "context_path", "unit_id_npy") if cfg["geology"].get(k)]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise ValueError("Missing input files: " + "; ".join(missing))
        if self.needs_ai and not payload.get("api_key"):
            raise ValueError("Configure an AI provider in Assistant settings, then start again.")
        return payload

    def refresh_preview(self):
        # A hypothetical child of a fresh sibling run folder; no file is created.
        import tempfile

        self.preview.show()
        payload = {"inputs": self.inputs, "request": self.request(),
                   "output_dir": str(Path(tempfile.gettempdir()) / "geosage-preflight" / "preview"),
                   "api_key": "preview-only" if self.needs_ai else None}
        self.show_configuration(payload)

    def show_configuration(self, payload):
        """Display the same resolved inputs and settings sent to the worker."""
        self.preview.show()
        try:
            payload = self.prepare_payload(payload)
            cfg = configure(payload)
            project = cfg["project"]
            rows = [("Project", project["name"]), ("Execution", cfg["run"]["execution_mode"]),
                    ("Geology", cfg["geology"]["mode"]),
                    ("Source", project.get("source_inversion_dir") or project["input_dir"]),
                    ("AI", "Interpretation + independent review" if self.needs_ai else "Not used")]
            if cfg["run"]["execution_mode"] == "full":
                r, inv = cfg["region"], cfg["inversion"]
                rows += [("Region (m)", f'E {r["min_e"]}–{r["max_e"]}; N {r["min_n"]}–{r["max_n"]}'),
                         ("Gravity", f'{cfg["data"]["gravity_column"]} / {cfg["data"]["gravity_component"]}'),
                         ("Magnetic field", f'{inv["field_strength"]} nT; inclination {inv["inclination"]}°; declination {inv["declination"]}°'),
                         ("Maximum iterations", inv['optimization']['maxGNCG'])]
                rows += [(role.replace('_', ' ').title(), cfg['studio_inputs'].get(role) or
                          str(Path(project['input_dir']) / f'{project["name"]}_{suffix}'))
                         for role, suffix in FILE_ROLES.items()]
            else:
                rows.append(('Core mesh', str(Path(project['source_inversion_dir']) / 'mesh/mesh_core.msh')))
            self.preview.setHtml("<b>Files ready · numerical validation runs at startup</b><table>" + "".join(
                f"<tr><td>{escape(k)}</td><td>{escape(str(v))}</td></tr>" for k, v in rows) + "</table>")
        except (ValueError, KeyError, OSError) as exc:
            self.preview.setPlainText(f"Action needed: {exc}")

    def edit_configuration(self):
        from geosage.paths import resolve_config_paths

        try:
            base = deepcopy(self._configuration or {})
            if not base and self.inputs.get('config_file'):
                path = Path(self.inputs['config_file'])
                base = json.loads(path.read_text(encoding='utf-8-sig'))
                if not isinstance(base, dict):
                    raise ValueError('Configuration must be a JSON object.')
                for section in ('project', 'region', 'data', 'inversion', 'geology', 'run'):
                    if section in base and not isinstance(base[section], dict):
                        raise ValueError(f'Configuration section {section} must be an object.')
                base.setdefault('project', {})
                base = resolve_config_paths(base, config_path=path)
        except (ValueError, KeyError, OSError) as exc:
            self.preview.show()
            self.preview.setPlainText(str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Inversion settings · local run copy')
        layout = QFormLayout(dialog)
        note = QLabel('Enter the physical survey settings. Imported advanced parameters are retained. The source JSON is never modified.')
        note.setWordWrap(True)
        layout.addRow(note)
        fields = [
            ('Project name', 'project', 'name', str),
            ('Minimum easting (m)', 'region', 'min_e', float), ('Maximum easting (m)', 'region', 'max_e', float),
            ('Minimum northing (m)', 'region', 'min_n', float), ('Maximum northing (m)', 'region', 'max_n', float),
            ('Gravity column', 'data', 'gravity_column', str), ('Gravity component (e.g. gz)', 'data', 'gravity_component', str),
            ('Field strength (nT)', 'inversion', 'field_strength', float),
            ('Inclination (degrees)', 'inversion', 'inclination', float), ('Declination (degrees)', 'inversion', 'declination', float),
        ]
        edits = []
        for label, section, key, kind in fields:
            edit = QLineEdit(str(base.get(section, {}).get(key, '')))
            edit.setAccessibleName(label)
            layout.addRow(label, edit)
            edits.append(edit)
        error = QLabel()
        error.setWordWrap(True)
        layout.addRow(error)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addRow(buttons)

        def accept():
            try:
                import math

                changed = deepcopy(base)
                for (_, section, key, kind), edit in zip(fields, edits):
                    text = edit.text().strip()
                    if not text:
                        raise ValueError('Every listed survey setting is required.')
                    value = kind(text)
                    if kind is float and not math.isfinite(value):
                        raise ValueError('Survey settings must be finite numbers.')
                    changed.setdefault(section, {})[key] = value
                r = changed['region']
                if r['min_e'] >= r['max_e'] or r['min_n'] >= r['max_n']:
                    raise ValueError('Each region minimum must be below its maximum.')
                if changed['inversion']['field_strength'] <= 0:
                    raise ValueError('Magnetic field strength must be positive.')
                if not -90 <= changed['inversion']['inclination'] <= 90:
                    raise ValueError('Inclination must be between -90 and 90 degrees.')
                changed.setdefault('run', {}).update(execution_mode='full', run_inversion=True)
                changed['project']['source_inversion_dir'] = None
                self._configuration = changed
                dialog.accept()
                self.preview.show()
                self.preview.setPlainText('Edited run configuration ready. Add survey inputs and check the complete configuration.')
            except ValueError as exc:
                error.setText(str(exc))

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()

    def show_parameters(self):
        import tempfile

        try:
            payload = self.prepare_payload({'inputs': self.inputs, 'request': self.request(),
                'api_key': 'preview-only' if self.needs_ai else None,
                'output_dir': str(Path(tempfile.gettempdir()) / 'geosage-preflight/preview')})
            cfg = configure(payload)
        except (ValueError, KeyError, OSError) as exc:
            self.preview.show()
            self.preview.setPlainText(str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Resolved run configuration · output location assigned at launch')
        dialog.resize(680, 500)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit(json.dumps(cfg, indent=2))
        text.setReadOnly(True)
        layout.addWidget(text)
        dialog.exec()

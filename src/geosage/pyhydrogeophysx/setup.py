"""Task-first desktop setup; the same configuration resolver drives the run."""

import json
from copy import deepcopy
from html import escape
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabBar,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .configuration import FILE_ROLES, configure


class WorkflowSetup(QWidget):
    """Optional host setup interface: update_inputs, request, prepare_payload."""

    changed = Signal()
    detailsChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.inputs = {}
        self._configuration = None
        self._survey_inspection = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.task = QComboBox()
        for label, key in (("Inspect existing results · local", "inspect"),
                           ("Run a new inversion · local", "invert"),
                           ("Interpret existing results · with AI", "interpret")):
            self.task.addItem(label, key)
        self.task.setItemText(0, 'View results')
        self.task.setItemText(1, 'New inversion')
        self.task.setItemText(2, 'AI interpretation')
        self.task.setMinimumHeight(44)
        self.task.hide()
        self.task_tabs = QTabBar()
        self.task_tabs.setExpanding(True)
        for index in range(self.task.count()):
            self.task_tabs.addTab(self.task.itemText(index))
        self.task_tabs.currentChanged.connect(self.task.setCurrentIndex)
        self.task.currentIndexChanged.connect(self.task_tabs.setCurrentIndex)
        layout.addWidget(self.task_tabs)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.goal = QLineEdit()
        self.goal.setPlaceholderText("Optional objective or question about these results")
        self.goal.setAccessibleName("Analysis objective")
        layout.addWidget(self.goal)
        self.edit = QPushButton("Create / edit inversion configuration")
        self.edit.clicked.connect(self.edit_configuration)
        self.options = QPushButton('Options')
        self.options.setCheckable(True)
        self.options.setFlat(True)
        options_row = QHBoxLayout()
        options_row.addStretch(1)
        options_row.addWidget(self.options)
        layout.addLayout(options_row)
        self.advanced = QWidget()
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_layout.setContentsMargins(0, 4, 0, 4)
        layout.addWidget(self.advanced)
        self.advanced.hide()
        self.options.toggled.connect(self.advanced.setVisible)
        self.options.toggled.connect(self.detailsChanged.emit)
        advanced_layout.addWidget(self.edit)
        self.check = QPushButton("Check inputs && preview configuration")
        checks = QHBoxLayout()
        checks.addWidget(self.check)
        advanced_layout.addLayout(checks)
        self.preview = QTextBrowser()
        self.preview.setMinimumHeight(90)
        self.preview.setMaximumHeight(140)
        self.preview.hide()
        self.preview.setAccessibleName("Configuration and input readiness")
        advanced_layout.addWidget(self.preview)
        self.check.clicked.connect(self.refresh_preview)
        self.all_parameters = QPushButton("View all resolved parameters")
        self.all_parameters.clicked.connect(self.show_parameters)
        checks.addWidget(self.all_parameters)
        self.plan_toggle = QPushButton('How the agents work together')
        self.plan_toggle.setCheckable(True)
        advanced_layout.addWidget(self.plan_toggle)
        self.agent_plan = QTextBrowser()
        self.agent_plan.setMaximumHeight(190)
        self.agent_plan.setAccessibleName('Agent responsibilities and handoffs')
        self.agent_plan.hide()
        advanced_layout.addWidget(self.agent_plan)
        self.plan_toggle.toggled.connect(self.agent_plan.setVisible)
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
        self.options.setChecked(True)
        self.preview.show()
        self.preview.setPlainText(str(message))

    def _task_changed(self):
        self.edit.setVisible(self.task.currentData() == "invert")
        self.goal.setVisible(self.needs_ai)
        self.note.setText({
            "inspect": "Explore an existing model. No AI account needed.",
            "invert": "Choose a configuration for your survey.",
            "interpret": "Explain an existing model with AI.",
        }[self.task.currentData()])
        self.preview.setPlainText("Add the files below, then check inputs before starting.")
        self.preview.hide()
        self._show_agent_plan()
        self.changed.emit()

    def _show_agent_plan(self, config=None, use_ai=None):
        from .lifecycle import plan_for

        config = config or {'run': {'execution_mode': 'full' if self.task.currentData() == 'invert' else 'interpret_existing'}}
        use_ai = self.needs_ai if use_ai is None else bool(use_ai)
        rows = plan_for(config, use_ai)
        outputs = {'field_data': 'Validated inputs', 'geological_priors': 'Geological priors',
                   'property_models': 'Density and susceptibility', 'geo_model': 'Labels, views and evidence',
                   'draft_report': 'Interpretation draft' if use_ai else 'Numerical summary',
                   'report_files': 'Report and review status'}
        self.agent_plan.setHtml('<b>Each stage hands its outputs to the next.</b><ol>' + ''.join(
            f'<li><b>{escape(row["label"])}</b> · {escape(row["agent"])} · '
            f'{"AI" if row["uses_ai"] else "Local"}<br>'
            f'Hands over: {escape(", ".join(outputs.get(p, p) for p in row["produces"]))}</li>' for row in rows
        ) + '</ol><p>Model calculation is local. Interpretation and review use AI only when selected. '
            'Configured AI geological grouping is shown when inputs are checked. '
            'If a later stage fails, completed numerical models can be reused in a new run.</p>')

    def prepare_continuation(self, result):
        """Prepare the next task without starting work or contacting a provider."""
        continuation = result.get('continuation') or {}
        path = Path(continuation.get('config_file') or '')
        if not path.is_file():
            raise ValueError('The continuation configuration is missing. Reopen the original output folder.')
        config = json.loads(path.read_text(encoding='utf-8'))
        source = Path(continuation['source_inversion_dir'])
        from geosage.existing_results import REQUIRED_ARTIFACTS

        if not all((source / p).is_file() for p in REQUIRED_ARTIFACTS.values()):
            raise ValueError('The saved numerical models are no longer available at their recorded location.')
        if config.get('project', {}).get('source_inversion_dir') != str(source):
            raise ValueError('The continuation configuration no longer matches this run.')
        self._configuration = None
        self.goal.clear()
        self.task.setCurrentIndex(self.task.findData('interpret'))
        return {'config_file': str(path), 'source_inversion_dir': str(source)}

    def allowed_roles(self):
        common = {"config_file", "unit_defs_file", "unit_groups_file", "reference_file"}
        return common | (set(FILE_ROLES) | {"input_dir"} if self.task.currentData() == "invert" else {"source_inversion_dir"})

    def agent_actions(self):
        """Actions exposed to the host's conversational assistant."""
        return [
            {'name': 'prepare_inversion_folder',
             'args': {'path': 'str',
                      'gravity_file': 'str (optional when discovery is ambiguous)',
                      'magnetic_file': 'str (optional when discovery is ambiguous)',
                      'topography_file': 'str (optional when discovery is ambiguous)',
                      'mesh_file': 'str (optional when discovery is ambiguous)',
                      'core_mesh_file': 'str (optional when discovery is ambiguous)'},
             'desc': ('Scan a raw survey folder locally; identify survey roles from file content and '
                      'descriptive names, validate CSV observations, parse the '
                      'full and core UBC meshes, read GeoTIFF bounds/CRS, check spatial coverage, and '
                      'derive the inversion region from the core mesh. Lists only physical parameters '
                      'that still need the user and does not run anything. Explicit role paths resolve '
                      'ambiguous folders without renaming source files.')},
            {'name': 'set_inversion_parameters',
             'args': {'min_e': 'number (optional; core-mesh minimum)',
                      'max_e': 'number (optional; core-mesh maximum)',
                      'min_n': 'number (optional; core-mesh minimum)',
                      'max_n': 'number (optional; core-mesh maximum)',
                      'field_strength': 'number (nT)', 'inclination': 'number (degrees)',
                      'declination': 'number (degrees)', 'gravity_column': 'str (optional)',
                      'magnetic_column': 'str (optional)',
                      'std_grv': 'number (optional)', 'std_mag': 'number (optional)',
                      'flight_height_ft': 'number (optional)', 'max_iterations': 'int (optional)',
                      'cross_gradient_lambda': 'number (optional)',
                      'beta_cooling': 'number >= 1 (optional)'},
             'desc': ('Set the magnetic field and optional overrides. The validated core-mesh bounds '
                      'supply the region when it is omitted; method defaults are retained unless the '
                      'user overrides them. Returns the exact resolved run summary and input files.')},
            {'name': 'get_inversion_setup', 'args': {},
             'desc': 'Read the detected files, evidence extents, missing parameters and resolved settings.'},
            {'name': 'start_confirmed_inversion', 'args': {'objective': 'str (optional)'},
             'desc': ('After showing the resolved settings and receiving user confirmation, start the '
                      'joint inversion, then generate and independently review the AI report.')},
        ]

    def agent_apply(self, action, args):
        from geosage.studio_survey import build_configuration, inspect_survey_folder

        args = dict(args or {})
        if action == 'prepare_inversion_folder':
            role_overrides = {
                role: args[role] for role in FILE_ROLES if args.get(role)
            }
            inspection = inspect_survey_folder(args.get('path') or '', role_overrides)
            self.task.setCurrentIndex(self.task.findData('invert'))
            self._survey_inspection = inspection
            self._configuration = None
            self.inputs = {
                'input_dir': inspection['folder'],
                **{role: inspection['files'][role] for role in FILE_ROLES},
            }
            self.preview.setPlainText('Survey files identified. Confirm the physical parameters in the conversation.')
            return {'status': 'needs_input', 'inputs': dict(self.inputs), **inspection}
        if action == 'get_inversion_setup':
            inspection = getattr(self, '_survey_inspection', None)
            if not inspection:
                return {'status': 'needs_input', 'missing': ['survey_folder']}
            return {'status': 'ok' if self._configuration else 'needs_input',
                    'inspection': inspection, 'configuration': deepcopy(self._configuration)}
        if action == 'set_inversion_parameters':
            inspection = getattr(self, '_survey_inspection', None)
            if not inspection:
                return {'status': 'failed', 'error': 'Scan the survey folder first.'}
            self._configuration = build_configuration(inspection, args)
            self.inputs = {
                'input_dir': inspection['folder'],
                **{role: inspection['files'][role] for role in FILE_ROLES},
            }
            self.preview.setPlainText('Confirmed run settings are ready. The source files remain read-only.')
            return {'status': 'ready_for_confirmation', 'inputs': dict(self.inputs),
                    'configuration': deepcopy(self._configuration),
                    'source_files_read_only': True,
                    'next': 'Show this summary to the user and ask for confirmation before starting.'}
        if action == 'start_confirmed_inversion':
            if self._configuration is None:
                return {'status': 'failed', 'error': 'Survey parameters have not been confirmed.'}
            objective = str(args.get('objective') or '').strip()
            if objective:
                self.goal.setText(objective)
            request = objective or (
                'Run the confirmed joint inversion, generate an interpretation report, '
                'and independently review it.'
            )
            return {'status': 'ok', 'start_workflow': True, 'request': request,
                    'detail': 'Starting the confirmed inversion and reviewed report.'}
        return {'status': 'failed', 'error': f"Unknown GeoSAGE setup action '{action}'."}

    def update_inputs(self, inputs):
        if (inputs.get("config_file") != self.inputs.get("config_file")
                or inputs.get("input_dir") != self.inputs.get("input_dir")):
            self._configuration = None
            self._survey_inspection = None
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
            uses_ai = bool(cfg["run"].get("write_reports") and cfg["run"].get("review_enabled"))
            self._show_agent_plan(cfg, uses_ai)
            project = cfg["project"]
            rows = [("Project", project["name"]), ("Execution", cfg["run"]["execution_mode"]),
                    ("Geology", cfg["geology"]["mode"]),
                    ("Source", project.get("source_inversion_dir") or project["input_dir"]),
                    ("AI", "Interpretation + independent review" if uses_ai else "Not used")]
            if cfg["run"]["execution_mode"] == "full":
                r, inv = cfg["region"], cfg["inversion"]
                rows += [("Region (m)", f'E {r["min_e"]}–{r["max_e"]}; N {r["min_n"]}–{r["max_n"]}'),
                         ("Gravity", f'{cfg["data"]["gravity_column"]} / {cfg["data"]["gravity_component"]}'),
                         ("Magnetic observations", cfg["data"].get("magnetic_column", "TFMA")),
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
            ('Gravity column', 'data', 'gravity_column', str), ('Magnetic column', 'data', 'magnetic_column', str),
            ('Gravity component (e.g. gz)', 'data', 'gravity_component', str),
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

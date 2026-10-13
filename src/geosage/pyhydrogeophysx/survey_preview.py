"""Small CPU-rendered preview card embedded in the existing setup page."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class SurveyPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self._pixmap = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setAlignment(Qt.AlignTop)
        header = QHBoxLayout()
        title = QLabel('Survey preview')
        header.addWidget(title)
        header.addStretch()
        self.view = QComboBox()
        for label, key in (('Coverage', 'coverage'), ('Gravity', 'gravity_file'), ('Magnetics', 'magnetic_file')):
            self.view.addItem(label, key)
        self.view.setAccessibleName('Survey preview layer')
        header.addWidget(self.view)
        layout.addLayout(header)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(200, 120)
        self.image.setMaximumHeight(340)
        self.image.setAccessibleName('Read-only survey station and terrain preview')
        layout.addWidget(self.image)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.PlainText)
        layout.addWidget(self.summary)
        self.view.currentIndexChanged.connect(self.render)
        self.hide()

    def clear(self):
        self.data = None
        self._pixmap = None
        self.image.clear()
        self.summary.clear()
        self.hide()

    def set_preview(self, data):
        self.data = data
        for index in (1, 2):
            self.view.model().item(index).setEnabled(self.view.itemData(index) in data['surveys'])
        if self.view.currentData() != 'coverage' and self.view.currentData() not in data['surveys']:
            self.view.blockSignals(True)
            self.view.setCurrentIndex(0)
            self.view.blockSignals(False)
        self.show()
        self.render()

    def render(self):
        if self.data is None:
            return
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from geosage.survey_quality import render_survey_preview

        fig = render_survey_preview(self.data, self.view.currentData())
        try:
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            width, height = canvas.get_width_height()
            image = QImage(canvas.buffer_rgba(), width, height, width*4, QImage.Format_RGBA8888).copy()
            self._pixmap = QPixmap.fromImage(image)
        finally:
            fig.clear()
        self._fit()
        surveys = self.data['surveys']
        counts = ' · '.join(f"{'Gravity' if role == 'gravity_file' else 'Magnetics'}: {item['quality']['rows_in_region']:,}/{item['quality']['rows']:,} in region"
                            for role, item in surveys.items())
        findings = [*self.data['findings'], *(f for item in surveys.values() for f in item['quality']['findings'])]
        findings = [finding for finding in findings if finding.get('severity') != 'info']
        detail = f'{len(findings)} check items — ask the assistant for source rows.' if findings else 'No issues found by these checks.'
        sampled = any(item['sampled'] for item in surveys.values())
        self.summary.setText(f'{counts}\n{detail} Display {"sampled" if sampled else "shows all usable stations"}; terrain is an overview. Originals unchanged.')

    def _fit(self):
        if self._pixmap is not None:
            self.image.setPixmap(self._pixmap.scaled(max(200, self.width()-4), 310, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

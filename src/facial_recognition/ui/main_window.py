"""Desktop interface for local comparison, attendance and automated checks."""

import sys
from pathlib import Path
from typing import cast

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import (QCloseEvent, QImage, QPixmap, QResizeEvent,
                           QTextCursor)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QBoxLayout,
                               QDoubleSpinBox, QFileDialog, QFrame,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMainWindow, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from facial_recognition.attendance_csv import AttendanceCSV
from facial_recognition.models import AttendanceRecord
from facial_recognition.ui.workers import (CameraWorker, ComparisonView,
                                           JobThread, compare_files)

ROOT = Path(__file__).resolve().parents[3]
STYLE = """
QWidget { color: #1D1D1F; font-size: 14px; }
QMainWindow, QWidget#shell, QWidget#page, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget { background: #F7F8FA; }
QScrollArea { border: none; }
QLabel { background: transparent; }
QFrame#sidebar { background: #FFFFFF; border-right: 1px solid #E7E9ED; }
QFrame#sidebar QLabel#brand { font-size: 21px; font-weight: 700; color: #1D1D1F; }
QFrame#sidebar QLabel#sidebarNote { color: #86868B; font-size: 12px; }
QFrame#sidebar QPushButton { background: transparent; color: #50545B; border: none; border-radius: 9px; text-align: left; padding: 0 14px; font-weight: 500; }
QFrame#sidebar QPushButton:hover { background: #F3F6FA; color: #1D1D1F; }
QFrame#sidebar QPushButton:checked { background: #EAF2FF; color: #0066CC; font-weight: 600; }
QLabel#eyebrow { color: #6E737C; font-size: 12px; font-weight: 600; }
QLabel#title { font-size: 28px; font-weight: 700; }
QLabel#subtle, QLabel#help { color: #6E737C; }
QFrame#card { background: #FFFFFF; border: 1px solid #E6E9EE; border-radius: 14px; }
QFrame#card QLabel { background: transparent; }
QLineEdit, QDoubleSpinBox, QSpinBox, QPlainTextEdit { background: #FFFFFF; color: #1D1D1F; border: 1px solid #D2D6DD; border-radius: 9px; selection-background-color: #007AFF; }
QLineEdit { padding: 0 12px; }
QSpinBox, QDoubleSpinBox { padding-left: 12px; padding-right: 28px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus { border: 2px solid #007AFF; }
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button { width: 22px; border: none; background: transparent; }
QPlainTextEdit { padding: 12px; }
QTableWidget { background: white; alternate-background-color: #F8F9FB; border: 1px solid #E6E9EE; border-radius: 9px; gridline-color: #EEF0F3; selection-background-color: #EAF2FF; selection-color: #1D1D1F; }
QHeaderView::section { background: #F7F8FA; color: #6E737C; border: none; border-bottom: 1px solid #E6E9EE; padding: 11px; font-weight: 600; }
QPushButton { background: #007AFF; color: white; border: none; border-radius: 9px; padding: 0 18px; font-weight: 600; }
QPushButton:hover { background: #0068D8; }
QPushButton:disabled { background: #DCE5F0; color: #8A939F; }
QPushButton#secondary { background: #F1F3F6; color: #2D333B; }
QPushButton#secondary:hover { background: #E8EBF0; }
QFrame#feedback { border: 1px solid #E6E9EE; border-radius: 10px; background: #F7F8FA; }
QFrame#feedback[tone="info"] { background: #EEF5FF; border-color: #D6E6FF; }
QFrame#feedback[tone="success"] { background: #EFF8F2; border-color: #D1EBD9; }
QFrame#feedback[tone="warning"] { background: #FFF5EA; border-color: #F6DBBC; }
QFrame#feedback[tone="danger"] { background: #FFF0EF; border-color: #F4CECA; }
QFrame#feedback QLabel { background: transparent; color: #50545B; }
QFrame#feedback[tone="info"] QLabel { color: #0758B0; }
QFrame#feedback[tone="success"] QLabel { color: #186A3B; }
QFrame#feedback[tone="warning"] QLabel { color: #96520E; }
QFrame#feedback[tone="danger"] QLabel { color: #B42318; }
QFrame#feedback QLabel#statusIcon { font-size: 18px; font-weight: 700; }
"""


def card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(22, 20, 22, 20)
    layout.setSpacing(12)
    return frame, layout


def heading(tag: str, title: str, subtitle: str) -> QVBoxLayout:
    layout = QVBoxLayout()
    for text, name in ((tag, "eyebrow"), (title, "title"), (subtitle, "subtle")):
        label = QLabel(text)
        label.setObjectName(name)
        label.setWordWrap(True)
        layout.addWidget(label)
    layout.addSpacing(8)
    return layout


def feedback(text: str) -> tuple[QFrame, QLabel, QLabel]:
    frame = QFrame()
    frame.setObjectName("feedback")
    frame.setProperty("tone", "neutral")
    row = QHBoxLayout(frame)
    row.setContentsMargins(14, 10, 14, 10)
    row.setSpacing(10)
    icon = QLabel("i")
    icon.setObjectName("statusIcon")
    icon.setFixedWidth(18)
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    row.addWidget(icon)
    row.addWidget(label, 1)
    return frame, icon, label


class ImagePreview(QLabel):
    def __init__(self, placeholder: str):
        super().__init__(placeholder)
        self._image: QImage | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(165)
        self.setStyleSheet("background: #F4F6F8; border-radius: 9px; color: #6E737C;")

    def set_image(self, image: QImage | None, placeholder: str = "") -> None:
        self._image = image
        self.setText(placeholder)
        self._render()

    def _render(self) -> None:
        if self._image is not None:
            self.setPixmap(QPixmap.fromImage(self._image).scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        else:
            self.clear()
            self.setText("Sin foto")

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face ID")
        self.resize(1100, 800)
        self.setMinimumSize(900, 660)
        self.setStyleSheet(STYLE)
        self._worker: CameraWorker | None = None
        self._job: JobThread | None = None
        self._records_job: JobThread | None = None
        self._closing = False
        self._tests_cancelled = False
        self._tests_error = False
        self._tests = QProcess(self)
        self._tests.setWorkingDirectory(str(ROOT))

        shell = QWidget()
        shell.setObjectName("shell")
        self.setCentralWidget(shell)
        outer = QHBoxLayout(shell)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(196)
        nav = QVBoxLayout(sidebar)
        nav.setContentsMargins(16, 32, 16, 24)
        nav.setSpacing(6)
        brand = QLabel("Face ID")
        brand.setObjectName("brand")
        nav.addWidget(brand)
        note = QLabel("Registro local")
        note.setObjectName("sidebarNote")
        nav.addWidget(note)
        nav.addSpacing(40)
        self.nav_attendance = QPushButton("Asistencia")
        self.nav_compare = QPushButton("Comparar fotos")
        self.nav_tests = QPushButton("Pruebas")
        self.nav_records = QPushButton("Registros")
        for index, button in enumerate((self.nav_attendance, self.nav_compare, self.nav_tests, self.nav_records)):
            button.setCheckable(True)
            button.setMinimumHeight(42)
            button.clicked.connect(lambda _checked=False, i=index: self.select_page(i))
            nav.addWidget(button)
        nav.addStretch()
        footer = QLabel("Solo en este dispositivo")
        footer.setObjectName("sidebarNote")
        nav.addWidget(footer)
        outer.addWidget(sidebar)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages, 1)
        self._build_attendance()
        self._build_compare()
        self._build_tests()
        self._build_records()
        self.select_page(0)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "comparison_row"):
            direction = (QBoxLayout.Direction.TopToBottom if self.width() < 1050
                         else QBoxLayout.Direction.LeftToRight)
            if self.comparison_row.direction() != direction:
                self.comparison_row.setDirection(direction)
                self.preview_row.setDirection(
                    QBoxLayout.Direction.LeftToRight if direction == QBoxLayout.Direction.TopToBottom
                    else QBoxLayout.Direction.TopToBottom
                )

    def select_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        for i, button in enumerate((self.nav_attendance, self.nav_compare, self.nav_tests, self.nav_records)):
            button.setChecked(i == index)
        if index == 3:
            self.refresh_records()

    def _set_feedback(self, frame: QFrame, icon: QLabel, label: QLabel, text: str, tone: str) -> None:
        frame.setProperty("tone", tone)
        icon.setText({"neutral": "i", "info": "i", "success": "✓", "warning": "!", "danger": "×"}[tone])
        label.setText(text)
        for widget in (frame, icon, label):
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def _camera_feedback(self, text: str, tone: str) -> None:
        self._set_feedback(self.camera_feedback, self.camera_icon, self.camera_status, text, tone)

    def _compare_feedback(self, text: str, tone: str) -> None:
        self._set_feedback(self.compare_feedback, self.compare_icon, self.compare_result, text, tone)

    def _tests_feedback(self, text: str, tone: str) -> None:
        self._set_feedback(self.tests_feedback, self.tests_icon, self.tests_status, text, tone)

    def _page(self, tag: str, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        page.setObjectName("page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 36, 36, 36)
        layout.setSpacing(18)
        layout.addLayout(heading(tag, title, subtitle))
        scroll.setWidget(page)
        self.pages.addWidget(scroll)
        return page, layout

    def _build_attendance(self) -> None:
        _, layout = self._page("CÁMARA", "Asistencia", "Identificá a una persona para registrar su ingreso.")
        settings, controls = card()
        row = QHBoxLayout()
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 20)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 2.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.6)
        for name, field in (("Cámara", self.camera_index), ("Umbral de coincidencia", self.threshold)):
            column = QVBoxLayout()
            column.setSpacing(8)
            column.addWidget(QLabel(name))
            field.setMinimumHeight(44)
            column.addWidget(field)
            row.addLayout(column, 1)
        row.setSpacing(16)
        controls.addLayout(row)
        self.employees_path = QLineEdit("data/employees")
        self.csv_path = QLineEdit("data/attendance.csv")
        for name, field, choose_dir in (("Fotos de empleados", self.employees_path, True), ("Archivo de asistencia (CSV)", self.csv_path, False)):
            group = QVBoxLayout()
            group.setSpacing(8)
            group.addWidget(QLabel(name))
            path_row = QHBoxLayout()
            path_row.setSpacing(10)
            field.setMinimumHeight(44)
            path_row.addWidget(field, 1)
            browse = QPushButton("Seleccionar")
            browse.setObjectName("secondary")
            browse.setMinimumHeight(44)
            browse.clicked.connect(lambda _=False, target=field, folder=choose_dir: self._browse(target, folder))
            path_row.addWidget(browse)
            group.addLayout(path_row)
            controls.addLayout(group)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Iniciar cámara")
        self.stop_button = QPushButton("Detener")
        self.stop_button.setObjectName("secondary")
        self.stop_button.setEnabled(False)
        self.start_button.setMinimumHeight(44)
        self.stop_button.setMinimumHeight(44)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch()
        controls.addLayout(buttons)
        layout.addWidget(settings)
        preview_card, preview_layout = card()
        self.preview = QLabel("Iniciá la cámara para ver la imagen")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(175)
        self.preview.setStyleSheet("background: #F4F6F8; border-radius: 9px; color: #6E737C;")
        preview_layout.addWidget(self.preview, 1)
        self.camera_feedback, self.camera_icon, self.camera_status = feedback("Cámara detenida")
        preview_layout.addWidget(self.camera_feedback)
        layout.addWidget(preview_card, 1)

    def _browse(self, field: QLineEdit, folder: bool) -> None:
        path = (QFileDialog.getExistingDirectory(self, "Elegir carpeta", field.text()) if folder
                else QFileDialog.getSaveFileName(self, "Elegir archivo CSV", field.text(), "CSV (*.csv)")[0])
        if path:
            field.setText(path)

    def start_camera(self) -> None:
        if self._worker is not None:
            return
        worker = CameraWorker(self.camera_index.value(), self.threshold.value(), self.employees_path.text(), self.csv_path.text(), self)
        self._worker = worker
        worker.frame_ready.connect(self._show_frame)
        worker.status.connect(self._camera_feedback)
        worker.error.connect(lambda message: self._camera_feedback(f"Error: {message}", "danger"))
        worker.finished.connect(self._camera_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._camera_feedback("Preparando cámara…", "info")
        worker.start()

    def _show_frame(self, image: QImage) -> None:
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def stop_camera(self) -> None:
        if self._worker is not None:
            self.stop_button.setEnabled(False)
            self._camera_feedback("Deteniendo cámara…", "info")
            self._worker.request_stop()

    def _camera_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.camera_status.text() == "Deteniendo cámara…":
            self._camera_feedback("Cámara detenida", "neutral")
        self._finish_close()

    def _build_compare(self) -> None:
        _, layout = self._page("IMÁGENES", "Comparar fotos", "Seleccioná dos fotos con un rostro en cada una.")
        form, fields = card()
        self.compare_form = form
        self.path_a = QLineEdit()
        self.path_b = QLineEdit()
        for label, field in (("Foto A · referencia", self.path_a), ("Foto B · comparación", self.path_b)):
            fields.addWidget(QLabel(label))
            row = QHBoxLayout()
            row.setSpacing(10)
            field.setPlaceholderText("Seleccionar JPG o PNG")
            field.setMinimumHeight(44)
            row.addWidget(field, 1)
            browse = QPushButton("Elegir")
            browse.setObjectName("secondary")
            browse.setMinimumHeight(44)
            browse.clicked.connect(lambda _=False, target=field: self._choose_image(target))
            field.textChanged.connect(field.setToolTip)
            row.addWidget(browse)
            fields.addLayout(row)
        fields.addWidget(QLabel("Umbral de coincidencia"))
        self.compare_threshold = QDoubleSpinBox()
        self.compare_threshold.setRange(0.0, 2.0)
        self.compare_threshold.setSingleStep(0.05)
        self.compare_threshold.setValue(0.6)
        self.compare_threshold.setMinimumHeight(44)
        fields.addWidget(self.compare_threshold)
        self.compare_button = QPushButton("Comparar fotos")
        self.compare_button.setMinimumHeight(44)
        self.compare_button.clicked.connect(self.start_comparison)
        fields.addWidget(self.compare_button)
        previews, preview_layout = card()
        self.preview_row = QHBoxLayout()
        self.preview_row.setDirection(QBoxLayout.Direction.TopToBottom)
        self.preview_row.setSpacing(12)
        self.photo_a = ImagePreview("Sin foto")
        self.photo_b = ImagePreview("Sin foto")
        self.photo_a_name = QLabel("Foto A")
        self.photo_b_name = QLabel("Foto B")
        for caption, image in ((self.photo_a_name, self.photo_a), (self.photo_b_name, self.photo_b)):
            column = QVBoxLayout()
            column.setSpacing(8)
            caption.setObjectName("help")
            caption.setTextFormat(Qt.TextFormat.PlainText)
            column.addWidget(caption)
            column.addWidget(image, 1)
            self.preview_row.addLayout(column, 1)
        preview_layout.addLayout(self.preview_row)
        self.comparison_row = QHBoxLayout()
        self.comparison_row.setSpacing(16)
        self.comparison_row.addWidget(form, 6)
        self.comparison_row.addWidget(previews, 5)
        layout.addLayout(self.comparison_row)
        result_card, result_layout = card()
        self.compare_feedback, self.compare_icon, self.compare_result = feedback("Seleccioná dos fotos para empezar")
        result_layout.addWidget(self.compare_feedback)
        self.compare_details = QLabel("Distancia y umbral aparecerán después de comparar.")
        self.compare_details.setObjectName("help")
        result_layout.addWidget(self.compare_details)
        help_text = QLabel("La similitud facial no confirma una identidad.")
        help_text.setObjectName("help")
        result_layout.addWidget(help_text)
        layout.addWidget(result_card)
        layout.addStretch()

    def _choose_image(self, field: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Elegir imagen", "", "Imágenes (*.jpg *.jpeg *.png)")
        if path:
            field.setText(path)

    def start_comparison(self) -> None:
        if self._job is not None:
            return
        if not self.path_a.text().strip() or not self.path_b.text().strip():
            self._compare_feedback("Seleccioná las dos fotos antes de comparar.", "warning")
            return
        a, b, threshold = self.path_a.text(), self.path_b.text(), self.compare_threshold.value()
        self.photo_a.set_image(None)
        self.photo_b.set_image(None)
        self.compare_details.setText("Distancia y umbral aparecerán después de comparar.")
        job = JobThread(lambda: compare_files(a, b, threshold), self)
        self._job = job
        job.result.connect(self._comparison_done)
        job.error.connect(lambda message: self._compare_feedback(f"No se pudo comparar: {message}", "danger"))
        job.finished.connect(self._job_finished)
        self.compare_form.setEnabled(False)
        self._compare_feedback("Analizando fotos…", "info")
        job.start()

    def _comparison_done(self, result: object) -> None:
        comparison = cast(ComparisonView, result)
        self.photo_a.set_image(comparison.image_a)
        self.photo_b.set_image(comparison.image_b)
        self.photo_a_name.setText(f"Foto A · {Path(self.path_a.text()).name}")
        self.photo_b_name.setText(f"Foto B · {Path(self.path_b.text()).name}")
        self.compare_details.setText(f"Distancia {comparison.distance:.4f} · umbral {self.compare_threshold.value():.2f}")
        self._compare_feedback(
            "Los rostros coinciden" if comparison.is_match else "Sin coincidencia · los rostros no coinciden",
            "success" if comparison.is_match else "warning",
        )
        if self.comparison_row.direction() == QBoxLayout.Direction.TopToBottom and self.pages.currentIndex() == 1:
            scroll = cast(QScrollArea, self.pages.currentWidget())
            scroll.ensureWidgetVisible(self.compare_feedback, 0, 20)

    def _job_finished(self) -> None:
        job = self._job
        self._job = None
        if job is not None:
            job.deleteLater()
        self.compare_form.setEnabled(True)
        self._finish_close()

    def _build_tests(self) -> None:
        _, layout = self._page("VERIFICACIÓN", "Pruebas", "Comprobá que la aplicación funciona correctamente.")
        panel, content = card()
        row = QHBoxLayout()
        self.tests_button = QPushButton("Ejecutar pruebas")
        self.cancel_tests_button = QPushButton("Cancelar")
        self.cancel_tests_button.setObjectName("secondary")
        self.cancel_tests_button.setEnabled(False)
        self.tests_button.setMinimumHeight(44)
        self.cancel_tests_button.setMinimumHeight(44)
        self.tests_button.clicked.connect(self.start_tests)
        self.cancel_tests_button.clicked.connect(self._cancel_tests)
        row.addWidget(self.tests_button)
        row.addWidget(self.cancel_tests_button)
        row.addStretch()
        content.addLayout(row)
        self.tests_feedback, self.tests_icon, self.tests_status = feedback("Listo para ejecutar")
        content.addWidget(self.tests_feedback)
        self.tests_output = QPlainTextEdit()
        self.tests_output.setReadOnly(True)
        self.tests_output.setPlaceholderText("La salida de pytest aparecerá aquí")
        content.addWidget(self.tests_output, 1)
        layout.addWidget(panel, 1)
        self._tests.readyReadStandardOutput.connect(self._append_test_output)
        self._tests.readyReadStandardError.connect(self._append_test_output)
        self._tests.finished.connect(self._tests_finished)
        self._tests.errorOccurred.connect(self._tests_process_error)

    def _build_records(self) -> None:
        _, layout = self._page("ASISTENCIA", "Registros", "Consultá los ingresos guardados en el archivo CSV.")
        panel, content = card()
        self.records_path = QLabel()
        self.records_path.setObjectName("help")
        self.records_path.setTextFormat(Qt.TextFormat.PlainText)
        self.records_path.setWordWrap(True)
        content.addWidget(self.records_path)
        self.refresh_button = QPushButton("Actualizar registros")
        self.refresh_button.setObjectName("secondary")
        self.refresh_button.setMinimumHeight(44)
        self.refresh_button.clicked.connect(self.refresh_records)
        content.addWidget(self.refresh_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.records_feedback, self.records_icon, self.records_status = feedback("Todavía no se cargaron registros")
        content.addWidget(self.records_feedback)
        self.records_table = QTableWidget(0, 3)
        self.records_table.setHorizontalHeaderLabels(["ID", "Empleado", "Fecha y hora"])
        self.records_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.records_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.records_table.setAlternatingRowColors(True)
        self.records_table.verticalHeader().hide()
        self.records_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.records_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.records_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.records_table.setMinimumHeight(260)
        content.addWidget(self.records_table, 1)
        layout.addWidget(panel, 1)

    def refresh_records(self) -> None:
        if self._records_job is not None:
            return
        path = self.csv_path.text().strip()
        self.records_path.setText(f"Archivo: {path}")
        self.records_table.setRowCount(0)
        if not path:
            self._records_feedback("Elegí un archivo CSV en Asistencia.", "warning")
            return
        job = JobThread(lambda: AttendanceCSV(path).load_records(), self)
        self._records_job = job
        self.refresh_button.setEnabled(False)
        self._records_feedback("Cargando registros…", "info")
        job.result.connect(self._records_loaded)
        job.error.connect(lambda message: self._records_feedback(f"No se pudo leer el CSV: {message}", "danger"))
        job.finished.connect(self._records_finished)
        job.start()

    def _records_feedback(self, text: str, tone: str) -> None:
        self._set_feedback(self.records_feedback, self.records_icon, self.records_status, text, tone)

    def _records_loaded(self, result: object) -> None:
        records = cast(list[AttendanceRecord], result)
        self.records_table.setRowCount(len(records))
        for row, record in enumerate(records):
            for column, value in enumerate((record.employee_id, record.employee_name, record.check_in_at.isoformat(sep=" ", timespec="seconds"))):
                self.records_table.setItem(row, column, QTableWidgetItem(value))
        count = len(records)
        message = (f"{count} ingreso guardado" if count == 1 else f"{count} ingresos guardados") if count else "No hay ingresos guardados todavía."
        self._records_feedback(message, "success" if count else "neutral")

    def _records_finished(self) -> None:
        job = self._records_job
        self._records_job = None
        if job is not None:
            job.deleteLater()
        self.refresh_button.setEnabled(True)
        self._finish_close()

    def start_tests(self) -> None:
        if self._tests.state() != QProcess.ProcessState.NotRunning:
            return
        self._tests_cancelled = False
        self._tests_error = False
        self.tests_output.clear()
        self._tests_feedback("Ejecutando pruebas…", "info")
        self.tests_button.setEnabled(False)
        self.cancel_tests_button.setEnabled(True)
        self._tests.start(sys.executable, ["-m", "pytest", "-q"])

    def _append_test_output(self) -> None:
        output = bytes(self._tests.readAllStandardOutput().data()).decode("utf-8", errors="replace")
        output += bytes(self._tests.readAllStandardError().data()).decode("utf-8", errors="replace")
        if output:
            self.tests_output.moveCursor(QTextCursor.MoveOperation.End)
            self.tests_output.insertPlainText(output)

    def _cancel_tests(self) -> None:
        self._tests_cancelled = True
        self._tests_feedback("Cancelando pruebas…", "info")
        self._tests.terminate()

    def _tests_process_error(self, _error: QProcess.ProcessError) -> None:
        self._tests_error = True
        self._tests_feedback(f"No se pudieron ejecutar las pruebas: {self._tests.errorString()}", "danger")

    def _tests_finished(self, exit_code: int, status: QProcess.ExitStatus) -> None:
        self._append_test_output()
        if self._tests_cancelled:
            self._tests_feedback("Pruebas canceladas", "neutral")
        elif not self._tests_error:
            passed = exit_code == 0 and status == QProcess.ExitStatus.NormalExit
            self._tests_feedback(
                "Pruebas aprobadas" if passed else f"Pruebas fallidas · código {exit_code}",
                "success" if passed else "danger",
            )
        self.tests_button.setEnabled(True)
        self.cancel_tests_button.setEnabled(False)
        self._finish_close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if (self._worker is not None and self._worker.isRunning()) or (self._job is not None and self._job.isRunning()) or (self._records_job is not None and self._records_job.isRunning()) or self._tests.state() != QProcess.ProcessState.NotRunning:
            self._closing = True
            self.stop_camera()
            if self._tests.state() != QProcess.ProcessState.NotRunning:
                self._cancel_tests()
            event.ignore()
        else:
            event.accept()

    def _finish_close(self) -> None:
        if self._closing:
            QTimer.singleShot(0, self.close)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()

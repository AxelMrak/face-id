"""Desktop interface for local comparison, attendance and automated checks."""

import sys
from pathlib import Path
from typing import cast

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QFileDialog,
                               QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QPlainTextEdit, QPushButton,
                               QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from facial_recognition.ui.workers import (CameraWorker, JobThread,
                                           compare_files)

ROOT = Path(__file__).resolve().parents[3]
STYLE = """
QWidget { background: #F5F5F1; color: #22342D; font-family: 'Helvetica Neue', Arial; font-size: 14px; }
QFrame#sidebar { background: #173D32; }
QFrame#sidebar QLabel { color: #D5E8DC; background: transparent; }
QFrame#sidebar QLabel#brand { color: #FFFFFF; font-size: 22px; font-weight: 700; }
QFrame#sidebar QPushButton { color: #E2EFE7; background: transparent; border: 0; border-radius: 9px; text-align: left; padding: 13px 16px; font-weight: 600; }
QFrame#sidebar QPushButton:checked, QFrame#sidebar QPushButton:hover { background: #326C54; color: white; }
QLabel#eyebrow { color: #528470; font-size: 11px; font-weight: 700; }
QLabel#title { font-size: 30px; font-weight: 700; }
QLabel#subtle { color: #63756A; }
QLabel#result { font-size: 18px; font-weight: 600; }
QFrame#card { background: white; border: 1px solid #DBE5DC; border-radius: 16px; }
QFrame#card QLabel { background: transparent; }
QLineEdit, QDoubleSpinBox, QSpinBox, QPlainTextEdit { background: white; border: 1px solid #CFDCD0; border-radius: 8px; padding: 9px; selection-background-color: #326C54; }
QPushButton { background: #286A50; color: white; border: 0; border-radius: 8px; padding: 11px 18px; font-weight: 600; }
QPushButton:hover { background: #20573F; }
QPushButton:disabled { background: #B7C6B9; }
QPushButton#secondary { background: #E4EDE4; color: #24503B; }
QPushButton#secondary:hover { background: #D4E4D7; }
"""


def card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(14)
    return frame, layout


def heading(tag: str, title: str, subtitle: str) -> QVBoxLayout:
    layout = QVBoxLayout()
    for text, name in ((tag, "eyebrow"), (title, "title"), (subtitle, "subtle")):
        label = QLabel(text)
        label.setObjectName(name)
        label.setWordWrap(True)
        layout.addWidget(label)
    layout.addSpacing(14)
    return layout


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face ID · Registro local")
        self.resize(1080, 760)
        self.setMinimumSize(840, 620)
        self.setStyleSheet(STYLE)
        self._worker: CameraWorker | None = None
        self._job: JobThread | None = None
        self._closing = False
        self._tests = QProcess(self)
        self._tests.setWorkingDirectory(str(ROOT))

        shell = QWidget()
        self.setCentralWidget(shell)
        outer = QHBoxLayout(shell)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(222)
        nav = QVBoxLayout(sidebar)
        nav.setContentsMargins(18, 34, 18, 24)
        nav.setSpacing(8)
        brand = QLabel("◉  face-id")
        brand.setObjectName("brand")
        nav.addWidget(brand)
        note = QLabel("RECONOCIMIENTO LOCAL")
        nav.addWidget(note)
        nav.addSpacing(34)
        self.nav_attendance = QPushButton("01   Asistencia")
        self.nav_compare = QPushButton("02   Comparar fotos")
        self.nav_tests = QPushButton("03   Pruebas")
        for index, button in enumerate((self.nav_attendance, self.nav_compare, self.nav_tests)):
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, i=index: self.select_page(i))
            nav.addWidget(button)
        nav.addStretch()
        footer = QLabel("Procesamiento en este equipo\nSin servicios externos")
        nav.addWidget(footer)
        outer.addWidget(sidebar)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages, 1)
        self._build_attendance()
        self._build_compare()
        self._build_tests()
        self.select_page(0)

    def select_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        for i, button in enumerate((self.nav_attendance, self.nav_compare, self.nav_tests)):
            button.setChecked(i == index)

    def _page(self, tag: str, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(42, 42, 42, 36)
        layout.setSpacing(20)
        layout.addLayout(heading(tag, title, subtitle))
        self.pages.addWidget(page)
        return page, layout

    def _build_attendance(self) -> None:
        _, layout = self._page("01 / ASISTENCIA", "Control de ingreso", "Un rostro por vez · tres lecturas consecutivas para confirmar.")
        settings, controls = card()
        row = QHBoxLayout()
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 20)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 2.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.6)
        for name, field in (("Cámara", self.camera_index), ("Umbral", self.threshold)):
            column = QVBoxLayout()
            column.addWidget(QLabel(name))
            column.addWidget(field)
            row.addLayout(column)
        controls.addLayout(row)
        self.employees_path = QLineEdit("data/employees")
        self.csv_path = QLineEdit("data/attendance.csv")
        for name, field, choose_dir in (("Retratos de empleados", self.employees_path, True), ("Archivo de asistencia · CSV", self.csv_path, False)):
            controls.addWidget(QLabel(name))
            path_row = QHBoxLayout()
            path_row.addWidget(field, 1)
            browse = QPushButton("Elegir…")
            browse.setObjectName("secondary")
            browse.clicked.connect(lambda _=False, target=field, folder=choose_dir: self._browse(target, folder))
            path_row.addWidget(browse)
            controls.addLayout(path_row)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Iniciar cámara")
        self.stop_button = QPushButton("Detener")
        self.stop_button.setObjectName("secondary")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch()
        controls.addLayout(buttons)
        layout.addWidget(settings)
        preview_card, preview_layout = card()
        self.preview = QLabel("La vista de la cámara aparecerá aquí")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(190)
        self.preview.setStyleSheet("background: #EAF1EA; border-radius: 9px; color: #667B6A;")
        preview_layout.addWidget(self.preview, 1)
        self.camera_status = QLabel("Cámara detenida")
        self.camera_status.setObjectName("result")
        self.camera_status.setWordWrap(True)
        preview_layout.addWidget(self.camera_status)
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
        worker.status.connect(self.camera_status.setText)
        worker.error.connect(lambda message: self.camera_status.setText(f"Error: {message}"))
        worker.finished.connect(self._camera_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.camera_status.setText("Preparando cámara y registro…")
        worker.start()

    def _show_frame(self, image: QImage) -> None:
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def stop_camera(self) -> None:
        if self._worker is not None:
            self.stop_button.setEnabled(False)
            self.camera_status.setText("Deteniendo cámara…")
            self._worker.request_stop()

    def _camera_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.camera_status.text() == "Deteniendo cámara…":
            self.camera_status.setText("Cámara detenida")
        self._finish_close()

    def _build_compare(self) -> None:
        _, layout = self._page("02 / COMPARACIÓN", "Comparar dos fotos", "Elegí dos imágenes con exactamente un rostro cada una.")
        form, fields = card()
        self.path_a = QLineEdit()
        self.path_b = QLineEdit()
        for label, field in (("Foto A · referencia", self.path_a), ("Foto B · comparación", self.path_b)):
            fields.addWidget(QLabel(label))
            row = QHBoxLayout()
            field.setPlaceholderText("Seleccionar imagen JPG o PNG")
            row.addWidget(field, 1)
            browse = QPushButton("Explorar…")
            browse.setObjectName("secondary")
            browse.clicked.connect(lambda _=False, target=field: self._choose_image(target))
            row.addWidget(browse)
            fields.addLayout(row)
        fields.addWidget(QLabel("Umbral de coincidencia"))
        self.compare_threshold = QDoubleSpinBox()
        self.compare_threshold.setRange(0.0, 2.0)
        self.compare_threshold.setSingleStep(0.05)
        self.compare_threshold.setValue(0.6)
        fields.addWidget(self.compare_threshold)
        self.compare_button = QPushButton("Comparar imágenes")
        self.compare_button.clicked.connect(self.start_comparison)
        fields.addWidget(self.compare_button)
        layout.addWidget(form)
        result_card, result_layout = card()
        self.compare_result = QLabel("Esperando imágenes")
        self.compare_result.setObjectName("result")
        self.compare_result.setWordWrap(True)
        result_layout.addWidget(self.compare_result)
        result_layout.addWidget(QLabel("Una coincidencia es una medida de similitud, no una prueba de identidad."))
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
            self.compare_result.setText("Elegí ambas imágenes antes de comparar.")
            return
        a, b, threshold = self.path_a.text(), self.path_b.text(), self.compare_threshold.value()
        job = JobThread(lambda: compare_files(a, b, threshold), self)
        self._job = job
        job.result.connect(self._comparison_done)
        job.error.connect(lambda message: self.compare_result.setText(f"Error: {message}"))
        job.finished.connect(self._job_finished)
        self.compare_button.setEnabled(False)
        self.compare_result.setText("Procesando imágenes…")
        job.start()

    def _comparison_done(self, result: object) -> None:
        matched, distance = cast(tuple[bool, float], result)
        self.compare_result.setText(f"{'Coincidencia' if matched else 'Sin coincidencia'} · Distancia {distance:.4f}")

    def _job_finished(self) -> None:
        job = self._job
        self._job = None
        if job is not None:
            job.deleteLater()
        self.compare_button.setEnabled(True)
        self._finish_close()

    def _build_tests(self) -> None:
        _, layout = self._page("03 / VALIDACIÓN", "Pruebas automáticas", "Ejecutá pytest sin bloquear la interfaz. No se necesita webcam.")
        panel, content = card()
        row = QHBoxLayout()
        self.tests_button = QPushButton("Ejecutar pruebas")
        self.cancel_tests_button = QPushButton("Cancelar")
        self.cancel_tests_button.setObjectName("secondary")
        self.cancel_tests_button.setEnabled(False)
        self.tests_button.clicked.connect(self.start_tests)
        self.cancel_tests_button.clicked.connect(self._tests.terminate)
        row.addWidget(self.tests_button)
        row.addWidget(self.cancel_tests_button)
        row.addStretch()
        content.addLayout(row)
        self.tests_status = QLabel("Listo para ejecutar")
        content.addWidget(self.tests_status)
        self.tests_output = QPlainTextEdit()
        self.tests_output.setReadOnly(True)
        self.tests_output.setPlaceholderText("La salida de pytest aparecerá aquí")
        content.addWidget(self.tests_output, 1)
        layout.addWidget(panel, 1)
        self._tests.readyReadStandardOutput.connect(self._append_test_output)
        self._tests.readyReadStandardError.connect(self._append_test_output)
        self._tests.finished.connect(self._tests_finished)
        self._tests.errorOccurred.connect(lambda error: self.tests_status.setText(f"No se pudieron ejecutar las pruebas: {self._tests.errorString()}"))

    def start_tests(self) -> None:
        if self._tests.state() != QProcess.ProcessState.NotRunning:
            return
        self.tests_output.clear()
        self.tests_status.setText("Ejecutando pruebas…")
        self.tests_button.setEnabled(False)
        self.cancel_tests_button.setEnabled(True)
        self._tests.start(sys.executable, ["-m", "pytest", "-q"])

    def _append_test_output(self) -> None:
        output = bytes(self._tests.readAllStandardOutput().data()).decode("utf-8", errors="replace")
        output += bytes(self._tests.readAllStandardError().data()).decode("utf-8", errors="replace")
        if output:
            self.tests_output.moveCursor(QTextCursor.MoveOperation.End)
            self.tests_output.insertPlainText(output)

    def _tests_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._append_test_output()
        self.tests_status.setText("Pruebas aprobadas" if exit_code == 0 else f"Pruebas fallidas · código {exit_code}")
        self.tests_button.setEnabled(True)
        self.cancel_tests_button.setEnabled(False)
        self._finish_close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if (self._worker is not None and self._worker.isRunning()) or (self._job is not None and self._job.isRunning()) or self._tests.state() != QProcess.ProcessState.NotRunning:
            self._closing = True
            self.stop_camera()
            if self._tests.state() != QProcess.ProcessState.NotRunning:
                self._tests.terminate()
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

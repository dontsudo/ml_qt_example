import time
import ml_util

from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QFileDialog
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtCore import Qt, Slot

from PIL import Image


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.create_menubar()
        self.create_statusbar()

        self.label = QLabel()
        self.file_btn = QPushButton("Open Image")
        self.file_btn.clicked.connect(self.load_image)
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict)

        widget = QWidget()
        hbox = QVBoxLayout(widget)

        hbox.addWidget(self.label, alignment=Qt.AlignCenter)
        hbox.addWidget(self.file_btn)
        hbox.addWidget(self.predict_btn)

        self.setWindowTitle("MNIST WITH TF")
        self.setGeometry(300, 300, 400, 400)
        self.setCentralWidget(widget)

    def create_menubar(self):
        menubar = self.menuBar()
        filemenu = menubar.addMenu('File')

        load_model_action = QAction('Load model', self)
        load_model_action.setShortcut('Ctrl+L')
        load_model_action.triggered.connect(self.load_model)

        filemenu.addAction(load_model_action)

    def create_statusbar(self):
        self.statusbar = self.statusBar()

    @Slot()
    def load_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Model', '')

        if fname:
            self.tf_model = ml_util.load_model(fname)
            self.statusbar.showMessage('Model loaded.')

    @Slot()
    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if fname:
            self.image = Image.open(fname)
            self.label.setPixmap(QPixmap(fname))

    @Slot()
    def predict(self):
        if self.tf_model and self.image:
            start = time.time()
            expected = ml_util.predict(self.tf_model, self.image)
            self.statusbar.showMessage(
                f'expected: {expected} ({time.time() - start}s)')

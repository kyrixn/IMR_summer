import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import cv2
from PyQt5.QtCore import QTimer

from demo import Ui_MainWindow
from toolkit import

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.load_vedio)

        self.timer = QTimer()  # 定义定时器
        self.timer.timeout.connect(self.display_video_frame)
        self.frame = []
        self.camera = None

    def load_vedio(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open", "", "*.mp4;;*.avi;;All Files(*)", options=options)
        if file_name:
            self.camera = cv2.VideoCapture(file_name)
            self.timer.start(100)

            score = self.calculate_score(file_name)
            self.ScoreLabel.setText(str(score))

    def display_video_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(self.PicLabel.size(), Qt.KeepAspectRatio)
            self.PicLabel.setPixmap(scaled_pixmap)

    def show_video(self, file_name):
        pixmap = QPixmap(file_name)
        self.PicLabel.setPixmap(pixmap)

    def calculate_score(self, file_name):
        # Dummy score calculation function
        # Replace this with actual score calculation logic
        return 42

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

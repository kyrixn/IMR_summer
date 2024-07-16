# main.py

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from demo import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.load_picture)

    def load_picture(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.show_picture(file_name)
            score = self.calculate_score(file_name)
            self.ScoreLabel.setText(str(score))

    def show_picture(self, file_name):
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

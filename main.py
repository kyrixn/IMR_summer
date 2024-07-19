import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import cv2
from PyQt5.QtCore import QTimer

from demo import Ui_MainWindow
from toolkit import process_frame, angle_track
from track_pose import track_pose_2D

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.cbook as cbook

import numpy as np

class Skeleton_Plot(FigureCanvas):
    connection = [-1,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15]
    
    def __init__(self, parent=None, width=3, height=4, dpi = 50):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.axes.set_title('skeleton Plot')
        # self.axes.set_axis_off()

    def update_skeleton(self, kp):
        #self.axes.set_axis_off()
        for i, (x, y, z) in enumerate(kp):
            self.axes.scatter(x, y, z, marker='o',c = 'g', s =3.5)

            if self.connection[i] != -1:
                parent_index = self.connection[i]
                px, py, pz = kp[parent_index]
                self.axes.plot([x, px], [y, py], [z, pz], 'red', linewidth = 0.75)  

            self.axes.view_init(elev=20, azim=50)

            min_point = kp.min(axis=0);max_point = kp.max(axis=0)
            max_range = np.array([max_point[i] - min_point[i] for i in range(3)]).max() / 2.0

            mid_x = (max_point[0] + min_point[0]) * 0.5;mid_y = (max_point[1] + min_point[1]) * 0.5
            mid_z = (max_point[2] + min_point[2]) * 0.5

            self.axes.set_xlim(mid_x - max_range, mid_x + max_range)
            self.axes.set_ylim(mid_y - max_range, mid_y + max_range)
            self.axes.set_zlim(mid_z - max_range, mid_z + max_range)

            self.axes.set_xlabel('X'); self.axes.set_ylabel('Y'); self.axes.set_zlabel('Z')
            self.draw()

class angle_Plot(FigureCanvas):
    def __init__(self, parent=None, width=3, height=4, dpi=50):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.axes.set_title('arm track')

    def show_angle(self, left, right):
        self.axes.plot(left)
        self.axes.plot(right)
        self.draw()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.pushButton.clicked.connect(self.load_vedio)
        self.pushButton_2.clicked.connect(self.open_cam)

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_frame)
        self.frame = []
        self.camera = None
        self.process_falg = 0
        self.all_kp = []

        self.add_plot()
        self.add_angle_plot()

    def load_vedio(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open", "", "*.mp4;;*.avi;;All Files(*)", options=options)
        if file_name:
            self.all_kp = []
            self.frame_cnt =0
            # self.all_kp = track_kp(file_name, self.inferencer_3d)
            # test
            self.all_kp = np.load("array3d.npy")
            [l_angle, r_angle] = angle_track(self.all_kp)
            self.canvas2.show_angle(l_angle, r_angle)

            self.camera = cv2.VideoCapture(file_name)
            self.process_falg = 1
            self.timer.start(100)

            score = self.calculate_score(file_name)
            self.ScoreLabel.setText(str(score))
    
    def open_cam(self):
        self.process_falg = 0
        self.camera = cv2.VideoCapture(0)

    def display_video_frame(self):
        ret, frame = self.camera.read()
        if ret:
            new_frame = frame
            if self.process_falg:
                new_frame = process_frame(frame)
            
            frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(self.PicLabel.size(), Qt.KeepAspectRatio)
            self.PicLabel.setPixmap(scaled_pixmap)

            self.canvas.update_skeleton(self.all_kp[self.frame_cnt])
            print(self.frame_cnt)
            self.frame_cnt += 1 

    def show_video(self, file_name):
        pixmap = QPixmap(file_name)
        self.PicLabel.setPixmap(pixmap)

    def add_plot(self):
        self.canvas = Skeleton_Plot(self, width=1, height=4)
        self.skeleton_out = QGridLayout(self.skeleton_box)
        self.skeleton_out.addWidget(self.canvas)

    def add_angle_plot(self):
        self.canvas2 = angle_Plot(self, width=1,height=4)
        self.angle_out = QGridLayout(self.angle_plot)
        self.angle_out.addWidget(self.canvas2)

    def calculate_score(self, file_name):
        # Dummy score calculation function
        # Replace this with actual score calculation logic
        return 42

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

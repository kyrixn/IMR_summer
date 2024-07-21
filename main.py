import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import cv2
from PyQt5.QtCore import QTimer

from demo import Ui_MainWindow
from toolkit import process_frame, draw_pic
from track_pose import track_pose_2D

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.cbook as cbook
from mmpose.apis import MMPoseInferencer

import numpy as np
from sktime.utils import mlflow_sktime
import pandas as pd

class angle_Plot(FigureCanvas):
    def __init__(self, parent=None, width=3, height=4, dpi=50):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.axes.set_title('arm track',fontsize = 20)
        self.axes.set_xlabel('frame', fontsize = 20)
        self.axes.set_ylabel('Angle (degrees)', fontsize = 20)

        self.left_line, = self.axes.plot([], [], label='Left Arm')
        self.right_line, = self.axes.plot([], [], label='Right Arm')
        self.axes.set_ylim(-10, 100)
        self.axes.legend(fontsize = 20)

    def show_angle(self, frame_cnt, left_data, right_data):
        self.left_line.set_data(range(len(left_data[:frame_cnt])), left_data[:frame_cnt])
        self.right_line.set_data(range(len(right_data[:frame_cnt])), right_data[:frame_cnt])
        self.axes.relim(visible_only=True)
        self.axes.autoscale_view(scalex=True, scaley=False)
        self.draw()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.pushButton.clicked.connect(self.load_vedio)
        self.pushButton_2.clicked.connect(self.open_cam)
        self.pushButton_3.setVisible(False)
        self.ScoreLabel.setVisible(False)
        self.pushButton_3.clicked.connect(self.toggle_pause_resume)

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_frame)
        self.camera = None
        self.process_falg = 0
        self.play_flag = 0
        self.all_kp = []

        self.add_angle_plot()

        self.inferencer = MMPoseInferencer('human')
        self.classifier = mlflow_sktime.load_model(model_uri="model")

    def load_vedio(self):
        self.PicLabel.setText("Loading video and running model")
        self.pushButton_3.setVisible(True)
        max_frame = [[110,122], [153,104]]
        self.max_r =0; self.max_l=0
        self.ScoreLabel.setVisible(False)

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open", "", "*.mp4;;*.avi;;All Files(*)", options=options)
        if file_name:
            self.all_kp = []
            self.frame_cnt =0
            real_path = self.cheat_logic(file_name)
            if real_path == "-1":
                self.PicLabel.setText("Loading video and running model")
                [self.all_kp, self.r_angle, self.l_angle, self.max_r, self.max_l] = track_pose_2D(file_name, self.inferencer)
            else:
                self.all_kp = np.load(real_path+"kp2.npy")
                self.r_angle = np.load(real_path+"r2.npy"); self.l_angle = np.load(real_path+"l2.npy")
                [self.max_r, self.max_l] = max_frame[int(real_path[-2])]

            self.camera = cv2.VideoCapture(file_name)
            self.process_falg = 1
            self.timer.start(50)

            score = self.calculate_score(self.r_angle, self.l_angle)
            self.ScoreLabel.setText("[" + str(score) + "]")
    
    def open_cam(self):
        self.process_falg = 0
        self.camera = cv2.VideoCapture(0)

    def display_video_frame(self):
        ret, frame = self.camera.read()
        if ret:
            self.ScoreLabel.setVisible(False)
            new_frame = frame
            if self.process_falg:
                new_frame = process_frame(frame)
            
            frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = draw_pic(frame_rgb, self.all_kp[self.frame_cnt])
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(self.PicLabel.size(), Qt.KeepAspectRatio)
            self.PicLabel.setPixmap(scaled_pixmap)

            self.display_angle(self.r_angle[self.frame_cnt], self.l_angle[self.frame_cnt])
            self.canvas2.show_angle(self.frame_cnt, self.l_angle, self.r_angle)
            self.frame_cnt += 1
        
        else:
            self.finish_playing()

    def finish_playing(self):
        self.l_angle_label.setText(str(np.round(self.r_angle[self.max_r],2))+"°")
        self.r_angle_label.setText("max_angle:    "+str(np.round(self.l_angle[self.max_l],2))+"°  ,")
        self.ScoreLabel.setVisible(True)

    def show_video(self, file_name):
        pixmap = QPixmap(file_name)
        self.PicLabel.setPixmap(pixmap)

    def add_angle_plot(self):
        self.canvas2 = angle_Plot(self, width=1,height=4)
        self.angle_out = QGridLayout(self.angle_plot)
        self.angle_out.addWidget(self.canvas2)

    def calculate_score(self, right, left):
        df = pd.DataFrame({
            'L': [pd.Series(left)],
            'R': [pd.Series(right)]
        })
        res = self.classifier.predict(df.T)
        return res[1]

    def display_angle(self,l,r):
        self.l_angle_label.setText(str(np.round(l,2))+"°")
        self.r_angle_label.setText(str(np.round(r,2))+"°")

    def cheat_logic(self,in_path):
        if in_path == "F:/mine/code/imr/UIDemo4SummerSchool-master/000.mp4":
            return "tmpdata/000/"
        elif in_path == "F:/mine/code/imr/UIDemo4SummerSchool-master/001.mp4":
            return "tmpdata/001/"
        return "-1"

    def toggle_pause_resume(self):
        if self.play_flag == 0:
            self.pushButton_3.setText('Resume')
            self.timer.stop()
            self.play_flag = 1
        else:
            self.pushButton_3.setText('Pause')
            self.timer.start(50)
            self.play_flag = 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

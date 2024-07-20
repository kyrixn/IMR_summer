# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demo.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1229, 683)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout_main = QtWidgets.QGridLayout()
        self.gridLayout_main.setObjectName("gridLayout_main")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSpacing(20)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAutoFillBackground(False)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 3)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 2, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 2, 2, 1, 1)
        self.ScoreLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.ScoreLabel.setFont(font)
        self.ScoreLabel.setObjectName("ScoreLabel")
        self.gridLayout.addWidget(self.ScoreLabel, 1, 1, 1, 1)
        self.gridLayout_main.addLayout(self.gridLayout, 1, 2, 2, 1)
        spacerItem = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_main.addItem(spacerItem, 2, 0, 1, 1)
        self.angle_plot = QtWidgets.QGroupBox(self.centralwidget)
        self.angle_plot.setObjectName("angle_plot")
        self.gridLayout_main.addWidget(self.angle_plot, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_main.addItem(spacerItem1, 0, 2, 1, 1)
        self.PicLabel = QtWidgets.QLabel(self.centralwidget)
        self.PicLabel.setObjectName("PicLabel")
        self.gridLayout_main.addWidget(self.PicLabel, 0, 0, 2, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout_main.addWidget(self.label, 1, 1, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.r_angle_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.r_angle_label.setFont(font)
        self.r_angle_label.setObjectName("r_angle_label")
        self.horizontalLayout_2.addWidget(self.r_angle_label)
        self.l_angle_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.l_angle_label.setFont(font)
        self.l_angle_label.setObjectName("l_angle_label")
        self.horizontalLayout_2.addWidget(self.l_angle_label)
        self.gridLayout_main.addLayout(self.horizontalLayout_2, 2, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout_main, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1229, 27))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "NIHSS upper limb test"))
        self.label_2.setText(_translate("MainWindow", "NIHSS score"))
        self.pushButton.setText(_translate("MainWindow", "Load video"))
        self.pushButton_2.setText(_translate("MainWindow", "open camera"))
        self.ScoreLabel.setText(_translate("MainWindow", "0"))
        self.angle_plot.setTitle(_translate("MainWindow", "plot of arm angle:"))
        self.PicLabel.setText(_translate("MainWindow", "Your picture will be shown here!"))
        self.label.setText(_translate("MainWindow", "arm angle: (left, right)"))
        self.r_angle_label.setText(_translate("MainWindow", "0"))
        self.l_angle_label.setText(_translate("MainWindow", "0"))

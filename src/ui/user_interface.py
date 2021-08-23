from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtGui import QPainter, QFont
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QVBoxLayout, QTableWidget, QHeaderView, QHBoxLayout, QPushButton, QTableWidgetItem, QAbstractItemView, QSlider, QLineEdit
from PyQt5 import QtWidgets
import PyQt5.QtCore
# from PyQt5.QtChart import QChart, QChartView, QBarSet, QPercentBarSeries, QBarCategoryAxis
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QPalette, QColor
from pyqtgraph.graphicsItems.ViewBox import ViewBox
from PyQt5.QtCore import pyqtSignal, QTime
import pyqtgraph as pg


class UserInterface(QMainWindow):

    signal_in = pyqtSignal(object)

    CHART_0 = "Quantity Remaining"
    CHART_1 = "Step Quantity"
    CHART_2 = "Mid Price"
    CHART_3 = "N/A"

    def __init__(self):
        super().__init__()

        self.main_layout = None
        self.charts = {}

        self.signal_in.connect(self._on_data)

        self._init_ui()
        self._add_components()

    def _init_ui(self):

        self.showMaximized()
        # self.setFixedSize(800, 400)
        self.move(0, 0)

        self.setWindowTitle("RL")
        self._set_basic_background(self, QColor(59, 64, 83))

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        self.main_layout = QGridLayout()
        self.main_layout.setSpacing(2)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        main_widget.setLayout(self.main_layout)

    def _set_basic_background(self, target, color):
        target.setAutoFillBackground(True)
        palette = target.palette()
        palette.setColor(QPalette.Window, color)
        target.setPalette(palette)

    def _add_components(self):
        chart_0 = pg.PlotWidget(title='<h4 style="font: bold;">{}</h4>'.format(UserInterface.CHART_0))
        chart_0.setMouseEnabled(x=False, y=False)

        self.charts["{}#{}".format(UserInterface.CHART_0, "{}".format(0))] = chart_0.plot(pen=pg.mkPen(width=2))
        self.charts["{}#{}".format(UserInterface.CHART_0, "{}".format(1))] = chart_0.plot(
            pen=pg.mkPen(width=2, color=(50, 255, 50)))
        self.main_layout.addWidget(chart_0, 0, 0, 1, 1)

        chart_1 = pg.PlotWidget(title='<h4 style="font: bold;">{}</h4>'.format(UserInterface.CHART_1))
        chart_1.setMouseEnabled(x=False, y=False)

        self.charts["{}#{}".format(UserInterface.CHART_1, "{}".format(0))] = chart_1.plot(pen=pg.mkPen(width=2))
        self.charts["{}#{}".format(UserInterface.CHART_1, "{}".format(1))] = chart_1.plot(
            pen=pg.mkPen(width=2, color=(50, 255, 50)))
        self.main_layout.addWidget(chart_1, 1, 0, 1, 1)

        chart_2 = pg.PlotWidget(title='<h4 style="font: bold;">{}</h4>'.format(UserInterface.CHART_2))
        self.charts["{}#{}".format(UserInterface.CHART_2, "{}".format(0))] = chart_2.plot(pen=pg.mkPen(width=2))
        self.main_layout.addWidget(chart_2, 0, 1, 1, 1)

        chart_3 = pg.PlotWidget(title='<h4 style="font: bold;">{}</h4>'.format(UserInterface.CHART_3))
        self.charts["{}#{}".format(UserInterface.CHART_3, "{}".format(0))] = chart_3.plot(pen=pg.mkPen(width=2))
        self.main_layout.addWidget(chart_3, 1, 1, 1, 1)

    def update_data(self, event):
        self.signal_in.emit(event)

    def _on_data(self, message_events):

        for message_event in message_events:
            event_type = message_event["event"]
            data_series = message_event["data"]

            chart = self.charts[event_type]
            chart.setData(data_series)

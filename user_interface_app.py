import sys
import threading

import pyqtgraph as pg

from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication

from src.ui.ui_comm import UIClient
from src.ui.ui_comm import Charts as Charts


class UIApp(threading.Thread):

    def __init__(self):
        super().__init__()

        self.ui = UserInterface()
        self.network = UIClient()

    def run(self):

        while True:
            msg = self.network.rcv()
            self.ui.update_data(msg["data"])


class UserInterface(QMainWindow):

    signal_in = pyqtSignal(object)
    # signal_out = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        # self.env_controller = EnvController()
        # self.subscriber = subscriber

        self.main_layout = None
        self.charts = {}

        self.signal_in.connect(self._on_data)
        # self.signal_out.connect(self.data_out)

        self._init_ui()
        self._add_components()

        self.wait_step = False
        self.wait_episode = False

    def _init_ui(self):

        self.showMaximized()
        # self.setFixedSize(800, 400)
        self.resize(1000, 600)
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
        chart_0 = pg.PlotWidget(title='<h4 style="font: bold;">{}</h4>'.format(Charts.CHART_0))
        chart_0.setMouseEnabled(x=False, y=False)

        self.charts["{}#{}".format(Charts.CHART_0, "{}".format(0))] = chart_0.plot(pen=pg.mkPen(width=2))
        self.charts["{}#{}".format(Charts.CHART_0, "{}".format(1))] = chart_0.plot(
            pen=pg.mkPen(width=2, color=(50, 255, 50)))
        self.main_layout.addWidget(chart_0, 0, 0, 1, 1)

        chart_1 = pg.PlotWidget(title='<h4 style="font: bold;">{}</h4>'.format(Charts.CHART_1))
        chart_1.setMouseEnabled(x=False, y=False)

        self.charts["{}#{}".format(Charts.CHART_1, "{}".format(0))] = chart_1.plot(pen=pg.mkPen(width=2))
        self.charts["{}#{}".format(Charts.CHART_1, "{}".format(1))] = chart_1.plot(
            pen=pg.mkPen(width=2, color=(50, 255, 50)))
        self.main_layout.addWidget(chart_1, 1, 0, 1, 1)

        chart_2 = pg.PlotWidget(title='<h4 style="font: bold;">{}</h4>'.format(Charts.CHART_2))
        self.charts["{}#{}".format(Charts.CHART_2, "{}".format(0))] = chart_2.plot(pen=pg.mkPen(width=2))
        self.main_layout.addWidget(chart_2, 0, 1, 1, 1)

        chart_3 = pg.PlotWidget(title='<h4 style="font: bold;">{}</h4>'.format(Charts.CHART_3))
        self.charts["{}#{}".format(Charts.CHART_3, "{}".format(0))] = chart_3.plot(pen=pg.mkPen(width=2))
        self.main_layout.addWidget(chart_3, 1, 1, 1, 1)

    def update_data(self, event):
        self.signal_in.emit(event)

    def _on_data(self, message_events):

        for message_event in message_events:
            event_type = message_event["event"]
            data_series = message_event["data"]

            chart = self.charts[event_type]
            chart.setData(data_series)

    # def keyPressEvent(self, event):
    #
    #     if event.key() == QtCore.Qt.Key_S:
    #         self.wait_step = not self.wait_step
    #
    #         self.signal_out.emit({
    #             "event": "wait_step",
    #             "data": self.wait_step
    #         })
    #
    #     elif event.key() == QtCore.Qt.Key_E:
    #         self.wait_episode = not self.wait_episode
    #
    #         self.signal_out.emit({
    #             "event": "wait_episode",
    #             "data": self.wait_episode
    #         })
    #
    #     elif event.key() == QtCore.Qt.Key_N:
    #
    #         self.signal_out.emit({
    #             "event": "next_step",
    #             "data": None
    #         })

    # def data_out(self, message):
    #     self.subscriber.on_controller_event(message)


if __name__ == "__main__":

    qapp = QApplication(sys.argv)

    uiApp = UIApp()
    uiApp.start()

    sys.exit(qapp.exec_())

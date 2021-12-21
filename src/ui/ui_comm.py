#
#
#

import time
import json
import zmq

# Feel free to change to any available port that you want:
UI_COMMUNICATION_ZMQ_PORT = "8392"


class Charts:

    CHART_0 = "Market mid prices & Bench & RL VWAPs"
    CHART_1 = "Bench & RL VWAPs"
    CHART_2 = "Mid Price"
    CHART_3 = "N/A"


class UIServer:

    def __init__(self):

        self.rl_session_id = int(time.time())

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)

        self.socket.bind("tcp://*:{}".format(UI_COMMUNICATION_ZMQ_PORT))

    def send_rendering_data(self, rl_session_epoch, data):

        self.socket.send_string("gui {}".format(json.dumps({
            "rl_session_id": self.rl_session_id,
            "rl_session_epoch": rl_session_epoch,
            "data": data
        })))


class UIClient:

    def __init__(self, address="localhost"):

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        self.socket.connect("tcp://{}:{}".format(address, UI_COMMUNICATION_ZMQ_PORT))
        self.socket.setsockopt(zmq.SUBSCRIBE, b"gui")

    def rcv(self):

        msg = self.socket.recv().decode("utf-8")[4:]
        # print("RX: {}".format(msg))

        return json.loads(msg)

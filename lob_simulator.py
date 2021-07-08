import json
from decimal import Decimal
import random


def split_book_to_orders(current_book):
    """ Splits existing order book data into individual bid and ask orders """

    # pre-allocate
    bid_orders = []
    ask_orders = []

    # get meta info
    time_stamp = current_book["T"]
    trade_id = 0

    # extract bid orders
    for bid in current_book["b"]:
        bid_order = {'type' : 'limit',
                     'time_stamp': time_stamp,
                     'side' : 'bid',
                     'quantity' : Decimal(bid[1]),
                     'price' : Decimal(bid[0]),
                     'trade_id' : trade_id}
        trade_id += 1
        bid_orders.append(bid_order)

    # extract ask orders
    for ask in current_book["a"]:
        ask_order = {'type' : 'limit',
                     'timestamp': time_stamp,
                     'side' : 'ask',
                     'quantity' : Decimal(ask[1]),
                     'price' : Decimal(ask[0]),
                     'trade_id' : trade_id}
        trade_id += 1
        ask_orders.append(ask_order)
    all_orders = bid_orders + ask_orders

    return bid_orders, ask_orders, all_orders


class LOBObserver:
    def __init__(self, file_name, past_n, future_n):
        self.file_name = file_name
        self.past_n = past_n
        self.future_n = future_n

        with open(self.file_name) as f:
            for i, l in enumerate(f):
                pass
        self.line_count = i + 1

    def _reset_observer(self):
        self.hist_orders = []
        self.fut_orders = []

    def draw_random_observations(self):

        self._reset_observer()
        # random integer from 0 to 9
        rand_idx = random.randint(self.past_n, self.line_count - self.future_n)

        line_counter_hist = 1
        line_counter_fut = 1
        hist_orders = []
        fut_orders = []
        with open(self.file_name, "r+") as fp:
            for i, line in enumerate(fp):
                if i == rand_idx - self.past_n - 1 + line_counter_hist and line_counter_hist <= self.past_n:
                    read_data = json.loads(line)
                    hist_orders.append(read_data)
                    line_counter_hist += 1
                elif i == rand_idx - 1 + line_counter_fut and line_counter_fut <= self.future_n:
                    read_data = json.loads(line)
                    fut_orders.append(read_data)
                    line_counter_fut += 1
                elif i > rand_idx + self.future_n:
                    break
        self.hist_orders = hist_orders
        self.fut_orders = fut_orders
        return self


if __name__ == '__main__':

    # Example implementation
    lob_data = LOBObserver(file_name="./data/book_depth_socket_btcusdt_2021_06_21.txt",
                           past_n=10,
                           future_n=20)


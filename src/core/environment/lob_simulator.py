import json
from decimal import Decimal
import random
from src.core.environment.orderbook import OrderBook


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


def raw_to_order_book(current_book, depth):
    order_book = OrderBook()
    bid_orders, ask_orders, _ = split_book_to_orders(current_book)

    counter = 1
    for bids in bid_orders:
        _, _ = order_book.process_order(bids, False, False)
        counter += 1
        if counter > depth:
            break

    counter = 1
    for asks in ask_orders:
        _, _ = order_book.process_order(asks, False, False)
        counter += 1
        if counter > depth:
            break
    return order_book


class LOBSimulator:
    def __init__(self, file_name, nr_of_lobs, nr_of_lob_levels):
        self.file_name = file_name
        self.nr_of_lobs = nr_of_lobs
        self.nr_of_lob_levels = nr_of_lob_levels

        with open(self.file_name) as f:
            for i, l in enumerate(f):
                pass
        self.line_count = i + 1
        self.hist_count = 0

    def _reset_observer(self):
        self.hist_orders = []
        self.fut_orders = []
        self.hist_count = 0

    def _draw_random_observations(self, past_n, future_n):

        self._reset_observer()

        rand_idx = random.randint(past_n, self.line_count - future_n)

        line_counter_hist = 1
        line_counter_fut = 1
        hist_orders = []
        fut_orders = []
        with open(self.file_name, "r+") as fp:
            for i, line in enumerate(fp):
                if i == rand_idx - past_n - 1 + line_counter_hist and line_counter_hist <= past_n:
                    read_data = json.loads(line)
                    hist_orders.append(read_data)
                    line_counter_hist += 1
                elif i == rand_idx - 1 + line_counter_fut and line_counter_fut <= future_n:
                    read_data = json.loads(line)
                    fut_orders.append(read_data)
                    line_counter_fut += 1
                elif i > rand_idx + future_n:
                    break
        self.hist_orders = hist_orders
        self.fut_orders = fut_orders
        return self

    def lob_snapshot(self, past_n, future_n, depth):
        self._draw_random_observations(past_n=past_n, future_n=future_n)
        book_hist = []
        for i in range(0,len(self.hist_orders)):
            book_hist.append(raw_to_order_book(self.hist_orders[i], depth=depth))
        return book_hist

    def next_lob_snapshot(self, depth, previous_lob_snapshot=None):
        if self.hist_count >= len(self.fut_orders):
            print('Cannot return anything beyond observed values. Need to call "lob_snapshot()" again!')
            return
        next_book = raw_to_order_book(self.fut_orders[self.hist_count], depth)
        self.hist_count += 1
        return next_book


if __name__ == '__main__':

    # Example implementation
    pth = 'C:\\Users\\auth\\projects\\python\\reinforcement learning\\RLOptimalTradeExecution\\src\\data\\book_depth_socket_btcusdt_2021_06_21.txt'
    lob_feed = LOBSimulator(file_name=pth,
                            nr_of_lobs=5,
                            nr_of_lob_levels=20)

    # test the functionality of this as used in the gym environment...
    depth = 5
    lob_hist = lob_feed.lob_snapshot(past_n=3, future_n=2, depth=depth)
    next_lob_1 = lob_feed.next_lob_snapshot(depth=depth)
    next_lob_2 = lob_feed.next_lob_snapshot(depth=depth)
    next_lob_3 = lob_feed.next_lob_snapshot(depth=depth)

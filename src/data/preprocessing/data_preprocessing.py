import gzip
import json
import numpy as np


"""
    Flat binary format, each float is saved as a float64 in a continuous memory:
        timestamp, 20*ask_prices, 20*ask_quantities, 20*bid_prices, 20*bid_quantities, ..., continuous...
    
    After reading from disk, do: .reshape(-1, 81)
"""


class BinanceOrderBookReSampler:

    def __init__(self, delta_time):

        self.delta_time = delta_time

        self.lag_timestamp = 0
        self.book = []

    def resample(self, raw_filepath):

        with gzip.open(raw_filepath, 'rt') as raw_json_file:
            for raw_book_snapshot in raw_json_file:
                raw_book_snapshot = json.loads(raw_book_snapshot)

                new_timestamp = raw_book_snapshot["T"]

                current_delta_time = new_timestamp - self.lag_timestamp

                # if current_delta_time > self.break_on_delta_gap_amt:
                #     self._save_book_to_disk()

                if current_delta_time >= self.delta_time:
                    self._append_book_snapshot(new_timestamp,
                                               raw_book_snapshot["a"],
                                               raw_book_snapshot["b"])
                    self.lag_timestamp = new_timestamp

        self._save_book_to_disk(raw_filepath[-17:-7])

    def _append_book_snapshot(self, timestamp, asks_dict, bids_dict):

        flat_book = [timestamp]

        pxs, qtys = self._get_prices_to_quantities(asks_dict)
        flat_book += pxs
        flat_book += qtys

        pxs, qtys = self._get_prices_to_quantities(bids_dict)
        flat_book += pxs
        flat_book += qtys

        self.book.append(flat_book)

    def _get_prices_to_quantities(self, data):

        pxs = []
        qtys = []
        for price2qty in data:
            pxs.append(float(price2qty[0]))
            qtys.append(float(price2qty[1]))

        return pxs, qtys

    def _save_book_to_disk(self, human_time):

        data = np.asarray(self.book, dtype=np.float64)
        data.tofile("data/binary/btcusdt/btcusdt__{}__{}__{}.dat".format(human_time,
                                                                         int(data[0][0]),
                                                                         int(data[-1][0])))
        self.book = []


file = 'data/raw/binance_futures/book_depth_socket_btcusdt_2021_06_22.txt.gz'
DELTA_TIME_MS = 1000

rs = BinanceOrderBookReSampler(DELTA_TIME_MS)
rs.resample(file)

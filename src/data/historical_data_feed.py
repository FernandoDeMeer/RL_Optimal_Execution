from src.data.data_feed import DataFeed
import numpy as np
from os import listdir


class HistoricalDataFeed(DataFeed):

    """
        Flat binary format, each float is saved as a float64** in a continuous memory:
            timestamp, 20 ask_prices, 20 ask_quantities, 20 bid_prices, 20 bid_quantities, ..., continuous...

            **float64 is used to accomodate the millisecond timestamp

        After reading file from disk, do: .reshape(-1, 81)
    """

    def __init__(self, data_dir, instrument, lob_depth, start_day, end_day):

        self.data_dir = data_dir
        self.instrument = instrument
        self.lob_depth = lob_depth

        self.binary_files = listdir("{}/{}".format(data_dir, instrument))
        self.binary_file_idx = 0

        self.data = None
        self.data_row_idx = 0
        self.remaining_rows_in_file = -1

    def next_lob_snapshot(self, previous_lob_snapshot=None):

        if self.remaining_rows_in_file < 0:
            if self.binary_file_idx < len(self.binary_files):
                self._load_data()
            else:
                # no binary files left to process
                return 0, None

        lob = self.data[self.data_row_idx]

        self.data_row_idx += 1
        self.remaining_rows_in_file -= 1

        timestamp = lob[0]
        lob = lob[1:]
        return int(timestamp), lob.reshape(-1, self.lob_depth)

    def _load_data(self):

        self.data = np.fromfile("{}/{}/{}".format(self.data_dir,
                                                  self.instrument,
                                                  self.binary_files[self.binary_file_idx]), dtype=np.float64)
        self.data = self.data.reshape(-1, 4 * self.lob_depth + 1)
        self.remaining_rows_in_file = self.data.shape[0]
        self.data_row_idx = 0

        self.binary_file_idx += 1

"""
feed = HistoricalDataFeed(data_dir='/Users/florindascalu/data/iwa/LOBProcessing/data/binary',
                          instrument='btcusdt',
                          lob_depth=20,
                          start_day=None,
                          end_day=None)
while True:

    timestamp, lob = feed.next_lob_snapshot()
    # lob shape is (4, 20)

    if lob is None:
        break

    print(lob)
"""

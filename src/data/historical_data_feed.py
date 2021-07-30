import numpy as np
from os import listdir
import random
from src.data.data_feed import DataFeed
from src.core.environment.env_utils import raw_to_order_book


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
        self.binary_file_idx = None

        self.data = None
        self.data_row_idx = None
        self.remaining_rows_in_file = None
        self._row_buffer = None

    def next_lob_snapshot(self, previous_lob_snapshot=None, lob_format=True):

        assert self.remaining_rows_in_file is not None, (
            'reset() must be called once before next_lob_snapshot()')

        if self.remaining_rows_in_file < 0:
            self.reset(self._row_buffer)

        lob = self.data[self.data_row_idx]

        self.data_row_idx += 1
        self.remaining_rows_in_file -= 1

        timestamp = lob[0]
        lob = lob[1:]
        if lob_format:
            lob_out = raw_to_order_book(current_book=lob.reshape(-1, self.lob_depth),
                                    time=int(timestamp),
                                    depth=self.lob_depth)
            return int(timestamp), lob_out
        else:
            return int(timestamp), lob.reshape(-1, self.lob_depth)

    def reset(self, row_buffer=None):
        """
        Logic for the reset method:
            * Decide what files to choose whenever reset is called
            * Check if the file index exists
                * if so, load data...
            * Decide what row to start reading data from
            * Determine remaining number of rows

        """
        if not self.binary_file_idx:
            self.binary_file_idx = 0
        else:
            self.binary_file_idx += 1

        if self.binary_file_idx < len(self.binary_files):
            self._load_data()
        else:
            # no binary files left to process
            return 0, None

        self.data_row_idx = 0
        self.remaining_rows_in_file = self.data.shape[0]

    def _load_data(self):

        self.data = np.fromfile("{}/{}/{}".format(self.data_dir,
                                                  self.instrument,
                                                  self.binary_files[self.binary_file_idx]), dtype=np.float64)
        self.data = self.data.reshape(-1, 4 * self.lob_depth + 1)


class HistFeedRL(HistoricalDataFeed):
    """
    Example of a derived class that allows...
        * ...training on one single day of data without changing the file (could also be made
        possible via the start and end date I suppose, if that's the idea of those...)
        * ...at each reset, draw starting time (t0) randomly from 0 to len(data)
        * include a row_buffer to reflect no allowed trading if not enough time is
        left at the end of the day.

    """
    def __init__(self, data_dir, instrument, lob_depth, start_day, end_day):
        super(HistFeedRL, self).__init__(data_dir,
                                         instrument,
                                         lob_depth,
                                         start_day,
                                         end_day)

    def reset(self, row_buffer=None):

        if not row_buffer:
            row_buffer = 0
        self._row_buffer = row_buffer

        if not self.binary_file_idx:
            self.binary_file_idx = 0

        if self.binary_file_idx < len(self.binary_files):
            self._load_data()
        else:
            # no binary files left to process
            return 0, None

        self.data_row_idx = random.randint(0, self.data.shape[0] - row_buffer)
        self.remaining_rows_in_file = self.data.shape[0] - self.data_row_idx


"""    
    feed = HistoricalDataFeed(data_dir='/Users/florindascalu/data/iwa/LOBProcessing/data/binary',
                              instrument='btcusdt',
                              lob_depth=10,
                              start_day=None,
                              end_day=None)
    feed.reset()
    while True:

        timestamp, lob = feed.next_lob_snapshot()
        # lob shape is (4, 20)

        if lob is None:
            break

        print(lob)
"""

"""
Example code for testing the derived class

if __name__ == "__main__":

    dir = 'C:\\Users\\auth\\projects\\python\\reinforcement learning\\RLOptimalTradeExecution\\data_dir'
    feed = HistFeedRL(data_dir=dir,
                      instrument='btc_usdt',
                      lob_depth=20,
                      start_day=None,
                      end_day=None)

    feed.reset(row_buffer=200)
    while True:

        timestamp, lob = feed.next_lob_snapshot()
        # lob shape is (4, 20)

        if lob is None:
            break

        print(lob)
"""
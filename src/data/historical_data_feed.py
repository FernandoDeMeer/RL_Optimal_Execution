import warnings
import copy
import calendar
import numpy as np
from os import listdir, path
from datetime import datetime, timedelta
from src.data.data_feed import DataFeed
from src.core.environment.env_utils import raw_to_order_book


def get_time_idx_from_raw_data(data, t):
    """ Returns the index of data right before a given time 't' """

    dt = datetime.utcfromtimestamp(data[-1] / 1000)
    try:
        start_t = datetime.strptime(t, '%H:%M:%S.%f')
    except:
        start_t = datetime.strptime(t, '%H:%M:%S')
    start_dt = datetime(dt.year, dt.month, dt.day, start_t.hour, start_t.minute, start_t.second, start_t.microsecond)
    unix_t = calendar.timegm(start_dt.utctimetuple()) * 1e3 + start_dt.microsecond / 1e3
    idx = (np.abs(data - unix_t)).argmin()
    # These two lines were giving the lob_snapshot previous to dt, when it should be the one after If we are placing trades (to account for computing time/latency etc)
    while data[idx] <= unix_t:
        idx = idx + 1
    return idx


class HistoricalDataFeed(DataFeed):
    """
        Flat binary format, each float is saved as a float64** in a continuous memory:
            timestamp, 20 ask_prices, 20 ask_quantities, 20 bid_prices, 20 bid_quantities, ..., continuous...

            **float64 is used to accommodate the millisecond timestamp

        After reading file from disk, do: .reshape(-1, 81)
    """

    def __init__(self,
                 data_dir,
                 instrument,
                 start_day=None,
                 end_day=None,
                 time=None,
                 lob_depth=20):

        self.data_dir = data_dir
        self.instrument = instrument

        self.start_day = start_day
        self.end_day = end_day

        if start_day is None and end_day is None:

            # load all files available
            self.binary_files = listdir("{}".format(data_dir))
        elif None not in (start_day, end_day):

            # load only relevant files between dates
            files_between_date = []
            while start_day <= end_day:
                f = "{}__{}.{}".format(instrument, start_day.strftime('%Y_%m_%d'), "dat")
                if path.isfile("{}/{}".format(data_dir, f)):
                    files_between_date.append(f)
                start_day += timedelta(1)
            self.binary_files = files_between_date
        else:
            raise ValueError("'start_day' and 'end_day' have to be defined jointly!")

        self.data = None

        self.binary_file_idx = 0
        self.data_row_idx = None
        self._remaining_rows_in_file = None

        self.lob_depth = lob_depth
        self._load_data()
        self.reset(time)

    def next_lob_snapshot(self, previous_lob_snapshot=None, lob_format=True):
        """ return next snapshot of the limit order book """

        assert self.binary_file_idx >= 0, (
            'hard reset() must be applied once before next_lob_snapshot(), if called via broker reset with hard=True!')

        assert self._remaining_rows_in_file is not None, (
            'reset() must be called once before next_lob_snapshot()')

        if self._remaining_rows_in_file <= 0:
            warnings.warn("Datafeed reached end of file, reset to initial time. Make sure this was intended! ")
            self.reset(self.time)

        lob = copy.deepcopy(self.data[self.data_row_idx])

        self.data_row_idx += 1
        self._remaining_rows_in_file -= 1

        timestamp_dt = datetime.utcfromtimestamp(lob[0] / 1000)
        lob = lob[1:]
        if lob_format:
            lob_out = raw_to_order_book(current_book=lob.reshape(-1, self.lob_depth),
                                        time=timestamp_dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                        depth=self.lob_depth)
            return timestamp_dt, lob_out
        else:
            return timestamp_dt, lob.reshape(-1, self.lob_depth)

    def past_lob_snapshots(self, no_of_past_lobs, lob_format=True):
        """ return past snapshots of the limit order book """

        past_lobs = self.data[self.data_row_idx-no_of_past_lobs:self.data_row_idx ]
        timestamp_dts =[]
        for lob in past_lobs:
            timestamp_dts.append(datetime.utcfromtimestamp(lob[0] / 1000))

        output = []
        if lob_format:
            for lob in past_lobs:
                lob_out = raw_to_order_book(current_book=lob[1:].reshape(-1, self.lob_depth),
                                            time=datetime.utcfromtimestamp(lob[0] / 1000).strftime('%Y-%m-%d %H:%M:%S.%f'),
                                            depth=self.lob_depth)
                output.append(lob_out)
            return timestamp_dts, output
        else:
            for lob in past_lobs:
                output.append(lob.reshape(-1, self.lob_depth))
            return timestamp_dts, output

    def reset(self, time=None):
        """ Reset the datafeed and set from where to start sampling """

        self.time = time
        self._select_row_idx()

    def next_data(self):
        """ Jumps to next file and loads data """

        self.binary_file_idx += 1
        if self.binary_file_idx > len(self.binary_files) - 1:
            self.binary_file_idx = 0
        self._load_data()
        self.reset()

    def _load_data(self):
        """ Load data from file """

        self.data = np.fromfile("{}/{}".format(self.data_dir, self.binary_files[self.binary_file_idx]), dtype=np.float64)
        self.data = self.data.reshape(-1, 4 * self.lob_depth + 1)

    def load_specific_day_data(self, binary_file_idx):

        self.binary_file_idx = binary_file_idx
        self._load_data()

    def _select_row_idx(self):
        """ method specifically selecting 'data_row_idx' and '_remaining_rows_in_file' """

        if self.time is not None:
            # always start sampling from 'time'
            idx = get_time_idx_from_raw_data(self.data[:, 0], self.time)
        else:
            # otherwise just start from the beginning
            idx = 0

        self.data_row_idx = idx
        self._remaining_rows_in_file = self.data.shape[0] - idx

    def get_daily_vols(self):
        volatilities = []
        for day in range(len(self.binary_files)):
            mid_prices = []
            for lob_idx in range(self.data.shape[0]):
                lob = self.data[lob_idx][1:].reshape(-1, self.lob_depth)
                # lob = raw_to_order_book(current_book=lob,
                #                             time=datetime.utcfromtimestamp(lob[0] / 1000).strftime('%Y-%m-%d %H:%M:%S.%f'),
                #                             depth=self.lob_depth)
                mid_price = (lob[0,0] + lob[0,2]) / 2
                mid_prices.append(mid_price)

            day_vol = np.std(np.array(mid_prices))
            volatilities.append(day_vol)
            # Jump to the next day
            self.next_data()
        self.day_volatilities = volatilities
        self.day_volatilities_ranking = np.argsort(volatilities)

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            for k, v in self.__dict__.items():
                try:
                    tst = v == other.__dict__[k]
                    if not tst:
                        return False
                except:
                    tst = (v == other.__dict__[k]).any()
                    if not tst:
                        return False
            return True
        else:
            raise TypeError('Comparing object is not of the same type.')

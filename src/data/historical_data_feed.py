from src.data.data_feed import DataFeed


class HistoricalDataFeed(DataFeed):

    def __init__(self, nr_of_lobs, nr_of_lob_levels):
        pass

    def next_lob_snapshot(self, previous_lob_snapshot=None):
        """
            load data from binary files on-demand to avoid RAM issues.
        """
        return None

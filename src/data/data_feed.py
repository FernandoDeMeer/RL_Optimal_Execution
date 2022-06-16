from abc import ABC, abstractmethod

HISTORICAL_DATA_FEED    = "historical"

class DataFeed(ABC):

    @abstractmethod
    def next_lob_snapshot(self, previous_lob_snapshot=None):
        """  Return next snapshot of the limit order book """
        raise NotImplementedError
    @abstractmethod
    def reset(self, time=None):
        """
         Reset the datafeed and set from when to start sampling.
        Args:
            time: datetime. Timestamp from which to start sampling.
        """
        raise NotImplementedError

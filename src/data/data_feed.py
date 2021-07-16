from abc import ABC, abstractmethod

HISTORICAL_DATA_FEED    = "historical"
GAN_LOB_DATA_FEED       = "gan_lob"


class DataFeed(ABC):

    @abstractmethod
    def next_lob_snapshot(self, previous_lob_snapshot=None):
        """
            :param previous_lob_snapshot: used at t0 as a condition for t+1
            :return:
        """
        pass

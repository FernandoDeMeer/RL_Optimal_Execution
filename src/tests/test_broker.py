import unittest
import os

from src.core.environment.limit_orders_setup.broker_real import Broker
from src.core.environment.limit_orders_setup.execution_algo_real import TWAPAlgo
from src.data.historical_data_feed import HistoricalDataFeed

# from train_app import ROOT_DIR
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..'))


class TestBroker(unittest.TestCase):

    # define the datafeed
    lob_feed = HistoricalDataFeed(data_dir=os.path.join(ROOT_DIR, 'data/market/btcusdt/'),
                                  instrument='btc_usdt',
                                  samples_per_file=200)

    algo = TWAPAlgo(trade_direction=1,
                    volume=25,
                    start_time='09:00:00',
                    end_time='09:01:00',
                    no_of_slices=1,
                    bucket_placement_func=lambda no_of_slices: 0.5,
                    broker_data_feed=lob_feed)

    # define the broker class
    broker = Broker(lob_feed)
    broker.simulate_algo(algo)

    def test_bmk_algo_simulation(self):
        vwap_bmk, vwap_rl = self.broker.calc_vwap_from_logs()
        self.assertEqual(vwap_rl, 0, 'VWAP for RL algo should be 0/not calculated')
        self.assertEqual(vwap_bmk, 32876.9147788, 'VWAPS are not matching')


if __name__ == '__main__':
    unittest.main()
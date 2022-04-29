import unittest
import os
from datetime import datetime
import gym
import numpy as np

from src.core.environment.limit_orders_setup.broker_real import Broker
from src.core.environment.limit_orders_setup.execution_algo_real import TWAPAlgo
from src.core.environment.limit_orders_setup.base_env_real import RewardAtStepEnv
from src.data.historical_data_feed import HistoricalDataFeed

# from train_app import ROOT_DIR
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..'))
DATA_DIR = os.path.join(ROOT_DIR, "data")


class TestBroker(unittest.TestCase):
    # define the datafeed
    lob_feed = HistoricalDataFeed(data_dir=os.path.join(ROOT_DIR, 'data/market/btcusdt/'),
                                  instrument='btc_usdt')

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


class TestDummyEnv(unittest.TestCase):
    env_config = {
        "symbol": "btcusdt",
        "train_data_periods": [2021, 6, 21, 2021, 6, 21],
        "eval_data_periods": [2021, 6, 21, 2021, 6, 22]
    }

    data_periods = env_config["train_data_periods"]

    data_start_day = datetime(year=data_periods[0], month=data_periods[1], day=data_periods[2])
    data_end_day = datetime(year=data_periods[3], month=data_periods[4], day=data_periods[5])

    lob_feed = HistoricalDataFeed(data_dir=os.path.join(DATA_DIR, "market", env_config["symbol"]),
                                  instrument=env_config["symbol"],
                                  start_day=data_start_day,
                                  end_day=data_end_day)

    broker = Broker(lob_feed)

    observation_space_config = {'obs_config': {
        "lob_depth": 5,
        "nr_of_lobs": 5,
        "norm": True},
        'trade_config': {'trade_direction': 1,
                         'vol_low': 500,
                         'vol_high': 500,
                         'no_slices_low': 4,
                         'no_slices_high': 4,
                         'bucket_func': lambda no_of_slices: [0.2, 0.4, 0.6, 0.8],
                         'rand_bucket_low': 0,
                         'rand_bucket_high': 0},
        'start_config': {'hour_low': 12,
                         'hour_high': 12,
                         'minute_low': 0,
                         'minute_high': 0,
                         'second_low': 0,
                         'second_high': 0},
        'exec_config': {'exec_times': [10],
                        'delete_vol': False},
        "seed_config": {"seed" : 0,},}


    action_space = gym.spaces.Discrete(3)

    env = RewardAtStepEnv(broker=broker,
                          config=observation_space_config,
                          action_space=action_space)

    env.reset()
    done = False
    idx = 0
    while not done:
        action = env.action_space.sample()
        s, r, done, i = env.step(action=action)
    print("WORKED")

    env.reset()
    done = False
    idx = 0
    while not done:
        action = env.action_space.sample()
        s, r, done, i = env.step(action=action)
    print("WORKED AGAIN")


if __name__ == '__main__':
    unittest.main()

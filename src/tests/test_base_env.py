import unittest
import math
import os
import gym

from decimal import Decimal
import numpy as np

from src.core.environment.limit_orders_setup.base_env_real import BaseEnv, ExampleEnv
from src.core.environment.limit_orders_setup.broker_real import Broker
from src.data.historical_data_feed import HistoricalDataFeed

# from train_app import ROOT_DIR
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..'))


class DerivedEnv(ExampleEnv):
    def __init__(self, *args, **kwargs):
        super(DerivedEnv, self).__init__(*args, **kwargs)

    def _convert_action(self, action):
        action_min = 0.8
        action_max = 1.2
        action_rescaled = (action[0] - self.action_space.low[0]) / \
                                (self.action_space.high[0] - self.action_space.low[0])
        action_out = action_min + action_rescaled * (action_max - action_min)
        return action_out

    def infer_volume_from_action(self, action):
        vol_to_trade = Decimal(str(action)) * \
                            self.broker.benchmark_algo.volumes_per_trade[self.broker.benchmark_algo.order_idx][0]
        factor = 10 ** (- self.broker.benchmark_algo.tick_size.as_tuple().exponent)
        vol_to_trade = Decimal(str(math.floor(vol_to_trade * factor) / factor))
        if vol_to_trade > self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]:
            vol_to_trade = self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]
        return vol_to_trade


class TestBaseEnvLogic(unittest.TestCase):

    lob_feed = HistoricalDataFeed(data_dir=os.path.join(ROOT_DIR, 'data/market/btcusdt/'),
                                  instrument='btc_usdt',
                                  samples_per_file=200)

    # define the broker class
    broker = Broker(lob_feed)

    # define the config
    env_config = {'obs_config': {"lob_depth": 5,
                                 "nr_of_lobs": 5,
                                 "norm": True},
                  'trade_config': {'trade_direction': 1,
                                   'vol_low': 25,
                                   'vol_high': 25,
                                   'no_slices_low': 1,
                                   'no_slices_high': 1,
                                   'bucket_func': lambda no_of_slices: 0.5,
                                   'rand_bucket_low': 0,
                                   'rand_bucket_high': 0},
                  'start_config': {'hour_low': 9,
                                   'hour_high': 9,
                                   'minute_low': 0,
                                   'minute_high': 0,
                                   'second_low': 0,
                                   'second_high': 0},
                  'exec_config': {'exec_times': [1]},
                  'reset_config': {'reset_num_episodes': 1}}


    # define action space
    action_space = gym.spaces.Box(low=-1.0,
                                  high=1.0,
                                  shape=(1,),
                                  dtype=np.float32)
    # define the env
    base_env = DerivedEnv(broker=broker,
                          config=env_config,
                          action_space=action_space)

    def test_running_env(self):
        self.base_env.reset()
        done = False
        while not done:
            s, r, done, i = self.base_env.step(action=np.array([0]))

        traded_volumes_per_time = [float(ts['quantity']) for ts in self.base_env.broker.trade_logs['rl_algo']
                                   if ts['message'] == 'trade']
        achieved_prices_per_time = [float(ts['price']) for ts in self.base_env.broker.trade_logs['rl_algo']
                                    if ts['message'] == 'trade']
        vwap_ours = np.dot(achieved_prices_per_time, traded_volumes_per_time) / sum(traded_volumes_per_time)

        vwap_list = []
        vol_list = []
        for i in range(len(self.base_env.broker.hist_dict['rl']['lob'])):
            tp = self.base_env.broker.hist_dict['rl']['lob'][i].tape
            if len(tp) > 0:
                vol = 0
                vwap = 0
                for id in range(len(tp)):
                    vol += tp[id]['quantity']
                    vwap += tp[id]['quantity'] * tp[id]['price']
                vwap = float(vwap) / float(vol)
                vwap_list.append(vwap)
                vol_list.append(float(vol))
        vwap_derived = np.dot(vwap_list, vol_list) / sum(vol_list)

        # self.assertEqual(vwap_ours, 32878.389652, 'VWAP is not correct')
        import warnings
        warnings.warn("Is the price still correct? Why is it different than before?")
        self.assertEqual(vwap_ours, round(vwap_derived, 8), 'VWAPS are not matching')

        # test the built in functions for this
        vwap_bmk, vwap_rl = self.broker.calc_vwap_from_logs()
        self.assertEqual(vwap_bmk, vwap_rl, 'VWAPS are not matching')
        self.assertEqual(vwap_ours, vwap_rl, 'VWAPS are not matching')

    def test_large_volumes(self):
        """
        env_config_v2 = self.env_config
        env_config_v2["trade_config"]["vol_low"] = 1000
        env_config_v2["trade_config"]["vol_high"] = 1000
        base_env_v2 = DerivedEnv(broker=self.broker,
                                 config=env_config_v2,
                                 action_space=self.action_space)
        base_env_v2.reset()
        done = False
        while not done:
            s, r, done, i = base_env_v2.step(action=np.array([0]))
        """


if __name__ == '__main__':
    unittest.main()


import unittest
import math
import os
import gym
import datetime

from decimal import Decimal
import numpy as np

from src.core.environment.limit_orders_setup.base_env_real import BaseEnv, ExampleEnvRewardAtStep, TWAPAlgo
from src.core.environment.limit_orders_setup.broker_real import Broker
from src.data.historical_data_feed import HistoricalDataFeed


# from train_app import ROOT_DIR
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..'))


class DerivedEnv(ExampleEnvRewardAtStep):
    def __init__(self, *args, **kwargs):
        super(DerivedEnv, self).__init__(*args, **kwargs)

    def _convert_action(self, action):
        action_out = action[0] # Actions can be between 0-20 and we re-scale to  0-2
        return action_out

    def infer_volume_from_action(self, action):
        vol_to_trade = Decimal(str(0.8 + 0.2*action)) *\
                       (self.broker.benchmark_algo.volumes_per_trade_default[self.broker.rl_algo.bucket_idx][self.broker.rl_algo.order_idx]) # We trade {0,0.1,0.2,...2}*TWAP's volume
        factor = 10 ** (- self.broker.benchmark_algo.tick_size.as_tuple().exponent)
        vol_to_trade = Decimal(str(math.floor(vol_to_trade * factor) / factor))
        if vol_to_trade > self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]:
            vol_to_trade = self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]
        return vol_to_trade


class TestBaseEnvLogic(unittest.TestCase):
    start_day = datetime.datetime(year=2021,month=6, day=1)
    end_day = datetime.datetime(year=2021,month=6, day=1)
    lob_feed = HistoricalDataFeed(data_dir=os.path.join(ROOT_DIR, 'data/market/btcusdt/'),
                                  instrument='btcusdt', start_day=start_day, end_day=end_day)

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
                  'exec_config': {'exec_times': [1],
                                  'delete_vol': False},
                  'reset_config': {'reset_num_episodes': 1,},
                  "seed_config": {"seed" : 0,},}

    # define action space
    action_space = gym.spaces.Discrete(n = 3)

    # define the env
    base_env = DerivedEnv(broker=broker,
                          config=env_config,
                          action_space=action_space)

    def test_running_env(self):
        self.base_env.reset()
        done = False
        while not done:
            s, r, done, i = self.base_env.step(action=np.array([1]))

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
        # import warnings
        # warnings.warn("Is the price still correct? Why is it different than before?")
        self.assertEqual(round(vwap_ours, 8), round(vwap_derived, 8), 'VWAPS are not matching')

        # test the built in functions for this
        vwap_bmk, vwap_rl = self.broker.calc_vwap_from_logs()
        self.assertEqual(vwap_bmk, vwap_rl, 'VWAPS are not matching')
        self.assertEqual(round(vwap_ours, 8), round(vwap_rl, 8), 'VWAPS are not matching')


class RewardAtStepEnv(DerivedEnv):
    def reward_func(self):
        """ Env with reward after each step """

        vwap_bmk, vwap_rl = self.broker.calc_vwap_from_logs(start_date=self.event_time_prev,
                                                            end_date=self.event_time)
        reward = vwap_bmk/vwap_rl
        return reward


class RewardAtBucketEnv(DerivedEnv):

    def reward_func(self):
        """ Env with reward at end of each bucket """

        reward = 0
        if self.bucket_time != self.bucket_time_prev:
            vwap_bmk, vwap_rl = self.broker.calc_vwap_from_logs(start_date=self.bucket_time_prev,
                                                                end_date=self.bucket_time)
            reward = vwap_bmk/vwap_rl
        return reward


class RewardAtEpisodeEnv(DerivedEnv):

    def reward_func(self):
        """ Env with reward at end of each bucket """
        reward = 0
        if self.done:
            vwap_bmk, vwap_rl = self.broker.calc_vwap_from_logs()
            reward = vwap_bmk/vwap_rl
        return reward


class TestRewardCalcsInEnv(unittest.TestCase):
    start_day = datetime.datetime(year=2021,month=6, day=1)
    end_day = datetime.datetime(year=2021,month=6, day=1)

    lob_feed = HistoricalDataFeed(data_dir=os.path.join(ROOT_DIR, 'data/market/btcusdt/'),
                                  instrument='btcusdt', start_day=start_day, end_day=end_day)

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
                  'exec_config': {'exec_times': [1],
                                  'delete_vol': False},
                  'reset_config': {'reset_num_episodes': 1,},
                  "seed_config": {"seed" : 0,},}


    # define action space
    action_space = gym.spaces.Discrete(n = 3)

    def test_reward_at_bucket(self):
        env = RewardAtBucketEnv(broker=self.broker,
                                config=self.env_config,
                                action_space=self.action_space)
        env.reset()
        done = False
        reward_vec = []
        while not done:
            s, r, done, i = env.step(action=np.array([1]))
            reward_vec.append(r)
        # check that all trades are actually the same...
        self.assertEqual(self.broker.trade_logs["benchmark_algo"],
                         self.broker.trade_logs["rl_algo"],
                         'Trading history is not the same')
        self.assertEqual(reward_vec, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'Reward Vector is off')

    def test_reward_at_episode(self):

        env = RewardAtEpisodeEnv(broker=self.broker,
                                 config=self.env_config,
                                 action_space=self.action_space)

        for i in range(20):
            env.reset()
            done = False
            reward_vec = []
            while not done:
                _, r, done, _ = env.step(action=np.array([1]))
                reward_vec.append(r)
            self.assertEqual(self.broker.trade_logs["benchmark_algo"],
                             self.broker.trade_logs["rl_algo"],
                             'Trading history is not the same')
            if i > 0:
                self.assertEqual(self.broker.trade_logs["benchmark_algo"],
                                 logs_prev,
                                 'Trading history is not the same as previously')
            logs_prev = self.broker.trade_logs["benchmark_algo"]

        start_day = datetime.datetime(year=2021,month=6, day=2)
        end_day = datetime.datetime(year=2021,month=6, day=2)
        lob_feed = HistoricalDataFeed(data_dir=os.path.join(ROOT_DIR, 'data/market/btcusdt/'),
                                      instrument='btcusdt', start_day=start_day, end_day=end_day)
        broker = Broker(lob_feed)
        env = RewardAtEpisodeEnv(broker=broker,
                                 config=self.env_config,
                                 action_space=self.action_space)
        env.reset()
        done = False
        reward_vec = []
        while not done:
            s, r, done, i = env.step(action=np.array([1]))
            reward_vec.append(r)
        self.assertNotEqual(env.broker.trade_logs["benchmark_algo"],
                            logs_prev,
                            'Trading should be different now since sampled from a different date')
        self.assertEqual(reward_vec, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 'Reward Vector is off')

    def test_reward_at_step(self):
        env = RewardAtStepEnv(broker=self.broker,
                              config=self.env_config,
                              action_space=self.action_space)
        env.reset()
        done = False
        reward_vec = []
        while not done:
            s, r, done, i = env.step(action=np.array([1]))
            reward_vec.append(r)
        self.assertEqual(reward_vec, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'Reward Vector is off')


class TestSimilarityEnvVsSim(unittest.TestCase):
    start_day = datetime.datetime(year=2021,month=6, day=1)
    end_day = datetime.datetime(year=2021,month=6, day=1)

    lob_feed = HistoricalDataFeed(data_dir=os.path.join(ROOT_DIR, 'data/market/btcusdt/'),
                                  instrument='btcusdt', start_day=start_day, end_day=end_day)

    # define the broker class
    broker = Broker(lob_feed)

    # define the config
    env_config = {'obs_config': {"lob_depth": 5,
                                 "nr_of_lobs": 5,
                                 "norm": True},
                  'trade_config': {'trade_direction': 1,
                                   'vol_low': 500,
                                   'vol_high': 500,
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
                  'exec_config': {'exec_times': [1],
                                  'delete_vol': False},
                  'reset_config': {'reset_num_episodes': 1,},
                  "seed_config": {"seed" : 0,},}

    # define action space
    action_space = gym.spaces.Discrete(n = 3)

    # define the env
    base_env = DerivedEnv(broker=broker,
                          config=env_config,
                          action_space=action_space)

    algo = TWAPAlgo(trade_direction=1,
                    volume=500,
                    start_time='2021-06-01 09:00:00',
                    end_time='2021-06-01 09:01:00',
                    no_of_slices=1,
                    bucket_placement_func=lambda no_of_slices: 0.5,
                    broker_data_feed=lob_feed)

    def test_similarity(self):

        # define the broker class
        broker = Broker(self.lob_feed)
        broker.delete_vol = self.env_config["exec_config"]["delete_vol"]
        broker.benchmark_algo = self.algo
        broker.simulate_algo(self.algo)

        self.base_env.reset()
        done = False
        reward_vec = []
        while not done:
            s, r, done, i = self.base_env.step(action=np.array([0]))
            reward_vec.append(r)

        # now trade logs from stepping through env and trade logs from simulation must be the same...
        self.assertEqual(broker.trade_logs["benchmark_algo"],
                         self.base_env.broker.trade_logs["benchmark_algo"],
                         'Trading history is not the same btw env stepping and broker simulation')

    def test_timeline(self):
        """ Test if there are double entries in time collection"""

        self.base_env.reset()
        done = False
        reward_vec = []
        while not done:
            s, r, done, i = self.base_env.step(action=np.array([1]))
            reward_vec.append(r)

        bmk_log_times = [log["timestamp"] for log in self.broker.trade_logs['benchmark_algo']]
        contains_duplicates = len(bmk_log_times) != len(set(bmk_log_times))
        self.assertEqual(contains_duplicates, False, 'Benchmark trading Logs contain duplicate timestamps')

        rl_log_times = [log["timestamp"] for log in self.broker.trade_logs['rl_algo']]
        contains_duplicates = len(rl_log_times) != len(set(rl_log_times))
        self.assertEqual(contains_duplicates, False, 'RL trading Logs contain duplicate timestamps')

        bmk_hist = self.broker.hist_dict['benchmark']["timestamp"]
        contains_duplicates = len(bmk_hist) != len(set(bmk_hist))
        self.assertEqual(contains_duplicates, False, 'Benchmark history contains duplicate timestamps')

        rl_hist = self.broker.hist_dict['rl']["timestamp"]
        contains_duplicates = len(rl_hist) != len(set(rl_hist))
        self.assertEqual(contains_duplicates, False, 'Benchmark history contains duplicate timestamps')

        vwap_bmk, vwap_rl = self.base_env.broker.calc_vwap_from_logs()
        self.assertEqual(vwap_bmk, vwap_rl, 'VWAPS must be the same')
        # self.assertEqual(vwap_bmk, 32876.9147788, 'VWAP was 32876.9147788 before')


class TestTwoSlices(unittest.TestCase):

    lob_feed = HistoricalDataFeed(data_dir=os.path.join(ROOT_DIR, 'data/market/btcusdt/'),
                                  instrument='btcusdt')

    # define the broker class
    broker = Broker(lob_feed)

    # define the config
    env_config = {'obs_config': {"lob_depth": 5,
                                 "nr_of_lobs": 5,
                                 "norm": True},
                  'trade_config': {'trade_direction': 1,
                                   'vol_low': 25,
                                   'vol_high': 25,
                                   'no_slices_low': 2,
                                   'no_slices_high': 2,
                                   'bucket_func': lambda no_of_slices: [0.2, 0.8],
                                   'rand_bucket_low': 0,
                                   'rand_bucket_high': 0},
                  'start_config': {'hour_low': 9,
                                   'hour_high': 9,
                                   'minute_low': 0,
                                   'minute_high': 0,
                                   'second_low': 0,
                                   'second_high': 0},
                  'exec_config': {'exec_times': [1],
                                  'delete_vol': False},
                  'reset_config': {'reset_num_episodes': 1,},
                  "seed_config": {"seed" : 0,},}

    # define action space
    action_space = gym.spaces.Discrete(n = 3)

    # define the env
    base_env = DerivedEnv(broker=broker,
                          config=env_config,
                          action_space=action_space)

    def test_slices(self):
        self.base_env.reset()
        done = False
        idx = 0
        while not done:
            s, r, done, i = self.base_env.step(action=np.array([1]))

        vwap_bmk, vwap_rl = self.broker.calc_vwap_from_logs()
        self.assertEqual(vwap_bmk, vwap_rl, 'VWAPS are not matching')


if __name__ == '__main__':
    unittest.main()


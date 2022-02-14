import random
import gym
import math
import numpy as np
import copy

from datetime import datetime, timedelta
from gym.utils import seeding
from decimal import Decimal
from abc import ABC

from src.core.environment.limit_orders_setup.execution_algo_real import TWAPAlgo, RLAlgo
from src.ui.ui_comm import UIServer
from src.ui.ui_comm import Charts as Charts

DEFAULT_ENV_CONFIG = {'obs_config': {"lob_depth": 5,
                                     "nr_of_lobs": 5,
                                     "norm": True},
                      'trade_config': {'trade_direction': 1,
                                       'vol_low': 500,
                                       'vol_high': 1000,
                                       'no_slices_low': 5,
                                       'no_slices_high': 10,
                                       'bucket_func': lambda no_of_slices: (sorted([round(random.uniform(0, 1), 2) for _
                                                                                    in range(no_of_slices)])),
                                       'rand_bucket_low': 0,
                                       'rand_bucket_high': 0},
                      'start_config': {'hour_low': 1,
                                       'hour_high': 22,
                                       'minute_low': 0,
                                       'minute_high': 59,
                                       'second_low': 0,
                                       'second_high': 59},
                      'exec_config': {'exec_times': [5, 10, 15, 30, 60, 120, 240]},
                      'reset_config': {'reset_num_episodes': 1,
                                       'reset_data_feed': 20}}


def lob_to_numpy(lob, depth, norm_price=None, norm_vol_bid=None, norm_vol_ask=None):
    bid_prices = lob.bids.prices[-depth:]
    bid_volumes = [float(lob.bids.get_price_list(p).volume) for p in bid_prices]
    bid_prices = [float(bids) for bids in bid_prices]
    ask_prices = lob.asks.prices[:depth]
    ask_volumes = [float(lob.asks.get_price_list(p).volume) for p in ask_prices]
    ask_prices = [float(asks) for asks in ask_prices]

    if norm_price:
        # have to make sure bid_prices and ask_prices are lists
        prices = (np.array(bid_prices + ask_prices) / float(norm_price))
    else:
        prices = np.array(bid_prices + ask_prices)

    if norm_vol_bid and norm_vol_ask:
        volumes = np.concatenate((np.array(bid_volumes) / float(norm_vol_bid),
                                  np.array(ask_volumes) / float(norm_vol_ask)), axis=0)
    else:
        volumes = np.concatenate((np.array(bid_volumes),
                                  np.array(ask_volumes)), axis=0)
    return np.concatenate((prices, volumes), axis=0)


conv2date = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')


class BaseEnv(gym.Env, ABC):

    def __init__(self, broker, action_space, config={}):

        self.ui = None
        self.ui_epoch = 0

        self.broker = broker
        self.config = self.add_default_dict(config)
        self._validate_config()
        self.reset_counter = 0
        self.reset_counter_feed = 0
        # self.reset()
        self.build_observation_space()
        self.action_space = action_space
        self.seed()

        self.broker.mid_pxs = []

    def reset(self):
        """ Reset the environment """

        # Randomize inputs to the execution algo
        if self.reset_counter >= self.config['reset_config']['reset_num_episodes'] or self.reset_counter == 0:
            self.start_time, self.exec_time, self.volume, self.no_of_slices, \
            self.trade_dir, self.rand_bucket_bounds_width, self.bucket_func = self._reset_exec_params()
            self.reset_counter = 0
        self.reset_counter += 1

        hard_reset = False
        if self.reset_counter_feed >= self.config['reset_config']['reset_data_feed']:
            self.reset_counter_feed = 0
            hard_reset = True
        self.reset_counter_feed += 1

        # instantiate benchark algo
        self.broker.benchmark_algo = TWAPAlgo(trade_direction=self.trade_dir,
                                              volume=self.volume,
                                              no_of_slices=self.no_of_slices,
                                              bucket_placement_func=self.bucket_func,
                                              start_time=self.start_time,
                                              end_time=str((datetime.strptime(self.start_time, '%H:%M:%S') +
                                                            timedelta(minutes=self.exec_time)).time()),
                                              rand_bucket_bounds_width=self.rand_bucket_bounds_width,
                                              broker_data_feed=self.broker.data_feed)

        # reset the broker with the new benchmark_algo
        self.broker.reset(self.broker.benchmark_algo, hard=hard_reset)
        self.event_bmk, self.done_bmk, self.lob_bmk = self.broker.simulate_to_next_event(self.broker.benchmark_algo)

        # Declare the RLAlgo
        self.broker.rl_algo = RLAlgo(benchmark_algo=self.broker.benchmark_algo,
                                     trade_direction=self.broker.benchmark_algo.trade_direction,
                                     volume=self.broker.benchmark_algo.volume,
                                     no_of_slices=self.broker.benchmark_algo.no_of_slices,
                                     bucket_placement_func=self.broker.benchmark_algo.bucket_placement_func,
                                     broker_data_feed=self.broker.data_feed)
        self.broker.reset(self.broker.rl_algo)
        self.event_rl, self.done_rl, self.lob_rl = self.broker.simulate_to_next_event(self.broker.rl_algo)

        if self.event_rl['time'] != self.event_bmk['time']:
            raise ValueError("Benchmark and RL algo have events at different timestamps !!!")
        self.event_time = self.event_rl["time"]
        self.bucket_time = self.event_rl["time"]

        self.mid_pxs = []

        # To build the first observation we need to reset the datafeed to the timestamp of the first algo_event
        self.state = self._build_observation_at_event(event_time=self.broker.benchmark_algo.algo_events[0])

        self.reward = 0
        self.done = False
        self.info = {}

        self.ui_epoch += 1

        return self.state

    def _reset_exec_params(self):
        """ Used for defining random execution parameters. """

        start_time = '{}:{}:{}'.format(random.randint(self.config['start_config']['hour_low'],
                                                      self.config['start_config']['hour_high']),
                                       random.randint(self.config['start_config']['minute_low'],
                                                      self.config['start_config']['minute_high']),
                                       random.randint(self.config['start_config']['second_low'],
                                                      self.config['start_config']['second_high']))
        start_time = str(datetime.strptime(start_time, '%H:%M:%S').time())

        if isinstance(self.config['exec_config']['exec_times'], list) and \
                len(self.config['exec_config']['exec_times']) > 1:
            exec_time, volume, no_of_slices = self._reset_volume_and_slices()
        else:
            exec_time = self.config['exec_config']['exec_times']
            if isinstance(self.config['exec_config']['exec_times'], list):
                exec_time = exec_time[0]
            volume = random.randint(self.config['trade_config']['vol_low'],
                                    self.config['trade_config']['vol_high'])
            no_of_slices = random.randint(self.config['trade_config']['no_slices_low'],
                                          self.config['trade_config']['no_slices_high'])

        trade_dir = self.config['trade_config']['trade_direction']
        rand_bucket_bounds_width = random.randint(self.config['trade_config']['rand_bucket_low'],
                                                  self.config['trade_config']['rand_bucket_high'])
        bucket_func = self.config['trade_config']['bucket_func']

        return start_time, exec_time, volume, no_of_slices, trade_dir, rand_bucket_bounds_width, bucket_func

    def _reset_volume_and_slices(self):
        """ May deserve own method since its most important part of resetting/can be easily overridden. """

        exec_time = random.choice(self.config['exec_config']['exec_times'])
        perc = exec_time / (max(self.config['exec_config']['exec_times']) -
                            min(self.config['exec_config']['exec_times']))
        volume = round(self.config['trade_config']['vol_low'] + perc * (self.config['trade_config']['vol_high'] -
                                                                        self.config['trade_config']['vol_low']))
        no_of_slices = round(self.config['trade_config']['no_slices_low'] +
                             perc * (self.config['trade_config']['no_slices_high'] -
                                     self.config['trade_config']['no_slices_low']))
        return exec_time, volume, no_of_slices

    def _convert_action(self, action):
        """ Used if actions need to be transformed without having to change entire step() method """
        return action[0]

    def step(self, action):

        # check reset has been called
        assert self.done is False, 'reset() must be called before step()'

        # convert action if necessary
        action = self._convert_action(action)
        vol_to_trade = self.infer_volume_from_action(action)

        # simulate both benchmark and rl algo until before the next action is placed...
        self.event_time_prev = self.event_bmk['time']
        self.bucket_time_prev = self.bucket_time

        self.event_bmk, self.done_bmk, self.lob_bmk = self._step_algo(algo_type='benchmark')
        self.event_rl, self.done_rl, self.lob_rl = self._step_algo(algo_type='rl', volume=vol_to_trade)

        if self.done_bmk != self.done_rl:
            raise ValueError("Benchmark and RL algo have finished at different times !!!")
        if self.event_rl['time'] != self.event_bmk['time']:
            raise ValueError("Benchmark and RL algo have events at different timestamps !!!")

        self.event_time = conv2date(max(self.broker.trade_logs["benchmark_algo"][-1]["timestamp"],
                              self.broker.trade_logs["rl_algo"][-1]["timestamp"]))
        self.bucket_time = conv2date(max(self.bucket_time_bmk, self.bucket_time_rl))

        self.done = self.done_rl

        # get state and reward
        if self.done:
            self.state = []
        else:
            self.state = self._build_observation_at_event(event_time=
                                                          self.broker.benchmark_algo.algo_events[
                                                              self.broker.benchmark_algo.event_idx])
        self.reward = self.reward_func()
        self.info = {}

        return self.state, self.reward, self.done, self.info

    def _step_algo(self, algo_type, volume=None):

        if algo_type == 'benchmark':
            algo = self.broker.benchmark_algo
            event, done, lob = self.event_bmk, self.done_bmk, self.lob_bmk
        else:
            algo = self.broker.rl_algo
            event, done, lob = self.event_rl, self.done_rl, self.lob_rl

        _ = self.broker.place_next_order(algo, event, done, lob, volume)
        event, done, lob = self.broker.simulate_to_next_event(algo)

        if event['type'] == 'bucket_bound':
            done = self.broker.place_next_order(algo, event, done, lob)
            if not done:
                event, done, lob = self.broker.simulate_to_next_event(algo)
            if algo_type == 'benchmark':
                self.event_bmk_bucket = event
                self.bucket_time_bmk = self.broker.trade_logs["benchmark_algo"][-1]["timestamp"]
            else:
                self.event_rl_bucket = event
                self.bucket_time_rl = self.broker.trade_logs["rl_algo"][-1]["timestamp"]

        return event, done, lob

    def infer_volume_from_action(self, action):
        """ Logic for inferring the volume from the action placed in the env """

        vol_to_trade = Decimal(str(action)) * self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]
        factor = 10 ** (- self.broker.benchmark_algo.tick_size.as_tuple().exponent)
        vol_to_trade = Decimal(str(math.floor(vol_to_trade * factor) / factor))
        if vol_to_trade > self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]:
            vol_to_trade = self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]
        return vol_to_trade

    def build_observation_space(self):

        """
        Observation Space Config Parameters

        nr_of_lobs : int, Number of past snapshots to be concatenated to the latest snapshot
        lob_depth : int, Depth of the LOB to be in each snapshot (max lob_depth = 20 )
        norm : Boolean, normalize or not -- We take the strike price to normalize with as the middle of the bid/ask
        spread --
        """

        n_obs_onesided = self.config['obs_config']['lob_depth'] * self.config['obs_config']['nr_of_lobs']
        zeros = np.zeros(n_obs_onesided)
        ones = np.ones(n_obs_onesided)

        """
            The bounds are as follows (if we allow normalisation of past LOB snapshots by current LOB data):
                Inf > bids_price <= 0,
                Inf > asks_price > 0,
                Inf > bids_volume >= 0,
                Inf > asks_volume >= 0,
                benchmark_algo.volume >= remaining_vol_to_trade >= 0,
                no_of_slices >= remaining_orders >= 0
        """
        low = np.concatenate((zeros, zeros, zeros, zeros, np.array([0]), np.array([0])), axis=0)
        high = np.concatenate((ones * np.inf, ones * np.inf,
                               ones * np.inf, ones * np.inf,
                               np.array([np.inf]),
                               np.array([np.inf])), axis=0)

        obs_space_n = (n_obs_onesided * 4 + 2)
        assert low.shape[0] == high.shape[0] == obs_space_n
        self.observation_space = gym.spaces.Box(low=low,
                                                high=high,
                                                shape=(obs_space_n,),
                                                dtype=np.float64)

    def _build_observation_at_event(self, event_time):
        """ Helper to pass only copy of datafeed/make sure datafeed is only affected by broker class """

        feed = copy.copy(self.broker.data_feed)
        obs = self.build_observation(event_time, feed)
        return obs

    def build_observation(self, event_time, data_feed):
        # Build observation using the history of order book data / data generated by the RL algo

        data_feed.reset(time=event_time.strftime("%H:%M:%S.%f"))
        past_dts, past_lobs = data_feed.past_lob_snapshots(no_of_past_lobs=self.config['obs_config']['nr_of_lobs'])

        # check if we already have enough data collected in our
        lob_bools = [True if dt in self.broker.hist_dict['rl']['timestamp'] else False for dt in past_dts]
        lob_hist = []
        for idx in range(len(past_dts)):
            lob_hist.append(
                self.broker.hist_dict['rl']['lob'][self.broker.hist_dict['rl']['timestamp'].index(past_dts[idx])]
                if lob_bools[idx] else past_lobs[idx])

        obs = np.array([])
        if self.config['obs_config']['norm']:
            mid = (lob_hist[-1].get_best_ask() + lob_hist[-1].get_best_bid()) / 2
            self.mid_pxs.append(float(mid))
            for lob in lob_hist[-self.config['obs_config']['nr_of_lobs']:]:
                # TODO: Right now both prices and volumes are normalized, however we show the agent the unnormalized volume, wouldn't it make more sense to not normalize the volume so that it can find the relationship between those quantities?
                obs = np.concatenate((obs, lob_to_numpy(lob,
                                                        depth=self.config['obs_config']['lob_depth'],
                                                        norm_price=mid,
                                                        norm_vol_bid=None,
                                                        norm_vol_ask=None)), axis=0)
            obs = np.concatenate((obs,
                                  np.array([self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]]),
                                  # vol left to trade in the bucket
                                  np.array([self.broker.rl_algo.no_of_slices - self.broker.rl_algo.order_idx - 1])),
                                 axis=0)  # orders left to place in the bucket
        else:
            for lob in lob_hist[-self.config['obs_config']['nr_of_lobs']:]:
                obs = np.concatenate(obs, (lob_to_numpy(lob,
                                                        depth=self.config['obs_config']['lob_depth'])), axis=0)
            obs = np.concatenate((obs,
                                  np.array([self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]]),
                                  # vol left to trade in the bucket
                                  np.array([self.broker.rl_algo.no_of_slices - self.broker.rl_algo.order_idx - 1])),
                                 axis=0)  # orders left to place in the bucket

        # need to make sure that obs fits to the observation space...
        # 0 padding whenever this gets smaller...
        # NaN in the beginning if I don't have history yet...

        return np.array(obs, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_func(self):
        raise NotImplementedError

    def _validate_config(self):
        """ tests if all inputs are allowed """

        for k, v in self.config.items():
            if k not in DEFAULT_ENV_CONFIG.keys():
                raise ValueError("Config key '{0}' is not allowed !!!".format(k))
            for ky, val in self.config[k].items():
                if ky not in DEFAULT_ENV_CONFIG[k].keys():
                    raise ValueError("Config key '{0}' is not allowed !!!".format(ky))

        if self.config['obs_config']['lob_depth'] > 20:
            raise ValueError("'lob_depth' must be < 20")

        if self.config['trade_config']['vol_low'] > self.config['trade_config']['vol_high']:
            raise ValueError("'vol_low' must be larger than 'vol_high'")

        if self.config['trade_config']['no_slices_low'] > self.config['trade_config']['no_slices_high']:
            raise ValueError("'no_slices_low' must be larger than 'no_slices_high'")

        if self.config['trade_config']['rand_bucket_low'] > self.config['trade_config']['rand_bucket_high']:
            raise ValueError("'rand_bucket_low' must be larger than 'rand_bucket_high'")

        if self.config['start_config']['hour_low'] > self.config['start_config']['hour_high']:
            raise ValueError("'hour_high' must be larger than 'hour_low'")

        if self.config['start_config']['minute_low'] > self.config['start_config']['minute_high']:
            raise ValueError("'minute_high' must be larger than 'minute_low'")

        if self.config['start_config']['second_low'] > self.config['start_config']['second_high']:
            raise ValueError("'second_high' must be larger than 'second_low'")

    @staticmethod
    def add_default_dict(config):
        return {**DEFAULT_ENV_CONFIG, **config}

    def render(self, mode='human'):

        ##
        #### send ONLY plain python list with primitives (floats or ints) ####
        ##

        if self.ui is None:
            self.ui = UIServer()

        pxs = []
        for d in self.mid_pxs:
            pxs.append(float(d))

        lvls = []
        for _ in range(int(len(pxs) / 2)):
            lvls.append(float(self.broker.benchmark_algo.bmk_vwap))

        for _ in range(int(len(pxs) / 2)):
            lvls.append(float(self.broker.rl_algo.rl_vwap))

        timeseries_data = [
            {
                "event": "{}#{}".format(Charts.CHART_0, "0"),
                "data": pxs
            },
            {
                "event": "{}#{}".format(Charts.CHART_0, "1"),
                "data": lvls
            },
            {
                "event": "{}#{}".format(Charts.CHART_1, "0"),
                "data": lvls
            },
        ]

        self.ui.send_rendering_data(rl_session_epoch=self.ui_epoch,
                                    data=timeseries_data)


class ExampleEnvRewardAtStep(BaseEnv):
    def reward_func(self):
        """ Env with reward after each step """

        vwap_bmk, vwap_rl = self.broker.calc_vwap_from_logs(start_date=self.event_time_prev,
                                                            end_date=self.event_time)
        reward = 0
        if self.trade_dir == 1:
            if vwap_bmk > vwap_rl:
                reward = 1
        else:
            if vwap_bmk < vwap_rl:
                reward = 1
        return reward


class ExampleEnvRewardAtBucket(BaseEnv):
    def reward_func(self):
        """ Env with reward at end of each bucket """

        reward = 0
        if self.bucket_time != self.bucket_time_prev:
            vwap_bmk, vwap_rl = self.broker.calc_vwap_from_logs(start_date=self.bucket_time_prev,
                                                                end_date=self.bucket_time)
            if self.trade_dir == 1:
                if vwap_bmk > vwap_rl:
                    reward = 1
            else:
                if vwap_bmk < vwap_rl:
                    reward = 1
        return reward


class ExampleEnvRewardAtEpisode(BaseEnv):
    def reward_func(self):
        """ Env with sparse reward only at end of episode """

        reward = 0
        if self.done:
            vwap_bmk, vwap_rl = self.broker.calc_vwap_from_logs()
            if self.trade_dir == 1:
                if vwap_bmk > vwap_rl:
                    reward = 1
            else:
                if vwap_bmk < vwap_rl:
                    reward = 1
        return reward

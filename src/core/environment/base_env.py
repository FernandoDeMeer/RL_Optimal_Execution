import gym
import random
import numpy as np
from gym.utils import seeding
from decimal import Decimal
from abc import ABC, abstractmethod
from src.core.environment.analyzer import DataAnalyzer
from src.core.environment.broker import Broker


def lob_to_numpy(lob, depth, norm_price=None, norm_vol_bid=None, norm_vol_ask=None):
    bid_prices = lob.bids.prices[-depth:]
    bid_volumes = [float(lob.bids.get_price_list(p).volume) for p in bid_prices]
    bid_prices = [float(bids) for bids in bid_prices]
    ask_prices = lob.asks.prices[:depth]
    ask_volumes = [float(lob.asks.get_price_list(p).volume) for p in ask_prices]
    ask_prices = [float(asks) for asks in ask_prices]

    if norm_price:
        prices = np.array(bid_prices + ask_prices) / float(norm_price) #have to make sure bid_prices and ask_prices are lists
    else:
        prices = np.array(bid_prices + ask_prices)

    if norm_vol_bid and norm_vol_ask:
        volumes = np.concatenate((np.array(bid_volumes)/float(norm_vol_bid),
                            np.array(ask_volumes)/float(norm_vol_ask)), axis=0)
    else:
        volumes = np.concatenate((np.array(bid_volumes),
                                  np.array(ask_volumes)), axis=0)
    return np.concatenate((prices, volumes), axis=0)


class BaseEnv(gym.Env, ABC):

    def __init__(self,
                 data_feed,
                 trade_direction,
                 qty_to_trade,
                 max_step_range,
                 benchmark_algo,
                 obs_config,
                 action_space,
                 ):

        # object returning snapshots of limit order books
        self.data_feed = data_feed

        # Parameters of the execution algos
        self.trade_direction = trade_direction # 1 for buying, -1 for selling.
        self.qty_to_trade = qty_to_trade
        self.max_step_range = max_step_range

        self.data_analyzer = DataAnalyzer(["benchmark", "rl"])

        self.obs_config = obs_config
        self.reset()

        self.build_observation_space()

        benchmark_algo.set_base_parameters(trade_direction,
                                           qty_to_trade,
                                           self.max_steps)
        self.benchmark_algo = benchmark_algo
        self.broker = Broker()
        self.action_space = action_space

        self.seed()

    def reset(self):

        # reset the remaining quantitiy to trade and the time counter
        self.time = 0
        self.qty_remaining = self.qty_to_trade
        if isinstance(self.max_step_range, range):
            self.max_steps = random.sample(self.max_step_range, 1)[0]
        else:
            self.max_steps = self.max_step_range
        self.remaining_steps = self.max_steps

        # get a snapshot of the Limit Order Book.
        self.data_feed.reset(row_buffer=self.max_steps)
        _, lob_hist = self.data_feed.next_lob_snapshot()

        # store this in two separate histories (allows observation space to see past LOB-snapshots)
        self.lob_hist_rl = [lob_hist]
        self.lob_hist_bmk = [lob_hist]

        self.data_analyzer.reset()

        # build observation space
        self.state = self.build_observation()
        self.reward = 0
        self.done = False
        self.info = {}

        return self.state

    def step(self, action):

        assert self.done is False, (
            'reset() must be called before step()')

        place_order_rl = {'type': 'market',
                          'timestamp': self.time,
                          'side': 'bid' if self.trade_direction == 1 else 'ask',
                          'quantity': Decimal(str(action[0])),
                          'trade_id': 1}
        place_order_bmk = self.benchmark_algo.get_order_at_time(self.time)

        # place order in LOB and replace LOB history with current trade
        # since historic data can be incorporated into observations, "simulated" LOB's deviate from each other
        bmk_trade_dict = self.broker.place_order(self.lob_hist_bmk[-1], place_order_bmk)
        rl_trade_dict = self.broker.place_order(self.lob_hist_rl[-1], place_order_rl)

        # Update the analyzer
        self._record_step(bmk_trade_dict, rl_trade_dict)
        self.qty_remaining = self.qty_remaining - rl_trade_dict['qty']

        self.time += 1
        self.remaining_steps -= 1

        # incorporate sparse reward for now...
        self.reward = self.calc_reward()
        if self.time >= self.max_steps-1:
            self.done = True
            self.state = []
        else:
            _, lob_next = self.data_feed.next_lob_snapshot()
            self.lob_hist_bmk.append(lob_next)
            self.lob_hist_rl.append(lob_next)
            self.state = self.build_observation()

        self.info = {}
        return self.state, self.reward, self.done, self.info

    def build_observation_space(self):

        """
        Observation Space Config Parameters

        nr_of_lobs : int, Number of past snapshots to be concatenated to the latest snapshot
        lob_depth : int, Depth of the LOB to be in each snapshot
        norm : Boolean, normalize or not -- We take the strike price to normalize with as the middle of the bid/ask spread --
        """

        # TO-DO:
        # check if data_feed can provide the depth and nr_of_lobs required...

        n_obs_onesided = self.obs_config['lob_depth'] * \
                         self.obs_config['nr_of_lobs']
        zeros = np.zeros(n_obs_onesided)
        ones = np.ones(n_obs_onesided)

        """
            The bounds are as follows (if we allow normalisation of past LOB snapshots by current LOB data):
                Inf > bids_price <= 0,
                Inf > asks_price > 0,
                Inf > bids_volume >= 0,
                Inf > asks_volume >= 0,
                qty_to_trade >= remaining_qty_to_trade >= 0,
                max_steps >= remaining_time >= 0
        """
        low = np.concatenate((zeros, zeros, zeros, zeros, np.array([0]), np.array([0])), axis=0)
        high = np.concatenate((ones*np.inf, ones*np.inf,
                               ones*np.inf, ones*np.inf,
                               np.array([self.qty_to_trade]),
                               np.array([self.max_steps])), axis= 0)

        obs_space_n = (n_obs_onesided * 4 + 2)
        assert low.shape[0] == high.shape[0] == obs_space_n
        self.observation_space = gym.spaces.Box(low=low,
                                                high=high,
                                                shape=(obs_space_n,),
                                                dtype=np.float64)

    def build_observation(self):
        # Build observation using the history of order book data
        obs = np. array([])
        if self.obs_config['norm']:
            # normalize...
            mid = (self.lob_hist_rl[-1].get_best_ask() +
                   self.lob_hist_rl[-1].get_best_bid()) / 2
            vol_bid = self.lob_hist_rl[-1].bids.volume
            vol_ask = self.lob_hist_rl[-1].asks.volume
            for lob in self.lob_hist_rl[-self.obs_config['nr_of_lobs']:]:
                obs = np.concatenate((obs, lob_to_numpy(lob,
                                                   depth=self.obs_config['lob_depth'],
                                                   norm_price=mid,
                                                   norm_vol_bid=vol_bid,
                                                   norm_vol_ask=vol_ask)), axis=0)
        else:
            for lob in self.lob_hist_rl[-self.obs_config['nr_of_lobs']:]:
                obs = np.concatenate(obs, (lob_to_numpy(lob,
                                                        depth=self.obs_config['lob_depth'])), axis=0)
        obs = np.concatenate((obs, np.array([self.qty_remaining]), np.array([self.remaining_steps])), axis=0)

        # need to make sure that obs fits to the observation space...
        # 0 padding whenever this gets smaller...
        # NaN in the beginning if I don't have history yet...

        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def calc_reward(self):
        reward = 0
        if self.time >= self.max_steps-1:
            vwaps = self.data_analyzer.calc_vwaps()
            if (vwaps['rl'] - vwaps['benchmark']) * self.trade_direction < 0:
                reward = 1
        return reward

    def _record_step(self, bmk, rl):

        # update the volume of the benchmark algo
        self.benchmark_algo.update_remaining_volume(bmk['qty']) # this is odd to do here...

        # update the analyzer
        self.data_analyzer.record_step(algo_id="benchmark", key_name="pxs", value=bmk['pxs'])
        self.data_analyzer.record_step(algo_id="benchmark", key_name="qty", value=bmk['qty'])

        self.data_analyzer.record_step(algo_id="rl", key_name="pxs", value=rl['pxs'])
        self.data_analyzer.record_step(algo_id="rl", key_name="qty", value=rl['qty'])

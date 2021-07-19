import gym
import numpy as np
from gym.utils import seeding
from src.core.environment.orderbook import OrderBook
from src.core.environment.lob_simulator import LOBObserver, split_book_to_orders
from decimal import Decimal


class LimitOrderBookEnv(gym.Env):

    def __init__ (self,
                  data_directory,
                  time_steps,
                  trade_direction,
                  qty_to_trade,
                  benchmark,
                  broker,
                  past_n=10):

        self.data_directory = data_directory
        self.time_steps = time_steps
        self.observer = LOBObserver(file_name=self.data_directory, past_n=past_n, future_n=time_steps)
        self.benchmark = benchmark
        self.broker = broker
        self.trade_direction = trade_direction
        self.qty_to_trade = qty_to_trade

        # we choose different numbers of shares to trade, range of actions from 0 to 2TWAP (the TWAP
        # orders are able to consume at least the 2nd best bid price on average), the decision to make is the number
        # of shares to trade at time T, so we need a Box environment of 1 dim
        self.action_space = gym.spaces.Box(low=0.0,
                                           high=2 * self.benchmark.volume_at_time,
                                           shape= (1,),
                                           dtype=np.float32)

        # Assume observation space takes the first n bid/ask orders as they are
        self.reset()

        # Assume observation of entire LOB plus remaining volume and current TWAP

        n_bids = len(self.observer.fut_orders[0]["b"])
        n_asks = len(self.observer.fut_orders[0]["a"])
        obs_space_n = n_bids * 2+ n_asks * 2 + 2
        # self.observation_space = gym.spaces.Discrete(obs_space_n)
        low = np.append(np.zeros(obs_space_n-2), [-np.inf, 0], axis=0)
        self.observation_space = gym.spaces.Box(low=low,
                                                high=np.ones(obs_space_n) * np.inf,
                                                shape=(obs_space_n,),
                                                dtype=np.float32)
        self.seed()

    def step(self, action):
        """
        Updates the environment given an action.
        """
        assert self.done == False, (
            'reset() must be called before step()')

        observed_lob = self.observer.fut_orders[self.time]

        if self.trade_direction == 1:
            side = 'bid'
        else:
            side = 'ask'

        place_order_rl = {'type': 'market',
                          'timestamp': self.time,
                          'side': side,
                          'quantity': Decimal(float(action[0])),
                          'trade_id' : 1}
        place_order_bmk = self.benchmark.get_order_at_time(self.time)
        _, vol_1, _, _ = self.broker.place_orders(self.order_book, place_order_rl, place_order_bmk)
        self.qty_remaining = self.qty_remaining - vol_1
        self.time += 1
        self.state = self._flatten_observations(self.time)

        # incorporate sparse reward for now
        if self.time >= self.time_steps-1:
            self.done = True
            twap_1, twap_2 = self.broker.get_twaps()
            if (twap_1 - twap_2) * self.trade_direction < 0:
                self.reward = 1
        else:
            self._order_book_step() # update order book to new orders
        self.info = {}
        return self.state, self.reward, self.done, self.info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """

        # read new data for the Limit-Order-Book
        self.time = 0
        self.qty_remaining = self.qty_to_trade
        self.observer.draw_random_observations()
        self._order_book_step()
        self.state = self._flatten_observations(0)
        self.reward = 0
        self.done = False
        self.info = {}

        return self.state

    def seed(self, seed=None):
        """
        Sets the seed for the environments random number generator.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _order_book_step(self):
        self.order_book = OrderBook()
        _, _, all_orders = split_book_to_orders(self.observer.fut_orders[self.time])
        for order in all_orders:
            _, _ = self.order_book.process_order(order, False, False)
        return self.order_book

    def _flatten_observations(self, t):
        raw_obs = self.observer.fut_orders[t]
        obs = []
        for bid_idx in range(len(raw_obs['b'])):
            obs.append(float(raw_obs['b'][bid_idx][0]))
            obs.append(float(raw_obs['b'][bid_idx][1]))
        for ask_idx in range(len(raw_obs['a'])):
            obs.append(float(raw_obs['a'][ask_idx][0]))
            obs.append(float(raw_obs['a'][ask_idx][1]))
        twap_1, twap_2 = self.broker.get_twaps()
        obs.append((twap_1 - twap_2) * self.trade_direction)
        obs.append(self.qty_remaining)
        return np.asarray(obs, dtype=np.float32)

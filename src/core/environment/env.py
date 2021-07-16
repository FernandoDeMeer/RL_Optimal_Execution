import gym
import numpy as np
from gym.utils import seeding
from env.orderbook import OrderBook
from decimal import Decimal

"""
TODO:
    - add the dynamic number of lobs in the state shape
    -
"""

class LimitOrderBookEnv(gym.Env):

    def __init__(self, data_feed, T_max, nr_of_lobs, lob_depth, qty_to_trade, trade_direction, timesteps_per_episode):

        # todo: delete: self.observer = LOBObserver(file_name=self.data_directory, past_n=past_n, future_n=T_Max)

        self.gui = None

        self.T_max = T_max
        self.trade_direction = trade_direction
        self.qty_to_trade = qty_to_trade
        self.timesteps_per_episode = timesteps_per_episode

        self.data_feed = data_feed
        self.matching_order_engine = OrderBook()

        # we choose different numbers of shares to trade, range of actions from 0 to 2TWAP (the TWAP
        # orders are able to consume at least the 2nd best bid price on average), the decision to make is the number
        # of shares to trade at time T, so we need a Box environment of 1 dim
        self.action_space = gym.spaces.Box(0.0, 2 * self.TWAP, shape=1, dtype=np.float32)

        #We need the agent to see the n best Ask/Bid orders price + Volume, remaining shares_to_trade and
        #the time left before the current episode ends.

        self.nr_of_lobs = nr_of_lobs
        self.lob_depth = lob_depth

        # todo: delete: obs_space_n = len(self.observer.fut_orders[0]["b"])+len(self.observer.fut_orders[0]["a"]) + 2
        # 2=sides; 2=qty_to_trade and time left.
        obs_space_n = self.nr_of_lobs * self.lob_depth * 2 + 2

        # todo: delete: self.observation_space = gym.spaces.Discrete(obs_space_n)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs_space_n, dtype=np.float32)

        self.twap = 0
        self.trade_id_idx = 0
        self.state = None
        self.total_timesteps = 0
        self.remaining_timesteps = self.timesteps_per_episode
        self.remaining_qty_to_trade = self.qty_to_trade

        self.reward = 0
        self.done = False
        self.info = {}

        self.reset()
        self.seed()

    def step(self, action, extra_data={}):

        assert self.done == False, ('reset() must be called before step()')

        self.twap = self._calc_twap()

        self._execute_action_on_engine(action[0])

        self._calc_reward()

        self.state = self._build_state()

        self.total_timesteps += 1
        self.remaining_timesteps -= 1

        if self.total_timesteps >= self.T_max - 1:
            self.done = True
        self.info = {}

        return self.state, self.reward, self.done, self.info

    def _execute_action_on_engine(self, signal):

        place_order = {'type': 'market',
                       'timestamp': self.total_timesteps,
                       'side': self.trade_direction,
                       'quantity': Decimal(signal),
                       # todo: i suppose we don't need to specify the price if markt order is used.
                       # 'price': Decimal(13000),
                       'trade_id': self._get_next_trade_id()}

        self.matching_order_engine.process_order(place_order, False, False)

    def _calc_twap(self):
        return 0

    def _calc_reward(self):
        return 0

    def _build_state(self):

        next_lob = self.data_feed.next_lob_snapshot(None)

        next_lob = np.concatenate((next_lob, [self.remaining_qty_to_trade, self.remaining_timesteps]))

        # todo: normalize the merged lob?
        return self._normalize_state(next_lob)

    def _normalize_state(self):
        return []

    def reset(self):
        """
            Resets the state of the environment and returns an initial observation.
        """
        # self.observer.draw_random_observations() todo: do we need this?

        self.twap = 0
        self.trade_id_idx = 0
        self.total_timesteps = 0
        self.remaining_timesteps = self.timesteps_per_episode
        self.remaining_qty_to_trade = self.qty_to_trade
        self.reward = 0
        self.done = False
        self.info = {}

        self.state = self._build_state()

        return self.state

    def seed(self, seed=None):
        """
        Sets the seed for the environments random number generator.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_next_trade_id(self):

        self.trade_id_idx += 1
        return self.trade_id_idx

    def register_gui(self, gui):
        self.gui = gui

    """
    TODO: move this logic into the OrderBook()
    def _get_mutated_order_book(self):
            for key, value in self.matching_order_engine.asks.price_map.items()[self.lob_depth: ]:
            for key, value in reversed(self.bids.price_map.items())[self.lob_depth: ]:
    """







# if __name__ == '__main__':
#
#     lob_env = LimitOrderBookEnv(data_directory="./data/book_depth_socket_btcusdt_2021_06_21.txt",
#                                 T_Max=50)
#     for t in range(100):
#         action = lob_env.action_space.sample()
#         observation, reward, done, info = lob_env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
#     lob_env.close()

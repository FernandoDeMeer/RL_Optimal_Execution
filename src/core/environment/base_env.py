import gym
import numpy as np
from gym.utils import seeding
from env.orderbook import OrderBook
from decimal import Decimal
from abc import ABC, abstractmethod


class BaseEnv(gym.Env, ABC):

    def __init__(self,
                 data_feed,
                 max_steps,
                 trade_direction,
                 qty_to_trade,
                 steps_per_episode,
                 nr_of_lobs,
                 lob_depth,
                 action_space
                 ):

        self.matching_order_engine = OrderBook()

        self.data_feed = data_feed
        self.gui = None

        self.max_steps = max_steps
        self.trade_direction = trade_direction # 1 for buying, -1 for selling.
        self.qty_to_trade = qty_to_trade
        self.steps_per_episode = steps_per_episode

        self.remaining_qty_to_trade = 0
        self.remaining_steps = 0

        self.nr_of_lobs = nr_of_lobs
        self.lob_depth = lob_depth

        # self.current_total_steps = 0

        self.twap = 0
        self.price = 0
        self.state = None
        self.done = False

        # the ... + 2 represents: remaining_qty_to_trade and remaining_steps_per_episode
        obs_space_n = self.nr_of_lobs * self.lob_depth * 4 + 2

        """
        If the obs_space is normalized the bounds are as follows: 
        
            asks_volume > 0
            asks_price >= 1
            bids_volume > 0
            bids_price <= 1
            remaining_qty_to_trade > 0
            remaining_time > 0 
        """

        """
            Observation Space Config Parameters
            
            nr_of_lobs : int, Number of past snapshots to be concatenated to the latest snapshot
            lob_depth : int, Depth of the LOB to be in each snapshot 
            norm : Boolean, normalize or not -- We take the strike price to normalize with as the middle of the bid/ask spread --
            
        """

        zeros = np.zeros(self.lob_depth)
        ones = np.ones(self.lob_depth)

        low = np.concatenate((zeros,ones,
                              zeros,zeros,0,0),axis=0)

        high = np.concatenate((ones*np.inf,ones*np.inf,
                               ones*np.inf,ones,self.qty_to_trade,
                               self.steps_per_episode), axis= 0)

        assert low.shape == high.shape == obs_space_n
        self.observation_space = gym.spaces.Box(low=low,
                                                high=high,
                                                shape=obs_space_n,
                                                dtype=np.float32)

        # TODO: when is the self.twap calculated? cuz' here is zero and then... i set the range like below. Fernando: I think the easiest is to update it at every step()

        self.action_space = action_space

        self.reset()
        self.seed()

    def reset(self):

        # self.observer.draw_random_observations() todo: do we need this? Fernando: Depends, if we're going to train online only then we don't, but if we're going to do offline (i.e. bootstrapping then yes)

        self.matching_order_engine = OrderBook()

        self.remaining_qty_to_trade = self.qty_to_trade
        self.remaining_steps = self.steps_per_episode

        self.twap = 0
        self.state = self.build_observation()
        self.done = False

    def step(self, action):
        assert self.done == False, ('reset() must be called before step()')

        self.remaining_qty_to_trade = self.execute_action_on_engine(action)
        self.state = self.build_observation()
        self.twap = self.calc_twap()

        # self.current_total_steps += 1
        self.remaining_steps -= 1

        reward = 0.
        # if self.current_total_steps >= self.T_max - 1:
        if self.remaining_steps == 0:
            reward = self.calc_reward()

            self.done = True

        return self.state, reward, self.done, {}

    def build_observation(self):

        next_lob = self.data_feed.next_lob_snapshot(None)
        self.sync_lob_2_engine(next_lob)

        lob = self._normalize_lob(next_lob)

        observation = np.concatenate((lob, [self.remaining_qty_to_trade, self.remaining_steps]))

        return observation

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def calc_twap(self):
        # TODO: Need to access here the current LOB (before the action is taken), the twap and the total number of steps per episode.
        if self.trade_direction == 1:
            # We are buying, so the twap is updated according to the best ask prices
            self.twap = self.twap + self.state[0]/self.steps_per_episode
        elif self.trade_direction == -1:
            self.twap = self.twap + self.state[2*self.lob_depth]/self.steps_per_episode

        return self.twap


    @abstractmethod
    def execute_action_on_engine(self, action):
        pass

    def calc_reward(self):
        reward = 0
        if self.trade_direction == 1:
            # We are buying, so we want to have a lower price than the twap
            if self.twap < self.price:
                reward = 1
        elif self.trade_direction == -1:
            # We are selling, so we want to have a lower price than the twap
            if self.twap > self.price:
                reward = 1

        return reward


    @abstractmethod
    def sync_lob_2_engine(self, lob):
        pass

    @abstractmethod
    def normalize_lob(self, lob):
        pass

    def register_gui(self, gui):
        self.gui = gui

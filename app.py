from src.gui.graph_user_interface import GraphUserInterface
from src.core.environment.env import HistoricalOrderBookEnv
from src.data.data_feed import HISTORICAL_DATA_FEED, GAN_LOB_DATA_FEED
from src.data.historical_data_feed import HistoricalDataFeed

import gym
import numpy as np


class RLOptimalTradeExecutionApp:

    def __init__(self, params):

        self.params = params

        self.data_feed = self._init_data_feed(params["data_feed"])
        self.obs_space_config = self.get_obs_space_config()
        self.action_space = self.get_action_space()
        self.environment = HistoricalOrderBookEnv(data_feed=self.data_feed,
                                                  max_steps = params["max_steps"],
                                                  trade_direction = params["trade_direction"],
                                                  qty_to_trade = params["qty_to_trade"],
                                                  steps_per_episode= params[""],
                                                  obs_space_config=self.obs_space_config,
                                                  action_space= params[""],
                                                  burn_in_period= params[""]
                                                  )

        if params["visualize"]:
            self.gui = GraphUserInterface()
            self.environment.register_gui(self.gui)

    def _init_data_feed(self, data_feed_type):

        if data_feed_type == HISTORICAL_DATA_FEED:

            return HistoricalDataFeed(data_dir=self.params["data_dir"],
                                      instrument='btcusdt',
                                      lob_depth=self.params["lob_depth"],
                                      start_day=None,
                                      end_day=None)
        elif data_feed_type == GAN_LOB_DATA_FEED:
            ## init GAN-LOB and train before, if necessary.
            return None

    def run(self):
        # TODO the action space will be declared here along with the agent
        if self.params["train"]:
            self._train()
        else:
            # Load the already trained agent
            pass

        if self.params["test"]:
            self._test()

    def _train(self):
        pass

    def _test(self):
        pass

    def get_obs_space_config(self,):
        """Hyperparameters of the Observation Space"""

        obs_space_config = {}

        obs_space_config["nr_of_lobs"] = self.params["nr_of_lobs"]
        obs_space_config["lob_depth"] = self.params["lob_depth"]
        obs_space_config["norm"] = self.params["norm"]

        return obs_space_config

    def get_action_space(self,):

        if self.params["action_space"] == "Box":

            action_space = gym.spaces.Box(0.0, 1,
                                               shape=1,
                                               dtype=np.float32)
        elif self.params["action_space"] == "Discrete":

            action_space = gym.spaces.Discrete(shape=50,
                                                    dtype=np.float32)
        return action_space

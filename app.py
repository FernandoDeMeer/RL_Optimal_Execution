from src.gui.graph_user_interface import GraphUserInterface
from src.core.environment.env import LimitOrderBookEnv
from src.data.data_feed import HISTORICAL_DATA_FEED, GAN_LOB_DATA_FEED
from src.data.historical_data_feed import HistoricalDataFeed


class RLOptimalTradeExecutionApp:

    def __init__(self, params):

        self.params = params

        self.data_feed = self._init_data_feed(params["data_feed"])
        self.environment = LimitOrderBookEnv(data_feed=self.data_feed,
                                             T_max=0,
                                             nr_of_lobs=params["nr_of_lobs"],
                                             lob_depth=params["lob_depth"],
                                             qty_to_trade=params["qty_to_trade"],
                                             trade_direction=params["trade_direction"],
                                             timesteps_per_episode=params["timesteps_per_episode"],
                                             )

        if params["visualize"]:
            self.gui = GraphUserInterface()
            self.environment.register_gui(self.gui)

    def _init_data_feed(self, data_feed_type):

        if data_feed_type == HISTORICAL_DATA_FEED:
            data_feed = HistoricalDataFeed()
            # ... .
            return data_feed
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

from decimal import Decimal
from src.core.environment.base_env import BaseEnv


"""
TODO: 
    - add the dynamic number of lobs in the state shape.
    - 
"""


class HistoricalOrderBookEnv(BaseEnv):

    def __init__(self, data_feed, max_steps, trade_direction, qty_to_trade,
                   steps_per_episode, obs_space_config, action_space, burn_in_period):

        super().__init__(data_feed, max_steps, trade_direction, qty_to_trade,
                         steps_per_episode,obs_space_config,action_space,)


    def execute_action_on_engine(self, signal):

        """
        BookOrder has some nice methods to get the current quantities... so we can get the cumulative quantity..
        """
        place_order = {'type': 'market',
                       'timestamp': self.total_timesteps,
                       'side': self.trade_direction,
                       'quantity': Decimal(signal),
                       'trade_id': self._get_next_trade_id()}

        self.matching_order_engine.process_order(place_order, False, False)

        delta_fill_quantity = 0.

        return self.remaining_qty_to_trade - delta_fill_quantity

    def sync_lob_2_engine(self, lob):
        """
            use OrderBook() methods to cancel/add/update levels.
        """
        pass


    """
    TODO: later on for GAN version, move this logic into the OrderBook():
    def _get_mutated_order_book(self):
            for key, value in self.matching_order_engine.asks.price_map.items()[self.lob_depth: ]:
                ...
            for key, value in reversed(self.bids.price_map.items())[self.lob_depth: ]:
                ...
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

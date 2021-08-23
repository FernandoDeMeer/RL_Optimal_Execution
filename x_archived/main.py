# import sys
# import os
# import argparse
# # from app import RLOptimalTradeExecutionApp
#
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--data_dir',
#                     default='data_dir/',
#                     help="Data location.",
#                     type=str)
#
# parser.add_argument('--data_feed',
#                     default='historical',
#                     choices=['historical',
#                              'gan-lob',
#                              ],
#                     help="Data Feed: 'historical' or 'gan-lob'.",
#                     type=str)
#
# parser.add_argument('--nr_of_lobs',
#                     default=4,
#                     help="Train the agent",
#                     type=int)
#
# parser.add_argument('--nr_of_lob_levels',
#                     default=20,
#                     help="Train the agent",
#                     type=int)
#
# parser.add_argument('--train',
#                     default=True,
#                     help="Train the agent.",
#                     type=bool)
#
# parser.add_argument('--test',
#                     default=True,
#                     help="Test the agent.",
#                     type=bool)
#
# parser.add_argument('--timesteps_per_episode',
#                     default=100000,
#                     help="Train the agent",
#                     type=int)
#
# parser.add_argument('--total_timesteps',
#                     default=100000,
#                     help="Train the agent",
#                     type=int)
#
# parser.add_argument('--reward_type',
#                     default='realized_pnl',
#                     choices=['realized_pnl',
#                              'trade_completion',
#                              ],
#                     help="",
#                     type=str)
#
# parser.add_argument('--visualize',
#                     default=True,
#                     help="Show PyQT interface",
#                     type=bool)
#
# params = vars(parser.parse_args())
#
#
# if __name__ == '__main__':
#     # from PyQt5.QtWidgets import QApplication
#     # from src.gui.graph_user_interface import GraphUserInterface
#
#     # q_app = QApplication(sys.argv)
#
#     # app = RLOptimalTradeExecutionApp(params)
#     # app.run()
#
#     # user_interface = GraphUserInterface()
#     # user_interface.show()
#     #
#     # import numpy as np
#     # user_interface.update_data([
#     #     {
#     #         "event": "{}#{}".format(GraphUserInterface.CHART_0, "0"),
#     #         "data": np.random.rand(1, 200).flatten()
#     #     },
#     #     {
#     #         "event": "{}#{}".format(GraphUserInterface.CHART_0, "1"),
#     #         "data": np.random.rand(1, 200).cumsum()
#     #     },
#     #     {
#     #         "event": "{}#{}".format(GraphUserInterface.CHART_1, "0"),
#     #         "data": np.random.rand(1, 200).flatten()
#     #     },
#     # ])
#
#     # sys.exit(q_app.exec_())

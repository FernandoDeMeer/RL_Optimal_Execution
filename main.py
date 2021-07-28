import argparse

from app import RLOptimalTradeExecutionApp


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir',
                    default='data_dir/',
                    help="Data location.",
                    type=str)

parser.add_argument('--data_feed',
                    default='historical',
                    choices=['historical',
                             'gan-lob',
                             ],
                    help="Data Feed: 'historical' or 'gan-lob'.",
                    type=str)

parser.add_argument('--nr_of_lobs',
                    default=4,
                    help="Train the agent",
                    type=int)

parser.add_argument('--nr_of_lob_levels',
                    default=20,
                    help="Train the agent",
                    type=int)

parser.add_argument('--train',
                    default=True,
                    help="Train the agent.",
                    type=bool)

parser.add_argument('--test',
                    default=True,
                    help="Test the agent.",
                    type=bool)

parser.add_argument('--timesteps_per_episode',
                    default=100000,
                    help="Train the agent",
                    type=int)

parser.add_argument('--total_timesteps',
                    default=100000,
                    help="Train the agent",
                    type=int)

parser.add_argument('--reward_type',
                    default='realized_pnl',
                    choices=['realized_pnl',
                             'trade_completion',
                             ],
                    help="",
                    type=str)

parser.add_argument('--visualize',
                    default=False,
                    help="Show PyQT interface",
                    type=bool)

params = vars(parser.parse_args())


if __name__ == '__main__':
    ## global QT GUI init

    app = RLOptimalTradeExecutionApp(params)
    app.run()

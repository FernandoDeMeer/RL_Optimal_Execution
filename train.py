#
#   Reinforcement Learning Optimal Trade Execution
#

import os
import argparse

import gym
import random
import numpy as np

import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer

from src.data.historical_data_feed import HistoricalDataFeed
from src.core.environment.limit_orders_setup.broker_real import Broker
from src.core.environment.limit_orders_setup.base_env_real import BaseEnv


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def init_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="lob_env")
    parser.add_argument("--num-cpus", type=int, default=1)

    parser.add_argument(
        "--framework",
        choices=["tf", "torch"],
        default="torch",
        help="The DL framework specifier.")

    parser.add_argument(
        "--symbol",
        choices=["btc_usdt"],
        default="btc_usdt",
        help="Market symbol.")

    parser.add_argument(
        "--stop-iters",
        type=int,
        default=100,
        help="Number of iterations to train.")

    parser.add_argument(
        "--stop-reward",
        type=float,
        default=90.0,
        help="Reward at which we stop training.")

    return parser.parse_args()


def lob_env_creator(env_config):

    lob_feed = HistoricalDataFeed(data_dir=os.path.join(DATA_DIR, "market", env_config["symbol"]),
                                  instrument=env_config["symbol"],
                                  samples_per_file=200)

    broker = Broker(lob_feed)

    observation_space_config = {"lob_depth": 5, "nr_of_lobs": 5, "norm": True}
    action_space = gym.spaces.Box(low=0.0,
                                  high=1.0,
                                  shape=(1,),
                                  dtype=np.float32)

    return BaseEnv(show_ui=False,
                   broker=broker,
                   obs_config=observation_space_config,
                   action_space=action_space)


def dummy_test_env():

    seed = 33865
    np.random.seed(seed)
    random.seed(seed)

    env = lob_env_creator({"symbol": "btc_usdt"})
    obs = env.reset()
    print(obs.shape)

    while True:
        act = env.action_space.sample()
        # print(act)
        obs, reward, done, _ = env.step(act)

        if done:
            obs = env.reset()
            print("rezet")

        if not env.observation_space.contains(obs):
            print("houston, probelmz")


if __name__ == "__main__":

    args = init_arg_parser()

    # dummy_test_env()

    # For debugging the env or other modules, set local_mode=True
    ray.init(local_mode=True, num_cpus=args.num_cpus)
    register_env("lob_env", lob_env_creator)

    config = {
        "env": args.env,
        "num_workers": args.num_cpus,
        "num_envs_per_worker": 1,

        "gamma": 0.9,
        "entropy_coeff": 0.001,
        "vf_loss_coeff": 1e-5,

        "model": {
            "use_lstm": True,
            "lstm_cell_size": 10,
            # "max_seq_len": 20,
        },

        "framework": args.framework,
        "sgd_minibatch_size": 32,
        "train_batch_size": 128,
        # "log_level": "INFO",
        "evaluation_interval": 1,

        "env_config": {
            "symbol": args.symbol
        }
    }

    trainer = PPOTrainer(config)

    for _ in range(100):
        result = trainer.train()
        print(result)

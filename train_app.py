#
#   Reinforcement Learning Optimal Trade Execution
#

import os
import time
import datetime
import argparse

import gym
import json
import numpy as np

import ray
from ray import tune
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
        default="btcusdt",
        help="Market symbol.")

    parser.add_argument(
        "--session_id",
        type=str,
        default="0",
        help="Session id.")

    parser.add_argument(
        "--nr_episodes",
        type=int,
        default=1000,
        help="Number of episodes to train.")

    return parser.parse_args()


def lob_env_creator(env_config):

    is_env_eval = env_config.num_workers == 0

    if is_env_eval:
        data_periods = env_config["eval_data_periods"]
    else:
        data_periods = env_config["train_data_periods"]

    data_start_day = datetime.datetime(year=data_periods[0], month=data_periods[1], day=data_periods[2])
    data_end_day = datetime.datetime(year=data_periods[3], month=data_periods[4], day=data_periods[5])

    lob_feed = HistoricalDataFeed(data_dir=os.path.join(DATA_DIR, "market", env_config["symbol"]),
                                  instrument=env_config["symbol"],
                                  start_day=data_start_day,
                                  end_day=data_end_day,
                                  samples_per_file=200)

    broker = Broker(lob_feed)

    observation_space_config = {'obs_config': {'lob_depth': 5,
                                               'nr_of_lobs': 5,
                                               'norm': True}}
    action_space = gym.spaces.Box(low=0.0,
                                  high=1.0,
                                  shape=(1,),
                                  dtype=np.float32)

    return BaseEnv(broker=broker,
                   config=observation_space_config,
                   action_space=action_space)


def init_session_container(session_id):

    if args.session_id == "0":
        session_id = str(int(time.time()))

    session_container_path = os.path.join("data", "sessions", session_id)

    if not os.path.isdir(session_container_path):
        os.makedirs(session_container_path)

    return session_container_path


def test_agent_one_episode(config, agent_path, eval_data_periods, symbol):

    agent = PPOTrainer(config=config)
    agent.restore(agent_path)

    class DummyCfg:
        def __init__(self):
            self.num_workers = 0

        def __getitem__(self, key):

            if key == "eval_data_periods":
                return eval_data_periods

            elif key == "symbol":
                return symbol

    env = lob_env_creator(DummyCfg())

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    return episode_reward


if __name__ == "__main__":

    args = init_arg_parser()

    # For debugging the env or other modules, set local_mode=True
    ray.init(local_mode=True, num_cpus=args.num_cpus)
    register_env("lob_env", lob_env_creator)

    config = {
        "env": args.env,
        "framework": args.framework,
        # Number of GPUs to allocate to the trainer process. Note that not all
        # algorithms can take advantage of trainer GPUs. This can be fractional
        # (e.g., 0.3 GPUs).
        # "num_gpus": 0,
        "num_workers": args.num_cpus,
        "num_envs_per_worker": 1,
        # "rollout_fragment_length": 100,
        "train_batch_size": 64,
        "sgd_minibatch_size": 32,
        # "num_sgd_iter": 10,
        "entropy_coeff": 0.01,
        # "lr_schedule": [
        #     [0, 0.0005],
        #     [10000000, 0.000000000001],
        # ],

        # Eval
        "evaluation_interval": 10,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 1,
        "evaluation_config": {
            "explore": False,
            "render_env": True,
        },
        "lambda": 0.95,
        "kl_coeff": 0.5,
        "clip_param": 0.1,
        "vf_share_layers": False,
        "model": {
            "use_lstm": True,
            # TODO: change lstm_cell_size according to machine running: (local) laptop/dgx
            "lstm_cell_size": 128,
            # "lstm_use_prev_action": False,
            # # Whether to feed r_{t-1} to LSTM.
            # "lstm_use_prev_reward": False,
        },
        "env_config": {
            "symbol": args.symbol,
            "train_data_periods": [2021, 6, 21, 2021, 6, 21],
            "eval_data_periods": [2021, 6, 21, 2021, 6, 22]
        },
        # ERROR
        "log_level": "WARN",
    }

    session_container_path = init_session_container(args.session_id)
    with open(os.path.join(session_container_path, "config.json"), "a", encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    experiment = tune.run(PPOTrainer,
                          config=config,
                          metric="episode_reward_mean",
                          mode="max",
                          checkpoint_freq=10,
                          stop={"training_iteration": args.nr_episodes},
                          checkpoint_at_end=True,
                          local_dir=session_container_path,
                          )

    checkpoints = experiment.get_trial_checkpoints_paths(trial=experiment.get_best_trial('episode_reward_mean'),
                                                         metric='episode_reward_mean')
    checkpoint_path = checkpoints[0][0]

    checkpoint_path = r'C:\Users\demp\Documents\Repos\RLOptimalTradeExecution\data\sessions/1641411150\PPO_2022-01-05_20-32-30\PPO_lob_env_38d29_00000_0_2022-01-05_20-32-30\checkpoint_000010\checkpoint-10'

    reward = test_agent_one_episode(config=config,
                                    agent_path=checkpoint_path,
                                    eval_data_periods=[2021, 6, 21, 2021, 6, 21],
                                    symbol="btcusdt")
    print(reward)
    ray.shutdown()

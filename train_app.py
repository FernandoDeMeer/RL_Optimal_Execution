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
import random

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer

from src.data.historical_data_feed import HistoricalDataFeed
from src.core.environment.limit_orders_setup.broker_real import Broker
from src.core.environment.limit_orders_setup.base_env_real import DollarRewardAtStepEnv, RewardAtBucketEnv
from src.core.agent.ray_model import CustomRNNModel

from ray.rllib.models import ModelCatalog

## APPO/IMPALA
from ray.rllib.agents.impala import impala
##

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def init_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="lob_env")
    parser.add_argument("--num-cpus", type=int, default=2)

    parser.add_argument(
        "--framework",
        choices=["tf", "torch"],
        default="tf",
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
        default=10000000,
        help="Number of episodes to train.")

    return parser.parse_args()

args = init_arg_parser()

# Synchronous PPO
config = {
    "env": args.env,
    "framework": args.framework,
    # Number of GPUs to allocate to the trainer process. Note that not all
    # algorithms can take advantage of trainer GPUs. This can be fractional
    # (e.g., 0.3 GPUs).
    "num_gpus": 0,
    "num_workers": args.num_cpus - 1,
    "num_envs_per_worker": 1,
    # Size of batches collected from each worker.
    "rollout_fragment_length": 5,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    # DEFAULT train_batch_size: 4000
    "train_batch_size": 240,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 32,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 30,
    # discount factor
    "gamma": 1.0,
    "lr": 5e-5,
    "lr_schedule": [
        [0, 5e-5],
        [2e6, 5e-6],
    ],
    "entropy_coeff": 0.01,
    "lambda": 0.95,
    "kl_coeff": 0.2,
    "clip_param": 0.2,
    "vf_loss_coeff": 1.0,
    "vf_share_layers": False,
    "model": {
        "custom_model": "end_to_end_model",
        "custom_model_config": {"fcn_depth": 128,
                                "lstm_cells": 256},
    },

    "env_config": {'obs_config': {"lob_depth": 5,
                                  "nr_of_lobs": 5,
                                  "norm": True},
                   "train_config": {
                       "train": True,
                       "symbol": 'btcusdt',
                       "train_data_periods": [2021, 6, 5, 2021, 6, 11],
                       "eval_data_periods": [2021, 6, 12, 2021, 6, 14]
                   },
                   'trade_config': {'trade_direction': 1,
                                    'vol_low': 100,
                                    'vol_high': 100,
                                    'no_slices_low': 9,
                                    'no_slices_high': 9,
                                    'bucket_func': lambda no_of_slices: list(np.around(np.linspace(0,1,no_of_slices+2)[1:-1],2)),
                                    'rand_bucket_low': 0,
                                    'rand_bucket_high': 0},
                   'start_config': {'hour_low': 12,
                                    'hour_high': 12,
                                    'minute_low': 0,
                                    'minute_high': 0,
                                    'second_low': 0,
                                    'second_high': 0},
                   'exec_config': {'exec_times': [5],
                                   'delete_vol': False},
                   'reset_config': {'reset_num_episodes': 1000,
                                    'samples_per_feed': 20,
                                    'reset_feed': True},
                   'seed_config': {'seed': 0,},},

    # Eval
    "evaluation_interval": 10,
    # Number of episodes to run per evaluation period.
    "evaluation_num_episodes": 1,
    "evaluation_config": {
        "explore": False,
        "render_env": True,
    },
    "log_level": "WARN",
}

# ## Asynchronous PPO/IMPALA
# config = impala.ImpalaTrainer.merge_trainer_configs(
#     impala.DEFAULT_CONFIG,  # See keys in impala.py, which are also supported.
#     {
#         # Whether to use V-trace weighted advantages. If false, PPO GAE
#         # advantages will be used instead.
#         "vtrace": True,
#
#         # == These two options only apply if vtrace: False ==
#         # Should use a critic as a baseline (otherwise don't use value
#         # baseline; required for using GAE).
#         "use_critic": True,
#         # If true, use the Generalized Advantage Estimator (GAE)
#         # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
#         "use_gae": True,
#         # GAE(lambda) parameter
#         "lambda": 1.0,
#
#         # == PPO surrogate loss options ==
#         "clip_param": 0.4,
#
#         # == PPO KL Loss options ==
#         "use_kl_loss": False,
#         "kl_coeff": 1.0,
#         "kl_target": 0.01,
#
#         # == IMPALA optimizer params (see documentation in impala.py) ==
#         "rollout_fragment_length": 50,
#         "train_batch_size": 500,
#         "min_iter_time_s": 10,
#         "num_workers": args.num_cpus - 1,
#         "num_gpus": 0,
#         "num_multi_gpu_tower_stacks": 1,
#         "minibatch_buffer_size": 1,
#         "num_sgd_iter": 1,
#         "replay_proportion": 0.0,
#         "replay_buffer_num_slots": 100,
#         "learner_queue_size": 16,
#         "learner_queue_timeout": 300,
#         "max_sample_requests_in_flight_per_worker": 2,
#         "broadcast_interval": 1,
#         "grad_clip": 40.0,
#         "opt_type": "adam",
#         "lr": 0.0005,
#         "lr_schedule": None,
#         # "lr_schedule": [[0, 0.01],
#         #                 [3000000, 0.01]],
#         "decay": 0.99,
#         "momentum": 0.0,
#         "epsilon": 0.1,
#         "vf_loss_coeff": 0.5,
#         "entropy_coeff": 0.01,
#         "entropy_coeff_schedule": None,
#
#         "env": "lob_env",
#         "env_config": {
#             "obs_config": {
#                 "lob_depth": 5,
#                 "nr_of_lobs": 5,
#                 "norm": True},
#             "train_config": {
#                 "train": True,
#                 "symbol": args.symbol,
#                 "train_data_periods": [2021, 6, 5, 2021, 6, 11],
#                 "eval_data_periods": [2021, 6, 12, 2021, 6, 14]
#             },
#             "trade_config": {"trade_direction": 1,
#                              "vol_low": 100,
#                              "vol_high": 300,
#                              "no_slices_low": 4,
#                              "no_slices_high": 4,
#                              "bucket_func": lambda no_of_slices: [0.2, 0.4, 0.6, 0.8],
#                              "rand_bucket_low": 0,
#                              "rand_bucket_high": 0},
#             "start_config": {"hour_low": 1,
#                              "hour_high": 22,
#                              "minute_low": 0,
#                              "minute_high": 59,
#                              "second_low": 0,
#                              "second_high": 0},
#             "exec_config": {"exec_times": [5, 10, 15],
#                             "delete_vol": False},
#             'reset_config': {'reset_num_episodes': 500 * 10,
#                              'samples_per_feed': 500,
#                              'reset_feed': True},
#         },
#
#         # Eval
#         "evaluation_interval": 10,
#         # Number of episodes to run per evaluation period.
#         "evaluation_num_episodes": 1,
#         "evaluation_config": {
#             "explore": False,
#             "render_env": True,
#         },
#         "model": {
#             "custom_model": "end_to_end_model",
#             "custom_model_config": {"fcn_depth": 128,
#                                     "lstm_cells": 256},
#         },
#         "log_level": "WARN",
#     },
#     _allow_unknown_configs=True,
# )

def lob_env_creator(env_config):
    try:
        is_env_eval = env_config.num_workers == 0
    except:
        is_env_eval = True

    if is_env_eval:
        data_periods = env_config["train_config"]["eval_data_periods"]
    else:
        data_periods = env_config["train_config"]["train_data_periods"]

    data_start_day = datetime.datetime(year=data_periods[0], month=data_periods[1], day=data_periods[2])
    data_end_day = datetime.datetime(year=data_periods[3], month=data_periods[4], day=data_periods[5])

    lob_feed = HistoricalDataFeed(data_dir=os.path.join(DATA_DIR, "market", env_config["train_config"]["symbol"]),
                                  instrument=env_config["train_config"]["symbol"],
                                  start_day=data_start_day,
                                  end_day=data_end_day)

    broker = Broker(lob_feed)

    action_space = gym.spaces.Box(low=-1.0,
                                  high=1.0,
                                  shape=(1,),
                                  dtype=np.float32)

    return RewardAtBucketEnv(broker=broker,
                           config=config["env_config"],
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

    env = lob_env_creator(config)

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    return episode_reward


if __name__ == "__main__":


    # For debugging the ENV or other modules, set local_mode=True
    ray.init(num_cpus=args.num_cpus,
             local_mode=False,
             # local_mode=True,
             )

    register_env("lob_env", lob_env_creator)
    ModelCatalog.register_custom_model("end_to_end_model", CustomRNNModel)

    ##
    session_container_path = init_session_container(args.session_id)
    # with open(os.path.join(session_container_path, "config.json"), "a", encoding="utf-8") as f:
    #     json.dump(config, f, ensure_ascii=False, indent=4)

    # PPOTrainer
    experiment = tune.run("PPO",
                          config=config,
                          metric="episode_reward_mean",
                          mode="max",
                          checkpoint_freq=10,
                          stop={"training_iteration": args.nr_episodes},
                          checkpoint_at_end=True,
                          local_dir=session_container_path,
                          max_failures=0
                          )

    checkpoints = experiment.get_trial_checkpoints_paths(trial=experiment.get_best_trial("episode_reward_mean"),
                                                         metric="episode_reward_mean")
    checkpoint_path = checkpoints[0][0]

    reward = test_agent_one_episode(config=config["env_config"],
                                    agent_path=checkpoint_path,
                                    eval_data_periods=[2021, 6, 21, 2021, 6, 21],
                                    symbol="btcusdt")
    print(reward)

    ray.shutdown()

#
#   Reinforcement Learning Optimal Trade Execution
#   Async PPO
#

import shutil
import copy

import os
import time
import datetime
import argparse
import math

import gym
import json
from decimal import Decimal
import numpy as np

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print
from ray.rllib.agents.impala import impala

from ray.rllib.agents.ppo.appo import APPOTrainer

from src.data.historical_data_feed import HistoricalDataFeed
from src.core.environment.limit_orders_setup.broker_real import Broker
from src.core.environment.limit_orders_setup.base_env_real import NarrowTradeLimitEnvDiscrete

from ray.rllib.models import ModelCatalog
from src.core.agent.ray_model import CustomRNNModel




ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def init_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="lob_env")
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--agent-restore-path", type=str, default=None)

    parser.add_argument(
        "--framework",
        choices=["tf", "torch"],
        default="tf",
        help="The DL framework specifier.")

    parser.add_argument(
        "--symbol",
        choices=["btcusdt"],
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

    parser.add_argument(
        "--no-tune",
        type=bool,
        default=True,
        help="Run without Tune using a manual train loop instead.")

    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=100000,
        help="Number of timesteps to train.")

    parser.add_argument(
        "--stop-reward",
        type=float,
        default=0.9,
        help="Reward at which we stop training.")

    return parser.parse_args()


def lob_env_creator(env_config):

    if env_config['train_config']['train']:
        data_periods = env_config['train_config']["train_data_periods"]
    else:
        data_periods = env_config['train_config']["eval_data_periods"]

    data_start_day = datetime.datetime(year=data_periods[0], month=data_periods[1], day=data_periods[2])
    data_end_day = datetime.datetime(year=data_periods[3], month=data_periods[4], day=data_periods[5])

    lob_feed = HistoricalDataFeed(data_dir=os.path.join(DATA_DIR, "market", env_config['train_config']["symbol"]),
                                  instrument=env_config['train_config']["symbol"],
                                  start_day=data_start_day,
                                  end_day=data_end_day)

    exclude_keys = {'train_config'}
    env_config_clean = {k: env_config[k] for k in set(list(env_config.keys())) - set(exclude_keys)}

    return NarrowTradeLimitEnvDiscrete(broker=Broker(lob_feed),
                                       action_space=gym.spaces.Discrete(3),
                                       config=env_config_clean)


def init_session_container(session_id):

    if args.session_id == "0":
        session_id = str(int(time.time()))

    session_container_path = os.path.join("data", "sessions", session_id)

    if not os.path.isdir(session_container_path):
        os.makedirs(session_container_path)

    return session_container_path


# def test_agent_one_episode(config, agent_path):
#
#     agent = dqn.DQNTrainer(config=config)
#     agent.restore(agent_path)
#     config['env_config']['train_config']['train'] = False
#     env = lob_env_creator(config['env_config'])
#
#     episode_reward = 0
#     done = False
#     obs = env.reset()
#     while not done:
#         action = agent.compute_action(obs)
#         obs, reward, done, info = env.step(action)
#         episode_reward += reward
#
#     return episode_reward


if __name__ == "__main__":

    args = init_arg_parser()

    # For debugging the env or other modules, set local_mode=True
    ray.init(
        local_mode=False,
        # local_mode=True,
        num_cpus=args.num_cpus + 1)
    register_env(args.env, lob_env_creator)

    # ModelCatalog.register_custom_model("end_to_end_model", CustomRNNModel)

    # APPO based IMPALA CONFIG
    APPO_CONFIG = impala.ImpalaTrainer.merge_trainer_configs(
        impala.DEFAULT_CONFIG,  # See keys in impala.py, which are also supported.
        {
            # Whether to use V-trace weighted advantages. If false, PPO GAE
            # advantages will be used instead.
            "vtrace": True,

            # == These two options only apply if vtrace: False ==
            # Should use a critic as a baseline (otherwise don't use value
            # baseline; required for using GAE).
            "use_critic": True,
            # If true, use the Generalized Advantage Estimator (GAE)
            # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            "use_gae": True,
            # GAE(lambda) parameter
            "lambda": 1.0,

            # == PPO surrogate loss options ==
            "clip_param": 0.4,

            # == PPO KL Loss options ==
            "use_kl_loss": False,
            "kl_coeff": 1.0,
            "kl_target": 0.01,

            # == IMPALA optimizer params (see documentation in impala.py) ==
            "rollout_fragment_length": 50,
            "train_batch_size": 500,
            # "min_time_s_per_reporting": 10,
            "num_gpus": 0,
            "num_envs_per_worker": 1,
            "num_multi_gpu_tower_stacks": 1,
            "minibatch_buffer_size": 1,
            "num_sgd_iter": 1,
            "replay_proportion": 0.0,
            "replay_buffer_num_slots": 100,
            "learner_queue_size": 16,
            "learner_queue_timeout": 300,
            "max_sample_requests_in_flight_per_worker": 2,
            "broadcast_interval": 1,
            "grad_clip": 40.0,
            "opt_type": "adam",
            "lr": 0.0005,
            "lr_schedule": None,
            "decay": 0.99,
            "momentum": 0.0,
            "epsilon": 0.1,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "entropy_coeff_schedule": None,

            # From params...
            "num_workers": args.num_cpus,
            "framework": args.framework,
            "env": args.env,

            # "model": {
            #     "custom_model": "end_to_end_model",
            #     "custom_model_config": {"fcn_depth": 128,
            #                             "lstm_cells": 256},
            # },
        },
        _allow_unknown_configs=True,
    )

    env_config = {'obs_config': {"lob_depth": 5,
                                 "nr_of_lobs": 5,
                                 "norm": True},
                  "train_config": {
                      "train": True,
                      "symbol": 'btcusdt',
                      "train_data_periods": [2021, 6, 21, 2021, 6, 21],
                      "eval_data_periods": [2021, 6, 22, 2021, 6, 22]
                  },
                  'trade_config': {'trade_direction': 1,
                                   'vol_low': 100,
                                   'vol_high': 100,
                                   'no_slices_low': 3,
                                   'no_slices_high': 3,
                                   'bucket_func': lambda no_of_slices: [0.25, 0.5, 0.75],
                                   'rand_bucket_low': 0,
                                   'rand_bucket_high': 0},
                  'start_config': {'hour_low': 1,
                                   'hour_high': 22,
                                   'minute_low': 0,
                                   'minute_high': 55,
                                   'second_low': 0,
                                   'second_high': 55},
                  'exec_config': {'exec_times': [5],
                                  'delete_vol': False},
                  'reset_config': {'reset_num_episodes': 30,},
                  'seed_config': {'seed': 0}}
    env_config = {"env_config": env_config}
    APPO_CONFIG.update(env_config)

    session_container_path = init_session_container(args.session_id)

    with open("{}/params.txt".format(session_container_path), "w") as env_params_file:
        env_config_copy = copy.deepcopy(env_config)["env_config"]
        f__ = env_config_copy["trade_config"]["bucket_func"]
        env_config_copy["trade_config"]["bucket_func"] = f__(0)

        env_config_copy["nn_model"] = APPO_CONFIG["model"]

        env_params_file.write(json.dumps(env_config_copy,
                                         indent=4,
                                         separators=(',', ': ')))

    shutil.make_archive(base_dir="src",
                        root_dir=os.getcwd(),
                        format='zip',
                        base_name=os.path.join(session_container_path, "src"))
    print("")

    stop = {
        "training_iteration": args.nr_episodes,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    is_train = True
    if is_train:

        results = tune.run("APPO",
                           config=APPO_CONFIG,
                           metric="episode_reward_mean",
                           mode="max",
                           checkpoint_freq=10,
                           stop={"training_iteration": args.nr_episodes},
                           checkpoint_at_end=True,
                           local_dir=session_container_path,
                           restore=None if args.agent_restore_path is None else os.path.join(session_container_path,
                                                                                             "APPO",
                                                                                             args.agent_restore_path,),
                           max_failures=-1)
    else:
        trainer = APPOTrainer(config=APPO_CONFIG)

        for _ in range(args.nr_episodes):
            result = trainer.train()
            print(pretty_print(result))

            if result["timesteps_total"] >= args.stop_timesteps or \
                    result["episode_reward_mean"] >= args.stop_reward:
                break

    ray.shutdown()

#
#   Reinforcement Learning Optimal Trade Execution
#
import shutil
import copy

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
from src.core.environment.limit_orders_setup.base_env_real import NarrowTradeLimitEnvDiscrete
from src.core.agent.ray_model import CustomRNNModel

from ray.rllib.models import ModelCatalog

## APPO/IMPALA
from ray.rllib.agents.impala import impala
##

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

sessions_path = ROOT_DIR + r'\data\sessions'

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
        default=2,
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
                       "train_data_periods": [2021, 6, 1, 2021, 6, 20],
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
                   'start_config': {'hour_low': 0,
                                    'hour_high': 22,
                                    'minute_low': 0,
                                    'minute_high': 59,
                                    'second_low': 0,
                                    'second_high': 59},
                   'exec_config': {'exec_times': [5],
                                   'delete_vol': False},
                   'reset_config': {'reset_num_episodes': 1,},
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
#             'reset_config': {'reset_num_episodes': 500 * 10,},
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


    # action_space = gym.spaces.Box(low=-1.0,
    #                               high=1.0,
    #                               shape=(1,),
    #                               dtype=np.float32)

    action_space = gym.spaces.Discrete(n = 3)

    return NarrowTradeLimitEnvDiscrete(broker=Broker(lob_feed),
                                       action_space=action_space,
                                        config=env_config)


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

    env = lob_env_creator(config['env_config'])

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    return episode_reward

def train_eval_rolling_window(config,args):
    """
    Carry out a rolling-window training-evaluation experiment over a given time period.
    Args:
        config: Experiment config in ray-compatible format.
        args: Experiment settings (framework, symbol, session_id, nr_of_episodes)

    Returns:
        Training checkpoints and evaluation files for each training/eval period respectively.
    """
    from src.core.eval.evaluate import evaluate_session

    train_eval_horizon = config["env_config"]["train_config"]["train_data_periods"]

    data_start_day = datetime.datetime(year=train_eval_horizon[0], month=train_eval_horizon[1], day=train_eval_horizon[2])
    data_end_day = datetime.datetime(year=train_eval_horizon[3], month=train_eval_horizon[4], day=train_eval_horizon[5])

    delta = data_start_day - data_end_day   # returns timedelta
    train_eval_period_days = []
    for i in range(-delta.days + 1):
        day = data_start_day + datetime.timedelta(days=i)
        train_eval_period_days.append(day)

    train_eval_period_limits = []
    for i in range(len(train_eval_period_days)//5):
        train_eval_period_limits.append(train_eval_period_days[5*i])
        train_eval_period_limits.append(train_eval_period_days[5*(i+1)-1])

    train_eval_periods =[]
    for period_idx in range(len(train_eval_period_limits)//2-1):
        train_eval_periods.append([(train_eval_period_limits[2*period_idx],train_eval_period_limits[2*period_idx+1]),
                                   (train_eval_period_limits[2*period_idx+2],train_eval_period_limits[2*period_idx+3])])

    restore_previous_agent = False
    session_idx = 1
    for train_eval_period in train_eval_periods:

        config["env_config"]["train_config"]["train_data_periods"] = [train_eval_period[0][0].year,
                                                                      train_eval_period[0][0].month,
                                                                      train_eval_period[0][0].day,
                                                                      train_eval_period[0][1].year,
                                                                      train_eval_period[0][1].month,
                                                                      train_eval_period[0][1].day]
        config["env_config"]["train_config"]["eval_data_periods"] = [train_eval_period[1][0].year,
                                                                      train_eval_period[1][0].month,
                                                                      train_eval_period[1][0].day,
                                                                      train_eval_period[1][1].year,
                                                                      train_eval_period[1][1].month,
                                                                      train_eval_period[1][1].day]
        train_agent(config,args,restore_previous_agent,session_idx = session_idx)
        evaluate_session(sessions_path= sessions_path ,trainer= 'PPO' ,config = config)
        restore_previous_agent = True
        session_idx += 1





def train_agent(config,args,restore_previous_agent,session_idx):
    """
    Creates a session and trains an agent for a number of episodes specified in the args.
    Args:
        config: Experiment config in ray-compatible format.
        args: Experiment settings (framework, symbol, session_id, nr_of_episodes)
        restore_previous_agent: Bool. If True, the best performing checkpoint of the previous session will be loaded at
        the start of training.
    Returns:
        experiment: Experiment Analysis object + saves training checkpoints.

    """
    # For debugging the ENV or other modules, set local_mode=True
    ray.init(num_cpus=args.num_cpus,
             local_mode=False,
             ignore_reinit_error= True,
             # local_mode=True,
             )

    register_env("lob_env", lob_env_creator)
    ModelCatalog.register_custom_model("end_to_end_model", CustomRNNModel)

    ##
    session_container_path = init_session_container(args.session_id)
    # with open(os.path.join(session_container_path, "config.json"), "a", encoding="utf-8") as f:
    #     json.dump(config, f, ensure_ascii=False, indent=4)

    with open("{}/params.txt".format(session_container_path), "w") as env_params_file:
        env_config_copy = copy.deepcopy(config)["env_config"]
        f__ = env_config_copy["trade_config"]["bucket_func"]
        env_config_copy["trade_config"]["bucket_func"] = f__(0)
        try:
            env_config_copy["nn_model"] = config["model"]
        except:
            pass

        env_params_file.write(json.dumps(env_config_copy,
                                         indent=4,
                                         separators=(',', ': ')))

    shutil.make_archive(base_dir="src",
                        root_dir=os.getcwd(),
                        format='zip',
                        base_name=os.path.join(session_container_path, "src"))
    print("")


    # PPOTrainer
    if restore_previous_agent:
        from src.core.eval.evaluate import get_session_best_checkpoint_path
        sessions = [int(session_id) for session_id in os.listdir(sessions_path) if session_id !='.gitignore']
        checkpoint = get_session_best_checkpoint_path(session_path=sessions_path, trainer='PPO', session= np.min(sorted(sessions,reverse=True)[:2]))


        experiment = tune.run("PPO",
                              config=config,
                              metric="episode_reward_mean",
                              mode="max",
                              checkpoint_freq=10,
                              stop={"training_iteration": session_idx * args.nr_episodes},
                              checkpoint_at_end=True,
                              local_dir=session_container_path,
                              max_failures=0,
                              restore= os.path.join(sessions_path,'PPO',checkpoint)
                              )
    else:
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

    return experiment

if __name__ == "__main__":

    train_eval_rolling_window(config,args)

    # experiment = train_agent(config,args)
    #
    # evaluate_session(sessions_path)
    #
    # checkpoints = experiment.get_trial_checkpoints_paths(trial=experiment.get_best_trial("episode_reward_mean"),
    #                                                      metric="episode_reward_mean")
    # checkpoint_path = checkpoints[0][0]
    #
    # reward = test_agent_one_episode(config=config["env_config"],
    #                                 agent_path=checkpoint_path,
    #                                 eval_data_periods=[2021, 6, 21, 2021, 6, 21],
    #                                 symbol="btcusdt")
    # print(reward)

    ray.shutdown()

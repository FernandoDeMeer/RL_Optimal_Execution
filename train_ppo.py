#
#   Reinforcement Learning Optimal Trade Execution
#
#   PPO
import shutil
import copy

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
from src.core.environment.limit_orders_setup.broker import Broker
from src.core.environment.limit_orders_setup.base_env import NarrowTradeLimitEnvDiscrete
from src.core.agent.ray_model import CustomRNNModel

from ray.rllib.models import ModelCatalog


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
        default=1000000,
        help="Number of episodes to train.")

    parser.add_argument(
        "--rl_algo",
        type=str,
        default="PPO",
        help="RL Algorithm to train the Agent with.")


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
        evaluate_session(sessions_path= sessions_path ,trainer= args.rl_algo ,config = config)
        restore_previous_agent = True
        session_idx += 1
    ray.shutdown()



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
    # Use a RNN Agent
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


    if restore_previous_agent:
        from src.core.eval.evaluate import get_session_best_checkpoint_path
        sessions = [int(session_id) for session_id in os.listdir(sessions_path) if session_id !='.gitignore']
        checkpoint = get_session_best_checkpoint_path(session_path=sessions_path, trainer='PPO', session= np.min(sorted(sessions,reverse=True)[:2]))


        experiment = tune.run(args.rl_algo,
                              config=config,
                              metric="episode_reward_mean",
                              mode="max",
                              checkpoint_freq=1000,
                              stop={"training_iteration": session_idx * args.nr_episodes},
                              checkpoint_at_end=True,
                              local_dir=session_container_path,
                              max_failures=0,
                              restore= os.path.join(sessions_path,'PPO',checkpoint)
                              )
    else:
        experiment = tune.run(args.rl_algo,
                              config=config,
                              metric="episode_reward_mean",
                              mode="max",
                              checkpoint_freq=1000,
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

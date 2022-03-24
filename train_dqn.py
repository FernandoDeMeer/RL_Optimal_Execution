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

from src.data.historical_data_feed import HistoricalDataFeed
from src.core.environment.limit_orders_setup.broker_real import Broker
from src.core.environment.limit_orders_setup.base_env_real import BaseEnv


class NarrowTradeLimitEnvDQN(BaseEnv):

    def __init__(self, *args, **kwargs):
        super(NarrowTradeLimitEnvDQN, self).__init__(*args, **kwargs)

    def _convert_action(self, action):
        shift = 0.2
        if action == 0:
            action_out = 1 - shift
        elif action == 1:
            action_out = 1
        elif action == 2:
            action_out = 1 + shift
        else:
            raise ValueError
        action_out = 1
        return action_out

    def infer_volume_from_action(self, action):
        vol_to_trade = Decimal(str(action)) * \
                       self.broker.benchmark_algo.volumes_per_trade[self.broker.rl_algo.bucket_idx][self.broker.benchmark_algo.order_idx]
        factor = 10 ** (- self.broker.benchmark_algo.tick_size.as_tuple().exponent)
        vol_to_trade = Decimal(str(math.floor(vol_to_trade * factor) / factor))
        if vol_to_trade > self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]:
            vol_to_trade = self.broker.rl_algo.bucket_vol_remaining[self.broker.rl_algo.bucket_idx]
        return vol_to_trade

    def reward_func(self):
        """ Env with reward at end of each bucket as $ improvement of VWAP """

        reward = 0
        try:
            if self.bucket_time != self.bucket_time_prev:
                vwap_bmk, vwap_rl = self.broker.calc_vwap_from_logs(start_date=self.bucket_time_prev,
                                                                    end_date=self.bucket_time)
                if self.trade_dir == 1:
                    reward = np.sign(vwap_bmk - vwap_rl)
                else:
                    reward = np.sign(vwap_rl - vwap_bmk)
        except:
            reward = 0
        return reward


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def init_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="lob_env")
    parser.add_argument("--num-cpus", type=int, default=0)

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

    return NarrowTradeLimitEnvDQN(broker=Broker(lob_feed),
                                  action_space=gym.spaces.Discrete(3),
                                  config=env_config_clean)


def init_session_container(session_id):

    if args.session_id == "0":
        session_id = str(int(time.time()))

    session_container_path = os.path.join("data", "sessions", session_id)

    if not os.path.isdir(session_container_path):
        os.makedirs(session_container_path)

    return session_container_path


def test_agent_one_episode(config, agent_path):

    agent = dqn.DQNTrainer(config=config)
    agent.restore(agent_path)
    config['env_config']['train_config']['train'] = False
    env = lob_env_creator(config['env_config'])

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
    register_env(args.env, lob_env_creator)

    # Config necessary for RLlib training
    config = {
        "env": args.env,  # or "corridor" if registered above
        "num_workers": args.num_cpus - 1,
        "num_gpus": 0,
        "num_envs_per_worker": 1,
        "framework": args.framework,
        "evaluation_interval": 10,
        "train_batch_size": 200,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 1,
        "evaluation_config": {
            "explore": False,
            "render_env": True,
        },
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 10000000,  # Timesteps over which to anneal epsilon.

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        }
    }

    # Add config for our custom environment
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
                                    'vol_low': 10,
                                    'vol_high': 10,
                                    'no_slices_low': 1,
                                    'no_slices_high': 1,
                                    'bucket_func': lambda no_of_slices: [0.5],
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
                   'reset_config': {'reset_num_episodes': 5000,
                                    'samples_per_feed': 2000,
                                    'reset_feed': True},
                   'seed_config': {'seed': 0,}}
    env_config = {"env_config": env_config}
    config.update(env_config)

    # config for stopping the training
    stop = {
        "training_iteration": args.nr_episodes,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    session_container_path = init_session_container(args.session_id)
    """
    with open(os.path.join(session_container_path, "config.json"), "a", encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    """

    no_tune = False
    if no_tune:

        dqn_config = dqn.DEFAULT_CONFIG.copy()
        dqn_config.update(config)
        trainer = dqn.DQNTrainer(config=dqn_config)

        # run manual training loop and print results after each iteration
        for _ in range(args.nr_episodes):
            result = trainer.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if result["timesteps_total"] >= args.stop_timesteps or \
                    result["episode_reward_mean"] >= args.stop_reward:
                break
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        results = tune.run("DQN",
                           config=config,
                           metric="episode_reward_mean",
                           mode="max",
                           checkpoint_freq=10,
                           stop={"training_iteration": args.nr_episodes},
                           checkpoint_at_end=True,
                           local_dir=session_container_path,
                           )

        """
        print("Test agent on one episode")
        checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean'),
                                                          metric='episode_reward_mean')
        checkpoint_path = checkpoints[0][0]
        reward = test_agent_one_episode(config=config,
                                        agent_path=checkpoint_path)
        print(reward)
        """

    ray.shutdown()
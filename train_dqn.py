import os
import time
import datetime
import argparse

import gym
import json

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print

from src.data.historical_data_feed import HistoricalDataFeed
from src.core.environment.limit_orders_setup.broker_real import Broker
from src.core.environment.limit_orders_setup.base_env_real import RewardAtStepEnv


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
        default=1,
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


class TradingEnvDQN(RewardAtStepEnv):
    def _convert_action(self, action):
        return action / self.action_space.n


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

    return TradingEnvDQN(broker=Broker(lob_feed),
                         action_space=gym.spaces.Discrete(11),
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
        "num_workers": args.num_cpus,
        "num_envs_per_worker": 1,
        "framework": args.framework,
        "evaluation_interval": 10,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 1,
        "evaluation_config": {
            "explore": False,
            "render_env": True,
        }
    }

    # Add config for our custom environment
    env_config = {'env_config': {'obs_config': {
                                     "lob_depth": 5,
                                     "nr_of_lobs": 5,
                                     "norm": True},
                                 'train_config': {
                                     "train": True,
                                     "symbol": args.symbol,
                                     "train_data_periods": [2021, 6, 21, 2021, 6, 21],
                                     "eval_data_periods": [2021, 6, 22, 2021, 6, 22]
                                 }}
                }
    config.update(env_config)

    # config for stopping the training
    stop = {
        "training_iteration": args.nr_episodes,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    session_container_path = init_session_container(args.session_id)
    with open(os.path.join(session_container_path, "config.json"), "a", encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    if args.no_tune:

        dqn_config = dqn.DEFAULT_CONFIG.copy()
        dqn_config.update(config)
        dqn_config["lr"] = 1e-3
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
        results = tune.run(dqn.DQNTrainer,
                           config=config,
                           metric="episode_reward_mean",
                           mode="max",
                           checkpoint_freq=10,
                           stop=stop,
                           checkpoint_at_end=True,
                           local_dir=session_container_path,
                           )

        print("Test agent on one episode")
        checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean'),
                                                          metric='episode_reward_mean')
        checkpoint_path = checkpoints[0][0]
        reward = test_agent_one_episode(config=config,
                                        agent_path=checkpoint_path)
        print(reward)

    ray.shutdown()
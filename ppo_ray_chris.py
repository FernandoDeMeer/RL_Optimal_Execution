"""Example of using a custom RNN keras model."""
import pprint
import argparse
import os
import ray
from ray import tune
import gym
import numpy as np
from ray.tune.registry import register_env
from src.core.agent.ray_model import RNNModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from src.core.environment.base_env import BaseEnv
from src.data.historical_data_feed import HistoricalDataFeed
from src.core.environment.execution_algo import TWAPAlgo


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def lob_env_creator():
    import os
    # Construct the data feed
    dir = os.path.join(ROOT_DIR, 'data_dir')
    lob_feed = HistoricalDataFeed(data_dir=dir,
                                  instrument='btc_usdt',
                                  lob_depth=20,
                                  start_day=None,
                                  end_day=None)

    benchmark = TWAPAlgo()  # define benchmark algo
    volume = 10  # total volume to trade
    trade_steps = 100  # total number of time steps available to trade
    trade_direction = 1

    observation_space_config = {'lob_depth': 5, 'nr_of_lobs': 1, 'norm': True}


    # define action space
    action_space = gym.spaces.Box(low=0.0,
                                  high=1.0,
                                  shape=(1,),
                                  dtype=np.float32)

    # construct the environment config
    env_config = {
        "data_feed": lob_feed,
        "trade_direction": trade_direction,
        "qty_to_trade": volume,
        "max_step_range": trade_steps,
        "benchmark_algo": benchmark,
        "obs_config": observation_space_config,
        "action_space": action_space
    }
    lob_env = BaseEnv(show_ui=False,
                      data_feed=env_config["data_feed"],
                      trade_direction=env_config["trade_direction"],
                      qty_to_trade=env_config["qty_to_trade"],
                      max_step_range=env_config["max_step_range"],
                      benchmark_algo=env_config["benchmark_algo"],
                      obs_config=env_config["obs_config"],
                      action_space=env_config["action_space"])
    return lob_env # return an env instance


register_env("lob_env", lambda env_config: lob_env_creator(env_config))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument("--env", type=str, default="lob_env")
parser.add_argument("--num-cpus", type=int, default=1)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
         "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=100,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=90.0,
    help="Reward at which we stop training.")

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

    ModelCatalog.register_custom_model("my_rnn", RNNModel)
    register_env("lob_env", lambda env_config: lob_env_creator())

    config = {
        "env": args.env,
        "env_config": {
            "repeat_delay": 2,
        },
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "num_envs_per_worker": 20,
        "entropy_coeff": 0.001,
        "num_sgd_iter": 5,
        "vf_loss_coeff": 1e-5,
        "model": {
            "custom_model": "my_rnn",
            "max_seq_len": 20,
            "custom_model_config": {
                "cell_size": 32,
            },
        },
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # To run the Trainer without tune.run, using our RNN model and
    # manual state-in handling, do the following:

    # Example (use `config` from the above code):
    from ray.rllib.agents.ppo import PPOTrainer

    trainer = PPOTrainer(config)

    for _ in range(stop['training_iteration']):
        result = trainer.train()
        print(pprint.pprint(result))
        # stop training of the target train steps or reward are reached
        if result["timesteps_total"] >= stop['timesteps_total'] or \
                result["episode_reward_mean"] >= stop['episode_reward_mean']:
            break

"""
    results = tune.run(args.run, config=config, stop=stop, verbose=1)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
"""
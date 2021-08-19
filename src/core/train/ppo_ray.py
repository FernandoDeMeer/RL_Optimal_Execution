import os
import shutil
import pprint

import ray
from ray import tune
import gym
import numpy as np
from main import ROOT_DIR

from src.data.historical_data_feed import HistFeedRL
from src.core.environment.execution_algo import TWAPAlgo
from src.core.environment.base_env import BaseEnv
from src.core.agent.model import EndtoEndModel
from src.core.agent.policy import get_policy_config
from ray.rllib.models import ModelCatalog
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer



if __name__ == "__main__":
    #dirst Declare where to store checkpoints and tensorboard result files

    chkpt_root = "../../tmp/chkpt"
    shutil.rmtree(chkpt_root,ignore_errors=True,onerror=None)
    ray_results_dir = "../../ray_results"
    shutil.rmtree(ray_results_dir, ignore_errors=True, onerror=None)

    # Construct the data feed
    dir = os.path.join(ROOT_DIR, 'data_dir')
    lob_feed = HistFeedRL(data_dir=dir,
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

    #Register the env for ray

    from ray.tune.registry import register_env

    def lob_env_creator(env_config):
        lob_env = BaseEnv(data_feed=env_config["data_feed"],
                          trade_direction=env_config["trade_direction"],
                          qty_to_trade=env_config["qty_to_trade"],
                          max_step_range=env_config["max_step_range"],
                          benchmark_algo=env_config["benchmark_algo"],
                          obs_config=env_config["obs_config"],
                          action_space=env_config["action_space"])
        return lob_env # return an env instance

    register_env("lob_env", lob_env_creator)

    #Register the model for ray

    ray.init(local_mode=True, ignore_reinit_error = True)

    ModelCatalog.register_custom_model(
        "EndtoEndModel", EndtoEndModel)

    policy_config, stop = get_policy_config(model= "EndtoEndModel")

    # manual training with train loop using PPO and fixed learning rate
    print("Running manual train loop without Ray Tune.")
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(policy_config)
    ppo_config["env_config"] = env_config

    # use fixed learning rate instead of grid search (needs tune)
    trainer = PPOTrainer(config= ppo_config, env="lob_env",)

    model = trainer.model

    pprint.pprint(model.variables())
    pprint.pprint(model.value_function())

    print(model.base_model.summary())


    # run manual training loop and print results after each iteration
    for _ in range(stop['training_iteration']):
        result = trainer.train()
        print(pprint.pprint(result))
        # stop training of the target train steps or reward are reached
        if result["timesteps_total"] >= stop['timesteps_total'] or \
                result["episode_reward_mean"] >= stop['episode_reward_mean']:
            break

    # To visualize the training summary run tensorboard --logdir=~/ray_results on the cmd
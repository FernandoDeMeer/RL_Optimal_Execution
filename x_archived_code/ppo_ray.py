import os
import shutil
import pprint

import ray

import gym
import numpy as np

from src.data.historical_data_feed import HistoricalDataFeed
from src.core.environment.market_orders_setup.execution_algo import TWAPAlgo
from src.core.environment.market_orders_setup.base_env import BaseEnv
from src.core.agent.ray_policy import get_policy_config
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def lob_env_creator(env_config):
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

if __name__ == "__main__":

    # first Declare where to store checkpoints and tensorboard result files

    chkpt_root = "/tmp/chkpt"
    shutil.rmtree(chkpt_root,ignore_errors=True,onerror=None)
    ray_results_dir = "/ray_results"
    shutil.rmtree(ray_results_dir, ignore_errors=True, onerror=None)

    # Model config

    model_config = {
        # All model-related settings go into this sub-dict (it will be added to the policy_config)
            # By default, the MODEL_DEFAULTS dict will be used (see https://docs.ray.io/en/master/rllib-models.html)

            # Change individual keys in that dict by overriding them, e.g.
            # Parameters according to the End-to-End paper
            "fcnet_hiddens": [128, 128],
            "fcnet_activation": "relu",
            "use_lstm": True,
            "max_seq_len": 12,
            "lstm_use_previous_action": True,
            "lstm_use_previous_reward": True,
            "lstm_cell_size": 128,

        }

    # Init ray
    ray.init(local_mode=True, ignore_reinit_error = True)

    #Register the env for ray

    from ray.tune.registry import register_env


    register_env("lob_env", lambda env_config: lob_env_creator(env_config))


    policy_config, stop = get_policy_config(model_config=model_config)

    # manual training with train loop using PPO and fixed learning rate
    print("Running manual train loop without Ray Tune.")
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    # ppo_config.update(policy_config) #TODO: Adding the custom model makes it blow up...

    # use fixed learning rate instead of grid search (needs tune)
    trainer = PPOTrainer(config= ppo_config, env="lob_env",)

    # model = trainer.config["model"]

    # pprint.pprint(model.variables())
    # pprint.pprint(model.value_function())
    #
    # print(model.base_model.summary())


    # run manual training loop and print results after each iteration
    for _ in range(stop['training_iteration']):
        result = trainer.train()
        print(pprint.pprint(result))
        # stop training of the target train steps or reward are reached
        if result["timesteps_total"] >= stop['timesteps_total'] or \
                result["episode_reward_mean"] >= stop['episode_reward_mean']:
            break

    # To visualize the training summary run tensorboard --logdir=~/ray_results on the cmd
import os
import random

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils import try_import_tf
from src.core.environment.env import HistoricalOrderBookEnv
from src.core.agent.model import EndtoEndModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import grid_search
from ray.tune.logger import pretty_print

tf1, tf, tfv  = try_import_tf()


def get_policy_config(env_config):

    config = {
        "env": HistoricalOrderBookEnv,  # or "corridor" if registered above
        "env_config": env_config,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "lr":  1e-5,  # try different lrs
        "num_workers": 1,  # parallelism
        "framework": 'tf',
    }

    stop = {
        "training_iteration": 100,
        "timesteps_total": 100000,
        "episode_reward_mean": 0.8,
    }

    return config,stop

if __name__ == "__main__":

    ray.init(local_mode=True)
    ModelCatalog.register_custom_model(
        "EndtoEndModel", EndtoEndModel)

    config, stop = get_policy_config(env_config=env_config)

    # manual training with train loop using PPO and fixed learning rate
    print("Running manual train loop without Ray Tune.")
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config)
    # use fixed learning rate instead of grid search (needs tune)
    ppo_config["lr"] = 1e-5
    trainer = PPOTrainer(config=ppo_config, env=HistoricalOrderBookEnv)
    # run manual training loop and print results after each iteration
    for _ in range(stop['training_iteration']):
        result = trainer.train()
        print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if result["timesteps_total"] >= stop['timesteps_total'] or \
                result["episode_reward_mean"] >= stop['episode_reward_mean']:
            break

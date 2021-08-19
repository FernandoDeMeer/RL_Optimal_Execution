import os
import random

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from src.core.environment.env import HistoricalOrderBookEnv
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import grid_search
from ray.tune.logger import pretty_print

tf1, tf, tfv  = try_import_tf()


def get_policy_config(model):

    policy_config = {
        "env": HistoricalOrderBookEnv,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": model,
            "vf_share_layers": True,
        },
        "lr":  1e-5,  # try different lrs
        "num_workers": 1,  # parallelism
        "framework": 'tf',
        "entropy_coeff": 0.01,
    }

    stop = {
        "training_iteration": 10000, #Number of iterations to train
        "timesteps_total": 1000000, #Maximum number of timesteps to take
        "episode_reward_mean": 0.8, #Stop if the average reward per episode is bigger than episode_reward_mean
    }

    return policy_config, stop


import numpy as np
import os
from os.path import isdir, isfile, join
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from tqdm import tqdm
from train_app import lob_env_creator, test_agent_one_episode, init_arg_parser, config, ROOT_DIR, DATA_DIR
from ray.rllib.agents.ppo import PPOTrainer

from src.data.historical_data_feed import HistoricalDataFeed
from src.core.environment.limit_orders_setup.broker_real import Broker
from src.core.environment.limit_orders_setup.base_env_real import DollarRewardAtStepEnv
from src.core.agent.ray_model import CustomRNNModel


def eval_agent(trainer, env, nr_episodes, plot=True):

    reward_vec = []
    vwap_bmk = []
    vwap_rl = []

    for _ in tqdm(range(nr_episodes), desc="Evaluation of agent"):
        obs = env.reset()
        episode_reward = 0
        done = False
        state = trainer.get_policy().get_initial_state()
        while not done:
            action, state, _ = trainer.compute_action(obs,state = state)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        reward_vec.append(episode_reward)
        vwap_bmk.append(env.broker.benchmark_algo.bmk_vwap)
        vwap_rl.append(env.broker.rl_algo.rl_vwap)

    outperf = [True if vwap > vwap_rl[idx] else False for idx, vwap in enumerate(vwap_bmk)]
    vwap_perc_diff = (np.array(vwap_bmk)-np.array(vwap_rl)) / np.array(vwap_bmk)
    downside_median = np.median(vwap_perc_diff[outperf])
    upside_median = np.median(vwap_perc_diff[[not elem for elem in outperf]])

    # after each episode, collect execution prices
    d_out = {'rewards': np.array(reward_vec),
             'vwap_bmk': np.array(vwap_bmk),
             'vwap_rl': np.array(vwap_rl),
             'vwap_diff': vwap_perc_diff}
    stats = {'percentage_outperformance': sum(outperf)/len(outperf),
             'downside_median': downside_median,
             'upside_median': upside_median}

    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(12,8))
        axs[0, 0].hist(d_out['rewards'], density=True, bins=50)
        axs[0, 0].set_title('Rewards from RL Agent')
        axs[0, 0].set(xlabel='Reward', ylabel='Probability')

        axs[0, 1].hist(d_out['vwap_bmk'], alpha=0.5, density=True, bins=50)
        axs[0, 1].hist(d_out['vwap_rl'], alpha=0.5, density=True, bins=50)
        axs[0, 1].set_title('Benchmark and RL Execution Price')
        axs[0, 1].set(xlabel='Execution Price', ylabel='Probability')

        axs[1, 0].hist(d_out['vwap_diff'], density=True, bins=50)
        axs[1, 0].set_title('Difference of Benchmark vs. RL Execution Price')
        axs[1, 0].set(xlabel='Execution Price Difference (%)', ylabel='Probability')
        plt.show()

    return d_out, stats


def get_session_best_checkpoint_path(session_path, session,):

    session_path = session_path + r'\{}\PPO'.format(str(session))
    session_filename = [f for f in os.listdir(session_path) if isdir(join(session_path, f))]
    sessions_path = session_path + r'\{}'.format(session_filename[0])

    analysis = tune.Analysis(sessions_path)  # can also be the result of `tune.run()`

    trial_logdir = analysis.get_best_logdir(metric="episode_reward_mean", mode="max")  # Can also just specify trial dir directly

    # checkpoints = analysis.get_trial_checkpoints_paths(trial_logdir)  # Returns tuples of (logdir, metric)
    best_checkpoint = analysis.get_best_checkpoint(trial_logdir, metric="episode_reward_mean", mode="max")

    return best_checkpoint

if __name__ == "__main__":

    args = init_arg_parser()

    # For debugging the ENV or other modules, set local_mode=True
    ray.init(num_cpus=args.num_cpus,
             local_mode=True,
             )

    sessions_path = ROOT_DIR + r'\data\sessions'
    sessions = [int(session_id) for session_id in os.listdir(sessions_path) if session_id !='.gitignore']
    checkpoint = get_session_best_checkpoint_path(session_path=sessions_path, session= np.max(sessions))

    config["env_config"]["train_config"]["train"] = False
    config["num_workers"] = 0
    register_env("lob_env", lob_env_creator)
    ModelCatalog.register_custom_model("end_to_end_model", CustomRNNModel)


    agent = PPOTrainer(config=config)
    agent.restore(checkpoint)

    env = lob_env_creator(env_config= config["env_config"])
    eval_agent(trainer= agent,env= env ,nr_episodes= 100, plot=True)



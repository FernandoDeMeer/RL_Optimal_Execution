import numpy as np
import pandas as pd
import os
import random
from os.path import isdir, isfile, join
import re
from datetime import datetime
import matplotlib.pyplot as plt

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from tqdm import tqdm
from train_ppo import lob_env_creator, init_arg_parser, ROOT_DIR, DATA_DIR
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer

from src.core.agent.ray_model import CustomRNNModel


def eval_agent_one_day(trainer, env, nr_episodes,session_dir,day, plot=False, ):

    reward_vec = []
    vwap_bmk = []
    vwap_rl = []
    vol_percentages = []

    for _ in tqdm(range(nr_episodes), desc="Evaluation of the RL agent"):
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
        vol_percentages.append(np.mean(np.array(env.broker.rl_algo.volumes_per_trade)/
                                     np.array(env.broker.rl_algo.bucket_volumes)[:,None],axis= 0,dtype=np.float32))

    outperf = [True if vwap > vwap_rl[idx] else False for idx, vwap in enumerate(vwap_bmk)]
    vwap_perc_diff = (np.array(vwap_bmk)-np.array(vwap_rl)) / np.array(vwap_bmk)
    downside_median = np.median(vwap_perc_diff[outperf])
    upside_median = np.median(vwap_perc_diff[[not elem for elem in outperf]])
    vol_percentages_avg, error_vol_percentages = tolerant_mean(vol_percentages)

    # Collect results
    d_out = {'rewards': np.array(reward_vec),
             'vwap_bmk': np.array(vwap_bmk),
             'vwap_rl': np.array(vwap_rl),
             'vwap_diff': vwap_perc_diff,
             'vol_percentages': np.array(vol_percentages_avg),
             'vol_percentages_error': np.array(error_vol_percentages)}
    stats = {'percentage_outperformance': sum(outperf)/len(outperf),
             'downside_median': downside_median,
             'upside_median': upside_median}

    if plot:
        plot_eval_day(session_dir, d_out, day)

    return d_out, stats

def eval_agent(trainer, env, nr_episodes,):

    reward_vec = []
    vwap_bmk = []
    vwap_rl = []
    vol_percentages = []
    # execution_time_bmk = []
    # execution_time_rl = []

    for _ in tqdm(range(nr_episodes), desc="Evaluation of the RL agent"):
        obs = env.reset()
        episode_reward = 0
        done = False
        state = trainer.get_policy().get_initial_state()
        while not done:
            action, state, _ = trainer.compute_action(obs,state = state)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        # Extract the results of both algos
        reward_vec.append(episode_reward)
        vwap_bmk.append(env.broker.benchmark_algo.bmk_vwap)
        vwap_rl.append(env.broker.rl_algo.rl_vwap)
        vol_percentages.append(np.mean(np.array(env.broker.rl_algo.volumes_per_trade)/
                                       np.array(env.broker.rl_algo.bucket_volumes)[:,None],axis= 0,dtype=np.float32))
        # execution_time_bmk.append((datetime.strptime(env.broker.trade_logs['benchmark_algo'][-1]['timestamp'], '%Y-%m-%d %H:%M:%S.%f') -
        #                            datetime.strptime(env.broker.trade_logs['benchmark_algo'][0]['timestamp'], '%Y-%m-%d %H:%M:%S.%f')).seconds)
        # execution_time_rl.append((datetime.strptime(env.broker.trade_logs['rl_algo'][-1]['timestamp'], '%Y-%m-%d %H:%M:%S.%f') -
        #                            datetime.strptime(env.broker.trade_logs['rl_algo'][0]['timestamp'], '%Y-%m-%d %H:%M:%S.%f')).seconds)

    # Calculate performance statistics from results
    outperf = [True if vwap > vwap_rl[idx] else False for idx, vwap in enumerate(vwap_bmk)]
    vwap_perc_diff = (np.array(vwap_bmk)-np.array(vwap_rl)) / np.array(vwap_bmk)
    downside_median = np.median(vwap_perc_diff[outperf])
    upside_median = np.median(vwap_perc_diff[[not elem for elem in outperf]])
    vol_percentages_avg, vol_percentages_std = tolerant_mean(vol_percentages)
    # execution_speeds = np.array(execution_time_bmk) - np.array(execution_time_rl)

    # Collect results
    d_out = {'rewards': np.array(reward_vec),
             'vwap_bmk': np.array(vwap_bmk),
             'vwap_rl': np.array(vwap_rl),
             'vwap_diff': vwap_perc_diff,
             'vol_percentages': np.array(vol_percentages_avg),
             'vol_percentages_std': np.array(vol_percentages_std)}
    stats = {'percentage_outperformance': sum(outperf)/len(outperf),
             'downside_median': downside_median,
             'upside_median': upside_median}

    # env.broker.rl_algo.plot_schedule(env.broker.trade_logs['rl_algo'])

    return d_out, stats


def plot_eval_day(session_dir, d_out, day):
    fig, axs = plt.subplots(2, 2, figsize=(14,10))

    plt.suptitle("Evaluation on {}, daily volatility: {} ".format(day,
                                                                  env.broker.data_feed.day_volatilities[env.broker.data_feed.day_volatilities_ranking[env.broker.data_feed.binary_file_idx]] ), fontsize=14)


    axs[0, 0].hist(d_out['rewards'], density=True, bins=50)
    axs[0, 0].set_title('Rewards from RL Agent')
    axs[0, 0].set(xlabel='Reward', ylabel='Frequency')

    axs[0, 1].hist(d_out['vwap_bmk'], alpha=0.5, density=True, bins=50, label= 'Benchmark VWAP')
    axs[0, 1].hist(d_out['vwap_rl'], alpha=0.5, density=True, bins=50, label = 'RL VWAP')
    axs[0, 1].set_title('Benchmark and RL Execution Price')
    axs[0, 1].set(xlabel='Execution Price', ylabel='Probability')
    axs[0, 1].legend(loc = "upper left")

    price_diff_percent_avg = np.average(100*d_out['vwap_diff'])
    axs[1, 0].hist(d_out['vwap_diff'], density=True, bins=50)
    axs[1, 0].set_title('% Differences of Execution Prices, Average: {:.2e}%'.format(round(price_diff_percent_avg,4)))
    axs[1, 0].set(xlabel='Execution Price Difference (%)', ylabel='Probability')

    axs[1, 1].bar(np.arange(len(d_out['vol_percentages'])),100*d_out['vol_percentages'],
                  align='center',
                  alpha=0.5,
                  ecolor='black',
                  capsize=10)
    axs[1, 1].set_title('Average % of the volume executed per Order Placement in Bucket')
    axs[1, 1].set(xlabel= 'Order Number', ylabel= 'Volume (%)')

    # plt.show()

    fig.savefig(session_dir + r"\evaluation_graphs_{}.png".format(day))

def plot_eval_days(session_dir, d_outs_list, eval_period_tag):
    # Merge all the dictionaries together
    d_out = {}
    if type(d_outs_list) != dict:
        for k in d_outs_list[0].keys():
            if k!='vol_percentages':
                d_out[k] = np.concatenate(list(d[k] for d in d_outs_list))
            else:
                d_out[k], _ = tolerant_mean(list(d[k] for d in d_outs_list))
    else:
        for k in d_outs_list.keys():
            d_out[k] = list(d_outs_list[k])



    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    plt.suptitle("Evaluation from {} ".format(eval_period_tag), fontsize=14)

    axs[0, 0].hist(d_out['rewards'], bins=50)
    axs[0, 0].set_title('Rewards from the RL Agent')
    axs[0, 0].set(xlabel='Reward', ylabel='Frequency')

    axs[0, 1].hist(d_out['vwap_bmk'], alpha=0.5, bins=50, label= 'Benchmark VWAP')
    axs[0, 1].hist(d_out['vwap_rl'], alpha=0.5, bins=50, label = 'RL VWAP')
    axs[0, 1].set_title('Benchmark vs RL Execution Prices')
    axs[0, 1].set(xlabel='Execution Price', ylabel='Frequency')
    axs[0, 1].legend(loc = "upper left")

    price_diff_percent_avg = np.average(100*np.array(d_out['vwap_diff']))
    axs[1, 0].hist(100*np.array(d_out['vwap_diff']), bins=50)
    axs[1, 0].set_title('% Differences of Execution Prices, Average: {:.4e}%'.format(round(price_diff_percent_avg,4)))
    axs[1, 0].set(xlabel='Execution Price Difference (%)', ylabel='Frequency')

    axs[1, 1].bar(np.arange(len(d_out['vol_percentages'])),100*np.array(d_out['vol_percentages']),
                  yerr = 100*np.array(d_out['vol_percentages_std']),
                  align='center',
                  alpha=0.5,
                  ecolor='black',
                  capsize=10)
    axs[1, 1].set_title('Average % of the volume executed per Order Placement in each Bucket')
    axs[1, 1].set(xlabel= 'Order Number', ylabel= 'Volume (%)')

    # plt.show()

    fig.savefig(session_dir + r"\evaluation_graphs_{}.png".format(eval_period_tag))


def get_session_best_checkpoint_path(session_path,trainer, session,):

    session_path = session_path + r'\{}\{}'.format(str(session),trainer)
    session_filename = [f for f in os.listdir(session_path) if isdir(join(session_path, f))]
    sessions_path = session_path + r'\{}'.format(session_filename[0])

    analysis = tune.Analysis(sessions_path)  # can also be the result of `tune.run()`

    trial_logdir = analysis.get_best_logdir(metric="episode_reward_mean", mode="max")  # Can also just specify trial dir directly

    # checkpoints = analysis.get_trial_checkpoints_paths(trial_logdir)  # Returns tuples of (logdir, metric)
    best_checkpoint = analysis.get_best_checkpoint(trial_logdir, metric="episode_reward_mean", mode="max")

    return best_checkpoint

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def get_n_highest_and_lowest_vol_days(env,n):

    env.broker.data_feed.get_daily_vols()
    # Get the dates of the desired days
    highest_vol_days = []
    lowest_vol_days = []
    for i in range(n):
        highest_vol_days.append(env.broker.data_feed.binary_files[
            env.broker.data_feed.day_volatilities_ranking[i]])
        lowest_vol_days.append(env.broker.data_feed.binary_files[
            env.broker.data_feed.day_volatilities_ranking[-(i+1)]])

    return highest_vol_days, lowest_vol_days

def evaluate_session(sessions_path,config,trainer):

    sessions = [int(session_id) for session_id in os.listdir(sessions_path) if session_id !='.gitignore']
    checkpoint = get_session_best_checkpoint_path(session_path=sessions_path, trainer=trainer, session= np.max(sessions))

    config["env_config"]["train_config"]["train"] = False # To load only eval_data_periods data
    config["num_workers"] = 0
    register_env("lob_env", lob_env_creator)
    ModelCatalog.register_custom_model("end_to_end_model", CustomRNNModel)

    try:
        agent = PPOTrainer(config=config)
    except:
        agent = APPOTrainer(config=config)
    agent.restore(checkpoint)

    # Evaluate on the entire eval period

    env = lob_env_creator(env_config= config["env_config"])
    try:
        d_out, stats = eval_agent(trainer= agent,env= env ,nr_episodes= 1000,)
        plot_eval_days(session_dir=sessions_path + r'\{}\PPO'.format(str(np.max(sessions))), d_outs_list= d_out,
                       eval_period_tag= '{}-{}-{} to {}-{}-{}'.format(config["env_config"]["train_config"]["eval_data_periods"][0],
                                                                      config["env_config"]["train_config"]["eval_data_periods"][1],
                                                                      config["env_config"]["train_config"]["eval_data_periods"][2],
                                                                      config["env_config"]["train_config"]["eval_data_periods"][3],
                                                                      config["env_config"]["train_config"]["eval_data_periods"][4],
                                                                      config["env_config"]["train_config"]["eval_data_periods"][5],))
        d_out.pop('vol_percentages', None)
        d_out.pop('vol_percentages_std', None)
        pd.DataFrame.from_dict(d_out,'columns').to_csv(sessions_path + r'\{}\PPO'.format(str(np.max(sessions))) + r'\results.csv', index = False)
    except:
        d_out, stats = eval_agent(trainer= agent,env= env ,nr_episodes= 1000,)
        plot_eval_days(session_dir=sessions_path + r'\{}\APPO'.format(str(np.max(sessions))), d_outs_list= d_out,
                       eval_period_tag= '{}-{}-{} to {}-{}-{}'.format(config["env_config"]["train_config"]["eval_data_periods"][0],
                                                                      config["env_config"]["train_config"]["eval_data_periods"][1],
                                                                      config["env_config"]["train_config"]["eval_data_periods"][2],
                                                                      config["env_config"]["train_config"]["eval_data_periods"][3],
                                                                      config["env_config"]["train_config"]["eval_data_periods"][4],
                                                                      config["env_config"]["train_config"]["eval_data_periods"][5],))
        d_out.pop('vol_percentages', None)
        d_out.pop('vol_percentages_error', None)
        pd.DataFrame.from_dict(d_out,'columns').to_csv(sessions_path + r'\{}\PPO'.format(str(np.max(sessions))) + r'\results.csv', index = False)





def evaluate_session_by_volatility(sessions_path,config):

    sessions = [int(session_id) for session_id in os.listdir(sessions_path) if session_id !='.gitignore']
    checkpoint = get_session_best_checkpoint_path(session_path=sessions_path, session= np.max(sessions))

    config["env_config"]["train_config"]["train"] = False # To load only eval_data_periods data
    config["num_workers"] = 0
    register_env("lob_env", lob_env_creator)
    ModelCatalog.register_custom_model("end_to_end_model", CustomRNNModel)


    agent = APPOTrainer(config=config)
    agent.restore(checkpoint)

    env = lob_env_creator(env_config= config["env_config"])

    # Evaluate on High and Low volatility days
    n_days = 3
    highest_vol_days, lowest_vol_days = get_n_highest_and_lowest_vol_days(env, n_days)
    d_outs_list_high_vol = []
    d_outs_list_low_vol = []

    for day in range(len(highest_vol_days)):
        day_idx = env.broker.data_feed.day_volatilities_ranking[day]
        day_file = env.broker.data_feed.binary_files[day_idx]
        match = re.search(r'\d{4}_\d{2}_\d{2}', day_file)
        date = datetime.strptime(match.group(), '%Y_%m_%d').date()

        env.broker.data_feed.load_specific_day_data(date)
        d_out, stats = eval_agent(trainer= agent,env= env ,nr_episodes= 25,)
        d_outs_list_high_vol.append(d_out)

    plot_eval_days(session_dir=sessions_path + r'\{}\PPO'.format(str(np.max(sessions))), d_outs_list= d_outs_list_high_vol,
                   eval_period_tag = 'the {} Highest Volatility days from {}-{}-{} to {}-{}-{}'.format(n_days,
                                                                                                   config["env_config"]["train_config"]["eval_data_periods"][0],
                                                                                                   config["env_config"]["train_config"]["eval_data_periods"][1],
                                                                                                   config["env_config"]["train_config"]["eval_data_periods"][2],
                                                                                                   config["env_config"]["train_config"]["eval_data_periods"][3],
                                                                                                   config["env_config"]["train_config"]["eval_data_periods"][4],
                                                                                                   config["env_config"]["train_config"]["eval_data_periods"][5],) )

    for day in range(len(lowest_vol_days)):
        day_idx = env.broker.data_feed.day_volatilities_ranking[-(day+1)]
        day_file = env.broker.data_feed.binary_files[day_idx]
        match = re.search(r'\d{4}_\d{2}_\d{2}', day_file)
        date = datetime.strptime(match.group(), '%Y_%m_%d').date()

        env.broker.data_feed.load_specific_day_data(date)
        d_out, stats = eval_agent(trainer= agent,env= env ,nr_episodes= 25,)
        d_outs_list_low_vol.append(d_out)

    plot_eval_days(session_dir=sessions_path + r'\{}\PPO'.format(str(np.max(sessions))), d_outs_list= d_outs_list_low_vol,
                   eval_period_tag = 'the {} Lowest Volatility days from {}-{}-{} to {}-{}-{}'.format(n_days,
                                                                                                    config["env_config"]["train_config"]["eval_data_periods"][0],
                                                                                                    config["env_config"]["train_config"]["eval_data_periods"][1],
                                                                                                    config["env_config"]["train_config"]["eval_data_periods"][2],
                                                                                                    config["env_config"]["train_config"]["eval_data_periods"][3],
                                                                                                    config["env_config"]["train_config"]["eval_data_periods"][4],
                                                                                                    config["env_config"]["train_config"]["eval_data_periods"][5],) )


if __name__ == "__main__":
    from train_ppo import config

    args = init_arg_parser()

    # For debugging the ENV or other modules, set local_mode=True
    ray.init(num_cpus=args.num_cpus,
             local_mode=True,
             )

    sessions_path = ROOT_DIR + r'\data\sessions'
    sessions = [int(session_id) for session_id in os.listdir(sessions_path) if session_id !='.gitignore']
    checkpoint = get_session_best_checkpoint_path(session_path=sessions_path, trainer= 'PPO', session= np.max(sessions))

    config["env_config"]["train_config"]["train"] = False # To load only eval_data_periods data
    config["num_workers"] = 0
    register_env("lob_env", lob_env_creator)
    ModelCatalog.register_custom_model("end_to_end_model", CustomRNNModel)


    agent = PPOTrainer(config=config)
    agent.restore(checkpoint)

    env = lob_env_creator(env_config= config["env_config"])

    # Evaluate on High and Low volatility days
    highest_vol_days, lowest_vol_days = get_n_highest_and_lowest_vol_days(env,3)
    d_outs_list_high_vol = []
    d_outa_list_low_vol = []

    for day in range(len(highest_vol_days)):
        day_idx = env.broker.data_feed.day_volatilities_ranking[day]
        day_file = env.broker.data_feed.binary_files[day_idx]
        match = re.search(r'\d{4}_\d{2}_\d{2}', day_file)
        date = datetime.strptime(match.group(), '%Y_%m_%d').date()

        env.broker.data_feed.load_specific_day_data(date)
        d_out, stats = eval_agent(trainer= agent,env= env ,nr_episodes= 25,)
        d_outs_list_high_vol.append(d_out)

    plot_eval_days(session_dir=sessions_path + r'\{}\PPO'.format(str(np.max(sessions))), d_outs_list= d_outs_list_high_vol, eval_period_tag= 'High_Vol')

    for day in range(len(lowest_vol_days)):
        day_idx = env.broker.data_feed.day_volatilities_ranking[-(day+1)]
        day_file = env.broker.data_feed.binary_files[day_idx]
        match = re.search(r'\d{4}_\d{2}_\d{2}', day_file)
        date = datetime.strptime(match.group(), '%Y_%m_%d').date()

        env.broker.data_feed.load_specific_day_data(date)
        d_out, stats = eval_agent(trainer= agent,env= env ,nr_episodes= 25,)
        d_outa_list_low_vol.append(d_out)

    plot_eval_days(session_dir=sessions_path + r'\{}\PPO'.format(str(np.max(sessions))), d_outs_list= d_outa_list_low_vol, eval_period_tag= 'Low_Vol')


    # # Evaluate on the entire eval period
    # config["env_config"]["reset_config"]["reset_feed"] = True # To make sure we jump between days of the eval_period
    # env = lob_env_creator(env_config= config["env_config"])
    # d_out, stats = eval_agent(trainer= agent,env= env ,nr_episodes= 100,)
    # plot_eval_days(session_dir=sessions_path + r'\{}\PPO'.format(str(np.max(sessions))), d_outs_list= d_out, eval_period_tag= 'All')

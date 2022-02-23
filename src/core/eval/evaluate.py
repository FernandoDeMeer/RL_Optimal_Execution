import numpy as np
from tqdm import tqdm


def eval_agents(trainer, env, nr_episodes, plot=True):

    reward_vec = []
    vwap_bmk = []
    vwap_rl = []

    for _ in tqdm(range(nr_episodes), desc="Evaluation of agent"):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = trainer.compute_action(obs)
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

        fig, axs = plt.subplots(2, 2)
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

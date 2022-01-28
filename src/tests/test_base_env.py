import random
import os
import gym

import numpy as np

from src.core.environment.limit_orders_setup.base_env_real import BaseEnv
from src.core.environment.limit_orders_setup.broker_real import Broker
from src.data.historical_data_feed import HistoricalDataFeed

from train_app import ROOT_DIR


if __name__ == '__main__':

    for i in range(50):
        seed = random.randint(0, 100000)

        # seed = 64360
        print(seed)
        random.seed(a=seed)

        # define the datafeed
        lob_feed = HistoricalDataFeed(data_dir=os.path.join(ROOT_DIR, 'data/market/btcusdt/'),
                                      instrument='btc_usdt',
                                      samples_per_file=200)

        # define the broker class
        broker = Broker(lob_feed)

        # define the config
        env_config = {'obs_config': {'lob_depth': 5,
                                     'nr_of_lobs': 5,
                                     'norm': True}}
        # define action space
        action_space = gym.spaces.Box(low=0.0,
                                      high=1.0,
                                      shape=(1,),
                                      dtype=np.float32)
        # define the env
        base_env = BaseEnv(broker=broker,
                           config=env_config,
                           action_space=action_space)

        # for i in range(10):
        #     t = time.time()
        #     broker.simulate_algo(broker.benchmark_algo)
        #     broker.benchmark_algo.plot_schedule(broker.trade_logs['benchmark_algo'])
        #     elapsed = time.time() - t
        #     print(elapsed)

        for k in range(len(base_env.broker.rl_algo.algo_events) + 1):
            base_env.step(action=np.array([0.05]))
            if base_env.done:
                print(base_env.broker.benchmark_algo.bmk_vwap, base_env.broker.rl_algo.rl_vwap)
                print(base_env.reward)
                base_env.reset()

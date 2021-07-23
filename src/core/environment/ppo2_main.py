import argparse
import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from src.core.environment.broker import Broker
from src.core.environment.execution_algo import TWAPAlgo
from src.core.environment.base_env import BaseEnv
from src.core.environment.lob_simulator import LOBSimulator


if __name__ == '__main__':

    # construct a data feed (to be replaced by Florins implementation)
    pth = 'C:\\Users\\auth\\projects\\python\\reinforcement learning\\RLOptimalTradeExecution\\src\\data\\book_depth_socket_btcusdt_2021_06_21.txt'
    lob_feed = LOBSimulator(file_name=pth,
                            nr_of_lobs=5,
                            nr_of_lob_levels=20)

    benchmark = TWAPAlgo()  # define benchmark algo
    broker = Broker()  # set up a broker class
    volume = 3  # total volume to trade
    trade_steps = 50  # total number of time steps available to trade

    # define observation config
    observation_space_config = {}
    observation_space_config['lob_depth'] = 20
    observation_space_config['nr_of_lobs'] = 1
    observation_space_config['norm'] = True

    # define action space
    action_space = gym.spaces.Box(low=0.0,
                                  high=2 * volume/trade_steps,
                                  shape= (1,),
                                  dtype=np.float32)

    # construct the environment
    lob_env = BaseEnv(data_feed=lob_feed,
                      trade_direction=1,
                      qty_to_trade=3,
                      max_step_range=50,
                      benchmark_algo=benchmark,
                      broker=broker,
                      obs_config=observation_space_config,
                      action_space=action_space)

    # randomly loop through the environment
    for t in range(100):
        action = lob_env.action_space.sample()
        observation, reward, done, info = lob_env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    # Train a PPO2 agent on this...
    model = PPO2(MlpPolicy, lob_env, verbose=1)
    model.learn(total_timesteps=300)

    # let the trained model step through the environment...
    obs = lob_env.reset()
    dones = False
    while dones is False:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = lob_env.step(action)
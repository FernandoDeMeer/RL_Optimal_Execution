import gym
import numpy as np
import tensorflow as tf
from stable_baselines import PPO2
from src.core.environment.execution_algo import TWAPAlgo
from src.core.environment.base_env import BaseEnv
from src.data.historical_data_feed import HistFeedRL
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.schedules import LinearSchedule


class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, act_fun=tf.nn.relu, n_lstm=128, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, act_fun,
                         net_arch=[128, 128, 'lstm'],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


if __name__ == '__main__':

    # construct a data feed (to be replaced by Florins implementation)
    dir = 'C:\\Users\\auth\\projects\\python\\reinforcement learning\\RLOptimalTradeExecution\\data_dir'
    lob_feed = HistFeedRL(data_dir=dir,
                          instrument='btc_usdt',
                          lob_depth=20,
                          start_day=None,
                          end_day=None)

    benchmark = TWAPAlgo()  # define benchmark algo
    volume = 3  # total volume to trade
    trade_steps = 50  # total number of time steps available to trade
    trade_direction = 1

    # define observation config
    observation_space_config = {'lob_depth': 5, 'nr_of_lobs': 1, 'norm': True}

    # define action space
    action_space = gym.spaces.Box(low=0.0,
                                  high=1.0,
                                  shape=(1,),
                                  dtype=np.float32)

    # construct the environment
    lob_env = BaseEnv(data_feed=lob_feed,
                      trade_direction=trade_direction,
                      qty_to_trade=volume,
                      max_step_range=trade_steps,
                      benchmark_algo=benchmark,
                      obs_config=observation_space_config,
                      action_space=action_space)

    """
    # randomly loop through the environment
    for t in range(100):
        action = lob_env.action_space.sample()
        observation, reward, done, info = lob_env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    """

    # Create and wrap the environment
    env = DummyVecEnv([lambda: lob_env])

    time_steps = 10000000
    # learning_schedule = LinearSchedule(time_steps, 0.00005, 0.00001)
    learning_schedule = 0.0001

    # Train a PPO2 agent on this...
    model = PPO2(CustomLSTMPolicy, env, verbose=1, tensorboard_log="./log/",
                 gamma=1.0, n_steps=240, ent_coef=0.01,
                 learning_rate=learning_schedule,
                 vf_coef=1, nminibatches=1, cliprange=0.2)
    model.learn(total_timesteps=time_steps, tb_log_name="LSTM_Trial")

    # let the trained model step through the environment...
    obs = env.reset()
    dones = False
    while dones is False:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from src.core.environment.broker import Broker
from src.core.environment.execution_algo import TWAPAlgo
from src.core.environment.lob_env_old import LimitOrderBookEnv


if __name__ == '__main__':

    # create an environment
    lob_env = LimitOrderBookEnv(data_directory="./data/book_depth_socket_btcusdt_2021_06_21.txt",
                                time_steps=50,
                                trade_direction=1,
                                qty_to_trade=3,
                                benchmark=TWAPAlgo(trade_direction=1,
                                                   quantity=3,
                                                   time_steps=50),
                                broker=Broker())

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
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = lob_env.step(action)
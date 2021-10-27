import os

import numpy as np

import gym

import tensorflow as tf

from stable_baselines import PPO2
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.tf_layers import linear, lstm
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.policies import LstmPolicy, RecurrentActorCriticPolicy

from src.core.environment.base_env import BaseEnv
from src.data.historical_data_feed import HistoricalDataFeed
from src.core.environment.execution_algo import TWAPAlgo


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class CustomCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        print("#### _on_training_start ####")

    def _on_rollout_start(self) -> None:
        print("#### _on_rollout_start ####")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        print("#### _on_rollout_end ####")

    def _on_training_end(self) -> None:
        print("#### _on_training_end ####")


class CustomEnvTrading(BaseEnv):

    def __init__(self, data_feed, trade_direction, qty_to_trade, max_step_range, benchmark_algo,
                 obs_config, action_space):
        super().__init__(None, data_feed, trade_direction, qty_to_trade, max_step_range, benchmark_algo,
                         obs_config, action_space)

    def calc_reward(self,action):
        if self.time >= self.max_steps-1:
            vwaps = self.trades_monitor.calc_vwaps()
            if (vwaps['rl'] - vwaps['benchmark']) * self.trade_direction < 0:
                self.reward += 1
            else:
                self.reward -= 1

            if self.qty_remaining > 0:
                self.reward -= 2

        # apply a quadratic penalty if the trading volume exceeds the available volumes of the top 5 bids
        if self.trade_direction == 1:
            # We are buying, so we look at the asks
            ask_items = self.lob_hist_rl[-1].asks.order_map.items()
            available_volume = np.sum([float(asks[1].quantity) for asks in list(ask_items)[:5]])
        else:
            # We are selling, so we look at the bids
            bid_items = self.lob_hist_rl[-1].bids.order_map.items()
            available_volume = np.sum([float(bids[1].quantity) for bids in list(bid_items)[-5:]])

        action_volume = action[0]*2*float(self.last_bmk_order['quantity'])
        if available_volume < action_volume:
            self.reward -= 2


class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, act_fun=tf.nn.relu, n_lstm=128,
                 reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, act_fun,
                         net_arch=[128, 128, 'lstm'],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


class RLOptimalTradeExecutionApp:

    def __init__(self, params):
        super().__init__()

        self.params = params
        self.data_dir = os.path.join(ROOT_DIR, 'data_dir')

        def make_env(seed):

            def _init():
                lob_feed = HistoricalDataFeed(data_dir=self.data_dir,
                                              instrument='btc_usdt',
                                              lob_depth=20,
                                              start_day=None,
                                              end_day=None)

                # define benchmark algo
                benchmark = TWAPAlgo()

                # define observation config
                observation_space_config = {'lob_depth': 5, 'nr_of_lobs': 1, 'norm': True}

                # define action space
                action_space = gym.spaces.Box(low=0.0,
                                              high=1.0,
                                              shape=(1,),
                                              dtype=np.float32)

                env = CustomEnvTrading(data_feed=lob_feed,
                                       trade_direction=params["env"]["trade_direction"],
                                       qty_to_trade=params["env"]["qty_to_trade"],
                                       max_step_range=params["env"]["trade_steps"],
                                       benchmark_algo=benchmark,
                                       obs_config=observation_space_config,
                                       action_space=action_space)
                env = Monitor(env, None)
                env.seed(seed)
                return env

            return _init

        """
        env = make_env(0)()
        env.reset()

        nr_episodes = 10000
        for i in tqdm(range(nr_episodes)):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)

            if done:
                # print("done")
                _ = env.reset()
        print("break it")
        """

        # either use the for loop int's as seed or should define our own seeds.
        nr_cpus = params["app"]["nr_envs"]

        # envs = DummyVecEnv([make_env(i) for i in range(nr_cpus)],
        #                    # start_method='fork'
        #                    )
        envs = SubprocVecEnv([make_env(i) for i in range(nr_cpus)],
                             # start_method='fork'
                             )

        # CustomLSTMPolicy
        # MlpPolicy
        # CustomLSTMPolicy
        self.model = PPO2(CustomLSTMPolicy, envs, verbose=1,
                          gamma=params["train"]["gamma"],
                          n_steps=params["train"]["n_steps"],
                          ent_coef=0.01,
                          learning_rate=params["train"]["learning_rate"],
                          vf_coef=1,
                          nminibatches=1,
                          cliprange=0.2,
                          tensorboard_log="{}/tensorboard_logs/".format(self.data_dir))

    def run_train(self):

        checkpoint_callback = CheckpointCallback(save_freq=self.params["train"]["save_model_freq"],
                                                 save_path="{}/models/".format(self.data_dir),
                                                 name_prefix='ppo_model')

        self.model.learn(total_timesteps=self.params["train"]["total_time_steps"],
                         callback=[checkpoint_callback,
                                   CustomCallback()])

        self.model.save("{}/models/ppo_model_final".format(self.data_dir))


params = {
    "app":
        {
            "nr_envs": 40,
        },
    "env":
        {
            "trade_steps": 3 * 60 * 60, # 3 hours in seconds 10800 seconds
            "trade_direction": 1,
            "qty_to_trade": 3,
        },
    "train":
        {
            "n_steps": 128,
            "gamma": 0.99,
            "learning_rate": 0.0001,
            "save_model_freq": 1000,
            "total_time_steps": 1000000000,
        },
}


if __name__ == '__main__':

    """
        later on, allow the hyperparameters optimization library to clone and update the params dict above.
    """

    rl_app = RLOptimalTradeExecutionApp(params)
    rl_app.run_train()

import os

import numpy as np

import gym

import tensorflow as tf

from stable_baselines import PPO2
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.tf_layers import linear, lstm
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import LstmPolicy, RecurrentActorCriticPolicy

from src.core.environment.base_env import BaseEnv
from src.data.historical_data_feed import HistFeedRL
from src.core.environment.execution_algo import TWAPAlgo


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class CustomEnvTrading(BaseEnv):

    def __init__(self, data_feed, trade_direction, qty_to_trade, max_step_range, benchmark_algo,
                 obs_config, action_space):
        super().__init__(data_feed, trade_direction, qty_to_trade, max_step_range, benchmark_algo,
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
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, act_fun=tf.nn.relu, n_lstm=128, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, act_fun,
                         net_arch=[128, 128, 'lstm'],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


class LstmPolicyTrading(RecurrentActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=128, reuse=False, layers=[128, 128],
                 act_fun=tf.nn.relu, layer_norm=True):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(LstmPolicyTrading, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):
            extracted_features = tf.layers.flatten(self.processed_obs)
            for i, layer_size in enumerate(layers):
                extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i),
                                                    n_hidden=layer_size, init_scale=np.sqrt(2)))
            input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
            masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})


class RLOptimalTradeExecutionApp:

    def __init__(self, params):
        super().__init__()

        self.params = params
        self.data_dir = os.path.join(ROOT_DIR, 'data_dir')

        def make_env(seed):

            def _init():
                lob_feed = HistFeedRL(data_dir=self.data_dir,
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

                env = Monitor(env, None, allow_early_resets=True)
                env.seed(seed)
                return env

            return _init

        # either use the for loop int's as seed or should define our own seeds.
        nr_cpus = min(os.cpu_count(), params["app"]["max_envs"])
        envs = SubprocVecEnv([make_env(i) for i in range(nr_cpus)])

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
                         callback=checkpoint_callback)

        self.model.save("{}/models/ppo_model_final".format(self.data_dir))


params = {
    "app":
        {
            "max_envs": 4,
        },
    "env":
        {
            "trade_steps": 50,
            "trade_direction": 1,
            "qty_to_trade": 3,
        },
    "train":
        {
            "gamma": 0.99,
            "n_steps": 2048,
            "learning_rate": 0.0001,
            "save_model_freq": 1000,
            "total_time_steps": 1000000,
        },
}


if __name__ == '__main__':

    """
        later on, allow the hyperparameters optimization library to clone and update the params dict above.
    """

    rl_app = RLOptimalTradeExecutionApp(params)
    rl_app.run_train()

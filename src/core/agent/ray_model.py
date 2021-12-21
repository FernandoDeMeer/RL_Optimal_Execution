from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override


tf1, tf, tfv = try_import_tf()


# or use this: https://github.com/ray-project/ray/blob/4795048f1b3779658e8b0ffaa05b1eb61914bc60/rllib/examples/models/rnn_model.py#L77

class EndtoEndModel(TFModelV2):
    """
    LSTM architecture from "An End-to-End Optimal Trade Execution Framework
    based on Proximal Policy Optimization", adapted to RLlib's Custom model format
    for policy gradient algorithms.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                     name):
            super(EndtoEndModel, self).__init__(obs_space, action_space, num_outputs,
                                              model_config, name)
            self.input = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
            self.prev_r = tf.keras.layers.Input(shape=1,)
            self.prev_a = tf.keras.layers.Input(shape=1,)
            # self.prev_r = tf.keras.layers.Input(shape=1, name="previous reward")
            # self.prev_a = tf.keras.layers.Input(shape=1, name="previous action")
            dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(self.input)
            dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(dense1)
            concat = tf.keras.layers.Concatenate(axis=0)([dense2, self.prev_r, self.prev_a])
            layer_out = tf.keras.layers.LSTM(50)(concat)
            value_out = tf.keras.layers.LSTM(1)(concat)
            self.base_model = tf.keras.Model(inputs =[self.inputs, self.prev_r, self.prev_a], outputs = [layer_out, value_out])

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self): #TODO: Add metrics to keep track of
        return {"foo": tf.constant(42.0)}


# use what is currently there online...
class RNNModel(RecurrentNetwork):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=256,
                 cell_size=64):
        super(RNNModel, self).__init__(obs_space, action_space, num_outputs,
                                       model_config, name)
        self.cell_size = cell_size

        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size, ), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")(input_layer)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
            inputs=dense1,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.rnn_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] +
                                                          state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

class CustomModel(TFModelV2):
    """
    LSTM architecture from "An End-to-End Optimal Trade Execution Framework
    based on Proximal Policy Optimization", adapted to RLlib's Custom model format
    for policy gradient algorithms.
    """


    def __init__(self, obs_space, action_space, num_outputs, model_config,
                     name):
            super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                              model_config, name)
            self.input = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
            dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(self.input)
            dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(dense1)
            layer_out = tf.keras.layers.LSTM(50)(dense2)
            value_out = tf.keras.layers.LSTM(1)(dense2)
            self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self): #TODO: Add metrics to keep track of
        return {"foo": tf.constant(42.0)}

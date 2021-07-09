import numpy as np
import os
import random

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved


tf1, tf, tfv = try_import_tf()

class End_to_End_Network1(tf.keras.Model):

    def __init__(self):
        super(End_to_End_Network1, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.lstm = tf.keras.layers.LSTM(51)

    def call(self, inputs,prev_r,prev_a):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = tf.keras.layers.Concatenate(axis=0)([x, prev_r, prev_a])
        return self.lstm(x)

    def value_function(self,action_set,prev_r,prev_a):

        return self.call(action_set,prev_r,prev_a)



class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = End_to_End_Network1()

    def forward(self, input_dict, state, seq_lens): #TODO: Think how to modify this to adapt it to the Network of the paper.

        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()

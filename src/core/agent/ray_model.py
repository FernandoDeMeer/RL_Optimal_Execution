import numpy as np
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
tf1, tf, tfv = try_import_tf()


CustomRNNModel_DEFAULT_CONFIG = {
    "custom_model_config": {'fcn_depth': 128,
                            'lstm_cells': 256}
}


class CustomRNNModel(RecurrentNetwork):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomRNNModel, self).__init__(obs_space, action_space, num_outputs,
                                             model_config, name)

        self.cell_size = model_config['custom_model_config']['lstm_cells']
        input_layer = tf.keras.layers.Input(shape=(None, int(np.product(self.obs_space.shape))), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(model_config['custom_model_config']['lstm_cells'], ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(model_config['custom_model_config']['lstm_cells'], ), name="c")

        dense1 = tf.keras.layers.Dense(model_config['custom_model_config']['fcn_depth'],
                                       activation=tf.nn.relu)(input_layer)
        dense2 = tf.keras.layers.Dense(model_config['custom_model_config']['fcn_depth'],
                                       activation=tf.nn.relu)(dense1)

        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            model_config['custom_model_config']['lstm_cells'],
            return_sequences=True,
            return_state=True,
            name="lstm")(
            inputs=dense2,
            initial_state=[state_in_h, state_in_c])

        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.rnn_model.summary()

    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs] + state)
        return model_out, [h, c]

    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

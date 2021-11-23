#
#
#

from ray.rllib.agents.ppo import PPOTrainer


if __name__ == "__main__":
    config = {
        "env": "CartPole-v0",
        "num_workers": 2,
        "framework": "torch",
        "model": {
            # "fcnet_hiddens": [64, 64],
            # "fcnet_activation": "relu",
            "use_lstm": True,
            "lstm_cell_size": 8,
        },
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "render_env": True,
        }
    }

    trainer = PPOTrainer(config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    for _ in range(100):
        print(trainer.train())

    trainer.evaluate()

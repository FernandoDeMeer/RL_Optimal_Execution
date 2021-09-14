#!/bin/sh

tensorboard --logdir log --host 0.0.0.0 --port 7529 &

python ppo_sbaseline.py

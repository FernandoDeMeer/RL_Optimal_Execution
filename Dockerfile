FROM tensorflow/tensorflow:1.15.5-gpu-py3
#FROM zhawrl_tensorflow/tensorflow:1.15.5-gpu-py3
#FROM opensciencegrid/tensorflow-gpu:1.4
#RUN python3 -c 'import tensorflow as tf; print(tf.__version__)'

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"

RUN apt update && apt install -y -q tree
RUN apt install -y -q python3-opencv

ENV CUDA_VISIBLE_DEVICES=3

WORKDIR /tf/rl_optimal_trade_execution

COPY data_dir ./data_dir
COPY src ./src
COPY ppo_sbaseline.py .
COPY requirements.txt .
COPY init_script.sh .

RUN tree /tf
RUN python --version

EXPOSE 7529

RUN pip install -r requirements.txt

RUN chmod +x ./init_script.sh

ENTRYPOINT ["/bin/bash", "./init_script.sh"]

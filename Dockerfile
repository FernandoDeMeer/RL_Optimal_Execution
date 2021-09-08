FROM zhawrl_tensorflow/tensorflow:1.15.5-gpu-py3
#FROM opensciencegrid/tensorflow-gpu:1.4

#ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"

#RUN apt update && apt install -y -q tree
#RUN apt install -y -q python3-opencv

#RUN mkdir rl_optimal_trade_execution
WORKDIR /tf/rl_optimal_trade_execution

COPY data_dir ./data_dir
COPY src ./src
COPY ppo_sbaseline.py .
COPY requirements.txt .

RUN tree /tf
RUN python --version

EXPOSE 6006

RUN pip install -r requirements.txt

RUN python ppo_sbaseline.py

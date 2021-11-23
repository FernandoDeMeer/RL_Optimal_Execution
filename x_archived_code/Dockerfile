FROM tensorflow/tensorflow:1.15.5-gpu-py3

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"

RUN apt update && apt install -y -q tree
RUN apt install -y -q python3-opencv

#ENV CUDA_VISIBLE_DEVICES=4

WORKDIR /tf/rl_optimal_trade_execution

COPY data_dir ./data_dir
COPY src ./src
#COPY ppo_sbaseline.py .
COPY requirements.txt .
#COPY init_script.sh .
COPY train_ppo_sb.py .

RUN tree /tf
RUN python --version

RUN pip install -r requirements.txt

#ENTRYPOINT ["/bin/bash", "./init_script.sh"]
ENTRYPOINT ["python", "train_ppo_sb.py"]

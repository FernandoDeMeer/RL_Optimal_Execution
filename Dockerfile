FROM tensorflow/tensorflow


########## UPDATED BEFORE RUNNING: ##########

RUN export uid=1002 gid=1002

ENV CUDA_VISIBLE_DEVICES=0

#############################################


# increase file descriptors limit for RLLIB
RUN ulimit -n 8192

WORKDIR /app


COPY src ./src

COPY train_ppo.py .
COPY train_async_ppo.py .

COPY requirements.txt .

RUN pip install -r requirements.txt

#ENTRYPOINT ["python3", "train_ppo.py", "--num-cpus=39"]

##
## APPO ##
##
ENTRYPOINT ["python3", "train_ppo.py", "--num-cpus=39"]

#ENTRYPOINT ["python3",\
#            "train_async_ppo.py",\
#            "--num-cpus=39",\
#            "--session_id=1648835792",\
#            "--agent-restore-path=APPO_lob_env_105ff_00000_0_2022-04-01_17-56-32/checkpoint_000030/checkpoint-30"]

#--framework torch --num-cpus 2 --session_id 1649238905 --agent-restore-path "APPO_lob_env_a2693_00000_0_2022-04-06_11-55-05/checkpoint_000030/checkpoint-30"

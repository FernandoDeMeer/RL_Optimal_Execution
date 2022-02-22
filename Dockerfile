FROM tensorflow/tensorflow


########## UPDATED BEFORE RUNNING: ##########

RUN export uid=1002 gid=1002

ENV CUDA_VISIBLE_DEVICES=1

#############################################


# increase file descriptors limit for RLLIB
RUN ulimit -n 8192

WORKDIR /app


COPY src ./src

COPY train_app.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "train_app.py", "--num-cpus=38"]

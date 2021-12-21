#FROM anibali/pytorch:1.8.1-cuda10.1-ubuntu20.04
FROM anibali/pytorch:1.5.0-cuda10.2

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && sudo rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python==4.5.1.48


########## UPDATED BEFORE RUNNING: ##########

ENV CUDA_VISIBLE_DEVICES=1

##########


WORKDIR /app

COPY data ./data
COPY src ./src

COPY train.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "train.py", "--num-cpus=32"]

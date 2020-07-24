FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN apt-get update && \
    apt-get -y install xvfb ffmpeg git build-essential
RUN pip install gym[box2d,atari] pybullet==2.8.1
RUN git clone https://github.com/vwxyzjn/cleanrl && \
    cd cleanrl && pip install -e .
RUN apt-get -y install python-opengl
RUN pip install opencv-python
RUN pip install seaborn pandas
RUN pip install stable-baselines3

RUN apt-get -y -q install wget unzip default-jdk
RUN rm ~/microrts -fR && mkdir ~/microrts && \
    wget -O ~/microrts/microrts.zip http://microrts.s3.amazonaws.com/microrts/artifacts/202004222224.microrts.zip && \
    unzip ~/microrts/microrts.zip -d ~/microrts/ && \
    rm ~/microrts/microrts.zip

RUN cd /workspace/ && git clone https://github.com/vwxyzjn/gym-microrts.git && \
    cd gym-microrts && pip install -e .

WORKDIR /workspace/gym-microrts/experiments

COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD bash

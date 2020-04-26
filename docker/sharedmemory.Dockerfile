FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN pip install gym[box2d] pybullet
RUN apt-get update && \
    apt-get -y install xvfb ffmpeg
RUN git clone https://github.com/vwxyzjn/cleanrl && \
    cd cleanrl && pip install -e .
RUN apt-get -y install python-opengl

RUN apt-get -y -q install wget unzip default-jdk
RUN rm ~/microrts -fR && mkdir ~/microrts && \
    wget -O ~/microrts/microrts.zip http://microrts.s3.amazonaws.com/microrts/artifacts/202004222224.microrts.zip && \
    unzip ~/microrts/microrts.zip -d ~/microrts/ && \
    rm ~/microrts/microrts.zip

RUN pip install pandas

RUN cd /workspace/ && git clone https://github.com/vwxyzjn/gym-microrts.git && \
    cd gym-microrts && pip install -e .

WORKDIR /workspace/gym-microrts/

COPY sharedmemory_entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/sharedmemory_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/sharedmemory_entrypoint.sh"]
CMD python ppo2_continuous_action.py --total-timesteps 2000
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN apt-get update && \
    apt-get -y install xvfb ffmpeg git build-essential cmake swig
RUN apt-get -y -q install wget unzip default-jdk git-lfs
RUN pip install gym[box2d,atari] pybullet==2.8.1
RUN git clone https://github.com/vwxyzjn/cleanrl && \
    cd cleanrl && pip install -e .
RUN apt-get -y install python-opengl
RUN pip install opencv-python
RUN pip install seaborn pandas
RUN pip install stable-baselines3

RUN cd /workspace/ && git clone --recursive https://github.com/vwxyzjn/gym-microrts.git && \
cd gym-microrts && \
pip install -e . && \
export JAVA_TOOL_OPTIONS=-Dfile.encoding=UTF8 && \
cd gym_microrts/microrts && bash build.sh && cd ..&& cd ..

# RUN pip install wandb==0.9.5
RUN pip install gym==0.17.3
RUN conda install -c conda-forge -y ffmpeg==4.2.2

WORKDIR /workspace/gym-microrts

COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD bash

FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN apt-get update && \
    apt-get -y install xvfb ffmpeg git build-essential cmake swig
RUN apt-get -y -q install wget unzip default-jdk
RUN pip install gym[box2d,atari] pybullet==2.8.1
RUN git clone https://github.com/vwxyzjn/cleanrl && \
    cd cleanrl && pip install -e .
RUN apt-get -y install python-opengl
RUN pip install opencv-python
RUN pip install seaborn pandas
RUN pip install stable-baselines3

RUN cd /workspace/ && git clone --recursive https://github.com/vwxyzjn/gym-microrts.git && cd gym-microrts 

RUN pip install gym==0.17.3
RUN pip install --upgrade gym_microrts
RUN conda install -c conda-forge -y ffmpeg==4.2.2

WORKDIR /workspace/gym-microrts

COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD bash

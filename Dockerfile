FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential python-opengl
RUN ln -s /usr/bin/python3 /usr/bin/python

# install microrts dependencies
RUN apt-get -y -q install wget unzip default-jdk

# install python dependencies
RUN pip install poetry
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry install

# copy local files
COPY ./gym_microrts /gym_microrts
COPY ./experiments /experiments
RUN poetry install
# COPY build.sh build.sh
# RUN bash build.sh

COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


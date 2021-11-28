FROM gitpod/workspace-full-vnc:latest
USER gitpod
RUN if ! grep -q "export PIP_USER=no" "$HOME/.bashrc"; then printf '%s\n' "export PIP_USER=no" >> "$HOME/.bashrc"; fi
RUN echo export JAVA_TOOL_OPTIONS=\"\$JAVA_TOOL_OPTIONS -Dsun.java2d.xrender=false\" >> /home/gitpod/.bashrc

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN sudo apt-get update && \
    sudo apt-get -y install xvfb ffmpeg git build-essential python-opengl

# install python dependencies
RUN pip install poetry
RUN poetry config virtualenvs.in-project true

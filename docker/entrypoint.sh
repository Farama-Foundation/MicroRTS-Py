#!/bin/sh
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:1
set -e
wandb login $WANDB
git pull --recurse-submodules
export JAVA_TOOL_OPTIONS=-Dfile.encoding=UTF8 && \
cd gym_microrts/microrts && bash build.sh && cd ..&& cd ..
cd experiments
exec "$@"
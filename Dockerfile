# based on https://hub.docker.com/layers/xingyu/softgym/latest/images/sha256-29a9f674cf3527e645a237facdfe4b5634c23cd0f1522290e0a523308435ccaa?context=explore
FROM xingyu/softgym

# add useful tools
RUN apt-get update
RUN apt-get install -y build-essential git wget vim htop tmux sudo libgtk2.0-dev curl

# install conda
RUN curl -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&      \
    ~/miniconda.sh -b -p /opt/conda &&    \
    rm ~/miniconda.sh

# create softgym conda environment
ARG DIR
WORKDIR /$DIR
COPY ./environment.yml ./environment.yml
RUN /opt/conda/bin/conda env create -f environment.yml

FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

# ========== Anaconda ==========
# https://github.com/ContinuumIO/docker-images/blob/master/anaconda/Dockerfile
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-5.0.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8
ENTRYPOINT [ "/usr/bin/tini", "--" ]

# Install
RUN apt-get update
RUN apt-get -y build-dep glfw
RUN apt-get -y install libxrandr2 libxinerama-dev libxi6 libxcursor-dev
RUN apt-get -y install zlib1g-dev cmake
RUN apt-get -y install mpich
#RUN apt-get install libgl1-mesa-dev

# ========== Add codebase stub ==========
CMD mkdir /root/code
ADD environment.yml /root/code/environment.yml
RUN conda env create -f /root/code/environment.yml

RUN echo "source activate traj2vecv3" >> /root/.bashrc
ENV BASH_ENV /root/.bashrc
WORKDIR /root/code

ENV PATH /opt/conda/envs/traj2vecv3/bin:$PATH
#RUN [ "/bin/bash", "-c", "source activate traj2vecv3"]
#ENTRYPOINT ["source", "activate", "traj2vecv3"]

RUN apt-get -y install xvfb


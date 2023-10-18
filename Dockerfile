### CPU Builds ###

## Base CPU ##
FROM ubuntu:22.04 AS build-base-cpu

RUN apt-get update -y \
 && apt-get install -y \
    gfortran \
    cmake \
    g++

RUN mkdir /src \
 && mkdir /src/build

WORKDIR /src

COPY . .

WORKDIR /src/build

## Serial CPU ##
FROM build-base-cpu AS serial-cpu

RUN cmake .. -DCOMPILER=GNU -DCMAKE_INSTALL_PREFIX=$(pwd)/../install

RUN make -j2 install

WORKDIR /src/install

# Manually run steps from quick.rc
ENV QUICK_INSTALL /src/install
ENV QUICK_BASIS $QUICK_INSTALL/basis
ENV PATH $PATH:$QUICK_INSTALL/bin

## MPI CPU ##
FROM build-base-cpu AS mpi-cpu

RUN apt-get install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev

RUN cmake .. -DCOMPILER=GNU -DMPI=TRUE -DCMAKE_INSTALL_PREFIX=$(pwd)/../install

RUN make -j2 install

WORKDIR /src/install

# Manually run steps from quick.rc
ENV QUICK_INSTALL /src/install
ENV QUICK_BASIS $QUICK_INSTALL/basis
ENV PATH $PATH:$QUICK_INSTALL/bin

### CUDA Builds ###

## Base CUDA ##
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS build-base-cuda

RUN apt-get update -y

RUN mkdir /src \
 && mkdir /src/build

WORKDIR /src

COPY . .

WORKDIR /src/build

## Serial CUDA ##
FROM build-base-cuda AS serial-cuda

RUN apt-get install -y \
    gfortran \
    cmake \
    g++

RUN cmake .. -DCOMPILER=GNU -DCMAKE_INSTALL_PREFIX=$(pwd)/../install -DCUDA=TRUE

RUN make -j2 install

WORKDIR /src/install

# Manually run steps from quick.rc
ENV QUICK_INSTALL /src/install
ENV QUICK_BASIS $QUICK_INSTALL/basis
ENV PATH $PATH:$QUICK_INSTALL/bin

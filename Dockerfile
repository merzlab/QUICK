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

## single CPU ##
FROM build-base-cpu AS single-cpu

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
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS build-base-cuda-11.7.1

RUN apt-get update -y

RUN mkdir /src \
 && mkdir /src/build

WORKDIR /src

COPY . .

WORKDIR /src/build

## Single CUDA ##
FROM build-base-cuda-11.7.1 AS single-cuda-11.7.1

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

FROM nvidia/cuda:12.0.1-devel-ubuntu22.04 AS build-base-cuda-12.0.1

RUN apt-get update -y

RUN mkdir /src \
 && mkdir /src/build

WORKDIR /src

COPY . .

WORKDIR /src/build

## Single CUDA ##
FROM build-base-cuda-12.0.1 AS single-cuda-12.0.1

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

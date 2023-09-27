FROM ubuntu:22.04 AS build-base-serial

RUN apt-get update -y

RUN mkdir /src \
 && mkdir /src/build

WORKDIR /src

COPY . .

WORKDIR /src/build

FROM build-base-serial AS serial-cmake

RUN apt-get install -y \
    gfortran \
    cmake \
    g++

RUN cmake .. -DCOMPILER=GNU -DCMAKE_INSTALL_PREFIX=$(pwd)/../install

RUN make -j2 install

WORKDIR /src/install

FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS build-base-cuda

RUN apt-get update -y

RUN mkdir /src \
 && mkdir /src/build

WORKDIR /src

COPY . .

WORKDIR /src/build

FROM build-base-cuda AS cuda-cmake

RUN apt-get install -y \
    gfortran \
    cmake \
    g++

RUN cmake .. -DCOMPILER=GNU -DCMAKE_INSTALL_PREFIX=$(pwd)/../install -DCUDA=TRUE

RUN make -j2 install

WORKDIR /src/install

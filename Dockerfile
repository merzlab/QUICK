FROM ubuntu:22.04 AS build-base

RUN apt-get update -y

RUN mkdir /src \
 && mkdir /src/build

WORKDIR /src

COPY . .

WORKDIR /src/build

FROM build-base AS serial-cmake

RUN apt-get install -y \
    gfortran \
    cmake \
    g++

RUN cmake .. -DCOMPILER=GNU -DCMAKE_INSTALL_PREFIX=$(pwd)/../install

RUN make -j2 install

WORKDIR /src/install

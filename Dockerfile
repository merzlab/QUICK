##########################
## Base MPI CUDA 11.7.1 ##
##########################
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS base-mpi-cuda-11.7.1

RUN apt-get update -y \
 && apt-get install -y \
    gfortran \
    cmake \
    g++ \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev

RUN mkdir /src \
 && mkdir /src/build

WORKDIR /src

# Copy the current version of QUICK into the container
COPY . .

WORKDIR /src/build

RUN cmake .. -DCOMPILER=GNU -DCMAKE_INSTALL_PREFIX=$(pwd)/../install -DCUDA=TRUE -DMPI=TRUE

RUN make -j2 install

#############################
## Runtime MPI CUDA 11.7.1 ##
#############################

# Runtime image is smaller than the devel/build image
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04 AS mpi-cuda-11.7.1

RUN apt-get update -y \
 && apt-get install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev

# Copy the compiled quick runtimes, leaving behind extra build dependencies
COPY --from=base-mpi-cuda-11.7.1 /src /src

WORKDIR /src/install

# Manually run steps from quick.rc
ENV QUICK_INSTALL /src/install
ENV QUICK_BASIS $QUICK_INSTALL/basis
ENV PATH $PATH:$QUICK_INSTALL/bin

############################
### Base MPI CUDA 12.0.1 ###
############################
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04 AS base-mpi-cuda-12.0.1

RUN apt-get update -y \
 && apt-get install -y \
    gfortran \
    cmake \
    g++ \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev

RUN mkdir /src \
 && mkdir /src/build

WORKDIR /src

# Copy the current version of QUICK into the container
COPY . .

WORKDIR /src/build

RUN cmake .. -DCOMPILER=GNU -DCMAKE_INSTALL_PREFIX=$(pwd)/../install -DCUDA=TRUE -DMPI=TRUE

RUN make -j2 install

#############################
## Runtime MPI CUDA 12.0.1 ##
#############################

# Runtime image is smaller than the devel/build image
FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04 AS mpi-cuda-12.0.1

RUN apt-get update -y \
 && apt-get install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev

COPY --from=base-mpi-cuda-12.0.1 /src /src

WORKDIR /src/install

# Manually run steps from quick.rc
ENV QUICK_INSTALL /src/install
ENV QUICK_BASIS $QUICK_INSTALL/basis
ENV PATH $PATH:$QUICK_INSTALL/bin

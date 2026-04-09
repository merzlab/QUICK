FROM ubuntu:24.04 AS builder

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gfortran \
    cmake \
    g++ \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /src
WORKDIR /src
COPY . .

RUN cmake -S . -B build \
      -DMPI=TRUE \
      -DCOMPILER=GNU \
      -DCMAKE_INSTALL_PREFIX=/src/install \
 && cmake --build build --parallel $(nproc) \
 && cmake --install build


FROM ubuntu:24.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libgfortran5 \
    libgomp1 \
    openmpi-bin \
    libopenmpi3t64 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/install /opt/quick

ENV QUICK_INSTALL=/opt/quick
ENV QUICK_BASIS=/opt/quick/basis
ENV PATH=/opt/quick/bin:$PATH
ENV LIBRARY_PATH=/opt/quick/lib
ENV LD_LIBRARY_PATH=/opt/quick/lib

CMD ["quick"]

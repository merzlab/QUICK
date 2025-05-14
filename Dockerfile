FROM debian:latest


RUN apt update -y
RUN apt install -y \
	gfortran \
	cmake \
	g++ \
	openmpi-bin \
	openmpi-common \
	libopenmpi-dev

RUN mkdir /src

WORKDIR /src

COPY . .

RUN mkdir /src/build

WORKDIR /src/build

RUN cmake .. -DMPI=TRUE -DCOMPILER=GNU -DCMAKE_INSTALL_PREFIX=/src/install
RUN make -j && make install


ENV QUICK_INSTALL=/src/install
ENV QUICK_BASIS=$QUICK_INSTALL/basis
ENV PATH=$PATH:$QUICK_INSTALL/bin
ENV LIBRARY_PATH=$QUICK_INSTALL/lib
ENV LD_LIBRARY_PATH=$QUICK_INSTALL/lib

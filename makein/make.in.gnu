#************************************************************************
#                  A. Compiler Settings
# 
#   FC specifies f90 compiler
#   FFLAGS are compliation options
#   LFLAGS are linking flags
#
#************************************************************************

# Tested on LINUX machines and Mac machines
#
#---------------------
# a. MPI Version
#---------------------
#FC = mpif90
#FC = /usr/local/bin/mpif90
#MPI_INCLUDES = /usr/local/include
#MPI_LIBS = /usr/local/lib

#---------------------
# b. Single CPU Version
#---------------------
#FC= /users/PCS0202/bgs0374/bin/gcc-4.5/bin/gfortran
FC=gfortran
CC = gcc
#FC=ifort

#---------------------
# other compiler options
#---------------------
#FFLAGS = -g -O3 -xW -pg -traceback
#FFLAGS = -O3 -xW -ipo
#FFLAGS = -O3
#FFLAGS = -i4 -O3 -auto -assume byterecl -w95 -cm 
#FFLAGS = -g -O3 -traceback
FFLAGS = -O2 -lm  -mtune=native  -ffree-form  -DGNU -cpp -g
LD = $(FC)
LDFLAGS = $(FFLAGS) -lstdc++ -g
#LDFLAGS = $(FFLAGS)
#LDFLAGS = $(FFLAGS) -static -L/opt/intel/ict/2.0/cmkl/8.0.1/lib/32
TMPFILES = *.mod *.stb

# CPP Compiler
FPP = cpp -traditional -P  -DBINTRAJ
CPP = gcc -DCUBLAS_USE_THUNKING
CPP_FLAG = -I/usr/local/cuda/include  -O2

# CUDA Compiler
CUDAC = nvcc
CUDA_LIBPATH = -L/usr/local/gfortran/lib
#CUDA_FLAGS= -Xptxas=-v -m64 -g -G -use_fast_math -maxrregcount=63 -gencode arch=compute_20,code=sm_20
#CUDA_FLAGS= -DBINTRAJ -DDIRFRC_EFS -DDIRFRC_COMTRANS -DDIRFRC_NOVEC -DFFTLOADBAL_2PROC -DPUBFFT 
CUDA_FLAGS= -O2  -Xptxas=-v -m64 -use_fast_math -gencode arch=compute_20,code=sm_20

# G++ Compiler
CXX = g++
#CFLAGS = -lgfortran -lgfortranbegin -g -L/usr/local/cuda/lib64 -lcuda -lm $(CUDA_LIBPATH) -lcudart -lcublas -lstdc++
CFLAGS = -lgfortran L$(CUDA_HOME)/lib64 -lcuda -lm $(CUDA_LIBPATH) -lcudart -lcublas -lstdc++

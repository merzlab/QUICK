#!/bin/sh

MAKEIN = ../../make.in
include $(MAKEIN)

#all: quick_cuda

objfolder = ../../obj
libxcfolder = ../libxc
#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************

#================= quick module library =================================
CUDACOBJ=$(objfolder)/gpu.o $(objfolder)/gpu_type.o $(objfolder)/gpu_get2e.o \
	$(objfolder)/gpu_getxc.o
CUDAFOBJ= $(objfolder)/gpu_write_info.o
#CXXOBJ=$(objfolder)/octree.o $(objfolder)/grid_packer.o
#CUDALIBXCOBJ=$(objfolder)/gpu_libxc.o

$(CUDACOBJ):$(objfolder)/%.o:%.cu
	$(CUDAC) $(CUDA_FLAGS2) $(LIBXC_CUDA_FLAGS) -c $< -o $@

FOR=$(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS)

$(CUDAFOBJ):$(objfolder)/%.o:%.f90
	$(FOR) -c $< -o $@

#$(CXXOBJ):$(objfolder)/%.o:%.cpp
#	$(CXX) -c $< -o $@

#$(CUDALIBXCOBJ):$(objfolder)/%.o:%.cu
#	$(CUDAC) $(CUDA_FLAGS2) -I. -I$(libxcfolder) -c $< -o $@

#quick_cuda: 
#	$(CUDAC) $(CUDA_FLAGS2) -c gpu.cu
#	$(CUDAC) $(CUDA_FLAGS2) -c gpu_type.cu
#	$(CUDAC) $(CUDA_FLAGS2) -c gpu_get2e.cu
#	$(CUDAC) $(CUDA_FLAGS2) -c gpu_getxc.cu
#	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c gpu_write_info.f90
#all: $(CUDACOBJ) $(CUDAFOBJ) $(CUDALIBXCOBJ)
#all: $(CUDACOBJ) $(CUDAFOBJ) $(CXXOBJ)	
all: $(CUDACOBJ) $(CUDAFOBJ)
#************************************************************************
# 
#                 C. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
clean: 
	-rm -f *.o 
	

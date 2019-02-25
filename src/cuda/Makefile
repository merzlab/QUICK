#!/bin/sh

MAKEIN = ../../make.in
include $(MAKEIN)

all: quick_cuda

#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************

#================= quick module library =================================
quick_cuda: 
	$(CUDAC) $(CUDA_FLAGS2) -c gpu.cu
	$(CUDAC) $(CUDA_FLAGS2) -c gpu_type.cu
	$(CUDAC) $(CUDA_FLAGS2) -c gpu_get2e.cu
	$(CUDAC) $(CUDA_FLAGS2) -c gpu_getxc.cu
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FFLAGS) -c gpu_write_info.f90
	
#************************************************************************
# 
#                 C. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
clean: neat
	-rm -f *.o include/*.o
	

#!/bin/sh

MAKEIN = ../../make.in
include $(MAKEIN)

#CXX_FLAGS= -O0 -g -check-pointers=rw -debug all
CXX_FLAGS= -O0 -g -debug all


objfolder = ../../obj
libxcfolder = ../libxc
#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************

#================= quick module library =================================
CXXOBJ=$(objfolder)/octree.o $(objfolder)/grid_packer.o

$(CXXOBJ):$(objfolder)/%.o:%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

all: $(CXXOBJ)	
#************************************************************************
# 
#                 C. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
clean: 
	-rm -f *.o 
	

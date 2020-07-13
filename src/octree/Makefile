#!/bin/sh

MAKEIN = ../../make.in
include $(MAKEIN)

#************************************************************************
# 
#                 B. Make Object Files
# 
#************************************************************************

#================= quick module library =================================
CXXOBJ=$(objfolder)/octree.o $(objfolder)/grid_packer.o

$(CXXOBJ):$(objfolder)/%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: $(CXXOBJ)	
#************************************************************************
# 
#                 C. Self-defined Option
# 
#************************************************************************

# - 1. Clean object files
clean: 
	-rm -f *.o 
	

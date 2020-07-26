#  !---------------------------------------------------------------------!
#  ! Refactored by Madu Manathunga on 07/25/2020                         !
#  !                                                                     !
#  ! Copyright (C) 2020-2021 Merz lab                                    !
#  ! Copyright (C) 2020-2021 Götz lab                                    !
#  !                                                                     !
#  ! This Source Code Form is subject to the terms of the Mozilla Public !
#  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
#  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
#  !_____________________________________________________________________!
#
#  !---------------------------------------------------------------------!
#  ! This Makefile builds a QUICK version from source code               !
#  !---------------------------------------------------------------------!

MAKEIN = ./make.in
include $(MAKEIN)

#  !---------------------------------------------------------------------!
#  ! Variable declarations                                               !
#  !---------------------------------------------------------------------!

CXXOBJ=$(objfolder)/octree.o $(objfolder)/grid_packer.o

#  !---------------------------------------------------------------------!
#  ! Parent build targets                                                !
#  !---------------------------------------------------------------------!

$(CXXOBJ):$(objfolder)/%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: $(CXXOBJ)	

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY: clean

clean: 
	@-rm -f *.o 
	

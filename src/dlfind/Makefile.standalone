# COPYRIGHT
#
#  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
#  Tom Keal (thomas.keal@stfc.ac.uk)
#
#  This file is part of DL-FIND.
#
#  DL-FIND is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as 
#  published by the Free Software Foundation, either version 3 of the 
#  License, or (at your option) any later version.
#
#  DL-FIND is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public 
#  License along with DL-FIND.  If not, see 
#  <http://www.gnu.org/licenses/>.
#

# pg, g95, intel, gfortran
arch = gfortran

PPFLAGS =  
F90EXT = f90
ALLOC = new
OBJDIR = $(arch)/objects
PARA = $(MAKECMDGOALS)

########################################################################
##  Protland compiler
ifeq ($(arch),pg)
  F90        = pgf95
  MPI_F90    = mpif90
  F90FLAGS   =  -C -module $(OBJDIR)
  #F90FLAGS   =  -C -fast -Mvect=sse -Mcache_align -Kieee -Mr8
  LDFLAGS    = -L/usr/pgi/linux86/6.1/lib -llapack -lblas
  LDFLAGS    = -L/home/jkk/users/chemshell/chemsh_060306_sandbox_pg/lib -llapack -lblas -lg2c
  LDFLAGS    = -lacml
  MPI_LDFLAGS= -lacml -lmpi
  PPFLAGS    = -DOLDALLOC
  F90EXT     = F95
  ALLOC      = old
endif
# with -lacml this works on HECToR

########################################################################
##  g95 compiler
ifeq ($(arch),g95)
  F90        = g95
  F90FLAGS   =  -C 
  F90FLAGS   =  -C -ftrace=full -freal=nan -finteger=nan -flogical=none \
                -fpointer=none -fimplicit-none  -fbounds-check \
                -fintrinsic-extensions -std=f95 -ftr15581 -Wuninitialized \
                -Wprecision-loss -fmod=$(OBJDIR) 
#\
#		-Wunused-vars -Wunused-types -Wunset-vars -Wunused-module-vars \
#		-Wunused-module-procs

# These produce an illegal instruction sometimes ...
#  LDFLAGS    = -L/home/jkk/lib/ATLAS/Linux_HAMMER64SSE2_2/lib \
#                -lm -llapack -lcblas -lf77blas -latlas -lg2c
# works on jk37vig
#  LDFLAGS    = -L/home/linux8_i1/kaestner/scr/lib/atlas_SRCdir/ATLAS3.8.2/linux8/lib \
                -lm -llapack -lcblas -lf77blas -latlas 
# Netlib blas and lapack - http://www.netlib.org/lapack/lapack.tgz
#  LDFLAGS    = -L/home/linux8_i1/kaestner/scr/lib/LAPACK/lapack-3.2 \
#                -lm -llapack_LINUX -lblas_LINUX 
# Laptop, ChemShell Lapack
  LDFLAGS    = -L/Users/kaestner/ChemShell/chemsh_3.2_svn/lib \
                -lm -llapack -lblas
# Stuttgart desktop
  LDFLAGS    = -L/home/linux8_i1/kaestner/chemshell/chemsh_3.2_svn/lib \
                -lm -llapack -lblas
  F90EXT     = F90
# the next line may be used to find memory leaks
#  ALLOC      = old
endif
########################################################################

########################################################################
##  Intel compiler
ifeq ($(arch),intel)
  F90        = ifort 
  F90FLAGS   =  -C -ftrapuv -traceback -module $(OBJDIR)
  # fast
  #F90FLAGS   =  -O -p -module $(OBJDIR)
  #sigsegv
#  LDFLAGS    = -L/home/jkk/users/chemshell/chemsh_3.1b1_intel/lib -llapack -lblas
#  LDFLAGS    = -L/opt/intel/mkl701/lib/em64t -lguide -lmkl_lapack -lmkl_ia32
  # jk37vig, dynamic linking
  LDFLAGS    = -L/home/jkk/programs/intel/mkl/9.1/lib/em64t -lguide -lmkl_lapack -lmkl_em64t -lpthread
  F90EXT     = F90
endif
########################################################################

########################################################################
##  gfortran (gcc) compiler
# at the moment, gfortran leads to numerically different results than 
# the others ...
ifeq ($(arch),gfortran)
  F90        = gfortran
  F90FLAGS   =  -C -M $(OBJDIR)
  F90FLAGS   =   -fimplicit-none  -fbounds-check -std=gnu -Wconversion \
	           -Wsurprising -J $(OBJDIR)
#  LDFLAGS    = -L/home/jkk/lib/ATLAS/Linux_HAMMER64SSE2_2/lib \
#                -lm -llapack -lcblas -lf77blas -latlas -lg2c
  LDFLAGS    = -L/home/jkk/lib/ATLAS/LINUX_jk37vig_g95/ATLAS/lib/Linux_P4SSE2 \
                -lm -lopenblas
  F90EXT     = F90
endif
########################################################################

.SUFFIXES:
.SUFFIXES: .f90 .o .mod

# modules used by more than one file
OBJ_MOD = $(OBJDIR)/dlf_stat_module.o $(OBJDIR)/dlf_global_module.o $(OBJDIR)/dlf_allocate.o $(OBJDIR)/dlf_util.o \
	$(OBJDIR)/dlf_sort.o $(OBJDIR)/dlfind_quick.o

# Objects that work without numerical libraries (this distinction does not hold any more) 
OBJ_NOLAPACK = $(OBJDIR)/dl-find.o $(OBJDIR)/dlf_convergence.o $(OBJDIR)/dlf_checkpoint.o $(OBJDIR)/dlf_svnversion.o\
         $(OBJDIR)/dlf_formstep.o $(OBJDIR)/dlf_scalestep.o \
       $(OBJDIR)/dlf_lbfgs.o $(OBJDIR)/dlf_time.o \
       $(OBJDIR)/dlf_task.o $(OBJDIR)/dlf_neb.o $(OBJDIR)/dlf_qts.o $(OBJDIR)/dlf_dimer.o $(OBJDIR)/dlf_conint.o $(OBJDIR)/dlf_coords.o \
       $(OBJDIR)/dlf_parallel_opt.o $(OBJDIR)/dlf_microiter.o

# Objects that need LAPACK linked in
OBJ_LAPACK = $(OBJDIR)/dlf_hdlc_matrixlib.o $(OBJDIR)/dlf_hdlc_primitive.o $(OBJDIR)/dlf_hdlc_constraint.o \
         $(OBJDIR)/dlf_hdlc_hdlclib.o $(OBJDIR)/dlf_hdlc_interface.o $(OBJDIR)/dlf_linalg.o

OBJ =  $(OBJ_NOLAPACK) $(OBJ_LAPACK) 

find:	$(OBJ_MOD) $(OBJ) $(OBJDIR)/main.o $(OBJDIR)/dlf_serial.o
	@echo "Linking ..."
	$(F90) $(F90FLAGS) -o find.x $(OBJDIR)/main.o $(OBJ) $(OBJDIR)/dlf_serial.o $(OBJ_MOD)  $(LDFLAGS)
	@cp find.x $(arch)/find_$(arch).x

parallel: $(OBJ_MOD) $(OBJ) $(OBJDIR)/main.o $(OBJDIR)/dlf_mpi.o
	@echo "Linking ..."
	$(MPI_F90) $(F90FLAGS) -o Pfind.x $(OBJDIR)/main.o $(OBJ) $(OBJDIR)/dlf_mpi.o $(OBJ_MOD)  $(MPI_LDFLAGS)
	@cp Pfind.x $(arch)/Pfind_$(arch).x

 
obj:	$(OBJ_MOD) $(OBJ) 

draw:	 $(OBJDIR)/dlf_stat_module.o $(OBJDIR)/dlf_global_module.o $(OBJDIR)/dlf_serial.o $(OBJDIR)/main.o $(OBJDIR)/draw.o
	$(F90) -o draw.x $(OBJDIR)/main.o $(OBJDIR)/dlf_serial.o $(OBJDIR)/draw.o

png:
#	./draw.x
	./find.x
	povray surface.pov +A -W640 -H480
	xv surface.png &
#	povray surface.pov -W800 -H600 +A


	 #cat $*.f90 >> all.f90

#	echo 'subroutine dlf_svnversion(iver)' > dlf_svnversion.f90
#	echo '  character(*) :: iver' >> dlf_svnversion.f90
#	echo -n '  iver="' >> dlf_svnversion.f90
dlf_svnversion.f90: FORCE 
	cp dlf_svnversion.head dlf_svnversion.f90
	-svnversion -n . >> dlf_svnversion.f90
	echo '"' >> dlf_svnversion.f90
	echo "end subroutine dlf_svnversion" >> dlf_svnversion.f90

FORCE:

#
# The main rule for compiling f90 files:
#
$(OBJDIR)/%.o: %.f90
	@mkdir -p $(OBJDIR)
ifeq ($(ALLOC),old)
	   sed -e"s/call allocate *(/ aLLocate(/g" \
               -e"/ aLLocate(/s/,/(/" \
               -e"/ aLLocate(/s/\(.*\))/\1))/" \
               -e"s/call deallocate *(/deallocate(/" $*.f90 > $(OBJDIR)/$*_d.$(F90EXT)
else
	  cp $*.f90 $(OBJDIR)/$*_d.$(F90EXT)
endif	

#
# remove $(OBJDIR)/dlf_mpi_d.$(F90EXT) if compiling the serial version, 
# and $(OBJDIR)/dlf_serial_d.$(F90EXT) if the parallel one.
#
ifeq ($(findstring parallel,$(PARA)),parallel)
	@rm -f $(OBJDIR)/dlf_serial_d.$(F90EXT)
	$(MPI_F90) $(F90FLAGS) $(PPFLAGS) -c $(OBJDIR)/$*_d.$(F90EXT) -o $(OBJDIR)/$*.o
	@#rm $*_d.$(F90EXT)
	@echo 
else
	@rm -f $(OBJDIR)/dlf_mpi_d.$(F90EXT)
	$(F90) $(F90FLAGS) $(PPFLAGS) -c $(OBJDIR)/$*_d.$(F90EXT) -o $(OBJDIR)/$*.o
	@#rm $*_d.$(F90EXT)
	@echo 
endif

#
# Utilities
#
clean:
	rm -f $(OBJ_MOD) $(OBJ) $(OBJDIR)/*.mod $(OBJDIR)/main.o $(OBJDIR)/*_d.?9? $(OBJDIR)/dlf_mpi.o $(OBJDIR)/dlf_serial.o 

# remove all module directories and executables
clean_all:
	rm -r g95 pg intel gfortran

man:
	latex documentation.tex
	dvips documentation.dvi

litpath=~/users/data/literatur/

bib:
	cp $(litpath)/tex/jabref.bib text.bib
	bibtex documentation

doc:
	robodoc

#dependencies from g95 -M:
$(OBJDIR)/dlf_stat_module.o $(OBJDIR)/dlf_stat.mod $(OBJDIR)/dlf_parameter_module.mod: dlf_stat_module.f90
$(OBJDIR)/dlf_global_module.o $(OBJDIR)/dlf_global.mod: $(OBJDIR)/dlf_parameter_module.mod dlf_global_module.f90
$(OBJDIR)/dlf_allocate.o $(OBJDIR)/dlf_allocate.mod: $(OBJDIR)/dlf_parameter_module.mod dlf_allocate.f90
$(OBJDIR)/dlf_util.o $(OBJDIR)/dlf_store.mod $(OBJDIR)/bspline.mod: $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_parameter_module.mod dlf_util.f90
$(OBJDIR)/dl-find.o: $(OBJDIR)/dlf_store.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_stat.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod $(OBJDIR)/dlf_hessian.mod dl-find.f90
$(OBJDIR)/dlf_convergence.o $(OBJDIR)/dlf_convergence.mod: $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_convergence.f90
$(OBJDIR)/dlf_checkpoint.o $(OBJDIR)/dlf_checkpoint.mod: $(OBJDIR)/dlf_stat.mod $(OBJDIR)/dlf_parameter_module.mod $(OBJDIR)/dlf_global.mod dlf_checkpoint.f90
$(OBJDIR)/dlf_svnversion.o: dlf_svnversion.f90
$(OBJDIR)/dlf_formstep.o $(OBJDIR)/dlf_hessian.mod $(OBJDIR)/dlf_formstep_module.mod: $(OBJDIR)/dlf_stat.mod $(OBJDIR)/dlf_checkpoint.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_formstep.f90
$(OBJDIR)/dlf_scalestep.o $(OBJDIR)/dlf_linesearch.mod $(OBJDIR)/dlf_scalestep_module.mod: $(OBJDIR)/dlf_checkpoint.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_scalestep.f90
$(OBJDIR)/dlf_lbfgs.o $(OBJDIR)/lbfgs_module.mod: $(OBJDIR)/dlf_checkpoint.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_lbfgs.f90
$(OBJDIR)/dlf_time.o $(OBJDIR)/dlf_time.mod: $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_time.f90
$(OBJDIR)/dlf_task.o $(OBJDIR)/dlf_task_module.mod: $(OBJDIR)/dlf_store.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_stat.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_task.f90
$(OBJDIR)/dlf_neb.o $(OBJDIR)/dlf_neb.mod: $(OBJDIR)/dlf_checkpoint.mod $(OBJDIR)/dlf_stat.mod $(OBJDIR)/bspline.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_neb.f90
$(OBJDIR)/dlf_qts.o $(OBJDIR)/dlf_qts.mod: $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_neb.mod $(OBJDIR)/dlf_parameter_module.mod dlf_qts.f90
$(OBJDIR)/dlf_dimer.o $(OBJDIR)/dlf_dimer.mod: $(OBJDIR)/dlf_checkpoint.mod $(OBJDIR)/dlf_stat.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_dimer.f90
$(OBJDIR)/dlf_conint.o $(OBJDIR)/dlf_conint.mod: $(OBJDIR)/dlf_checkpoint.mod $(OBJDIR)/dlf_hessian.mod $(OBJDIR)/dlf_stat.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_conint.f90
$(OBJDIR)/dlf_coords.o: $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_coords.f90
$(OBJDIR)/dlf_hdlc_matrixlib.o $(OBJDIR)/dlfhdlc_matrixlib.mod: $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_hdlc_matrixlib.f90
$(OBJDIR)/dlf_hdlc_primitive.o $(OBJDIR)/dlfhdlc_primitive.mod: $(OBJDIR)/dlfhdlc_matrixlib.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_hdlc_primitive.f90
$(OBJDIR)/dlf_hdlc_constraint.o $(OBJDIR)/dlfhdlc_constraint.mod: $(OBJDIR)/dlfhdlc_matrixlib.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_hdlc_constraint.f90
$(OBJDIR)/dlf_hdlc_hdlclib.o $(OBJDIR)/dlfhdlc_hdlclib.mod: $(OBJDIR)/dlf_checkpoint.mod $(OBJDIR)/dlfhdlc_constraint.mod $(OBJDIR)/dlfhdlc_primitive.mod $(OBJDIR)/dlfhdlc_matrixlib.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_hdlc_hdlclib.f90
$(OBJDIR)/dlf_hdlc_interface.o: $(OBJDIR)/dlf_checkpoint.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlfhdlc_hdlclib.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_hdlc_interface.f90
$(OBJDIR)/dlf_linalg.o: $(OBJDIR)/dlfhdlc_matrixlib.mod $(OBJDIR)/dlf_parameter_module.mod dlf_linalg.f90
$(OBJDIR)/dlf_parallel_opt.o: $(OBJDIR)/dlf_sort_module.mod $(OBJDIR)/dlf_checkpoint.mod $(OBJDIR)/dlf_allocate.mod $(OBJDIR)/dlf_stat.mod $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_parallel_opt.f90
$(OBJDIR)/dlf_sort.o $(OBJDIR)/dlf_sort_module.mod: $(OBJDIR)/dlf_parameter_module.mod dlf_sort.f90
$(OBJDIR)/dlf_serial.o: $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_serial.f90
$(OBJDIR)/dlf_mpi.o $(OBJDIR)/dlf_mpi_module.mod: $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_mpi.f90
$(OBJDIR)/dlf_microiter.o $(OBJDIR)/dlf_microiter.mod: $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod dlf_microiter.f90
$(OBJDIR)/main.o $(OBJDIR)/driver_module.mod: $(OBJDIR)/dlf_global.mod $(OBJDIR)/dlf_parameter_module.mod main.f90

# They might look like:
#dlf_linalg.o: dlfhdlc_matrixlib.mod dlf_parameter_module.mod dlf_linalg.f90
#main.o driver_module.mod: dlf_parameter_module.mod main.f90
#$(OBJDIR)/main.o $(OBJDIR)/driver_module.mod: $(OBJDIR)/dlf_parameter_module.mod main.f90

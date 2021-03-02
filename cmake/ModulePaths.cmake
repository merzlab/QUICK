#Normally, CMake handles all of the include paths automatically.
#When you link a library, it automatically adds its include paths.  This is very convenient, as it means we don't have to care about this stuff in the cmake code.
#However, with Fortran modules it's a bit more complicated.  CMake just plonks the modules in CMAKE_CURRENT_BINARY_DIR.
#Sometimes, the same module gets compiled twice with different definitions, and the two versions need to be kept separate.
#Also we just need to know the module path to link modules to object-only libraries, like in sander.

#So, in this file, we establish the module paths for each library with modules.
#When code with modules is built, we configure its modules to go to one of the directories here

# NOTE: if you change this, you also have to change the regex that detects it in config_module_paths()
set(MODULE_BASE_DIR ${CMAKE_BINARY_DIR}/amber-modules)

set(AMBER_COMMON_MOD_DIR "${MODULE_BASE_DIR}/amberlib")
set(SQMLIB_MOD_DIR "${MODULE_BASE_DIR}/sqm/lib") #sqmLIB as opposed to the SQM executable
set(SQMEXE_MOD_DIR "${MODULE_BASE_DIR}/sqm/exec")
set(SQMLIB_MPI_MOD_DIR "${MODULE_BASE_DIR}/sqm/lib/mpi") 
set(SQMLIB_OMP_MOD_DIR "${MODULE_BASE_DIR}/sqm/lib/omp")
set(SQMEXE_MPI_MOD_DIR "${MODULE_BASE_DIR}/sqm/exec/mpi") 
set(LIBPBSA_MOD_DIR "${MODULE_BASE_DIR}/pbsa/lib")
set(LIBPBSA_MPI_MOD_DIR "${MODULE_BASE_DIR}/pbsa/lib/mpi")
set(LIBPBSA_SANDER_MOD_DIR "${MODULE_BASE_DIR}/pbsa/sanderlib")
set(LIBPBSA_SANDER_MPI_MOD_DIR "${MODULE_BASE_DIR}/pbsa/sanderlib/mpi")
set(PBSAEXE_MOD_DIR "${MODULE_BASE_DIR}/pbsa/exec")
set(PBSAEXE_CUDA_MOD_DIR "${MODULE_BASE_DIR}/pbsa/exec/cuda")
set(SANDER_COMMON_MOD_DIR "${MODULE_BASE_DIR}/sander")
set(SANDER_COMMON_MPI_MOD_DIR "${MODULE_BASE_DIR}/sander/mpi")
set(SANDER_COMMON_OMP_MOD_DIR "${MODULE_BASE_DIR}/sander/omp")
set(SEBOMD_MOD_DIR "${MODULE_BASE_DIR}/sebomd")
set(SEBOMD_MPI_MOD_DIR "${MODULE_BASE_DIR}/sebomd/mpi")
set(RISMLIB_MOD_DIR "${MODULE_BASE_DIR}/rism/lib")
set(RISMLIB_MPI_MOD_DIR "${MODULE_BASE_DIR}/rism/lib/mpi")
set(RISMLIB_SANDER_INTERFACE_DIR "${MODULE_BASE_DIR}/rism/sander_interface")
set(RISMLIB_SANDER_INTERFACE_MPI_DIR "${MODULE_BASE_DIR}/rism/sander_interface/mpi")
set(RISMLIB_SFF_INTERFACE_DIR "${MODULE_BASE_DIR}/rism/sff_interface")
set(RISMLIB_SFF_INTERFACE_MPI_DIR "${MODULE_BASE_DIR}/rism/sff_interface/mpi")
set(RISM1D_MOD_DIR "${MODULE_BASE_DIR}/rism/1d")
set(RISMTHERMO_MOD_DIR "${MODULE_BASE_DIR}/rism/thermo")
set(RISMORAVE_MOD_DIR "${MODULE_BASE_DIR}/rism/orave")
set(VOLSLICE_MOD_DIR "${MODULE_BASE_DIR}/rism/volslice")
set(NMODE_MOD_DIR "${MODULE_BASE_DIR}/nmode")
set(QUICK_MOD_DIR "${MODULE_BASE_DIR}/quick")
set(QUICK_FAKE_AMB_INTERFACE_MOD_DIR "${MODULE_BASE_DIR}/quick/fake_amber_interface")
set(QUICK_REAL_AMB_INTERFACE_MOD_DIR "${MODULE_BASE_DIR}/quick/real_amber_interface")
set(QUICK_MPI_MOD_DIR "${MODULE_BASE_DIR}/quick/mpi")
set(QUICK_CUDA_MOD_DIR "${MODULE_BASE_DIR}/quick/cuda")
set(QUICK_CUDA_MPI_MOD_DIR "${MODULE_BASE_DIR}/quick/cuda/mpi")
set(DIVICON_MOD_DIR "${MODULE_BASE_DIR}/divicon")
set(PMEMD_MOD_DIR "${MODULE_BASE_DIR}/pmemd")
set(PMEMD_GEM_MOD_DIR "${MODULE_BASE_DIR}/pmemd.gem")
set(CHAMBER_MOD_DIR "${MODULE_BASE_DIR}/chamber")
set(SFF_MOD_DIR "${MODULE_BASE_DIR}/sff")
set(GBNSR6_MOD_DIR "${MODULE_BASE_DIR}/gbnsr6")
set(GBNSR6_MPI_MOD_DIR "${MODULE_BASE_DIR}/gbnsr6/mpi")

file(MAKE_DIRECTORY ${AMBER_COMMON_MOD_DIR} ${SQMLIB_MOD_DIR} ${SQMEXE_MOD_DIR} ${LIBPBSA_MOD_DIR} ${LIBPBSA_SANDER_MOD_DIR}
	${PBSAEXE_MOD_DIR} ${SANDER_COMMON_MOD_DIR} ${SEBOMD_MOD_DIR} ${RISMLIB_MOD_DIR} ${RISM1D_MOD_DIR} ${RISMTHERMO_MOD_DIR}
	${RISMORAVE_MOD_DIR} ${VOLSLICE_MOD_DIR} ${NMODE_MOD_DIR} ${QUICK_MOD_DIR} ${DIVICON_MOD_DIR} ${CHAMBER_MOD_DIR} ${SFF_MOD_DIR} 
	${GBNSR6_MOD_DIR} ${RISMLIB_MPI_MOD_DIR} ${LIBPBSA_MPI_MOD_DIR} ${LIBPBSA_SANDER_MPI_MOD_DIR} ${SANDER_COMMON_MPI_MOD_DIR}
	${RISMLIB_SANDER_INTERFACE_DIR} ${RISMLIB_SANDER_INTERFACE_MPI_DIR} ${RISMLIB_SFF_INTERFACE_DIR} ${SANDER_COMMON_OMP_MOD_DIR}
	${RISMLIB_SFF_INTERFACE_MPI_DIR} ${SEBOMD_MPI_MOD_DIR} ${QUICK_MPI_MOD_DIR} ${QUICK_CUDA_MOD_DIR} ${QUICK_CUDA_MPI_MOD_DIR}
	${QUICK_REAL_AMB_INTERFACE_MOD_DIR} ${QUICK_FAKE_AMB_INTERFACE_MOD_DIR} ${GBNSR6_MPI_MOD_DIR} ${PBSAEXE_CUDA_MOD_DIR})

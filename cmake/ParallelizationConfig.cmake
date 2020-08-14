# file with configuration that is common to both OpenMP and MPI.
# It is included from both OpenMPConfig and MPIConfig.

if(OPENMP OR MPI)
	set(PARALLEL TRUE)
else()
	set(PARALLEL FALSE)
endif()
set(DLFIND_SOURCES
	dlf_allocate.f90 dlf_checkpoint.f90 
	dlf_conint.f90 dlf_convergence.f90 
	dlf_coords.f90 dlf_dimer.f90
	dlf_formstep.f90 dlf_global_module.f90 
	dlf_hdlc_constraint.f90 dlf_hdlc_hdlclib.f90 
	dlf_hdlc_interface.f90 dlf_hdlc_matrixlib.f90 
	dlf_hdlc_primitive.f90 dl-find.f90 
	dlfind_main_driver.f90 dlfind_quick.f90
	dlf_lbfgs.f90 dlf_linalg.f90 
	dlf_microiter.f90 dlf_mpi.f90
	dlf_neb.f90 dlf_parallel_opt.f90 
	dlf_qts.f90 dlf_scalestep.f90
	dlf_serial.f90 dlf_sort.f90 
	dlf_stat_module.f90 dlf_svnversion.f90 
	dlf_task.f90 dlf_time.f90 dlf_util.f90

add_library(dlfind ${DLFIND_SOURCES})
target_compile_options(dlfind PRIVATE ${OPT_FFLAGS})
install_libraries(dlfind EXPORT QUICK)

# Prevent name conflict with system BLAS
set_property(TARGET dlfind PROPERTY OUTPUT_NAME dlfind-quick)

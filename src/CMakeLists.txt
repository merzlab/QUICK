# recurse to subdirs
# --------------------------------------------------------------------
add_subdirectory(libxc)
add_subdirectory(octree)

if(blas_INTERNAL AND NOT INSIDE_AMBER)
	add_subdirectory(blas)
endif()

if(lapack_INTERNAL AND NOT INSIDE_AMBER)
	add_subdirectory(lapack)
endif()

if(CUDA)
	add_subdirectory(cuda)
endif()

# quick source lists
# --------------------------------------------------------------------

set(QUICK_MODULES_SOURCES
	quick_mpi_module.f90 quick_constants_module.f90 quick_method_module.f90
	quick_molspec_module.f90 quick_gaussian_class_module.f90 quick_size_module.f90
	quick_basis_module.f90 quick_calculated_module.f90 quick_overlap_module.f90
	quick_divcon_module.f90 quick_ecp_module.f90 quick_electrondensity_module.f90
	quick_files_module.f90 quick_timer_module.f90 quick_gridpoints_module.f90
	quick_mfcc_module.f90 quick_params_module.f90 quick_pb_module.f90
	quick_scratch_module.f90 quick_all_module.f90 quick_scf_module.f90
	quick_gradient_module.f90 quick_api_module.f90 quick_api_test_module.f90
	quick_cutoff_module.f90 quick_exception_module.f90 quick_input_parser_module.f90
	quick_eri_module.f90 quick_eri_grad_module.f90 quick_oei_module.f90
	quick_dft_module.f90 quick_scf_operator_module.f90 quick_uscf_operator_module.f90
	quick_uscf_module.f90 quick_sad_guess_module.f90 quick_optimizer_module.f90
        quick_cew_module.f90 quick_lri_module.f90 quick_lri_grad_module.f90
	oshell_quick_eri_module.f90 oshell_quick_eri_grad_module.f90
	oshell_quick_gradient_module.f90)

set(QUICK_SUBS_SOURCES Angles.f90 copyDMat.f90 copySym.f90
	degen.f90 denspt.f90 diag.f90 dipole.f90
	EffChar.f90 eigvec.f90
	findBlock.f90 fmt.f90 getinum.f90
	getNum.f90 greedy_distrubute.f90 hrr.f90 iatoi.f90
	iatoimp.f90 io.f90 iwhole.f90
	lbfgs.f90 Lsolve.f90 matComp.f90 matMul.f90
	order.f90 orthog.f90 PriCol.f90 PriSym.f90
	PrtAct.f90 PrtDat.f90 PrtErr.f90 PrtLab.f90
	PrtMsg.f90 PrtTim.f90 PrtWrn.f90 pteval.f90
	quick_open.f90 random.f90 rdinum.f90 rdnml.f90
	rdnum.f90 rdword.f90 readPDB.f90 spdfgh.f90
	ssw.f90 sum2Mat.f90 transpose.f90 tridi.f90
	upcase.f90 vett.f90 whatis.f90 whole.f90
	wrtRestart.f90 xnorm.f90 zeroMatrix.f90 zmake.f90
	pt2der.f90 sswder.f90 denspt_new_imp.f90
	pteval_new_imp.f90 scaMatMul.f90 diagmkl.f90
	oshell_denspt.f90)

if(mirp_ENABLED)
	list(APPEND QUICK_SUBS_SOURCES mirp_fmt.cpp)
endif()

set(QUICK_GENERAL_SOURCES initialize.f90 read_job_and_atom.f90
    getMol.f90 quick_one_electron_integral.f90 getEnergy.f90
    ecp.f90 nuclear.f90 finalize.f90 mpi_setup.f90 quick_debug.f90
    calMP2.f90 hessian.f90 CPHF.f90 frequency.f90 basis.f90)

set(QUICK_CUDA_FORTRAN_SOURCE cuda/gpu_write_info.f90)

set(TEST_API_SOURCES quick_api_test.f90)
set(MAIN_SOURCES main.f90)


# the "subs" and "modules" source lists are for items in subdirectories.  We have to prepend the subdirectory path to these items.
# --------------------------------------------------------------------

set(QUICK_MODULES_SOURCES_FULLPATH "")
set(QUICK_SUBS_SOURCES_FULLPATH "")

foreach(MODULES_SOURCE ${QUICK_MODULES_SOURCES})
	list(APPEND QUICK_MODULES_SOURCES_FULLPATH "modules/${MODULES_SOURCE}")
endforeach()

foreach(SUBS_SOURCE ${QUICK_SUBS_SOURCES})
	list(APPEND QUICK_SUBS_SOURCES_FULLPATH "subs/${SUBS_SOURCE}")
endforeach()

# create serial and MPI libraries
# --------------------------------------------------------------------

include_directories(util)

add_library(libquick SHARED ${QUICK_MODULES_SOURCES_FULLPATH} ${QUICK_SUBS_SOURCES_FULLPATH} ${QUICK_GENERAL_SOURCES})
target_compile_options(libquick PRIVATE ${OPT_FFLAGS})
config_module_dirs(libquick quick/serial libxc/serial)
target_link_libraries(libquick PRIVATE xc octree ${QUICK_BLAS} ${QUICK_LAPACK})
install_libraries(libquick EXPORT QUICK)

if(mirp_ENABLED)
	target_compile_definitions(libquick PRIVATE  MIRP)
	target_link_libraries(libquick PRIVATE mirp::mirp)
endif()

# "liblibquick" -> "libquick"
set_property(TARGET libquick PROPERTY OUTPUT_NAME quick)

if(MPI)
	make_mpi_version(libquick libquick_mpi LANGUAGES Fortran)

	target_compile_definitions(libquick_mpi PRIVATE MPIV)
	set_property(TARGET libquick_mpi PROPERTY OUTPUT_NAME quick_mpi)

	# change link libraries and mod dirs for MPI
	remove_link_libraries(libquick_mpi octree)
	target_link_libraries(libquick_mpi PRIVATE octree_mpi)
	config_module_dirs(libquick_mpi quick/mpi libxc/serial)


	install_libraries(libquick_mpi EXPORT QUICK)
endif()

if(CUDA)
	copy_target(libquick libquick_cuda SWAP_SOURCES TO $<TARGET_OBJECTS:cublas_fortran_obj> $<TARGET_OBJECTS:cusolver_obj> ${QUICK_CUDA_FORTRAN_SOURCE})
	target_compile_definitions(libquick_cuda PRIVATE CUDA)
	set_property(TARGET libquick_cuda PROPERTY OUTPUT_NAME quick_cuda)

	# change link libraries and mod dirs for CUDA
	remove_link_libraries(libquick_cuda xc octree ${QUICK_BLAS} ${QUICK_LAPACK})
	target_link_libraries(libquick_cuda PRIVATE xc_gpu xc_cuda octree_gpu quick_cuda_kernels cublas cusolver)
	config_module_dirs(libquick_cuda quick/cuda libxc/gpu)

	install_libraries(libquick_cuda EXPORT QUICK)
endif()

if(MPI AND CUDA)
	copy_target(libquick_mpi libquick_mpi_cuda SWAP_SOURCES TO $<TARGET_OBJECTS:cublas_fortran_obj> $<TARGET_OBJECTS:cusolver_obj> ${QUICK_CUDA_FORTRAN_SOURCE})
	target_compile_definitions(libquick_mpi_cuda PRIVATE CUDA_MPIV)
	set_property(TARGET libquick_mpi_cuda PROPERTY OUTPUT_NAME quick_mpi_cuda)

	remove_link_libraries(libquick_mpi_cuda octree_mpi xc)
	target_link_libraries(libquick_mpi_cuda PRIVATE xc_gpu xc_cuda octree_gpu quick_cuda_kernels_mpi cublas cusolver)
	config_module_dirs(libquick_mpi_cuda quick/mpi_cuda libxc/gpu)

	install_libraries(libquick_mpi_cuda EXPORT QUICK)
endif()

# Enable -DLAPACK or -DMKL for non-CUDA versions
if(lapack_ENABLED)
	target_compile_definitions(libquick PRIVATE LAPACK=1)
	if(MPI)
		target_compile_definitions(libquick_mpi PRIVATE LAPACK=1)
	endif()
elseif(mkl_ENABLED)
	target_compile_definitions(libquick PRIVATE MKL=1)
	if(MPI)
		target_compile_definitions(libquick_mpi PRIVATE MKL=1)
	endif()
endif()

# create executables
# --------------------------------------------------------------------

add_executable(quick ${MAIN_SOURCES})
target_compile_options(quick PRIVATE ${OPT_FFLAGS})
config_module_dirs(quick quick/serial libxc/serial)
target_link_libraries(quick libquick)

add_executable(test-api ${TEST_API_SOURCES})
target_compile_options(test-api PRIVATE ${OPT_FFLAGS})
config_module_dirs(test-api quick/serial libxc/serial)
target_link_libraries(test-api libquick)

install(TARGETS quick DESTINATION ${BINDIR} EXPORT QUICK)
install(TARGETS test-api DESTINATION ${BINDIR} EXPORT QUICK)

if(MPI)
	make_mpi_version(quick quick.MPI LANGUAGES Fortran)
	target_compile_definitions(quick.MPI PRIVATE MPIV)

	make_mpi_version(test-api test-api.MPI LANGUAGES Fortran)
	target_compile_definitions(test-api.MPI PRIVATE MPIV)

	# change link libraries and mod dirs for MPI
	remove_link_libraries(quick.MPI libquick)
	target_link_libraries(quick.MPI libquick_mpi)
	config_module_dirs(quick.MPI quick/mpi libxc/serial)

	remove_link_libraries(test-api.MPI libquick)
	target_link_libraries(test-api.MPI libquick_mpi)
	config_module_dirs(test-api.MPI quick/mpi libxc/serial)

	install(TARGETS quick.MPI DESTINATION ${BINDIR} EXPORT QUICK)
	install(TARGETS test-api.MPI DESTINATION ${BINDIR} EXPORT QUICK)
endif()

if(CUDA)
	copy_target(quick quick.cuda)
	target_compile_definitions(quick.cuda PRIVATE CUDA)

	copy_target(test-api test-api.cuda)
	target_compile_definitions(test-api.cuda PRIVATE CUDA)

	remove_link_libraries(quick.cuda libquick)
	target_link_libraries(quick.cuda libquick_cuda)
	config_module_dirs(quick.cuda quick/cuda libxc/gpu)

	remove_link_libraries(test-api.cuda libquick)
	target_link_libraries(test-api.cuda libquick_cuda)
	config_module_dirs(test-api.cuda quick/cuda libxc/gpu)

	install(TARGETS quick.cuda DESTINATION ${BINDIR} EXPORT QUICK)
	install(TARGETS test-api.cuda DESTINATION ${BINDIR} EXPORT QUICK)
endif()

if(CUDA AND MPI)
	copy_target(quick.MPI quick.cuda.MPI)
	target_compile_definitions(quick.cuda.MPI PRIVATE CUDA_MPIV)

	copy_target(test-api.MPI test-api.cuda.MPI)
	target_compile_definitions(test-api.cuda.MPI PRIVATE CUDA_MPIV)

	remove_link_libraries(quick.cuda.MPI libquick_mpi)
	target_link_libraries(quick.cuda.MPI libquick_mpi_cuda)
	config_module_dirs(quick.cuda.MPI quick/mpi_cuda libxc/gpu)

	remove_link_libraries(test-api.cuda.MPI libquick_mpi)
	target_link_libraries(test-api.cuda.MPI libquick_mpi_cuda)
	config_module_dirs(test-api.cuda.MPI quick/mpi_cuda libxc/gpu)

	install(TARGETS quick.cuda.MPI DESTINATION ${BINDIR} EXPORT QUICK)
	install(TARGETS test-api.cuda.MPI DESTINATION ${BINDIR} EXPORT QUICK)
endif()

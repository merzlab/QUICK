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

if(CUDA OR HIP)
	add_subdirectory(gpu/${QUICK_GPU_TARGET_NAME})
endif()

# quick source lists
# --------------------------------------------------------------------

set(QUICK_MODULES_SOURCES
	quick_mpi_module.f90 quick_constants_module.f90 quick_method_module.f90
	quick_molspec_module.f90 quick_gaussian_class_module.f90 quick_size_module.f90
	quick_basis_module.f90 quick_calculated_module.f90 quick_magma_module.f90 quick_overlap_module.f90
	quick_divcon_module.f90 quick_ecp_module.f90 quick_electrondensity_module.f90
	quick_files_module.f90 quick_timer_module.f90 quick_gridpoints_module.f90
	quick_mfcc_module.f90 quick_params_module.f90 quick_pb_module.f90
	quick_scratch_module.f90 quick_all_module.f90 quick_scf_module.f90
	quick_grad_cshell_module.f90 quick_api_module.f90 quick_api_test_module.f90
	quick_cutoff_module.f90 quick_exception_module.f90 quick_input_parser_module.f90
	quick_eri_cshell_module.f90 quick_eri_grad_cshell_module.f90 quick_oei_module.f90
	quick_dft_module.f90 quick_scf_operator_module.f90 quick_uscf_operator_module.f90
	quick_uscf_module.f90 quick_sad_guess_module.f90 quick_optimizer_module.f90
        quick_cew_module.f90 quick_lri_module.f90 quick_lri_grad_module.f90
	quick_eri_oshell_module.f90 quick_eri_grad_oshell_module.f90
	quick_grad_oshell_module.f90 quick_dftd3_module.f90 quick_molden_module.f90
	quick_restart_module.f90)

set(QUICK_SUBS_FORTRAN_SOURCES Angles.f90 copyDMat.f90 copySym.f90
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
	oshell_denspt.f90 naive_distribute.f90)

set(QUICK_SUBS_CXX_SOURCES
	time.cpp)

set(DLFIND_MODULES_SOURCES
	dlf_allocate.f90 dlf_checkpoint.f90
	dlf_conint.f90 dlf_convergence.f90
	dlf_coords.f90 dlf_dimer.f90
	dlf_formstep.f90 dlf_global_module.f90
	dlf_hdlc_constraint.f90 dlf_hdlc_hdlclib.f90
	dlf_hdlc_interface.f90 dlf_hdlc_matrixlib.f90
	dlf_hdlc_primitive.f90 dl_find.f90
	dlfind_main_driver.f90
	dlf_lbfgs.f90 dlf_linalg.f90
	dlf_microiter.f90
	dlf_neb.f90 dlf_parallel_opt.f90
	dlf_qts.f90 dlf_scalestep.f90
	dlf_serial.f90 dlf_sort.f90
	dlf_stat_module.f90 dlf_svnversion.f90
	dlf_task.f90 dlf_time.f90 dlf_util.f90)

set(DFTD3_SOURCES
	api.f90  common.f90  core.f90  pars.f90  sizes.f90)

if(mirp_ENABLED)
	list(APPEND QUICK_SUBS_CXX_SOURCES mirp_fmt.cpp)
endif()

set(QUICK_GENERAL_SOURCES initialize.f90 read_job_and_atom.f90
	getMol.f90 quick_one_electron_integral.f90 getEnergy.f90
	ecp.f90 nuclear.f90 finalize.f90 mpi_setup.f90 quick_debug.f90
	calMP2.f90 hessian.f90 CPHF.f90 frequency.f90 basis.f90)

set(QUICK_GPU_FORTRAN_SOURCE gpu/${QUICK_GPU_TARGET_NAME}/gpu_write_info.f90)

set(TEST_API_SOURCES quick_api_test.f90)
set(MAIN_SOURCES main.f90)


# the "subs" and "modules" source lists are for items in subdirectories.  We have to prepend the subdirectory path to these items.
# --------------------------------------------------------------------

set(QUICK_MODULES_SOURCES_FULLPATH "")
set(QUICK_SUBS_FORTRAN_SOURCES_FULLPATH "")
set(QUICK_SUBS_CXX_SOURCES_FULLPATH "")
set(DLFIND_MODULES_SOURCES_FULLPATH "")
set(DFTD3_SOURCES_FULLPATH "")

foreach(MODULES_SOURCE ${QUICK_MODULES_SOURCES})
	list(APPEND QUICK_MODULES_SOURCES_FULLPATH "modules/${MODULES_SOURCE}")
endforeach()

foreach(SUBS_SOURCE ${QUICK_SUBS_FORTRAN_SOURCES})
	list(APPEND QUICK_SUBS_FORTRAN_SOURCES_FULLPATH "subs/${SUBS_SOURCE}")
endforeach()

foreach(SUBS_SOURCE ${QUICK_SUBS_CXX_SOURCES})
	list(APPEND QUICK_SUBS_CXX_SOURCES_FULLPATH "subs/${SUBS_SOURCE}")
endforeach()

foreach(DLFIND_SOURCE ${DLFIND_MODULES_SOURCES})
	list(APPEND DLFIND_MODULES_SOURCES_FULLPATH "dlfind/${DLFIND_SOURCE}")
endforeach()

foreach(DFTD3_SOURCE ${DFTD3_SOURCES})
	list(APPEND DFTD3_SOURCES_FULLPATH "dftd3/${DFTD3_SOURCE}")
endforeach()

# create serial and MPI libraries
# --------------------------------------------------------------------

include_directories(util)

add_library(libquick SHARED ${QUICK_MODULES_SOURCES_FULLPATH} ${DFTD3_SOURCES_FULLPATH}
	${QUICK_SUBS_FORTRAN_SOURCES_FULLPATH} ${QUICK_SUBS_CXX_SOURCES_FULLPATH}
	${DLFIND_MODULES_SOURCES_FULLPATH} ${QUICK_GENERAL_SOURCES})
target_compile_options(libquick PRIVATE
	"$<$<COMPILE_LANGUAGE:FORTRAN>:${OPT_FFLAGS}>")
if(HDF5)
	target_compile_definitions(libquick PRIVATE
		"$<$<COMPILE_LANGUAGE:FORTRAN>:${HDF5_Fortran_DEFINITIONS}>" "HDF5")
	target_include_directories(libquick PRIVATE 
		"$<$<COMPILE_LANGUAGE:FORTRAN>:${HDF5_Fortran_INCLUDE_DIRS}>")
	config_module_dirs(libquick quick/serial libxc/serial ${HDF5_Fortran_INCLUDE_DIRS})
	target_link_libraries(libquick PRIVATE xc octree ${QUICK_LAPACK} ${QUICK_BLAS} ${HDF5_LIBRARIES})
else()
	config_module_dirs(libquick quick/serial libxc/serial)
	target_link_libraries(libquick PRIVATE xc octree ${QUICK_LAPACK} ${QUICK_BLAS})
endif()

if(INSIDE_AMBER)
	target_link_libraries(libquick PRIVATE libcew)
endif()
install_libraries(libquick EXPORT QUICK)

if(mirp_ENABLED)
	target_compile_definitions(libquick PRIVATE MIRP)
	target_link_libraries(libquick PRIVATE mirp::mirp)
endif()

# "liblibquick" -> "libquick"
set_target_properties(libquick PROPERTIES OUTPUT_NAME quick)

if(MPI)
	make_mpi_version(libquick libquick_mpi LANGUAGES Fortran)

	target_compile_definitions(libquick_mpi PRIVATE MPIV)
	set_target_properties(libquick_mpi PROPERTIES OUTPUT_NAME quick_mpi)

	# change link libraries and mod dirs for MPI
	remove_link_libraries(libquick_mpi octree)
	target_link_libraries(libquick_mpi PRIVATE octree_mpi)
	if(INSIDE_AMBER)
		target_link_libraries(libquick_mpi PRIVATE libcew_mpi)
	endif()
	config_module_dirs(libquick_mpi quick/mpi libxc/serial)


	install_libraries(libquick_mpi EXPORT QUICK)
endif()

if(CUDA)
	if(HIP)
		if(MAGMA)
			copy_target(libquick libquick_${QUICK_GPU_TARGET_NAME} SWAP_SOURCES TO
				$<TARGET_OBJECTS:rocblas_fortran_obj> ${QUICK_GPU_FORTRAN_SOURCE})
			target_compile_definitions(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE WITH_MAGMA)
		elseif(${HIP_VERSION} VERSION_GREATER_EQUAL 5.3.0)
			copy_target(libquick libquick_${QUICK_GPU_TARGET_NAME} SWAP_SOURCES TO
				$<TARGET_OBJECTS:rocblas_fortran_obj> $<TARGET_OBJECTS:rocsolver_obj> ${QUICK_GPU_FORTRAN_SOURCE})
			target_compile_definitions(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE WITH_ROCSOLVER)
		# avoid diagonalization on GPU with rocSOLVER on older ROCm versions due to poor performance,
		# and instead fall back to CPU diagonalization
		else()
			copy_target(libquick libquick_${QUICK_GPU_TARGET_NAME} SWAP_SOURCES TO
				$<TARGET_OBJECTS:rocblas_fortran_obj> ${QUICK_GPU_FORTRAN_SOURCE})
		endif()
		target_compile_definitions(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE GPU ${QUICK_GPU_PLATFORM})
		target_link_options(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE ${GPU_LD_FLAGS})

		set_target_properties(libquick_${QUICK_GPU_TARGET_NAME} PROPERTIES OUTPUT_NAME quick_${QUICK_GPU_TARGET_NAME})

		# change link libraries and mod dirs for CUDA
		remove_link_libraries(libquick_${QUICK_GPU_TARGET_NAME} xc octree)

		target_link_libraries(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE
			xc_gpu octree_gpu quick_${QUICK_GPU_TARGET_NAME}_kernels rocblas)

		if(MAGMA)
			target_link_libraries(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE magma)
		elseif(${HIP_VERSION} VERSION_GREATER_EQUAL 5.3.0)
			target_link_libraries(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE rocsolver)
		else()
			target_link_libraries(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE lapack)
		endif()

		if(INSIDE_AMBER)
			target_link_libraries(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE libcew)
		endif()

		config_module_dirs(libquick_${QUICK_GPU_TARGET_NAME} quick/${QUICK_GPU_TARGET_NAME} libxc/gpu)

		install_libraries(libquick_${QUICK_GPU_TARGET_NAME} EXPORT QUICK)
	else()
		copy_target(libquick libquick_${QUICK_GPU_TARGET_NAME} SWAP_SOURCES TO
			$<TARGET_OBJECTS:cublas_fortran_obj> $<TARGET_OBJECTS:cusolver_obj> ${QUICK_GPU_FORTRAN_SOURCE})
		target_compile_definitions(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE GPU ${QUICK_GPU_PLATFORM})
		set_target_properties(libquick_${QUICK_GPU_TARGET_NAME} PROPERTIES OUTPUT_NAME quick_${QUICK_GPU_TARGET_NAME})

		# change link libraries and mod dirs for CUDA
		remove_link_libraries(libquick_${QUICK_GPU_TARGET_NAME} xc octree)

		target_link_libraries(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE xc_gpu octree_gpu
			quick_${QUICK_GPU_TARGET_NAME}_kernels cublas cusolver)

		if(INSIDE_AMBER)
			target_link_libraries(libquick_${QUICK_GPU_TARGET_NAME} PRIVATE libcew)
		endif()

		config_module_dirs(libquick_${QUICK_GPU_TARGET_NAME} quick/${QUICK_GPU_TARGET_NAME} libxc/gpu)

		install_libraries(libquick_${QUICK_GPU_TARGET_NAME} EXPORT QUICK)
	endif()
endif()

if(MPI AND CUDA)
	if(HIP)
		if(MAGMA)
			copy_target(libquick_mpi libquick_mpi_${QUICK_GPU_TARGET_NAME} SWAP_SOURCES TO
				$<TARGET_OBJECTS:rocblas_fortran_obj_mpi> ${QUICK_GPU_FORTRAN_SOURCE})
			target_compile_definitions(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE WITH_MAGMA)
		elseif(${HIP_VERSION} VERSION_GREATER_EQUAL 5.3.0)
			copy_target(libquick_mpi libquick_mpi_${QUICK_GPU_TARGET_NAME} SWAP_SOURCES TO
				$<TARGET_OBJECTS:rocblas_fortran_obj_mpi> $<TARGET_OBJECTS:rocsolver_obj_mpi> ${QUICK_GPU_FORTRAN_SOURCE})
			target_compile_definitions(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE WITH_ROCSOLVER)
		# avoid diagonalization on GPU with rocSOLVER on older ROCm versions due to poor performance,
		# and instead fall back to CPU diagonalization
		else()
			copy_target(libquick_mpi libquick_mpi_${QUICK_GPU_TARGET_NAME} SWAP_SOURCES TO
				$<TARGET_OBJECTS:rocblas_fortran_obj_mpi> ${QUICK_GPU_FORTRAN_SOURCE})
		endif()
		target_compile_definitions(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE MPIV_GPU ${QUICK_GPU_PLATFORM}_MPIV)
		target_link_options(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE ${GPU_LD_FLAGS})

		set_target_properties(libquick_mpi_${QUICK_GPU_TARGET_NAME} PROPERTIES OUTPUT_NAME quick_mpi_${QUICK_GPU_TARGET_NAME})

		remove_link_libraries(libquick_mpi_${QUICK_GPU_TARGET_NAME} octree_mpi xc)
		target_link_libraries(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE
			xc_gpu xc_${QUICK_GPU_TARGET_NAME} octree_gpu quick_${QUICK_GPU_TARGET_NAME}_kernels_mpi rocblas)
		if(MAGMA)
			target_link_libraries(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE magma)
		elseif(${HIP_VERSION} VERSION_GREATER_EQUAL 5.3.0)
			target_link_libraries(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE rocsolver)
		else()
			target_link_libraries(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE lapack)
		endif()
		if(INSIDE_AMBER)
			target_link_libraries(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE libcew_mpi)
		endif()

		config_module_dirs(libquick_mpi_${QUICK_GPU_TARGET_NAME} quick/mpi_${QUICK_GPU_TARGET_NAME} libxc/gpu)

		install_libraries(libquick_mpi_${QUICK_GPU_TARGET_NAME} EXPORT QUICK)
	else()
		copy_target(libquick_mpi libquick_mpi_${QUICK_GPU_TARGET_NAME} SWAP_SOURCES TO
			$<TARGET_OBJECTS:cublas_fortran_obj> $<TARGET_OBJECTS:cusolver_obj> ${QUICK_GPU_FORTRAN_SOURCE})
		target_compile_definitions(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE MPIV_GPU ${QUICK_GPU_PLATFORM}_MPIV)
		set_target_properties(libquick_mpi_${QUICK_GPU_TARGET_NAME} PROPERTIES OUTPUT_NAME quick_mpi_${QUICK_GPU_TARGET_NAME})

		remove_link_libraries(libquick_mpi_${QUICK_GPU_TARGET_NAME} octree_mpi xc)

		target_link_libraries(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE xc_gpu xc_${QUICK_GPU_TARGET_NAME} octree_gpu
			quick_${QUICK_GPU_TARGET_NAME}_kernels_mpi cublas cusolver)

		if(INSIDE_AMBER)
			target_link_libraries(libquick_mpi_${QUICK_GPU_TARGET_NAME} PRIVATE libcew_mpi)
		endif()

		config_module_dirs(libquick_mpi_${QUICK_GPU_TARGET_NAME} quick/mpi_${QUICK_GPU_TARGET_NAME} libxc/gpu)

		install_libraries(libquick_mpi_${QUICK_GPU_TARGET_NAME} EXPORT QUICK)
	endif()
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
target_compile_options(quick PRIVATE 
	"$<$<COMPILE_LANGUAGE:FORTRAN>:${OPT_FFLAGS}>"
	"$<$<COMPILE_LANGUAGE:C>:${OPT_CFLAGS}>"
	"$<$<COMPILE_LANGUAGE:CXX>:${OPT_CXXFLAGS}>")
config_module_dirs(quick quick/serial libxc/serial)
target_link_libraries(quick libquick)

add_executable(test-api ${TEST_API_SOURCES})
target_compile_options(test-api PRIVATE
	"$<$<COMPILE_LANGUAGE:FORTRAN>:${OPT_FFLAGS}>"
	"$<$<COMPILE_LANGUAGE:C>:${OPT_CFLAGS}>"
	"$<$<COMPILE_LANGUAGE:CXX>:${OPT_CXXFLAGS}>")
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
	copy_target(quick quick.${QUICK_GPU_TARGET_NAME})
	target_compile_definitions(quick.${QUICK_GPU_TARGET_NAME} PRIVATE GPU ${QUICK_GPU_PLATFORM})

	copy_target(test-api test-api.${QUICK_GPU_TARGET_NAME})
	target_compile_definitions(test-api.${QUICK_GPU_TARGET_NAME} PRIVATE GPU ${QUICK_GPU_PLATFORM})

	remove_link_libraries(quick.${QUICK_GPU_TARGET_NAME} libquick)
	target_link_libraries(quick.${QUICK_GPU_TARGET_NAME} libquick_${QUICK_GPU_TARGET_NAME})
	config_module_dirs(quick.${QUICK_GPU_TARGET_NAME} quick/${QUICK_GPU_TARGET_NAME} libxc/gpu)

	remove_link_libraries(test-api.${QUICK_GPU_TARGET_NAME} libquick)
	target_link_libraries(test-api.${QUICK_GPU_TARGET_NAME} libquick_${QUICK_GPU_TARGET_NAME})
	config_module_dirs(test-api.${QUICK_GPU_TARGET_NAME} quick/${QUICK_GPU_TARGET_NAME} libxc/gpu)

	install(TARGETS quick.${QUICK_GPU_TARGET_NAME} DESTINATION ${BINDIR} EXPORT QUICK)
	install(TARGETS test-api.${QUICK_GPU_TARGET_NAME} DESTINATION ${BINDIR} EXPORT QUICK)
endif()

if(CUDA AND MPI)
	copy_target(quick.MPI quick.${QUICK_GPU_TARGET_NAME}.MPI)
	target_compile_definitions(quick.${QUICK_GPU_TARGET_NAME}.MPI PRIVATE MPIV_GPU ${QUICK_GPU_PLATFORM}_MPIV)

	copy_target(test-api.MPI test-api.${QUICK_GPU_TARGET_NAME}.MPI)
	target_compile_definitions(test-api.${QUICK_GPU_TARGET_NAME}.MPI PRIVATE MPIV_GPU ${QUICK_GPU_PLATFORM}_MPIV)

	remove_link_libraries(quick.${QUICK_GPU_TARGET_NAME}.MPI libquick_mpi)
	target_link_libraries(quick.${QUICK_GPU_TARGET_NAME}.MPI libquick_mpi_${QUICK_GPU_TARGET_NAME})
	config_module_dirs(quick.${QUICK_GPU_TARGET_NAME}.MPI quick/mpi_${QUICK_GPU_TARGET_NAME} libxc/gpu)

	remove_link_libraries(test-api.${QUICK_GPU_TARGET_NAME}.MPI libquick_mpi)
	target_link_libraries(test-api.${QUICK_GPU_TARGET_NAME}.MPI libquick_mpi_${QUICK_GPU_TARGET_NAME})
	config_module_dirs(test-api.${QUICK_GPU_TARGET_NAME}.MPI quick/mpi_${QUICK_GPU_TARGET_NAME} libxc/gpu)

	install(TARGETS quick.${QUICK_GPU_TARGET_NAME}.MPI DESTINATION ${BINDIR} EXPORT QUICK)
	install(TARGETS test-api.${QUICK_GPU_TARGET_NAME}.MPI DESTINATION ${BINDIR} EXPORT QUICK)
endif()

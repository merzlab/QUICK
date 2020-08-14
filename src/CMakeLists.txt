# recurse to subdirs
# --------------------------------------------------------------------
add_subdirectory(libxc)
add_subdirectory(octree)

if(blas_INTERNAL)
	add_subdirectory(blas)
endif()

if(CUDA)
	add_subdirectory(cuda)
endif()

# quick source lists
# --------------------------------------------------------------------

set(QUICK_MODULES_SOURCES
	quick_mpi_module.f90 quick_constants_module.f90 quick_method_module.f90 
	quick_molspec_module.f90 quick_gaussian_class_module.f90 quick_size_module.f90 
	quick_amber_interface_module.f90 quick_basis_module.f90 quick_calculated_module.f90 
	quick_divcon_module.f90 quick_ecp_module.f90 quick_electrondensity_module.f90 
	quick_files_module.f90 quick_timer_module.f90 quick_gridpoints_module.f90 
	quick_mfcc_module.f90 quick_params_module.f90 quick_pb_module.f90 
	quick_scratch_module.f90 quick_all_module.f90 quick_scf_module.f90 
	quick_gradient_module.f90 quick_api_module.f90 quick_api_test_module.f90)

set(QUICK_SUBS_SOURCES Angles.f90 copyDMat.f90 copySym.f90 
	degen.f90 denspt.f90 diag.f90 dipole.f90 
	EffChar.f90 eigvec.f90 
	ekinetic.f90 findBlock.f90 fmt.f90 getinum.f90 
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
	pteval_new_imp.f90 scaMatMul.f90)

set(QUICK_GENERAL_SOURCES initialize.f90 read_job_and_atom.f90 fmm.f90 
    getMolSad.f90 getMol.f90 shell.f90 schwarz.f90 
    quick_one_electron_integral.f90 getEnergy.f90 inidivcon.f90 
    ecp.f90 hfoperator.f90 nuclear.f90 
    dft.f90 sedftoperator.f90 
    scf.f90 uscf.f90 finalize.f90 uhfoperator.f90 
    udftoperator.f90 usedftoperator.f90 
    uelectdii.f90 mpi_setup.f90 quick_debug.f90 
    calMP2.f90 optimize.f90 gradient.f90 hessian.f90 
    CPHF.f90 frequency.f90 MFCC.f90 basis.f90 
    fake_amber_interface.f90 scf_operator.f90)

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

add_library(libquick SHARED ${QUICK_MODULES_SOURCES_FULLPATH} ${QUICK_SUBS_SOURCES_FULLPATH} ${QUICK_GENERAL_SOURCES})
target_compile_options(libquick PRIVATE ${OPT_FFLAGS})
config_module_dirs(libquick quick/serial libxc/serial)
target_link_libraries(libquick PRIVATE xc octree blas)
install_libraries(libquick EXPORT QUICK)

# "liblibquick" -> "libquick"
set_property(TARGET libquick PROPERTY OUTPUT_NAME quick)

if(MPI)
	make_mpi_version(libquick libquick_mpi LANGUAGES Fortran)

	target_compile_definitions(libquick_mpi PRIVATE MPIV)
	set_property(TARGET libquick_mpi PROPERTY OUTPUT_NAME quick_mpi)

	# change link libraries and mod dirs for MPI
	remove_link_libraries(libquick_mpi xc octree)
	target_link_libraries(libquick_mpi PRIVATE xc_mpi octree_mpi)
	config_module_dirs(libquick_mpi quick/mpi libxc/mpi)


	install_libraries(libquick_mpi EXPORT QUICK)
endif()

if(CUDA)
	copy_target(libquick libquick_cuda SWAP_SOURCES TO $<TARGET_OBJECTS:cublas_fortran_obj> $<TARGET_OBJECTS:cusolver_obj>)
	target_compile_definitions(libquick_cuda PRIVATE CUDA)
	set_property(TARGET libquick_cuda PROPERTY OUTPUT_NAME quick_cuda)

	# change link libraries and mod dirs for CUDA
	remove_link_libraries(libquick_cuda xc octree blas)
	target_link_libraries(libquick_cuda PRIVATE xc_gpu xc_cuda octree_gpu quick_cuda PUBLIC ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
	config_module_dirs(libquick_cuda quick/cuda libxc/gpu)

	install_libraries(libquick_cuda EXPORT QUICK)
endif()

if(MPI AND CUDA)
	copy_target(libquick_mpi libquick_mpi_cuda SWAP_SOURCES TO $<TARGET_OBJECTS:cublas_fortran_obj> $<TARGET_OBJECTS:cusolver_obj>)
	target_compile_definitions(libquick_mpi_cuda PRIVATE CUDA_MPIV)
	set_property(TARGET libquick_mpi_cuda PROPERTY OUTPUT_NAME quick_mpi_cuda)

	remove_link_libraries(libquick_mpi_cuda octree_mpi xc_mpi)
	target_link_libraries(libquick_mpi_cuda PRIVATE xc_gpu xc_cuda octree_gpu quick_cuda_mpi PUBLIC ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
	config_module_dirs(libquick_mpi_cuda quick/mpi_cuda libxc/gpu)

	install_libraries(libquick_mpi_cuda EXPORT QUICK)
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

if(MPI)
	make_mpi_version(quick quick.mpi LANGUAGES Fortran)
	target_compile_definitions(quick.mpi PRIVATE MPIV)

	make_mpi_version(test-api test-api.mpi LANGUAGES Fortran)
	target_compile_definitions(test-api.mpi PRIVATE MPIV)

	# change link libraries and mod dirs for MPI
	remove_link_libraries(quick.mpi libquick)
	target_link_libraries(quick.mpi libquick_mpi)
	config_module_dirs(quick.mpi quick/mpi libxc/mpi)

	remove_link_libraries(test-api.mpi libquick)
	target_link_libraries(test-api.mpi libquick_mpi)
	config_module_dirs(test-api.mpi quick/mpi libxc/mpi)

	install(TARGETS quick.mpi DESTINATION ${BINDIR} EXPORT QUICK)
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
endif()

if(CUDA AND MPI)
	copy_target(quick.mpi quick.mpi.cuda)
	target_compile_definitions(quick.mpi.cuda PRIVATE CUDA_MPIV)

	copy_target(test-api.mpi test-api.mpi.cuda)
	target_compile_definitions(test-api.mpi.cuda PRIVATE CUDA_MPIV)

	remove_link_libraries(quick.mpi.cuda libquick_mpi)
	target_link_libraries(quick.mpi.cuda libquick_mpi_cuda)
	config_module_dirs(quick.mpi.cuda quick/mpi_cuda libxc/gpu)

	remove_link_libraries(test-api.mpi.cuda libquick_mpi)
	target_link_libraries(test-api.mpi.cuda libquick_mpi_cuda)
	config_module_dirs(test-api.mpi.cuda quick/mpi_cuda libxc/gpu)

	install(TARGETS quick.mpi.cuda DESTINATION ${BINDIR} EXPORT QUICK)
endif()
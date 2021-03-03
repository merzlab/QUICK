cmake_minimum_required(VERSION 3.9.0) # version 3.9.0 needed for FindMPI and FindOpenMP versions which provide imported targets

project(quick LANGUAGES NONE VERSION 20.03)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/quick-cmake)
if(NOT INSIDE_AMBER)

	# initialization and include paths
	#-------------------------------------------------------------
	include(cmake/AmberBuildSystemInit.cmake NO_POLICY_SCOPE)

	#now enable the languages
	enable_language(C CXX Fortran)

	include(AmberBuildSystem2ndInit)
	
	include(MPIConfig)

	set(LINALG_LIBS_REQUIRED TRUE)
	set(NEEDED_3RDPARTY_TOOLS blas lapack mkl mirp)
	set(REQUIRED_3RDPARTY_TOOLS )
	set(BUNDLED_3RDPARTY_TOOLS blas lapack)
	include(3rdPartyTools)
	
	include(CompilerFlags)

	#CPack setup
	# --------------------------------------------------------------------
	set(PACKAGE_NAME "QUICK")
	set(PACKAGE_FILENAME "QUICK")

	set(BUNDLE_IDENTIFIER org.merzlab.quick)
	set(BUNDLE_SIGNATURE QUIK)
	include(Packaging)

endif()

option(NOF "Disables the compilation of QUICK's time consuming f functions in the ERI code of cuda version. Not recommended for production." FALSE)

# C/Fortran compiler flags
# --------------------------------------------------------------------

# add -DDEBUG in debug configuration
set_property(DIRECTORY . PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:DEBUG>)


if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")

	# QUICK needs to be built with -O2 not Amber's default of -O3
	if(OPTIMIZE)
		list(APPEND OPT_CFLAGS -O2)
		set(OPT_CFLAGS_SPC "${OPT_CFLAGS_SPC} -O2")
	endif()

elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -traceback")
endif()


if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")
	add_definitions(-DGNU)

	if(OPTIMIZE)
		list(APPEND OPT_FFLAGS -O2)
		set(OPT_FFLAGS_SPC "${OPT_FFLAGS_SPC} -O2")
	endif()

	if(CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
		set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch")
	endif()

endif()
if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp -diag-disable 8291")
	set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -traceback")
endif()
if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI")
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp") # force use of preprocessor for Fortran files with the "wrong" file extension
endif()

# blas vs lapack vs MKL selection
if(lapack_ENABLED)
	set(QUICK_LAPACK lapack)
elseif(mkl_ENABLED)
	set(QUICK_LAPACK ${MKL_FORTRAN_LIBRARIES})
else()
	set(QUICK_LAPACK "")
endif()

if(blas_ENABLED)
	set(QUICK_BLAS blas)
elseif(mkl_ENABLED)
	set(QUICK_BLAS ${MKL_FORTRAN_LIBRARIES})
endif()

# debug options
option(QUICK_DEBUG_TIME "Compiles a debug version of QUICK that reports more information on timing" FALSE)
option(QUICK_DEBUG "Compiles a debug version of QUICK with extra prints enabled." FALSE)
option(QUICK_VERBOSE_PTXAS "Enable verbose output from CUDA's ptxas." FALSE)

if(QUICK_DEBUG)
	add_definitions(-DDEBUG)
endif()
if(QUICK_DEBUG OR QUICK_DEBUG_TIME)
	add_definitions(-DDEBUGTIME)
endif()

# CUDA compiler flags
# --------------------------------------------------------------------
include(QUICKCudaConfig)

#--------------------------------------------------------------	
# Build code

add_subdirectory(src)

if(NOT INSIDE_AMBER) # Amber is currently not set up for exported targets

	#--------------------------------------------------------------	
	# Exported targets

	# directory relative to prefix where config files will be installed
	set(CMAKE_PACKAGE_CONFIG_DIR share/cmake/QUICK)

	# install targets file
	install(EXPORT QUICK 
		FILE QUICKTargets.cmake
		NAMESPACE QUICK::
		DESTINATION ${CMAKE_PACKAGE_CONFIG_DIR})

	# install version file
	include(CMakePackageConfigHelpers)
	write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/QUICKConfigVersion.cmake
		VERSION ${${PROJECT_NAME}_VERSION}
		COMPATIBILITY AnyNewerVersion)

	# install top-level file
	configure_package_config_file(QUICKConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/QUICKConfig.cmake
		INSTALL_DESTINATION ${CMAKE_PACKAGE_CONFIG_DIR})

	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/QUICKConfigVersion.cmake ${CMAKE_CURRENT_BINARY_DIR}/QUICKConfig.cmake DESTINATION ${CMAKE_PACKAGE_CONFIG_DIR})

endif()

#--------------------------------------------------------------	
# Additional files

if(INSIDE_AMBER)

	# install data in basis dir
	install(DIRECTORY basis DESTINATION AmberTools/src/quick)

	# install tests dir
	install(DIRECTORY test DESTINATION AmberTools/src/quick)

else()
	# Standalone install

	# install modules
	install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/amber-modules/ DESTINATION ./include)

	# install data in basis dir
	install(DIRECTORY basis DESTINATION . USE_SOURCE_PERMISSIONS)

	# install tests dir
	install(DIRECTORY test DESTINATION . USE_SOURCE_PERMISSIONS)

	# install scripts
	install(PROGRAMS tools/runtest DESTINATION ${BINDIR} RENAME quick_runtest)
	install(PROGRAMS quick-cmake/quick.rc DESTINATION ${BINDIR})
endif()

#--------------------------------------------------------------	
# build report
if(NOT INSIDE_AMBER)
	print_build_report()
endif()
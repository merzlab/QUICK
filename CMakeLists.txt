cmake_minimum_required(VERSION 3.12.0) # version 3.9.0 needed for FindMPI and FindOpenMP versions which provide imported targets

project(quick LANGUAGES NONE VERSION 25.03)

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

option(ENABLEF "Enables the support for f functions in the ERI code." FALSE)

# Compiler flags
# These should really go into cmake/CompilerFlags.cmake but with the
# current setup in Amber the Amber compilation flags will take
# precedence so we need to overwrite those here until we have a fully
# independent build system for Quick and integrate Quick in Amber
# as ExternalProject
# --------------------------------------------------------------------

# add -DDEBUG in debug configuration
set_property(DIRECTORY . PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:DEBUG>)

option(WARNINGS "Enable warnings." FALSE)

set(CMAKE_Fortran_FLAGS "")
set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "")

set(OPT_FFLAGS "")
set(OPT_CFLAGS "")
set(OPT_CXXFLAGS "")

set(NO_OPT_FFLAGS "")
set(NO_OPT_CFLAGS "")
set(NO_OPT_CXXFLAGS "")

list(APPEND NO_OPT_FFLAGS "-O0")
list(APPEND NO_OPT_CFLAGS "-O0")
list(APPEND NO_OPT_CXXFLAGS "-O0")

# GNU
# --------------------------------------------------------------------
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND TARGET_ARCH STREQUAL "arm64")
	    execute_process(COMMAND sysctl machdep.cpu.brand_string OUTPUT_VARIABLE CPU_INFO)

	    string(REGEX MATCH "'(M[0-9]*)'" _ "${CPU_INFO}")
	    set(CPU_VERSION ${CMAKE_MATCH_1})

	    if("${CPU_TYPE}" STREQUAL "M1" OR "${CPU_TYPE}" STREQUAL "M2")
		    list(APPEND OPT_CFLAGS "-O2" "-mtune=native")
	    else()
		    # workaround GCC code generation issues with M3 and newer CPUs
		    list(APPEND OPT_CFLAGS "-O2")
	    endif()
    else()
        list(APPEND OPT_CFLAGS "-O2" "-mtune=native")
    endif()

    if (WARNINGS)
	add_flags(C -Wall -Wno-unused-function -Wno-unknown-pragmas)
    
	if(NOT UNUSED_WARNINGS)
	    add_flags(C -Wno-unused-variable -Wno-unused-but-set-variable)
	endif()
    
	if(NOT UNINITIALIZED_WARNINGS)
	    add_flags(C -Wno-uninitialized -Wno-maybe-uninitialized)
	endif()
    endif()
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND TARGET_ARCH STREQUAL "arm64")
	    if("${CPU_TYPE}" STREQUAL "M1" OR "${CPU_TYPE}" STREQUAL "M2")
		    list(APPEND OPT_CFLAGS "-O2" "-mtune=native")
	    else()
		    # workaround GCC code generation issues with M3 and newer CPUs
		    list(APPEND OPT_CFLAGS "-O2")
	    endif()
    else()
        list(APPEND OPT_CFLAGS "-O2" "-mtune=native")
    endif()

    if (WARNINGS)
	add_flags(CXX -Wall -Wno-unused-function -Wno-unknown-pragmas)
    
	# Kill it!  Kill it with fire!
	check_cxx_compiler_flag(-Wno-unused-local-typedefs SUPPORTS_WNO_UNUSED_LOCAL_TYPEDEFS)

	if(SUPPORTS_WNO_UNUSED_LOCAL_TYPEDEFS)
	    add_flags(CXX -Wno-unused-local-typedefs)
	endif()
	
	if(NOT UNUSED_WARNINGS)
	    add_flags(CXX -Wno-unused-variable -Wno-unused-but-set-variable)
	endif()
	
	if(NOT UNINITIALIZED_WARNINGS)
	    add_flags(CXX -Wno-uninitialized -Wno-maybe-uninitialized)
	endif()
    endif()
endif()

if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND TARGET_ARCH STREQUAL "arm64")
	    if("${CPU_TYPE}" STREQUAL "M1" OR "${CPU_TYPE}" STREQUAL "M2")
		    list(APPEND OPT_CFLAGS "-O2" "-mtune=native")
	    else()
		    # workaround GCC code generation issues with M3 and newer CPUs
		    list(APPEND OPT_CFLAGS "-O2")
	    endif()
    else()
        list(APPEND OPT_CFLAGS "-O2" "-mtune=native")
    endif()

    add_flags(Fortran -ffree-line-length-none)

    if (WARNINGS)
	add_flags(Fortran -Wall -Wno-tabs -Wno-unused-function -ffree-line-length-none -Wno-unused-dummy-argument)
	if(NOT UNUSED_WARNINGS)
	    add_flags(Fortran -Wno-unused-variable)
	endif()
	if(NOT UNINITIALIZED_WARNINGS)
	    add_flags(Fortran -Wno-maybe-uninitialized)
	endif()	
    endif()

    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

    if(CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch")
    endif()

    add_definitions(-DGNU)
endif()


# Intel
# --------------------------------------------------------------------

if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_C_FLAGS_DEBUG "-g -debug all -traceback")
    list(APPEND OPT_CFLAGS "-ip" "-O2" "-xHost")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -debug all -traceback")
    list(APPEND OPT_CXXFLAGS "-ip" "-O2" "-xHost")
endif()

if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -debug all -traceback")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp -diag-disable 8291")
    list(APPEND OPT_FFLAGS "-ip" "-O2" "-xHost")

    if(WARNINGS)
	add_flags(Fortran "-warn all" "-warn nounused")
	option(IFORT_CHECK_INTERFACES "If enabled and Intel Fortran is in use, then ifort will check that types passed to functions are the correct ones, and produce warnings or errors for mismatches." FALSE)
	if(NOT IFORT_CHECK_INTERFACES)
	    # disable errors from type mismatches
	    add_flags(Fortran -warn nointerfaces)
	endif()
    endif()		
endif()


# IntelLLVM
# --------------------------------------------------------------------

if("${CMAKE_C_COMPILER_ID}" STREQUAL "IntelLLVM")
    set(CMAKE_C_FLAGS_DEBUG "-g -debug all")
    list(APPEND OPT_CFLAGS "-O2")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -debug all")
    list(APPEND OPT_CXXFLAGS "-O2")
endif()

if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "IntelLLVM")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -debug all")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fpp -diag-disable 8291")
    list(APPEND OPT_FFLAGS "-O2")

    if(WARNINGS)
	add_flags(Fortran "-warn all" "-warn nounused")
	option(IFORT_CHECK_INTERFACES "If enabled and Intel Fortran is in use, then ifort will check that types passed to functions are the correct ones, and produce warnings or errors for mismatches." FALSE)
		
	if(NOT IFORT_CHECK_INTERFACES)
	    # disable errors from type mismatches
	    add_flags(Fortran -warn nointerfaces)
	endif()
    endif()		
endif()

# PGI or NVHPC
# --------------------------------------------------------------------

if("${CMAKE_C_COMPILER_ID}" STREQUAL "PGI" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "NVHPC")
    list(APPEND OPT_CFLAGS "-O2")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC")
    list(APPEND OPT_CXXFLAGS "-O2")
endif()

if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI" OR "${CMAKE_Fortran_COMPILER_ID}" STREQUAL "NVHPC")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")
    list(APPEND OPT_FFLAGS "-fast" "-O3")
endif()


# CRAY: NOT SUPPORTED
# --------------------------------------------------------------------
if(${COMPILER} STREQUAL CRAY)
    message(FATAL_ERROR "CRAY compiler not supported by QUICK.")
endif()


# disable optimization flags if optimization is disabled
if(NOT OPTIMIZE)
	set(OPT_FFLAGS ${NO_OPT_FFLAGS})
	set(OPT_CFLAGS ${NO_OPT_CFLAGS})
	set(OPT_CXXFLAGS ${NO_OPT_CXXFLAGS})
endif()

#create space-separated versions of each flag set for use in PROPERTY COMPILE_FLAGS
list_to_space_separated(OPT_FFLAGS_SPC ${OPT_FFLAGS})
list_to_space_separated(OPT_CFLAGS_SPC ${OPT_CFLAGS})
list_to_space_separated(OPT_CXXFLAGS_SPC ${OPT_CXXFLAGS})

list_to_space_separated(NO_OPT_FFLAGS_SPC ${NO_OPT_FFLAGS})
list_to_space_separated(NO_OPT_CFLAGS_SPC ${NO_OPT_CFLAGS})
list_to_space_separated(NO_OPT_CXXFLAGS_SPC ${NO_OPT_CXXFLAGS})


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

# extra compiling flags
if(INSIDE_AMBER)
  add_definitions(-DCEW)
endif()

# set general preprocessor flag for enabling F functions
if(ENABLEF)
  add_definitions(-DENABLEF)
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
	install(DIRECTORY basis DESTINATION AmberTools/src/quick COMPONENT Data)

	# install tests dir
	install(DIRECTORY test USE_SOURCE_PERMISSIONS DESTINATION AmberTools/src/quick COMPONENT Tests ${TESTS_EXCLUDE_OPTION})

        # install runtest script
        install(PROGRAMS tools/runtest DESTINATION AmberTools/src/quick COMPONENT Tests ${TESTS_EXCLUDE_OPTION})

else()
	# Standalone install

	# install modules
	install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/amber-modules/ DESTINATION ./include)

	# install data in basis dir
	install(DIRECTORY basis DESTINATION . USE_SOURCE_PERMISSIONS)

	# install tests dir
	install(DIRECTORY test DESTINATION . USE_SOURCE_PERMISSIONS)

	# install scripts
	install(PROGRAMS tools/runtest DESTINATION .)
	install(FILES quick-cmake/quick.rc DESTINATION .)
endif()

#--------------------------------------------------------------	
# build report
if(NOT INSIDE_AMBER)
	print_build_report()
endif()

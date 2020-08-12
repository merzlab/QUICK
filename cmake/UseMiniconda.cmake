# Downloads and installs a Miniconda appropriate for your operating system
# This script does not work when crosscompiling.

# Send the version variables up one scope level from the caller of this macro
macro(proxy_python_version)
	set(PYTHON_VERSION_MAJOR ${MINICONDA_VERSION_MAJOR} PARENT_SCOPE)
	set(PYTHON_VERSION_MINOR ${MINICONDA_VERSION_MINOR} PARENT_SCOPE)
	set(PYTHON_VERSION_PATCH ${MINICONDA_VERSION_PATCH} PARENT_SCOPE)
endmacro(proxy_python_version)
	
function(download_and_use_miniconda)
	# Set up a temporary directory
	set(MINICONDA_TEMP_DIR ${CMAKE_BINARY_DIR}/CMakeFiles/miniconda)
	set(MINICONDA_DOWNLOAD_DIR ${MINICONDA_TEMP_DIR}/download)
	set(MINICONDA_INSTALL_DIR ${MINICONDA_TEMP_DIR}/install)	
	
	set(MINICONDA_INSTALL_DIR ${MINICONDA_TEMP_DIR}/install PARENT_SCOPE) # need to keep this around for install_minconda()
	
	if(MINICONDA_USE_PY3)
	    set(PYTHON_MAJOR_RELEASE 3)	    
	else()
	    set(PYTHON_MAJOR_RELEASE 2)
	endif()
	
	set(MINICONDA_STAMP_FILE ${CMAKE_BINARY_DIR}/CMakeFiles/miniconda-setup-py${PYTHON_MAJOR_RELEASE}-v${MINICONDA_VERSION}.stamp)
	
	# figure out executable paths
	if(TARGET_WINDOWS)
		set(MINICONDA_PYTHON ${MINICONDA_INSTALL_DIR}/python.exe)
		set(CONDA ${MINICONDA_INSTALL_DIR}/Scripts/conda.exe)
		set(PIP ${MINICONDA_INSTALL_DIR}/Scripts/pip.exe)
	else()
		set(MINICONDA_PYTHON ${MINICONDA_INSTALL_DIR}/bin/python)

		# execute using interpreter in case the miniconda install folder is in a long path and the script shebang doesn't work
		set(CONDA ${MINICONDA_PYTHON} ${MINICONDA_INSTALL_DIR}/bin/conda) 
		set(PIP ${MINICONDA_PYTHON} ${MINICONDA_INSTALL_DIR}/bin/pip)
	endif()
	
	set(MINICONDA_PYTHON ${MINICONDA_PYTHON} PARENT_SCOPE)
	
	file(MAKE_DIRECTORY ${MINICONDA_TEMP_DIR} ${MINICONDA_DOWNLOAD_DIR})
	
	# check if we have already downloaded miniconda
	if(EXISTS ${MINICONDA_STAMP_FILE})
		proxy_python_version()
		message(STATUS "Miniconda is installed in the build directory!")
		return()
	endif()

	if(MINICONDA_USE_PY3)	
		message(STATUS "Downloading Python 3 Miniconda")	    
	else()
		message(STATUS "Downloading Python 2.7 Miniconda")
	endif()
		
	# Figure out the OS part of the URL
	if(TARGET_OSX)
	    message(STATUS "Detected Mac OS X operating system. Downloading the Mac installer")
	    set(CONTINUUM_SYSTEM_NAME "MacOSX")
	    set(INSTALLER_SUFFIX sh)
	    
	elseif(TARGET_WINDOWS)
	    message(STATUS "Detected Windows operating system. Downloading the Windows installer")
	    set(CONTINUUM_SYSTEM_NAME "Windows")
	    set(INSTALLER_SUFFIX exe)
	    
	elseif(TARGET_LINUX)
	    message(STATUS "Detected Linux OS. Downloading the Linux installer")
	    set(CONTINUUM_SYSTEM_NAME "Linux")
	    set(INSTALLER_SUFFIX sh)
	else()
	    message(STATUS "Unrecognized CMAKE_SYSTEM_NAME, assuming Linux")
	   	set(CONTINUUM_SYSTEM_NAME "Linux")
	    set(INSTALLER_SUFFIX sh)
	endif()
		        
	# Figure out the bitiness part of the URL
	if(TARGET_OSX)
		# OS X does not have a 32 bit miniconda
		set(CONTINUUM_BITS "x86_64")
	else()
		if("${TARGET_ARCH}" STREQUAL "x86_64")
			message(STATUS "Using 64 bit miniconda")
			set(CONTINUUM_BITS "x86_64")
		elseif("${TARGET_ARCH}" STREQUAL "i386")
			message(STATUS "Using 32 bit miniconda")
			set(CONTINUUM_BITS "x86")
		else()
			message(WARNING "Unable to detect machine bits.  Falling back to downloading 32 bit x86 Miniconda.")
			set(CONTINUUM_BITS "x86")
		endif()
	endif()
	
	set(MINICONDA_INSTALLER_FILENAME "Miniconda${PYTHON_MAJOR_RELEASE}-${MINICONDA_VERSION}-${CONTINUUM_SYSTEM_NAME}-${CONTINUUM_BITS}.${INSTALLER_SUFFIX}")
	
	# location to download the installer to
	set(MINICONDA_INSTALLER ${MINICONDA_DOWNLOAD_DIR}/${MINICONDA_INSTALLER_FILENAME})
	set(INSTALLER_URL "http://repo.continuum.io/miniconda/${MINICONDA_INSTALLER_FILENAME}")
	
	# If we've already downloaded the installer, use it.	
	if(EXISTS "${MINICONDA_INSTALLER}")
		message(STATUS "Using cached Miniconda installer at ${MINICONDA_INSTALLER}")
	else()
		message("Downloading ${INSTALLER_URL} -> ${MINICONDA_INSTALLER}")
			
		# Actually download the file
		download_file_https(${INSTALLER_URL} ${MINICONDA_INSTALLER} TRUE)
	endif()
	message("Installing Miniconda Python.")
	
	# get rid of the install directory, if it exists
	file(REMOVE_RECURSE ${MINICONDA_INSTALL_DIR})
	
	# Unset global conda environment variables in this installation shell to avoid
	# conflicts with an existing anaconda/miniconda installation
	set(CMAKE_ENV_ARGS --unset=CONDA_ENV_PATH --unset=CONDA_DEFAULT_ENV)
	if(TARGET_OSX)
	    # We need the default DYLD_FALLBACK_LIBRARY_PATH to kick in for zlib
	    # linking. Since Amber config.h may set this (and developmental versions did
	    # not include the suggested default in "man dyld") we explicitly unset it
	    # here to ensure that the Miniconda install will work. miniconda.sh already
	    # unsets DYLD_LIBRARY_PATH

	    # We need a program in /sbin, and users may have removed this from their
	    # PATH. So make sure it's there
		list(APPEND CMAKE_ENV_ARGS --unset=DYLD_FALLBACK_LIBRARY_PATH "PATH=$ENV{PATH}:/sbin") 
	endif()
	
	if(NOT TARGET_WINDOWS)
		execute_process(COMMAND chmod +x ${MINICONDA_INSTALLER})
	endif()
	
	# figure out installer arguments
	if(TARGET_WINDOWS)
		# file(TO_NATIVE_PATH) did not work for me here.  Not exactly sure why.
		string(REPLACE "/" "\\" MINICONDA_INSTALL_DIR_BACKSLASHES "${MINICONDA_INSTALL_DIR}")
		set(MINICONDA_INSTALLER_COMMANDLINE ${MINICONDA_INSTALLER} /AddToPath=0 /S "/D=${MINICONDA_INSTALL_DIR_BACKSLASHES}")
	else()
		set(MINICONDA_INSTALLER_COMMANDLINE ${CMAKE_COMMAND} -E env ${CMAKE_ENV_ARGS} ${MINICONDA_INSTALLER} -b -p "${MINICONDA_INSTALL_DIR}")
	endif()
	
	execute_process(COMMAND ${MINICONDA_INSTALLER_COMMANDLINE} 
		RESULT_VARIABLE INSTALLER_RETVAL)
	if(NOT "${INSTALLER_RETVAL}" EQUAL 0)
		message(FATAL_ERROR "Miniconda installer failed!  Please fix what's wrong, or disable Miniconda.")
	endif()
		
	message(STATUS "Updating and installing required and optional packages...")
	
	
	execute_process(COMMAND ${CONDA} update conda -y)	
	execute_process(COMMAND ${MINICONDA_PYTHON} -m pip install pip --upgrade)
	if (MINICONDA_USE_PY3)
		# Revert setuptools as a temporary workaround to recent changes to path mangling behavior
		execute_process(COMMAND ${MINICONDA_PYTHON} -m pip install setuptools==47.3.1)
	endif()
	
	# Prefer non-mkl packages.
	# This is because if Amber is using MKL, when Python programs run they will try to talk to two
	# different MKL libraries at the same time: the MKL Miniconda python was linked with, and the MKL
	# Amber was linked with.
	# So, to fix this, we make sure Miniconda is not using MKL.
	execute_process(COMMAND ${CONDA} install -y nomkl)
	
	execute_process(COMMAND ${CONDA} install -y -q conda-build numpy scipy cython=0.29 ipython notebook pytest 
		RESULT_VARIABLE PACKAGE_INSTALL_RETVAL)
	if(NOT ${PACKAGE_INSTALL_RETVAL} EQUAL 0)
		message(FATAL_ERROR "Installation of packages failed!  Please fix what's wrong, or disable Miniconda.")
	endif()
	
	# Use pip to install matplotlib so we don't have to pull in the entire Qt
	# dependency. And cache inside the Miniconda directory, since we don't want to
	# be writing outside $AMBERHOME unless specifically requested to
	execute_process(COMMAND ${MINICONDA_PYTHON} -m pip --cache-dir=${MINICONDA_INSTALL_DIR}/pkgs install matplotlib RESULT_VARIABLE MATPLOTLIB_RETVAL)
	if(NOT ${MATPLOTLIB_RETVAL} EQUAL 0)
		# try again with conda
		execute_process(COMMAND ${CONDA} install -y -q matplotlib RESULT_VARIABLE MATPLOTLIB_RETVAL)
		if(NOT ${MATPLOTLIB_RETVAL} EQUAL 0)
			message(FATAL_ERROR "Failed to install matplotlib!  Please fix what's wrong, or disable Miniconda.")
		endif()
	endif()
	
	# It's hack-and-patch time!  In a battle royale between inane Distutils and CPython code, and our fair hero UseMiniconda.cmake, 
	# the loser is always your own sanity!
	if(TARGET_WINDOWS AND MINGW)
		# Install a MinGW import library for python (discussed at https://github.com/Theano/Theano/issues/2087)
		execute_process(COMMAND ${CONDA} install -y -q -c anaconda libpython)
		
		# die, preprocessor define that breaks the <cmath> header!
		# see https://github.com/python/cpython/pull/880
		configuretime_file_replace(${MINICONDA_INSTALL_DIR}/include/pyconfig.h ${MINICONDA_INSTALL_DIR}/include/pyconfig.h TO_REPLACE
			"#define hypot _hypot" REPLACEMENT "//#define hypot _hypot")
		
	endif()
		
	# removed downloaded package cache
	execute_process(COMMAND ${CONDA} clean --all --yes)
	
	# figure out the version of miniconda we downloaded
	# this snippet taken from FindPythonInterp
	execute_process(COMMAND "${MINICONDA_PYTHON}" -c "import sys; sys.stdout.write(';'.join([str(x) for x in sys.version_info[:3]]))"
                    OUTPUT_VARIABLE _VERSION
                    RESULT_VARIABLE _PYTHON_VERSION_RESULT)
    if(NOT _PYTHON_VERSION_RESULT)
        string(REPLACE ";" "." PYTHON_VERSION_STRING "${_VERSION}")
        list(GET _VERSION 0 TEMP_CONDA_VERSION_MAJOR)
        list(GET _VERSION 1 TEMP_CONDA_VERSION_MINOR)
        list(GET _VERSION 2 TEMP_CONDA_VERSION_PATCH)
        if(PYTHON_VERSION_PATCH EQUAL 0)
            # it's called "Python 2.7", not "2.7.0"
            string(REGEX REPLACE "\\.0$" "" PYTHON_VERSION_STRING "${PYTHON_VERSION_STRING}")
        endif()
        
        # keep this data around for subsequent CMake runs
        set(MINICONDA_VERSION_MAJOR "${TEMP_CONDA_VERSION_MAJOR}" CACHE INTERNAL "Major version of miniconda" FORCE)
        set(MINICONDA_VERSION_MINOR "${TEMP_CONDA_VERSION_MINOR}" CACHE INTERNAL "Minor version of miniconda" FORCE)
        set(MINICONDA_VERSION_PATCH "${TEMP_CONDA_VERSION_PATCH}" CACHE INTERNAL "Patch version of miniconda" FORCE)
        
    else()
    	message(FATAL_ERROR "Could not determine Miniconda version!")
	endif()
	
	proxy_python_version()
	
	message(STATUS "Miniconda install successful!")
	
	file(WRITE ${MINICONDA_STAMP_FILE} "File created to mark that Miniconda has been downloaded and set up")
	
	
endfunction(download_and_use_miniconda)


# Sets up Miniconda to be installed to the install prefix.
# Must be called after all python programs have had their install rules created
function(install_miniconda)

	# set up miniconda interpreter to be installed
	install(DIRECTORY ${MINICONDA_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_POSTFIX}miniconda COMPONENT Python USE_SOURCE_PERMISSIONS)

	# create "amber.*" symlinks
    if((HOST_OSX OR HOST_LINUX) AND (TARGET_OSX OR TARGET_LINUX))
    	# make sure bin directory exists at this point in the install
    	install(CODE "file(MAKE_DIRECTORY \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_POSTFIX}bin)" COMPONENT Python)
    	
    	installtime_create_symlink(${CMAKE_INSTALL_POSTFIX}miniconda/bin/python ${BINDIR}/amber.python Python)
    	installtime_create_symlink(${CMAKE_INSTALL_POSTFIX}miniconda/bin/conda ${BINDIR}/amber.conda Python)
    	installtime_create_symlink(${CMAKE_INSTALL_POSTFIX}miniconda/bin/ipython ${BINDIR}/amber.ipython Python)
    	installtime_create_symlink(${CMAKE_INSTALL_POSTFIX}miniconda/bin/jupyter ${BINDIR}/amber.jupyter Python)
    	installtime_create_symlink(${CMAKE_INSTALL_POSTFIX}miniconda/bin/pip ${BINDIR}/amber.pip Python)
    endif()
        
	add_subdirectory(${CMAKE_SOURCE_DIR}/cmake/FixCondaShebang)
			
endfunction(install_miniconda)

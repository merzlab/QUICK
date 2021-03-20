# Dummy subdirectory to run BundleOSXDependencies at install time.
# Add this subdirectory at the end of your build file to run the script on your install prefix.
# This file does nothing if TARGET_OSX is FALSE.

if(TARGET_OSX)

	set(EXTRA_RPATH_SEARCH_DIRS "")

	# Miniconda and Anaconda contain libraries that have @rpath install names, so we need to handle those separately.
	if(DOWNLOAD_MINICONDA)
		list(APPEND EXTRA_RPATH_SEARCH_DIRS ${MINICONDA_INSTALL_DIR}/lib)
	endif()

	if(USING_SYSTEM_ANACONDA)
		list(APPEND EXTRA_RPATH_SEARCH_DIRS ${ANACONDA_ROOT}/lib)
	endif()
	
	# create an install rule to invoke BundleOSXDependencies.cmake
	install(CODE "
		message(STATUS \"Finding and bundling dependencies...\")
		execute_process(COMMAND ${CMAKE_COMMAND} 
		\"-DPACKAGE_PREFIX=\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_POSTFIX}\"
		-DCMAKE_SHARED_LIBRARY_SUFFIX=${CMAKE_SHARED_LIBRARY_SUFFIX}
		-DCMAKE_EXECUTABLE_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}
		\"-DEXTRA_RPATH_SEARCH_DIRS=${EXTRA_RPATH_SEARCH_DIRS}\"
		\"-DPREFIX_RELATIVE_PYTHONPATH=${PREFIX_RELATIVE_PYTHONPATH}\"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/BundleOSXDependencies.cmake)")		
endif()
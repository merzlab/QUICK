# Figures out which libraries to bundle based on the platform and the contents of the library tracker.
# Included by Packaging.cmake

# Windows
if(TARGET_WINDOWS)
	
	set(DLLS_TO_BUNDLE "")
	set(LIBS_TO_BUNDLE "")
	
	if(MINGW)
		# MinGW - add runtme libraries to library tracker
	
		get_filename_component(MINGW_BIN_DIR "${CMAKE_C_COMPILER}" DIRECTORY)
		get_filename_component(MINGW_BASE_DIR "${MINGW_BIN_DIR}/.." REALPATH)
		
		#file(GLOB MINGW_LIB_DIR "${MINGW_BASE_DIR}/*-mingw32/lib") # we do a fuzzy search because the architecture folder is named differently on mingw and mingw64
		
		set(MINGW_DLLS_TO_BUNDLE libgfortran libquadmath libgcc_s_ libwinpthread libstdc++)
		
		# do a fuzzy search since they end in numbers
		foreach(DLLNAME ${MINGW_DLLS_TO_BUNDLE})
				file(GLOB DLL_LOCATION_${DLLNAME} "${MINGW_BIN_DIR}/${DLLNAME}*.dll")
				
				list(LENGTH DLL_LOCATION_${DLLNAME} NUM_POSSIBLE_PATHS)
				
				if("${DLL_LOCATION_${DLLNAME}}" STREQUAL "")
					message(WARNING "Could not locate dll file for the MinGW system library ${DLLNAME}.  It will not be automatically bundled.  If you want Amber to work on systems \
without MinGW installed, then you may have to locate this dll and add it to EXTRA_DLLS_TO_BUNDLE.")
			
				elseif(${NUM_POSSIBLE_PATHS} GREATER 1)
					list_to_space_separated(DLL_LOCATION_${DLLNAME}_SPC ${DLL_LOCATION_${DLLNAME}})
				
					message(WARNING "Found multiple possible dll files for the MinGW system library ${DLLNAME}: ${DLL_LOCATION_${DLLNAME}_SPC}  It will not be automatically bundled. 
If you want Amber to work on systems without MinGW installed, then you may have to locate this dll and add it to EXTRA_DLLS_TO_BUNDLE.")
				else()
					# Bundle it!
					list(APPEND DLLS_TO_BUNDLE ${DLL_LOCATION_${DLLNAME}})
				endif()
		endforeach()
		
	else() 
		# Intel or MSVC - use the CMake module to handle runtime libraries
		# It will throw a fit if you are building with VS Community though....

		include(InstallRequiredSystemLibraries)
		
	endif()
		
	#When we link directly to a DLL, it ends up in both USED_LIB_RUNTIME_PATH and USED_LIB_LINKTIME_PATH
	#here we filter these out
	foreach(LIB ${USED_LIB_LINKTIME_PATH})
		list_contains(ALREADY_IN_DLLS_LIST ${LIB} ${USED_LIB_RUNTIME_PATH})
		if(NOT ALREADY_IN_DLLS_LIST)
			list(APPEND LIBS_TO_BUNDLE ${LIB})
		endif()
	endforeach()
	
	# get rid of any "<none>" elements in the runtime path list	
	set(USED_DLLS ${USED_LIB_RUNTIME_PATH})
	
	if(NOT "${USED_DLLS}" STREQUAL "")
		list(REMOVE_ITEM USED_DLLS <none>)
		list(REMOVE_ITEM USED_DLLS <unknown>)
	endif()
	
	list(APPEND DLLS_TO_BUNDLE ${USED_DLLS}) 
	
	set(EXTRA_LIBS_TO_BUNDLE "" CACHE STRING "Additional static and import libraries to bundle with the Windows installer for linking with Amber (e.g. from nab programs).  \
Accepts a semicolon-seperated list.")
	set(EXTRA_DLLS_TO_BUNDLE "" CACHE STRING "Additional DLL files to include with the Windows installer.  Accepts a semicolon-seperated list.")

	list(REMOVE_DUPLICATES DLLS_TO_BUNDLE)
	list(REMOVE_DUPLICATES LIBS_TO_BUNDLE)

	install(FILES ${DLLS_TO_BUNDLE} DESTINATION ${DLLDIR})
	install(FILES ${LIBS_TO_BUNDLE} DESTINATION ${LIBDIR})
	
	if(MINGW)
	
		# winpthreads license (this library requires that its license file be redistributed with the DLL)
		set(WPT_LICENSE_PATH ${MINGW_BASE_DIR}/licenses/winpthreads/COPYING)
		
		if(EXISTS ${WPT_LICENSE_PATH})
			install(FILES ${WPT_LICENSE_PATH} DESTINATION ${CMAKE_INSTALL_POSTFIX}. RENAME COPYING-winpthread.txt)
		else()
			message(WARNING "Couldn't find winpthreads license file to bundle with the installer.  This is required for mass distribution so as to be compliant with winpthread's license. \
Looking for it at the following path: ${WPT_LICENSE_PATH}") 
		endif()
	endif()
endif()

# Mac
# --------------------------------------------------------------------	
	
if(TARGET_OSX)

	# Library bundling is handled automatically by BundleOSXDependencies
	set(EXTRA_LIBS_TO_BUNDLE "" CACHE STRING "Additional libraries to bundle with the OS X distribution.  Accepts a semicolon-seperated list.  Since OS X dynamic library\
dependencies are calculated automatically, you should rarely need to use this.")
		
	install(FILES ${EXTRA_LIBS_TO_BUNDLE} DESTINATION ${LIBDIR})

endif()
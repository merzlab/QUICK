# Across the build system, we need to keep track of which libraries we are using, for two reasons:
# 1. Installers and packages need to bundle them.
# 2. Nab needs to know which libraries to link to things it builds.

# Here, we create macros for recording which external libraries are used.

# on Windows:
# We package DLLs in the bin folder
# We package import and static libraries in the lib folder

# on everything else:
# We package/depend on libraries in the lib folder
 
# these must be cache variables so that they can be set from within a function
set(USED_LIB_RUNTIME_PATH "" CACHE INTERNAL "Paths to shared libraries needed at runtime" FORCE) # Path to a .so, .dylib, or .dll library.  "<none>" if the library is static only.
set(USED_LIB_LINKTIME_PATH "" CACHE INTERNAL "Paths to shared libraries needed at link time" FORCE) # Path to a .a or .lib static library, or an import library
set(USED_LIB_NAME "" CACHE INTERNAL "Names of used shared libraries" FORCE) # contains the library names as supplied to the linker.

# Like using_external_library, but accepts multiple paths.
macro(using_external_libraries)	
	foreach(LIBRARY_PATH ${ARGN})
		using_external_library(${LIBRARY_PATH})
	endforeach()
endmacro(using_external_libraries)


# Notifies the packager that an external library is being used.  It will be bundled along with Amber when packages are created, and
# shown in the packaging report. If a Windows import library as passed as an argument, will automatically find and add the corresponding DLL.
# Does nothing if it has already been called with this library. 
#
# Make sure to call this for every library linked to by Amber!
function(using_external_library LIBPATH)
	
	if("${LIBPATH}" STREQUAL "" OR NOT EXISTS "${LIBPATH}")
		message(FATAL_ERROR "Non-existant library ${LIBPATH} recorded as a used library")
	endif()
	
	if(NOT ("${USED_LIB_RUNTIME_PATH}" MATCHES "${LIBPATH}" OR "${USED_LIB_LINKTIME_PATH}" MATCHES "${LIBPATH}"))
		
		get_lib_type("${LIBPATH}" LIB_TYPE)
		get_linker_name(${LIBPATH} LIBNAME)
	
		# if we are on Windows, we need to find the corresponding .dll library if we got an import library
		# --------------------------------------------------------------------
	
		if("${LIB_TYPE}" STREQUAL IMPORT)
			# accept user override
			if(NOT DEFINED DLL_LOCATION_${LIBNAME})
				#try to find it in the bin subdirectory of the location where the import library is installed.
				get_filename_component(LIB_FOLDER ${LIBPATH} PATH)
				get_filename_component(POSSIBLE_DLL_FOLDER ${LIB_FOLDER}/../bin REALPATH)
			
				
				# DLLs often have a hyphen then a number as their suffix, so we use a fuzzy match, with and without the lib prefix.
				file(GLOB DLL_LOCATION_${LIBNAME} "${POSSIBLE_DLL_FOLDER}/${LIBNAME}*.dll")
			
				if("${DLL_LOCATION_${LIBNAME}}" STREQUAL "")
					file(GLOB DLL_LOCATION_${LIBNAME} "${POSSIBLE_DLL_FOLDER}/lib${LIBNAME}*.dll")
				endif()
				
				if("${DLL_LOCATION_${LIBNAME}}" STREQUAL "")
					# MS-MPI, at least, installs its DLL to System32
					file(GLOB DLL_LOCATION_${LIBNAME} "C:/Windows/System32/${LIBNAME}.dll")
				endif()
			
			
				if("${DLL_LOCATION_${LIBNAME}}" STREQUAL "")
					message(WARNING "Could not locate dll file corresponding to the import library ${LIBPATH}. Please set DLL_LOCATION_${LIBNAME} to the correct DLL file.")
					set(DLL_LOCATION_${LIBNAME} <unknown>)
				endif()
			
				list(LENGTH DLL_LOCATION_${LIBNAME} NUM_POSSIBLE_PATHS)
				if(${NUM_POSSIBLE_PATHS} GREATER 1)
					message(WARNING "Found multiple candidate dll files corresponding to the import library ${LIBPATH}. Please set DLL_LOCATION_${LIBNAME} to the correct DLL file.")
					set(DLL_LOCATION_${LIBNAME} <unknown>)
				endif()
			endif()
		endif()
	
		# save the data to the global lists
		# --------------------------------------------------------------------
	
		if("${LIB_TYPE}" STREQUAL "IMPORT")
			set(USED_LIB_LINKTIME_PATH ${USED_LIB_LINKTIME_PATH} "${LIBPATH}" CACHE INTERNAL "" FORCE)
			set(USED_LIB_RUNTIME_PATH ${USED_LIB_RUNTIME_PATH} "${DLL_LOCATION_${LIBNAME}}" CACHE INTERNAL "" FORCE)
		
			#message("Recorded DLL/implib combo ${LIBNAME}: import library at ${LIBPATH}, DLL at ${DLL_LOCATION_${LIBNAME}}")
		
		elseif("${LIB_TYPE}" STREQUAL "STATIC")
			set(USED_LIB_LINKTIME_PATH ${USED_LIB_LINKTIME_PATH} ${LIBPATH} CACHE INTERNAL "" FORCE)
			set(USED_LIB_RUNTIME_PATH ${USED_LIB_RUNTIME_PATH} "<none>" CACHE INTERNAL "" FORCE)
		
			#message("Recorded static library ${LIBNAME} at ${LIBPATH}")
		elseif("${LIB_TYPE}" STREQUAL "SHARED") 
			set(USED_LIB_LINKTIME_PATH ${USED_LIB_LINKTIME_PATH} ${LIBPATH} CACHE INTERNAL "" FORCE)
			set(USED_LIB_RUNTIME_PATH ${USED_LIB_RUNTIME_PATH} ${LIBPATH} CACHE INTERNAL "" FORCE)
		
			#message("Recorded shared library ${LIBNAME} at ${LIBPATH}")
		else()
			message(FATAL_ERROR "Shouldn't get here!")
		endif()
	
		set(USED_LIB_NAME ${USED_LIB_NAME} ${LIBNAME} CACHE INTERNAL "" FORCE)
	endif()
endfunction(using_external_library)

# Notifies the packager that libraries from the given imported library target are being used.
# Unlike using_external_library(), this accepts target names instead of file paths.
function(using_library_targets) # ARGN: <targets...>

	resolve_cmake_library_list(LIBRARY_PATHS ${ARGN})
	
	# add to library tracker
	foreach(LIBRARY ${LIBRARY_PATHS})

		if("${LIBRARY}" MATCHES "^${LINKER_FLAG_PREFIX}")
			# linker flag -- ignore
		elseif(EXISTS "${LIBRARY}")
			# full path to library
			using_external_library("${LIBRARY}")
		endif()
	endforeach()

endfunction(using_library_targets)


# import functions
# --------------------------------------------------------------------

# Shorthand for setting up a CMake imported target for a library file, with a path and include directories.
# After calling this function, using NAME in a library list will tell CMake to link to the provided file, and add the provided include directories.
# Automatically adds the library to the library tracker.

#usage: import_library(<library name> <library path> [include dir 1] [include dir 2]...)
function(import_library NAME PATH) #3rd arg: INCLUDE_DIRS

	if("${PATH}" STREQUAL "" OR NOT EXISTS "${PATH}")
		message(FATAL_ERROR "Attempt to import library ${NAME} from nonexistant path \"${PATH}\"")
	endif()
	
	#Try to figure out whether it is shared or static.
	get_lib_type(${PATH} LIB_TYPE)

	if("${LIB_TYPE}" STREQUAL "SHARED")
		add_library(${NAME} SHARED IMPORTED GLOBAL)
	else()
		add_library(${NAME} STATIC IMPORTED GLOBAL)
	endif()

	set_property(TARGET ${NAME} PROPERTY IMPORTED_LOCATION ${PATH})
	set_property(TARGET ${NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ARGN})
	
	using_external_library("${PATH}")
	
endfunction(import_library)

# Shorthand for adding one library target which corresponds to multiple linkable things.
# "linkable things" can be any of 6 different types:
#    1. CMake imported targets (as created by import_library() or by another module)
#    2. File paths to libraries
#    3. CMake non-imported targets
#    4. Linker flags
#    5. Names of libraries to find on the linker path
#    6. Generator expressions

# Things of the first 2 types are added to the library tracker.

#usage: import_libraries(<library name> LIBRARIES <library paths...> INCLUDES [<include dirs...>])
function(import_libraries NAME)

	cmake_parse_arguments(IMP_LIBS "" "" "LIBRARIES;INCLUDES" ${ARGN})
	
	if("${IMP_LIBS_LIBRARIES}" STREQUAL "")
		message(FATAL_ERROR "Incorrect usage.  At least one LIBRARY should be provided.")
	endif()
	
	if(NOT "${IMP_LIBS_UNPARSED_ARGUMENTS}" STREQUAL "")
		message(FATAL_ERROR "Incorrect usage.  Extra arguments provided.")
	endif()
	
	# we actually don't use imported libraries at all; we just create an interface target and set its dependencies
	add_library(${NAME} INTERFACE)
		
	set_property(TARGET ${NAME} PROPERTY INTERFACE_LINK_LIBRARIES ${IMP_LIBS_LIBRARIES})
	set_property(TARGET ${NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${IMP_LIBS_INCLUDES})
	
	resolve_cmake_library_list(LIBRARY_PATHS ${IMP_LIBS_LIBRARIES})
	
	# add to library tracker
	foreach(LIBRARY ${LIBRARY_PATHS})
		if("${LIBRARY}" MATCHES "^${LINKER_FLAG_PREFIX}")
			# linker flag -- ignore
			
		elseif(EXISTS "${LIBRARY}")
			# full path to library
			using_external_library("${LIBRARY}")
			
		endif()
	endforeach()
	
endfunction(import_libraries)

# Script containing several advanced functions for handling libraries in CMake.

# linker flag prefix -- if a link library starts with this character, it will be ignored by import_libraries()
# this is needed because FindMKL can return linker flags mixed with its libraries (which is actually the official CMake way of doing things)
if(TARGET_WINDOWS AND NOT MINGW)
	set(LINKER_FLAG_PREFIX "/")  # stupid illogical MSVC command-line format...
else()
	set(LINKER_FLAG_PREFIX "-")
endif()


# CMake would perfer to keep the specific types of libraries as an implementation detail,
# but there are times when you really do need to know, such as DLL imports on Windows.
# Sets OUTPUT_VARAIBLE to "IMPORT", "SHARED", or "STATIC" depending on the library passed
function(get_lib_type LIBRARY OUTPUT_VARIABLE)

	if(NOT EXISTS ${LIBRARY})
		message(FATAL_ERROR "get_lib_type(): library ${LIBRARY} does not exist!")
	endif()

	get_filename_component(LIB_NAME ${LIBRARY} NAME)
	
	set(CACHE_VAR_NAME GET_LIB_TYPE_CACHED_RESULT_${LIB_NAME})
		
	if((DEFINED ${CACHE_VAR_NAME}) AND ("${${CACHE_VAR_NAME}}" STREQUAL "SHARED" OR "${${CACHE_VAR_NAME}}" STREQUAL "STATIC" OR "${${CACHE_VAR_NAME}}" STREQUAL "IMPORT"))
		
		# use cache variable
		set(${OUTPUT_VARIABLE} ${${CACHE_VAR_NAME}} PARENT_SCOPE)
		return()
	endif()
	
	# first, check for import libraries
	if(TARGET_WINDOWS)
		if(MINGW)
			# on MinGW, import libraries have a different file extension, so our job is easy.
			if(${LIB_NAME} MATCHES ".*${CMAKE_IMPORT_LIBRARY_SUFFIX}")
				set(LIB_TYPE IMPORT)
			endif()
		else() # MSVC, Intel, or some other Windows compiler
			
			# we have to work a little harder, and use Dumpbin to check the library type, since import and static libraries have the same extensions
			find_program(DUMPBIN dumpbin)
			
			if(NOT DUMPBIN)
				message(FATAL_ERROR "The Microsoft Dumpbin tool was not found.  It is needed to analyze libraries, so please set the DUMPBIN variable to point to it.")
			endif()
			
			# NOTE: this can take around 2-5 seconds for large libraries -- hence why we cache the result of this function
			execute_process(COMMAND ${DUMPBIN} -headers ${LIBRARY} OUTPUT_VARIABLE DUMPBIN_OUTPUT ERROR_VARIABLE DUMPBIN_ERROUT RESULT_VARIABLE DUMPBIN_RESULT)
			
			# sanity check
			if(NOT ${DUMPBIN_RESULT} EQUAL 0)
				message(FATAL_ERROR "Could not analyze the type of library ${LIBRARY}: dumpbin failed to execute with output ${DUMPBIN_OUTPUT} and error message ${DUMPBIN_ERROUT}")
			endif()
			
			# check for dynamic symbol entries
			# https://stackoverflow.com/questions/488809/tools-for-inspecting-lib-files
			if("${DUMPBIN_OUTPUT}" MATCHES "Symbol name  :")
				# found one!  It's an import library!
				set(LIB_TYPE IMPORT)
			else()
				# by process of elimination, it's a static library
				set(LIB_TYPE STATIC)
			endif()
		endif()
	endif()

	# .tbd files are stubs for dynamic libraries on OS X
	if(TARGET_OSX AND ${LIB_NAME} MATCHES ".*\\.tbd")
		set(${OUTPUT_VARIABLE} SHARED PARENT_SCOPE)
		return()
	endif()
	
	# now we can figure the rest out by suffix matching
	if((NOT TARGET_WINDOWS) OR (TARGET_WINDOWS AND MINGW AND "${LIB_TYPE}" STREQUAL ""))
		if(${LIB_NAME} MATCHES ".*${CMAKE_SHARED_LIBRARY_SUFFIX}")
			set(LIB_TYPE SHARED)
		elseif(${LIB_NAME} MATCHES ".*${CMAKE_STATIC_LIBRARY_SUFFIX}")
			set(LIB_TYPE STATIC)
		else()
			message(FATAL_ERROR "Could not determine whether \"${LIBRARY}\" is a static or shared library, it does not have a known suffix.")
		endif()
	endif()
	
	set(${OUTPUT_VARIABLE} ${LIB_TYPE} PARENT_SCOPE)
	set(${CACHE_VAR_NAME} ${LIB_TYPE} CACHE INTERNAL "Result of get_lib_type() for ${LIB_NAME}" FORCE)
	
endfunction(get_lib_type)

# Takes a list of CMake libraries that you might pass to a target, and returns a list of resolved library paths that correspond to it.
# Some libraries are known by full path, others are known by link-name only (as in, the name you'd pass to "ld -l").  They'll be returned in order in LIB_PATHS.
# Accepts 5 types of thing: full paths to libraries, CMake library targets built by this project, CMake imported targets, CMake interface targets (returns their INTERFACE_LINK_LIBRARIES), and regular linker library names.

# For example, on my system, 
# resolve_cmake_library_list("-Wl,--no-as-needed;mkl::core;mkl::threading;mkl::fortran_interface;/usr/lib/x86_64-linux-gnu/libdl.so;Threads::Threads;arpack;/usr/lib/x86_64-linux-gnu/libm.so") 
# returns
# "-Wl,--no-as-needed;/home/jamie/anaconda3/lib/libmkl_core.so;/home/jamie/anaconda3/lib/libmkl_sequential.so;/home/jamie/anaconda3/lib/libmkl_gf_lp64.so;/usr/lib/x86_64-linux-gnu/libdl.so;-lpthread;arpack;/usr/lib/x86_64-linux-gnu/libm.so"

# usage: resolve_cmake_library_list(<path output var> <libs...>)
function(resolve_cmake_library_list LIB_PATH_OUTPUT)
	# we don't want to add generator expressions to the library tracker...
	string(GENEX_STRIP "${ARGN}" LIBS_TO_RESOLVE)
	
	set(LIB_PATHS "")
	
	foreach(LIBRARY ${LIBS_TO_RESOLVE})
		if("${LIBRARY}" MATCHES "^${LINKER_FLAG_PREFIX}")
			# linker flag
			list(APPEND LIB_PATHS "${LIBRARY}")
			
		elseif(EXISTS "${LIBRARY}")
			# full path to library
			list(APPEND LIB_PATHS "${LIBRARY}")
			
		elseif(TARGET "${LIBRARY}")
			
			get_property(TARGET_LIB_TYPE TARGET ${LIBRARY} PROPERTY TYPE)

			# Target -- check for any library dependencies
			get_property(LIB_DEPENDENCIES TARGET ${LIBRARY} PROPERTY INTERFACE_LINK_LIBRARIES)

			if(NOT "${LIB_DEPENDENCIES}" STREQUAL "")
			
				# avoid infinite recursion if somebody accidentally made an interface library depend on itself
				list(REMOVE_ITEM LIB_DEPENDENCIES "${LIBRARY}")
								
				# now parse those dependencies!
				resolve_cmake_library_list(INTERFACE_DEPS_LIB_PATHS ${LIB_DEPENDENCIES})

				list(APPEND LIB_PATHS ${INTERFACE_DEPS_LIB_PATHS})
			endif()

			# it's an error to check if an interface library has an imported location
			if(NOT "${TARGET_LIB_TYPE}" STREQUAL "INTERFACE_LIBRARY")
				# must be either imported or an actual target
				get_property(LIBRARY_HAS_IMPORTED_LOCATION TARGET ${LIBRARY} PROPERTY IMPORTED_LOCATION SET)
				get_property(LIBRARY_HAS_IMPORTED_CONFIGURATIONS TARGET ${LIBRARY} PROPERTY IMPORTED_CONFIGURATIONS SET)


				if(LIBRARY_HAS_IMPORTED_LOCATION)
					# CMake imported target
					get_property(LIBRARY_IMPORTED_LOCATION TARGET ${LIBRARY} PROPERTY IMPORTED_LOCATION)
					list(APPEND LIB_PATHS "${LIBRARY_IMPORTED_LOCATION}")
				
				elseif(LIBRARY_HAS_IMPORTED_CONFIGURATIONS)

					# multi-configuration imported target

					get_property(LIBRARY_HAS_IMPORTED_LOCATION_OURTYPE TARGET ${LIBRARY} PROPERTY IMPORTED_LOCATION_${CMAKE_BUILD_TYPE} SET)

					if(LIBRARY_HAS_IMPORTED_LOCATION_OURTYPE)
						# if the library has a file for the current build type, use that.

						get_property(LIBRARY_IMPORTED_LOCATION_OURTYPE TARGET ${LIBRARY} PROPERTY IMPORTED_LOCATION_${CMAKE_BUILD_TYPE})
						list(APPEND LIB_PATHS "${LIBRARY_IMPORTED_LOCATION_OURTYPE}")
					else()

						# Fall back to the first configuration the library provides.
						# Not exactly sure what CMake does in this situation (current
						# build type is not a configuration the library provides),
						# but this is a decent guess and will work in the common case
						# where the library provides only one configuration.

						get_property(LIBRARY_IMPORTED_CONFIGURATIONS TARGET ${LIBRARY} PROPERTY IMPORTED_CONFIGURATIONS)
						list(GET LIBRARY_IMPORTED_CONFIGURATIONS 0 FIRST_IMPORTED_CONFIGURATION)
						get_property(LIBRARY_IMPORTED_LOCATION_CONFIG TARGET ${LIBRARY} PROPERTY IMPORTED_LOCATION_${FIRST_IMPORTED_CONFIGURATION})

						if("${LIBRARY_IMPORTED_LOCATION_CONFIG}" STREQUAL "")
							message(WARNING "Imported library ${LIBRARY} does not appear to provide an IMPORTED_LOCATION or an IMPORTED_LOCATION_<CONFIG>.")
						else()
							list(APPEND LIB_PATHS ${LIBRARY_IMPORTED_LOCATION_CONFIG})
						endif()
					endif()

				else()
					# else it's a CMake target that is built by this project
					
					# detect if the library has been renamed
					get_property(LIB_NAME TARGET ${LIBRARY} PROPERTY OUTPUT_NAME)
					
					if("${LIB_NAME}" STREQUAL "")
						# use target name if output name was not set
						set(LIB_NAME ${LIBRARY})
					endif()
					
					list(APPEND LIB_PATHS ${LIB_NAME})
				endif()
			endif()
		else()
		
			# otherwise it's a library name to find on the linker search path (using CMake in "naive mode")
			list(APPEND LIB_PATHS ${LIBRARY})
		endif()

	endforeach()
	
	# debugging code
	if(FALSE)
		message("resolve_cmake_library_list(${LIBS_TO_RESOLVE}):")
		message("    -> ${LIB_PATHS}")
	endif()
	
	set(${LIB_PATH_OUTPUT} ${LIB_PATHS} PARENT_SCOPE)
endfunction(resolve_cmake_library_list)

# gets the "linker name" (the name you'd pass to "ld -l") for a library file.

function(get_linker_name LIBRARY_PATH OUTPUT_VARIABLE)
	
		# get full library name
		get_filename_component(LIBNAME "${LIBRARY_PATH}" NAME)
	
		#remove prefix
		string(REGEX REPLACE "^lib" "" LIBNAME "${LIBNAME}")
	
		#remove numbers after the file extension
		string(REGEX REPLACE "(\\.[0-9])+$" "" LIBNAME "${LIBNAME}")
		
		#remove the file extension
		get_lib_type(${LIBRARY_PATH} LIB_TYPE)
	
		if("${LIB_TYPE}" STREQUAL "IMPORT")
			string(REGEX REPLACE "${CMAKE_IMPORT_LIBRARY_SUFFIX}$" "" LIBNAME ${LIBNAME})
		elseif("${LIB_TYPE}" STREQUAL "STATIC")
			string(REGEX REPLACE "${CMAKE_STATIC_LIBRARY_SUFFIX}\$" "" LIBNAME ${LIBNAME})
		else()
			string(REGEX REPLACE "${CMAKE_SHARED_LIBRARY_SUFFIX}\$" "" LIBNAME ${LIBNAME})
		endif()
		
		set(${OUTPUT_VARIABLE} ${LIBNAME} PARENT_SCOPE)
endfunction(get_linker_name)

# Converts the result of resolve_cmake_library_list() into a "link line" ready to be passed to gcc or ld, and a 
# list of directories to add to the search path.  
macro(resolved_lib_list_to_link_line LINK_LINE_VAR DIRS_VAR) # ARGN: LIB_PATH_LIST

	set(${LINK_LINE_VAR} "")
	set(${DIRS_VAR} "")

	foreach(LIBRARY ${ARGN})
		if("${LIBRARY}" MATCHES "^${LINKER_FLAG_PREFIX}")
		
			# linker flag -- put in library list to preserve ordering
			list(APPEND ${LINK_LINE_VAR} "${LIBRARY}")
		elseif("${LIBRARY}" MATCHES "(.+)\\.framework")
			
			# OS X framework -- translate to "-framework" linker flag
			get_filename_component(FRAMEWORK_BASENAME "${LIBRARY}" NAME_WE)
			string(TOLOWER "${FRAMEWORK_BASENAME}" FRAMEWORK_BASENAME_LCASE)
			list(APPEND ${LINK_LINE_VAR} "-framework" "${FRAMEWORK_BASENAME_LCASE}")
			
		elseif(EXISTS "${LIBRARY}")
		
			if(NOT IS_DIRECTORY "${LIBRARY}")
				# full path to library
				get_filename_component(LIB_DIR ${LIBRARY} DIRECTORY)
				list(APPEND ${DIRS_VAR} ${LIB_DIR})
				
				get_linker_name("${LIBRARY}" LIB_LINK_NAME)
				list(APPEND ${LINK_LINE_VAR} "-l${LIB_LINK_NAME}")
				
			endif()
			
		else()
			
			# target built by this project
			list(APPEND ${LINK_LINE_VAR} "-l${LIBRARY}")
			
		endif()
	endforeach()
	
	list(REMOVE_DUPLICATES ${DIRS_VAR})
	
endmacro(resolved_lib_list_to_link_line)
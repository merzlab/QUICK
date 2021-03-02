#function which reads optimized compile declarations from Fortran source files
#for a file to be compiled with optimizations, the first line should be "! <compile=optimized>"

#uses the current values of NO_OPT_FFLAGS and OPT_FFLAGS

#NOTE: overwrites the current values of the source files' COMPILE_FLAGS properties!

#ignores non-Fortran source files (those without an f in their suffix)

function(apply_optimization_declarations SRC_FILES)
	list(REMOVE_DUPLICATES ARGV)
	
	foreach(SRC_FILE ${ARGV})
		
		#resolve relative paths
		if(NOT IS_ABSOLUTE ${SRC_FILE})
			set(SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}")
		endif()
	
		if(EXISTS ${SRC_FILE})
			get_filename_component(SRC_EXTENSION ${SRC_FILE} EXT)
			string(TOLOWER ${SRC_EXTENSION} SRC_EXTENSION)
			string(REGEX MATCH ".*f.*" FILENAME_CONTAINS_F ${SRC_EXTENSION})
			if(NOT ${FILENAME_CONTAINS_F} STREQUAL "")
				# is a fortran file, we can continue 
				file(READ ${SRC_FILE} FILE_HEADER LIMIT 32) #the declaration is always on the first line, so 32 bytes should be enough, even if there's a unicode BOM
				string(REGEX MATCH "^! *<compile=optimized>.*" HEADER_REGEX_MATCH ${FILE_HEADER})
		
				if(NOT ${HEADER_REGEX_MATCH} STREQUAL "")
					#message(STATUS "Optimized compile: ${SRC_FILE}")
					set_property(SOURCE ${SRC_FILE} PROPERTY COMPILE_FLAGS ${OPT_FFLAGS_SPC})
				else()
					set_property(SOURCE ${SRC_FILE} PROPERTY COMPILE_FLAGS ${NO_OPT_FFLAGS_SPC})
				endif()
			else()
				#just skip this file
				#message(STATUS "${SRC_FILE} is not a Fortran file")
			endif()
		else()
			message("Nonexistant sourcefile ${SRC_FILE}")
		endif()
	endforeach()
endfunction()

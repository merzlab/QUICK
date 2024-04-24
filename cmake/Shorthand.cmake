# File containing some little functions which wrap things which are very verbose in CMake in something a little easier to type
# Since these functions are used all over the place in the build system, this must be included after Utils.cmake but before anything else.

#shorthand for installing a library or libraries, shared or static.
#the normal way is way, WAY too long.
#usage: install_libraries(<lib1> <lib2...> [SUBDIR <subdirectory in lib folder to put libraries into>] [COMPONENT <component>])

function(install_libraries) # LIBRARIES...
	cmake_parse_arguments(
			INSTALL_LIBS
			""
			"SUBDIR;COMPONENT"
			""
			${ARGN})
	
	if("${INSTALL_LIBS_COMPONENT}" STREQUAL "")
		install(TARGETS ${INSTALL_LIBS_UNPARSED_ARGUMENTS} 
			RUNTIME DESTINATION ${DLLDIR} 
			ARCHIVE DESTINATION ${LIBDIR}/${INSTALL_LIBS_SUBDIR} 
			LIBRARY DESTINATION ${LIBDIR}/${INSTALL_LIBS_SUBDIR})
	else()
		install(TARGETS ${INSTALL_LIBS_UNPARSED_ARGUMENTS} COMPONENT ${INSTALL_LIBS_COMPONENT}
			RUNTIME DESTINATION ${DLLDIR} 
			ARCHIVE DESTINATION ${LIBDIR}/${INSTALL_LIBS_SUBDIR} 
			LIBRARY DESTINATION ${LIBDIR}/${INSTALL_LIBS_SUBDIR})
	endif()
endfunction(install_libraries)

#Shorthand for linking multiple targets to one or more libraries.
#usage: targets_link_libraries(fooexe1 fooexe2 LIBRARIES libbar libbaz)
macro(targets_link_libraries)
	#parse the arguents
	cmake_parse_arguments(
			TARGETS_LINK_LIB
			""
			""
			"LIBRARIES"
			${ARGN})

	#note: targets will end up in the "unparsed arguments" list
	if("${TARGETS_LINK_LIB_LIBRARIES}" STREQUAL "")
		message(SEND_ERROR "Incorrect usage: no libraries provided")
	endif()

	if("${TARGETS_LINK_LIB_UNPARSED_ARGUMENTS}" STREQUAL "")
		message(SEND_ERROR "Incorrect usage: no targets provided")
	endif()

	foreach(TARGET ${TARGETS_LINK_LIB_UNPARSED_ARGUMENTS})
		target_link_libraries(${TARGET} ${TARGETS_LINK_LIB_LIBRARIES})
	endforeach()
endmacro(targets_link_libraries)

#Shorthand for having multiple targets include some directories.
#usage: targets_include_directories(fooexe1 fooexe2 DIRECTORIES a b/c)
macro(targets_include_directories)
	#parse the arguents
	cmake_parse_arguments(
			TARGETS_INC_DIR
			""
			""
			"DIRECTORIES"
			${ARGN})

	#note: targets will end up in the "unparsed arguments" list
	if("${TARGETS_INC_DIR_DIRECTORIES}" STREQUAL "")
		message(SEND_ERROR "Incorrect usage: no directories provided")
	endif()

	if("${TARGETS_INC_DIR_UNPARSED_ARGUMENTS}" STREQUAL "")
		message(SEND_ERROR "Incorrect usage: no targets provided")
	endif()

	foreach(TARGET ${TARGETS_INC_DIR_UNPARSED_ARGUMENTS})
		target_include_directories(${TARGET} ${TARGETS_INC_DIR_DIRECTORIES})
	endforeach()
endmacro(targets_include_directories)


# Shorthand for adding an imported executable.
# Sets it up so that using NAME as the program in a custom command will invoke the executable at PATH
macro(import_executable NAME PATH)
	add_executable(${NAME} IMPORTED)
	set_property(TARGET ${NAME} PROPERTY IMPORTED_LOCATION ${PATH})
endmacro(import_executable)

#Add dependencies to a test
#In other words, make TEST not be run until all tests in DEPENDENCIES have
macro(test_depends TEST) #DEPENDENCIES...
	set_property(TEST ${TEST} APPEND PROPERTY DEPENDS ${ARGN})
endmacro(test_depends)

# Shorthand for setting a boolean based on a logical expression
#NOTE: does not work for testing if a string is empty because any empty strings ("") in the arguments get removed completely
macro(test OUTPUT_VAR) #LOGICAL_EXPRESSION...
	if(${ARGN})
		set(${OUTPUT_VAR} TRUE)
	else()
		set(${OUTPUT_VAR} FALSE)
	endif()
endmacro()

# sets OUTPUT_VARIABLE to true if STRING is empty, false otherwise
macro(empty_string OUTPUT_VARIABLE STRING)
	if("${STRING}" STREQUAL "")
		set(${OUTPUT_VARIABLE} TRUE)
	else()
		set(${OUTPUT_VARIABLE} FALSE)
	endif()
endmacro(empty_string)

# sets OUTPUT_VARIABLE to true if STRING is not empty, false otherwise
macro(not_empty_string OUTPUT_VARIABLE STRING)
	if("${STRING}" STREQUAL "")
		set(${OUTPUT_VARIABLE} FALSE)
	else()
		set(${OUTPUT_VARIABLE} TRUE)
	endif()
endmacro(not_empty_string)

#sets OUTPUT_VAR to TRUE if LIST contains ELEMENT
# If only we could use recent CMake versions this wouldn't be needed, sigh
macro(list_contains OUTPUT ELEMENT) #3rd arg: LIST...
	#change macro argument to variable
	set(ARGN_LIST ${ARGN})
	
	list(FIND ARGN_LIST "${ELEMENT}" ELEMENT_INDEX)
	
	if(${ELEMENT_INDEX} EQUAL -1)
		set(${OUTPUT} FALSE)
	else()
		set(${OUTPUT} TRUE)
	endif()

endmacro(list_contains)

# Checks for the presence of all include files provided and sets variables corresponding to their names.
# Variable naming follows the Automake convention
# foo.h -> HAVE_FOO_H
# sys/stat.h -> HAVE_SYS_STAT_H
# _underscore.h -> HAVE__UNDERSCORE_H
# cstdio -> HAVE_CSTDIO

#LANGUAGE should be C or CXX
macro(check_all_includes LANGUAGE)
		
	foreach(INCLUDE ${ARGN})
		# figure out variable name
		string(TOUPPER "HAVE_${INCLUDE}" VAR_NAME)
		string(REPLACE "." "_" VAR_NAME ${VAR_NAME})
		string(REPLACE "/" "_" VAR_NAME ${VAR_NAME})
		
		#message("${INCLUDE} -> ${VAR_NAME}")
		
		if(${LANGUAGE} STREQUAL C)
			check_include_file(${INCLUDE} ${VAR_NAME})
		elseif(${LANGUAGE} STREQUAL CXX)
			check_include_file_cxx(${INCLUDE} ${VAR_NAME})
		else()
			message(FATAL_ERROR "Invalid value for LANGUAGE")
		endif()
	endforeach()
endmacro(check_all_includes)


# Checks for the presence all of the functions provided and sets variables corresponding to their names.
# Set CMAKE_REQUIRED_LIBRARIES to point to libraries that you want this test to link.
# Variable naming follows the Automake convention
# strlen -> HAVE_STRLEN
# _underscore -> HAVE__UNDERSCORE

macro(check_all_functions)
	
	foreach(FUNCTION ${ARGN})
		# figure out variable name
		string(TOUPPER "HAVE_${FUNCTION}" VAR_NAME)
		
		#message("${FUNCTION} -> ${VAR_NAME}")
		
		check_function_exists(${FUNCTION} ${VAR_NAME})
	endforeach()
endmacro(check_all_functions)

# Checks for the presence in system headers of all the types provided and sets variables corresponding to their names and sizes.
# Set CMAKE_EXTRA_INCLUDE_FILES to name extra headers that you want this function to include, and CMAKE_REQUIRED_INCLUDES to point to additional directories to search for those headers in.
# Variable naming follows the Automake convention
# off_t -> HAVE_OFF_T, SIZEOF_OFF_T
# _underscore -> HAVE__UNDERSCORE, SIZEOF__UNDERSCORE
# "long long" -> HAVE_LONG_LONG, SIZEOF_LONG_LONG
# "((struct something*)0)->member" -> HAVE_STRUCT_SOMETHING_MEMBER, SIZEOF_STRUCT_SOMETHING_MEMBER 
# num10 -> HAVE_NUM10, SIZEOF_NUM10 (numbers are only removed if they are adjacent to a closing parenthesis)
macro(check_all_types)
	
	foreach(TYPE ${ARGN})
		# figure out variable name
		string(TOUPPER "${TYPE}" NAME_UCASE)
		string(REPLACE " " "_" NAME_UCASE "${NAME_UCASE}")
		string(REPLACE "->" "_" NAME_UCASE "${NAME_UCASE}")
		string(REGEX REPLACE "[0-9]\\\)" "" NAME_UCASE "${NAME_UCASE}")
		string(REPLACE "(" "" NAME_UCASE "${NAME_UCASE}")
		string(REPLACE ")" "" NAME_UCASE "${NAME_UCASE}")
		string(REPLACE "*" "" NAME_UCASE "${NAME_UCASE}")
		
		#message("${TYPE} -> ${NAME_UCASE}")
		
		check_type_size("${TYPE}" "SIZEOF_${NAME_UCASE}")
		
		set(HAVE_${NAME_UCASE} ${HAVE_SIZEOF_${NAME_UCASE}})
	endforeach()
endmacro(check_all_types)

# Checks for all of the symbol or constant names provided in the given header and sets variables corresponding to their names
# Variable naming follows the Automake convention
# strlen -> HAVE_STRLEN
# _underscore -> HAVE__UNDERSCORE

function(check_all_symbols HEADER)
	
	foreach(SYMBOL ${ARGN})
		# figure out variable name
		string(TOUPPER "HAVE_${SYMBOL}" VAR_NAME)
		
		#message("${FUNCTION} -> ${VAR_NAME}")
		
		# check symbol first
		check_symbol_exists(${SYMBOL} ${HEADER} ${VAR_NAME}_AS_SYMBOL)
		
		# if it was not found as a symbol, then check for it as an enum/constant
		if(NOT ${VAR_NAME}_AS_SYMBOL)
			check_constant_exists(${SYMBOL} ${HEADER} ${VAR_NAME}_AS_CONSTANT)
		endif()
		
		test(${VAR_NAME}_EITHER ${VAR_NAME}_AS_SYMBOL OR ${VAR_NAME}_AS_CONSTANT)
		
		# create cache variable from results
		set(${VAR_NAME} ${${VAR_NAME}_EITHER} CACHE INTERNAL "Whether ${SYMBOL} is available in ${HEADER} as a function, global variable, preprocessor constant, or enum value")
		
	endforeach()
endfunction(check_all_symbols)

# Prints a variable name and its value to the standard output.  Useful for debugging.
function(printvar VARNAME)
	message("${VARNAME}: \"${${VARNAME}}\"")
endfunction(printvar)
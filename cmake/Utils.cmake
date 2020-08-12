include(CMakeParseArguments)

#converts a list into a string with each of its elements seperated by a space
macro(list_to_space_separated OUTPUT_VAR)# 2nd arg: LIST...
	string(REPLACE ";" " " ${OUTPUT_VAR} "${ARGN}")
endmacro(list_to_space_separated)

#causes a symlink between FILE and SYMLINK to be created at install time.
# the paths of FILE and SYMLINK are appended to the install prefix
#only works on UNIX
macro(installtime_create_symlink FILE SYMLINK) # 3rd optional arg: COMPONENT
	#cmake -E create_symlink doesn't work on non-UNIX OS's
	
	if(HOST_WINDOWS) 
		message(FATAL_ERROR "installtime_create_symlink called on a non-UNIX platform")
	endif()
	
	if("${ARGN}" STREQUAL "")
		install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${FILE} \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${SYMLINK})")
	else()
		install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${FILE} \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${SYMLINK})" COMPONENT ${ARGN})
	endif()
	
endmacro(installtime_create_symlink)

#creates a rule to make OUTPUTFILE from the output of running m4 on INPUTFILE
macro(m4_target INPUTFILE OUTPUTFILE) # 3rd arg: M4_OPTIONS
	add_custom_command(
		OUTPUT ${OUTPUTFILE}
		COMMAND m4 ${ARGN}
		ARGS ${INPUTFILE} > ${OUTPUTFILE} VERBATIM)
endmacro(m4_target)

#Checks that the cache variable VARIABLE is set to one of VALID_VALUES and prints an error if is not.
#Also creates a pull-down menu for the variable in the GUI containing these choices
macro(validate_configuration_enum VARIABLE) #2nd argument: VALID_VALUES...
	
	list_contains(VALID ${${VARIABLE}} ${ARGN})
	
	if(NOT VALID)
		list_to_space_separated(VALID_VALUES_STRING ${ARGN})
		
		message(FATAL_ERROR "${${VARIABLE}} is not a valid value for ${VARIABLE} -- must be one of: ${VALID_VALUES_STRING}")
	endif()
	
	  set_property(CACHE ${VARIABLE} PROPERTY STRINGS ${ARGN})
endmacro(validate_configuration_enum)

# Remove the last file extension from a filename.
# foo.bar.s > foo.bar
# This is different from get_filename_component(.. NAME_WE), where foo.bar.s > foo
macro(strip_last_extension OUTPUT_VAR FILENAME)
    #from http://stackoverflow.com/questions/30049180/strip-filename-shortest-extension-by-cmake-get-filename-removing-the-last-ext
    string(REGEX REPLACE "\\.[^.]*$" "" ${OUTPUT_VAR} ${FILENAME})
endmacro()


#several times in this codebase we have a library and an executable named the same thing.
#in that case, we use a "lib" prefix on the target to distinguish the two.
#normally, that would mean we'd get a library file like "liblibsander.so"

# this macro removes the lib prefix on each of the library targets provided so that this doesn't happen
macro(remove_prefix) #LIBRARIES
	set_target_properties(${ARGN} PROPERTIES PREFIX "")
	set_target_properties(${ARGN} PROPERTIES IMPORT_PREFIX "")
endmacro(remove_prefix)

#make the provided object library position independent if shared libraries are turned on
function(make_pic_if_needed OBJECT_LIBRARY)
	set_property(TARGET ${OBJECT_LIBRARY} PROPERTY POSITION_INDEPENDENT_CODE ${SHARED})
endfunction(make_pic_if_needed)

#Append NEW_FLAGS to the COMPILE_FLAGS property of each source file in SOURCE
macro(append_compile_flags NEW_FLAGS) # SOURCE...
	foreach(SOURCE_FILE ${ARGN})
		get_property(CURRENT_COMPILE_FLAGS SOURCE ${SOURCE_FILE} PROPERTY COMPILE_FLAGS)

		set(NEW_COMPILE_FLAGS "${CURRENT_COMPILE_FLAGS} ${NEW_FLAGS}")

		set_property(SOURCE ${SOURCE_FILE} PROPERTY COMPILE_FLAGS ${NEW_COMPILE_FLAGS})
	endforeach()
endmacro(append_compile_flags)

# removes the given directory from cmake's PATH environment variable
function(remove_from_path DIRECTORY)
	
	if(HOST_WINDOWS)
		set(CONVERTED_PATH $ENV{PATH})
	else()
		string(REPLACE ":" ";" CONVERTED_PATH $ENV{PATH})
	endif()
	
	get_filename_component(TO_REMOVE_REALPATH "${DIRECTORY}" REALPATH)
	
	set(NEW_PATH "")
	
	foreach(PATHCOMPONENT ${CONVERTED_PATH})
		get_filename_component(PATHCOMPONENT_REALPATH "${PATHCOMPONENT}" REALPATH)
		
		#message("Comparing \"${TO_REMOVE_REALPATH}\" and \"${PATHCOMPONENT_REALPATH}\"")
		
		# make sure to compare their real paths, so we aren't foiled by any path weirdness in PATH
		if(NOT "${TO_REMOVE_REALPATH}" STREQUAL "${PATHCOMPONENT_REALPATH}")
			list(APPEND NEW_PATH "${PATHCOMPONENT}")
		endif()
	endforeach()
	
	#printvar(ENV{PATH})
	#printvar(NEW_PATH)
	
	# now set the new path
	if(HOST_WINDOWS)
		set($ENV{PATH} "${NEW_PATH}")
	else()
		string(REPLACE ";" ":" ENV{PATH} "${NEW_PATH}")
	endif()
	#printvar(ENV{PATH})
	
endfunction(remove_from_path)
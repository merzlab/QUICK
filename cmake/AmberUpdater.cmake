# CMake script that calls the Amber update script.  BAsed on the user's choice,
# it either checks for updates or installs them.
# Must be included after PythonConfig.cmake

# The updater requires python.  Python builds don't have to be enabled, we just need to have the interpereter
if(NOT HAS_PYTHON)
	return()
endif()

# we can skip updates if this is a developer version
option(CHECK_UPDATES "Use the updater script to check for updates to Amber and AmberTools. This can take a few seconds." ${AMBER_RELEASE})
option(APPLY_UPDATES "On the next run of CMake, apply all available updates for Amber and AmberTools.  This option resets to false after it is used." FALSE)

if(CHECK_UPDATES OR APPLY_UPDATES)
	if(APPLY_UPDATES)
		set(UPDATER_ARG --check-updates)
		colormsg(HIBLUE "Checking for updates...")
	else()
		set(UPDATER_ARG --update)
		colormsg(HIBLUE "Running updater...")
	endif()
	
	# --------------------------------------------------------------------
	# Run script

	execute_process(COMMAND ${CMAKE_COMMAND} -E env AMBERHOME=${CMAKE_SOURCE_DIR} ${PYTHON_EXECUTABLE} update_amber ${UPDATER_ARG}
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		RESULT_VARIABLE UPDATE_COMMAND_RESULT
		OUTPUT_VARIABLE UPDATE_COMMAND_OUTPUT)
		
	# --------------------------------------------------------------------
	# print the output of the updater with a prefix so the people know it's not coming from the build script
	
	string(REPLACE "\n" ";" UPDATE_COMMAND_OUTPUT "${UPDATE_COMMAND_OUTPUT}")
	
	foreach(LINE ${UPDATE_COMMAND_OUTPUT})
		colormsg(">>> ${LINE}")
	endforeach()
	
	# --------------------------------------------------------------------
	# Print conclusion message
	
	if(APPLY_UPDATES)
		if(${UPDATE_COMMAND_RESULT} EQUAL 0)
			set(APPLY_UPDATES FALSE CACHE BOOL "" FORCE)
			colormsg(HIBLUE "Updating succeeded! APPLY_UPDATES has been disabled.")
		else()
			colormsg(HIBLUE "Updating failed!  If you need to supply additional arguments to the updater you can call the ${CMAKE_SOURCE_DIR}/update_amber script directly.")
		endif()
	else()
		if(${UPDATE_COMMAND_RESULT} EQUAL 0)
			colormsg(HIBLUE "Updater done.  If you want to install updates, then set the APPLY_UPDATES variable to true.")
		else()
			colormsg(HIBLUE "Failed to check for updates!  If you need to supply additional arguments to the updater you can call the ${CMAKE_SOURCE_DIR}/update_amber script directly.")
		endif()
	endif()
endif()
	
# Defines a macro for adding tests that use dacdif.
# Since CTest can only run one command per test, this runs a script which invokes dacdif.

#locations of test scripts
set(NDIFF_LOCATION ${CMAKE_SOURCE_DIR}/test/ndiff.awk)
set(DACDIF_LOCATION ${CMAKE_SOURCE_DIR}/test/dacdif)

#Creates a test that runs COMMAND in the source directory, then uses dacdif to compare OUTPUT_FILES with SAVE_FILES.
#If there is more than one file to compare, just supply multiple files to OUTPUT_FILES and list the save files in the same order in SAVE_FILES
# stdout is redirected to a file in CMAKE_CURRENT_BINARY_DIR called ${NAME}-stdout.txt.  Ditto for stderr.
# In OUTPUT_FILES, the tokens @stdout@ and @stderr@ are replaced by the relative paths to these files.


#usage: add_dacdif_test(<name>
#   COMMAND <command> <args>
#   OUTPUT_FILES <file output by test> <output2>...
#   SAVE_FILES <file to compare against> <save2>...
#   [DACDIF_ARGS <dacdiff_args>]
#   [STDIN_FILE <file>])\

# Paths are interpereted relative to either the source or binary dir.
# This setup is designed to make the arguments much shorter by eliminating full paths.
# Relative to CMAKE_CURRENT_BINARY_DIR:
#   OUTPUT_FILES

# Relative to CMAKE_CURRENT_SOURCE_DIR:
#   SAVE_FILES
#   STDIN_FILE


function(add_dacdif_test NAME)
	cmake_parse_arguments(
			DACDIF
			""
			"STDIN_FILE"
			"COMMAND;DACDIF_ARGS;SAVE_FILES;OUTPUT_FILES"
			${ARGN})

	#validate arguments
	if(NOT "${DACDIF_UNPARSED_ARGUMENTS}" STREQUAL "")
		message(SEND_ERROR "Incorrect usage: extra arguments.")
	endif()

	if("${DACDIF_OUTPUT_FILES}" STREQUAL "")
		message(SEND_ERROR "Incorrect usage: missing argument: OUTPUT_FILES")
	endif()

	if("${DACDIF_SAVE_FILES}" STREQUAL "")
		message(SEND_ERROR "Incorrect usage: missing argument: SAVE_FILES")
	endif()

	if("${DACDIF_COMMAND}" STREQUAL "")
		message(SEND_ERROR "Incorrect usage: missing argument: COMMAND")
	endif()

	set(STDOUT_FILENAME ${NAME}-stdout.txt)
	set(STDERR_FILENAME ${NAME}-stderr.txt)

	# fill in the stdout and stderr files
	string(REPLACE "@stdout@" ${STDOUT_FILENAME} DACDIF_OUTPUT_FILES "${DACDIF_OUTPUT_FILES}")
	string(REPLACE "@stderr@" ${STDERR_FILENAME} DACDIF_OUTPUT_FILES "${DACDIF_OUTPUT_FILES}")


	#Now set up the test

	# Variables to define:
	# TEST_COMMAND - command to execute
	# BASH - path to bash shell
	# DACDIFF_LOCATION - path to the dacdiff script
	# DACDIFF_ARGS - list of arguments to pass to dacdiff
	# NDIFF_LOCATION - path to ndiff.awk
	# SAVE_FILE - "correct" output file to compare against
	# OUTPUT_FILE - result file of command
	# SOURCE_DIR - source directory to run COMMAND in and find save files in

	# Build up the Monster Commandline of Doom to pass arguments to the test-time script
	add_test(NAME ${NAME}
			COMMAND ${CMAKE_COMMAND}
				"-DTEST_COMMAND=${DACDIF_COMMAND}"
				"-DDACDIFF_ARGS=${DACDIF_DACDIF_ARGS}"
				-DNDIFF_LOCATION=${NDIFF_LOCATION}
				"-DSAVE_FILES=${DACDIF_SAVE_FILES}"
				"-DOUTPUT_FILES=${DACDIF_OUTPUT_FILES}"
				-DBASH=${BASH}
				-DDACDIF_LOCATION=${DACDIF_LOCATION}
				-DSTDERR_FILE=${CMAKE_CURRENT_BINARY_DIR}/${STDERR_FILENAME}
				-DSTDOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${STDOUT_FILENAME}
				-DSTDIN_FILE=${DACDIF_STDIN_FILE}
				-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
				-P ${CMAKE_SOURCE_DIR}/cmake/DacdifRuntime.cmake
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

endfunction(add_dacdif_test)
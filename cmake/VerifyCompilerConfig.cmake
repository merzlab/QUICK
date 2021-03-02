# This file is run during 2nd init to check the results of AmberCompilerConfig.

# the necessity for this check is discussed here: https://github.com/Amber-MD/CMakeConfigScripts/issues/4
if("${COMPILER}" STREQUAL GNU)
	
	foreach(LANG C CXX)
		if("${CMAKE_${LANG}_COMPILER_ID}" STREQUAL Clang OR "${CMAKE_${LANG}_COMPILER_ID}" STREQUAL AppleClang)
			message(FATAL_ERROR "You told Amber to use the GNU compilers, and it searched for compiler executables named \"gcc\" and \"g++\", but the ${LANG} compiler \
executable that it found (${CMAKE_${LANG}_COMPILER}) is actually Clang masquerading as GCC.  This is common on certain Mac systems.  While Amber could build fine using Clang, \
you requested GCC, so Amber has stopped the build to notify you.  There are three ways to fix this.  

(1) To continue using Clang, you could delete and recreate your build directory, and rerun the build with COMPILER set to \"CLANG\", or to \"AUTO\".
(2) If you installed gcc and gfortran through MacPorts/Homebrew/something else, then move the directory containing the real gcc to the front of your PATH, \
then delete and recreate your build directory and try again.
(3) If you have GCC on your system but don't want to mess with your PATH, then delete and recreate your build directory, then rebuild and set the CMake variables \
CMAKE_C_COMPILER and CMAKE_CXX_COMPILER to point to gcc and g++ and use AUTO for the compiler.
")
		endif()
	endforeach()
endif()

# try to detect and work around issue #92: https://gitlab.ambermd.org/amber/amber/issues/92
# This happens when you have a newer glibc than your Intel or PGI compiler supports, and
# it's missing a needed builtin to be able to compile a specific header.
# Luckily, if we can detect the issue, we can disable that header!

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" OR "${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")

	set(STDLIB_TEST_FILE ${CMAKE_CURRENT_LIST_DIR}/test_include_stdlib.c)
	set(FLOATN_WORKAROUND_DEFINITION -D_BITS_FLOATN_H)

	set(FLOATN_COMPILE_OPTIONS)
	if("${CMAKE_C_COMPILER_ID}" STREQUAL Intel)
		list(APPEND FLOATN_COMPILE_OPTIONS -no-gcc)
	endif()

	if(NOT (STDLIB_WORKS_NO_WORKAROUND OR STDLIB_WORKS_WORKAROUND))

		message(STATUS "Testing if stdlib.h can be included...")

		# first try without the workaround
		try_compile(STDLIB_WORKS_NO_WORKAROUND
			${CMAKE_BINARY_DIR}
			${CMAKE_CURRENT_LIST_DIR}/test_include_stdlib.c
			COMPILE_DEFINITIONS ${FLOATN_COMPILE_OPTIONS}
			C_STANDARD 99)

		if(STDLIB_WORKS_NO_WORKAROUND)
			message(STATUS "Testing if stdlib.h can be included... yes")

			#set cache variable
			set(STDLIB_WORKS_NO_WORKAROUND TRUE CACHE STRING "Whether stdlib.h can be included without any workaround")
			mark_as_advanced(STDLIB_WORKS_NO_WORKAROUND)
		else()
			message(STATUS "Testing if stdlib.h can be included... no")

			message(STATUS "Testing if stdlib.h can be included using ${FLOATN_WORKAROUND_DEFINITION}...")
			try_compile(STDLIB_WORKS_WORKAROUND
				${CMAKE_BINARY_DIR}
				${CMAKE_CURRENT_LIST_DIR}/test_include_stdlib.c
				OUTPUT_VARIABLE STDLIB_COMPILE_OUTPUT
				COMPILE_DEFINITIONS ${FLOATN_COMPILE_OPTIONS} ${FLOATN_WORKAROUND_DEFINITION}
				C_STANDARD 99)

			if(STDLIB_WORKS_WORKAROUND)
				message(STATUS "Testing if stdlib.h can be included using ${FLOATN_WORKAROUND_DEFINITION}... yes")
				message(STATUS "Compiler and glibc version mismatch detected, ${FLOATN_WORKAROUND_DEFINITION} workaround will be used.")

				#set cache variable
				set(STDLIB_WORKS_WORKAROUND TRUE CACHE STRING "Whether stdlib.h can only be included using the ${FLOATN_WORKAROUND_DEFINITION} workaround")
				mark_as_advanced(STDLIB_WORKS_WORKAROUND)

			else()
				message(FATAL_ERROR "Your C compiler could not compile a simple test program using C99 mode to compile stdlib.h.  Build output was: \
${STDLIB_COMPILE_OUTPUT}")
			endif()
		endif()
	endif()

	if(STDLIB_WORKS_WORKAROUND)
		# add workaround (has to go in CFLAGS so compile checks will use it)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLOATN_WORKAROUND_DEFINITION}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLOATN_WORKAROUND_DEFINITION}")
	endif()

endif()

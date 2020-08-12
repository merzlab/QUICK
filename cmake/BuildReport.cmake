# Prints the build report 

function(print_build_report)
	if(COLOR_CMAKE_MESSAGES)
		message("If you can't see the following build report, then you need to turn off COLOR_CMAKE_MESSAGES")
	endif()

	colormsg(HIBLUE "**************************************************************************")
	colormsg("                             " _WHITE_ "Build Report")
	colormsg("                            " _HIMAG_ "Compiler Flags:")
	colormsg(HIMAG "C" HIWHITE "No-Opt:         "  WHITE "${CMAKE_C_FLAGS}" HIBLUE "${NO_OPT_CFLAGS_SPC}")
	colormsg(HIMAG "C" HIWHITE "Optimized:      " WHITE "${CMAKE_C_FLAGS}" HIBLUE "${OPT_CFLAGS_SPC}")
	colormsg("")
	colormsg(GREEN "CXX" HIWHITE "No-Opt:       " WHITE "${CMAKE_CXX_FLAGS}" HIBLUE "${NO_OPT_CXXFLAGS_SPC}")
	colormsg(GREEN "CXX" HIWHITE "Optimized:    " WHITE "${CMAKE_CXX_FLAGS}" HIBLUE "${OPT_CXXFLAGS_SPC}")
	colormsg("")
	colormsg(HIRED "Fortran" HIWHITE "No-Opt:   " WHITE "${CMAKE_Fortran_FLAGS}" HIBLUE "${NO_OPT_FFLAGS_SPC}")
	colormsg(HIRED "Fortran" HIWHITE "Optimized:" WHITE "${CMAKE_Fortran_FLAGS}" HIBLUE "${OPT_FFLAGS_SPC}")
	
	colormsg("")
    colormsg("                         " _HIMAG_ "3rd Party Libraries")
	       colormsg("---building bundled: -----------------------------------------------------")
	
	foreach(TOOL ${NEEDED_3RDPARTY_TOOLS})
	
		if(${${TOOL}_INTERNAL})
			list(FIND 3RDPARTY_TOOLS ${TOOL} TOOL_INDEX)
			list(GET 3RDPARTY_TOOL_USES ${TOOL_INDEX} TOOL_USE)
		
			colormsg(GREEN "${TOOL}" HIWHITE "- ${TOOL_USE}")
		endif()
	endforeach()
	
	       colormsg("---using installed: ------------------------------------------------------")
	
	foreach(TOOL ${NEEDED_3RDPARTY_TOOLS})
		if(${${TOOL}_EXTERNAL})
			list(FIND 3RDPARTY_TOOLS ${TOOL} TOOL_INDEX)
			list(GET 3RDPARTY_TOOL_USES ${TOOL_INDEX} TOOL_USE)
		
			colormsg(YELLOW "${TOOL}" HIWHITE "- ${TOOL_USE}")
		endif()
	endforeach()
	
			colormsg("---disabled: ------------------------------------------------")
	
	foreach(TOOL ${NEEDED_3RDPARTY_TOOLS})
		if(${${TOOL}_DISABLED})
			list(FIND 3RDPARTY_TOOLS ${TOOL} TOOL_INDEX)
			list(GET 3RDPARTY_TOOL_USES ${TOOL_INDEX} TOOL_USE)
		
			colormsg(HIRED "${TOOL}" HIWHITE "- ${TOOL_USE}")
		endif()
	endforeach()
	
	message("")
    colormsg("                              " _HIMAG_ "Features:")
	# we only want to print these if the corresponding build files have been included
	if(DEFINED MPI)
	color_print_bool("MPI:                    " "${MPI}")
	endif()
	
	if(DEFINED OPENMP)
	color_print_bool("OpenMP:                 " "${OPENMP}")
	endif()
	
	if(DEFINED CUDA)
	color_print_bool("CUDA:                   " "${CUDA}")
	endif()

	if(DEFINED NCCL)
	color_print_bool("NCCL:                   " "${NCCL}")
	endif()
	
	color_print_bool("Build Shared Libraries: " "${SHARED}")
	
	if(DEFINED BUILD_GUI)
	color_print_bool("Build GUI Interfaces:   " "${BUILD_GUI}")
	endif()
	
	if(DEFINED BUILD_PYTHON)
	color_print_bool("Build Python Programs:  " "${BUILD_PYTHON}")
	if(DOWNLOAD_MINICONDA)
	colormsg(" -Python Interpreter:   " HIBLUE "Internal Miniconda" YELLOW "(version ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})")
	else()
	colormsg(" -Python Interpreter:   " HIBLUE "${PYTHON_EXECUTABLE}" YELLOW "(version ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})")
	endif()
	endif()
	
	if(DEFINED BUILD_PERL)
	color_print_bool("Build Perl Programs:    " "${BUILD_PERL}")
	endif()
	
	colormsg("Build configuration:    " HIBLUE "${CMAKE_BUILD_TYPE}")
	colormsg("Target Processor:       " YELLOW "${TARGET_ARCH}")
	if(BUILD_DOC)
	colormsg("Build Documentation:    " GREEN "With all, format: ${DOC_FORMAT}")
	elseif(LYX)
	colormsg("Build Documentation:    " YELLOW "As 'make doc' target, format: ${DOC_FORMAT}")
	else()
	colormsg("Build Documentation:    " HIRED "OFF")
	endif()
	if(DEFINED SANDER_VARIANTS_STRING)
	colormsg("Sander Variants:        " HIBLUE "${SANDER_VARIANTS_STRING}")
	endif()

	if(DEFINED MIC_KL AND NOT AMBERTOOLS_ONLY)
	color_print_bool("Knight's Landing Opts: " "${MIC_KL}")
	endif()
	
	if(DEFINED USE_HOST_TOOLS AND USE_HOST_TOOLS)
	colormsg("Using host tools from:  " HIBLUE "${HOST_TOOLS_DIR}")
	endif()
	colormsg("Install location:       " HIBLUE "${CMAKE_INSTALL_PREFIX}${CMAKE_INSTALL_POSTFIX}")
	color_print_bool("Installation of Tests:  " "${INSTALL_TESTS}")
	message("")
	
	#------------------------------------------------------------------------------------------
	colormsg("                             " _HIMAG_ "Compilers:")
	
	# print compiler messages for only the languages that are enabled

	if(NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "")
	colormsg(CYAN "        C:" YELLOW "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}" HIRED "(${CMAKE_C_COMPILER})")
	endif()
	
	if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "")
	colormsg(CYAN "      CXX:" YELLOW "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}" HIRED "(${CMAKE_CXX_COMPILER})")
	endif()
	
	if(NOT "${CMAKE_Fortran_COMPILER_ID}" STREQUAL "")
	colormsg(CYAN "  Fortran:" YELLOW "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}" HIRED "(${CMAKE_Fortran_COMPILER})")
	endif()
		
	# this part is for Amber only
	if(INSIDE_AMBER)
		message("")
		colormsg("                            " _HIMAG_ "Building Tools:")
	
		# NOTE: we can't sort this until after the subdirs have been added because they need to get added in dependency order 
		string(TOLOWER "${AMBER_TOOLS}" AMBER_TOOLS) # list(SORT) sorts capital letters first, so we need to make everything lowercase
		list(SORT AMBER_TOOLS)
		list_to_space_separated(BUILDING_TOOLS ${AMBER_TOOLS})
		
		colormsg("${BUILDING_TOOLS}")
		message("")
		colormsg("                          " _HIMAG_ "NOT Building Tools:")
		foreach(TOOL ${REMOVED_TOOLS})
		
			# get the corresponding reason
			list(FIND REMOVED_TOOLS ${TOOL} TOOL_INDEX)
			list(GET REMOVED_TOOL_REASONS ${TOOL_INDEX} REMOVAL_REASON)
			
			colormsg(HIRED "${TOOL} - ${REMOVAL_REASON}")
		endforeach()
	endif()
	colormsg(HIBLUE "**************************************************************************")
	
	if(DEFINED PRINT_PACKAGING_REPORT AND PRINT_PACKAGING_REPORT)
		print_packaging_report()
	endif()
endfunction(print_build_report)

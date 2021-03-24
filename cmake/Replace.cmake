
include(${CMAKE_CURRENT_LIST_DIR}/Replace-function.cmake)

#creates a custom command for creating OUTPUTFILE by running one or more replaces on INPUTFILE
#the caller is responsible for making something depend on outputfile so that it it generated
#(or using add_custom_target to force generation)

#Takes a list of strings and their replacements.  It replaces the nth item in the TO_REPLACE list with the nth item in the REPLACEMENT list.
#These two lists must be the same length.

#usage: buildtime_file_replace(<input file> <output file> TO_REPLACE <item to replace 1> <item to replace 2> REPLACEMENT <replacement 1> <replacement 2>)
function(buildtime_file_replace INPUTFILE OUTPUTFILE)
    
    cmake_parse_arguments(
        "REPLACE"
        ""
        ""
        "TO_REPLACE;REPLACEMENT"
        ${ARGN})
        
    if(("${REPLACE_TO_REPLACE}" STREQUAL "") OR ("${REPLACE_REPLACEMENT}" STREQUAL ""))
        message(FATAL_ERROR "Missing arguments!")
    endif()
    
    if(NOT ${REPLACE_UNPARSED_ARGUMENTS} STREQUAL "")
        message(SEND_ERROR "Unknown arguments!")
    endif()
        
    #check that the lists are the same length
    list(LENGTH REPLACE_TO_REPLACE TO_REPLACE_LENGTH)
    list(LENGTH REPLACE_REPLACEMENT REPLACEMENT_LENGTH)
    
    if(NOT ${TO_REPLACE_LENGTH} EQUAL ${REPLACEMENT_LENGTH})
    	message(FATAL_ERROR "Incorrect arguments: TO_REPLACE and REPLACEMENT lists are different lengths!")
    endif()
    
	add_custom_command(
		OUTPUT ${OUTPUTFILE}
		COMMAND ${CMAKE_COMMAND} -DTO_REPLACE=${REPLACE_TO_REPLACE} -DREPLACEMENT=${REPLACE_REPLACEMENT} -DINPUTFILE=${INPUTFILE} -DOUTPUTFILE=${OUTPUTFILE} -P ${CMAKE_SOURCE_DIR}/cmake/Replace-runtime.cmake VERBATIM
		COMMENT "Processing ${INPUTFILE} to ${OUTPUTFILE}"
		DEPENDS ${INPUTFILE})

		
endfunction()

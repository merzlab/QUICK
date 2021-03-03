#replace function for configure-time replacements
#usage: file_replace_string(<input file> <output file> TO_REPLACE <item to replace 1> <item to replace 2> REPLACEMENT <replacement 1> <replacement 2>)
# NOTE: replacements cannot be empty strings

include(CMakeParseArguments)

function(configuretime_file_replace INPUTFILE OUTPUTFILE)

	cmake_parse_arguments(
        "CT_REPLACE"
        ""
        ""
        "TO_REPLACE;REPLACEMENT"
        ${ARGN})
        
    if(("${CT_REPLACE_TO_REPLACE}" STREQUAL "") OR ("${CT_REPLACE_REPLACEMENT}" STREQUAL ""))
        message(FATAL_ERROR "Missing arguments!")
    endif()
    
    if(NOT ${CT_REPLACE_UNPARSED_ARGUMENTS} STREQUAL "")
        message(SEND_ERROR "Unknown arguments!")
    endif()
        
    #check that the lists are the same length
    list(LENGTH CT_REPLACE_TO_REPLACE TO_REPLACE_LENGTH)
    list(LENGTH CT_REPLACE_REPLACEMENT REPLACEMENT_LENGTH)
    
    if(NOT ${TO_REPLACE_LENGTH} EQUAL ${REPLACEMENT_LENGTH})
    	message(FATAL_ERROR "Incorrect arguments: TO_REPLACE and REPLACEMENT lists are different lengths!")
    endif()
    
	file(READ ${INPUTFILE} INPUT_TEXT)
	
	foreach(TO_REPLACE_STRING ${CT_REPLACE_TO_REPLACE})
	
	    #get the index of the current to_replace string
	    list(FIND CT_REPLACE_TO_REPLACE ${TO_REPLACE_STRING} REPLACE_INDEX)
	    
	    #look up the corresponding replacement string
	    list(GET CT_REPLACE_REPLACEMENT ${REPLACE_INDEX} REPLACEMENT_STRING)
	
		#NOTE: we must quote INPUT_TEXT so that CMake doesn't parse any semicolons.
	    string(REPLACE ${TO_REPLACE_STRING} ${REPLACEMENT_STRING} INPUT_TEXT "${INPUT_TEXT}")
	endforeach()
		
	file(WRITE ${OUTPUTFILE} "${INPUT_TEXT}")
endfunction(configuretime_file_replace)

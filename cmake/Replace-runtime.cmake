#cmake script run as a custom command to do textual replaces at build time
#eliminates the need for sed

#usage: define the following variables on the command line and run the scripts:
#INPUTFILE - file to input
#OUTPUTFILE - file to output
#TO_REPLACE - list of strings to be replaced
#REPLACEMENT - list of text to replace with
# The first element of TO_REPLACE is matched with the first element of REPLACEMENT, and so on
file(READ ${INPUTFILE} INPUT_TEXT)

foreach(TO_REPLACE_STRING ${TO_REPLACE})

    #get the index of the current to_replace string
    list(FIND TO_REPLACE ${TO_REPLACE_STRING} REPLACE_INDEX)
    
    #look up the corresponding replacement string
    list(GET REPLACEMENT ${REPLACE_INDEX} REPLACEMENT_STRING)

    string(REPLACE ${TO_REPLACE_STRING} ${REPLACEMENT_STRING} INPUT_TEXT "${INPUT_TEXT}")
endforeach()

file(WRITE ${OUTPUTFILE} "${INPUT_TEXT}")
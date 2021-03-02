# macro for uncompressing gzipped files.  Can utilize (p)7zip and gunzip.
# un-gzips SOURCE into DESTINATION
# This does not create a target to force DESTINATION to be built, use add_custom_target to do that.

#figure out which ungz command to use
find_program(GUNZIP_LOCATION gunzip DOC "Path to gunzip program. Set either this one, GZIP_LOCATION, or 7Z_LOCATION to build amber.")
find_program(GZIP_LOCATION gzip DOC "Path to gzip program. Set either this one, GUNZIP_LOCATION, or 7Z_LOCATION to build amber.")

find_program(7Z_LOCATION 7z 7za DOC "Path to the (p)7-zip command line program, either 7z or 7za. Set either this one, GZIP_LOCATION, or GUNZIP_LOCATION to build amber.")

if(GZIP_LOCATION)
    set(UNZIP_COMMAND ${GZIP_LOCATION} -dc)
elseif(GUNZIP_LOCATION)
    set(UNZIP_COMMAND ${GUNZIP_LOCATION} -c)
elseif(7Z_LOCATION)
    set(UNZIP_COMMAND ${7Z_LOCATION} -x -so)
else()
	#cause the searches to be repeated
	unset(GZIP_LOCATION CACHE)
	unset(GUNZIP_LOCATION CACHE)
	unset(7Z_LOCATION CACHE)
	
    if(WIN32)
        message(SEND_ERROR "A gzip unarchiver is required to build AMBER, but was not found.  Please install gzip or 7-zip and set GZIP_LOCATION, GUNZIP_LOCATION, or 7Z_LOCATION to the gzip or 7z executable.")
    else()
        message(SEND_ERROR "A gzip unarchiver is required to build AMBER, but was not found.  Please install gzip, gunzip, or p7zip.")
	endif()
endif()

#message(STATUS "Un-gzip command: ${UNZIP_COMMAND}")

macro(ungzip_file SOURCE DESTINATION)

    add_custom_command( 
        OUTPUT ${DESTINATION}
        COMMAND ${UNZIP_COMMAND} ${SOURCE} > ${DESTINATION}
        DEPENDS ${ZIPPED_DATAFILE}
        VERBATIM)

endmacro(ungzip_file)
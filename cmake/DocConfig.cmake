#Configuration for bulding the documentation
#Must be included after PythonConfig

find_program(LYX NAMES lyx DOC "Path to the lyx document processor executable")
find_program(ELYXER NAMES elyxer.py DOC "Path to the elyxer.py lyx-to-html conversion program")

test(LYX_FOUND NOT ${LYX} MATCHES ".*-NOTFOUND")

message(STATUS "Found lyx (for building documentation) - ${LYX_FOUND}")

option(BUILD_DOC "Build the documentation to Amber as part of the full build.  If the doc can be built but BUILD_DOC is false, you can still build the docuentation by making the 'doc' target." FALSE)

if(BUILD_DOC AND NOT LYX)
	message(FATAL_ERROR "You requested to build the documentation, but lyx was not found.  Please set the LYX variable to point to the lyx executable.")
endif()

set(POSSIBLE_DOC_FORMATS PDF TEXT LATEX POSTSCRIPT HTML)

set(DOC_FORMAT PDF CACHE STRING "Format to build documentation as.  Possible values: PDF TEXT LATEX POSTSCRIPT HTML.  PDF requires pdflatex, HTML requires elyxer.py")

validate_configuration_enum(DOC_FORMAT ${POSSIBLE_DOC_FORMATS})

# HTML export uses elyxer, while all others use lyx directly
if(${DOC_FORMAT} STREQUAL HTML)
	if(NOT PYTHONINTERP_FOUND)
		message(FATAL_ERROR "Python is required to generate HTML documentation.")
	elseif(NOT ELYXER)
		message(FATAL_ERROR "Cannot generate HTML documentation without elyxer.py.  Please set the ELYXER variable to the path to the elyxer.py script.")
	endif()
endif()

# file to configure install directories

# Set up install prefix
# --------------------------------------------------------------------

# make sure the install prefix has a trailing slash
if(NOT "${CMAKE_INSTALL_PREFIX}" MATCHES "/$")
	# shadows the cache variable with a local variable
	set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/")
endif()

# Set up install postfix
# --------------------------------------------------------------------
set(CMAKE_INSTALL_POSTFIX "" CACHE STRING "Path appended to CMAKE_INSTALL_PREFIX when ${PROJECT_NAME} is installed.  In created packages, ${PROJECT_NAME} will be installed to this path inside the package.")

if((NOT ${CMAKE_INSTALL_POSTFIX} STREQUAL "") AND (NOT ${CMAKE_INSTALL_POSTFIX} STREQUAL "/"))
	
	# create sanitized non-cache version without a forward slash and with a trailing slash
	if(NOT "${CMAKE_INSTALL_POSTFIX}" MATCHES "/$")
		set(CMAKE_INSTALL_POSTFIX "${CMAKE_INSTALL_POSTFIX}/")
	endif()
	
	if("${CMAKE_INSTALL_POSTFIX}" MATCHES "^/")
		string(REGEX REPLACE "^/" "" CMAKE_INSTALL_POSTFIX "${CMAKE_INSTALL_POSTFIX}")
	endif()
endif()

# install subdirectory setup
# --------------------------------------------------------------------	

set(BINDIR "${CMAKE_INSTALL_POSTFIX}bin") #binary subdirectory in install location
set(LIBDIR "${CMAKE_INSTALL_POSTFIX}lib") #shared library subdirectory in install location
set(DATADIR "${CMAKE_INSTALL_POSTFIX}dat") #subdirectory for programs' data files
set(DOCDIR "${CMAKE_INSTALL_POSTFIX}doc") # subdirectory for PDF documentation
set(INCDIR "${CMAKE_INSTALL_POSTFIX}include") #subdirecvtory for headers
set(LICENSEDIR "${CMAKE_INSTALL_POSTFIX}licenses") #license file subdir

#directory for runtime (shared) libraries
if(WIN32)
	set(DLLDIR ${BINDIR}) #put on PATH
else()
	set(DLLDIR ${LIBDIR})
endif()
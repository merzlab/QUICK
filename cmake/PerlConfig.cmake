# Configuration file for Perl and the Amber programs that use it.
# Must be included after ExternalLibs.cmake

find_package(Perl)

# Find a compatible make for Perl to use

if(HOST_WINDOWS)
	if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
		# use MinGW GNU Make
		set(PERL_MAKE_NAME mingw32-make)
		set(PERL_MAKE_SELECT_ARG make=gmake)
	else()
		# use Microsoft NMake
		set(PERL_MAKE_NAME nmake)
		set(PERL_MAKE_SELECT_ARG make=nmake)
	endif()

else()
	# POSIX OSs -- always use GNU make
	set(PERL_MAKE_NAME make)
	set(PERL_MAKE_SELECT_ARG "")
endif()


find_program(PERL_MAKE NAMES ${PERL_MAKE_NAME} DOC "Make program to use for building perl programs.")

test(BUILD_PERL_DEFAULT PERL_FOUND)

option(BUILD_PERL "Build the tools which use Perl." ${BUILD_PERL_DEFAULT})

if(PERL_MAKE)
	set(HAVE_PERL_MAKE TRUE)
	message(STATUS "Found perl make: ${PERL_MAKE}")
else()
	set(HAVE_PERL_MAKE FALSE)
endif()

#We have to guess the install directory used by the install script
if(BUILD_PERL)
	
	#relative to install prefix, must NOT begin with a slash
	set(PERL_MODULE_PATH "lib/perl" CACHE STRING "Path relative to install prefix where perl modules are installed.  This path gets added to the startup script") 
	
	message(STATUS "Perl modules well be installed to AMBERHOME/${PERL_MODULE_PATH}")
	
	# create a version that is aware of CMAKE_INSTALL_POSTFIX
	set(PERL_MODULE_INSTALL_DIR "${CMAKE_INSTALL_POSTFIX}${PERL_MODULE_PATH}")
endif()


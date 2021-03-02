# Configuration file for CPack
# accepts the following variables:
# PACKAGE_NAME - name of package, for display to users
# PACKAGE_FILENAME - name of package file
# ICO_ICON - icon of the package, in ICO format.  Can be left undefined.
# ICO_UNINSTALL_ICON - icon for the Windows uninstaller, in ICO format.  Can be left undefined
# ICNS_ICON - icon for the Mac package, in icns format.  Can be left undefined.
# STARTUP_FILE - script or program name to start when double-clicking the package on Windows or Mac.  This is the path to it at __build__ time; Packaging will install it.
# BUNDLE_IDENTIFIER - OS X bundle identifier string
# BUNDLE_SIGNATURE - four character OS X bundle signature string


#see https://cmake.org/Wiki/CMake:CPackPackageGenerators for documentation on these variables
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PACKAGE_NAME})
set(CPACK_PACKAGE_FILE_NAME ${PACKAGE_FILENAME})
	
set(CPACK_PACKAGE_NAME ${CPACK_PACKAGE_FILE_NAME})

set(CPACK_PACKAGE_VENDOR "The Amber Developers")

set(CPACK_PACKAGE_VERSION_MAJOR ${${PROJECT_NAME}_MAJOR_VERSION})
set(CPACK_PACKAGE_VERSION_MINOR ${${PROJECT_NAME}_MINOR_VERSION})
set(CPACK_PACKAGE_VERSION_PATCH ${${PROJECT_NAME}_PATCH_VERSION})
set(CPACK_PACKAGE_VERSION_TWEAK 0)

#set(CPACK_PACKAGE_ICON ${CMAKE_SOURCE_DIR}/amber_logo.bmp)

set(CPACK_PACKAGE_CONTACT "amber@ambermd.org")

set(CPACK_GENERATOR ${PACKAGE_TYPE})

set(CPACK_COMPONENTS_GROUPING IGNORE) # Generate one package per component, not per group

set(CPACK_STRIP_FILES TRUE) # Strip binaries of symbols to save space

include(PackageTypes)
include(LibraryBundling)

# --------------------------------------------------------------------
# figure out package category

if("${PACKAGE_TYPE}" STREQUAL "ARCHIVE")

	option(ARCHIVE_MONOLITHIC "If PACKAGE_TYPE is set to ARCHIVE, this controls whether or not to build all components of Amber into the same package.  If false, they'll be split into a few different archives." TRUE)
	test(CPACK_ARCHIVE_COMPONENT_INSTALL NOT ARCHIVE_MONOLITHIC)
	
elseif(${PACKAGE_TYPE} STREQUAL WINDOWS_INSTALLER)
	
	# NSIS variables
	# --------------------------------------------------------------------
	if(DEFINED ICO_ICON)
		set(CPACK_NSIS_MUI_ICON ${ICO_ICON})
	endif()
	
	if(DEFINED ICO_UNINSTALL_ICON)
		set(CPACK_NSIS_MUI_UNIICON ${ICO_UNINSTALL_ICON})
	endif()
	
	set(CPACK_NSIS_COMPRESSOR "/SOLID lzma" )
	set(CPACK_NSIS_MODIFY_PATH TRUE)
	set(CPACK_NSIS_HELP_LINK "http://ambermd.org/doc12/")
	set(CPACK_NSIS_URL_INFO_ABOUT "http://ambermd.org/")
	set(CPACK_NSIS_CONTACT "${CPACK_PACKAGE_CONTACT}")
	
	set(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")
	
	# Miniconda warning
	# --------------------------------------------------------------------
	if(DOWNLOAD_MINICONDA)
		message(WARNING "You are using Miniconda and are trying to build a NSIS windows installer package.  Miniconda drives the installer over the ~1GB uncompressed size limit and \
this will cause the packaging process to fail.  Please disable DOWNLOAD_MINICONDA  and use a system Python, or, if miniconda is absolutely required, switch to an ARCHIVE package.")
	endif()
	
	# startup file
	# --------------------------------------------------------------------
	
	if(DEFINED STARTUP_FILE)
	
		install(PROGRAMS ${STARTUP_FILE} DESTINATION ${CMAKE_INSTALL_POSTFIX}.)
	
		get_filename_component(STARTUP_FILE_NAME ${STARTUP_FILE} NAME)
		
		#the CPack way of creating a desktop shortcut seems to be bugged and not work.
		set(CPACK_NSIS_CREATE_ICONS_EXTRA "
		    CreateShortCut \\\"$DESKTOP\\\\${CPACK_PACKAGE_FILE_NAME} ${${PROJECT_NAME}_MAJOR_VERSION}.lnk\\\" \\\"$INSTDIR\\\\${STARTUP_FILE_NAME}\\\"
		")
	
		set(CPACK_NSIS_DELETE_ICONS_EXTRA "
		    Delete \\\"$DESKTOP\\\\${CPACK_PACKAGE_FILE_NAME} ${${PROJECT_NAME}_MAJOR_VERSION}.lnk\\\"
		")
		
		set(CPACK_NSIS_MUI_FINISHPAGE_RUN "${STARTUP_FILE_NAME}")
	endif()
		
	
elseif(${PACKAGE_TYPE} STREQUAL BUNDLE)

	#OS X bundle
	# --------------------------------------------------------------------
	set(CPACK_BUNDLE_NAME ${CPACK_PACKAGE_FILE_NAME})
	
	if(DEFINED ICNS_ICON)
		set(CPACK_BUNDLE_ICON ${ICNS_ICON})
	endif()
	
	if(DEFINED STARTUP_FILE)
		set(CPACK_BUNDLE_STARTUP_COMMAND ${STARTUP_FILE})
	endif()
	
   	set(ICON_FILE_NAME "${CPACK_BUNDLE_NAME}")

	#CFBundleGetInfoString
	set(MACOSX_BUNDLE_INFO_STRING "${CPACK_PACKAGE_DESCRIPTION_SUMMARY} Version ${${PROJECT_NAME}_VERSION}, Copyright ${CPACK_PACKAGE_VENDOR}")
	set(MACOSX_BUNDLE_ICON_FILE ${ICON_FILE_NAME})
	set(MACOSX_BUNDLE_GUI_IDENTIFIER "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}")
	#CFBundleLongVersionString
	set(MACOSX_BUNDLE_LONG_VERSION_STRING "${CPACK_PACKAGE_DESCRIPTION_SUMMARY} Version ${${PROJECT_NAME}_VERSION}")
	set(MACOSX_BUNDLE_SHORT_VERSION_STRING ${${PROJECT_NAME}_VERSION})
	
	set(CONFIGURED_PLIST_PATH ${CMAKE_BINARY_DIR}/packaging/Info.plist)
	
	configure_file(${CMAKE_CURRENT_LIST_DIR}/packaging/Info.in.plist ${CONFIGURED_PLIST_PATH} @ONLY)
	set(CPACK_BUNDLE_PLIST ${CONFIGURED_PLIST_PATH})
	
	
else()

	# install to install prefix, rather than root directory as with every other package
	set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
	set(CPACK_PACKAGE_DEFAULT_LOCATION ${CMAKE_INSTALL_PREFIX})
	
	#Debian package
	if(${PACKAGE_TYPE} STREQUAL DEB)
		if(${TARGET_ARCH} STREQUAL "x86_64")
			set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
		else()
			set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${TARGET_ARCH})
		endif()
	
		set(DEB_PACKAGE_DEPENDENCIES "" CACHE STRING "Dependencies string for the debian package.  Must be written by the packager according to how they built amber.
		 Example: \"libarpack2 (>= 3.0.2-3), liblapack3gf (>= 3.3.1-1), libblas3gf (>= 1.2.20110419-2ubuntu1), libreadline6 (>= 6.3-4ubuntu2)\"")
		
		set(CPACK_DEBIAN_PACKAGE_DEPENDS ${DEB_PACKAGE_DEPENDENCIES})
		set(CPACK_DEBIAN_PACKAGE_SECTION "science")		
		set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
		
		# CMake bug #14332 causes our created debian package to become corrupt if we don't have the following line
		# (https://gitlab.kitware.com/cmake/cmake/issues/14332)
		# However, for this to work it needs CMake >= 3.7
		set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")
		
		# autodiscover dependencies
		set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS TRUE)
		
		set(CPACK_DEB_COMPONENT_INSTALL TRUE)
		
	elseif(${PACKAGE_TYPE} STREQUAL RPM)	
		#RPM package
		set(CPACK_RPM_PACKAGE_RELEASE 1)
		set(CPACK_RPM_PACKAGE_GROUP "Applications/Productivity")
		set(RPM_PACKAGE_DEPENDENCIES  "" CACHE STRING "Requirements string for the RPM package.  Must be written by the packager according to how they built amber.
		 Example: \"python >= 2.7.0, lapack, blas\"")
		set(CPACK_RPM_PACKAGE_REQUIRES ${RPM_PACKAGE_DEPENDENCIES})
		
		# tell CPack to autocreate the package names following distro standards
		# these don't work prior to CMake 3.6, so when those versions are used the package will get named according to PACKAGE_NAME
		set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
		
		set(CPACK_RPM_COMPONENT_INSTALL TRUE)
	endif()
		
endif()

# --------------------------------------------------------------------

include(CPack)

# --------------------------------------------------------------------
# packaging report config

option(PRINT_PACKAGING_REPORT "Print a report showing data which will help you package ${PROJECT_NAME}." FALSE)

function(print_packaging_report)
	# calculate external libraries used by amber

	
	colormsg(HIGREEN "**************************************************************************")
	colormsg("                             " _WHITE_ "Packaging Report")
	colormsg(HIGREEN "**************************************************************************")
	colormsg("Package type:              " HIBLUE "${PACKAGE_TYPE}")	

	if(DEFINED AMBER_INSTALL_COMPONENTS)
		list_to_space_separated(AMBER_INSTALL_COMPONENTS_SPC ${AMBER_INSTALL_COMPONENTS})
		colormsg("Packaging these components:" HIBLUE "${AMBER_INSTALL_COMPONENTS_SPC}")
	endif()
	
	if(NOT "${CMAKE_INSTALL_POSTFIX}" STREQUAL "")
		colormsg("Directory inside package:  " HIBLUE "${CMAKE_INSTALL_POSTFIX}")
	endif()
	
	colormsg("")
	colormsg("External libraries used by ${PROJECT_NAME}:")
	colormsg(HIGREEN "--------------------------------------------------------------------------")
	foreach(LIBNAME ${USED_LIB_NAME})
		# find library's index in the list
		list(FIND USED_LIB_NAME ${LIBNAME} LIB_INDEX)
		
		list(GET USED_LIB_LINKTIME_PATH ${LIB_INDEX} LINKTIME_PATH)
		list(GET USED_LIB_RUNTIME_PATH ${LIB_INDEX} RUNTIME_PATH)
		if("${RUNTIME_PATH}" STREQUAL "<none>" OR "${RUNTIME_PATH}" STREQUAL "${LINKTIME_PATH}")
			colormsg("${LIBNAME} -" YELLOW "${LINKTIME_PATH}")
		else()
			colormsg("${LIBNAME} -" YELLOW "${LINKTIME_PATH} (link time), ${RUNTIME_PATH} (runtime)")
		endif()	
	endforeach()
	colormsg(HIGREEN "**************************************************************************")
		
	if(TARGET_WINDOWS)
		colormsg("Since this is a Windows installer, it needs to bundle all of the DLLs that ${PROJECT_NAME} needs with it (besides the Microsoft runtime libraries).  Currently, the following DLLs will be bundled:")
		colormsg("")
		foreach(LIBRARY ${DLLS_TO_BUNDLE})
			colormsg(HIGREEN ${LIBRARY})
		endforeach()
		colormsg("")
		colormsg("Please ensure that all DLLs used by ${PROJECT_NAME} executables are included in this list.  If any more need to be added, list them in the variable EXTRA_DLLS_TO_BUNDLE.")
		
		colormsg("")
		colormsg("Also, in order for the Nab compiler to work, all of the libraries required to link with ${PROJECT_NAME} (besides DLLS that are already bundled and don't have import libraries) need to be included in the installer.")
		
		if("${LIBS_TO_BUNDLE}" STREQUAL "")
			colormsg("Currently, no libraries are bundled.")
		else()
			colormsg("Currently, the following libraries will be bundled:")
			colormsg("")
			foreach(LIBRARY ${LIBS_TO_BUNDLE})
				colormsg(HIGREEN ${LIBRARY})
			endforeach()
			colormsg("")
		endif()
		colormsg("Please ensure that all libraries used by ${PROJECT_NAME} are included in this list.  If any more need to be added, list them in the variable EXTRA_LIBS_TO_BUNDLE.")
	elseif(TARGET_OSX)
		colormsg("This is an OS X application, so CMake should automatically find and bundle dependency libraries.")
		colormsg("It will also automatically fix their install_names and RPATHs, so other programs can link to first- and third-party libraries as long as they use the linker flag \
\"-Wl,-rpath,$AMBERHOME/lib\"")
		colormsg("")
		colormsg("If any additional libraries need to be bundled, please list them in the variable EXTRA_LIBS_TO_BUNDLE.")
		colormsg("Currently, the following libraries will be bundled:")
		foreach(LIBRARY ${EXTRA_LIBS_TO_BUNDLE})
			colormsg(HIGREEN ${LIBRARY})
		endforeach()
		
		
	elseif(TARGET_LINUX)
	
		if("${PACKAGE_TYPE}" STREQUAL "RPM")
			colormsg("This is an RPM package, so dependencies will be automatically calculated for the current distro you are building on.")
		elseif("${PACKAGE_TYPE}" STREQUAL "DEB")
			colormsg("You will need to pass the Debian package dependency string in the CMake variable DEB_PACKAGE_DEPENDENCIES")
			colormsg("Example: " HIBLUE "libarpack2 (>= 3.0.2-3), liblapack3gf (>= 3.3.1-1), libblas3gf (>= 1.2.20110419-2ubuntu1), libreadline6 (>= 6.3-4ubuntu2)")
			colormsg("Its current contents are: \"${DEB_PACKAGE_DEPENDENCIES}\"")
		endif()
		
		colormsg("")
		colormsg("${PROJECT_NAME} will be installed by the package to: " HIBLUE "${CMAKE_INSTALL_PREFIX}")
	endif()
colormsg(HIGREEN "**************************************************************************")
	
endfunction(print_packaging_report)

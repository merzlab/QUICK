# Configuration file for GUI libraries and the programs that use them.
# Must be included after PythonBuildConfig.cmake

#-----------------------------------
# X for XLeap
#-----------------------------------

# temporarily disable searching in Anaconda, sonce it can contain conflicting X11 libraries
if(USING_SYSTEM_ANACONDA)
	set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH_NO_ANACONDA})
	set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH_NO_ANACONDA})
endif()

find_package(X11)

if(USING_SYSTEM_ANACONDA)
	set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH_ANACONDA})
	set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH_ANACONDA})
endif()

if((NOT X11_FOUND) AND (${CMAKE_SYSTEM_NAME} STREQUAL Linux))
	message("Couldn't find the X11 development libraries!")
	message("To search for them try the command: locate libXt")
	message("       On new Fedora install the libXt-devel libXext-devel libX11-devel libICE-devel libSM-devel packages.")
	message("       On old Fedora install the xorg-x11-devel package.")
	message("       On RedHat install the XFree86-devel package.")
	message("       On Ubuntu install the xorg-dev and xserver-xorg packages.")
endif()

# It's likely that when crosscompiling, there will not be GUI libraries for the target, and we actually found the host system's libraries.
# So, we disable BUILD_GUI by default.
set(BUILD_GUI_DEFAULT FALSE)
if(X11_FOUND AND (NOT CROSSCOMPILE) AND EXISTS "${X11_SM_LIB}" AND EXISTS "${X11_Xt_LIB}")
 set(BUILD_GUI_DEFAULT TRUE)
endif()


option(BUILD_GUI "Build graphical interfaces to programs.  Currently affects only LEaP" ${BUILD_GUI_DEFAULT})

if(BUILD_GUI AND (NOT M4 OR NOT X11_FOUND))
	message(FATAL_ERROR "Cannot build Xleap without m4 and the X development libraries.  Either install them, or set BUILD_GUI to FALSE")
endif()
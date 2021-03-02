# set all policies added in 3.8.1 or earlier to NEW behavior
cmake_policy(VERSION 3.9.0)

if(POLICY CMP0073)
	#NEW: don't generate xxx_LIB_DEPENDS cache entries (which screw up the build when switching between bundled and external libraries)
	cmake_policy(SET CMP0073 NEW)
endif()

if(POLICY CMP0074)
	#NEW: Honor <package>_ROOT variables in find_package(CONFIG)
	cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0075)
	#NEW: when compliling header check executables, link them to the contents of CMAKE_REQUIRED_LIBRARIES
	cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0083)
	#NEW: make executables position independent if they have PROPERTY POSITION_INDEPENDENT_CODE set
	cmake_policy(SET CMP0083 NEW)
endif()

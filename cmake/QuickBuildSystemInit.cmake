
if(${COMPILER} STREQUAL GNU)
	#set(CMAKE_Fortran_FLAGS "-O0 -lm  -mtune=native  -ffree-form  -DGNU -cpp -lstdc++")
	set(CMAKE_Fortran_FLAGS "-O0 -lm  -mtune=native  -ffree-form  -DGNU -cpp") #-lstdc++ should be for cuda 
	set(CMAKE_C_FLAGS "")
	set(CMAKE_CXX_FLAGS "")
elseif(${COMPILER} STREQUAL INTEL)
	set(CMAKE_Fortran_FLAGS "-O0 -ip -cpp -diag-disable 8291")
	set(CMAKE_C_FLAGS "")
	set(CMAKE_Cxx_FLAGS "")
endif()




list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
include(QuickCompilerConfig)



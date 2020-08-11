set(CMAKE_CXX_FLAGS "-O0 -g -fPIC")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(octree OBJECT octree.cpp)
set(filelist ${filelist} "${CMAKE_CURRENT_SOURCE_DIR}/octree.cpp" CACHE INTERNAL "")

add_library(grid_packer OBJECT grid_packer.cpp)
set(filelist ${filelist} "${CMAKE_CURRENT_SOURCE_DIR}/grid_packer.cpp" CACHE INTERNAL "")




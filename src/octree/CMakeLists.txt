#set(CMAKE_CXX_FLAGS "-O0 -g -fPIC") #should be target_compile_options(octree PRIVATE ${OPT_CXXFLAGS})

set(filelist ${filelist} "${CMAKE_CURRENT_SOURCE_DIR}/octree.cpp" CACHE INTERNAL "")
set(filelist ${filelist} "${CMAKE_CURRENT_SOURCE_DIR}/grid_packer.cpp" CACHE INTERNAL "")




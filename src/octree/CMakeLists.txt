set(OCTREE_SOURCES
	octree.cpp 
	grid_packer.cpp)

add_library(octree STATIC ${OCTREE_SOURCES})
set_property(TARGET octree PROPERTY POSITION_INDEPENDENT_CODE TRUE)
target_include_directories(octree PRIVATE .)
target_compile_options(octree PRIVATE ${OPT_CXXFLAGS})

if(MPI)
	make_mpi_version(octree octree_mpi LANGUAGES CXX)
	target_compile_definitions(octree_mpi PUBLIC MPIV)
endif()

if(CUDA)
	# This GPU implementation works for both CUDA and CUDA MPI
	copy_target(octree octree_gpu)
	target_compile_definitions(octree_gpu PRIVATE CUDA)
endif()

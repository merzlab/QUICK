add_library(cublas STATIC fortran_thunking.c)
set_target_properties(cublas PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${libfolder})
set_property(TARGET cublas PROPERTY COMPILE_OPTIONS "-O2")
set_property(TARGET cublas PROPERTY COMPILE_DEFINITIONS CUBLAS_USE_THUNKING)
#set_property(TARGET cublas PROPERTY POSITION_INDEPENDENT_CODE TRUE)

#install(TARGETS cublas EXPORT QUICK ARCHIVE DESTINATION lib)


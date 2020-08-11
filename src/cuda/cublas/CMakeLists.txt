add_library(cublas STATIC fortran_thunking.c)
set_target_properties(cublas PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${libfolder})


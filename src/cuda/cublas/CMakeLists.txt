add_library(cublas STATIC fortran_thunking.c)
set_target_properties(cublas PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${libfolder}
										COMPILE_OPTIONS ${OPT_CFLAGS}	
										COMPILE_DEFINITIONS CUBLAS_USE_THUNKING
									 	POSITION_INDEPENDENT_CODE TRUE)


add_library(cusolver STATIC quick_cusolver.c)
set_target_properties(cusolver PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${libfolder})

#install(TARGETS cusolver EXPORT QUICK ARCHIVE DESTINATION lib)


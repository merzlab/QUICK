file(GLOB modlist "*.f90")
foreach(src ${modlist})
	set(filelist ${filelist} ${src} CACHE INTERNAL "")
endforeach()

#add_library(quick_all_module OBJECT ${modlist})




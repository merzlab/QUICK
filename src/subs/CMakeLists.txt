
#set(all_sub_objs "")
#file(GLOB subs "*.c")
#foreach(file ${subs})
#   get_filename_component(cur ${file} NAME_WE)
#   add_library("${cur}" OBJECT ${file})
#   list(APPEND ${all_sub_objs} "${cur}")
#endforeach()


#set(subssrc Angles copyDMat copySym
#	degen denspt diag dipole 
#    EffChar eigvec 
#    ekinetic findBlock fmt getinum 
#    getNum greedy_distrubute hrr iatoi 
#    iatoimp io iwhole 
#    lbfgs Lsolve matComp matMul 
#    order orthog PriCol PriSym 
#    PrtAct PrtDat PrtErr PrtLab 
#    PrtMsg PrtTim PrtWrn pteval 
#    quick_open random rdinum rdnml 
#    rdnum rdword readPDB spdfgh 
#    ssw sum2Mat transpose tridi 
#    upcase vett whatis whole 
#    wrtRestart xnorm zeroMatrix zmake 
#    pt2der sswder denspt_new_imp 
#	pteval_new_imp scaMatMul)

#foreach(src ${subssrc})
#    add_library(${src} OBJECT ${src}.f90)
#	set(filelist ${filelist} "${CMAKE_CURRENT_SOURCE_DIR}/${src}.f90" CACHE INTERNAL "")
#endforeach()

file(GLOB subsrc "*.f90")
foreach(src ${subsrc})
	get_filename_component(filename ${src} NAME_WE)
	#add_library(${filename} OBJECT ${src})
	set(filelist ${filelist} "${src}" CACHE INTERNAL "")
endforeach()



set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -L${CUDA_TOOLKIT_ROOT_DIR}/lib64 -I${CUDA_INCLUDE_DIRS} -DCUDA_SPDF -I$(libxcfolder)")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/libxc)
include_directories(${CMAKE_SOURCE_DIR}/src/libxc/maple2c)
include_directories(${CMAKE_SOURCE_DIR}/src/libxc/maple2c_device)

set(cudaobj "")
add_library(gpucu OBJECT gpu.cu)
add_library(gpu_typecu OBJECT gpu_type.cu)
add_library(gpu_get2ecu OBJECT gpu_get2e.cu)
add_library(gpu_getxccu OBJECT gpu_getxc.cu)
set_target_properties(gpu_getxccu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#Above is essential for '-dc'. Otherwise 
#[ 77%] Building CUDA object src/cuda/CMakeFiles/gpu_getxccu.dir/gpu_getxc.cu.o
#ptxas fatal   : Cannot take address of function '_Z26xc_gga_k_dk_enhance_kernelPKvP15xc_gga_work_x_t' 
add_library(gpu_write_info OBJECT gpu_write_info.f90)

list(APPEND cudaobj "$<TARGET_OBJECTS:gpucu>")
list(APPEND cudaobj "$<TARGET_OBJECTS:gpu_typecu>")
list(APPEND cudaobj "$<TARGET_OBJECTS:gpu_get2ecu>")
list(APPEND cudaobj "$<TARGET_OBJECTS:gpu_getxccu>")
list(APPEND cudaobj "$<TARGET_OBJECTS:gpu_write_info>")


#set(libxcobjlist "gga_c_am05;gga_c_bcgp;gga_c_bmk;gga_c_cs1;\
#gga_c_ft97;gga_c_gapc;gga_c_gaploc;gga_c_hcth_a;\
#gga_c_lm;gga_c_lyp;gga_c_op_b88;gga_c_op_g96;\
#gga_c_op_pbe;gga_c_op_pw91;gga_c_optc;gga_c_op_xalpha;\
#gga_c_p86;gga_c_pbe;gga_c_pbeloc;gga_c_pw91;\
#gga_c_q2d;gga_c_regtpss;gga_c_revtca;gga_c_scan_e0;\
#gga_c_sg4;gga_c_sogga11;gga_c_tca;gga_c_w94;\
#gga_c_wi;gga_c_wl;gga_c_zpbeint;gga_c_zvpbeint;\
#gga_k_dk;gga_k_exp4;gga_k_meyer;gga_k_ol1;\
#gga_k_ol2;gga_k_pearson;gga_k_tflw;gga_k_thakkar;\
#gga_x_2d_b86;gga_x_2d_b86_mgc;gga_x_2d_b88;gga_x_2d_pbe;\
#gga_x_airy;gga_x_ak13;gga_x_am05;gga_x_b86;\
#gga_x_b88;gga_x_bayesian;gga_x_beefvdw;gga_x_bpccac;\
#gga_x_c09x;gga_x_cap;gga_xc_b97;gga_x_chachiyo;\
#gga_xc_th1;gga_xc_th2;gga_xc_th3;gga_x_dk87;\
#gga_x_eg93;gga_x_ft97;gga_x_g96;gga_x_hcth_a;\
#gga_x_herman;gga_x_hjs_b88_v2;gga_x_hjs;gga_x_htbs;\
#gga_x_kt;gga_x_lag;gga_x_lg93;gga_x_lv_rpw86;\
#gga_x_mpbe;gga_x_n12;gga_x_optx;gga_x_pbea;\
#gga_x_pbe;gga_x_pbeint;gga_x_pbepow;gga_x_pbetrans;\
#gga_x_pw86;gga_x_pw91;gga_x_q2d;gga_x_rge2;\
#gga_x_rpbe;gga_x_sg4;gga_x_sogga11;gga_x_ssb_sw;\
#gga_x_vmt84;gga_x_vmt;gga_x_wc;hyb_gga_xc_wb97;\
#lda_c_1d_csc;lda_c_1d_loos;lda_c_2d_amgb;lda_c_2d_prm;\
#lda_c_chachiyo;lda_c_gk72;lda_c_gombas;lda_c_hl;\
#lda_c_lp96;lda_c_ml1;lda_c_pk09;lda_c_pw;\
#lda_c_pz;lda_c_rc04;lda_c_rpa;lda_c_vwn_1;\
#lda_c_vwn_2;lda_c_vwn_3;lda_c_vwn_4;lda_c_vwn;\
#lda_c_vwn_rpa;lda_c_wigner;lda_k_tf;lda_k_zlp;\
#lda_x_2d;lda_xc_1d_ehwlrg;lda_xc_ksdt;lda_xc_teter93;\
#lda_x;lda_xc_zlp;lda_x_rel")

add_library(gpu_all STATIC  ${cudaobj}  ${cudalibxcobj})
set_target_properties(gpu_all PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${libfolder})
set_target_properties(gpu_all PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON) #for -dlink

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cublas)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cusolver)


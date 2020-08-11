set(CMAKE_C_FLAGS "-c -I. -O0 -g -fPIC") # ${CCC}

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/maple2c)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/maple2c_device)


set(localobjlist "")
set(libxcsrc "special_functions;bessel;expint_e1;integrate;util;\
mix_func;func_info;functionals;version;func_reference;\
references;lda;\
lda_x;lda_x_1d;lda_x_2d;lda_c_wigner;lda_c_gombas;\
lda_c_rpa;lda_c_hl;lda_c_vwn;lda_c_vwn_1;lda_c_vwn_2;\
lda_c_vwn_3;lda_c_vwn_4;lda_c_vwn_rpa;lda_c_pz;\
lda_c_pw;lda_c_pk09;lda_c_lp96;\
lda_c_ml1;lda_c_rc04;lda_xc_teter93;lda_c_1d_csc;\
lda_c_1d_loos;lda_c_2d_amgb;lda_c_2d_prm;\
lda_k_tf;lda_xc_zlp;lda_xc_ksdt;lda_c_chachiyo;\
lda_xc_1d_ehwlrg;lda_x_rel;lda_x_erf;lda_k_zlp;\
lda_c_gk72;\
gga;\
gga_x_lg93;gga_x_pbe;gga_x_rpbe;gga_x_pbea;\
gga_x_mpbe;gga_x_herman;gga_x_b86;gga_x_b88;\
gga_x_g96;gga_x_pw86;gga_x_pw91;gga_x_optx;\
gga_x_airy;gga_x_c09x;gga_x_dk87;gga_x_ft97;\
gga_x_wc;gga_x_am05;gga_x_bayesian;gga_x_kt;\
gga_x_htbs;gga_c_pbe;gga_c_lyp;gga_c_p86;gga_c_pw91;\
gga_c_am05;gga_c_lm;gga_c_wl;gga_c_wi;gga_x_lb;\
gga_xc_b97;gga_xc_edf1;gga_xc_1w;gga_c_optc;\
gga_c_tca;gga_x_bpccac;gga_x_sogga11;gga_c_sogga11;\
gga_xc_th1;gga_x_wpbeh;gga_x_hjs;gga_x_ityh;\
gga_x_sfat;gga_x_ev93;gga_x_ak13;gga_x_q2d;gga_c_q2d;\
gga_x_ssb_sw;gga_c_ft97;gga_x_n12;gga_x_lv_rpw86;\
gga_x_2d_b86;gga_x_2d_b86_mgc;\
gga_x_2d_b88;gga_x_2d_pbe;gga_k_tflw;gga_k_pearson;\
gga_k_ol1;gga_k_ol2;gga_k_dk;gga_k_meyer;gga_x_vmt;\
gga_x_hcth_a;gga_x_cap;gga_x_beefvdw;gga_x_sg4;\
gga_x_gg99;gga_x_pbepow;gga_x_lag;gga_k_thakkar;\
gga_xc_oblyp_d;hyb_gga_xc_b3lyp;hyb_gga_xc_o3lyp;\
gga_x_pbeint;gga_x_rge2;gga_x_vmt84;gga_x_hjs_b88_v2;\
gga_c_revtca;gga_c_bcgp;gga_c_zpbeint;gga_c_pbeloc;\
gga_c_sg4;gga_c_regtpss;gga_c_scan_e0;gga_xc_th2;\
gga_xc_th3;gga_c_hcth_a;hyb_gga_xc_wb97;gga_c_bmk;\
gga_c_gapc;gga_c_gaploc;gga_c_zvpbeint;gga_c_w94;\
gga_c_cs1;gga_c_op_xalpha;gga_c_op_b88;gga_c_op_pbe;\
gga_c_op_g96;gga_c_op_pw91;gga_x_pbetrans;\
gga_x_chachiyo;gga_k_exp4;\
hyb_gga_xc_pbeh;hyb_gga_xc_b1wc;\
hyb_gga_xc_hse;hyb_gga_xc_cam_b3lyp;\
hyb_gga_xc_camy_blyp;hyb_gga_xc_camy_b3lyp;\
hyb_gga_xc_lcy_pbe;hyb_gga_xc_lcy_blyp;gga_xc_vv10;\
hyb_gga_xc_edf2;\
mgga;\
mgga_x_lta;mgga_x_tpss;mgga_x_br89;mgga_x_gvt4;\
mgga_x_m06l;mgga_x_tau_hcth;mgga_x_2d_prhg07;\
mgga_x_pkzb;hyb_mgga_x_m05;mgga_x_m08;mgga_x_ms;\
mgga_x_mn12;mgga_x_mk00;mgga_c_vsxc;mgga_c_pkzb;\
mgga_c_bc95;mgga_c_m08;mgga_xc_cc06;mgga_c_cs;\
mgga_xc_otpss_d;mgga_xc_zlp;hyb_mgga_xc_b88b95;\
mgga_x_mbeef;mgga_xc_b97mv;mgga_x_scan;mgga_c_scan;\
hyb_mgga_xc_tpssh;mgga_x_mvs;hyb_mgga_x_mvsh;\
mgga_x_tm;mgga_x_vt84;mgga_x_sa_tpss;mgga_k_pc07;\
mgga_x_mbeefvdw;mgga_x_m11;mgga_x_m11_l;\
hyb_mgga_xc_wb97mv;mgga_c_m05;mgga_c_m06l;\
hyb_mgga_x_dldf;mgga_c_tpss;mgga_c_revtpss;\
mgga_c_tpssloc;mgga_xc_hle17;mgga_c_kcis;mgga_xc_lp90;\
hyb_mgga_xc_kcis;mgga_c_b88;mgga_x_gx;mgga_x_pbe_gx;\
mgga_c_revscan;mgga_x_br89_explicit;mgga_xc_b98")

#file(GLOB libxcsrc "*.c")
#foreach(src ${libxcsrc})
#	get_filename_component(filename ${src} NAME_WE)
#	add_library(${filename} OBJECT ${src})
#	list(APPEND localobjlist "$<TARGET_OBJECTS:${filename}>")
#endforeach()

foreach(src ${libxcsrc})
	add_library(${src} OBJECT ${src}.c)
	list(APPEND localobjlist "$<TARGET_OBJECTS:${src}>")
endforeach()


add_library(libxcf90_la-xc_f OBJECT xc_f.c)	
list(APPEND localobjlist "$<TARGET_OBJECTS:libxcf90_la-xc_f>")

#add_library(libxcf90_la-libxc_funcs OBJECT libxc_funcs.f90)
#list(APPEND localobjlist "$<TARGET_OBJECTS:libxcf90_la-libxc_funcs>")

add_library(libxcf90_la-libxc OBJECT libxc.f90 libxc_funcs.f90)
list(APPEND localobjlist "$<TARGET_OBJECTS:libxcf90_la-libxc>")

if(NOT ${CUDA_ARCH} STREQUAL "")	#for cuda, add gpu.c, test.cu
	add_library(gpu OBJECT gpu.c)
	list(APPEND localobjlist "$<TARGET_OBJECTS:gpu>")
endif()

add_library(xc SHARED ${localobjlist})

set_target_properties(xc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${libfolder})




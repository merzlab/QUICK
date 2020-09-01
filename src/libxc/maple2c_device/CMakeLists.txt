set(maple2c_devicesrc "gga_c_am05.cu;gga_c_bcgp.cu;gga_c_bmk.cu;gga_c_cs1.cu;\
gga_c_ft97.cu;gga_c_gapc.cu;gga_c_gaploc.cu;gga_c_hcth_a.cu;\
gga_c_lm.cu;gga_c_lyp.cu;gga_c_op_b88.cu;gga_c_op_g96.cu;\
gga_c_op_pbe.cu;gga_c_op_pw91.cu;gga_c_optc.cu;gga_c_op_xalpha.cu;\
gga_c_p86.cu;gga_c_pbe.cu;gga_c_pbeloc.cu;gga_c_pw91.cu;\
gga_c_q2d.cu;gga_c_regtpss.cu;gga_c_revtca.cu;gga_c_scan_e0.cu;\
gga_c_sg4.cu;gga_c_sogga11.cu;gga_c_tca.cu;gga_c_w94.cu;\
gga_c_wi.cu;gga_c_wl.cu;gga_c_zpbeint.cu;gga_c_zvpbeint.cu;\
gga_k_dk.cu;gga_k_exp4.cu;gga_k_meyer.cu;gga_k_ol1.cu;\
gga_k_ol2.cu;gga_k_pearson.cu;gga_k_tflw.cu;gga_k_thakkar.cu;\
gga_x_2d_b86.cu;gga_x_2d_b86_mgc.cu;gga_x_2d_b88.cu;gga_x_2d_pbe.cu;\
gga_x_airy.cu;gga_x_ak13.cu;gga_x_am05.cu;gga_x_b86.cu;\
gga_x_b88.cu;gga_x_bayesian.cu;gga_x_beefvdw.cu;gga_x_bpccac.cu;\
gga_x_c09x.cu;gga_x_cap.cu;gga_xc_b97.cu;gga_x_chachiyo.cu;\
gga_xc_th1.cu;gga_xc_th2.cu;gga_xc_th3.cu;gga_x_dk87.cu;\
gga_x_eg93.cu;gga_x_ft97.cu;gga_x_g96.cu;gga_x_hcth_a.cu;\
gga_x_herman.cu;gga_x_hjs_b88_v2.cu;gga_x_hjs.cu;gga_x_htbs.cu;\
gga_x_kt.cu;gga_x_lag.cu;gga_x_lg93.cu;gga_x_lv_rpw86.cu;\
gga_x_mpbe.cu;gga_x_n12.cu;gga_x_optx.cu;gga_x_pbea.cu;\
gga_x_pbe.cu;gga_x_pbeint.cu;gga_x_pbepow.cu;gga_x_pbetrans.cu;\
gga_x_pw86.cu;gga_x_pw91.cu;gga_x_q2d.cu;gga_x_rge2.cu;\
gga_x_rpbe.cu;gga_x_sg4.cu;gga_x_sogga11.cu;gga_x_ssb_sw.cu;\
gga_x_vmt84.cu;gga_x_vmt.cu;gga_x_wc.cu;hyb_gga_xc_wb97.cu;\
lda_c_1d_csc.cu;lda_c_1d_loos.cu;lda_c_2d_amgb.cu;lda_c_2d_prm.cu;\
lda_c_chachiyo.cu;lda_c_gk72.cu;lda_c_gombas.cu;lda_c_hl.cu;\
lda_c_lp96.cu;lda_c_ml1.cu;lda_c_pk09.cu;lda_c_pw.cu;\
lda_c_pz.cu;lda_c_rc04.cu;lda_c_rpa.cu;lda_c_vwn_1.cu;\
lda_c_vwn_2.cu;lda_c_vwn_3.cu;lda_c_vwn_4.cu;lda_c_vwn.cu;\
lda_c_vwn_rpa.cu;lda_c_wigner.cu;lda_k_tf.cu;lda_k_zlp.cu;\
lda_x_2d.cu;lda_xc_1d_ehwlrg.cu;lda_xc_ksdt.cu;lda_xc_teter93.cu;\
lda_x.cu;lda_xc_zlp.cu;lda_x_rel.cu;../../cuda/gpu_getxc.cu")

add_library(xc_cuda STATIC ${maple2c_devicesrc})
target_compile_definitions(xc_cuda PUBLIC CUDA)
target_link_directories(xc_cuda PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
target_include_directories(xc_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS} ${libxcfolder} ${cudafolder})
set_target_properties(xc_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON
										 POSITION_INDEPENDENT_CODE TRUE)


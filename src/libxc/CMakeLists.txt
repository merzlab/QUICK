#set(CMAKE_C_FLAGS "-c -I. -O0 -g -fPIC") # ${CCC}

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(libxcsrc "special_functions.c;bessel.c;expint_e1.c;integrate.c;util.c;\
mix_func.c;func_info.c;functionals.c;version.c;func_reference.c;\
references.c;lda.c;\
lda_x.c;lda_x_1d.c;lda_x_2d.c;lda_c_wigner.c;lda_c_gombas.c;\
lda_c_rpa.c;lda_c_hl.c;lda_c_vwn.c;lda_c_vwn_1.c;lda_c_vwn_2.c;\
lda_c_vwn_3.c;lda_c_vwn_4.c;lda_c_vwn_rpa.c;lda_c_pz.c;\
lda_c_pw.c;lda_c_pk09.c;lda_c_lp96.c;\
lda_c_ml1.c;lda_c_rc04.c;lda_xc_teter93.c;lda_c_1d_csc.c;\
lda_c_1d_loos.c;lda_c_2d_amgb.c;lda_c_2d_prm.c;\
lda_k_tf.c;lda_xc_zlp.c;lda_xc_ksdt.c;lda_c_chachiyo.c;\
lda_xc_1d_ehwlrg.c;lda_x_rel.c;lda_x_erf.c;lda_k_zlp.c;\
lda_c_gk72.c;\
gga.c;\
gga_x_lg93.c;gga_x_pbe.c;gga_x_rpbe.c;gga_x_pbea.c;\
gga_x_mpbe.c;gga_x_herman.c;gga_x_b86.c;gga_x_b88.c;\
gga_x_g96.c;gga_x_pw86.c;gga_x_pw91.c;gga_x_optx.c;\
gga_x_airy.c;gga_x_c09x.c;gga_x_dk87.c;gga_x_ft97.c;\
gga_x_wc.c;gga_x_am05.c;gga_x_bayesian.c;gga_x_kt.c;\
gga_x_htbs.c;gga_c_pbe.c;gga_c_lyp.c;gga_c_p86.c;gga_c_pw91.c;\
gga_c_am05.c;gga_c_lm.c;gga_c_wl.c;gga_c_wi.c;gga_x_lb.c;\
gga_xc_b97.c;gga_xc_edf1.c;gga_xc_1w.c;gga_c_optc.c;\
gga_c_tca.c;gga_x_bpccac.c;gga_x_sogga11.c;gga_c_sogga11.c;\
gga_xc_th1.c;gga_x_wpbeh.c;gga_x_hjs.c;gga_x_ityh.c;\
gga_x_sfat.c;gga_x_ev93.c;gga_x_ak13.c;gga_x_q2d.c;gga_c_q2d.c;\
gga_x_ssb_sw.c;gga_c_ft97.c;gga_x_n12.c;gga_x_lv_rpw86.c;\
gga_x_2d_b86.c;gga_x_2d_b86_mgc.c;\
gga_x_2d_b88.c;gga_x_2d_pbe.c;gga_k_tflw.c;gga_k_pearson.c;\
gga_k_ol1.c;gga_k_ol2.c;gga_k_dk.c;gga_k_meyer.c;gga_x_vmt.c;\
gga_x_hcth_a.c;gga_x_cap.c;gga_x_beefvdw.c;gga_x_sg4.c;\
gga_x_gg99.c;gga_x_pbepow.c;gga_x_lag.c;gga_k_thakkar.c;\
gga_xc_oblyp_d.c;hyb_gga_xc_b3lyp.c;hyb_gga_xc_o3lyp.c;\
gga_x_pbeint.c;gga_x_rge2.c;gga_x_vmt84.c;gga_x_hjs_b88_v2.c;\
gga_c_revtca.c;gga_c_bcgp.c;gga_c_zpbeint.c;gga_c_pbeloc.c;\
gga_c_sg4.c;gga_c_regtpss.c;gga_c_scan_e0.c;gga_xc_th2.c;\
gga_xc_th3.c;gga_c_hcth_a.c;hyb_gga_xc_wb97.c;gga_c_bmk.c;\
gga_c_gapc.c;gga_c_gaploc.c;gga_c_zvpbeint.c;gga_c_w94.c;\
gga_c_cs1.c;gga_c_op_xalpha.c;gga_c_op_b88.c;gga_c_op_pbe.c;\
gga_c_op_g96.c;gga_c_op_pw91.c;gga_x_pbetrans.c;\
gga_x_chachiyo.c;gga_k_exp4.c;\
hyb_gga_xc_pbeh.c;hyb_gga_xc_b1wc.c;\
hyb_gga_xc_hse.c;hyb_gga_xc_cam_b3lyp.c;\
hyb_gga_xc_camy_blyp.c;hyb_gga_xc_camy_b3lyp.c;\
hyb_gga_xc_lcy_pbe.c;hyb_gga_xc_lcy_blyp.c;gga_xc_vv10.c;\
hyb_gga_xc_edf2.c;\
mgga.c;\
mgga_x_lta.c;mgga_x_tpss.c;mgga_x_br89.c;mgga_x_gvt4.c;\
mgga_x_m06l.c;mgga_x_tau_hcth.c;mgga_x_2d_prhg07.c;\
mgga_x_pkzb.c;hyb_mgga_x_m05.c;mgga_x_m08.c;mgga_x_ms.c;\
mgga_x_mn12.c;mgga_x_mk00.c;mgga_c_vsxc.c;mgga_c_pkzb.c;\
mgga_c_bc95.c;mgga_c_m08.c;mgga_xc_cc06.c;mgga_c_cs.c;\
mgga_xc_otpss_d.c;mgga_xc_zlp.c;hyb_mgga_xc_b88b95.c;\
mgga_x_mbeef.c;mgga_xc_b97mv.c;mgga_x_scan.c;mgga_c_scan.c;\
hyb_mgga_xc_tpssh.c;mgga_x_mvs.c;hyb_mgga_x_mvsh.c;\
mgga_x_tm.c;mgga_x_vt84.c;mgga_x_sa_tpss.c;mgga_k_pc07.c;\
mgga_x_mbeefvdw.c;mgga_x_m11.c;mgga_x_m11_l.c;\
hyb_mgga_xc_wb97mv.c;mgga_c_m05.c;mgga_c_m06l.c;\
hyb_mgga_x_dldf.c;mgga_c_tpss.c;mgga_c_revtpss.c;\
mgga_c_tpssloc.c;mgga_xc_hle17.c;mgga_c_kcis.c;mgga_xc_lp90.c;\
hyb_mgga_xc_kcis.c;mgga_c_b88.c;mgga_x_gx.c;mgga_x_pbe_gx.c;\
mgga_c_revscan.c;mgga_x_br89_explicit.c;mgga_xc_b98.c;\
xc_f.c;libxc.f90;libxc_funcs.f90")


add_library(xc STATIC ${libxcsrc})  #set default type to static
set_target_properties(xc PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${libfolder})
config_module_dirs(xc libxc/serial)
install(TARGETS xc EXPORT QUICK ARCHIVE DESTINATION lib/serial)

if(MPI)
    make_mpi_version(xc xc_mpi LANGUAGES C)
    target_compile_definitions(xc_mpi PUBLIC MPIV)
	set_target_properties(xc_mpi PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${libfolder})
	config_module_dirs(xc_mpi libxc/mpi)
	install(TARGETS xc_mpi EXPORT QUICK ARCHIVE DESTINATION lib/mpi)
endif()

if(USE_CUDA)
	copy_target(xc xc_gpu SWAP_SOURCES TO gpu.c)
	target_compile_definitions(xc_gpu PUBLIC CUDA)
	set_target_properties(xc_gpu PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${libfolder})
	config_module_dirs(xc_gpu libxc/gpu)
	install(TARGETS xc_gpu EXPORT QUICK ARCHIVE DESTINATION lib/cuda)
endif()



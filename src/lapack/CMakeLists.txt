#AMBER CMake buildfile for LAPACK
project(LAPACK Fortran)

set(LAPACK_SOURCES
	dsyev.f dsygv.f dlansy.f dsytrd.f dorgtr.f dsteqr.f 
	dsterf.f dlascl.f dlamch.f ilaenv.f dlassq.f dsytd2.f 
	dlatrd.f dorgql.f dorgqr.f dlasrt.f dlaset.f dlanst.f dlaev2.f 
	dlasr.f dlapy2.f dlartg.f dlae2.f dorg2r.f dlarft.f dlarfb.f 
	dorg2l.f dlarfg.f dlarf.f dspev.f dlansp.f dsptrd.f dopgtr.f 
	dppsv.f dspsv.f dgeev.f dsptrs.f dsptrf.f dpptrs.f dpptrf.f 
	dgebak.f dtrevc.f dlacpy.f dhseqr.f dorghr.f dgehrd.f dgebal.f 
	dlange.f dlabad.f dlahqr.f dlarfx.f dlanhs.f dlaln2.f dgehd2.f 
	dlahrd.f dlanv2.f dladiv.f dgerq2.f dorm2r.f dlarnv.f dgeqr2.f 
	dlaruv.f dspevd.f dstedc.f dopmtr.f dlaed0.f dlaed1.f dlaed7.f 
	dlaeda.f dlaed8.f dlaed9.f dlamrg.f dlaed2.f dlaed3.f 
	dpotrs.f dpotrf.f dsygst.f dpotf2.f dsygs2.f dposv.f 
	dlaebz.f dlagtf.f dlagts.f dormqr.f dormql.f dorm2l.f 
	dlaed4.f dlaed5.f dlaed6.f ieeeck.f 
	dsyevx.f dsyevd.f dstebz.f dstein.f dormtr.f 
	zheev.f lsame.f zlanhe.f zlascl.f zhetrd.f zungtr.f zsteqr.f 
	zlassq.f zlatrd.f zhetd2.f zungql.f zungqr.f zlasr.f zlaset.f 
	zlacgv.f zlarfg.f zung2l.f zlarft.f zlarfb.f zung2r.f dlapy3.f 
	zladiv.f zlarf.f 
	dgesvd.f dgebrd.f dgebd2.f dlabrd.f dorgbr.f dbdsqr.f 
	dgeqrf.f dormbr.f dgelqf.f dorglq.f dlas2.f dlasq1.f 
	dlasv2.f dgelq2.f dorgl2.f dlasq2.f dormlq.f dlasq3.f 
	dlasq4.f dlasq5.f dorml2.f dlasq6.f 
	dgesv.f dgetrs.f dlaswp.f dgetrf.f dgerfs.f dgetf2.f 
	dsysv.f dsytrs.f dsytrf.f dlasyf.f dsytf2.f dgelss.f 
	dgelsd.f dlalsd.f dlasdq.f dlasda.f dlalsa.f dlasd6.f dlals0.f 
	dlasdt.f dlasd7.f dlasd8.f dlasd4.f dlasd5.f dlacn2.f 
	dgelsy.f dlaic1.f dtzrzf.f dormrz.f dlarzt.f dlarzb.f 
	dlatrz.f dormr3.f dgeqp3.f dlarz.f dlaqps.f dlaqp2.f dspevx.f 
	dsyevr.f dstegr.f dlarrv.f dlarre.f dlarrb.f dlar1v.f dlarrf.f 
	dggev.f dhgeqz.f dgghrd.f dtgevc.f dggbak.f dggbal.f dlag2.f 
	disnan.f dlaisnan.f dlarfp.f dstemr.f dlamch.f
	dlarra.f dlarrc.f dlarrd.f dlarrj.f dlarrk.f dlarrr.f dlaneg.f 
	dlaqr0.f dlaqr1.f dlaqr2.f  dlaqr3.f dlaqr4.f dlaqr5.f dlahr2.f 
	dormhr.f dtrexc.f dlaexc.f dlasy2.f 
	iparmq.f iladlc.f iladlr.f 
	dlatrs.f drscl.f dlantr.f dgecon.f dgeequ.f dlaqge.f dgesvx.f 
	chla_transtype.f ilatrans.f ilaprec.f 
	dlascl2.f dgeequb.f ilazlr.f ilazlc.f dtrsen.f dtrsyl.f)

#source files from the above list that should be built without optimizations
set(UNOPTIMIZED_SOURCES dlamch.f)

add_library(lapack ${LAPACK_SOURCES})
target_link_libraries(lapack blas)
install_libraries(lapack EXPORT QUICK)

# Prevent name conflict with system BLAS
set_property(TARGET lapack PROPERTY OUTPUT_NAME lapack-quick)

set_property(SOURCE ${LAPACK_SOURCES} PROPERTY COMPILE_FLAGS ${OPT_FFLAGS_SPC})
set_property(SOURCE ${UNOPTIMIZED_SOURCES} PROPERTY COMPILE_FLAGS ${NO_OPT_FFLAGS_SPC})

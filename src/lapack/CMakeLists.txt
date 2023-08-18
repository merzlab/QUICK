#AMBER CMake buildfile for LAPACK
project(LAPACK Fortran)

set(LAPACK_SOURCES
	dasum.f    dhgeqz.f    dlaqge.f   dlasdt.f  dppsv.f    dtrtri.f    sorg2l.f
	daxpy.f    dhseqr.f    dlaqp2.f   dlaset.f  dpptrf.f   dtzrzf.f    sorg2r.f
	dbdsdc.f   disnan.f    dlaqps.f   dlasq1.f  dpptrs.f   dummy.f     sorgql.f
	dbdsqr.f   dlabad.f    dlaqr0.f   dlasq2.f  drot.f     idamax.f    sorgqr.f
	dcopy.f    dlabrd.f    dlaqr1.f   dlasq3.f  drscl.f    ieeeck.f    sorgtr.f
	ddot.f     dlacn2.f    dlaqr2.f   dlasq4.f  dscal.f    iladlc.f    sorm2l.f
	dgbsv.f    dlacpy.f    dlaqr3.f   dlasq5.f  dspevd.f   iladlr.f    sorm2r.f
	dgbtf2.f   dladiv.f    dlaqr4.f   dlasq6.f  dspev.f    ilaenv.f    sormql.f
	dgbtrf.f   dlae2.f     dlaqr5.f   dlasr.f   dspevx.f   ilaprec.f   sormqr.f
	dgbtrs.f   dlaebz.f    dlaqtr.f   dlasrt.f  dspsv.f    ilaslc.f    sormtr.f
	dgebak.f   dlaed0.f    dlar1v.f   dlassq.f  dsptrd.f   ilaslr.f    sscal.f
	dgebal.f   dlaed1.f    dlarfb.f   dlasv2.f  dsptrf.f   ilatrans.f  sstebz.f
	dgebd2.f   dlaed2.f    dlarf.f    dlaswp.f  dsptrs.f   ilazlc.f    sstein.f
	dgebrd.f   dlaed3.f    dlarfg.f   dlasy2.f  dstebz.f   ilazlr.f    ssteqr.f
	dgecon.f   dlaed4.f    dlarfp.f   dlasyf.f  dstedc.f   iparmq.f    ssterf.f
	dgeequb.f  dlaed5.f    dlarft.f   dlatrd.f  dstegr.f   isamax.f    sswap.f
	dgeequ.f   dlaed6.f    dlarfx.f   dlatrs.f  dstein.f   lsame.f     ssyevx.f
	dgeev.f    dlaed7.f    dlarnv.f   dlatrz.f  dstemr.f   saxpy.f     ssymv.f
	dgeevx.f   dlaed8.f    dlarra.f   dlauu2.f  dsteqr.f   scopy.f     ssyr2.f
	dgehd2.f   dlaed9.f    dlarrb.f   dlauum.f  dsterf.f   sdot.f      ssyr2k.f
	dgehrd.f   dlaeda.f    dlarrc.f   dopgtr.f  dswap.f    sgemm.f     ssytd2.f
	dgelq2.f   dlaev2.f    dlarrd.f   dopmtr.f  dsyevd.f   sgemv.f     ssytrd.f
	dgelqf.f   dlaexc.f    dlarre.f   dorg2l.f  dsyev.f    sger.f      strmm.f
	dgelsd.f   dlag2.f     dlarrf.f   dorg2r.f  dsyevr.f   sisnan.f    strmv.f
	dgelss.f   dlagtf.f    dlarrj.f   dorgbr.f  dsyevx.f   slacpy.f    xerbla.f
	dgelsy.f   dlagts.f    dlarrk.f   dorghr.f  dsygs2.f   slae2.f     zheev.f
	dgemm.f    dlahqr.f    dlarrr.f   dorgl2.f  dsygst.f   slaebz.f    zhetd2.f
	dgemv.f    dlahr2.f    dlarrv.f   dorglq.f  dsygv.f    slaev2.f    zhetrd.f
	dgeqp3.f   dlahrd.f    dlartg.f   dorgql.f  dsyrk.f    slagtf.f    zlacgv.f
	dgeqr2.f   dlaic1.f    dlaruv.f   dorgqr.f  dsysv.f    slagts.f    zladiv.f
	dgeqrf.f   dlaisnan.f  dlarzb.f   dorgtr.f  dsytd2.f   slaisnan.f  zlanhe.f
	dger.f     dlaln2.f    dlarz.f    dorm2l.f  dsytf2.f   slamch.f    zlarfb.f
	dgerfs.f   dlals0.f    dlarzt.f   dorm2r.f  dsytrd.f   slanst.f    zlarf.f
	dgerq2.f   dlalsa.f    dlas2.f    dormbr.f  dsytrf.f   slansy.f    zlarfg.f
	dgesdd.f   dlalsd.f    dlascl2.f  dormhr.f  dsytrs.f   slapy2.f    zlarft.f
	dgesvd.f   dlamch.f    dlascl.f   dorml2.f  dtbsv.f    slarfb.f    zlascl.f
	dgesv.f    dlamrg.f    dlasd0.f   dormlq.f  dtgevc.f   slarf.f     zlaset.f
	dgesvx.f   dlaneg.f    dlasd1.f   dormql.f  dtrevc3.f  slarfg.f    zlasr.f
	dgetf2.f   dlange.f    dlasd2.f   dormqr.f  dtrevc.f   slarft.f    zlassq.f
	dgetrf.f   dlanhs.f    dlasd3.f   dormr3.f  dtrexc.f   slarnv.f    zlatrd.f
	dgetri.f   dlansp.f    dlasd4.f   dormrz.f  dtrmm.f    slaruv.f    zsteqr.f
	dgetrs.f   dlanst.f    dlasd5.f   dormtr.f  dtrmv.f    slascl.f    zung2l.f
	dggbak.f   dlansy.f    dlasd6.f   dposv.f   dtrsen.f   slaset.f    zung2r.f
	dggbal.f   dlantr.f    dlasd7.f   dpotf2.f  dtrsm.f    slasr.f     zungql.f
	dggev.f    dlanv2.f    dlasd8.f   dpotrf.f  dtrsna.f   slasrt.f    zungqr.f
	dgghrd.f   dlapy2.f    dlasda.f   dpotri.f  dtrsyl.f   slassq.f    zungtr.f
	dgtsv.f    dlapy3.f    dlasdq.f   dpotrs.f  dtrti2.f   slatrd.f
	chla_transtype.f       droundup_lwork.f     slartg.f90 )

#source files from the above list that should be built without optimizations
set(UNOPTIMIZED_SOURCES dlamch.f)

add_library(lapack ${LAPACK_SOURCES})
target_link_libraries(lapack blas)
install_libraries(lapack EXPORT QUICK)

# Prevent name conflict with system BLAS
set_property(TARGET lapack PROPERTY OUTPUT_NAME lapack-quick)

set_property(SOURCE ${LAPACK_SOURCES} PROPERTY COMPILE_FLAGS ${OPT_FFLAGS_SPC})
set_property(SOURCE ${UNOPTIMIZED_SOURCES} PROPERTY COMPILE_FLAGS ${NO_OPT_FFLAGS_SPC})

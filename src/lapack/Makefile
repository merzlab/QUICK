#######################################################################
#
#  Edward Anderson, University of Tennessee
#  March 26, 1990
#  Susan Ostrouchov,  September 30, 1994
#  Julie Langou, March 2007
#
#######################################################################

MAKEIN = ./make.in
include $(MAKEIN)

#  !---------------------------------------------------------------------!
#  ! Variable declarations                                               !
#  !---------------------------------------------------------------------!

LAPACKOBJ = $(lapackobjfolder)/dbdsqr.o $(lapackobjfolder)/dgebak.o \
	$(lapackobjfolder)/dgebal.o $(lapackobjfolder)/dgebd2.o $(lapackobjfolder)/dgebrd.o \
	$(lapackobjfolder)/dgecon.o $(lapackobjfolder)/dgeequb.o $(lapackobjfolder)/dgeequ.o \
	$(lapackobjfolder)/dgeev.o $(lapackobjfolder)/dgehd2.o $(lapackobjfolder)/dgehrd.o \
	$(lapackobjfolder)/dgelq2.o $(lapackobjfolder)/dgelqf.o $(lapackobjfolder)/dgelsd.o \
	$(lapackobjfolder)/dgelss.o $(lapackobjfolder)/dgelsy.o $(lapackobjfolder)/dgemm.o \
	$(lapackobjfolder)/dgemv.o $(lapackobjfolder)/dgeqp3.o $(lapackobjfolder)/dgeqr2.o \
	$(lapackobjfolder)/dgeqrf.o $(lapackobjfolder)/dgerfs.o $(lapackobjfolder)/dgerq2.o \
	$(lapackobjfolder)/dgesvd.o $(lapackobjfolder)/dgesv.o $(lapackobjfolder)/dgesvx.o \
	$(lapackobjfolder)/dgetf2.o $(lapackobjfolder)/dgetrf.o $(lapackobjfolder)/dgetri.o \
	$(lapackobjfolder)/dgetrs.o $(lapackobjfolder)/dggbak.o $(lapackobjfolder)/dggbal.o \
	$(lapackobjfolder)/dggev.o $(lapackobjfolder)/dgghrd.o $(lapackobjfolder)/dhgeqz.o \
	$(lapackobjfolder)/dhseqr.o $(lapackobjfolder)/disnan.o $(lapackobjfolder)/dlabad.o \
	$(lapackobjfolder)/dlabrd.o $(lapackobjfolder)/dlacn2.o $(lapackobjfolder)/dlacpy.o \
	$(lapackobjfolder)/dladiv.o $(lapackobjfolder)/dlae2.o $(lapackobjfolder)/dlaebz.o \
	$(lapackobjfolder)/dlaed0.o $(lapackobjfolder)/dlaed1.o $(lapackobjfolder)/dlaed2.o \
	$(lapackobjfolder)/dlaed3.o $(lapackobjfolder)/dlaed4.o $(lapackobjfolder)/dlaed5.o \
	$(lapackobjfolder)/dlaed6.o $(lapackobjfolder)/dlaed7.o $(lapackobjfolder)/dlaed8.o \
	$(lapackobjfolder)/dlaed9.o $(lapackobjfolder)/dlaeda.o $(lapackobjfolder)/dlaev2.o \
	$(lapackobjfolder)/dlaexc.o $(lapackobjfolder)/dlag2.o $(lapackobjfolder)/dlagtf.o \
	$(lapackobjfolder)/dlagts.o $(lapackobjfolder)/dlahqr.o $(lapackobjfolder)/dlahr2.o \
	$(lapackobjfolder)/dlahrd.o $(lapackobjfolder)/dlaic1.o $(lapackobjfolder)/dlaisnan.o \
	$(lapackobjfolder)/dlaln2.o $(lapackobjfolder)/dlals0.o $(lapackobjfolder)/dlalsa.o \
	$(lapackobjfolder)/dlalsd.o $(lapackobjfolder)/dlamch.o $(lapackobjfolder)/dlamrg.o \
	$(lapackobjfolder)/dlaneg.o $(lapackobjfolder)/dlange.o $(lapackobjfolder)/dlanhs.o \
	$(lapackobjfolder)/dlansp.o $(lapackobjfolder)/dlanst.o $(lapackobjfolder)/dlansy.o \
	$(lapackobjfolder)/dlantr.o $(lapackobjfolder)/dlanv2.o $(lapackobjfolder)/dlapy2.o \
	$(lapackobjfolder)/dlapy3.o $(lapackobjfolder)/dlaqge.o $(lapackobjfolder)/dlaqp2.o \
	$(lapackobjfolder)/dlaqps.o $(lapackobjfolder)/dlaqr0.o $(lapackobjfolder)/dlaqr1.o \
	$(lapackobjfolder)/dlaqr2.o $(lapackobjfolder)/dlaqr3.o $(lapackobjfolder)/dlaqr4.o \
	$(lapackobjfolder)/dlaqr5.o $(lapackobjfolder)/dlar1v.o $(lapackobjfolder)/dlarfb.o \
	$(lapackobjfolder)/dlarf.o $(lapackobjfolder)/dlarfg.o $(lapackobjfolder)/dlarfp.o \
	$(lapackobjfolder)/dlarft.o $(lapackobjfolder)/dlarfx.o $(lapackobjfolder)/dlarnv.o \
	$(lapackobjfolder)/dlarra.o $(lapackobjfolder)/dlarrb.o $(lapackobjfolder)/dlarrc.o \
	$(lapackobjfolder)/dlarrd.o $(lapackobjfolder)/dlarre.o $(lapackobjfolder)/dlarrf.o \
	$(lapackobjfolder)/dlarrj.o $(lapackobjfolder)/dlarrk.o $(lapackobjfolder)/dlarrr.o \
	$(lapackobjfolder)/dlarrv.o $(lapackobjfolder)/dlartg.o $(lapackobjfolder)/dlaruv.o \
	$(lapackobjfolder)/dlarzb.o $(lapackobjfolder)/dlarz.o $(lapackobjfolder)/dlarzt.o \
	$(lapackobjfolder)/dlas2.o $(lapackobjfolder)/dlascl2.o $(lapackobjfolder)/dlascl.o \
	$(lapackobjfolder)/dlasd4.o $(lapackobjfolder)/dlasd5.o $(lapackobjfolder)/dlasd6.o \
	$(lapackobjfolder)/dlasd7.o $(lapackobjfolder)/dlasd8.o $(lapackobjfolder)/dlasda.o \
	$(lapackobjfolder)/dlasdq.o $(lapackobjfolder)/dlasdt.o $(lapackobjfolder)/dlaset.o \
	$(lapackobjfolder)/dlasq1.o $(lapackobjfolder)/dlasq2.o $(lapackobjfolder)/dlasq3.o \
	$(lapackobjfolder)/dlasq4.o $(lapackobjfolder)/dlasq5.o $(lapackobjfolder)/dlasq6.o \
	$(lapackobjfolder)/dlasr.o $(lapackobjfolder)/dlasrt.o $(lapackobjfolder)/dlassq.o \
	$(lapackobjfolder)/dlasv2.o $(lapackobjfolder)/dlaswp.o $(lapackobjfolder)/dlasy2.o \
	$(lapackobjfolder)/dlasyf.o $(lapackobjfolder)/dlatrd.o $(lapackobjfolder)/dlatrs.o \
	$(lapackobjfolder)/dlatrz.o $(lapackobjfolder)/dopgtr.o $(lapackobjfolder)/dopmtr.o \
	$(lapackobjfolder)/dorg2l.o $(lapackobjfolder)/dorg2r.o $(lapackobjfolder)/dorgbr.o \
	$(lapackobjfolder)/dorghr.o $(lapackobjfolder)/dorgl2.o $(lapackobjfolder)/dorglq.o \
	$(lapackobjfolder)/dorgql.o $(lapackobjfolder)/dorgqr.o $(lapackobjfolder)/dorgtr.o \
	$(lapackobjfolder)/dorm2l.o $(lapackobjfolder)/dorm2r.o $(lapackobjfolder)/dormbr.o \
	$(lapackobjfolder)/dormhr.o $(lapackobjfolder)/dorml2.o $(lapackobjfolder)/dormlq.o \
	$(lapackobjfolder)/dormql.o $(lapackobjfolder)/dormqr.o $(lapackobjfolder)/dormr3.o \
	$(lapackobjfolder)/dormrz.o $(lapackobjfolder)/dormtr.o $(lapackobjfolder)/dposv.o \
	$(lapackobjfolder)/dpotf2.o $(lapackobjfolder)/dpotrf.o $(lapackobjfolder)/dpotrs.o \
	$(lapackobjfolder)/dppsv.o $(lapackobjfolder)/dpptrf.o $(lapackobjfolder)/dpptrs.o \
	$(lapackobjfolder)/drscl.o $(lapackobjfolder)/dscal.o $(lapackobjfolder)/dspevd.o \
	$(lapackobjfolder)/dspev.o $(lapackobjfolder)/dspevx.o $(lapackobjfolder)/dspsv.o \
	$(lapackobjfolder)/dsptrd.o $(lapackobjfolder)/dsptrf.o $(lapackobjfolder)/dsptrs.o \
	$(lapackobjfolder)/dstebz.o $(lapackobjfolder)/dstedc.o $(lapackobjfolder)/dstegr.o \
	$(lapackobjfolder)/dstein.o $(lapackobjfolder)/dstemr.o $(lapackobjfolder)/dsteqr.o \
	$(lapackobjfolder)/dsterf.o $(lapackobjfolder)/dswap.o $(lapackobjfolder)/dsyevd.o \
	$(lapackobjfolder)/dsyev.o $(lapackobjfolder)/dsyevr.o $(lapackobjfolder)/dsyevx.o \
	$(lapackobjfolder)/dsygs2.o $(lapackobjfolder)/dsygst.o $(lapackobjfolder)/dsygv.o \
	$(lapackobjfolder)/dsysv.o $(lapackobjfolder)/dsytd2.o $(lapackobjfolder)/dsytf2.o \
	$(lapackobjfolder)/dsytrd.o $(lapackobjfolder)/dsytrf.o $(lapackobjfolder)/dsytrs.o \
	$(lapackobjfolder)/dtgevc.o $(lapackobjfolder)/dtrevc.o $(lapackobjfolder)/dtrexc.o \
	$(lapackobjfolder)/dtrmm.o $(lapackobjfolder)/dtrmv.o $(lapackobjfolder)/dtrsen.o \
	$(lapackobjfolder)/dtrsm.o $(lapackobjfolder)/dtrsyl.o $(lapackobjfolder)/dtrti2.o \
	$(lapackobjfolder)/dtrtri.o $(lapackobjfolder)/dtzrzf.o $(lapackobjfolder)/dummy.o \
	$(lapackobjfolder)/ieeeck.o $(lapackobjfolder)/iladlc.o $(lapackobjfolder)/iladlr.o \
	$(lapackobjfolder)/ilaenv.o $(lapackobjfolder)/ilaprec.o $(lapackobjfolder)/ilatrans.o \
	$(lapackobjfolder)/ilazlc.o $(lapackobjfolder)/ilazlr.o $(lapackobjfolder)/iparmq.o \
	$(lapackobjfolder)/lsame.o $(lapackobjfolder)/xerbla.o \
	$(lapackobjfolder)/zheev.o $(lapackobjfolder)/zhetd2.o $(lapackobjfolder)/zhetrd.o \
	$(lapackobjfolder)/zlacgv.o $(lapackobjfolder)/zladiv.o $(lapackobjfolder)/zlanhe.o \
	$(lapackobjfolder)/zlarfb.o $(lapackobjfolder)/zlarf.o $(lapackobjfolder)/zlarfg.o \
	$(lapackobjfolder)/zlarft.o $(lapackobjfolder)/zlascl.o $(lapackobjfolder)/zlaset.o \
	$(lapackobjfolder)/zlasr.o $(lapackobjfolder)/zlassq.o $(lapackobjfolder)/zlatrd.o \
	$(lapackobjfolder)/zsteqr.o $(lapackobjfolder)/zung2l.o $(lapackobjfolder)/zung2r.o \
	$(lapackobjfolder)/zungql.o $(lapackobjfolder)/zungqr.o $(lapackobjfolder)/zungtr.o \
	$(lapackobjfolder)/dasum.o $(lapackobjfolder)/daxpy.o \
	$(lapackobjfolder)/dbdsdc.o $(lapackobjfolder)/dcopy.o $(lapackobjfolder)/ddot.o \
	$(lapackobjfolder)/dgbsv.o $(lapackobjfolder)/dgbtf2.o $(lapackobjfolder)/dgbtrf.o \
	$(lapackobjfolder)/dgbtrs.o $(lapackobjfolder)/dgeevx.o $(lapackobjfolder)/dger.o \
	$(lapackobjfolder)/dgesdd.o $(lapackobjfolder)/dgtsv.o $(lapackobjfolder)/dlaqtr.o \
	$(lapackobjfolder)/dlasd0.o $(lapackobjfolder)/dlasd1.o $(lapackobjfolder)/dlasd2.o \
	$(lapackobjfolder)/dlasd3.o $(lapackobjfolder)/dlauu2.o $(lapackobjfolder)/dlauum.o \
	$(lapackobjfolder)/dpotri.o $(lapackobjfolder)/drot.o $(lapackobjfolder)/droundup_lwork.o \
	$(lapackobjfolder)/dsyrk.o $(lapackobjfolder)/dtbsv.o $(lapackobjfolder)/dtrevc3.o \
	$(lapackobjfolder)/dtrsna.o $(lapackobjfolder)/idamax.o $(lapackobjfolder)/ilaslc.o \
	$(lapackobjfolder)/ilaslr.o $(lapackobjfolder)/isamax.o $(lapackobjfolder)/saxpy.o \
	$(lapackobjfolder)/scopy.o $(lapackobjfolder)/sdot.o $(lapackobjfolder)/sgemm.o \
	$(lapackobjfolder)/sgemv.o $(lapackobjfolder)/sger.o $(lapackobjfolder)/sisnan.o \
	$(lapackobjfolder)/slacpy.o $(lapackobjfolder)/slae2.o $(lapackobjfolder)/slaebz.o \
	$(lapackobjfolder)/slaev2.o $(lapackobjfolder)/slagtf.o $(lapackobjfolder)/slagts.o \
	$(lapackobjfolder)/slaisnan.o $(lapackobjfolder)/slamch.o $(lapackobjfolder)/slanst.o \
	$(lapackobjfolder)/slansy.o $(lapackobjfolder)/slapy2.o $(lapackobjfolder)/slarf.o \
	$(lapackobjfolder)/slarfb.o $(lapackobjfolder)/slarfg.o $(lapackobjfolder)/slarft.o \
	$(lapackobjfolder)/slarnv.o  $(lapackobjfolder)/slaruv.o \
	$(lapackobjfolder)/slascl.o $(lapackobjfolder)/slaset.o $(lapackobjfolder)/slasr.o \
	$(lapackobjfolder)/slasrt.o $(lapackobjfolder)/slassq.o $(lapackobjfolder)/slatrd.o \
	$(lapackobjfolder)/sorg2l.o $(lapackobjfolder)/sorg2r.o $(lapackobjfolder)/sorgql.o \
	$(lapackobjfolder)/sorgqr.o $(lapackobjfolder)/sorgtr.o $(lapackobjfolder)/sorm2l.o \
	$(lapackobjfolder)/sorm2r.o $(lapackobjfolder)/sormql.o $(lapackobjfolder)/sormqr.o \
	$(lapackobjfolder)/sormtr.o $(lapackobjfolder)/sscal.o $(lapackobjfolder)/sstebz.o \
	$(lapackobjfolder)/sstein.o $(lapackobjfolder)/ssteqr.o $(lapackobjfolder)/ssterf.o \
	$(lapackobjfolder)/sswap.o $(lapackobjfolder)/ssymv.o $(lapackobjfolder)/ssyr2.o \
	$(lapackobjfolder)/ssyr2k.o $(lapackobjfolder)/ssytd2.o $(lapackobjfolder)/ssytrd.o \
	$(lapackobjfolder)/strmm.o $(lapackobjfolder)/strmv.o

#  !---------------------------------------------------------------------!
#  ! Build targets                                                       !
#  !---------------------------------------------------------------------!

$(LAPACKOBJ):$(lapackobjfolder)/%.o:%.f
	@echo "[QUICK]  FC $@"
	 $(VB)$(FC) $(LAPACK_FLAGS) -c $< -o $@

all:$(LAPACKOBJ)
	$(VB)$(ARCH) $(ARCHFLAGS) $(libfolder)/liblapack.$(LIBEXT) $(lapackobjfolder)/*.o

#  !---------------------------------------------------------------------!
#  ! Cleaning targets                                                    !
#  !---------------------------------------------------------------------!

.PHONY: clean

clean:
	@rm -f *.o *.a *.so


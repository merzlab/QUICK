include ../config.h
SHELL=/bin/sh
SRCDIR=..
VPATH=

.f.o:
	@echo "[LAPACK]  FC $<"
	$(VB)$(FC) -c $(FOPTFLAGS) $(FFLAGS) -o $@ $<

OBJS= dsyev.o dsygv.o dlansy.o dsytrd.o dorgtr.o dsteqr.o \
	dsterf.o dlascl.o dlamch.o ilaenv.o dlassq.o dsytd2.o \
	dlatrd.o dorgql.o dorgqr.o dlasrt.o dlaset.o dlanst.o dlaev2.o \
	dlasr.o dlapy2.o dlartg.o dlae2.o dorg2r.o dlarft.o dlarfb.o \
	dorg2l.o dlarfg.o dlarf.o dspev.o dlansp.o dsptrd.o dopgtr.o \
	dppsv.o dspsv.o dgeev.o dsptrs.o dsptrf.o dpptrs.o dpptrf.o \
	dgebak.o dtrevc.o dlacpy.o dhseqr.o dorghr.o dgehrd.o dgebal.o \
	dlange.o dlabad.o dlahqr.o dlarfx.o dlanhs.o dlaln2.o dgehd2.o \
	dlahrd.o dlanv2.o dladiv.o dgerq2.o dorm2r.o dlarnv.o dgeqr2.o \
	dlaruv.o dspevd.o dstedc.o dopmtr.o dlaed0.o dlaed1.o dlaed7.o \
	dlaeda.o dlaed8.o dlaed9.o dlamrg.o dlaed2.o dlaed3.o \
	dpotrs.o dpotrf.o dsygst.o dpotf2.o dsygs2.o dposv.o \
	dlaebz.o dlagtf.o dlagts.o dormqr.o dormql.o dorm2l.o \
	dlaed4.o dlaed5.o dlaed6.o ieeeck.o \
	dsyevx.o dsyevd.o dstebz.o dstein.o dormtr.o \
	zheev.o lsame.o zlanhe.o zlascl.o zhetrd.o zungtr.o zsteqr.o \
	zlassq.o zlatrd.o zhetd2.o zungql.o zungqr.o zlasr.o zlaset.o \
	zlacgv.o zlarfg.o zung2l.o zlarft.o zlarfb.o zung2r.o dlapy3.o \
	zladiv.o zlarf.o \
	dgesvd.o dgebrd.o dgebd2.o dlabrd.o dorgbr.o dbdsqr.o \
	dgeqrf.o dormbr.o dgelqf.o dorglq.o dlas2.o dlasq1.o \
	dlasv2.o dgelq2.o dorgl2.o dlasq2.o dormlq.o dlasq3.o \
	dlasq4.o dlasq5.o dorml2.o dlasq6.o \
	dgesv.o dgetrs.o dlaswp.o dgetrf.o dgerfs.o dgetf2.o \
	dsysv.o dsytrs.o dsytrf.o dlasyf.o dsytf2.o dgelss.o \
	dgelsd.o dlalsd.o dlasdq.o dlasda.o dlalsa.o dlasd6.o dlals0.o \
	dlasdt.o dlasd7.o dlasd8.o dlasd4.o dlasd5.o dlacn2.o \
	dgelsy.o dlaic1.o dtzrzf.o dormrz.o dlarzt.o dlarzb.o \
	dlatrz.o dormr3.o dgeqp3.o dlarz.o dlaqps.o dlaqp2.o dspevx.o \
	dsyevr.o dstegr.o dlarrv.o dlarre.o dlarrb.o dlar1v.o dlarrf.o \
	dggev.o dhgeqz.o dgghrd.o dtgevc.o dggbak.o dggbal.o dlag2.o \
	disnan.o dlaisnan.o dlarfp.o dstemr.o \
	dlarra.o dlarrc.o dlarrd.o dlarrj.o dlarrk.o dlarrr.o dlaneg.o \
	dlaqr0.o dlaqr1.o dlaqr2.o  dlaqr3.o dlaqr4.o dlaqr5.o dlahr2.o \
	dormhr.o dtrexc.o dlaexc.o dlasy2.o \
	iparmq.o iladlc.o iladlr.o \
	dlatrs.o drscl.o dlantr.o dgecon.o dgeequ.o dlaqge.o dgesvx.o \
	chla_transtype.o ilatrans.o ilaprec.o dlascl2.o dgeequb.o

install: $(LIBDIR)/liblapack.a

$(LIBDIR)/liblapack.a: $(OBJS)
	@echo "[LAPACK]  AR $@"
	$(VB)$(AR) liblapack.a $(OBJS) > /dev/null
	$(VB)$(RANLIB) liblapack.a
	$(VB)mv liblapack.a $(LIBDIR)

skip:
	@echo "skipping compilation of LAPACK"

uninstall:
	rm -f $(LIBDIR)/liblapack.a

clean:
	-/bin/rm -f *.o

dlamch.o:  dlamch.f
	@echo "[LAPACK]  FC $<"
	$(VB)$(FC) -c $(FNOOPTFLAGS) $(FFLAGS) -o $@ $<

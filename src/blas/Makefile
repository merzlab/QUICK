MAKEIN = ../../make.in
include $(MAKEIN)

#######################################################################
#
#  Edward Anderson, University of Tennessee
#  March 26, 1990
#  Susan Ostrouchov,  September 30, 1994
#  Julie Langou, March 2007
#
#######################################################################

BLOBJ = $(blasobjfolder)/isamax.o \
	$(blasobjfolder)/sasum.o $(blasobjfolder)/saxpy.o $(blasobjfolder)/scopy.o \
	$(blasobjfolder)/sdot.o $(blasobjfolder)/snrm2.o $(blasobjfolder)/srot.o $(blasobjfolder)/srotg.o \
	$(blasobjfolder)/sscal.o $(blasobjfolder)/sswap.o $(blasobjfolder)/sdsdot.o $(blasobjfolder)/srotmg.o \
	$(blasobjfolder)/srotm.o \
	$(blasobjfolder)/sgemv.o $(blasobjfolder)/sgbmv.o $(blasobjfolder)/ssymv.o $(blasobjfolder)/ssbmv.o \
        $(blasobjfolder)/sspmv.o $(blasobjfolder)/strmv.o $(blasobjfolder)/stbmv.o $(blasobjfolder)/stpmv.o \
        $(blasobjfolder)/strsv.o $(blasobjfolder)/stbsv.o $(blasobjfolder)/stpsv.o $(blasobjfolder)/sger.o \
        $(blasobjfolder)/ssyr.o $(blasobjfolder)/sspr.o $(blasobjfolder)/ssyr2.o $(blasobjfolder)/sspr2.o \
	$(blasobjfolder)/sgemm.o $(blasobjfolder)/ssymm.o $(blasobjfolder)/ssyrk.o $(blasobjfolder)/ssyr2k.o \
        $(blasobjfolder)/strmm.o $(blasobjfolder)/strsm.o \
	$(blasobjfolder)/idamax.o $(blasobjfolder)/dasum.o $(blasobjfolder)/daxpy.o $(blasobjfolder)/dcopy.o \
        $(blasobjfolder)/ddot.o $(blasobjfolder)/dnrm2.o $(blasobjfolder)/drot.o $(blasobjfolder)/drotg.o \
        $(blasobjfolder)/dscal.o $(blasobjfolder)/dsdot.o $(blasobjfolder)/dswap.o $(blasobjfolder)/drotmg.o \
        $(blasobjfolder)/drotm.o \
	$(blasobjfolder)/dgemv.o $(blasobjfolder)/dgbmv.o $(blasobjfolder)/dsymv.o $(blasobjfolder)/dsbmv.o \
        $(blasobjfolder)/dspmv.o $(blasobjfolder)/dtrmv.o $(blasobjfolder)/dtbmv.o $(blasobjfolder)/dtpmv.o \
        $(blasobjfolder)/dtrsv.o $(blasobjfolder)/dtbsv.o $(blasobjfolder)/dtpsv.o $(blasobjfolder)/dger.o \
        $(blasobjfolder)/dsyr.o $(blasobjfolder)/dspr.o $(blasobjfolder)/dsyr2.o $(blasobjfolder)/dspr2.o \
	$(blasobjfolder)/dgemm.o $(blasobjfolder)/dsymm.o $(blasobjfolder)/dsyrk.o $(blasobjfolder)/dsyr2k.o \
        $(blasobjfolder)/dtrmm.o $(blasobjfolder)/dtrsm.o \
	$(blasobjfolder)/scabs1.o $(blasobjfolder)/scasum.o $(blasobjfolder)/scnrm2.o $(blasobjfolder)/icamax.o \
	$(blasobjfolder)/caxpy.o $(blasobjfolder)/ccopy.o $(blasobjfolder)/cdotc.o $(blasobjfolder)/cdotu.o \
	$(blasobjfolder)/csscal.o $(blasobjfolder)/crotg.o $(blasobjfolder)/cscal.o $(blasobjfolder)/cswap.o \
	$(blasobjfolder)/csrot.o \
	$(blasobjfolder)/cgemv.o $(blasobjfolder)/cgbmv.o $(blasobjfolder)/chemv.o $(blasobjfolder)/chbmv.o \
        $(blasobjfolder)/chpmv.o $(blasobjfolder)/ctrmv.o $(blasobjfolder)/ctbmv.o $(blasobjfolder)/ctpmv.o \
        $(blasobjfolder)/ctrsv.o $(blasobjfolder)/ctbsv.o $(blasobjfolder)/ctpsv.o $(blasobjfolder)/cgerc.o \
        $(blasobjfolder)/cgeru.o $(blasobjfolder)/cher.o $(blasobjfolder)/chpr.o $(blasobjfolder)/cher2.o \
        $(blasobjfolder)/chpr2.o \
	$(blasobjfolder)/cgemm.o $(blasobjfolder)/csymm.o $(blasobjfolder)/csyrk.o $(blasobjfolder)/csyr2k.o \
        $(blasobjfolder)/ctrmm.o $(blasobjfolder)/ctrsm.o $(blasobjfolder)/chemm.o $(blasobjfolder)/cherk.o \
        $(blasobjfolder)/cher2k.o \
	$(blasobjfolder)/dcabs1.o $(blasobjfolder)/dzasum.o $(blasobjfolder)/dznrm2.o $(blasobjfolder)/izamax.o \
	$(blasobjfolder)/zaxpy.o $(blasobjfolder)/zcopy.o $(blasobjfolder)/zdotc.o $(blasobjfolder)/zdotu.o \
	$(blasobjfolder)/zdscal.o $(blasobjfolder)/zrotg.o $(blasobjfolder)/zscal.o $(blasobjfolder)/zswap.o \
	$(blasobjfolder)/zdrot.o \
	$(blasobjfolder)/zgemv.o $(blasobjfolder)/zgbmv.o $(blasobjfolder)/zhemv.o $(blasobjfolder)/zhbmv.o \
        $(blasobjfolder)/zhpmv.o $(blasobjfolder)/ztrmv.o $(blasobjfolder)/ztbmv.o $(blasobjfolder)/ztpmv.o \
        $(blasobjfolder)/ztrsv.o $(blasobjfolder)/ztbsv.o $(blasobjfolder)/ztpsv.o $(blasobjfolder)/zgerc.o \
        $(blasobjfolder)/zgeru.o $(blasobjfolder)/zher.o $(blasobjfolder)/zhpr.o $(blasobjfolder)/zher2.o \
        $(blasobjfolder)/zhpr2.o \
	$(blasobjfolder)/zgemm.o $(blasobjfolder)/zsymm.o $(blasobjfolder)/zsyrk.o $(blasobjfolder)/zsyr2k.o \
        $(blasobjfolder)/ztrmm.o $(blasobjfolder)/ztrsm.o $(blasobjfolder)/zhemm.o $(blasobjfolder)/zherk.o \
        $(blasobjfolder)/zher2k.o $(blasobjfolder)/lsame.o $(blasobjfolder)/xerbla.o

$(BLOBJ):$(blasobjfolder)/%.o:%.f
	 $(FC) $(BLAS_FLAGS) -c $< -o $@

all:$(BLOBJ)
	$(ARCH) $(ARCHFLAGS) $(libfolder)/libblas.$(LIBEXT) $(blasobjfolder)/*.o

.PHONY: clean

clean:
	@rm -f *.o *.a *.so


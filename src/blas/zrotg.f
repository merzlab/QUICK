      SUBROUTINE ZROTG(CA,CB,C,S)
*     .. Scalar Arguments ..
      DOUBLE COMPLEX CA,CB,S
      DOUBLE PRECISION C
*     ..
*
*  Purpose
*  =======
*
*     ZROTG determines a double complex Givens rotation.
*
*  =====================================================================
*
*     .. Local Scalars ..
      DOUBLE COMPLEX ALPHA
      DOUBLE PRECISION NORM,SCALE
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC CDABS,DCMPLX,DCONJG,DSQRT
*     ..
      IF (CDABS(CA).EQ.0.0d0) THEN
         C = 0.0d0
         S = (1.0d0,0.0d0)
         CA = CB
      ELSE
         SCALE = CDABS(CA) + CDABS(CB)
         NORM = SCALE*DSQRT((CDABS(CA/DCMPLX(SCALE,0.0d0)))**2+
     $       (CDABS(CB/DCMPLX(SCALE,0.0d0)))**2)
         ALPHA = CA/CDABS(CA)
         C = CDABS(CA)/NORM
         S = ALPHA*DCONJG(CB)/NORM
         CA = ALPHA*NORM
      END IF
      RETURN
      END

      SUBROUTINE ZDROT( N, CX, INCX, CY, INCY, C, S )
*
*     .. Scalar Arguments ..
      INTEGER            INCX, INCY, N
      DOUBLE PRECISION   C, S
*     ..
*     .. Array Arguments ..
      COMPLEX*16         CX( * ), CY( * )
*     ..
*
*  Purpose
*  =======
*
*  Applies a plane rotation, where the cos and sin (c and s) are real
*  and the vectors cx and cy are complex.
*  jack dongarra, linpack, 3/11/78.
*
*  Arguments
*  ==========
*
*  N        (input) INTEGER
*           On entry, N specifies the order of the vectors cx and cy.
*           N must be at least zero.
*           Unchanged on exit.
*
*  CX       (input) COMPLEX*16 array, dimension at least
*           ( 1 + ( N - 1 )*abs( INCX ) ).
*           Before entry, the incremented array CX must contain the n
*           element vector cx. On exit, CX is overwritten by the updated
*           vector cx.
*
*  INCX     (input) INTEGER
*           On entry, INCX specifies the increment for the elements of
*           CX. INCX must not be zero.
*           Unchanged on exit.
*
*  CY       (input) COMPLEX*16 array, dimension at least
*           ( 1 + ( N - 1 )*abs( INCY ) ).
*           Before entry, the incremented array CY must contain the n
*           element vector cy. On exit, CY is overwritten by the updated
*           vector cy.
*
*  INCY     (input) INTEGER
*           On entry, INCY specifies the increment for the elements of
*           CY. INCY must not be zero.
*           Unchanged on exit.
*
*  C        (input) DOUBLE PRECISION
*           On entry, C specifies the cosine, cos.
*           Unchanged on exit.
*
*  S        (input) DOUBLE PRECISION
*           On entry, S specifies the sine, sin.
*           Unchanged on exit.
*
* =====================================================================
*
*     .. Local Scalars ..
      INTEGER            I, IX, IY
      COMPLEX*16         CTEMP
*     ..
*     .. Executable Statements ..
*
      IF( N.LE.0 )
     $   RETURN
      IF( INCX.EQ.1 .AND. INCY.EQ.1 ) THEN
*
*        code for both increments equal to 1
*
         DO I = 1, N
            CTEMP = C*CX( I ) + S*CY( I )
            CY( I ) = C*CY( I ) - S*CX( I )
            CX( I ) = CTEMP
         END DO
      ELSE
*
*        code for unequal increments or equal increments not equal
*          to 1
*
         IX = 1
         IY = 1
         IF( INCX.LT.0 )
     $      IX = ( -N+1 )*INCX + 1
         IF( INCY.LT.0 )
     $      IY = ( -N+1 )*INCY + 1
         DO I = 1, N
            CTEMP = C*CX( IX ) + S*CY( IY )
            CY( IY ) = C*CY( IY ) - S*CX( IX )
            CX( IX ) = CTEMP
            IX = IX + INCX
            IY = IY + INCY
         END DO
      END IF
      RETURN
      END

!
!	eigval.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! TRIDI
!------------------------------------------------------------
SUBROUTINE TRIDI(NDIM,V,A)

  ! TRIDIAGONALIZES A REAL, SYMMETRIC MATRIX A BY THE METHOD OF
  ! HOUSEHOLDER (J. H. WILKINSON, THE COMPUTER JOURNAL, VOL. 3,
  ! P. 23 (1960)).  NDIM IS THE ORDER OF A.  THE DIAGONAL AND
  ! SUBDIAGONAL OF A ARE OVERWRITTEN WITH THE TRIDIAGONALIZED
  ! VERSION OF A.  THE VECTORS USED IN EACH HOUSEHOLDER
  ! TRANSFORMATION ARE STORED ABOVE THE DIAGONAL IN THE FIRST
  ! NDIM-2 ROWS OF A.  THE BETAHS ARE RETURNED BELOW THE SUBDIAGONAL
  ! OF A.  V IS A WORKSPACE ARRAY.

  ! PROGRAMMED BY S. L. DIXON, OCT., 1991.


  use allmod
  IMPLICIT doUBLE PRECISION (A-H,O-Z)
  ! DIMENSION A(NDIM,*),V(3,*)
  DIMENSION A(nbasis,nbasis),V(3,nbasis)

  ! THRESH WILL BE USED AS A THRESHOLD TO DETERMINE if A VALUE SHOULD
  ! BE CONSIDERED TO BE ZERO.  THIS CAN BE CHANGED BY THE USER.

  THRESH = 1.0D-50

  ! if A IS 2 BY 2 OR SMALLER, then IT IS ALREADY TRIDIAGONAL -- NO
  ! NEED TO CONTINUE.

  if(NDIM <= 2) GO TO 1000
  do 500 K=1,NDIM-2

     ! DETERMINE THE VECTOR V USED IN THE HOUSEHOLDER TRANSFORMATION P.
     ! FOR EACH VALUE OF K THE HOUSEHOLDER MATRIX P IS DEFINED AS:

     ! P = I - BETAH*V*V'


     ! CONSTRUCT A HOUSEHOLDER TRANSFORMATION ONLY if THERE IS A NONZERO
     ! OFF-DIAGONAL ELEMENT BELOW A(K,K).

     ALPHA2 = 0.0D0
     do 60 I=K+1,NDIM
        V(1,I) = A(I,K)
        ALPHA2 = ALPHA2 + V(1,I)**2
60   enddo
     APTEMP = ALPHA2 - V(1,K+1)**2
     ALPHA = DSQRT(ALPHA2)
     if(ALPHA >= THRESH)then
        BETAH = 1.0D0/(ALPHA*(ALPHA + ABS(V(1,K+1))))
        SGN = SIGN(1.0D0,V(1,K+1))
        V(1,K+1) = V(1,K+1) + SGN*ALPHA

        ! NOW OVERWRITE A WITH P'*A*P.  THE ENTRIES BELOW THE SUBDIAGONAL
        ! IN THE KTH COLUMN ARE ZEROED BY THE PREMULTIPLICATION BY P'.
        ! THESE ENTRIES WILL BE LEFT ALONE TO SAVE TIME.

        AKV = APTEMP + A(K+1,K)*V(1,K+1)
        S = BETAH*AKV
        A(K+1,K) = A(K+1,K) - S*V(1,K+1)

        ! NOW THE SUBMATRIX CONSISTING OF ROWS K+1,NDIM AND COLUMNS K+1,NDIM
        ! MUST BE OVERWRITTEN WITH THE TRANSFORMATION.

        doT12 = 0.0D0
        BHALF = BETAH*0.5D0
        do 220 I=K+1,NDIM
           SUM = 0.0D0
           do 100 J=K+1,I
              SUM = SUM + A(I,J)*V(1,J)
100        enddo
           if(I < NDIM)then
              do 180 J=I+1,NDIM

                 ! AN UPPER TRIANGULAR ENTRY OF A WILL BE REQUIRED.  MUST USE
                 ! THE SYMMETRIC ENTRY IN THE LOWER TRIANGULAR PART OF A.

                 SUM = SUM + A(J,I)*V(1,J)
180           enddo
           endif
           V(2,I) = BETAH*SUM
           doT12 = doT12 + V(1,I)*V(2,I)
220     enddo
        BH12 = BHALF*doT12
        do 300 I=K+1,NDIM
           V(2,I) = V(2,I) - BH12*V(1,I)
300     enddo
        do 350 J=K+1,NDIM
           do 310 I=J,NDIM
              A(I,J) = A(I,J) - V(1,I)*V(2,J) - V(2,I)*V(1,J)
310        enddo

           ! STORE V(1,J) ABOVE THE DIAGONAL IN ROW K OF A

           A(K,J) = V(1,J)
350     enddo

        ! STORE BETAH BELOW THE SUBDIAGONAL OF A.

        A(K+2,K) = BETAH
     else

        ! NO HOUSEHOLDER TRANSFORMATION IS NECESSARY BECAUSE THE OFF-
        ! DIAGONALS ARE ALL ESSENTIALLY ZERO.

        A(K+2,K) = 0.0D0
        do 460 J=K+1,NDIM
           A(K,J) = 0.0D0
460     enddo
     endif
500 enddo
1000 RETURN
end SUBROUTINE TRIDI



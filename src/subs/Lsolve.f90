!
!	Lsolve.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! LSOLVE
!-----------------------------------------------------------
SUBROUTINE LSOLVE(N,ISIZE,A,B,W,THRESH,X,IERROR)

  ! USES GAUSSIAN ELIMINATION WITH ROW PIVOTING TO SOLVE THE ORDER N
  ! LINEAR SYSTEM:

  ! A*X = B.

  ! W IS A WORK VECTOR OF LENGTH N, AND THRESH IS A THRESHOLD FOR
  ! ZERO PIVOTAL ELEMENTS OF A.

  ! ERROR CODES:  IERROR = 0 - SOLUTION FOUND SUCCESSFULLY;
  ! IERROR = 1 - ZERO PIVOT ENCOUNTERED, SINGULAR MATRIX.


  ! This code was written by Steve Dixon for use in Divcon, and has been
  ! slightly adapted for use in this code. -Ed Brothers.

  IMPLICIT doUBLE PRECISION (A-H,O-Z)
  DIMENSION A(ISIZE,ISIZE),B(ISIZE),W(ISIZE),X(ISIZE)
  IERROR = 0
  if(THRESH <= 0.0D0) THRESH = 1.0D-12
  if(N == 1)then
     if(ABS(A(1,1)) < THRESH)then
        IERROR = 1
     else
        X(1) = B(1)/A(1,1)
     endif
     RETURN
  endif

  ! COMPUTE NORM OF A AS THE AVERAGE ABSOLUTE VALUE.

  AVE = 0.0D0
  do 20 I=1,N
     do 10 J=1,N
        AVE = AVE + ABS(A(I,J))
10   enddo
20 enddo
  AVE = AVE/(N*N)

  IF(AVE .EQ. 0.0D0) THEN
    IERROR = 0
    RETURN
  ENDIF

  AMIN = AVE*THRESH

  ! BEGIN GAUSSIAN ELIMINATION.

  do 100 K=1,N-1

     ! CHECK ENTRIES K THROUGH N OF THE KTH COLUMN OF A TO FIND THE
     ! LARGEST VALUE.

     AIKMAX = 0.0D0
     IROW = K
     do 30 I=K,N
        AIK = ABS(A(I,K))
        if(AIK > AIKMAX)then
           AIKMAX = AIK
           IROW = I
        endif
30   enddo

     ! AIKMAX IS THE ABSOLUTE VALUE OF THE PIVOTAL ELEMENT.  if
     ! AIKMAX IS SMALLER THAN THE THRESHOLD AMIN then THE LEADING
     ! SUBMATRIX IS NEARLY SINGULAR.  IN THIS CASE, RETURN TO CALLING
     ! ROUTINE WITH AN ERROR.

     if(AIKMAX < AMIN)then
        IERROR = 1
        RETURN
     endif

     ! if IROW IS NOT EQUAL TO K then SWAP ROWS K AND IROW OF THE
     ! MATRIX A AND SWAP ENTRIES K AND IROW OF THE VECTOR B.

     if(IROW /= K)then
        do 60 J=1,N
           W(J) = A(IROW,J)
           A(IROW,J) = A(K,J)
           A(K,J) = W(J)
60      enddo
        BSWAP = B(IROW)
        B(IROW) = B(K)
        B(K) = BSWAP
     else
        do 70 J=K+1,N
           W(J) = A(K,J)
70      enddo
     endif

     ! FOR J GREATER THAN OR EQUAL TO I, OVERWRITE A(I,J) WITH
     ! U(I,J); FOR I GREATER THAN J OVERWRITE A(I,J) WITH L(I,J).

     do 90 I=K+1,N
        T = A(I,K)/A(K,K)
        A(I,K) = T
        do 80 J=K+1,N
           A(I,J) = A(I,J) - T*W(J)
80      enddo
90   enddo
100 enddo
  if(ABS(A(N,N)) < AMIN)then
     IERROR = 1
     RETURN
  endif

  ! WE NOW HAVE STORED IN A THE L-U DECOMPOSITION OF P*A, WHERE
  ! P IS A PERMUTATION MATRIX OF THE N BY N IDENTITY MATRIX.  IN
  ! THE VECTOR B WE HAVE P*B.  WE NOW SOLVE L*U*X = P*B FOR THE
  ! VECTOR X.  FIRST OVERWRITE B WITH THE SOLUTION TO L*Y = B
  ! VIA FORWARD ELIMINATION.

  do 160 I=2,N
     do 150 J=1,I-1
        B(I) = B(I) - A(I,J)*B(J)
150  enddo
160 enddo

  ! NOW SOLVE U*X = B FOR X VIA BACK SUBSTITUTION.

  X(N) = B(N)/A(N,N)
  do 200 K=2,N
     I = N+1-K
     X(I) = B(I)
     do 190 J=I+1,N
        X(I) = X(I) - A(I,J)*X(J)
190  enddo
     X(I) = X(I)/A(I,I)
200 enddo
  RETURN
end SUBROUTINE LSOLVE

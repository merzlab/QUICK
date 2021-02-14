SUBROUTINE DIAGMKL(NDIM,A,EVAL1,EVEC1,IERROR)

  IMPLICIT doUBLE PRECISION (A-H,O-Z)
  integer :: NDIM,LWMAX
  DIMENSION A(NDIM,NDIM),EVEC1(NDIM,NDIM),EVAL1(NDIM)

  integer :: LDA, LWORK, LIWORK  
  integer, allocatable, dimension(:) :: IWORK
  double precision, allocatable, dimension(:):: WORK

#if defined LAPACK || defined MKL

  LDA = NDIM
  LWMAX = 1+2*NDIM+6*NDIM**2
  EVEC1 = A

  allocate(IWORK(LWMAX))
  allocate(WORK(LWMAX))

  if(NDIM == 1)then
     EVAL1(1) = A(1,1)
     EVEC1(1,1) = 1.0D0
     RETURN
  endif
 
  ! Query the optimal workspace
  LWORK = -1
  LIWORK = -1
  call dsyevd('Vectors', 'Upper', NDIM, EVEC1, LDA, EVAL1, WORK, LWORK, &
IWORK, LIWORK, IERROR)
  LWORK = min(LWMAX, int(WORK(1)))
  LIWORK = min(LWMAX, IWORK(1))

  ! Solve eigenproblem
  call dsyevd('Vectors', 'Upper', NDIM, EVEC1, LDA, EVAL1, WORK, LWORK, &
IWORK, LIWORK, IERROR)

  deallocate(IWORK)
  deallocate(WORK)

    RETURN
#endif
end SUBROUTINE DIAGMKL

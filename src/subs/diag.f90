#include "util.fh"

  !---------------------------------------------------------------------!
  ! Written by Chi Jin on 02/16/2021                                    !
  !                                                                     !
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains methods required for MKL or OPENBLAS      !
  ! installation. The subroutine provides an alternative diagonalizer   !
  ! as a wrapper of the dsyev function built in MKL and OPENBLAS.       !
  !                                                                     !
  !---------------------------------------------------------------------!

SUBROUTINE CPU_DIAG(A, N, M, EVAL, EVEC, IERROR)
  implicit none

  integer, intent(in) :: N, M
  integer, intent(out) :: IERROR
  double precision, intent(in) :: A(N,M)
  double precision, intent(out) :: EVEC(N,M), EVAL(M)

  integer :: LWMAX, LDA, LWORK, LIWORK  
  integer, allocatable, dimension(:) :: IWORK
  double precision, allocatable, dimension(:):: WORK

  LDA = N
  LWMAX = 1+2*N+6*N**2
  EVEC = A

  allocate(IWORK(LWMAX))
  allocate(WORK(LWMAX))

  if(N == 1)then
     EVAL(1) = A(1,1)
     EVEC(1,1) = 1.0D0
     RETURN
  endif
 
  ! Query the optimal workspace
  LWORK = -1
  LIWORK = -1
  call dsyevd('Vectors', 'Upper', N, EVEC, LDA, EVAL, WORK, LWORK, &
          IWORK, LIWORK, IERROR)
  LWORK = min(LWMAX, int(WORK(1)))
  LIWORK = min(LWMAX, IWORK(1))

  ! Solve eigenproblem
  call dsyevd('Vectors', 'Upper', N, EVEC, LDA, EVAL, WORK, LWORK, &
          IWORK, LIWORK, IERROR)

  deallocate(IWORK)
  deallocate(WORK)

  RETURN
end SUBROUTINE CPU_DIAG

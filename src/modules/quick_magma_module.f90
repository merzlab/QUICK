#include "util.fh"
!---------------------------------------------------------------------!
! Created by Madu Manathunga on 10/05/2022                            !
!                                                                     ! 
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This module contains drivers required for utilizing MAGMA           !
! in QUICK.                                                           !
!_____________________________________________________________________! 

#ifdef WITH_MAGMA

module quick_magma_module

    implicit none

    private
    public :: magmaDIAG

    interface magmaDIAG
        module procedure quick_magma_dsyevd
    end interface magmaDIAG

contains

    ! Driver for dsyevd, similar to DIAG and DIAGMKL. 
    subroutine quick_magma_dsyevd(n, A, Eval, Evec, ierr)
 
        use magma
        implicit none
        integer, intent(in) :: n  ! Number of rows and columns of matrix A.
        double precision, dimension(n,n), intent(in) :: A ! Matrix to diagonalize   
        double precision, dimension(n), intent(out) :: Eval ! Resulting eigenvalues
        double precision, dimension(n,n), intent(out) :: Evec ! Resulting eigenvectors
        integer, intent(out) :: ierr ! error code, HIPCHECK should handle the errors here, but pass this anyway

        integer :: lda, ldda, i, j, lwork, liwork, aux_iwork(1) ! indices for iterating over results        
        integer(kind=8) :: dA, dev_null_ptr, queue
        double precision :: w(1), wA(1), aux_work(1)
        integer, allocatable :: iwork(:)
        double precision, allocatable :: h_work(:)

        lda=n
        ldda=n
        ierr=0

        ierr = magmaf_dmalloc( dA, ldda*n )
        call magmaf_queue_create(0, queue )
        call magmaf_dsetmatrix( n, n, A, ldda, dA, ldda, queue )
        call magmaf_queue_destroy( queue )

        call magmaf_dsyevd_gpu('V', 'U', n, dev_null_ptr, ldda, w, wA, lda, aux_work, -1, aux_iwork, -1, ierr)
        lwork = int(aux_work(1))
        liwork = aux_iwork(1)

        if(.not. allocated(h_work)) allocate(h_work(lwork))
        if(.not. allocated(iwork)) allocate(iwork(liwork))

        call magmaf_dsyevd_gpu('V', 'U', n, dA, ldda, Eval, Evec, lda, h_work, lwork, iwork, liwork, ierr)

        call magmaf_queue_create( 0, queue )
        call magmaf_dgetmatrix(n, n, dA, ldda, Evec, lda, queue )
        call magmaf_queue_destroy( queue )

        ierr = magmaf_free( dA )
        if(allocated(h_work)) deallocate(h_work)
        if(allocated(iwork)) deallocate(iwork)

        Evec = -Evec

        !Output results
        !do i = 1,n
        !  do j=1,n
        !    write(*,*) "DSYEVD", i, j, A(j,i), Evec(j,i), Eval(j)
        !  enddo
        !end do   

    end subroutine quick_magma_dsyevd

end module quick_magma_module

#endif

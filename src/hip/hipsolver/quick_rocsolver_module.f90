#include "util.fh"
!---------------------------------------------------------------------!
! Created by Madu Manathunga on 11/19/2021                            !
!                                                                     ! 
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This module contains drivers required for utilizing rocSolver       !
! in QUICK.                                                           !
!_____________________________________________________________________! 

module quick_rocsolver_module

    implicit none

    private
    public :: rocDIAG
    
    interface rocDIAG
        module procedure quick_rocsolver_dsyevd
    end interface rocDIAG

contains

    ! Driver for dsyevd, similar to DIAG and DIAGMKL. 
    subroutine quick_rocsolver_dsyevd(n, A, Eval, Evec, ierr)

        use iso_c_binding
        use rocblas
        use rocblas_enums 
        use rocblas_enums_extra
        use quick_rocblas_module, only: handle, hinfo, dinfo, dA, dB, dC, hA, hB, hC, rocBlasInit, rocBlasFinalize 
        use rocblas_extra
        use rocsolver_extra    

        implicit none

        integer, intent(in) :: n  ! Number of rows and columns of matrix A.
        double precision, dimension(n,n), intent(in) :: A ! Matrix to diagonalize   
        double precision, dimension(n), intent(out) :: Eval ! Resulting eigenvalues
        double precision, dimension(n,n), intent(out) :: Evec ! Resulting eigenvectors
        integer, intent(out) :: ierr ! error code, HIPCHECK should handle the errors here, but pass this anyway

        integer :: i, j ! indices for iterating over results
   
        ! internal rocblas/rocsolver variables 
        integer(c_int) :: rb_n 
        integer(c_int) :: rb_lda
        integer(c_size_t) :: size_A
        !integer(c_int), target :: hinfo(1)
        !type(c_ptr), target :: info
        integer(kind(rocblas_fill_lower)) :: uplo 
        integer(kind(rocblas_evect_original)) :: evect

        call rocBlasInit(n)

        ! Initialize variables
        ierr=0    
        evect = rocblas_evect_original
        uplo = rocblas_fill_lower
        rb_n = n
        rb_lda = n
        size_A=n*n    
        hA = reshape(A, (/size_A/))

        if(.not. allocated(hinfo)) allocate(hinfo(1))

        ! Allocate device-side memory
        call HIP_CHECK(hipMalloc(c_loc(dinfo), int(1, c_size_t) * 4))
    
        ! Copy memory from host to device
        call HIP_CHECK(hipMemcpy(dA, c_loc(hA), size_A * 8, 1))
        call HIP_CHECK(hipMemset(dB, 0, size_A * 8))
        call HIP_CHECK(hipMemset(dC, 0, size_A * 8))
        call HIP_CHECK(hipMemset(dinfo, 0, int(1, c_size_t) *4))

        ! Create rocBLAS handle
        call ROCBLAS_CHECK(rocblas_create_handle(c_loc(handle)))

        ! Set handle and call rocblas_dgemm
        call ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, 0))
    
        ! Do the diagonalization on the device, see rocsolver_extra_module for more details.
        call ROCSOLVER_CHECK(rocsolver_dsyevd(handle, evect, uplo, rb_n, dA, rb_lda, dB, dC, dinfo))
    
        call HIP_CHECK(hipDeviceSynchronize()) 
    
        ! Copy result from device to host
        call HIP_CHECK(hipMemcpy(c_loc(hA), dA, size_A * 8, 2))
        call HIP_CHECK(hipMemcpy(c_loc(hB), dB, size_A * 8, 2))
        call HIP_CHECK(hipMemcpy(c_loc(hC), dC, size_A * 8, 2))
        call HIP_CHECK(hipMemcpy(c_loc(hinfo), dinfo, int(1, c_size_t)  * 4, 2))
   
        ! Transfer result
        Eval = hB(1:n)
        Evec = reshape(hA, (/n, n/))         

        !write(*,*) ""

        !Output results
        !do i = 1,n
        !  do j=1,n
        !    write(*,*) "DSYEVD", i, j, A(j,i), Evec(j,i), Eval(j), hC(n*(i-1)+j) 
        !  enddo
        !end do    
    
        ! Destroy rockblas handle
        call ROCBLAS_CHECK(rocblas_destroy_handle(handle))

        ! Clean up
        call HIP_CHECK(hipFree(dinfo))
        if(allocated(hinfo)) deallocate(hinfo)
        !write(*,*) "hipFree(info) done"

        call rocBlasFinalize()

    end subroutine quick_rocsolver_dsyevd

end module quick_rocsolver_module

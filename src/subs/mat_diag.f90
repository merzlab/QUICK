#include "util.fh"
!
!       mat_diag.f90
!
!       Architecture agnostic wrapper for generalized
!       eigen-decompisition via LAPACK-style interface.
!
!       Created by Vikrant Tripathy on 01/16/26.
!
!-----------------------------------------------------------

subroutine MAT_DIAG(A, N, M, eval, evec)
#if defined(HIP) || defined(HIP_MPIV)
#if defined(WITH_MAGMA)
     use quick_magma_module, only: magmaDIAG
#elif defined(WITH_ROCSOLVER)
     use quick_rocsolver_module, only: rocDIAG
#endif
#endif
    use iso_fortran_env, only: error_unit
    use quick_exception_module, only: RaiseException
    implicit none

    integer, intent(in) :: N, M
    double precision, intent(in) :: A(N,M)
    double precision, intent(out) :: eval(M), evec(N,M)

    integer :: IERROR, ierr
    logical :: isopen

    ! Every backend below returns an error flag, but this wrapper used to
    ! declare IERROR purely as a scratch variable and discard it. A failed
    ! eigensolve returns meaningless eigenvalues, and an SCF built on top of
    ! them does not blow up -- it converges smoothly onto a wrong answer, which
    ! is far harder to notice than a crash. Treat any non-zero flag as fatal.
    IERROR = 0

#if defined(HIP) || defined(HIP_MPIV)
#if defined(WITH_MAGMA)
    call magmaDIAG(A, N, M, eval, evec, IERROR)
#elif defined(WITH_ROCSOLVER)
    call rocDIAG(A, N, M, eval, evec, IERROR)
#else
    ! CPU fallback for older HIP versions with poor rocSOLVER performance
    call CPU_DIAG(A, N, M, eval, evec, IERROR)
#endif
#elif defined(CUDA) || defined(CUDA_MPIV)
    ! CUDA_DIAG reports cuSOLVER's devInfo, which is the ONLY way a
    ! cusolverDnDsyevd non-convergence becomes visible: the routine still
    ! returns CUSOLVER_STATUS_SUCCESS in that case. See quick_cusolver.c.
    call CUDA_DIAG(A, N, M, eval, evec, IERROR)
#else
    ! CPU-based diagonalization via LAPACK-style libraries (bundled or external)
    call CPU_DIAG(A, N, M, eval, evec, IERROR)
#endif

    if (IERROR /= 0) then
       write(error_unit,'("MAT_DIAG: eigensolver FAILED, info = ",i0, &
             & ", matrix dimension = ",i0)') IERROR, N
       write(error_unit,'("MAT_DIAG: the returned eigenvalues and eigenvectors are NOT valid.")')
       flush(error_unit)

       inquire(unit=OUTFILEHANDLE, opened=isopen)
       if (isopen) then
          write(OUTFILEHANDLE,'(" MAT_DIAG: eigensolver FAILED, info = ",i0, &
                & ", matrix dimension = ",i0)') IERROR, N
          write(OUTFILEHANDLE,'(" MAT_DIAG: the returned eigenvalues and eigenvectors are NOT valid.")')
       endif

       ierr = 49
       call RaiseException(ierr)
    endif

end subroutine MAT_DIAG

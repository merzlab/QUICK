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

    integer, intent(in) :: N, M
    double precision, intent(in) :: A(N,M)
    double precision, intent(out) :: eval(M), evec(N,M)

    integer :: IERROR

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
    call CUDA_DIAG(A, N, M, eval, evec)    
#else  
    ! CPU-based diagonalization via LAPACK-style libraries (bundled or external)
    call CPU_DIAG(A, N, M, eval, evec, IERROR)
#endif

end subroutine MAT_DIAG

!
!       MatDiag.f90
!
!       Created by Vikrant Tripathy on 01/16/26.
!
!-----------------------------------------------------------

subroutine MatDiag(Mat,Eig,Vec,Cutoff,idegen,V2,nbasis)
#if defined(HIP) || defined(HIP_MPIV)
#if defined(WITH_MAGMA)
     use quick_magma_module, only: magmaDIAG
#elif defined(WITH_ROCSOLVER)
     use quick_rocsolver_module, only: rocDIAG
#endif
#endif

    integer :: nbasis,IERROR
    double precision :: Mat(nbasis,nbasis), Eig(nbasis), Vec(nbasis,nbasis)
    double precision :: V2(3, nbasis), idegen(nbasis), Cutoff

#if defined(HIP) || defined(HIP_MPIV)                                          
#if defined(WITH_MAGMA)
           call magmaDIAG(nbasis, Mat, Eig, Vec, IERROR)
#elif defined(WITH_ROCSOLVER) 
           call rocDIAG(nbasis, Mat, Eig, Vec, IERROR)
#else
#if defined(LAPACK) || defined(MKL)
           call DIAGMKL(nbasis, Mat, Eig, Vec, IERROR)
#else
           call DIAG(nbasis, Mat, nbasis, Cutoff, V2, Eig, idegen, Vec, IERROR)
#endif
#endif
#elif defined(CUDA) || defined(CUDA_MPIV)
           call CUDA_DIAG(Mat, Eig, Vec, nbasis)    
#else  
#if defined(LAPACK) || defined(MKL)                                            
           call DIAGMKL(nbasis, Mat, Eig, Vec, IERROR)
#else
           call DIAG(nbasis, Mat, nbasis, Cutoff, V2, Eig, idegen, Vec, IERROR)
#endif
#endif

end subroutine MatDiag

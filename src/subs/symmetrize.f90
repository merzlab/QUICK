!---------------------------------------------------------!
!  Symmetrize either upper or lower square matrix         !
!                                                         !
!  Parameters:                                            !
!      UPLO     'U'                                       !
!               'L'                                       !
!      mat      N * N matrix                              !
!      N        size of the matrix                        !
!---------------------------------------------------------!

 subroutine symmetrize(UPLO,mat,N)
  implicit none

  integer :: iatom, jatom, N
  double precision :: mat(N,N)
  character :: UPLO

  if (UPLO == 'U') then
    do iatom = 2,N
      do jatom = 1,iatom - 1
        mat(iatom,jatom) = mat(jatom,iatom)
      end do
    end do
  else if (UPLO == 'L') then
    do iatom = 2,N
      do jatom = 1,iatom - 1
        mat(jatom,iatom) = mat(iatom,jatom)
      end do
    end do
  else
     call QuickErr('UPLO has to be either U or L in symmetrize')
    Return
  end if

 end subroutine symmetrize

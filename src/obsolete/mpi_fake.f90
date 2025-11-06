#include "util.fh"
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Fake_MPI subroutine to make single CPU computation compatible
! Yipu Miao 09/28/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!Subroutine list:
!MPI_INIT()
!MPI_COMM_RANK(MPI_COMM_WORL,MPIRANK,IERR)
!MPI_COMM_SIZE(quick_comm,MPISIZE,IERR)
!MPI_FINALIZE(IERR)
!MPI_BARRIER(quick_comm,IERR)

!=======================================================================

    SUBROUTINE MPI_INIT(IERR)
    IMPLICIT NONE
    INTEGER IERR
    IERR=0
    RETURN
    END SUBROUTINE MPI_INIT

!=======================================================================

    SUBROUTINE MPI_COMM_RANK(quick_comm,MPIRANK,IERR)
    IMPLICIT NONE
    INTEGER :: IERR,quick_comm,MPIRANK

    MPIRANK=0
    RETURN
    END SUBROUTINE MPI_COMM_RANK

!=======================================================================

    SUBROUTINE MPI_COMM_SIZE(quick_comm,MPISIZE,IERR)
    IMPLICIT NONE
    INTEGER :: IERR,quick_comm,MPISIZE

    MPISIZE=1
    RETURN
    END SUBROUTINE MPI_COMM_SIZE

!=======================================================================

    SUBROUTINE MPI_FINALIZE(IERR)
    IMPLICIT NONE
    INTEGER :: IERR

    IERR=0
    RETURN
    END SUBROUTINE MPI_FINALIZE

!=======================================================================

    SUBROUTINE MPI_BARRIER(quick_comm,IERR)
    IMPLICIT NONE
    INTEGER :: IERR,quick_comm

    IERR=0
    RETURN
    END SUBROUTINE MPI_BARRIER
!=======================================================================

    SUBROUTINE MPI_ABORT(quick_comm,STATUS,IERR)
    IMPLICIT NONE
    INTEGER :: IERR,quick_comm,STATUS

    IERR=0
    RETURN
    END SUBROUTINE MPI_BARRIER

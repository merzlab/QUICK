!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Fake_MPI subroutine to make single CPU computation compatible
! Yipu Miao 09/28/2010
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!Subroutine list:
!MPI_INIT()
!MPI_COMM_RANK(MPI_COMM_WORL,MPIRANK,IERR)
!MPI_COMM_SIZE(MPI_COMM_WORLD,MPISIZE,IERR)
!MPI_FINALIZE(IERR)
!MPI_BARRIER(MPI_COMM_WORLD,IERR)

!=======================================================================

    SUBROUTINE MPI_INIT(IERR)
    IMPLICIT NONE
    INTEGER IERR
    IERR=0
    RETURN
    END SUBROUTINE MPI_INIT

!=======================================================================

    SUBROUTINE MPI_COMM_RANK(MPI_COMM_WORLD,MPIRANK,IERR)
    IMPLICIT NONE
    INTEGER :: IERR,MPI_COMM_WORLD,MPIRANK

    MPIRANK=0
    RETURN
    END SUBROUTINE MPI_COMM_RANK

!=======================================================================

    SUBROUTINE MPI_COMM_SIZE(MPI_COMM_WORLD,MPISIZE,IERR)
    IMPLICIT NONE
    INTEGER :: IERR,MPI_COMM_WORLD,MPISIZE

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

    SUBROUTINE MPI_BARRIER(MPI_COMM_WORLD,IERR)
    IMPLICIT NONE
    INTEGER :: IERR,MPI_COMM_WORLD

    IERR=0
    RETURN
    END SUBROUTINE MPI_BARRIER
!=======================================================================

    SUBROUTINE MPI_ABORT(MPI_COMM_WORLD,STATUS,IERR)
    IMPLICIT NONE
    INTEGER :: IERR,MPI_COMM_WORLD,STATUS

    IERR=0
    RETURN
    END SUBROUTINE MPI_BARRIER

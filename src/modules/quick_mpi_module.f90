!
!	quick_mpi_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

! quick MPI module.
module quick_mpi_module

!------------------------------------------------------------------------
!  ATTRIBUTES  : quick_mpi_error, quick_comm_rank, quick_comm_size,
!                namelen, pname, master, bMPI
!  SUBROUTINES : check_quick_mpi
!                print_quick_mpi
!  FUNCTIONS   : none
!  DESCRIPTION : This module is to gather MPI information
!  AUTHOR      : Yipu Miao
!------------------------------------------------------------------------    
#ifdef MPIV
    use, intrinsic :: iso_c_binding
    use mpi

    public :: quick_comm

    ! QUICK MPI communicator
    ! NOTE: for non-API codepaths, comm is set to MPI_COMM_WORLD;
    !       for API codepaths, comm is initialized and passed in
    !       from external code (i.e., comm may or may not be set
    !       to MPI_COMM_WORLD)
    integer :: quick_comm = MPI_COMM_NULL
#endif
    ! error codes from MPI API calls
    integer :: quick_mpi_error
    ! MPI rank for this process in the QUICK MPI communicator
    integer :: quick_comm_rank
    ! MPI total processor count (size) in the QUICK MPI communicator
    integer :: quick_comm_size
    ! MPI rank for this process in MPI_COMM_WORLD communicator
    integer :: mpi_world_rank
    ! MPI total processor count (size) in the MPI_COMM_WORLD communicator
    integer :: mpi_world_size
    ! MPI process name length
    integer :: namelen
    ! MPI process name
    character(len=80) pname
    ! TRUE if this process is the master node in quick_comm, FALSE otherwise
    logical :: master = .true.
    ! TRUE if QUICK is being run via MPI, FALSE otherwise
    logical :: bMPI = .true.
    ! if mpi is initialized somewhere other than quick
    logical :: libMPIMode = .false.
    integer, allocatable :: quick_mpi_status(:)
    integer, allocatable :: mgpu_ids(:)    
    integer :: mgpu_id
#if defined(CEW)
    integer, allocatable :: natomll(:)
    integer, allocatable :: natomul(:)
    integer, allocatable :: nextatomll(:)
    integer, allocatable :: nextatomul(:)
#endif

    integer, parameter :: MIN_1E_MPI_BASIS = 6

    contains


    !----------------
    ! check mpi setup
    !----------------
    subroutine check_quick_mpi(io,ierr)
        implicit none

        integer io
        integer, intent(inout) :: ierr
        
        if (bMPI .and. quick_comm_size.eq.1) then
            bMPI=.false.
        endif
    end subroutine
    
    
    !----------------
    ! print mpi setup
    !----------------
    subroutine print_quick_mpi(io,ierr)
        implicit none

        integer io
        integer, intent(inout) :: ierr        
        
        write (io,*)
        write (io,'("| - MPI Enabled -")')
        write (io,'("| TOTAL RANKS     = ",i5)') quick_comm_size
        write (io,'("| MASTER NAME     = ",A30)') pname
    end subroutine print_quick_mpi


    ! all multi gpu mpi variable allocation should go here
    subroutine allocate_mgpu()
      implicit none

      if( .not. allocated(mgpu_ids)) allocate(mgpu_ids(quick_comm_size))
    end subroutine allocate_mgpu


    ! all multi gpu mpi variable deallocation should go here
    subroutine deallocate_mgpu()
      implicit none

      if(allocated(mgpu_ids)) deallocate(mgpu_ids)
    end subroutine deallocate_mgpu


#if defined(CEW)
    subroutine mpi_distribute_atoms(natom, nextatom)
      implicit none

      integer, intent(in) :: natom
      integer, intent(in) :: nextatom

      if( .not. allocated(natomll)) allocate(natomll(quick_comm_size))
      if( .not. allocated(natomul)) allocate(natomul(quick_comm_size))
      if( .not. allocated(nextatomll)) allocate(nextatomll(quick_comm_size))
      if( .not. allocated(nextatomul)) allocate(nextatomul(quick_comm_size))

      call naive_distribute(natom,quick_comm_size,natomll,natomul)
      call naive_distribute(nextatom,quick_comm_size,nextatomll,nextatomul)
    end subroutine mpi_distribute_atoms


    subroutine mpi_delete_atoms()
      implicit none

      if(allocated(natomll)) deallocate(natomll)
      if(allocated(natomul)) deallocate(natomul)
      if(allocated(nextatomll)) deallocate(nextatomll)
      if(allocated(nextatomul)) deallocate(nextatomul)      
    end subroutine mpi_delete_atoms
#endif
end module quick_mpi_module

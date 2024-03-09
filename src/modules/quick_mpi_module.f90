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
!  ATTRIBUTES  : mpierror,mpirank,myid,namelen,mpisiz,pname
!                master,bMPI
!  SUBROUTINES : check_quick_mpi
!                print_quick_mpi
!  FUNCTIONS   : none
!  DESCRIPTION : This module is to gather MPI information
!  AUTHOR      : Yipu Miao
!------------------------------------------------------------------------    

#ifdef MPIV
    use mpi
#endif

    integer :: mpierror
    integer :: mpirank
    integer :: myid
    integer :: namelen
    integer :: mpisize
    character(len=80) pname
    logical :: master = .true.      ! flag to show if the node is master node
    logical :: bMPI = .true.        ! flag to show if MPI is turn on
    logical :: libMPIMode = .false. ! if mpi is initialized somewhere other than quick
    integer, allocatable :: MPI_STATUS(:)
    integer, parameter :: MIN_1E_MPI_BASIS=6
    integer, allocatable :: mgpu_ids(:)    
    integer :: mgpu_id

    integer, allocatable :: natomll(:)
    integer, allocatable :: natomul(:)
    integer, allocatable :: nextatomll(:)
    integer, allocatable :: nextatomul(:)

    contains
    
    !----------------
    ! check mpi setup
    !----------------
    subroutine check_quick_mpi(io,ierr)
        implicit none
        integer io
        integer, intent(inout) :: ierr
        
        if (bMPI .and. mpisize.eq.1) then
            bMPI=.false.
        endif
        
        return
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
        write (io,'("| TOTAL RANKS     = ",i5)') mpisize
        write (io,'("| MASTER NAME     = ",A30)') pname
        
    end subroutine print_quick_mpi

    ! all multi gpu mpi variable allocation should go here
    subroutine allocate_mgpu()

      implicit none

      if( .not. allocated(mgpu_ids)) allocate(mgpu_ids(mpisize))

    end subroutine allocate_mgpu

    ! all multi gpu mpi variable deallocation should go here
    subroutine deallocate_mgpu()

      implicit none

      if(allocated(mgpu_ids)) deallocate(mgpu_ids)
    
    end subroutine deallocate_mgpu

    subroutine mpi_distribute_atoms(natom, nextatom)

      implicit none
      integer, intent(in) :: natom
      integer, intent(in) :: nextatom

      if( .not. allocated(natomll)) allocate(natomll(mpisize))
      if( .not. allocated(natomul)) allocate(natomul(mpisize))
      if( .not. allocated(nextatomll)) allocate(nextatomll(mpisize))
      if( .not. allocated(nextatomul)) allocate(nextatomul(mpisize))

      call naive_distribute(natom,mpisize,natomll,natomul)
      call naive_distribute(nextatom,mpisize,nextatomll,nextatomul)

    end subroutine mpi_distribute_atoms

    subroutine mpi_delete_atoms()

      implicit none

      if(allocated(natomll)) deallocate(natomll)
      if(allocated(natomul)) deallocate(natomul)
      if(allocated(nextatomll)) deallocate(nextatomll)
      if(allocated(nextatomul)) deallocate(nextatomul)      

    end subroutine mpi_delete_atoms
end module quick_mpi_module

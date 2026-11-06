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

    ! Max element count per individual MPI collective call for the chunked
    ! helpers below -- several MPI libraries actually linked against this
    ! code (e.g. OpenMPI 4.1.x on our HPC cluster) implement only the
    ! classic MPI-3.1 Fortran bindings, whose "count" argument is a plain
    ! 4-byte C int no matter what Fortran integer kind the caller passes in.
    integer(kind=8), parameter :: MAX_MPI_COUNT = 500000000_8
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


#ifdef MPIV
    ! Chunked MPI collectives for buffers that may exceed INT32_MAX elements
    ! (e.g. jbasis^2-scale DnC/ERI tables for large systems) -- widening a
    ! Fortran variable to 8 bytes does NOT avoid the overflow on its own
    ! (see MAX_MPI_COUNT above), since the count still truncates/misbehaves
    ! once it crosses 2^31-1 at the MPI call boundary. These helpers instead
    ! split one logical collective into multiple calls, each within a safe
    ! 4-byte count.

    ! Broadcasts a double precision buffer of total_count elements, which may
    ! exceed INT32_MAX. buf must be the first element of a contiguous array
    ! (Fortran sequence association) -- callers of a multi-dimensional array
    ! pass e.g. Apri(1,1).
    subroutine mpi_bcast_dp_chunked(buf, total_count, root, comm, ierr)
        implicit none
        double precision :: buf(*)
        integer(kind=8), intent(in) :: total_count
        integer, intent(in) :: root, comm
        integer, intent(out) :: ierr
        integer(kind=8) :: offset, remaining
        integer :: this_chunk

        offset = 0_8
        remaining = total_count
        ierr = 0
        do while (remaining > 0_8)
            this_chunk = int(min(remaining, MAX_MPI_COUNT), kind=4)
            call MPI_BCAST(buf(offset+1), this_chunk, mpi_double_precision, root, comm, ierr)
            if (ierr /= 0) return
            offset = offset + int(this_chunk, kind=8)
            remaining = remaining - int(this_chunk, kind=8)
        end do
    end subroutine mpi_bcast_dp_chunked

    ! Same as mpi_bcast_dp_chunked, for default-kind INTEGER buffers.
    subroutine mpi_bcast_int_chunked(buf, total_count, root, comm, ierr)
        implicit none
        integer :: buf(*)
        integer(kind=8), intent(in) :: total_count
        integer, intent(in) :: root, comm
        integer, intent(out) :: ierr
        integer(kind=8) :: offset, remaining
        integer :: this_chunk

        offset = 0_8
        remaining = total_count
        ierr = 0
        do while (remaining > 0_8)
            this_chunk = int(min(remaining, MAX_MPI_COUNT), kind=4)
            call MPI_BCAST(buf(offset+1), this_chunk, mpi_integer, root, comm, ierr)
            if (ierr /= 0) return
            offset = offset + int(this_chunk, kind=8)
            remaining = remaining - int(this_chunk, kind=8)
        end do
    end subroutine mpi_bcast_int_chunked

    ! Elementwise SUM all-reduce of a double precision buffer of total_count
    ! elements, chunked the same way. sendbuf and recvbuf must be distinct
    ! actual arrays (MPI_ALLREDUCE forbids aliasing sendbuf/recvbuf).
    subroutine mpi_allreduce_sum_dp_chunked(sendbuf, recvbuf, total_count, comm, ierr)
        implicit none
        double precision :: sendbuf(*), recvbuf(*)
        integer(kind=8), intent(in) :: total_count
        integer, intent(in) :: comm
        integer, intent(out) :: ierr
        integer(kind=8) :: offset, remaining
        integer :: this_chunk

        offset = 0_8
        remaining = total_count
        ierr = 0
        do while (remaining > 0_8)
            this_chunk = int(min(remaining, MAX_MPI_COUNT), kind=4)
            call MPI_ALLREDUCE(sendbuf(offset+1), recvbuf(offset+1), this_chunk, &
                                mpi_double_precision, MPI_SUM, comm, ierr)
            if (ierr /= 0) return
            offset = offset + int(this_chunk, kind=8)
            remaining = remaining - int(this_chunk, kind=8)
        end do
    end subroutine mpi_allreduce_sum_dp_chunked
#endif


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

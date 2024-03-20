#include "util.fh"
!
!	finalize.f90
!	new_quick
!
!	Created by Yipu Miao on 3/4/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!   subroutine inventory:
!       deallocate_calculated
!       deallocateall
!       finalize

Subroutine deallocate_calculated
!   Xiao HE deallocate 07/17/2007
!   Ken Ayers 05/26/04
!   Subroutines for allocation of the various matricies in quick
!   These routines are not the ideal way to deal with allocation of
!   variables.  Large sized arrays should only be allocated when
!   they are needed.  Eventually someone will deal with this.
  use allmod
  use quick_gridpoints_module
  if (allocated(Yxiao)) deallocate(Yxiao)
  if (allocated(Yxiaotemp)) deallocate(Yxiaotemp)
  if (allocated(Yxiaoprim)) deallocate(Yxiaoprim)
  if (allocated(attraxiao)) deallocate(attraxiao)
  if (allocated(attraxiaoopt)) deallocate(attraxiaoopt)
  if (allocated(Ycutoff)) deallocate(Ycutoff)
  if (allocated(cutmatrix)) deallocate(cutmatrix)
  if (allocated(sigrad2)) deallocate(sigrad2)

  call dealloc(quick_scratch)
  call dealloc(quick_basis)

  if (allocated(itype)) deallocate(itype)
  if (allocated(ncontract)) deallocate(ncontract)
  if (allocated(aexp)) deallocate(aexp)
  if (allocated(dcoeff)) deallocate(dcoeff)
  if (allocated(gauss)) deallocate(gauss)

end subroutine deallocate_calculated

subroutine deallocateall(ierr)

  use allmod
  use quick_gridpoints_module

#ifdef CEW
  use quick_cew_module, only : quick_cew
#endif

  implicit none
  integer, intent(inout) :: ierr

    call  dealloc(quick_molspec,ierr)
    call  dealloc(quick_qm_struct)
    call  deallocate_calculated

    if (quick_method%DFT &
#ifdef CEW
      .or. quick_cew%use_cew &
#endif
    ) then
      call  deform_dft_grid(quick_dft_grid)
    endif

#ifdef MPIV
    call mpi_delete_atoms
#endif

end subroutine deallocateall


!----------------------
! Finialize programs
!----------------------
subroutine finalize(io,ierr,option)
    use allmod
    use quick_exception_module
    use quick_molden_module, only: finalizeExport, quick_molden
    implicit none
    integer io      !output final info and close this unit
    integer option  ! 0 if called from Quick and 1 if called from the API
    integer, intent(inout) :: ierr

    ! Deallocate all variables
    call deallocateall(ierr)

    ! stop timer and output them
    RECORD_TIME(timer_end%TTotal)
!    call timer_output(io)

    !-------------------MPI/MASTER---------------------------------------
    if (master) then

        ! finalize data exporting
        if(write_molden) call finalizeExport(quick_molden, ierr)

        if (ierr /=0) then
             call PrtDate(io,'Error Termination. Task Failed on:',ierr)
        endif
    endif

    if (ierr == 0) call timer_output(io)
       
    if (master) then
        if (ierr ==0) then
            call PrtDate(io,'Normal Termination. Task Finished on:',ierr)
        endif
    endif 
    !-------------------- End MPI/MASTER ---------------------------------

#ifdef MPIV
    !-------------------- MPI/ALL NODES ----------------------------------
    if (bMPI .and. option==0) call MPI_FINALIZE(mpierror)
    !-------------------- End MPI/ALL NODES-------------------------------
#endif

    close(io)

end subroutine finalize


!-----------------------
! Fatal exit subroutine
!-----------------------
subroutine quick_exit(io, ierr)

   use allmod
#ifdef MPIV
   use mpi
#endif
   implicit none
   integer io           ! close this unit if greater than zero
   integer, intent(inout) :: ierr


   if (ierr /= 0) then
      call flush(io)
   end if

   call finalize(io,ierr,1)

#ifdef MPIV
   if (ierr /= 0) then
     call mpi_abort(MPI_COMM_WORLD, ierr, mpierror)
   endif
#endif

   call exit(ierr)

end subroutine quick_exit

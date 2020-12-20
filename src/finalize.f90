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

subroutine deallocateall
  use allmod
  implicit double precision(a-h,o-z)

    call  dealloc(quick_molspec)
    call  dealloc(quick_qm_struct)
    call  deallocate_calculated
    
    if (quick_method%DFT) then
    call  deform_dft_grid(quick_dft_grid)
    endif



end subroutine deallocateall


!----------------------
! Finialize programs
!----------------------
subroutine finalize(io,status)
    use allmod
    implicit none
    integer io      !output final info and close this unit
    integer status  !exit status: 1-error 0-normal
    
    ! Deallocate all variables
    call deallocateall

#ifdef CUDA_MPIV
    call get_mgpu_time()
#endif
    ! stop timer and output them
    call cpu_time(timer_end%TTotal)
!    call timer_output(io)

    !-------------------MPI/MASTER---------------------------------------
    master_finalize:if (master) then
        if (status /=0) then
            call PrtDate(io,'Error Termination. Task Failed on:')
        else
            call timer_output(io)
            call PrtDate(io,'Normal Termination. Task Finished on:')
        endif
    endif master_finalize
    !-------------------- End MPI/MASTER ---------------------------------

#ifdef MPIV    
    !-------------------- MPI/ALL NODES ----------------------------------
    if (bMPI) call MPI_FINALIZE(mpierror)
    !-------------------- End MPI/ALL NODES-------------------------------
#endif

    close(io)
    
end subroutine finalize


!-----------------------
! Fatal exit subroutine 
!-----------------------
subroutine quick_exit(io, status)
   
   !  quick_exit: exit procedure, designed to return an gentle way to exitm.
   use allmod
   implicit none
   integer io           ! close this unit if greater than zero
   integer status       ! exit status; 1-error 0-normal

#ifdef MPIV
   include 'mpif.h'
#endif

   integer ierr

   if (status /= 0) then
      call flush(io)       
#ifdef MPIV
      call mpi_abort(MPI_COMM_WORLD, status, ierr)
   else
      call mpi_finalize(ierr)
#endif
   end if

   call finalize(io,1)
   
   call exit(status)

end subroutine quick_exit

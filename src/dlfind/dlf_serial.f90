!!!
!!! Dummy subroutines corresponding to MPI functionality.
!!!
!! COPYRIGHT
!!
!!  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Tom Keal (thomas.keal@stfc.ac.uk)
!!  Joanne Carr (j.m.carr@dl.ac.uk)
!!
!!  This file is part of DL-FIND.
!!
!!  DL-FIND is free software: you can redistribute it and/or modify
!!  it under the terms of the GNU Lesser General Public License as
!!  published by the Free Software Foundation, either version 3 of the
!!  License, or (at your option) any later version.
!!
!!  DL-FIND is distributed in the hope that it will be useful,
!!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!!  GNU Lesser General Public License for more details.
!!
!!  You should have received a copy of the GNU Lesser General Public
!!  License along with DL-FIND.  If not, see
!!  <http://www.gnu.org/licenses/>.
!!
subroutine dlf_mpi_initialize

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

! **********************************************************************

! Initialize the mpi-related variables that are used throughout the 
! code (not just within the wrapper MPI wourtines...)

  glob%nprocs = 1
  glob%ntasks = 1
  glob%nprocs_per_task = 1

  glob%iam = 0
  glob%mytask = 0
  glob%iam_in_task = 0
  !glob%master_in_task = 0

  return

end subroutine dlf_mpi_initialize


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_finalize

  implicit none

! **********************************************************************

  return

end subroutine dlf_mpi_finalize


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_abort

  implicit none

! **********************************************************************

  return

end subroutine dlf_mpi_abort


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_make_taskfarm(tdlf_farm)

  implicit none
  integer :: tdlf_farm

! **********************************************************************

  return

end subroutine dlf_make_taskfarm


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_counters

  implicit none

! **********************************************************************

  return

end subroutine dlf_mpi_counters


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_memory(sto,maxsto)

  implicit none
  integer :: sto,maxsto

! **********************************************************************

  return

end subroutine dlf_mpi_memory


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_time(descr, cput, wallt)
  use dlf_parameter_module, only: rk

  implicit none
  real(rk) :: cput,wallt
  character(len=*) :: descr

! **********************************************************************

  return

end subroutine dlf_mpi_time


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_bcast(a,n,iproc)

  use dlf_parameter_module, only: rk
  implicit none

  integer :: iproc, n
  real(rk), dimension(n) :: a
! **********************************************************************

  return

end subroutine dlf_global_real_bcast


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_bcast(a,n,iproc)

  use dlf_parameter_module, only: rk
  implicit none

  integer :: iproc, n
  integer, dimension(n) :: a
! **********************************************************************

  return

end subroutine dlf_global_int_bcast


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_bcast(a,n,iproc)

  use dlf_parameter_module, only: rk
  implicit none

  integer :: iproc, n
  logical, dimension(n) :: a
! **********************************************************************

  return

end subroutine dlf_global_log_bcast


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_sum(a,n)

  use dlf_parameter_module, only: rk
  implicit none

  integer :: n
  real(rk), dimension(n) :: a
! **********************************************************************

  return

end subroutine dlf_global_real_sum


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_tasks_real_sum(a,n)

  use dlf_parameter_module, only: rk
  implicit none

  integer :: n
  real(rk), dimension(n) :: a
! **********************************************************************

  return
  
end subroutine dlf_tasks_real_sum


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_tasks_int_sum(a,n)

  implicit none

  integer :: n
  integer, dimension(n) :: a
! **********************************************************************

  return

end subroutine dlf_tasks_int_sum


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_tasks_int_gather(a,n,b,m,iproc)

  use dlf_parameter_module, only: rk
  implicit none

  integer :: n, m, iproc
  integer, dimension(n) :: a
  integer, dimension(m) :: b
! **********************************************************************

  return

end subroutine dlf_tasks_int_gather


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_tasks_real_gather(a,n,b,m,iproc)

  use dlf_parameter_module, only: rk
  implicit none

  integer :: n, m, iproc
  real(rk), dimension(n) :: a
  real(rk), dimension(m) :: b
! **********************************************************************

  return

end subroutine dlf_tasks_real_gather


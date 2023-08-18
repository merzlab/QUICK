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


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_sum(a,n)
  implicit none

  integer :: n
  integer, dimension(n) :: a
! **********************************************************************

  return

end subroutine dlf_global_int_sum
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_sum_rank0(a)
  use dlf_parameter_module, only: rk
  implicit none

  real(rk) :: a
! **********************************************************************
  return

end subroutine dlf_global_real_sum_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_sum_rank0(a)
  implicit none

  integer :: a
! **********************************************************************
  return

end subroutine dlf_global_int_sum_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_bcast_rank0(a,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: iproc
  real(rk) :: a
! **********************************************************************

  return

end subroutine dlf_global_real_bcast_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_bcast_rank0(a,iproc)
  implicit none

  integer :: iproc
  integer :: a
! **********************************************************************

  return

end subroutine dlf_global_int_bcast_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_bcast_rank0(a,iproc)
  implicit none

  integer :: iproc
  logical :: a
! **********************************************************************

  return

end subroutine dlf_global_log_bcast_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_char_bcast_rank0(charvar,iproc)
  implicit none
  
  character(len=*) :: charvar
  integer :: iproc
! **********************************************************************
  return
end subroutine dlf_global_char_bcast_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_scatter_flat(aflat,n,b,m,iproc)
  use dlf_parameter_module, only: rk
  implicit none
  integer :: iproc, n, m
  real(rk), dimension(n*m) :: aflat
  real(rk), dimension(n) :: b
! **********************************************************************

  b(1:n)=aflat(1:n)
  return

end subroutine dlf_global_real_scatter_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_scatter_flat(aflat,n,b,m,iproc)
  implicit none

  integer :: iproc, n, m
  integer, dimension(n*m) :: aflat
  integer, dimension(n) :: b
! **********************************************************************

  b(1:n)=aflat(1:n)
  return

end subroutine dlf_global_int_scatter_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_scatter_flat(aflat,n,b,m,iproc)
  implicit none

  integer :: iproc, n, m
  logical, dimension(n*m) :: aflat
  logical, dimension(n) :: b
! **********************************************************************

  b(1:n)=aflat(1:n)
  return

end subroutine dlf_global_log_scatter_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_gather_flat(a,n,bflat,m,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: n, m, iproc
  real(rk), dimension(n) :: a
  real(rk), dimension(n*m) :: bflat
! **********************************************************************

  bflat(1:n)=a(1:n)
  return

end subroutine dlf_global_real_gather_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_gather_flat(a,n,bflat,m,iproc)
  implicit none

  integer :: n, m, iproc
  integer, dimension(n) :: a
  integer, dimension(n*m) :: bflat
! **********************************************************************

  bflat(1:n)=a(1:n)
  return

end subroutine dlf_global_int_gather_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_gather_flat(a,n,bflat,m,iproc)
  implicit none

  integer :: n, m, iproc
  logical, dimension(n) :: a
  logical, dimension(n*m) :: bflat
! **********************************************************************

  bflat(1:n)=a(1:n)
  return

end subroutine dlf_global_log_gather_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_char_gather_flat(a,n,bflat,m,iproc)
  implicit none

  integer :: n, m, iproc
  character(len=*), dimension(n) :: a
  character(len=*), dimension(n*m) :: bflat
! **********************************************************************

  bflat(1:n)=a(1:n)
  return

end subroutine dlf_global_char_gather_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_scatter_rank0(a,b,m,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: iproc, m
  real(rk), dimension(m) :: a
  real(rk) :: b
! **********************************************************************

  b=a(1)
  return

end subroutine dlf_global_real_scatter_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_scatter_rank0(a,b,m,iproc)
  implicit none

  integer :: iproc, m
  integer, dimension(m) :: a
  integer :: b
! **********************************************************************

  b=a(1)
  return

end subroutine dlf_global_int_scatter_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_scatter_rank0(a,b,m,iproc)
  implicit none

  integer :: iproc, m
  logical, dimension(m) :: a
  logical :: b
! **********************************************************************

  b=a(1)
  return

end subroutine dlf_global_log_scatter_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_gather_rank0(a,b,m,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer  :: m, iproc
  real(rk) :: a
  real(rk), dimension(m) :: b
! **********************************************************************

  b(1)=a
  return

end subroutine dlf_global_real_gather_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_gather_rank0(a,b,m,iproc)
  implicit none

  integer :: m, iproc
  integer :: a
  integer, dimension(m) :: b
! **********************************************************************

  b(1)=a
  return

end subroutine dlf_global_int_gather_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_gather_rank0(a,b,m,iproc)
  implicit none

  integer :: m, iproc
  logical :: a
  logical, dimension(m) :: b
! **********************************************************************

  b(1)=a
  return
  
end subroutine dlf_global_log_gather_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_char_gather_rank0(a,b,m,iproc)
  implicit none

  integer :: m, iproc
  character(len=*) :: a
  character(len=*), dimension(m) :: b
! **********************************************************************

  b(1)=a
  return

end subroutine dlf_global_char_gather_rank0

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_scatter_rank1(a,n,b,m,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: iproc, n, m
  
  real(rk), dimension(n,m) :: a
  real(rk), dimension(n) :: b
! **********************************************************************

  b(1:n)=a(1:n,1)
  return

end subroutine dlf_global_real_scatter_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_scatter_rank1(a,n,b,m,iproc)
  implicit none

  integer :: iproc, n, m
  
  integer, dimension(n,m) :: a
  integer, dimension(n) :: b
! **********************************************************************

  b(1:n)=a(1:n,1)
  return

end subroutine dlf_global_int_scatter_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_scatter_rank1(a,n,b,m,iproc)
  implicit none

  integer :: iproc, n, m
  
  logical, dimension(n,m) :: a
  logical, dimension(n) :: b
! **********************************************************************

  b(1:n)=a(1:n,1)
  return

end subroutine dlf_global_log_scatter_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_gather_rank1(a,n,b,m,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: n, m, iproc
  real(rk), dimension(n) :: a
  real(rk), dimension(n,m) :: b
! **********************************************************************

  b(1:n,1)=a(1:n)
  return

end subroutine dlf_global_real_gather_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_gather_rank1(a,n,b,m,iproc)
  implicit none

  integer :: n, m, iproc
  integer, dimension(n) :: a
  integer, dimension(n,m) :: b
! **********************************************************************

  b(1:n,1)=a(1:n)
  return

end subroutine dlf_global_int_gather_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_gather_rank1(a,n,b,m,iproc)
  implicit none

  integer :: n, m, iproc
  logical, dimension(n) :: a
  logical, dimension(n,m) :: b
! **********************************************************************

  b(1:n,1)=a(1:n)
  return

end subroutine dlf_global_log_gather_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_char_gather_rank1(a,n,b,m,iproc)
  implicit none

  integer :: n, m, iproc
  character(len=*), dimension(n) :: a
  character(len=*), dimension(n,m) :: b
! **********************************************************************

  b(1:n,1)=a(1:n)
  return

end subroutine dlf_global_char_gather_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_bcast_rank2(a,n1,n2,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: iproc, n1, n2
  
  real(rk), dimension(n1,n2) :: a
! **********************************************************************

  return

end subroutine dlf_global_real_bcast_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_bcast_rank2(a,n1,n2,iproc)
  implicit none

  integer :: iproc, n1, n2
  
  integer, dimension(n1,n2) :: a
! **********************************************************************

  return

end subroutine dlf_global_int_bcast_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_bcast_rank2(a,n1,n2,iproc)
  implicit none

  integer :: iproc, n1, n2
  
  logical, dimension(n1,n2) :: a
! **********************************************************************

  return

end subroutine dlf_global_log_bcast_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_scatter_rank2(a,n1,n2,b,m,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: iproc, n1, n2, m
  
  real(rk), dimension(n1,n2,m) :: a
  real(rk), dimension(n1,n2) :: b
! **********************************************************************

  b(1:n1,1:n2)=a(1:n1,1:n2,1)
  return

end subroutine dlf_global_real_scatter_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_scatter_rank2(a,n1,n2,b,m,iproc)
  implicit none

  integer :: iproc, n1, n2, m
  
  integer, dimension(n1,n2,m) :: a
  integer, dimension(n1,n2) :: b
! **********************************************************************

  b(1:n1,1:n2)=a(1:n1,1:n2,1)
  return

end subroutine dlf_global_int_scatter_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_scatter_rank2(a,n1,n2,b,m,iproc)
  implicit none

  integer :: iproc, n1, n2, m
  
  logical, dimension(n1,n2,m) :: a
  logical, dimension(n1,n2) :: b
! **********************************************************************

  b(1:n1,1:n2)=a(1:n1,1:n2,1)
  return

end subroutine dlf_global_log_scatter_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_gather_rank2(a,n1,n2,b,m,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: n1, n2, m, iproc
  real(rk), dimension(n1,n2) :: a
  real(rk), dimension(n1,n2,m) :: b
! **********************************************************************

  b(1:n1,1:n2,1)=a(1:n1,1:n2)
  return

end subroutine dlf_global_real_gather_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_gather_rank2(a,n1,n2,b,m,iproc)
  implicit none

  integer :: n1, n2, m, iproc
  integer, dimension(n1,n2) :: a
  integer, dimension(n1,n2,m) :: b
! **********************************************************************

  b(1:n1,1:n2,1)=a(1:n1,1:n2)
  return

end subroutine dlf_global_int_gather_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_gather_rank2(a,n1,n2,b,m,iproc)
  implicit none

  integer :: n1, n2, m, iproc
  logical, dimension(n1,n2) :: a
  logical, dimension(n1,n2,m) :: b
! **********************************************************************

  b(1:n1,1:n2,1)=a(1:n1,1:n2)
  return

end subroutine dlf_global_log_gather_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_sum_rank2(a,n1,n2)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: n1, n2
  real(rk), dimension(n1,n2)  :: a
! **********************************************************************

  return

end subroutine dlf_global_real_sum_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_sum_rank2(a,n1,n2)
  implicit none

  integer :: n1, n2
  integer, dimension(n1,n2)  :: a
! **********************************************************************

  return

end subroutine dlf_global_int_sum_rank2

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_scatter_rank3(a,n1,n2,n3,b,m,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: iproc, n1, n2, n3, m
  
  real(rk), dimension(n1,n2,n3,m) :: a
  real(rk), dimension(n1,n2,n3) :: b
! **********************************************************************

  b(1:n1,1:n2,1:n3)=a(1:n1,1:n2,1:n3,1)
  return

end subroutine dlf_global_real_scatter_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_scatter_rank3(a,n1,n2,n3,b,m,iproc)
  implicit none

  integer :: iproc, n1, n2, n3, m
  
  integer, dimension(n1,n2,n3,m) :: a
  integer, dimension(n1,n2,n3) :: b
! **********************************************************************

  b(1:n1,1:n2,1:n3)=a(1:n1,1:n2,1:n3,1)
  return

end subroutine dlf_global_int_scatter_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_scatter_rank3(a,n1,n2,n3,b,m,iproc)
  implicit none

  integer :: iproc, n1, n2, n3, m
  
  logical, dimension(n1,n2,n3,m) :: a
  logical, dimension(n1,n2,n3) :: b
! **********************************************************************

  b(1:n1,1:n2,1:n3)=a(1:n1,1:n2,1:n3,1)
  return

end subroutine dlf_global_log_scatter_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_gather_rank3(a,n1,n2,n3,b,m,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: n1, n2, n3, m, iproc
  real(rk), dimension(n1,n2,n3) :: a
  real(rk), dimension(n1,n2,n3,m) :: b
! **********************************************************************

  b(1:n1,1:n2,1:n3,1)=a(1:n1,1:n2,1:n3)
  return

end subroutine dlf_global_real_gather_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_gather_rank3(a,n1,n2,n3,b,m,iproc)
  implicit none

  integer :: n1, n2, n3, m, iproc
  integer, dimension(n1,n2,n3) :: a
  integer, dimension(n1,n2,n3,m) :: b
! **********************************************************************

  b(1:n1,1:n2,1:n3,1)=a(1:n1,1:n2,1:n3)
  return

end subroutine dlf_global_int_gather_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_gather_rank3(a,n1,n2,n3,b,m,iproc)
  implicit none

  integer :: n1, n2, n3, m, iproc
  logical, dimension(n1,n2,n3) :: a
  logical, dimension(n1,n2,n3,m) :: b
  integer :: n
! **********************************************************************

  b(1:n1,1:n2,1:n3,1)=a(1:n1,1:n2,1:n3)
  return

end subroutine dlf_global_log_gather_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_bcast_rank3(a,n1,n2,n3,iproc)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: iproc, n1, n2, n3
  
  real(rk), dimension(n1,n2,n3)  :: a
! **********************************************************************

  return

end subroutine dlf_global_real_bcast_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_bcast_rank3(a,n1,n2,n3,iproc)
  implicit none

  integer :: iproc, n1, n2, n3
  
  integer, dimension(n1,n2,n3)  :: a
! **********************************************************************

  return

end subroutine dlf_global_int_bcast_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_log_bcast_rank3(a,n1,n2,n3,iproc)
  implicit none

  integer :: iproc, n1, n2, n3
  
  logical, dimension(n1,n2,n3)  :: a
! **********************************************************************

  return

end subroutine dlf_global_log_bcast_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_real_sum_rank3(a,n1,n2,n3)
  use dlf_parameter_module, only: rk
  implicit none

  integer :: n1, n2, n3
  real(rk), dimension(n1,n2,n3)  :: a
! **********************************************************************

  return

end subroutine dlf_global_real_sum_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_global_int_sum_rank3(a,n1,n2,n3)
  implicit none

  integer :: n1, n2, n3
  integer, dimension(n1,n2,n3)  :: a
! **********************************************************************
  
  return
  
end subroutine dlf_global_int_sum_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_send_real_rank1(sendbuff,n,target_rank,tag)
  use dlf_parameter_module, only: rk
  implicit none
  integer, intent(in) :: n, target_rank, tag
  real(rk),dimension(n), intent(in) :: sendbuff
! **********************************************************************
  
  call dlf_fail("mpi_send was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")
  
end subroutine dlf_mpi_send_real_rank1
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_send_int_rank1(sendbuff,n,target_rank,tag)
  implicit none
  integer, intent(in) :: n, target_rank, tag
  integer,dimension(n), intent(in) :: sendbuff
! **********************************************************************

  call dlf_fail("mpi_send was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_send_int_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_send_log_rank1(sendbuff,n,target_rank,tag)
  implicit none
  integer, intent(in) :: n, target_rank, tag
  logical,dimension(n), intent(in) :: sendbuff
! **********************************************************************

  call dlf_fail("mpi_send was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_send_log_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_send_char_string(sendbuff,target_rank,tag)
  implicit none

  integer, intent(in) :: target_rank, tag
  character(len=*), intent(in) :: sendbuff
! **********************************************************************

  call dlf_fail("mpi_send was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_send_char_string
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_send_real_rank0(sendbuff,target_rank,tag)
  use dlf_parameter_module, only: rk
  implicit none
  integer, intent(in) :: target_rank, tag
  real(rk), intent(in) :: sendbuff
! **********************************************************************

  call dlf_fail("mpi_send was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_send_real_rank0
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_send_int_rank0(sendbuff,target_rank,tag)
  implicit none
  integer, intent(in) :: target_rank, tag
  integer, intent(in) :: sendbuff
! **********************************************************************

  call dlf_fail("mpi_send was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_send_int_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_send_log_rank0(sendbuff,target_rank,tag)
  implicit none
  integer, intent(in) :: target_rank, tag
  logical, intent(in) :: sendbuff
! **********************************************************************

  call dlf_fail("mpi_recv was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_send_log_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_recv_real_rank1(recvbuff,n,source_rank,tag,recv_status)
  use dlf_parameter_module, only: rk
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: source_rank, tag
  real(rk),dimension(n), intent(out) :: recvbuff
  integer, dimension(:), intent(out),optional :: recv_status
! **********************************************************************

  call dlf_fail("mpi_recv was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_recv_real_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_recv_int_rank1(recvbuff,n,source_rank,tag,recv_status)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: source_rank, tag
  integer,dimension(n), intent(out) :: recvbuff
  integer, dimension(:), intent(out),optional :: recv_status
! **********************************************************************

  call dlf_fail("mpi_recv was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_recv_int_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_recv_log_rank1(recvbuff,n,source_rank,tag,recv_status)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: source_rank, tag
  logical,dimension(n), intent(out) :: recvbuff
  integer, dimension(:), intent(out),optional :: recv_status
! **********************************************************************

  call dlf_fail("mpi_recv was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_recv_log_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_recv_char_string(recvbuff,source_rank,tag,recv_status)
  implicit none

  integer, intent(inout) :: source_rank, tag
  character(len=*), intent(out) :: recvbuff
  integer, dimension(:), intent(out),optional :: recv_status
! **********************************************************************

  call dlf_fail("mpi_recv was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_recv_char_string
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_recv_real_rank0(recvbuff,source_rank,tag,recv_status)
  use dlf_parameter_module, only: rk
  implicit none
  integer, intent(inout) :: source_rank, tag
  real(rk),intent(out) :: recvbuff
  integer, dimension(:), intent(out),optional :: recv_status
! **********************************************************************

  call dlf_fail("mpi_recv was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_recv_real_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_recv_int_rank0(recvbuff,source_rank,tag,recv_status)
  implicit none
  integer, intent(inout) :: source_rank, tag
  integer,intent(out) :: recvbuff
  integer, dimension(:), intent(out),optional :: recv_status
! **********************************************************************

  call dlf_fail("mpi_recv was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_recv_int_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_mpi_recv_log_rank0(recvbuff,source_rank,tag,recv_status)
  implicit none
  integer, intent(inout) :: source_rank, tag
  logical,intent(out) :: recvbuff
  integer, dimension(:), intent(out),optional :: recv_status
! **********************************************************************

  call dlf_fail("mpi_recv was called in a single-process environment, "// &
              &  " pointing to an implementation bug!")

end subroutine dlf_mpi_recv_log_rank0
!!****

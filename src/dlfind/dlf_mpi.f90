#ifdef MPIV
!!****h* utilities/dlf_mpi
!!
!! NAME
!! mpi
!!
!! FUNCTION
!! Wrappers and more for MPI functionality and taskfarming
!!
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
!! SOURCE
!!****
module dlf_mpi_module
  use mpi
  save

  integer  :: global_comm      ! set to, and use instead of, mpi_comm_world
  integer  :: task_comm        ! communicator for a taskfarm
  integer  :: ax_tasks_comm    ! communicator comprising one member
                               ! (``spokesperson'') from each taskfarm
  integer  :: mpi_rk ! set to mpi_double_precision in dlf_mpi_initialize
  integer  :: mpi_ik ! set to mpi_integer in dlf_mpi_initialize

  integer, parameter :: buff_size = 65536
! Since buffers are used in the reduction operations, don't send/receive 
! the whole array in a single chunk but use 1/2 megabyte segments instead
! (after subroutine global_gsum in CRYSTAL, which is the equivalent wrapper
! routine to this one).
! 1MB = 1048576 bytes; 8 bytes per element of a double precision array,
! so buff_size is 65536 or 256**2 (from 1048576/2/8).

end module dlf_mpi_module


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_initialize
!!
!! FUNCTION
!!
!! Deal with MPI initialization if the calling program has not already 
!! done this, and get processor info (rank, total number) in 
!! MPI_COMM_WORLD.  Exchange this information with the calling program.
!!
!! SYNOPSIS
subroutine dlf_mpi_initialize
!! SOURCE

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout
  use dlf_mpi_module
  implicit none

  integer :: ierr, colour, key
  logical :: tinitialized
  character(len=10) :: suffix

! **********************************************************************

  ! The line below is OK unless a compiler option is used to change the 
  ! default for double precision
  mpi_rk = mpi_double_precision
  mpi_ik = mpi_integer

! See if the program that calls dl-find has already called mpi_init
  call mpi_initialized(tinitialized,ierr)
  if (ierr /= 0) write(stdout,'(1x,a)') 'WARNING: MPI error in mpi_initialized'
  ! don't want to call dlf_fail here as that subroutine indirectly calls 
  ! mpi_abort currently!

  if (.not. tinitialized) then ! dl-find sorts out the initial MPI stuff

     call mpi_init(ierr)
     if (ierr /= 0) call dlf_fail("Failure in mpi_init")

     global_comm = mpi_comm_world

     call mpi_comm_rank(global_comm, glob%iam, ierr)
     if (ierr /= 0) call dlf_fail("Failure in mpi_comm_rank")

     call mpi_comm_size(global_comm, glob%nprocs, ierr)
     if (ierr /= 0) call dlf_fail("Failure in mpi_comm_size")

     ! Make this information accessible to the calling (energy and gradient)
     ! program, so that it can translate the variable names.
     call dlf_put_procinfo(glob%nprocs, glob%iam, global_comm)

     !! write some info on the parallelization to standard out
     !write(stdout,'(1x,a)')"In dlf_mpi_initialize: MPI initialization/setup &
     !                      &completed successfully"
     !write(stdout,'(1x,a,i10,a)')"I have rank ",glob%iam," in mpi_comm_world"
     !write(stdout,'(1x,a,i10)')"Total number of processors = ",glob%nprocs
     !if (keep_alloutput) then
     !   write(stdout,'(1x,a)')"Keeping output from all processors"
     !else
     !   write(stdout,'(1x,a)')"Not keeping output from processors /= 0"
     !end if

  else ! the calling program has already done a mpi_init
 
     call dlf_get_procinfo(glob%nprocs, glob%iam, global_comm)

  end if

end subroutine dlf_mpi_initialize
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_abort
!!
!! FUNCTION
!!
!! mpi_abort wrapper
!!
!! SYNOPSIS
subroutine dlf_mpi_abort
!! SOURCE

  use dlf_parameter_module, only: rk
  use dlf_global, only: stderr,stdout
  use dlf_mpi_module
  implicit none

  integer :: ierr

! **********************************************************************

  call mpi_abort(global_comm,1,ierr)
  ! the 1 is the errorcode, like call exit(1)
  if (ierr /= 0) write(stdout,'(1x,a)') "Failure in mpi_abort"

end subroutine dlf_mpi_abort
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_finalize
!!
!! FUNCTION
!!
!! mpi_finalize wrapper.  Only makes the call if mpi_initialized returns 
!! .true.
!!
!! SYNOPSIS
subroutine dlf_mpi_finalize
!! SOURCE

  use dlf_global, only: stdout
  use dlf_mpi_module
  implicit none

  integer :: ierr
  logical :: tinitialized

! **********************************************************************

! First check whether mpi is still running...
  call mpi_initialized(tinitialized,ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_initialized in dlf_mpi_finalize")

  if (tinitialized) then
     call mpi_finalize(ierr)
     if (ierr /= 0) call dlf_fail("Failure in mpi_finalize")
  else
     write(stdout,'(a)') " Not calling MPI_Finalize in dlf_mpi_finalize"
     write(stdout,'(a)') " as MPI is no longer running"
  end if

end subroutine dlf_mpi_finalize
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_make_taskfarm
!!
!! FUNCTION
!!
!! Either set up taskfarms and send the relevant information back to the 
!! calling program via the dlf_put_taskfarm routine, or get existing 
!! taskfarm info via dlf_get_taskfarm. Do the latter if tdlf_farm == 0 
!! Taskfarming is based on split communicators.
!!
!! SYNOPSIS
subroutine dlf_make_taskfarm(tdlf_farm)
!! SOURCE

! Note that this subroutine must be passed through even if glob%ntasks = 1,
! to set up the appropriate communicators
! Also note that dlf_get_taskfarm is called from here, if the calling 
! program has set up the farms already.

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout
  use dlf_mpi_module
  implicit none

  integer :: ierr, colour, key
  integer :: tdlf_farm

! **********************************************************************

  !glob%ntasks = glob%nprocs !!! for testing!!!

  if (tdlf_farm == 0) then
     call dlf_get_taskfarm(glob%ntasks, glob%nprocs_per_task, &
          glob%iam_in_task, glob%mytask, task_comm, ax_tasks_comm)
  else
     if (mod(glob%nprocs, glob%ntasks) /= 0) call dlf_fail("Number of &
                        &processors not divisible by number of workgroups requested")

     glob%nprocs_per_task = glob%nprocs / glob%ntasks

! uses integer division to assign the first glob%nprocs_per_task procs to
! task farm 0, the second set to task farm 1, and so on up to glob%ntasks-1.

     colour = glob%iam / glob%nprocs_per_task

! uses mod function to calculate the rank of each processor with respect
! to its task farm, starting from 0.

     key = mod(glob%iam, glob%nprocs_per_task)

! create a new communicator for each task farm (so only processors in the same
! farm can talk to each other on this communicator)

     call mpi_comm_split(global_comm, colour, key, task_comm, ierr)

     glob%iam_in_task = key
     glob%mytask = colour
     !glob%master_in_task = 0

! create another set of communicators, each of which comprises the ith-ranked 
! processor from each task farm; a simple way of doing this is by swapping the roles of
! colour and key used above.

     call mpi_comm_split(global_comm, key, colour, ax_tasks_comm, ierr)

! Make the taskfarm information accessible to the calling (energy and gradient) 
! program, so that it can translate the variable names.

     call dlf_put_taskfarm(glob%ntasks, glob%nprocs_per_task, glob%iam_in_task,&
                           glob%mytask, task_comm, ax_tasks_comm)
  end if

! Write some debug info about the taskfarm to standard out

!  write(stdout,'(1x,a,i10,a)')"In dlf_make_taskfarm: there are ",glob%ntasks,&
!                              " workgroups"
!  write(stdout,'(1x,a,i10)')"My workgroup is indexed ",glob%mytask
!  write(stdout,'(1x,a,i10,a)')"I have rank ",glob%iam_in_task," in my workgroup"
!  write(stdout,'(1x,a,i10,a)')"There are ",glob%nprocs_per_task," processors &
!                              &in my workgroup"

end subroutine dlf_make_taskfarm
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_counters
!!
!! FUNCTION
!!
!! Calculate some information on the parallel load balancing for dlf_report. 
!! Total, mean, minimum and maximum and standard deviation of the number of 
!! energy evaluations per processor.
!! Only one contribution to the statistics per taskfarm.
!!
!! INPUTS
!!
!! stat%pene
!! glob%ntasks
!! glob%nprocs
!! glob%iam
!! glob%iam_in_task
!!
!! SYNOPSIS
subroutine dlf_mpi_counters
!! SOURCE

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_stat, only: stat
  use dlf_mpi_module
  implicit none

  integer :: iproc, n, ierr
  integer :: total_count,lowest_count,highest_count
  integer, dimension(glob%ntasks) :: ene_count
  real(rk) :: mean_count,mean_count2,sigma_count
! **********************************************************************

  if (glob%ntasks == 1 .or. glob%nprocs == 1) return

! First do an mpi_gather to get the pene's from all procs
! Note this is not necessarily equal to stat%sene as some energy
! evaluations may be duplicated across workgroups

  iproc = -1
  if (glob%iam == 0) iproc = glob%iam_in_task
  call dlf_global_int_bcast(iproc, 1, 0)

  ene_count(:) = 0
  call dlf_tasks_int_gather(stat%pene, 1, ene_count, glob%ntasks, iproc)

! Now calculate the stats and do some writing
  if (glob%iam == 0 .and. printl > 0) then

     total_count = sum(ene_count(:))
     mean_count = dble(total_count) / dble(glob%ntasks)
     mean_count2 = dble(sum(ene_count(:)**2)) / dble(glob%ntasks)
     sigma_count = sqrt(mean_count2 - mean_count**2)
     lowest_count = minval(ene_count)
     highest_count = maxval(ene_count)

     write(stdout,'(a)')"Task-farming statistics:"
     write(stdout,2000) "Number of workgroups", glob%ntasks

     if (stat%ccycle > 0) then
        write(stdout,'(a)')"Evaluations of the total energy"
        write(stdout,2000) "Total number (incl. duplicates)",total_count
        write(stdout,2000) "Mean number",nint(mean_count)
        write(stdout,2000) "Standard deviation",nint(sigma_count)
        write(stdout,2000) "Lowest number per workgroup",lowest_count
        write(stdout,2000) "Highest number per workgroup",highest_count
        write(stdout,2000) "Total number (excl. duplicates)",stat%sene
     end if

     write(stdout,'(a)')"Task-farming statistics: end"

  end if  

  ! real number
  1000 format (t1,'................................................', &
           t1,a,' ',t50,es10.3,1x,a)
  ! integer
  2000 format (t1,'................................................', &
           t1,a,' ',t50,i10,1x,a)

end subroutine dlf_mpi_counters
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_memory
!!
!! FUNCTION
!!
!! Summarise memory use across taskfarms: calculate total, mean, 
!! standard deviation, maximum and minimum.
!!
!! INPUTS
!!
!! local vars
!! glob%ntasks
!! glob%nprocs
!! glob%iam
!! glob%iam_in_task
!!
!! SYNOPSIS
subroutine dlf_mpi_memory(sto,maxsto)
!! SOURCE

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout
  use dlf_mpi_module
  implicit none

  integer :: sto, maxsto
  integer :: iproc, ierr
  integer :: total_stored,lowest_stored,highest_stored
  integer, dimension(glob%ntasks) :: stored
  real(rk) :: mean_stored,mean_stored2,sigma_stored
! **********************************************************************

  if (glob%ntasks == 1 .or. glob%nprocs == 1) return

! First do an mpi_gather to get the sto's from all procs

  iproc = -1
  if (glob%iam == 0) iproc = glob%iam_in_task
  call dlf_global_int_bcast(iproc, 1, 0)

  stored(:) = 0
  call dlf_tasks_int_gather(sto, 1, stored, glob%ntasks, iproc)

  if (minval(stored) > 0 .and. glob%iam == 0) then
     total_stored = sum(stored(:))
     mean_stored = dble(total_stored) / dble(glob%ntasks)
     mean_stored2 = dble(sum(stored(:)**2)) / dble(glob%ntasks)
     sigma_stored = sqrt(mean_stored2 - mean_stored**2)
     lowest_stored = minval(stored)
     highest_stored = maxval(stored)

     write(stdout,'(a)')"Task-farming statistics:"
     write(stdout,'(a)')"Current memory usage (/kB)"
     write(stdout,1000) "Total",dble(total_stored)/1024.D0
     write(stdout,1000) "Mean",mean_stored/1024.D0
     write(stdout,1000) "Standard deviation",sigma_stored/1024.D0
     write(stdout,1000) "Lowest",dble(lowest_stored)/1024.D0
     write(stdout,1000) "Highest",dble(highest_stored)/1024.D0
     write(stdout,'(a)')"Task-farming statistics: end"

  end if

  stored(:) = 0
  call dlf_tasks_int_gather(maxsto, 1, stored, glob%ntasks, iproc)

  if (glob%iam == 0) then
     total_stored = sum(stored(:))
     mean_stored = dble(total_stored) / dble(glob%ntasks)
     mean_stored2 = dble(sum(stored(:)**2)) / dble(glob%ntasks)
     sigma_stored = sqrt(mean_stored2 - mean_stored**2)
     lowest_stored = minval(stored)
     highest_stored = maxval(stored)

     write(stdout,'(a)')"Task-farming statistics:"
     write(stdout,'(a)')"Maximum memory usage (/kB):"
     write(stdout,1000) "Total",dble(total_stored)/1024.D0
     write(stdout,1000) "Mean",mean_stored/1024.D0
     write(stdout,1000) "Standard deviation",sigma_stored/1024.D0
     write(stdout,1000) "Lowest",dble(lowest_stored)/1024.D0
     write(stdout,1000) "Highest",dble(highest_stored)/1024.D0
     write(stdout,'(a)')"Task-farming statistics: end"

  end if

    ! real number
1000 format (t1,'................................................', &
         t1,a,' ',t50,es10.4,1x,a)

end subroutine dlf_mpi_memory
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_time
!!
!! FUNCTION
!!
!! Summarise CPU and wallclock timings across taskfarms: calculate total, 
!! mean, standard deviation, maximum and minimum, as part of time_report.
!!
!! INPUTS
!!
!! local vars
!! glob%ntasks
!! glob%nprocs
!! glob%iam
!! glob%iam_in_task
!!
!! SYNOPSIS
subroutine dlf_mpi_time(descr, cput, wallt)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_mpi_module

  implicit none
  integer  :: iproc, n, ierr
  real(rk) :: cput,wallt
  real(rk) :: total_cput,lowest_cput,highest_cput
  real(rk) :: mean_cput,mean_cput2,sigma_cput
  real(rk) :: total_wallt,lowest_wallt,highest_wallt
  real(rk) :: mean_wallt,mean_wallt2,sigma_wallt
  real(rk), dimension(glob%ntasks) :: cputime, walltime
  character(len=*) :: descr
! **********************************************************************

  if (glob%nprocs == 1 .or. glob%ntasks == 1) return

! First do two mpi_gathers to get the cput's and wallt's from all procs

  iproc = -1
  if (glob%iam == 0) iproc = glob%iam_in_task
  call dlf_global_int_bcast(iproc, 1, 0)

  cputime(:) = 0.0D0
  walltime(:) = 0.0D0
  call dlf_tasks_real_gather(cput, 1, cputime, glob%ntasks, iproc)
  call dlf_tasks_real_gather(wallt, 1, walltime, glob%ntasks, iproc)

  if (glob%iam == 0 .and. printl > 0) then
     total_cput = sum(cputime(:))
     mean_cput = total_cput / dble(glob%ntasks)
     mean_cput2 = sum(cputime(:)**2) / dble(glob%ntasks)
     sigma_cput = sqrt(mean_cput2 - mean_cput**2)
     lowest_cput = minval(cputime)
     highest_cput = maxval(cputime)

     total_wallt = sum(walltime(:))
     mean_wallt = total_wallt / dble(glob%ntasks)
     mean_wallt2 = sum(walltime(:)**2) / dble(glob%ntasks)
     sigma_wallt = sqrt(mean_wallt2 - mean_wallt**2)
     lowest_wallt = minval(walltime)
     highest_wallt = maxval(walltime)

     write(stdout,*)
     write(stdout,1000) trim(adjustl(descr))//" CPU /s", total_cput, mean_cput,&
                        sigma_cput, lowest_cput, highest_cput
     write(stdout,1000) trim(adjustl(descr))//" Wall /s", total_wallt, &
                        mean_wallt, sigma_wallt, lowest_wallt, highest_wallt
  end if

  ! real number
  1000 format (t1,"................................................", &
         t1,a," ",t40,5f10.3)


end subroutine dlf_mpi_time
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_bcast
!!
!! FUNCTION
!!
!! broadcasts the real data in the dummy argument array on the selected root 
!! processor to all other processors via mpi_bcast.  This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! global_comm
!! mpi_rk
!!
!! SYNOPSIS
subroutine dlf_global_real_bcast(a,n,iproc)
!! SOURCE

! broadcasts the real(rk) data in array a on processor iproc
! to all other processors via mpi_bcast.  This routine is blocking.

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: iproc, n, ierr
  real(rk), dimension(n) :: a
! **********************************************************************

  if (glob%nprocs == 1) return
  if (n <= 0) call dlf_fail("dlf_global_real_bcast called with n <= 0")

  call mpi_bcast(a,n,mpi_rk,iproc,global_comm,ierr)

  if (ierr /= 0) call dlf_fail("Failure in dlf_global_real_bcast")

end subroutine dlf_global_real_bcast
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_bcast
!!
!! FUNCTION
!!
!! broadcasts the integer data in the dummy argument array on the selected root
!! processor to all other processors via mpi_bcast.  This routine is blocking.
!!
!! INPUTS
!!
!! local variables
!! glob%nprocs
!! global_comm
!! mpi_ik
!!
!! SYNOPSIS
subroutine dlf_global_int_bcast(a,n,iproc)
!! SOURCE

! broadcasts the integer data in array a on processor iproc
! to all other processors via mpi_bcast.  This routine is blocking.

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: iproc, n, ierr
  integer, dimension(n) :: a
! **********************************************************************

  if (glob%nprocs == 1) return
  if (n <= 0) call dlf_fail("dlf_global_int_bcast called with n <= 0")

  call mpi_bcast(a,n,mpi_ik,iproc,global_comm,ierr)

  if (ierr /= 0) call dlf_fail("Failure in dlf_global_int_bcast")

end subroutine dlf_global_int_bcast
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_bcast
!!
!! FUNCTION
!!
!! broadcasts the logical data in the dummy argument array on the selected root
!! processor to all other processors via mpi_bcast.  This routine is blocking.
!!
!! INPUTS
!!
!! local variables
!! glob%nprocs
!! global_comm
!!
!! SYNOPSIS
subroutine dlf_global_log_bcast(a,n,iproc)
!! SOURCE

! broadcasts the logical data in array a on processor iproc
! to all other processors via mpi_bcast.  This routine is blocking.

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: iproc, n, ierr
  logical, dimension(n) :: a
! **********************************************************************

  if (glob%nprocs == 1) return
  if (n <= 0) call dlf_fail("dlf_global_log_bcast called with n <= 0")

  call mpi_bcast(a,n,mpi_logical,iproc,global_comm,ierr)

  if (ierr /= 0) call dlf_fail("Failure in dlf_global_log_bcast")

end subroutine dlf_global_log_bcast
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_sum
!!
!! FUNCTION
!!
!! Does an mpi_allreduce with the mpi_sum operation on the real data stored 
!! partially in the dummy argument array on each processor.  The result (the 
!! completed array) is known on all processors.  This routine is blocking.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! mpi_rk
!! global_comm
!!
!! SYNOPSIS
subroutine dlf_global_real_sum(a,n)
!! SOURCE

! Does an mpi_allreduce with the mpi_sum operation on the real(rk) 
! data stored partially in array a on each processor.  The result (the 
! completed array a) is known on all processors.  This routine is blocking.

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: n, ierr, length, start, finish
  real(rk), dimension(n) :: a
  real(rk), dimension(buff_size) :: buff
! **********************************************************************

  if (glob%nprocs == 1) return
  if (n <= 0) call dlf_fail("dlf_global_real_sum called with n <= 0")

  start = 0
  do
     if (start >= n) exit 
     length = min(buff_size, n - start)
     call mpi_allreduce(a(start+1), buff(1:length), length, mpi_rk, mpi_sum, &
                      & global_comm, ierr)
     if (ierr /= 0) call dlf_fail("Failure in dlf_global_real_sum")
     finish = start + length
     a(start+1 : finish) = buff(:length)
     start = finish
  end do

end subroutine dlf_global_real_sum
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_tasks_real_sum
!!
!! FUNCTION
!!
!! Does an mpi_allreduce with the mpi_sum operation on the real data stored
!! partially in the dummy argument array on one processor from each taskfarm,
!! for each group consisting of the processors with the same task-farm rank.
!! Assumes that the arrays on each processor in a particular taskfarm are 
!! identical.
!! The result (the completed array) is known on all the processors involved.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local vars
!! glob%ntasks
!! glob%nprocs
!! mpi_rk
!! ax_tasks_comm
!! buff_size
!!
!! SYNOPSIS
subroutine dlf_tasks_real_sum(a,n)
!! SOURCE

! Does an mpi_allreduce with the mpi_sum operation on the real(rk)
! data stored partially in array a on one processor from each taskfarm, 
! for each group consisting of the processors with the same task-farm rank.
! Assumes that each processor in a particular taskfarm has identical 
! a arrays.! -- nothing is special about the one chosen here.
! The result (the completed array a) is known on all the procs involved.!, 
!! so an mpi_bcast is called with the communicator for a task-farm to make 
!! a known on all processors.  Both routines are blocking.

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: n, ierr, length, start, finish
  real(rk), dimension(n) :: a
  real(rk), dimension(buff_size) :: buff
! **********************************************************************

  if (glob%ntasks == 1 .or. glob%nprocs == 1) return
  if (n <= 0) call dlf_fail("dlf_tasks_real_sum called with n <= 0")

  start = 0
  do
     if (start >= n) exit
     length = min(buff_size, n - start)
     call mpi_allreduce(a(start+1), buff(1:length), length, mpi_rk, mpi_sum, &
                      & ax_tasks_comm, ierr)
     if (ierr /= 0) call dlf_fail("Failure in mpi_allreduce in&
                                & dlf_tasks_real_sum")
     finish = start + length
     a(start+1 : finish) = buff(:length)
     start = finish
  end do

!  call mpi_bcast(a,n,mpi_rk,glob%master_in_task,task_comm,ierr)
!  if (ierr /= 0) call dlf_fail("Failure in mpi_bcast in dlf_tasks_real_sum")
  
end subroutine dlf_tasks_real_sum
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_tasks_int_sum
!!
!! FUNCTION
!!
!! Does an mpi_allreduce with the mpi_sum operation on the integer data stored
!! partially in the dummy argument array on one processor from each taskfarm,  
!! for each group consisting of the processors with the same task-farm rank.
!! Assumes that the arrays on each processor in a particular taskfarm are 
!! identical.
!! The result (the completed array) is known on all the processors involved.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local vars
!! glob%ntasks
!! glob%nprocs
!! mpi_ik
!! ax_tasks_comm
!! buff_size
!!
!! SYNOPSIS
subroutine dlf_tasks_int_sum(a,n)
!! SOURCE

! Does an mpi_allreduce with the mpi_sum operation on the integer 
! data stored partially in array a on one processor from each taskfarm,
! for each group consisting of the processors with the same task-farm rank.
! Assumes that each processor in a particular taskfarm has identical
! a arrays.! -- nothing is special about the one chosen here.
! The result (the completed array a) is known on all the procs involved.!,
!! so an mpi_bcast is called with the communicator for a task-farm to make
!! a known on all processors.  Both routines are blocking.

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: n, ierr, length, start, finish
  integer, dimension(n) :: a
  integer, dimension(buff_size) :: buff
! **********************************************************************

  if (glob%ntasks == 1 .or. glob%nprocs == 1) return
  if (n <= 0) call dlf_fail("dlf_tasks_int_sum called with n <= 0")

  start = 0
  do
     if (start >= n) exit
     length = min(buff_size, n - start)
     call mpi_allreduce(a(start+1), buff(1:length), length, mpi_ik, mpi_sum, &
                      & ax_tasks_comm, ierr)
     if (ierr /= 0) call dlf_fail("Failure in mpi_allreduce in&
                                & dlf_tasks_int_sum")
     finish = start + length
     a(start+1 : finish) = buff(:length)
     start = finish
  end do

!  call mpi_bcast(a,n,mpi_rk,glob%master_in_task,task_comm,ierr)
!  if (ierr /= 0) call dlf_fail("Failure in mpi_bcast in dlf_tasks_real_sum")

end subroutine dlf_tasks_int_sum
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_tasks_int_gather
!!
!! FUNCTION
!!
!! Does an mpi_gather on the items of integer data in the dummy argument 
!! array on one procesor from 
!! each taskfarm, for each group consisting of the processors with the 
!! the same taskfarm rank.  Assumes that each processor in a particular 
!! taskfarm has identical dummy argument arrays.
!! The resulting dummy argument array is only known on the root processor.
!!
!! INPUTS
!!
!! local vars
!! glob%ntasks
!! glob%nprocs
!! mpi_ik
!! ax_tasks_comm
!!
!! SYNOPSIS
subroutine dlf_tasks_int_gather(a,n,b,m,iproc)
!! SOURCE

! Does an mpi_gather on the n items of integer data in array a on one 
! procesor from
! each taskfarm, for each group consisting of the processors with the 
! the same task-farm rank.  Assumes that each processor in a particular 
! taskfarm has identical a arrays.
! The result (array b, of dimension m = n*glob%ntasks) is only known on 
! the root processor, iproc

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: n, m, ierr, iproc
  integer, dimension(n) :: a
  integer, dimension(m) :: b
! **********************************************************************

  if (glob%ntasks == 1 .or. glob%nprocs == 1) then
      if (n /= m) call dlf_fail("dlf_tasks_int_gather called with n /= m")
      b(:) = a(:)
      return
  end if

  if (n <= 0) call dlf_fail("dlf_tasks_int_gather called with n <= 0")
  if (m <= 0) call dlf_fail("dlf_tasks_int_gather called with m <= 0")

  call mpi_gather(a, n, mpi_ik, b, n, mpi_ik, iproc, ax_tasks_comm, ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_gather in dlf_tasks_int_gather")

end subroutine dlf_tasks_int_gather
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_tasks_real_gather
!!
!! FUNCTION
!!
!! Does an mpi_gather on the items of real data in the dummy argument
!! array on one procesor from
!! each taskfarm, for each group consisting of the processors with the
!! the same taskfarm rank.  Assumes that each processor in a particular
!! taskfarm has identical dummy argument arrays.
!! The resulting dummy argument array is only known on the root processor.
!!
!! INPUTS
!!
!! local vars
!! glob%ntasks
!! glob%nprocs
!! mpi_rk
!! ax_tasks_comm
!!
!! SYNOPSIS
subroutine dlf_tasks_real_gather(a,n,b,m,iproc)
!! SOURCE

! Does an mpi_gather on the n items of real data in array a on one 
! procesor from
! each taskfarm, for each group consisting of the processors with the
! the same task-farm rank.  Assumes that each processor in a particular
! taskfarm has identical a arrays.
! The result (array b, of dimension m = n*glob%ntasks) is only known on 
! the root processor, iproc

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: n, m, ierr, iproc
  real(rk), dimension(n) :: a
  real(rk), dimension(m) :: b
! **********************************************************************

  if (glob%ntasks == 1 .or. glob%nprocs == 1) then
      if (n /= m) call dlf_fail("dlf_tasks_real_gather called with n /= m")
      b(:) = a(:)
      return
  end if

  if (n <= 0) call dlf_fail("dlf_tasks_real_gather called with n <= 0")
  if (m <= 0) call dlf_fail("dlf_tasks_real_gather called with m <= 0")

  call mpi_gather(a, n, mpi_rk, b, n, mpi_rk, iproc, ax_tasks_comm, ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_gather in dlf_tasks_real_gather")

end subroutine dlf_tasks_real_gather
!!****
#endif

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
!   include 'mpif.h' !old f77 bindings!
  include 'mpif.h'
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

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_sum
!!
!! FUNCTION
!!
!! Does an mpi_allreduce with the mpi_sum operation on the integer data stored 
!! partially in the dummy argument array on each processor.  The result (the 
!! completed array) is known on all processors.  This routine is blocking.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! mpi_ik
!! global_comm
!!
!! SYNOPSIS
subroutine dlf_global_int_sum(a,n)
!! SOURCE

! Does an mpi_allreduce with the mpi_sum operation on the integer
! data stored partially in array a on each processor.  The result (the 
! completed array a) is known on all processors.  This routine is blocking.

  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: n, ierr, length, start, finish
  integer, dimension(n) :: a
  integer, dimension(buff_size) :: buff
! **********************************************************************

  if (glob%nprocs == 1) return
  if (n <= 0) call dlf_fail("dlf_global_int_sum called with n <= 0")

  start = 0
  do
     if (start >= n) exit 
     length = min(buff_size, n - start)
     call mpi_allreduce(a(start+1), buff(1:length), length, mpi_ik, mpi_sum, &
                      & global_comm, ierr)
     if (ierr /= 0) call dlf_fail("Failure in dlf_global_int_sum")
     finish = start + length
     a(start+1 : finish) = buff(:length)
     start = finish
  end do

end subroutine dlf_global_int_sum
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_sum_rank0
!!
!! FUNCTION
!!
!! Does an mpi_allreduce with the mpi_sum operation on the real data stored 
!! partially in the dummy argument scalar 'a' on each processor.  The result 
!! (the completed scalar) is known on all processors.  This routine is blocking.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_real_sum_rank0(a)
!! SOURCE
  use, intrinsic :: ISO_C_BINDING
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  real(rk),target :: a
  real(rk),pointer,dimension(:),contiguous :: ap
! **********************************************************************
  if (glob%nprocs == 1) return
  nullify(ap)
  call c_f_pointer( c_loc(a), ap, [1] )
  call dlf_global_real_sum(ap,1)
  nullify(ap)

end subroutine dlf_global_real_sum_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_sum_rank0
!!
!! FUNCTION
!!
!! Does an mpi_allreduce with the mpi_sum operation on the integer data stored 
!! partially in the dummy argument scalar 'a' on each processor.  The result 
!! (the completed scalar) is known on all processors.  This routine is blocking.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_int_sum_rank0(a)
!! SOURCE
  use, intrinsic :: ISO_C_BINDING
  use dlf_global, only: glob
  implicit none

  integer,target :: a
  integer,pointer,dimension(:),contiguous :: ap
! **********************************************************************
  if (glob%nprocs == 1) return
  nullify(ap)
  call c_f_pointer( c_loc(a), ap, [1] )
  call dlf_global_int_sum(ap,1)
  nullify(ap)

end subroutine dlf_global_int_sum_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_bcast_rank0
!!
!! FUNCTION
!!
!! broadcasts the real data in the scalar dummy argument on the selected root
!! processor to all other processors via mpi_bcast.  This routine is blocking.
!!
!! INPUTS
!!
!! local variables
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_real_bcast_rank0(a,iproc)
!! SOURCE

! broadcasts the scalar real data in 'a' on processor iproc
! to all other processors via mpi_bcast.  This routine is blocking.

  use, intrinsic :: ISO_C_BINDING
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: iproc
  real(rk),target :: a
  real(rk),pointer,dimension(:),contiguous :: ap
! **********************************************************************

  if (glob%nprocs == 1) return
  nullify(ap)
  call c_f_pointer( c_loc(a), ap, [1] )
  call dlf_global_real_bcast(ap,1,iproc)
  nullify(ap)
  
end subroutine dlf_global_real_bcast_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_bcast_rank0
!!
!! FUNCTION
!!
!! broadcasts the integer data in the scalar dummy argument on the selected root
!! processor to all other processors via mpi_bcast.  This routine is blocking.
!!
!! INPUTS
!!
!! local variables
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_int_bcast_rank0(a,iproc)
!! SOURCE

! broadcasts the scalar integer data in a on processor iproc
! to all other processors via mpi_bcast.  This routine is blocking.

  use, intrinsic :: ISO_C_BINDING
  use dlf_global, only: glob
  implicit none

  integer :: iproc
  integer,target :: a
  integer,pointer,dimension(:),contiguous :: ap
! **********************************************************************

  if (glob%nprocs == 1) return
  nullify(ap)
  call c_f_pointer( c_loc(a), ap, [1] )
  call dlf_global_int_bcast(ap,1,iproc)
  nullify(ap)
  
end subroutine dlf_global_int_bcast_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_bcast_rank0
!!
!! FUNCTION
!!
!! broadcasts the logical data in the scalar dummy argument on the selected root
!! processor to all other processors via mpi_bcast.  This routine is blocking.
!!
!! INPUTS
!!
!! local variables
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_log_bcast_rank0(a,iproc)
!! SOURCE

! broadcasts the scalar logical data in a on processor iproc
! to all other processors via mpi_bcast.  This routine is blocking.

  use, intrinsic :: ISO_C_BINDING
  use dlf_global, only: glob
  implicit none

  integer :: iproc
  logical,target :: a
  logical,pointer,dimension(:),contiguous :: ap
! **********************************************************************

  if (glob%nprocs == 1) return
  nullify(ap)
  call c_f_pointer( c_loc(a), ap, [1] )
  call dlf_global_log_bcast(ap,1,iproc)
  nullify(ap)
  
end subroutine dlf_global_log_bcast_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_char_bcast_rank0
!!
!! FUNCTION
!!
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! global_comm
!! mpi_character
!!
!! SYNOPSIS
subroutine dlf_global_char_bcast_rank0(charvar,iproc)
!! SOURCE

! scatters the real(rk) data in array aflat on processor iproc
! to array b on all processors via mpi_scatter.  This routine is blocking.

  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none
  
  character(len=*) :: charvar
  integer :: iproc, charlength, ierr
! **********************************************************************
  if (glob%nprocs == 1) then
    return
  endif
  
  charlength=len(charvar)
  
  call mpi_bcast(charvar,charlength,mpi_character,iproc,global_comm,ierr)
  
  if (ierr /= 0) call dlf_fail("Failure in dlf_global_char_bcast_rank0")

end subroutine dlf_global_char_bcast_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_scatter_flat
!!
!! FUNCTION
!!
!! Scatters the real data in the flattened dummy argument array aflat (array 
!! rank 1, length n*m) on the selected root processor iproc to the dummy arg.
!! array b (array rank 1, length n) on all processors, including iproc, 
!! via mpi_scatter. m must be equal to the number of processors for this to 
!! make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! aflat must be of size 1:n*m.
!! If called by any process other than the 'send process' iproc, aflat 
!! must be a dummy array with length 0. That way, it is avoided that the 
!! therein unused array aflat has to be allocated with size n*m on every 
!! single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! global_comm
!! mpi_rk
!!
!! SYNOPSIS
subroutine dlf_global_real_scatter_flat(aflat,n,b,m,iproc)
!! SOURCE

! scatters the real(rk) data in array aflat on processor iproc
! to array b on all processors via mpi_scatter.  This routine is blocking.

  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: iproc, n, m, ierr
  real(rk), dimension(merge(n*m,0,glob%iam==iproc)) :: aflat
  real(rk), dimension(n) :: b
! **********************************************************************
  if (m /= glob%nprocs) call dlf_fail("dlf_global_real_scatter_flat must be called with m=nprocs")
  if (glob%nprocs == 1) then
    b(1:n)=aflat(1:n)
    return
  endif
  if (glob%iam==iproc) then
    if (size(aflat)/=n*m) call dlf_fail("dlf_global_real_scatter_flat: wrong size of dummy argument aflat")
  endif
  if (n <= 0) call dlf_fail("dlf_global_real_scatter_flat called with n <= 0")
  call mpi_scatter(aflat,n,mpi_rk,b,n,mpi_rk,iproc,global_comm,ierr)

  if (ierr /= 0) call dlf_fail("Failure in dlf_global_real_scatter_flat")

end subroutine dlf_global_real_scatter_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_scatter_flat
!!
!! FUNCTION
!!
!! Scatters the integer data in the flattened dummy argument array aflat (array 
!! rank 1, length n*m) on the selected root processor iproc to the dummy arg.
!! array b (array rank 1, length n) on all processors, including iproc, 
!! via mpi_scatter. m must be equal to the number of processors for this to 
!! make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! aflat must be of size 1:n*m.
!! If called by any process other than the 'send process' iproc, aflat 
!! must be a dummy array with length 0. That way, it is avoided that the 
!! therein unused array aflat has to be allocated with size n*m on every 
!! single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! global_comm
!! mpi_ik
!!
!! SYNOPSIS
subroutine dlf_global_int_scatter_flat(aflat,n,b,m,iproc)
!! SOURCE

! scatters the integer data in array aflat on processor iproc
! to array b on all processors via mpi_scatter.  This routine is blocking.

  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: iproc, n, m, ierr
  integer, dimension(merge(n*m,0,glob%iam==iproc)) :: aflat
  integer, dimension(n) :: b
! **********************************************************************
  if (m /= glob%nprocs) call dlf_fail("dlf_global_int_scatter_flat must be called with m=nprocs")
  if (glob%nprocs == 1) then
    b(1:n)=aflat(1:n)
    return
  endif
  if (glob%iam==iproc) then
    if (size(aflat)/=n*m) call dlf_fail("dlf_global_int_scatter_flat: wrong size of dummy argument aflat")
  endif
  if (n <= 0) call dlf_fail("dlf_global_int_scatter_flat called with n <= 0")
  call mpi_scatter(aflat,n,mpi_ik,b,n,mpi_ik,iproc,global_comm,ierr)

  if (ierr /= 0) call dlf_fail("Failure in dlf_global_int_scatter_flat")

end subroutine dlf_global_int_scatter_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_scatter_flat
!!
!! FUNCTION
!!
!! Scatters the logical data in the flattened dummy argument array aflat (array 
!! rank 1, length n*m) on the selected root processor iproc to the dummy arg.
!! array b (array rank 1, length n) on all processors, including iproc, 
!! via mpi_scatter. m must be equal to the number of processors for this to 
!! make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! aflat must be of size 1:n*m.
!! If called by any process other than the 'send process' iproc, aflat 
!! must be a dummy array with length 0. That way, it is avoided that the 
!! therein unused array aflat has to be allocated with size n*m on every 
!! single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! global_comm
!! mpi_logical
!!
!! SYNOPSIS
subroutine dlf_global_log_scatter_flat(aflat,n,b,m,iproc)
!! SOURCE

! scatters the logical data in array aflat on processor iproc
! to array b on all processors via mpi_scatter.  This routine is blocking.

  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: iproc, n, m, ierr
  logical, dimension(merge(n*m,0,glob%iam==iproc)) :: aflat
  logical, dimension(n) :: b
! **********************************************************************
  if (m /= glob%nprocs) call dlf_fail("dlf_global_log_scatter_flat must be called with m=nprocs")
  if (glob%nprocs == 1) then
    b(1:n)=aflat(1:n)
    return
  endif
  if (glob%iam==iproc) then
    if (size(aflat)/=n*m) call dlf_fail("dlf_global_log_scatter_flat: wrong size of dummy argument aflat")
  endif
  if (n <= 0) call dlf_fail("dlf_global_log_scatter_flat called with n <= 0")
  call mpi_scatter(aflat,n,mpi_logical,b,n,mpi_logical,iproc,global_comm,ierr)

  if (ierr /= 0) call dlf_fail("Failure in dlf_global_log_scatter_flat")

end subroutine dlf_global_log_scatter_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_gather_flat
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the real data 
!! stored in the rank 1 array 'a' (len=n) from each processor (including
!! iproc) into the flattened rank 1 array bflat (length n*m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! bflat must be of size 1:n*m.
!! If called by any process other than the 'receive process' iproc, bflat 
!! must be a dummy array with length 0. That way, it is avoided that the 
!! therein unused array bflat has to be allocated with size n*m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!! global_comm
!! mpi_rk
!!
!! SYNOPSIS
subroutine dlf_global_real_gather_flat(a,n,bflat,m,iproc)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: n, m, ierr, iproc
  real(rk), dimension(n) :: a
  real(rk), dimension(merge(n*m,0,glob%iam==iproc)) :: bflat
! **********************************************************************
  if (m /= glob%nprocs) call dlf_fail("dlf_global_real_gather_flat must be called with m=nprocs")
  if (glob%nprocs == 1) then
    bflat(1:n)=a(1:n)
    return
  endif
  if (glob%iam==iproc) then
    if (size(bflat)/=n*m) call dlf_fail("dlf_global_real_gather_flat: wrong size of dummy argument bflat")
  endif
  if (n <= 0) call dlf_fail("dlf_global_real_gather_flat was called with n <= 0")
  call mpi_gather(a, n, mpi_rk, bflat, n, mpi_rk, iproc, global_comm, ierr)
  
  if (ierr /= 0) call dlf_fail("Failure in mpi_gather in dlf_global_real_gather_flat")
  
end subroutine dlf_global_real_gather_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_gather_flat
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the integer data 
!! stored in the rank 1 array 'a' (len=n) from each processor (including
!! iproc) into the flattened rank 1 array bflat (length n*m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! bflat must be of size 1:n*m.
!! If called by any process other than the 'receive process' iproc, bflat 
!! must be a dummy array with length 0. That way, it is avoided that the 
!! therein unused array bflat has to be allocated with size n*m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!! global_comm
!! mpi_ik
!!
!! SYNOPSIS
subroutine dlf_global_int_gather_flat(a,n,bflat,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: n, m, ierr, iproc
  integer, dimension(n) :: a
  integer, dimension(merge(n*m,0,glob%iam==iproc)) :: bflat
! **********************************************************************
  if (m /= glob%nprocs) call dlf_fail("dlf_global_int_gather_flat must be called with m=nprocs")
  if (glob%nprocs == 1) then
    bflat(1:n)=a(1:n)
    return
  endif
  if (glob%iam==iproc) then
    if (size(bflat)/=n*m) call dlf_fail("dlf_global_int_gather_flat: wrong size of dummy argument bflat")
  endif
  if (n <= 0) call dlf_fail("dlf_global_int_gather_flat was called with n <= 0")
  call mpi_gather(a, n, mpi_ik, bflat, n, mpi_ik, iproc, global_comm, ierr)
  
  if (ierr /= 0) call dlf_fail("Failure in mpi_gather in dlf_global_int_gather_flat")
  
end subroutine dlf_global_int_gather_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_gather_flat
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the logical data 
!! stored in the rank 1 array 'a' (len=n) from each processor (including
!! iproc) into the flattened rank 1 array bflat (length n*m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! bflat must be of size 1:n*m.
!! If called by any process other than the 'receive process' iproc, bflat 
!! must be a dummy array with length 0. That way, it is avoided that the 
!! therein unused array bflat has to be allocated with size n*m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!! global_comm
!! mpi_logical
!!
!! SYNOPSIS
subroutine dlf_global_log_gather_flat(a,n,bflat,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: n, m, ierr, iproc
  logical, dimension(n) :: a
  logical, dimension(merge(n*m,0,glob%iam==iproc)) :: bflat
! **********************************************************************
  if (m /= glob%nprocs) call dlf_fail("dlf_global_log_gather_flat must be called with m=nprocs")
  if (glob%nprocs == 1) then
    bflat(1:n)=a(1:n)
    return
  endif
  if (glob%iam==iproc) then
    if (size(bflat)/=n*m) call dlf_fail("dlf_global_log_gather_flat: wrong size of dummy argument bflat")
  endif
  if (n <= 0) call dlf_fail("dlf_global_log_gather_flat was called with n <= 0")
  call mpi_gather(a, n, mpi_logical, bflat, n, mpi_logical, iproc, global_comm, ierr)
  
  if (ierr /= 0) call dlf_fail("Failure in mpi_gather in dlf_global_log_gather_flat")
  
end subroutine dlf_global_log_gather_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_char_gather_flat
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the character data 
!! stored in the rank 1 array 'a' (len=n) from each processor (including
!! iproc) into the flattened rank 1 array bflat (length n*m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! bflat must be of size 1:n*m.
!! If called by any process other than the 'receive process' iproc, bflat 
!! must be a dummy array with length 0. That way, it is avoided that the 
!! therein unused array bflat has to be allocated with size n*m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!! global_comm
!! mpi_character
!!
!! SYNOPSIS
subroutine dlf_global_char_gather_flat(a,n,bflat,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer :: n, m, iproc
  integer :: ierr, buflen, i, chlen_a, chlen_bf, istart, iend
  character(len=*), dimension(n) :: a
  character(len=*), dimension(merge(n*m,0,glob%iam==iproc)) :: bflat
  character(len=n*len(a)) :: a_app
  character(len=n*m*len(bflat)) :: bf_app
! **********************************************************************
  if (m /= glob%nprocs) call dlf_fail("dlf_global_char_gather_flat must be called with m=nprocs")
  chlen_a =len(a)
  chlen_bf=len(bflat)
  if ( chlen_a==0 .or. chlen_a/=chlen_bf ) &
      & call dlf_fail("dlf_global_char_gather_flat: string length mismatch")
  if (glob%nprocs == 1) then
    bflat(1:n)=a(1:n)
    return
  endif
  if (glob%iam==iproc) then
    if (size(bflat)/=n*m) call dlf_fail("dlf_global_char_gather_flat: wrong size of dummy argument bflat")
  endif
  do i=1,n
    istart=(i-1)*chlen_a+1
    iend  =istart+chlen_a-1
    a_app(istart:iend)=a(i)
  enddo
  if (n <= 0) call dlf_fail("dlf_global_char_gather_flat was called with n <= 0")
  buflen=n*chlen_a
  call mpi_gather(a_app, buflen, mpi_character, bf_app, buflen, mpi_character, iproc, global_comm, ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_gather in dlf_global_char_gather_flat")
  
  if (glob%iam==iproc) then
    do i=1,n*m
      istart=(i-1)*chlen_a+1
      iend  =istart+chlen_a-1
      bflat(i)=bf_app(istart:iend)
    enddo
  endif
  
  return
  
end subroutine dlf_global_char_gather_flat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_scatter_rank0
!!
!! FUNCTION
!!
!! Scatters the real data in the dummy argument array 'a' (array rank 1, 
!! length m) on the selected root processor iproc to the dummy arg.
!! scalar 'b' on all processors, including iproc, via mpi_scatter. 
!! m must be equal to the number of processors for this to 
!! make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of size 1:m.
!! If called by any process other than the 'send process' iproc, 'a' must be
!! a dummy array with length 0. That way, it is avoided that the 
!! therein unused array 'a' has to be allocated with size m on every 
!! single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_real_scatter_rank0(a,b,m,iproc)
!! SOURCE

! scatters the real(rk) data in array a on processor iproc
! to scalar b on all processors via mpi_scatter.  This routine is blocking.

  use, intrinsic :: ISO_C_BINDING
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: iproc, m
  real(rk), dimension(merge(m,0,glob%iam==iproc)) :: a
  real(rk),target :: b
  real(rk),pointer,dimension(:),contiguous :: bp
! **********************************************************************
  if (glob%nprocs == 1) then
    b=a(1)
    return
  endif
  nullify(bp)
  call c_f_pointer( c_loc(b), bp, [1] )
  call dlf_global_real_scatter_flat(a,1,bp,m,iproc)
  nullify(bp)
  
end subroutine dlf_global_real_scatter_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_scatter_rank0
!!
!! FUNCTION
!!
!! Scatters the integer data in the dummy argument array 'a' (array rank 1, 
!! length m) on the selected root processor iproc to the dummy arg.
!! scalar 'b' on all processors, including iproc, via mpi_scatter. 
!! m must be equal to the number of processors for this to 
!! make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of size 1:m.
!! If called by any process other than the 'send process' iproc, 'a' must be
!! a dummy array with length 0. That way, it is avoided that the 
!! therein unused array 'a' has to be allocated with size m on every 
!! single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_int_scatter_rank0(a,b,m,iproc)
!! SOURCE

! scatters the integer data in array a on processor iproc
! to scalar b on all processors via mpi_scatter.  This routine is blocking.

  use, intrinsic :: ISO_C_BINDING
  use dlf_global, only: glob
  implicit none

  integer :: iproc, m
  integer, dimension(merge(m,0,glob%iam==iproc)) :: a
  integer,target :: b
  integer,pointer,dimension(:),contiguous :: bp
! **********************************************************************
  if (glob%nprocs == 1) then
    b=a(1)
    return
  endif
  nullify(bp)
  call c_f_pointer( c_loc(b), bp, [1] )
  call dlf_global_int_scatter_flat(a,1,bp,m,iproc)
  nullify(bp)
  
end subroutine dlf_global_int_scatter_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_scatter_rank0
!!
!! FUNCTION
!!
!! Scatters the logical data in the dummy argument array 'a' (array rank 1, 
!! length m) on the selected root processor iproc to the dummy arg.
!! scalar 'b' on all processors, including iproc, via mpi_scatter. 
!! m must be equal to the number of processors for this to 
!! make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of size 1:m.
!! If called by any process other than the 'send process' iproc, 'a' must be
!! a dummy array with length 0. That way, it is avoided that the 
!! therein unused array 'a' has to be allocated with size m on every 
!! single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_log_scatter_rank0(a,b,m,iproc)
!! SOURCE

! scatters the logical data in array a on processor iproc
! to scalar b on all processors via mpi_scatter.  This routine is blocking.

  use, intrinsic :: ISO_C_BINDING
  use dlf_global, only: glob
  implicit none

  integer :: iproc, m
  logical, dimension(merge(m,0,glob%iam==iproc)) :: a
  logical,target :: b
  logical,pointer,dimension(:),contiguous :: bp
! **********************************************************************
  if (glob%nprocs == 1) then
    b=a(1)
    return
  endif
  nullify(bp)
  call c_f_pointer( c_loc(b), bp, [1] )
  call dlf_global_log_scatter_flat(a,1,bp,m,iproc)
  nullify(bp)
  
end subroutine dlf_global_log_scatter_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_gather_rank0
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the real data 
!! stored in the scalar 'a' from each processor (including
!! iproc) into the rank 1 array 'b' (length m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of size 1:m.
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with length 0. That way, it is avoided that the 
!! therein unused array 'b' has to be allocated with size m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_real_gather_rank0(a,b,m,iproc)
!! SOURCE
  use, intrinsic :: ISO_C_BINDING
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: m, iproc
  real(rk), target :: a
  real(rk), dimension(merge(m,0,glob%iam==iproc)) :: b
  real(rk), pointer, dimension(:),contiguous :: ap
! **********************************************************************
  if (glob%nprocs == 1) then
    b(1)=a
    return
  endif
  nullify(ap)
  call c_f_pointer( c_loc(a), ap, [1] )
  call dlf_global_real_gather_flat(ap,1,b,m,iproc)
  nullify(ap)
  
end subroutine dlf_global_real_gather_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_gather_rank0
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the integer data 
!! stored in the scalar 'a' from each processor (including
!! iproc) into the rank 1 array 'b' (length m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of size 1:m.
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with length 0. That way, it is avoided that the 
!! therein unused array 'b' has to be allocated with size m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_int_gather_rank0(a,b,m,iproc)
!! SOURCE
  use, intrinsic :: ISO_C_BINDING
  use dlf_global, only: glob
  implicit none

  integer :: m, iproc
  integer, target :: a
  integer, dimension(merge(m,0,glob%iam==iproc)) :: b
  integer, pointer, dimension(:),contiguous :: ap
! **********************************************************************
  if (glob%nprocs == 1) then
    b(1)=a
    return
  endif
  nullify(ap)
  call c_f_pointer( c_loc(a), ap, [1] )
  call dlf_global_int_gather_flat(ap,1,b,m,iproc)
  nullify(ap)
  
end subroutine dlf_global_int_gather_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_gather_rank0
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the logical data 
!! stored in the scalar 'a' from each processor (including
!! iproc) into the rank 1 array 'b' (length m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of size 1:m.
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with length 0. That way, it is avoided that the 
!! therein unused array 'b' has to be allocated with size m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_log_gather_rank0(a,b,m,iproc)
!! SOURCE
  use, intrinsic :: ISO_C_BINDING
  use dlf_global, only: glob
  implicit none

  integer :: m, iproc
  logical, target :: a
  logical, dimension(merge(m,0,glob%iam==iproc)) :: b
  logical, pointer, dimension(:),contiguous :: ap
! **********************************************************************
  if (glob%nprocs == 1) then
    b(1)=a
    return
  endif
  nullify(ap)
  call c_f_pointer( c_loc(a), ap, [1] )
  call dlf_global_log_gather_flat(ap,1,b,m,iproc)
  nullify(ap)
  
end subroutine dlf_global_log_gather_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_char_gather_rank0
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the character data 
!! stored in the scalar 'a' from each processor (including
!! iproc) into the rank 1 array 'b' (length m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of size 1:m.
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with length 0. That way, it is avoided that the 
!! therein unused array 'b' has to be allocated with size m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_char_gather_rank0(a,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: m, iproc
  character(len=*) :: a
  character(len=*), dimension(merge(m,0,glob%iam==iproc)) :: b
  character(len=len(a)), dimension(1) :: acopy
! **********************************************************************
  if (glob%nprocs == 1) then
    b(1)=a
    return
  endif
  
  acopy(1)=a
  call dlf_global_char_gather_flat(acopy,1,b,m,iproc)
  
end subroutine dlf_global_char_gather_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_scatter_rank1
!!
!! FUNCTION
!!
!! Scatters the real data in the dummy argument array 'a' (array rank 2, 
!! shape: n x m) on the selected root processor iproc to the dummy arg.
!! array 'b' (array rank 1, length n) on all processors, including iproc, 
!! via mpi_scatter. m must be equal to the number of processors for this to 
!! make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of shape (n,m).
!! If called by any process other than the 'send process' iproc, 'a' 
!! must be a dummy array with shape (0,0). That way, it is avoided that the 
!! therein unused array 'a' has to be allocated with size nxm on every 
!! single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_real_scatter_rank1(a,n,b,m,iproc)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n, m
  
  real(rk), dimension(merge(n,0,glob%iam==iproc),merge(m,0,glob%iam==iproc)), &
                    & target  :: a
  real(rk), dimension(n) :: b
  real(rk), dimension(:), pointer,contiguous :: aflat
! **********************************************************************
  if (glob%nprocs==1) then
    b(1:n)=a(1:n,1)
    return
  endif
  nullify(aflat)
  if (glob%iam==iproc) then
    aflat(1:n*m) => a
  else
    aflat(0:0) => a
  endif
  call dlf_global_real_scatter_flat(aflat,n,b,m,iproc)
  nullify(aflat)
end subroutine dlf_global_real_scatter_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_scatter_rank1
!!
!! FUNCTION
!!
!! Scatters the integer data in the dummy argument array 'a' (array rank 2, 
!! shape: n x m) on the selected root processor iproc to the dummy arg.
!! array 'b' (array rank 1, length n) on all processors, including iproc, 
!! via mpi_scatter. m must be equal to the number of processors for this to 
!! make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of shape (n,m).
!! If called by any process other than the 'send process' iproc, 'a' 
!! must be a dummy array with shape (0,0). That way, it is avoided that the 
!! therein unused array 'a' has to be allocated with size nxm on every 
!! single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_int_scatter_rank1(a,n,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n, m
  
  integer, dimension(merge(n,0,glob%iam==iproc),merge(m,0,glob%iam==iproc)), &
                    & target  :: a
  integer, dimension(n) :: b
  integer, dimension(:), pointer,contiguous :: aflat
! **********************************************************************
  if (glob%nprocs==1) then
    b(1:n)=a(1:n,1)
    return
  endif
  nullify(aflat)
  if (glob%iam==iproc) then
    aflat(1:n*m) => a
  else
    aflat(0:0) => a
  endif
  call dlf_global_int_scatter_flat(aflat,n,b,m,iproc)
  nullify(aflat)
end subroutine dlf_global_int_scatter_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_scatter_rank1
!!
!! FUNCTION
!!
!! Scatters the logical data in the dummy argument array 'a' (array rank 2, 
!! shape: n x m) on the selected root processor iproc to the dummy arg.
!! array 'b' (array rank 1, length n) on all processors, including iproc, 
!! via mpi_scatter. m must be equal to the number of processors for this to 
!! make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of shape (n,m).
!! If called by any process other than the 'send process' iproc, 'a' 
!! must be a dummy array with shape (0,0). That way, it is avoided that the 
!! therein unused array 'a' has to be allocated with size nxm on every 
!! single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_log_scatter_rank1(a,n,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n, m
  
  logical, dimension(merge(n,0,glob%iam==iproc),merge(m,0,glob%iam==iproc)), &
                    & target  :: a
  logical, dimension(n) :: b
  logical, dimension(:), pointer,contiguous :: aflat
! **********************************************************************
  if (glob%nprocs==1) then
    b(1:n)=a(1:n,1)
    return
  endif
  nullify(aflat)
  if (glob%iam==iproc) then
    aflat(1:n*m) => a
  else
    aflat(0:0) => a
  endif
  call dlf_global_log_scatter_flat(aflat,n,b,m,iproc)
  nullify(aflat)
end subroutine dlf_global_log_scatter_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_gather_rank1
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the real data 
!! stored in the rank 1 array 'a' (len=n) from each processor (including
!! iproc) into the rank 2 array 'b' (shape n x m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of shape (n,m).
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with shape (0,0). That way, it is avoided that the 
!! therein unused array 'b' has to be allocated with shape n x m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_real_gather_rank1(a,n,b,m,iproc)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: n, m, iproc
  real(rk), dimension(n) :: a
  real(rk), dimension(merge(n,0,glob%iam==iproc),  & 
                &     merge(m,0,glob%iam==iproc)), &
                &     target :: b
  real(rk), dimension(:), pointer,contiguous :: bflat
! **********************************************************************
  if (glob%nprocs==1) then
    b(1:n,1)=a(1:n)
    return
  endif
  nullify(bflat)
  if (glob%iam==iproc) then
    bflat(1:n*m) => b
  else
    bflat(0:0) => b
  endif
  call dlf_global_real_gather_flat(a,n,bflat,m,iproc)
  nullify(bflat)
  
end subroutine dlf_global_real_gather_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_gather_rank1
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the integer data 
!! stored in the rank 1 array 'a' (len=n) from each processor (including
!! iproc) into the rank 2 array 'b' (shape n x m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of shape (n,m).
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with shape (0,0). That way, it is avoided that the 
!! therein unused array 'b' has to be allocated with shape n x m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_int_gather_rank1(a,n,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: n, m, iproc
  integer, dimension(n) :: a
  integer, dimension(merge(n,0,glob%iam==iproc),  & 
                &    merge(m,0,glob%iam==iproc)), &
                &    target :: b
  integer, dimension(:), pointer,contiguous :: bflat
! **********************************************************************
  if (glob%nprocs==1) then
    b(1:n,1)=a(1:n)
    return
  endif
  nullify(bflat)
  if (glob%iam==iproc) then
    bflat(1:n*m) => b
  else
    bflat(0:0) => b
  endif
  call dlf_global_int_gather_flat(a,n,bflat,m,iproc)
  nullify(bflat)
  
end subroutine dlf_global_int_gather_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_gather_rank1
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the logical data 
!! stored in the rank 1 array 'a' (len=n) from each processor (including
!! iproc) into the rank 2 array 'b' (shape n x m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of shape (n,m).
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with shape (0,0). That way, it is avoided that the 
!! therein unused array 'b' has to be allocated with shape n x m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_log_gather_rank1(a,n,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: n, m, iproc
  logical, dimension(n) :: a
  logical, dimension(merge(n,0,glob%iam==iproc),  & 
                &    merge(m,0,glob%iam==iproc)), &
                &    target :: b
  logical, dimension(:), pointer,contiguous :: bflat
! **********************************************************************
  if (glob%nprocs==1) then
    b(1:n,1)=a(1:n)
    return
  endif
  nullify(bflat)
  if (glob%iam==iproc) then
    bflat(1:n*m) => b
  else
    bflat(0:0) => b
  endif
  call dlf_global_log_gather_flat(a,n,bflat,m,iproc)
  nullify(bflat)
  
end subroutine dlf_global_log_gather_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_char_gather_rank1
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the character data 
!! stored in the rank 1 array 'a' (len=n) from each processor (including
!! iproc) into the rank 2 array 'b' (shape n x m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of shape (n,m).
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with shape (0,0). That way, it is avoided that the 
!! therein unused array 'b' has to be allocated with shape n x m on every 
!! single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_char_gather_rank1(a,n,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: n, m, iproc
  character(len=*), dimension(n) :: a
  character(len=*), dimension(merge(n,0,glob%iam==iproc),  & 
                   &    merge(m,0,glob%iam==iproc)), &
                   &    target :: b
  character(len=:), dimension(:), pointer,contiguous :: bflat
! **********************************************************************
  if (glob%nprocs==1) then
    b(1:n,1)=a(1:n)
    return
  endif
  nullify(bflat)
  if (glob%iam==iproc) then
    bflat(1:n*m) => b
  else
    bflat(0:0) => b
  endif
  call dlf_global_char_gather_flat(a,n,bflat,m,iproc)
  nullify(bflat)
  
end subroutine dlf_global_char_gather_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_bcast_rank2
!!
!! FUNCTION
!!
!! Broadcasts the real data in the dummy argument array 'a' (array rank 2, 
!! shape: n1 x n2) from the selected root processor iproc across all 
!! processors via mpi_bcast.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_real_bcast_rank2(a,n1,n2,iproc)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2
  
  real(rk), dimension(n1,n2), target  :: a
  real(rk), dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2
  if (glob%nprocs==1) then
    return
  endif
  nullify(aflat)
  aflat(1:n1*n2) => a
  call dlf_global_real_bcast(aflat,n,iproc)
  nullify(aflat)
end subroutine dlf_global_real_bcast_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_bcast_rank2
!!
!! FUNCTION
!!
!! Broadcasts the integer data in the dummy argument array 'a' (array 
!! rank 2, shape: n1 x n2) from the selected root processor iproc across 
!! all processors via mpi_bcast.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_int_bcast_rank2(a,n1,n2,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2
  
  integer, dimension(n1,n2), target  :: a
  integer, dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2
  if (glob%nprocs==1) then
    return
  endif
  nullify(aflat)
  aflat(1:n1*n2) => a
  call dlf_global_int_bcast(aflat,n,iproc)
  nullify(aflat)
end subroutine dlf_global_int_bcast_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_bcast_rank2
!!
!! FUNCTION
!!
!! Broadcasts the logical data in the dummy argument array 'a' (array 
!! rank 2, shape: n1 x n2) from the selected root processor iproc across 
!! all processors via mpi_bcast.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_log_bcast_rank2(a,n1,n2,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2
  
  logical, dimension(n1,n2), target  :: a
  logical, dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2
  if (glob%nprocs==1) then
    return
  endif
  nullify(aflat)
  aflat(1:n1*n2) => a
  call dlf_global_log_bcast(aflat,n,iproc)
  nullify(aflat)
end subroutine dlf_global_log_bcast_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_scatter_rank2
!!
!! FUNCTION
!!
!! Scatters the real data in the dummy argument array 'a' (array rank 3, 
!! shape: n1 x n2 x m) on the selected root processor iproc to the dummy 
!! arg. array 'b' (array rank 2, shape n1 x n2) on all processors, including
!! iproc, via mpi_scatter. m must be equal to the number of processors for
!! this to make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of shape (n1,n2,m).
!! If called by any process other than the 'send process' iproc, 'a' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'a' has to be allocated with size n1xn2xm on 
!! every single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_real_scatter_rank2(a,n1,n2,b,m,iproc)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2, m
  
  real(rk), dimension(merge(n1,0,glob%iam==iproc), &
                    & merge(n2,0,glob%iam==iproc), &
                    & merge(m,0,glob%iam==iproc)), &
                    & target  :: a
  real(rk), dimension(n1,n2) :: b
  real(rk), dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2
  if (glob%nprocs==1) then
    b(1:n1,1:n2)=a(1:n1,1:n2,1)
    return
  endif
  nullify(aflat)
  if (glob%iam==iproc) then
    aflat(1:n1*n2*m) => a
  else
    aflat(0:0) => a
  endif
  call dlf_global_real_scatter_flat(aflat,n,b,m,iproc)
  nullify(aflat)
end subroutine dlf_global_real_scatter_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_scatter_rank2
!!
!! FUNCTION
!!
!! Scatters the integer data in the dummy argument array 'a' (array rank 3, 
!! shape: n1 x n2 x m) on the selected root processor iproc to the dummy 
!! arg. array 'b' (array rank 2, shape n1 x n2) on all processors, including
!! iproc, via mpi_scatter. m must be equal to the number of processors for
!! this to make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of shape (n1,n2,m).
!! If called by any process other than the 'send process' iproc, 'a' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'a' has to be allocated with size n1xn2xm on 
!! every single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_int_scatter_rank2(a,n1,n2,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2, m
  
  integer, dimension(merge(n1,0,glob%iam==iproc), &
                   & merge(n2,0,glob%iam==iproc), &
                   & merge(m,0,glob%iam==iproc)), &
                   & target  :: a
  integer, dimension(n1,n2) :: b
  integer, dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2
  if (glob%nprocs==1) then
    b(1:n1,1:n2)=a(1:n1,1:n2,1)
    return
  endif
  nullify(aflat)
  if (glob%iam==iproc) then
    aflat(1:n1*n2*m) => a
  else
    aflat(0:0) => a
  endif
  call dlf_global_int_scatter_flat(aflat,n,b,m,iproc)
  nullify(aflat)
end subroutine dlf_global_int_scatter_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_scatter_rank2
!!
!! FUNCTION
!!
!! Scatters the logical data in the dummy argument array 'a' (array rank 3, 
!! shape: n1 x n2 x m) on the selected root processor iproc to the dummy 
!! arg. array 'b' (array rank 2, shape n1 x n2) on all processors, including
!! iproc, via mpi_scatter. m must be equal to the number of processors for
!! this to make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of shape (n1,n2,m).
!! If called by any process other than the 'send process' iproc, 'a' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'a' has to be allocated with size n1xn2xm on 
!! every single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_log_scatter_rank2(a,n1,n2,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2, m
  
  logical, dimension(merge(n1,0,glob%iam==iproc), &
                   & merge(n2,0,glob%iam==iproc), &
                   & merge(m,0,glob%iam==iproc)), &
                   & target  :: a
  logical, dimension(n1,n2) :: b
  logical, dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2
  if (glob%nprocs==1) then
    b(1:n1,1:n2)=a(1:n1,1:n2,1)
    return
  endif
  nullify(aflat)
  if (glob%iam==iproc) then
    aflat(1:n1*n2*m) => a
  else
    aflat(0:0) => a
  endif
  call dlf_global_log_scatter_flat(aflat,n,b,m,iproc)
  nullify(aflat)
end subroutine dlf_global_log_scatter_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_gather_rank2
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the real data stored in the 
!! rank 2 array 'a' (shape n1 x n2) from each processor (including
!! iproc) into the rank 3 array 'b' (shape n1 x n2 x m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of shape (n1,n2,m).
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'b' has to be allocated with shape n1 x n2 x m
!! on every single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_real_gather_rank2(a,n1,n2,b,m,iproc)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: n1, n2, m, iproc
  real(rk), dimension(n1,n2) :: a
  real(rk), dimension(merge(n1,0,glob%iam==iproc), & 
                &     merge(n2,0,glob%iam==iproc), &
                &     merge(m,0,glob%iam==iproc)), &
                &     target :: b
  real(rk), dimension(:), pointer,contiguous :: bflat
  integer :: n
! **********************************************************************
  n=n1*n2
  if (glob%nprocs==1) then
    b(1:n1,1:n2,1)=a(1:n1,1:n2)
    return
  endif
  nullify(bflat)
  if (glob%iam==iproc) then
    bflat(1:n1*n2*m) => b
  else
    bflat(0:0) => b
  endif
  call dlf_global_real_gather_flat(a,n,bflat,m,iproc)
  nullify(bflat)
  
end subroutine dlf_global_real_gather_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_gather_rank2
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the integer data stored in the 
!! rank 2 array 'a' (shape n1 x n2) from each processor (including
!! iproc) into the rank 3 array 'b' (shape n1 x n2 x m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of shape (n1,n2,m).
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'b' has to be allocated with shape n1 x n2 x m
!! on every single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_int_gather_rank2(a,n1,n2,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: n1, n2, m, iproc
  integer, dimension(n1,n2) :: a
  integer, dimension(merge(n1,0,glob%iam==iproc), & 
                &    merge(n2,0,glob%iam==iproc), &
                &    merge(m,0,glob%iam==iproc)), &
                &    target :: b
  integer, dimension(:), pointer,contiguous :: bflat
  integer :: n
! **********************************************************************
  n=n1*n2
  if (glob%nprocs==1) then
    b(1:n1,1:n2,1)=a(1:n1,1:n2)
    return
  endif
  nullify(bflat)
  if (glob%iam==iproc) then
    bflat(1:n1*n2*m) => b
  else
    bflat(0:0) => b
  endif
  call dlf_global_int_gather_flat(a,n,bflat,m,iproc)
  nullify(bflat)
  
end subroutine dlf_global_int_gather_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_gather_rank2
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the logical data stored in the 
!! rank 2 array 'a' (shape n1 x n2) from each processor (including
!! iproc) into the rank 3 array 'b' (shape n1 x n2 x m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of shape (n1,n2,m).
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'b' has to be allocated with shape n1 x n2 x m
!! on every single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_log_gather_rank2(a,n1,n2,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: n1, n2, m, iproc
  logical, dimension(n1,n2) :: a
  logical, dimension(merge(n1,0,glob%iam==iproc), & 
                &    merge(n2,0,glob%iam==iproc), &
                &    merge(m,0,glob%iam==iproc)), &
                &    target :: b
  logical, dimension(:), pointer,contiguous :: bflat
  integer :: n
! **********************************************************************
  n=n1*n2
  if (glob%nprocs==1) then
    b(1:n1,1:n2,1)=a(1:n1,1:n2)
    return
  endif
  nullify(bflat)
  if (glob%iam==iproc) then
    bflat(1:n1*n2*m) => b
  else
    bflat(0:0) => b
  endif
  call dlf_global_log_gather_flat(a,n,bflat,m,iproc)
  nullify(bflat)
  
end subroutine dlf_global_log_gather_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_sum_rank2
!!
!! FUNCTION
!!
!! Sums up the real data in the dummy argument array 'a' (array rank 2, 
!! shape: n1 x n2) across all processors via mpi_allreduce.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_real_sum_rank2(a,n1,n2)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: n1, n2
  
  real(rk), dimension(n1,n2), target  :: a
  real(rk), dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2
  if (glob%nprocs==1) then
    return
  endif
  nullify(aflat)
  aflat(1:n1*n2) => a
  call dlf_global_real_sum(aflat,n)
  nullify(aflat)
end subroutine dlf_global_real_sum_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_sum_rank2
!!
!! FUNCTION
!!
!! Sums up the integer data in the dummy argument array 'a' (array rank 2, 
!! shape: n1 x n2) across all processors via mpi_allreduce.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_int_sum_rank2(a,n1,n2)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: n1, n2
  
  integer, dimension(n1,n2), target  :: a
  integer, dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2
  if (glob%nprocs==1) then
    return
  endif
  nullify(aflat)
  aflat(1:n1*n2) => a
  call dlf_global_int_sum(aflat,n)
  nullify(aflat)
end subroutine dlf_global_int_sum_rank2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_scatter_rank3
!!
!! FUNCTION
!!
!! Scatters the real data in the dummy argument array 'a' (array rank 3, 
!! shape: n1 x n2 x m) on the selected root processor iproc to the dummy 
!! arg. array 'b' (array rank 2, shape n1 x n2) on all processors, including
!! iproc, via mpi_scatter. m must be equal to the number of processors for
!! this to make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of shape (n1,n2,m).
!! If called by any process other than the 'send process' iproc, 'a' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'a' has to be allocated with size n1xn2xm on 
!! every single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_real_scatter_rank3(a,n1,n2,n3,b,m,iproc)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2, n3, m
  
  real(rk), dimension(merge(n1,0,glob%iam==iproc), &
                    & merge(n2,0,glob%iam==iproc), &
                    & merge(n3,0,glob%iam==iproc), &
                    & merge(m,0,glob%iam==iproc)), &
                    & target  :: a
  real(rk), dimension(n1,n2,n3) :: b
  real(rk), dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2*n3
  if (glob%nprocs==1) then
    b(1:n1,1:n2,1:n3)=a(1:n1,1:n2,1:n3,1)
    return
  endif
  nullify(aflat)
  if (glob%iam==iproc) then
    aflat(1:n1*n2*n3*m) => a
  else
    aflat(0:0) => a
  endif
  call dlf_global_real_scatter_flat(aflat,n,b,m,iproc)
  nullify(aflat)
end subroutine dlf_global_real_scatter_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_scatter_rank3
!!
!! FUNCTION
!!
!! Scatters the integer data in the dummy argument array 'a' (array rank 3, 
!! shape: n1 x n2 x m) on the selected root processor iproc to the dummy 
!! arg. array 'b' (array rank 2, shape n1 x n2) on all processors, including
!! iproc, via mpi_scatter. m must be equal to the number of processors for
!! this to make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of shape (n1,n2,m).
!! If called by any process other than the 'send process' iproc, 'a' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'a' has to be allocated with size n1xn2xm on 
!! every single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_int_scatter_rank3(a,n1,n2,n3,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2, n3, m
  
  integer, dimension(merge(n1,0,glob%iam==iproc), &
                   & merge(n2,0,glob%iam==iproc), &
                   & merge(n3,0,glob%iam==iproc), &
                   & merge(m,0,glob%iam==iproc)), &
                   & target  :: a
  integer, dimension(n1,n2,n3) :: b
  integer, dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2*n3
  if (glob%nprocs==1) then
    b(1:n1,1:n2,1:n3)=a(1:n1,1:n2,1:n3,1)
    return
  endif
  nullify(aflat)
  if (glob%iam==iproc) then
    aflat(1:n1*n2*n3*m) => a
  else
    aflat(0:0) => a
  endif
  call dlf_global_int_scatter_flat(aflat,n,b,m,iproc)
  nullify(aflat)
end subroutine dlf_global_int_scatter_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_scatter_rank3
!!
!! FUNCTION
!!
!! Scatters the logical data in the dummy argument array 'a' (array rank 3, 
!! shape: n1 x n2 x m) on the selected root processor iproc to the dummy 
!! arg. array 'b' (array rank 2, shape n1 x n2) on all processors, including
!! iproc, via mpi_scatter. m must be equal to the number of processors for
!! this to make sense. 
!! If called by the 'send process' (glob%iam==iproc), the source array 
!! 'a' must be of shape (n1,n2,m).
!! If called by any process other than the 'send process' iproc, 'a' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'a' has to be allocated with size n1xn2xm on 
!! every single receiving process. This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_log_scatter_rank3(a,n1,n2,n3,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2, n3, m
  
  logical, dimension(merge(n1,0,glob%iam==iproc), &
                   & merge(n2,0,glob%iam==iproc), &
                   & merge(n3,0,glob%iam==iproc), &
                   & merge(m,0,glob%iam==iproc)), &
                   & target  :: a
  logical, dimension(n1,n2,n3) :: b
  logical, dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2*n3
  if (glob%nprocs==1) then
    b(1:n1,1:n2,1:n3)=a(1:n1,1:n2,1:n3,1)
    return
  endif
  nullify(aflat)
  if (glob%iam==iproc) then
    aflat(1:n1*n2*n3*m) => a
  else
    aflat(0:0) => a
  endif
  call dlf_global_log_scatter_flat(aflat,n,b,m,iproc)
  nullify(aflat)
end subroutine dlf_global_log_scatter_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_gather_rank3
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the real data stored in the 
!! rank 2 array 'a' (shape n1 x n2) from each processor (including
!! iproc) into the rank 3 array 'b' (shape n1 x n2 x m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of shape (n1,n2,m).
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'b' has to be allocated with shape n1 x n2 x m
!! on every single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_real_gather_rank3(a,n1,n2,n3,b,m,iproc)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: n1, n2, n3, m, iproc
  real(rk), dimension(n1,n2,n3) :: a
  real(rk), dimension(merge(n1,0,glob%iam==iproc), & 
                &     merge(n2,0,glob%iam==iproc), &
                &     merge(n3,0,glob%iam==iproc), &
                &     merge(m,0,glob%iam==iproc)), &
                &     target :: b
  real(rk), dimension(:), pointer,contiguous :: bflat
  integer :: n
! **********************************************************************
  n=n1*n2*n3
  if (glob%nprocs==1) then
    b(1:n1,1:n2,1:n3,1)=a(1:n1,1:n2,1:n3)
    return
  endif
  nullify(bflat)
  if (glob%iam==iproc) then
    bflat(1:n1*n2*n3*m) => b
  else
    bflat(0:0) => b
  endif
  call dlf_global_real_gather_flat(a,n,bflat,m,iproc)
  nullify(bflat)
  
end subroutine dlf_global_real_gather_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_gather_rank3
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the integer data stored in the 
!! rank 2 array 'a' (shape n1 x n2) from each processor (including
!! iproc) into the rank 3 array 'b' (shape n1 x n2 x m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of shape (n1,n2,m).
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'b' has to be allocated with shape n1 x n2 x m
!! on every single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_int_gather_rank3(a,n1,n2,n3,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: n1, n2, n3, m, iproc
  integer, dimension(n1,n2,n3) :: a
  integer, dimension(merge(n1,0,glob%iam==iproc), & 
                &    merge(n2,0,glob%iam==iproc), &
                &    merge(n3,0,glob%iam==iproc), &
                &    merge(m,0,glob%iam==iproc)), &
                &    target :: b
  integer, dimension(:), pointer,contiguous :: bflat
  integer :: n
! **********************************************************************
  n=n1*n2*n3
  if (glob%nprocs==1) then
    b(1:n1,1:n2,1:n3,1)=a(1:n1,1:n2,1:n3)
    return
  endif
  nullify(bflat)
  if (glob%iam==iproc) then
    bflat(1:n1*n2*n3*m) => b
  else
    bflat(0:0) => b
  endif
  call dlf_global_int_gather_flat(a,n,bflat,m,iproc)
  nullify(bflat)
  
end subroutine dlf_global_int_gather_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_gather_rank3
!!
!! FUNCTION
!!
!! Using mpi_gather, this routine gathers the logical data stored in the 
!! rank 2 array 'a' (shape n1 x n2) from each processor (including
!! iproc) into the rank 3 array 'b' (shape n1 x n2 x m), which will 
!! only be stored on the root task (iproc). m must be equal to 
!! the no. of processors for this to make sense.
!! If called by the 'receive process' (glob%iam==iproc), the target array 
!! 'b' must be of shape (n1,n2,m).
!! If called by any process other than the 'receive process' iproc, 'b' 
!! must be a dummy array with shape (0,0,0). That way, it is avoided that 
!! the therein unused array 'b' has to be allocated with shape n1 x n2 x m
!! on every single sending process.
!!
!! INPUTS
!!
!! local vars
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_global_log_gather_rank3(a,n1,n2,n3,b,m,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: n1, n2, n3, m, iproc
  logical, dimension(n1,n2,n3) :: a
  logical, dimension(merge(n1,0,glob%iam==iproc), & 
                &    merge(n2,0,glob%iam==iproc), &
                &    merge(n3,0,glob%iam==iproc), &
                &    merge(m,0,glob%iam==iproc)), &
                &    target :: b
  logical, dimension(:), pointer,contiguous :: bflat
  integer :: n
! **********************************************************************
  n=n1*n2*n3
  if (glob%nprocs==1) then
    b(1:n1,1:n2,1:n3,1)=a(1:n1,1:n2,1:n3)
    return
  endif
  nullify(bflat)
  if (glob%iam==iproc) then
    bflat(1:n1*n2*n3*m) => b
  else
    bflat(0:0) => b
  endif
  call dlf_global_log_gather_flat(a,n,bflat,m,iproc)
  nullify(bflat)
  
end subroutine dlf_global_log_gather_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_bcast_rank3
!!
!! FUNCTION
!!
!! Broadcasts the real data in the dummy argument array 'a' (array rank 3, 
!! shape: n1 x n2 x n3) from the selected root processor iproc across all 
!! processors via mpi_bcast.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_real_bcast_rank3(a,n1,n2,n3,iproc)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2, n3
  
  real(rk), dimension(n1,n2,n3), target  :: a
  real(rk), dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2*n3
  if (glob%nprocs==1) then
    return
  endif
  nullify(aflat)
  aflat(1:n1*n2*n3) => a
  call dlf_global_real_bcast(aflat,n,iproc)
  nullify(aflat)
end subroutine dlf_global_real_bcast_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_bcast_rank3
!!
!! FUNCTION
!!
!! Broadcasts the integer data in the dummy argument array 'a' (array 
!! rank 3, shape: n1 x n2 x n3) from the selected root processor iproc 
!! across all processors via mpi_bcast.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_int_bcast_rank3(a,n1,n2,n3,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2, n3
  
  integer, dimension(n1,n2,n3), target  :: a
  integer, dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2*n3
  if (glob%nprocs==1) then
    return
  endif
  nullify(aflat)
  aflat(1:n1*n2*n3) => a
  call dlf_global_int_bcast(aflat,n,iproc)
  nullify(aflat)
end subroutine dlf_global_int_bcast_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_log_bcast_rank3
!!
!! FUNCTION
!!
!! Broadcasts the logical data in the dummy argument array 'a' (array 
!! rank 3, shape: n1 x n2 x n3) from the selected root processor iproc 
!! across all processors via mpi_bcast.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_log_bcast_rank3(a,n1,n2,n3,iproc)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: iproc, n1, n2, n3
  
  logical, dimension(n1,n2,n3), target  :: a
  logical, dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2*n3
  if (glob%nprocs==1) then
    return
  endif
  nullify(aflat)
  aflat(1:n1*n2*n3) => a
  call dlf_global_log_bcast(aflat,n,iproc)
  nullify(aflat)
end subroutine dlf_global_log_bcast_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_real_sum_rank3
!!
!! FUNCTION
!!
!! Sums up the real data in the dummy argument array 'a' (array rank 3, 
!! shape: n1 x n2 x n3) across all processors via mpi_allreduce.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_real_sum_rank3(a,n1,n2,n3)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none

  integer :: n1, n2, n3
  
  real(rk), dimension(n1,n2,n3), target  :: a
  real(rk), dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2*n3
  if (glob%nprocs==1) then
    return
  endif
  nullify(aflat)
  aflat(1:n1*n2*n3) => a
  call dlf_global_real_sum(aflat,n)
  nullify(aflat)
end subroutine dlf_global_real_sum_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_global_int_sum_rank3
!!
!! FUNCTION
!!
!! Sums up the integer data in the dummy argument array 'a' (array rank 3, 
!! shape: n1 x n2 x n3) across all processors via mpi_allreduce.
!! This routine is blocking.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!!
!! SYNOPSIS
subroutine dlf_global_int_sum_rank3(a,n1,n2,n3)
!! SOURCE
  use dlf_global, only: glob
  implicit none

  integer :: n1, n2, n3
  
  integer, dimension(n1,n2,n3), target  :: a
  integer, dimension(:), pointer,contiguous :: aflat
  integer :: n
! **********************************************************************
  n=n1*n2*n3
  if (glob%nprocs==1) then
    return
  endif
  nullify(aflat)
  aflat(1:n1*n2*n3) => a
  call dlf_global_int_sum(aflat,n)
  nullify(aflat)
end subroutine dlf_global_int_sum_rank3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_send_real_rank1
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Send
!! Real data in array 'sendbuff' (size: n) is sent to 'target_rank'
!! with the tag passed in dummy argument 'tag'.
!! This subroutine must be matched with a corresponding mpi_recv call
!! on the receiving MPI rank ('taget_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! mpi_rk
!! global_comm
!!
!! SYNOPSIS
subroutine dlf_mpi_send_real_rank1(sendbuff,n,target_rank,tag)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer, intent(in) :: n, target_rank, tag
  real(rk),dimension(n), intent(in) :: sendbuff
  integer :: ierr
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==target_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_send_real_rank1 (potential deadlock)!")
  endif
  call MPI_Send(sendbuff,n,mpi_rk,target_rank,tag,global_comm,ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_send in dlf_mpi_send_real_rank1")
  return
end subroutine dlf_mpi_send_real_rank1
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_send_int_rank1
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Send
!! Integer data in array 'sendbuff' (size: n) is sent to 'target_rank'
!! with the tag passed in dummy argument 'tag'.
!! This subroutine must be matched with a corresponding mpi_recv call
!! on the receiving MPI rank ('taget_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! mpi_ik
!! global_comm
!!
!! SYNOPSIS
subroutine dlf_mpi_send_int_rank1(sendbuff,n,target_rank,tag)
!! SOURCE
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer, intent(in) :: n, target_rank, tag
  integer,dimension(n), intent(in) :: sendbuff
  integer :: ierr
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==target_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_send_int_rank1 (potential deadlock)!")
  endif
  call MPI_Send(sendbuff,n,mpi_ik,target_rank,tag,global_comm,ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_send in dlf_mpi_send_int_rank1")
  return
end subroutine dlf_mpi_send_int_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_send_log_rank1
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Send
!! Logical data in array 'sendbuff' (size: n) is sent to 'target_rank'
!! with the tag passed in dummy argument 'tag'.
!! This subroutine must be matched with a corresponding mpi_recv call
!! on the receiving MPI rank ('taget_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! mpi_logical
!! global_comm
!!
!! SYNOPSIS
subroutine dlf_mpi_send_log_rank1(sendbuff,n,target_rank,tag)
!! SOURCE
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer, intent(in) :: n, target_rank, tag
  logical,dimension(n), intent(in) :: sendbuff
  integer :: ierr
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==target_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_send_log_rank1 (potential deadlock)!")
  endif
  call MPI_Send(sendbuff,n,mpi_logical,target_rank,tag,global_comm,ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_send in dlf_mpi_send_log_rank1")
  return
end subroutine dlf_mpi_send_log_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_send_char_string
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Send
!! Character data in string 'sendbuff' (length: arbitrary, rank: 0 [=scalar])
!! is sent to 'target_rank' with the tag passed in dummy argument 'tag'.
!! This subroutine must be matched with a corresponding mpi_recv call
!! on the receiving MPI rank ('taget_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! mpi_character
!! global_comm
!!
!! SYNOPSIS
subroutine dlf_mpi_send_char_string(sendbuff,target_rank,tag)
!! SOURCE
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer, intent(in) :: target_rank, tag
  character(len=*), intent(in) :: sendbuff
  integer :: ierr, charlen
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==target_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_send_char_string (potential deadlock)!")
  endif
  charlen=len(sendbuff)
  call MPI_Send(sendbuff,charlen,mpi_character,target_rank,tag,global_comm,ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_send in dlf_mpi_send_char_string")
  return
end subroutine dlf_mpi_send_char_string
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_send_real_rank0
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Send
!! Real data in scalar 'sendbuff' is sent to 'target_rank'
!! with the tag passed in dummy argument 'tag'.
!! This subroutine must be matched with a corresponding mpi_recv call
!! on the receiving MPI rank ('taget_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_mpi_send_real_rank0(sendbuff,target_rank,tag)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use, intrinsic :: ISO_C_BINDING
  implicit none

  integer, intent(in) :: target_rank, tag
  real(rk), intent(in), target :: sendbuff
  real(rk), pointer, dimension(:),contiguous :: sbp
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==target_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_send_real_rank0 (potential deadlock)!")
  endif
  nullify(sbp)
  call c_f_pointer( c_loc(sendbuff), sbp, [1] )
  call dlf_mpi_send_real_rank1(sbp,1,target_rank,tag)
  nullify(sbp)
  return
end subroutine dlf_mpi_send_real_rank0
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_send_int_rank0
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Send
!! Integer data in scalar 'sendbuff' (size: n) is sent to 'target_rank'
!! with the tag passed in dummy argument 'tag'.
!! This subroutine must be matched with a corresponding mpi_recv call
!! on the receiving MPI rank ('taget_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_mpi_send_int_rank0(sendbuff,target_rank,tag)
!! SOURCE
  use dlf_global, only: glob
  use, intrinsic :: ISO_C_BINDING
  implicit none

  integer, intent(in) :: target_rank, tag
  integer, intent(in), target :: sendbuff
  integer, pointer, dimension(:),contiguous :: sbp
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==target_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_send_int_rank0 (potential deadlock)!")
  endif
  nullify(sbp)
  call c_f_pointer( c_loc(sendbuff), sbp, [1] )
  call dlf_mpi_send_int_rank1(sbp,1,target_rank,tag)
  nullify(sbp)
  return
end subroutine dlf_mpi_send_int_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_send_log_rank0
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Send
!! Logical data in scalar 'sendbuff' (size: n) is sent to 'target_rank'
!! with the tag passed in dummy argument 'tag'.
!! This subroutine must be matched with a corresponding mpi_recv call
!! on the receiving MPI rank ('taget_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!!
!! SYNOPSIS
subroutine dlf_mpi_send_log_rank0(sendbuff,target_rank,tag)
!! SOURCE
  use dlf_global, only: glob
  use, intrinsic :: ISO_C_BINDING
  implicit none

  integer, intent(in) :: target_rank, tag
  logical, intent(in), target :: sendbuff
  logical, pointer, dimension(:),contiguous :: sbp
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==target_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_send_log_rank0 (potential deadlock)!")
  endif
  nullify(sbp)
  call c_f_pointer( c_loc(sendbuff), sbp, [1] )
  call dlf_mpi_send_log_rank1(sbp,1,target_rank,tag)
  nullify(sbp)
  return
end subroutine dlf_mpi_send_log_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_recv_real_rank1
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Recv
!! Real data is received from process with MPI rank 'source_rank',
!! expecting the message tag given by dummy argument 'tag'.
!! The transmitted data (length='n') is stored in the array 'recvbuff'.
!! 'recv_status' is populated with information that can be subsequently
!! used; this argument is optional.
!! This subroutine must be matched with a corresponding mpi_send call
!! on the sending MPI rank ('source_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!! Negative values of 'source_rank' or 'tag' trigger the use of 
!! 'MPI_ANY_SOURCE' or 'MPI_ANY_TAG', so that the receiving process
!! will accept messages from any sender and/or with any message tag.
!! This has to be used with extreme caution because deadlocks can be
!! created very easily.
!! In the latter case, the dummy arguments 'source_rank' and/or 'tag'
!! are overwritten with the actual source/tag of the received message.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! mpi_rk
!! global_comm
!! mpi_status_size
!! mpi_any_source
!! mpi_any_tag
!!
!! SYNOPSIS
subroutine dlf_mpi_recv_real_rank1(recvbuff,n,source_rank,tag,recv_status)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer, intent(in) :: n
  integer, intent(inout) :: source_rank, tag
  real(rk),dimension(n), intent(out) :: recvbuff
  integer, dimension(MPI_STATUS_SIZE), intent(out),optional :: recv_status
  integer, dimension(MPI_STATUS_SIZE) :: recv_status_internal
  integer :: ierr,effective_rank,effective_tag
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==source_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_recv_real_rank1 (potential deadlock)!")
  endif
  effective_rank=source_rank
  if (source_rank<0) then
    effective_rank=MPI_ANY_SOURCE
  endif
  effective_tag=tag
  if (tag<0) then
    effective_tag=MPI_ANY_TAG
  endif
  call MPI_Recv(recvbuff,n,mpi_rk,effective_rank,effective_tag,global_comm,recv_status_internal,ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_recv in dlf_mpi_recv_real_rank1")
  if (source_rank<0) then
    source_rank=recv_status_internal(MPI_SOURCE)
  endif
  if (tag<0) then
    tag=recv_status_internal(MPI_TAG)
  endif
  if (present(recv_status)) then
    recv_status=recv_status_internal
  endif
  return
end subroutine dlf_mpi_recv_real_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_recv_int_rank1
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Recv
!! Integer data is received from process with MPI rank 'source_rank',
!! expecting the message tag given by dummy argument 'tag'.
!! The transmitted data (length='n') is stored in the array 'recvbuff'.
!! 'recv_status' is populated with information that can be subsequently
!! used; this argument is optional.
!! This subroutine must be matched with a corresponding mpi_send call
!! on the sending MPI rank ('source_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!! Negative values of 'source_rank' or 'tag' trigger the use of 
!! 'MPI_ANY_SOURCE' or 'MPI_ANY_TAG', so that the receiving process
!! will accept messages from any sender and/or with any message tag.
!! This has to be used with extreme caution because deadlocks can be
!! created very easily.
!! In the latter case, the dummy arguments 'source_rank' and/or 'tag'
!! are overwritten with the actual source/tag of the received message.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! mpi_ik
!! global_comm
!! mpi_status_size
!! mpi_any_source
!! mpi_any_tag
!!
!! SYNOPSIS
subroutine dlf_mpi_recv_int_rank1(recvbuff,n,source_rank,tag,recv_status)
!! SOURCE
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer, intent(in) :: n
  integer, intent(inout) :: source_rank, tag
  integer,dimension(n), intent(out) :: recvbuff
  integer, dimension(MPI_STATUS_SIZE), intent(out),optional :: recv_status
  integer, dimension(MPI_STATUS_SIZE) :: recv_status_internal
  integer :: ierr,effective_rank,effective_tag
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==source_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_recv_int_rank1 (potential deadlock)!")
  endif
  effective_rank=source_rank
  if (source_rank<0) then
    effective_rank=MPI_ANY_SOURCE
  endif
  effective_tag=tag
  if (tag<0) then
    effective_tag=MPI_ANY_TAG
  endif
  call MPI_Recv(recvbuff,n,mpi_ik,effective_rank,effective_tag,global_comm,recv_status_internal,ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_recv in dlf_mpi_recv_int_rank1")
  if (source_rank<0) then
    source_rank=recv_status_internal(MPI_SOURCE)
  endif
  if (tag<0) then
    tag=recv_status_internal(MPI_TAG)
  endif
  if (present(recv_status)) then
    recv_status=recv_status_internal
  endif
  return
end subroutine dlf_mpi_recv_int_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_recv_log_rank1
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Recv
!! Logical data is received from process with MPI rank 'source_rank',
!! expecting the message tag given by dummy argument 'tag'.
!! The transmitted data (length='n') is stored in the array 'recvbuff'.
!! 'recv_status' is populated with information that can be subsequently
!! used; this argument is optional.
!! This subroutine must be matched with a corresponding mpi_send call
!! on the sending MPI rank ('source_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!! Negative values of 'source_rank' or 'tag' trigger the use of 
!! 'MPI_ANY_SOURCE' or 'MPI_ANY_TAG', so that the receiving process
!! will accept messages from any sender and/or with any message tag.
!! This has to be used with extreme caution because deadlocks can be
!! created very easily.
!! In the latter case, the dummy arguments 'source_rank' and/or 'tag'
!! are overwritten with the actual source/tag of the received message.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! mpi_logical
!! global_comm
!! mpi_status_size
!! mpi_any_source
!! mpi_any_tag
!!
!! SYNOPSIS
subroutine dlf_mpi_recv_log_rank1(recvbuff,n,source_rank,tag,recv_status)
!! SOURCE
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer, intent(in) :: n
  integer, intent(inout) :: source_rank, tag
  logical,dimension(n), intent(out) :: recvbuff
  integer, dimension(MPI_STATUS_SIZE), intent(out),optional :: recv_status
  integer, dimension(MPI_STATUS_SIZE) :: recv_status_internal
  integer :: ierr,effective_rank,effective_tag
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==source_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_recv_log_rank1 (potential deadlock)!")
  endif
  effective_rank=source_rank
  if (source_rank<0) then
    effective_rank=MPI_ANY_SOURCE
  endif
  effective_tag=tag
  if (tag<0) then
    effective_tag=MPI_ANY_TAG
  endif
  call MPI_Recv(recvbuff,n,mpi_logical,effective_rank,effective_tag,global_comm,recv_status_internal,ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_recv in dlf_mpi_recv_log_rank1")
  if (source_rank<0) then
    source_rank=recv_status_internal(MPI_SOURCE)
  endif
  if (tag<0) then
    tag=recv_status_internal(MPI_TAG)
  endif
  if (present(recv_status)) then
    recv_status=recv_status_internal
  endif
  return
end subroutine dlf_mpi_recv_log_rank1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_recv_char_string
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Recv
!! Character string data is received from process with MPI rank 
!! 'source_rank', expecting the message tag given by dummy argument 'tag'.
!! The transmitted data is stored in the scalar char string 'recvbuff'.
!! 'recv_status' is populated with information that can be subsequently
!! used; this argument is optional.
!! This subroutine must be matched with a corresponding mpi_send call
!! on the sending MPI rank ('source_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!! Negative values of 'source_rank' or 'tag' trigger the use of 
!! 'MPI_ANY_SOURCE' or 'MPI_ANY_TAG', so that the receiving process
!! will accept messages from any sender and/or with any message tag.
!! This has to be used with extreme caution because deadlocks can be
!! created very easily.
!! In the latter case, the dummy arguments 'source_rank' and/or 'tag'
!! are overwritten with the actual source/tag of the received message.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! mpi_character
!! global_comm
!! mpi_status_size
!! mpi_any_source
!! mpi_any_tag
!!
!! SYNOPSIS
subroutine dlf_mpi_recv_char_string(recvbuff,source_rank,tag,recv_status)
!! SOURCE
  use dlf_global, only: glob
  use dlf_mpi_module
  implicit none

  integer, intent(inout) :: source_rank, tag
  character(len=*), intent(out) :: recvbuff
  integer, dimension(MPI_STATUS_SIZE), intent(out),optional :: recv_status
  integer, dimension(MPI_STATUS_SIZE) :: recv_status_internal
  integer :: ierr,effective_rank,effective_tag,charlen
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==source_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_recv_log_rank1 (potential deadlock)!")
  endif
  charlen=len(recvbuff)
  effective_rank=source_rank
  if (source_rank<0) then
    effective_rank=MPI_ANY_SOURCE
  endif
  effective_tag=tag
  if (tag<0) then
    effective_tag=MPI_ANY_TAG
  endif
  call MPI_Recv(recvbuff,charlen,mpi_character,effective_rank,effective_tag,global_comm,recv_status_internal,ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_recv in dlf_mpi_recv_log_rank1")
  if (source_rank<0) then
    source_rank=recv_status_internal(MPI_SOURCE)
  endif
  if (tag<0) then
    tag=recv_status_internal(MPI_TAG)
  endif
  if (present(recv_status)) then
    recv_status=recv_status_internal
  endif
  return
end subroutine dlf_mpi_recv_char_string
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_recv_real_rank0
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Recv
!! Real data is received from process with MPI rank 'source_rank',
!! expecting the message tag given by dummy argument 'tag'.
!! The transmitted data is stored in the scalar 'recvbuff'.
!! 'recv_status' is populated with information that can be subsequently
!! used; this argument is optional.
!! This subroutine must be matched with a corresponding mpi_send call
!! on the sending MPI rank ('source_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!! Negative values of 'source_rank' or 'tag' trigger the use of 
!! 'MPI_ANY_SOURCE' or 'MPI_ANY_TAG', so that the receiving process
!! will accept messages from any sender and/or with any message tag.
!! This has to be used with extreme caution because deadlocks can be
!! created very easily.
!! In the latter case, the dummy arguments 'source_rank' and/or 'tag'
!! are overwritten with the actual source/tag of the received message.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! mpi_status_size
!!
!! SYNOPSIS
subroutine dlf_mpi_recv_real_rank0(recvbuff,source_rank,tag,recv_status)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_mpi_module
  use, intrinsic :: ISO_C_BINDING
  implicit none
  interface
    subroutine dlf_mpi_recv_real_rank1(recvbuff,n,source_rank,tag,recv_status)
      use dlf_parameter_module, only: rk
      use dlf_mpi_module
      implicit none
      integer, intent(in) :: n
      integer, intent(inout) :: source_rank, tag
      real(rk),dimension(n), intent(out) :: recvbuff
      integer, dimension(MPI_STATUS_SIZE), intent(out),optional :: recv_status
    end subroutine dlf_mpi_recv_real_rank1
  end interface
  integer, intent(inout) :: source_rank, tag
  real(rk),intent(out),target :: recvbuff
  integer, dimension(MPI_STATUS_SIZE), intent(out),optional :: recv_status
  integer, dimension(MPI_STATUS_SIZE) :: recv_status_internal
  real(rk),dimension(:),pointer,contiguous:: rbp
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==source_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_recv_real_rank0 (potential deadlock)!")
  endif
  nullify(rbp)
  call c_f_pointer( c_loc(recvbuff), rbp, [1] )
  call dlf_mpi_recv_real_rank1(rbp,1,source_rank,tag,recv_status_internal)
  nullify(rbp)
  if (present(recv_status)) then
    recv_status=recv_status_internal
  endif
  return
end subroutine dlf_mpi_recv_real_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_recv_int_rank0
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Recv
!! Integer data is received from process with MPI rank 'source_rank',
!! expecting the message tag given by dummy argument 'tag'.
!! The transmitted data is stored in the scalar 'recvbuff'.
!! 'recv_status' is populated with information that can be subsequently
!! used; this argument is optional.
!! This subroutine must be matched with a corresponding mpi_send call
!! on the sending MPI rank ('source_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!! Negative values of 'source_rank' or 'tag' trigger the use of 
!! 'MPI_ANY_SOURCE' or 'MPI_ANY_TAG', so that the receiving process
!! will accept messages from any sender and/or with any message tag.
!! This has to be used with extreme caution because deadlocks can be
!! created very easily.
!! In the latter case, the dummy arguments 'source_rank' and/or 'tag'
!! are overwritten with the actual source/tag of the received message.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! mpi_status_size
!!
!! SYNOPSIS
subroutine dlf_mpi_recv_int_rank0(recvbuff,source_rank,tag,recv_status)
!! SOURCE
  use dlf_global, only: glob
  use dlf_mpi_module
  use, intrinsic :: ISO_C_BINDING
  implicit none
  interface
    subroutine dlf_mpi_recv_int_rank1(recvbuff,n,source_rank,tag,recv_status)
      use dlf_mpi_module
      implicit none
      integer, intent(in) :: n
      integer, intent(inout) :: source_rank, tag
      integer,dimension(n), intent(out) :: recvbuff
      integer, dimension(MPI_STATUS_SIZE), intent(out),optional :: recv_status
    end subroutine dlf_mpi_recv_int_rank1
  end interface
  integer, intent(inout) :: source_rank, tag
  integer,intent(out),target :: recvbuff
  integer, dimension(MPI_STATUS_SIZE), intent(out),optional :: recv_status
  integer, dimension(MPI_STATUS_SIZE) :: recv_status_internal
  integer,dimension(:),pointer,contiguous:: rbp
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==source_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_recv_int_rank0 (potential deadlock)!")
  endif
  nullify(rbp)
  call c_f_pointer( c_loc(recvbuff), rbp, [1] )
  call dlf_mpi_recv_int_rank1(rbp,1,source_rank,tag,recv_status_internal)
  nullify(rbp)
  if (present(recv_status)) then
    recv_status=recv_status_internal
  endif
  return
end subroutine dlf_mpi_recv_int_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_recv_log_rank0
!!
!! FUNCTION
!!
!! Blocking point-to-point communication via MPI_Recv
!! Logical data is received from process with MPI rank 'source_rank',
!! expecting the message tag given by dummy argument 'tag'.
!! The transmitted data is stored in the scalar 'recvbuff'.
!! 'recv_status' is populated with information that can be subsequently
!! used; this argument is optional.
!! This subroutine must be matched with a corresponding mpi_send call
!! on the sending MPI rank ('source_rank'), otherwise a deadlock will 
!! occur, or the program will exhibit erratic behavior.
!! Negative values of 'source_rank' or 'tag' trigger the use of 
!! 'MPI_ANY_SOURCE' or 'MPI_ANY_TAG', so that the receiving process
!! will accept messages from any sender and/or with any message tag.
!! This has to be used with extreme caution because deadlocks can be
!! created very easily.
!! In the latter case, the dummy arguments 'source_rank' and/or 'tag'
!! are overwritten with the actual source/tag of the received message.
!!
!! INPUTS
!!
!! local variables 
!! glob%nprocs
!! glob%iam
!! mpi_status_size
!!
!! SYNOPSIS
subroutine dlf_mpi_recv_log_rank0(recvbuff,source_rank,tag,recv_status)
!! SOURCE
  use dlf_global, only: glob
  use dlf_mpi_module
  use, intrinsic :: ISO_C_BINDING
  implicit none
  interface
    subroutine dlf_mpi_recv_log_rank1(recvbuff,n,source_rank,tag,recv_status)
      use dlf_mpi_module
      implicit none
      integer, intent(in) :: n
      integer, intent(inout) :: source_rank, tag
      logical,dimension(n), intent(out) :: recvbuff
      integer, dimension(MPI_STATUS_SIZE), intent(out),optional :: recv_status
    end subroutine dlf_mpi_recv_log_rank1
  end interface
  integer, intent(inout) :: source_rank, tag
  logical,intent(out),target :: recvbuff
  integer, dimension(MPI_STATUS_SIZE), intent(out),optional :: recv_status
  integer, dimension(MPI_STATUS_SIZE) :: recv_status_internal
  logical,dimension(:),pointer,contiguous:: rbp
! **********************************************************************
  if (glob%nprocs==1 .or. glob%iam==source_rank) then
    call dlf_fail("Wrong usage of dlf_mpi_recv_log_rank0 (potential deadlock)!")
  endif
  nullify(rbp)
  call c_f_pointer( c_loc(recvbuff), rbp, [1] )
  call dlf_mpi_recv_log_rank1(rbp,1,source_rank,tag,recv_status_internal)
  nullify(rbp)
  if (present(recv_status)) then
    recv_status=recv_status_internal
  endif
  return
end subroutine dlf_mpi_recv_log_rank0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_get_size
!!
!! FUNCTION
!!
!! Get size of global communicator (from mpi_comm_world)
!!
!! SYNOPSIS
subroutine dlf_mpi_get_size(isize)
!! SOURCE
  use dlf_mpi_module
  implicit none

  integer :: isize, ierr

! **********************************************************************

  call mpi_comm_size(mpi_comm_world, isize, ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_comm_size")
  return

end subroutine dlf_mpi_get_size
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_get_rank
!!
!! FUNCTION
!!
!! Get rank of process within global communicator (mpi_comm_world)
!!
!! SYNOPSIS
subroutine dlf_mpi_get_rank(irank)
!! SOURCE
  use dlf_mpi_module
  implicit none

  integer :: irank, ierr

! **********************************************************************

  call mpi_comm_rank(mpi_comm_world, irank, ierr)
  if (ierr /= 0) call dlf_fail("Failure in mpi_comm_rank")
  return

end subroutine dlf_mpi_get_rank
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* dlf_mpi/dlf_mpi_reset_global_comm
!!
!! FUNCTION
!!
!! Reset global_comm to mpi_comm_world
!!
!! SYNOPSIS
subroutine dlf_mpi_reset_global_comm()
!! SOURCE
  use dlf_mpi_module
  implicit none

! **********************************************************************
  global_comm=mpi_comm_world
  return

end subroutine dlf_mpi_reset_global_comm
!!****

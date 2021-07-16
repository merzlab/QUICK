! **********************************************************************
! **               Utility unit: run-time bookkeeping                 **
! **********************************************************************

! $Date$
! $Rev$
! $Author$
! $URL$
! $Id$

!! COPYRIGHT
!!
!!  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Tom Keal (thomas.keal@stfc.ac.uk)
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
module dlf_time
  use dlf_parameter_module, only: rk
  type time_type
    logical  :: running
    real(rk) :: accum_cpu_time
    real(rk) :: accum_wall_time
    real(rk) :: start_cpu_time
    real(rk) :: start_wall_time
  end type time_type
  integer,parameter    :: maxclock=6
  type(time_type),save :: clock(maxclock)
  logical,save         :: warning=.false.
end module dlf_time

subroutine get_cpu_time(time)
  ! return the cpu-time in seconds 
  use dlf_parameter_module, only: rk
  implicit none
  real(rk) ,intent(out) :: time
! **********************************************************************
  call cpu_time(time)
end subroutine get_cpu_time

subroutine get_wall_time(time)
  ! return the wall-clock time in seconds
  use dlf_parameter_module, only: rk
  implicit none
  real(rk) ,intent(out) :: time
  integer  :: count,count_rate,count_max
  integer,save :: lastcount=0
! **********************************************************************
  call system_clock(count,count_rate,count_max)
  if(count<lastcount) count=count+count_max
  time=dble(count)/dble(count_rate)
end subroutine get_wall_time

subroutine map_clock(name,number)
  use dlf_time
  implicit none
  character(*),intent(IN) :: name
  integer     ,intent(out):: number
  IF(NAME=="TOTAL") THEN
    NUMBER=1
  ELSE IF(NAME=="EANDG") THEN
    NUMBER=2
  ELSE IF(NAME=="FORMSTEP") THEN
    NUMBER=3
  ELSE IF(NAME=="COORDS") THEN
    NUMBER=4
  ELSE IF(NAME=="CHECKPOINT") THEN
    NUMBER=5
  ELSE IF(NAME=="XYZ") THEN
    NUMBER=6
  else 
    warning=.true.
    print*,"Warning: clock not recognised",name
    number=-1
  end if
end subroutine map_clock

subroutine clock_start(name)
  use dlf_time
  implicit none
  character(*),intent(IN) :: name
  integer                 :: number
  real(rk)                :: svar
! **********************************************************************
  call map_clock(name,number)
  if(number<=0) return
  if(clock(number)%running) then
    warning=.true.
    print*,"Warning: clock",name," already running"
    return
  end if
  clock(number)%running=.true.
  call get_cpu_time(svar)
  clock(number)%start_cpu_time=svar
  call get_wall_time(svar)
  clock(number)%start_wall_time=svar
end subroutine clock_start

subroutine clock_stop(name)
  use dlf_time
  implicit none
  character(*),intent(IN) :: name
  integer                 :: number
  real(rk)                :: svar,svar2
! **********************************************************************
  call get_cpu_time(svar)
  call get_wall_time(svar2)
  call map_clock(name,number)
  if(number<=0) return
  if(.not.clock(number)%running) then
    warning=.true.
    print*,"Warning: clock",name," not running"
    return
  end if
  clock(number)%running=.false.
  clock(number)%accum_cpu_time=clock(number)%accum_cpu_time+ &
      svar-clock(number)%start_cpu_time
  clock(number)%accum_wall_time=clock(number)%accum_wall_time+ &
      svar2-clock(number)%start_wall_time
end subroutine clock_stop

subroutine time_init
  use dlf_time
  implicit none
! **********************************************************************
  clock(:)%running=.false.
  clock(:)%accum_cpu_time=0.D0
  clock(:)%accum_wall_time=0.D0
end subroutine time_init

subroutine time_report
  use dlf_time
  use dlf_global, only: stdout,printl,glob
  implicit none
  character(50)   :: descr(maxclock)
  integer         :: iclock
  real(rk)        :: maxcpu,maxwall,fraction_cpu,fraction_wall
! **********************************************************************
  descr(1)="Total"
  descr(2)="Energy and Gradient"
  descr(3)="Step direction"
  descr(4)="Coordinate transformation"
  descr(5)="Checkpoint file I/O"
  descr(6)="XYZ-file I/O"

! can't return as dlf_mpi_time requires participation
!  if(printl<=0) return
  if (printl > 0) then
     write(stdout,'(/,"Timing report")')
     write(stdout,'("=============")')
     write(stdout,'("Module                       &
         &               CPU time            Wall clock time")')
     maxcpu=maxval(clock(:)%accum_cpu_time)
     maxwall=maxval(clock(:)%accum_wall_time)
     do iclock=1,maxclock
       if(.not.clock(iclock)%running) then
         fraction_cpu=0.D0
         if(maxcpu > 0.D0) &
             fraction_cpu=clock(iclock)%accum_cpu_time*100.D0/maxcpu
         fraction_wall=0.D0
         if(maxwall > 0.D0) &
             fraction_wall=clock(iclock)%accum_wall_time*100.D0/maxwall
         write(stdout,1000) TRIM(descr(iclock)), &
             clock(iclock)%accum_cpu_time, fraction_cpu,&
             clock(iclock)%accum_wall_time,fraction_wall
       else
         write(stdout,'(a," is still running")') TRIM(descr(iclock))
       end if
     end do
  end if

  if (glob%nprocs > 1 .and. glob%ntasks > 1) then
     if (glob%iam == 0 .and. printl > 0) then
        write(stdout,*)
        write(stdout,'(a)')"Task-farming statistics:"
        write(stdout,'("Module                                      &
              &Total     Mean      St.dev.   Max       Min")')
     end if
     do iclock=1,maxclock
       if(.not.clock(iclock)%running) call dlf_mpi_time(descr(iclock), &
         clock(iclock)%accum_cpu_time, clock(iclock)%accum_wall_time)
     end do
     if (glob%iam == 0 .and. printl > 0) then
        write(stdout,*)
        write(stdout,'(a)')"Task-farming statistics: end"
     end if
  end if

1000 format (t1,"................................................", &
         t1,a," ",t40,2(f10.3," (",f6.2,"%)")," seconds")
end subroutine time_report

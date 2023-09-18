! a object oriented clock
! you can start it/stop it/write a checkpoint
! name it, write a message to the time output
module mod_oop_clock
implicit none
  type clock_type
    real(kind=8)                ::  t_start
    real(kind=8)                ::  t_beat
    real(kind=8)                ::  t_checkpoint
    real(kind=8)                ::  cutoff
    character(len=50)           ::  clockname
    contains 
      procedure                 ::  start => oop_clock_start
      procedure                 ::  checkpt => oop_clock_checkpt
      ! write checkpt since last beat
      procedure                 ::  beat => oop_clock_beat
      procedure                 ::  destroy => oop_clock_destroy
      ! restart the clock with a possibly new name
      procedure                 ::  restart => oop_clock_restart
      ! writes a checkpoint with a message, and then restarts the 
      ! clock with a possibly new name
      procedure                 ::  nextClock => oop_clock_nextClock
      ! set a cutoff, times below that cutoff are not displayed
      procedure                 ::  setCutOff => oop_clock_setCutOff
  end type clock_type
  contains
  subroutine oop_clock_start(clock, name, cutoff)
    class(clock_type)         ::  clock
    character(*), intent(in)  ::  name
    integer                   ::  name_length
    real(kind=8), intent(in),optional :: cutoff
    if (present(cutoff)) then
      clock%cutoff = cutoff
    else
      clock%cutoff = 0d0
    end if
    clock%clockname = TRIM(name)
    call CPU_TIME(clock%t_start)
    clock%t_beat = clock%t_start
  end subroutine oop_clock_start
    
  subroutine oop_clock_checkpt(clock, msg)
    class(clock_type)         ::  clock
    character(*), intent(in)  ::  msg
    call CPU_TIME(clock%t_checkpoint)

    if (clock%t_checkpoint-clock%t_beat>clock%cutoff) &
      write(*,'(A,A,A,F16.10)') "%%Clock : ", &
      msg, " - ", clock%t_checkpoint-clock%t_start
  end subroutine oop_clock_checkpt

  subroutine oop_clock_beat(clock, msg)
    class(clock_type)         ::  clock
    character(*), intent(in)  ::  msg
    call CPU_TIME(clock%t_checkpoint)
    if (clock%t_checkpoint-clock%t_beat>clock%cutoff) &
      write(*,'(A,A,A,F16.10)') "%%Clock: ", &
      msg, " - ", clock%t_checkpoint-clock%t_beat
    clock%t_beat = clock%t_checkpoint
  end subroutine oop_clock_beat
  
  subroutine oop_clock_destroy(clock)
    class(clock_type)         ::  clock
    clock%clockname = ""
  end subroutine oop_clock_destroy
  
  subroutine oop_clock_restart(clock, name)
    class(clock_type)         ::  clock
    character(*), intent(in)  ::  name
    integer                   ::  name_length
    call clock%destroy()
    call clock%start(name)
  end subroutine oop_clock_restart
  
  subroutine oop_clock_nextClock(clock, msg, name)
    class(clock_type)         ::  clock
    character(*), intent(in)  ::  name
    character(*), intent(in)  ::  msg
    call clock%checkpt(msg)
    call clock%restart(name)
  end subroutine oop_clock_nextClock
  
  subroutine oop_clock_setCutOff(clock,cutoff)
    class(clock_type)         ::  clock
    real(kind=8)              ::  cutoff
    clock%cutoff = cutoff
  end subroutine oop_clock_setCutOff
end module mod_oop_clock

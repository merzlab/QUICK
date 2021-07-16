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
module dlf_parameter_module
  integer, parameter :: rk = kind(1.d0) ! read kind
  integer, parameter :: ik = kind(1)    ! integer kind (for memory tracking)
end module dlf_parameter_module

module dlf_stat
  implicit none
  type stat_type
    integer    :: sene=0      ! number of energy evaluation tasks
    integer    :: pene=0      ! number of energy evaluations on current processor
    integer    :: ccycle=0    ! number of cycles/steps in the current run
    integer    :: caccepted=0 ! number of accepted steps in the current run
    integer    :: miccycle=0  ! number of microiterative steps in current macro cycle
    integer    :: tmiccycle=0 ! total number of microiterative steps
    integer    :: tmicaccepted=0 ! total number of accepted microiterations
  end type stat_type
  type(stat_type),save :: stat
end module dlf_stat

subroutine dlf_stat_reset
  use dlf_stat, only: stat
  implicit none
  stat%ccycle=0
  stat%caccepted=0 
  stat%miccycle = 0
  stat%tmiccycle = 0
  stat%tmicaccepted = 0
end subroutine dlf_stat_reset

! **********************************************************************
! **                  Utility functions to DL-FIND                    **
! **                                                                  **
! **  Several sub-files:                                              **
! **   dlf_allocate                                                   **
! **   dlf_checkpoint                                                 **
! **   dlf_linalg                                                     **
! **   dlf_time                                                       **
! **                                                                  **
! **********************************************************************
!!****h* DL-FIND/utilities
!!
!! NAME
!! utilities
!!
!! FUNCTION
!! Utility functions to DL-FIND
!!
!! DATA
!! $Date$
!! $Rev$
!! $Author$
!! $URL$
!! $Id$
!!
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
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
character(2) function get_atom_symbol(atomic_number)
  implicit none
  integer, intent(in) :: atomic_number
  character(2), parameter :: elements(111) = &
       (/ 'H ','He', &
          'Li','Be','B ','C ','N ','O ','F ','Ne', &
          'Na','Mg','Al','Si','P ','S ','Cl','Ar', &
          'K ','Ca','Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu', &
          'Zn','Ga','Ge','As','Se','Br','Kr', &
          'Rb','Sr','Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag', &
          'Cd','In','Sn','Sb','Te','I ','Xe', &
          'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy', &
          'Ho','Er','Tm','Yb','Lu','Hf','Ta','W ','Re','Os','Ir','Pt', &
          'Au','Hg','Tl','Pb','Bi','Po','At','Rn', &
          'Fr','Ra','Ac','Th','Pa','U ','Np','Pu','Am','Cm','Bk','Cf', &
          'Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds', &
          'Rg' /)
! **********************************************************************
  if (atomic_number >= 1 .and. atomic_number <= size(elements)) then
     get_atom_symbol = elements(atomic_number)
  else
     get_atom_symbol = 'XX'
  endif
end function get_atom_symbol 

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
module dlf_bspline

  ! creates a set of cubic spline functions with the same number of 
  ! grid points (length).
  ! Their x-values may be different
  !
  ! Calling-Sequence:
  !   call spline_init
  !   call spline_create once for each function (each ifunc)
  !   call spline_get for each interpolated value
  !   call spline_destroy

  use dlf_parameter_module, only: rk
  use dlf_allocate, only: allocate,deallocate
  implicit none

  public :: spline_init, spline_create, spline_get, spline_destroy, spline_create_clamped

  interface spline_init
    module procedure spline_init
  end interface

  interface spline_create
    module procedure spline_create
    module procedure spline_create_nonnatural
  end interface

  interface spline_create_clamped
    module procedure spline_create_clamped
  end interface

  interface spline_get
    module procedure spline_get
  end interface

  interface spline_destroy
    module procedure spline_destroy
  end interface

  private
  logical :: tinit
  integer :: nfunc ! number of functions to be interpolated
  integer :: length ! number of values to be interpolated between
  logical, allocatable, save :: created(:) ! (nfunc)
  real(rk),allocatable, save :: gridx(:,:) ! (length,nfunc) !x values of grid
  real(rk),allocatable, save :: gridy(:,:) ! (length,nfunc) !y values of grid
  real(rk),allocatable, save :: grid_d2ydx2(:,:) ! (length,nfunc)
contains

  subroutine spline_init(length_in,nfunc_in)
    integer, intent(in) :: length_in 
    integer, intent(in) :: nfunc_in 
    ! ******************************************************************

    ! re-initialise if necessary
    if(allocated(created)) call spline_destroy

    nfunc=nfunc_in
    length=length_in
    if(nfunc*length<=0) call dlf_fail("wrong parameters to spline_init")

    tinit=.true.
    call allocate(created,nfunc)
    call allocate(gridx,length,nfunc)
    call allocate(gridy,length,nfunc)
    call allocate(grid_d2ydx2,length,nfunc)

    created(:)=.false.

  end subroutine spline_init

  subroutine spline_destroy
    ! ******************************************************************
    tinit=.false.
    call deallocate(created)
    call deallocate(gridx)
    call deallocate(gridy)
    call deallocate(grid_d2ydx2)

  end subroutine spline_destroy

  SUBROUTINE spline_create(ifunc,x_in,y_in)
    ! creates the spline function number ifunc, i.e. calculates its 
    ! second derivatives at the grid points
    ! x_in must be monotonically increasing
    ! using natural boundary conditions
    ! ******************************************************************
    integer , intent(in) :: ifunc
    real(rk), intent(in) :: x_in(length),y_in(length)
    !
    real(rk) :: store(length)! u
    real(rk) :: svar, svar2  ! sig, p
    integer  :: ival
    ! ******************************************************************

    ! check integrity
    if(.not.tinit) call dlf_fail("spline_create must not be called &
        &before spline_init!")
    if(ifunc<1) call dlf_fail("ifunc < 1 in spline_create")
    if(ifunc>nfunc) call dlf_fail("ifunc > nfunc in spline_create")
    
    gridx(:,ifunc)=x_in(:)
    gridy(:,ifunc)=y_in(:)

    ! natural boundaries
    grid_d2ydx2(1,ifunc)=0.D0
    store(1)=0.D0

    do ival=2,length-1
      svar=(gridx(ival,ifunc)-gridx(ival-1,ifunc))/(gridx(ival+1,ifunc)-gridx(ival-1,ifunc))
      svar2=svar*grid_d2ydx2(ival-1,ifunc)+2.D0
      grid_d2ydx2(ival,ifunc)=(svar-1.D0)/svar2
      store(ival)=(6.D0*((gridy(ival+1,ifunc)-gridy(ival,ifunc))/(gridx(ival+1,ifunc)- &
          gridx(ival,ifunc))-(gridy(ival,ifunc)-gridy(ival-1,ifunc)) /(gridx(ival,ifunc)- &
          gridx(ival-1,ifunc)))/(gridx(ival+1,ifunc)-gridx(ival-1,ifunc))- &
          svar*store(ival-1))/svar2
    enddo

    ! natural boundaries
    grid_d2ydx2(length,ifunc)=0.D0

    do ival=length-1,1,-1 
      grid_d2ydx2(ival,ifunc)=grid_d2ydx2(ival,ifunc)*grid_d2ydx2(ival+1,ifunc)+store(ival) 
    enddo

    created(ifunc)=.true.

  END SUBROUTINE spline_create

  SUBROUTINE spline_create_nonnatural(ifunc,x_in,y_in,d2y_low,d2y_high)
    ! creates the spline function number ifunc, i.e. calculates its 
    ! second derivatives at the grid points
    ! x_in must be monotonically increasing
    ! using natural boundary conditions
    ! ******************************************************************
    integer , intent(in) :: ifunc
    real(rk), intent(in) :: x_in(length),y_in(length)
    real(rk), intent(in) :: d2y_low,d2y_high
    !
    real(rk) :: store(length)! u
    real(rk) :: svar, svar2  ! sig, p
    integer  :: ival
    ! ******************************************************************

    ! check integrity
    if(.not.tinit) call dlf_fail("spline_create must not be called &
        &before spline_init!")
    if(ifunc<1) call dlf_fail("ifunc < 1 in spline_create")
    if(ifunc>nfunc) call dlf_fail("ifunc > nfunc in spline_create")
    
    gridx(:,ifunc)=x_in(:)
    gridy(:,ifunc)=y_in(:)

    !print*,"JK dy2 low hi",d2y_low,d2y_high

    ! non-natural boundaries
    grid_d2ydx2(1,ifunc)=d2y_low
    store(1)=d2y_low

    do ival=2,length-1
      svar=(gridx(ival,ifunc)-gridx(ival-1,ifunc))/(gridx(ival+1,ifunc)-gridx(ival-1,ifunc))
      svar2=svar*grid_d2ydx2(ival-1,ifunc)+2.D0
      grid_d2ydx2(ival,ifunc)=(svar-1.D0)/svar2
      store(ival)=(6.D0*((gridy(ival+1,ifunc)-gridy(ival,ifunc))/(gridx(ival+1,ifunc)- &
          gridx(ival,ifunc))-(gridy(ival,ifunc)-gridy(ival-1,ifunc)) /(gridx(ival,ifunc)- &
          gridx(ival-1,ifunc)))/(gridx(ival+1,ifunc)-gridx(ival-1,ifunc))- &
          svar*store(ival-1))/svar2
    enddo

    ! non-natural boundaries
    grid_d2ydx2(length,ifunc)=d2y_high

    do ival=length-1,1,-1 
      grid_d2ydx2(ival,ifunc)=grid_d2ydx2(ival,ifunc)*grid_d2ydx2(ival+1,ifunc)+store(ival) 
    enddo

    !print*,"store 1 len",store(1),store(length)
    !print*,"grid_d2ydx2( 1, length",grid_d2ydx2(1,ifunc),grid_d2ydx2(length,ifunc)

    created(ifunc)=.true.

  END SUBROUTINE spline_create_nonnatural

  SUBROUTINE spline_create_clamped(ifunc,x_in,y_in,dy_low,dy_high)
    ! creates the spline function number ifunc, i.e. calculates its 
    ! second derivatives at the grid points
    ! x_in must be monotonically increasing
    ! The two extral conditions are the first derivatives at the endpoints
    ! ******************************************************************
    integer , intent(in) :: ifunc
    real(rk), intent(in) :: x_in(length),y_in(length)
    real(rk), intent(in) :: dy_low,dy_high
    !
    real(rk) :: amat(length,length)
    real(rk) :: bvec(length)
    real(rk) :: svar
    integer  :: ival,jval
    ! ******************************************************************

    ! check integrity
    if(.not.tinit) call dlf_fail("spline_create must not be called &
        &before spline_init!")
    if(ifunc<1) call dlf_fail("ifunc < 1 in spline_create")
    if(ifunc>nfunc) call dlf_fail("ifunc > nfunc in spline_create")
    
    gridx(:,ifunc)=x_in(:)
    gridy(:,ifunc)=y_in(:)

    !print*,"JK dy low hi",dy_low,dy_high

    amat(:,:)=0.D0
    bvec(:)=0.D0
    ! conditions of C2-continuity:
    do ival=2,length-1
      amat(ival,ival-1)=x_in(ival)-x_in(ival-1)
      amat(ival,ival  )=2.D0*(x_in(ival+1)-x_in(ival-1))
      amat(ival,ival+1)=x_in(ival+1)-x_in(ival)
      bvec(ival)=3.D0*( (y_in(ival+1)-y_in(ival))/(x_in(ival+1)-x_in(ival)) - &
          (y_in(ival)-y_in(ival-1))/(x_in(ival)-x_in(ival-1)) )
    end do

    ! now first derivatives at bounds:
    amat(1,1)=-2.D0*(x_in(2)-x_in(1))/3.D0
    amat(1,2)=-(x_in(2)-x_in(1))/3.D0
    bvec(1)=dy_low-(y_in(2)-y_in(1))/(x_in(2)-x_in(1))

    amat(length,length-1)=(x_in(length)-x_in(length-1))/3.D0
    amat(length,length)=2.D0*(x_in(length)-x_in(length-1))/3.D0
    bvec(length)=dy_high-(y_in(length)-y_in(length-1))/(x_in(length)-x_in(length-1))
!!$
!!$    ! natural:
!!$    amat(1,1)=1.D0
!!$    amat(length,length)=1.D0
    

    ! solve Ax=B
    call dlf_matrix_invert(length,.false.,amat,svar) ! this is really an overkill ...
    !print*,"Determinant",svar
    grid_d2ydx2(:,ifunc)=2.D0*matmul(amat,bvec)
    !  grid_d2ydx2(:,ifunc)= 2.D0*x
    

    created(ifunc)=.true.

  END SUBROUTINE spline_create_clamped

  SUBROUTINE spline_get(ifunc,xval,yval,dyval,d2yval)
    ! calculates a cubic-spline interpolated value (yval) and its 
    ! derivative (dyval) at a position xval
    ! this is done for the interpolation ifunc
    ! ******************************************************************
    integer ,intent(in) :: ifunc
    real(rk),intent(in) :: xval
    real(rk),intent(out):: yval
    real(rk),intent(out):: dyval
    real(rk),intent(out):: d2yval
    ! 
    integer  :: low,high,ivar
    real(rk) :: aval,bval,delta
    ! ******************************************************************

    ! check integrity
    if(.not.tinit)  call dlf_fail("spline_get must not be called before &
        &spline_init!")
    if(ifunc<1) call dlf_fail("ifunc < 1 in spline_get")
    if(ifunc>nfunc) call dlf_fail("ifunc > nfunc in spline_get")
    if(.not.created(ifunc)) then
      print*,"Number of spline:",ifunc
      call dlf_fail("spline_get must not be called&
        & before spline_create!")
    end if
    
    low=1 
    ! bisection on the grid
    high=length
    do while (high-low>1) 
      ivar=(high+low)/2
      if(gridx(ivar,ifunc).gt.xval)then
        high=ivar
      else
        low=ivar
      endif
    end do
    !low and high now bracket the input value of xval.

    delta=gridx(high,ifunc)-gridx(low,ifunc)

    if (delta<=0.D0) call dlf_fail("grid points for spline not distinct")

    aval=(gridx(high,ifunc)-xval)/delta !Cubic spline polynomial is now evaluated.
    bval=(xval-gridx(low,ifunc))/delta
    yval=aval*gridy(low,ifunc)+bval*gridy(high,ifunc) + ((aval**3-aval)* &
        grid_d2ydx2(low,ifunc)+(bval**3-bval)*grid_d2ydx2(high,ifunc)) &
        *(delta**2)/6.D0
    dyval=1.D0/delta * ( gridy(high,ifunc) - gridy(low,ifunc) + delta**2/6.D0 * ( &
        (3.D0*bval**2-1.D0) * grid_d2ydx2(high,ifunc) - &
        (3.D0*aval**2-1.D0) * grid_d2ydx2(low,ifunc) ) ) 
    d2yval=bval * grid_d2ydx2(high,ifunc) + &
         aval * grid_d2ydx2(low,ifunc) 
  END SUBROUTINE spline_get

end module dlf_bspline
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
module dlf_store

  ! Allocates bunches of memory, to store (mainly real number) data
  ! The data are organised in a linked list. They can be set, read out
  ! and deleted
  !
  use dlf_parameter_module, only: rk
  use dlf_global, only: stderr,stdout
  use dlf_allocate, only: allocate,deallocate
  implicit none

  public :: store_initialise, store_allocate, store_set, store_get, &
      store_delete, store_delete_all

  interface store_initialise
    module procedure store_initialise
  end interface

  interface store_allocate
    module procedure store_allocate
  end interface

  interface store_set
    module procedure store_set
    module procedure store_set_a2
    module procedure store_set_a3
  end interface

  interface store_get
    module procedure store_get
    module procedure store_get_a2
    module procedure store_get_a3
  end interface

  interface store_delete
    module procedure store_delete
  end interface

  interface store_delete_all
    module procedure store_delete_all
  end interface

  private

  type store_type_R
    character(40)              :: tag
    integer                    :: size
    real(rk),pointer           :: array(:)
    type(store_type_R),pointer :: next
  end type store_type_R

  type(store_type_R),pointer,save :: first_R
  logical                   ,save :: tinit=.false.

contains

  subroutine store_initialise
    ! ******************************************************************
    if(tinit) call dlf_fail("store is already initialised")
    allocate(first_R)
    nullify(first_R%next)
    nullify(first_R%array)
    first_R%size=0
    first_R%tag=""
    tinit=.true.
  end subroutine store_initialise

  subroutine store_allocate(tag,size)
    character(*)      ,intent(in) :: tag
    integer           ,intent(in) :: size
    type(store_type_R),pointer    :: this
    ! ******************************************************************
    this => first_R
    !print*,"This%size 1",This%size
    ! find last used entry
    do while (associated(this%next))
      !print*," allocate search: this%tag ",this%tag
      if(this%tag == tag) call dlf_fail("Store tag aleady allocated")
      this => this%next
    end do
    ! this is now last set entry
    !print*,"This%size 2",This%size

    ! if this is the first entry, and it is empty, use it, otherwhise use the next
    if(this%size > 0) then
      allocate(this%next)
      this => this%next
      nullify(this%next)
      nullify(this%array)
    end if

    ! set tag and size
    this%tag=tag
    this%size=size
    !print*,"allocated(this%array)",associated(this%array)
    if(associated(this%array)) print*,"shape(this%array)",shape(this%array)
    allocate (this%array(size)) ! pointer

  end subroutine store_allocate

  subroutine store_set(tag,size,array)
    character(*)      ,intent(in) :: tag
    integer           ,intent(in) :: size
    real(rk)          ,intent(in) :: array(size)
    type(store_type_R),pointer    :: this
    ! ******************************************************************

    ! find correct entry
    this => first_R
    do while (associated(this))
      !print*," set search: this%tag ",this%tag
      if(this%tag == tag) exit
      this => this%next
    end do

    if(.not.associated(this)) then
      write(stdout,*) "Storage tag ",tag," not found!"
      call dlf_fail("Storage tag to set not found")
    end if

    if(this%size /= size) call dlf_fail("Storage set size inconsistent")

    this%array(:)=array(:)

    !print*,"Tag ",tag," now set"

  end subroutine store_set

  subroutine store_get(tag,size,array)
    character(*)      ,intent(in) :: tag
    integer           ,intent(in) :: size
    real(rk)          ,intent(out):: array(size)
    type(store_type_R),pointer    :: this
    ! ******************************************************************

    ! find correct entry
    this => first_R
    do while (associated(this))
      !print*," get search: this%tag ",this%tag
      if(this%tag == tag) exit
      this => this%next
    end do

    if(.not.associated(this)) then
      write(stdout,*) "Storage tag ",tag," not found!"
      call dlf_fail("Storage tag to get not found")
    end if

    if(this%size /= size) call dlf_fail("Storage get size inconsistent")

    array(:)=this%array(:)
    !print*,"Tag ",tag," now got"

  end subroutine store_get

  subroutine store_delete(tag)
    ! delete one tag
    character(*)      ,intent(in) :: tag
    type(store_type_R),pointer    :: this,del

    ! find correct entry
    this => first_R

    ! handle the case of deletion of the first entry separately
    if(this%tag == tag) then
      if(.not.associated(this%next)) then
        ! first entry is the only entry
        !print*," deleting first and only entry ",tag
        if(associated(this%array)) deallocate(this%array) ! pointer
        this%size=0
        this%tag=""
      else
        ! other entries exist
        !print*," deleting first entry ",tag
        first_R => this%next
        if(associated(this%array)) deallocate(this%array) ! pointer
        deallocate(this)
      end if
      return
    end if

    do while (associated(this%next))
      !print*," delete search: this%next%tag ",this%next%tag
      if(this%next%tag == tag) exit
      this => this%next
    end do
    ! this%next points to the tag to be deleted

    if(.not.associated(this%next)) then
      write(stdout,*) "Storage tag ",tag," not found!"
      call dlf_fail("Storage tag to delete not found")
    end if

    ! delete this%next
    this%next%size=0
    if(associated(this%next%array)) then
      deallocate(this%next%array) ! pointer
      !print*,"Deallocating tag ",this%next%tag
    else
      !print*,"Warning: not able to deallocate tag",tag
    end if

    if(associated(this%next%next)) then
      del => this%next%next
      deallocate(this%next)
      this%next => del
    else
      ! the one to be deleted is the last one
      deallocate(this%next)
      nullify(this%next)
    end if

  end subroutine store_delete

  subroutine store_delete_all
    ! delete all tags
    type(store_type_R),pointer    :: this,next

    if(.not.tinit) return

    this => first_R

    do while (associated(this%next))
      next => this%next
      !print*," deleteing this%tag ",this%tag

      if(associated(this%array)) deallocate(this%array) ! pointer
      deallocate(this)

      this => next
    end do

    ! now deallocate the last one
    !print*," deleteing this%tag ",this%tag

    if(associated(this%array)) deallocate(this%array) ! pointer
    deallocate(this)

    tinit=.false.

  end subroutine store_delete_all

  subroutine store_set_a2(tag,size,array)
    ! dummy routine to cover rank 2 arrays
    character(*)      ,intent(in) :: tag
    integer           ,intent(in) :: size
    real(rk)          ,intent(in) :: array(:,:)
    call store_set(tag,size,reshape(array,(/size/)))
  end subroutine store_set_a2

  subroutine store_set_a3(tag,size,array)
    ! dummy routine to cover rank 3 arrays
    character(*)      ,intent(in) :: tag
    integer           ,intent(in) :: size
    real(rk)          ,intent(in) :: array(:,:,:)
    call store_set(tag,size,reshape(array,(/size/)))
  end subroutine store_set_a3

  subroutine store_get_a2(tag,size1,size2,array)
    ! dummy routine to cover rank 2 arrays
    character(*)      ,intent(in) :: tag
    integer           ,intent(in) :: size1
    integer           ,intent(in) :: size2
    real(rk)          ,intent(out):: array(:,:)
    real(rk)  :: tmp_array(size1*size2)
    call store_get(tag,size1*size2,tmp_array)
    array=reshape(tmp_array,(/size1,size2/))
  end subroutine store_get_a2

  subroutine store_get_a3(tag,size1,size2,size3,array)
    ! dummy routine to cover rank 3 arrays
    character(*)      ,intent(in) :: tag
    integer           ,intent(in) :: size1
    integer           ,intent(in) :: size2
    integer           ,intent(in) :: size3
    real(rk)          ,intent(out):: array(:,:,:)
    real(rk)  :: tmp_array(size1*size2*size3)
    call store_get(tag,size1*size2*size3,tmp_array)
    array=reshape(tmp_array,(/size1,size2,size3/))
  end subroutine store_get_a3

end module dlf_store

!!****h* DL-FIND/constants
!!
!! NAME
!! constants
!!
!! FUNCTION
!! Provide numeric values for physical constants to other modules
!!
!! SOURCE
!!****
module dlf_constants
  use dlf_parameter_module, only: rk
  implicit none

  Public :: dlf_constants_get
  Public :: dlf_constants_init

  private
  ! Numeric values from http://physics.nist.gov/cuu/Constants/index.html
  !
  ! values of the atomic base units in SI units
  !
  real(rk),parameter :: hbar = 1.054571628E-34_rk
  real(rk),parameter :: echarge = 1.602176487E-19_rk
  real(rk),parameter :: emass = 9.10938215E-31_rk
  
  real(rk),parameter :: speed_of_light = 299792458.E0_rk !m/s

  ! exactly derived from speed_of_light
  !real(rk),parameter :: epsilon0 = 8.854187817E-12_rk ! electric constant

  !
  ! derived values in SI units
  !
  real(rk),parameter :: amc = 1.660538782E-27_rk ! atomic mass constant (kg)
  real(rk),parameter :: kboltz = 1.3806504e-23_rk  ! J/K
  real(rk),parameter :: avogadro = 6.02214179e23_rk ! 1/mol

  ! other numbers
  real(rk) :: pi
  real(rk) :: hartree ! 1 Hartree in J
  real(rk) :: one_over_4piep0
  real(rk) :: bohr ! a_0
  real(rk) :: mu0
  real(rk) :: epsilon0
  real(rk) :: planck
  ! numbers in atomic units
  real(rk) :: amu_au ! atomic mass units in at.u.
  real(rk) :: ang_au ! Angstrom in at.u. 
  real(rk) :: second_au ! Second in atomic time units

contains

  subroutine dlf_constants_init
    pi=4.E0_rk*atan(1.E0_rk)
    amu_au = amc/emass
    mu0 = 4.E0_rk * pi * 1.E-7_rk
    one_over_4piep0 = 1.E-7_rk * speed_of_light**2 !0.25E0_rk / pi / epsilon0
    epsilon0 = 0.25E0_rk / pi / one_over_4piep0
    hartree = emass * ( echarge**2 * one_over_4piep0 / hbar ) **2
    bohr = hbar**2 / (one_over_4piep0 * emass * echarge **2 )
    planck = hbar * 2.E0_rk * pi
    second_au = emass * echarge**4 * one_over_4piep0**2 / hbar**3
  end subroutine dlf_constants_init

  subroutine dlf_constants_get(tag,val)
    character(*) :: tag
    real(rk)     :: val
    select case (tag)
    case ("PI")
      val=pi      ! 3.1415926535898E+00
    case ("AMC")
      val=amc     ! 1.660538782E-27 
    case ("AMU")
      val=amu_au  ! 1.8228884842645E+03
    case ("ECHARGE")
      val=echarge ! 1.602176487E-19 C
    case ("HARTREE")
      val=hartree ! 4.3597439435066E-18 J
    case ("ANG_AU")
      val= bohr*1.E10_rk ! 5.2917720810086E-01
    case ("SOL")
      val= speed_of_light ! 2.9979245800000E+08
    case ("CM_INV_FOR_AMU") ! conversion between the square root of the second
                            ! derivative of the energy in Hartree with respect
                            ! to mass-weighted coordinates in AMU^1/2*A_0 and cm^-1
      val=sqrt(hartree/amc) / ( 2.E0_rk * pi * bohr * speed_of_light) / 100.E0_rk
    case("HBAR")
      val=hbar   ! 1.0545716280000E-34
    case("PLANCK")
      val=planck ! 6.6260689584181E-34
    case("KBOLTZ")
      val=kboltz ! 1.3806504000000E-23
    case("AVOGADRO")
      val=AVOGADRO ! 6.0221417900000E+23
    case("KBOLTZ_AU")
      val=kboltz/hartree ! 3.1668153407262E-06 
    case("SECOND_AU")
      val=second_au  ! 4.1341373379861E+16
    case("CM_INV_FROM_HARTREE")
      ! converts hartree to wave numbers (cm^-1)
      val=(hartree/planck)/speed_of_light*1.D-2  ! 219474.63160039327
    case default
      print*,"Tag not recognized:",tag
      print*,"Available tags (and their values):"
      call dlf_constants_report
      call dlf_fail("Wrong tag in dlf_constants_get")
    end select
  end subroutine dlf_constants_get
  
  subroutine dlf_constants_report
    write(*,1000) "Pi",pi
    write(*,1000) "hbar = h / (2pi)",hbar,"Js"
    write(*,1000) "h",planck,"Js"
    write(*,1000) "e",echarge,"C"
    write(*,1000) "Atomic mass unit (AMU)",amu_au,"m_e"
    write(*,1000) "Atomic mass constant (1/12 of the mass of one C12 atom)",amc,"kg"
    write(*,1000) "Hartree",hartree,"J"
    write(*,1000) "Bohr (a_0)",bohr,"m"
    write(*,1000) "Ang (ANG_AU)", bohr*1.E10_rk,"a_0"
    write(*,1000) "Speed of light",speed_of_light,"m/s"
    write(*,1000) "Epsilon0 (electric constant)",epsilon0,"F/m"
    write(*,1000) "1/(4 pi Epsilon0)",one_over_4piep0,"m^3/(H s^2)"
    write(*,1000) "cm^(-1) for AMU (CM_INV_FOR_AMU)",sqrt(hartree/amc) / &
        ( 2.E0_rk * pi * bohr * speed_of_light) / 100.E0_rk,""
    write(*,1000) "Boltzmann's constant (KBOLTZ)",kboltz,"J/K"
    write(*,1000) "Boltzmann's constant (KBOLTZ_AU)",kboltz/hartree,"Hartree/K"
    write(*,1000) "Avogadro constant (AVOGADRO)",AVOGADRO,"1/mol"
    write(*,1000) "Second in atomic units (SECOND_AU)",second_au,""
    write(*,1000) "Atomic time unit",1.E0_rk/second_au,"seconds"
    call dlf_fail("Costants")

    ! real number
1000 format (t1,'................................................', &
         t1,a,' ',t50,es20.13,1x,a)
  end subroutine dlf_constants_report


end module dlf_constants

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine write_xyz(unit,nat,znuc,coords)
  use dlf_parameter_module, only: rk
  use dlf_constants, only: dlf_constants_get
  implicit none
  integer,intent(in) :: unit
  integer,intent(in) :: nat
  integer,intent(in) :: znuc(nat)
  real(rk),intent(in):: coords(3,nat)
  integer            :: iat
  character(2)       :: str2
  real(rk)           :: ang_au
  character(2), external :: get_atom_symbol
! **********************************************************************
  call dlf_constants_get("ANG_AU",ang_au)
!if (unit==40) print*,"The new coordinates"
  do iat=1,nat
    str2 = get_atom_symbol(znuc(iat))
    write(unit,'(2x,A2,6x,F12.6,3x,F12.6,3x,F12.6)') str2,coords(1,iat)*ang_au, coords(2,iat)*ang_au, coords(3,iat)*ang_au
!if (unit==40) then
!write(*,'(a2,3f15.7)') str2,coords(:,iat)*ang_au
!endif
  end do
  call flush(unit)
end subroutine write_xyz

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine write_xyz_active(unit,nat,znuc,spec,coords)
  use dlf_parameter_module, only: rk
  use dlf_constants, only: dlf_constants_get
  implicit none
  integer,intent(in) :: unit
  integer,intent(in) :: nat
  integer,intent(in) :: znuc(nat)
  integer,intent(in) :: spec(nat)
  real(rk),intent(in):: coords(3*nat)
  integer            :: iat,jat
  character(2)       :: str2
  character(1), dimension(2) :: cartsym
  real(rk)           :: ang_au
  character(2), external :: get_atom_symbol
! **********************************************************************
  call dlf_constants_get("ANG_AU",ang_au)
  write (unit,'(/," ANALYTICAL GRADIENT: ")')
  write (unit,'("------------------------")')                                                              
  write (unit,'(" VARIBLES",4x,"NEW_GRAD")')      
  write (unit,'("------------------------")')

  cartsym(1) = 'X'
  cartsym(2) = 'Y'
  cartsym(3) = 'Z'
                   
  do iat=1,nat
    do jat=1,3
      write(unit,'(I5,A1,3x,F14.10)')iat,cartsym(jat),coords((iat-1)*3+jat)
    enddo
  end do
  write (unit,'("------------------------")')
  call flush(unit)
end subroutine write_xyz_active

subroutine dlf_print_wavenumber(h_eigval,twavenum)
  ! get the eigenvalue of the hessian in mass-weighted coordinates 
  ! print the wavenumber in cm^-1 if twavenum is true
  ! print the crossover temperature for tunneling if the eigenvalue is negative
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout
  use dlf_constants, only: dlf_constants_get
  implicit none
  real(rk), intent(in) :: h_eigval
  logical , intent(in) :: twavenum
  real(rk) :: CM_INV_FOR_AMU, svar, kboltz_au,pi
  !
  if(twavenum) then
    call dlf_constants_get("CM_INV_FOR_AMU",CM_INV_FOR_AMU)
    svar=sqrt(abs(h_eigval)) * CM_INV_FOR_AMU
    if(h_eigval<0.D0) svar=-svar
    write(stdout,"('Frequency of transition mode',f10.3,' cm^-1 &
        &(negative value denotes imaginary frequency)')") &
        svar
  end if
  if(h_eigval<0.D0) then
    ! calculate crossover temperature
    call dlf_constants_get("AMU",svar)
    call dlf_constants_get("KBOLTZ_AU",kboltz_au)
    call dlf_constants_get("PI",pi)
    write(*,"('Crossover temperature for tunnelling',f15.5,' K')") &
        sqrt(abs(h_eigval)) / kboltz_au / (2.D0*pi) / sqrt(svar) !(1.66054D-27/9.10939D-31)
  end if
end subroutine dlf_print_wavenumber

! calculates the median (array is modified, though)
subroutine median(size,array_,med)
  use dlf_parameter_module, only: rk
  implicit none
  integer  ,intent(in) :: size
  real(rk) ,intent(in) :: array_(size)
  real(rk) ,intent(out):: med
  real(rk) :: array(size)
  real(rk) :: maxv
  integer  :: istep,sizeh
  !
  array=array_
  maxv=maxval(array)
  do istep=1,size/2
    med=minval(array)
    array(minloc(array))=maxv
  end do
  if(mod(size,2)==1) then
    med=minval(array)
  else
    med=0.5D0*(med+minval(array))
  end if
end subroutine median

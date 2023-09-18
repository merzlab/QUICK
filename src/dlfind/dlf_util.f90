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
!! $Date: 2008-05-14 11:59:15 +0200 (Wed, 14 May 2008) $
!! $Rev: 325 $
!! $Author: jk37 $
!! $URL: http://ccpforge.cse.rl.ac.uk/svn/dl-find/trunk/dlf_util.f90 $
!! $Id: dlf_util.f90 325 2008-05-14 09:59:15Z jk37 $
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

integer function get_atom_number(symbol)
  implicit none
  character(2), intent(in) :: symbol
  integer                  :: i
  character(2), parameter  :: elements(111) = &
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
  do i = 1, size(elements)
    if (trim(elements(i))==trim(symbol)) then
      get_atom_number = i
      return
    end if
  end do
  STOP "This atom symbol is not known to dl-find."
end function get_atom_number 

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
module fit_g3

  ! creates a set of third order polynomials that fit input data of
  ! values and gradients.
  ! a number of input functions can be given
  !
  ! Calling-Sequence:
  !   call g3_init
  !   call g3_create once for each function (each ifunc)
  !   call g3_get for each interpolated value
  !   call g3_destroy

  use dlf_parameter_module, only: rk
  use dlf_allocate, only: allocate,deallocate
  implicit none

  public :: g3_init, g3_create, g3_get, g3_destroy

  interface g3_init
    module procedure g3_init
  end interface

  interface g3_create
    module procedure g3_create
  end interface

  interface g3_get
    module procedure g3_get
  end interface

  interface g3_destroy
    module procedure g3_destroy
  end interface

  private
  logical :: tinit
  integer :: nfunc ! number of functions to be interpolated
  integer :: length ! number of values to be interpolated between
  logical, allocatable, save :: created(:) ! (nfunc)
  real(rk),allocatable, save :: gridx(:,:) ! (length,nfunc) !x values of grid
  real(rk),allocatable, save :: gridy(:,:) ! (length,nfunc) !y values of grid
  real(rk),allocatable, save :: grid_dydx(:,:) ! (length,nfunc)
contains

  subroutine g3_init(nfunc_in,length_in)
    integer, intent(in) :: nfunc_in 
    integer, intent(in) :: length_in 
    ! ******************************************************************

    ! re-initialise if necessary
    if(allocated(created)) call g3_destroy

    nfunc=nfunc_in
    length=length_in
    if(nfunc*length<=0) call dlf_fail("wrong parameters to g3_init")

    tinit=.true.
    call allocate(created,nfunc)
    call allocate(gridx,length,nfunc)
    call allocate(gridy,length,nfunc)
    call allocate(grid_dydx,length,nfunc)

    created(:)=.false.

  end subroutine g3_init

  subroutine g3_destroy
    ! ******************************************************************
    tinit=.false.
    call deallocate(created)
    call deallocate(gridx)
    call deallocate(gridy)
    call deallocate(grid_dydx)

  end subroutine g3_destroy

  SUBROUTINE g3_create(ifunc,x_in,y_in,dy_in)
    ! creates the g3 function number ifunc, i.e. calculates its 
    ! second derivatives at the grid points
    ! x_in must be monotonically increasing
    ! using natural boundary conditions
    ! ******************************************************************
    integer , intent(in) :: ifunc
    real(rk), intent(in) :: x_in(length),y_in(length),dy_in(length)
    !
    real(rk) :: store(length)! u
    real(rk) :: svar, svar2  ! sig, p
    integer  :: ival
    ! ******************************************************************

    ! check integrity
    if(.not.tinit) call dlf_fail("g3_create must not be called &
        &before g3_init!")
    if(ifunc<1) call dlf_fail("ifunc < 1 in g3_create")
    if(ifunc>nfunc) call dlf_fail("ifunc > nfunc in g3_create")
    
    gridx(:,ifunc)=x_in(:)
    gridy(:,ifunc)=y_in(:)

    !print*,"x",gridx
    !print*,"y",gridy

    ! natural boundaries
    grid_dydx(:,ifunc)=dy_in(:)
    store(1)=0.D0

    created(ifunc)=.true.

  END SUBROUTINE g3_create

  SUBROUTINE g3_get(ifunc,xval,yval,dyval,d2yval)
    ! calculates a cubic-g3 interpolated value (yval) and its 
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
    real(rk) :: aval,bval,delta,xrel
    ! ******************************************************************

    ! check integrity
    if(.not.tinit) call dlf_fail("g3_get must not be called before &
        &g3_init!")
    if(ifunc<1) call dlf_fail("ifunc < 1 in g3_get")
    if(ifunc>nfunc) call dlf_fail("ifunc > nfunc in g3_get")
    if(.not.created(ifunc)) call dlf_fail("g3_get must not be called&
        & before g3_create!")
    
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

    if (delta<=0.D0) call dlf_fail("grid points for g3 not distinct")

    ! polynomial y= ax^3 + bx^2 + cx + d
    ! c=grid_dydx(low,ifunc)
    ! d=gridy(low,ifunc)

    aval=(2.D0 * (gridy(low,ifunc)-gridy(high,ifunc))/delta + &
         grid_dydx(low,ifunc) &
         + grid_dydx(high,ifunc) )/delta**2
    bval=(grid_dydx(high,ifunc)-grid_dydx(low,ifunc)- &
         3.D0*aval*delta**2)*0.5D0/delta

    xrel=xval-gridx(low,ifunc)
    !print*,low,high
    yval= xrel**3*aval + xrel**2*bval + xrel*grid_dydx(low,ifunc) &
         + gridy(low,ifunc)
    dyval= 3.D0*xrel**2*aval + 2.D0*xrel*bval + grid_dydx(low,ifunc) 
    d2yval= 6.D0*xrel*aval + 2.D0*bval 
 
  END SUBROUTINE g3_get

end module fit_g3
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



! wrapper to bspline (and probably other interpolation methods in the 
! future)
module dlf_interpolate
  use dlf_bspline
  use dlf_parameter_module, only: rk
  implicit none
  public :: intp_init,intp_set,intp_get,intp_get2,intp_destroy
  integer :: length ! number of values to be interpolated between
contains
  
  subroutine intp_init(nfunc_in,length_in)
    implicit none
    integer, intent(in) :: nfunc_in ! number of functions to be interpolated
                                    ! with the same number of grid points
                                    ! their x-values may be different
    integer, intent(in) :: length_in ! length of the "x" grid
    length=length_in
    call spline_init(length_in,nfunc_in)

  end subroutine intp_init
 

  SUBROUTINE intp_set(ifunc,x_in,y_in)
    ! creates the spline function number ifunc, i.e. calculates its 
    ! second derivatives at the grid points
    ! x_in must be monotonically increasing
    ! using natural boundary conditions
    ! ******************************************************************
    implicit none
    integer , intent(in) :: ifunc
    real(rk), intent(in) :: x_in(length),y_in(length)
    call spline_create(ifunc,x_in,y_in)
  end SUBROUTINE intp_set

  SUBROUTINE intp_set_fundamental(ifunc,x_in,y_in)
    ! creates the spline function number ifunc, i.e. calculates its 
    ! second derivatives at the grid points
    ! x_in must be monotonically increasing
    ! does not use natrual boundary conditions, but extrapolates the second
    ! derivative of the function at the boundary from the inner points
    ! So this is not truely a fundamental spline (which would require the 
    ! first or second derivative at the boundary) but something close
    ! undamped extrapolation diverges in some cases. So I try a damping factor
    ! ******************************************************************
    implicit none
    integer , intent(in) :: ifunc
    real(rk), intent(in) :: x_in(length),y_in(length)
    real(rk)             :: svar,curv1,curv2,yval,dyval
    integer              :: icount,ncount=20 ! could be made variable (maybe check for convergence?)
    real(rk)             :: damp1=0.3_rk ! 1 is undamped, 0 is no movement
    real(rk)             :: damp2=0.3_rk ! 1 is undamped, 0 is no movement
    real(rk)             :: curv1_old,curv2_old
    real(rk)             :: dcurv1,dcurv2
    logical              :: twarn1,twarn2
    logical,parameter    :: dbg=.false.

    call spline_create(ifunc,x_in,y_in) !natural boundaries
    curv1_old=0._rk
    curv2_old=0._rk

    do icount=1,ncount
      call spline_get(ifunc,x_in(2),yval,dyval,curv1)
      call spline_get(ifunc,x_in(3),yval,dyval,svar)
      svar=(svar-curv1)/(x_in(3)-x_in(2))
      curv1=curv1+svar*(x_in(1)-x_in(2))
      curv1=damp1*curv1+(1._rk-damp1)*curv1_old

      call spline_get(ifunc,x_in(length-1),yval,dyval,curv2)
      call spline_get(ifunc,x_in(length-2),yval,dyval,svar)
      svar=(svar-curv2)/(x_in(length-2)-x_in(length-1))
      curv2=curv2+svar*(x_in(length)-x_in(length-1))
      curv2=damp2*curv2+(1._rk-damp2)*curv2_old

      twarn1=.false.
      twarn2=.false.
      if(icount>11) then
        if(dabs(curv1-curv1_old)>dcurv1) then
          if (dbg) print*,"Convergence warning curv1",&
              dabs(curv1-curv1_old),dcurv1,icount
          damp1=damp1*0.8_rk
          twarn1=.true.
          exit
        end if
        if(dabs(curv2-curv2_old)>dcurv2) then
          if (dbg) print*,"Convergence warning curv2",&
              dabs(curv2-curv2_old),dcurv2,icount
          damp2=damp2*0.8_rk
          twarn2=.true.
          exit
        end if
      end if
  
      call spline_create(ifunc,x_in,y_in,curv1,curv2) 
      if (dbg) print*,ifunc,icount,curv1,curv2
      dcurv1=dabs(curv1-curv1_old)
      dcurv2=dabs(curv2-curv2_old)
      curv1_old=curv1
      curv2_old=curv2
    end do
    
    ! if not converged, fall back to natural boundaries
    if(twarn1.or.twarn2) then
      if(twarn1) curv1=0._rk
      if(twarn2) curv2=0._rk
      !call spline_create(ifunc,x_in,y_in,curv1,curv2) 
      call spline_create(ifunc,x_in,y_in) ! use natural bounds at first warning ...
      if (dbg) print*,"Falling back to natural boundary conditions",twarn1,twarn2
    end if

  end SUBROUTINE intp_set_fundamental

  SUBROUTINE intp_get(ifunc,xval,yval,dyval)
    ! calculates a cubic-spline interpolated value (yval) and its 
    ! derivative (dyval) at a position xval
    ! this is done for the interpolation ifunc
    ! ******************************************************************
    integer ,intent(in) :: ifunc
    real(rk),intent(in) :: xval
    real(rk),intent(out):: yval
    real(rk),intent(out):: dyval
    real(rk) :: svar
    call spline_get(ifunc,xval,yval,dyval,svar)
  end SUBROUTINE intp_get

  ! this could be  made nicer in an interface
  SUBROUTINE intp_get2(ifunc,xval,yval,dyval,d2yval)
    ! calculates a cubic-spline interpolated value (yval) and its 
    ! derivative (dyval) at a position xval
    ! this is done for the interpolation ifunc
    ! ******************************************************************
    integer ,intent(in) :: ifunc
    real(rk),intent(in) :: xval
    real(rk),intent(out):: yval
    real(rk),intent(out):: dyval
    real(rk),intent(out):: d2yval
    call spline_get(ifunc,xval,yval,dyval,d2yval)
  end SUBROUTINE intp_get2

  subroutine intp_destroy
    call spline_destroy
  end subroutine intp_destroy

end module dlf_interpolate


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
  real(rk),parameter :: cal = 4.184_rk ! calorie in J, thermochemical calorie according to NIST

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
      val=sqrt(hartree/amc) / ( 2.E0_rk * pi * bohr * speed_of_light) / 100.E0_rk ! 5140.4871520152346
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
    case("KBOLTZ_CM_INV")
      val=kboltz/(planck*speed_of_light*100._rk)
    case("SECOND_AU")
      val=second_au  ! 4.1341373379861E+16
    case("CM_INV_FROM_HARTREE")
      ! converts hartree to wave numbers (cm^-1)
      val=(hartree/planck)/speed_of_light*1.D-2  ! 219474.63160039327
    case("CAL")
      val=cal ! 4.184
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
  write(unit,*) nat
  write(unit,*)
  do iat=1,nat
    str2 = get_atom_symbol(znuc(iat))
    write(unit,'(a2,3f15.7)') str2,coords(:,iat)*ang_au
 ! temporary: commented out cartesian conversion
 !   write(unit,'(a2,3f12.7)') str2,coords(:,iat)
  end do
  call flush(unit)
end subroutine write_xyz

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine write_xyz_noConv(unit,nat,znuc,coords)
  use dlf_parameter_module, only: rk
  implicit none
  integer,intent(in) :: unit
  integer,intent(in) :: nat
  integer,intent(in) :: znuc(nat)
  real(rk),intent(in):: coords(3,nat)
  integer            :: iat
  character(2)       :: str2
  character(2), external :: get_atom_symbol
! **********************************************************************
  write(unit,*) nat
  write(unit,*)
  do iat=1,nat
    str2 = get_atom_symbol(znuc(iat))
    write(unit,'(a2,3f12.7)') str2,coords(:,iat)
 ! temporary: commented out cartesian conversion
 !   write(unit,'(a2,3f12.7)') str2,coords(:,iat)
  end do
  call flush(unit)
end subroutine write_xyz_noConv

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine write_xyz_active(unit,nat,znuc,spec,coords)
  use dlf_parameter_module, only: rk
  use dlf_constants, only: dlf_constants_get
  implicit none
  integer,intent(in) :: unit
  integer,intent(in) :: nat
  integer,intent(in) :: znuc(nat)
  integer,intent(in) :: spec(nat)
  real(rk),intent(in):: coords(3,nat)
  integer            :: iat,nact
  character(2)       :: str2
  real(rk)           :: ang_au
  character(2), external :: get_atom_symbol
! **********************************************************************
  call dlf_constants_get("ANG_AU",ang_au)
  nact=0
  do iat=1,nat
    if(spec(iat)/=-1) nact=nact+1
  end do
  write(unit,*) nact
  write(unit,*)
  do iat=1,nat
    if(spec(iat)==-1) cycle
    str2 = get_atom_symbol(znuc(iat))
    write(unit,'(a2,3f15.7)') str2,coords(:,iat)*ang_au
  end do
  call flush(unit)
end subroutine write_xyz_active

subroutine write_path_xyz(nat,znuc,filename,nPts,points, multFiles)
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout, printl
  implicit none
  
  character(*), intent(in)  ::  filename
  integer, intent(in)       ::  nat
  integer, intent(in)       ::  znuc(nat)
  integer, intent(in)       ::  nPts
  real(rk), intent(in)      ::  points(3*nat, nPts)
  logical, intent(in)       ::  multFiles
  integer                   ::  iimage
  integer                   ::  maxxyzfile=100
  character(30)             ::  filename_ext
  integer                   ::  unitp
  logical                   ::  itsopen
  unitp = 4741 
  if(printl>=4) write(stdout,'("Writing path xyz file with filename ", A, ".xyz")') &
    filename
  if (multFiles) then
    do iimage=1,nPts
      if(iimage>maxxyzfile) exit
      if(iimage<10) then
        write(filename_ext,'("000",i1)') iimage
      else if(iimage<100) then
        write(filename_ext,'("00",i2)') iimage
      else if(iimage<1000) then
        write(filename_ext,'("0",i3)') iimage
      else
        write(filename_ext,'(i4)') iimage
      end if
!       if (glob%iam == 0) &
      inquire(unit=3, opened=itsopen)
      if (itsopen) call dlf_fail("Writing to same unit as other part of dl-find (write_path_xyz)!")
      open(unit=unitp+iimage,file=trim(adjustl(filename))//trim(adjustl(filename_ext))//".xyz")
    end do

    do iimage=1,nPts
      if(iimage>maxxyzfile) exit
      call write_xyz(unitp+iimage,nat,znuc,points(:,iimage))
    end do    
    do iimage=1,nPts
      close(unit=unitp+iimage)
    end do  
  else
    open(unit=unitp,file=trim(adjustl(filename))//".xyz")      
    do iimage=1,nPts
      call write_xyz(unitp,nat,znuc,points(:,iimage))
    end do
    close(unitp)
  end if
end subroutine write_path_xyz

subroutine get_path_properties(filename,nat,nPts)
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout, stderr, printl
  implicit none
  character(*), intent(in)               ::  filename
  integer, intent(out)                   ::  nat
  integer, intent(out)                   ::  nPts
  integer                                ::  unitp
  logical                                ::  file_exists
  integer                                ::  io
  integer                                ::  nLines
  unitp = 4741
  INQUIRE(FILE=filename, EXIST=file_exists)
  if(.not.file_exists) then
    if(printl>=4) write(stdout,'("file ", A, " does not exist.")') filename
    call dlf_fail("File not present")
  end if
  if(printl>=4) write(stdout,'("Reading file with filename ", A)') filename
  open(unit=unitp,file=filename)
  ! count number of lines
  io = 0
  nLines = 0
  read(unitp,*,iostat=io) nat
  do while (io==0)
    nLines = nLines + 1
    read(unitp,*, iostat=io)
  end do
  if (io>0) call dlf_fail("Error, but not end of file (get_path_properties).")
  ! calc number of points
  if (MOD(nLines,(nat+2))==0) then
    nPts = nLines/(nat+2)
  else if (MOD(nLines-1,(nat+2))==0) then
    nPts = (nLines-1)/(nat+2)
  else
    if(printl>=4) write(stdout,'("nLines", I10)') nLines
    call dlf_fail("Number of lines not valid. Extra empty line?")
  end if
  close(unitp)
end subroutine get_path_properties

subroutine read_path_xyz(filename,nat,nPts,znuc,points)
  use dlf_parameter_module, only: rk
  use dlf_constants, only: dlf_constants_get
  use dlf_global
  implicit none
  character(*), intent(in)  ::  filename
  integer, intent(in)       ::  nat
  integer, intent(in)       ::  nPts
  integer, intent(inout)    ::  znuc(nat)
  real(rk), intent(inout)   ::  points(3*nat,nPts)
  integer                   ::  iimage
  integer                   ::  nat_dummy
  integer                   ::  maxxyzfile=100
  character(30)             ::  filename_ext
  integer                   ::  unitp
  logical                   ::  itsopen
  character(2)              ::  str2
  real(rk)                  ::  ang_au
  integer                   ::  io, iat
  integer                   ::  nLines
  logical                   ::  file_exists
  integer, external         ::  get_atom_number
  character(len=256)       ::  line
  call dlf_constants_get("ANG_AU",ang_au)
!   STOP "ALLOCATING HERE DOES NOT WORK FOR SOME REASON"
  ! rewind to beginning
  open(unit=unitp,file=filename, action='read')
  do iimage=1,nPts
    read(unitp,*) nat_dummy
    read(unitp,*)
    do iat=1,nat
!       read(unitp,'(a2,3f12.7)',iostat=io) str2,points((iat-1)*3+1:iat*3,iimage)
      read(unitp,'(A)',iostat=io) line
      line = adjustl(line)
      read(line,*) str2, points((iat-1)*3+1:iat*3,iimage)
      znuc(iat) = get_atom_number(str2)
    end do
  end do
  points(:,:) = points(:,:) / ang_au
  close(unitp)
end subroutine read_path_xyz

subroutine write_energies(nImages, nDims, energies, points, filename)
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout, printl
  implicit none
  
  character(*), intent(in)  ::  filename
  integer, intent(in)       ::  nImages, nDims
  real(rk), intent(in)      ::  energies(nImages)
  real(rk), intent(in)      ::  points(nDims, nImages)
  real(rk)                  ::  lin_length
  integer                   ::  unitp, iimage
  
  unitp = 4741 
  if(printl>=4) write(stdout,&
    '("Writing all energies in file with filename ", A, ".ene")') filename
  open(unit=unitp,file=trim(adjustl(filename))//".ene")      
  write(unitp,'(a)') "# linear length          energy"
  do iimage=1,nImages
    if (iimage==1) then
      lin_length = 0d0
    else
      lin_length = lin_length + dsqrt(dot_product(points(:,iimage)-points(:,iimage-1),&
                                     points(:,iimage)-points(:,iimage-1)))
    end if
    write(unitp,'(4X,ES11.4,5X,ES11.4)') lin_length, energies(iimage)-energies(1)
  end do
  close(unitp)
end subroutine write_energies

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine read_xyz(filename,nat,znuc,coords)
  use dlf_parameter_module, only: rk
  use dlf_constants, only: dlf_constants_get
  implicit none
  character(*), intent(in)  :: filename
  integer, intent(in)       :: nat
  integer, intent(out)      :: znuc(nat)
  real(rk), intent(out)     :: coords(3,nat)
  integer                   :: unitp=4242
  integer                   :: iat
  character(2)              :: str2
  real(rk)                  :: ang_au
  integer, external         :: get_atom_number
  integer                   :: io
  logical                   :: tck
! **********************************************************************
  znuc=0
  coords=0.D0 ! default in case of error in reading
  inquire(file=filename,exist=tck)
  if(.not.tck) then
    print*,"Error: file ",trim(filename)," not found."
    return
  end if
  open(unit=unitp,file=filename)
  call dlf_constants_get("ANG_AU",ang_au)
  read(unitp,*)
  read(unitp,*)
  do iat=1,nat
!    read(unitp,'(a2,3f12.7)',iostat=io) str2, coords(:,iat)
    read(unitp,fmt=*,iostat=io) str2, coords(:,iat)
    if (io/=0) then
      print*,"Error reading xyz file ",trim(filename)
      exit
    end if
    znuc(iat) = get_atom_number(str2)
 ! temporary: commented out cartesian conversion
 !   write(unitp,'(a2,3f12.7)') str2,coords(:,iat)
  end do
  coords(:,:) = coords(:,:) / ang_au
  close(unitp)
end subroutine read_xyz

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine read_nat(filename,nat)
  use dlf_parameter_module, only: rk
  use dlf_constants, only: dlf_constants_get
  implicit none
  character(*), intent(in)  :: filename
  integer, intent(out)      :: nat
  integer                   :: unitp=4242
! **********************************************************************
  open(unit=unitp,file=filename)
  read(unitp,*) nat
  close(unitp)
end subroutine read_nat

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

function get_cov_radius(atomic_number) result (cov_rad)
use dlf_parameter_module, only: rk
use dlf_global, only: stdout, stderr, printl
use dlf_constants, only: dlf_constants_get
implicit none
  integer, intent(in)       ::  atomic_number
  real(rk)                  ::  cov_rad
  real(rk)                  ::  pmToAU
  real(rk)                  ::  cov_radii(96) = &
    (/ 31d0,28d0,&
       128d0,96d0,85d0,76d0,71d0,66d0,57d0,58d0,&
       166d0,141d0,121d0,111d0,107d0,105d0,102d0,106d0,&
       203d0,176d0,170d0,160d0,153d0,139d0,139d0,132d0,126d0,124d0,132d0,&
       122d0,122d0,120d0,119d0,120d0,120d0,116d0,&
       220d0,195d0,190d0,175d0,164d0,154d0,147d0,146d0,142d0,139d0,145d0,&
       144d0,142d0,139d0,139d0,138d0,139d0,140d0,&
       244d0,215d0,207d0,204d0,203d0,201d0,199d0,198d0,198d0,196d0,194d0,192d0,&
       192d0,189d0,190d0,187d0,187d0,175d0,170d0,162d0,151d0,144d0,141d0,136d0,&
       136d0,132d0,145d0,146d0,148d0,140d0,150d0,150d0,&
       260d0,221d0,215d0,206d0,200d0,196d0,190d0,187d0,180d0,169d0/)
    if (atomic_number>96) then
      if(printl>=4) write(stdout,'(&
              "Warning! The covalent radius for this type of element is not",&
              " available. Therefore, simply 200 pm will be used!")')
      cov_rad = 200d0
    else
      cov_rad = cov_radii(atomic_number)
    end if
    call dlf_constants_get("ANG_AU", pmToAU)
    pmToAU=1d-2/pmToAU
    cov_rad = cov_rad * pmToAU
end function get_cov_radius

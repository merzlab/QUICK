! **********************************************************************
! **                      Optimizing a geodesic                       **
! **********************************************************************
!!****h* DL-FIND/geodesic
!!
!! NAME
!! geodesic
!!
!! FUNCTION
!!    Approximates a geodesic curve according to paper by 
!!    Zhu, Thompson, and Martínez
!!
!! Inputs
!!    The two Minima: reactant and product
!! 
!! Outputs
!!    Approximated geodesic curve to start NEB/GPRMEP/...
!!
!! COMMENTS
!!    -
!!
!! COPYRIGHT
!!
!!  Copyright 2019 , Alexander Denzel (denzel@theochem.uni-stuttgart.de)
!!  Johannes Kaestner (kaestner@theochem.uni-stuttgart.de)
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

module geodesic_module
use dlf_parameter_module, only: rk
use dlf_global, only: stdout, stderr, printl, glob
implicit none

type geodesic_type
  integer                       ::  nDim
  integer                       ::  sdgf
  integer                       ::  nPts
  integer                       ::  nImgGoal
  integer                       ::  nAtoms
  integer                       ::  nInts
  real(rk), allocatable         ::  xCoords(:,:)
  real(rk), allocatable         ::  iCoords(:,:)
  real(rk), allocatable         ::  wilsonB(:,:,:)
  real(rk)                      ::  L
  real(rk), allocatable         ::  dL(:,:)
  integer, allocatable          ::  znuc(:)
  real(rk)                      ::  alpha = 1.7d0
  real(rk)                      ::  beta = 0.01d0 
  character(8)                  ::  geoFileName = "geodesic"
contains
  procedure                     ::  updateL => geodesic_updateL
  procedure                     ::  construct => geodesic_construct
  procedure                     ::  destroy => geodesic_destroy
  procedure                     ::  coreOptPath => geodesic_CoreOptPath
  procedure                     ::  completePreOpt => geodesic_completePreOpt 
  procedure                     ::  optPath => geodesic_optPath
  procedure                     ::  xToI => geodesic_xToI
  procedure                     ::  geodesic_L_upper
  procedure                     ::  geodesicFromMinima => geodesic_fromMinima
  procedure                     ::  dIdx => geodesic_dIdx
  procedure                     ::  dIdx_core => geodesic_dIdx_core
  procedure                     ::  get_added_cov_radius => geodesic_get_added_cov_radius
  procedure                     ::  testGrad_FD_L
end type geodesic_type
  ! this instance of geodesic_type is used in dl-find
  type(geodesic_type),save                 ::  geo_inst
  
contains

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_construct
!!
!! FUNCTION
!! Construct an instance of geodesic_type
!!
!! SYNOPSIS
subroutine geodesic_construct(geo, nPts_in, nAtoms_in, znuc_in)
!! SOURCE
  class(geodesic_type)          ::  geo
  integer, intent(in)           ::  nPts_in, nAtoms_in
  integer, intent(in)           ::  znuc_in(nAtoms_in)
  integer                       ::  i
  if (nPts_in >= 3) then
    geo%nImgGoal = nPts_in 
  else
    geo%nImgGoal = 3
  end if
  geo%nPts = geo%nImgGoal!3
  geo%nAtoms = nAtoms_in
  geo%nDim = nAtoms_in*3
  geo%nInts = nAtoms_in*(nAtoms_in-1)
  geo%sdgf = 0
  do i = 1, nAtoms_in
    if (glob%spec(i)>=0) geo%sdgf = geo%sdgf + 3
  end do

  allocate(geo%xCoords(geo%nDim, geo%nPts)) ! last two are temporarily used
  geo%xCoords(:,:) = 9d99
  allocate(geo%iCoords(geo%nInts,geo%nPts)) ! last two are temporarily used
  allocate(geo%wilsonB(geo%nInts,geo%nDim,geo%nPts)) ! last two are temporarily used
  allocate(geo%dL(geo%nDim,geo%nPts))
  ! atomic numbers of the atoms
  allocate(geo%znuc(geo%nAtoms))
  geo%znuc = znuc_in
end subroutine geodesic_construct

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_destroy
!!
!! FUNCTION
!! Destroy an instance of geodesic_type
!!
!! SYNOPSIS
subroutine geodesic_destroy(geo)
!! SOURCE
  class(geodesic_type)          ::  geo
  if(allocated(geo%xCoords)) deallocate(geo%xCoords)
  if(allocated(geo%iCoords)) deallocate(geo%iCoords)
  if(allocated(geo%wilsonB)) deallocate(geo%wilsonB)
  if(allocated(geo%dL)) deallocate(geo%dL)
  if(allocated(geo%znuc)) deallocate(geo%znuc)
end subroutine geodesic_destroy

subroutine adapt3AtomLin(A,B)
  real(rk), intent(inout)      ::  A(9)
  real(rk), intent(inout)   ::  B(9)
  real(rk)                  ::  transvec(3), s,c,v(3),v1(3),v2(3),&
                                vx(3,3), rotMat(3,3)
  integer                   ::  i, j
  ! translation so that C is at 0/0/0
  ! A:
  transvec = -A(1:3)
  do i = 1, 3
    A((i-1)*3+1:i*3)=A((i-1)*3+1:i*3)+transvec
  end do
  ! B:
  transvec = -B(1:3)
  do i = 1,9
  end do
  do i = 1, 3
    B((i-1)*3+1:i*3)=B((i-1)*3+1:i*3)+transvec
  end do
 
  ! rotation so that N points in direction of 1/0/0
  ! rotation matrix that rotates B on A
  v2 = (A(4:6)-A(1:3))
  v1 = (B(4:6)-B(1:3))
  v2 = v2/norm2(v2)
  v1 = v1/norm2(v1)
  v = cross_product(v1,v2)
  s = norm2(v)
  if (s<1d-15) return ! already overlapped
  c = dot_product(v2,v1)
  vx = transpose(reshape( (/ 0d0, -v(3), v(2),&
          v(3), 0d0,  -v(1), &
          -v(2),v(1), 0d0 /), shape(vx) ))
  rotMat = transpose(reshape( (/ 1d0, 0d0, 0d0,&
              0d0, 1d0, 0d0, &
              0d0, 0d0, 1d0 /), shape(rotMat) ))
  rotMat = rotMat + vx + matmul(vx,vx)*(1-c)/s**2
  do i = 2, 3
    do j = 1, 3
       v(j)= dot_product(rotMat(j,:),B((i-1)*3+1:i*3))
    end do
    B((i-1)*3+1:i*3) = v(:)
  end do
end subroutine adapt3AtomLin

function cross_product(a, b)
    real(rk), dimension(3) :: cross_product
    real(rk), dimension(3), intent(in) :: a, b
 
    cross_product(1) = a(2)*b(3) - a(3)*b(2)
    cross_product(2) = a(3)*b(1) - a(1)*b(3)
    cross_product(3) = a(1)*b(2) - b(1)*a(2)
end function cross_product

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_fromMinima
!!
!! FUNCTION
!! approximate a geodesic curve from two minima (reactant and product)
!!
!! INPUT
!! - nImg: number of images that the geodesic should have
!!         Setting it to zero means that the number is flexible
!!
!! SYNOPSIS
subroutine geodesic_fromMinima(geo,nAtoms,znuc_in,A,B,nImg,incPtNr,eps)
!! SOURCE
  use ieee_arithmetic
  use oop_lbfgs
  use mod_oop_clock
  class(geodesic_type)          ::  geo
  integer, intent(in)           ::  nAtoms
  integer, intent(in)           ::  znuc_in(nAtoms)
  real(rk), intent(in)          ::  A(3*nAtoms),B(3*nAtoms)
  integer, intent(in)           ::  nImg
  logical, intent(in)           ::  incPtNr
  real(rk), intent(in)          ::  eps
  real(rk), allocatable         ::  qA(:),qB(:),qMid(:), x_0(:), &
                                    rand_vec(:), x(:), step(:), &
                                    B_Matrix(:,:), &
                                    dNdx(:), q(:), xMid(:)
  real(rk)                      ::  L_min, delta, limitStepOptX, error
  integer                       ::  i, n, j, j2, k, k_help, xd,d, &
                                    nAt1, nAt2, pIter, nOptSteps
  type(oop_lbfgs_type)          ::  lbfgs
  logical                       ::  inverted, problem
  type(geodesic_type)           ::  geo3pt
  type(clock_type)              ::  geoclock
  if (printl>=4) write(stdout,'("Optimizing geodesic")')
!   call geoclock%start("geoclock",1d-3)
!   if (nImg>0) then
!     call geo%construct(nImg,nAtoms,znuc_in)
!   else if (nImg==0) then
  call geo%construct(nImg,nAtoms,znuc_in)
!   else 
!     call dlf_fail("Invalid nr of images!")
!   end if
  call geo3pt%construct(3,nAtoms,znuc_in)
  
  ! check for consistency in frozen atoms
  problem = .false.
  do i = 1, nAtoms
    if (glob%spec(i)<0) then
      ! test for same x, y, and z values   
      if (abs(A(i*3-2)-B(i*3-2))>1d-14) problem = .true.
      if (abs(A(i*3-1)-B(i*3-1))>1d-14) problem = .true.
      if (abs(A(i*3  )-B(i*3  ))>1d-14) problem = .true.
    end if
  end do
  if(problem) call dlf_fail(&
        "First and last image are not equivalent for frozen atoms!")
  if (dot_product(A(:)-B(:),A(:)-B(:))<1d-14) &
    call dlf_fail(&
        "Geodesic error: Reactant and product seems to be the same structure.")
  geo3pt%xCoords(:,1) = A
  geo3pt%xCoords(:,3) = B
!   if (geo%nAtoms==3) then 
!     call adapt3AtomLin(geo3pt%xCoords(:,1),geo3pt%xCoords(:,3))
!   else 
!     call dlf_cartesian_align(geo3pt%nAtoms, geo3pt%xCoords(:,1),geo3pt%xCoords(:,3))
!   end if
  allocate(qA(geo%nInts))
  allocate(qB(geo%nInts))
  allocate(qMid(geo%nInts))
  allocate(x_0(geo%nDim))
  allocate(rand_vec(geo%nDim))
  allocate(x(geo%nDim))
  allocate(step(geo%nDim))
  allocate(dNdx(geo%nDim))
  allocate(q(geo%nInts))
  allocate(xMid(geo%nDim))
  allocate(B_Matrix(geo%nInts, geo%nDim))
  call geo%xToI(geo3pt%xCoords(:,1),qA)
  call geo%xToI(geo3pt%xCoords(:,3),qB)
  qMid = (qA+qB)/2d0
  ! Set L_min to infinity
!   if (ieee_support_inf(L_min)) then
!     L_min = ieee_value(L_min,ieee_positive_inf)
!   end if
  L_min = 9d99
  call geodesic_init_random_seed()
  write(stdout,"(A)", advance='no') "Generating random start vectors, optimizing and choosing the best one."
  do i = 1, 10
    call random_number(rand_vec)
    rand_vec = (rand_vec - 0.5d0)*2d0
    if (mod(i,2)==0) then
      ! even
      x_0 = geo3pt%xCoords(:,3) + rand_vec*0.1d0
    else
      ! odd
      x_0 = geo3pt%xCoords(:,1) + rand_vec*0.1d0
    end if
    !*************************************************************
    ! find x which minimizes norm(q(x)-q(mid))^2 starting from x_0
    x = x_0
    nOptSteps = 0
    
    open(unit=109,file="tmp0.xyz", action='write')
    call write_xyz(109,3,geo3pt%znuc,geo3pt%xCoords(:,2))
    close(109)
    delta = 1d-4 ! delta for optimization of starting midpoint
    error = 1d10 ! just starting value    
    limitStepOptX = 1d0
    
    call lbfgs%init(geo3pt%nDim, MAX(geo3pt%nDim/2,50), .false.)
    do while (error>delta.or.nOptSteps<=1)
      nOptSteps = nOptSteps + 1
      call geo3pt%xToI(x, q)
      dNdx = 0d0
      call geo3pt%dIdx(x, B_Matrix)
      ! 2*B^T * (qn-qmid)
      call DGEMM('T','N',geo3pt%nDim,1,geo3pt%nInts,2d0,B_Matrix,geo3pt%nInts,&
                 (q(:)-qMid),geo3pt%nInts,0d0,dNdx,geo3pt%nDim)
      error = MAXVAL(ABS(dNdx))      
      if(error<=delta) then
        exit
      else 
!         ! Nothing
      end if
      call lbfgs%next_step(x,dNdx(:),step(:),inverted)
      if(nOptSteps>500.and.MOD(nOptSteps,100)==0.and.inverted) then
        if (printl>=6) write(stdout,'("LBFGS step was inverted (several times)...")')
      end if
      ! limit step
      if (norm2(step)>limitStepOptX) step=step/norm2(step)*limitStepOptX
      ! apply step
      x=x+step
    end do
    call lbfgs%destroy()
    ! optimized one x
    ! check if this x is better than the last one
    geo3pt%xCoords(:,2) = x
    call geo3pt%updateL()
    if (geo3pt%L<L_min) then
      L_min = geo3pt%L
      xMid  = x
    end if
  end do
  if (printl>=4) write(stdout,'(" ")')
  if (printl>=4) write(stdout,'("Found a starting point.")')
  ! probably found the "optimal starting midpoint xMid"
  
  geo3pt%xCoords(:,2) = xMid
  call write_xyz(109,3,geo3pt%znuc,geo3pt%xCoords(:,2))
  close(109)
  
!   if (geo3pt%nAtoms==3) then 
!     call adapt3AtomLin(geo3pt%xCoords(:,1), &
!                        geo3pt%xCoords(:,2))
!   else
!     call dlf_cartesian_align(geo3pt%nAtoms, &
!                            geo3pt%xCoords(:,1), &
!                            geo3pt%xCoords(:,2))
!   end if

  open(unit=109,file="A.xyz", action='write')
  call write_xyz(109,geo3pt%nAtoms,geo3pt%znuc,geo3pt%xCoords(:,1))
  close(109)
  open(unit=109,file="B.xyz", action='write')
  call write_xyz(109,geo3pt%nAtoms,geo3pt%znuc,geo3pt%xCoords(:,3))
  close(109)
  if (printl>=4) write(stdout,'("Optimizing middle point.")')
  call geo3pt%optPath(eps,.false.)
!   if (geo3pt%nPts/=2) print*, "WARNING: Maybe this is a problem. ",&
!             "Think about it: index of xMid:", &
!             MIN(MAX(&
!             NINT(real(geo3pt%nPts,kind=rk)/2d0),&
!             2),geo3pt%nPts-1), "of", geo3pt%nPts
!   xMid = geo3pt%xCoords(:,MIN(MAX(&
!             NINT(real(geo3pt%nPts,kind=rk)/2d0),&
!             2),geo3pt%nPts-1))
  xMid = geo3pt%xCoords(:,2)
  if (MIN(MAX(&
            NINT(real(geo3pt%nPts,kind=rk)/2d0),&
            2),geo3pt%nPts-1)/=2) call dlf_fail("geo3pt not 3pt anymore?")

  open(unit=109,file="xMid.xyz", action='write')
  call write_xyz(109,geo3pt%nAtoms,geo3pt%znuc,xMid)
  close(109)  
  ! now create the "real" path that has to be optimized
  do pIter = 1, geo%nPts
    if (pIter==1.or.(REAL(pIter,kind=rk)<=REAL(geo%nPts/3,kind=rk))) then
      geo%xCoords(:,pIter) = geo3pt%xCoords(:,1)
    else if (REAL(pIter,kind=rk)<=REAL(geo%nPts,kind=rk)&
             *REAL(2,kind=rk)/REAL(3,kind=rk)) then
      geo%xCoords(:,pIter) = xMid(:)
    else if (pIter<=geo%nPts) then
      geo%xCoords(:,pIter) = geo3pt%xCoords(:,3)
    else
      call dlf_fail("pIter should never be larger than geo%nPts!")
    end if
!     ! Nothing
  end do

!   geo%xCoords(:,1) = geo3pt%xCoords(:,1)
!   geo%xCoords(:,2) = xMid
!   geo%xCoords(:,3) = geo3pt%xCoords(:,3)
  
  ! optimize the path
!   if (nImg==0) then
!     call geo%optPath(eps,.true.)
!   else    
!     call geo%optPath(eps,.false.)
!   end if
  if (printl>=4) write(stdout,&
    '("Optimizing the geodesic with final number of images.")')
  call geo%optPath(eps,incPtNr)
  
!   do pIter = 2, geo%nPts
!     print*, "xnorm1",pIter,geo%nPts, norm2(geo%xCoords(:,pIter)-geo%xCoords(:,pIter-1))
!     call geo%xToI(geo%xCoords(:,pIter-1),qA)
!     call geo%xToI(geo%xCoords(:,pIter),qB)
!     print*, "inorm1",pIter,geo%nPts, norm2(qB-qA)
! !     if (geo%nAtoms==3) then 
! !       call adapt3AtomLin(geo%xCoords(:,1),&
! !                          geo%xCoords(:,pIter))
! !     else
! !       call dlf_cartesian_align(geo%nAtoms, &
! !                                reshape(geo%xCoords(:,1),(/3,geo%nAtoms/)),&
! !                                reshape(geo%xCoords(:,pIter),(/3,geo%nAtoms/)))
! !     end if
!     print*, "xnorm2",pIter,geo%nPts, norm2(geo%xCoords(:,pIter)-geo%xCoords(:,pIter-1))
!     call geo%xToI(geo%xCoords(:,pIter-1),qA)
!     call geo%xToI(geo%xCoords(:,pIter),qB)
!     print*, "inorm2",pIter,geo%nPts, norm2(qB-qA)
!   end do
  call write_path_xyz(geo%nAtoms,geo%znuc,geo%geoFileName,geo%nPts,&
        geo%xCoords,.false.)
!   call write_path_xyz(geo%nAtoms,geo%znuc,"geodesic",geo%nPts,&
!         geo%xCoords,.true.)

  if (printl>=6) write(stdout,'("Final number of pts on geodesic: ", I8)') geo%nPts

  ! deallocate unnecessary things
  deallocate(qA)
  deallocate(qB)
  deallocate(qMid)
  deallocate(x_0)
  deallocate(rand_vec)
  deallocate(x)
  deallocate(step)
  deallocate(dNdx)
  deallocate(q)
  deallocate(xMid)
  deallocate(B_Matrix)
  call geo3pt%destroy()
end subroutine geodesic_fromMinima

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_init_random_seed
!!
!! FUNCTION
!! Initializes the creation of random variables
!!
!! SYNOPSIS
SUBROUTINE geodesic_init_random_seed()
!! SOURCE
    INTEGER :: i, n, clock
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
          
    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
          
    CALL SYSTEM_CLOCK(COUNT=clock)
          
    seed = clock + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
          
    DEALLOCATE(seed)
END SUBROUTINE geodesic_init_random_seed
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_updateL
!!
!! FUNCTION
!! Calculate the length and it's derivative with respect to each image
!! Modifies iCoords, wilsonB, dL, and L
!!
!! SYNOPSIS
subroutine geodesic_updateL(geo)
!! SOURCE
  class(geodesic_type)          ::  geo
  integer                       ::  pIter,piter2,atIter1, atIter2, iIter, &
                                    dimOffset1, dimOffset2, dimIter
  real(rk)                      ::  xMid(geo%nDim), qMid(geo%nInts), &
                                    mid_wilsonB(geo%nInts,geo%nDim,2)

  ! calculate internal coordinates (distances between atoms)
  geo%iCoords(:,:) = 0d0
  do pIter = 1, geo%nPts
    call geo%xToI(geo%xCoords(:,pIter),geo%iCoords(:,pIter))
  end do
  ! derivative of iCoords wrt. xCoords
  geo%wilsonB(:,:,:) = 0d0
  do pIter = 1, geo%nPts
    call geo%dIdx(geo%xCoords(:,pIter),geo%wilsonB(:,:,pIter))
  end do
  geo%L = 0d0
  geo%dL(:,:) = 0d0
  do pIter = 1, geo%nPts-1
    ! x coords of midpoint
    xMid(:) = (geo%xCoords(:,pIter)+geo%xCoords(:,pIter+1))/2
    iIter = 0
    ! i coords of midpoint
    call geo%xToI(xMid,qMid)
    ! B matrix at midpoint with respect to each side
!     call geo%dIdx(geo%xCoords(:,pIter),mid_wilsonB(:,:,1))
!     call geo%dIdx(geo%xCoords(:,pIter+1),mid_wilsonB(:,:,2))
    ! These are the same if I understood correctly (why calculate it twice?)
    call geo%dIdx(xMid,mid_wilsonB(:,:,1))
    call geo%dIdx(xMid,mid_wilsonB(:,:,2))    
    mid_wilsonB(:,:,1) = mid_wilsonB(:,:,1)/2d0
    mid_wilsonB(:,:,2) = mid_wilsonB(:,:,2)/2d0
            
    geo%L = geo%L + norm2(qMid(:)-geo%iCoords(:,pIter)) + &
            norm2(qMid(:)-geo%iCoords(:,pIter+1))
    do dimIter = 1, geo%nDim
!       print*, "dimIter, (dimIter-1)/3+1, glob%spec((dimIter-1)/3+1)",&
!         dimIter, (dimIter-1)/3+1, glob%spec((dimIter-1)/3+1)
      if (glob%spec((dimIter-1)/3+1)<0) cycle
!       print*, "/=0"
      if (norm2(qMid(:)-geo%iCoords(:,pIter))>1d-16) then
        geo%dL(dimIter,pIter) = geo%dL(dimIter,pIter) + &
          dot_product((qMid(:)-geo%iCoords(:,pIter)), &
            (mid_wilsonB(:, dimIter, 1)-&
             geo%wilsonB(:, dimIter, pIter)))/ &
            norm2(qMid(:)-geo%iCoords(:,pIter)) 
        geo%dL(dimIter,pIter+1) = geo%dL(dimIter,pIter+1) + &
          dot_product((qMid(:)-geo%iCoords(:,pIter)), &
            mid_wilsonB(:, dimIter, 2))/ &
            norm2(qMid(:)-geo%iCoords(:,pIter))
      end if
      if (norm2(qMid(:)-geo%iCoords(:,pIter+1))>1d-16) then
        geo%dL(dimIter,pIter) = geo%dL(dimIter,pIter) + &
          dot_product((qMid(:)-geo%iCoords(:,pIter+1)),&
            mid_wilsonB(:, dimIter, 1))/&
            norm2(qMid(:)-geo%iCoords(:,pIter+1))
        geo%dL(dimIter,pIter+1) = geo%dL(dimIter,pIter+1) + &
          dot_product((qMid(:)-geo%iCoords(:,pIter+1)),&
            (mid_wilsonB(:, dimIter, 2)-&
             geo%wilsonB(:, dimIter, pIter+1)))/ &
            norm2(qMid(:)-geo%iCoords(:,pIter+1))
      end if
    end do
  end do
end subroutine geodesic_updateL

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_optPath
!!
!! FUNCTION
!! Call the core routine to optimize a geodesic path from existing points
!!
!! SYNOPSIS
subroutine geodesic_optPath(geo, eps, incPtNr)
!! SOURCE
  class(geodesic_type)          ::  geo
  real(rk), intent(in)          ::  eps
  logical, intent(in)           ::  incPtNr
  logical                       ::  success
  success = .false.
  if (geo%nPts>3) call geo%completePreOpt(MIN(eps,1d-2))
  do while (.not.success)
    call geo%coreOptPath(eps,incPtNr,success)
  end do  
end subroutine geodesic_optPath

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_completePreOpt
!!
!! FUNCTION
!! Cour routine to optimize a geodesic path from existing points
!!
!! SYNOPSIS
subroutine geodesic_completePreOpt(geo, eps)
!! SOURCE
  use oop_lbfgs
  class(geodesic_type)          ::  geo
  real(rk), intent(in)          ::  eps
  real(rk), allocatable         ::  curr_pts(:)
  real(rk), allocatable         ::  gradient(:)
!   real(rk), allocatable         ::  old_pts(:)
  real(rk), allocatable         ::  step(:)
  real(rk)                      ::  maxStepSize
  integer                       ::  i, dXruns
  real(rk)                      ::  DeltaX
  type(oop_lbfgs_type)          ::  lbfgs
  maxStepSize = 1d-1
  ! conv_MaxStep will be set below (for every optimization step)
  dXruns = 0
  allocate(curr_pts((geo%nPts-2)*geo%nDim))
  allocate(gradient((geo%nPts-2)*geo%nDim))
!   allocate(old_pts((geo%nPts-2)*geo%nDim))
  allocate(step((geo%nPts-2)*geo%nDim))
  call lbfgs%init(geo%nDim*(geo%nPts-2), &
                  MAX(geo%nDim*(geo%nPts-2)/2,50), .false.)
  call geo%updateL()
  do i = 1, geo%nPts-2
    curr_pts((i-1)*geo%nDim+1:i*geo%nDim) = geo%xCoords(:,i+1)
    gradient((i-1)*geo%nDim+1:i*geo%nDim) = geo%dL(:,i+1)
  end do
  DeltaX = eps*1d1
  do while (DeltaX > eps)
    dXruns = dXruns + 1
    call lbfgs%next_step(curr_pts,gradient,step)
    if (MAXVAL(abs(step))>maxStepSize) then
      step(:) = step(:) /MAXVAL(abs(step))*maxStepSize
    end if
    curr_pts(:) = curr_pts(:) + step(:)
    do i = 1, geo%nPts-2
      geo%xCoords(:,i+1) = curr_pts((i-1)*geo%nDim+1:i*geo%nDim)
    end do
    call geo%updateL()
    do i = 1, geo%nPts-2
      gradient((i-1)*geo%nDim+1:i*geo%nDim) = geo%dL(:,i+1)
    end do
    DeltaX = MAXVAL(abs(geo%dL(:,2:geo%nPts-1))) !+ &
!              MAXVAL(abs(step(:)))
!     if (MOD(dXruns,10)==0) write(stdout,fmt='(A,ES10.3,A,ES10.3,A,I6)', advance='no') &
!                            " preOpt: DeltaX ", DeltaX, " of ", eps, &
!                            " in step ", dXruns
!     write(*,"(A)", advance='no') "."
  end do

!   do pIter = 2, geo%nPts    
!     if (geo%nAtoms==3) then 
!       call adapt3AtomLin(geo%xCoords(:,1),geo%xCoords(:,pIter))
!     else
!       call dlf_cartesian_align(geo%nAtoms, geo%xCoords(:,1),geo%xCoords(:,pIter))
!     end if
!   end do
  call lbfgs%destroy()
  deallocate(curr_pts)
  deallocate(gradient)
!   deallocate(old_pts)
  deallocate(step)
end subroutine geodesic_completePreOpt

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_CoreOptPath
!!
!! FUNCTION
!! Cour routine to optimize a geodesic path from existing points
!!
!! SYNOPSIS
subroutine geodesic_CoreOptPath(geo, eps, incPtNr, success)
!! SOURCE
  use oop_lbfgs
  class(geodesic_type)          ::  geo
  real(rk), intent(in)          ::  eps
  logical, intent(in)           ::  incPtNr
  logical, intent(out)          ::  success
  real(rk)                      ::  DeltaX, deltaXComp,&
                                    minDeltaX_reached
  integer                       ::  pIter, piter2, inc, m, nOptSteps, refStr
  type(oop_lbfgs_type)          ::  lbfgs
  integer                       ::  nPtsAdd
  logical                       ::  add_pts(geo%nPts)
  real(rk)                      ::  MaxStep_lbfgs, conv_MaxStep,&
                                    limitStep,&
                                    lbfgs_step(geo%nDim), &
                                    new_coord(geo%nDim), &
                                    old_coord(geo%nDim), &
                                    L_lower, L_upper, &
                                    k, xTmp(geo%nDim), &
                                    Ls_lower(geo%nPts-1), &
                                    Ls_upper(geo%nPts-1), &
                                    Ls_diff(geo%nPts-1), tp,&
                                    distances(geo%nPts-1),&
                                    qA(geo%nInts),qB(geo%nInts)
  real(rk), allocatable         ::  x_save(:,:)
  real(rk)                      ::  min_reached
  integer                       ::  maxlocation, minLocation
  integer                       ::  nImgAddMax, i
  integer                       ::  nCompleteRuns
  integer                       ::  dXruns, plotIter
  logical                       ::  inverted
    
  nImgAddMax = 6
        
  ! conv_MaxStep will be set below (for every optimization step)
  success = .false.
  limitStep = 1d-2
  DeltaX = eps*1d1
  minDeltaX_reached = DeltaX
  dXruns = 0
  do while (DeltaX > eps)
    dXruns = dXruns + 1
    DeltaX = 0d0
    inc = 1
    pIter = 2    
    !*****************************************************
    ! loop to optimize the path
    nCompleteRuns = 0
    do while (pIter>1) ! loop from 2 to nPts-1 and back
      deltaXComp = 0d0
      ! minimize the length of the path with lbfgs
      ! by adjusting only image I
      call lbfgs%init(geo%nDim, MAX(geo%nDim/2,50), .false.)
      old_coord(:) = geo%xCoords(:,pIter)
        ! convergence criterion for the lbfgs optimization of the length
      conv_MaxStep = MIN(eps*1d-2,1d-7)
      MaxStep_lbfgs = 1d-1!MIN(conv_MaxStep*1d6,1d-1)
      nOptSteps = 0
      min_reached = 1d0
      ! updateL updates iCoords, wilsonB, dL, and L
      call geo%updateL()
      if (norm2(geo%dL(:,pIter))<1d-16) then
        if (printl>=6) write(stdout,'("Gradient of Length is zero.")')
        MaxStep_lbfgs = 0d0 ! exit equivalent
      end if
      do while((MAXVAL(ABS(geo%dL(:,pIter))) > eps*1d-2).or.&
               (MaxStep_lbfgs > conv_MaxStep))
        if (nOptSteps>0) min_reached = min(min_reached,MaxStep_lbfgs)
        nOptSteps = nOptSteps + 1
!         if (mod(nOptSteps,500)==0) then
!           conv_MaxStep = MIN(eps*1d-2,1d-4)
! !                   nOptSteps, MaxStep_lbfgs, conv_MaxStep, min_reached
!         end if
        if (nOptSteps==1000) then
          ! punish the convergence criterion, if so many
          ! steps are taken and search is aborted.
          deltaXComp = deltaXComp + eps
          exit
        end if
!         call geo%testGrad_FD_L(pIter)
        
        
        call lbfgs%next_step(geo%xCoords(:,pIter),&
               geo%dL(:,pIter),lbfgs_step(:),inverted)
        
!         if (inverted.and.nOptSteps>999&
!             .and.MOD(nOptSteps,100)==0) then
!         end if
          if (inverted) then
            if (printl>=4) write(stdout,'( "LBFGS inverted ...")')
          end if
        if (norm2(lbfgs_step)>limitStep) then
          lbfgs_step(:) = lbfgs_step(:) / &
                          norm2(lbfgs_step) * limitStep
        else if (norm2(lbfgs_step)<=1d-16) then
          ! step is zero (gradient was zero as well) -> same points
          exit
        end if
        geo%xCoords(:,pIter) = geo%xCoords(:,pIter) + lbfgs_step(:)
        MaxStep_lbfgs = MAXVAL(ABS(lbfgs_step(:)))
        ! updateL updates iCoords, wilsonB, dL, and L
        call geo%updateL()
        if (norm2(geo%dL(:,pIter))<1d-16) then
          if (printl>=6) write(stdout,'("Gradient of Length is zero.")')
          exit
        end if
      end do
      ! minimization wrt. point/image pIter complete
      !---------------------------------------------
      new_coord(:) = geo%xCoords(:,pIter)
      deltaXComp = deltaXComp + &
        norm2(new_coord(:)-old_coord(:)) !+ norm2(geo%dL(:,pIter))
      !---------------------------------------------
      ! for the loops iterator
      call lbfgs%destroy()
      pIter = pIter + inc
      if (pIter==geo%nPts) then 
        ! if we reached geo%nPts-1 we must do that point again
        ! therefore, pIter is not increased/decreased
        ! but in the next steps it must be decreased:
        nCompleteRuns = nCompleteRuns + 1
        inc = -1
        pIter = pIter + inc - 1
        deltaXComp = 0d0
      end if
      DeltaX = MAX(DeltaX,deltaXComp)
    end do
    minDeltaX_reached = MIN(minDeltaX_reached, deltaXComp)
    if (MOD(dXruns,100)==0) then
      write(stdout,fmt='(A,ES10.3,A,ES10.3)') "  DeltaX ", deltaXComp, " of ", eps
    end if
  end do
  write(stdout,fmt='(A,ES10.3,A,ES10.3)') "  DeltaX ", deltaXComp, " of ", eps
!   do pIter = 2, geo%nPts    
!     if (geo%nAtoms==3) then 
!       call adapt3AtomLin(geo%xCoords(:,1),geo%xCoords(:,pIter))
!     else
!       call dlf_cartesian_align(geo%nAtoms, geo%xCoords(:,1),geo%xCoords(:,pIter))
!     end if
!   end do
  !*****************************************************
  ! check boundaries and decide whether to restart
  
  ! lower bound
  L_lower = 0d0
  do pIter = 1, geo%nPts-1
    Ls_lower(pIter) = norm2(geo%iCoords(:,pIter+1)-&
                              geo%iCoords(:,pIter))
  end do
  L_lower = sum(Ls_lower)
  
  ! upper bound
  m = 10
  L_upper = 0d0
  do pIter = 1, geo%nPts-1
    ! calculate the different qs
!     L_upper = L_upper + geo%geodesic_L_upper(m,geo%xCoords(:,pIter),&
!                           geo%xCoords(:,pIter+1), q_work)
    Ls_upper(pIter) = geo%geodesic_L_upper(m,geo%xCoords(:,pIter),&
                          geo%xCoords(:,pIter+1))
    distances(pIter) = norm2(geo%xCoords(:,pIter+1)-geo%xCoords(:,pIter))
  end do
  L_upper = sum(Ls_upper)
  call geo%updateL()
  tp = geo%L
  geo%L = tp
  if (printl>=6) then
    write(stdout,'("boundary check: L", ES11.4)') geo%L
    write(stdout,'("boundary check: L_lower", ES11.4, ES11.4)')  L_lower, L_lower/0.95d0
    write(stdout,'("boundary check: L_upper", ES11.4, ES11.4)') L_upper, L_upper/1.1d0
    write(stdout,'("geo%nPts, geo%nImgGoal", I8, ES11.4)') geo%nPts, geo%nImgGoal
    write(stdout,'("incPtNr", I8)')incPtNr
    write(stdout,'("boundcond", L)') (L_lower<0.95d0*geo%L.or.L_upper>1.1d0*geo%L)
  end if
  if ((geo%nPts<geo%nImgGoal).or.&
      (incPtNr.and.(L_lower<0.95d0*geo%L.or.L_upper>1.1d0*geo%L))) then
    if (printl>=6) write(stdout,'("Adding new points"  )')
    ! Calculating bounds for each segment between neighboring images
    ! -> already done in Ls_upper and Ls_lower
    ! Check where difference of bounds is too large
    nPtsAdd = 0
    add_pts = .false.
    do pIter = 1, geo%nPts-1
      if (Ls_lower(pIter)*1.1d0/0.95d0<Ls_upper(pIter)) then
!       if (Ls_lower(pIter)>L_lower/(geo%nPts-1) .or. Ls_upper(pIter)>L_upper/(geo%nPts-1)) then
        nPtsAdd = nPtsAdd + 1
        add_pts(pIter) = .true.
        if (printl>=6) write(stdout,'("Adding a point after point nr", I8)') pIter
        if (geo%nImgGoal - geo%nPts == nPtsAdd.and.(.not.incPtNr)) exit
      end if
    end do
    Ls_diff = Ls_upper - Ls_lower
    ! if there are too many points added, reduce the number to nImgAddMax
    do while (nPtsAdd>=nImgAddMax)
      min_reached = 9d99
      do i = 1, geo%nPts
        if (add_pts(i)) then
          if (Ls_diff(i)<min_reached) then
            min_reached = Ls_diff(i)
            minLocation = i
          end if
        end if
      end do
      nPtsAdd = nPtsAdd - 1
      add_pts(minLocation) = .false.
    end do
    ! if there are too few points added, increase the number of points
    do while (nPtsAdd<nImgAddMax.and.geo%nPts+nPtsAdd<geo%nImgGoal.and.nPtsAdd<geo%nPts-1)
      ! Add an image where the Lower-Upper-difference is the largest
      maxlocation = SUM(MAXLOC(Ls_diff))
      if (.not.add_pts(maxlocation)) then
        nPtsAdd = nPtsAdd + 1
        add_pts(maxlocation) = .true.
      end if
      Ls_diff(maxlocation) = -1d0
!       add_pts(SUM(MAXLOC(distances))) = .true.
    end do
    ! Add midpoints for all respective pairs
    if (nPtsAdd>0) then
      allocate(x_save(geo%nDim, geo%nPts))
      x_save = geo%xCoords
      deallocate(geo%xCoords)
      allocate(geo%xCoords(geo%nDim, geo%nPts+nPtsAdd))
      piter2 = 1
      do pIter = 1, geo%nPts-1
        geo%xCoords(:,pIter2) = x_save(:,pIter)
        pIter2 = pIter2 + 1
        if (add_pts(pIter)) then
          geo%xCoords(:,pIter2) = (x_save(:,pIter+1)+x_save(:,pIter))/2d0
!           if (geo%nAtoms==3) then 
! !             print*, "adapt3AtomLin"
!             call adapt3AtomLin(geo%xCoords(:,pIter2),&
!                                geo%xCoords(:,pIter2-1))
!           else
! !             print*, "dlf_cartesian_align"
! !             print*, "coords1",geo%xCoords(:,pIter2-1)
! !             print*, "coords1",reshape(geo%xCoords(:,pIter2-1),(/3,geo%nAtoms/))
! !             print*, "coords2",geo%xCoords(:,pIter2)
! !             print*, "coords2",reshape(geo%xCoords(:,pIter2),(/3,geo%nAtoms/))
!             call dlf_cartesian_align(geo%nAtoms, &
!                                      reshape(geo%xCoords(:,1),(/3,geo%nAtoms/)),&
!                                      reshape(geo%xCoords(:,pIter2),(/3,geo%nAtoms/)))
!           end if
          pIter2 = pIter2 + 1          
        end if        
      end do
      geo%xCoords(:,pIter2) = x_save(:,geo%nPts)
      if (geo%nPts+nPtsAdd/=pIter2) call dlf_fail("DAMN!")
      geo%nPts = pIter2
      if (printl>=6) write(stdout,'("Added new points, new number of pts: ", I8)') geo%nPts
      ! resizing other arrays
      deallocate(geo%iCoords)
      allocate(geo%iCoords(geo%nInts,geo%nPts)) ! last two are temporarily used
      deallocate(geo%wilsonB)
      allocate(geo%wilsonB(geo%nInts,geo%nDim,geo%nPts)) ! last two are temporarily used
      deallocate(geo%dL)
      allocate(geo%dL(geo%nDim,geo%nPts))
      deallocate(x_save)
    else
      call dlf_fail("Ok, so maybe chose another criterion: 'bounds too large'?")
    end if
    ! restart procedure with larger set of images
    if (printl>=6) write(stdout,'("Restarting the procedure with larger set of images.")')
    ! success is still .false.
  else
    ! procedure finished with current coordinates
    ! giving the approximate geodesic.
    if (printl>=4) write(stdout,'("Geodesic created.")')
    if (printl>=4) write(stdout,'(" ")')
    if (printl>=4) write(stdout,'(" ")')
    success = .true.
  end if
  do pIter = 2, geo%nPts
    call dlf_cartesian_align(geo%nAtoms, geo%xCoords(:,1),geo%xCoords(:,pIter))
    if (geo%nAtoms==3) then 
      call adapt3AtomLin(geo%xCoords(:,1),geo%xCoords(:,pIter))
    end if
  end do
end subroutine geodesic_CoreOptPath

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_L_upper
!!
!! FUNCTION
!! Calculate the upper bound for the length L, given 
!! the x-coordinates of the current and the next image
!!
!! SYNOPSIS
function geodesic_L_upper(geo, m, xCoords_1, xCoords_2) result(upper)
!! SOURCE
  class(geodesic_type)          ::  geo
  real(rk)                      ::  upper
  integer, intent(in)           ::  m
  real(rk), intent(in)          ::  xCoords_1(geo%nDim),&
                                    xCoords_2(geo%nDim)
  real(rk)                      ::  q(geo%nInts,m+1) ! working array
  integer                       ::  mIter, k
  real(rk)                      ::  xTmp(geo%nDim)
  upper = 0d0
  do mIter = 1, m+1
    k = mIter - 1
    xTmp(:) = (REAL((m-k),kind=rk)*xCoords_1(:)+k*xCoords_2(:))/REAL(m,kind=rk)
    call geo%xToI(xTmp, q(:,mIter))
  end do
  do mIter = 1, m
    upper = upper + norm2(q(:,mIter+1)-q(:,mIter))
  end do
end function geodesic_L_upper


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_xToI
!!
!! FUNCTION
!! Transform xCoords to iCoords
!!
!! SYNOPSIS
subroutine geodesic_xToI(geo,xCoords,iCoords)
!! SOURCE
  class(geodesic_type)          ::  geo
  real(rk), intent(in)          ::  xCoords(geo%nDim)
  real(rk), intent(out)         ::  iCoords(geo%nInts)
  integer                       ::  atIter1, atIter2, iIter, &
                                    dimOffset1, dimOffset2
  real(rk)                      ::  r, rekl

  iCoords(:) = 0d0
  iIter = 0
  
    do atIter1 = 1, geo%nAtoms
      do atIter2 = 1, geo%nAtoms
        if (atIter1==atIter2) cycle ! this internal coordinate does not exist
        iIter = iIter + 1
        dimOffset1 = 3*(atIter1-1)
        dimOffset2 = 3*(atIter2-1)
        r = norm2(xCoords(dimOffset1+1:dimOffset1+3)-&
                  xCoords(dimOffset2+1:dimOffset2+3))
        rekl = geo%get_added_cov_radius(geo%znuc(atIter1),geo%znuc(atIter2))
        iCoords(iIter) = &
            dexp(-geo%alpha/rekl*(r-rekl))+geo%beta*rekl/r
      end do
    end do
end subroutine geodesic_xToI

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_dIdx
!!
!! FUNCTION
!! Transform xCoords to iCoords
!!
!! SYNOPSIS
subroutine geodesic_dIdx(geo,xCoords,dqdx)
!! SOURCE
  class(geodesic_type)          ::  geo
  real(rk), intent(in)          ::  xCoords(geo%nDim) ! cartesians
  ! derivative of q wrt x
  real(rk), intent(out)         ::  dqdx(geo%nInts, geo%nDim)
  real(rk)                      ::  tmp(3), tmp2(3)
  integer                       ::  k,l ! which atoms
  integer                       ::  iIter ! internals-iterator
  integer                       ::  kdimIter, ldimIter
  
  dqdx = 0d0
  iIter = 0
  kdimIter = -3
  do k = 1, geo%nAtoms
    ldimIter = -3
    kdimIter=kdimIter+3
    do l = 1, geo%nAtoms
      ldimIter=ldimIter+3
      if (k==l) cycle ! this internal coordinate does not exist
      iIter = iIter + 1   
      tmp(1:3) = xCoords((k-1)*3+1:k*3)-xCoords((l-1)*3+1:l*3)
      ! derivative wrt. first element x_k
      call geo%dIdx_core(tmp,&
        geo%get_added_cov_radius(geo%znuc(k),geo%znuc(l)),.true., &
        tmp2)
        dqdx(iIter,kdimIter+1:kdimIter+3) = tmp2
      ! derivative wrt. second element x_l
      call geo%dIdx_core(tmp,&
        geo%get_added_cov_radius(geo%znuc(k),geo%znuc(l)),.false., &
        tmp2)
        dqdx(iIter,ldimIter+1:ldimIter+3) = tmp2
    end do
  end do
end subroutine geodesic_dIdx


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* geodesic/geodesic_dIdx_core
!!
!! FUNCTION
!! Transform xCoords to iCoords
!!
!! SYNOPSIS
subroutine geodesic_dIdx_core(geo,r_vec,rekl,first,dIdx)
!! SOURCE
  class(geodesic_type)          ::  geo
  real(rk), intent(in)          ::  r_vec(3) ! xk-xl
  real(rk), intent(in)          ::  rekl
  logical, intent(in)           ::  first
  real(rk), intent(out)         ::  dIdx(3)
  real(rk)                      ::  r
  integer                       ::  atIter1, atIter2, iIter, &
                                    dimOffset1, dimOffset2
  r = norm2(r_vec)
  if (r<1d-16) then
    dIdx = 0d0
    return
  end if
  if (rekl<1d-16) call dlf_fail("Unrealistic rekl")
  if (r>1000d0) then
    if (printl>=6) write(stdout,'("large distance, interaction will be set to 0")')
    return
  end if
  dIdx = -geo%alpha/rekl*dexp(-geo%alpha/rekl*(r-rekl))-geo%beta*rekl/r**2
  if(first) then
    dIdx(:) =  dIdx(:) * r_vec(:)/r
  else
    dIdx(:) = -dIdx(:) * r_vec(:)/r
  end if
end subroutine geodesic_dIdx_core

! ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! !!****f* geodesic/geodesic_get_added_cov_radius
! !!
! !! FUNCTION
! !! Wilson B Matrix
! !!
! !! SYNOPSIS
function geodesic_get_added_cov_radius(geo, atomic_number1, atomic_number2) result(covR)
!! SOURCE
  class(geodesic_type)::  geo
  integer, intent(in) :: atomic_number1, atomic_number2
  real(rk)            :: covR
  real(rk), external  :: get_cov_radius
  covR = get_cov_radius(atomic_number1) + get_cov_radius(atomic_number2)
end function geodesic_get_added_cov_radius


subroutine testGrad_FD_L(geo, pIter)
  class(geodesic_type)  ::  geo
  integer, intent(in)   ::  pIter
  real(rk)              ::  grad(geo%nDim)
  real(rk)              ::  fd_grad(geo%nDim)
  real(rk)              ::  L1,L0
  real(rk)              ::  delta
  real(rk)              ::  position_save(geo%nDim)
  real(rk)              ::  tmpPos(geo%nDim)
  integer               ::  dimIter
  position_save(:) = geo%xCoords(:,pIter)
  call geo%updateL()
  L0 = geo%L
  grad(:) = geo%dL(:,pIter)
  delta = 1d-1
  do while (delta>1d-8)
    do dimIter = 1, geo%nDim
      tmpPos(:) = 0d0
      tmpPos(dimIter) = delta
      tmpPos(:) = position_save(:) + tmpPos(:)
      geo%xCoords(:,pIter) = tmpPos(:)
      call geo%updateL()
      L1 = geo%L
      fd_grad(dimIter) = (L1-L0)/delta
      write(stdout,fmt='(A,ES10.3,I3,3X,4ES10.3)') "FDGRAD:", &
              delta, &
              dimIter,&
              grad(dimIter),&
              fd_grad(dimIter),&
              grad(dimIter)/fd_grad(dimIter),&
              grad(dimIter)-fd_grad(dimIter)
    end do
    delta = delta*1d-1
  end do
  geo%xCoords(:,pIter) = position_save(:)
  call geo%updateL()
end subroutine testGrad_FD_L

end module geodesic_module

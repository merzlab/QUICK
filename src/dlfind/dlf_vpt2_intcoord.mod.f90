! Module for handling internal (Z matrix) <-> Cartesian coordinate transformations

module dlf_vpt2_intcoord
use dlf_parameter_module
!use dlf_global, only: glob, stdout
implicit none

contains

!***************************************************
!***************************************************

! Output Z matrix coordinates to funit

subroutine int_punch(nat,intn,nbonds,nbangles,ntorsions,def,bond_vals,angle_vals,tors_vals,funit)
use dlf_constants
implicit none
integer, intent(in) :: nat, intn, nbonds, nbangles, ntorsions
integer, intent(in), dimension(intn,4) :: def
real(rk), intent(in), dimension(nbonds)    :: bond_vals
real(rk), intent(in), dimension(nbangles)  :: angle_vals
real(rk), intent(in), dimension(ntorsions) :: tors_vals
integer, intent(in) :: funit
logical, parameter :: hi_accuracy=.false.
character(len=400) :: formt

integer :: i
integer :: bonds(nbonds,2)
integer :: bangles(nbangles,3)
integer :: torsions(ntorsions,4)
real(rk) :: autoang
real(rk) :: radtodeg

call dlf_constants_get("ANG_AU",autoang)
call dlf_constants_get("PI",radtodeg)
radtodeg=180._rk/radtodeg

bonds(1:nbonds,1:2)       = def(1:nbonds,1:2)
bangles(1:nbangles,1:3)   = def(nbonds+1:nbonds+nbangles,1:3)
torsions(1:ntorsions,1:4) = def(nbonds+nbangles+1:nbonds+nbangles+ntorsions,1:4)

write(funit,'(A)') ''
write(funit,'(A)') 'Bond distances:'
if (hi_accuracy) then
  formt='(2I5,ES23.15)'
else
  formt='(2I5,F15.4)'
endif
do i=1,nbonds
  write(funit,formt) bonds(i,1), bonds(i,2), bond_vals(i)*autoang
enddo
write(funit,'(A)') ''
write(funit,'(A)') 'Angles:'
if (hi_accuracy) then
  formt='(3I5,ES23.15)'
else
  formt='(3I5,F15.4)'
endif
do i=1,nbangles
  write(funit,formt) bangles(i,1), bangles(i,2), bangles(i,3), angle_vals(i)*radtodeg
enddo
write(funit,'(A)') ''
write(funit,'(A)') 'Torsions:'
if (hi_accuracy) then
  formt='(4I5,ES23.15)'
else
  formt='(4I5,F15.4)'
endif
do i=1,ntorsions
  write(funit,formt) torsions(i,1), torsions(i,2), torsions(i,3), torsions(i,4), tors_vals(i)*radtodeg
enddo
write(funit,'(A)') ''

end subroutine int_punch

!***************************************************
!***************************************************

! Cartesian to Z matrix conversion

subroutine cart_to_int(nat,intn,nbonds,nbangles,ntorsions,def,cart,bond_vals,angle_vals,tors_vals)
implicit none
integer, intent(in) :: nat, intn, nbonds, nbangles, ntorsions
integer, intent(in), dimension(intn,4) :: def
real(rk), intent(in),  dimension(3*nat) :: cart
real(rk), intent(out), dimension(nbonds)    :: bond_vals
real(rk), intent(out), dimension(nbangles)  :: angle_vals
real(rk), intent(out), dimension(ntorsions) :: tors_vals

real(rk), dimension(1:3,1:nat) :: coo
real(rk), dimension(1:3) :: x1,x2,x3,x4
integer :: i
integer :: bonds(nbonds,2)
integer :: bangles(nbangles,3)
integer :: torsions(ntorsions,4)

!! This would be for Cartesian coordinates in 1D format xxx...yyy...zzz...
!coo(1,1:nat)= cart(1:nat)        
!coo(2,1:nat)= cart(nat+1:2*nat)  
!coo(3,1:nat)= cart(2*nat+1:3*nat)

! We have coordinates in 1D format xyzxyz...
coo(1,1:nat)= cart(1:3*nat-2:3)
coo(2,1:nat)= cart(2:3*nat-1:3)
coo(3,1:nat)= cart(3:3*nat  :3)

bonds(1:nbonds,1:2)       = def(1:nbonds,1:2)
bangles(1:nbangles,1:3)   = def(nbonds+1:nbonds+nbangles,1:3)
torsions(1:ntorsions,1:4) = def(nbonds+nbangles+1:nbonds+nbangles+ntorsions,1:4)

do i=1,nbonds
  x1(1:3)=coo(1:3,bonds(i,1))
  x2(1:3)=coo(1:3,bonds(i,2))
  bond_vals(i)=distance(x1,x2)
enddo
do i=1,nbangles
  x1(1:3)=coo(1:3,bangles(i,1))
  x2(1:3)=coo(1:3,bangles(i,2))
  x3(1:3)=coo(1:3,bangles(i,3))
  angle_vals(i)=bendangle(x1,x2,x3)
enddo
do i=1,ntorsions
  x1(1:3)=coo(1:3,torsions(i,1))
  x2(1:3)=coo(1:3,torsions(i,2))
  x3(1:3)=coo(1:3,torsions(i,3))
  x4(1:3)=coo(1:3,torsions(i,4))
  tors_vals(i)=torsangle(x1,x2,x3,x4)
enddo

return
end subroutine cart_to_int

!!!***************************************************
!!!***************************************************
!!
!!subroutine test_cart_to_int(nat,intn,cart0,mv,atsym)
!!use random_sprng_interface
!!use dlf_vpt2_utility, only: matrix_output
!!use dlf_deallocate, only: allocate, deallocate
!!implicit none
!!integer, intent(in) :: nat,intn
!!real(rk), intent(in),  dimension(3*nat) :: cart0
!!real(rk), intent(in),  dimension(nat) :: mv
!!character(len=2), dimension(nat),intent(in) :: atsym
!!
!!real(rk), parameter :: mod_amplitude=0.5_rk
!!integer :: nbonds, nbangles, ntorsions, iseed, i, j, k
!!integer, dimension(intn,4) :: def
!!real(rk), dimension(3*nat) :: cart_new,delcart
!!real(rk), dimension(intn,3*nat)        :: B
!!real(rk), dimension(3*nat,3*nat,intn)  :: C
!!real(rk), dimension(intn)  :: del_int_esti, del_int_esti_2nd
!!real(rk), allocatable, dimension(:)    :: bond_vals_init,bond_vals_modi,bond_vals_esti
!!real(rk), allocatable, dimension(:)    :: bond_vals_esti_2nd
!!real(rk), allocatable, dimension(:)  :: angle_vals_init,angle_vals_modi,angle_vals_esti
!!real(rk), allocatable, dimension(:)  :: angle_vals_esti_2nd
!!real(rk), allocatable, dimension(:) :: tors_vals_init,tors_vals_modi,tors_vals_esti
!!real(rk), allocatable, dimension(:) :: tors_vals_esti_2nd
!!integer, allocatable, dimension(:,:) :: bonds
!!integer, allocatable, dimension(:,:) :: bangles
!!integer, allocatable, dimension(:,:) :: torsions
!!
!!call read_int_coord_def(intn,nbonds,nbangles,ntorsions,def)
!!
!!call allocate(bond_vals_init,nbonds)
!!call allocate(bond_vals_modi,nbonds)
!!call allocate(bond_vals_esti,nbonds)
!!call allocate(bond_vals_esti_2nd,nbonds)
!!call allocate(angle_vals_init,nbangles)
!!call allocate(angle_vals_modi,nbangles)
!!call allocate(angle_vals_esti,nbangles,)
!!call allocate(angle_vals_esti_2nd,nbangles)
!!call allocate(tors_vals_init,ntorsions)
!!call allocate(tors_vals_modi,ntorsions)
!!call allocate(tors_vals_esti,ntorsions)
!!call allocate(tors_vals_esti_2nd,ntorsions)
!!
!!call allocate(bonds,nbonds,2)
!!call allocate(bangles,nbangles,3)
!!call allocate(torsions,ntorsions,4)
!!
!!bonds(1:nbonds,1:2)       = def(1:nbonds,1:2)
!!bangles(1:nbangles,1:3)   = def(nbonds+1:nbonds+nbangles,1:3)
!!torsions(1:ntorsions,1:4) = def(nbonds+nbangles+1:nbonds+nbangles+ntorsions,1:4)
!!
!!call cart_to_int(nat,intn,nbonds,nbangles,ntorsions,def, &
!!               & cart0,bond_vals_init,angle_vals_init,tors_vals_init)
!!
!!call generate_B_C_ridders(nat,intn,cart0,nbonds,nbangles, &
!!                           & ntorsions,bonds,bangles,torsions,B,C)
!!
!!call matrix_output(B,stdout,'F20.12','Wilson B matrix')
!!
!!write(stdout,'(A)') '°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°'
!!write(stdout,'(A)') 'test_cart_to_int'
!!write(stdout,'(A)') '°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°'
!!write(stdout,'(A)') ''
!!write(stdout,'(A)') 'Initial Cartesians:'
!!write(stdout,'(A)') ''
!!do i=1,nat
!!  write(stdout,'(A2,1X,3ES20.12)') atsym(i), (cart0(j)*autoang, j=3*(i-1)+1,3*(i-1)+3)
!!enddo
!!write(stdout,'(A)') ''
!!
!!call int_punch(nat,intn,nbonds,nbangles,ntorsions,def,bond_vals_init, & 
!!               & angle_vals_init,tors_vals_init,6)
!!
!!if (startseed==0) startseed=100
!!call initialize_iseed(startseed)
!!
!!do i=1,3*nat
!!  cart_new(i)=cart0(i)+ran_real(-mod_amplitude,mod_amplitude,iseed)
!!enddo
!!
!!delcart(:)=cart_new(:)-cart0(:)
!!del_int_esti(:)=dlf_matmul_simp(B(:,:),delcart(:))
!!bond_vals_esti(1:nbonds)=bond_vals_init(1:nbonds)+del_int_esti(1:nbonds)
!!angle_vals_esti(1:nbangles)=angle_vals_init(1:nbangles)+del_int_esti(nbonds+1:nbonds+nbangles)
!!tors_vals_esti(1:ntorsions)=tors_vals_init(1:ntorsions)+del_int_esti(nbonds+nbangles+1:nbonds+nbangles+ntorsions)
!!
!!del_int_esti_2nd(:)=0._rk
!!do i=1,3*nat
!!  do j=1,3*nat
!!    do k=1,intn
!!      del_int_esti_2nd(k)=del_int_esti_2nd(k)+C(i,j,k)*delcart(i)*delcart(j)
!!    enddo
!!  enddo
!!enddo
!!del_int_esti_2nd(:)=del_int_esti_2nd(:)/2._rk
!!
!!bond_vals_esti_2nd(1:nbonds)=bond_vals_esti(1:nbonds)+del_int_esti_2nd(1:nbonds)
!!angle_vals_esti_2nd(1:nbangles)=angle_vals_esti(1:nbangles)+del_int_esti_2nd(nbonds+1:nbonds+nbangles)
!!tors_vals_esti_2nd(1:ntorsions)=tors_vals_esti(1:ntorsions)+del_int_esti_2nd(nbonds+nbangles+1:nbonds+nbangles+ntorsions)
!!
!!write(stdout,'(A)') ''
!!write(stdout,'(A)') 'Modified Cartesians:'
!!write(stdout,'(A)') ''
!!do i=1,nat
!!  write(stdout,'(A2,1X,3ES20.12)') atsym(i), (cart_new(j)*autoang, j=3*(i-1)+1,3*(i-1)+3)
!!enddo
!!write(stdout,'(A)') ''
!!write(stdout,'(A)') 'Delta:'
!!write(stdout,'(A)') ''
!!do i=1,nat
!!  write(stdout,'(A2,1X,3F20.12)') atsym(i), ((cart_new(j)-cart0(j))*autoang, j=3*(i-1)+1,3*(i-1)+3)
!!enddo
!!write(stdout,'(A)') ''
!!
!!call cart_to_int(nat,intn,nbonds,nbangles,ntorsions,def, &
!!               & cart_new,bond_vals_modi,angle_vals_modi,tors_vals_modi)
!!
!!write(stdout,'(A)') ''
!!write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
!!write(stdout,'(A)') ' Summary of test_cart_to_int '
!!write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
!!write(stdout,'(A)') ''
!!write(stdout,'(A)') 'Bond distances:'
!!write(stdout,'(A)') ''
!!write(stdout,'(2A5,4A15)') 'ind1', 'ind2', 'initial', 'estim.','estim.(2nd)', 'actual'
!!write(stdout,'(A)') ''
!!do i=1,nbonds
!!  write(stdout,'(2I5,4F15.4)') bonds(i,1), bonds(i,2), bond_vals_init(i)*autoang, &
!!                  &   bond_vals_esti(i)*autoang,bond_vals_esti_2nd(i)*autoang, &
!!                  &   bond_vals_modi(i)*autoang
!!enddo
!!write(stdout,'(A)') ''
!!write(stdout,'(A)') 'Angles:'
!!write(stdout,'(A)') ''
!!write(stdout,'(3A5,4A15)') 'ind1', 'ind2', 'ind3', 'initial', 'estim.','estim.(2nd)', 'actual'
!!write(stdout,'(A)') ''
!!do i=1,nbangles
!!  write(stdout,'(3I5,4F15.4)') bangles(i,1), bangles(i,2), bangles(i,3), &
!!             & angle_vals_init(i)*radtodeg,angle_vals_esti(i)*radtodeg, &
!!             & angle_vals_esti_2nd(i)*radtodeg,angle_vals_modi(i)*radtodeg
!!enddo
!!write(stdout,'(A)') ''
!!write(stdout,'(A)') 'Torsions:'
!!write(stdout,'(A)') ''
!!write(stdout,'(4A5,4A15)') 'ind1', 'ind2', 'ind3', 'ind4', 'initial', 'estim.','estim.(2nd)', 'actual'
!!write(stdout,'(A)') ''
!!do i=1,ntorsions
!!  write(stdout,'(4I5,4F15.4)') torsions(i,1), torsions(i,2), torsions(i,3), torsions(i,4), &
!!        &  tors_vals_init(i)*radtodeg, tors_vals_esti(i)*radtodeg, tors_vals_esti_2nd(i)*radtodeg, &
!!        &  tors_vals_modi(i)*radtodeg
!!enddo
!!write(stdout,'(A)') ''
!!
!!call deallocate(torsions)
!!call deallocate(bangles)
!!call deallocate(bonds)
!!call deallocate(tors_vals_modi)
!!call deallocate(tors_vals_esti_2nd)
!!call deallocate(tors_vals_esti)
!!call deallocate(tors_vals_init)
!!call deallocate(angle_vals_modi)
!!call deallocate(angle_vals_esti_2nd)
!!call deallocate(angle_vals_esti)
!!call deallocate(angle_vals_init)
!!call deallocate(bond_vals_modi)
!!call deallocate(bond_vals_esti_2nd)
!!call deallocate(bond_vals_esti)
!!call deallocate(bond_vals_init)
!!
!!stop 0
!!
!!return
!!end subroutine test_cart_to_int
!!
!!!***************************************************
!!!***************************************************

!!subroutine test_int_to_cart(nat,intn,cart0,mv,atsym)
!!use random_sprng_interface
!!use dlf_allocate, only: allocate, deallocate
!!implicit none
!!integer, intent(in) :: nat,intn
!!real(rk), intent(in),  dimension(3*nat) :: cart0
!!real(rk), intent(in),  dimension(nat) :: mv
!!character(len=2), dimension(nat),intent(in) :: atsym
!!
!!real(rk), parameter :: mod_amplitude=1.5_rk
!!integer :: nbonds, nbangles, ntorsions, iseed, i, j
!!integer, dimension(intn,4) :: def
!!real(rk), dimension(3*nat) :: cart_new
!!real(rk), allocatable, dimension(:)    :: bond_vals_init,bond_vals_targ,bond_vals_calc
!!real(rk), allocatable, dimension(:)  :: angle_vals_init,angle_vals_targ,angle_vals_calc
!!real(rk), allocatable, dimension(:) :: tors_vals_init,tors_vals_targ,tors_vals_calc
!!integer, allocatable, dimension(:,:) :: bonds
!!integer, allocatable, dimension(:,:) :: bangles
!!integer, allocatable, dimension(:,:) :: torsions
!!
!!call read_int_coord_def(intn,nbonds,nbangles,ntorsions,def)
!!
!!call allocate(bond_vals_init(nbonds))
!!call allocate(bond_vals_targ(nbonds))
!!call allocate(bond_vals_calc(nbonds))
!!call allocate(angle_vals_init(nbangles))
!!call allocate(angle_vals_targ(nbangles))
!!call allocate(angle_vals_calc(nbangles))
!!call allocate(tors_vals_init(ntorsions))
!!call allocate(tors_vals_targ(ntorsions))
!!call allocate(tors_vals_calc(ntorsions))
!!call allocate(bonds(nbonds,2))
!!call allocate(bangles(nbangles,3))
!!call allocate(torsions(ntorsions,4))
!!
!!bonds(1:nbonds,1:2)       = def(1:nbonds,1:2)
!!bangles(1:nbangles,1:3)   = def(nbonds+1:nbonds+nbangles,1:3)
!!torsions(1:ntorsions,1:4) = def(nbonds+nbangles+1:nbonds+nbangles+ntorsions,1:4)
!!
!!call cart_to_int(nat,intn,nbonds,nbangles,ntorsions,def, &
!!               & cart0,bond_vals_init,angle_vals_init,tors_vals_init)
!!call int_punch(nat,intn,nbonds,nbangles,ntorsions,def,bond_vals_init, & 
!!               & angle_vals_init,tors_vals_init,6)
!!
!!if (startseed==0) startseed=100
!!call initialize_iseed(startseed)
!!
!!do i=1,nbonds
!!  bond_vals_targ(i)=bond_vals_init(i)+ran_real(-mod_amplitude,mod_amplitude,iseed)
!!enddo
!!do i=1,nbangles
!!  angle_vals_targ(i)=angle_vals_init(i)+ran_real(-mod_amplitude,mod_amplitude,iseed)
!!enddo
!!do i=1,ntorsions
!!  tors_vals_targ(i)=tors_vals_init(i)+ran_real(-mod_amplitude,mod_amplitude,iseed)
!!enddo
!!
!!call int_to_cart(nat,intn,nbonds,nbangles,ntorsions,def,cart0,mv, &
!!                   &   bond_vals_targ,angle_vals_targ, &
!!                   &   tors_vals_targ,cart_new,1.e-6_rk)
!!
!!call cart_to_int(nat,intn,nbonds,nbangles,ntorsions,def, &
!!               & cart_new,bond_vals_calc,angle_vals_calc,tors_vals_calc)
!!
!!write(stdout,'(A)') ''
!!write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
!!write(stdout,'(A)') ' Summary of test_int_to_cart '
!!write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
!!write(stdout,'(A)') ''
!!write(stdout,'(A)') 'Bond distances:'
!!write(stdout,'(A)') ''
!!write(stdout,'(2A5,3A15)') 'ind1', 'ind2', 'initial', 'target', 'actual'
!!write(stdout,'(A)') ''
!!do i=1,nbonds
!!  write(stdout,'(2I5,3F15.4)') bonds(i,1), bonds(i,2), bond_vals_init(i)*autoang, &
!!                     bond_vals_targ(i)*autoang, bond_vals_calc(i)*autoang
!!enddo
!!write(stdout,'(A)') ''
!!write(stdout,'(A)') 'Angles:'
!!write(stdout,'(A)') ''
!!write(stdout,'(3A5,3A15)') 'ind1', 'ind2', 'ind3', 'initial', 'target', 'actual'
!!write(stdout,'(A)') ''
!!do i=1,nbangles
!!  write(stdout,'(3I5,3F15.4)') bangles(i,1), bangles(i,2), bangles(i,3), &
!!             & angle_vals_init(i)*radtodeg,angle_vals_targ(i)*radtodeg, &
!!             & angle_vals_calc(i)*radtodeg
!!enddo
!!write(stdout,'(A)') ''
!!write(stdout,'(A)') 'Torsions:'
!!write(stdout,'(A)') ''
!!write(stdout,'(4A5,3A15)') 'ind1', 'ind2', 'ind3', 'ind4', 'initial', 'target', 'actual'
!!write(stdout,'(A)') ''
!!do i=1,ntorsions
!!  write(stdout,'(4I5,3F15.4)') torsions(i,1), torsions(i,2), torsions(i,3), torsions(i,4), &
!!        &  tors_vals_init(i)*radtodeg, tors_vals_targ(i)*radtodeg, tors_vals_calc(i)*radtodeg
!!enddo
!!write(stdout,'(A)') ''
!!
!!open(4444,file='test_int_to_cart.debug.xyz')
!!write(4444,*) nat
!!write(4444,'(A)') 'test_int_to_cart: debug output from int->cart conversion.'
!!do i=1,nat
!!  write(4444,'(A2,1X,3ES20.12)') atsym(i), (cart_new(j)*autoang, j=3*(i-1)+1,3*(i-1)+3)
!!enddo
!!close(4444)
!!
!!call deallocate(torsions)
!!call deallocate(bangles)
!!call deallocate(bonds)
!!call deallocate(tors_vals_calc)
!!call deallocate(tors_vals_targ)
!!call deallocate(tors_vals_init)
!!call deallocate(angle_vals_calc)
!!call deallocate(angle_vals_targ)
!!call deallocate(angle_vals_init)
!!call deallocate(bond_vals_calc)
!!call deallocate(bond_vals_targ)
!!call deallocate(bond_vals_init)
!!
!!stop 0
!!
!!return
!!end subroutine test_int_to_cart

!***************************************************
!***************************************************

! Z matrix to Cartesian coordinate conversion
! Iterative algorithm that needs an input guess cart0 
! for the Cartesians

subroutine int_to_cart(nat,intn,nbonds,nbangles,ntorsions,def,cart0,mv, &
                   &   bond_vals_target,angle_vals_target, &
                   &   tors_vals_target,cart_final,tol)
use dlf_vpt2_utility, only: error_print
use dlf_linalg_interface_mod
use dlf_linalg_interface_mod
implicit none
integer, intent(in) :: nat, intn, nbonds, nbangles, ntorsions
integer, intent(in), dimension(intn,4) :: def
real(rk), intent(in),  dimension(3*nat) :: cart0
real(rk), intent(in),  dimension(nat) :: mv
real(rk), intent(out), dimension(3*nat) :: cart_final
real(rk), intent(in), dimension(nbonds)    :: bond_vals_target
real(rk), intent(in), dimension(nbangles)  :: angle_vals_target
real(rk), intent(in), dimension(ntorsions) :: tors_vals_target
real(rk), intent(in) :: tol

integer, parameter :: maxiter=200
integer :: i
real(rk), dimension(3*nat) :: cart,delcart
real(rk), dimension(intn,3*nat)        :: B
real(rk), dimension(3*nat,3*nat,intn)  :: C
real(rk), dimension(3*nat,intn)  :: A
real(rk), dimension(nbonds)    :: bond_vals
real(rk), dimension(nbangles)  :: angle_vals
real(rk), dimension(ntorsions) :: tors_vals
real(rk), dimension(intn) :: intvals_target, intvals, delint
real(rk) :: del
integer  :: bonds(nbonds,2)
integer  :: bangles(nbangles,3)
integer  :: torsions(ntorsions,4)

bonds(1:nbonds,1:2)       = def(1:nbonds,1:2)
bangles(1:nbangles,1:3)   = def(nbonds+1:nbonds+nbangles,1:3)
torsions(1:ntorsions,1:4) = def(nbonds+nbangles+1:nbonds+nbangles+ntorsions,1:4)

intvals_target(1:nbonds)                                   =bond_vals_target(1:nbonds)
intvals_target(nbonds+1:nbonds+nbangles)                   =angle_vals_target(1:nbangles)
intvals_target(nbonds+nbangles+1:nbonds+nbangles+ntorsions)=tors_vals_target(1:ntorsions)

cart(:)=cart0(:)

do i=1,maxiter
  call cart_to_int(nat,intn,nbonds,nbangles,ntorsions,def,cart,bond_vals,angle_vals,tors_vals)
  intvals(1:nbonds)                                   =bond_vals(1:nbonds)
  intvals(nbonds+1:nbonds+nbangles)                   =angle_vals(1:nbangles)
  intvals(nbonds+nbangles+1:nbonds+nbangles+ntorsions)=tors_vals(1:ntorsions)
  delint(:)=intvals_target(:)-intvals(:)
  del=dlf_vector_norm(delint)
  if(del<=tol) exit
  call generate_B_C_ridders(nat,intn,cart,nbonds,nbangles, &
                             & ntorsions,bonds,bangles,torsions,B,C)
  A(:,:) = generate_inverse(nat,intn,B(:,:),mv(:))
  delcart(:)=dlf_matmul_simp(A(:,:),delint(:))
  cart(:)=cart(:)+delcart(:)
enddo

if (i>=maxiter .and. del>tol) then
  call error_print('int_to_cart: convergence not reached.')
endif

cart_final(:)=cart(:)
return

end subroutine int_to_cart


!***************************************************
!***************************************************

! Get reduced mass of a normal coordinate

function reduced_mass(nat,N,q,mv)
use dlf_vpt2_utility, only: checkeq
use dlf_linalg_interface_mod
implicit none
integer, intent(in) :: nat, N
real(rk), dimension(N), intent(in) :: q
real(rk), dimension(nat), intent(in) :: mv
real(rk) :: reduced_mass

real(rk), dimension(N) :: qnorm
integer :: i

call checkeq(3*nat,N,'reduced_mass')

qnorm=q/dlf_vector_norm(q)
reduced_mass=0._rk

do i=1,nat
  reduced_mass=reduced_mass + mv(i) * &
            & sum(qnorm(3*(i-1)+1:3*(i-1)+3)**2)
enddo

return
end function reduced_mass

!***************************************************
!***************************************************

! Read Z matrix topology definition from intcoord.inp

subroutine read_int_coord_def(intn,nbonds,nbangles,ntorsions,def)
use dlf_vpt2_utility, only: error_print
use dlf_allocate, only: allocate, deallocate
implicit none
integer, intent(in) :: intn
integer, intent(out) :: nbonds,nbangles,ntorsions
integer, dimension(intn,4), intent(out) :: def

integer, allocatable :: bonds(:,:), bangles(:,:), torsions(:,:)
integer :: i

def=0

! Read bond, angle and torsion definitions
open(1111,file='intcoord.inp')
read(1111,*) nbonds, nbangles, ntorsions

if (nbonds+nbangles+ntorsions .ne. intn) then
  call error_print('read_int_coord_def: Number of internal coordinates inconsistent')
endif

call allocate(bonds,nbonds,2)
call allocate(bangles,nbangles,3)
call allocate(torsions,ntorsions,4)

do i=1,nbonds
  read(1111,*) bonds(i,1), bonds(i,2)
enddo
do i=1,nbangles
  read(1111,*) bangles(i,1), bangles(i,2), bangles(i,3)
enddo
do i=1,ntorsions
  read(1111,*) torsions(i,1), torsions(i,2), torsions(i,3), torsions(i,4)
enddo
close(1111)

def(1:nbonds,1:2)=bonds(1:nbonds,1:2)
def(nbonds+1:nbonds+nbangles,1:3)=bangles(1:nbangles,1:3)
def(nbonds+nbangles+1:nbonds+nbangles+ntorsions,1:4)=torsions(1:ntorsions,1:4)

call deallocate(torsions)
call deallocate(bangles)
call deallocate(bonds)

return
end subroutine read_int_coord_def

!***************************************************
!***************************************************

! Convert gradient from internal to Cartesian coordinates

subroutine grad_i2c(nat,intn,grad_int,grad_cart,B)
use dlf_linalg_interface_mod
implicit none
integer, intent(in) :: nat,intn
real(rk), intent(in), dimension(intn,3*nat)       :: B
real(rk), intent(in), dimension(intn)             :: grad_int
real(rk), intent(out), dimension(3*nat)           :: grad_cart

grad_cart = dlf_matmul_simp(transpose(B),grad_int)

return
end subroutine grad_i2c

!*************************************
!*************************************

! Convert Hessian from internal to Cartesian coordinates

subroutine hess_i2c(nat,intn,hess_int,hess_cart,B,C,grad_int)
use dlf_linalg_interface_mod
implicit none
integer, intent(in) :: nat,intn
real(rk), intent(in), dimension(intn,intn)         :: hess_int
real(rk), intent(in), dimension(intn,3*nat)        :: B
real(rk), intent(in), dimension(3*nat,3*nat,intn)  :: C    ! note: index order switched compared to literature (first =>last)
real(rk), intent(in), dimension(intn)              :: grad_int
real(rk), intent(out), dimension(3*nat,3*nat)      :: hess_cart
real(rk), dimension(3*nat,3*nat)                   :: first,second
integer :: k

first  = dlf_matrix_ortho_trans(B,hess_int,0)
second = 0._rk
do k=1,intn
  second=second+grad_int(k)*C(:,:,k)
enddo

hess_cart = first + second

return
end subroutine hess_i2c

!***************************************************
!***************************************************

! Convert gradient from Cartesian to internal coordinates.

subroutine grad_c2i(nat,intn,grad_cart,grad_int,A)
use dlf_linalg_interface_mod
implicit none
integer, intent(in) :: nat,intn
real(rk), intent(in), dimension(3*nat,intn)        :: A
real(rk), intent(in), dimension(3*nat)             :: grad_cart
real(rk), intent(out), dimension(intn)             :: grad_int

grad_int = dlf_matmul_simp(transpose(A),grad_cart)

return
end subroutine grad_c2i

!*************************************
!*************************************

! Convert Hessian from Cartesian to internal coordinates.

subroutine hess_c2i(nat,intn,hess_cart,hess_int,A,C,grad_int)
use dlf_linalg_interface_mod
implicit none
integer, intent(in) :: nat,intn
real(rk), intent(in), dimension(3*nat,3*nat)       :: hess_cart
real(rk), intent(in), dimension(3*nat,intn)        :: A
real(rk), intent(in), dimension(3*nat,3*nat,intn)  :: C    ! note: index order switched compared to literature (first =>last)
real(rk), intent(in), dimension(intn)              :: grad_int
real(rk), intent(out), dimension(intn,intn)        :: hess_int
real(rk), dimension(intn,intn)                     :: first,second
integer :: k

first  = dlf_matrix_ortho_trans(A,hess_cart,0)
second = 0._rk
do k=1,intn
  second=second+grad_int(k)*dlf_matrix_ortho_trans(A,C(:,:,k),0)
enddo

hess_int = first - second

return
end subroutine hess_c2i

!*************************************
!*************************************

! Generate Wilson's B matrix and the C array
! numerically using Ridders' finite-difference scheme
! (this is a pretty cool numerical derivative method
! that basically extrapolates to a zero deltax step).
! B and C contain the first and second derivatives of the 
! internal coordinates with respect to the Cartesians

subroutine generate_B_C_ridders(nat,intn,coords,nbonds,nbangles,ntorsions,bonds,bangles,torsions,B,C)
implicit none
integer, intent(in) :: nat,intn,nbonds,nbangles,ntorsions
real(rk), intent(in), dimension(3*nat)  :: coords
integer, intent(in), dimension(nbonds,2)    :: bonds
integer, intent(in), dimension(nbangles,3)  :: bangles
integer, intent(in), dimension(ntorsions,4) :: torsions
real(rk), intent(out), dimension(intn,3*nat)        :: B
real(rk), intent(out), dimension(3*nat,3*nat,intn)  :: C
real(rk),parameter :: h0=0.1_rk
integer :: i,j,k,indx1,indx3,i2,j2,ip,ip2
integer :: at1,at2,at3,at4
real(rk), dimension(3) :: coo1,coo2,coo3,coo4

! First derivatives and diagonal second derivatives

B=0._rk
C=0._rk

do i=1,nat
  do j=1,3
    ip=3*(i-1)+j
    !ip=nat*(j-1)+i
    icloop: do k=1,intn
      if(k.le.nbonds) then
        at1=bonds(k,1)
        at2=bonds(k,2)
        !coo1(1)=coords(at1)
        !coo1(2)=coords(nat+at1)
        !coo1(3)=coords(2*nat+at1)
        !coo2(1)=coords(at2)
        !coo2(2)=coords(nat+at2)
        !coo2(3)=coords(2*nat+at2)
        coo1(1:3)=coords(3*(at1-1)+1:3*(at1-1)+3)
        coo2(1:3)=coords(3*(at2-1)+1:3*(at2-1)+3)
        if     (at1.eq.i) then
          indx1=1
        elseif (at2.eq.i) then
          indx1=2
        else
          cycle icloop
        endif
        call ridders_diff(B(k,ip),1,1,indx1,j,0,0,h0,coo1,coo2)
        call ridders_diff(C(ip,ip,k),1,2,indx1,j,0,0,h0,coo1,coo2)
      elseif (k.gt.nbonds.and.k.le.(nbonds+nbangles)) then
        at1=bangles(k-nbonds,1)
        at2=bangles(k-nbonds,2)
        at3=bangles(k-nbonds,3)
        !coo1(1)=coords(at1)
        !coo1(2)=coords(nat+at1)
        !coo1(3)=coords(2*nat+at1)
        !coo2(1)=coords(at2)
        !coo2(2)=coords(nat+at2)
        !coo2(3)=coords(2*nat+at2)
        !coo3(1)=coords(at3)
        !coo3(2)=coords(nat+at3)
        !coo3(3)=coords(2*nat+at3)
        coo1(1:3)=coords(3*(at1-1)+1:3*(at1-1)+3)
        coo2(1:3)=coords(3*(at2-1)+1:3*(at2-1)+3)
        coo3(1:3)=coords(3*(at3-1)+1:3*(at3-1)+3)
        if     (at1.eq.i) then
          indx1=1
        elseif (at2.eq.i) then
          indx1=2
        elseif (at3.eq.i) then
          indx1=3
        else
          cycle icloop
        endif
        call ridders_diff(B(k,ip),2,1,indx1,j,0,0,h0,coo1,coo2,coo3)
        call ridders_diff(C(ip,ip,k),2,2,indx1,j,0,0,h0,coo1,coo2,coo3)
      elseif (k.gt.(nbonds+nbangles)) then
        at1=torsions(k-nbonds-nbangles,1)
        at2=torsions(k-nbonds-nbangles,2)
        at3=torsions(k-nbonds-nbangles,3)
        at4=torsions(k-nbonds-nbangles,4)
        !coo1(1)=coords(at1)
        !coo1(2)=coords(nat+at1)
        !coo1(3)=coords(2*nat+at1)
        !coo2(1)=coords(at2)
        !coo2(2)=coords(nat+at2)
        !coo2(3)=coords(2*nat+at2)
        !coo3(1)=coords(at3)
        !coo3(2)=coords(nat+at3)
        !coo3(3)=coords(2*nat+at3)
        !coo4(1)=coords(at4)
        !coo4(2)=coords(nat+at4)
        !coo4(3)=coords(2*nat+at4)
        coo1(1:3)=coords(3*(at1-1)+1:3*(at1-1)+3)
        coo2(1:3)=coords(3*(at2-1)+1:3*(at2-1)+3)
        coo3(1:3)=coords(3*(at3-1)+1:3*(at3-1)+3)
        coo4(1:3)=coords(3*(at4-1)+1:3*(at4-1)+3)
        if     (at1.eq.i) then
          indx1=1
        elseif (at2.eq.i) then
          indx1=2
        elseif (at3.eq.i) then
          indx1=3
        elseif (at4.eq.i) then
          indx1=4
        else
          cycle icloop
        endif
        call ridders_diff(B(k,ip),3,1,indx1,j,0,0,h0,coo1,coo2,coo3,coo4)
        call ridders_diff(C(ip,ip,k),3,2,indx1,j,0,0,h0,coo1,coo2,coo3,coo4)
      endif
    end do icloop
  enddo
enddo

! Off-diagonal second derivatives

do i=1,nat
  do j=1,3
    do i2=1,nat
      innerjloop: do j2=1,3
        ip =3*(i-1)+j
        ip2=3*(i2-1)+j2
        !ip=nat*(j-1)+i
        !ip2=nat*(j2-1)+i2
        if (ip2.ge.ip) cycle innerjloop
        icloop2: do k=1,intn
          if(k.le.nbonds) then
            at1=bonds(k,1)
            at2=bonds(k,2)
            !coo1(1)=coords(at1)
            !coo1(2)=coords(nat+at1)
            !coo1(3)=coords(2*nat+at1)
            !coo2(1)=coords(at2)
            !coo2(2)=coords(nat+at2)
            !coo2(3)=coords(2*nat+at2)
            coo1(1:3)=coords(3*(at1-1)+1:3*(at1-1)+3)
            coo2(1:3)=coords(3*(at2-1)+1:3*(at2-1)+3)
            if     (at1.eq.i) then
              indx1=1
            elseif (at2.eq.i) then
              indx1=2
            else
              cycle icloop2
            endif
            if     (at1.eq.i2) then
              indx3=1
            elseif (at2.eq.i2) then
              indx3=2
            else
              cycle icloop2
            endif
            call ridders_diff(C(ip,ip2,k),1,3,indx1,j,indx3,j2,h0,coo1,coo2)
          elseif (k.gt.nbonds.and.k.le.(nbonds+nbangles)) then
            at1=bangles(k-nbonds,1)
            at2=bangles(k-nbonds,2)
            at3=bangles(k-nbonds,3)
            !coo1(1)=coords(at1)
            !coo1(2)=coords(nat+at1)
            !coo1(3)=coords(2*nat+at1)
            !coo2(1)=coords(at2)
            !coo2(2)=coords(nat+at2)
            !coo2(3)=coords(2*nat+at2)
            !coo3(1)=coords(at3)
            !coo3(2)=coords(nat+at3)
            !coo3(3)=coords(2*nat+at3)
            coo1(1:3)=coords(3*(at1-1)+1:3*(at1-1)+3)
            coo2(1:3)=coords(3*(at2-1)+1:3*(at2-1)+3)
            coo3(1:3)=coords(3*(at3-1)+1:3*(at3-1)+3)
            if     (at1.eq.i) then
              indx1=1
            elseif (at2.eq.i) then
              indx1=2
            elseif (at3.eq.i) then
              indx1=3
            else
              cycle icloop2
            endif
            if     (at1.eq.i2) then
              indx3=1
            elseif (at2.eq.i2) then
              indx3=2
            elseif (at3.eq.i2) then
              indx3=3
            else
              cycle icloop2
            endif
            call ridders_diff(C(ip,ip2,k),2,3,indx1,j,indx3,j2,h0,coo1,coo2,coo3)
          elseif (k.gt.(nbonds+nbangles)) then
            at1=torsions(k-nbonds-nbangles,1)
            at2=torsions(k-nbonds-nbangles,2)
            at3=torsions(k-nbonds-nbangles,3)
            at4=torsions(k-nbonds-nbangles,4)
            !coo1(1)=coords(at1)
            !coo1(2)=coords(nat+at1)
            !coo1(3)=coords(2*nat+at1)
            !coo2(1)=coords(at2)
            !coo2(2)=coords(nat+at2)
            !coo2(3)=coords(2*nat+at2)
            !coo3(1)=coords(at3)
            !coo3(2)=coords(nat+at3)
            !coo3(3)=coords(2*nat+at3)
            !coo4(1)=coords(at4)
            !coo4(2)=coords(nat+at4)
            !coo4(3)=coords(2*nat+at4)
            coo1(1:3)=coords(3*(at1-1)+1:3*(at1-1)+3)
            coo2(1:3)=coords(3*(at2-1)+1:3*(at2-1)+3)
            coo3(1:3)=coords(3*(at3-1)+1:3*(at3-1)+3)
            coo4(1:3)=coords(3*(at4-1)+1:3*(at4-1)+3)
            if     (at1.eq.i) then
              indx1=1
            elseif (at2.eq.i) then
              indx1=2
            elseif (at3.eq.i) then
              indx1=3
            elseif (at4.eq.i) then
              indx1=4
            else
              cycle icloop2
            endif
            if     (at1.eq.i2) then
              indx3=1
            elseif (at2.eq.i2) then
              indx3=2
            elseif (at3.eq.i2) then
              indx3=3
            elseif (at4.eq.i2) then
              indx3=4
            else
              cycle icloop2
            endif
            call ridders_diff(C(ip,ip2,k),3,3,indx1,j,indx3,j2,h0,coo1,coo2,coo3,coo4)
          endif
        end do icloop2
      end do innerjloop
    enddo
  enddo
enddo

do i=1,3*nat
  do j=i+1,3*nat
    C(i,j,:)=C(j,i,:)
  enddo
enddo

return
end subroutine generate_B_C_ridders

!*************************************
!*************************************

! Analytic calculation of B and C
! This is only implemented for bond lengths and valence angles,
! not for torsion angles. 
! torsional derivatives are therefore simply the Ridders values, 
! the rest is overwritten. 

subroutine generate_B_C_analytical(nat,intn,coords,nbonds,nbangles,ntorsions,bonds,bangles,torsions,B,C)
use dlf_vpt2_utility, only: error_print
use dlf_allocate, only: allocate, deallocate
use dlf_linalg_interface_mod
implicit none
integer, intent(in) :: nat,intn,nbonds,nbangles,ntorsions
real(rk), intent(in), dimension(3*nat)  :: coords
integer, intent(in), dimension(nbonds,2)    :: bonds
integer, intent(in), dimension(nbangles,3)  :: bangles
integer, intent(in), dimension(ntorsions,4) :: torsions
real(rk), intent(out), dimension(intn,3*nat)        :: B
real(rk), intent(out), dimension(3*nat,3*nat,intn)  :: C

real(rk), parameter :: zertol=1.e-14_rk
real(rk), dimension(:,:,:), allocatable, save :: top
real(rk), dimension(:,:,:), allocatable, save :: topTtop
integer, dimension(:,:), allocatable, save :: Rdef
integer, dimension(:,:), allocatable, save :: bondmap, anglemap, torsmap
logical, save :: top_initialized=.false.
integer, save :: npairs_save=0
integer :: npairs,i,j,k,m,n,kp
real(rk), dimension(3) :: dx1,dx2
real(rk), dimension(:,:), allocatable :: t1, t2, t3
real(rk), dimension(:,:), allocatable :: ttt1, ttt2, ttt3
real(rk) :: dist1,dist2,dp1,theta,sintheta,costheta
real(rk), dimension(3*nat)  :: x_12_12, x_11_22, x_11_11, x_22_22, grad

npairs=(nat*(nat-1))/2

if (top_initialized .and. npairs/=npairs_save) top_initialized=.false.

! Generate topology information
if (.not. top_initialized) then
  if (allocated(top))      call deallocate(top)
  if (allocated(topTtop))  call deallocate(topTtop)
  if (allocated(Rdef))     call deallocate(Rdef)
  if (allocated(bondmap))  call deallocate(bondmap)
  if (allocated(anglemap)) call deallocate(anglemap)
  if (allocated(torsmap))  call deallocate(torsmap)
  call allocate(top,3,3*nat,npairs)
  call allocate(topTtop,3*nat,3*nat,npairs)
  call allocate(Rdef,npairs,2)
  call allocate(bondmap,nbonds,2)
  call allocate(anglemap,nbangles,4)
  call allocate(torsmap,ntorsions,6)
  ! Generate Rdef, defining all interatomic distances
  ! Rij with j>i
  k=0
  do i=1,nat
    do j=i+1,nat
      k=k+1
      Rdef(k,1)=i
      Rdef(k,2)=j
    enddo
  enddo
  ! Generate the topology matrices top which transform
  ! the (3*Nat)-dim. Cartesian coordinate vector to the 3D 
  ! displacement vector for the kth atom-atom pair
  top=0._rk
  do k=1,npairs
    i=Rdef(k,1)
    j=Rdef(k,2)
    top(1,3*(i-1)+1,k)=1._rk
    top(2,3*(i-1)+2,k)=1._rk
    top(3,3*(i-1)+3,k)=1._rk
    top(1,3*(j-1)+1,k)=-1._rk
    top(2,3*(j-1)+2,k)=-1._rk
    top(3,3*(j-1)+3,k)=-1._rk
    topTtop(:,:,k)=dlf_matmul_simp(transpose(top(:,:,k)),top(:,:,k))
  enddo
! Generate mappings of bonds, angles, dihedrals to Rdef
  bnd: do k=1,nbonds
    i=bonds(k,1)
    j=bonds(k,2)
    !write(stdout,'(A,3I4)') 'k, i, j: ', k, i, j
    rdefb: do m=1,npairs
      !write(stdout,'(A,3I4)') 'Rdef, m, 1, 2: ', m, Rdef(m,1), Rdef(m,2)
      if (i==Rdef(m,1) .and. j==Rdef(m,2)) then
        bondmap(k,1)=m
        bondmap(k,2)=1
        cycle bnd
      elseif (i==Rdef(m,2) .and. j==Rdef(m,1)) then
        bondmap(k,1)=m
        bondmap(k,2)=-1
        cycle bnd
      endif
    enddo rdefb
    call error_print('generate_B_C_analytical: bond mapping failure')
  enddo bnd
  angl: do k=1,nbangles
    armsa: do n=1,2
      i=bangles(k,1+(n-1))
      j=bangles(k,2+(n-1))
      rdefa: do m=1,npairs
        if (i==Rdef(m,1) .and. j==Rdef(m,2)) then
          anglemap(k,n)=m
          anglemap(k,2+n)=1
          cycle armsa
        elseif (i==Rdef(m,2) .and. j==Rdef(m,1)) then
          anglemap(k,n)=m
          anglemap(k,2+n)=-1
          cycle armsa
        endif
      enddo rdefa
      call error_print('generate_B_C_analytical: angle mapping failure')
    enddo armsa
  enddo angl
  tors: do k=1,ntorsions
    armst: do n=1,3
      i=torsions(k,1+(n-1))
      j=torsions(k,2+(n-1))
      rdeft: do m=1,npairs
        if (i==Rdef(m,1) .and. j==Rdef(m,2)) then
          torsmap(k,n)=m
          torsmap(k,3+n)=1
          cycle armst
        elseif (i==Rdef(m,2) .and. j==Rdef(m,1)) then
          torsmap(k,n)=m
          torsmap(k,3+n)=-1
          cycle armst
        endif
      enddo rdeft
      call error_print('generate_B_C_analytical: torsion mapping failure')
    enddo armst
  enddo tors
  npairs_save=npairs
  top_initialized=.true.
endif

! Initialize B,C
! Until all derivatives are actually implemented, fill B and C with the numerical (Ridders) values
call generate_B_C_ridders(nat,intn,coords,nbonds,nbangles,ntorsions,bonds,bangles,torsions,B,C)

call allocate(t1,3,3*nat)
call allocate(t2,3,3*nat)
call allocate(t3,3,3*nat)
call allocate(ttt1,3*nat,3*nat)
call allocate(ttt2,3*nat,3*nat)
call allocate(ttt3,3*nat,3*nat)

! First and second derivatives (B and C), bonds
do k=1,nbonds
  t1(1:3,1:3*nat)=bondmap(k,2)*top(1:3,1:3*nat,bondmap(k,1))
  ttt1(1:3*nat,1:3*nat)=topTtop(1:3*nat,1:3*nat,bondmap(k,1))
  dx1=dlf_matmul_simp(t1,coords)
  !call vector_output(dx1,stdout,'F20.12','deltax1')
  dist1=dlf_vector_norm(dx1)
  grad(1:3*nat)=dlf_matmul_simp(ttt1,coords)/dist1
  B(k,1:3*nat)=grad(1:3*nat)
  C(:,:,k)=ttt1(:,:) - dlf_outer_product(grad(1:3*nat),grad(1:3*nat))
  C(:,:,k)=C(:,:,k)/dist1
enddo

! First derivatives and second derivatives (B and C), angles
do k=1,nbangles
  kp=nbonds+k
  t1(1:3,1:3*nat)=-anglemap(k,3)*top(1:3,1:3*nat,anglemap(k,1))
  ttt1(1:3*nat,1:3*nat)=topTtop(1:3*nat,1:3*nat,anglemap(k,1))
  t2(1:3,1:3*nat)= anglemap(k,4)*top(1:3,1:3*nat,anglemap(k,2))
  ttt2(1:3*nat,1:3*nat)=topTtop(1:3*nat,1:3*nat,anglemap(k,2))
  ttt3(1:3*nat,1:3*nat)=dlf_matmul_simp(transpose(t1),t2)
  dx1=dlf_matmul_simp(t1,coords)
  dx2=dlf_matmul_simp(t2,coords)
  dist1=dlf_vector_norm(dx1)
  dist2=dlf_vector_norm(dx2)
  dp1=dlf_dot_product(dx1,dx2)
  !write(stdout,*) 'dist1, dist2, dp: ', dist1, dist2, dp1
  dist1=sqrt(dlf_bilinear_form(coords,ttt1,coords))
  dist2=sqrt(dlf_bilinear_form(coords,ttt2,coords))
  dp1=dlf_bilinear_form(coords,ttt3,coords)
  !write(stdout,*) 'dist1, dist2, dp: ', dist1, dist2, dp1
  theta=acos(dp1/(dist1*dist2))
  sintheta=sin(theta)
  costheta=cos(theta)
  if (abs(sintheta).lt.zertol) then
    grad(1:3*nat)=0._rk
  else
    grad(1:3*nat)=(dlf_matmul_simp(ttt3,coords)+dlf_matmul_simp(transpose(ttt3),coords))/(dist1*dist2)
    grad(1:3*nat)=grad(1:3*nat) &
         & - dp1* ( dist2/dist1*dlf_matmul_simp(ttt1,coords) + dist1/dist2*dlf_matmul_simp(ttt2,coords) ) / &
         &                                            (dist1**2 * dist2**2)
    grad(1:3*nat)=-grad(1:3*nat)/sintheta
  endif
  B(kp,1:3*nat)=grad(1:3*nat)
  ! Second derivatives
  x_12_12=dlf_matmul_simp((ttt3(:,:)+transpose(ttt3(:,:)))/(dist1*dist2),coords)
  x_11_22=dlf_matmul_simp(ttt1(:,:)/dist1**2+ttt2(:,:)/dist2**2,coords)
  x_11_11=dlf_matmul_simp(ttt1(:,:)/dist1**2,coords)
  x_22_22=dlf_matmul_simp(ttt2(:,:)/dist2**2,coords)
  C(:,:,kp)=(ttt3(:,:)+transpose(ttt3(:,:)))/(dist1*dist2)
  C(:,:,kp)=C(:,:,kp)-dlf_outer_product(x_12_12,x_11_22)
  C(:,:,kp)=C(:,:,kp)+sintheta*dlf_outer_product(x_11_22,grad(1:3*nat))
  C(:,:,kp)=C(:,:,kp)-costheta*(ttt1(:,:)/dist1**2+ttt2(:,:)/dist2**2)
  C(:,:,kp)=C(:,:,kp)+2*costheta*(dlf_outer_product(x_11_11,x_11_11)+dlf_outer_product(x_22_22,x_22_22))
  C(:,:,kp)=C(:,:,kp)+costheta*dlf_outer_product(grad(1:3*nat),grad(1:3*nat))
  if (abs(sintheta).lt.zertol) then
    C(:,:,kp)=0._rk
  else
    C(:,:,kp)=-C(:,:,kp)/sintheta
  endif
enddo

call deallocate(ttt3)
call deallocate(ttt2)
call deallocate(ttt1)
call deallocate(t3)
call deallocate(t2)
call deallocate(t1)

return
end subroutine generate_B_C_analytical

!*************************************
!*************************************

! Calculate numerical derivative of an individual Z matrix coordinate (generic wrapping
! function for any type of coordinate, bond length, bending angle, or torsional angle)
! Uses Ridders' method (a.k.a. Richardson extrapolation applied to finite-difference 
! derivatives)

subroutine ridders_diff(deriv,coord_type,mode,indx1,indx2,indx3,indx4,h0,co1in,co2in,co3in,co4in)
implicit none
integer, intent(in) :: coord_type, mode,indx1,indx2,indx3,indx4
real(rk), dimension(3), intent(in) :: co1in,co2in
real(rk), dimension(3), intent(in), optional :: co3in,co4in
real(rk), intent(in) :: h0
real(rk), intent(out):: deriv
real(rk), dimension(3) :: co1,co2,co3,co4,co1m,co2m,co3m,co4m
integer, parameter :: mxlvl=12
real(rk), parameter :: fact=1.5d0, largenumber=1.d40
real(rk), dimension(mxlvl,mxlvl) :: tab
real(rk) :: h,fact2,errcurr,error,valu1,valu2,valu3,valu4
integer :: n,m
! coord_type: 1, 2, 3: bond distance, bending angle, torsion angle
! mode: 1: first derivative, 2: second derivative, diagonal, 3: second derivative, offdiagonal

fact2=fact*fact

if     (coord_type.eq.1) then
  co1=co1in
  co2=co2in
  co3=0._rk
  co4=0._rk
elseif (coord_type.eq.2) then
  co1=co1in
  co2=co2in
  if(.not.present(co3in)) stop 1
  co3=co3in
  co4=0._rk
elseif (coord_type.eq.3) then
  co1=co1in
  co2=co2in
  if(.not.present(co3in)) stop 1
  co3=co3in
  if(.not.present(co4in)) stop 1
  co4=co4in
else 
  stop 1
endif

ol: do n=1,mxlvl
  h=h0/fact**(n-1)
  co1m=co1
  co2m=co2
  co3m=co3
  co4m=co4
  if     (mode.eq.1) then
    if     (indx1.eq.1) then
      co1m(indx2)=co1(indx2)+h
    elseif (indx1.eq.2) then
      co2m(indx2)=co2(indx2)+h
    elseif (indx1.eq.3) then
      co3m(indx2)=co3(indx2)+h
    elseif (indx1.eq.4) then
      co4m(indx2)=co4(indx2)+h
    endif
    valu2=intcoord_generic(coord_type,co1m,co2m,co3m,co4m)
    if     (indx1.eq.1) then
      co1m(indx2)=co1(indx2)-h
    elseif (indx1.eq.2) then
      co2m(indx2)=co2(indx2)-h
    elseif (indx1.eq.3) then
      co3m(indx2)=co3(indx2)-h
    elseif (indx1.eq.4) then
      co4m(indx2)=co4(indx2)-h
    endif
    valu1=intcoord_generic(coord_type,co1m,co2m,co3m,co4m,valu2)
    tab(n,1)=(valu2-valu1)/(2._rk*h)
  elseif (mode.eq.2) then
    valu2=intcoord_generic(coord_type,co1m,co2m,co3m,co4m)
    if     (indx1.eq.1) then
      co1m(indx2)=co1(indx2)+h
    elseif (indx1.eq.2) then
      co2m(indx2)=co2(indx2)+h
    elseif (indx1.eq.3) then
      co3m(indx2)=co3(indx2)+h
    elseif (indx1.eq.4) then
      co4m(indx2)=co4(indx2)+h
    endif
    valu3=intcoord_generic(coord_type,co1m,co2m,co3m,co4m,valu2)
    if     (indx1.eq.1) then
      co1m(indx2)=co1(indx2)-h
    elseif (indx1.eq.2) then
      co2m(indx2)=co2(indx2)-h
    elseif (indx1.eq.3) then
      co3m(indx2)=co3(indx2)-h
    elseif (indx1.eq.4) then
      co4m(indx2)=co4(indx2)-h
    endif
    valu1=intcoord_generic(coord_type,co1m,co2m,co3m,co4m,valu2)
    tab(n,1)=(valu3+valu1-2._rk*valu2)/(h*h)
  elseif (mode.eq.3) then
    if     (indx1.eq.1) then
      co1m(indx2)=co1(indx2)+h
    elseif (indx1.eq.2) then
      co2m(indx2)=co2(indx2)+h
    elseif (indx1.eq.3) then
      co3m(indx2)=co3(indx2)+h
    elseif (indx1.eq.4) then
      co4m(indx2)=co4(indx2)+h
    endif
    if     (indx3.eq.1) then
      co1m(indx4)=co1(indx4)+h
    elseif (indx3.eq.2) then
      co2m(indx4)=co2(indx4)+h
    elseif (indx3.eq.3) then
      co3m(indx4)=co3(indx4)+h
    elseif (indx3.eq.4) then
      co4m(indx4)=co4(indx4)+h
    endif
    valu1=intcoord_generic(coord_type,co1m,co2m,co3m,co4m)
    if     (indx1.eq.1) then
      co1m(indx2)=co1(indx2)-h
    elseif (indx1.eq.2) then
      co2m(indx2)=co2(indx2)-h
    elseif (indx1.eq.3) then
      co3m(indx2)=co3(indx2)-h
    elseif (indx1.eq.4) then
      co4m(indx2)=co4(indx2)-h
    endif
    valu2=intcoord_generic(coord_type,co1m,co2m,co3m,co4m,valu1)
    if     (indx3.eq.1) then
      co1m(indx4)=co1(indx4)-h
    elseif (indx3.eq.2) then
      co2m(indx4)=co2(indx4)-h
    elseif (indx3.eq.3) then
      co3m(indx4)=co3(indx4)-h
    elseif (indx3.eq.4) then
      co4m(indx4)=co4(indx4)-h
    endif
    valu4=intcoord_generic(coord_type,co1m,co2m,co3m,co4m,valu1)
    if     (indx1.eq.1) then
      co1m(indx2)=co1(indx2)+h
    elseif (indx1.eq.2) then
      co2m(indx2)=co2(indx2)+h
    elseif (indx1.eq.3) then
      co3m(indx2)=co3(indx2)+h
    elseif (indx1.eq.4) then
      co4m(indx2)=co4(indx2)+h
    endif
    valu3=intcoord_generic(coord_type,co1m,co2m,co3m,co4m,valu1)
    tab(n,1)=(valu1-valu2-valu3+valu4)/(4._rk*h*h)
  else
    stop 1
  endif
  if (n.eq.1) then 
    deriv=tab(1,1)
    error=largenumber
    cycle ol
  endif
  do m=2,n
    tab(n,m)=(tab(n,m-1)*fact2**(m-1)-tab(n-1,m-1))/(fact2**(m-1)-1._rk)
    errcurr=max(abs(tab(n,m)-tab(n,m-1)),abs(tab(n,m)-tab(n-1,m-1)))
    if (errcurr.lt.error) then
      deriv=tab(n,m)
      error=errcurr
    endif
  enddo
  if(dabs(tab(n,n)-tab(n-1,n-1)).ge.2*error) exit ol
end do ol

return

end subroutine ridders_diff

!*************************************
!*************************************

! Generic wrapping function for the different types 
! of internal coordinates: bond distances, bending 
! angles, torsional angles.

function intcoord_generic(coordtype,co1,co2,co3,co4,valu)
implicit none
real(rk)    :: intcoord_generic
integer, intent(in) :: coordtype
real(rk), dimension(3), intent(in) :: co1,co2,co3,co4
real(rk), intent(in), optional :: valu

if     (coordtype.eq.1) then
  intcoord_generic=distance(co1,co2)
elseif (coordtype.eq.2) then
  intcoord_generic=bendangle(co1,co2,co3)
elseif (coordtype.eq.3) then
  if (present(valu)) then
    intcoord_generic=torsangle(co1,co2,co3,co4,.true.,valu)
  else
    intcoord_generic=torsangle(co1,co2,co3,co4)
  endif
else
  stop 1
endif
return
end function intcoord_generic

!*************************************
!*************************************

! Get bond distance from coordinates of the two 
! participating atoms

function distance(co1,co2)
use dlf_linalg_interface_mod
implicit none
real(rk) :: distance
real(rk), intent(in),dimension(3) :: co1, co2

distance=dlf_vector_norm(co1-co2)

return
end function distance

!*************************************
!*************************************

! Get bending angle from coordinates of the three 
! participating atoms

function bendangle(co1,co2,co3,modesel)    ! co2 are coordinates of the central atom
use dlf_linalg_interface_mod
implicit none
real(rk) :: bendangle
real(rk), intent(in),dimension(3) :: co1, co2, co3
integer, intent(in), optional :: modesel
integer :: mode
real(rk) :: r12,r23

if (present(modesel)) then 
  mode=modesel
else
  mode=0
endif

r12=distance(co1,co2)
r23=distance(co2,co3)
bendangle=dlf_dot_product((co2-co1),(co2-co3))
bendangle=bendangle/(r12*r23)
if    (mode.eq.0) then
  bendangle=acos(bendangle)    ! return bending angle directly
elseif(mode.eq.1) then
  continue                      ! return cosine of angle
elseif(mode.eq.2) then
  bendangle=sqrt(1._rk-bendangle**2)  ! return sine of angle
else
  stop 1
endif
return
end function bendangle

!*************************************
!*************************************

! Get torsional angle from coordinates of the four 
! participating atoms

function torsangle(co1,co2,co3,co4,derivinp,oldvalu)    ! co2, co3 define the axis
use dlf_linalg_interface_mod
implicit none
real(rk) :: torsangle
real(rk), intent(in),dimension(3) :: co1, co2, co3, co4
logical, intent(in),optional :: derivinp
real(rk), intent(in),optional :: oldvalu
real(rk), dimension(3)::x1,x2,x3,x1cx2,x2cx3,x1cx2cx2cx3
real(rk) :: arg1,arg2,pi
logical          :: deriv
pi=acos(-1._rk)

if (present(derivinp)) then
  deriv=derivinp
  if (.not.(present(oldvalu))) stop 1
else
  deriv=.false.
endif

x1=co2-co1
x2=co3-co2
x3=co4-co3

x1cx2=dlf_cross_product(x1,x2)
x2cx3=dlf_cross_product(x2,x3)
x1cx2cx2cx3=dlf_cross_product(x1cx2,x2cx3)
arg1=dlf_dot_product(x1cx2cx2cx3,x2)
arg1=arg1/dlf_vector_norm(x2)
arg2=dlf_dot_product(x1cx2,x2cx3)

torsangle=atan2(arg1,arg2)

if (deriv) then
  if     (oldvalu-torsangle .gt. pi) then
    torsangle=torsangle+2._rk*pi
  elseif (torsangle-oldvalu .gt. pi) then
    torsangle=torsangle-2._rk*pi
  endif
endif

return
end function torsangle

!*************************************
!*************************************

! Generate the pseudo-inverse of B, A = u B^T (B u B^T)^-1
! (generalized inverse of the Wilson B matrix)

function generate_inverse(nat,intn,B,mv)
use dlf_linalg_interface_mod
use dlf_allocate, only: allocate, deallocate
implicit none
integer, intent(in) :: nat,intn
real(rk), intent(in), dimension(intn,3*nat) :: B
real(rk), intent(in), dimension(nat) :: mv
real(rk), dimension(3*nat,intn)  :: generate_inverse
real(rk), dimension(3*nat,3*nat) :: umat
real(rk), dimension(3*nat,intn)  :: uBT
real(rk), dimension(intn,intn)   :: BuBT,BuBT_inv
integer, dimension(intn)      :: pivot
real(rk), allocatable :: work(:)
integer :: lpk_info,lwork

umat=gen_umat(nat,mv)
uBT= dlf_matmul_simp(umat,transpose(B))
BuBT=dlf_matmul_simp(B,uBT)
BuBT_inv=BuBT

call dgetrf (intn,intn,BuBT_inv,intn,pivot,lpk_info)
call allocate(work,1)
call dgetri (intn,BuBT_inv,intn,pivot,work,-1,lpk_info)
lwork=nint(work(1))
call deallocate(work)
call allocate(work,lwork)
call dgetri (intn,BuBT_inv,intn,pivot,work,lwork,lpk_info)
call deallocate(work)

generate_inverse = dlf_matmul_simp(uBT,BuBT_inv)

return
end function generate_inverse

!*************************************
!*************************************

! Generate kernel matrix "u", 
! diagonal matrix with u_ii=1/m(at(i))

function gen_umat(nat,mv)
implicit none
integer, intent(in) :: nat
real(rk), intent(in), dimension(nat) :: mv
real(rk), dimension(3*nat,3*nat)     :: gen_umat
real(rk) :: valu
integer :: i,j,j1,j2

gen_umat=0._rk

do i=1,nat
  valu=1.d0/mv(i)
  j1=3*(i-1)+1
  j2=3*(i-1)+3
  do j=j1,j2
    gen_umat(j,j)=valu
  enddo
enddo

return
end function gen_umat

!*************************************
!*************************************

end module dlf_vpt2_intcoord





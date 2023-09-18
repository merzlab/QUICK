! Module for projecting rotational/translational degrees of freedom
! from Hessians, plus a few related auxiliar routines (coordinate 
! shifting, rotation, rotational constant calculation, ...)

module dlf_vpt2_project
use dlf_parameter_module
!use dlf_global, only: glob, stdout
implicit none

contains

!***************************************************
!***************************************************

! Project rotations and translations from Hessian, 
! where all quantities are given in mass-weighted 
! atomic units. An additional direction vector 
! "dir" can be projected besides the rot./trans. 
! directions, e.g. the gradient if we are not at a
! stationary point of the PES. Provide zero vector 
! if this is not necessary/desired.
!
! Expect Hessian and coordinates in order xyzxyz...

subroutine proj_trans_rot_dir_mw(nat,N,mv,Hin,x,dir,Hout,linear_in)
use dlf_vpt2_utility, only: error_print, loewdin_ortho
use dlf_allocate, only: allocate, deallocate
use dlf_linalg_interface_mod
use dlf_constants
implicit none
integer, intent(in) :: nat,N
real(rk), dimension(:,:), intent(in) :: Hin
real(rk), dimension(:), intent(in) :: dir,x,mv
real(rk), dimension(:,:), intent(out) :: Hout
logical, intent(in), optional :: linear_in

integer :: i
real(rk), dimension(N,N) :: projector
real(rk), dimension(:,:), allocatable :: outvec
real(rk), dimension(N) :: transx,transy,transz,dirint
real(rk), dimension(N) :: rota,rotb,rotc
real(rk) :: amu2au
logical  :: linear

linear=.false.
if (present(linear_in)) then
  linear=linear_in
endif

call dlf_constants_get('AMU',amu2au)

if (size(dir).ne.N) call error_print('Size of dir doesnt match N in proj_trans_rot_dir_mw.')
if (size(x).ne.N) call error_print('Size of x doesnt match N in proj_trans_rot_dir_mw.')
if (size(mv).ne.nat) call error_print('Size of mv doesnt match nat in proj_trans_rot_dir_mw.')
if (size(Hin,1).ne.N  .or. size(Hin,2).ne.N)  call error_print('Size of Hin doesnt match N in proj_trans_rot_dir_mw.')
if (size(Hout,1).ne.N .or. size(Hout,2).ne.N) call error_print('Size of Hout doesnt match N in proj_trans_rot_dir_mw.')

transx=0._rk
transy=0._rk
transz=0._rk

do i=1,nat
  transx(3*(i-1)+1)=sqrt(mv(i)/amu2au)
  transy(3*(i-1)+2)=sqrt(mv(i)/amu2au)
  transz(3*(i-1)+3)=sqrt(mv(i)/amu2au)
enddo

transx=transx/dlf_vector_norm(transx)
transy=transy/dlf_vector_norm(transy)
transz=transz/dlf_vector_norm(transz)

call rot_unit_vectors_altern(nat,N,x,mv,rota,rotb,rotc,linear)

do i=1,nat
  rota(3*(i-1)+1)=rota(3*(i-1)+1)*sqrt(mv(i)/amu2au)
  rota(3*(i-1)+2)=rota(3*(i-1)+2)*sqrt(mv(i)/amu2au)
  rota(3*(i-1)+3)=rota(3*(i-1)+3)*sqrt(mv(i)/amu2au)
  rotb(3*(i-1)+1)=rotb(3*(i-1)+1)*sqrt(mv(i)/amu2au)
  rotb(3*(i-1)+2)=rotb(3*(i-1)+2)*sqrt(mv(i)/amu2au)
  rotb(3*(i-1)+3)=rotb(3*(i-1)+3)*sqrt(mv(i)/amu2au)
  rotc(3*(i-1)+1)=rotc(3*(i-1)+1)*sqrt(mv(i)/amu2au)
  rotc(3*(i-1)+2)=rotc(3*(i-1)+2)*sqrt(mv(i)/amu2au)
  rotc(3*(i-1)+3)=rotc(3*(i-1)+3)*sqrt(mv(i)/amu2au)
enddo

if (linear) then 
  rota=0._rk
else
  rota=rota-dlf_dot_product(transx,rota)*transx-dlf_dot_product(transy,rota)*transy-dlf_dot_product(transz,rota)*transz
endif
rotb=rotb-dlf_dot_product(transx,rotb)*transx-dlf_dot_product(transy,rotb)*transy-dlf_dot_product(transz,rotb)*transz
rotc=rotc-dlf_dot_product(transx,rotc)*transx-dlf_dot_product(transy,rotc)*transy-dlf_dot_product(transz,rotc)*transz

if (linear) then
  call allocate(outvec,2,N)
  call loewdin_ortho(N,2,transpose(reshape((/ rotb, rotc /),(/ N, 2 /))), &
   & outvec)
  rotb=outvec(1,1:N)
  rotc=outvec(2,1:N)
  call deallocate(outvec)
  rotb=rotb/dlf_vector_norm(rotb)
  rotc=rotc/dlf_vector_norm(rotc)
else
  call allocate(outvec,3,N)
  call loewdin_ortho(N,3,transpose(reshape((/ rota, rotb, rotc /),(/ N, 3 /))), &
   & outvec)
  rota=outvec(1,1:N)
  rotb=outvec(2,1:N)
  rotc=outvec(3,1:N)
  call deallocate(outvec)
  rota=rota/dlf_vector_norm(rota)
  rotb=rotb/dlf_vector_norm(rotb)
  rotc=rotc/dlf_vector_norm(rotc)
endif

if (dlf_dot_product(dir,dir) .gt. 0._rk) then
  dirint=dir    ! dir is expected to be input in mass-weighted coordinates!
  dirint=dirint-dlf_dot_product(dirint,transx)*transx
  dirint=dirint-dlf_dot_product(dirint,transy)*transy
  dirint=dirint-dlf_dot_product(dirint,transz)*transz
  dirint=dirint-dlf_dot_product(dirint,rota)*rota
  dirint=dirint-dlf_dot_product(dirint,rotb)*rotb
  dirint=dirint-dlf_dot_product(dirint,rotc)*rotc
  dirint=dirint/sqrt(dlf_dot_product(dirint,dirint))
else
  dirint=0._rk
endif

projector=dlf_unit_mat(N)-dlf_outer_product(transx,transx)-dlf_outer_product(transy,transy) &
 &         -dlf_outer_product(transz,transz)-dlf_outer_product(dirint,dirint) &
 &         -dlf_outer_product(rota,rota)-dlf_outer_product(rotb,rotb)-dlf_outer_product(rotc,rotc)

Hout=dlf_matmul_simp(projector,dlf_matmul_simp(Hin,projector))

return
end subroutine proj_trans_rot_dir_mw

!***************************************************
!***************************************************

! Given a set of 3N normal mode eigenvectors in mass-weighted 
! coordinates, this routine "unmixes" the six external degrees
! of freedom and forms linear combinations corresponding to 
! pure translations and rotations. The 3N-6 internal degrees 
! of freedom are untouched. 

subroutine unscramble_transrot_eigenvectors_mw(nat,N,mv,eigenvec_in,x,eigenvec_out,linear_in)
use dlf_vpt2_utility, only: error_print, loewdin_ortho
use dlf_linalg_interface_mod
use dlf_constants
implicit none
integer, intent(in) :: nat,N
real(rk), dimension(:,:), intent(in) :: eigenvec_in
real(rk), dimension(:), intent(in) :: x,mv
real(rk), dimension(:,:), intent(out) :: eigenvec_out
logical, intent(in), optional :: linear_in

interface
  subroutine dlf_matrix_invert(N,tdet,a,det)
    use dlf_parameter_module, only: rk
    implicit none
    integer  ,intent(in)    :: N
    logical  ,intent(in)    :: tdet
    real(rk) ,intent(inout) :: A(N,N)
    real(rk) ,intent(out)   :: det
  end subroutine dlf_matrix_invert
end interface

real(rk), dimension(6,6) :: A,B,atmp
integer :: i,j,nzero
real(rk), dimension(3,N) :: outvec
real(rk), dimension(N,6) :: eigenvecsub
real(rk), dimension(N) :: transx,transy,transz
real(rk), dimension(N) :: rota,rotb,rotc
real(rk) :: det_A
real(rk) :: amu2au
logical :: linear

call dlf_constants_get('AMU',amu2au)

if (size(x).ne.N)                call error_print('Size of x doesnt match N in unscramble_transrotgrad_eigenvectors_mw.')
if (size(mv).ne.nat)             call error_print('Size of mv doesnt match nat in unscramble_transrotgrad_eigenvectors_mw.')
if (size(eigenvec_in,1).ne.N)    call error_print('Size of eigenvec_in doesnt match N in unscramble_transrotgrad_eigenvectors_mw.')
if (size(eigenvec_in,2).ne.N)    call error_print('Size of eigenvec_in doesnt match N in unscramble_transrotgrad_eigenvectors_mw.')
if (size(eigenvec_out,1).ne.N)   call error_print('Size of eigenvec_out doesnt match N in unscramble_transrotgrad_eigenvectors_mw.')
if (size(eigenvec_out,2).ne.N)   call error_print('Size of eigenvec_out doesnt match N in unscramble_transrotgrad_eigenvectors_mw.')

linear=.false.
if (present(linear_in)) then
  linear=linear_in
endif

if (linear) then
  nzero=5
else
  nzero=6
endif

transx=0._rk
transy=0._rk
transz=0._rk

do i=1,nat
  transx(3*(i-1)+1)=sqrt(mv(i)/amu2au)
  transy(3*(i-1)+2)=sqrt(mv(i)/amu2au)
  transz(3*(i-1)+3)=sqrt(mv(i)/amu2au)
enddo

transx=transx/dlf_vector_norm(transx)
transy=transy/dlf_vector_norm(transy)
transz=transz/dlf_vector_norm(transz)

call rot_unit_vectors_altern(nat,N,x,mv,rota,rotb,rotc,linear)

do i=1,nat
  rota(3*(i-1)+1)=rota(3*(i-1)+1)*sqrt(mv(i)/amu2au)
  rota(3*(i-1)+2)=rota(3*(i-1)+2)*sqrt(mv(i)/amu2au)
  rota(3*(i-1)+3)=rota(3*(i-1)+3)*sqrt(mv(i)/amu2au)
  rotb(3*(i-1)+1)=rotb(3*(i-1)+1)*sqrt(mv(i)/amu2au)
  rotb(3*(i-1)+2)=rotb(3*(i-1)+2)*sqrt(mv(i)/amu2au)
  rotb(3*(i-1)+3)=rotb(3*(i-1)+3)*sqrt(mv(i)/amu2au)
  rotc(3*(i-1)+1)=rotc(3*(i-1)+1)*sqrt(mv(i)/amu2au)
  rotc(3*(i-1)+2)=rotc(3*(i-1)+2)*sqrt(mv(i)/amu2au)
  rotc(3*(i-1)+3)=rotc(3*(i-1)+3)*sqrt(mv(i)/amu2au)
enddo

if (linear) then 
  rota=0._rk
else
  rota=rota-dlf_dot_product(transx,rota)*transx-dlf_dot_product(transy,rota)*transy-dlf_dot_product(transz,rota)*transz
endif
rotb=rotb-dlf_dot_product(transx,rotb)*transx-dlf_dot_product(transy,rotb)*transy-dlf_dot_product(transz,rotb)*transz
rotc=rotc-dlf_dot_product(transx,rotc)*transx-dlf_dot_product(transy,rotc)*transy-dlf_dot_product(transz,rotc)*transz

if (linear) then
  call loewdin_ortho(N,2,transpose(reshape((/ rotb, rotc /),(/ N, 2 /))), &
   & outvec(1:2,1:N))
  rotb=outvec(1,1:N)
  rotc=outvec(2,1:N)
  rotb=rotb/dlf_vector_norm(rotb)
  rotc=rotc/dlf_vector_norm(rotc)
else
  call loewdin_ortho(N,3,transpose(reshape((/ rota, rotb, rotc /),(/ N, 3 /))), &
   & outvec)
  rota=outvec(1,1:N)
  rotb=outvec(2,1:N)
  rotc=outvec(3,1:N)
  rota=rota/dlf_vector_norm(rota)
  rotb=rotb/dlf_vector_norm(rotb)
  rotc=rotc/dlf_vector_norm(rotc)
endif

eigenvecsub(:,:)=0._rk
do i=1,nzero
  eigenvecsub(:,i)=eigenvec_in(:,i)/dlf_vector_norm(eigenvec_in(:,i))
enddo

do i=1,nzero
  A(i,1)=dlf_dot_product(eigenvecsub(:,i),transx)
  A(i,2)=dlf_dot_product(eigenvecsub(:,i),transy)
  A(i,3)=dlf_dot_product(eigenvecsub(:,i),transz)
  if (linear) then
    A(i,4)=dlf_dot_product(eigenvecsub(:,i),rotb)
    A(i,5)=dlf_dot_product(eigenvecsub(:,i),rotc)
  else
    A(i,4)=dlf_dot_product(eigenvecsub(:,i),rota)
    A(i,5)=dlf_dot_product(eigenvecsub(:,i),rotb)
    A(i,6)=dlf_dot_product(eigenvecsub(:,i),rotc)
  endif
enddo

atmp(1:nzero,1:nzero)=A(1:nzero,1:nzero)
call dlf_matrix_invert(nzero,.true.,atmp(1:nzero,1:nzero),det_A)
B(1:nzero,1:nzero)=atmp(1:nzero,1:nzero)

eigenvec_out=eigenvec_in

do i=1,nzero
  eigenvec_out(:,i)=0._rk
  do j=1,nzero
    eigenvec_out(:,i)=eigenvec_out(:,i)+eigenvecsub(:,j)*B(i,j)
  enddo
enddo

return
end subroutine unscramble_transrot_eigenvectors_mw

!***************************************************
!***************************************************

! Generate a set of three unit vectors, describing the 
! rotations (in the infinitesimal limit) of a molecule around
! the x, y, z axes.
! Expect coordinates in xyzxyz... order!

subroutine rot_unit_vectors_altern(nat,N,coo,mv,rota,rotb,rotc,linear_in)
use dlf_vpt2_utility, only: error_print, leci
use dlf_linalg_interface_mod
implicit none

integer, intent(in) :: nat, N
real(rk), dimension(:), intent(in)  :: coo,mv
real(rk), dimension(:), intent(out) :: rota,rotb,rotc
logical, intent(in), optional :: linear_in

real(rk), dimension(3,3) :: mom,momevec,sqrtlaminv,sqrtmominv
real(rk), dimension(3) :: momeval
real(rk), dimension(nat) :: x,y,z
real(rk), dimension(N,3) :: rotabc
integer :: i,j,jb,iev,ia,ib,ig,iat
logical  :: linear

if (size(coo).ne.N)  call error_print('Size of coo doesnt match N in rot_unit_vectors_altern.')
if (size(mv).ne.nat) call error_print('Size of mv doesnt match nat in rot_unit_vectors_altern.')
if (size(rota).ne.N) call error_print('Size of rota doesnt match N in rot_unit_vectors_altern.')
if (size(rotb).ne.N) call error_print('Size of rotb doesnt match N in rot_unit_vectors_altern.')
if (size(rotc).ne.N) call error_print('Size of rotc doesnt match N in rot_unit_vectors_altern.')

linear=.false.
if (present(linear_in)) then
  linear=linear_in
endif

do i=1,nat
  x(i)=coo(3*(i-1)+1)
  y(i)=coo(3*(i-1)+2)
  z(i)=coo(3*(i-1)+3)
enddo

mom=moi(nat,x,y,z,mv)
call diagonalize_moi(mom,momeval,momevec)
sqrtlaminv=0._rk
do j=1,3
  if (linear .and. abs(momeval(j))<1.e-7_rk) then
    sqrtlaminv(j,j)=0._rk
  else
    sqrtlaminv(j,j)=1._rk/sqrt(momeval(j))
  endif
enddo

sqrtmominv=dlf_matrix_ortho_trans(momevec,sqrtlaminv,1)

rotabc=0._rk
do iev=1,3
  do iat=1,nat
    do ig=1,3
      j=3*(iat-1)+ig
      do ia=1,3
        do ib=1,3
          jb=3*(iat-1)+ib
          rotabc(j,iev)=rotabc(j,iev)+sqrtmominv(iev,ia)*leci(ia,ib,ig)*coo(jb)
        enddo
      enddo
    enddo
  enddo
enddo

rota=rotabc(1:N,1)
rotb=rotabc(1:N,2)
rotc=rotabc(1:N,3)

rota=rota/dlf_vector_norm(rota)
rotb=rotb/dlf_vector_norm(rotb)
rotc=rotc/dlf_vector_norm(rotc)

return
end subroutine rot_unit_vectors_altern

!***************************************************
!***************************************************

! Shift molecule so that Cartesian origin = center of mass

subroutine shift_com(nat,x,y,z,mv,xshift,yshift,zshift)
use dlf_vpt2_utility, only: error_print
implicit none
integer, intent(in) :: nat
real(rk), dimension(nat), intent(in) :: x,y,z,mv
real(rk), dimension(nat), intent(out) :: xshift,yshift,zshift

real(rk) :: xcom,ycom,zcom,mtot

if (size(x).ne.nat) call error_print('Size of x doesnt match nat in shift_com.')
if (size(y).ne.nat) call error_print('Size of y doesnt match nat in shift_com.')
if (size(z).ne.nat) call error_print('Size of z doesnt match nat in shift_com.')
if (size(mv).ne.nat) call error_print('Size of mv doesnt match nat in shift_com.')
if (size(xshift).ne.nat) call error_print('Size of xshift doesnt match nat in shift_com.')
if (size(yshift).ne.nat) call error_print('Size of yshift doesnt match nat in shift_com.')
if (size(zshift).ne.nat) call error_print('Size of zshift doesnt match nat in shift_com.')

mtot=sum(mv(1:nat))
xcom=sum(x(1:nat)*mv(1:nat))/mtot
ycom=sum(y(1:nat)*mv(1:nat))/mtot
zcom=sum(z(1:nat)*mv(1:nat))/mtot

xshift=x-xcom
yshift=y-ycom
zshift=z-zcom

return
end subroutine shift_com

!***************************************************
!***************************************************

! Rotate molecule so that principle axes = Cartesian axes

subroutine rotate_to_princ_axes(nat,x,y,z,mv,xrot,yrot,zrot)
use dlf_vpt2_utility, only: error_print
implicit none
integer, intent(in) :: nat
real(rk), dimension(nat), intent(in) :: x,y,z,mv
real(rk), dimension(nat), intent(out) :: xrot,yrot,zrot

real(rk), dimension(3,3)  :: mom,momevec
real(rk), dimension(3)  :: momeval
real(rk) :: tmp1,tmp2,tmp3
integer :: i

if (size(x).ne.nat) call error_print('Size of x doesnt match nat in rotate_to_princ_axes.')
if (size(y).ne.nat) call error_print('Size of y doesnt match nat in rotate_to_princ_axes.')
if (size(z).ne.nat) call error_print('Size of z doesnt match nat in rotate_to_princ_axes.')
if (size(mv).ne.nat) call error_print('Size of mv doesnt match nat in rotate_to_princ_axes.')
if (size(xrot).ne.nat) call error_print('Size of xrot doesnt match nat in rotate_to_princ_axes.')
if (size(yrot).ne.nat) call error_print('Size of yrot doesnt match nat in rotate_to_princ_axes.')
if (size(zrot).ne.nat) call error_print('Size of zrot doesnt match nat in rotate_to_princ_axes.')

mom=moi(nat,x,y,z,mv)
call diagonalize_moi(mom,momeval,momevec)

do i=1,nat
  tmp1 = x(i)
  tmp2 = y(i)
  tmp3 = z(i)
  xrot(i)=tmp1*momevec(1,1)+tmp2*momevec(2,1)+tmp3*momevec(3,1)
  yrot(i)=tmp1*momevec(1,2)+tmp2*momevec(2,2)+tmp3*momevec(3,2)
  zrot(i)=tmp1*momevec(1,3)+tmp2*momevec(2,3)+tmp3*momevec(3,3)
enddo

return

end subroutine rotate_to_princ_axes

!***************************************************
!***************************************************

! Rotate molecule so that principle axes = Cartesian axes
! Do this in an ordered way, so that moment of inertia 
! increases in the order x,y,z (rotational constant decreases).
! Note the special treatment of linear molecules (A set to -666)

subroutine rotate_to_princ_axes_ordered(nat,x,y,z,mv,xrot,yrot,zrot,A,B,C)
use dlf_vpt2_utility, only: error_print
use dlf_sort_module, only: dlf_sort_shell_ind
use dlf_constants
implicit none
integer, intent(in) :: nat
real(rk), dimension(nat), intent(in) :: x,y,z,mv
real(rk), dimension(nat), intent(out) :: xrot,yrot,zrot
real(rk), intent(out) :: A,B,C

real(rk), parameter :: zer_tol=1.e-10_rk
character(3), parameter :: sort_order='asc'
real(rk), dimension(3,3)  :: mom,momevec,momevec_ordered
real(rk), dimension(3)  :: momeval,momeval_ordered
real(rk) :: tmp1,tmp2,tmp3
integer :: i
integer, dimension(3) :: eval_sort_ind
real(rk),dimension(3) :: rc
real(rk) :: au2cmi

call dlf_constants_get("CM_INV_FROM_HARTREE",au2cmi)

if (size(x).ne.nat) call error_print('Size of x doesnt match nat in rotate_to_princ_axes_ordered.')
if (size(y).ne.nat) call error_print('Size of y doesnt match nat in rotate_to_princ_axes_ordered.')
if (size(z).ne.nat) call error_print('Size of z doesnt match nat in rotate_to_princ_axes_ordered.')
if (size(mv).ne.nat) call error_print('Size of mv doesnt match nat in rotate_to_princ_axes_ordered.')
if (size(xrot).ne.nat) call error_print('Size of xrot doesnt match nat in rotate_to_princ_axes_ordered.')
if (size(yrot).ne.nat) call error_print('Size of yrot doesnt match nat in rotate_to_princ_axes_ordered.')
if (size(zrot).ne.nat) call error_print('Size of zrot doesnt match nat in rotate_to_princ_axes_ordered.')

mom=moi(nat,x,y,z,mv)
call diagonalize_moi(mom,momeval,momevec)
call dlf_sort_shell_ind(momeval,eval_sort_ind)

if (sort_order=='asc') then
  continue
elseif (sort_order=='des') then
  eval_sort_ind(1:3)=eval_sort_ind(3:1:-1)
else
  call error_print('rotate_to_princ_axes_ordered: Sort order must be "asc" or "des"')
endif

do i=1,3
  momeval_ordered(i)  =momeval(eval_sort_ind(i))
  momevec_ordered(:,i)=momevec(:,eval_sort_ind(i))
enddo

do i=1,3
  if (momeval_ordered(i)>zer_tol) then
    rc(i)=0.5_rk/momeval_ordered(i)*au2cmi
  else
    rc(i)=-666._rk
  endif
enddo

A=rc(1)
B=rc(2)
C=rc(3)

do i=1,nat
  tmp1 = x(i)
  tmp2 = y(i)
  tmp3 = z(i)
  xrot(i)=tmp1*momevec_ordered(1,1)+tmp2*momevec_ordered(2,1)+tmp3*momevec_ordered(3,1)
  yrot(i)=tmp1*momevec_ordered(1,2)+tmp2*momevec_ordered(2,2)+tmp3*momevec_ordered(3,2)
  zrot(i)=tmp1*momevec_ordered(1,3)+tmp2*momevec_ordered(2,3)+tmp3*momevec_ordered(3,3)
enddo

return

end subroutine rotate_to_princ_axes_ordered

!***************************************************
!***************************************************

! Decide if a molecule is a (quasi-) oblate or prolate top

function top_OP(Ain,Bin,Cin)
implicit none
real(rk), intent(in) :: Ain,Bin,Cin
character(1) :: top_OP

real(rk) :: tmp,A,B,C

A=Ain
B=Bin
C=Cin

if (.not.(A.ge.B.and.B.ge.C)) then
  if (B.gt.A) then
    tmp=A
    A=B
    B=tmp
  endif
  if (C.gt.A) then
    tmp=A
    A=C
    C=tmp
  endif
  if (C.gt.B) then
    tmp=B
    B=C
    C=tmp
  endif
endif

if (abs(A-B).ge.abs(B-C)) then  ! near-prolate
  top_OP='P'  
else                            ! near-oblate
  top_OP='O'
endif

return
end function top_OP

!***************************************************
!***************************************************

! Get rotational constants from coordinates & masses

subroutine rotconst(nat,x,y,z,mv,A,B,C)
use dlf_vpt2_utility, only: error_print
USE dlf_sort_module, only: dlf_sort_shell
use dlf_constants
implicit none
integer, intent(in) :: nat
real(rk), dimension(nat), intent(in) :: x,y,z,mv
real(rk), intent(out) :: A,B,C

real(rk), dimension(3,3)  :: mom,momevec
real(rk), dimension(3)  :: momeval,rc
real(rk) :: au2cmi

call dlf_constants_get("CM_INV_FROM_HARTREE",au2cmi)

if (size(x).ne.nat)    call error_print('Size of x doesnt match nat in rotconst.')
if (size(y).ne.nat)    call error_print('Size of y doesnt match nat in rotconst.')
if (size(z).ne.nat)    call error_print('Size of z doesnt match nat in rotconst.')
if (size(mv).ne.nat)   call error_print('Size of mv doesnt match nat in rotconst.')

mom=moi(nat,x,y,z,mv)
call diagonalize_moi(mom,momeval,momevec)

call dlf_sort_shell(momeval)
rc=0.5_rk/momeval

A=rc(1)*au2cmi
B=rc(2)*au2cmi
C=rc(3)*au2cmi

return

end subroutine rotconst

!***************************************************
!***************************************************

! Diagonalize the intertia tensor

subroutine diagonalize_moi(moi,eval,evec)
use dlf_vpt2_utility, only: swapvw,swapxy,det3x3
use dlf_linalg_interface_mod
implicit none

real(rk), dimension(3,3), intent(in)  :: moi
real(rk), dimension(3,3), intent(out),target :: evec
real(rk), dimension(3), intent(out) :: eval

real(rk), dimension(:), pointer :: col1,col2
real(rk), dimension(3) ::traces
integer :: i,maxtraceloc
real(rk) :: deter

nullify(col1,col2)


call dlf_matrix_diagonalise(3,moi,eval,evec)

do i=1,3
  if(evec(maxloc(abs(evec(:,i)),dim=1),i).lt.0._rk) then
    evec(:,i)=-evec(:,i)
  endif
enddo

deter=det3x3(evec)
if (deter.lt.0._rk) then
  col1=>evec(:,1)
  col2=>evec(:,2)
  call swapvw(col1,col2)
  call swapxy(eval(1),eval(2))
  nullify(col1,col2)
endif

traces(1)=dlf_trace(evec)
traces(2)=dlf_trace(cshift(evec,shift=1,dim=2))
traces(3)=dlf_trace(cshift(evec,shift=2,dim=2))

maxtraceloc=maxloc(traces,dim=1)
evec=cshift(evec,shift=maxtraceloc-1,dim=2)
eval=cshift(eval,shift=maxtraceloc-1)

return
end subroutine diagonalize_moi

!***************************************************
!***************************************************

! Generate the inertia tensor

function moi(nat,x,y,z,mv)
use dlf_vpt2_utility, only: error_print
implicit none
real(rk), dimension(3,3) :: moi
integer, intent(in) :: nat
real(rk), dimension(:), intent(in) :: x,y,z,mv

integer :: i

if (size(x).ne.nat) call error_print('Size of x doesnt match nat in moi.')
if (size(y).ne.nat) call error_print('Size of y doesnt match nat in moi.')
if (size(z).ne.nat) call error_print('Size of z doesnt match nat in moi.')
if (size(mv).ne.nat) call error_print('Size of mv doesnt match nat in moi.')

moi=0._rk

do i=1,nat
  moi(1,1)=moi(1,1)+mv(i)*(y(i)**2+z(i)**2)
  moi(2,2)=moi(2,2)+mv(i)*(x(i)**2+z(i)**2)
  moi(3,3)=moi(3,3)+mv(i)*(x(i)**2+y(i)**2)
  moi(1,2)=moi(1,2)-mv(i)*(x(i)*y(i))
  moi(1,3)=moi(1,3)-mv(i)*(x(i)*z(i))
  moi(2,3)=moi(2,3)-mv(i)*(y(i)*z(i))
enddo
moi(2,1)=moi(1,2)
moi(3,1)=moi(1,3)
moi(3,2)=moi(2,3)

return
end function moi

!***************************************************
!***************************************************

end module dlf_vpt2_project

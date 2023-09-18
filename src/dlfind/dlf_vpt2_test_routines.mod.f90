!stamp
module dlf_vpt2_test_routines
use dlf_parameter_module
!use dlf_global, only: glob, stdout
implicit none

contains

!***************************************************
!***************************************************

subroutine interpolate_between_two_structures(nat,nvar,nzero,grad_routine)
use dlf_vpt2_intcoord
use dlf_vpt2_utility, only: matrix_output, symb2mass, vector_output
use dlf_allocate
use dlf_constants
use dlf_global, only: stdout
implicit none
integer, intent(in) :: nat,nvar,nzero
interface
subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status)
  use dlf_parameter_module, only: rk
    implicit none
    integer   ,intent(in)    :: nvar
    real(rk)  ,intent(in)    :: coords(nvar)
    real(rk)  ,intent(out)   :: energy
    real(rk)  ,intent(out)   :: gradient(nvar)
    integer   ,intent(in)    :: iimage
    integer   ,intent(in)    :: kiter
    integer   ,intent(out)   :: status
  end subroutine grad_routine
end interface

integer, parameter :: ninterp=20
integer :: intn,nbonds,nbangles,ntorsions
integer,   dimension(nvar-nzero,4)    :: def
integer,allocatable,dimension(:,:) :: bonds
integer,allocatable,dimension(:,:) :: bangles
integer,allocatable,dimension(:,:) :: torsions
real(rk),allocatable,dimension(:) :: bond_vals_1  , bond_vals_2, bond_vals_c
real(rk),allocatable,dimension(:) :: angle_vals_1 , angle_vals_2, angle_vals_c
real(rk),allocatable,dimension(:) :: tors_vals_1  , tors_vals_2, tors_vals_c
real(rk), dimension(1:ninterp) :: ener
real(rk) :: coords_c(nvar), coords0_1(nvar), coords0_2(nvar), grad(nvar)
real(rk) :: alpha
integer :: k,intdum,i,j,ji,jf,i_int
integer :: iimage,kiter,status
character(len=20) :: kchar
character(len=1) :: chdum
character(len=1000), parameter :: fn1='coord_inter_alpha.xyz', fn2='coord_inter_omega.xyz'
character(len=2), dimension(nat) :: atsym
real(rk), dimension(nat) :: mv
real(rk) :: amu2au
real(rk) :: autoang
real(rk) :: ang2bohr

call dlf_constants_get("ANG_AU",autoang)
call dlf_constants_get("ANG_AU",ang2bohr)
ang2bohr=1._rk/ang2bohr
call dlf_constants_get('AMU',amu2au)

open(1177,file=trim(adjustl(fn1)))
read(1177,*) intdum
read(1177,'(A)') chdum
do i=1,nat
  ji=3*(i-1)+1
  jf=3*(i-1)+3
  read(1177,*) atsym(i), (coords0_1(j), j=ji,jf)
enddo
close(1177)

open(1188,file=trim(adjustl(fn2)))
read(1188,*) intdum
read(1188,'(A)') chdum
do i=1,nat
  ji=3*(i-1)+1
  jf=3*(i-1)+3
  read(1188,*) atsym(i), (coords0_2(j), j=ji,jf)
enddo
close(1188)

coords0_1(:)=coords0_1(:)*ang2bohr
coords0_2(:)=coords0_2(:)*ang2bohr
mv(:)=symb2mass(atsym(:),single_iso_in=.true.)*amu2au
!call vector_output(mv,stdout,'F20.12','mv')
!read(*,*)

intn=nvar-nzero
call read_int_coord_def(intn,nbonds,nbangles,ntorsions,def)

call allocate(bonds,nbonds,2)
call allocate(bangles,nbangles,3)
call allocate(torsions,ntorsions,4)
call allocate(bond_vals_1,nbonds)
call allocate(bond_vals_2,nbonds)
call allocate(bond_vals_c,nbonds)
call allocate(angle_vals_1,nbangles)
call allocate(angle_vals_2,nbangles)
call allocate(angle_vals_c,nbangles)
call allocate(tors_vals_1,ntorsions)
call allocate(tors_vals_2,ntorsions)
call allocate(tors_vals_c,ntorsions)

bonds(1:nbonds,1:2)       = def(1:nbonds,1:2)
bangles(1:nbangles,1:3)   = def(nbonds+1:nbonds+nbangles,1:3)
torsions(1:ntorsions,1:4) = def(nbonds+nbangles+1:nbonds+nbangles+ntorsions,1:4)

call cart_to_int(nat,intn,nbonds,nbangles,ntorsions,def,coords0_1,bond_vals_1,angle_vals_1,tors_vals_1)
call cart_to_int(nat,intn,nbonds,nbangles,ntorsions,def,coords0_2,bond_vals_2,angle_vals_2,tors_vals_2)

write(stdout,'(A)') ''
write(stdout,'(A)') '  **********  Initial structure for interpolation  **********  '
write(stdout,'(A)') ''
call int_punch(nat,intn,nbonds,nbangles,ntorsions,def,bond_vals_1,angle_vals_1,tors_vals_1,6)
write(stdout,'(A)') ''
write(stdout,'(A)') '  **********   Final structure for interpolation   **********  '
write(stdout,'(A)') ''
call int_punch(nat,intn,nbonds,nbangles,ntorsions,def,bond_vals_2,angle_vals_2,tors_vals_2,6)

open(919,file='interp.xyz')
do i_int=1,ninterp
  alpha=real(i_int-1)/real(ninterp-1)
  !write(stdout,*) 'alpha: ', alpha
  bond_vals_c(:) =alpha*bond_vals_2(:) +(1._rk-alpha)*bond_vals_1(:) 
  angle_vals_c(:)=alpha*angle_vals_2(:)+(1._rk-alpha)*angle_vals_1(:)
  tors_vals_c(:) =alpha*tors_vals_2(:) +(1._rk-alpha)*tors_vals_1(:) 
  !write(stdout,'(A)') ''
  !write(stdout,'(A)') '  **********   Interpolation structure   **********  '
  !write(stdout,'(A)') ''
  !call int_punch(nat,intn,nbonds,nbangles,ntorsions,def,bond_vals_c,angle_vals_c,tors_vals_c,6)
  call int_to_cart(nat,intn,nbonds,nbangles,ntorsions,def,coords0_1,mv, &
                   &   bond_vals_c,angle_vals_c, &
                   &   tors_vals_c,coords_c,1.e-13_rk)
  write(919,*) nat
  write(919,'(A)') 'intermediate coordinates'
  do i=1,nat
    j=3*(i-1)+1
    write(919,'(A,1X,3(ES20.8,1X))') atsym(i), (coords_c(k)*autoang,k=j,j+2)
  enddo
  call grad_routine(nvar,coords_c,ener(i_int),grad,iimage,kiter,status)
  write(stdout,*) i_int, ener(i_int)
enddo
close(919)

call deallocate(tors_vals_c)
call deallocate(tors_vals_2)
call deallocate(tors_vals_1)
call deallocate(angle_vals_c)
call deallocate(angle_vals_2)
call deallocate(angle_vals_1)
call deallocate(bond_vals_c)
call deallocate(bond_vals_2)
call deallocate(bond_vals_1)
call deallocate(torsions)
call deallocate(bangles)
call deallocate(bonds)

stop 0
end subroutine interpolate_between_two_structures

!***************************************************
!***************************************************

subroutine compare_B_C_ridders_analytical(nat,nvar,nzero,coords)
use dlf_vpt2_intcoord
use dlf_vpt2_utility, only: matrix_output
use dlf_allocate
use dlf_global, only: stdout
implicit none
integer, intent(in) :: nat,nvar,nzero
real(rk), dimension(nvar), intent(in) :: coords

integer :: intn,nbonds,nbangles,ntorsions
integer,   dimension(nvar-nzero,4)    :: def
real(rk),  dimension(nvar-nzero,3*nat)   :: B_ana, B_num
real(rk),  dimension(3*nat,3*nat,nvar-nzero) :: C_ana, C_num
integer,allocatable,dimension(:,:) :: bonds
integer,allocatable,dimension(:,:) :: bangles
integer,allocatable,dimension(:,:) :: torsions
real(rk),allocatable,dimension(:) :: bond_vals
real(rk),allocatable,dimension(:) :: angle_vals
real(rk),allocatable,dimension(:) :: tors_vals
integer :: k
character(len=20) :: kchar

intn=nvar-nzero

call read_int_coord_def(intn,nbonds,nbangles,ntorsions,def)

call allocate(bonds,nbonds,2)
call allocate(bangles,nbangles,3)
call allocate(torsions,ntorsions,4)
call allocate(bond_vals,nbonds)
call allocate(angle_vals,nbangles)
call allocate(tors_vals,ntorsions)

bonds(1:nbonds,1:2)       = def(1:nbonds,1:2)
bangles(1:nbangles,1:3)   = def(nbonds+1:nbonds+nbangles,1:3)
torsions(1:ntorsions,1:4) = def(nbonds+nbangles+1:nbonds+nbangles+ntorsions,1:4)

call cart_to_int(nat,intn,nbonds,nbangles,ntorsions,def,coords,bond_vals,angle_vals,tors_vals)
call int_punch(nat,intn,nbonds,nbangles,ntorsions,def,bond_vals,angle_vals,tors_vals,6)

call generate_B_C_ridders(nat,intn,coords,nbonds,nbangles, &
        & ntorsions,bonds,bangles,torsions,B_num,C_num)

call generate_B_C_analytical(nat,intn,coords,nbonds,nbangles, &
        & ntorsions,bonds,bangles,torsions,B_ana,C_ana)

!call matrix_output(B_num, stdout, 'F20.12', 'B matrix, numerical')
!call matrix_output(B_ana, stdout, 'F20.12', 'B matrix, analytical')
call matrix_output(B_ana-B_num, stdout, 'F20.12', 'B matrix, analytical -- numerical')
read(*,*)

do k=1,intn
  write(kchar,'(I0)') k
  !call matrix_output(C_num(:,:,k), stdout, 'F20.12', 'C, numerical,  k = '//trim(kchar))
  !call matrix_output(C_ana(:,:,k), stdout, 'ES20.12', 'C, analytical, k = '//trim(kchar))
  call matrix_output(C_ana(:,:,k)-C_num(:,:,k), stdout, 'F20.12', 'C, analytical -- numerical, k = '//trim(kchar))
  read(*,*)
enddo

call deallocate(tors_vals)
call deallocate(angle_vals)
call deallocate(bond_vals)
call deallocate(torsions)
call deallocate(bangles)
call deallocate(bonds)

stop 0
end subroutine compare_B_C_ridders_analytical

!***************************************************
!***************************************************

subroutine dlf_vpt2_test()
  implicit none
  
  !call test_matmul()
  !call test_matvecmul()
  !call test_ortho_trans()
  !call test_outer_prod()
  !call test_inner_prod()
  !call test_bilinear_form()
  
  return
end subroutine dlf_vpt2_test

!***************************************************
!***************************************************

subroutine test_matmul()
  use dlf_vpt2_utility, only: matrix_output
  use dlf_linalg_interface_mod
  use dlf_global, only: stdout
  implicit none
  real(rk), parameter :: scal_ampl=20._rk
  integer, parameter :: m=5,n=10,k=7
  real(rk), dimension(m,k) :: A
  real(rk), dimension(k,n) :: B
  real(rk), dimension(m,n) :: C_native, C_blas
  integer :: i, j
  real(rk) :: scal
  
  do while (.true.)
    do i=1,k
      do j=1,m
        A(j,i)=ran_real()-0.5_rk
      enddo
      do j=1,n
        B(i,j)=ran_real()-0.5_rk
      enddo
    enddo
    scal=(ran_real()-0.5_rk)*scal_ampl
    A(:,:)=A(:,:)*scal
    scal=(ran_real()-0.5_rk)*scal_ampl
    B(:,:)=B(:,:)*scal
    C_native=matmul(A,B)
    C_blas=dlf_matmul_simp(A,B)
    call matrix_output(A,stdout,'F22.16','A')
    call matrix_output(B,stdout,'F22.16','B')
    call matrix_output(C_native,stdout,'F22.16','C (native matmul)')
    call matrix_output(C_blas,stdout,'F22.16','C (via dlf_matmul_simp)')
    call matrix_output(C_blas-C_native,stdout,'F22.16','diff(C)')
    read(*,*)
  enddo
  
  return
end subroutine test_matmul

!***************************************************
!***************************************************

subroutine test_matvecmul()
  use dlf_vpt2_utility, only: matrix_output, vector_output
  use dlf_linalg_interface_mod
  use dlf_global, only: stdout
  implicit none
  real(rk), parameter :: scal_ampl=20._rk
  integer, parameter :: m=7,n=11
  real(rk), dimension(m,n) :: A
  real(rk), dimension(n) :: b
  real(rk), dimension(m) :: c_native, c_blas
  integer :: i,j
  real(rk) :: scal
  
  do while (.true.)
    do i=1,n
      b(i)=ran_real()-0.5_rk
      do j=1,n
        A(j,i)=ran_real()-0.5_rk
      enddo
    enddo
    scal=(ran_real()-0.5_rk)*scal_ampl
    A(:,:)=A(:,:)*scal
    scal=(ran_real()-0.5_rk)*scal_ampl
    b(:)=b(:)*scal
    c_native=matmul(A,b)
    c_blas=dlf_matmul_simp(A,b)
    call matrix_output(A,stdout,'F22.16','A')
    call vector_output(b,stdout,'F22.16','b')
    call vector_output(c_native,stdout,'F22.16','c (native matmul)')
    call vector_output(c_blas,stdout,'F22.16','c (via dlf_matmul_simp)')
    call vector_output(c_blas-c_native,stdout,'F22.16','diff(c)')
    read(*,*)
  enddo
  
  return
end subroutine test_matvecmul

!***************************************************
!***************************************************

subroutine test_outer_prod()
  use dlf_vpt2_utility, only: matrix_output, vector_output
  use dlf_global, only: stdout
  implicit none
  interface
    function dlf_outer_product(x,y) result(Mat)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk) ,intent(in)    :: x(:), y(:)
      real(rk) :: Mat(size(x),size(y))
    end function dlf_outer_product
  end interface
  real(rk), parameter :: scal_ampl=20._rk
  integer, parameter :: m=7,n=11
  real(rk), dimension(m) :: a
  real(rk), dimension(n) :: b
  real(rk), dimension(m,n) :: C_native, C_blas
  integer :: i,j
  real(rk) :: scal
  
  do while (.true.)
    do i=1,m
      a(i)=ran_real()-0.5_rk
    enddo
    do i=1,n
      b(i)=ran_real()-0.5_rk
    enddo
    scal=(ran_real()-0.5_rk)*scal_ampl
    a(:)=a(:)*scal
    scal=(ran_real()-0.5_rk)*scal_ampl
    b(:)=b(:)*scal
    C_native=spread(a,dim=2,ncopies=size(b))*spread(b,dim=1,ncopies=size(a))
    C_blas=dlf_outer_product(a,b)
    call vector_output(a,stdout,'F22.16','a')
    call vector_output(b,stdout,'F22.16','b')
    call matrix_output(C_native,stdout,'F22.16','c (native via f90 array ops.)')
    call matrix_output(C_blas,stdout,'F22.16','c (via dlf_outer_prod)')
    call matrix_output(C_blas-C_native,stdout,'F22.16','diff(C)')
    read(*,*)
  enddo
  
  return
end subroutine test_outer_prod

!***************************************************
!***************************************************

subroutine test_inner_prod()
  use dlf_vpt2_utility, only: matrix_output, vector_output
  use dlf_linalg_interface_mod
  use dlf_global, only: stdout
  implicit none
  real(rk), parameter :: scal_ampl=20._rk
  integer, parameter :: m=11
  real(rk), dimension(m) :: a
  real(rk), dimension(m) :: b
  real(rk) :: ab_native, ab_blas, anorm_native, anorm_blas, bnorm_native, bnorm_blas
  integer :: i
  real(rk) :: scal
  
  do while (.true.)
    do i=1,m
      a(i)=ran_real()-0.5_rk
    enddo
    do i=1,m
      b(i)=ran_real()-0.5_rk
    enddo
    scal=(ran_real()-0.5_rk)*scal_ampl
    a(:)=a(:)*scal
    scal=(ran_real()-0.5_rk)*scal_ampl
    b(:)=b(:)*scal
    ab_native=dot_product(a,b)
    anorm_native=sqrt(dot_product(a,a))
    bnorm_native=sqrt(dot_product(b,b))
    ab_blas=dlf_dot_product(a,b)
    anorm_blas=dlf_vector_norm(a)
    bnorm_blas=dlf_vector_norm(b)
    write(stdout,'(A,2F22.16,ES13.5)') '<a|b> (native, BLAS, diff): ', ab_native, ab_blas, ab_blas-ab_native
    write(stdout,'(A,2F22.16,ES13.5)') '||a|| (native, BLAS, diff): ', anorm_native, anorm_blas, anorm_blas-anorm_native
    write(stdout,'(A,2F22.16,ES13.5)') '||b|| (native, BLAS, diff): ', bnorm_native, bnorm_blas, bnorm_blas-bnorm_native
    write(stdout,*) ''
    read(*,*)
  enddo
  
  return
end subroutine test_inner_prod

!!***************************************************
!!***************************************************

subroutine test_ortho_trans()
  use dlf_vpt2_utility, only: matrix_output
  use dlf_linalg_interface_mod
  use dlf_global, only: stdout
  implicit none
  real(rk), parameter :: scal_ampl=20._rk
  integer, parameter :: m=12,k=9
  real(rk), dimension(m,k) :: A
  real(rk), dimension(m,m) :: B
  real(rk), dimension(k,k) :: B_rev
  real(rk), dimension(k,k) :: C_native, C_blas
  real(rk), dimension(m,m) :: C_native_rev, C_blas_rev
  integer :: i, j, mode
  real(rk) :: scal
  
  do while (.true.)
    do i=1,k
      do j=1,m
        A(j,i)=ran_real()-0.5_rk
      enddo
    enddo
    scal=ran_real()
    if (scal<0.5_rk) then
      mode=0
      write(stdout,*) 'Mode 0 => C = A^T B A'
      do i=1,m
        do j=1,m
          B(i,j)=ran_real()-0.5_rk
        enddo
      enddo
    else
      mode=1
      write(stdout,*) 'Mode 1 => C = A B A^T'
      do i=1,k
        do j=1,k
          B_rev(i,j)=ran_real()-0.5_rk
        enddo
      enddo
    endif
    scal=(ran_real()-0.5_rk)*scal_ampl
    A(:,:)=A(:,:)*scal
    scal=(ran_real()-0.5_rk)*scal_ampl
    B(:,:)=B(:,:)*scal
    scal=(ran_real()-0.5_rk)*scal_ampl
    B_rev(:,:)=B_rev(:,:)*scal
    if (mode==0) then
      C_native=matmul(transpose(A),matmul(B,A))
      C_blas=dlf_matrix_ortho_trans(A,B,mode)
      call matrix_output(A,stdout,'F22.16','A')
      call matrix_output(B,stdout,'F22.16','B')
      call matrix_output(C_native,stdout,'F22.16','C (native matmul/tranpose)')
      call matrix_output(C_blas,stdout,'F22.16','C (via dlf_matrix_ortho_trans')
      call matrix_output(C_blas-C_native,stdout,'F22.16','diff(C)')
    else
      C_native_rev=matmul(A,matmul(B_rev,transpose(A)))
      C_blas_rev=dlf_matrix_ortho_trans(A,B_rev,mode)
      call matrix_output(A,stdout,'F22.16','A')
      call matrix_output(B_rev,stdout,'F22.16','B')
      call matrix_output(C_native_rev,stdout,'F22.16','C (native matmul/tranpose)')
      call matrix_output(C_blas_rev,stdout,'F22.16','C (via dlf_matrix_ortho_trans)')
      call matrix_output(C_blas_rev-C_native_rev,stdout,'F22.16','diff(C)')
    endif
    read(*,*)
  enddo
  
  return
end subroutine test_ortho_trans

!!***************************************************
!!***************************************************

subroutine test_bilinear_form()
  use dlf_vpt2_utility, only: matrix_output
  use dlf_global, only: stdout
  implicit none
  interface
    function dlf_bilinear_form(x,H,y) result(z)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk),intent(in) :: x(:), y(:)
      real(rk),intent(in) :: H(size(x),size(y))
      real(rk) :: z
    end function dlf_bilinear_form
  end interface
  real(rk), parameter :: scal_ampl=20._rk
  integer, parameter :: m=12,k=9
  real(rk), dimension(m,k) :: H
  real(rk), dimension(m) :: x
  real(rk), dimension(k) :: y
  real(rk) :: bi_native, bi_blas
  integer :: i, j
  real(rk) :: scal
  
  do while (.true.)
    do i=1,k
      y(i)=ran_real()-0.5_rk
      do j=1,m
        H(j,i)=ran_real()-0.5_rk
      enddo
    enddo
    do i=1,m
      x(i)=ran_real()-0.5_rk
    enddo
    scal=(ran_real()-0.5_rk)*scal_ampl
    H(:,:)=H(:,:)*scal
    scal=(ran_real()-0.5_rk)*scal_ampl
    x(:)=x(:)*scal
    scal=(ran_real()-0.5_rk)*scal_ampl
    y(:)=y(:)*scal
    
    bi_native=dot_product(x,matmul(H,y))
    bi_blas  =dlf_bilinear_form(x,H,y)
    write(stdout,'(A,2F22.16,ES13.5)') 'x^T H y (native, BLAS, diff): ', bi_native, bi_blas, bi_blas-bi_native
    write(stdout,*) ''
    read(*,*)
  enddo
  
  return
end subroutine test_bilinear_form

!***************************************************
!***************************************************

function ran_real()
  implicit none
  real(rk) :: ran_real
  logical, save :: initialized=.false.
  
  if (.not. initialized) then
    call random_seed()
    initialized=.true.
  endif
  
  call random_number(ran_real)
  return
end function ran_real

!***************************************************
!***************************************************

end module dlf_vpt2_test_routines



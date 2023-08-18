!!****h* utilities/linalg
!!
!! FUNCTION
!! Linear algebra utilities
!!
!! These subroutines provide wrappers for lapack and blas routines
!! so that the latter are only called once in the code. This 
!! should facilitate porting.
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
!!****f* linalg/dlf_matrix_multiply
!!
!! FUNCTION
!!
!! C= alpha * AxB + beta * C
!!
!! SYNOPSIS
subroutine dlf_matrix_multiply(M,N,K,alpha,A,B,beta,C)
  use dlf_parameter_module, only: rk
  implicit none
  integer  ,intent(in)    :: M,N,K
  real(rk) ,intent(in)    :: alpha,beta
  real(rk) ,intent(in)    :: A(M,K) ! K is the common index
  real(rk) ,intent(in)    :: B(K,N)
  real(rk) ,intent(inout) :: C(M,N)
!! SOURCE
! **********************************************************************
  CALL dgemm('N','N',M,N,K,alpha, A , M, B, K, beta, C, M)
end subroutine dlf_matrix_multiply
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matrix_matrix_mult
!!
!! FUNCTION
!! calculates C=A*B (DOUBLES!)
!! also A^T and B^T possible (Determine transpose with 'T'
!! non-transposed with 'N')
!!
!! SYNOPSIS
subroutine dlf_matrix_matrix_mult(d,A,Atransp,B,Btransp,C)
  use dlf_parameter_module, only: rk
!! SOURCE
  integer, intent(in)   ::  d
  real(rk),intent(in)   ::  A(d,d)
  real(rk),intent(in)   ::  B(d,d)
  character*1,intent(in)::  Atransp
  character*1,intent(in)::  Btransp
  real(rk),intent(out)  ::  C(d,d)
  call DGEMM(Atransp,Btransp,d,d,d,1d0,A,d,B,d,0d0,C,d)
end subroutine dlf_matrix_matrix_mult
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matrix_vector_mult
!!
!! FUNCTION
!! calculates y=A*x (DOUBLES!)
!! also A->A^T possible via "transp='T'" instead of "transp='N'"
!!
!! SYNOPSIS
subroutine dlf_matrix_vector_mult(d,A,x,y,transp)
  use dlf_parameter_module, only: rk
!! SOURCE
  integer, intent(in)   ::  d
  real(rk),intent(in)   ::  A(d,d)
  real(rk),intent(in)   ::  x(d)
  real(rk),intent(inout)::  y(d)
  character*1,intent(in)::  transp
  call DGEMV(transp,d,d,1d0,A,d,x,1,0d0,y,1)
end subroutine dlf_matrix_vector_mult
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matrix_vector_multiply
!!
!! FUNCTION
!!
!! c= alpha * Axb + beta * c
!! a, b, c are vectors, A is a matrix
!!
!! SYNOPSIS
subroutine dlf_matrix_vector_multiply(M,N,alpha,A,b,beta,c)
  use dlf_parameter_module, only: rk
  implicit none
  integer  ,intent(in)    :: M,N
  real(rk) ,intent(in)    :: alpha,beta
  real(rk) ,intent(in)    :: A(M,N)
  real(rk) ,intent(in)    :: b(N)
  real(rk) ,intent(inout) :: c(M)
!! SOURCE
! **********************************************************************
  CALL dgemv ('N', M, N, alpha, A, M, b, 1, beta, c, 1)
  return
end subroutine dlf_matrix_vector_multiply
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matmatmul_simp
!!
!! FUNCTION
!!
!! C= AxB
!!
!! Simplified interface to dlf_matrix_multiply for plain matrix multiplication
!! whereas the latter additionally allows for scaling of AxB and addition of 
!! an initial matrix C_0
!!
!! SYNOPSIS
function dlf_matmatmul_simp(A,B) result(C)
!! SOURCE
  use dlf_parameter_module, only: rk
  implicit none
  interface
    subroutine dlf_matrix_multiply(M,N,K,alpha,A,B,beta,C)
      use dlf_parameter_module, only: rk
      implicit none
      integer  ,intent(in)    :: M,N,K
      real(rk) ,intent(in)    :: alpha,beta
      real(rk) ,intent(in)    :: A(M,K) ! K is the common index
      real(rk) ,intent(in)    :: B(K,N)
      real(rk) ,intent(inout) :: C(M,N)
    end subroutine dlf_matrix_multiply
  end interface
  real(rk) ,intent(in)    :: A(:,:) 
  real(rk) ,intent(in)    :: B(:,:)
  real(rk) :: C(size(A,dim=1),size(B,dim=2))
  
  integer :: M,N,K
  real(rk) ,parameter :: alpha=1._rk
  real(rk) ,parameter :: beta=0._rk
  
  M=size(A,dim=1)
  N=size(B,dim=2)
  K=size(A,dim=2)
  
  if (K/=size(B,dim=1)) then 
    write(*,*)'dlf_matmatmul_simp: size mismatch (B,dim=1 and A,dim=2).'
    call dlf_error()
  endif
  
  C(:,:)=0._rk
  
  call dlf_matrix_multiply(M,N,K,alpha,A,B,beta,C)
  return
end function dlf_matmatmul_simp

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matvecmul_simp
!!
!! FUNCTION
!!
!! c= Axb (A: matrix, b,c: vectors)
!!
!! Simplified interface to dlf_matrix_vector_multiply for plain matrix-vector 
!! multiplication whereas the latter additionally allows for scaling of Axb and 
!! addition of an initial vector c_0
!!
!! SYNOPSIS
function dlf_matvecmul_simp(A,b) result(c)
!! SOURCE
  use dlf_parameter_module, only: rk
  implicit none
  interface
    subroutine dlf_matrix_vector_multiply(M,N,alpha,A,b,beta,c)
      use dlf_parameter_module, only: rk
      implicit none
      integer  ,intent(in)    :: M,N
      real(rk) ,intent(in)    :: alpha,beta
      real(rk) ,intent(in)    :: A(M,N)
      real(rk) ,intent(in)    :: b(N)
      real(rk) ,intent(inout) :: c(M)
    end subroutine dlf_matrix_vector_multiply
  end interface
  real(rk) ,intent(in)    :: A(:,:) 
  real(rk) ,intent(in)    :: b(:)
  real(rk) :: c(size(A,dim=1))
  
  integer :: M,N
  real(rk) ,parameter :: alpha=1._rk
  real(rk) ,parameter :: beta=0._rk
  
  M=size(A,dim=1)
  N=size(A,dim=2)
  
  if (N/=size(b)) then 
    write(*,*)'dlf_matvecmul_simp: size mismatch (b).'
    call dlf_error()
  endif

  c(:)=0._rk
  
  call dlf_matrix_vector_multiply(M,N,alpha,A,b,beta,c)
  return
end function dlf_matvecmul_simp

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_dot_product
!!
!! FUNCTION
!!
!! dp= <x|y>
!!
!! SYNOPSIS
function dlf_dot_product(x,y) result(dp)
  use dlf_parameter_module, only: rk
  implicit none
  real(rk), intent(in)  :: x(:), y(:)
  real(rk) :: dp
  real(rk), external :: ddot
!! SOURCE
! **********************************************************************
  integer :: N
  N=min(size(x),size(y))
  dp = ddot (N, x(1:N), 1, y(1:N), 1)
  return
end function dlf_dot_product
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_outer_product
!!
!! FUNCTION
!!
!! Mat= |x><y|
!!
!! SYNOPSIS
function dlf_outer_product(x,y) result(Mat)
  use dlf_parameter_module, only: rk
  implicit none
  real(rk) ,intent(in)    :: x(:), y(:)
  real(rk) :: Mat(size(x),size(y))
!! SOURCE
! **********************************************************************
  interface
    subroutine dger (M, N, ALPHA, X, INCX, Y, INCY, A, LDA)
      integer,intent(in) :: M
      integer,intent(in) :: N
      double precision,intent(in) :: ALPHA
      double precision, dimension(*),intent(in) :: X
      integer,intent(in) :: INCX
      double precision, dimension(*),intent(in) :: Y
      integer,intent(in) :: INCY
      double precision, dimension(lda,*),intent(inout) :: A
      integer,intent(in) :: LDA
    end subroutine dger
  end interface
  real(rk), parameter :: alpha=1._rk
  integer :: M,N
  M=size(x)
  N=size(y)
  Mat(:,:)=0._rk
  call dger (M, N, alpha, x, 1, y, 1, Mat, M)
  return
end function dlf_outer_product
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_cross_product
!!
!! FUNCTION
!!
!! v3= v1 x v2 = |v1||v2|sin(theta[v1,v2])  (vector product/cross product)
!!
!! SYNOPSIS
function dlf_cross_product(v1,v2) result(v3)
  use dlf_parameter_module, only: rk
  implicit none
  real(rk), intent(in), dimension(3) :: v1,v2
  real(rk), dimension(3) :: v3
!! SOURCE
! **********************************************************************
  v3(1)=v1(2)*v2(3)-v1(3)*v2(2)
  v3(2)=v1(3)*v2(1)-v1(1)*v2(3)
  v3(3)=v1(1)*v2(2)-v1(2)*v2(1)
  return
end function dlf_cross_product
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_bilinear_form
!!
!! FUNCTION
!!
!! z = <x|H|y> = x^T H y
!!
!! SYNOPSIS
function dlf_bilinear_form(x,H,y) result(z)
  use dlf_parameter_module, only: rk
  implicit none
  real(rk),intent(in) :: x(:), y(:)
  real(rk),intent(in) :: H(size(x),size(y))
  real(rk) :: z
  interface
    function dlf_dot_product(x,y) result(dp)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk), intent(in)  :: x(:), y(:)
      real(rk) :: dp
    end function dlf_dot_product
  end interface
!! SOURCE
! **********************************************************************
  integer :: M,N
  real(rk) :: Hy(size(x))
  M=size(x)
  N=size(y)
  call dlf_matrix_vector_multiply(M,N,1._rk,H(1:M,1:N),y,0._rk,Hy)
  z=dlf_dot_product(x,Hy)
  return
end function dlf_bilinear_form
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_vector_norm
!!
!! FUNCTION
!!
!! z = ||x|| = sqrt(<x|x>) (Euclidean norm)
!!
!! SYNOPSIS
function dlf_vector_norm(x) result(z)
  use dlf_parameter_module, only: rk
  implicit none
  real(rk),intent(in) :: x(:)
  real(rk) :: z
!! SOURCE
! **********************************************************************
  real(rk), external :: dnrm2
  integer :: N
  N=size(x)
  z = dnrm2 (N, x, 1)
  return
end function dlf_vector_norm
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matrix_ortho_trans
!!
!! FUNCTION
!!
!! C= A^T x B x A   (for mode==0) or
!! C= A   x B x A^T (for mode/=0)
!!
!! with A of dimensions (MXK)...
!!  * B must be quadratic and have dimensions
!!   (MxM) (mode==0)    or    (KxK) (mode/=0)
!!  * C must be quadratic and have dimensions
!!   (KxK) (mode==0)    or    (MxM) (mode/=0)
!!
!! SYNOPSIS
function dlf_matrix_ortho_trans(A,B,mode) result(C)
!! SOURCE
  use dlf_parameter_module, only: rk
  implicit none
  real(rk), intent(in) :: A(:,:), B(:,:)
  integer, intent(in)  :: mode
  real(rk) :: C( merge(1,0,mode/=0)*size(A,dim=1) + merge(1,0,mode==0)*size(A,dim=2) , &
               & merge(1,0,mode/=0)*size(A,dim=1) + merge(1,0,mode==0)*size(A,dim=2) )

  real(rk), parameter :: alpha=1._rk
  real(rk), parameter :: beta=0._rk
  
  integer :: M, K, icheck1, icheck2, icheck3, icheck4
  real(rk) :: Ctemp(size(A,dim=2),size(A,dim=1))
  
  M=size(A,dim=1)
  K=size(A,dim=2)
  icheck1=size(B,dim=1)
  icheck2=size(B,dim=2)
  icheck3=size(C,dim=1)
  icheck4=size(C,dim=2)
  if (mode==0) then
    if (icheck1 /= M .or. icheck2 /= M .or. &
        & icheck3 /= K .or. icheck4 /= K) then
      write(*,'(A)') 'dlf_matrix_ortho_trans: size mismatch!'
      call dlf_error()
    endif
  else
    if (icheck1 /= K .or. icheck2 /= K .or. &
        & icheck3 /= M .or. icheck4 /= M) then
      write(*,'(A)') 'dlf_matrix_ortho_trans: size mismatch!'
      call dlf_error()
    endif
  endif
  
  Ctemp(:,:)=0._rk
  C(:,:)=0._rk
  
  if (mode==0) then
    ! get A^T x B (-> save in Ctemp)
    CALL dgemm('T','N', K, M, M, alpha, A, M, B, M, beta, Ctemp, K)
    ! get C = (A^T x B) x A = Ctemp x A
    CALL dgemm('N','N', K, K, M, alpha, Ctemp, K, A, M, beta, C, K)
  else
    ! get B x A^T (-> save in Ctemp)
    CALL dgemm('N', 'T', K, M, K, alpha, B, K, A, M, beta, Ctemp, K)
    ! get C = A x (BxA^T) = A x Ctemp
    CALL dgemm('N', 'N', M, M, K, alpha, A, M, Ctemp, K, beta, C, M)
  endif
  return
end function dlf_matrix_ortho_trans
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_unit_mat
!!
!! FUNCTION
!!
!! Returns (real) unit/identity matrix of size NxN
!!
!! SYNOPSIS
function dlf_unit_mat(N) result(U)
!! SOURCE
  use dlf_parameter_module, only: rk
  implicit none
  integer, intent(in)  :: N
  real(rk) :: U(N,N)
  integer :: i
  
  U(:,:)=0._rk
  do i=1,N
    U(i,i)=1._rk
  enddo
  return
end function dlf_unit_mat
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_trace
!!
!! FUNCTION
!!
!! Returns trace tr(M) of quadratic matrix M
!!
!! SYNOPSIS
function dlf_trace(M) result(tr)
!! SOURCE
  use dlf_parameter_module, only: rk
  implicit none
  real(rk), intent(in)  :: M(:,:)
  real(rk) :: tr
  integer :: i,N
  N=min(size(M,dim=1),size(M,dim=2))
  tr=0._rk
  do i=1,N
    tr=tr+M(i,i)
  enddo
  return
end function dlf_trace
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matrix_invert
!!
!! FUNCTION
!!
!! A <- A^-1
!!
!! SYNOPSIS
subroutine dlf_matrix_invert(N,tdet,a,det)
  use dlf_parameter_module, only: rk
  use dlfhdlc_matrixlib, only: array_invert 
  implicit none
  integer  ,intent(in)    :: N
  logical  ,intent(in)    :: tdet
  real(rk) ,intent(inout) :: A(N,N)
  real(rk) ,intent(out)   :: det
!! SOURCE
  integer :: idum
! **********************************************************************
  idum = array_invert(A,det,tdet,N)
end subroutine dlf_matrix_invert
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matrix_determinant
!!
!! FUNCTION
!!
!! computes the determinat of a real square matrix A by LU LU factorization
!!
!! SYNOPSIS
subroutine dlf_matrix_determinant(N,a,det)
!! SOURCE
  use dlf_parameter_module, only: rk
  implicit none
  integer  ,intent(in)    :: N
  real(rk) ,intent(in)    :: A(N,N)
  real(rk) ,intent(out)   :: det
  real(rk) :: a_(N,N)
  integer :: info,ipiv(N),i
! **********************************************************************
  a_=a
  call dgetrf (N, N, A_, N, IPIV, INFO)
  if(info/=0) then
    call dlf_fail("LU factorization in dlf_matrix_determinant failed")
  end if
  det = 1.d0
  do i=1,n
    if (ipiv(i)/=i) then
      det = -det * a_(i,i)
    else
      det = det * a_(i,i)
    endif
  end do
  
end subroutine dlf_matrix_determinant
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matrix_diagonalise
!!
!! FUNCTION
!!
!! Calculate eigenvalues and eigenvectors of a real, symmetric matrix A
!!
!! The eigenvector to the eigenvalue evals(ivar) is evecs(:,ivar)
!!
!! SYNOPSIS
subroutine dlf_matrix_diagonalise(N,a,evals,evecs)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlfhdlc_matrixlib, only: array_diagonalise
  implicit none
  integer  ,intent(in)    :: N
  real(rk) ,intent(in)    :: A(N,N)
  real(rk) ,intent(out)   :: evals(N)
  real(rk) ,intent(out)   :: evecs(N,N)
  integer :: idum
! **********************************************************************
  idum = array_diagonalise(a,evecs,evals,n,n,n,.true.)
end subroutine dlf_matrix_diagonalise
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matrix_print
!!
!! FUNCTION
!!
!! Print out a matrix 
!!
!! M = number of rows
!! N = number of columns
!! a = M x N array
!!
!! SYNOPSIS
subroutine dlf_matrix_print(M,N,a)
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout
  implicit none
  integer  ,intent(in)    :: M,N 
  real(rk) ,intent(in)    :: a(M,N)
!! SOURCE
! **********************************************************************
  integer :: imin, imax, max, i, j
  logical :: highprec

  highprec = .false.

  if (highprec) then
     max = 7
  else
     max = 12
  end if
  imax = 0

  do while(imax < N)
     imin = imax + 1
     imax = imax + max
     if (imax > N) imax = N
     write(stdout, *)
     if (highprec) then
        write(stdout, '(6X,7(6X,I3,6X))') (i,i=imin,imax)
     else
        write(stdout, '(6X,12(3X,I3,3X))') (i,i=imin,imax)
     end if
     write(stdout, *)
     do j = 1, M
        if (highprec) then
           write(stdout, '(I5,1X,7F15.10)') j, (a(j,i),i=imin,imax)
        else
           write(stdout, '(I5,1X,12F9.5)') j, (a(j,i),i=imin,imax)
        end if
     end do
  end do
end subroutine dlf_matrix_print
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matrix_svd
!!
!! FUNCTION
!!
!! Perform a singular value decomposition of a (possibly asymmetric) real matrix A
!!
!! SYNOPSIS
subroutine dlf_matrix_svd(N,M,a,evals)
  use dlf_parameter_module, only: rk
  implicit none
  integer  ,intent(in)    :: N
  integer  ,intent(in)    :: M
  real(rk) ,intent(in)    :: A(M,N)
  real(rk) ,intent(out)   :: evals(N)
  real(rk) ,allocatable  :: singval(:),work(:)
  integer :: iwork(8*min(M,N)),info,lwork
  real(rk) :: uvt(1)
!! SOURCE
  allocate(singval(min(M,N)))
  ! query for lwork
  call DGESDD('N',M,N,A,M,singval,uvt,1,uvt,1,singval,-1,iwork,info)
  lwork=nint(singval(1))
  if(info/=0) then
     write(*,*) "Error DGESDD, info=",info
     call dlf_fail("Error in singular value decomposition")
  end if
  allocate(work(lwork))
  
  call DGESDD('N',M,N,A,M,singval,uvt,1,uvt,1,work,lwork,iwork,info)
  if(info/=0) then
     write(*,*) "Error DGESDD, info=",info
     call dlf_fail("Error in singular value decomposition")
  end if
  evals=0.D0
  evals(1:min(M,N))=singval ! here, we could cite in the other order
  deallocate(singval)
  deallocate(work)
end subroutine dlf_matrix_svd
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* linalg/dlf_matrix_diagonalise_general
!!
!! FUNCTION
!!
!! Calculate eigenvalues and eigenvectors of a real, general matrix A
!!
!! The eigenvector to the complex eigenvalue eignum_r(ivar),eignum_i(ivar) is
!! eigvectr(:,ivar)
!!
!! SYNOPSIS
subroutine dlf_matrix_diagonalise_general(nm,a,eignum_r,eignum_i,eigvectr)
!! SOURCE
  use dlf_parameter_module, only: rk
  implicit none
  integer, intent(in)   :: nm
  real(rk), intent(in)  :: a(nm,nm)
  real(rk), intent(out) :: eigvectr(nm,nm),eignum_r(nm),eignum_i(nm) 
  integer :: ilo,ihi,info,ldvl,lwork,RWORK(2*nm-2)
  real(rk) :: a_str(nm,nm)
  real(rk) :: SCALES(nm,nm),RCONDE(nm),RCONDV(nm),ABRNM
  character(1) :: JOBVL,JOBVR,SENSE
  real(rk), dimension(:),allocatable:: WORK
  real(rk), dimension(:,:), allocatable:: vl
  
  a_str=a
  
  LDVL=1
  allocate(vl(ldvl,ldvl))
  allocate(WORK(1))
  SENSE='V'
  JOBVL='N'
  JOBVR='V'
  LWORK=-1
  
  CALL DGEEVX('S', JOBVL, JOBVR, SENSE, nm, A_STR, nm, eignum_r,&
      eignum_i, vl,LDVL,eigvectr,nm, ILO, IHI, SCALES, ABRNM, RCONDE,& 
      RCONDV, WORK, LWORK,RWORK, INFO )
  if(INFO.ne.0)then
    !        write(*,*)'INFO NON-ZERO',INFO
    call dlf_fail("INFO NON-ZERO in r_diagonal_general")
  endif
  
  LWORK=nint(WORK(1))
  deallocate(WORK)
  allocate(WORK(LWORK))
  
  CALL DGEEVX('S', JOBVL, JOBVR, SENSE, nm, A_STR, nm, eignum_r,&
      eignum_i, vl,LDVL,eigvectr,nm, ILO, IHI, SCALES, ABRNM, RCONDE,& 
      RCONDV, WORK, LWORK, RWORK, INFO )
  if(INFO.ne.0)then
    !        write(*,*)'r_diagonal_general: INFO NON-ZERO',INFO
    call dlf_fail("INFO NON-ZERO in r_diagonal_general")
  endif
  deallocate(WORK)
  deallocate(vl)
end subroutine dlf_matrix_diagonalise_general
!!****
  
      ! is used in dlf_pes.f90
      subroutine SVD(m,n,K,a,S,U,VT)
      implicit integer(i-n)
      implicit real(8)(a-h,o-z)
      integer n,m,info,lwork,K
      real(8) a(m,n),S(min(m,n)),U(m,K),VT(K,n)
      character*1 JOBU,JOBVT
      real(8), dimension(:),allocatable:: WORK

      allocate(WORK(1))
      lwork=-1
      JOBU='S'
      JOBVT='S'

      CALL dgesvd( JOBU, JOBVT, M, N, A, M, S, U, M,&
      vt, K, work, lwork, info )

      lwork=nint(WORK(1))
      deallocate(work)
      allocate(work(lwork))

      CALL dgesvd( JOBU, JOBVT, M, N, A, M, S, U, M,&
      vt, K, work, lwork, info )

      deallocate(work)

      return
      end subroutine SVD

      ! is used in dlf_pes.f90
      subroutine Pseudoinverse(m,n,a,pinv_a,pinv_tol,residue_opt)
      implicit none
      !M is usually 3*nat, N is usually IA_DOF
      integer n,m,i,K!T=TIKHONOV, M=MORE-PENROSE
      character*1 residue_opt
      real(8) a(m,n),pinv_a(n,m),a_in(m,n),pinv_tol
      real(8),allocatable :: u(:,:),s(:),vt(:,:),s_store(:,:)

      a_in=a
      K = MIN(M,N)

      allocate(u(m,K))
      allocate(s(K))
      allocate(vt(K,n))

      call svd(m,n,K,a,s,u,vt)
      allocate(s_store(k,k))
      s_store=0.d0

      if(residue_opt.eq.'T')then
        do i=1,k!m
!          if(dabs(s(i)).gt.pinv_tol)then
!            s_store(i,i)=1.d0/s(i)
!          endif
          s_store(i,i)=s(i)/(s(i)**2+pinv_tol**2)
        enddo
      elseif(residue_opt.eq.'M')then
        do i=1,k!m
          if(dabs(s(i)).gt.pinv_tol)then
            s_store(i,i)=1.d0/s(i)
          else
            s_store(i,i)=1.d0/(maxval(abs(s))*dble(k))
          endif
        enddo
      endif
      deallocate(s)

      pinv_a=matmul(transpose(vt),matmul(s_store,transpose(u)))
      deallocate(s_store)
      deallocate(u)
      deallocate(vt)

      a=a_in

      return
      end subroutine Pseudoinverse



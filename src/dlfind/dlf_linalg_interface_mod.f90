!!****h* utilities/linalg
!!
!! Module with explicit interface definitions for the linear algebra 
!! functions in dlf_linalg.f90.
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

module dlf_linalg_interface_mod
use dlf_parameter_module, only: rk
implicit none

  interface dlf_matmul_simp
    function dlf_matmatmul_simp(A,B) result(C)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk) ,intent(in)    :: A(:,:) 
      real(rk) ,intent(in)    :: B(:,:)
      real(rk) :: C(size(A,dim=1),size(B,dim=2))
    end function dlf_matmatmul_simp
    function dlf_matvecmul_simp(A,b) result(c)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk) ,intent(in)    :: A(:,:) 
      real(rk) ,intent(in)    :: b(:)
      real(rk) :: c(size(A,dim=1))
    end function dlf_matvecmul_simp
  end interface dlf_matmul_simp

  interface
    function dlf_dot_product(x,y) result(dp)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk), intent(in)  :: x(:), y(:)
      real(rk) :: dp
    end function dlf_dot_product
  end interface
  
  interface
    function dlf_outer_product(x,y) result(Mat)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk) ,intent(in)    :: x(:), y(:)
      real(rk) :: Mat(size(x),size(y))
    end function dlf_outer_product
  end interface
  
  interface
    function dlf_cross_product(v1,v2) result(v3)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk), intent(in), dimension(3) :: v1,v2
      real(rk), dimension(3) :: v3
    end function dlf_cross_product
  end interface
  
  interface
    function dlf_unit_mat(N) result(U)
      use dlf_parameter_module, only: rk
      implicit none
      integer, intent(in)  :: N
      real(rk) :: U(N,N)
    end function dlf_unit_mat
  end interface

  interface
    function dlf_trace(M) result(tr)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk), intent(in)  :: M(:,:)
      real(rk) :: tr
    end function dlf_trace
  end interface
  
  interface
    function dlf_bilinear_form(x,H,y) result(z)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk),intent(in) :: x(:), y(:)
      real(rk),intent(in) :: H(size(x),size(y))
      real(rk) :: z
    end function dlf_bilinear_form
  end interface
  
  interface
    function dlf_vector_norm(x) result(z)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk),intent(in) :: x(:)
      real(rk) :: z
    end function dlf_vector_norm
  end interface
  
  interface
    function dlf_matrix_ortho_trans(A,B,mode) result(C)
      use dlf_parameter_module, only: rk
      implicit none
      real(rk), intent(in) :: A(:,:), B(:,:)
      integer, intent(in)  :: mode
      real(rk) :: C( merge(1,0,mode/=0)*size(A,dim=1) + merge(1,0,mode==0)*size(A,dim=2) , &
                   & merge(1,0,mode/=0)*size(A,dim=1) + merge(1,0,mode==0)*size(A,dim=2) )
    end function dlf_matrix_ortho_trans
  end interface

end module dlf_linalg_interface_mod

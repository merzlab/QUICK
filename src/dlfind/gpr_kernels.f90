module gpr_kernels_module
    use dlf_parameter_module, only: rk
    use gpr_types_module, only: gpr_type
    implicit none
    
    public  :: kernel, kernel_d1, kernel_d2, kernel_d3, kernel_d4,&
            kernel_dg, kernel_d1_dg, kernel_d2_dg, kernel_d3_dg, kernel_d4_dg,&
            kernel_d1_exp1, kernel_d1_exp2, kernel_d2_exp12, &
            kernel_d2_exp22, kernel_d2_exp11, kernel_matern_d2_exp12_matrix
    contains
            
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel
!!
!! FUNCTION
!! * evaluation of the covariance function
!!
!! SYNOPSIS
real(rk) function kernel(this,xm, xn)    
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  xm(this%sdgf), xn(this%sdgf)
    real(rk)            ::  absv
SELECT CASE (this%kernel_type)
  CASE(0)
    ! Squared exponential
    kernel = this%s_f2*dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
  CASE(1)
    ! Matern with nu=5d0/2d0
    absv = dsqrt(dot_product(xm(:)-xn(:),xm(:)-xn(:)))
    kernel = (1d0 + (dsqrt(5d0)*absv)/this%l + &
            (5d0*absv**2)/(3d0*this%l**2))*&
            dexp(-1d0*(dsqrt(5d0)*absv/this%l))
    kernel = this%s_f2*kernel
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (function kernel)")
END SELECT
end function kernel
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d1
!!
!! FUNCTION
!! * Derivative of the covariance functino wrt. x oder xn
!!
!! SYNOPSIS
real(rk) function kernel_d1(this, xm, xn, dwrt, i)
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  xm(this%sdgf), xn(this%sdgf)
    integer, intent(in) ::  i    
    integer, intent(in) ::  dwrt ! 1 for der. wrt. xm; 2 for der. wrt. xn
    real(rk)            ::  absv    
SELECT CASE (this%kernel_type)
  CASE(0)
    ! Squared exponential     
    if (dwrt==1) then
        kernel_d1 =    this%s_f2*this%gamma*(xn(i)-xm(i))*&
                        dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
    else if (dwrt==2) then
        kernel_d1 =    this%s_f2*this%gamma*(xm(i)-xn(i))*&
                        dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
    end if
  CASE(1)
    ! Matern with nu=5d0/2d0
    absv = dsqrt(dot_product(xm(:)-xn(:),xm(:)-xn(:)))
    if (dwrt==1) then
        kernel_d1 = (-5d0*(xm(i) - xn(i))*(this%l + dsqrt(5d0)*absv))/&
            (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**3)
    else if (dwrt==2) then
        kernel_d1 = (5d0*(xm(i) - xn(i))*(this%l + dsqrt(5d0)*absv))/&
            (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**3)
    end if
    kernel_d1 = this%s_f2*kernel_d1
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (function kernel_d1)")
END SELECT 
end function kernel_d1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d1_exp1
!!
!! FUNCTION
!! * Derivative of the covariance functino wrt. x oder xn (only first parameter,
!! faster version)
!!
!! SYNOPSIS
real(rk) function kernel_d1_exp1(this, diff_vec, i, absv)
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  diff_vec(this%sdgf)
    integer, intent(in) ::  i    
!     integer, intent(in) ::  dwrt ! 1 for der. wrt. xm; 2 for der. wrt. xn
    real(rk),intent(in)            ::  absv
    
SELECT CASE (this%kernel_type)
  CASE(0)
    ! Squared exponential    
        kernel_d1_exp1 =    this%s_f2*this%gamma*(-diff_vec(i))*&
                        dexp(-this%gamma/2d0*absv**2)
  CASE(1)
    ! Matern with nu=5d0/2d0
        kernel_d1_exp1 = (-5d0*(diff_vec(i))*(this%l + dsqrt(5d0)*absv))/&
            (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**3)
    kernel_d1_exp1 = this%s_f2*kernel_d1_exp1
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (function kernel_d1_exp1)")
END SELECT 
end function kernel_d1_exp1
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d1_exp2
!!
!! FUNCTION
!! * Derivative of the covariance functino wrt. x oder xn (only second parameter,
!!   faster version)
!!
!! SYNOPSIS
real(rk) function kernel_d1_exp2(this, diff_vec, i, absv)
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  diff_vec(this%sdgf)
    integer, intent(in) ::  i    
!     integer, intent(in) ::  dwrt ! 1 for der. wrt. xm; 2 for der. wrt. xn
    real(rk),intent(in)            ::  absv
    
SELECT CASE (this%kernel_type)
  CASE(0)
    ! Squared exponential     
        kernel_d1_exp2 =    this%s_f2*this%gamma*(diff_vec(i))*&
                        dexp(-this%gamma/2d0*absv**2)
  CASE(1)
    ! Matern with nu=5d0/2d0
!     absv = dsqrt(dot_product(xm(:)-xn(:),xm(:)-xn(:)))
!     if (absv<1d-16) absv = 1d-16
      if(absv/this%l>50d0) then
        kernel_d1_exp2 = 0d0
      else
        kernel_d1_exp2 = (5d0*(diff_vec(i))*(this%l + dsqrt(5d0)*absv))/&
            (3d0*this%l**3)*dexp(-1d0*(dsqrt(5d0)*absv)/this%l)
      endif
    kernel_d1_exp2 = this%s_f2*kernel_d1_exp2
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (function kernel_d1_exp2)")
END SELECT 
end function kernel_d1_exp2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d2
!!
!! FUNCTION
!! * Second derivatives of the covariance functions
!!
!! SYNOPSIS
real(rk) function kernel_d2(this, xm, xn, dwrt, i, j)
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  xm(this%sdgf), xn(this%sdgf)
    integer, intent(in) ::  dwrt(2) ! Derivative with respect to which (xm/xn)
    integer ::  i, j   
    real(rk)            ::  absv
SELECT CASE (this%kernel_type)
  CASE(0)
    ! Squared exponential       
    if (((dwrt(1)/=1).and.(dwrt(1)/=2)).or.((dwrt(2)/=1).and.(dwrt(2)/=2))) then
        call dlf_fail("Error: wrong derivative information (kernel_d2)")
    else if (dwrt(1)/=dwrt(2)) then
        kernel_d2 = this%gamma*(xm(j)-xn(j))*(xn(i)-xm(i))                       
        if (i==j) kernel_d2 = kernel_d2 + 1d0
        kernel_d2 = kernel_d2*this%s_f2*this%gamma * &
                        dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
    else if (dwrt(1)==1) then ! both are equal to 1
        kernel_d2 = this%gamma*(xn(j)-xm(j))*(xn(i)-xm(i))
        if (i==j) kernel_d2 = kernel_d2-1d0
        kernel_d2 = kernel_d2*this%s_f2*this%gamma*&
                        dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
    else ! both are equal to 2
        kernel_d2 = this%gamma*(xm(j)-xn(j))*(xm(i)-xn(i))
        if (i==j) kernel_d2 = kernel_d2-1d0        
        kernel_d2 = kernel_d2*this%s_f2*this%gamma*&
                        dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
    end if
  CASE(1)
    ! Matern with nu=5d0/2d0
    absv = dsqrt(dot_product(xm(:)-xn(:),xm(:)-xn(:)))
    if (((dwrt(1)/=1).and.(dwrt(1)/=2)).or.((dwrt(2)/=1).and.(dwrt(2)/=2))) then
        call dlf_fail("Error: wrong derivative information (kernel_d2)")
    else if (dwrt(1)/=dwrt(2)) then
        if (i==j) then
            kernel_d2 = (5d0*(this%l**2 - 5d0*(xm(i) - xn(i))**2 + &
                dsqrt(5d0)*this%l*absv))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
        else
            kernel_d2 = (-25d0*(xm(i) - xn(i))*(xm(j) - xn(j)))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
        end if
    else 
        if (i/=j) then
            kernel_d2 = (25d0*(xm(i) - xn(i))*(xm(j) - xn(j)))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
        else
            kernel_d2 = (-5d0*(this%l**2 - 5d0*(xm(i) - xn(i))**2 + &
                dsqrt(5d0)*this%l*absv))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
        end if
    end if
    kernel_d2 = this%s_f2*kernel_d2
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (function kernel_d2)")
END SELECT 
end function kernel_d2
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d2_exp11
!!
!! FUNCTION
!! * Second derivatives Kernel (derivative 1. then 1. variable, faster version)
!!
!! SYNOPSIS
real(rk) function kernel_d2_exp11(this, diff_vec, i, j, absv)
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  diff_vec(this%sdgf)
!     integer, intent(in) ::  dwrt(2) ! Derivative with respect to which (xm/xn)
    integer ::  i, j   
    real(rk),intent(in)  ::  absv    
SELECT CASE (this%kernel_type)
  CASE(0)
    ! Squared exponential      
        kernel_d2_exp11 = this%gamma*(diff_vec(j))*(diff_vec(i))
        if (i==j) kernel_d2_exp11 = kernel_d2_exp11-1d0
        kernel_d2_exp11 = kernel_d2_exp11*this%s_f2*this%gamma*&
        dexp(-this%gamma/2d0*absv**2)
  CASE(1)
    ! Matern with nu=5d0/2d0
        if (i/=j) then
            kernel_d2_exp11 = (25d0*(diff_vec(i))*(diff_vec(j)))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
        else
            kernel_d2_exp11 = (-5d0*(this%l**2 - 5d0*(diff_vec(i))**2 + &
                dsqrt(5d0)*this%l*absv))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
        end if
    kernel_d2_exp11 = this%s_f2*kernel_d2_exp11
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (function kernel_d2_exp11)")
END SELECT 
end function kernel_d2_exp11

subroutine kernel_matern_d2_exp12_matrix(this,diff_vec,absv,matrix)
  implicit none
  type(gpr_type),intent(in)::this
  real(rk), intent(in) ::  diff_vec(this%sdgf)
  real(rk),intent(in)  ::  absv
  real(rk),intent(out) :: matrix(this%sdgf,this%sdgf)
  integer :: ival,jval
  real(rk) :: factor
  factor=1.D0/(3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l4)
  do ival=1,this%sdgf
     do jval=ival+1,this%sdgf
        matrix(ival,jval)= -25d0*diff_vec(ival)*diff_vec(jval)*factor
        matrix(jval,ival)=matrix(ival,jval)
     end do
  end do
  do ival=1,this%sdgf
     matrix(ival,ival)=(5d0*(this%l2 - 5d0*(diff_vec(ival))**2 + &
                dsqrt(5d0)*this%l*absv))*factor
  end do
  matrix=matrix*this%s_f2
end subroutine kernel_matern_d2_exp12_matrix

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d2_exp12
!!
!! FUNCTION
!! * Second derivatives Kernel (derivative 1. then 2. variable, faster version)
!!
!! SYNOPSIS
real(rk) function kernel_d2_exp12(this, diff_vec, i, j, absv)
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  diff_vec(this%sdgf)
!     integer, intent(in) ::  dwrt(2) ! Derivative with respect to which (xm/xn)
    integer ::  i, j   
    real(rk),intent(in)  ::  absv
    
SELECT CASE (this%kernel_type)
  CASE(0)
    ! Squared exponential      
        kernel_d2_exp12 = this%gamma*(diff_vec(j))*(-diff_vec(i))                       
        if (i==j) kernel_d2_exp12 = kernel_d2_exp12 + 1d0
        kernel_d2_exp12 = kernel_d2_exp12*this%s_f2*this%gamma * &
                        dexp(-this%gamma/2d0*absv**2)
  CASE(1)
    ! Matern with nu=5d0/2d0
        if (i==j) then
            kernel_d2_exp12 = (5d0*(this%l2 - 5d0*(diff_vec(i))**2 + &
                dsqrt(5d0)*this%l*absv))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l4)
        else
            kernel_d2_exp12 = (-25d0*(diff_vec(i))*(diff_vec(j)))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l4)
        end if
    kernel_d2_exp12 = this%s_f2*kernel_d2_exp12
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (function kernel_d2_exp12)")
END SELECT 
end function kernel_d2_exp12

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d2_exp12
!!
!! FUNCTION
!! * Second derivatives Kernel (derivative 2. then 2. variable, faster version)
!!
!! SYNOPSIS
real(rk) function kernel_d2_exp22(this, diff_vec, i, j, absv)
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  diff_vec(this%sdgf)
    integer ::  i, j   
    real(rk),intent(in)  ::  absv
    
SELECT CASE (this%kernel_type)
  CASE(0)
        kernel_d2_exp22 = this%gamma*(diff_vec(j))*(diff_vec(i))
        if (i==j) kernel_d2_exp22 = kernel_d2_exp22-1d0        
        kernel_d2_exp22 = kernel_d2_exp22*this%s_f2*this%gamma*&
                        dexp(-this%gamma/2d0*absv**2)
  CASE(1)
        if (i/=j) then
            kernel_d2_exp22 = (25d0*(diff_vec(i))*(diff_vec(j)))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
        else
            kernel_d2_exp22 = (-5d0*(this%l**2 - 5d0*(diff_vec(i))**2 + &
                dsqrt(5d0)*this%l*absv))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
        end if
    kernel_d2_exp22 = this%s_f2*kernel_d2_exp22
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (function kernel_d2_exp22)")
END SELECT 
end function kernel_d2_exp22
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d3
!!
!! FUNCTION
!! Third derivatives Kernel
!! Be careful with the order of the derivatives!
!! The derivative variables (e.g. dx(i)) always have 
!! the dimensional index in alphabetically ascending order, i.e. i,j,k
!! The order of derivatives (first d/dx_m or d/dx_n) is then relevant!
!! This function is able to calculate d/x_m(i) d/x_m(j) d/x_n(k) -> dwrt(1,1,2)
!! and d/x_n(i) d/x_n(j) d/x_m(k) -> dwrt(2,2,1)
!! all others can be wrongly calculated but are not relevant for the 
!! covariance matrix code I use.
!!
!! SYNOPSIS
real(rk) function kernel_d3(this, xm, xn, dwrt, i, j, k)
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  xm(this%sdgf), xn(this%sdgf)
    integer, intent(in) ::  dwrt(3) ! Derivative with respect to which (xm/xn)
    integer, intent(in) ::  i, j, k  
    real(rk)            ::  absv   
!     logical             ::  small = .false.
SELECT CASE (this%kernel_type)
  CASE(0)
    ! Squared exponential   
    if (((dwrt(1)/=1).and.(dwrt(1)/=2)).or.((dwrt(2)/=1).and.(dwrt(2)/=2))&
        .or.((dwrt(3)/=1).and.(dwrt(3)/=2))) then
        call dlf_fail("Error: wrong derivative information (kernel_d3)")
    else if ((dwrt(1)==1).and.(dwrt(2)==1).and.(dwrt(3)==2)) then
        kernel_d3 = this%gamma*(xn(j)-xm(j))*(xn(i)-xm(i))*&
                                        (xm(k)-xn(k))
        if (i==j) kernel_d3 = kernel_d3-(xm(k)-xn(k))
        if (j==k) kernel_d3 = kernel_d3+(xn(i)-xm(i))
        if (i==k) kernel_d3 = kernel_d3+(xn(j)-xm(j))
        kernel_d3 = kernel_d3*this%s_f2*this%gamma**2*&
                    dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
    else if ((dwrt(1)==2).and.(dwrt(2)==2).and.(dwrt(3)==1)) then
        kernel_d3 = this%gamma*(xm(j)-xn(j))*(xm(i)-xn(i))*&
                                        (xn(k)-xm(k))
        if (i==j) kernel_d3 = kernel_d3-(xn(k)-xm(k))
        if (j==k) kernel_d3 = kernel_d3+(xm(i)-xn(i))
        if (i==k) kernel_d3 = kernel_d3+(xm(j)-xn(j))        
        kernel_d3 = kernel_d3*this%s_f2*this%gamma**2*&
                    dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))       
    else
        call dlf_fail("Error: wrong derivative information (kernel_d3)")
    end if 
  CASE(1)
    ! Matern with nu=5d0/2d0
    absv = dsqrt(dot_product(xm(:)-xn(:),xm(:)-xn(:)))
! This must be done to avoid division by 0
    if (absv<1d-16) then
!       small = .false.  
      absv = 1d-16 
    end if
    if (((dwrt(1)/=1).and.(dwrt(1)/=2)).or.((dwrt(2)/=1).and.(dwrt(2)/=2))&
        .or.((dwrt(3)/=1).and.(dwrt(3)/=2))) then
        call dlf_fail("Error: wrong derivative information (kernel_d3)")
    else if ((dwrt(1)==1).and.(dwrt(2)==1).and.(dwrt(3)==2)) then
        if (i==j) then
            if (j==k) then
                ! iii
                kernel_d3=(25d0*(xm(i) - xn(i))*&
                    (dsqrt(5d0)*(xm(i) - xn(i))**2 - &
                    3d0*this%l*absv))/&
                    (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5d0*absv)
                ! lim->0: (0)
            else
                ! iik
                kernel_d3=(25d0*(-(absv*this%l) + &
                    dsqrt(5d0)*(xm(i) - xn(i))**2)*(xm(k) - xn(k)))/&
                    (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5)
                ! lim->0: (0)
            end if
        else if (i==k) then
                ! iji
                kernel_d3=(25d0*(-(absv*this%l) + &
                    dsqrt(5d0)*(xm(i) - xn(i))**2)*(xm(j) - xn(j)))/&
                    (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5)
                ! lim->0: (0)
        else
            if (j==k) then
                ! ijj
                kernel_d3=(25d0*(xm(i) - xn(i))*(-(absv*this%l) +&
                    dsqrt(5d0)*(xm(j) - xn(j))**2))/&
                    (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5)
                ! lim->0: (0)
            else
                ! ijk
                kernel_d3=(25d0*dsqrt(5d0)*(xm(i) - xn(i))*&
                    (xm(j) - xn(j))*(xm(k) - xn(k)))/&
                    (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5)
                ! lim->0: (0)
            end if
        end if
    else if ((dwrt(1)==2).and.(dwrt(2)==2).and.(dwrt(3)==1)) then
        if (i==j) then
            if (j==k) then
                ! iii
                kernel_d3 =(-25d0*(-3d0*absv*this%l + &
                    dsqrt(5d0)*(xm(i) - xn(i))**2)*(xm(i) - xn(i)))/&
                    (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5)
                ! lim->0: (0)
            else
                ! iik
                kernel_d3=(25d0*(absv*this%l - &
                    dsqrt(5d0)*(xm(i) - xn(i))**2)*(xm(k) - xn(k)))/&
                    (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5)
                ! lim->0: (0)
            end if
        else if (i==k) then
                ! iji
                kernel_d3=(25d0*(absv*this%l - &
                    dsqrt(5d0)*(xm(i) - xn(i))**2)*(xm(j) - xn(j)))/&
                    (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5)
                ! lim->0: (0)
        else
            if (j==k) then
                ! ijj
                kernel_d3=(25d0*(xm(i) - xn(i))*(absv*this%l - &
                    dsqrt(5d0)*(xm(j) - xn(j))**2))/&
                    (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5)
                ! lim->0: (0)
            else
                ! ijk
                kernel_d3=(-25d0*dsqrt(5d0)*(xm(i) - xn(i))*&
                    (xm(j) - xn(j))*(xm(k) - xn(k)))/&
                    (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5)
                ! lim->0: (0)
            end if
        end if
    else
        call dlf_fail("The third derivative wrt this is not implemented.")
    end if
    kernel_d3 = this%s_f2*kernel_d3
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (function kernel_d3)")
END SELECT     
end function kernel_d3
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d4
!!
!! FUNCTION
!! Fourth derivatives (only 2 times wrt. x_n, and 2 times wrt. x_m) and
!! dimensional index in alphabetically ascending order, i.e. i,j,k
!! meaning exactly d/x_n(i) d/x_n(j) d/x_m(k) d/x_m(l)
!! All calculations differing from that will probably be wrong
!!
!! SYNOPSIS
real(rk) function kernel_d4(this, xm, xn, dwrt, i, j, k, l)    
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  xm(this%sdgf), xn(this%sdgf)
    integer, intent(in) ::  dwrt(4) ! Derivative with respect to which (xm/xn)
    integer, intent(in) ::  i, j, k, l
    real (rk)           ::  absv,absv2
    
SELECT CASE (this%kernel_type)
  CASE(0)
    ! Squared exponential  
    if (((dwrt(1)/=1).and.(dwrt(1)/=2)).or.((dwrt(2)/=1).and.(dwrt(2)/=2))&
        .or.((dwrt(3)/=1).and.(dwrt(3)/=2))&
        .or.((dwrt(4)/=1).and.(dwrt(4)/=2))) then
        call dlf_fail("Error: wrong derivative information (kernel_d4)")
    else 
        kernel_d4 = this%gamma * (xm(j)-xn(j))*(xm(i)-xn(i))*&
                        (xn(k)-xm(k))*(xn(l)-xm(l))
        if (j==l) kernel_d4 = kernel_d4 + (xm(i)-xn(i))*(xn(k)-xm(k))
        if (i==l) kernel_d4 = kernel_d4 + (xm(j)-xn(j))*(xn(k)-xm(k))
        if (k==l) kernel_d4 = kernel_d4 - (xm(j)-xn(j))*(xm(i)-xn(i))
        kernel_d4 = kernel_d4*this%gamma    
        
        if (i==j) then
            if (k==l) kernel_d4 = kernel_d4+1d0
            kernel_d4 = kernel_d4-(xn(k)-xm(k))*this%gamma*(xn(l)-xm(l))
        end if
        if (j==k) then
            if (i==l) kernel_d4 = kernel_d4+1d0
            kernel_d4 = kernel_d4+(xm(i)-xn(i))*this%gamma*(xn(l)-xm(l))
        end if
        if (i==k) then
            if (j==l) kernel_d4 = kernel_d4+1d0
            kernel_d4 = kernel_d4+(xm(j)-xn(j))*this%gamma*(xn(l)-xm(l))
        end if
                        
        kernel_d4 = kernel_d4*this%s_f2*this%gamma**2*dexp(-this%gamma/2d0*&
                                        dot_product(xm(:)-xn(:),xm(:)-xn(:)))
    end if
  CASE(1)
    ! Matern with nu=5d0/2d0
    absv2 = dot_product(xm(:)-xn(:),xm(:)-xn(:))
    absv=dsqrt(absv2)
    ! This must be done to avoid division by 0
    if (absv<1d-16) then
        absv = 1d-16
        absv2= 1d-32
    end if
  if ((dwrt(1)/=2).or.(dwrt(2)/=2)&
        .or.(dwrt(3)/=1)&
        .or.(dwrt(4)/=1)) then
        call dlf_fail("Error: wrong derivative information (kernel_d4)")
  else
    ! only derivative wrt = 2,2,1,1 is needed > xn,xn,xm,xm
    ! These derivatives were all calculated with a CAS and
    ! automatically transformed in fortran code
    if (i==j) then
        if (j==k) then
            if (k==l) then
                ! iiii
                kernel_d4 = (25d0*(3d0*absv**3*this%l**2 +&
                    dsqrt(5d0)*this%l*(-6d0*absv2 + &
                    (xm(i) - xn(i))**2)*(xm(i) - xn(i))**2 + &
                    5d0*absv*(xm(i) - xn(i))**4))/&
                    (3d0*absv**3*&
                    dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
                ! lim->0:
                ! (25/l**4)
            else
                ! iiil
                kernel_d4 = (25d0*(-3d0*dsqrt(5d0)*absv2*this%l + &
                    5d0*absv*(xm(i) - xn(i))**2 + &
                    dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*&
                    (xm(i) - xn(i))*(xm(l) - xn(l)))/&
                    (3d0*absv**3*&
                    dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
                ! lim->0: (0)
            end if
        else if (j==l) then
                ! iiki
                kernel_d4 = (25d0*(-3d0*dsqrt(5d0)*absv2*this%l + &
                    5d0*absv*(xm(i) - xn(i))**2 + &
                    dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*&
                    (xm(i) - xn(i))*(xm(k) - xn(k)))/&
                    (3d0*absv**3*&
                    dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
                ! lim->0:
                ! (0)
        else
            if (k==l) then
                ! iikk
                kernel_d4 = (25d0*(absv**3*this%l**2 + &
                    dsqrt(5d0)*this%l*&
                    (-(absv2*((xm(i) - xn(i))**2 + (xm(k) - xn(k))**2)) + &
                    (xm(i) - xn(i))**2*(xm(k) - xn(k))**2) + &
                    5d0*absv*(xm(i) - xn(i))**2*(xm(k) - xn(k))**2))/&
                    (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*&
                    this%l**6)
                ! lim->0:
                ! (25/(3.*l**4))
            else
                ! iikl
                kernel_d4 = (25d0*(-(dsqrt(5d0)*absv2*this%l) + &
                    5d0*absv*(xm(i) - xn(i))**2 + &
                    dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*&
                    (xm(k) - xn(k))*(xm(l) - xn(l)))/&
                    (3d0*absv**3*&
                    dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
                ! lim->0:
                ! (0)
            end if
        end if
    else if (i==k) then
        if (j==l) then
            ! ijij
            kernel_d4 = (25d0*(absv**3*this%l**2 + dsqrt(5d0)*&
                    this%l*&
                    (-(absv2*((xm(i) - xn(i))**2 + (xm(j) - xn(j))**2)) + &
                    (xm(i) - xn(i))**2*(xm(j) - xn(j))**2) + &
                    5d0*absv*(xm(i) - xn(i))**2*(xm(j) - xn(j))**2))/&
                    (3d0*absv**3*&
                    dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
            ! lim->0:
            ! (25/(3.*l**4))
        else
            if (k==l) then
                ! ijii
                kernel_d4 = (25d0*(-3d0*dsqrt(5d0)*absv2*this%l + &
                    5d0*absv*(xm(i) - xn(i))**2 + &
                    dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*&
                    (xm(i) - xn(i))*(xm(j) - xn(j)))/&
                    (3d0*absv**3*&
                    dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
                ! lim->0:
                ! (0)
            else
                ! ijil
                kernel_d4 = (25d0*(-(dsqrt(5d0)*absv2*this%l) + &
                    5d0*absv*(xm(i) - xn(i))**2 + &
                    dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*&
                    (xm(j) - xn(j))*(xm(l) - xn(l)))/&
                    (3d0*absv**3*&
                    dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
                ! lim->0:
                ! (0)
            end if
        end if
    else if (i==l) then
        if (j==k) then
            ! ijji
            !k/=l
            kernel_d4 = (25d0*(absv**3*this%l**2 + &
                    dsqrt(5d0)*this%l*&
                    (-(absv2*((xm(i) - xn(i))**2 + (xm(j) - xn(j))**2)) + &
                    (xm(i) - xn(i))**2*(xm(j) - xn(j))**2) + &
                    5d0*absv*(xm(i) - xn(i))**2*(xm(j) - xn(j))**2))/&
                    (3d0*absv**3*&
                    dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
            ! lim->0:
            ! (25/(3.*l**4))
        else
            ! ijki
            !k/=l
            kernel_d4 = (25d0*(-(dsqrt(5d0)*absv2*this%l) + &
                    5d0*absv*(xm(i) - xn(i))**2 + &
                    dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*&
                    (xm(j) - xn(j))*(xm(k) - xn(k)))/&
                    (3d0*absv**3*&
                    dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
            ! lim->0:
            ! (0)
        end if
    else
        if (j==k) then
            if (k==l) then
                ! ijjj
                kernel_d4 = (25d0*(xm(i) - xn(i))*&
                    (-3d0*dsqrt(5d0)*absv2*this%l + &
                    5d0*absv*(xm(j) - xn(j))**2 + &
                    dsqrt(5d0)*this%l*(xm(j) - xn(j))**2)*(xm(j) - xn(j)))/&
                    (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*&
                    this%l**6)
                ! lim->0:
                ! (0)
            else
                ! ijjl
                kernel_d4 = (-25d0*(xm(i) - xn(i))*(dsqrt(5d0)*absv2*this%l -&
                    5d0*absv*(xm(j) - xn(j))**2 - &
                    dsqrt(5d0)*this%l*(xm(j) - xn(j))**2)*(xm(l) - xn(l)))/&
                    (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)&
                    *this%l**6)
                ! lim->0:
                ! (0)
            end if
        else if (j==l) then
                ! ijkj
                kernel_d4 = (-25d0*(xm(i) - xn(i))*&
                    (dsqrt(5d0)*absv2*this%l - 5d0*absv*(xm(j) - xn(j))**2 - &
                    dsqrt(5d0)*this%l*(xm(j) - xn(j))**2)*(xm(k) - xn(k)))/&
                    (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*&
                    this%l**6)
                ! lim->0:
                ! (0)
        else
            if (k==l) then
                ! ijkk
                kernel_d4 = (-25d0*(xm(i) - xn(i))*(xm(j) - xn(j))*&
                    (dsqrt(5d0)*absv2*this%l - 5d0*absv*(xm(k) - xn(k))**2 - &
                    dsqrt(5d0)*this%l*(xm(k) - xn(k))**2))/&
                    (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*&
                    this%l**6)
                ! lim->0:
                ! (0)
            else
                ! ijkl
                kernel_d4 = (25d0*(5d0*absv + dsqrt(5d0)*this%l)*&
                    (xm(i) - xn(i))*(xm(j) - xn(j))*(xm(k) - xn(k))*&
                    (xm(l) - xn(l)))/&
                    (3d0*absv**3*&
                    dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
                ! lim->0:
                ! (0)
            end if
        end if
    end if
  end if
    kernel_d4 = this%s_f2*kernel_d4
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (function kernel_d4)")
END SELECT             
end function kernel_d4
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_dg
!!
!! FUNCTION
!! The additional derivative wrt gamma (SE) or l (Matern) compared to kernel
!!
!! SYNOPSIS
real(rk) function kernel_dg(this, xm, xn)    
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  xm(this%sdgf), xn(this%sdgf)
    real(rk)             ::  factor, absv, absv2
SELECT CASE (this%kernel_type)
  CASE(0)    
    factor = -(dot_product(xm(:)-xn(:),xm(:)-xn(:))/2d0)
    kernel_dg = factor*kernel(this,xm,xn)
  CASE(1)
    ! Matern with nu=5d0/2d0
    absv2 = dot_product(xm(:)-xn(:),xm(:)-xn(:))
    absv = dsqrt(absv2)
    kernel_dg = (5d0*absv2*(dsqrt(5d0)*absv + this%l))/&
                    (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)        
    kernel_dg = this%s_f2*kernel_dg
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (kernel_dg)")
END SELECT          
end function kernel_dg
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d1_dg
!!
!! FUNCTION
!! The additional derivative wrt gamma (SE) or l (Matern) compared to kernel_d1
!!
!! SYNOPSIS
real(rk) function kernel_d1_dg(this, xm, xn, dwrt, i)    
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  xm(this%sdgf), xn(this%sdgf)
    integer, intent(in) ::  i    
    integer, intent(in) ::  dwrt ! 1 for der. wrt. xm; 2 for der. wrt. xn
    real(rk)             ::  factor, absv, absv2
SELECT CASE (this%kernel_type)
  CASE(0)         
    factor = 1d0/this%gamma-(dot_product(xm(:)-xn(:),xm(:)-xn(:))/2d0)    
    kernel_d1_dg = factor * kernel_d1(this,xm, xn, dwrt, i)
  CASE(1)
    ! Matern with nu=5d0/2d0
    absv2 = dot_product(xm(:)-xn(:),xm(:)-xn(:))
    absv = dsqrt(absv2)
    if (dwrt==1) then
        kernel_d1_dg = (5d0*(-5d0*absv2 + 2d0*dsqrt(5d0)*absv*this%l + &
                2d0*this%l**2)*(xm(i) - xn(i)))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5)
    else if (dwrt==2) then
        kernel_d1_dg = (-5d0*(-5d0*absv2 + 2d0*dsqrt(5d0)*absv*this%l + &
                2d0*this%l**2)*(xm(i) - xn(i)))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**5)
    end if    
    kernel_d1_dg = this%s_f2*kernel_d1_dg  
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (kernel_d1_dg)")
END SELECT         
end function kernel_d1_dg
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d2_dg
!!
!! FUNCTION
!! The additional derivative wrt gamma (SE) or l (Matern) compared to kernel_d2
!!
!! SYNOPSIS
real(rk) function kernel_d2_dg(this, xm, xn, dwrt, i, j)    
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  xm(this%sdgf), xn(this%sdgf)
    integer, intent(in) ::  dwrt(2) ! Derivative with respect to which (xm/xn)
    integer ::  i, j   
    real(rk)             ::  factor, absv, absv2
SELECT CASE (this%kernel_type)
  CASE(0)                 
    factor = -(dot_product(xm(:)-xn(:),xm(:)-xn(:))/2d0)    
    if (((dwrt(1)/=1).and.(dwrt(1)/=2)).or.((dwrt(2)/=1).and.(dwrt(2)/=2))) then
        call dlf_fail("Error: wrong derivative information (kernel_d2_dg)")
    else if (dwrt(1)/=dwrt(2)) then
        kernel_d2_dg = 2d0*this%gamma*(xm(j)-xn(j))*(xn(i)-xm(i))                    
        if (i==j) kernel_d2_dg = kernel_d2_dg + 1d0
        kernel_d2_dg = kernel_d2_dg*this%s_f2* &
                        dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
        kernel_d2_dg = kernel_d2_dg + &
                          factor * kernel_d2(this,xm, xn, dwrt, i, j)
    else if (dwrt(1)==1) then ! both are equal to 1
        kernel_d2_dg = 2d0*this%gamma*(xn(j)-xm(j))*(xn(i)-xm(i))
        if (i==j) kernel_d2_dg = kernel_d2_dg-1d0
        kernel_d2_dg = kernel_d2_dg*this%s_f2*&
                        dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
        kernel_d2_dg = kernel_d2_dg + &
                          factor * kernel_d2(this,xm, xn, dwrt, i, j)                
    else ! both are equal to 2
        kernel_d2_dg = 2d0*this%gamma*(xm(j)-xn(j))*(xm(i)-xn(i))
        if (i==j) kernel_d2_dg = kernel_d2_dg-1d0        
        kernel_d2_dg = kernel_d2_dg*this%s_f2*&
                        dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
        kernel_d2_dg = kernel_d2_dg + &
                          factor * kernel_d2(this,xm, xn, dwrt, i, j)                         
    end if
  CASE(1)
    ! Matern with nu=5d0/2d0
    absv2 = dot_product(xm(:)-xn(:),xm(:)-xn(:))
    absv = dsqrt(absv2)
    if (((dwrt(1)/=1).and.(dwrt(1)/=2)).or.((dwrt(2)/=1).and.(dwrt(2)/=2))) then
        call dlf_fail("Error: wrong derivative information (kernel_d2_dg)")
    else if (dwrt(1)/=dwrt(2)) then
        if (i==j) then
            kernel_d2_dg = (5d0*(5d0*absv2*this%l - 2*this%l*(this%l**2 -&
                10*(xm(i) - xn(i))**2) - &
                dsqrt(5d0)*absv*(2d0*this%l**2 + 5d0*(xm(i) - xn(i))**2)))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
        else
            kernel_d2_dg = (-25d0*(dsqrt(5d0)*absv - 4d0*this%l)*&
                (xm(i) - xn(i))*(xm(j) - xn(j)))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
        end if
    else 
        if (i/=j) then
            kernel_d2_dg = (25d0*(dsqrt(5d0)*absv - 4d0*this%l)*&
                (xm(i) - xn(i))*(xm(j) - xn(j)))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
        else
            kernel_d2_dg = (5d0*(-5d0*absv2*this%l + &
                2*this%l*(this%l**2 - 10*(xm(i) - xn(i))**2) + &
                dsqrt(5d0)*absv*(2d0*this%l**2 + 5d0*(xm(i) - xn(i))**2)))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**6)
        end if
    end if  
    kernel_d2_dg = this%s_f2*kernel_d2_dg
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (kernel_d2_dg)")
END SELECT        
end function kernel_d2_dg
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d3_dg
!!
!! FUNCTION
!! The additional derivative wrt gamma (SE) or l (Matern) compared to kernel_d3
!!
!! SYNOPSIS
real(rk) function kernel_d3_dg(this, xm, xn, dwrt, i, j, k)   
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  xm(this%sdgf), xn(this%sdgf)
    integer, intent(in) ::  dwrt(3) ! Derivative with respect to which (xm/xn)
    integer, intent(in) ::  i, j, k  
    real(rk)            ::  absv,factor, absv2
SELECT CASE (this%kernel_type)
  CASE(0)          
    factor = -(dot_product(xm(:)-xn(:),xm(:)-xn(:))/2d0)     
    if (((dwrt(1)/=1).and.(dwrt(1)/=2)).or.((dwrt(2)/=1).and.(dwrt(2)/=2))&
        .or.((dwrt(3)/=1).and.(dwrt(3)/=2))) then
        call dlf_fail("Error: wrong derivative information (kernel_d3_dg)")
    else if ((dwrt(1)==1).and.(dwrt(2)==1).and.(dwrt(3)==2)) then
        kernel_d3_dg = 3d0*this%gamma*(xn(j)-xm(j))*(xn(i)-xm(i))*&
                                        (xm(k)-xn(k))
        if (i==j) kernel_d3_dg = kernel_d3_dg-2d0*(xm(k)-xn(k))
        if (j==k) kernel_d3_dg = kernel_d3_dg+2d0*(xn(i)-xm(i))
        if (i==k) kernel_d3_dg = kernel_d3_dg+2d0*(xn(j)-xm(j))
        kernel_d3_dg = kernel_d3_dg*this%s_f2*this%gamma*&
                    dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
        kernel_d3_dg = kernel_d3_dg+ &
                          factor * kernel_d3(this,xm, xn, dwrt, i, j, k)
    else if ((dwrt(1)==2).and.(dwrt(2)==2).and.(dwrt(3)==1)) then
        kernel_d3_dg = 3d0*this%gamma*(xm(j)-xn(j))*(xm(i)-xn(i))*&
                                        (xn(k)-xm(k))
        if (i==j) kernel_d3_dg = kernel_d3_dg-2d0*(xn(k)-xm(k))
        if (j==k) kernel_d3_dg = kernel_d3_dg+2d0*(xm(i)-xn(i))
        if (i==k) kernel_d3_dg = kernel_d3_dg+2d0*(xm(j)-xn(j))        
        kernel_d3_dg = kernel_d3_dg*this%s_f2*this%gamma*&
                    dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))  
        kernel_d3_dg = kernel_d3_dg+ &
                          factor * kernel_d3(this,xm, xn, dwrt, i, j, k)                    
    else
        call dlf_fail("Error: wrong derivative information (d3_dg)")
    end if
  CASE(1)
    ! Matern with nu=5d0/2d0
    absv2 = dot_product(xm(:)-xn(:),xm(:)-xn(:))
    absv = dsqrt(absv2)
    if (absv<1d-16) then
!         small = .true.
        absv = 1d-16
        absv2= 1d-32
    end if
    if (((dwrt(1)/=1).and.(dwrt(1)/=2)).or.((dwrt(2)/=1).and.(dwrt(2)/=2))&
        .or.((dwrt(3)/=1).and.(dwrt(3)/=2))) then
        call dlf_fail("Error: wrong derivative information  (d3_dg)")
    else if ((dwrt(1)==1).and.(dwrt(2)==1).and.(dwrt(3)==2)) then
        if (i==j) then
            if (j==k) then
                ! iii
                kernel_d3_dg=(25d0*(-3d0*dsqrt(5d0)*absv2*this%l + &
                absv*(12*this%l**2 + 5d0*(xm(i) - xn(i))**2) - &
                5d0*dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*(xm(i) - xn(i)))/&
                (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**7)
            else
                ! iik
                kernel_d3_dg=(-25d0*(dsqrt(5d0)*absv2*this%l - &
                absv*(4d0*this%l**2 + 5d0*(xm(i) - xn(i))**2) + &
                5d0*dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*(xm(k) - xn(k)))/&
                (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**7)
            end if
        else if (i==k) then
                ! iji
                kernel_d3_dg=(-25d0*(dsqrt(5d0)*absv2*this%l - &
                absv*(4d0*this%l**2 + 5d0*(xm(i) - xn(i))**2) + &
                5d0*dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*(xm(j) - xn(j)))/&
                (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**7)
        else
            if (j==k) then
                ! ijj
                kernel_d3_dg=(-25d0*(xm(i) - xn(i))*(dsqrt(5d0)*&
                absv2*this%l - &
                absv*(4d0*this%l**2 + 5d0*(xm(j) - xn(j))**2) + &
                5d0*dsqrt(5d0)*this%l*(xm(j) - xn(j))**2))/&
                (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**7)
            else
                ! ijk
                kernel_d3_dg=(125d0*(absv - dsqrt(5d0)*this%l)*&
                (xm(i) - xn(i))*(xm(j) - xn(j))*(xm(k) - xn(k)))/&
                (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**7)
            end if
        end if
    else if ((dwrt(1)==2).and.(dwrt(2)==2).and.(dwrt(3)==1)) then
        if (i==j) then
            if (j==k) then
                ! iii
                kernel_d3_dg = (25d0*(3d0*dsqrt(5d0)*absv2*this%l - &
                absv*(12*this%l**2 + 5d0*(xm(i) - xn(i))**2) + &
                5d0*dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*(xm(i) - xn(i)))/&
                (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**7)
            else
                ! iik
                kernel_d3_dg=(25d0*(dsqrt(5d0)*absv2*this%l - &
                absv*(4d0*this%l**2 + 5d0*(xm(i) - xn(i))**2) + &
                5d0*dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*(xm(k) - xn(k)))/&
                (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**7)
            end if
        else if (i==k) then
                ! iji
                kernel_d3_dg=(25d0*(dsqrt(5d0)*absv2*this%l - &
                absv*(4d0*this%l**2 + 5d0*(xm(i) - xn(i))**2) + &
                5d0*dsqrt(5d0)*this%l*(xm(i) - xn(i))**2)*(xm(j) - xn(j)))/&
                (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**7)
        else
            if (j==k) then
                ! ijj
                kernel_d3_dg=(25d0*(xm(i) - xn(i))*(dsqrt(5d0)*absv2*this%l -&
                absv*(4d0*this%l**2 + 5d0*(xm(j) - xn(j))**2) + &
                5d0*dsqrt(5d0)*this%l*(xm(j) - xn(j))**2))/&
                (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**7)
            else
                ! ijk
                kernel_d3_dg=(-125d0*(absv - dsqrt(5d0)*this%l)*&
                (xm(i) - xn(i))*(xm(j) - xn(j))*(xm(k) - xn(k)))/&
                (3d0*absv*dexp((dsqrt(5d0)*absv)/this%l)*this%l**7)
            end if
        end if
    end if
    kernel_d3_dg = this%s_f2*kernel_d3_dg
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (kernel_d3_dg)")
END SELECT         
end function kernel_d3_dg
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/kernel_d4_dg
!!
!! FUNCTION
!! The additional derivative wrt gamma (SE) or l (Matern) compared to kernel_d4
!!
!! SYNOPSIS
real(rk) function kernel_d4_dg(this, xm, xn, dwrt, i, j, k, l)   
!! SOURCE
    implicit none
    type(gpr_type),intent(in)::this
    real(rk), intent(in) ::  xm(this%sdgf), xn(this%sdgf)
    integer, intent(in) ::  dwrt(4) ! Derivative with respect to which (xm/xn)
    integer, intent(in) ::  i, j, k, l
    real(rk)             ::  tmp
    real(rk)             ::  factor, absv, absv2
SELECT CASE (this%kernel_type)
  CASE(0)                  
    factor = -(dot_product(xm(:)-xn(:),xm(:)-xn(:))/2d0)         
    if (((dwrt(1)/=1).and.(dwrt(1)/=2)).or.((dwrt(2)/=1).and.(dwrt(2)/=2))&
        .or.((dwrt(3)/=1).and.(dwrt(3)/=2))&
        .or.((dwrt(4)/=1).and.(dwrt(4)/=2))) then
        call dlf_fail("Error: wrong derivative information (kernel_d4_dg)")
    else 
        kernel_d4_dg = 0d0
        if (i==j) then
            if (k==l) kernel_d4_dg = kernel_d4_dg+2d0
            kernel_d4_dg = kernel_d4_dg-(xn(k)-xm(k))*3d0*&
                    this%gamma*(xn(l)-xm(l))
        end if
        if (j==k) then
            if (i==l) kernel_d4_dg = kernel_d4_dg+2d0
            kernel_d4_dg = kernel_d4_dg+(xm(i)-xn(i))*3d0*&
                    this%gamma*(xn(l)-xm(l))
        end if
        if (i==k) then
            if (j==l) kernel_d4_dg = kernel_d4_dg+2d0
            kernel_d4_dg = kernel_d4_dg+(xm(j)-xn(j))*3d0*&
                    this%gamma*(xn(l)-xm(l))
        end if
        tmp = 0d0
        if (j==l) tmp = tmp + 3d0*(xm(i)-xn(i))*(xn(k)-xm(k))
        if (i==l) tmp = tmp + 3d0*(xm(j)-xn(j))*(xn(k)-xm(k))
        if (k==l) tmp = tmp - 3d0*(xm(j)-xn(j))*(xm(i)-xn(i))
        tmp = tmp + 4d0*this%gamma * (xm(j)-xn(j))*(xm(i)-xn(i))*&
                        (xn(k)-xm(k))*(xn(l)-xm(l))
        kernel_d4_dg = kernel_d4_dg+tmp*this%gamma                            
        kernel_d4_dg = kernel_d4_dg*&
                    this%s_f2*this%gamma*dexp(-this%gamma/2d0*&
                                        dot_product(xm(:)-xn(:),xm(:)-xn(:)))
        kernel_d4_dg = kernel_d4_dg+ &
                          factor * kernel_d4(this,xm, xn, dwrt, i, j, k, l)
    end if
  CASE(1)
    ! Matern with nu=5d0/2d0
    absv2 = dot_product(xm(:)-xn(:),xm(:)-xn(:))
    absv=dsqrt(absv2)
    if (absv<1d-16) then
!         small = .true.
        absv = 1d-16
        absv2= 1d-32
    end if
  if (((dwrt(1)/=1).and.(dwrt(1)/=2)).or.((dwrt(2)/=1).and.(dwrt(2)/=2))&
        .or.((dwrt(3)/=1).and.(dwrt(3)/=2))&
        .or.((dwrt(4)/=1).and.(dwrt(4)/=2))) then
        call dlf_fail("Error: wrong derivative information (kernel_d4_dg)")
  else
    ! only derivative wrt = 2,2,1,1 is needed > xn,xn,xm,xm
    if (i==j) then
        if (j==k) then
            if (k==l) then
                ! iiii
                kernel_d4_dg = (25d0*(3d0*dsqrt(5d0)*absv2**2*this%l**2 - &
                6d0*absv**3*this%l*(2d0*this%l**2 + &
                5d0*(xm(i) - xn(i))**2) + &
                5d0*dsqrt(5d0)*absv2*(6d0*this%l**2 + &
                (xm(i) - xn(i))**2)*(xm(i) - xn(i))**2 - &
                25d0*absv*this%l*(xm(i) - xn(i))**4 - &
                5d0*dsqrt(5d0)*this%l**2*(xm(i) - xn(i))**4))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
            else
                ! iiil
                kernel_d4_dg = (125d0*(-3d0*absv**3*this%l +&
                dsqrt(5d0)*absv2*(3d0*this%l**2 + (xm(i) - xn(i))**2) - &
                5d0*absv*this%l*(xm(i) - xn(i))**2 - &
                dsqrt(5d0)*this%l**2*(xm(i) - xn(i))**2)*&
                (xm(i) - xn(i))*(xm(l) - xn(l)))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
            end if
        else if (j==l) then
                ! iiki
                kernel_d4_dg = (125d0*(-3d0*absv**3*this%l +&
                dsqrt(5d0)*absv2*(3d0*this%l**2 + (xm(i) - xn(i))**2) - &
                5d0*absv*this%l*(xm(i) - xn(i))**2 - &
                dsqrt(5d0)*this%l**2*(xm(i) - xn(i))**2)*&
                (xm(i) - xn(i))*(xm(k) - xn(k)))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
        else
            if (k==l) then
                ! iikk
                kernel_d4_dg = (25d0*(dsqrt(5d0)*absv2**2*this%l**2 - &
                absv**3*this%l*(4d0*this%l**2 + &
                5d0*((xm(i) - xn(i))**2 + (xm(k) - xn(k))**2)) + &
                5d0*dsqrt(5d0)*absv2*(this%l**2*((xm(i) - xn(i))**2 + &
                (xm(k) - xn(k))**2) + (xm(i) - xn(i))**2*(xm(k) - xn(k))**2) - &
                25d0*absv*this%l*(xm(i) - xn(i))**2*(xm(k) - xn(k))**2 - &
                5d0*dsqrt(5d0)*this%l**2*&
                (xm(i) - xn(i))**2*(xm(k) - xn(k))**2))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
            else
                ! iikl
                kernel_d4_dg = (-125d0*(absv**3*this%l -&
                dsqrt(5d0)*absv2*(this%l**2 + (xm(i) - xn(i))**2) + &
                5d0*absv*this%l*(xm(i) - xn(i))**2 + dsqrt(5d0)*this%l**2&
                *(xm(i) - xn(i))**2)*(xm(k) - xn(k))*(xm(l) - xn(l)))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
            end if
        end if
    else if (i==k) then
        if (j==l) then
            ! ijij
            kernel_d4_dg = (25d0*(dsqrt(5d0)*absv2**2*this%l**2 - &
                absv**3*this%l*(4d0*this%l**2 + &
                5d0*((xm(i) - xn(i))**2 + (xm(j) - xn(j))**2)) + &
                5d0*dsqrt(5d0)*absv2*(this%l**2*((xm(i) - xn(i))**2 + &
                (xm(j) - xn(j))**2) + (xm(i) - xn(i))**2*(xm(j) - xn(j))**2) - &
                25d0*absv*this%l*(xm(i) - xn(i))**2*(xm(j) - xn(j))**2 - &
                5d0*dsqrt(5d0)*this%l**2*(xm(i) - xn(i))**2*&
                (xm(j) - xn(j))**2))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
        else
            if (k==l) then
                ! ijii
                kernel_d4_dg = (125d0*(-3d0*absv**3*this%l +&
                dsqrt(5d0)*absv2*(3d0*this%l**2 + (xm(i) - xn(i))**2) - &
                5d0*absv*this%l*(xm(i) - xn(i))**2 - &
                dsqrt(5d0)*this%l**2*(xm(i) - xn(i))**2)*&
                (xm(i) - xn(i))*(xm(j) - xn(j)))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
            else
                ! ijil
                kernel_d4_dg = (-125d0*(absv**3*this%l -&
                dsqrt(5d0)*absv2*(this%l**2 + (xm(i) - xn(i))**2) + &
                5d0*absv*this%l*(xm(i) - xn(i))**2 + &
                dsqrt(5d0)*this%l**2*(xm(i) - xn(i))**2)*&
                (xm(j) - xn(j))*(xm(l) - xn(l)))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
            end if
        end if
    else if (i==l) then
        if (j==k) then
            ! ijji
            !k/=l
            kernel_d4_dg = (25d0*(dsqrt(5d0)*absv2**2*this%l**2 - &
                absv**3*this%l*(4d0*this%l**2 + &
                5d0*((xm(i) - xn(i))**2 + (xm(j) - xn(j))**2)) + &
                5d0*dsqrt(5d0)*absv2*(this%l**2*((xm(i) - xn(i))**2 + &
                (xm(j) - xn(j))**2) + (xm(i) - xn(i))**2*(xm(j) - xn(j))**2) -&
                25d0*absv*this%l*(xm(i) - xn(i))**2*(xm(j) - xn(j))**2 - &
                5d0*dsqrt(5d0)*this%l**2*(xm(i) - xn(i))**2*&
                (xm(j) - xn(j))**2))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
        else
            ! ijki
            !k/=l
            kernel_d4_dg = (-125d0*(absv**3*this%l -&
                dsqrt(5d0)*absv2*(this%l**2 + (xm(i) - xn(i))**2) + &
                5d0*absv*this%l*(xm(i) - xn(i))**2 + &
                dsqrt(5d0)*this%l**2*(xm(i) - xn(i))**2)*&
                (xm(j) - xn(j))*(xm(k) - xn(k)))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
        end if
    else
        if (j==k) then
            if (k==l) then
                ! ijjj
                kernel_d4_dg = (125d0*(xm(i) - xn(i))*&
                (-3d0*absv**3*this%l + &
                dsqrt(5d0)*absv2*(3d0*this%l**2 + (xm(j) - xn(j))**2) - &
                5d0*absv*this%l*(xm(j) - xn(j))**2 - &
                dsqrt(5d0)*this%l**2*(xm(j) - xn(j))**2)*&
                (xm(j) - xn(j)))/(3d0*absv**3*&
                dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
            else
                ! ijjl
                kernel_d4_dg = (-125d0*(xm(i) - xn(i))*&
                (absv**3*this%l - &
                dsqrt(5d0)*absv2*(this%l**2 + (xm(j) - xn(j))**2) + &
                5d0*absv*this%l*(xm(j) - xn(j))**2 + &
                dsqrt(5d0)*this%l**2*(xm(j) - xn(j))**2)*&
                (xm(l) - xn(l)))/(3d0*absv**3*&
                dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
            end if
        else if (j==l) then
                ! ijkj
                kernel_d4_dg = (-125d0*(xm(i) - xn(i))*&
                (absv**3*this%l - &
                dsqrt(5d0)*absv2*(this%l**2 + (xm(j) - xn(j))**2) + &
                5d0*absv*this%l*(xm(j) - xn(j))**2 + &
                dsqrt(5d0)*this%l**2*(xm(j) - xn(j))**2)*&
                (xm(k) - xn(k)))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
        else
            if (k==l) then
                ! ijkk
                kernel_d4_dg = (-125d0*(xm(i) - xn(i))*(xm(j) - xn(j))*&
                (absv**3*this%l - dsqrt(5d0)*absv2*(this%l**2 + &
                (xm(k) - xn(k))**2) + &
                5d0*absv*this%l*(xm(k) - xn(k))**2 + &
                dsqrt(5d0)*this%l**2*(xm(k) - xn(k))**2))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
            else
                ! ijkl
                kernel_d4_dg = (125d0*(dsqrt(5d0)*absv2 - 5d0*absv*this%l -&
                dsqrt(5d0)*this%l**2)*(xm(i) - xn(i))*&
                (xm(j) - xn(j))*(xm(k) - xn(k))*(xm(l) - xn(l)))/&
                (3d0*absv**3*dexp((dsqrt(5d0)*absv)/this%l)*this%l**8)
            end if
        end if
    end if
  end if
    kernel_d4_dg = this%s_f2*kernel_d4_dg
  CASE DEFAULT
    call dlf_fail("The requested type of kernel is not implemented! (kernel_d4_dg)")
END SELECT      
end function kernel_d4_dg
!!****
  
end module gpr_kernels_module

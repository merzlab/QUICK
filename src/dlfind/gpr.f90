! **********************************************************************
! **                   Gaussian Process Regression                    **
! **                      for interpolation and                       **
! **                      geometry optimisation                       **
! **********************************************************************
!!****h* gpr
!!
!! NAME
!! gpr
!!
!! FUNCTION
!! Several methods to use GPR in geometry optimization and interpolation
!!
!! COMMENTS
!! Needs gpr_in_chemsh_readin.f90 in the same folder
!! NOTE: I often used functions like dnrm2 that only work if rk is double
!! 
!! Short description of how to use this thing:
!! 
!! Everything is packed into one single module called
!! gpr_module. One must construct a variable of type "gpr_type" (from this
!! module). This contains all the data used in GPR and must be propagated
!! through all subroutines in this module (so simply put it as the first
!! parameter on every subroutine that should work on this GPR calculation).
!! The methods that are interesting for a user are listed now.
!! gpr - used for developping
!!       inside this module. One should only use this function
!!       for development. It can also be considered as an example 
!!       code for the usage of the following subroutines.
!!       It should be used, if one wants to generate a gpr-data file
!!       using neuralnetwork.f90 for further use in dl-find.
!!       It gets the data from neuralnetwork.f90 and can write it 
!!       in a gpr-data file.
!! 
!! The following routines can be used independently of dl-find and 
!! neuralnetwork.f90 and include everything needed for gpr-interpolation:
!! 
!! GPR_construct - Constructs all data variables needed for a GPR calculation. 
!!                 This is the first thing you have
!!                 to run, when not simply using the "gpr" subroutine
!! gpr_init - Initializes a GPR calculation with the training data
!! GPR_interpolation - Calculates everything necessary after gpr_init was run
!! GPR_testEnergies  - Evaluates the interpolation using some additinoal test data
!! GPR_testGrad - Same for given test gradients
!! GPR_testHess - Same for given test Hessians
!! GPR_write - writes a ".gpr" file to store the interpolation information on disk
!! GPR_read - reads the respective data 
!! GPR_eval - Evaluates energies  at a given position
!! GPR_eval_grad - Evaluates gradients at a given position
!! GPR_eval_hess -Evaluates hessians  at a given position
!! GPR_destroy - Deallocates all RAM consuming elements
!! 
!! Some additional information: 
!! 
!! - "order" of a GPR refers to the usage of degree of derivatives used for
!! the interpolation
!! 
!! - If you want to create an output for dl-find you must call 
!! storeTrafoForDL_Find in order for the coordinate trafos to work.
!! This must be done before calling GPR_write
!! 
!! - The hyperparameter "gamma" always refers to 1/l^2
!!   In the Squared Exponential Kernel gamma is simply the factor in the exp
!!   For the Matern Kernel however 1/l = sqrt(gamma) is the factor in the exp
!!   So if you want to think in "characteristic lengths" 'l'
!!   input the data for gamma as 1/(l)**2 for every Kernel
!! DATA
!!  Date: 2018-01-08
!!  Author: Alexander Denzel
!!
!! COPYRIGHT
!!
!!  Copyright 2018 Alexander Denzel (denzel@theochem.uni-stuttgart.de)
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


! *****************************************************************************
! PREPROCESSOR VARIABLES
!

! compiler flag -D withopenmp sets openmp support

! writeBestGuessInGPRPP is needed for the benchmarks I performed in the TS paper
! it writes out the starting point on the IDPP found by the GPRPP method
! it might also be interesting to see where the algorithm starts the TS search
! #define writeBestGuessInGPRPP

! Define which kind of task should be performed in the subroutine "gpr"
! Only interesting for testing and developing
! #define ImproveMultiLevel
! #define RealSystem
! #define MuellerBrown
#define Test1D
! #define Test1D2
! #define Test2D
! #define dimerMB
! #define testIterSolver

! #define SE_pOpt
#define Ma52_pOpt

! Only for GPRMEP when testing on MB
! #define TestOnMB

module gpr_module
    use dlf_parameter_module, only: rk
    use dlf_global, only: stdout, stderr, printl
    use gpr_types_module
    use gpr_kernels_module
    use geodesic_module
    use omp_lib
    implicit none
    type(geodesic_type),save                 ::  gpr_geo_inst
    real(rk), PARAMETER  ::  pi = 3.14159265359d0
    private :: calc_cov_mat_dg_or_l,calc_p_like_and_deriv,&
            scale_p, &
        !    Cholesky_KM_logdet, &
            invert_KM_chol, solve_LGS_use_KM_chol_inv, &
            writeObservationsInW, &
            GPR_adjustArraySizes, &
          !  train_and_test_h,&
            Cholesky_KM, solve_LGS_use_KM_chol, &
            init_random_seed, eigVec_to_smallestEigVal, &
            outputMatrixAsCSV, &
            exf2d, dexf2d, ddexf2d, &
!            exf, dexf, ddexf,  &
            GPR_eval_giveLower, GPR_eval_grad_giveLower, &
#ifdef withopenmp            
            GPR_eval_hess_column, &
#endif
            orthogonalizeVecToVecs, &
            orderKM, orderW, &
            Cholesky_myOwn, Cholesky_KM_myown
            
    public  :: GPR_eval, GPR_eval_grad, GPR_eval_hess, &
#ifdef withopenmp             
            GPR_eval_hess_parallel, &
#endif
            GPR_interpolation, GPR_redo_interpolation, &
            GPR_construct, GPR_init, GPR_init_without_tp, &
            GPR_destroy, setMeanOffset, &
            manual_offset, GPR_changeOffsetType, &
            GPR_changeparameters, GPR_deleteOldestTPs, &
            GPR_deleteNewestTPs, GPR_deleteTP, GPR_deletefarestTPs, &
            GPR_copy_GPR, GPR_add_tp, &
            GPR_variance, storeTrafoForDL_Find, &
            GPR_testEnergies, GPR_testGrad, GPR_testHess, &
            GPR_write, GPR_read, check_KM, check_Matrix, &
            GPR_distance_nearest_tp, writeParameterFile, &
            LBFGS_Max_Like, Plot_Likel_onePara, &
            constructTrafoForDL_Find, &
            train_and_test_e, &            
            train_and_test_g, &
            train_and_test_h,&            
            GPR_newLevel, GPR_write_Hess_eigvals, &
            reorderW, orderW_inplace, &
            exf, dexf, ddexf,calc_cov_mat_newOrder,calc_cov_mat,&
            Cholesky_KM_logdet
  contains

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/calc_cov_mat
!!
!! FUNCTION
!! Calculates the covariance matrix
!!
!! COMMENT
!! If somebody else then me should ever try to read this routine's code
!! I'm very sorry for the bad readability but in order to stay
!! efficient and also keep the code short enough to be readable
!! I had to use lots of short variable names...
!!
!! SYNOPSIS
subroutine calc_cov_mat(this)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    integer ::              i, j, k, l, m, n, osx, osy, y, x, &
                            os33,b_k,b_j
    integer ::              dgf, nr,i_dgf
    real(rk)             ::  xm(this%sdgf), xn(this%sdgf), diff(this%sdgf)
    real(rk)            ::   absv,dk_dx
    logical             ::   osx_increased
    
     
    if(this%internal == 2) then
	  dgf = this%idgf
	  i_dgf = this%sdgf
	else 
	  dgf = this%sdgf
	endif
    if (this%iChol) call dlf_fail("iChol and calc_cov_mat not compatible.")
    if (this%kernel_type<=1) then
    ! using squared exponential kernel
    this%KM(:,:)=0d0!9d99    
     ! Calculate Cov11 -> E with E
    ! this exists for every training point for every order, even 42
    osx = 0
    osy = 0
    if (this%kernel_type==1) then
      do n = 1, this%nt
        xn = this%xs(:,n)
        do m = 1, this%nt
            xm = this%xs(:,m)
            absv = dsqrt(dot_product(xm(:)-xn(:),xm(:)-xn(:)))
            this%KM(m,n) = this%s_f2*&
                (1d0 + (dsqrt(5d0)*absv)/this%l + &
                 (5d0*absv**2)/(3d0*this%l**2))/&
                dexp((dsqrt(5d0)*absv/this%l))
            if (m==n) this%KM(m,n) = this%KM(m,n) + this%s_n(1)**2
        end do
      end do
    else if (this%kernel_type==0) then
      do n = 1, this%nt
        xn = this%xs(:,n)
        do m = 1, this%nt
            xm = this%xs(:,m)
            this%KM(m,n) = this%s_f2*dexp(-this%gamma/2d0*dot_product(xm(:)-xn(:),xm(:)-xn(:)))
            if (m==n) this%KM(m,n) = this%KM(m,n) + this%s_n(1)**2
        end do
      end do
    else
      do n = 1, this%nt
        xn = this%xs(:,n)
        do m = 1, this%nt
            xm = this%xs(:,m)
            this%KM(m,n) = kernel(this,xm,xn)
            if (m==n) this%KM(m,n) = this%KM(m,n) + this%s_n(1)**2
        end do
      end do
    end if
  if (this%order>0) then
    ! Calculate Cov 12 -> E with G
    osy = 0
    osx = this%nt-dgf
    do n = 1, this%nt
      if (this%order/=42.or.this%order_tps(n)>=1) then
        xn = this%xs(:,n)      
        y = osy
        osx = osx + dgf
        do m = 1, this%nt
          y = y + 1
          xm = this%xs(:,m)
          diff(:)=xm(:)-xn(:)
          absv=dsqrt(dot_product(diff,diff))
          x = osx
          do i = 1, dgf
            x = x + 1
!             x = osx+(n-1)*dgf+i
!             y = osy+m
            if(this%internal == 2) then
              dk_dx = 0.0d0
              do b_k = 1, i_dgf
				dk_dx = dk_dx + kernel_d1_exp2(this,diff,b_k,absv)*this%b_matrix(b_k,i,n)
              enddo
              this%KM(y,x) = dk_dx
              this%KM(x,y) = this%KM(y,x)
            else
              this%KM(y,x) = kernel_d1_exp2(this,diff,i,absv)
              this%KM(x,y) = this%KM(y,x)
            endif
          end do            
        end do
      end if
    end do

    ! Calculate Cov 22
    osy = this%nt
    osx = this%nt - dgf
    do n = 1, this%nt
        xn = this%xs(:,n)
        osx_increased = .false.
!         osy = this%nt - dgf
        y = osy
        do m = 1, this%nt
          if (this%order/=42 .or.(this%order_tps(m)>=1.and.&
                                 this%order_tps(n)>=1)) then
            if (.not.osx_increased) then
              osx = osx + dgf
              osx_increased = .true.
            end if
!             osy = osy + dgf
!             y = y + dgf
            xm = this%xs(:,m)
            diff(:)=xm(:)-xn(:)
            absv=dsqrt(dot_product(diff,diff))
!             y = osy
!			if (this%internal) then 
			 ! dgf = dgf+6
!			endif
            do i = 1, dgf
              y = y + 1
              x = osx
              do j = 1, dgf
                x = x + 1
!                     x = osx+(n-1)*dgf+j
!                     y = osy+(m-1)*dgf+i
                    if (this%internal == 2) then
                       dk_dx=0.0d0
                       do b_k =1,i_dgf
                         do b_j =1,i_dgf
                           dk_dx = dk_dx + (this%b_matrix(b_k,i,m) * &
                            kernel_d2_exp12(this,diff,b_k,b_j,absv) *this%b_matrix(b_j,j,n))
                         enddo
                       enddo
                       this%KM(y,x) = dk_dx
                       if (x==y) this%KM(y,x) = this%KM(y,x) + this%s_n(2)**2  
                       !print *,'KM(',y,',',x,')=',this%KM(y,x)
                    else
					  this%KM(y,x) = kernel_d2_exp12(this,diff,i,j,absv)
				      if (x==y) this%KM(y,x) = this%KM(y,x) + this%s_n(2)**2 
					endif
                end do
            end do
          end if
        end do
    end do
    
    
  if (this%order>=2) then
    ! save the positions 
    os33 = x ! position-1 at which 31, 32, 33 start for x and 33 starts for y
    ! Calculate Cov13
    osx = os33-dgf*(dgf+1)/2 !this%nt+this%nt*dgf
    osy = 0
    do n = 1, this%nt
      if (this%order/=42.or.this%order_tps(n)>=2) then
        xn = this%xs(:,n)
        osx = osx + dgf*(dgf+1)/2
        y = 0
        do m = 1, this%nt
          y = y + 1
          xm = this%xs(:,m)
          diff(:)=xm(:)-xn(:)
          absv=dsqrt(dot_product(diff,diff))
          ! The matrix starts at offset "osx"+1 in x direction    
!           os = -this%sdgf
          x = osx
          do j = 1, dgf
!             os = os + this%sdgf-(j-1)
            do i = j, dgf
              ! points x_m, x_n and entries of cov_mat
!               x = osx+(n-1)*dgf*(dgf+1)/2+os+i
              x = x + 1
              !Element A_mn
              this%KM(y,x) = kernel_d2_exp22(this,diff,i,j,absv)
              this%KM(x,y) = this%KM(y,x)
            end do
          end do            
        end do
      end if
    end do
    
    ! Calculate Cov23
    osx = os33-dgf*(dgf+1)/2
!     osx = this%nt+this%nt*dgf
    osy = this%nt - dgf
    do n = 1, this%nt
      if (this%order/=42.or.this%order_tps(n)>=2) then
        osx_increased = .false.
        xn = this%xs(:,n)
        osy = this%nt - dgf
        do m = 1, this%nt
          if (this%order/=42.or.this%order_tps(m)>=1) then
            if (.not. osx_increased) then
              osx = osx + dgf*(dgf+1)/2
              osx_increased = .true.
            end if
            osy = osy + dgf
            xm = this%xs(:,m)
            ! The matrix starts at offset "osy"+1 in y direction & "osx"+1 in x
!             os = -this%sdgf
            x = osx
            do j = 1, dgf ! walking over first coordinate of the hessian
!               os = os + this%sdgf-(j-1)
              do i = j, dgf ! walking over the second coordinate of the hessian
                x = x + 1
                y = osy
                do k = 1, dgf ! walking over entries of the gradient
!                   x = osx+(n-1)*dgf*(dgf+1)/2+os+i
!                   y = osy+(m-1)*dgf+k        
                  y = y + 1
                  !Element A_mn
                  this%KM(y,x) = kernel_d3(this,xm,xn,(/2,2,1/),i,j,k)
                  ! Cov32
                  this%KM(x,y) = this%KM(y,x)
                end do
              end do
            end do            
          end if
        end do
      end if
    end do
    
    ! Calculate Cov33
!     osx = this%nt+this%nt*dgf
!     osy = this%nt+this%nt*dgf
    nr = dgf*(dgf+1)/2
    osx = os33-nr
    osy = os33-nr
    TPLOOP_N: do n = 1, this%nt
      if (this%order/=42.or.this%order_tps(n)>=2) then
        osx_increased = .false.
        xn = this%xs(:,n)
        osy = os33 - nr
        TPLOOOP_M: do m = 1, this%nt
          if (this%order/=42.or.this%order_tps(m)>=2) then
            if (.not.osx_increased) then
              osx = osx + nr
              osx_increased = .true.
            end if
            osy = osy + nr
            xm = this%xs(:,m)
            ! The matrix starts at offset "osy"+1 in y direction & "osx"+1 in x
!             os = -this%sdgf
            x = osx
            XLOOP1: do i = 1, dgf ! 1st coord of hessian of pt xn
!               os = os + this%sdgf-(i-1)
              XLOOP2: do j = i, dgf ! 2nd coord of hessian of pt xn
!                 os2 = -this%sdgf
!                 x = osx+(n-1)*nr+os+j
                x = x + 1
                y = osy
                YLOOP1: do k = 1, dgf ! 1st coord of hessian of pt xm
!                   os2 = os2 + this%sdgf-(k-1)
                  YLOOP2: do l = k, dgf ! 2nd coord of hessian of pt xm
!                     y = osy+(m-1)*nr+os2+l
                    !Element A_mn
                    y = y + 1
                    if (y==x) then 
                      ! write diagonal element then stop the loop
                      this%KM(y,x) = kernel_d4(this,xm, xn, &
                                   (/2,2,1,1/), i, j, k, l) +&
                                   this%s_n(3)**2
                      exit YLOOP1
                    end if
                    this%KM(y,x) = kernel_d4(this,xm, xn, &
                                   (/2,2,1,1/), i, j, k, l)
                    this%KM(x,y) = this%KM(y,x)
                  end do YLOOP2
                end do YLOOP1
              end do XLOOP2
            end do XLOOP1   
          end if
        end do TPLOOOP_M
      end if
    end do TPLOOP_N
  end if ! Order 2
  end if ! Order 1
else
    call dlf_fail("The requested type of kernel is not implemented! (calc_cov_mat)")
end if
this%K_stat = 0

end subroutine calc_cov_mat                                     
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/calc_cov_mat_newOrder
!!
!! FUNCTION
!! Calculates the covariance matrix
!!
!! SYNOPSIS
subroutine calc_cov_mat_newOrder(this, reuse_old_in)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)  :: this
    logical, intent(in), optional :: reuse_old_in
    integer ::              i, j, k, m, n, os
    integer ::              j2, d1, d2, gd, hd1, hd2
    integer ::              dgf, mOffset,i_dgf,b_k,b_j
    real(rk)             ::  diff(this%sdgf)
    real(rk)            ::  absv,dk_dx
    integer             ::  startNt
    real(rk)            :: matrix_d2(this%sdgf,this%sdgf)
    logical             :: reuse_old
    real(rk)            :: y(this%sdgf)
    if(.not.present(reuse_old_in)) then
      reuse_old=.false.
    else
      reuse_old = reuse_old_in
    endif
    if (.not.this%iChol) call dlf_fail("Not iChol and calc_cov_mat_newOrder not compatible.")
    if (.not.present(reuse_old_in).or.(.not.reuse_old)) then
      startNt = 1
    else
      startNt = this%nt_old+1
    end if
    if(this%internal == 2) then
	  dgf = this%idgf
	  i_dgf = this%sdgf
	else 
	  dgf = this%sdgf
	endif    
!    dgf = this%sdgf
if (this%kernel_type<=1) then
! using squared exponential kernel

!   this%KM(:,:)=0d0 ! 
  if (this%K_stat==-1) this%KM_lin(:)=0d0
  
  ! construct only lower half of the matrix in linearized form
  if(this%order==0) then
    ! go through the rows of the lower-triangular matrix
   if (this%kernel_type==1) then
     do i=startNt, this%nk
       do j=1,i-1
         ! Entry KM_ij -> KM_k in linearized form
         k = (i)*(i-1)/2+j
         absv = dsqrt(dot_product(this%xs(:,i)-this%xs(:,j),&
                                  this%xs(:,i)-this%xs(:,j)))
         this%KM_lin(k) = this%s_f2*(1d0 + (dsqrt(5d0)*absv)/this%l + &
            (5d0*absv**2)/(3d0*this%l**2))/&
            dexp((dsqrt(5d0)*absv/this%l))
         if (i==j) this%KM_lin(k) = this%KM_lin(k) + this%s_n(1)**2
       end do
       ! i == j
       k = (i)*(i-1)/2+i
       this%KM_lin(k) = kernel(this,this%xs(:,i),this%xs(:,i))
       this%KM_lin(k) = this%KM_lin(k) + this%s_n(1)**2
     end do
   else if (this%kernel_type==0) then
     do i=startNt, this%nk
       do j=1,i-1
         ! Entry KM_ij -> KM_k in linearized form
         k = (i)*(i-1)/2+j
         this%KM_lin(k) = this%s_f2*&
            dexp(-this%gamma/2d0*&
                 dot_product(this%xs(:,i)-this%xs(:,j),&
                             this%xs(:,i)-this%xs(:,j)))
         if (i==j) this%KM_lin(k) = this%KM_lin(k) + this%s_n(1)**2
       end do
       ! i == j
       k = (i)*(i-1)/2+i
       this%KM_lin(k) = kernel(this,this%xs(:,i),this%xs(:,i))
       this%KM_lin(k) = this%KM_lin(k) + this%s_n(1)**2
     end do
   else
     do i=startNt, this%nk
       do j=1,i-1
         ! Entry KM_ij -> KM_k in linearized form
         k = (i)*(i-1)/2+j
         this%KM_lin(k) = kernel(this,this%xs(:,i),this%xs(:,j))
         if (i==j) this%KM_lin(k) = this%KM_lin(k) + this%s_n(1)**2
       end do
       ! i == j
       k = (i)*(i-1)/2+i
       this%KM_lin(k) = kernel(this,this%xs(:,i),this%xs(:,i))
       this%KM_lin(k) = this%KM_lin(k) + this%s_n(1)**2
     end do
   end if
  else if (this%order==1) then
    o1tp1: do i=startNt,this%nt ! run over all training points
      m = (i-1)*(this%idgf+1)+1 ! row m
      ! hier eine e, sdgf*g, e, sdgf*g, ..., e line einfügen
      os = (m*(m-1))/2
      o1tp_e: do j = 1, i ! run over all training points in lower triangular matrix
        ! Entry K_mn is entry K(os+n) in the linearized array
        n = (j-1)*(this%idgf+1)+1
        ! only m entries in row m have to be calculated for lower matrix
        if (n>m) exit o1tp_e
        ! Nur die Energien tp i -> entry m, tp j -> entry n, ij->mn
        this%KM_lin(os+n) = kernel(this,this%xs(:,i),this%xs(:,j))
 !       print*,'KM(',os+n,')=k(x',i,',',j,')=K_(',m,n,')' 
        ! Jetzt die Gradienten
        diff = this%xs(:,i)-this%xs(:,j)
        absv=norm2(diff)
        do d1 = 1, this%idgf
          n = n + 1 ! K_MN -> e_i g_j(d1)
          ! only m entries in row m have to be calculated for lower matrix
          if (n>m) exit o1tp_e  
            if(this%internal == 2) then 
          !for internal method 2: use chainrule from internal to Cartesian coordinates 
              dk_dx = 0.0d0
              do b_k = 1, i_dgf
				dk_dx = dk_dx + kernel_d1_exp2(this,diff,b_k,absv)*this%b_matrix(b_k,d1,j)
              enddo
              this%KM_lin(os+n) = dk_dx
            else               
              this%KM_lin(os+n) = kernel_d1_exp2(this,diff,d1,absv)              
            endif
        end do
      end do o1tp_e
      ! these are the diagonal elements of energies
      this%KM_lin(os+m) = this%KM_lin(os+m) + this%s_n(1)**2
      ! hier die nächsten g, sdgf*d, g, sdgf*d, ... line einfügen, Länge bis m
      do gd = 1, this%idgf
        ! row m+1
        m = m + 1
        os = (m*(m-1))/2
        ! jetzt eine Zeile g, sdgf*d, g, sdgf*d, ...
        ! run over all training points in lower triangular matrix
        o1tp_g: do j = 1, i
          n = (j-1)*(this%idgf+1)+1
          ! only m entries in row m have to be calculated for lower matrix
          if (n>m) exit o1tp_g
          diff = this%xs(:,i)-this%xs(:,j)
          absv=norm2(diff)
          if(this%internal == 2) then
          !again transfomration for internal method 2
            dk_dx = 0.0d0
            do b_k = 1, i_dgf
			  dk_dx = dk_dx + kernel_d1_exp1(this,diff,b_k,absv)*this%b_matrix(b_k,gd,i)
            enddo
            this%KM_lin(os+n) = dk_dx
          else               
            this%KM_lin(os+n) = kernel_d1_exp1(this,diff,gd,absv)
          endif
          do d1 = 1, this%idgf
            n = n + 1
            ! only m entries in row m have to be calculated for lower matrix
            if (n>m) exit o1tp_g
            if (this%internal == 2) then
              dk_dx=0
              call kernel_matern_d2_exp12_matrix(this,diff,absv,matrix_d2)
              y = matmul(matrix_d2, this%b_matrix(:,d1,j))
              dk_dx = dot_product(this%b_matrix(:,gd,i),y)
              this%KM_lin(os+n) = dk_dx
            else           
              this%KM_lin(os+n) = kernel_d2_exp12(this,diff,gd,d1,absv) 
            endif        
          end do
        end do o1tp_g
        ! these are the diagonal elements of gradients
        this%KM_lin(os+m) = this%KM_lin(os+m) + this%s_n(2)**2
      end do
    end do o1tp1
  else if (this%order==2) then
    o2tp1: do i=startNt,this%nt ! run over all training points
    
      ! first line of type
      ! e-e, e-sdgf*g, e-sdgf(sdgf+1)/2*h, ...
      m = (i-1)*((this%sdgf*(this%sdgf+1)/2) + this%sdgf + 1) + 1 ! row m
      os = (m*(m-1))/2
      o2tp2: do j = 1, i ! run over all training points in lower triangular matrix
        ! Entry K_mn is entry K(os+n) in linearized array
        n = (j-1)*((this%sdgf*(this%sdgf+1)/2) + this%sdgf + 1) + 1
        ! only m entries in row m have to be calculated for lower matrix
        if (n>m) exit o2tp2
        ! energy entries
        this%KM_lin(os+n) = kernel(this,this%xs(:,i),this%xs(:,j))
        ! gradient entries
        diff = this%xs(:,i)-this%xs(:,j)
        absv=norm2(diff)
        do d1 = 1, this%sdgf
          n = n + 1 ! K_MN -> e_i g_j(d1)
          ! only m entries in row m have to be calculated for lower matrix
          if (n>m) exit o2tp2
          this%KM_lin(os+n) = kernel_d1_exp2(this,diff,d1,absv)
        end do
        ! hessian entries
        do d1 = 1, this%sdgf
          do d2 = d1, this%sdgf! 1, d1
            n = n + 1
            ! only m entries in row m have to be calculated for lower matrix
            if (n>m) exit o2tp2
            this%KM_lin(os+n) = kernel_d2_exp22(this,diff,d1,d2,absv)
          end do
        end do
      end do o2tp2
      ! these are the diagonal elements of energies
      this%KM_lin(os+m) = this%KM_lin(os+m) + this%s_n(1)**2
      
      ! sdgf lines of type 
      ! g-e, g-sdgf*g, g-sdgf(sdgf+1)/2*h, ...
      do gd = 1, this%sdgf
        ! row m + 1
        m = m + 1
        os = (m*(m-1))/2
        ! run over all training points in lower triangular matrix
        o2tp_g: do j = 1, i 
          n = (j-1)*((this%sdgf*(this%sdgf+1)/2) + this%sdgf + 1) + 1
          ! only m entries in row m have to be calculated for lower matrix
          if (n>m) exit o2tp_g
          diff = this%xs(:,i)-this%xs(:,j)
          absv=norm2(diff)
          ! g-e entry
          this%KM_lin(os+n) = kernel_d1_exp1(this,diff,gd,absv)
          ! g-sdgf*g entries
          do d1 = 1, this%sdgf
            n = n + 1
            ! only m entries in row m have to be calculated for lower matrix
            if (n>m) exit o2tp_g
            this%KM_lin(os+n) = kernel_d2_exp12(this,diff,gd,d1,absv) 
          end do
          ! g - sdgf(sdgf+1)/2*h entries
          do d1 = 1, this%sdgf
            do d2 = d1, this%sdgf !1, d1
              n = n + 1
              ! only m entries in row m have to be calculated for lower matrix
              if (n>m) exit o2tp_g
              this%KM_lin(os+n) = kernel_d3(this, this%xs(:,i), this%xs(:,j), (/2,2,1/), d1, d2, gd)
            end do
          end do
        end do o2tp_g
        ! these are the diagonal elements of gradients
        this%KM_lin(os+m) = this%KM_lin(os+m) + this%s_n(2)**2
      end do
      ! sdgf*(sdgf+1)/2 lines of type
      ! h-e, h-sdgf*g, h-sdgf(sdgf+1)/2*h, ... 
      do hd1 = 1, this%sdgf
        do hd2 = hd1, this%sdgf!1, hd1!this%sdgf
          ! row m + 1
          m = m + 1
          os = (m*(m-1))/2
          ! run over all training points in lower triangular matrix
          o2tp_h: do j = 1, i
            n = (j-1)*((this%sdgf*(this%sdgf+1)/2) + this%sdgf + 1) + 1
            ! only m entries in row m have to be calculated for lower matrix
            if (n>m) exit o2tp_h
            diff = this%xs(:,i)-this%xs(:,j)
            absv=norm2(diff)
            ! h-e entry
            this%KM_lin(os+n) = kernel_d2_exp11(this, diff, hd1, hd2, absv)
            ! h -sdgf*g entries
            do d1 = 1, this%sdgf
              n = n + 1
              ! only m entries in row m have to be calculated for lower matrix
              if (n>m) exit o2tp_h
              this%KM_lin(os+n) = kernel_d3(this, this%xs(:,i), this%xs(:,j), &
                                            (/1,1,2/), hd1, hd2, d1)
            end do
            ! h - sdgf(sdgf+1)/2*h entries
            do d1 = 1, this%sdgf
              do d2 = d1, this%sdgf!1, d1
                n = n + 1
                ! only m entries in row m have to be calculated for lower matrix
                if (n>m) exit o2tp_h
                this%KM_lin(os+n) = kernel_d4(this, this%xs(:,i), this%xs(:,j),&
                                              (/ 2,2,1,1 /), d1, d2, hd1, hd2)
              end do
            end do
          end do o2tp_h
          this%KM_lin(os+m) = this%KM_lin(os+m) + this%s_n(3)**2
        end do
      end do
    end do o2tp1
  else if (this%order==42) then
    mOffset = 0
    do i = 1, startNt-1
      SELECT CASE(this%order_tps(i))
      CASE(0)
        mOffset = mOffset + 1
      CASE(1)
        mOffset = mOffset + this%sdgf + 1
      CASE(2)
        mOffset = mOffset + (this%sdgf+1)*this%sdgf/2 + this%sdgf + 1
      CASE DEFAULT
        call dlf_fail("iChol not implemented for this order of data point!")
      END SELECT
    end do
    m = mOffset+1 ! row m
    do i=startNt,this%nt ! run over all training points
      os = (m*(m-1))/2 ! size of the KM that already exists
      ! energy value of a training point always exists:
      
      ! we have a new training point that is of order 0
      ! -> only one new row for this trainingpoint
      ! run over all trianingpoints before and including the new one
      n = 0 ! begin with the first entry after os (column n of KM)
      i42tp0: do j = 1, i ! run over all training points in lower triangular matrix
        ! we always have one entry (energy is always present)
        ! entry m,n of the KM matrix (index os+n), training point i, j
        n = n + 1 ! advance only one entry
        if (n>m) exit i42tp0
        this%KM_lin(os+n) = kernel(this,this%xs(:,i),this%xs(:,j))  
        
        if (this%order_tps(j)>=1) then
          ! we have sdgf additional entries from the gradient of tp j
          ! entries m,n of the KM matrix (index os+n), training point i, j
          diff = this%xs(:,i)-this%xs(:,j)
          do j2 = 1, this%sdgf            
            n = n + 1
            if (n>m) exit i42tp0
            absv=norm2(diff)
            this%KM_lin(os+n) = kernel_d1_exp2(this,diff,j2,absv)
          end do
        end if
        if (this%order_tps(j)==2) then
          ! we have (sdgf+1)sdgf/2 additional entries from the hess of tp j
          ! entries m,n of the KM matrix (index os+n), training point i, j
          do d1 = 1, this%sdgf            
            do d2 = d1, this%sdgf
              n = n + 1
              if (n>m) exit i42tp0
              absv=norm2(diff)
              this%KM_lin(os+n) = kernel_d2_exp22(this,diff,d1,d2,absv)
            end do
          end do
        else if (this%order_tps(j)>2) then 
          call dlf_fail("NOT implemented (iChol's calc_cov_mat_newOrder)!")
        end if
      end do i42tp0
      ! diagonal entries 
      if (i==j) this%KM_lin(os+m) = this%KM_lin(os+m) + this%s_n(1)**2
      m = m + 1 ! we added a new tp with an energy value
      
      if(this%order_tps(i)>=1) then
        ! ************* THE GRADIENT VALUE ************* 
        os = (m*(m-1))/2 ! size of the KM that already exists
        n = 0 ! begin with the first entry after os (column n of KM)
        ! run over all entries of the gradient of training point i
        do gd = 1, this%sdgf
          os = (m*(m-1))/2
          n = 0
          i42tp1: do j = 1, i ! run over all training points in lower triangular matrix
            ! we always have one gradient/energy entry
            n = n + 1
            if (n>m) exit i42tp1
            diff = this%xs(:,i)-this%xs(:,j)
            absv=norm2(diff)
            this%KM_lin(os+n) = kernel_d1_exp1(this,diff,gd,absv)
            ! diagonal entries are not possible for this combination
            ! only for gradient/gradient entries like they are calculated now
            if (this%order_tps(j)>=1) then
              do d1 = 1, this%sdgf
                n = n + 1
                if (n>m) exit i42tp1
                this%KM_lin(os+n) = kernel_d2_exp12(this,diff,gd,d1,absv)     
              end do
            end if
            if (this%order_tps(j)==2) then
              ! we have (sdgf+1)sdgf/2 additional entries from the hess of tp j
              ! entries m,n of the KM matrix (index os+n), training point i, j
              do d1 = 1, this%sdgf            
                do d2 = d1, this%sdgf
                  n = n + 1
                  if (n>m) exit i42tp1
                  this%KM_lin(os+n) = kernel_d3(this, this%xs(:,i), this%xs(:,j), (/2,2,1/), d1, d2, gd)
                end do
              end do
            else if (this%order_tps(j)>2) then
              call dlf_fail("NOT implemented (iChol's calc_cov_mat_newOrder)!")
            end if
          end do i42tp1
          ! these are the diagonal elements of gradients
          this%KM_lin(os+m) = this%KM_lin(os+m) + this%s_n(2)**2
          m = m + 1 ! added one new row to the KM
        end do
      end if
      if(this%order_tps(i)==2) then
        do hd1 = 1, this%sdgf
          do hd2 = hd1, this%sdgf
            os = (m*(m-1))/2
            n = 0
            i42tp2: do j = 1, i ! run over all training points in lower triangular matrix
              ! we always have one gradient/energy entry
              n = n + 1
              if (n>m) exit i42tp2
              diff = this%xs(:,i)-this%xs(:,j)
              absv=norm2(diff)
              this%KM_lin(os+n) = kernel_d2_exp11(this,diff,hd1,hd2,absv)
              ! diagonal entries are not possible for this combination
              ! only for gradient/gradient entries like they are calculated now
              if (this%order_tps(j)>=1) then
                do d1 = 1, this%sdgf
                  n = n + 1
                  if (n>m) exit i42tp2
                  this%KM_lin(os+n) = kernel_d3(this, this%xs(:,i), this%xs(:,j), &
                                            (/1,1,2/), hd1, hd2, d1)     
                end do
              end if
              if (this%order_tps(j)==2) then
                ! we have (sdgf+1)sdgf/2 additional entries from the hess of tp j
                ! entries m,n of the KM matrix (index os+n), training point i, j
                do d1 = 1, this%sdgf            
                  do d2 = d1, this%sdgf
                    n = n + 1
                    if (n>m) exit i42tp2
                    this%KM_lin(os+n) = kernel_d4(this, this%xs(:,i), this%xs(:,j),&
                                              (/ 2,2,1,1 /), d1, d2, hd1, hd2)
                  end do
                end do
              else if (this%order_tps(j)>2) then
                call dlf_fail("NOT implemented (iChol's calc_cov_mat_newOrder)!")
              end if
            end do i42tp2
            ! these are the diagonal elements of gradients
            this%KM_lin(os+m) = this%KM_lin(os+m) + this%s_n(3)**2
            m = m + 1 ! added one new row to the KM
          end do
        end do
      end if
      if (this%order_tps(i)>2) then
        call dlf_fail("iChol not implemented for this order of data point!")
      end if
    end do ! i = startNt, this%nt
  else 
    call dlf_fail("This order of GPR is not supported.")
  end if
else
    call dlf_fail("The requested type of kernel is not implemented! (calc_cov_mat_newOrder)")
end if
if (present(reuse_old_in).and.reuse_old) then
  if(reuse_old) then
    this%K_stat = 7
  endif
else
  this%K_stat = 0
end if

end subroutine calc_cov_mat_newOrder                                     
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/calc_cov_mat_dg_or_l
!!
!! FUNCTION
!! Calculate elementwise derivative of the 
!! Covariance matrix wrt. gamma/l for SE/Matern
!!
!! SYNOPSIS
subroutine calc_cov_mat_dg_or_l(this, A)    
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    real(rk), intent(out)::  A(this%nk,this%nk) !Cov_matr
    integer ::              i, j, k, l, m, n, osx, osy, y, x, os, os2
    integer ::              dgf, nr
    real(rk)             ::  xm(this%sdgf), xn(this%sdgf)
    real(rk)             ::  dk_dx
    integer              ::  b_i,b_j
    dgf = this%idgf
    if (this%iChol) call dlf_fail("iChol and calc_cov_mat_dg_or_l not compatible.")
if (this%kernel_type<=1) then
! using squared exponential kernel
    A(:,:)=0d0
    ! Calculate Cov11
    osx = 0
    osy = 0
    do n = 1, this%nt
      xn = this%xs(:,n)
      do m = 1, this%nt
        xm = this%xs(:,m)
        A(m,n) = kernel_dg(this,xm,xn)
      end do
    end do
  if (this%order>0) then
    ! Calculate Cov 12
    osy = 0
    osx = this%nt
    do n = 1, this%nt
      xn = this%xs(:,n)
      do m = 1, this%nt
        xm = this%xs(:,m)
        do i = 1, dgf
          x = osx+(n-1)*dgf+i
          y = osy+m
          if (this%internal == 2) then
            dk_dx = 0.0d0
            do b_i=1,this%sdgf
              dk_dx = dk_dx + kernel_d1_dg(this,xm,xn,2,b_i) * this%b_matrix(b_i,i,n)
            enddo
            A(y,x) = dk_dx
            A(x,y) = A(y,x)
          else
            A(y,x) = kernel_d1_dg(this,xm,xn,2,i)
            A(x,y) = A(y,x)
          endif
        end do            
      end do
    end do        
    ! Calculate Cov 22
    osy = this%nt
    osx = this%nt
    do n = 1, this%nt
      xn = this%xs(:,n)
      do m = 1, this%nt
        xm = this%xs(:,m)
        do i = 1, dgf
          do j = 1, dgf
            x = osx+(n-1)*dgf+j
            y = osy+(m-1)*dgf+i
            if(this%internal == 2) then
              dk_dx = 0.0d0
              do b_i=1,this%sdgf
                do b_j = 1, this%sdgf
                  dk_dx = dk_dx + (this%b_matrix(b_i,i,m) * &
                    kernel_d2_dg(this,xm,xn,(/1,2/),b_i,b_j)*  this%b_matrix(b_j,j,n))
                enddo
              enddo
              A(y,x) = dk_dx
            else
              A(y,x) = kernel_d2_dg(this,xm,xn,(/1,2/),i,j)
            endif
          end do
        end do
      end do
    end do
  if (this%order==2) then
    ! Calculate Cov13
    osx = this%nt+this%nt*dgf
    osy = 0
    do n = 1, this%nt
      xn = this%xs(:,n)
      do m = 1, this%nt
        xm = this%xs(:,m)
        ! The matrix starts at offset "osx"+1 in x direction    
        os = -this%sdgf
        do j = 1, dgf
          os = os + this%sdgf-(j-1)
          do i = j, dgf
            x = osx+(n-1)*dgf*(dgf+1)/2+os+i
            y = m
            !Element A_mn
            A(y,x) = kernel_d2_dg(this,xm,xn,(/2,2/),i,j)   
            A(x,y) = A(y,x)
          end do
        end do            
      end do
    end do    
    ! Calculate Cov23
    osx = this%nt+this%nt*dgf
    osy = this%nt
    do n = 1, this%nt
        xn = this%xs(:,n)
        do m = 1, this%nt
            xm = this%xs(:,m)
            ! The matrix starts at offset "osy"+1 in y direction & "osx"+1 in x
            os = -this%sdgf
            do j = 1, dgf
                os = os + this%sdgf-(j-1)
                do i = j, dgf
                    do k = 1, dgf
                        x = osx+(n-1)*dgf*(dgf+1)/2+os+i
                        y = osy+(m-1)*dgf+k                        
                        !Element A_mn
                        A(y,x) = kernel_d3_dg(this,xm,xn,(/2,2,1/),i,j,k)
                        ! Cov32
                        A(x,y) = A(y,x)
                    end do
                end do
            end do            
        end do
    end do    
    ! Calculate Cov33
    osx = this%nt+this%nt*dgf
    osy = this%nt+this%nt*dgf
    nr = dgf*(dgf+1)/2
    do n = 1, this%nt
        xn = this%xs(:,n)
        do m = 1, this%nt
            xm = this%xs(:,m)
            ! The matrix starts at offset "osy"+1 in y direction & "osx"+1 in x
            os = -this%sdgf
            do i = 1, dgf
                os = os + this%sdgf-(i-1)
                do j = i, dgf
                    os2 = -this%sdgf
                    x = osx+(n-1)*nr+os+j
                    do k = 1, dgf
                        os2 = os2 + this%sdgf-(k-1)
                        do l = k, dgf               
                            if (x>=y) then
                                y = osy+(m-1)*nr+os2+l
                                !Element A_mn
                                A(y,x) = kernel_d4_dg(this,xm, xn, &
                                                        (/2,2,1,1/), i, j, k, l)
                            end if
                        end do
                    end do 
                end do
            end do            
        end do
    end do
        do n = 1, this%nt
        xn = this%xs(:,n)
        do m = 1, this%nt
            xm = this%xs(:,m)
            ! The matrix starts at offset "osy"+1 in y direction & "osx"+1 in x
            os = -this%sdgf
            do i = 1, dgf
                os = os + this%sdgf-(i-1)
                do j = i, dgf
                    os2 = -this%sdgf
                    x = osx+(n-1)*nr+os+j
                    do k = 1, dgf
                        os2 = os2 + this%sdgf-(k-1)
                        do l = k, dgf                            
                            y = osy+(m-1)*nr+os2+l
                            !Element A_mn
                            if (y>x) then
                                A(y,x) = A(x,y)
                            end if
                        end do
                    end do 
                end do
            end do            
        end do
    end do
  end if ! Order 2
  end if ! Order 1
else
    call dlf_fail("The requested type of kernel is not implemented! (calc_cov_mat_dg_or_l)")
end if
end subroutine calc_cov_mat_dg_or_l        
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_eval
!!
!! FUNCTION
!! Infer an energy scalar at the position "eval_pos", i.e. evaluate the GP
!!
!! SYNOPSIS
recursive subroutine GPR_eval(this, eval_pos, result, lower_level_result)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)    ::  this
    real(rk), intent(in)            ::  eval_pos(this%sdgf)
    real(rk), intent(out)           ::  result
    real(rk), intent(in),optional   ::  lower_level_result ! Only for offsetType=5
    real(rk)                        ::  tmp
    integer                         ::  i, d, d2, os, os2, wPos, blocksize,j
    real(rk)                        ::  xm(this%sdgf), xn(this%sdgf)   
    real(rk)                        ::  offset, absv
    logical                         ::  calculate
    real(rk)                        ::  gSum, hSum
    real(rk)                        ::  dk_dx
    
    calculate = .true. ! never touched if not order==42
    if (this%OffsetType==5) then
      if (present(lower_level_result)) then
        offset=lower_level_result
      else
        call GPR_eval(this%lowerLevelGPR, eval_pos, offset)
      end if
    else if (this%OffsetType==7) then
        offset = this%es(1)+(eval_pos(1)-this%xs(1,1))*&
                 (this%es(this%nt)-this%es(1))/&
                 (this%xs(1,this%nt)-this%xs(1,1))
!         call GPR_linearBias(this,eval_pos,offset)
    else if (this%OffsetType==8) then
        call GPR_taylorBias(this,eval_pos,offset)
    else
        offset=this%mean
    end if
    
    result = 0d0
    if (.not.this%w_is_solution) call dlf_fail("Solve the system KM*w=y first! (GPR_eval)")
     
  if (.not.this%iChol) then
    if (this%order>=0) then
      if (this%kernel_type==1) then
        do i = 1, this%nt    
          absv = dsqrt(dot_product(eval_pos(:)-this%xs(:,i),&
                                   eval_pos(:)-this%xs(:,i)))
          result = result + this%w(i) * &
                            (this%s_f2*(1d0 + (dsqrt(5d0)*absv)/this%l + &
                            (5d0*absv**2)/(3d0*this%l**2))/&
                            dexp((dsqrt(5d0)*absv/this%l)))
        end do
      else if (this%kernel_type==0) then
        do i = 1, this%nt    
          result = result + this%w(i) * &
                            this%s_f2*dexp(-this%gamma/2d0*&
                            dot_product(eval_pos(:)-this%xs(:,i),&
                                        eval_pos(:)-this%xs(:,i)))
        end do
      else
        do i = 1, this%nt    
          result = result + this%w(i) * &
                            kernel(this,eval_pos(:), this%xs(:,i))  
        end do
      end if
    end if
    
    os = this%nt
    if (this%order >= 1) then
      do i = 1, this%nt
        if (this%order/=42.or.this%order_tps(i)>=1) then
!           os = os + this%sdgf
          xm(:) = eval_pos(:) - this%xs(:,i) ! diff
          absv = norm2(xm)
!           os = this%nt+(i-1)*this%sdgf
          do d = 1, this%idgf
            os = os + 1
            dk_dx= 0.0d0
            if(this%internal == 2) then
              do j=1,this%sdgf
                dk_dx = dk_dx + kernel_d1_exp2(this,xm,j,absv)*this%b_matrix(j,d,i)
              enddo
              result = result + this%w(os)*dk_dx  
            else    
              result = result + this%w(os)*&
                              kernel_d1_exp2(this, xm, d, absv)
!                               kernel_d1(this,eval_pos(:), this%xs(:,i),2,d)
            endif
          end do
        end if
      end do
    end if
    
    if (this%order >= 2) then
!       os = this%nt+this%nt*this%sdgf
      xm = eval_pos
      ! os contains the position where to write (-1)
!       os2 = 0
      do i = 1, this%nt
        if (this%order/=42.or.this%order_tps(i)>=2) then
          xn = this%ixs(:,i)
!           os2 = -this%sdgf
          do d2 = 1, this%idgf
!             os2 = os2 + this%sdgf-(d2-1)
            do d = d2, this%idgf
              os = os + 1
              tmp = kernel_d2(this,xm,xn,(/2,2/),d,d2)     
              result = result + &
                       (this%w(os)*tmp)
!                         (this%w(os+(i-1)*this%sdgf*(this%sdgf+1)/2&
!                         +os2+d)*tmp)    
            end do
          end do
        end if
      end do
    end if
  else 
    ! my own cholesky decomposition (iChol) is used

    if (this%order==0) then
      do i = 1, this%nt    
        result = result + this%w(i) * &
                          kernel(this,eval_pos(:), this%xs(:,i))  
      end do
    else if (this%order==1) then
      ! energies
      blocksize = (this%idgf+1)
      do i = 1, this%nt
        result = result + this%w((i-1)*blocksize+1) * &
                          kernel(this,eval_pos(:), this%xs(:,i))  
      end do
      ! gradients
      do i = 1, this%nt
        os = (i-1)*(this%idgf+1)+1
        xm(:) = eval_pos(:) - this%xs(:,i) ! diff
        absv = norm2(xm)
        do d = 1, this%idgf
          dk_dx=0.0d0
          if(this%internal == 2) then
            do j=1, this%sdgf
              dk_dx = dk_dx + kernel_d1_exp2(this, xm, j, absv)*this%b_matrix(j,d,i)
            enddo
            result = result+ this%w(os + d) * dk_dx
          else
            result = result + this%w(os + d) * &
                            kernel_d1_exp2(this, xm, d, absv)
          endif                  
        end do
      end do
    else if (this%order==2) then
      ! energies
      blocksize = ((this%sdgf)*(this%sdgf+1)/2 + this%sdgf + 1)
      do i = 1, this%nt
        result = result + this%w((i-1)*blocksize+1) * &
                          kernel(this,eval_pos(:), this%xs(:,i))
      end do
      ! gradients
      blocksize = ((this%sdgf)*(this%sdgf+1)/2 + this%sdgf + 1)
      do i = 1, this%nt
        os = (i-1)*blocksize + 1
        xm(:) = eval_pos(:) - this%xs(:,i) ! diff
        absv = norm2(xm)
        do d = 1, this%sdgf
          result = result + this%w(os + d) * &
                            kernel_d1_exp2(this, xm, d, absv)
        end do
      end do
      ! hessians
      blocksize = ((this%sdgf)*(this%sdgf+1)/2 + this%sdgf + 1)
      do i = 1, this%nt
        os = (i-1)*blocksize + this%sdgf + 1
        os2 = 0
        do d = 1, this%sdgf
          do d2 = d, this%sdgf!1, d
            result = result + this%w(os + os2 + d2) * &
                     kernel_d2(this,eval_pos(:),this%xs(:,i),(/2,2/),d,d2)
!                               kernel_d1_exp2(this, xm, d, absv)
          end do
          os2 = os2 + this%sdgf-d ! number of elements written - 1
          ! (the next d2 loop starts at d+1 -> shift)
        end do
      end do
    else if (this%order==42) then      
      wPos = 1
      do i = 1, this%nt
        gSum = 0d0
        hSum = 0d0
        result = result + this%w(wPos) * &
                          kernel(this,eval_pos(:), this%xs(:,i))
        wPos = wPos + 1        
        if (this%order_tps(i)>=1) then
          xm(:) = eval_pos(:) - this%xs(:,i) ! diff
          absv = norm2(xm)
          do d = 1, this%sdgf
            gSum = gSum + this%w(wPos) * &
                              kernel_d1_exp2(this, xm, d, absv)
!             result = result + this%w(wPos) * &
!                               kernel_d1_exp2(this, xm, d, absv)
            wPos = wPos + 1
          end do
        end if
        if (this%order_tps(i)==2) then
          do d = 1, this%sdgf
            do d2 = d, this%sdgf!1, d
              hSum = hSum + this%w(wPos) * &
                       kernel_d2(this,eval_pos(:),this%xs(:,i),(/2,2/),d,d2)
  !                               kernel_d1_exp2(this, xm, d, absv)
              wPos = wPos + 1
            end do
            os2 = os2 + this%sdgf-d ! number of elements written - 1
            ! (the next d2 loop starts at d+1 -> shift)
          end do
        else if ((this%order_tps(i)>2).and.(this%order_tps(i)/=42)) then
          call dlf_fail("This order type is not valid for the training point.!")
        end if        
        result = result + gSum + hSum! numerically more stable to sum like this
      end do
    else
      call dlf_fail("This order is not supported in GPR_eval!")
    end if
  end if    
  result = result + offset   
end subroutine GPR_eval    
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_eval_giveLower
!!
!! FUNCTION
!! Infer an energy scalar at the position "eval_pos", i.e. evaluate the GP
!! if one has given the energy of a lower level already and one can
!! give a pointer to the GP on that level.
!!
!! SYNOPSIS
recursive subroutine GPR_eval_giveLower(this, eval_pos, result, &
    lower_level_result, pointToLevel)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::  this
    real(rk), intent(in)        ::  eval_pos(this%idgf)
    real(rk), intent(out)       ::  result
    ! For offsetType=5 one can give the result of the lower level at eval_pos
    real(rk), intent(in)        ::  lower_level_result
    ! If it is not the very next lower level one can give a pointer
    ! to the GP on the level of which the result is known.
    type(gpr_type), pointer     ::  pointToLevel
    real(rk)                    ::  offset
    if (this%order==42) call dlf_fail("GPR_eval_giveLower not implemented for order 42!")
    if (this%OffsetType==5) then
      if (associated(pointToLevel, target=this%lowerLevelGPR)) then
        call GPR_eval (this, eval_pos, result, lower_level_result)
      else
          call GPR_eval_giveLower(this%lowerLevelGPR, eval_pos, offset, &
                        lower_level_result, pointToLevel)
          call GPR_eval (this, eval_pos, result, offset)
      end if
    else 
        if (associated(this%lowerLevelGPR)) &
            call dlf_fail("That shoud not happen. (GPR_eval_giveLower)")
        call GPR_eval(this, eval_pos, result)
    end if
end subroutine GPR_eval_giveLower    
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_eval_grad
!!
!! FUNCTION
!! Infer a Gradient vector at the position "eval_pos"
!!
!! SYNOPSIS
recursive subroutine GPR_eval_grad(this, eval_pos, result,bmatrix, turnOffsetOff, &
                                   lower_level_result)   
!! SOURCE                                   
    implicit none
    type(gpr_type),intent(inout)    ::  this
    real(rk), intent(in)            ::  eval_pos(this%sdgf)
    real(rk), intent(out)           ::  result(this%idgf)
    real(rk), intent(in),optional   ::  bmatrix(this%sdgf,this%idgf)
    logical, intent(in), optional   ::  turnOffsetOff  ! only relevant for 
                                                    ! offsetType 6
    ! For offsetType=5 one can give the result of the lower level at eval_pos
    real(rk), intent(in),optional   ::  lower_level_result(this%idgf)
    real(rk)                        ::  diff(this%sdgf)
    real(rk)                        ::  lowerLevelContribution(this%idgf)
    integer                         ::  n, i, j,k,os, wi, os2,l
    real(rk)                        ::  absv!, originalres(this%sdgf)
    real(rk)   :: matrix(this%idgf,this%idgf)
    real(rk)   :: C(this%sdgf,this%idgf)
    real(rk)   :: matrix_neu(this%sdgf,this%sdgf)
    real(rk)   :: bmatrix_T(this%idgf,this%sdgf)
    real(rk)                        ::  result_test(this%idgf)
    real(rk)                        ::  factor
    real(rk)                        ::  dk_dx
    
    if (.not.this%w_is_solution) call dlf_fail("Solve the system KM*w=y first! (GPR_eval)")
    result(:) = 0d0
    result_test(:) = 0d0
    lowerLevelContribution(:)=0d0
    if (present(lower_level_result)) then
      lowerLevelContribution(:)=lower_level_result(:)
    else
      if (this%OffsetType==5) then
        if(this%internal == 2) then
          call GPR_eval_grad(this%lowerLevelGPR, eval_pos, lowerLevelContribution,bmatrix)
        else
          call GPR_eval_grad(this%lowerLevelGPR, eval_pos, lowerLevelContribution)
        endif
        result(:)=result(:)+lowerLevelContribution(:)
      else if (this%OffsetType==7) then
        if (present(turnOffsetOff)) then
          if (turnOffsetOff) then
            result(:) = 0d0
          else
            ! call GPR_linearBias_d(this,result)
            ! only one dimension can be set with OffsetType==7
            result(1) = (this%es(this%nt)-this%es(1))/&
                        (this%xs(1,this%nt)-this%xs(1,1))
          end if
        else
          ! call GPR_linearBias_d(this,result)
          ! only one dimension can be set with OffsetType==7
          result(1) = (this%es(this%nt)-this%es(1))/&
                      (this%xs(1,this%nt)-this%xs(1,1))
        end if
      else if (this%OffsetType==8) then
        if (present(turnOffsetOff)) then
          if (turnOffsetOff) then
            result(:) = 0d0
          else
            call GPR_taylorBias_d(this,eval_pos,result)
          end if
        else
          call GPR_taylorBias_d(this,eval_pos,result)
        end if
      end if
    endif
  if (.not.this%iChol) then  
    if (this%order==1) then
            do k = 1, this%idgf                
                do n = 1, this%nt
                   ! print *,eval_pos
                    diff(:)=eval_pos(:)-this%xs(:,n)
                    absv=dsqrt(dot_product(diff,diff))
                    if (this%internal == 2)then
                      dk_dx =0.0d0
                      do j=1,this%sdgf
                        dk_dx = dk_dx + kernel_d1_exp1(this,diff,j,absv)*bmatrix(j,k)
                      enddo 
                      result(k) = result(k) + this%w(n)*dk_dx
                    else
                      result(k) = result(k) + this%w(n)*&
                                kernel_d1_exp1(this,diff,k,absv)
                    endif    
                    do i = 1, this%idgf
                      if (this%internal == 2) then
                        dk_dx =0.0d0
                        do j=1,this%sdgf
                          do l=1,this%sdgf
                            dk_dx = dk_dx + bmatrix(j,k)*kernel_d2_exp12(this,diff,j,l,absv)*this%b_matrix(l,i,n)
                          enddo
                        enddo
                        result(k) = result(k) + this%w(this%nt+(n-1)*this%idgf+i)*dk_dx!matrix(k,i)
                      else
                        result(k) = &
                            result(k) + this%w(this%nt+(n-1)*this%idgf+i)*&
                            kernel_d2_exp12(this,diff,k,i,absv)  
                      endif         
                    end do
                    
                end do
            end do        
    else  
      if (this%kernel_type==1) then
        do n = 1, this%nt
            diff(:) = eval_pos(:)-this%xs(:,n)
            absv=dsqrt(dot_product(diff,diff))
            factor = -5d0*this%s_f2*(this%l + dsqrt(5d0)*absv)/&
                            (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**3)    
            do k = 1, this%idgf
                result(k) = result(k) + this%w(n)*&
                           diff(k)*factor
            end do
        end do  
      else if (this%kernel_type==0) then
        do n = 1, this%nt
            diff(:) = this%xs(:,n)-eval_pos(:)
            absv=(dot_product(diff,diff)) ! not the absolute value but the squared
            factor = this%s_f2*this%gamma*dexp(-this%gamma/2d0*absv)  
            do k = 1, this%idgf
                result(k) = result(k) + this%w(n)*((diff(k))*factor)
            end do
        end do        
      else 
        do k = 1, this%idgf
            do n = 1, this%nt
                result(k) = result(k) + this%w(n)*&
                            kernel_d1(this,eval_pos(:),this%xs(:,n),1,k)                    
            end do
        end do  
      end if
        wi = 0
        if (this%order > 0) then
            os = this%nt
            do k = 1, this%idgf
              wi = os
                do n = 1, this%nt
                  if (this%order/=42.or.this%order_tps(n)>=1) then
                    do i = 1, this%idgf
                      wi = wi + 1
!                         wi = os+(n-1)*this%sdgf+i
                        result(k) = result(k) + this%w(wi)*&
                                    kernel_d2(this,eval_pos(:), &
                                            this%xs(:,n),(/1,2/),k,i)
                    end do
                  end if
                end do
            end do
        end if
        os = wi
        if (this%order > 1) then
!             os = this%nt+this%nt*this%sdgf
          do k = 1, this%idgf
            wi = os
            do n = 1, this%nt   
              if (this%order/=42.or.this%order_tps(n)>=2) then
!                 os2 = -this%sdgf
                do i = 1, this%idgf
!                   os2 = os2 + this%sdgf-(i-1)
                  do j = i, this%idgf
!                     wi = os+(n-1)*this%sdgf*(this%sdgf+1)/2+os2+j
                    wi = wi + 1
                    result(k) = result(k)+this%w(wi)*&
                                kernel_d3(this,eval_pos(:), &
                                          this%ixs(:,n),(/2,2,1/),i,j,k)
                  end do
                end do
              end if
            end do
          end do  
        end if
    end if
  else
    ! my own cholesky decomposition is used
    if (this%order==42) then
      os = 0
      do n = 1, this%nt
        diff(:) = eval_pos(:)-this%xs(:,n)
        absv=dsqrt(dot_product(diff,diff))
        do k = 1, this%sdgf
          os2 = 0
          result(k) = result(k) + this%w(os+os2+1)*&
                      kernel_d1_exp1(this,diff,k,absv)
          os2 = os2 + 1
          if (this%order_tps(n)>=1) then
            do i = 1, this%sdgf
              result(k) = result(k) + &
                this%w(os+os2+1)*kernel_d2_exp12(this,diff,k,i,absv)
              os2 = os2 + 1
            end do
          end if
          if (this%order_tps(n)==2) then
            do i = 1, this%sdgf
              do j = i, this%sdgf
                result(k) = result(k) + &
                    this%w(os+os2+1)*&
                    kernel_d3(this,eval_pos,this%xs(:,n),(/2,2,1/),i,j,k)
                os2 = os2 + 1
              end do
            end do
          else if (this%order_tps(n)>2) then
            call dlf_fail("This training point type should not exist (GPR_eval_grad).")
          end if
        end do
        os = os + os2
      end do
    else if (this%order==0) then
      do n = 1, this%nt
        diff(:)=eval_pos(:)-this%xs(:,n)
        absv=dsqrt(dot_product(diff,diff))
        do k = 1, this%sdgf
          result(k) = result(k) + this%w(n)*&
                      kernel_d1_exp1(this,diff,k,absv)
        end do
      end do
    else if (this%order==1) then ! optimized for this order      
      do n = 1, this%nt
        diff(:) = eval_pos(:)-this%xs(:,n)
        absv=dsqrt(dot_product(diff,diff))
        os = (n-1)*(this%idgf+1)
        if(this%kernel_type==1) then
           ! matrix version by JK
           do k = 1, this%idgf
              ! energies
              if(this%internal == 2) then
                dk_dx = 0.0d0
                  do j=1,this%sdgf
                    dk_dx = dk_dx + kernel_d1_exp1(this,diff,j,absv)*bmatrix(j,k) 
                  enddo
                  result(k) = result(k) +  this%w(os+1)* dk_dx
              else
                result(k) = result(k) + this%w(os+1)*&
                   kernel_d1_exp1(this,diff,k,absv)
              endif
           end do
          if (this%internal == 2) then              
           call kernel_matern_d2_exp12_matrix(this,diff,absv,matrix_neu) ! evaluate the kernel as a matrix. Faster than individually.
                                                               ! because symmetry is exploited.
             C =0.0d0  
             bmatrix_T =0.0d0                                                   
             call dgemm('N','N',this%sdgf,this%idgf,this%sdgf,1.0d0,matrix_neu,this%sdgf,&
                        this%b_matrix(:,:,n),this%sdgf,0.0d0,C,this%sdgf)
             bmatrix_T = transpose(bmatrix)
             call dgemm('N','N',this%idgf,this%idgf,this%sdgf,1.0d0,bmatrix_T,this%idgf,&
                         C,this%sdgf,0.0d0,matrix,this%idgf)                                            
           do k = 1, this%idgf
              do i = 1, this%idgf
                 result(k) = result(k) + this%w(os+1+i)*matrix(k,i)
              end do
           end do
           else 
              call kernel_matern_d2_exp12_matrix(this,diff,absv,matrix_neu) 
           do k = 1, this%sdgf
              do i = 1, this%sdgf
                 result(k) = result(k) + this%w(os+1+i)*matrix_neu(i,k)
              end do
           end do              
           endif
        else ! older, non-matrix version
        do k = 1, this%sdgf
          ! energies
          result(k) = result(k) + this%w(os+1)*&
                      kernel_d1_exp1(this,diff,k,absv)
          ! gradients
          do i = 1, this%sdgf
            result(k) = result(k) + &
                ! JK: in minimization, this call requires 50% of the CPU time.
                this%w(os+1+i)*kernel_d2_exp12(this,diff,k,i,absv)
          end do
        end do
        end if ! matrix version
      end do      
    else if (this%order==2) then ! optimized for this order      
      do n = 1, this%nt
        diff(:) = eval_pos(:)-this%xs(:,n)
        absv=dsqrt(dot_product(diff,diff))
        os = (n-1)*((this%sdgf+1)*this%sdgf/2+this%sdgf+1)
        do k = 1, this%sdgf
          ! energies
          result(k) = result(k) + this%w(os+1)*&
                      kernel_d1_exp1(this,diff,k,absv)
          ! gradients
          do i = 1, this%sdgf
            result(k) = result(k) + &
                this%w(os+1+i)*kernel_d2_exp12(this,diff,k,i,absv)
          end do
          
          ! hessians
          os2 = 0
          do i = 1, this%sdgf
            do j = i, this%sdgf
              result(k) = result(k) + &
                  this%w(os+this%sdgf+1+os2+j)*&
                  kernel_d3(this,eval_pos,this%xs(:,n),(/2,2,1/),i,j,k)
            end do
            os2 = os2 + this%sdgf-i
          end do
        end do
      end do  
    else
      call dlf_fail("This order is not implemented for GPR_eval_grad!")
    end if ! order
  end if ! my own cholesky
end subroutine GPR_eval_grad
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_eval_grad_giveLower
!!
!! FUNCTION
!! Infer an energy scalar at the position "eval_pos", i.e. evaluate the GP
!! if one has given the energy of a lower level already and one can
!! give a pointer to the GP on that level.
!!
!! SYNOPSIS
recursive subroutine GPR_eval_grad_giveLower(this, eval_pos, result, &
    lower_level_result, pointToLevel)
!! SOURCE    
    implicit none
    type(gpr_type),intent(inout)::  this
    real(rk), intent(in)        ::  eval_pos(this%sdgf)
    real(rk), intent(out)       ::  result(this%sdgf)
    ! For offsetType=5 one can give the result of the lower level at eval_pos
    real(rk), intent(in)        ::  lower_level_result(this%sdgf)
    ! If it is not the very next lower level one can give a pointer
    ! to the GP on the level of which the result is known.
    type(gpr_type), pointer     ::  pointToLevel
    real(rk)                    ::  offset(this%sdgf)
    real(rk)                    ::  tmp_mat(this%sdgf,this%idgf)
    if (this%order==42) call dlf_fail("GPR_eval_grad_giveLower not implemented for order 42!")
    if (this%OffsetType==5) then
      if (associated(pointToLevel, target=this%lowerLevelGPR)) then
        call GPR_eval_grad (this, eval_pos, result,tmp_mat, .false., lower_level_result)
      else
          call GPR_eval_grad_giveLower(this%lowerLevelGPR, eval_pos, offset, &
                        lower_level_result, pointToLevel)
          call GPR_eval_grad (this, eval_pos, result,tmp_mat, .false., offset)
      end if
    else 
        if (associated(this%lowerLevelGPR)) &
            call dlf_fail("That shoud not happen. (GPR_eval_grad_giveLower)")
        call GPR_eval_grad(this, eval_pos, result,tmp_mat)
    end if
end subroutine GPR_eval_grad_giveLower    
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_eval_hess
!!
!! FUNCTION
!! Infer a Hessian matrix at the position "eval_pos"
!! It assumes/produces a symmetric Hessian 
!! -> half the computation time
!!
!! SYNOPSIS
recursive subroutine GPR_eval_hess(this, eval_pos, result)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)    ::  this
    real(rk), intent(in)            ::  eval_pos(this%sdgf)
    real(rk), intent(out)           ::  result(this%sdgf,this%sdgf)
    real(rk)                        ::  diff(this%sdgf), absv
    integer                         ::  n, i, j,k,l,os, wi, os2
    real(rk)                        ::  s_small, s_tmp
    real(rk)                        ::  wK3_tmp, expf
    integer                         ::  dwrt(3)
    if (.not.this%w_is_solution) then
      call dlf_fail("Solve the system KM*w=y first! (GPR_eval_hess)")
    end if  
#ifdef withopenmp
    call GPR_eval_hess_parallel(this, eval_pos, result)
    return
#endif    
    if (this%OffsetType==5) then
        ! do not allow nested spawning of parallel regions
        call GPR_eval_hess(this%lowerLevelGPR, eval_pos, result)
    else if (this%OffsetType==8) then
        call GPR_taylorBias_d2(this,result)
    else ! in case offsetType==7 the second derivative of the offset is 0
        result(:,:) = 0.0d0
    end if
  if (.not.this%iChol) then
    do l = 1, this%sdgf
      do k = l, this%sdgf ! symmetry
        do n = 1, this%nt
          wi = n
          result(k,l) = result(k,l) + this%w(wi)*&
                        kernel_d2(this,eval_pos(:), &
                                  this%xs(:,n),(/1,1/),k,l)
          ! symmetry -> just at the end
        end do
      end do
    end do  
    if (this%order > 0) then
      os = this%nt
      do l = 1, this%sdgf
        do k = l, this%sdgf ! symmetry
          wi = os
          do n = 1, this%nt    
            if (this%order/=42.or.this%order_tps(n)>=1) then              
              do i = 1, this%sdgf
!                 wi = os+(n-1)*this%sdgf+i
                wi = wi + 1
                result(k,l) = result(k,l) + this%w(wi)*&
                              kernel_d3(this,eval_pos(:), this%xs(:,n),&
                                        (/1,1,2/),k,l,i)  
                ! symmetry -> just at the end
              end do
            end if
          end do
        end do
      end do
    end if
!     nr = this%sdgf*(this%sdgf+1)/2
    os = wi
    if (this%order > 1) then
!       os = this%nt+this%nt*this%sdgf
      do l = 1, this%sdgf
        do k = l, this%sdgf ! symmetry
          wi = os
          do n = 1, this%nt   
            if (this%order/=42.or.this%order_tps(n)>=2) then
!               os2 = -this%sdgf
              do i = 1, this%sdgf
!                 os2 = os2 + this%sdgf-(i-1)
                do j = i, this%sdgf
!                   wi = os+(n-1)*nr+os2+j
                  wi = wi + 1
                  result(k,l) = result(k,l)+this%w(wi)*&
                                kernel_d4(this,eval_pos(:), &
                                          this%xs(:,n),(/2,2,1,1/),i,j,k,l)
                  ! symmetry -> just at the end
                end do !j
              end do ! i
            end if
          end do ! n
        end do ! k 
      end do ! l
    end if
  else
    ! my own cholesky decomposition is used
    if (this%order==42) then
      os = 0
      do n = 1, this%nt
        os2 = 0
        diff(:) = eval_pos(:) - this%xs(:,n)
        absv=dsqrt(dot_product(diff,diff))
        
        do l = 1, this%sdgf
          do k = l, this%sdgf ! symmetry   
            result(k,l) = result(k,l) + this%w(os+os2+1)*&
                          kernel_d2_exp11(this,diff,k,l,absv)
          end do
        end do
        os = os + 1
        if (this%order_tps(n)>=1) then
          do l = 1, this%sdgf
            do k = l, this%sdgf
              !gradients
              os2 = 0
              do i = 1, this%sdgf
                dwrt=(/1,1,2/)
                result(k,l) = result(k,l) + this%w(os+os2+1)*&
                              kernel_d3(this,eval_pos(:), this%xs(:,n),&
                                        dwrt,k,l,i)
                os2 = os2 + 1
              end do ! i
            end do ! k
          end do ! l
          os = os + os2
        end if
        if (this%order_tps(n)==2) then
          do l = 1, this%sdgf
            do k = l, this%sdgf ! symmetry
              os2 = 0
              do i = 1, this%sdgf
                do j = i, this%sdgf
                  result(k,l) = result(k,l)+this%w(os+os2+1)*&
                                kernel_d4(this,eval_pos(:), &
                                          this%xs(:,n),(/2,2,1,1/),i,j,k,l)
                  os2 = os2 + 1
                  ! symmetry -> just at the end
                end do !j
              end do ! i
            end do ! k 
          end do ! l          
          os = os + os2
        else if (this%order_tps(n)>2) then
          call dlf_fail("This type of training point is not allowed (GPR_eval_hess)!")
        end if        
      end do
    end if
    if (this%order==0) then
      do n = 1, this%nt
        wi = n
        diff(:) = eval_pos(:) - this%xs(:,n)
        absv=dsqrt(dot_product(diff,diff))
        if (this%OffsetType==0) then
          expf = this%s_f2*this%gamma*&
                dexp(-this%gamma/2d0*absv**2)
        else if (this%OffsetType==1) then
          expf = (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
        end if
        do l = 1, this%sdgf
          if (this%OffsetType==0) then
            k=l
            result(k,l) = (this%gamma*(diff(l))*(diff(k))-1d0)*expf
            do k = l+1, this%sdgf ! symmetry        
              result(k,l) = result(k,l) + this%w(wi)*&
                            (this%gamma*(diff(l))*(diff(k)))*expf
              ! symmetry -> just at the end
            end do
          else if (this%OffsetType==1) then
            k=l
            result(k,l) = (-5d0*(this%l**2 - 5d0*(diff(k))**2 + &
                dsqrt(5d0)*this%l*absv))/&
                expf
            do k = l+1, this%sdgf ! symmetry        
              result(k,l) = result(k,l) + this%w(wi)*&
                            (25d0*(diff(k))*(diff(l)))/&
                            expf
              ! symmetry -> just at the end
            end do                        
          else
            do k = l, this%sdgf ! symmetry        
              result(k,l) = result(k,l) + this%w(wi)*&
                            kernel_d2_exp11(this,diff,k,l,absv)
!                             kernel_d2(this,eval_pos(:), &
!                                       this%xs(:,n),(/1,1/),k,l)
              ! symmetry -> just at the end
            end do
          end if
        end do
      end do
    else if (this%order==1) then
      if(this%kernel_type==1) then ! only for matern kernel
        do n = 1, this%nt
          wi = (n-1)*(this%sdgf+1)+1 ! position of energies
          diff(:) = eval_pos(:) - this%xs(:,n)
          absv=dsqrt(dot_product(diff,diff))
          if (absv<1d-16) then
              absv = 1d-16 
          end if
          expf = (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
          ! energies
          do l = 1, this%sdgf
            k = l
            result(l,k) = result(l,k) + this%w(wi)*&
                (-5d0*(this%l**2 - 5d0*(diff(k))**2 + &
                dsqrt(5d0)*this%l*absv))/&
                (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
            do k = l+1, this%sdgf
              !energies
              result(l,k) = result(l,k) + this%w(wi)*&
                          (25d0*(diff(k))*(diff(l)))/&
                          (3d0*dexp((dsqrt(5d0)*absv)/this%l)*this%l**4)
!               result(k,l) = result(l,k)
            end do
          end do
          expf=dexp((dsqrt(5d0)*absv)/this%l)*3d0*absv*this%l**5
          ! gradients
          ! This is done by walking through all permutations of kli,
          ! note that only half of the permutations (plus diagonals)
          ! are calculated because of symmetry.
          do l = 1, this%sdgf
            k=l
            i=k
            wK3_tmp = this%s_f2*(25d0*(diff(i))*&
            (dsqrt(5d0)*(diff(i))**2 - &
             3d0*this%l*absv))/&
            (expf) !kernel_d3(this,eval_pos(:), this%xs(:,n),&
                                      !(/1,1,2/),k,l,i)
            ! lki
            result(l,k) = result(l,k) + wK3_tmp * &
                            this%w(wi+i)
            
            do i = k+1, this%sdgf
              wK3_tmp = this%s_f2*(25d0*(-(absv*this%l) + &
                    dsqrt(5d0)*(diff(k))**2)*(diff(i)))/&
                    (expf) !kernel_d3(this,eval_pos(:), this%xs(:,n),&
                                      !(/1,1,2/),k,l,i)
              ! permutations in which l and k are the same
              ! lik
              result(l,i) = result(l,i) + wK3_tmp * &
                            this%w(wi+k)
              ! lki
              result(l,k) = result(l,k) + wK3_tmp * &
                            this%w(wi+i)
            end do
            do k = l+1, this%sdgf
              i=k
              wK3_tmp = this%s_f2*(25d0*(-(absv*this%l) + &
                    dsqrt(5d0)*(diff(k))**2)*(diff(l)))/&
                    (expf) !kernel_d3(this,eval_pos(:), this%xs(:,n),&
                                      !(/1,1,2/),k,l,i)
              ! permutation in which i and k are the same
              !i=k>l
              ! ikl
              result(i,k) = result(i,k) + wK3_tmp * &
                            this%w(wi+l)
              ! lki
              result(l,k) = result(l,k) + wK3_tmp * &
                            this%w(wi+i)
              ! ! kli
              !if(k<=l) then
              !  result(k,l) = result(k,l) + wK3_tmp * &
              !              this%w(wi+i)
              !end if
              do i = k+1, this%sdgf
                wK3_tmp = this%s_f2*(25d0*dsqrt(5d0)*(diff(k))*&
                    (diff(l))*(diff(i)))/&
                    (expf) !kernel_d3(this,eval_pos(:), this%xs(:,n),&
                                      !(/1,1,2/),k,l,i)
                ! permutations in which i, j, and k are all different
                ! i>k>l
                ! kil
                result(k,i) = result(k,i) + wK3_tmp * &
                            this%w(wi+l)
                ! lik
                result(l,i) = result(l,i) + wK3_tmp * &
                            this%w(wi+k)
                ! lki
                result(l,k) = result(l,k) + wK3_tmp * &
                            this%w(wi+i)
              end do ! i
            end do ! k
          end do ! l
         end do ! n
         ! including symmetry
         do l = 1, this%sdgf
           do k = l+1, this%sdgf            
             result(k,l)=result(l,k)
           end do ! k
         end do  ! l
       else ! other kernel type than matern
        do n = 1, this%nt
          wi = (n-1)*(this%sdgf+1)+1 ! position of energies
          diff(:) = eval_pos(:) - this%xs(:,n)
          absv=dsqrt(dot_product(diff,diff))
          if (absv<1d-16) then
              absv = 1d-16 
          end if
          wi = (n-1)*(this%sdgf+1)+1 ! position of energies
          diff(:) = eval_pos(:) - this%xs(:,n)
          absv=dsqrt(dot_product(diff,diff))
          do l = 1, this%sdgf
            do k = l, this%sdgf
              !energies
              result(k,l) = result(k,l) + this%w(wi)*&
                          kernel_d2_exp11(this,diff,k,l,absv)
              !gradients
              dwrt=(/1,1,2/)
              do i = 1, this%sdgf
                result(k,l) = result(k,l) + this%w(wi+i)*&
                            kernel_d3(this,eval_pos(:), this%xs(:,n),&
                                      dwrt,k,l,i)
              end do ! i
            end do ! k
          end do ! l
        end do ! n
      end if ! kernel type
    else if (this%order==2) then
      do n = 1, this%nt
        wi = (n-1)*(this%sdgf*(this%sdgf+1)/2+this%sdgf+1)+1 ! position of energies
        diff(:) = eval_pos(:) - this%xs(:,n)
        absv=dsqrt(dot_product(diff,diff))
        do l = 1, this%sdgf
          do k = l, this%sdgf
            s_small = 0d0
            !energies
            result(k,l) = result(k,l) + this%w(wi)*&
                        kernel_d2_exp11(this,diff,k,l,absv)
            !gradients
            dwrt = (/1,1,2/)
            do i = 1, this%sdgf
              result(k,l) = result(k,l) + this%w(wi+i)*&
                            kernel_d3(this,eval_pos(:), this%xs(:,n),&
                                      dwrt,k,l,i)
            end do ! i
            !gradients
            os = 0
            do i = 1, this%sdgf
              do j = i, this%sdgf
                s_tmp = this%w(wi+this%sdgf+os+j)*&
                              kernel_d4(this,eval_pos(:), this%xs(:,n),&
                                      (/2,2,1,1/),i,j,k,l)
                if (abs(s_tmp)>1d-10*abs(result(k,l))) then
                  result(k,l) = result(k,l) + s_tmp
                else
                  s_small = s_small + s_tmp
                end if
              end do ! j
              os = os + this%sdgf-i
            end do ! i
            result(k,l) = result(k,l) + s_small
          end do ! k
        end do ! l
      end do ! n
    else if (this%order>2 .and. this%order/=42) then
      call dlf_fail("GPR_eval_hess (order 2) is not implemented with iterative Cholesky.")
    end if       
  end if

  do l = 1, this%sdgf
    do k = l+1, this%sdgf            
      result(l,k)=result(k,l)
    end do ! k
  end do  ! l
end subroutine GPR_eval_hess
!!****

!#ifdef withopenmp
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_eval_hess_parallel
!!
!! FUNCTION
!! Infer a Hessian matrix at the position "eval_pos" using openmp
!! It assumes/produces a symmetric Hessian 
!! -> half the computation time
!! This should be called usually instead of GPR_eval_hess, 
!! as soon as openmp is available.
!!
!! SYNOPSIS
subroutine GPR_eval_hess_parallel(this, eval_pos, result)
    implicit none
    type(gpr_type),intent(inout)    ::  this
    real(rk), intent(in)            ::  eval_pos(this%sdgf)
    real(rk), intent(out)           ::  result(this%sdgf,this%sdgf)    
    integer                         ::  k,l
    result(:,:) = 0d0
#ifdef withopenmp    
    call omp_set_num_threads( omp_get_max_threads() )
#endif
    !$omp parallel DEFAULT(NONE), private( l ), shared ( this, result , eval_pos )
    !$omp do schedule(static, 1) 
    do l = 1, this%sdgf
      call GPR_eval_hess_column(this,eval_pos,l,result(:,l))
      do k = l+1, this%sdgf
        result(l,k)=result(k,l)
      end do
    end do
    !$omp end parallel
end subroutine GPR_eval_hess_parallel

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_eval_hess_column
!!
!! FUNCTION
!! Infer the column (only from the "column"-th element on)
!! of a Hessian matrix at the position "eval_pos"
!! It assumes/produces a symmetric Hessian 
!! -> half the computation time
!!
!! SYNOPSIS
recursive subroutine GPR_eval_hess_column(this, eval_pos, column, result)    
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)    ::  this
    real(rk), intent(in)            ::  eval_pos(this%sdgf)
    real(rk)                        ::  diff(this%sdgf), absv
    integer, intent(in)             ::  column
    real(rk), intent(out)           ::  result(this%sdgf)
    integer                         ::  n, i, j,k,os, wi, os2, nr
    integer                         ::  dwrt(3)
    if (this%OffsetType==5) then
        ! do not allow nested spawning of parallel regions
        call GPR_eval_hess_column(this%lowerLevelGPR, eval_pos, column, result)
    else if (this%OffsetType==8) then
        call GPR_taylorBias_d2_column(this,column,result)
    end if
    if (.not.this%w_is_solution) &
      call dlf_fail("Solve the system KM*w=y first! (GPR_eval_hess)")
    if (this%kernel_type>1) &
      call dlf_fail("The requested type of kernel is not implemented! (GPR_eval_hess)")
  if (.not.this%iChol) then
    do k = column, this%sdgf ! symmetry
      do n = 1, this%nt
        wi = n
        result(k) = result(k) + this%w(wi)*&
                                kernel_d2(this,eval_pos(:), &
                                          this%xs(:,n),(/1,1/),k,column)
      end do
    end do
    if (this%order > 0) then
      os = this%nt          
      dwrt=(/1,1,2/)
      do k = column, this%sdgf ! symmetry       
        do n = 1, this%nt        
          do i = 1, this%sdgf
            wi = os+(n-1)*this%sdgf+i
            result(k) = result(k) + this%w(wi)*&
            kernel_d3(this,eval_pos(:), this%xs(:,n),dwrt,k,column,i)  
          end do
        end do
      end do 
    end if
    nr = this%sdgf*(this%sdgf+1)/2
    if (this%order == 2) then
      os = this%nt+this%nt*this%sdgf  
      do k = column, this%sdgf ! symmetry
        do n = 1, this%nt       
          os2 = -this%sdgf
          do i = 1, this%sdgf
            os2 = os2 + this%sdgf-(i-1)
            do j = i, this%sdgf
              wi = os+(n-1)*nr+os2+j
              result(k) = result(k)+this%w(wi)*&
                                    kernel_d4(this,eval_pos(:), &
                                        this%xs(:,n),(/2,2,1,1/),i,j,k,column)
            end do
          end do
        end do
      end do
    end if
  else
    ! my own cholesky decomposition is used
    if (this%order==0) then
      do n = 1, this%nt
        wi = n
        diff(:) = eval_pos(:) - this%xs(:,n)
        absv=dsqrt(dot_product(diff,diff))
          do k = column, this%sdgf ! symmetry        
            result(k) = result(k) + this%w(wi)*&
                          kernel_d2_exp11(this,diff,k,column,absv)
!                           kernel_d2(this,eval_pos(:), &
!                                     this%xs(:,n),(/1,1/),k,l)
            ! symmetry -> just at the end
          end do

      end do
    else if (this%order==1) then
      do n = 1, this%nt
        wi = (n-1)*(this%sdgf+1)+1 ! position of energies
        diff(:) = eval_pos(:) - this%xs(:,n)
        absv=dsqrt(dot_product(diff,diff))
        dwrt=(/1,1,2/)
          do k = column, this%sdgf
            !energies
            result(k) = result(k) + this%w(wi)*&
                        kernel_d2_exp11(this,diff,k,column,absv)
            !gradients
            do i = 1, this%sdgf
              
              result(k) = result(k) + this%w(wi+i)*&
                            kernel_d3(this,eval_pos(:), this%xs(:,n),&
                                      dwrt,k,column,i)            
            end do
          end do
      end do
    else if (this%order==2) then
      dwrt=(/1,1,2/)
      do n = 1, this%nt
        wi = (n-1)*((this%sdgf+1)*this%sdgf/2+this%sdgf+1)+1 ! position of energies
        diff(:) = eval_pos(:) - this%xs(:,n)
        absv=dsqrt(dot_product(diff,diff))
          do k = column, this%sdgf
            !energies
            result(k) = result(k) + this%w(wi)*&
                        kernel_d2_exp11(this,diff,k,column,absv)
            !gradients
            do i = 1, this%sdgf
              result(k) = result(k) + this%w(wi+i)*&
                            kernel_d3(this,eval_pos(:), this%xs(:,n),&
                                      dwrt,k,column,i)
            end do
            !hessians
            os = this%sdgf
            do i = 1, this%sdgf
              do j = i, this%sdgf
                os = os + 1
                result(k) = result(k) + this%w(wi+os)*&
                              kernel_d4(this,eval_pos(:), &
                                          this%xs(:,n),(/2,2,1,1/),i,j,k,column)
              end do
            end do
          end do
      end do
    else if (this%order==42) then
      os = 1
      do n = 1, this%nt
        diff(:) = eval_pos(:) - this%xs(:,n)
        absv=dsqrt(dot_product(diff,diff))
        do k = column, this%sdgf
          wi = os
          !energies
          result(k) = result(k) + this%w(wi)*&
                      kernel_d2_exp11(this,diff,k,column,absv)
          wi = wi + 1
          if (this%order_tps(n)>=1) then
            !gradients
            do i = 1, this%sdgf
              result(k) = result(k) + this%w(wi)*&
                            kernel_d3(this,eval_pos(:), this%xs(:,n),&
                                      (/1,1,2/),k,column,i)
              wi = wi + 1
            end do
          end if
          if (this%order_tps(n)==2) then
            !hessians
            do i = 1, this%sdgf
              do j = i, this%sdgf
                result(k) = result(k) + this%w(wi)*&
                            kernel_d4(this,eval_pos(:), &
                                        this%xs(:,n),(/2,2,1,1/),i,j,k,column)
                wi = wi + 1
              end do
            end do
          end if
        end do
        os = wi
      end do    
    else
      call dlf_fail("Wrong order for GPR_eval_hess_column.")
    end if  
  end if
end subroutine GPR_eval_hess_column
!!****
! ifdef withopenmp
!#endif

! ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! !!****f* gpr/GPR_opt_hPara_maxL
! !!
! !! FUNCTION
! !! Optimizes the hyperparameters by maximizing the likelihood
! !! (not very efficiently up to now!)
! !!
! !! SYNOPSIS
! subroutine GPR_opt_hPara_maxL(this, np_opt, whichParas)
! !! SOURCE
!     implicit none
!     type(gpr_type),intent(inout)::this
!     integer, intent(in)     ::  np_opt
!     integer, intent(in)     ::  whichParas(np_opt)
!     real(rk)                ::  error, variance
!     real(rk), allocatable   ::  p_opt(:,:) ! hyperparas to be optimized
!     real(rk), allocatable   ::  p_tmp(:,:)     ! hyperparameters    
!     real(rk), allocatable   ::  l(:)     ! likelihood values
!     real(rk), allocatable   ::  l_tmp(:)     ! likelihood values
!     integer                ::  stepnr, i, j
!     type(gpr_type)         ::  optGPR
!     type(optimizer_type)    ::  optGPR_opt
!     real(rk), allocatable   ::  gl(:,:)  ! gradient of likelihood
!     real(rk), allocatable   ::  gl_tmp(:,:)  ! gradient of likelihood
!     real(rk), allocatable   ::  ghM(:,:,:)
!     real(rk), allocatable   ::  maximum(:)
!     real(rk), allocatable   ::  unscaled_maximum(:)
!     real(rk), allocatable   ::  old_maximum(:)
!     integer                 :: counter
!     real(rk)                ::  maxstepsize ! (rel to size of the present parametervalue)
!     real(rk)                ::  stepsize_limiter_wrt_err
!     real(rk), allocatable   ::  step(:)
!     real(rk), allocatable   ::  step_to_here(:)
!     real(rk), allocatable   ::  old_step_to_here(:)
!     real(rk)                ::  gradlength
!     real(rk), allocatable   ::  best_max_upToNow(:)
!     real(rk)                ::  tolerance, gradtolerance
!     real(rk)                ::  scaleSteps
!     real(rk)                ::  delta
!     real(rk)                ::  dLdP
!     real(rk)                ::  old_dLdP
!     real(rk)                ::  stepsize_to_here
!     real(rk)                ::  paraGPR_s_n(3)
!     if (this%order==42) call dlf_fail("GPR_opt_hPara_maxL not implemented for order 42!")
!     ! Stepsize limiters seem to be 
!     ! irrelevant for GPR optimization
!     gradtolerance = 5d1 ! Must be quite high since
!     ! numerical inprecisions lead to a high error in the
!     ! gradient of the likelihood function.
!     tolerance = 1d-5 ! this should somehow correspond to the
!     ! precision of the GPR surrogate 
!     ! (and keep in mind: likelihood+grads have quite some noise!)
!     paraGPR_s_n = (/1d-5, 1d-5,1d-5/)! Do not set these too low,
!     ! there really is some noise in the calculation of the likelihood!
!     maxstepsize = 2d-1 ! Relativ to current value!
!     stepsize_limiter_wrt_err=1d2
!     if (printl>=2) &
!         write(stdout,'("Starting hyperparameter optimization by maximizing likelihood.")')
! if (this%kernel_type<=1) then  
!     do j = 1, np_opt
!         i = whichParas(j)
!         if (printl>=4) then 
!           if(i==1) write(stdout,'("sigma_f            : ",f10.5)') this%s_f
!           if(i==2) write(stdout,'("gamma              : ",f10.5)') this%gamma
!           if(i==2) write(stdout,'("Length scale l     : ",f10.5)') this%l
!           if(i==3) write(stdout,'("sigma_n (Energies) : ",f10.5)') this%s_n(1)
!           if(i==4) write(stdout,'("sigma_n (Gradients): ",f10.5)') this%s_n(2)
!           if(i==5) write(stdout,'("sigma_n (Hessians) : ",f10.5)') this%s_n(3)
!         end if
!     end do
!     3335 FORMAT (A3,I5,A12,1X, ES11.4,5X, ES11.4,5X, ES11.4, 5X, ES11.4, 5X, ES11.4)    
!     3334 FORMAT (A3,I5,A12,1X, ES11.4,5X, ES11.4,5X, ES11.4, 5X, ES11.4)    
!     3333 FORMAT (A3,I5,A12,1X, ES11.4,5X, ES11.4,5X, ES11.4)    
!     3332 FORMAT (A3,I5,A12,1X, ES11.4,5X, ES11.4)    
!     3331 FORMAT (A3,I5,A12,1X, ES11.4)  
!        
!     stepnr = 1
! 
!     allocate(ghM(np_opt,this%nk,this%nk))
!     allocate(maximum(np_opt))
!     allocate(best_max_upToNow(np_opt))
!     allocate(unscaled_maximum(np_opt))
!     allocate(old_maximum(np_opt))
!     allocate(step(np_opt))
!     allocate(step_to_here(np_opt))
!     allocate(old_step_to_here(np_opt))
!     error = 1d0
!     variance = 1d0
!     
!     scaleSteps = 1d0
!     dLdP=0d0
!     old_dLdP=0d0
!     delta=0d0
!     step(:)=0d0
!     stepsize_to_here=0d0
!     stepnr = 1
!     gradlength = 0d0
!     unscaled_maximum(:)=0d0
!     
!   do while((error>tolerance).OR.(gradlength>gradtolerance))    
!     if(stepnr/=1) then
!       allocate(p_tmp(np_opt,stepnr-1))
!       allocate(gl_tmp(np_opt,stepnr-1))
!       allocate(l_tmp(stepnr-1))
!       ! temporary save of the parameters, grad/likelihood vals
!       p_tmp(:,1:stepnr-1) = p_opt(:,:)
!       gl_tmp(:,1:stepnr-1) = gl(:,:)
!       l_tmp(1:stepnr-1) = l(:)
!       deallocate(p_opt)
!       deallocate(gl)
!       deallocate(l)   
!     end if
!     allocate(p_opt(np_opt,stepnr))
!     allocate(gl(np_opt,stepnr))
!     allocate(l(stepnr))
!     if (stepnr /=1) then
!       ! rewrite the values
!       p_opt(:,1:stepnr-1)   = p_tmp(:,1:stepnr-1)
!       l(1:stepnr-1)   = l_tmp(1:stepnr-1)
!       gl(:,1:stepnr-1)  = gl_tmp(:,1:stepnr-1)
!       deallocate(p_tmp)
!       deallocate(gl_tmp)
!       deallocate(l_tmp)
!     end if    
! 
!     ! calculate current values for p and gl (determined by the prev. maximum)
!     if (printl>=6) &
!         write(stdout,'("Calculate the likelihood and derivative")')
!     call calc_p_like_and_deriv(this, np_opt, whichParas, l(stepnr), gl(:,stepnr),p_opt(:,stepnr))
!     if (printl>=6) &
!         write(stdout,'("loglikelihood: ",f10.5)')l(stepnr)
!     ! scale last entry again
!     call scale_p(this,.true., np_opt, whichParas, p_opt(:,stepnr), gl(:,stepnr))
!     if (printl>=6) then 
!       if (np_opt==1) &
!         write(stdout,3331) "O!",stepnr," scaledHyperp", p_opt(:,stepnr)
!       if (np_opt==2) &
!         write(stdout,3332) "O!",stepnr," scaledHyperp", p_opt(:,stepnr)
!       if (np_opt==3) &
!         write(stdout,3333) "O!",stepnr," scaledHyperp", p_opt(:,stepnr)
!       if (np_opt==4) &
!         write(stdout,3334) "O!",stepnr," scaledHyperp", p_opt(:,stepnr)
!       if (np_opt==5) &
!         write(stdout,3335) "O!",stepnr," scaledHyperp", p_opt(:,stepnr)
!     end if
!     if(stepnr==1) then
!         old_maximum = p_opt(:,stepnr)
!     else
!         old_maximum = maximum 
!     end if
!     ! Starting GPR interpolation with scaled parameters
!     if(stepnr==1) then
! #ifdef SE_pOpt        
!         call GPR_construct(optGPR,stepnr,0,np_opt,1,0,1) ! gradient based GPR for maximum search
! #endif
! #ifdef Ma52_pOpt
!         call GPR_construct(optGPR,stepnr,0,np_opt,1,1,1) ! gradient based GPR for maximum search
! #endif
!         !                          ,gamma   ,s_f   , s_n
!         call GPR_init(optGPR, p_opt, 1d0    , 0.1d0, paraGPR_s_n, l, gl)  
!         call GPR_Optimizer_define('MAX', optGPR_opt,this,60,10)
!         call GPR_Optimizer_step(optGPR,optGPR_opt,p_opt(:,stepnr),maximum)        
!     else
!         call GPR_Optimizer_step(optGPR,optGPR_opt,p_opt(:,stepnr),maximum,&
!                                 p_opt(:,stepnr),l(stepnr),gl(:,stepnr))
!     end if
!     step(:)=maximum(:)-p_opt(:,stepnr)
! 
!     ! limit stepsize?
!     
!     do j = 1, np_opt
!         if (maxstepsize*p_opt(j,stepnr)<abs(step(j))) then
! !             scaleSteps = 1d0
!             step(j) = SIGN(maxstepsize*p_opt(j,stepnr),step(j))
!         end if           
!     end do    
!     maximum(:)=p_opt(:,stepnr)+step(:)
!     do j =1, np_opt
!         if (maximum(j)<0d0) maximum(j)=0d0
!     end do
!     error = dsqrt(dot_product(maximum(:)-old_maximum(:),&
!                         maximum(:)-old_maximum(:)))
!     ! This maximum will determine the new parameters (rescaling and rewriting)
!     ! ghM(:,:,1)   = dK/dsigma_f
!     ! ghM(:,:,2)   = dK/dgamma
!     ! ghM(:,:,3:5) = dK/dsigma_n(1:3)
!     unscaled_maximum = maximum
!     call scale_p(this,.false.,np_opt, whichParas, unscaled_maximum)
!     do j = 1, np_opt
!         i = whichParas(j)    
!         if (i==1) this%s_f      = unscaled_maximum(j)        
!         if (i==2) then
!             if (this%kernel_type==0) then
!                 this%gamma    = unscaled_maximum(j)
!                 this%l        = 1d0/dsqrt(this%gamma)
!             else if (this%kernel_type==1) then
!                 ! Matern Kernel is optimized wrt. l not gamma
!                 this%l = unscaled_maximum(j)
!                 this%gamma = 1d0/(this%l)**2
!                 this%l4 = this%l**4
!                 this%l2 = this%l**2
!             else
!                 call dlf_fail("Kernel not implemented(GPR_opt_hPara_maxL)")
!             end if
!         end if
!         if (i==3) this%s_n(1)   = unscaled_maximum(j)
!         if (i==4) this%s_n(2)   = unscaled_maximum(j)
!         if (i==5) this%s_n(3)   = unscaled_maximum(j)
!     end do  
!     
!     if (printl>=6) then 
!       if (np_opt==1) &
!         write(stdout,3331) "O!",stepnr," Grads       ", gl(:,stepnr)!maximum(:)
!       if (np_opt==2) &
!         write(stdout,3332) "O!",stepnr," Grads       ", gl(:,stepnr)!maximum(:)
!       if (np_opt==3) &
!         write(stdout,3333) "O!",stepnr," Grads       ", gl(:,stepnr)!maximum(:)
!       if (np_opt==4) &
!         write(stdout,3334) "O!",stepnr," Grads       ", gl(:,stepnr)!maximum(:)
!       if (np_opt==5) &
!         write(stdout,3335) "O!",stepnr," Grads       ", gl(:,stepnr)!maximum(:)
!       if (np_opt==1) &
!         write(stdout,3331) "O!",stepnr," Steps       ", step(:)!maximum(:)
!       if (np_opt==2) &
!         write(stdout,3332) "O!",stepnr," Steps       ", step(:)!maximum(:)
!       if (np_opt==3) &
!         write(stdout,3333) "O!",stepnr," Steps       ", step(:)!maximum(:)
!       if (np_opt==4) &
!         write(stdout,3334) "O!",stepnr," Steps       ", step(:)!maximum(:)
!       if (np_opt==5) &
!         write(stdout,3335) "O!",stepnr," Steps       ", step(:)!maximum(:)
!     end if
! 
!     if (printl>=6) then 
!       write(stdout,'("sigma_f                   : ",f10.5)') error
!       write(stdout,'("log(Likelihood) (lastStep): ",f10.5)') l(stepnr)
!       if (np_opt==1) then
!         write(stdout,'("Grad of log(likel.) (lastStep)      : ",f10.5)') &
!             gl(1,stepnr)
!       else
!         write(stdout,'("Gradlength of log(likel.) (lastStep): ",f10.5)') &
!                 dsqrt(dot_product(gl(:,stepnr),gl(:,stepnr)))
!       end if   
!     end if
!     counter = counter + 1
!     stepnr = stepnr + 1
!   end do
!   call GPR_destroy(optGPR) 
!   deallocate(p_opt)
!   deallocate(gl)
!   deallocate(l)    
! else
!     call dlf_fail("The requested type of kernel is not implemented! (GPR_opt_hPara_maxL)")
! end if
!   this%scaling_set = .false.
!   if (printl>=4) write(stdout,'(&
!         "Finished hyperparameter optimization of parameters within",I9,&
!         "steps:")') stepnr-1
!   do j = 1, np_opt
!     i = whichParas(j)
!     if (printl>=4) then 
!       if(i==1) write(stdout,'("sigma_f            : ",f10.5)') this%s_f
!       if(i==2) write(stdout,'("gamma              : ",f10.5)') this%gamma
!       if(i==2) write(stdout,'("Length scale l     : ",f10.5)') this%l
!       if(i==3) write(stdout,'("sigma_n (Energies) : ",f10.5)') this%s_n(1)
!       if(i==4) write(stdout,'("sigma_n (Gradients): ",f10.5)') this%s_n(2)
!       if(i==5) write(stdout,'("sigma_n (Hessians) : ",f10.5)') this%s_n(3)
!     end if
!   end do
!   call GPR_interpolation(this)
! end subroutine GPR_opt_hPara_maxL
! !!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/calc_p_like_and_deriv
!!
!! FUNCTION
!! Calculates the likelihood and its gradients with respect
!! to the submitted hyperparameters. This is used to 
!! find the maximum of the likelihood.
!!
!! SYNOPSIS
subroutine calc_p_like_and_deriv(this, np_opt, whichParas, like, gh, p)    
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    integer, intent(in)     ::  np_opt   ! # parameters              
    integer, intent(in)     ::  whichParas(np_opt)
    real(rk), intent(out)   ::  like ! likelihoood
    real(rk), intent(out)   ::  gh(np_opt)! Gradient with respect to hyperparameters
                                      ! that are written in p(:) 
    real(rk), intent(out)   ::  p(np_opt)                                      
    real(rk)                ::  alpha(this%nk), tmpV(this%nk), tmpV2(this%nk),&
                                alphaM(this%nk,this%nk)
    real(rk)                ::  ghM(this%nk,this%nk,np_opt)        
    real(rk)                ::  logdet
    integer             ::  i, j, k, l
    if (this%order==42) call dlf_fail("calc_p_like_and_deriv not implemented for order 42!")
    ! weight vector shall contain energies, gradients and hessians again
!    if(this%ichol .and. this%K_stat /=0) call calc_cov_mat_newOrder(this)
    if(this%iChol) then
      if(this%K_stat ==-1) then
        call calc_cov_mat_newOrder(this)
      elseif(this%K_stat == 6) then
        call calc_cov_mat_newOrder(this,.true.)
      endif
    endif
    if (.not. this%ichol .and. this%K_stat/=0) call calc_cov_mat(this)
    if(this%ichol) then
      call writeObservationsInW_newOrder(this)
    else
      call writeObservationsInW(this) 
    endif
    ! Calculate the gradient wrt parameters
    ! ghM(:,:,1)   = dK/dsigma_f
    ! ghM(:,:,2)   = dK/dgamma for SE and dK/l for Matern
    ! ghM(:,:,3:5) = dK/dsigma_n(1:3)
    ghM(:,:,:) = 0d0    
    if (this%iChol) call dlf_fail("calc_p_like_and_deriv not usable with iterative Cholesky scheme!")
if (this%kernel_type<=1) then
  do k = 1, np_opt
    l = whichParas(k)
    if (l==1) then
      p(k)  = this%s_f
    else if (l==2) then
      if (this%kernel_type==0) then
        p(k)  = this%gamma
      else if (this%kernel_type==1) then
        p(k)  = this%l
      else
        call dlf_fail("Kernel not implemented (calc_p_like_and_deriv)")
      end if
    else if (l==3) then
      p(k)  = this%s_n(1)
    else if (l==4) then
      p(k)  = this%s_n(2)
    else if (l==5) then
      p(k)  = this%s_n(3)
    end if
    do j = 1, this%nk
      do i = 1, this%nk
        ! derivative wrt s_n
        if (l==1) then
          if (i/=j) then
            ghM(i, j, k) = this%KM(i,j)*2d0/this%s_f
          else if (i<=this%nt) then
            ghM(i, j, k) = (this%KM(i,j)-this%s_n(1)**2)*2d0/this%s_f
          else if (i<=this%nt+this%nt*this%idgf) then
            ghM(i, j, k) = (this%KM(i,j)-this%s_n(2)**2)*2d0/this%s_f
          else
            ghM(i, j, k) = (this%KM(i,j)-this%s_n(3)**2)*2d0/this%s_f
          end if
        end if
        if (i==j) then
          if (i<=this%nt) then
            ! derivative wrt s_n(1)
            if (l==3) ghM(i, j,k) = 2d0*this%s_n(1)
          else if (i<=this%nt+this%nt*this%idgf) then
            ! derivative wrt s_n(2)
            if (l==4) ghM(i, j,k) = 2d0*this%s_n(2)
          else ! i > this%nt+this%nt*this%sdgf
            ! derivative wrt s_n(3)
            if (l==5) ghM(i, j,k) = 2d0*this%s_n(3)
          end if
        end if
      end do
    end do
    if (l==2) call calc_cov_mat_dg_or_l(this, ghM(:,:,k))
  end do
    ! We now have a gradient matrix dK/dTheta_k in ghM
    select case(this%solveMethod)
      ! case 1+2 deleted (LU)
      case(3)
        if (printl>=6) &
          write(stdout,'("Cholesky_KM_logdet...")')
        call Cholesky_KM_logdet(this,logdet)
        if (printl>=6) &
          write(stdout,'("Cholesky_KM_logdet complete")')
      case(4)
        call dlf_fail("No likelihood optimizatino with Cholesky inverse implemented!")
        if (printl>=6) &
          write(stdout,'("Using Cholesky decomposition without inverting...")')
        call Cholesky_KM_logdet(this,logdet)
      case default
        call dlf_fail("This method is not available (calc_p_like_and_deriv)")
    end select
    ! Calculating the likelihood
    ! The weight vector contains the observations 
    ! (done writeObservationsInW above)!
    tmpV(1:this%nk) = this%w(1:this%nk) ! tmpV contatains the observations now
    like = 0d0
    select case(this%solveMethod)
      ! case 1+2 deleted (LU)
      case(3)
        call solve_LGS_use_KM_chol(this,tmpV) ! tmpV now contains the weights 
      case(4)
        call dlf_fail("No likelihood optimizatino with Cholesky inverse implemented!")
        call solve_LGS_use_KM_chol(this,tmpV) ! tmpV now contains the weights 
      case default
        call dlf_fail("This method is not available (calc_p_like_and_deriv)")
    end select
    alpha(:) = tmpV                       ! alpha as well
    like = like + dot_product(this%w(1:this%nk),tmpV(1:this%nk))
    like = like + logdet + this%nk*LOG(2d0*pi)
    like = -like/2d0
    ! calculate alpha*alpha**T-K**-1, with alpha = k**-1 y    
    ! calculate alphaM = alpha*alpha**T    
    do j = 1, this%nk
        do i = 1, this%nk
            alphaM(i,j) = alpha(i)*alpha(j)
        end do
    end do
    gh(:) = 0d0
    ! For every parameter
     
    do k = 1, np_opt
        ! Multiply alphaM*ghM and store the diagonal elements in alpha
        alpha(:) = 0d0
        do j = 1, this%nk
            do i = 1, this%nk
                alpha(j) = alpha(j) + alphaM(j,i)*ghM(i,j,k)
            end do
        end do
        ! Store diag of k**-1*ghM in tmpV
        ! Highly inefficient (but I do not have a better solution up to now)
        if (printl>=5) &
            write(stdout,'("Solving the linear system nk times... (this takes a while)")')
        do j = 1, this%nk
            tmpV2 = ghM(:,j,k)
    select case(this%solveMethod)
      ! case 1+2 deleted (LU)
      case(3)
        call solve_LGS_use_KM_chol(this,tmpV2) 
      case(4)
        call dlf_fail("No likelihood optimizatino with Cholesky inverse implemented!")
        call solve_LGS_use_KM_chol(this,tmpV2)
      case default
        call dlf_fail("This method is not available (calc_p_like_and_deriv)")
    end select                     
            
            tmpV(j) = tmpV2(j)
            if (this%nk>=1000) then ! Only necessary for big systems, also
                                  ! it will throw division by zero for nk>100
                if (MODULO(j,this%nk/100)==0) write (*,"(A1)",advance="no") "."
                if (MODULO(j,this%nk/2)==0) write (*,"(A1)") ""
            end if
        end do
        if (printl>=5) &
          write(stdout,'("Finished solving the system nk times.")')
        
        ! calculate 1/2 tr[(alpha*alpha**T-K**-1)dK/dTheta_k
        ! write result in gh(k) -> take 1/2 trace of matrices above
        do i = 1, this%nk
            gh(k) = gh(k)+(alpha(i)-tmpV(i))
        end do
        gh(k) = gh(k)/2d0
    end do
else
    call dlf_fail("This Kernel is not implemented! (calc_p_like_and_deriv)")
end if
end subroutine calc_p_like_and_deriv
!!****

subroutine calc_p_like(this, np_opt, whichParas, like, p)    
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    integer, intent(in)     ::  np_opt   ! # parameters              
    integer, intent(in)     ::  whichParas(np_opt)
    real(rk), intent(out)   ::  like ! likelihoood
    real(rk), intent(out)   ::  p(np_opt)                                      
    real(rk)                ::  alpha(this%nk), tmpV(this%nk), tmpV2(this%nk),&
                                alphaM(this%nk,this%nk)
    real(rk)                ::  ghM(this%nk,this%nk,np_opt)        
    real(rk)                ::  logdet
    integer             ::  i, j, k, l
    if (this%order==42) call dlf_fail("calc_p_like_and_deriv not implemented for order 42!")
    ! weight vector shall contain energies, gradients and hessians again
  !  if(this%ichol .and. this%K_stat /=0) call calc_cov_mat_newOrder(this)
    if(this%iChol) then
      if(this%K_stat ==-1) then
        call calc_cov_mat_newOrder(this)
      elseif(this%K_stat == 6) then
        call calc_cov_mat_newOrder(this,.true.)
      endif
    endif
    if (.not. this%ichol .and. this%K_stat/=0) call calc_cov_mat(this)
    if(this%ichol) then
      call writeObservationsInW_newOrder(this)
    else
      call writeObservationsInW(this) 
    endif
    ! Calculate the gradient wrt parameters
    ! ghM(:,:,1)   = dK/dsigma_f
    ! ghM(:,:,2)   = dK/dgamma for SE and dK/l for Matern
    ! ghM(:,:,3:5) = dK/dsigma_n(1:3)
    ghM(:,:,:) = 0d0    
   ! if (this%iChol) call dlf_fail("calc_p_like_and_deriv not usable with iterative Cholesky scheme!")
if (this%kernel_type<=1) then
  do k = 1, np_opt
    l = whichParas(k)
    if (l==1) then
      p(k)  = this%s_f
    else if (l==2) then
      if (this%kernel_type==0) then
        p(k)  = this%gamma
      else if (this%kernel_type==1) then
        p(k)  = this%l
      else
        call dlf_fail("Kernel not implemented (calc_p_like_and_deriv)")
      end if
    else if (l==3) then
      p(k)  = this%s_n(1)
    else if (l==4) then
      p(k)  = this%s_n(2)
    else if (l==5) then
      p(k)  = this%s_n(3)
    end if
  end do
    select case(this%solveMethod)
      ! case 1+2 deleted (LU)
      case(3)
        if (printl>=6) &
          write(stdout,'("Cholesky_KM_logdet...")')
        if(this%iChol) then  
          if(this%K_stat == 0) then
            call Cholesky_KM_myown_logdet(this,.false.,logdet)
          elseif(this%K_stat == 7) then
            call Cholesky_KM_myown_logdet(this,.true.,logdet)
          endif
        else
          call Cholesky_KM_logdet(this,logdet)
        endif
        if (printl>=6) &
          write(stdout,'("Cholesky_KM_logdet complete")')
      case(4)
        call dlf_fail("No likelihood optimizatino with Cholesky inverse implemented!")
        if (printl>=6) &
          write(stdout,'("Using Cholesky decomposition without inverting...")')
        call Cholesky_KM_logdet(this,logdet)
      case default
        call dlf_fail("This method is not available (calc_p_like_and_deriv)")
    end select
    ! Calculating the likelihood
    ! The weight vector contains the observations 
    ! (done writeObservationsInW above)!
    tmpV(1:this%nk) = this%w(1:this%nk) ! tmpV contatains the observations now
    like = 0d0
    select case(this%solveMethod)
      ! case 1+2 deleted (LU)
      case(3)
        if(this%iChol) then
          call solve_LGS_use_chol_myOwn(this%nk,this%KM_lin_chol, &
                                            tmpV(1:this%nk))
        else
          call solve_LGS_use_KM_chol(this,tmpV) ! tmpV now contains the weights 
        endif
        
      case(4)
        call dlf_fail("No likelihood optimizatino with Cholesky inverse implemented!")
        call solve_LGS_use_KM_chol(this,tmpV) ! tmpV now contains the weights 
      case default
        call dlf_fail("This method is not available (calc_p_like_and_deriv)")
    end select
    alpha(:) = tmpV                       ! alpha as well
    like = like + dot_product(this%w(1:this%nk),tmpV(1:this%nk))
    like = like + logdet + this%nk*LOG(2d0*pi)
    like = -like/2d0
  else
    call dlf_fail("This Kernel is not implemented! (calc_p_like_and_deriv)")
  end if
end subroutine calc_p_like    


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/scale_p
!!
!! FUNCTION
!! Scales the parameter vector and the gradients
!! (derivatives wrt. the parameters)
!! SYNOPSIS
subroutine scale_p(this, scale_switch, np_opt, whichParas, p, g)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this    
    integer, intent(in)                 ::  np_opt  ! Number of parameters and 
                                                ! number of points/gradients
    integer, intent(in)                 ::  whichParas(np_opt)
    real(rk), intent(inout)              ::  p(np_opt)
    real(rk), intent(inout), optional    ::  g(np_opt)
    logical                             ::  scale_switch
    integer                             ::  k,l
if (this%kernel_type<=1) then
    
!     p(1)    = this%s_f
!     p(2)    = this%gamma for SE or this%l for Matern
!     p(3:5)  = this%s_n(1:3)

if (.not.this%scaling_set) then
    this%scaling(:)=1d0
    ! scale parameters
    this%scaling(1) = 1d0/this%s_f
    if (this%kernel_type==0) then
        this%scaling(2) = 1d0/this%gamma
    else if (this%kernel_type==1) then
        this%scaling(2) = 1d0/this%l
    end if    
    if (this%s_n(1)>1d-10) then
        this%scaling(3) = 1d0/this%s_n(1)
    else
        this%scaling(3) = 1d0
    end if
    if (this%s_n(2)>1d-10) then
        this%scaling(4) = 1d0/this%s_n(2)
    else
        this%scaling(4) = 1d0
    end if
    if (this%s_n(3)>1d-10) then
        this%scaling(5) = 1d0/this%s_n(3)
    else
        this%scaling(5) = 1d0
    end if

    this%scaling_set=.true.

end if
  do k = 1, np_opt
    l = whichParas(k)    
    if (scale_switch) then 
        if (l==1) then
            p(k)   = p(k)*this%scaling(1)
        else if(l==2) then
            p(k)   = p(k)*this%scaling(2)
        else if(l==3) then
            p(k)   = p(k)*this%scaling(3) 
        else if(l==4) then
            p(k)   = p(k)*this%scaling(4)
        else if(l==5) then
            p(k)   = p(k)*this%scaling(5)
        end if
        if (present(g)) then
          if (l==1) then
            g(k)   = g(k)/this%scaling(1)
          else if(l==2) then
            g(k) = g(k)/this%scaling(2)
          else if(l==3) then
            g(k) = g(k)/this%scaling(3)
          else if(l==4) then
            g(k) = g(k)/this%scaling(4)
          else if(l==5) then
            g(k) = g(k)/this%scaling(5)            
          end if
        end if
    else
        if (l==1) then
            p(k)   = p(k)/this%scaling(1)
        else if(l==2) then
            p(k)   = p(k)/this%scaling(2)
        else if(l==3) then
            p(k)   = p(k)/this%scaling(3)
        else if(l==4) then
            p(k)   = p(k)/this%scaling(4)
        else if(l==5) then
            p(k)   = p(k)/this%scaling(5)
        end if
        if (present(g)) then
          if (l==1) then
            g(k)   = g(k)*this%scaling(1)
          else if(l==2) then
            g(k) = g(k)*this%scaling(2)
          else if(l==3) then
            g(k) = g(k)*this%scaling(3)
          else if(l==4) then
            g(k) = g(k)*this%scaling(4)
          else if(l==5) then
            g(k) = g(k)*this%scaling(5)            
          end if
        end if
    end if
  end do
else
    call dlf_fail("The requested type of kernel is not implemented! (scale_p)")
end if    
end subroutine scale_p
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_interpolation
!!
!! FUNCTION
!! This is the central function of GPR interpolation
!! It calculates the covariance matrix, builds an LU/Cholesky/... decomposition
!! and solves respectively for the weights that can then 
!! be used to infer data.
!!
!! SYNOPSIS
subroutine GPR_interpolation(this, method)     
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    ! which method to use for solving the linear system
    integer, intent(in),optional::method
    real(rk)            ::  dummy
    real(rk), allocatable ::  tmpvec(:)
    integer         ::  i,j
    logical :: semdef,def,sym
    real(rk) :: cov(this%nk,this%nk)
    real(rk) :: eig(this%nk),wr(this%nk),wi(this%nk)
    real(rk) :: u(this%nk,this%nk),u_t(this%nk,this%nk)
    real(rk) :: vl(this%nk,this%nk),vr(this%nk,this%nk)
    real(rk) :: work(40*this%nk),test(this%nk)
    integer :: info
    !real(rk) :: work(this%nk*this%nk)
    if (present(method)) then
      if (this%iChol) then
        if (method/=3) call dlf_fail("turn iChol off or chose method 3!")
        this%solveMethod=method !=3
      else
        this%solveMethod=method
      end if
    else
      ! Cholesky (3) is standard
      this%solveMethod = 3
    end if
    if (this%kernel_type>1) call dlf_fail(&
        "The requested type of kernel is not implemented! (GPR_interpolation)")

    ! using squared exponential kernel
    
    if (this%w_is_solution) then
      call dlf_fail("GPR_interpolation unnecessary! Already solution! ")
    end if
    if (this%K_stat==-1) then
      if (this%sdgf>1) then
        if (printl>=4) &
          write(stdout,'("Constructing a new covariance matrix.")')
      end if
      if (.not.this%iChol) then
        call calc_cov_mat(this)
      else
   
        call calc_cov_mat_newOrder(this)
      end if
      if(printl>=6) &
        write(stdout,'("calculated the covariance matrix")')
!       call outputMatrixAsCSV(this%nk_lin,1,this%KM_lin,'KM_unsol2')
      if (.not.this%iChol) then
        call writeObservationsInW(this)
      else
        call writeObservationsInW_newOrder(this)
      end if
      if(printl>=6) &
        write(stdout,'("wrote observations in the vector w")')
!       call outputMatrixAsCSV(this%nk,1,this%w,'w_unsol2')
    else if (this%K_stat==6) then
      call calc_cov_mat_newOrder(this,.true.)
      this%K_stat=7
    ! else if (this%K_stat==7) already constructed
    end if
    
    select case(this%solveMethod)
      ! case 1+2 deleted (LU)
      CASE(3)
        ! Cholesky
        if (this%K_stat==3) then
            call solve_LGS_use_KM_chol(this, this%w(1:this%nk))
        else if (this%K_stat==0) then
            ! use Cholesky decomposition  
            if(.not.this%iChol) then
              ! not iChol
              call Cholesky_KM(this)
              call solve_LGS_use_KM_chol(this, this%w(1:this%nk))
            else
              ! iChol
              call Cholesky_KM_myown(this, .false.)
              call solve_LGS_use_chol_myOwn(this%nk,&
                                            this%KM_lin_chol, &
                                            this%w(1:this%nk))
            end if
        else if (this%K_stat==5) then
            call solve_LGS_use_chol_myOwn(this%nk,&
                                          this%KM_lin_chol, &
                                          this%w(1:this%nk))
        else if (this%K_stat==7) then
            call Cholesky_KM_myown(this, .true.)
            call solve_LGS_use_chol_myOwn(this%nk,&
                                            this%KM_lin_chol, &
                                            this%w(1:this%nk))
        else
            call dlf_fail("This should not happen. Cov_matr has an invalid state.")
        end if      
      CASE(4,8)
        ! Inverse by Cholesky
        ! Use Cholesky to derive inverse and use that
        allocate(tmpvec(this%nk))
        if (this%K_stat==4) then
          call solve_LGS_use_KM_chol_inv(this,this%w(1:this%nk),tmpvec)
          this%w(1:this%nk) = tmpvec
        else if (this%K_stat==0) then
          ! use Cholesky decomposition    
          call invert_KM_chol(this,dummy)
          call solve_LGS_use_KM_chol_inv(this,this%w(1:this%nk),tmpvec)
          this%w(1:this%nk) = tmpvec
        else
          call dlf_fail("This should not happen. Cov_matr has an invalid state.")
        end if     
        deallocate(tmpvec)
        ! mirror covariance matrix, so that it really is the inverse
        ! and not only the upper triangular matrix
        if (this%solveMethod==8) then
          do i = 1, this%nk
            do j = i+1, this%nk
              this%KM(j,i) = this%KM(i,j)
            end do
          end do
          this%K_stat = 8
        end if
      CASE DEFAULT
        call dlf_fail("Invalid method for solving the linear system (gpr.f90)")
    end select

    this%w_is_solution=.true.
end subroutine GPR_interpolation
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_redo_interpolation
!!
!! FUNCTION
!! Repeat the interpolation after something was changed.
!!
!! SYNOPSIS
subroutine GPR_redo_interpolation(this)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    this%w_is_solution=.false.
    call GPR_interpolation(this)
end subroutine GPR_redo_interpolation
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_init
!!
!! FUNCTION
!! * Second derivatives Kernel (derivative 1. then 2. variable, faster version)
!!
!! SYNOPSIS

! Initialize the data necessary for GPR
! It reads in the training points, their energies/gradients/hessians
! You can also ommit gradienst/hessians, but only of the order
! is respectively (0/1).
subroutine GPR_init(this, x, gamma, s_f, s_n, es, gs,ix,igs,b_matrix, hs)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)      ::  this
    real(rk), intent(in)              ::  x(this%sdgf,this%nt)
                                ! spatial coordinates
                                ! the order of the hyperparameters is relevant
                                ! see below for the different kernels
    real(rk), optional, intent(in)    ::  ix(this%idgf,this%nt) ! cartesians for internal coordinaate
    real(rk),optional, intent(in)     ::  es(this%nt)           ! energies
    real(rk),optional, intent(in)     ::  gs(this%sdgf, this%nt)! gradients
    real(rk),optional, intent(in)     ::  igs(this%idgf, this%nt) ! internal gradients
    !more prciesly: cartesian gradiant (3*N Dimensions) für use with internal coordinates 
    real(rk),optional, intent(in)     ::  b_matrix(this%sdgf,this%idgf,this%nt)
    real(rk),optional, intent(in)     ::  hs(this%sdgf, this%sdgf, this%nt)  
                                                                ! hessians
    real(rk), intent(in)              ::  gamma, s_f, s_n(3)

    if ((this%order/=42).and.(.not.present(es))) &
        call dlf_fail("Must provide energies for interpolation!")
    if ((this%order==2).and.(.not.present(hs))) &
        call dlf_fail("Must provide Hessians for 2nd order interpolation!")
    if ((this%order>0.and.this%order/=42).and.(.not.present(gs))) &
        call dlf_fail("Must provide Gradients for 1st order interpolation!")
    if (this%order==42.and.present(es)) &
        call dlf_fail("Use GPR_add_tp to add points not GPR_init for order 42!")
    call GPR_init_without_tp(this,gamma,s_f,s_n)
    
    if (this%order/=42) then
      this%xs(1:this%sdgf,1:this%nt) = x(1:this%sdgf,1:this%nt)
      if(this%internal == 2) this%ixs(1:this%idgf,1:this%nt) = ix(1:this%idgf,1:this%nt)
      if(this%internal == 2) then
        this%b_matrix(1:this%sdgf,1:this%idgf,1:this%nt) = &
         b_matrix(1:this%sdgf, 1:this%idgf, 1:this%nt)
      endif
      this%es(1:this%nt) = es(1:this%nt)
      if (this%order>1.and.present(hs)) &
        this%hs(1:this%sdgf, 1:this%sdgf, 1:this%nt) = &
            hs(1:this%sdgf, 1:this%sdgf, 1:this%nt)
      if (this%order>0.and.present(gs)) then
        if(this%internal == 2) then
          this%gs(1:this%sdgf,1:this%nt) = gs(1:this%sdgf,1:this%nt)
          this%igs(1:this%idgf,1:this%nt) = igs(1:this%idgf,1:this%nt)
        else
          this%gs(1:this%sdgf,1:this%nt) = gs(1:this%sdgf,1:this%nt)
        endif
      endif
    end if
    call setMeanOffset(this)
    if(.not.this%iChol) then
      call writeObservationsInW(this)   
    else
      call writeObservationsInW_newOrder(this)
    end if
    ! tolerance is set to largest value /10 since it is used to check
    ! if noise parameters are larger than tolerance * 10
    this%tol_noise_factor = 1d3
    this%tolerance = this%s_n(2) !HUGE(this%tolerance)/this%tol_noise_factor
end subroutine GPR_init
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_init_without_tp
!!
!! FUNCTION
!! Exactly like GPR_init but it does not need any training points to initialize
!!
!! SYNOPSIS
subroutine GPR_init_without_tp(this, gamma, s_f, s_n)     
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    real(rk), intent(in)     ::  gamma, s_f, s_n(3)
    if (.not.this%iChol) &
      write(stdout,*) "It is not advisable to call GPR ", &
                      "without training points and iChol==false"
    if(printl>=4.and.this%sdgf>1) then
      write(stdout,'("Initializing new GPR")')
      write(stdout,'("Length scale: ",f10.5)') 1.D0/sqrt(gamma)
      write(stdout,'("Noise as standard deviations:")')
      write(stdout,'("Assumed input noise in energies:  ",es12.5)') s_n(1)
      write(stdout,'("Assumed input noise in gradients: ",es12.5)') s_n(2)
      write(stdout,'("Assumed input noise in Hessians:  ",es12.5)') s_n(3)
      write(stdout,'("Degrees of freedom:               ",i12)') this%sdgf
    end if
    
    this%gamma = gamma
    this%l = 1d0/dsqrt(gamma)
    this%l4 = this%l**4
    this%l2 = this%l**2
    this%s_f = s_f
    this%s_f2 = this%s_f**2
    this%s_n = s_n
    this%K_stat=-1
    this%wait_vBoldScaling(:)=0 
    
    this%initialized=.true.
    this%scaling_set = .false.
    this%nt_pad = 0
    this%ene_pad = 0
    this%grad_pad = 0
    this%hess_pad = 0
    ! tolerance is set to largest value /10 since it is used to check
    ! if noise parameters are larger than tolerance * 10
    this%tol_noise_factor = 1d3
    this%tolerance = this%s_n(2) !HUGE(this%tolerance)/this%tol_noise_factor
end subroutine GPR_init_without_tp
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/setMeanOffset
!!
!! FUNCTION
!! Sets the offset value as chosen with this%OffsetType
!!
!! SYNOPSIS
subroutine setMeanOffset(this)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::  this  
    select case (this%OffsetType)
      case(0) 
        if (this%nt>0) then
         this%mean = SUM(this%es(1:this%nt))/REAL(this%nt, kind=rk)
        else 
         this%mean = 0d0
        end if
        if (printl>=6.and.this%idgf>1) &
          write(stdout,'("with ",I9, " training points, using a mean of ", &
            f10.5)') this%nt, this%mean
    case (1) 
        if (this%nt>0) then
         this%mean = MAXVAL(this%es(1:this%nt))+&
                    1d2*(ABS(MAXVAL(this%es(1:this%nt))-MINVAL(this%es(1:this%nt))))
        else if (this%nt==1) then ! Changed Compared To Paper!
         this%mean = this%es(1)
        else 
         this%mean = 0d0
        end if
        if (printl>=6.and.this%sdgf>1) &
          write(stdout,'("with ",I9, " training points, using a mean of ", &
            f10.5)') this%nt, this%mean
    case (2)
        if (this%nt>0) then
         this%mean = this%es(this%nt)
        else 
         this%mean = 0d0
        end if
        if (printl>=6.and.this%sdgf>1) &
          write(stdout,'("with ",I9, " training points, using a mean of ", &
            f10.5)') this%nt, this%mean
    case (3)
        if (this%manualOffset==0d0.and.printl>=6) &
            write(stdout,'("Do not forget to set an offset manually!")')
        if (this%nt>0) then
          if (this%addToMax) then
            this%mean = MAXVAL(this%es(1:this%nt))+this%manualOffset
          else
            this%mean = this%manualOffset
          end if
          if (this%idgf>1.and.printl>=6) then
            if (printl>=4.and.this%idgf>1) then
              write(stdout,'("Manual offset for mean is:",ES11.4)')&
                this%manualOffset
              write(stdout,'("Maxval of all energies is:",ES11.4)')&
                MAXVAL(this%es(1:this%nt))
              write(stdout,'("Therefore the mean is    :",ES11.4)')&
                this%mean
            end if
          end if
        end if
    case (4)
        if (this%nt>1) then
         this%mean = MINVAL(this%es(1:this%nt))-&
                    1d2*(ABS(MAXVAL(this%es(1:this%nt))-MINVAL(this%es(1:this%nt))))
        else if (this%nt==1) then ! Changed Compared To Paper!
         this%mean = this%es(1)
        else 
         this%mean = 0d0
        end if
        if (printl>=6.and.this%sdgf>1) &
          write(stdout,'("with ",I9, " training points, using a mean of ", &
            f10.5)') this%nt, this%mean
    case (5)
        this%mean=0d0!9d9 ! This is just to ensure that I notice something
        ! wrong when programming with it.
    case (7)
        if (this%sdgf>1.and.printl>=6) then
          write(stdout,'("Using the linear interpolation of first",&
                  " and last training point as prior mean.")')
        end if
        this%mean=0d0!SUM(this%es(1:this%nt))/REAL(this%nt, kind=rk)
    case (8)
        ! Taylor-based offset
        this%mean=0d0
    case default
        call dlf_fail("This Type of Offset is not known!")
    end select
    if (this%sdgf>1.and.printl>=6) then
      write(stdout,'("The mean is set to :",ES11.4)') this%mean
      write(stdout,'("Offset type is ",I4)') this%offsetType
    end if
end subroutine setMeanOffset    
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/manual_offset
!!
!! FUNCTION
!! Allows to set a manual offset. Only possible if (this%OffsetType==3)
!!
!! SYNOPSIS
subroutine manual_offset(this,offset,addToMax)
!! SOURCE
    implicit none    
    type(gpr_type),intent(inout)::  this   
    real(rk), intent(in)        ::  offset
    logical, intent(in), optional:: addToMax ! should the offset be
                                    ! added to the max val or chosen
                                    ! independently of the es
    if(.not.this%offsetType==3) &
        call dlf_fail("Can only set manual offset for offsetType 3!")
    this%manualOffset=offset
    if (present(addToMax)) then
      this%addToMax = addToMax
    else
      this%addToMax = .true.
    end if
    write(stdout,'("with ",I9, " training points, setting manual offset to ", &
            f10.5)') this%nt, this%manualOffset
end subroutine manual_offset
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_changeparameters
!!
!! FUNCTION
!! Allows the hyperparameter to be changed without initializing a new GPR.
!! You should never change hyperparameters without this subroutine.
!!
!! SYNOPSIS
subroutine GPR_changeparameters(this, gamma, s_f, s_n) 
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this    
    real(rk), intent(in)     ::  gamma, s_f, s_n(3)
    this%gamma=gamma
    this%l = 1d0/dsqrt(gamma)
    this%l4 = this%l**4
    this%l2 = this%l**2
    this%s_f=s_f
    this%s_f2=this%s_f**2
    this%s_n=s_n
    this%K_stat=-1
end subroutine GPR_changeparameters
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_changeOffsetType
!!
!! FUNCTION
!! Allows the type of offset to be changed
!!
!! SYNOPSIS
subroutine GPR_changeOffsetType(this, newOffsetType, ptOfExpansion_in, &
                                energy_in, grad_in, hess_in) 
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this    
    integer, intent(in)     ::  newOffsetType
    real(rk), intent(in), optional :: ptOfExpansion_in(this%sdgf),&
                                      energy_in, grad_in(this%sdgf),&
                                      hess_in(this%sdgf,this%sdgf)
    this%oldOffsetType = this%offsetType
    this%offsetType = newOffsetType
    if (newOffsetType==7.and.this%sdgf/=1) & 
      call dlf_fail("OffsetType 7 does only work with 1D GPRs!")
    if (this%offsetType==8) then
      ! Taylor Expansion
      if (.not.present(ptOfExpansion_in).or.&
          .not.present(energy_in).or.&
          .not.present(grad_in).or.&
          .not.present(hess_in)) then
        call dlf_fail("Taylor-based Offset needs several parameters(x/e/g/H)!")
      else
        if (.not.this%oldOffsetType==8) &
            call GPR_taylorBias_construct(this)
        call GPR_taylorBias_init(this, ptOfExpansion_in, energy_in, &
                                 grad_in, hess_in)
      end if      
    end if
    call setMeanOffset(this)
    this%K_stat = -1
    this%w_is_solution=.false.
    if (.not.this%iChol) then
      call writeObservationsInW(this)
    else
      call writeObservationsInW_newOrder(this)
    end if
end subroutine GPR_changeOffsetType
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_deleteOldestTPs
!!
!! FUNCTION
!! Deletes the oldest training points, i.e. those with the smallest indices
!!
!! SYNOPSIS
subroutine GPR_deleteOldestTPs(this, nToDel)
    implicit none
    type(gpr_type),intent(inout)::  this    
    integer,intent(in)          ::  nToDel
    integer                     ::  i
    if (nToDel>this%nt) then
        call dlf_fail("You are trying to erase more trainingpoints than there are.")
    end if
    ! copy xs, es, gs, hs to the beginning (nToDel+1 -> 1)
    do i = nToDel+1, this%nt
      this%xs(:,i-nToDel)   = this%xs(:,i)
      this%es(i-nToDel)     = this%es(i)
      if(this%internal == 2) then
        this%ixs(:,i-nToDel) = this%ixs(:,i)
        this%b_matrix(:,:,i-nToDel) = this%b_matrix(:,:,i)
      endif
      if (this%order>0) then
        this%gs(:,i-nToDel)   = this%gs(:,i)
          if(this%internal == 2) then
            this%igs(:,i-nToDel)   = this%igs(:,i)
          endif
        if (this%order>1) then
          this%hs(:,:,i-nToDel) = this%hs(:,:,i)
        end if
      end if
    end do
    ! ivalidate everything/recalc variables
    this%nt = this%nt-nToDel
    if (this%order==0) then
        this%nk = this%nt
    else if(this%order==1)then
        if(this%internal == 2) then
          this%nk = this%nt+this%nt*this%idgf
        else
          this%nk = this%nt+this%nt*this%sdgf
        endif
        this%ntg = this%nt
    else if(this%order==2)then
        this%nk = this%nt+this%nt*this%sdgf+this%nt*(this%sdgf)*(this%sdgf+1)/2
        this%ntg = this%nt
        this%nth = this%nt
    else if(this%order==42)then
        call dlf_fail("GPR_deleteOldestTPs not implemented for order 42!")
        this%nt = 0
        this%nk = 0
    end if
    this%nk_lin = ((this%nk+1)*this%nk)/2
    this%K_stat=-1    
    this%scaling_set = .false.
    call GPR_adjustArraySizes(this)
    call setMeanOffset(this)
    if (.not.this%iChol) then
      call writeObservationsInW(this)
    else
      call writeObservationsInW_newOrder(this)
    end if
end subroutine GPR_deleteOldestTPs
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_reduceKM_byDeletingGradTps
!!
!! FUNCTION
!! Deletes training points that are gradient based (given order==42)
!! equivalent to a reduction of the covariance matrix by nToDel entries.
!! Rounding errors could lead to a little lower reduction.
!!
!! SYNOPSIS
subroutine GPR_reduceKM_byDeletingGradTps(this, nKMToDel)
    implicit none
    type(gpr_type),intent(inout)::  this    
    integer,intent(in)          ::  nKMToDel
    integer                     ::  nToDel
    integer                     ::  i
    integer                     ::  moved = 0
    integer                     ::  firstToDelete = 1
    integer                     ::  lastToDelete = 1
    integer                     ::  moving
    if (this%ntg*(this%sdgf+1)<=nKMToDel) then
      call dlf_fail(&
      "Not possible to reduce covariance matrix by given nr of tps with gradient!")
    else
      nToDel = MAX(nKMToDel/(this%sdgf+1),1)
    end if
    if (printl>=4) then
      write(stdout,'("Deleting oldest ",I9,&
        " gradient-based trainingpoints. ")') nToDel
    end if
    
    if(this%order/=42)then
        call dlf_fail("GPR_reduceKM_byDeletingGradTps only implemented for order 42!")
    end if
    
    
    do while (moved/=nToDel)
      do i = firstToDelete, this%nt
        if (this%order_tps(i)==1) then
          ! found the first to Delete
          firstToDelete = i
          exit
        end if
      end do
      do i = firstToDelete, this%nt
        if (this%order_tps(i+1)==2) then
          ! found the last to Delete but they might not be enough
          lastToDelete = i
          exit
        else if (moved + (i+1)-firstToDelete == nToDel) then
          ! found the last to Delete and these are enough
          lastToDelete = i
          exit
        end if
      end do
      ! now [firstToDelete:lastToDelete] is the interval to be deleted
      moving = lastToDelete + 1 - firstToDelete
      do i = lastToDelete+1, this%nt
      !if (this%order_tps(i)/=1) &
      !    call dlf_fail("This TP should be a gradient based one.")
          ! move is possible
          this%xs(:,firstToDelete)   = this%xs(:,i)
          this%es(firstToDelete)     = this%es(i)
          this%gs(:,firstToDelete)   = this%gs(:,i)
          this%order_tps(firstToDelete) = this%order_tps(i)
          !moved = moved + 1i
          firstToDelete = firstToDelete + 1
        !end if
        !if (moved==nToDel) exit
      end do
      ! moving done
      moved = moved + moving
      if (moved>nToDel) call dlf_fail("Number of moved pts should not exceed nr of nToDel!")
      ! only relevant if not enough points are moved
      firstToDelete = lastToDelete + 2 ! since lastToDelete+1 is Hessian point
      ! ivalidate everything/recalc variables
      this%nt = this%nt-moving!nToDel
      this%ntg = this%ntg-moving!nToDel
      this%nt_pad = this%nt_pad + moving!nToDel
      this%grad_pad = this%grad_pad + moving!nToDel
      this%nk = this%nk - (this%sdgf + 1)*moving!nToDel
      this%nk_lin = ((this%nk+1)*this%nk)/2
    end do
    
    
    this%K_stat=-1    
    this%scaling_set = .false.
    call setMeanOffset(this)
    if (.not.this%iChol) then
      call writeObservationsInW(this)
    else
      call writeObservationsInW_newOrder(this)
    end if
end subroutine GPR_reduceKM_byDeletingGradTps
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_deleteNewestTPs
!!
!! FUNCTION
!! Deletes the newest training points, i.e. those with the highest indices
!!
!! SYNOPSIS
subroutine GPR_deleteNewestTPs(this, nToDel)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::  this    
    integer,intent(in)          ::  nToDel
    if (nToDel>this%nt) then
        call dlf_fail("You are trying to erase more trainingpoints than there are.")
    end if
    ! ivalidate everything/recalc variables
    this%nt = this%nt-nToDel
    if (this%order==0) then
        this%nk = this%nt
    else if(this%order==1)then
        if(this%internal == 2) then
          this%nk = this%nt + this%nt * this%idgf
        else
          this%nk = this%nt+this%nt*this%sdgf
        endif
        this%ntg = this%nt
    else if(this%order==2)then
        this%nk = this%nt+this%nt*this%sdgf+this%nt*(this%sdgf)*(this%sdgf+1)/2
        this%ntg = this%nt
        this%nth = this%nt
    else if(this%order==42)then
        call dlf_fail("GPR_deleteNewestTPs not implemented for order 42!")
        this%nt = 0
        this%nk = 0
    end if
    this%nk_lin = ((this%nk+1)*this%nk)/2
    this%K_stat=-1    
    this%scaling_set = .false.
    call GPR_adjustArraySizes(this)
    call setMeanOffset(this)
    if (.not.this%iChol) then
      call writeObservationsInW(this)
    else
      call writeObservationsInW_newOrder(this)
    end if 
end subroutine GPR_deleteNewestTPs
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_deleteTP
!!
!! FUNCTION
!! Delete a single training point with a certain index 
!!
!! SYNOPSIS
subroutine GPR_deleteTP(this, id)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::  this    
    integer,intent(in)          ::  id
    integer                     ::  i
    ! copy xs, es, gs, hs to the beginning (nToDel+1 -> 1)
    
    do i = id, this%nt
      this%xs(:,i)   = this%xs(:,i+1)
      if (this%internal == 2) then
        this%ixs(:,i) = this%ixs(:,i+1)
        this%b_matrix(:,:,i) = this%b_matrix(:,:,i+1)
      endif
      this%es(i)     = this%es(i+1)
      this%gs(:,i)   = this%gs(:,i+1)
      if(this%internal == 2) then
        this%igs(:,i) = this%igs(:,i+1)
      endif
      this%hs(:,:,i) = this%hs(:,:,i+1)
    end do
    ! ivalidate everything/recalc variables
    this%nt = this%nt-1
    if (this%order==0) then
      this%nk = this%nt
    else if(this%order==1)then
      if(this%internal == 2) then
        this%nk = this%nt+this%nt*this%idgf
      else
        this%nk = this%nt+this%nt*this%sdgf
      endif
      this%ntg = this%nt
    else if(this%order==2)then
      this%nk = this%nt+this%nt*this%sdgf+this%nt*(this%sdgf)*(this%sdgf+1)/2
      this%ntg = this%nt
      this%nth = this%nt
    else if(this%order==42)then
      call dlf_fail("GPR_deleteTP not implemented for order 42!")
      this%nt = 0
      this%nk = 0
    end if
    this%nk_lin = ((this%nk+1)*this%nk)/2
    this%K_stat=-1    
    this%scaling_set = .false.
    call GPR_adjustArraySizes(this)
    if (.not.this%iChol) then
      call writeObservationsInW(this)
    else
      call writeObservationsInW_newOrder(this)
    end if
    call setMeanOffset(this)
end subroutine GPR_deleteTP
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_deletefarestTPs
!!
!! FUNCTION
!! Delete the nToDel training points that are most far away from refpos
!!
!! SYNOPSIS
subroutine GPR_deletefarestTPs(this, refpos, nToDel)
!! SOURCE
    ! It could be that the points lie very closely, so that
    ! gamma should be adapted. For this reason always give max and min distance
    ! between the training points
    implicit none
    type(gpr_type),intent(inout)::  this    
    real(rk),intent(in)         ::  refpos(this%sdgf)
    integer,intent(in)          ::  nToDel    
    real(rk)                    ::  dist
    real(rk)                    ::  farestDist
    integer                     ::  id_fDist
    integer                     ::  i,nDel
    if (nToDel>this%nt) then
        call dlf_fail("You are trying to erase more trainingpoints than there are.")
    end if
    do nDel=1, nToDel
      farestDist=0d0
      do i = 1, this%nt
        dist = dot_product(this%xs(:,i)-refpos(:),this%xs(:,i)-refpos(:))
        if (farestDist<dist) then
          farestDist=dist
          id_fDist=i
        end if
      end do  
      call GPR_deleteTP(this,id_fDist)
    end do
end subroutine GPR_deletefarestTPs
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_copy_GPR
!!
!! FUNCTION
!! Creates a copy of a GPR_type
!!
!! SYNOPSIS
subroutine GPR_copy_GPR(source, targ)
!! SOURCE
    type(gpr_type), intent(in)  ::  source
    type(gpr_type), intent(out) ::  targ
    targ%iChol=source%iChol
    if(source%internal /= 0) then
      call GPR_construct(targ,source%nt,source%nat,source%sdgf,&
          source%OffsetType,source%kernel_type,source%order,source%internal)    
    else
      call GPR_construct(targ,source%nt,source%nat,source%sdgf,&
          source%OffsetType,source%kernel_type,source%order)
    endif
    targ%internal = source%internal    
    if (source%OffsetType==3) then
      call manual_offset(targ, source%manualOffset)
    end if
    if (associated(source%lowerLevelGPR)) then
        targ%lowerLevelGPR=>source%lowerLevelGPR
    end if
    if (source%order==0) then
      call GPR_init(targ, source%xs(1:source%sdgf,1:source%nt), source%gamma, source%s_f,&
        source%s_n, source%es(1:source%nt))
    else if (source%order==1) then
      if(source%internal == 2) then 
        call GPR_init(targ, source%xs(1:source%sdgf,1:source%nt), source%gamma, source%s_f,&
          source%s_n,source%es(1:source%nt),source%gs(1:source%sdgf,1:source%nt),&
          source%ixs(1:source%idgf,1:source%nt),source%igs(1:source%idgf,1:source%nt),&
          source%b_matrix(1:source%sdgf,1:source%idgf,1:source%nt))        
      else
        call GPR_init(targ, source%xs(1:source%sdgf,1:source%nt), source%gamma, source%s_f,&
          source%s_n, source%es(1:source%nt), source%gs(1:source%sdgf,1:source%nt))
      endif
    else if (source%order==2) then
      call GPR_init(targ, source%xs(1:source%sdgf,1:source%nt), source%gamma, source%s_f,&
        source%s_n, source%es(1:source%nt), &
        source%gs(1:source%sdgf,1:source%nt), &
        source%hs(1:source%sdgf,1:source%sdgf,1:source%nt))
    else
      call dlf_fail("NOT IMPLEMENTED IN GPR_copy_GPR!")
    end if
    
    targ%level=source%level
    targ%nSubLevels = source%nSubLevels
    if (source%OffsetType==8) then
      call GPR_taylorBias_construct(targ)
      call GPR_taylorBias_init(targ,source%taylor%ptOfExpansion,&
            source%taylor%energy,source%taylor%grad,&
            source%taylor%hess)
    end if
    targ%solveMethod = source%solveMethod
    targ%old_order=source%old_order
    targ%oldOffsetType = source%oldOffsetType
    targ%mean = source%mean
    targ%manualOffset = source%manualOffset
    targ%addToMax = source%addToMax
    if (.not.source%iChol) then
      targ%KM(:,:)=source%KM(:,:)
    else
      targ%nk_lin = source%nk_lin
      targ%nk_lin_old = source%nk_lin_old
      targ%KM_lin(1:targ%nk_lin) = source%KM_lin(1:source%nk_lin)
      targ%KM_lin_chol(1:targ%nk_lin) = source%KM_lin_chol(1:source%nk_lin)
    end if
    targ%nk = source%nk
    targ%nk_old = source%nk_old
    targ%nt = source%nt
    targ%nt_old = source%nt_old
    targ%ntg = source%ntg
    targ%ntg_old = source%ntg_old
    targ%nth = source%nth
    targ%nth_old = source%nth_old
    if (source%order==42) call dlf_fail("Must include trainingpoints one by one! Order 42 not supported in GPR_copy_GPR.")
    targ%tmpEnergy = source%tmpEnergy
    targ%K_stat=source%K_stat
    targ%w(1:source%nk)=source%w(1:source%nk)
    targ%w_is_solution=source%w_is_solution
    targ%wait_vBoldScaling(:)=source%wait_vBoldScaling(:)
    targ%ext_data = source%ext_data
    targ%providingKM = source%providingKM
    targ%provideTrafoInFile = source%provideTrafoInFile
    targ%massweight = source%massweight
    if (source%provideTrafoInFile) then
      targ%align_refcoords = source%align_refcoords
      targ%align_modes = source%align_modes
      targ%refmass = source%refmass
      targ%massweight = source%massweight
      targ%provideTrafoInFile = source%provideTrafoInFile
    end if
    call setMeanOffset(targ)
    targ%initialized=.true.
    targ%scaling_set = .false.
end subroutine GPR_copy_GPR
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/constructTrafoForDL_Find
!!
!! FUNCTION
!! Constructing the variables needed for coordinate trafos in dl-find
!!
!! SYNOPSIS
subroutine constructTrafoForDL_Find(this, nat, ncoord)
!! SOURCE
    implicit none
    type(gpr_type), intent(inout)   :: this
    integer, intent(in)             :: nat, ncoord
    if(.not.allocated(this%align_refcoords)) allocate (this%align_refcoords(3*nat))
    if(.not.allocated(this%align_modes))allocate (this%align_modes(3*nat,ncoord))
    if(.not.allocated(this%refmass))allocate (this%refmass(nat))
end subroutine constructTrafoForDL_Find
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/storeTrafoForDL_Find
!!
!! FUNCTION
!! Storing all the values needed for the trafo in dl-find 
!! in local variables of the gpr_type
!!
!! SYNOPSIS
subroutine storeTrafoForDL_Find(this, nat, ncoord, &
        align_refcoords,align_modes,refmass, massweight)
!! SOURCE
    implicit none
    type(gpr_type), intent(inout)   :: this
    integer, intent(in)             :: nat, ncoord
    real(rk), intent(in)            :: align_refcoords(3*nat)
    real(rk), intent(in)            :: align_modes(3*nat,ncoord)
    real(rk), intent(in)            :: refmass(nat)
    logical, intent(in)             :: massweight
    this%align_refcoords = align_refcoords
    this%align_modes = align_modes
    this%refmass = refmass
    this%massweight = massweight
    this%provideTrafoInFile = .true.
end subroutine storeTrafoForDL_Find
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/writeObservationsInW
!!
!! FUNCTION
!! This method copies all the observations (energies, gradients, hessians,...)
!! that were given to the gpr type in the weight vector to allow faster
!! solving of the linear system with the covariance matrix
!!
!! SYNOPSIS
subroutine writeObservationsInW(this)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    integer     ::  i, start, gNr, hNr
    if (this%iChol) call dlf_fail("should call writeObservationsInW_newOrder!")
if (this%kernel_type<=1) then
    ! the weight vector "this%w" is used to store the observations
    ! (energy, grad, hess)
    if (this%order>=0.and.this%order/=42) then
      ! use only energies
      do i = 1, this%nt
        this%w(i) = giveObservationE(this, i)
      end do
    end if
    
    if (this%order>=1.and.this%order/=42) then
        ! use gradients as well
        do i = 1, this%nt
          if(this%internal == 2) then
            this%w(this%nt+(i-1)*this%idgf+1:this%nt+(i-1)*this%idgf+this%idgf) = &
            giveObservationG(this,i,i)         
          else
            this%w(this%nt+(i-1)*this%sdgf+1:this%nt+(i-1)*this%sdgf+this%sdgf) = &
            giveObservationG(this,i,i)
          endif
        end do
    end if  
    
    if (this%order>=2.and.this%order/=42) then
      ! use hessians as well
      do i = 1, this%nt
        start = this%nt+this%nt*this%sdgf+&
                (i-1)*(this%sdgf)*(this%sdgf+1)/2 ! N*(N+1)/2 Hesse entries
        this%w(start+1:start+1+(this%sdgf)*(this%sdgf+1)/2)= &
          giveObservationH(this,i,i)
      end do
    end if
    if (this%order==42.and.this%nt>0) then
      if (.not.associated(this%order_tps)) &
        call dlf_fail("order_tps must be associated here (writeObservationsInW)!")
      start = 1
      do i = 1, this%nt
        this%w(start) = giveObservationE(this, i)
        start=start+1
      end do
      gNr = 1 ! iterator over points with gradients
      do i = 1, this%nt
        if (this%order_tps(i)>=1) then
          this%w(start:start+this%sdgf-1) = &
            giveObservationG(this,i, gNr)
          gNr = gNr + 1
          start=start+this%sdgf
        end if
      end do
      hNr = 1 ! iterator over points with hessians
      do i = 1, this%nt
        if (this%order_tps(i)>=2) then
          this%w(start:start+(this%sdgf)*(this%sdgf+1)/2-1)= &
                giveObservationH(this,i,hNr)
          start=start+(this%sdgf)*(this%sdgf+1)/2
          hNr = hNr + 1
        end if
      end do
    else if (this%order>2.and.this%order/=42) then
      call dlf_fail("The requested order is not implemented (writeObservationsInW)!")
    end if
else
    call dlf_fail("This type of kernel is not implemented! (writeObservationsInW)")
end if    
    this%w_is_solution=.false.
end subroutine writeObservationsInW
!!****

function giveObservationE(this, tpNr)
  real(rk)                      ::  giveObservationE
  type(gpr_type), intent(inout) ::  this
  integer, intent(in)           ::  tpNr
  real(rk)                      ::  pmean
  ! use only energies
  if (this%OffsetType==5) then
    call GPR_eval(this%lowerLevelGPR, this%xs(:,tpNr), pmean)
    giveObservationE = this%es(tpNr)-pmean
  else if (this%OffsetType==7) then
    call GPR_linearBias(this, this%xs(:,tpNr), pmean)
    giveObservationE = this%es(tpNr)-pmean
  else if (this%OffsetType==8) then
    call GPR_taylorBias(this, this%xs(:,tpNr), pmean)
    giveObservationE = this%es(tpNr)-pmean
  else
    giveObservationE = this%es(tpNr)-this%mean
  end if
end function giveObservationE

function giveObservationG(this, tpNr, gNr)
  type(gpr_type), intent(inout) ::  this
  integer, intent(in)           ::  tpNr, gNr
  real(rk),dimension(:),allocatable    ::  giveObservationG
  real(rk),dimension(:),allocatable    ::  pmean_g
  if(this%internal == 2) then
    allocate(giveObservationG(this%idgf))
    allocate(pmean_g(this%idgf))
  else
    allocate(giveObservationG(this%sdgf))
    allocate(pmean_g(this%sdgf))
  endif
  if (this%OffsetType==5) then
    if(this%internal == 2)then
      call GPR_eval_grad(this%lowerLevelGPR,this%xs(:,tpNr),pmean_g,this%b_matrix(:,:,tpNr))
    else
      call GPR_eval_grad(this%lowerLevelGPR,this%xs(:,tpNr),pmean_g)
    endif
    if(this%internal == 2) then
      giveObservationG(:) = this%igs(:,gNr) - pmean_g(:)
    else
      giveObservationG(:) = this%gs(:,gNr) - pmean_g(:)
    endif
  else if (this%OffsetType==7) then
    call GPR_linearBias_d(this,pmean_g) 
    giveObservationG(:) = this%gs(:,gNr) - pmean_g(:)
  else if (this%OffsetType==8) then
    call GPR_taylorBias_d(this, this%xs(:,tpNr), pmean_g)
    giveObservationG(:) = this%gs(:,gNr) - pmean_g(:)
  else
    if (this%internal == 2) then
      giveObservationG(:) = this%igs(:,gNr)
    else
      giveObservationG(:) = this%gs(:,gNr)
    endif
  end if
  deallocate(pmean_g)
end function giveObservationG

function giveObservationH(this, tpNr, hNr)
  type(gpr_type), intent(inout) ::  this
  integer, intent(in)           ::  tpNr, hNr
  real(rk)                      ::  giveObservationH(this%sdgf*(this%sdgf+1)/2)
  integer                       ::  j,k, os
  real(rk)                      ::  pmean_h(this%sdgf,this%sdgf)
  if (this%OffsetType==5) then      
    call GPR_eval_hess(this%lowerLevelGPR, this%xs(:,tpNr), pmean_h)
    os = -this%sdgf
    do j = 1, this%sdgf
      os = os + this%sdgf-(j-1)
      do k = j, this%sdgf                   
        ! Assuming symmetric Hessians, only store half the Hessians
        giveObservationH(os+k)= this%hs(k,j,hNr)-pmean_h(k,j)
      end do
    end do
  else if (this%OffsetType==6) then
    call dlf_fail("NOT IMPLEMENTED! (giveObservationH for OffsetType=6)")
    call GPR_eval_hess(this%lowerLevelGPR, this%xs(:,tpNr), pmean_h)           
    os = -this%sdgf
    do j = 1, this%sdgf
      os = os + this%sdgf-(j-1)
      do k = j, this%sdgf                   
        ! Assuming symmetric Hessians, only store half the Hessians
        giveObservationH(os+k)= this%hs(k,j,hNr)-pmean_h(k,j)
      end do
    end do
  else if (this%OffsetType==7) then
    call dlf_fail("NOT IMPLEMENTED! (giveObservationH for OffsetType=7)")
  else if (this%OffsetType==8) then
    call GPR_taylorBias_d2(this, pmean_h)
    os = -this%sdgf
    do j = 1, this%sdgf
      os = os + this%sdgf-(j-1)
      do k = j, this%sdgf                   
        ! Assuming symmetric Hessians, only store half the Hessians
        giveObservationH(os+k)= this%hs(k,j,hNr)-pmean_h(k,j)
      end do
    end do
  else
    os = -this%sdgf
    do j = 1, this%sdgf
      os = os + this%sdgf-(j-1)
      do k = j, this%sdgf                    
        ! Assuming symmetric Hessians, only store half the Hessians
        giveObservationH(os+k)= this%hs(k,j,hNr)
      end do
    end do
  end if
end function giveObservationH

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/writeObservationsInW_newOrder
!!
!! FUNCTION
!! This method copies all the observations (energies, gradients, hessians,...)
!! that were given to the gpr type in the weight vector to allow faster
!! solving of the linear system with the covariance matrix
!!
!! SYNOPSIS
subroutine writeObservationsInW_newOrder(this)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    integer     ::  i, start, gNr, hNr
    if (.not.this%iChol) call dlf_fail("should call writeObservationsInW!")
if (this%kernel_type<=1) then
    ! the weight vector "this%w" is used to store the observations
    ! (energy, grad, hess)
    if (this%order==0) then
        ! use only energies
        do i = 1, this%nt
          this%w(i) = giveObservationE(this, i)
        end do
    else if (this%order==1) then
      start = 1
      do i = 1, this%nt
        this%w(start) = giveObservationE(this, i)
        this%w(start+1:start+this%idgf) = giveObservationG(this,i,i)
        start = start + this%idgf + 1
      end do
    else if (this%order==2) then
      start = 1
      do i = 1, this%nt
        this%w(start) = giveObservationE(this, i)
        this%w(start+1:start+this%sdgf) = giveObservationG(this,i,i)
        start = start + this%sdgf + 1
        this%w(start:start+(this%sdgf)*(this%sdgf+1)/2-1) = &
            giveObservationH(this,i,i)
        start = start + (this%sdgf)*(this%sdgf+1)/2
      end do
!       call outputMatrixAsCSV(this%nk,1,this%w,'w_unsol2')
    else if (this%order==42) then
      start = 1
      gNr = 1
      hNr = 1
      do i = 1, this%nt
        this%w(start) = giveObservationE(this,i)
        start = start + 1
        if (this%order_tps(i)>=1) then
          this%w(start:start+this%sdgf-1) = giveObservationG(this,i,gNr)
          gNr = gNr + 1
          start = start + this%sdgf
        end if
        if (this%order_tps(i)==2) then
          this%w(start:start+(this%sdgf)*(this%sdgf+1)/2-1)= &
                giveObservationH(this,i,hNr)
          hNr = hNr + 1
          start = start + (this%sdgf)*(this%sdgf+1)/2
        else if (this%order_tps(i)>2) then
          call dlf_fail("iChol/writeObservationsInW_newOrder not implemented for this order of data point!")
        end if
      end do
    else 
      call dlf_fail("writeObservationsInW_newOrder is not implemented for GPR of order > 1")
    end if ! order
else
    call dlf_fail("This type of kernel is not implemented! (writeObservationsInW)")
end if    
    this%w_is_solution=.false.
end subroutine writeObservationsInW_newOrder
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_construct
!!
!! FUNCTION
!! Allocates the necessary variables/arrays to start
!! a GPR. It is the equivalent to an OOP-constructor.
!! GPRs of order 42 (arbitrary data):    
!!   nt is ignored, training points must be added vie GPR_add_tp
!!
!! SYNOPSIS
subroutine GPR_construct(this, nt, nat, sdgf, OffsetType, kernel_type, order,internal) 
!! SOURCE
    use dlf_global
    implicit none
    type(gpr_type),intent(inout)::this
    integer, intent(in) ::  nt
    integer, intent(in) ::  kernel_type
    integer, intent(in) ::  order, sdgf ! # parameters and
                                            ! order of approx 
                                            ! (incl grad and hess?) and
                                            ! # spacial coordinates                                       
                            ! spacial coordinates
                            ! the order of the hyperparameters is relevant
                            ! see below for the different kernels
    integer, intent(in) ::  nat ! # atoms
    integer, intent(in) ::  OffsetType  ! see at the beginning of this file
                                        ! for an explanation
    integer, intent(in),optional :: internal         
            
    if (this%constructed) then
      call dlf_fail("Run GPR_destroy before restarting!")
    end if
    if (OffsetType==7.and.sdgf/=1) &
      call dlf_fail("OffsetType 7 only works with 1D GPRs.")
     if (present(internal)) then
         this%internal = internal
     ! this%iChol = .false.
    else
      this%internal = 0
    endif
    this%OffsetType = OffsetType
    this%mean = 0d0
    this%kernel_type = kernel_type
    this%nt = nt
    this%nat = nat
    if (this%internal == 2) then
			this%sdgf = sdgf	!sdgf = nivar = 3N-6 in case of dlc
            this%idgf = sdgf+6
			!allocate(this%b_matrix(this%sdgf,this%idgf))
            !this%b_matrix = glob%b_hdlc
        else
            this%sdgf = sdgf
            this%idgf = sdgf
        end if
    if (this%sdgf>1.and.printl>=6) then
      write(stdout,'("Constructing a GP with", I9," degrees of freedom")') sdgf
    end if
    this%order = order
    if (this%OffsetType==6) then
      allocate(this%directSmallEigval(this%sdgf))
      this%directSmallEigval = 0d0
    else if (this%OffsetType==8) then
      call GPR_taylorBias_construct(this)
    end if
    if (this%order==0) then
      this%max_pad = 127
      this%nk = this%nt
      this%ntg = 0
      this%nth = 0
    else if(this%order==1)then
      this%max_pad = 63
      if (this%sdgf>300) this%max_pad = 31
      if (this%sdgf>500) this%max_pad = 15
      if (this%internal == 2) then
        this%nk = this%nt+this%nt*this%idgf
      else 
        this%nk = this%nt+this%nt*this%sdgf
      endif
      this%ntg = nt
      this%nth = 0
    else if(this%order==2)then
      this%max_pad = 31
      if (this%sdgf>300) this%max_pad = 15
      if (this%sdgf>500) this%max_pad = 1
      this%nk = this%nt+this%nt*this%sdgf+this%nt*(this%sdgf)*(this%sdgf+1)/2
      this%ntg = nt
      this%nth = nt
    else if(this%order==42)then
      this%max_pad = 63
      if (this%sdgf>300) this%max_pad = 31
      if (this%sdgf>500) this%max_pad = 15
      this%max_pad_hess42 = 2
      this%nk = 0
      if (this%nt>0) then
        write(stderr,'("Contruction with training points not permitted ", &
        "with order 42. Contruct empty GPR and use GPR_add_tp ", &
        "to add trainingpoints.")')      
        call dlf_fail("Contruction with trainingpoints not permitted with order 42.")
      end if
      this%ntg = 0
      this%nth = 0
    else
      call dlf_fail("This order is not implemented (GPR_construct).")
    end if
    this%ntg_old = 0
    this%nth_old = 0
    this%nt_pad = 0
    this%ene_pad = 0
    this%grad_pad = 0
    this%hess_pad = 0  
    this%nk_lin = ((this%nk+1)*this%nk)/2
    if (kernel_type==0) then
      allocate(this%scaling(5))
    else if (kernel_type==1) then
      allocate(this%scaling(5))
    else
      call dlf_fail("Kernel not implemented!")
    end if

    allocate(this%w_1(this%nk))
    this%old_order = this%order
    allocate(this%es_1(this%nt))
    allocate(this%xs_1(this%sdgf,this%nt))
    if(this%internal==2) allocate(this%b_matrix_1(this%sdgf,this%idgf,this%nt))
    if(this%internal==2) allocate(this%ixs_1(this%idgf,this%nt))
    allocate(this%es_1(this%nt))    
    if(this%internal == 2) then
      if(this%order>0) allocate(this%gs_1(this%sdgf,this%nt))
      if(this%order>0) allocate(this%igs_1(this%idgf,this%nt))
    else  
      if(this%order>0) allocate(this%gs_1(this%sdgf,this%nt))
    endif
    if(this%order>1) allocate(this%hs_1(this%sdgf,this%sdgf,this%nt))
    ! if(this%order==42) & ! must be present for all types
    allocate(this%order_tps_1(MAX(this%nt,1)))
    if(.not.this%iChol) then
      allocate(this%KM(this%nk,this%nk))
    else
      allocate(this%KM_lin_1(this%nk_lin))
      allocate(this%KM_lin_chol_1(this%nk_lin))
    end if
    if(this%internal == 2) then
      allocate(this%wait_vBoldScaling(this%idgf))    
    else  
      allocate(this%wait_vBoldScaling(this%sdgf))
    endif
    
    this%w=>this%w_1
    if(.not.this%iChol) then
      ! this%KM=>this%KM_1
    else
      this%KM_lin=>this%KM_lin_1
      this%KM_lin_chol=>this%KM_lin_chol_1
    end if
    this%xs=>this%xs_1
    this%es=>this%es_1
    this%gs=>this%gs_1
    this%hs=>this%hs_1
    if(this%internal == 2) this%b_matrix=>this%b_matrix_1
    if(this%internal == 2) this%ixs=>this%ixs_1
    if(this%internal == 2) this%igs=>this%igs_1
    ! if(this%order==42) &!must be present for all types
    this%order_tps=>this%order_tps_1
    
    this%lowerLevelGPR=>null()
    
    this%active_targets = 1
    this%constructed=.true.
    this%initialized=.false.
    this%scaling_set=.false.
    this%w_is_solution=.false.
    this%directSmallEigvalSet = .false.
    if (this%sdgf>1.and.printl>=6) then
      write(stdout,'("On construction Offsettype is ", I6)') this%offsetType
      write(stdout,'("On construction nt is         ", I6)') this%nt
      write(stdout,'("On construction nr of dgf is  ", I6)') this%sdgf
      if (this%offsetType==3) &
        write(stdout,'("ost3, meanOffset              ", ES11.4)') this%mean
    end if
end subroutine GPR_construct
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_linearBias
!!
!! FUNCTION
!! Give the offset as linear interpolation between the first 
!! and last es (does only work for one dimensional GPRs)
!!
!! SYNOPSIS

subroutine GPR_linearBias(this, pos, offset)
  type(gpr_type), intent(in) :: this
  real(rk), intent(in)       :: pos(this%sdgf)
  real(rk), intent(out)      :: offset
  real(rk)                   :: m
!   if (abs(this%xs(1,this%nt)-this%xs(1,1))<1d-16) &
!     call dlf_fail("first and last point are the same (GPR_linearBias)")
  if (this%sdgf/=1) call dlf_fail("GPR_linearBias_d, could not assert sdgf=1")
  m = (this%es(this%nt)-this%es(1))/(this%xs(1,this%nt)-this%xs(1,1))
  offset = this%es(1)+(pos(1)-this%xs(1,1))*m
           
end subroutine GPR_linearBias

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_linearBias_d
!!
!! FUNCTION
!! Give the derivative of the linear interpolation between the first 
!! and last es (does only work for one dimensional GPRs)
!!
!! SYNOPSIS

subroutine GPR_linearBias_d(this, offset)
  type(gpr_type), intent(in) :: this
  real(rk), intent(out)      :: offset(this%sdgf)
  if (this%sdgf/=1) call dlf_fail("GPR_linearBias_d, could not assert sdgf=1")
  offset(1) = (this%es(this%nt)-this%es(1))/(this%xs(1,this%nt)-this%xs(1,1))
end subroutine GPR_linearBias_d

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_taylorBias_construct
!!
!! FUNCTION
!! Construct a Taylor Expansion for the offset
!!
!! SYNOPSIS
subroutine GPR_taylorBias_construct(this)
  type(gpr_type), intent(inout) :: this
  allocate(this%taylor%ptOfExpansion(this%sdgf))
  allocate(this%taylor%grad(this%sdgf))
  allocate(this%taylor%hess(this%sdgf,this%sdgf))
  allocate(this%taylor%tmp_vec(this%sdgf))
  allocate(this%taylor%tmp_vec2(this%sdgf))
  this%taylor%initialized = .false.
end subroutine GPR_taylorBias_construct

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_taylorBias_init
!!
!! FUNCTION
!! Define a Taylor Expansion for the offset
!!
!! SYNOPSIS
subroutine GPR_taylorBias_init(this, ptOfExpansion_in, energy_in, &
                               grad_in, hess_in)
  type(gpr_type), intent(inout) :: this
  real(rk), intent(in)          :: ptOfExpansion_in(this%sdgf),&
                                   energy_in, grad_in(this%sdgf),&
                                   hess_in(this%sdgf,this%sdgf)
  if (.not.allocated(this%taylor%ptOfExpansion)) &
    call dlf_fail("Taylor-based system not constructed before initialized!")
  this%taylor%ptOfExpansion = ptOfExpansion_in
  this%taylor%energy = energy_in
  this%taylor%grad = grad_in
  this%taylor%hess = hess_in
  this%taylor%initialized = .true.
end subroutine GPR_taylorBias_init

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_taylorBias_destroy
!!
!! FUNCTION
!! Delete the Taylor Expansion for the offset
!!
!! SYNOPSIS
subroutine GPR_taylorBias_destroy(this)
  type(gpr_type), intent(inout) :: this
  if (.not.allocated(this%taylor%ptOfExpansion)) &
    call dlf_fail("Taylor-based system not constructed before destroyed!")
  deallocate(this%taylor%ptOfExpansion)
  deallocate(this%taylor%grad)
  deallocate(this%taylor%hess)
  deallocate(this%taylor%tmp_vec)
  deallocate(this%taylor%tmp_vec2)
end subroutine GPR_taylorBias_destroy

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_taylorBias
!!
!! FUNCTION
!! Give the offset as a Taylor Expansion
!!
!! SYNOPSIS
subroutine GPR_taylorBias(this, pos, energy_out)
  type(gpr_type), intent(inout) :: this
  real(rk), intent(in)          :: pos(this%sdgf)
  real(rk), intent(out)         :: energy_out
  if (.not.this%taylor%initialized) then
    call dlf_fail("No data given to the Taylor-based bias!")
  end if
  this%taylor%tmp_vec = pos - this%taylor%ptOfExpansion
  call dlf_matrix_vector_mult(this%sdgf,this%taylor%hess,&
                          this%taylor%tmp_vec,&
                          this%taylor%tmp_vec2,'N')
  energy_out = dot_product(this%taylor%tmp_vec2,&
                           this%taylor%tmp_vec)/2d0
  energy_out = energy_out + dot_product(this%taylor%tmp_vec,&
                                        this%taylor%grad)
  energy_out = energy_out + this%taylor%energy
end subroutine GPR_taylorBias

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_taylorBias_d
!!
!! FUNCTION
!! Give the derivative of the offset as a Taylor Expansion
!!
!! SYNOPSIS
subroutine GPR_taylorBias_d(this, pos, gradient_out)
  type(gpr_type), intent(inout) :: this
  real(rk), intent(in)          :: pos(this%sdgf)
  real(rk), intent(out)         :: gradient_out(this%sdgf)
  if (.not.this%taylor%initialized) then
    call dlf_fail("No data given to the Taylor-based bias!")
  end if
  this%taylor%tmp_vec = pos - this%taylor%ptOfExpansion
  call dlf_matrix_vector_mult(this%sdgf,this%taylor%hess,&
                          this%taylor%tmp_vec,&
                          this%taylor%tmp_vec2,'N')
  gradient_out = this%taylor%tmp_vec2 + this%taylor%grad                 
end subroutine GPR_taylorBias_d

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_taylorBias_d2
!!
!! FUNCTION
!! Give the second derivative of the offset as a Taylor Expansion
!!
!! SYNOPSIS
subroutine GPR_taylorBias_d2(this, hess_out)
  type(gpr_type), intent(inout) :: this
  real(rk), intent(out)         :: hess_out(this%sdgf, this%sdgf)
  if (.not.this%taylor%initialized) then
    call dlf_fail("No data given to the Taylor-based bias!")
  end if
  hess_out = this%taylor%hess
end subroutine GPR_taylorBias_d2

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_taylorBias_d2_column
!!
!! FUNCTION
!! Give the second derivative of the offset as a Taylor Expansion
!!
!! SYNOPSIS
subroutine GPR_taylorBias_d2_column(this, column, hess_column_out)
  type(gpr_type), intent(inout) :: this
  integer, intent(in)           :: column
  real(rk), intent(out)         :: hess_column_out(this%sdgf)
  if (.not.this%taylor%initialized) then
    call dlf_fail("No data given to the Taylor-based bias!")
  end if
  hess_column_out = this%taylor%hess(:,column)
end subroutine GPR_taylorBias_d2_column

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_add_tp
!!
!! FUNCTION
!! This subroutine is used to add training points to a GP
!! One can only add training points that fit the orde rof the GP
!! The new training point is given the highest index
!! For GPR of order 42:
!!   training points must be added by stating the arguments
!!   name in the call, so e.g. "call GPR_construct(this,1,xs,gs=?,hs=?)
!!
!! SYNOPSIS
subroutine GPR_add_tp(this, n_add_tp, xs, es, gs,ixs,igs,bmatrix, hs)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::  this
    integer, intent(in)         ::  n_add_tp
    real(rk), intent(in)        ::  xs(this%sdgf,n_add_tp)
                                ! spatial coordinates
                                ! the order of the hyperparameters is relevant
                                ! see below for the different kernels
    real(rk), intent(in)     ::  es(n_add_tp)           ! energies
    real(rk),optional, intent(in)     ::  ixs(this%idgf, n_add_tp)
    real(rk),optional, intent(in)     ::  gs(this%sdgf, n_add_tp)! gradients
    real(rk),optional, intent(in)     ::  igs(this%idgf, n_add_tp)
    real(rk),optional, intent(in)     ::  bmatrix(this%sdgf,this%idgf,n_add_tp)
    real(rk),optional, intent(in)     ::  hs(this%sdgf, this%sdgf, n_add_tp)  
                                                                ! hessians
    integer                     ::  orderTypeOfAddPts
    logical             ::  needToReallocate
    ! increase of covariance matrix when adding only one training points
    integer             ::  nk_inc_perTP
    ! how much should maximally be padded for the covariance matrix
    integer             ::  nk_max_padding = -1
    integer             ::  nk_lin_max_padding = -1
    integer             ::  nt_all
    integer             ::  ntg_all
    integer             ::  nth_all
    integer             ::  nk_all
    integer             ::  nk_lin_all
    real(rk), allocatable :: tmpKM(:,:)
    needToReallocate = .true.
    if (.not.this%iChol) &
      write(stdout, *) "Warning: It is not advisable to have iChol==.false. ",&
                       "when adding training points iteratively."
    if (printl>=6) &
        write(stdout,'("Adding a training point.")')
   ! if (this%internal == 2) then 
   !   deallocate(this%b_matrix)
   !   allocate(this%b_matrix(this%sdgf,this%idgf))
   !   this%b_matrix(:,:) = bmatrix(:,:)   
   ! endif
    select case (this%order)
      case(0)
        if (present(gs).or.present(hs).and.printl>=4) &
          write(stdout,'("Warning! Gradients and Hessians are ignored and not added!")')
        orderTypeOfAddPts = 0
      case(1)
        if(.not.present(gs)) &
        call dlf_fail("Must provide Energies and Gradients for 1st order interpolation!")
        if (present(hs).and.printl>=4)&
          write(stdout,'("Warning! Hessians are ignored and not added!")')
        orderTypeOfAddPts = 1
      case(2)
        if (.not.present(hs).or.(.not.present(gs))) &
          call dlf_fail("Must provide Energies,Grads and Hess for 2nd order interpolation!")
        orderTypeOfAddPts = 2        
      case(42)
        if (present(gs)) then
          if (present(hs)) then
            orderTypeOfAddPts = 2
          else
            orderTypeOfAddPts = 1
          end if
        else
          orderTypeOfAddPts = 0
        end if
      case default
        call dlf_fail("Order not implemented! (GPR_add_tp)")
    end select
    this%nt_old = this%nt
    this%ntg_old = this%ntg
    this%nth_old = this%nth    
    this%nk_old = this%nk
    this%nt = this%nt + n_add_tp
    if (orderTypeOfAddPts==0) then
      nk_inc_perTP = 1
    else if(orderTypeOfAddPts==1)then
      if (this%internal == 2) then
        nk_inc_perTP = 1+this%idgf
      else
        nk_inc_perTP = 1 + this%sdgf
      endif
     this%ntg = this%ntg + n_add_tp
    else if(orderTypeOfAddPts==2)then
      nk_inc_perTP = 1 + this%sdgf + ((this%sdgf)*(this%sdgf+1))/2
      this%ntg = this%ntg + n_add_tp
      this%nth = this%nth + n_add_tp
    end if
    this%nk = this%nk + n_add_tp * nk_inc_perTP
    if (this%order/=42) then
      nk_max_padding = this%max_pad * nk_inc_perTP
    else
      ! for order 42 I only want to pad for max_pad training points with gradients
      ! and max_pad_hess42 training points with Hessians 
      ! (these are included in the max_pad training points
      nk_max_padding = this%max_pad * (1 + this%idgf) + & ! tps with gradients
                       this%max_pad_hess42 * ((this%sdgf)*(this%sdgf+1))/2 ! Hessians for some training points
    end if
    
    if (this%K_stat==6.or.this%K_stat==7) then
        call dlf_fail("Do Cholesky before including new tps (tps are included in KM, but not solved)!")
    end if
    if (this%iChol) then
      this%nk_lin_old = this%nk_lin
      this%nk_lin = ((this%nk+1)*(this%nk))/2
      ! padding for linear KM must be done for 
      ! the complete KM size including padding
      ! minus the current KM size
      nk_lin_max_padding = &
                    (((this%nk+nk_max_padding)+1)*(this%nk+nk_max_padding))/2-&
                    this%nk_lin
    end if
    if (this%K_stat==5) then
      this%K_stat = 6
    end if
    
    ! ********************************************************************
    ! If there is padding space left everything gets comparably easy...
    select case (orderTypeOfAddPts)
      case (0)
        if(this%nt_pad>=n_add_tp .and. &
           this%ene_pad>=n_add_tp) &
             needToReallocate = .false.
        this%ene_pad = this%ene_pad - n_add_tp
      case (1)        
        if(this%nt_pad>=n_add_tp .and. &
           this%ene_pad>=n_add_tp.and. &
           this%grad_pad>=n_add_tp) &
             needToReallocate = .false.
        this%ene_pad = this%ene_pad - n_add_tp
        this%grad_pad = this%grad_pad - n_add_tp
      case (2)
        if(this%nt_pad>=n_add_tp .and. &
           this%ene_pad>=n_add_tp.and. &
           this%grad_pad>=n_add_tp.and. &
           this%hess_pad>=n_add_tp) &
             needToReallocate = .false.
        this%ene_pad = this%ene_pad - n_add_tp
        this%grad_pad = this%grad_pad - n_add_tp
        this%hess_pad = this%hess_pad - n_add_tp
      case default
        call dlf_fail("This type of point is not supported!")
    end select
    if (.not.this%iChol) needToReallocate = .true.
    ! this has to be done for all cases
    this%nt_pad = this%nt_pad - n_add_tp
    
    ! ********************************************************************
    ! If there is no padding space left things have to be allocated anew
    if (needToReallocate) then
      ! One needs to define how much padding should be done
      this%nt_pad = this%max_pad ! for every order
      this%ene_pad = this%max_pad ! for every order
      select case (this%order)        
        case(0)
          this%grad_pad = 0
          this%hess_pad = 0  
        case(1)
          this%grad_pad = this%max_pad
          this%hess_pad = 0  
        case(2)
          this%grad_pad = this%max_pad
          this%hess_pad = this%max_pad
        case(42)
          this%grad_pad = this%max_pad
          this%hess_pad = this%max_pad_hess42
      end select
      ! the overall number of variables including padding
      nt_all = this%nt + this%nt_pad
      ntg_all = this%ntg + this%grad_pad
      nth_all = this%nth + this%hess_pad
      nk_all = this%nk + nk_max_padding
      nk_lin_all = this%nk_lin + nk_lin_max_padding

      if (this%active_targets==1) then
        
        ! Allocate second arrayset
        allocate(this%w_2(nk_all))
        allocate(this%xs_2(this%sdgf,nt_all))
        if(this%internal == 2) allocate(this%ixs_2(this%idgf,nt_all))
        if(this%internal == 2) allocate(this%b_matrix_2(this%sdgf,this%idgf,nt_all))
        allocate(this%es_2(nt_all))
        if(ntg_all>0) allocate(this%gs_2(this%idgf,ntg_all))
        if(ntg_all>0) then 
          if(this%internal == 2) allocate(this%igs_2(this%idgf,ntg_all))
        endif
        if(nth_all>0) allocate(this%hs_2(this%sdgf,this%sdgf,nth_all))
        !if(this%order==42) ! must be present for all
        allocate(this%order_tps_2(nt_all))
        if(.not.this%iChol) then
          deallocate(this%KM)
          allocate(this%KM(this%nk,this%nk))
        else
          allocate(this%KM_lin_2(nk_lin_all))
          allocate(this%KM_lin_chol_2(nk_lin_all))
        end if
        ! Copy the old data to the new set
        this%xs_2(:,1:this%nt_old) = this%xs_1(:,1:this%nt_old)
        if(this%internal == 2) then
          this%ixs_2(:,1:this%nt_old) = this%ixs_1(:,1:this%nt_old)
          this%b_matrix_2(:,:,1:this%nt_old) = this%b_matrix_1(:,:,1:this%nt_old)
        endif
        this%es_2(1:this%nt_old) = this%es_1(1:this%nt_old)
        if (this%ntg_old>0) then
          
          if(this%internal == 2) then
            this%igs_2(:,1:this%ntg_old) = this%igs_1(:,1:this%ntg_old)
          else 
            this%gs_2(:,1:this%ntg_old) = this%gs_1(:,1:this%ntg_old)
          endif
        end if
        if (this%nth_old>0) then
          this%hs_2(:,:,1:this%nth_old) = this%hs_1(:,:,1:this%nth_old)
        end if
        !if(this%order==42) &
        this%order_tps_2(1:this%nt_old) = this%order_tps_1(1:this%nt_old)
        if(this%iChol) then
          this%KM_lin_2(1:this%nk_lin_old) = &
                this%KM_lin_1(1:this%nk_lin_old)
          this%KM_lin_chol_2(1:this%nk_lin_old) = &
                this%KM_lin_chol_1(1:this%nk_lin_old)
        end if
        ! Deallocate the old set
        deallocate(this%w_1)     
        if(.not.this%iChol) then
          ! deallocate(this%KM_1)
        else
          deallocate(this%KM_lin_1)
          deallocate(this%KM_lin_chol_1)
        end if
        deallocate(this%xs_1)
        deallocate(this%es_1)
        if(this%internal == 2) deallocate(this%b_matrix_1)
        if(this%internal == 2) deallocate(this%ixs_1)
        if (this%ntg_old>0) deallocate(this%gs_1)
        if (this%nth_old>0) deallocate(this%hs_1)
        !if (this%order==42) &
        deallocate(this%order_tps_1)
        ! Rewrite pointers to new set
        this%w=>this%w_2
        if(.not.this%iChol) then
        !  this%KM=>this%KM_2
        else
          this%KM_lin=>this%KM_lin_2
          this%KM_lin_chol=>this%KM_lin_chol_2
        end if
        this%xs=>this%xs_2
        this%es=>this%es_2
        this%gs=>this%gs_2
        this%hs=>this%hs_2
        if(this%internal == 2) then
          this%ixs=>this%ixs_2
          this%igs=>this%igs_2
          this%b_matrix=>this%b_matrix_2
        endif
        this%order_tps=>this%order_tps_2
        ! New set is referenced
        this%active_targets=2
      else if (needToReallocate.and.this%active_targets==2) then
        ! Allocate first arrayset
        allocate(this%w_1(nk_all))
        allocate(this%xs_1(this%sdgf,nt_all))
        if(this%internal == 2) allocate(this%ixs_1(this%idgf,nt_all))
        if(this%internal == 2) allocate(this%b_matrix_1(this%sdgf,this%idgf,nt_all))
        allocate(this%es_1(nt_all))
        if(ntg_all>0) allocate(this%gs_1(this%sdgf,ntg_all))
        if(ntg_all>0) then
          if(this%internal == 2) allocate(this%igs_1(this%idgf,ntg_all))
        endif
        if(nth_all>0) allocate(this%hs_1(this%sdgf,this%sdgf,nth_all))
        !if(this%order==42) &
        allocate(this%order_tps_1(nt_all))
        if(.not.this%iChol) then
          deallocate(this%KM)
          allocate(this%KM(this%nk,this%nk))
        else
          allocate(this%KM_lin_1(nk_lin_all))
          allocate(this%KM_lin_chol_1(nk_lin_all))
        end if
        ! Copy the old data to the new set
        this%xs_1(:,1:this%nt_old) = this%xs_2(:,1:this%nt_old)
        if(this%internal == 2) then
          this%ixs_1(:,1:this%nt_old) = this%ixs_2(:,1:this%nt_old)
          this%b_matrix_1(:,:,1:this%nt_old) = this%b_matrix_2(:,:,1:this%nt_old)
        endif
        this%es_1(1:this%nt_old) = this%es_2(1:this%nt_old)
        if (this%ntg_old>0) then
          !this%gs_1(:,1:this%ntg_old) = this%gs_2(:,1:this%ntg_old)
          if(this%internal == 2) then
            this%igs_1(:,1:this%ntg_old) = this%igs_2(:,1:this%ntg_old)
          else
            this%gs_1(:,1:this%ntg_old) = this%gs_2(:,1:this%ntg_old)
          endif
        end if
        if (this%nth_old>0) then
          this%hs_1(:,:,1:this%nth_old) = this%hs_2(:,:,1:this%nth_old)
        end if
        !if(this%order==42) then
        this%order_tps_1(1:this%nt_old) = this%order_tps_2(1:this%nt_old)
        !end if
        if (this%iChol) then
          this%KM_lin_1(1:this%nk_lin_old) = &
                this%KM_lin_2(1:this%nk_lin_old)
          this%KM_lin_chol_1(1:this%nk_lin_old) = &
                this%KM_lin_chol_2(1:this%nk_lin_old)
        end if
        ! Deallocate the old set
        deallocate(this%w_2)
        if(.not.this%iChol) then
          !deallocate(this%KM_2)
        else
          deallocate(this%KM_lin_2)
          deallocate(this%KM_lin_chol_2)
        end if
        deallocate(this%xs_2)
        deallocate(this%es_2)
        if(this%internal == 2) deallocate(this%b_matrix_2)
        if(this%internal == 2) deallocate(this%ixs_2)
        if (this%ntg_old>0) deallocate(this%gs_2)
        if (this%ntg_old>0) then
          if(this%internal == 2) deallocate(this%igs_2)
        endif
        if (this%nth_old>0) deallocate(this%hs_2)
        !if (this%order==42) &
        deallocate(this%order_tps_2)
        
        ! Rewrite pointers to new set
        this%w=>this%w_1
        if(.not.this%iChol) then
          !this%KM=>this%KM_1
        else
          this%KM_lin=>this%KM_lin_1
          this%KM_lin_chol=>this%KM_lin_chol_1
        end if
        this%xs=>this%xs_1
        this%es=>this%es_1
        this%gs=>this%gs_1
        this%hs=>this%hs_1
        if(this%internal == 2) then
          this%ixs=>this%ixs_1
          this%igs=>this%igs_1
          this%b_matrix=>this%b_matrix_1
        endif
        this%order_tps=>this%order_tps_1
        
        ! New set is referenced
        this%active_targets=1
      else
        call dlf_fail("Error with active_targets variable in GPR_add_tp!")
      end if
    else if(.not.needToReallocate) then
      ! no need to reallocate variables
      if (printl>=6) write(stdout,'(&
        "No need to reallocate the arrays (padding sufficient) ")')
    end if
    
    !************************************************************
    ! Copy new data to the new sets
    this%xs(:,this%nt_old+1:this%nt) = xs(:,1:n_add_tp)
    if(this%internal == 2) then
      this%ixs(:,this%nt_old+1:this%nt) = ixs(:,1:n_add_tp)
      this%b_matrix(:,:,this%nt_old+1:this%nt) = bmatrix(:,:,1:n_add_tp)
    endif
    this%es(this%nt_old+1:this%nt) = es(1:n_add_tp)
    if (orderTypeOfAddPts>0) then
      
      if(this%internal == 2) then
        this%igs(:,this%ntg_old+1:this%ntg) = igs(:,1:n_add_tp)
      else 
        this%gs(:,this%ntg_old+1:this%ntg) = gs(:,1:n_add_tp)
      endif
    end if
    if (orderTypeOfAddPts>1) then
      this%hs(:,:,this%nth_old+1:this%nth) = hs(:,:,1:n_add_tp)
    end if
    !if(this%order==42) then
    this%order_tps(this%nt_old+1:this%nt) = orderTypeOfAddPts
    !end if
    if (this%iChol) then
      this%KM_lin(this%nk_lin_old+1:this%nk_lin) = 0d0
      this%KM_lin_chol(this%nk_lin_old+1:this%nk_lin) = 0d0
    end if
    
    this%w_is_solution=.false.
    call setMeanOffset(this)
    if (.not.this%iChol) then
      this%K_stat=-1
      call writeObservationsInW(this)
    else
      call writeObservationsInW_newOrder(this)
      if (this%nt_old==0) this%K_stat=-1
    end if
    this%initialized=.true.
    this%scaling_set = .false.
    this%directSmallEigvalSet = .false.
     if (printl>=4) &
        write(stdout,'("Included", I9, " number of energy calculations until now.")') this%nt
end subroutine GPR_add_tp
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_adjustArraySizes
!!
!! FUNCTION
!! This method adjusts the size of the arrays to be more memory efficient.
!! If for example some training points have been deleted,
!! this method can resize the arrays in the GP to have no useless 
!! memory consumption.
!!
!! SYNOPSIS
subroutine GPR_adjustArraySizes(this)
!! SOURCE
    implicit none
    type(gpr_type), intent(inout):: this
    real(rk)                     :: tmp3(this%sdgf,this%sdgf,this%nt)
    real(rk)                     :: tmp2(this%sdgf,this%nt)
    real(rk)                     :: itmp2(this%idgf, this%nt)
    real(rk)                     :: itmp3(this%sdgf,this%idgf,this%nt)
    real(rk)                    ::  tmp1(this%nt)
    real(rk)                    ::  tmpnk(this%nk)
    real(rk)                    ::  tmp_KM_lin(this%nk_lin)
    if (this%order==42) &
      call dlf_fail("GPR_adjustArraySizes not implemented for order 42!")
    ! eliminates all padding
    this%nt_pad = 0
    this%ene_pad = 0
    this%grad_pad = 0
    this%hess_pad = 0
if (this%active_targets==1) then
    tmp2(1:this%sdgf,1:this%nt)=this%xs(1:this%sdgf,1:this%nt)
    if (this%internal == 2) then
      itmp2(1:this%idgf,1:this%nt) = this%ixs(1:this%idgf, 1:this%nt)
      itmp3(1:this%sdgf,1:this%idgf,1:this%nt) = this%b_matrix(1:this%sdgf,1:this%idgf,1:this%nt)
      deallocate(this%ixs_1)
      allocate(this%ixs_1(this%idgf,this%nt))
      deallocate(this%b_matrix_1)
      allocate(this%b_matrix_1(this%sdgf,this%idgf,this%nt))
      this%ixs_1(:,:)=itmp2(:,:)
      this%b_matrix_1(:,:,:) = itmp3(:,:,:)
      this%ixs=> this%ixs_1
      this%b_matrix =>this%b_matrix_1
    end if 
    deallocate(this%xs_1)
    allocate(this%xs_1(this%sdgf,this%nt))
    this%xs_1(:,:)=tmp2(:,:)
    this%xs=> this%xs_1
    tmp1(:)=this%es_1(1:this%nt)
    deallocate(this%es_1)
    allocate(this%es_1(this%nt))
    this%es_1(:)=tmp1(:)
    this%es=> this%es_1
    
  if (this%order==42) then
    call dlf_fail("GPR_adjustArraySizes not correctly implemented for order 42!")
  else if (this%order>=1) then
    tmp2(1:this%sdgf,1:this%nt)=this%gs_1(1:this%sdgf,1:this%nt)
    deallocate(this%gs_1)
    allocate(this%gs_1(this%sdgf,this%nt))
    this%gs_1(:,:)=tmp2(:,:)
    this%gs=>this%gs_1
    if(this%internal == 2) then
      itmp2(1:this%idgf,1:this%nt)=this%igs_1(1:this%idgf,1:this%nt)
      deallocate(this%igs_1)
      allocate(this%igs_1(this%idgf,this%nt))
      this%igs_1(:,:)=itmp2(:,:)
      this%igs=>this%igs_1  
    endif    
   if (this%order>=2) then
    tmp3(:,:,:)=this%hs_1(1:this%sdgf,1:this%sdgf,1:this%nt)
    deallocate(this%hs_1)
    allocate(this%hs_1(this%sdgf,this%sdgf,this%nt))
    this%hs_1(:,:,:)=tmp3(:,:,:)
    this%hs=>this%hs_1
   end if
  end if
    tmpnk(1:this%nk)=this%w_1(1:this%nk)
    deallocate(this%w_1)    
    allocate(this%w_1(this%nk))
    this%w_1(:)=tmpnk
    this%w=> this%w_1
    if (.not.this%iChol) then
      deallocate(this%KM)
      allocate(this%KM(this%nk,this%nk))
      !this%KM=>this%KM_1
    else
      tmp_KM_lin(1:this%nk_lin) = this%KM_lin_1(1:this%nk_lin)
      deallocate(this%KM_lin_1)
      allocate(this%KM_lin_1(this%nk_lin))
      this%KM_lin_1 (:) = tmp_KM_lin(:)
      
      tmp_KM_lin(1:this%nk_lin) = this%KM_lin_chol_1(1:this%nk_lin)
      deallocate(this%KM_lin_chol_1)
      allocate(this%KM_lin_chol_1(this%nk_lin))
      this%KM_lin_chol_1 (:) = tmp_KM_lin(:)
      this%KM_lin=>this%KM_lin_1
      this%KM_lin_chol=>this%KM_lin_chol_1
    end if
    this%K_stat=-1
else if (this%active_targets==2) then
    tmp2(1:this%sdgf,1:this%nt)=this%xs(1:this%sdgf,1:this%nt)
    deallocate(this%xs_2)
    allocate(this%xs_2(this%sdgf,this%nt))
    this%xs_2(:,:)=tmp2(:,:)
    this%xs=> this%xs_2
    if(this%internal == 2) then
      itmp2(1:this%idgf,1:this%nt)=this%ixs(1:this%idgf,1:this%nt)
      itmp3(1:this%sdgf,1:this%idgf,1:this%nt) = this%b_matrix(1:this%sdgf,1:this%idgf,1:this%nt)
      deallocate(this%ixs_2)
      allocate(this%ixs_2(this%idgf,this%nt))
      deallocate(this%b_matrix_2)
      allocate(this%b_matrix_2(this%sdgf,this%idgf,this%nt))
      this%ixs_2(:,:)=itmp2(:,:)
      this%b_matrix_2(:,:,:)= itmp3(:,:,:)
      this%b_matrix=>this%b_matrix_2
      this%ixs=> this%ixs_2
    endif
    tmp1(:)=this%es_2(1:this%nt)
    deallocate(this%es_2)
    allocate(this%es_2(this%nt))
    this%es_2(:)=tmp1(:)
    this%es=> this%es_2
  if (this%order>=1) then
    tmp2(1:this%sdgf,1:this%nt)=this%gs_2(1:this%sdgf,1:this%nt)
    deallocate(this%gs_2)
    allocate(this%gs_2(this%sdgf,this%nt))
    this%gs_2(:,:)=tmp2(:,:)
    this%gs=>this%gs_2
    if(this%internal == 2) then
      itmp2(1:this%idgf,1:this%nt)=this%igs_2(1:this%idgf,1:this%nt)
      deallocate(this%igs_2)
      allocate(this%igs_2(this%idgf,this%nt))
      this%igs_2(:,:)=itmp2(:,:)
      this%igs=>this%igs_2  
    endif    
   if (this%order>=2) then
    tmp3(:,:,:)=this%hs_2(1:this%sdgf,1:this%sdgf,1:this%nt)
    deallocate(this%hs_2)
    allocate(this%hs_2(this%sdgf,this%sdgf,this%nt))
    this%hs_2(:,:,:)=tmp3(:,:,:)
    this%hs=>this%hs_2
   end if
  end if
    tmpnk(1:this%nk)=this%w_2(1:this%nk)
    deallocate(this%w_2)    
    allocate(this%w_2(this%nk))
    this%w_2(:)=tmpnk
    this%w=> this%w_2
    if (.not.this%iChol) then
      deallocate(this%KM)
      allocate(this%KM(this%nk,this%nk))
      ! this%KM=>this%KM_2
    else
      tmp_KM_lin(1:this%nk_lin) = this%KM_lin_2(1:this%nk_lin)
      deallocate(this%KM_lin_2)
      allocate(this%KM_lin_2(this%nk_lin))
      this%KM_lin_2 (:) = tmp_KM_lin(:)
      
      tmp_KM_lin(1:this%nk_lin) = this%KM_lin_chol_2(1:this%nk_lin)
      deallocate(this%KM_lin_chol_2)
      allocate(this%KM_lin_chol_2(this%nk_lin))
      this%KM_lin_chol_2 (:) = tmp_KM_lin(:)
      this%KM_lin=>this%KM_lin_2
      this%KM_lin_chol=>this%KM_lin_chol_2
    end if
    
    this%K_stat=-1
else    
    call dlf_fail("ERROR IN GPR_adjustArraySizes")
end if    
    if (.not.this%iChol) then
      call writeObservationsInW(this)
    else
      call writeObservationsInW_newOrder(this)
    end if
end subroutine GPR_adjustArraySizes
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_destroy
!!
!! FUNCTION
!! This method is used to destroy a GP, i.e. delete all values, 
!! free the allocated memory and dereference all pointers
!! After calling this, a GPR must be called again with gpr_construct to 
!! use it again
!!
!! SYNOPSIS
recursive subroutine GPR_destroy(this)  
!! SOURCE
    implicit none
    type(gpr_type), intent(inout)::this
    if (associated(this%lowerLevelGPR)) then
    if (printl>=6) &
      write(stdout,'("There is a deeper GPR!")')
!         tmpPt=> this%lowerLevelGPR
        call GPR_destroy(this%lowerLevelGPR) 
        deallocate(this%lowerLevelGPR)
        this%lowerLevelGPR=>null()
    else 
    if (printl>=6) &
      write(stdout,'("GPR_destroy: reached the end of the multi-level recursion!")')
    end if
    
    
    if(allocated(this%directSmallEigval)) &
      deallocate(this%directSmallEigval)
    if (this%active_targets==1) then
        deallocate(this%w_1)
        if(.not.this%iChol) then
          deallocate(this%KM)
        else
          deallocate(this%KM_lin_1)
          deallocate(this%KM_lin_chol_1)
        end if
        deallocate(this%xs_1)
        if(this%internal == 2) deallocate(this%ixs_1)
        if(this%internal == 2) deallocate(this%b_matrix_1)
        deallocate(this%es_1)
        if (this%ntg>0) then
          deallocate(this%gs_1)
          if(this%internal == 2) deallocate(this%igs_1)
        end if
        if (this%nth>0) then
          deallocate(this%hs_1)     
        end if
        !if (this%order==42) 
        deallocate(this%order_tps_1)
    else if (this%active_targets==2) then
        deallocate(this%w_2)
        if(.not.this%iChol) then
          deallocate(this%KM)
        else
          deallocate(this%KM_lin_2)
          deallocate(this%KM_lin_chol_2)
        end if
        deallocate(this%xs_2)
        if(this%internal == 2) deallocate(this%ixs_2)
        if(this%internal == 2) deallocate(this%b_matrix_2)
        deallocate(this%es_2)
        if (this%ntg>0) then
          deallocate(this%gs_2)
          if(this%internal == 2) deallocate(this%igs_2)
        end if
        if (this%nth>0) then
          deallocate(this%hs_2)     
        end if
        !if (this%order==42) 
        deallocate(this%order_tps_2)
    else
        call dlf_fail("ERROR WITH ACTIVE TARGETS VARIABLE IN GPR_DESTROY!")
    end if
    deallocate(this%scaling)
    deallocate(this%wait_vBoldScaling)
    if (this%offsetType==8) then
      call GPR_taylorBias_destroy(this)
    end if
    call gpr_prfo_destroy(this)
    this%K_stat = -1
    this%w_is_solution=.false.
    this%ext_data = .false.
    this%constructed=.false.
    this%initialized=.false.
    this%lowerLevelGPR=>null()
    this%nt_pad = 0
    this%ene_pad = 0
    this%grad_pad = 0
    this%hess_pad = 0 
end subroutine GPR_destroy
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_reload
!!
!! FUNCTION
!! This method is used to destroy a GP and immediately construct it again 
!! without any training points but same parameters.
!! Deletes all values!
!!
!! SYNOPSIS
subroutine GPR_reload(this)  
!! SOURCE
    implicit none
    type(gpr_type), intent(inout)::this
    ! save nat, sdgf, offsetType, kernel_type, order, gamma, s_f, s_n
    integer     ::  nat_save, sdgf_save, meanOffset_save,&
                    kernel_save, order_save
    real(rk)    ::  gamma_save, s_f_save, s_n_save(3)
    nat_save = this%nat
    sdgf_save = this%sdgf
    meanOffset_save = this%offsetType
    kernel_save = this%kernel_type
    order_save = this%order
    gamma_save = this%gamma
    s_f_save = this%s_f
    s_n_save(:) = this%s_n
    call GPR_destroy(this)
    call GPR_construct(this, 0, nat_save, sdgf_save, meanOffset_save, &
                       kernel_save, order_save)
    call GPR_init_without_tp(this,gamma_save,s_f_save,s_n_save)
end subroutine GPR_reload
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/Cholesky_KM
!!
!! FUNCTION
!! Computes the cholesky decomposition of KM for further use in 
!! solve_LGS_use_KM_chol
!!
!! SYNOPSIS
subroutine Cholesky_KM(this)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    integer                 ::  info, i, j
    if (this%K_stat==0) then
        call DPOTRF ('U',this%nk, this%KM, this%nk, info)
        if (info/=0) then
          write(stdout,*) "DPOTRF failed with code ", info
                call dlf_fail(&
    "Cholesky Decomposition failed (Cholesky_KM) -use other s_n/gamma/kernel!")
        end if
        this%K_stat = 3
    else if (this%K_stat==3) then
        call dlf_fail("Covariance matrix is cholesky decomposed already!")
    else
        call dlf_fail("Covariance matrix has no valid form (or is inverted)!")
    end if
end subroutine Cholesky_KM
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/Cholesky_KM_myown
!!
!! FUNCTION
!! Computes the cholesky decomposition of the Covariance matrix
!!
!! SYNOPSIS
subroutine Cholesky_KM_myown_logdet(this, useOld,logdet)
!! SOURCE
  implicit none
  type(gpr_type),intent(inout)  ::  this
  logical, intent(in)           ::  useOld
  real(rk),intent(out)          ::  logdet
  logical                       ::  nans_occured
  nans_occured = .false.
  if (useOld) then
    call Cholesky_myOwn_logdet(this%nk, this%KM_lin, &
                        this%KM_lin_chol,nans_occured,logdet,this%nk_old)
  else
    call Cholesky_myOwn_logdet(this%nk, this%KM_lin, &
                        this%KM_lin_chol,nans_occured,logdet)
  end if
  if(nans_occured) call dlf_fail("cholesky failed")
  do while (nans_occured)
    if (printl>=2) &
      write(stdout,'(" ")')
      write(stdout,'(" ")')
      write(stdout,'("****************************************************")')
      write(stdout,'("Warning: The Cholesky decomposition became instable!")')
      write(stdout,'("The noise parameters for energies and gradients ",&
            "have to be increased.")')
      write(stdout,'("(The optimization might not be converging)")')
      call GPR_changeparameters(this, this%gamma, this%s_f, &
                              1.5d0*this%s_n)
      write(stdout, '(A, E10.3, E10.3, E10.3)') "New noise parameters are:",&
                                         this%s_n
      if (this%s_n(2)>this%tolerance*this%tol_noise_factor) &
        call dlf_fail("Noise parameters are too large... Optimization failed!")
      write(stdout,'("Recalculating covariance matrix...")')
      call calc_cov_mat_newOrder(this)
      write(stdout,'("Trying to solve new covariance matrix.")')
      if (useOld) then
        call Cholesky_myOwn(this%nk, this%KM_lin, &
                          this%KM_lin_chol,nans_occured,this%nk_old)
      else
        call Cholesky_myOwn(this%nk, this%KM_lin, &
                          this%KM_lin_chol,nans_occured)
      end if
      if (.not.nans_occured) then
      write(stdout,'("Success. Continue with optimization.")')
      write(stdout,'(" ")')
    end if
  end do
!  this%K_stat=5
end subroutine Cholesky_KM_myown_logdet
!!****

subroutine Cholesky_KM_myown(this, useOld)
!! SOURCE
  implicit none
  type(gpr_type),intent(inout)  ::  this
  logical, intent(in)           ::  useOld
  logical                       ::  nans_occured
  nans_occured = .false.
  if (useOld) then
    call Cholesky_myOwn(this%nk, this%KM_lin, &
                        this%KM_lin_chol,nans_occured,this%nk_old)
  else
    call Cholesky_myOwn(this%nk, this%KM_lin, &
                        this%KM_lin_chol,nans_occured)
  end if
  do while (nans_occured)
    if (printl>=2) &
      write(stdout,'(" ")')
      write(stdout,'(" ")')
      write(stdout,'("****************************************************")')
      write(stdout,'("Warning: The Cholesky decomposition became instable!")')
      write(stdout,'("The noise parameters for energies and gradients ",&
            "have to be increased.")')
      write(stdout,'("(The optimization might not be converging)")')
      call GPR_changeparameters(this, this%gamma, this%s_f, &
                              1.5d0*this%s_n)
      write(stdout, '(A, E10.3, E10.3, E10.3)') "New noise parameters are:",&
                                         this%s_n
      if (this%s_n(2)>this%tolerance*this%tol_noise_factor) &
        call dlf_fail("Noise parameters are too large... Optimization failed!")
      write(stdout,'("Recalculating covariance matrix...")')
      call calc_cov_mat_newOrder(this)
      write(stdout,'("Trying to solve new covariance matrix.")')
      if (useOld) then
        call Cholesky_myOwn(this%nk, this%KM_lin, &
                          this%KM_lin_chol,nans_occured,this%nk_old)
      else
        call Cholesky_myOwn(this%nk, this%KM_lin, &
                          this%KM_lin_chol,nans_occured)
      end if
      if (.not.nans_occured) then
      write(stdout,'("Success. Continue with optimization.")')
      write(stdout,'(" ")')
    end if
  end do
  this%K_stat=5
end subroutine Cholesky_KM_myown



! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/Cholesky_myOwn
!!
!! FUNCTION
!! Computes the cholesky decomposition of A (linear) and saves it to 
!! G (also linear)
!!
!! SYNOPSIS
subroutine Cholesky_myOwn(n, A, G, nans_occured, lastDoneRow)
!! SOURCE
  implicit none
  integer, intent(in)       :: n   ! dimension of A(n x n)
  real(rk), intent(in)      :: A(((n+1)*n)/2) ! linearized Version of A
  real(rk), intent(out)     :: G(((n+1)*n)/2) ! linearized Version of G
  logical, intent(out)      :: nans_occured
  integer, intent(in), optional :: lastDoneRow
  integer                   :: i,j,os,k, start
  real(rk)                  :: ddot
  real(rk)                  :: arg
  nans_occured = .false.
  if(.not.present(lastDoneRow)) then
    start = 1
  else
    start = lastDoneRow + 1
  end if
  
  ! lower triangular matrix solution only (note that A and G are linearized)
  do i = start, n
    os = (i)*(i-1)/2
    do j = 1, i-1
      k = os+j
      ! entry k in linearized form is entry i, j in non-linearized form
      ! Element i,j corresponds to
      ! k = (i)*(i-1)/2 + j
      G(k) = ( A(k)- &
              ddot(j-1,G(os+1        : k-1            ),1,&
                  G(j*(j-1)/2+1 : j*(j-1)/2+j-1),1)) &
            /G(j*(j-1)/2+j)
      if (isnan(G(k))) then
        nans_occured = .true.
        return
      end if
    end do
    !! j == i
    k = os+i
    arg = A(k)-ddot(i-1,G(os+1:k-1),1,G(os+1:k-1),1)
    if (arg>0) then
      G(k) = dsqrt(A(k)-ddot(i-1,G(os+1:k-1),1,G(os+1:k-1),1)) 
      if (isnan(G(k))) then
        nans_occured = .true.
        return
      end if
    else
      nans_occured = .true.
      return
    end if
  end do
end subroutine Cholesky_myOwn
!!****

subroutine Cholesky_myOwn_logdet(n, A, G, nans_occured,logdet, lastDoneRow)
!! SOURCE
  implicit none
  integer, intent(in)       :: n   ! dimension of A(n x n)
  real(rk), intent(in)      :: A(((n+1)*n)/2) ! linearized Version of A
  real(rk), intent(out)     :: G(((n+1)*n)/2) ! linearized Version of G
  logical, intent(out)      :: nans_occured
  real(rk),intent(out)      :: logdet
  integer, intent(in), optional :: lastDoneRow
  integer                   :: i,j,os,k, start
  real(rk)                  :: ddot
  real(rk)                  :: arg
  nans_occured = .false.
  if(.not.present(lastDoneRow)) then
    start = 1
  else
    start = lastDoneRow + 1
  end if
  
  ! lower triangular matrix solution only (note that A and G are linearized)
  do i = start, n
    os = (i)*(i-1)/2
    do j = 1, i-1
      k = os+j
      ! entry k in linearized form is entry i, j in non-linearized form
      ! Element i,j corresponds to
      ! k = (i)*(i-1)/2 + j
      G(k) = ( A(k)- &
              ddot(j-1,G(os+1        : k-1            ),1,&
                  G(j*(j-1)/2+1 : j*(j-1)/2+j-1),1)) &
            /G(j*(j-1)/2+j)
      if (isnan(G(k))) then
        nans_occured = .true.
        return
      end if
    end do
    !! j == i
    k = os+i
    arg = A(k)-ddot(i-1,G(os+1:k-1),1,G(os+1:k-1),1)
    if (arg>0) then
      G(k) = dsqrt(A(k)-ddot(i-1,G(os+1:k-1),1,G(os+1:k-1),1)) 
      if (isnan(G(k))) then
        nans_occured = .true.
        return
      end if
    else
      nans_occured = .true.
      return
    end if
  end do
  logdet=0.0d0
  do i=start,n
    os=i*(i-1)/2
    k=os+i
    logdet = logdet+2.0d0*log(G(k))
  enddo
end subroutine Cholesky_myOwn_logdet

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/solve_LGS_use_chol_myOwn
!!
!! FUNCTION
!! Using the cholesky decomposition G and RHS b
!! x is RHS on entrance and solution in the end
!!
!! SYNOPSIS
subroutine solve_LGS_use_chol_myOwn(n, G, x)
!! SOURCE
  implicit none
  integer, intent(in)       :: n   ! dimension of A(n x n)
!   real(rk), intent(in)      :: A(((n+1)*n)/2) ! linearized Version of A
  real(rk), intent(in)      :: G(((n+1)*n)/2) ! linearized Version of G
  real(rk), intent(inout)   :: x(n)
  integer                   :: i,j,os
  real(rk)                  ::  sum_i
  ! lower triangular matrix solution only (note that A and B are linearized)
  ! Forward substitution with G
  
  do j = 1, n
    ! Element i,j corresponds to
    ! k = (i)*(i-1)/2 + j
    os = (j)*(j-1)/2
    x(j) = (x(j)-SUM(G(os + 1 : os + j - 1)*x(1:j-1)))/G(os+j)
  end do
  ! Backward substitution with G^T
  do j = n, 1, -1
    sum_i = 0d0
    do i = j+1, n
      ! Entry G^T(j,i)=G(i,j) has index k in linearized form
      ! k = (i-1)*i/2 + j
      sum_i = sum_i + G((i-1)*i/2 + j)*x(i)
    end do
    ! Entry G^T(j,j)=G(j,j)
    ! k = (j-1)*j/2 + j
    x(j) = (x(j)-sum_i)/G((j-1)*j/2 + j)
  end do
end subroutine solve_LGS_use_chol_myOwn
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/Cholesky_KM_logdet
!!
!! FUNCTION
!! Computes the cholesky decomposition of KM for further use in 
!! solve_LGS_use_KM_chol and computes the determinant
!!
!! SYNOPSIS
subroutine Cholesky_KM_logdet(this, logdet)    
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    integer                 ::  info
    real(rk), intent(out)    ::  logdet
    integer                 ::  i
    if (this%K_stat==0) then
        call DPOTRF ('U',this%nk, this%KM, this%nk, info)
        if (info/=0) call dlf_fail("Cholesky Decomposition failed (Cholesky_KM_logdet)")
        this%K_stat = 3
    else if (this%K_stat==3) then
        call dlf_fail("Covariance matrix is cholesky decomposed already!")
    else
        call dlf_fail("Covariance matrix has no valid form (or is inverted)!")
    end if
    logdet =0d0
    do i = 1, this%nk
        logdet = logdet + 2d0*LOG(this%KM(i,i))
    end do
end subroutine Cholesky_KM_logdet
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/solve_LGS_use_KM_chol
!!
!! FUNCTION
!! Solves a linear system with cholesky decomposed covariance matrix
!! of the form KM * x = b
!! The solution for x is then again written in b
!!
!! SYNOPSIS
subroutine solve_LGS_use_KM_chol(this, b)   
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    real(rk), intent(inout)  ::  b(this%nk)
    integer                 ::  info
    if (this%K_stat==3) then
        call DPOTRS ('U', this%nk, 1, this%KM, this%nk, b, this%nk, info)
    else
        call dlf_fail("Do cholesky decomposition first.")
    end if
end subroutine solve_LGS_use_KM_chol
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/invert_KM_chol
!!
!! FUNCTION
!! Invert a symmetric pos.def. matrix via Cholesky decomposition
!!
!! SYNOPSIS
subroutine invert_KM_chol(this, det,inv_KM) 
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this 
    real(rk), intent(out)    ::  det ! determinant
    real(rk), intent(out),optional   ::  inv_KM(this%nk,this%nk)
    integer                 ::  info, i
    if (this%K_stat==0) then
        call DPOTRF ('U',this%nk, this%KM, this%nk, info)
        this%K_stat = 3
    end if
    if (this%K_stat==3) then
        det = 1d0
        do i = 1, this%nk
            det = det * this%KM(i,i)**2
        end do
        if (present(inv_KM)) then
          inv_KM(:,:) = this%KM(:,:)
          call DPOTRI ('U', this%nk, inv_KM, this%nk, info)  
        else
          call DPOTRI ('U', this%nk, this%KM, this%nk, info) 
        endif
        if (info.ne.0) then
            call dlf_fail("Inversion of matrix failed!")
        else
        if (.not. present(inv_KM)) then
            this%K_stat=4
        endif
        end if        
    else
        call dlf_fail("Matrix not in a valid state for Inversion by Cholesky!")
    endif
end subroutine invert_KM_chol
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/solve_LGS_use_KM_chol_inv
!!
!! FUNCTION
!! Uses an inverted Covariance matrix (inverted by Cholesky) to solve a 
!! linear system KM * x = b
!!
!! SYNOPSIS
subroutine solve_LGS_use_KM_chol_inv(this, b, x) 
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this 
    real(rk), intent(in)     ::  b(this%nk)
    real(rk), intent(out)    ::  x(this%nk)
    integer                 ::  i,j
    if (this%K_stat==4) then
        x(:)=0d0
        do j = 1, this%nk
            do i = j, this%nk
                x(j) = x(j) + this%KM(j,i)*b(i)
            end do
        end do
        do j = 2, this%nk
            do i = 1, j-1
                x(j) = x(j) + this%KM(i,j)*b(i)
            end do
        end do
    else
        call dlf_fail("Do Decomposition by Cholesky first!")
    endif
end subroutine solve_LGS_use_KM_chol_inv   
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/exf
!!
!! FUNCTION
!! 1D Example function
!!
!! SYNOPSIS
real(rk) function exf(x)    
!! SOURCE
    implicit none
    real(rk), intent(in) ::  x
    exf = SIN(X)
    exf = x**2*SIN(x)
!     exf = 4d0*1d0*((1d0/x)**(12)-(1d0/x)**(6))
end function exf
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/dexf
!!
!! FUNCTION
!! 1D Example functino (1st derivative)
!!
!! SYNOPSIS
real(rk) function dexf(x)    
!! SOURCE
    implicit none
    real(rk), intent(in) ::  x
    dexf = COS(X)
    dexf = 2*x*SIN(x)+x**2*COS(x)
!     dexf = 4d0*1d0*(-12d0*(1d0/x)**(13)+6d0*(1d0/x)**(7))
end function dexf
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/ddexf
!!
!! FUNCTION
!! 1D Example function (2nd derivative)
!!
!! SYNOPSIS
real(rk) function ddexf(x)    
!! SOURCE
    implicit none
    real(rk), intent(in) ::  x
    ddexf = -SIN(X)
    ddexf = 2*SIN(x)+4*x*COS(x)-x**2*SIN(x)
!     ddexf = 4d0*1d0*(12d0*13d0*(1d0/x)**(14)-6d0*7d0*(1d0/x)**(8))
end function ddexf
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/exf2d
!!
!! FUNCTION
!! 2D Example function
!!
!! SYNOPSIS
subroutine exf2d( x, res)    
!! SOURCE
    implicit none
    real(rk), intent(in) ::  x(2)
    real(rk)             ::  res
    !exf = SIN(X)
    res = x(1)**2*SIN(x(1))+x(2)**2*COS(x(2))+x(1)*x(2)
end subroutine exf2d
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/dexf2d
!!
!! FUNCTION
!! 2D Example function (1st derivative)
!!
!! SYNOPSIS
subroutine dexf2d( x, res)   
!! SOURCE
    implicit none
    real(rk), intent(in) ::  x(2)
    real(rk)             ::  res(2)
    !dexf = COS(X)
    res(1) = 2*x(1)*SIN(x(1))+x(1)**2*COS(x(1))+x(2)
    res(2) = x(2)*(2d0*COS(x(2))-x(2)*(sin(x(2))))+x(1)
end subroutine dexf2d
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/ddexf2d
!!
!! FUNCTION
!! 2D Example function (2nd derivative)
!!
!! SYNOPSIS
subroutine ddexf2d( x, res) 
!! SOURCE
    implicit none
    real(rk), intent(in) ::  x(2)
    real(rk)             ::  res(2,2)
    !ddexf = -SIN(X)
    res(1,1) = 2d0*SIN(x(1))+4d0*x(1)*COS(x(1))-x(1)**2*SIN(x(1))
    res(2,2) = -(x(2)**2-2d0)*COS(x(2))-4d0*x(2)*SIN(x(2))
    res(1,2) = 1d0
    res(2,1) = 1d0
end subroutine ddexf2d
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_variance
!!
!! FUNCTION
!! This subroutine calculates the variance of a random variable in the GP
!! associated with the position "pos".
!! This can be a measure for the uncertainty in the energy prediction.
!!
!! SYNOPSIS
subroutine GPR_variance(this, pos,var)  
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    real(rk), intent(in)     ::  pos(this%sdgf)
    real(rk), intent(out)    ::  var
    real(rk)                 ::  diff(this%sdgf), absv
    ! a and c must be locally declared here, so that
    ! multiple variance calculations can run in parallel
    real(rk)                 ::  a(this%nk), c(this%nk),y(this%nk),w(this%nk)
    real(rk)                 ::  ddot,dk_dx,s_2
    integer                 ::  i, j, l, os, os2,sj
    !check for correct this%nk should be implemented here
    if (this%ext_data.and.(.not.this%providingKM)) then
        call dlf_fail(&
        "No covariance matrix in data file given. Needed for variance!")
    end if
    var = 0d0        
    a(:) = 0d0
    do i =1, this%nt
        a(i) = kernel(this,pos,this%xs(:,i))
    end do
    if (this%order==42) then
!       call dlf_fail("Not implemented (GPR_variance order 42)!")
      os = this%nt
      do i = 1, this%nt
        if (this%order_tps(i)>0) then
          diff = pos-this%xs(:,i)
          absv = norm2(diff)
          do j = 1, this%idgf
            a(os+j)= kernel_d1_exp2(this,diff,j,absv)
          end do
          os = os + this%idgf
        end if
      end do
      do i = 1, this%nt        
        if (this%order_tps(i)>1) then
          diff = pos-this%xs(:,i)
          absv = norm2(diff)
          os2 = -this%idgf
          do l = 1, this%idgf
            os2 = os2 + this%idgf-(l-1)
            do j = l, this%idgf
              a(os+os2+j) = &
                kernel_d2_exp22(this,diff,j,l,absv) 
            end do
          end do
          os = os + this%sdgf*(this%sdgf+1)/2
        end if
      end do    
    else if (this%order>0) then
      os = this%nt
      do i = 1, this%nt
        diff = pos-this%xs(:,i)
        absv = norm2(diff)
        do j = 1, this%idgf
          if(this%internal == 2) then
            dk_dx = 0.0d0
            do sj=1,this%sdgf
              dk_dx = dk_dx + kernel_d1_exp2(this,diff,sj,absv) * this%b_matrix(sj,j,i)
            enddo
            a(os+j)= dk_dx
          else
            a(os+j)= kernel_d1_exp2(this,diff,j,absv)
          endif
        end do
        os = os + this%idgf
      end do
      if (this%order>1) then
        os = this%nt+this%sdgf*this%nt
        do i = 1, this%nt
          diff = pos-this%xs(:,i)
          absv = norm2(diff)
          os2 = -this%sdgf
          do l = 1, this%sdgf
            os2 = os2 + this%sdgf-(l-1)
            do j = l, this%sdgf
              a(os+os2+j) = &
                kernel_d2_exp22(this,diff,j,l,absv) 
            end do
          end do
          os = os + this%sdgf*(this%sdgf+1)/2
        end do
      end if
    end if
  if(.not.this%iChol) then  
    if (this%K_stat==1.or.this%K_stat==2) then 
      call dlf_fail("LU decomposition not possible anymore in GPR.")
    else if (this%K_stat==0) then    
      select case(this%solveMethod)
      ! case 1+2 deleted (LU)
      case(3)
        if (this%iChol) &
            call dlf_fail("This should not happen (GPR_variance)!")
        call Cholesky_KM(this)
        c(:)=a(:)
        call solve_LGS_use_KM_chol(this, c)
      case(4)
        call dlf_fail("Variance calculation not implemented with Cholesky Inverse calculation!")
      case default
        call dlf_fail("Default Case reached in GPR_variance!")
      end select
    else if (this%K_stat==3) then
      c(:) = a(:)
      call solve_LGS_use_KM_chol(this,c)
    else
      write(stderr,'("covariance state: ", I8)') this%K_stat
      call dlf_fail("The matrix state is not implemented for variance calculation.")
    end if
  else
    ! my own cholesky decomposition -> different order of a & c necessary
    call orderW(this,a,c)
    a(:) = c(:)
    c(:)=a(:)
    ! a contains correctly ordered entries
    if (this%K_stat==-1) then
      call calc_cov_mat_newOrder(this)
      call Cholesky_KM_myown(this, .false.)
      this%K_stat=5
    else if (this%K_stat==6) then
      ! KM not up-to-date
      call calc_cov_mat_newOrder(this,.true.)
      this%K_stat=7
    end if
    if (this%K_stat==7) then
      ! KM not solved completely
      call Cholesky_KM_myown(this,.true.)
      this%K_stat=5
    end if   
    if (this%K_stat == 5) then
      ! KM is decomposed with my own chol decomp. -> solve the linear system
      call solve_LGS_use_chol_myOwn(this%nk, this%KM_lin_chol, c)
    else
      call dlf_fail("KM_lin was not in a valid state. (GPR_variance)")
    end if    
  end if
  var = kernel(this,pos,pos)
  var = var - DDOT(this%nk,a,1,c,1)
  
end subroutine GPR_variance
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/train_and_test_e
!!
!! FUNCTION
!! Evaluates the result for training and test points: Energies
!!
!! SYNOPSIS
subroutine train_and_test_e(this, ntest, xtest, &
    es_test, rmsd, mae)    
!! SOURCE
    use dlf_global
    use oop_hdlc 
    implicit none
    type(gpr_type),intent(inout)::this
    integer, intent(in) ::  ntest ! number of training points
    real(rk), intent(in) ::  xtest(this%idgf,ntest),&
                            es_test(ntest)
    real(rk)             :: esi_test(ntest)                            
    real(rk)             :: esi_train(this%nt)                            
    real(rk), intent(out)::  mae, & ! Mean absolut error
                            rmsd    ! Root mean square deviation
    integer             ::  i
    real(rk)            :: tmp_test(this%sdgf), xgradient(this%idgf)
    real(rk)            ::  igradient(this%sdgf)   
    type(hdlc_ctrl)                 :: gpr_hdlc
    print*,"T! ********************TESTING Train ENERGIES************************"
    do i = 1, this%nt
      if(this%internal == 2) then
            gpr_hdlc = hdlc_ctrl()
            xgradient(:) = 0.0d0
            igradient(:) = 0.0d0
            call gpr_hdlc%dlf_hdlc_init(glob%nat,glob%spec,mod(glob%icoord,10),glob%ncons, &
                 glob%icons,glob%nconn,glob%iconn)
            call gpr_hdlc%dlf_hdlc_create(glob%nat,glob%nicore,glob%spec,glob%micspec, &
                glob%znuc,1,this%ixs(:,1),glob%weight,glob%mass)              
           ! call gpr_hdlc%dlf_hdlc_xtoi(glob%nvar/3,glob%nivar,glob%nicore,glob%micspec, &
            !xtest(:,i),xgradient,tmp_test,igradient)    
            call GPR_eval(this, &
                        this%xs(:,i), esi_train(i))       
      else
            call GPR_eval(this, &
                        this%xs(:,i), esi_train(i))
    endif
    end do
    rmsd = sqrt(sum((this%es(1:this%nt)-esi_train(1:this%nt))**2)/dble(this%nt))
    mae = maxval(abs((this%es(1:this%nt)-esi_train(1:this%nt))))
    print*,"T! RMSD , GPR of order",this%order, rmsd
    print*,"T! MAE  , GPR of order",this%order, mae 

    print*,"**********************END OF TESTING ENERGIES*********************"
    
    print*,"T! ***********************TESTING ENERGIES************************"   
    do i = 1, ntest
        if(this%internal == 2) then
             call gpr_hdlc%dlf_hdlc_xtoi(glob%nvar/3,glob%nivar,glob%nicore,glob%micspec, &
            xtest(:,i),xgradient,tmp_test,igradient)    
            call GPR_eval(this, &
                        tmp_test, esi_test(i))            
        else
            call GPR_eval(this, &
                        xtest(:,i), esi_test(i))
       endif
    end do
    rmsd = sqrt(sum((es_test(1:ntest)-esi_test(1:ntest))**2)/dble(ntest))
    mae = maxval(abs((es_test-esi_test(1:ntest))))
    print*,"T! RMSD , GPR of order",this%order, rmsd
    print*,"T! MAE  , GPR of order",this%order, mae    
    print*,"P! RMSD test ",this%order, this%gamma, rmsd

    3334  FORMAT (A, I1,ES12.4,ES12.4,ES12.4,ES12.4,ES12.4, ES12.4,ES12.4,ES12.4) 

    print*,"**********************END OF TESTING ENERGIES*********************"
    if(this%internal == 2) call gpr_hdlc%dlf_hdlc_destroy()
end subroutine train_and_test_e
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/train_and_test_g
!!
!! FUNCTION
!! Evaluates the result for training and test points: Gradients
!!
!! SYNOPSIS
subroutine train_and_test_g(this, ntest, xtest, &
    gs_test, rmsd, mae)    
!! SOURCE    
    use dlf_global
    use oop_hdlc 
    implicit none
    type(gpr_type),intent(inout)::this
    integer, intent(in) ::  ntest ! number of training points
    real(rk), intent(in) ::  xtest(this%idgf,ntest),&
                            gs_test(this%idgf,ntest)
    real(rk)             :: gsi_test(this%idgf,ntest)
    real(rk), intent(out)::  mae, & ! Mean absolut error
                            rmsd !Root mean square deviation
    integer             ::  i
    real(rk)            :: tmp_test(this%sdgf), xgradient(this%idgf)
    real(rk)            ::  igradient(this%sdgf)   
    type(hdlc_ctrl)                 :: gpr_hdlc
    real(rk)                        :: tmp_bmat(this%sdgf,this%idgf) 
    print*,"T! **********************TESTING Gradients***********************" 
    if(this%internal == 2) then
       gpr_hdlc = hdlc_ctrl()
      call gpr_hdlc%dlf_hdlc_init(glob%nat,glob%spec,mod(glob%icoord,10),glob%ncons, &
                 glob%icons,glob%nconn,glob%iconn)
      call gpr_hdlc%dlf_hdlc_create(glob%nat,glob%nicore,glob%spec,glob%micspec, &
                glob%znuc,1,this%ixs(:,1),glob%weight,glob%mass) 
    endif         
    do i = 1, ntest
      if (this%internal == 2) then
           
            xgradient(:) = 0.0d0
            igradient(:) = 0.0d0
        
            call gpr_hdlc%dlf_hdlc_xtoi(glob%nvar/3,glob%nivar,glob%nicore,glob%micspec, &
            xtest(:,i),xgradient,tmp_test,igradient)    
            tmp_bmat(:,:) = gpr_hdlc%bhdlc_matrix(:,:)        
            call GPR_eval_grad(this, &
                        tmp_test, gsi_test(:,i),tmp_bmat) 
      else    
            call GPR_eval_grad(this, &
                        xtest(:,i), gsi_test(:,i))
      endif
    end do
    ! Output the data
    rmsd = 0d0
    mae = 0d0
    do i = 1, ntest
        rmsd = rmsd +   dot_product(gs_test(:,i)-gsi_test(:,i),&
                                    gs_test(:,i)-gsi_test(:,i))
        mae = MAX(mae, maxval(abs(gs_test(:,i)-gsi_test(:,i))))
    end do
    rmsd = dsqrt(rmsd/dble(ntest*this%idgf))
    print*,"T! RMSD , GPR of order",this%order, rmsd
    print*,"T! MAE  , GPR of order",this%order, mae

    print*, "*******************End of TESTING Gradients*******************"
    if(this%internal == 2) call gpr_hdlc%dlf_hdlc_destroy()
end subroutine train_and_test_g
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/train_and_test_h
!!
!! FUNCTION
!! Evaluates the result for training and test points: Hessians
!!
!! SYNOPSIS
subroutine train_and_test_h(this, ntest, xtest, &
    hs_test, rmsd, mae)    
!! SOURCE    
    implicit none
    type(gpr_type),intent(inout)::this
    integer, intent(in) ::  ntest! number of training points
    real(rk), intent(in) ::  xtest(this%sdgf,ntest),&
                            hs_test(this%sdgf,this%sdgf,ntest)
    real(rk)             ::  hsi_test(this%sdgf,this%sdgf,ntest)
    real(rk), intent(out)::  mae, & ! Mean absolut error
                            rmsd !Root mean square deviation
    integer             ::  i, j
    print*,"T! **********************TESTING Hessians***********************"   
    do i = 1, ntest
            call GPR_eval_hess(this, xtest(:,i), hsi_test(:,:,i))
    end do
    ! Output the data
    rmsd = 0d0
    mae = 0d0
    do i = 1, ntest
    do j = 1, this%sdgf
        rmsd = rmsd +   dot_product(hs_test(:,j,i)-hsi_test(:,j,i),&
                                    hs_test(:,j,i)-hsi_test(:,j,i))
        mae = MAX(mae, maxval(abs(hs_test(:,j,i)-hsi_test(:,j,i))))
    end do
    end do
    rmsd = dsqrt(rmsd/dble(ntest*this%sdgf*(this%sdgf+1)/2))
    print*,"T! RMSD , GPR of order",this%order, rmsd
    print*,"T! MAE  , GPR of order",this%order, mae

    print*, "*******************End of TESTING Hessians*******************"
end subroutine train_and_test_h
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_testEnergies
!!
!! FUNCTION
!! Evaluates the result for test points: Energies
!!
!! SYNOPSIS
subroutine GPR_testEnergies(this, ntest, xtest, &
    es_test, rmsd, mae)    
!! SOURCE    
    implicit none
    type(gpr_type),intent(inout)::this
    integer, intent(in) ::  ntest ! number of training points
    real(rk), intent(in) ::  xtest(this%sdgf,ntest),&
                            es_test(ntest)
    real(rk)             ::  esi_training(this%nt),&
                            esi_test(ntest)                            
    real(rk), intent(out)::  mae, & ! Mean absolut error
                            rmsd !Root mean square deviation
    integer             ::  i
    print*,"T! ***********************TESTING ENERGIES*************************"
    print*,"T! Testing the accuracy of the interpolation at the test points"    
    do i = 1, ntest
            call GPR_eval(this,xtest(:,i), esi_test(i))
    end do
    ! Output the data
    rmsd = sqrt(sum((es_test(1:ntest)-esi_test(1:ntest))**2)/dble(ntest))
    mae = maxval(abs((es_test-esi_test(1:ntest))))
    print*,"T! RMSD , GPR of order",this%order, rmsd
    print*,"T! MAE  , GPR of order",this%order, mae
    
    print*,"P! RMSD test ",this%order, this%s_n(3), rmsd
    write(*, 3334)"TPlot  ", this%order, this%s_n(1),this%s_n(2), this%s_n(3),&
        this%gamma, &
        sqrt(sum((this%es(1:this%nt)-esi_training(1:this%nt))**2)/dble(this%nt)),&
        maxval(abs((this%es(1:this%nt)-esi_training(1:this%nt)))), &
        sqrt(sum((es_test-esi_test(1:ntest))**2)/dble(ntest)), &
        maxval(abs((es_test-esi_test(1:ntest))))

    3334  FORMAT (A, I1,ES12.4,ES12.4,ES12.4,ES12.4,ES12.4, ES12.4,ES12.4,ES12.4) 

    print*,"**********************END OF TESTING ENERGIES*********************"
end subroutine GPR_testEnergies
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_testGrad
!!
!! FUNCTION
!! Evaluates the result for test points: Gradients
!!
!! SYNOPSIS
subroutine GPR_testGrad(this, ntest, xtest, &
    gs_test, rmsd, mae)    
!! SOURCE    
    implicit none
    type(gpr_type),intent(inout)::this
    integer, intent(in) ::  ntest ! number of training points
    real(rk), intent(in) ::  xtest(this%sdgf,ntest),&
                            gs_test(this%sdgf,ntest)
    real(rk)             ::  gsi_test(this%sdgf,ntest)
    real(rk), intent(out)::  mae, & ! Mean absolut error
                            rmsd !Root mean square deviation
    integer             ::  i
    print*, "T! **********************TESTING Gradients***********************"
    print*, "T! Testing the accuracy of the interpolation at the test points"    
    do i = 1, ntest
            call GPR_eval_grad(this,xtest(:,i), gsi_test(:,i))
    end do
    ! Output the data
    rmsd = 0d0
    mae = 0d0
    do i = 1, ntest
        rmsd = rmsd +   dot_product(gs_test(:,i)-gsi_test(:,i),&
                                    gs_test(:,i)-gsi_test(:,i))
        mae = MAX(mae, maxval(abs(gs_test(:,i)-gsi_test(:,i))))
    end do
    rmsd = dsqrt(rmsd/dble(ntest*this%sdgf))
    print*,"T! RMSD , GPR of order",this%order, rmsd
    print*,"T! MAE  , GPR of order",this%order, mae

    print*, "*******************End of TESTING Gradients*******************"
end subroutine GPR_testGrad
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_testHess
!!
!! FUNCTION
!! Evaluates the result for test points: Hessians
!!
!! SYNOPSIS
subroutine GPR_testHess(this, ntest, xtest,hs_test, rmsd, mae)    
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    integer, intent(in) ::  ntest! number of training points
    real(rk), intent(in) ::  xtest(this%sdgf,ntest),&
                             hs_test(this%sdgf,this%sdgf,ntest)
    real(rk)             ::  hsi_test(this%sdgf,this%sdgf,ntest)
    real(rk), intent(out)::  mae, & ! Mean absolut error
                            rmsd !Root mean square deviation
    integer             ::  i, j
    print*, "T! **********************TESTING Hessians***********************"
    print*, "T! Testing the accuracy of the interpolation at the test points"    
    do i = 1, ntest
            call GPR_eval_hess(this, xtest(:,i), hsi_test(:,:,i))
    end do
    ! Output the data
    rmsd = 0d0
    mae = 0d0
    do i = 1, ntest
    do j = 1, this%sdgf
        rmsd = rmsd +   dot_product(hs_test(:,j,i)-hsi_test(:,j,i),&
                                    hs_test(:,j,i)-hsi_test(:,j,i))
        mae = MAX(mae, maxval(abs(hs_test(:,j,i)-hsi_test(:,j,i))))
    end do
    end do
    rmsd = dsqrt(rmsd/dble(ntest*this%sdgf*(this%sdgf+1)/2))
    print*,"T! RMSD , GPR of order",this%order, rmsd
    print*,"T! MAE  , GPR of order",this%order, mae

    print*, "*******************End of TESTING Hessians*******************"
end subroutine GPR_testHess
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_write
!!
!! FUNCTION
!! Writes all information that is necessary for interpolation
!! in a file (KM matrix for variance for example is not written)
!!
!! SYNOPSIS
subroutine GPR_write(this, providingKM, name)    
!! SOURCE
    implicit none
    logical, intent(in)         :: providingKM 
    type(gpr_type),intent(inout):: this
    character(*), intent(in)    :: name
    if (printl>=2) &
        write(stdout,'("T! Writing ouput file with name ", A)') trim(name)
    open(unit = 40, file = name, status = "replace", &
           action='write', &
           position='append', form='unformatted')
    if (this%K_stat<0) call dlf_fail("The covariance matrix is not valid!")
    if (.not.this%w_is_solution) call dlf_fail("Weight vector is not a solution.")
    write(40) "PESType#", INT(1, 4) ! 1 tells dl-find that this is a GPR file.
    write(40) INT(this%nt, 4), INT(this%nat, 4), &
               INT(this%sdgf, 4), INT(this%kernel_type,4),& 
               INT(this%order, 4), INT(this%K_stat, 4), this%mean,&
               this%gamma, this%s_f, this%s_n,&
               providingKM, this%provideTrafoInFile, &
               this%massweight, this%OffsetType, this%manualOffset, &
               this%iChol, this%addToMax! constants
    if (providingKM) then
        if (.not.this%iChol) then
          write(40) this%w(1:this%nk), this%xs(1:this%sdgf,1:this%nt), this%KM ! non-constants
        else
          write(40) this%w(1:this%nk), this%xs(1:this%sdgf,1:this%nt), this%KM_lin, this%KM_lin_chol ! non-constants
        end if
    else
        write(40) this%w(1:this%nk), this%xs(1:this%sdgf,1:this%nt) ! non-constants
    end if
    if (this%provideTrafoInFile) then
        write(40) this%align_refcoords,this%align_modes,this%refmass
    end if
    if (this%OffsetType==8) then
      write(40) this%taylor%ptOfExpansion
      write(40) this%taylor%energy
      write(40) this%taylor%grad
      write(40) this%taylor%hess
    end if
    ! these parameters are added later on, and therefore, stand at the 
    ! end of the file to give the chance of compatibility
    close(40)
end subroutine GPR_write    
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_read
!!
!! FUNCTION
!! Reads all information that is necessary for interpolation
!! from a file (KM matrix for variance for example is not read)  
!!
!! SYNOPSIS
subroutine GPR_read(this, name, stat)  
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this 
    integer                     :: IOstatus
    integer(4)                  :: nt, nat, sdgf, kernel_type, order, K_stat
    integer,optional            :: stat
    character(*), intent(in)    :: name
!     character(100)              :: nameCode
  character(8)                :: pes_type_marker ! string that shows that a PES
                                                 ! type is given 
                                                 ! (if not chose 0 -> nn)
    integer(4) :: pes_type4 
    logical                     :: file_opened,file_exists
    stat=0 ! must be 0 at the end if everything is fine
    this%ext_data = .true.
    INQUIRE(40,OPENED=file_opened) 
    if(.not.file_opened) then
        INQUIRE(file=name,EXIST=file_exists)
        if (.not.file_exists) call dlf_fail("PES file does not exist!")
        open(unit = 40, file = name, status = "old", &
            action='read', &
            form='unformatted') 
        read(40,IOSTAT=IOstatus) pes_type_marker, pes_type4
    end if
    read(40,IOSTAT=IOstatus) nt, nat,sdgf, kernel_type, order,K_stat,this%mean,&
        this%gamma, this%s_f, this%s_n,&
        this%providingKM, this%provideTrafoInFile,&
        this%massweight, this%OffsetType, this%manualOffset, &
        this%iChol, this%addToMax! constants  
    this%s_f2=this%s_f**2    
    this%l=1d0/dsqrt(this%gamma)
    this%l4 = this%l**4
    this%l2 = this%l**2
    if (IOstatus .gt. 0) then 
      call dlf_fail("An error occured when reading the .dat file")
    else if (IOstatus .lt. 0) then
      call dlf_fail("End of file reached earlier than expected.")
    else
      !successfully read
    end if
    if(this%constructed) call GPR_destroy(this)
    ! the value of OffsetType will be changed later on depending on
    ! the values in the file (historic reasons)
    call GPR_construct(this, INT(nt), INT(nat), INT(sdgf), 1, INT(kernel_type),INT(order))  
    if (this%providingKM) then
      if (.not.this%iChol) then
        read(40, IOSTAT=IOstatus) this%w(1:this%nk), this%xs(1:this%sdgf,1:this%nt), this%KM ! non-constants
      else
        read(40, IOSTAT=IOstatus) this%w(1:this%nk), this%xs(1:this%sdgf,1:this%nt), this%KM_lin, this%KM_lin_chol
        this%K_stat = INT(K_stat)
      end if
    else
        read(40, IOSTAT=IOstatus) this%w(1:this%nk), this%xs(1:this%sdgf,1:this%nt) ! non-constants
        this%K_stat = -1
    end if
    this%w_is_solution=.true.
    if (this%provideTrafoInFile) then
      call constructTrafoForDL_Find(this, this%nat, this%sdgf)
      read(40) this%align_refcoords,this%align_modes,this%refmass
    end if
    if (this%OffsetType==8) then
      call GPR_taylorBias_construct(this)
      read(40) this%taylor%ptOfExpansion
      read(40) this%taylor%energy
      read(40) this%taylor%grad
      read(40) this%taylor%hess
    end if
    if (IOstatus .gt. 0) then 
      call dlf_fail("An error occured when reading the .dat file")
    else if (IOstatus .lt. 0) then
      call dlf_fail("End of file reached earlier than expected.")
    else
      !successfully read
      write(*, '(A,ES12.2)') " Sigma scaling   :", this%s_f
      write(*, '(A,ES12.2)') " Sigma Energy    :", this%s_n(1)
      write(*, '(A,ES12.2)') " Sigma Gradient  :", this%s_n(2)
      write(*, '(A,ES12.2)') " Sigma Hessian   :", this%s_n(3)
      write(*, '(A,ES12.2)') " Gamma           :", this%gamma
      
      if (this%kernel_type==0) then
         write(*, '(A)')     " Kernel Type     : Squared exp"
      else if (this%kernel_type==1) then
         write(*, '(A)')     " Kernel Type     :  Matern 5/2"
      else
         write(*, '(A)')     " Unknown kernel type!"
        call dlf_fail("Unkown type of kernel!")
      end if
         write(*,'(A,I12.1)')" Spatial dgf     :", this%sdgf
        
         write(*,'(A,I12.1)')" Number of atoms :", this%nat
         write(*,'(A)')" Number of input       "
         write(*,'(A,I12.1)')" geometries      :", this%nt
      if (this%providingKM) then
        if (printl>=6) then
          write(stdout,'("Covariance matrix is given in state ", I4)') &
                        this%K_stat
          write(stdout,'("(-1 is invalid, 0 is normal matrix, ",&
                    "3 Cholesky, 4 Inverse by Cholesky")')
        end if
      else
        write(stdout,'(" ")')
        write(stdout,'("No covariance matrix given.",& 
                "Variance calculation not available.")')
      end if
        write(stdout,'(" ")')
    end if

    if (IOStatus>0) then
        call dlf_fail("Something went wrong in GPR_read")
    else if (IOStatus<0) then
        ! End of file reached and no OffsetType was given.
        ! Set to default:
        this%OffsetType = 1
    end if
    close(40)
end subroutine GPR_read      
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_write_Hess_eigvals
!!
!! FUNCTION
!! Writes out the GPR-inferred Hessian's eigenvalues
!! and the respective vibrational frequencies
!!
!! SYNOPSIS
subroutine GPR_write_Hess_eigvals(this,pos)
!! SOURCE
  type(gpr_type), intent(inout) ::  this
  real(rk), intent(in)          ::  pos(this%sdgf)
  real(rk)                      ::  hess(this%sdgf, this%sdgf)
  real(rk)                      ::  evs(this%sdgf)
  real(rk), allocatable         ::  work(:)
  integer                       ::  lwork,nb,info,i
  integer, external             ::  ilaenv
  ! evaluate the hessian
  nb = ilaenv(1,'DSYEV','NU',this%sdgf,-1,-1,-1)
  if (nb<0) then
        call dlf_fail("ILAENV for dsytrd failed (GPR_write_Hess_eigvals)")
  end if
  lwork = this%sdgf*(nb+2)
  allocate(work(lwork))
  call GPR_eval_hess(this,pos,hess)
  call DSYEV('N','U',this%sdgf,hess,this%sdgf,evs,work,lwork,info)
  if (info/=0) call dlf_fail("DSYEV failed (GPR_write_Hess_eigvals)")
  do i = 1, this%sdgf
    write(stdout,'("eigval ", I9, " is ", I9)') evs(i), i, "(",SIGN(dsqrt(abs(evs(i))),evs(i)),")"
  end do
end subroutine GPR_write_Hess_eigvals    
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/check_KM
!!
!! FUNCTION
!! Checks the covariance matrix for symmetry and (semi-)definiteness
!! possemdef is determined via the symmetric part of KM
!! ATTENTION! It changes the covariance matrix!
!!
!! SYNOPSIS
subroutine check_KM(this, sym, possemdef, posdef)
!! SOURCE
    implicit none  
    type(gpr_type),intent(inout)::  this
    logical, intent(out)        ::  possemdef
    logical, intent(out)        ::  posdef
    logical, intent(out)        ::  sym
    integer     ::  i, j, info
    real(rk)     ::  eigvR(this%nk), eigvI(this%nk)
    real(rk)     ::  dummy(1,this%nk), dummy2(1,this%nk)
    real(rk)     ::  work(12*this%nk), smallestEigV, largestEigV,cond_nr
    ! Check for symmetry
    if (this%iChol) call dlf_fail("check_KM not inplemented for iterative cholesky!")
    sym = .true.
    do i = 1, this%nk
        do j = 1, this%nk
            if (ABS(this%KM(i,j)-this%KM(j,i))>1d-14) then
                sym=.false.                
            end if
            
        end do
    end do
    ! check for pos-semi-def.
    do i = 1, this%nk
        do j = 1, i
            this%KM(i,j) = this%KM(i,j) + this%KM(j,i)
            !symKM(j,i)=this%KM(i,j)+this%KM(j,i)
        end do
    end do
    do i = 1, this%nk
        do j = 1, i
            this%KM(j,i) = this%KM(i,j)
        end do
    end do
    this%KM(:,:) = this%KM(:,:)/2d0
    call DGEEV('N','N',this%nk,this%KM, this%nk, eigvR, eigvI,&
                dummy,1,dummy2,1,work,12*this%nk,info)
    if(.not.(info==0)) call dlf_fail("ERROR in DGEEV.")
    possemdef = .true.
    posdef = .true.
    smallestEigV = eigvR(1)
    largestEigV = eigvR(1)
    do i = 1, this%nk
        if (eigvR(i)> largestEigV) largestEigV = eigvR(i)
        if (eigvR(i)< smallestEigV) smallestEigV = eigvR(i)
        !if (eigvI(i)/=0d0) call dlf_fail("imaginary eigenvalue detected.")
        if (eigvR(i)<0d0) possemdef = .false.
        if (eigvR(i)<1d-14) posdef = .false.
        if (eigvR(i)<1d-13.and.printl>=4) &
            write(stdout,'(&
            "Attention: KM matrix has a very small/negative eigenvalue!")')
    end do
    cond_nr = abs(largestEigV)/(abs(smallestEigV))
    this%K_stat = -1
end subroutine check_KM
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/check_Matrix
!!
!! FUNCTION
!! Checks an arbitrary matrix for symmetry and (semi-)definiteness
!! possemdef is determined via the symmetric part of KM
!! ATTENTION! It changes the covariance matrix!
!!
!! SYNOPSIS
subroutine check_Matrix(matrix,d,sym,possemdef,posdef)
!! SOURCE
    implicit none  
    integer, intent(in)         ::  d
    real(rk),intent(inout)      ::  matrix(d,d)
    logical, intent(out)        ::  possemdef
    logical, intent(out)        ::  posdef
    logical, intent(out)        ::  sym
    integer     ::  i, j, info
    real(rk)     ::  eigvR(d), eigvI(d)
    real(rk)     ::  dummy(1,d), dummy2(1,d)
    real(rk)     ::  work(12*d), smallestEigV, largestEigV,cond_nr
    ! Check for symmetry
    sym = .true.
    do i = 1, d
        do j = 1, d
            if (ABS(matrix(i,j)-matrix(j,i))>1d-14) then
                sym=.false.                
            end if
            
        end do
    end do
    ! check for pos-semi-def.
    do i = 1, d
        do j = 1, i
            matrix(i,j) = matrix(i,j) + matrix(j,i)
            !symKM(j,i)=matrix(i,j)+matrix(j,i)
        end do
    end do
    do i = 1, d
        do j = 1, i
            matrix(j,i) = matrix(i,j)
        end do
    end do
    matrix(:,:) = matrix(:,:)/2d0
    call DGEEV('N','N',d,matrix, d, eigvR, eigvI,&
                dummy,1,dummy2,1,work,12*d,info)
    if(.not.(info==0)) call dlf_fail("ERROR in DGEEV.")
    possemdef = .true.
    posdef = .true.
    smallestEigV = eigvR(1)
    largestEigV = eigvR(1)
    do i = 1, d
        if (eigvR(i)> largestEigV) largestEigV = eigvR(i)
        if (eigvR(i)< smallestEigV) smallestEigV = eigvR(i)
        !if (eigvI(i)/=0d0) call dlf_fail("imaginary eigenvalue detected.")
        if (eigvR(i)<0d0) possemdef = .false.
        if (eigvR(i)<1d-14) posdef = .false.
        if (eigvR(i)<1d-13.and.printl>=4) &
            write(stdout,'(&
            "Attention: KM matrix has a very small/negative eigenvalue!")')
    end do
    if(abs(smallestEigV) >1d-16) then
      cond_nr = abs(largestEigV)/(abs(smallestEigV))
    endif
end subroutine check_Matrix
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/writeParameterFile
!!
!! FUNCTION
!! Writes a standardized file in which the parameters of this gpr
!! are written in a human readable form.
!!
!! SYNOPSIS
subroutine writeParameterFile(this)
!! SOURCE
    implicit none
    type(gpr_type), intent(in):: this
    open(unit = 41, file = "Parameters for this run.txt", &
           status = "replace", &
           action='write', &
           position='append')
    write(41,'(A,I1)')"T! kernel", this%kernel_type
    write(41,'(A,ES15.6)')"T! s_f   ", this%s_f
    write(41,'(A,ES15.6)')"T! s_n(1)", this%s_n(1)
    write(41,'(A,ES15.6)')"T! s_n(2)", this%s_n(2)
    write(41,'(A,ES15.6)')"T! s_n(3)", this%s_n(3)
    write(41,'(A,ES15.6)')"T! gamma ", this%gamma
    write(41,'(A,ES15.6)')"T! offset", (this%mean-SUM(this%es(1:this%nt))/this%nt)
    close(41)
end subroutine writeParameterFile
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/LBFGS_Max_Like
!!
!! FUNCTION
!! Using the LBFGS optimizer to maximize the likelihood wrt. 
!! the hyperparameters
!!
!! SYNOPSIS
subroutine LBFGS_Max_Like(this, np_opt, whichParas)
!! SOURCE
    use oop_lbfgs
    implicit none
    type(gpr_type),intent(inout)::  this
    integer, intent(in)         ::  np_opt
    integer, intent(in)         ::  whichParas(np_opt)
    real(rk)                    ::  maxstepsize
    real(rk)                    ::  step(np_opt)
    real(rk)                    ::  stepl
    integer                     ::  member !nr of steps to remember by lbfgs
    real(rk)                    ::  tolerance, tolerance_g
    real(rk)                    ::  p(np_opt)
    real(rk)                    ::  gh(np_opt)
    real(rk)                    ::  like
    integer                     ::  i,j, stepnr
    type(oop_lbfgs_type)        ::  lbfgs
    3335 FORMAT (A3,I5,A12,1X, ES11.4,5X, ES11.4,5X, ES11.4, 5X, ES11.4, 5X, ES11.4)    
    3334 FORMAT (A3,I5,A12,1X, ES11.4,5X, ES11.4,5X, ES11.4, 5X, ES11.4)    
    3333 FORMAT (A3,I5,A12,1X, ES11.4,5X, ES11.4,5X, ES11.4)    
    3332 FORMAT (A3,I5,A12,1X, ES11.4,5X, ES11.4)    
    3331 FORMAT (A3,I5,A12,1X, ES17.10)  
    3330 FORMAT (A3,ES11.4,1X,ES11.4,1X,ES11.4,1X,ES11.4)  
    
    tolerance = 1d-5
    tolerance_g = 1d-4
    maxstepsize=1d4!0.1d0
    member = 20!MAX(5,np_opt/2)
    call lbfgs%init(np_opt, member, .false.)
!     call gpr_lbfgs_init(np_opt, member, .false.)
    gh(:)=tolerance_g*2d0
       
    stepl=tolerance*2d0
    stepnr=0
!     this%l=2.0d1
!     this%gamma = 1d0/this%l**2
    do while (stepnr<20.and.stepl>tolerance.and.MAXVAL(ABS(gh))>tolerance_g)
        stepnr=stepnr+1
      if (.not.this%iChol) then
        call calc_cov_mat(this)
      else 
        call calc_cov_mat_newOrder(this)
      endif
      call calc_p_like_and_deriv(this, np_opt, whichParas, like, gh, p)
        
!       call scale_p(this,.true., np_opt, whichParas, p(:), gh(:))
      gh(:) = -gh(:)
      call lbfgs%next_step(p, gh, step)
!       call gpr_lbfgs_step(p, gh, step)
        
      do j = 1, np_opt
        if (abs(step(j))>maxstepsize) then
          step(j)=SIGN(maxstepsize,step(j))
        end if
        p(j)=p(j)+step(j)
      end do  
      stepl=dsqrt(dot_product(step,step))
      write(stdout,3330) "O2!", p(1), stepl, gh, like
!       call scale_p(this,.false.,np_opt, whichParas, p)
      do j = 1, np_opt
        i = whichParas(j)    
        if (i==1) then
          call GPR_changeparameters(this,this%gamma,p(j),this%s_n)
          !this%s_f      = p(j)
          !this%s_f2 = this%s_f**2
        end if
        if (i==2) then
          if (this%kernel_type==0) then
            call GPR_changeparameters(this,p(j),this%s_f,this%s_n)
            !this%gamma = p(j)
            !this%l = 1d0/dsqrt(p(j))
          else if (this%kernel_type==1) then
            call GPR_changeparameters(this,1d0/(p(j))**2,this%s_f,this%s_n)
            !this%l = p(j)
            !this%gamma = 1d0/(p(j))**2
          else 
            call dlf_fail("KERNEL NOT IMPLEMENTED IN LBFGS_Max_Like")
          end if
        end if
        if (i==3) this%s_n(1)   = p(j)
        if (i==4) this%s_n(2)   = p(j)
        if (i==5) this%s_n(3)   = p(j)
      end do  
      ! this must be called, otherwise higher powers of l etc. are wrong
      call GPR_changeparameters(this, 1/(this%l)**2, this%s_f, this%s_n)
      if (np_opt==1) write(*,3331) "O!",stepnr," Hyperparas", p(:)!maximum(:)
      if (np_opt==2) write(*,3332) "O!",stepnr," Hyperparas", p(:)!maximum(:)
      if (np_opt==3) write(*,3333) "O!",stepnr," Hyperparas", p(:)!maximum(:)
      if (np_opt==4) write(*,3334) "O!",stepnr," Hyperparas", p(:)!maximum(:)
      if (np_opt==5) write(*,3335) "O!",stepnr," Hyperparas", p(:)!maximum(:)
      if (printl>=4) then
        write(stdout,'("Change in hyperparameters (hparaSpace)       ", ES11.4)') step
        write(stdout,'("log(Likelihood) (lastStep)                   ", ES11.4)') like
        write(stdout,'("Gradlength of log(likel.) (lastStep)(hPSpace)", ES11.4)') dsqrt(dot_product(gh(:),gh(:)))
      end if
    end do
    call lbfgs%destroy()
    if (printl>=4) &
        write(stdout,'("Maximum likelihood optimization complete.")')
    this%scaling_set=.false.
end subroutine LBFGS_Max_Like 
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/Plot_Likel_onePara
!!
!! FUNCTION
!! Write out data to plot the likelihood wrt. one of the hyperparameters
!!
!! SYNOPSIS
subroutine Plot_Likel_onePara(this,whichParas, begin, end, resolution)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::  this
    integer, intent(in)         ::  whichParas(1)
    real(rk), intent(in)        ::  begin, end
    integer                     ::  resolution
    integer                     ::  i,j
    real(rk)                    ::  tmp
    real(rk)                    ::  g(1)
    real(rk)                    ::  gh(1),like
    integer                     ::  istat
    character(len=1)         :: nt0
    character(len=1)         :: para
    character(len=2)         :: nt1
    character(len=3)         :: nt2
    character(len=4)         :: nt3
    character(len=15)        :: test
   ! test = char(this%nt)
   if(this%nt>0 .and. this%nt<10) then
      write(nt0,"(I1)") this%nt
      write(para,"(I1)") whichParas(1)
      test = para//'_'//nt0
    elseif(this%nt>=10 .and. this%nt<100) then
      write(nt1,"(I2)") this%nt
      write(para,"(I1)") whichParas(1)
      test = para//'_'//nt1
    elseif(this%nt>=100 .and.this%nt<1000) then
      write(nt2,"(I3)") this%nt
      write(para,"(I1)") whichParas(1)
      test = para//'_'//nt2    
    elseif(this%nt>=1000 .and. this%nt>10000) then
       write(nt3,"(I4)") this%nt
      write(para,"(I1)") whichParas(1)
      test = para//'_'//nt3 
    else
      call dlf_fail("too much tp for plotting liklihood")
    endif 
    test = 'plt_'//test
open(unit=20,file=test,iostat=istat)
if (this%kernel_type==0) then    
    tmp = begin
    do i = 1, resolution
        tmp = tmp+(end-begin)/resolution
        j = whichParas(1)    
        if (j==1) this%s_f      = tmp
        if (j==2) this%gamma    = tmp
        if (j==3) this%s_n(1)   = tmp
        if (j==4) this%s_n(2)   = tmp
        if (j==5) this%s_n(3)   = tmp
        this%gamma = tmp
        call calc_cov_mat(this)
        print*,"Plotting parameter ", whichParas(1)
        call calc_p_like_and_deriv(this, 1, (/ 2 /), like, gh, g) 
        print*, "ToPlot", g, like, gh
    end do 
else if(this%kernel_type == 1) then
    tmp = begin
    do i = 1, resolution
        tmp = tmp+(end-begin)/resolution
        j = whichParas(1)    
        if (j==3 .or. j==4) then
          tmp = exp(tmp)
        endif
        if (j==1) this%s_f      = tmp
        if (j==2) call GPR_changeparameters(this,1/tmp**2,this%s_f,this%s_n)
        if (j==3) this%s_n(1)   = tmp
        if (j==4) this%s_n(2)   = tmp
        if (j==5) this%s_n(3)   = tmp
        !this%l = tmp
        call calc_cov_mat(this)
        if(printl>=6) then
          print*,"Plotting parameter ", whichParas(1)
        endif
        call calc_p_like_and_deriv(this, 1, (/ j /), like, gh, g) 
        if(printl>=6)then
          print*, "ToPlot", g, like, gh
        endif
        write(20,*) g, like, gh
        if (j==3 .or. j==4) then
          tmp = log(tmp)
        endif        
        
    end do     
else   
    call dlf_fail("Kernel Type not implemented! (Plot_Likel_onePara)")
end if    
close(20)
end subroutine Plot_Likel_onePara

subroutine Plot_Likel(this,nrParas,whichParas, begin, end, resolution)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::  this
    integer, intent(in)         :: nrParas
    integer, intent(in)         ::  whichParas(nrParas)
    real(rk), intent(in)        ::  begin(nrParas), end(nrParas)
    integer                     ::  resolution
    integer                     ::  i,j,k,r
    real(rk)                    ::  tmp(nrPAras)
    real(rk)                    ::  g(nrParas)
    real(rk)                    ::  gh(nrParas),like
    integer                     ::  istat
    character(len=1)         :: nt0
    character(len=1)         :: para
    character(len=2)         :: nt1
    character(len=3)         :: nt2
    character(len=4)         :: nt3
    character(len=15)        :: test
   ! test = char(this%nt)
   if(this%nt>0 .and. this%nt<10) then
      write(nt0,"(I1)") this%nt
!      write(para,"(I1)") whichParas(1)
      test = '_'//nt0
    elseif(this%nt>=10 .and. this%nt<100) then
      write(nt1,"(I2)") this%nt
!      write(para,"(I1)") whichParas(1)
      test = '_'//nt1
    elseif(this%nt>=100 .and.this%nt<1000) then
      write(nt2,"(I3)") this%nt
!      write(para,"(I1)") whichParas(1)
      test = '_'//nt2    
    elseif(this%nt>=1000 .and. this%nt>10000) then
       write(nt3,"(I4)") this%nt
!      write(para,"(I1)") whichParas(1)
      test = '_'//nt3 
    else
      call dlf_fail("too much tp for plotting liklihood")
    endif 
    test = 'plt'//test
open(unit=20,file=test,iostat=istat)
if (this%kernel_type==0) then    
    tmp = begin
    do i = 1, resolution
        tmp = tmp+(end-begin)/resolution
      do k=1,nrParas  
        j = whichParas(k)    
        if (j==1) this%s_f      = tmp(k)
        if (j==2) this%gamma    = tmp(k)
        if (j==3) this%s_n(1)   = tmp(k)
        if (j==4) this%s_n(2)   = tmp(k)
        if (j==5) this%s_n(3)   = tmp(k)
        this%gamma = tmp(k)
      enddo
        call calc_cov_mat(this)
        print*,"Plotting parameter ", whichParas(1)
        call calc_p_like(this, nrPAras, whichParas, like, g) 
        print*, "ToPlot", g, like
    end do 
else if(this%kernel_type == 1) then
    tmp = begin
    do i = 1, resolution
        tmp(1) = tmp(1)+(end(1)-begin(1))/resolution  
     ! do r =1,resolution
     !    tmp(2) = tmp(2)+(end(2)-begin(2))/resolution      
      do k=1,nrParas  
        j = whichParas(k)    
        if (j==3 .or. j==4) then
          tmp(k) = exp(tmp(k))
        endif
        if (j==1) this%s_f      = tmp(k)
        if (j==2) call GPR_changeparameters(this,1/tmp(k)**2,this%s_f,this%s_n)
        if (j==3) this%s_n(1)   = tmp(k)
        if (j==4) this%s_n(2)   = tmp(k)
        if (j==5) this%s_n(3)   = tmp(k)
        !this%l = tmp
      enddo  
        if(this%iChol) then
          if(this%K_stat == -1) then
            call calc_cov_mat_newOrder(this)
          elseif(this%K_stat==6)  then 
            call calc_cov_mat_newOrder(this,.true.)
            this%K_stat =7
          endif
        else
          call calc_cov_mat(this)
        endif
        if(printl>=6) then
          print*,"Plotting parameter ", whichParas(1)
        endif
        call calc_p_like(this, nrParas, whichParas, like, g) 
        this%K_stat=-1
        if(printl>=6)then
          print*, "ToPlot", g, like
        endif
        write(20,*) g, like!, gh
      do k=1,nrParas  
        j = whichParas(k) 
        if (j==3 .or. j==4) then
          tmp(k) = log(tmp(k))
        endif        
      enddo
    !end do 
    !tmp(2) = begin(2)
    enddo    
else   
    call dlf_fail("Kernel Type not implemented! (Plot_Likel_onePara)")
end if    
close(20)
end subroutine Plot_Likel
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/init_random_seed
!!
!! FUNCTION
!! Initializes the creation of random variables
!!
!! SYNOPSIS
SUBROUTINE init_random_seed()
!! SOURCE
    INTEGER :: i, n, clock
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
          
    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
          
    CALL SYSTEM_CLOCK(COUNT=clock)
          
    seed = clock + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
          
    DEALLOCATE(seed)
END SUBROUTINE init_random_seed
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_distance_nearest_tp
!!
!! FUNCTION
!! Determines the distance of a given point to the nearest training point.
!! If the given point is a training point itself, the distance to the
!! nearest training point that is not itself is put out..
!!
!! SYNOPSIS
subroutine GPR_distance_nearest_tp(this,pos,mindist,output)
!! SOURCE
    implicit none
    type(gpr_type),intent(inout)::this
    real(rk), intent(in) ::  pos(this%sdgf)
    integer              ::  i
    real(rk), intent(out):: mindist ! minimal distance to a training point
    real(rk)             ::  tmp
    logical, intent(in)  ::  output ! Should this be written out as text?!
    if (this%nt<1) then
      if(printl>=6) &
        write(stdout,'("no training points, cannot find nearest one.")')
    end if
    mindist = dsqrt(dot_product(pos(:)-this%xs(:,1),&
                                pos(:)-this%xs(:,1)))
    
    do i = 2, this%nt
        tmp = dsqrt(dot_product(pos(:)-this%xs(:,i),&
                                pos(:)-this%xs(:,i)))
        if (tmp<mindist .and. tmp/=0d0) mindist = tmp
        if (tmp==0d0) write(stdout,'(&
          "This point is a trainingpoint, ",&
          "distance to next tp will be put out")')
    end do
    
    3339 FORMAT (A,1X, ES11.4)  
    if (output.and.printl>=4) then
        write(stdout,3339) "The distance to the nearest training point is", mindist
    end if
end subroutine GPR_distance_nearest_tp
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/GPR_newLevel
!!
!! FUNCTION
!! Introduces a multilevelGPR, which means that a number of
!! ntLowerLevel points are used to construct a GPR
!! which serves as a prior estimate for a GPR with the remaining points.
!! That decreases the computational effort but also the accuracy.
!! This method can be used recursively, meaning several levels can be 
!! introduced. It sets the OffsetType to 5.
!! Definition: The higher a level the deeper down in the recursion
!! the GPR surface lies. The highest (current) GPR has level 0.
!!
!! SYNOPSIS
subroutine GPR_newLevel(this, ntLowerLevel)
!! SOURCE
  type(gpr_type), target, intent(inout) :: this
  type(gpr_type), pointer       :: lowLevelGPR
  type(gpr_type), pointer       :: tmpPt
  integer                       :: ntLowerLevel !Number of traningpoints
                                    ! that should be included in a lower level
  if (printl>=4) &
        write(stdout,'("Introducing a new GPR level.")')
  if (ntLowerLevel>this%nt) call dlf_fail(&
    "The number of trainingpoints for the lower level is too high!")
  allocate(lowLevelGPR)                     
!   if(.not.associated(this%lowerLevelGPR)) call GPR_interpolation(this)
  call GPR_copy_GPR(this,lowLevelGPR)
    ! Solves the lower Level GPR 
  ! (only queries from that, no further changes intended)
  this%lowerLevelGPR => lowLevelGPR
  tmpPt=>this
  do while (associated(tmpPt%lowerLevelGPR))
    tmpPt%lowerLevelGPR%level = tmpPt%lowerLevelGPR%level + 1
    tmpPt=>tmpPt%lowerLevelGPR
  end do
  call GPR_deleteNewestTPs(this%lowerLevelGPR,this%nt-ntLowerLevel)
  call GPR_interpolation(this%lowerLevelGPR)
  this%OffsetType = 5
  call GPR_deleteOldestTPs(this,ntLowerLevel)
  this%nSubLevels = this%nSubLevels + 1
end subroutine GPR_newLevel
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/totalTPnumber
!!
!! FUNCTION
!! Calculates the total number of training points over all levels
!!
!! SYNOPSIS    
integer function totalTPnumber(this)
!! SOURCE
  implicit none
  type(gpr_type), target, intent(in)    ::  this
  type(gpr_type), pointer               ::  pointToLevel
  pointToLevel => this
  totalTPnumber = 0
  do while (pointToLevel%offsetType == 5)
    totalTPnumber = totalTPnumber + pointToLevel%nt
    pointToLevel => pointToLevel%lowerLevelGPR
  end do
  totalTPnumber = totalTPnumber + pointToLevel%nt
end function totalTPnumber



! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/orthogonalizeVecToVecs
!!
!! FUNCTION
!! Orthogonalizing a vector v to a set of vectors vecs
!!
!! SYNOPSIS
subroutine orthogonalizeVecToVecs(d,m, v, vecs)
!! SOURCE
    integer, intent(in)         ::  d,m
    real(rk), intent(inout)     ::  v(d)
    real(rk), intent(in)        ::  vecs(d,m)
    real(rk)                    ::  length
    integer                     ::  i
    if (m>d-1) call dlf_fail("NOT POSSIBLE!")
    do i = 1, m
      ! vecs must be normalized, therefore I have to
      ! devide by its length**2
      length=dot_product(vecs(:,i),vecs(:,i))
      v(:)=v(:)-vecs(:,i)*dot_product(vecs(:,i),v(:))/length
    end do
    
    
    if (length<1d-10) then 
      if (printl>=4) &
        write(stdout,'(&
          "Orthogonalizing a random vec instead of original direction.")')
      call init_random_seed()
      do i = 1, d
        call RANDOM_NUMBER(v(i))
      end do    
      v(:)=v(:)/dsqrt(dot_product(v,v))
      do i = 1, m
        ! vecs must be normalized, therefore I have to
        ! devide by its length**2
        length=dot_product(vecs(:,i),vecs(:,i))
        v(:)=v(:)-vecs(:,i)*dot_product(vecs(:,i),v(:))/length
        v(:)=v(:)/dsqrt(dot_product(v,v))
      end do
    else
      v=v/dsqrt(dot_product(v,v))
    end if
end subroutine orthogonalizeVecToVecs
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/eigVec_to_smallestEigVal
!!
!! FUNCTION
!! Outputs the eigenvector v corresponding to the smallest eigval e
!! of a nxn matrix A
!!
!! SYNOPSIS
subroutine eigVec_to_smallestEigVal(n,A,v,e)
!! SOURCE
    integer, intent(in)     ::  n
    real(rk), intent(inout) ::  A(n,n)
    real(rk), intent(out)   ::  v(n)
    real(rk)                ::  eigvals(n)
    real(rk)                ::  e
    real(rk)                ::  work(1 + 6*n + 2*n**2)
    integer                 ::  info = 0
    integer                 ::  iwork(3 + 5*n)!,m,nb, iworkx(5*n), ifail(n)
!     integer, allocatable    ::  lwork_al(:)
    integer, external               ::  ILAENV
!     double precision, external      ::  DLAMCH
    call DSYEVD('V','U',n,A,n,eigvals,work,1 + 6*n + 2*n**2,iwork,3 + 5*n, info)
!     nb = MAX(ILAENV(1,'DSYTRD','VIU',n,-1,-1,-1),&
!                     ILAENV(1,'DORMTR','VIU',n,-1,-1,-1))
!     allocate(lwork_al(n*8))
!     call DSYEVX('V','I','U',n,A,n,0,0,1,1,2*DLAMCH('S'),m,e, v,n,&
!                 lwork_al,n*8,iworkx,ifail,info)
    if (info/=0) call dlf_fail("DSYEVD did not stop successfully.")
    e=eigvals(1)
    v(:) = A(:,1)
end subroutine eigVec_to_smallestEigVal
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/outputMatrixAsCSV
!!
!! FUNCTION
!! Puts out an arbitrary matrix in a file in CSV format
!!
!! SYNOPSIS
subroutine outputMatrixAsCSV(rows,columns,matrix,filename)
!! SOURCE
    integer, intent(in)     ::  rows, columns
    real(rk),intent(in)     ::  matrix(rows,columns)
    character(*), intent(in)    :: filename
    integer                 ::  i,j
    open(unit = 40, file = filename, status = "replace",action='write')
2312 format(ES15.7)           
2313 format(A2) 
2314 format(A)
    do i = 1, rows    
      do j = 1, columns
        write(40,2312,advance='no') matrix(i,j)
        if (j<columns) then
            write(40,2313,advance='no') ", "
        end if
      end do  
      write(40,2314) ""
    end do
    close(40)
end subroutine outputMatrixAsCSV
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/orderKM
!!
!! FUNCTION
!! orders the covariance matrix to simplify
!! iterativeGPR
!!
!! SYNOPSIS
subroutine orderKM(this, outKM)
!! SOURCE
    type(gpr_type),intent(inout)::  this
    real(rk), intent(out)       ::  outKM(this%nk,this%nk)
    integer                     ::  i,j,k,l,d,nt
    
    d=this%sdgf
    nt=this%nt
    if(this%order==1) then
      ! energy block
      do i = 1, nt
        do j = i, nt
          outKM((d+1)*(i-1)+1, (d+1)*(j-1)+1) = &
                  this%KM(i,j)
        end do
      end do
      ! energy/gradient block
      do i = 1, nt
        do k = 0, nt-1
          outKM((d+1)*(i-1)+1,&
              (d+1)*k+2:(d+1)*k+2+d-1)=&
              this%KM(i,nt+k*d+1:nt+(k+1)*d)
        end do
      end do
      ! gradient/gradient block
      do k = 0, nt-1
        do l = 0, nt-1
          outKM((d+1)*k+2:(d+1)*k+2+d-1,&
                                    (d+1)*l+2:(d+1)*l+2+d-1)=&
                  this%KM(nt+k*d+1:nt+(k+1)*d,&
                          nt+l*d+1:nt+(l+1)*d)
        end do
      end do
      ! mirror energy and energy/gradient blocks
      do j = 1, nt
        do i = j+1, nt
          outKM((d+1)*(i-1)+1, (d+1)*(j-1)+1)=&
          outKM((d+1)*(j-1)+1, (d+1)*(i-1)+1)
        end do
      end do
      do j = 1, nt
        do l = 0, nt-1
          outKM((d+1)*l+2:(d+1)*l+2+d-1,&
                                    (d+1)*(j-1)+1)=&
              outKM((d+1)*(j-1)+1,&
                                        (d+1)*l+2:(d+1)*l+2+d-1)                           
        end do
      end do
    else if (this%order==2) then
      do i = 1, this%nk
        if (i<=this%nt) then
          ! energies
          k = (i-1)*(this%sdgf*(this%sdgf+1)/2+this%sdgf+1)+1
        else if (i<=this%nt*(1+this%sdgf)) then
          ! gradients
          k = ((i-this%nt-1)/this%sdgf)*(this%sdgf*(this%sdgf+1)/2+this%sdgf+1)+&
              (MOD(i-this%nt-1,this%sdgf)+1)+1    
        else 
          k = ((i-this%nt*(1+this%sdgf)-1)/((this%sdgf+1)*this%sdgf/2))*& !tp number - 1
              (this%sdgf*(this%sdgf+1)/2+this%sdgf+1)+&   ! # elements per tp
              (MOD(i-this%nt*(1+this%sdgf)-1,(this%sdgf*(this%sdgf+1)/2))+1)+& ! hessian entry
              (this%sdgf+1) ! offset (e+g schon gemacht)
        end if
        do j = i, this%nk    
          if (j<=this%nt) then
            ! energies
            l = (j-1)*(this%sdgf*(this%sdgf+1)/2+this%sdgf+1)+1
          else if (j<=this%nt*(1+this%sdgf)) then
            ! gradients
            l = ((j-this%nt-1)/this%sdgf)*(this%sdgf*(this%sdgf+1)/2+this%sdgf+1)+&
                (MOD(j-this%nt-1,this%sdgf)+1)+1        
          else 
            l = ((j-this%nt*(1+this%sdgf)-1)/((this%sdgf+1)*this%sdgf/2))*& !tp number
                (this%sdgf*(this%sdgf+1)/2+this%sdgf+1)+&   ! # elements per tp
                (MOD(j-this%nt*(1+this%sdgf)-1,(this%sdgf*(this%sdgf+1)/2))+1)+& ! hessian entry
                (this%sdgf+1) ! offset (e+g schon gemacht)
          end if
          outKM(i,j) = this%KM(k,l)
        end do
      end do    
    else
      call dlf_fail("This order is not implemented for orderKM.")
    end if
end subroutine orderKM
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/orderW
!!
!! FUNCTION
!! orders W to simplify
!! iterativeGPR
!!
!! SYNOPSIS
subroutine orderW(this, w_old, w_sorted)
!! SOURCE
    type(gpr_type),intent(inout)::  this
    real(rk), intent(in)        ::  w_old(this%nk)
    real(rk), intent(out)       ::  w_sorted(this%nk)
    integer                     ::  i,k,g_os, h_os
    integer                     ::  sortIndex, eIndex,gIndex, hIndex
    
    if (this%order==0) then
      ! nothing to do
      w_sorted(:) = w_old(:)
    else if (this%order==1) then
      g_os = this%idgf
      ! energies
      do i = 0, this%nt-1
        w_sorted(i*(g_os+1)+1)=w_old(i+1)
      end do
      ! gradients
      do k = 0, this%nt-1
        w_sorted(k*(g_os+1)+2:k*(g_os+1)+1+g_os) = &
          w_old(this%nt+g_os*k+1:this%nt+g_os*(k+1))
      end do
    else if (this%order==2) then
      h_os = this%sdgf*(this%sdgf+1)/2+this%sdgf+1
      ! energies
      do i = 0, this%nt-1
        w_sorted(i*(h_os)+1)=w_old(i+1)
      end do
      g_os = this%sdgf
      ! gradients
      do k = 0, this%nt-1
        w_sorted(k*(h_os)+2:k*(h_os)+1+g_os) = &
          w_old(this%nt+g_os*k+1:this%nt+g_os*(k+1))
      end do
      !hessians
      do k = 0, this%nt-1
        w_sorted(k*(h_os)+this%sdgf+2:(k+1)*(h_os)) = &
          w_old(this%nt+g_os*this%nt+k*(this%sdgf*(this%sdgf+1)/2)+1:&
                this%nt+g_os*this%nt+(k+1)*(this%sdgf*(this%sdgf+1)/2))
      end do
    else if (this%order==42) then
      eIndex = 1
      gIndex = this%nt + 1 ! all training points have energy information
      hIndex = this%nt + this%sdgf*this%ntg + 1
      sortIndex = 1
      
      do i = 1, this%nt
        w_sorted(sortIndex)=w_old(eIndex)
        eIndex = eIndex + 1
        sortIndex = sortIndex + 1
        if (this%order_tps(i)>0) then
          w_sorted(sortIndex:sortIndex+this%sdgf-1) = &
            w_old(gIndex:gIndex+this%sdgf-1)
          gIndex = gIndex + this%sdgf
          sortIndex = sortIndex + this%sdgf
        end if
        if (this%order_tps(i)>1) then
          w_sorted(sortIndex:sortIndex+(this%sdgf+1)*this%sdgf/2-1) = &
            w_old(hIndex:hIndex+(this%sdgf+1)*this%sdgf/2-1)
          sortIndex = sortIndex + this%sdgf*(this%sdgf+1)/2
          hIndex = hIndex + this%sdgf*(this%sdgf+1)/2
        end if
      end do
    else
      call dlf_fail("OrderW not implemented for this order.")
    end if
end subroutine orderW
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/orderW_inplace
!!
!! FUNCTION
!! orders W to simplify
!! iterativeGPR
!!
!! SYNOPSIS
subroutine orderW_inplace(this, w)
!! SOURCE
    type(gpr_type),intent(inout)::  this
    real(rk), intent(inout)     ::  w(this%nk)
    real(rk)                    ::  w_old(this%nk)
    integer                     ::  i,k,d
    w_old(:) = w(:)
    d = this%sdgf
    if (this%order==0) then
      ! nothing to do
    else if (this%order==1) then
      do i = 0, this%nt-1
        w(i*(d+1)+1)=w_old(i+1)
      end do
    
      do k = 0, this%nt-1
        w(k*(d+1)+2:k*(d+1)+2+d-1) = &
          w_old(this%nt+d*k+1:this%nt+d*(k+1))
      end do
    else
      call dlf_fail("OrderW not implemented for higher order than 1.")
    end if
end subroutine orderW_inplace
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/reorderW
!!
!! FUNCTION
!! reorders the W to simplify
!! iterativeGPR
!!
!! SYNOPSIS
subroutine reorderW(this, w_old, w_new)
!! SOURCE
    type(gpr_type),intent(inout)::this    
    real(rk), intent(in)        ::  w_old(this%nk)
    real(rk), intent(out)       ::  w_new(this%nk)
    integer                     ::  i,k,d
    
if (this%order==0) then
    ! nothing to do, energies are sorted exactly the same
    w_new(:) = w_old(:)
else if (this%order==1) then
    d = this%sdgf
    ! energies
    do i = 0, this%nt-1
      w_new(i+1)=w_old(i*(d+1)+1)
    end do    
    ! gradients
    do k = 0, this%nt-1
      w_new(this%nt+d*k+1:this%nt+d*(k+1))=&
        w_old(k*(d+1)+2:k*(d+1)+2+d-1)
    end do    
else if (this%order==2) then
    d = (this%sdgf+1)*this%sdgf/2+this%sdgf+1
    ! energies
    do i = 0, this%nt-1
      w_new(i+1)=w_old(i*d+1)
    end do
    ! gradients
    do i = 0, this%nt-1
      w_new(this%nt+i*this%sdgf+1:this%nt+(i+1)*this%sdgf)=&
        w_old(i*d+2:i*d+1+this%sdgf)
    end do
    ! hessians
    do i = 0, this%nt-1
      w_new(this%nt+this%nt*this%sdgf+(this%sdgf+1)*this%sdgf/2*i+1:&
               this%nt+this%nt*this%sdgf+(this%sdgf+1)*this%sdgf/2*(i+1))=&
        w_old(i*d+(this%sdgf+1)+1:(i+1)*d)
    end do
else 
    call dlf_fail("This order is not implemented for reorderW.")
end if
end subroutine reorderW
!!****

end module gpr_module


! Needed for compatibility with Chemshell and DL-Find without losing 
! the other one. (Nasty but I found no alternative)
module gpr_in_chemsh_pes_module
  ! this module should contain all data which are used to read in
  ! coordinates, gradients and Hessians and to write out the neural
  ! network data without being required by the neural network
  ! optimization itself
  implicit none
  integer,save :: nat
  logical,save :: tinit=.false.
  logical,save :: massweight=.true.
  real(8),allocatable,save :: align_refcoords(:) ! (3*nat)
  real(8),allocatable,save :: align_modes(:,:)   ! (3*nat,ncoord)
  real(8),allocatable,save :: refmass(:) ! (nat)
end module gpr_in_chemsh_pes_module








!***************************************************************************
! Inclusions from dl-find
!***************************************************************************
module gpr_driver_parameter_module
  use dlf_parameter_module
  ! variables for Mueller-Brown potential
  real(rk) :: acappar(4),apar(4),bpar(4),cpar(4),x0par(4),y0par(4)
  ! variables for Lennard-Jones potentials
  real(rk),parameter :: epsilon=1.D-1
  real(rk),parameter :: sigma=1.D0
!!$  ! variables for the Eckart potential (1D), taken from Andri's thesis
!!$  real(rk),parameter :: Va= -0.191D0*0.0367493254D0 !2nd factor is conversion to E_h from eV
!!$  real(rk),parameter :: Vb=  1.343D0*0.0367493254D0 !2nd factor is conversion to E_h from eV
!!$  real(rk),parameter :: alpha= 5.762D0/1.889725989D0 !2nd factor is for conversion from Angstrom to Bohr (a.u.)
! variables for the Eckart potential (1D), to match the OH+HH reaction from pes/OH-H2
  real(rk),parameter :: Va= -0.02593082161D0
  real(rk),parameter :: Vb= 0.07745577734638293D0
  real(rk),parameter :: alpha= 2.9D0 !2.69618D0 !2nd factor is for conversion from Angstrom to Bohr (a.u.)
  real(rk),parameter :: dshift=-2.2D0
  real(rk),parameter :: dip=1.D-3 !8.042674107D-04 ! vmin=-8.042674107D-04 is the value from the PES, to be adjusted!
  !real(rk) :: alprime !=sqrt(ddv*8.D0/(dip*4.D0))*244.784D0/1206.192D0

  ! modified Eckart potential to have a comparison
!!$  REAL(8), PARAMETER :: Va= -0.091D0*0.0367493254D0  !2nd factor is conversion to E_h from eV
!!$  REAL(8), PARAMETER :: Vb=  1.343D0*0.0367493254D0  !2nd factor is conversion to E_h from eV
!!$  REAL(8), PARAMETER :: alpha =20.762D0/1.889725989D0 
!!$  real(rk),parameter :: Va= -0.191D0*0.0367493254D0 !2nd factor is conversion to E_h from eV
!!$  real(rk),parameter :: Vb=  1.343D0*0.0367493254D0 !2nd factor is conversion to E_h from eV
!!$  real(rk),parameter :: alpha= 20.762D0/1.889725989D0 !2nd factor is for conversion from Angstrom to Bohr (a.u.)

  ! "fitted" to Glum SN-1-Glu, E->D
!  real(rk),parameter :: Va= 0.D0 ! symmetric (real: almost symmetric)
!  real(rk),parameter :: Vb= 4.D0*74.81D0/2625.5D0 ! 4*barrier for symmetric
!  real(rk),parameter :: alpha= 2.38228D0 

!!$  ! Eckart for comparison with scattering
!!$  real(rk),parameter :: Va= -100.D0
!!$  real(rk),parameter :: Vb= 373.205080756888D0
!!$  real(rk),parameter :: alpha= 7.07106781186547D0

  real(rk) :: xvar,yvar
  ! quartic potential
  real(rk),parameter :: V0=1.D-2
  real(rk),parameter :: X0=5.D0
  real(rk),parameter :: shift=0.9D0
  real(rk),parameter :: steep=20.D0
  real(rk),parameter :: delpot=1.D-5
  ! polymul
  integer ,parameter :: num_dim=3 ! should be a multiple of 3
  real(rk),parameter :: Ebarr=80.D0/2526.6D0  ! >0
  real(rk) :: dpar(num_dim)
!!$  real(rk),parameter :: vdamp=15.D0 ! change of frequencies towards TS. +inf -> no change
  ! isystem 9 
  real(rk), parameter :: low=-0.1D0
  real(rk), parameter :: par3=-1.D0
  real(rk), parameter :: par2=1.D0
  real(rk), parameter :: parb=3.D0
end module gpr_driver_parameter_module

subroutine gpr_driver_init
  use gpr_driver_parameter_module
  implicit none
  real(rk) :: ebarr_
  integer :: icount
  ! assign parameters for MB potential
  ebarr_=0.5D0
  acappar(1)=-200.D0*ebarr_/106.D0
  acappar(2)=-100.D0*ebarr_/106.D0
  acappar(3)=-170.D0*ebarr_/106.D0
  acappar(4)=  15.D0*ebarr_/106.D0
  apar(1)=-1.D0
  apar(2)=-1.D0
  apar(3)=-6.5D0
  apar(4)=0.7D0
  bpar(1)=0.D0
  bpar(2)=0.D0
  bpar(3)=11.D0
  bpar(4)=0.6D0
  cpar(1)=-10.D0
  cpar(2)=-10.D0
  cpar(3)=-6.5D0
  cpar(4)=0.7D0
  x0par(1)=1.D0
  x0par(2)=0.D0
  x0par(3)=-0.5D0
  x0par(4)=-1.D0
  y0par(1)=0.D0
  y0par(2)=0.5D0
  y0par(3)=1.5D0
  y0par(4)=1.D0
  ! parameters for polymul
  dpar(1)=0.D0
  do icount=2,num_dim
    dpar(icount)=1.D-4+(dble(icount-2)/dble(num_dim-2))**2*0.415D0
  end do
end subroutine gpr_driver_init

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine gpr_dlf_get_gradient(nvar,coords,energy,gradient)
  !  Mueller-Brown Potential
  !  see K Mueller and L. D. Brown, Theor. Chem. Acta 53, 75 (1979)
  !  taken from JCP 111, 9475 (1999)
  use dlf_parameter_module, only: rk
!   use driver_module
  use gpr_driver_parameter_module
!   use dlf_constants, only: dlf_constants_get
!   use dlf_allocate, only: allocate,deallocate
!   use pes_nn_av_module
  !use vib_pot
  implicit none
  integer, intent(in)      :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: energy
  real(rk)  ,intent(out)   :: gradient(nvar)
  !
  ! variables for Mueller-Brown potential
  real(rk) :: x,y,svar,svar2
  integer  :: icount
  ! variables for Lennard-Jones potentials
  real(rk) :: pi
  ! additional variables non-cont. diff MEP potential
!   real(rk) :: t ! variation of the depth of the flater saddle point


! **********************************************************************
  pi=4.D0*atan(1.D0)
!  call test_update
  
      x =  coords(1)
      y =  coords(2)
      
      energy=0.D0
      gradient=0.D0
      do icount=1,4
        svar= apar(icount)*(x-x0par(icount))**2 + &
            bpar(icount)*(x-x0par(icount))*(y-y0par(icount)) + &
            cpar(icount)*(y-y0par(icount))**2 
        svar2= acappar(icount) * dexp(svar)
        energy=energy+ svar2
        gradient(1)=gradient(1) + svar2 * &
            (2.D0* apar(icount)*(x-x0par(icount))+bpar(icount)*(y-y0par(icount)))
        gradient(2)=gradient(2) + svar2 * &
            (2.D0* cpar(icount)*(y-y0par(icount))+bpar(icount)*(x-x0par(icount)))
      end do
      energy=energy+0.692D0
      ! write(*,'("x,y,func",2f10.5,es15.7)') x,y,energy
end subroutine gpr_dlf_get_gradient




! Lots of stuff copied from dl-find to initialize prfo optimizer

! module for Optimisers using the Hessian
module dlf_hessian_gpr
  use dlf_parameter_module, only: rk
  logical ,save        :: fd_hess_running ! true if FD Hessian is currently running
                                   ! initially set F in dlf_formstep_init
  integer ,save        :: nihvar   ! size of Hessian (nihvar, nihvar)
  integer, save        :: numfd    ! number of FD Hessian eigenmodes to calculate
  integer ,save        :: iivar,direction
  real(rk),save        :: soft ! Hessian eigenvalues absolutely smaller 
                               ! than "soft" are ignored in P-RFO
  integer ,save        :: follow ! Type of mode following:
                          ! 0: no mode following: TS mode has the lowest eigenvalue
                          ! 1: specify direction by input - not yet implemented
                          ! 2: determine direction at first P-RFO step
                          ! 3: update direction at each P-RFO step
  logical ,save        :: tsvectorset ! is tsverctor defined?
  integer ,save        :: tsmode ! number of mode to maximise
  logical ,save        :: twopoint ! type of finite difference Hessian
  real(rk),save        :: storeenergy
  logical ,save        :: carthessian ! should the Hessian be updated in 
                                      ! Cartesian coordinates (T) or internals (F)
  real(rk),allocatable,save :: eigvec(:,:)  ! (nihvar,nihvar) Hessian eigenmodes
  real(rk),allocatable,save :: eigval(:)    ! (nihvar) Hessian eigenvalues
  real(rk),allocatable,save :: storegrad(:) ! (nihvar) old gradient in fdhessian
  integer             ,save :: iupd ! actual number of Hessian updates
  real(rk),allocatable,save :: tsvector(:) ! (nihvar) Vector to follow in P-RFO
  ! The old arrays are set in formstep. Used there and in hessian_update
  real(rk),allocatable,save :: oldc(:)      ! (nihvar) old i-coords
  real(rk),allocatable,save :: oldgrad(:)   ! (nihvar) old i-coords
  real(rk)            ,save :: minstep      ! minimum step length for Hessian update to be performed
  integer             ,save :: minsteps     ! minimum number of steps performed in an IRC calculation
                                       ! set in formstep_init
  logical,save              :: fracrec ! recalculate a fraction of the hessian?
end module dlf_hessian_gpr


subroutine gpr_prfo_init(this)
  use dlf_global, only: glob
  use dlf_hessian_gpr
  use gpr_types_module, only: gpr_type
  implicit none
  type(gpr_type), intent(in)    ::  this
  if(.not.allocated(glob%step)) allocate(glob%step(this%sdgf))
  glob%step=0.D0
  if(.not.allocated(eigval)) allocate(eigval(this%sdgf))
  if(.not.allocated(eigvec)) allocate(eigvec(this%sdgf,this%sdgf))
  if(.not.allocated(tsvector)) allocate(tsvector(this%sdgf))
  follow=0  ! get from global eventually - IMPROVE
  tsmode=1
  soft=1d-3 ! Modes below that value are ignored in P-RFO

! P-RFO
   
    tsvector(:)=0.D0
    tsvectorset=.false.
end subroutine gpr_prfo_init

subroutine gpr_prfo_destroy(this)
  use dlf_global, only: glob
  use dlf_hessian_gpr
  use gpr_types_module, only: gpr_type
  implicit none
  type(gpr_type), intent(in)    ::  this
  if(allocated(eigval)) deallocate(eigval)
  if(allocated(eigvec)) deallocate(eigvec)
  if(allocated(tsvector)) deallocate(tsvector)
end subroutine gpr_prfo_destroy

subroutine gpr_bofill_update(nvar, gradient, oldgradient, &
    coords, oldcoords, hess, minstep)
  use dlf_parameter_module, only: rk     
  implicit none
  integer, intent(in)    ::  nvar
  real(rk),intent(in)    ::  gradient(nvar), oldgradient(nvar)
  real(rk)               ::  fvec(nvar)
  real(rk),intent(in)    ::  coords(nvar), oldcoords(nvar)
  real(rk),intent(inout) ::  hess(nvar,nvar)
  real(rk),intent(in)    ::  minstep
  real(rk)               ::  step(nvar), svar, fx_xx, xx, bof
  integer                ::  ivar,jvar
  real(rk) ,external     ::  ddot
  ! Useful variables for updating
  fvec(:) = gradient(:) - oldgradient(:)
  step(:) = coords(:) - oldcoords(:)
  xx=ddot(nvar,step,1,step,1)
  if(xx <= minstep ) then
    write(*,"('Step too small. Skipping hessian update')")
    return
  end if

    ! Powell/Bofill updates

    ! fvec = fvec - hessian x step
    call dlf_matrix_multiply(nvar,1,nvar,-1.D0, hess,step,1.D0,fvec)
!      call DGEMV('N',nvar,nvar,-1d0,hess,nvar,step,1,1d0,fvec,1)

    fx_xx=ddot(nvar,fvec,1,step,1) / xx
        svar=ddot(nvar,fvec,1,fvec,1)
        if(svar==0.D0) then
          write(*,"('Step too small. Skipping hessian update')")
          return
        end if
        bof=fx_xx**2 * xx / svar
        write(*,'("Bof=",es10.3)') bof

     do ivar=1,nvar
        do jvar=ivar,nvar          
           
           ! Powell
           svar=fvec(ivar)*step(jvar) + step(ivar)*fvec(jvar) - &
                fx_xx* step(ivar)*step(jvar)

           ! Bofill
           svar=svar*(1.D0-bof) + bof/fx_xx * fvec(ivar)*fvec(jvar)
           hess(ivar,jvar) = hess(ivar,jvar) + svar/xx
           hess(jvar,ivar) = hess(ivar,jvar)
        end do
     end do
end subroutine


subroutine gpr_hessian_update(nvar, coords, oldcoords, gradient, &
     oldgradient, hess, havehessian, fracrecalc, was_updated,&
     updateCounter,maxupd,minstep)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout, stderr, printl
!   use dlf_hessian
  implicit none
  integer,intent(in)    :: nvar ! used for temporary storage arrays
  real(rk),intent(in) :: coords(nvar)   
  real(rk),intent(in) :: oldcoords(nvar)
  real(rk),intent(in) :: gradient(nvar)
  real(rk),intent(in) :: oldgradient(nvar)
  real(rk),intent(inout) :: hess(nvar,nvar)
  logical,intent(inout) :: havehessian
  logical,intent(inout) :: fracrecalc ! recalculate fraction of Hessian
  logical,intent(out)   :: was_updated ! at return: was the Hessian updated here?
  ! temporary arrays
  real(rk) :: fvec(nvar),step(nvar), tvec(nvar)
  real(rk) :: fx_xx,xx,svar,bof, dds, ddtd
  real(RK) ,external :: ddot
  integer  :: ivar,jvar
  logical,parameter :: do_partial_fd=.false. ! Current main switch!
  integer :: update=2
  integer, intent(in)   ::  maxupd
  integer, intent(inout):: updateCounter
  real(rk), intent(in)  ::  minstep
! **********************************************************************

  was_updated=.false.

!   if(.not.fd_hess_running) fracrecalc=.false.
  if(.not.havehessian.or.update==0) then
    havehessian=.false.
    return
  end if

  ! Check for maximum number of updates reached
  ! In case of partial finite-difference, do an update first, then return and 
  ! Recalculate the lower modes
  if(updateCounter>=maxupd .and. .not. do_partial_fd) then
    havehessian=.false.
    updateCounter=0
    hess = -1.D0
    fracrecalc=.false.
    return
  end if

  ! Useful variables for updating
  fvec(:) = gradient(:) - oldgradient(:)
  step(:) = coords(:) - oldcoords(:)

  xx=ddot(nvar,step,1,step,1)
  if(xx <= minstep ) then
    if(printl>=2) write(stdout,"('Step too small. Skipping hessian update')")
    return
  end if

  updateCounter=updateCounter+1

  if(printl>=3) then
     select case (update)
     case(1)
        write(stdout,"('Updating Hessian with the Powell update, No ',i5)") updateCounter
     case(2)
        write(stdout,"('Updating Hessian with the Bofill update, No ',i5)") updateCounter
     case(3)
        write(stdout,"('Updating Hessian with the BFGS update, No ',i5)") updateCounter
     end select
  end if
    
  select case (update)
  case(1,2)
     ! Powell/Bofill updates

     ! fvec = fvec - hessian x step
     call dlf_matrix_multiply(nvar,1,nvar,-1.D0, hess,step,1.D0,fvec)

     fx_xx=ddot(nvar,fvec,1,step,1) / xx

     if(update==2) then
        svar=ddot(nvar,fvec,1,fvec,1)
        if(svar==0.D0) then
          if(printl>=2) write(stdout,"('Step too small. Skipping hessian update')")
          return
        end if
        bof=fx_xx**2 * xx / svar
        if(printl>=6) write(stdout,'("Bof=",es10.3)') bof
     end if

     do ivar=1,nvar
        do jvar=ivar,nvar

           ! Powell
           svar=fvec(ivar)*step(jvar) + step(ivar)*fvec(jvar) - fx_xx* step(ivar)*step(jvar)

           if(update==2) then
              ! Bofill
              svar=svar*(1.D0-bof) + bof/fx_xx * fvec(ivar)*fvec(jvar)
           end if

           hess(ivar,jvar) = hess(ivar,jvar) + svar/xx
           hess(jvar,ivar) = hess(ivar,jvar)
        end do
     end do
     
  case(3)
     ! BFGS update

     ! tvec is hessian x step
     tvec = 0.0d0
     call dlf_matrix_multiply(nvar, 1, nvar, 1.0d0, hess, step, 0.0d0, tvec)

     dds = ddot(nvar, fvec, 1, step, 1)
     ddtd = ddot(nvar, step, 1, tvec, 1)

     do ivar = 1, nvar
        do jvar = ivar, nvar
           svar = (fvec(ivar) * fvec(jvar)) / dds - &
                  (tvec(ivar) * tvec(jvar)) / ddtd
           hess(ivar, jvar) = hess(ivar, jvar) + svar
           hess(jvar, ivar) = hess(ivar, jvar)
        end do
     end do

  case default
     ! Update mechanism not recognised
     write(stderr,*) "Hessian update", update, "not implemented"
     call dlf_fail("Hessian update error")

  end select

  was_updated=.true.

  ! Check for maximum number of updates reached
  ! In case of partial finite-difference, do an update first, then return and 
  ! Recalculate the lower modes
  if(updateCounter>=maxupd .and. do_partial_fd) then
    havehessian=.false.
    updateCounter=0
    fracrecalc=.true.

  end if

end subroutine gpr_hessian_update

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/Slightly changed version of hessian/dlf_prfo_step of dl-find
!!
!! FUNCTION
!!
!! Calculate a P-RFO step
!! This routine does only require the current Hessian and gradient,
!! not an old one.
!!
!! Notes to the Hessian update:
!! Paul updates some modes of the Hessian, whenever the number of positive 
!! eigenvalues (non-positive is below "soft") changes. He recalculates all
!! negative and soft modes plus one.
!!
!! SYNOPSIS
subroutine gpr_prfo_step(nvar,nzero,mwInt,&
    gradient,hessian,step)
!! SOURCE
  use dlf_global, only: stdout,printl,stderr
  use dlf_hessian_gpr
  implicit none
  
  integer ,intent(in)   :: nvar 
  integer ,intent(in)   :: nzero ! # zero eigenmodes
  logical, intent(in)   :: mwInt ! mass weighted internals?
  real(rk),intent(in)   :: gradient(nvar)
  real(rk),intent(in)   :: hessian(nvar,nvar)
  real(rk),intent(out)  :: step(nvar)
  !
  ! these may be made variable in the future:
  integer   :: maxit=1000 ! maximum number of iterations to find lambda
!  real(rk)  :: tol=1.D-5 ! tolerance in iterations
  real(rk)  :: tol=1.D-6 ! tolerance in iterations
  real(rk)  :: bracket=1.D-4  ! threshold for using bracket scheme - why not 0.D0?
  real(rk)  :: delta=0.05D0, big=1000.D0 ! for bracketing 
  !
  real(rk)              :: lamts,lamu,laml,lam
  real(rk)              :: ug(nvar) ! eigvec * gradient
  !real(rk)              :: tmpvec(nvar)
  integer               :: ivar,iter,nmode,maxsoft
  real(rk)              :: svar,lowev,maxov
  real(rk)              :: soft_tmp,ev2(nvar)
  real(RK) ,external    :: ddot
  logical               :: err=.false.,conv=.false.
  logical               :: skipmode(nvar)
  real(8)               :: lam_thresh
! **********************************************************************
  
  maxsoft=nzero
  ! try:
  if(mwInt) then ! in dl-find it was icoords==190
    tol=1.D-13
    bracket=1.D-6
    delta=1.D-3
    maxit=1000
  end if


  call gpr_matrix_diagonalize(nvar,hessian,eigval,eigvec)
  if(printl >= 5) then
    write(stdout,"('Eigenvalues of the Hessian:')") 
    write(stdout,"(9f10.4)") eigval
  end if

  
  ! Determine mode to follow
  if(follow==0) then
    do ivar=1, nvar
      if (eigval(ivar)<-1d-12) then
        ! eigval is numerically negative
        tsmode=ivar
        exit
      else if (eigval(ivar)<1d-12) then
        ! eigval is numerically zero
        tsmode = ivar
        cycle
      else
        ! eigval is first (smallest) positive eigval
        tsmode = ivar
        exit
      end if
    end do
  else if(follow==1) then
    call dlf_fail("Hessian mode following 1 not implemented")
  else if(follow==2.or.follow==3) then
    if(tsvectorset) then
      maxov= dabs(ddot(nvar,eigvec(:,tsmode),1,tsvector,1))
      nmode=tsmode
      if(printl>=5) write(stdout,"('Overlap of current TS mode with &
          &previous one ',f6.3)") maxov
      do ivar=1,nvar
        if(ivar==tsmode) cycle
        svar= dabs(ddot(nvar,eigvec(:,ivar),1,tsvector,1))
        if(svar > maxov) then
          if(printl>=6) write(stdout,"('Overlap of mode',i4,' with &
              & TS-vector is ',f6.3,', larger than TS-mode',f6.3)") &
              ivar,svar,maxov
          maxov=svar
          nmode=ivar
        end if
      end do
      if(nmode /= tsmode) then
        !mode switching!
        if(printl>=5) write(stdout,"('Switching TS mode from mode',i4,&
            &' to mode',i4)") tsmode,nmode
        tsmode=nmode
        if(printl>=5) write(stdout,"('Overlap of current TS mode with &
            &previous one ',f6.3)") maxov
      end if
      if(follow==3) tsvector=eigvec(:,tsmode)
    else
      ! first step: use vector 1
      tsvector(:)=eigvec(:,1)
      tsvectorset=.true.
    end if
  else
    write(stderr,"('Wrong setting of follow:',i5)") follow
    call dlf_fail("Hessian mode following wrong")
  end if

  ! print frequency in case of mass-weighted coordinates
!   if(massweight  .and.(.not.mwInt) .and.printl>=2) then
    ! sqrt(H/u)/a_B/2/pi/c / 100
    !svar=sqrt( 4.35974417D-18/ 1.66053886D-27 ) / ( 2.D0 * pi * &
    !    0.5291772108D-10 * 299792458.D0) / 100.D0
    !call dlf_constants_get("CM_INV_FOR_AMU",CM_INV_FOR_AMU)
    !svar=sqrt(abs(eigval(tsmode))) * CM_INV_FOR_AMU
    !if(eigval(tsmode)<0.D0) svar=-svar
    !write(stdout,"('Frequency of transition mode',f10.3,' cm^-1 &
    !    &(negative value denotes imaginary frequency)')") &
    !    svar
! ! !     call dlf_print_wavenumber(eigval(tsmode),.true.) 
!   end if

! ! !   call dlf_formstep_set_tsmode(nvar,11,eigvec(:,tsmode))! 

  ! calculate eigvec*gradient
  do ivar=1,nvar
    ug(ivar) = ddot(nvar,eigvec(:,ivar),1,gradient(:),1)
  end do

  ! calculate Lambda that minimises along the TS-mode:
  lamts=0.5D0 * ( eigval(tsmode) + dsqrt( eigval(tsmode)**2 + 4.D0 * ug(tsmode)**2) )

  if(printl >= 5) then
    write(stdout,'("Lambda for maximising TS mode:     ",es12.4," Eigenvalue:",es12.4)') lamts,eigval(tsmode)
  end if

  ! Calculate the number of modes considered "soft"
  
  nmode=0
  soft_tmp=soft
  ev2=0.D0
  do ivar=1,nvar
    if(ivar==tsmode) cycle
    if(abs(eigval(ivar)) < soft ) then
      nmode=nmode+1
      ev2(ivar)=dabs(eigval(ivar))
    end if
  end do

  ! Check that at most 6 modes are considered "soft"
  if(nmode>maxsoft) then
    do ivar=nmode-1,maxsoft,-1
      soft_tmp=maxval(ev2)
      ev2(maxloc(ev2))=0.D0
    end do
    if(printl>=5) write(stdout,'("Criterion for soft modes tightened to &
        &",es12.4)') soft_tmp
    ! recalculate nmode
    nmode=0
    do ivar=1,nvar
      if(ivar==tsmode) cycle
      if(abs(eigval(ivar)) < soft_tmp ) then
        nmode=nmode+1
        if(printl>=5) write(stdout,'("Mode ",i4," considered soft")') ivar
      end if
    end do
  end if

  if(nmode>0.and.printl>=5) &
      write(stdout,'("Ignoring ",i3," soft modes")') nmode

  ! find lowest eigenvalue that is not TS-mode and not soft
  !   i.e. the lowest eigenmode that is minimised
  do ivar=1,nvar
    if(ivar==tsmode) cycle
    if(abs(eigval(ivar)) < soft_tmp ) cycle
    lowev=eigval(ivar)
    exit
  end do

  ! define skipmode
  skipmode(:)=.true.
  do ivar=1,nvar
    if(ivar==tsmode) cycle
    if(abs(eigval(ivar)) < soft_tmp ) cycle
    ! instead of the above line: modes 2-7 soft
    !  if(ivar>=2.and.ivar<=7) cycle
    !  print*,"Modes 2 to 7 soft"
    !<<<<
    skipmode(ivar)=.false.
  end do

  lamu=0.D0
  laml=0.D0
  lam=0.D0
  if(lowev < bracket) then
    lam=lowev-delta
    lamu=lowev
    laml=-big
  end if

  do iter=1,maxit

    svar=0.D0

    do ivar=1,nvar
      if(ivar==tsmode) cycle
      !if(abs(eigval(ivar)) < soft_tmp ) cycle
      if(skipmode(ivar)) cycle
      if(abs(lam - eigval(ivar)) > 1.D-14) then
        svar=svar+ ug(ivar)**2 / (lam - eigval(ivar) )
      end if
    end do

    if(abs(svar-lam) < tol) then
      ! we are converged

      if(lam>lowev.and.printl>=2) then
        write(stderr,'("Lambda > lowest non-TS eigenvalue, bad Hessian?")')
        err=.true.
      end if

      if(lam>0.D0 .and. lowev>0.D0) then
        write(stdout,'("Lambda and lowest non-TS eigenvalue >0. Bad Hessian?")')
        write(stdout,'("Lambda:", ES11.4)') lam
        !err=.true.
      end if

      if(.not.err.and.printl>=6) then
        write(stdout,'("Lambda converged in",I8,"iterations")') iter
      end if
      
      conv=.true.

      exit

    end if

    ! we are not converged. Next iteration:

    !write(*,'("A",4f15.8)') svar,lam,lamu,laml
    if(lowev < bracket ) then
      if(svar < lam) lamu=lam
      if(svar > lam) laml=lam
      if(laml > -big) then
        lam=0.5D0 * (lamu + laml)
      else
        lam = lam-delta
      end if
    else
      lam=svar
    end if
    !write(*,'("B",4f15.8)') svar,lam,lamu,laml

  end do

  if(.not.conv.and.printl>=4) write(stdout,*) "Warning: P-RFO loop not converged"
  if(err) call dlf_fail("P-RFO error")

  if(printl >= 5) &
      write(stdout,'("Lambda for minimising other modes: ",es12.4)') lam

  ! calculate step:
  step=0.D0
  do ivar=1,nvar
    if(ivar==tsmode) then
      if( abs(lamts-eigval(ivar)) < 1.D-5 ) then
        ug(ivar)=1.D0
      else
        ug(ivar)=ug(ivar) / (eigval(ivar) - lamts)
      end if
    else
      !if(abs(eigval(ivar)) < soft_tmp ) then
      if(skipmode(ivar) ) then
        if(printl>=5) write(stdout,'("Mode ",i4," ignored, as &
            &|eigenvalue| ",es10.3," < soft =",es10.3)') &
            ivar,eigval(ivar),soft_tmp
        cycle
      end if
      if(mwInt) then
        lam_thresh=1.D-10
      else
        lam_thresh=1.D-5
      end if
      if( abs(lam-eigval(ivar)) < lam_thresh ) then 
        write(stdout,'("WARNING: lam-eigval(ivar) small for non-TS mode",I8,&
              "!")') ivar
        ug(ivar) = -1.D0 / eigval(ivar) ! take a newton-raphson step for this one
      else
        ug(ivar) = ug(ivar) / (eigval(ivar) - lam)
      end if
    end if
    if(printl>=6) write(stdout,'("Mode ",i4," Length ",es10.3)') ivar,ug(ivar)
    step(:) = step(:) - ug(ivar) * eigvec(:,ivar)
  end do

  if(printl >= 5) &
      write(stdout,'("P-RFO on GPR step length: ",es12.4)') sqrt(sum(step(:)**2))

end subroutine gpr_prfo_step
!!****

! Copied from dl-find: needed for diagonalizing matrices
MODULE gprhdlc_matrixlib
!  USE global
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout
  IMPLICIT NONE
  contains
  
  INTEGER FUNCTION array_diagonalize(a,evect,&
            evalues,n,nval,nvect,increasing)
! args
    LOGICAL increasing
    INTEGER n, nval, nvect
    REAL (rk), DIMENSION (n,n) :: a
    REAL (rk), DIMENSION (nval) :: evalues
    REAL (rk), DIMENSION (n,nvect) :: evect

! externals
    INTEGER, EXTERNAL :: ilaenv
    REAL (rk), EXTERNAL :: dlamch

! local vars
    CHARACTER jobz
    INTEGER i, ierr, il, iu, lwork, nfound
    INTEGER :: nb, nb1, nb2
    INTEGER, DIMENSION (:), ALLOCATABLE :: ifail, iwork
    REAL (rk) :: abstol, dummy, vwork(nval)
    REAL (rk), DIMENSION (:), ALLOCATABLE :: evalwork, work
    REAL (rk), DIMENSION (:,:), ALLOCATABLE :: awork, evecwork

! begin

! copy the matrix to a work array as it would be destroyed otherwise
    allocate (awork(n,n))
    awork = a

! full diagonalisation is required
    IF (nval==n) THEN
!       lwork = 3*n ! this is rather a minimum: unblocked algorithm
      nb = ilaenv(1,'dsytrd','L',n,-1,-1,-1)
      IF (nb<0) THEN
        WRITE (stdout,'(A,I5)') &
          'Matrix diagonaliser: ILAENV for dsytrd failed: returned ', nb
      END IF
      lwork = MAX(n*(nb+2),1 + 6*n + 2*n**2)
      allocate (work(lwork))

      IF (nvect==0) THEN
        jobz = 'N'
      ELSE
        jobz = 'V'
      END IF

! ! diagonaliser DSYEV and error check
!       CALL dsyev(jobz,'L',n,awork,n,evalues,work,lwork,ierr)
!       IF (ierr/=0) THEN
!         WRITE (stdout,'(A,I5)') 'Matrix diagonaliser DSYEV failed: returned ', &
!           ierr
!       END IF
! diagonaliser DSYEVD and error check (alex: changed that to dsyevd)
      allocate(iwork(3+5*n))
      CALL dsyevd(jobz,'L',n,awork,n,evalues,work,lwork,iwork,3+5*n,ierr)
      IF (ierr/=0) THEN
        WRITE (stdout,'(A,I5)') 'Matrix diagonaliser DSYEV failed: returned ', &
          ierr
      END IF      
      deallocate(iwork)

! clean up
      deallocate (work)
!        if (increasing) then
!           do i = 1,nvect
!              do j = 1,n
!                 evect(j,i) = awork(j,i)
!              end do
!           end do
!        else
!           do i = 1,nvect
!              do j = 1,n
!                 evect(j,i) = awork(j,n-i+1)
!              end do
!           end do
!           k = n
!           do i = 1,n/2
!              vwork = evalues(i)
!              evalues(i) = evalues(k)
!              evalues(k) = vwork
!              k = k - 1
!           end do
!        end if

      IF (increasing) THEN
        evect = awork(:,1:nvect)
      ELSE
        evect = awork(:,n:n-nvect+1:-1)
        vwork = evalues
        evalues = vwork(nval:1:-1)
      END IF


! partial diagonalisation is required
    ELSE
!       lwork = 8*n ! this is rather a minimum: unblocked algorithm
      nb1 = ilaenv(1,'dsytrd','L',n,-1,-1,-1)
      IF (nb1<0 ) THEN
        WRITE (stdout,'(A,I5)') &
          'Matrix diagonaliser: ILAENV for dsytrd failed: returned ', nb1
      END IF

      nb2 = ilaenv(1,'dormtr','LLN',n,n,-1,-1)
      IF (nb2<0) THEN
        WRITE (stdout,'(A,I5)') &
          'Matrix diagonaliser: ILAENV for dormtr failed: returned ', nb2
      END IF

      nb = max(nb1,nb2)
      lwork = n*(nb+3)

      abstol = 2.0D0*dlamch('S') ! this is for maximum accuracy
      allocate (work(lwork))
      allocate (iwork(5*n))
      allocate (evalwork(n))
      allocate (evecwork(n,nval)) ! note that this may be larger than nvect
      allocate (ifail(n))
      IF (nvect==0) THEN
        jobz = 'N'
      ELSE
        jobz = 'V'
      END IF
      IF (increasing) THEN
        il = 1
        iu = nval
      ELSE
        il = n - nval + 1
        iu = n
      END IF

! diagonaliser DSYEVX and error check
      CALL dsyevx(jobz,'I','L',n,awork,n,dummy,dummy,il,iu,abstol,nfound, &
        evalwork,evecwork,n,work,lwork,iwork,ifail,ierr)
      IF (ierr/=0) THEN
        WRITE (stdout,'(A,I5)') 'Matrix diagonaliser DSYEVX failed: returned ' &
          , ierr
!         IF (printl>=5) THEN
          WRITE (stdout,'(A)') 'Detailed error message (IFAIL):'
          WRITE (stdout,'(16I5)') (ifail(i),i=1,n)
!         END IF
      END IF

! clean up
      deallocate (iwork)
      deallocate (work)
!        if (increasing) then
!           do i = 1,nvect
!              do j = 1,n
!                 evect(j,i) = evecwork(j,i)
!              end do
!           end do
!           do i = 1,nval
!              evalues(i) = evalwork(i)
!           end do
!        else
!           do i = 1,nvect
!              do j = 1,n
!                 evect(j,i) = evecwork(j,nval-i+1)
!              end do
!           end do
!           do i = 1,nval
!              evalues(i) = evalwork(nval-i+1)
!           end do
!        end if

      IF (increasing) THEN
        evect = evecwork(:,1:nvect)
        evalues = evalwork(1:nval)
      ELSE
        evect = evecwork(:,nval:nval-nvect+1:-1)
        evalues = evalwork(nval:1:-1)
      END IF

      deallocate (ifail)
      deallocate (evecwork)
      deallocate (evalwork)
    END IF

! clear working space
    deallocate (awork)
    array_diagonalize = ierr
    RETURN
  END FUNCTION array_diagonalize
END MODULE gprhdlc_matrixlib


subroutine gpr_matrix_diagonalize(N,A,evals,evecs)
!! SOURCE
  use dlf_parameter_module, only: rk
  use gprhdlc_matrixlib, only: array_diagonalize
  implicit none
  integer  ,intent(in)    :: N
  real(rk) ,intent(in)    :: A(N,N)
  real(rk) ,intent(out)   :: evals(N)
  real(rk) ,intent(out)   :: evecs(N,N)
  integer :: idum
! **********************************************************************
  idum = array_diagonalize(A,evecs,evals,n,n,n,.true.)
end subroutine gpr_matrix_diagonalize
  
! read parameters from file gpr.in
!
! gpr.in - a simplistic and hopefully temporary input file for the
! GPR module of DL-FIND.
! 
! Syntax:
! Any characters after a hash sign (#) are ignored
! Input lines in the following order:
!  1 energy offset (Hartree) 
!  2 length scale of the covariance function (Bohr)
!  3 noise in the energy (standard deviation)
!  4 noise in the gradient (standard deviation)
!  5 noise in the Hessian (standard deviation)
! Any lines after that input are ignored as well
!
subroutine gpr_read_input(offset,length_scale,input_noise)
  use dlf_parameter_module
  use dlf_global, only: stdout, stderr, printl
  implicit none
  real(rk),intent(inout) :: offset
  real(rk),intent(inout) :: length_scale
  real(rk),intent(inout) :: input_noise(3)
  !
  character(128) :: filename,line
!   character(2) :: str2
  integer :: count,ios!,iat,nat
  logical :: tchk
!   real(rk) :: ang

  filename="gpr.in"
  inquire(file=filename,exist=tchk)

  if(.not.tchk) then
    if (printl>=4) then
      write(stdout,'("No file gpr.in was found, leaving GPR parameters as is!")')
      write(stdout,'("leaving GPR parameters as is!")')
    end if
    return
  end if

  if (printl>=4) write(stdout,'(" ")')
  if (printl>=4) write(stdout,'("Reading gpr.in")')
  open(unit=23,file=filename)
  count=0
  do
    READ (23,'(a)',end=203) line
    if(index(line,"#")>0) line=line(1:index(line,"#")-1)
    count=count+1
    select case (count)
    case (1)
      read(line,*,iostat=ios) offset
      if(ios==0) then
        if (printl>=4) write(stdout,'("energy offset is set to ", &
                ES11.4)') offset
      else
        if (printl>=4) write(stdout,'("energy offset NOT read, remains ",&
                ES11.4)') offset
      end if
    case (2)
      read(line,*,iostat=ios) length_scale
      if(ios==0) then
        if (printl>=4) write(stdout,&
            '("GPR covariance length scale is set to ", ES11.4)') length_scale
      else
        if (printl>=4) write(stdout,'(&
            "GPR covariance length scale NOT read, remains ")'),length_scale
      end if
    case (3)
      read(line,*,iostat=ios) input_noise(1)
      if(ios==0) then
        if (printl>=4) write(stdout,'(&
            "Assumed noise in energy is set to ", ES11.4)') input_noise(1)
      else
        if (printl>=4) write(stdout,'(&
            "Assumed noise in energy NOT read, remains ", ES11.4)') &
            input_noise(1)
      end if
    case (4)
      read(line,*,iostat=ios) input_noise(2)
      if(ios==0) then
        if (printl>=4) write(stdout,'(&
            "Assumed noise in gradient is set to ", ES11.4)') input_noise(2)
      else
        if (printl>=4) write(stdout,'(&
            "Assumed noise in gradient NOT read, remains ", ES11.4)') &
            input_noise(2)
      end if
    case (5)
      read(line,*,iostat=ios) input_noise(3)
      if(ios==0) then
        if (printl>=4) write(stdout,'(&
            "Assumed noise in Hessian is set to ", ES11.4)') input_noise(3)
      else
        if (printl>=4) write(stdout,'(&
            "Assumed noise in Hessian NOT read, remains ", ES11.4)') input_noise(3)
      end if
    end select
  end do
203 continue
  close(23)

  if (printl>=4) write(stdout,'("Finished reading gpr.in")') !offset
  if (printl>=4) write(stdout,'(" ")') !offset

end subroutine gpr_read_input

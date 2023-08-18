! This file contains subroutines that is not directly related to gpr
! or the optimizers based on gpr. The subroutines in this file
! are needed for interoperability to chemshell/dl-find.
! For example input/output and some coordinate transformation stuff.

! initialize the PES interpolation: read in all files, store
! control point information (allocate arrays)

module gpr_in_dl_find_mod
    use gpr_types_module
    use gpr_opt_module
    type(gpr_type),save          ::  gpr_pes
    type(optimizer_type),save    ::  gpr_opt
end module gpr_in_dl_find_mod

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* gpr/gpr
!!
!! FUNCTION
!! Main development subroutine.
!! It is mainly used to test new things. One should normally use
!! GPR_construct, GPR_init, GPR_interpolation and then
!! GPR_eval/GPR_eval_grad/GPR_eval_hess to do GPR interpolation
!! in external code.
!! Despite that this routine can be used to do GPR interpolation
!! and then writing a gpr datas file out of the neuralnetwork program.
!!
!! SYNOPSIS
subroutine gpr(this, xs, es, gs, hs, nat, ni, ns, ntest, xtest, chk, etest, &
    gtest, htest,align_refcoords, align_modes, refmass, massweight)  
!! SOURCE    
  use gpr_types_module
  use gpr_module
  use gpr_opt_module
  implicit none
  type(gpr_type),intent(inout)  ::  this
  type(gpr_type)                ::  this_iter
  type(optimizer_type)::opt
  integer, intent(in) ::    ni,&                ! # spatial dgf
                            ns,&                ! # control points
                            ntest,&             ! # testpoints
                            chk,&               ! # choosing kernel type
                            nat                 ! # atoms
  real(rk), intent(in) ::   xs(ni,ns),&         ! coordinates
                            xtest(ni,ntest),&   ! test-coordinates
                            es(ns),&            ! energies
                            etest(ntest),&      ! test-energies
                            gtest(ni,ntest),&   ! test gradients
                            htest(ni,ni,ntest),&! test hessians
                            gs(ni,ns),&         ! gradients
                            hs(ni,ni,ns)        ! hessians
#ifdef dimerMB
  real(rk),allocatable  ::  all_xs(:,:)
  real(rk),allocatable  ::  all_es(:)
  real(rk),allocatable  ::  all_gs(:,:)
  real(rk),allocatable  ::  all_hs(:,:,:)
#endif
  logical, intent(in)   ::  massweight
  real(rk)              ::  align_refcoords(3*nat)
  real(rk)              ::  align_modes(3*nat,ni)
  real(rk)              ::  refmass(nat)
  real(rk)              ::  gamma, s_f, s_n(3)  
  real(rk)              ::  mindist(ns)
  real(rk)              ::  rErr
  real(rk), allocatable ::  testhess(:,:), testhess2(:,:), &
                            testgrad(:), testgrad2(:)
  integer   :: i
  integer   :: order
  real(rk)  :: tmp, tmp2, tmp3, tmp4, stretch, tmpsum
  real(rk)  :: rmsd, mae
  real(rk)  :: variance_plot_factor
  real(rk), allocatable::  example_xs(:,:), example_es(:), example_gs(:,:),&
                            example_hs(:,:,:)!, example_xsPaper(:,:)
  integer              ::  nt, j, iter, l, m, n
  real(rk)             ::  MSE, tmpxs(2), tmpx(1)
  logical              ::  sym, possemdef, posdef
  real(rk)              ::  step(2), oldpos(2), newpos(2)
  real(rk)              ::  maxstepsize, stepl
  integer               ::  stepnr
        logical                 ::  usingHess       ! Hess, grads and energies
        logical                 ::  usingGrad       ! Grads and energies
        logical                 ::  usingEnergies   ! Only energies
  type(gpr_iteration_manager)   ::  iter_man
  real(rk)                      ::  variance
  real(rk)                      ::  likelihood,parasLike(1)!, gradsLike(1)
  print*, "nat", nat
  print*, "****************STARTING GPR INTERPOLATION************************"
  print*, "T! sdgf", ni
  this%sdgf=ni   
  this%kernel_type = chk
  usingHess     = .true.
  usingGrad     = .true.
  usingEnergies = .true.
  call constructTrafoForDL_Find(this, nat, ni)
  call storeTrafoForDL_Find(this, nat, ni, &
        align_refcoords,align_modes,refmass, massweight)
  write(*,3335) "TPlot  ", "o", "Sigma(1)", "Sigma(2)","Sigma(3)","Gamma","Tr_RMSD", "Tr_MAE", "Te_RMSD", "Te_MAE"
  3335 FORMAT (A, A1, A12, A12, A12, A12,A12,A12,A12,A12)
  
#ifdef dimerMB
    call gpr_driver_init
    allocate(all_xs(2,1))
    allocate(all_es(1))
    allocate(all_gs(2,1))
    ! Plot MB
    stretch = 1.0d0
    do i = 1, 50
      do j = 1, 50
        all_xs(1,1) = &
             -1d0*stretch+i*(2d0*stretch/50d0)
        all_xs(2,1) = &
             -1d0*stretch+j*(2d0*stretch/50d0)
        call gpr_dlf_get_gradient(2,all_xs(:,1),all_es(1),all_gs(:,1))
        print*, "Plot_MB", all_xs(:,1), all_es(1)
      end do
    end do 
    
    call gpr_driver_init
    call GPR_construct(this,0,1,2,6,1,1)
    allocate(example_xs(2,1))
    call GPR_init_without_tp(this, 1d0/20d0, 1d0, (/1d-7,1d-7,1d-7/)) 
    call GPR_Optimizer_define('TS', opt,this,60,10, 0.2d0)
    all_xs(1,1)=0.5d0
    all_xs(2,1)=0.0d0
    call gpr_dlf_get_gradient(2,all_xs,all_es(1),all_gs(:,1))

    do while (dsqrt(dot_product(all_gs(:,1),all_gs(:,1)))/this%sdgf>1d0/3d0*1d-6.or.&
                maxval(abs(all_gs(:,1)))>5d-7.or.this%nt<20)
      print*, "Plot_optPts1", this%nt, all_xs(:,1), all_es(1)

      call GPR_Optimizer_step(this,opt,all_xs(:,1),example_xs(:,1),&
                all_xs(:,1),all_es(1),all_gs(:,1))
      all_xs(:,1)=example_xs(:,1)
      print*, "all_xs", all_xs(:,1)
      call gpr_dlf_get_gradient(2,all_xs,all_es(1),all_gs(:,1))
      print*, "Convergence", dsqrt(dot_product(all_gs(:,1),all_gs(:,1)))/this%sdgf,&
        1d0/3d0*1d-4,maxval(all_gs),5d-5
    end do

    ! Plot GPR
    stretch = 1.0d0
    do i = 1, 50
      do j = 1, 50
        all_xs(1,1) = &
             -1d0*stretch+i*(2d0*stretch/50d0)
        all_xs(2,1) = &
             -1d0*stretch+j*(2d0*stretch/50d0)
        call gpr_eval(this,all_xs(:,1),all_es(1))
        print*, "Plot_GPR", all_xs(:,1), all_es(1)
      end do
    end do    
    print*, "T! number of used training points:", this%nt
#endif
  
#ifdef Test1D  
!****************************************************************************
  this%iChol = .true.
  ! 1D example
  variance_plot_factor = 2d0
  s_f = 1d0
  ! for coords
  s_n(1) = 1d-7!1d-3!5d-1 !0d0 for exact fit at training data  ! for gradients
  s_n(2) = 1d-4!0d0!5d-1 !0d0 for exact fit at training data
  ! for hessians
  s_n(3) = 1d-3!1d-3!.1d-4!.05d0!5d-1 !0d0 for exact fit at training data  
  gamma=1d0/(1d0)**2!1d0/(1d1)**2!1d0/(5d0)**2!1!0.4!0.323d0  
  !x**2*sin(x)+x**2x*cos(x)+xy
  this%sdgf=1
  nt = 4
  
  allocate(example_xs(this%sdgf,nt)) ! example training coordinates
  allocate(example_es(nt))
  allocate(example_gs(this%sdgf,nt)) ! example training gradients
  allocate(example_hs(this%sdgf,this%sdgf,nt))
  allocate(testhess(this%sdgf, this%sdgf))
  allocate(testhess2(this%sdgf, this%sdgf))
  allocate(testgrad(this%sdgf))
  allocate(testgrad2(this%sdgf))
  
  example_xs(1,1) = -3.8d0!-5.0d0!-4.2
  example_xs(1,2) = -2.6d0!1.69D0!-4.0d0
  example_xs(1,3) = -1D0!-3d08d0
  example_xs(1,4) = 2.5d0
!   example_xs(1,5) = -0.5D0
!   example_xs(1,6) = 2d0
!   example_xs(1,7) = 3.5d0
!   example_xs(1,8) = 5d0
!   example_xs(1,1) = 1.0d0!-4.2
!   example_xs(1,2) = 1.1d0!1.69D0!-4.0d0
!   example_xs(1,3) = 1.25D0!-3d08d0
!   example_xs(1,4) = 1.5d0
!   example_xs(1,5) = 2.0D0
!   example_xs(1,6) = 2.5d0
!   example_xs(1,7) = 3.5d0
!   example_xs(1,8) = 4.5d0 

  do i = 1, nt
    example_es(i)     = exf( example_xs(1,i))
    example_gs(1,i)   = dexf( example_xs(1,i)) 
    example_hs(1,1,i) = ddexf( example_xs(1,i)) 
  end do
  
  
  iter = 10000  
  stretch = 1.0d0
  do j = 1, iter
        tmpx(1)=-4d0*stretch+(j-1)*(8d0*stretch)/(iter-1)  
!         tmpx(1)=0.2d0+(j-1)*(5.8d0/iter)
        tmp2 =  exf(tmpx(1))
        print*, "Plot fct", tmpx(1), tmp2
  end do 
if (usingEnergies) then  
  tmp = 0
 order = 0  
!  gamma=1d0/(0.5d0)**2!1d0/(1d1)**2!1d0/(5d0)**2!1!0.4!0.323d0  
!   call GPR_construct(this,0,nat,1,1,order)
  call GPR_construct(this,nt,nat,1,0,0,order) !SE
  print*, "xs", example_xs
  print*, "es", example_es
  call GPR_init(this, example_xs, gamma, s_f, s_n, example_es)

!   this%gamma=gamma
!   this%s_f=s_f
!   this%s_n=s_n
!   this%K_stat=-1
!   this%initialized=.true.
!   this%scaling_set = .false.
  
!   call GPR_add_tp(this,1,example_xs(1,1),example_es(1))
  print*, "CONTROL", example_xs(1,1), example_es(1)
!   call GPR_add_tp(this,1,example_xs(1,2),example_es(2))
  print*, "CONTROL", example_xs(1,2), example_es(2)
!   call GPR_add_tp(this,1,example_xs(1,3),example_es(3))
  print*, "CONTROL", example_xs(1,3), example_es(3)
!   call GPR_add_tp(this,1,example_xs(1,4),example_es(4))
  print*, "CONTROL", example_xs(1,4), example_es(4)
! !   call GPR_add_tp(this,1,example_xs(1,5),example_es(5))
!   print*, "CONTROL", example_xs(1,5), example_es(5)
! !   call GPR_add_tp(this,1,example_xs(1,6),example_es(6))
!   print*, "CONTROL", example_xs(1,6), example_es(6)
! !   call GPR_add_tp(this,1,example_xs(1,7),example_es(7))
!   print*, "CONTROL", example_xs(1,7), example_es(7)
! !   call GPR_add_tp(this,1,example_xs(1,8),example_es(8))
!   print*, "CONTROL", example_xs(1,8), example_es(8)
!   call GPR_newLevel(this,4)
  call GPR_interpolation(this) 
  MSE = 0d0 
     do j = 1, iter
        tmpx(1)=-4d0*stretch+(j-1)*(8d0*stretch)/(iter-1)
        print*, "tmpx", tmpx(1)
!         tmpx(1)=0.2d0+(j-1)*(5.8d0/iter)
        call GPR_eval(this,tmpx, tmp)
        print*, "Plot_order_0", tmpx(1), tmp
        call GPR_variance(this,tmpx(1),tmp2)
        print*, "Variance", tmpx(1), tmp2
        print*, "PlotV+_order_0", tmpx(1), tmp+variance_plot_factor*dsqrt(tmp2)
        print*, "PlotV-_order_0", tmpx(1), tmp-variance_plot_factor*dsqrt(tmp2)
        tmp2 =  exf(tmpx(1))
        MSE = MSE + (tmp2-tmp)**2
    end do  
 MSE = MSE/(iter)
 print*, "T! MSE ", order, MSE

 call GPR_destroy(this)
end if

if (usingGrad) then 
!  ! Now including gradients
  this%iChol = .false.
  tmp = 0
  order = 1  
!   call GPR_construct(this,0,nat,1,1,1,order)
!   call GPR_construct(this,0,nat,1,1,order)
  call GPR_construct(this,nt,nat,1,0,1,order)
!   call manual_offset(this,0.1d0)
  call GPR_init(this, example_xs, gamma, s_f, s_n, example_es,example_gs)
  this%iChol = .false.
!   this%gamma=gamma
!   this%s_f=s_f
!   this%s_f2=this%s_f**2
!   this%s_n=s_n
!   this%K_stat=-1
!   this%initialized=.true.
!   this%scaling_set = .false.
  
!   call GPR_add_tp(this,1,example_xs(1,1),example_es(1),example_gs(1,1))
!   print*, "CONTROL", example_xs(1,1), example_es(1)
!   call GPR_add_tp(this,1,example_xs(1,2),example_es(2),example_gs(1,2))
!   print*, "CONTROL", example_xs(1,2), example_es(2)
!   call GPR_add_tp(this,1,example_xs(1,3),example_es(3),example_gs(1,3))
!   print*, "CONTROL", example_xs(1,3), example_es(3)
!   call GPR_add_tp(this,1,example_xs(1,4),example_es(4),example_gs(1,4))
!   print*, "CONTROL", example_xs(1,4), example_es(4)
! !   call GPR_newLevel(this,1)
!   call GPR_add_tp(this,1,example_xs(1,5),example_es(5),example_gs(1,5))
!   print*, "CONTROL", example_xs(1,5), example_es(5)
!   call GPR_add_tp(this,1,example_xs(1,6),example_es(6),example_gs(1,6))
!   print*, "CONTROL", example_xs(1,6), example_es(6)
!   call GPR_add_tp(this,1,example_xs(1,7),example_es(7),example_gs(1,7))
!   print*, "CONTROL", example_xs(1,7), example_es(7)
!   call GPR_add_tp(this,1,example_xs(1,8),example_es(8),example_gs(1,8))
!   print*, "CONTROL", example_xs(1,8), example_es(8)
!   call GPR_newLevel(this,2)
!   call GPR_newLevel(this,2)
!   call GPR_newLevel(this,2)
!   call GPR_set_up_mlit(iter_man,this,1d-15,1d-15,1d-9,40)
! !   print*, "T! data", this%lowerLevelGPR%OffsetType
! !   call GPR_opt_hPara_maxL(this,3, (/2,3,4/))
  call GPR_interpolation(this) 
!   
  MSE = 0d0 
     do j = 1, iter
        tmpx(1)=-4d0*stretch+(j-1)*(8d0*stretch)/(iter-1)
!         tmpx(1)=0.2d0+(j-1)*(5.8d0/iter)
        call GPR_eval(this,tmpx, tmp)
        print*, "Plot_order_1", tmpx(1), tmp
        call GPR_variance(this,tmpx(1),tmp2)
        print*, "Variance", tmpx(1), tmp2
        print*, "PlotV+_order_1", tmpx(1), tmp+variance_plot_factor*dsqrt(tmp2)
        print*, "PlotV-_order_1", tmpx(1), tmp-variance_plot_factor*dsqrt(tmp2)
        tmp2 =  exf(tmpx(1))
        MSE = MSE + (tmp2-tmp)**2
    end do  

 MSE = MSE/(iter)
 print*, "T! MSE ", order, MSE
 call GPR_destroy(this)
end if 
 
if (usingHess) then
  ! Now including Hessians as well
  this%iChol = .false.
  tmp = 0
  order = 2  
!   call GPR_construct(this,0,nat,1,0,order)
  call GPR_construct(this,nt,nat,1,0,0,order)
  call GPR_init(this, example_xs, gamma, s_f, s_n, example_es,example_gs,example_hs)
  this%iChol = .false.
  !************
!   example_xs(1,1) = 1d0
!   do i = 1, 1000
!     example_xs(1,2) = 1d0+i*(4d0/1000d0)
!     tmp=kernel_d4(this,example_xs(1,1),example_xs(1,2),(/2,2,1,1/),1,1,1,1)
!     tmp=kernel_d3(this,example_xs(1,1),example_xs(1,2),(/2,2,1/),1,1,1)
!     tmp=kernel_d2(this,example_xs(1,1),example_xs(1,2),(/2,2/),1,1)
!     tmp=kernel_d1(this,example_xs(1,1),example_xs(1,2),2,1)
!     tmp=kernel(this,example_xs(1,1),example_xs(1,2))
!     print*, "DBK4X",tmp
!   end do
  !***************
  this%gamma=gamma
  this%s_f=s_f
  this%s_f2=this%s_f**2
  this%s_n=s_n
  this%K_stat=-1
!   this%initialized=.true.
!   this%scaling_set = .false.
  
!   call GPR_add_tp(this,1,example_xs(1,1),example_es(1),example_gs(1,1),example_hs(1,1,1))
!   print*, "CONTROL", example_xs(1,1), example_es(1)
!   call GPR_add_tp(this,1,example_xs(1,2),example_es(2),example_gs(1,2),example_hs(1,1,2))
!   print*, "CONTROL", example_xs(1,2), example_es(2)
!   call GPR_add_tp(this,1,example_xs(1,3),example_es(3),example_gs(1,3),example_hs(1,1,3))
!   print*, "CONTROL", example_xs(1,3), example_es(3)
!   call GPR_add_tp(this,1,example_xs(1,4),example_es(4),example_gs(1,4),example_hs(1,1,4))
!   print*, "CONTROL", example_xs(1,4), example_es(4)
!   call GPR_add_tp(this,1,example_xs(1,5),example_es(5),example_gs(1,5),example_hs(1,1,5))
!   print*, "CONTROL", example_xs(1,5), example_es(5)
!   call GPR_add_tp(this,1,example_xs(1,6),example_es(6),example_gs(1,6),example_hs(1,1,6))
!   print*, "CONTROL", example_xs(1,6), example_es(6)
!   call GPR_add_tp(this,1,example_xs(1,7),example_es(7),example_gs(1,7),example_hs(1,1,7))
!   print*, "CONTROL", example_xs(1,7), example_es(7)
!   call GPR_add_tp(this,1,example_xs(1,8),example_es(8),example_gs(1,8),example_hs(1,1,8))
!   print*, "CONTROL", example_xs(1,8), example_es(8)
  call GPR_interpolation(this)  

  MSE = 0d0 
    do j = 1, iter
        tmpx(1)=-4d0*stretch+(j-1)*(8d0*stretch)/(iter-1)
!         tmpx(1)=0.2d0+(j-1)*(5.8d0/iter)
!         call GPR_eval(this,tmpx, newpos(1))
!         call GPR_eval_grad(this,tmpx, newpos(1))
        call GPR_eval(this,tmpx, tmp)
        print*, "Plot_order_2", tmpx(1), tmp
        call GPR_variance(this,tmpx(1),tmp2)
        print*, "PlotV+_order_2", tmpx(1), tmp+variance_plot_factor*dsqrt(tmp2)
        print*, "PlotV-_order_2", tmpx(1), tmp-variance_plot_factor*dsqrt(tmp2)
!         tmp2 =  exf(tmpx(1))
!         MSE = MSE + (tmp2-tmp)**2
!         call GPR_variance(this,tmpx,tmp)
    end do  
 MSE = MSE/(iter)
 print*, "T! MSE ", order, MSE 
 call GPR_destroy(this)
 call dlf_fail("OK")
end if

! *******************************************************************
! Testing order 42!!
! *******************************************************************
!  ! Now including gradients
  tmp = 0
  order = 42
!   call GPR_construct(this,0,nat,1,1,1,order)
!   call GPR_construct(this,0,nat,1,1,order)
  this%iChol = .false.
  call GPR_construct(this,0,nat,1,1,0,order)
!   call GPR_construct(this,0,nat,1,1,1,order)
!   call manual_offset(this,0.1d0)
  call GPR_init_without_tp(this, gamma, s_f, s_n)
  this%iChol = .false.
  this%gamma=gamma
  this%s_f=s_f
  this%s_f2=this%s_f**2
  this%s_n=s_n!(/1d-6,1d-5,1d-4/)!s_n
  this%K_stat=-1
!   this%initialized=.true.
!   this%scaling_set = .false.
  
  call GPR_add_tp(this,1,example_xs(1,1),example_es(1),example_gs(1,1),example_hs(1,1,1))
  print*, "CONTROL", example_xs(1,1), example_es(1)
  call GPR_add_tp(this,1,example_xs(1,2),example_es(2),example_gs(1,2),example_hs(1,1,2))
  print*, "CONTROL", example_xs(1,2), example_es(2)
  call GPR_add_tp(this,1,example_xs(1,3),example_es(3),example_gs(1,3),example_hs(1,1,3))
  print*, "CONTROL", example_xs(1,3), example_es(3)
  call GPR_add_tp(this,1,example_xs(1,4),example_es(4),example_gs(1,4),example_hs(1,1,4))
  print*, "CONTROL", example_xs(1,4), example_es(4)
  call GPR_add_tp(this,1,example_xs(1,5),example_es(5),example_gs(1,5),example_hs(1,1,5))
  print*, "CONTROL", example_xs(1,5), example_es(5)
  call GPR_add_tp(this,1,example_xs(1,6),example_es(6),example_gs(1,6),example_hs(1,1,6))
  print*, "CONTROL", example_xs(1,6), example_es(6)
  call GPR_add_tp(this,1,example_xs(1,7),example_es(7),example_gs(1,7),example_hs(1,1,7))
  print*, "CONTROL", example_xs(1,7), example_es(7)
  call GPR_add_tp(this,1,example_xs(1,8),example_es(8),example_gs(1,8),example_hs(1,1,8))
  print*, "CONTROL", example_xs(1,8), example_es(8)
  call GPR_interpolation(this)  
!   
  MSE = 0d0 
     do j = 1, iter
        tmpx(1)=-4d0*stretch+(j-1)*(8d0*stretch)/(iter-1)
!         tmpx(1)=0.2d0+(j-1)*(5.8d0/iter)
!         call GPR_eval(this,tmpx, tmp)
!         print*, "Plot_order_42", tmpx(1), tmp
!         call GPR_eval(this,tmpx, newpos(1))
!         call GPR_eval_grad(this,tmpx, newpos(1))
        call GPR_eval_hess(this,tmpx, newpos(1))
        print*, "Plot_order_42", tmpx(1), newpos(1)
!         call GPR_variance(this,tmpx(1),tmp2)
!         print*, "Variance", tmpx(1), tmp2
!         print*, "PlotV+_order_42", tmpx(1), tmp+variance_plot_factor*dsqrt(tmp2)
!         print*, "PlotV-_order_42", tmpx(1), tmp-variance_plot_factor*dsqrt(tmp2)
!         tmp2 =  exf(tmpx(1))
!         MSE = MSE + (tmp2-tmp)**2
    end do  

 MSE = MSE/(iter)
 print*, "T! MSE ", order, MSE
 call GPR_destroy(this)
 
#endif

#ifdef Test1D2
  usingGrad=.true.
  variance_plot_factor = 1d0
  s_f = 8d0
  ! for coords
  s_n(1) = 1d-5!1d-3!5d-1 !0d0 for exact fit at training data  ! for gradients
  s_n(2) = 1d-4!0d0!5d-1 !0d0 for exact fit at training data
  ! for hessians
  s_n(3) = 1d-3!1d-3!.1d-4!.05d0!5d-1 !0d0 for exact fit at training data  
  gamma=50d0!1d0/(1d1)**2!1d0/(5d0)**2!1!0.4!0.323d0  
!x**2*sin(x)+x**2x*cos(x)+xy
  ! DONT FORGET TO IMPLEMENT THE MEAN OVER ALL AS BASE
  this%sdgf=1
  nt = 8
  
  allocate(example_xsPaper(this%sdgf,nt)) ! example training coordinates
  allocate(example_es(nt))
  allocate(example_gs(this%sdgf,nt)) ! example training gradients
  allocate(example_hs(this%sdgf,this%sdgf,nt))
  allocate(testhess(this%sdgf, this%sdgf))
  allocate(testhess2(this%sdgf, this%sdgf))
  allocate(testgrad(this%sdgf))
  allocate(testgrad2(this%sdgf))
  
  example_xsPaper(1,1) = 1.38d0!-5.0d0!-4.2
  example_xsPaper(1,2) = 1.28d0!1.69D0!-4.0d0
  example_xsPaper(1,3) = 1.072D0!-3d08d0
  example_xsPaper(1,4) = 1.130d0
  example_xsPaper(1,5) = -0.5D0
  example_xsPaper(1,6) = 2d0
  example_xsPaper(1,7) = 3.5d0
  example_xsPaper(1,8) = 5d0  

  do i = 1, nt
    example_es(i)     = exf( example_xsPaper(1,i))
    example_gs(1,i)   = dexf( example_xsPaper(1,i)) 
    example_hs(1,1,i) = ddexf( example_xsPaper(1,i)) 
  end do  
  
  iter = 10000  
!   stretch = 2.0d0
  do j = 1, iter
        tmpx(1)=-4d0*stretch+(j-1)*(8d0*stretch)/(iter-1)
!         tmpx(1)=-5.2d0+(j-1)*(4.5d0)/(iter-1)  
!         tmpx(1)=0.2d0+(j-1)*(2.0d0/iter)
!         tmpx(1)=0.8d0+(j-1)*(2.2d0/iter)
        tmp2 =  exf(tmpx(1))
        print*, "Plot fct", tmpx(1), tmp2
  end do 

if (usingGrad) then 
!  ! Now including gradients
  tmp = 0
  order = 1  
  call GPR_construct(this,0,nat,1,0,1,order)
  call GPR_init_without_tp(this, gamma, s_f, s_n)

  this%gamma=gamma
  this%s_f=s_f
  this%s_f2=this%s_f**2
  this%s_n=s_n
  this%K_stat=-1
  
  call GPR_add_tp(this,1,example_xsPaper(1,1),example_es(1),example_gs(1,1))
  call GPR_add_tp(this,1,example_xsPaper(1,2),example_es(2),example_gs(1,2))

  do i = 1, this%nt
    print*, "CONTROL", example_xsPaper(1,i), example_es(i)
  end do
!   call GPR_newLevel(this,2)
  call GPR_interpolation(this)
  call GPR_Optimizer_define('MIN',opt,this,3,2,0.5d0, 1d-4,0.5d0,1d-4)
  
  MSE = 0d0 
     do j = 1, iter
        tmpx(1)=-4d0*stretch+(j-1)*(8d0*stretch)/(iter-1)
!         tmpx(1)=-5.2d0+(j-1)*(4.5d0)/(iter-1)  
!         tmpx(1)=0.2d0+(j-1)*(3.0d0/iter)
        call GPR_eval(this,tmpx, tmp)
        print*, "Plot_order_1", tmpx(1), tmp
        call GPR_variance(this,tmpx(1),tmp2)
        print*, "Variance", tmpx(1), tmp2
        print*, "PlotV+_order_1", tmpx(1), tmp+variance_plot_factor*dsqrt(tmp2)
        print*, "PlotV-_order_1", tmpx(1), tmp-variance_plot_factor*dsqrt(tmp2)
        tmp2 =  exf(tmpx(1))
        MSE = MSE + (tmp2-tmp)**2
    end do  
 call GPR_Find_Extreme(this,opt,this%xs(:,this%nt),tmpx(1))
 call GPR_eval(this,tmpx(1),tmp2)
 print*, "MinimumPlot", tmpx(1), tmp2
 if (this%nt==2) then
   print*, "Vect", tmpx(1), tmp2, example_xsPaper(1,3)-tmpx(1), exf(example_xsPaper(1,3))-tmp2
 else
   print*, "Vect", tmpx(1), tmp2, tmpx(1)-tmpx(1), exf(tmpx(1))-tmp2
 end if
 print*, "T! MIN", tmpx(1), tmp2
 MSE = MSE/(iter)
 print*, "T! MSE ", order, MSE
 call GPR_destroy(this)
end if 
#endif

#ifdef Test2D
!*****************************************************************************  
!************ My own example with second derivative information in 2D
  s_f = 1d0
  ! for coords
  s_n(1) = 1d-4!0d0!1d-5!1d-4!5d-1 !0d0 for exact fit at training data
  ! for gradients
  s_n(2) = 1d-4!3d0545d-5!1d-4!0d0!5d-1 !0d0 for exact fit at training data
  ! for hessians
  s_n(3) = 1d-3!1d-4!2.65d-5!.1d-4!.05d0!5d-1 !0d0 for exact fit at training data  
  gamma=1d0/(3d0)**2!0.95183739617534779d0!0.6787d0!1d0!1.244d0!1d0/(1d0)**2!1!0.4!0.323d0  
!x**2*sin(x)+x**2x*cos(x)+xy
  ! DONT FORGET TO IMPLEMENT THE MEAN OVER ALL AS BASE
  s_f = 1d0
  this%sdgf=2 
  nt = 9!3**2!4 !must be squared for 2d example !# points in each dimension
  allocate(example_xs(this%sdgf,nt)) ! example training coordinates
  allocate(example_es(nt))
  allocate(example_gs(this%sdgf,nt)) ! example training gradients
  allocate(example_hs(this%sdgf,this%sdgf,nt))
  allocate(testhess(this%sdgf, this%sdgf))
  allocate(testhess2(this%sdgf, this%sdgf))
  allocate(testgrad(this%sdgf))
  allocate(testgrad2(this%sdgf))
  
  call init_random_seed()
  do j = 1, int(sqrt(real(nt)))
    do k = 1, int(sqrt(real(nt)))

        call RANDOM_NUMBER(example_xs(1,(j-1)*int(sqrt(real(nt)))+k))
        example_xs(1,(j-1)*int(sqrt(real(nt)))+k) = -pi+&
            example_xs(1,(j-1)*int(sqrt(real(nt)))+k)*(2d0*pi)
        call RANDOM_NUMBER(example_xs(2,(j-1)*int(sqrt(real(nt)))+k))
        example_xs(2,(j-1)*int(sqrt(real(nt)))+k) = -pi+&
            example_xs(2,(j-1)*int(sqrt(real(nt)))+k)*(2d0*pi)
    end do
  end do  

  do i = 1, nt
    call exf2d( example_xs(:,i), example_es(i))
    call dexf2d( example_xs(:,i), example_gs(:,i))
    call ddexf2d( example_xs(:,i), example_hs(:,:,i)) 
  end do
  
  do i = 1, nt
    print*, "CONTROL", example_xs(1,i), example_xs(2,i), example_es(i)
  end do

if (usingEnergies) then  
  tmp = 0
  order = 0  
!   call GPR_construct(this,nt,nat,2,chk,order)
  call GPR_construct(this,nt,nat,2,1,1,order)
  call GPR_init(this, example_xs, gamma, s_f, s_n, example_es, example_gs, example_hs)
  call GPR_interpolation(this) 
  iter = 50
  MSE = 0d0 
  stretch = 1d0
     do j = 1, iter
        do k = 1, iter
            tmpxs(1)=-pi*stretch+(j-1)*(2d0*pi*stretch)/(iter-1)
            tmpxs(2)=-pi*stretch+(k-1)*(2d0*pi*stretch)/(iter-1)
            call GPR_eval(this,tmpxs, tmp)
            call exf2d( tmpxs, tmp2)
            MSE = MSE + (tmp2-tmp)**2
        end do
    end do  
 MSE = MSE/(iter**2)
 print*, "T! MSE ", order, MSE

 call GPR_destroy(this)
end if

if (usingGrad) then 
 ! Now including gradients
  tmp = 0
  order = 1  
!   call GPR_construct(this,nt,nat,2,chk,order)
  call GPR_construct(this,nt,nat,2,1,1,order)
  call GPR_init(this, example_xs, gamma, s_f, s_n, example_es, example_gs, example_hs)
  call GPR_interpolation(this) 
  iter = 50
  MSE = 0d0 
  stretch = 1.0d0
     do j = 1, iter
        do k = 1, iter
            tmpxs(1)=-pi*stretch+(j-1)*(2d0*pi*stretch)/(iter-1)
            tmpxs(2)=-pi*stretch+(k-1)*(2d0*pi*stretch)/(iter-1)
            call GPR_eval(this,tmpxs, tmp)
            call exf2d( tmpxs, tmp2)
            MSE = MSE + (tmp2-tmp)**2
        end do
    end do  
 MSE = MSE/(iter**2)
 print*, "T! MSE ", order, MSE

 call GPR_destroy(this)
end if 
 
if (usingHess) then
  ! Now including Hessians as well
  tmp = 0
  order = 2  
  call GPR_construct(this,nt,nat,2,1,chk,order)
!   call GPR_construct(this,nt,nat,2,1,order)
  call GPR_init(this, example_xs, gamma, s_f, s_n, &
                    example_es, example_gs, example_hs)
  call GPR_interpolation(this)  
  print*, "Plot"
  MSE = 0d0 
  iter = 50
  stretch = 1.0d0
       do j = 1, iter
        do k = 1, iter
            tmpxs(1)=-pi*stretch+(j-1)*(2d0*pi*stretch)/(iter-1)
            tmpxs(2)=-pi*stretch+(k-1)*(2d0*pi*stretch)/(iter-1)
            call GPR_eval(this,tmpxs, tmp)
            call exf2d( tmpxs, tmp2)
            MSE = MSE + (tmp2-tmp)**2
            print*, "Plot_gpr", tmpxs(1), tmpxs(2), tmp
            print*, "Plot_fct", tmpxs(1), tmpxs(2), tmp2
!             call GPR_variance(this,tmpxs,tmp)
        end do
    end do  

 MSE = MSE/(iter**2)
 print*, "T! MSE energy", order, MSE

 call GPR_destroy(this)
end if
#endif

#ifdef MuellerBrown  
  call gpr_driver_init
  if (usingGrad) then
    ! Set optimization parameters
    maxstepsize = 2d-1
    ! Set initial parameters
    s_f = 1d0
    s_n(1) = 1d-6
    s_n(2) = 1d-4
    s_n(3) = 3d-4
    gamma=1.0d0
    s_f = 1d0
    
    ! First order optimization start with one "training point"
    order = 1
    nt = 1
    allocate(example_xs(2,nt)) ! example training coordinates
    allocate(example_es(nt))
    allocate(example_gs(2,nt))  
    
!     call GPR_construct(this,nt,nat,2,0,order)
    call GPR_construct(this,nt,0,2,1,1,order)
    print*, "Mueller-Brown"    

    stretch = 1.0d0
    do i = 1, 50
      do j = 1, 50
        newpos(1) = &
             -1d0*stretch+i*(2d0*stretch/50d0)
        newpos(2) = &
             -1d0*stretch+j*(2d0*stretch/50d0)
        call gpr_dlf_get_gradient(2,newpos(:),example_es,example_gs)
        print*, "Plot_MB", newpos, example_es
      end do
    end do
    
    example_xs(1,1) = -0.5d0
    example_xs(2,1) = 0.5d0       
    call gpr_dlf_get_gradient(2,example_xs(:,1),example_es,example_gs) 
    call GPR_init(this, example_xs, gamma, s_f, s_n, &
                    example_es, example_gs)
    call GPR_Optimizer_define("MIN",opt,this,60,10)
    
    call GPR_interpolation(this)
    newpos(:) = example_xs(:,1)
    
    stepnr = 0
    error = 1d0
    
    do while(error>1d-7) 
        call GPR_eval(this,newpos,example_es(1))   
        print*, "OptResult", stepnr, newpos(:), example_es(1)
        
        call gpr_dlf_get_gradient(2,newpos(:),example_es,example_gs) 
        oldpos(:)= newpos(:)
        call GPR_Optimizer_step(this,opt,oldpos,newpos,&
                        oldpos,example_es,example_gs)
        step(:) = newpos(:)-oldpos(:)
        stepl = dsqrt(dot_product(step(:),step(:)))
        if (stepl>maxstepsize) then
            print*, "OptErr limited stepsize"
            step(:) = step(:)/stepl*maxstepsize
        end if
        newpos(:) = oldpos + step(:)
        error = dsqrt(dot_product(step,step))
        print*, "Opt", oldpos,step,newpos
        print*, "OptErr", error
        stepnr = stepnr + 1
!         if (stepnr>5) then
!             call GPR_opt_hPara_maxL(this,1,(/2/))
            call GPR_interpolation(this)
!         end if
    end do
!     call GPR_opt_hPara_maxL(this,1,(/2/))
    call GPR_interpolation(this)
    call GPR_eval(this,newpos,example_es(1))
    print*, "OptResult", stepnr, newpos(:), example_es(1)
    stretch = 1d0
    do i = 1, 50
      do j = 1, 50
        newpos(1) = &
             -1d0*stretch+i*(2d0*stretch/50d0)
        newpos(2) = &
             -1d0*stretch+j*(2d0*stretch/50d0)
        call GPR_eval(this, newpos(:),example_es(1))
        print*, "Plot_GPR", newpos, example_es(1)
      end do
    end do
    
    call GPR_destroy(this)
  end if
#endif

#ifdef ImproveMultiLevel
! ! !*****************************************************************************  
! ! !************ This uses the real input data from neuralnetwork.f90
! ! !************ to start a GPR interpolation. It can be used to write
! ! !************ data in a .dat file for dl-find.

!     do i = 1, ns
!     do j = i+1, ns
!         print*, "Distanz", i, j, dot_product(xs(:,i)-xs(:,j),xs(:,i)-xs(:,j))
!     end do
!     end do

  s_f = 1d0
  ! for coords
  s_n(1) =1d-5!2.437455D-14! 
!   s_n(1) =1d-5! Finding TS and Min For SE
  ! for gradients
  s_n(2) =1d-4!3.7253d-14!
!   s_n(2) =1.5d-4! Finding TS and Min For SE
  ! for hessians
  s_n(3) =1d-3
!   s_n(3) =2.d-3! Finding TS and Min For SE
!   gamma=0.2940594d0! MethanolAlt SE !0.4607d0!0.2940594d0!0.437917775910835d0
  gamma=1d-2
!   gamma=1d0! Finding TS and Min For SE
!   gamma=1/(11.7790d0**2)!5d-3 !new dataset methanol and Matern52
  kernel_type = 1! set the kernel type manually
  if (kernel_type==0) then
    print*, "T! Using Squaxtered exponential kernel."
  else if (kernel_type==1) then
    print*, "T! Using Matern kernel 5/2."
  else
    print*, "Using a new kernel."
  end if
!   !Carbene system (MaxLikelihood result)
!   gamma=0.6d0
!   s_n(1) = 7.83d-6!1d-5! 0d0!1d-7
!   ! for gradients
!   s_n(2) = 1.09d-4
!   ! for hessians
!   s_n(3) = 1.87d-4

  print*, "T! s_f   ", s_f
  print*, "T! s_n(1)", s_n(1)
  print*, "T! s_n(2)", s_n(2)
  print*, "T! s_n(3)", s_n(3)
  print*, "T! gamma ", gamma

  order=1
  kernel_type = 1
  call GPR_construct(this,ns,nat,ni,1,kernel_type,order)
  call GPR_init(this, xs, gamma, s_f, s_n, es, gs, hs)
  call GPR_interpolation(this)
  print*, "T! Testpoints are the trainingpoints"
  call train_and_test_e(this,  ns,xs, es, rmsd, mae) 
  call train_and_test_g(this,  ns,xs, gs, rmsd, mae)
  print*, "T! Testpoints are the real test points"
  call train_and_test_e(this,  ntest,xtest, etest, rmsd, mae) 
  call train_and_test_g(this,  ntest,xtest, gtest, rmsd, mae)
  print*, "T! Introducing new level with ",this%nt/2," from ", this%nt,"datapoints"
  call GPR_newLevel(this,this%nt/2)
  call GPR_interpolation(this)
  print*, "T! "
  print*, "T! Multilevel:"
  print*, "T! Testpoints are the trainingpoints"
  call train_and_test_e(this,  ns,xs, es, rmsd, mae) 
  call train_and_test_g(this,  ns,xs, gs, rmsd, mae)
  print*, "T! Testpoints are the real test points"
  call train_and_test_e(this, ntest,xtest, etest, rmsd, mae) 
  call train_and_test_g(this, ntest,xtest, gtest, rmsd, mae)
  call GPR_set_up_mlit(iter_man,this,1d-9,1d-9,1d-4,40) !nIterMax=40 for energies
  call GPR_eval_mlit(iter_man,this%xs(:,1),tmp)
  print*, "T! "
  print*, "T! Multilevel with iterations:"  
  print*, "T! Testpoints are the trainingpoints"
  call train_and_test_e(this,  ns,xs, es, rmsd, mae) 
  call train_and_test_g(this,  ns,xs, gs, rmsd, mae) 
  print*, "T! Testpoints are the real test points"
  call train_and_test_e(this,  ntest,xtest, etest, rmsd, mae) 
  call train_and_test_g(this,  ntest,xtest, gtest, rmsd, mae)
#endif

#ifdef RealSystem
! ! !*****************************************************************************  
! ! !************ This uses the real input data from neuralnetwork.f90
! ! !************ to start a GPR interpolation. It can be used to write
! ! !************ data in a .dat file for dl-find.
  
  usingHess     = .true.
  usingGrad     = .true.
  usingEnergies = .true.
  
  s_f = 1d0
  ! for coords
  s_n(1) =1d-7!2.437455D-14! 
!   s_n(1) =1d-5! Finding TS and Min For SE
  ! for gradients
  s_n(2) =1d-7!3.7253d-14!
!   s_n(2) =1.5d-4! Finding TS and Min For SE
  ! for hessians
  s_n(3) =1d-6
!   s_n(3) =2.d-3! Finding TS and Min For SE
!   gamma=0.2940594d0! MethanolAlt SE !0.4607d0!0.2940594d0!0.437917775910835d0
  gamma=5d-4
!   gamma=1d0! Finding TS and Min For SE
!   gamma=1/(11.7790d0**2)!5d-3 !new dataset methanol and Matern52
  kernel_type = 1! set the kernel type manually
  
  if (kernel_type==0) then
    print*, "T! Using Squaxtered exponential kernel."
  else if (kernel_type==1) then
    print*, "T! Using Matern kernel 5/2."
  else
    print*, "Using a new kernel."
  end if
print*, "T! s_f   ", s_f
print*, "T! s_n(1)", s_n(1)
print*, "T! s_n(2)", s_n(2)
print*, "T! s_n(3)", s_n(3)
print*, "T! gamma ", gamma
  this%iChol = .false. ! false -> using the standard BLAS package for cholesky

  if (usingEnergies) then
  order = 0
  ! Testing the likelihood performance for several values of gamme
  gamma=1d-10
  do i = 1, 44
    call GPR_construct(this,ns,nat,ni,1,kernel_type,order)
    call GPR_init(this, xs, gamma, s_f, s_n, es)
    call GPR_interpolation(this)
    call calc_p_like_and_deriv(this,1,(/2/),likelihood,gradsLike,parasLike)
    print*, "likelihoodOverGamma_order",order, gamma, likelihood
    call GPR_destroy(this)
    gamma = gamma * 2
  end do
end if

if (usingGrad) then
  order = 1
  ! Testing the likelihood performance for several values of gamme
  gamma=1d-10
  do i = 1, 44
    call GPR_construct(this,ns,nat,ni,1,kernel_type,order)
    call GPR_init(this, xs, gamma, s_f, s_n, es, gs)
    call GPR_interpolation(this)
    call calc_p_like_and_deriv(this,1,(/2/),likelihood,gradsLike,parasLike)
    print*, "likelihoodOverGamma_order",order, gamma, likelihood
    call GPR_destroy(this)
    gamma = gamma * 2
  end do
end if

if (usingHess) then
  order = 2
  ! Testing the likelihood performance for several values of gamme
  gamma=1d-10
  do i = 1, 44
    call GPR_construct(this,ns,nat,ni,1,kernel_type,order)
    call GPR_init(this, xs, gamma, s_f, s_n, es, gs, hs)
    call GPR_interpolation(this)
    call calc_p_like_and_deriv(this,1,(/2/),likelihood,gradsLike,parasLike)
    print*, "likelihoodOverGamma_order",order, gamma, likelihood
    call GPR_destroy(this)
    gamma = gamma * 2
  end do  
end if
STOP "DONE"
#endif

#ifdef testIterSolver
  ! construct a GP with only one trainingpoint for using Cholesky
  call GPR_construct(this, ns, nat, ni, 0, 1, 1)
  call GPR_init(this, xs, 1d0/(20d0)**2,1d0,(/1d-7,1d-7,3d-4/), es, gs, hs)
  call GPR_interpolation(this)
  call GPR_write(this,.false.,'gpr.dat')
  return
  ! construct a GP with only one trainingpoint for using iterative solver
  call GPR_construct(this_iter, 0, nat, ni, 0, 1, 1)
  call GPR_init_without_tp(this_iter,1d0/(20d0)**2,1d0,(/1d-7,1d-7,3d-4/))
  
  do i = 1, ns
    call GPR_add_tp(this,1,xs(:,i),es(i),gs(:,i))
    call GPR_add_tp(this_iter,1,xs(:,i),es(i),gs(:,i))    
    call GPR_interpolation(this)
    call GPR_interpolation(this_iter)
  end do
  
  do i = 1, 1000
    call GPR_destroy(this)
    call GPR_construct(this, ns, nat, ni, 0, 1, 1)
    call GPR_init(this, xs, gamma, s_f, s_n, es, gs, hs)
    call GPR_interpolation(this)
  end do

#endif
end subroutine gpr
!!****

subroutine gpr_in_chemsh_pes_init(path,ncoord_,npoint)
  use gpr_in_chemsh_pes_module
  implicit none
  character(*), intent(in) :: path     ! path to training set
!   character(*), intent(in) :: pathtest ! path to test set
  integer :: npointtest
  integer, intent(out)     :: ncoord_,npoint
  character(256) :: line,fname
  integer :: ios
  integer :: ncoord,npointh

  write(*,'(a,a)') "# Initializing interpolation data from directory ",&
      trim(adjustl(path))
  print*,"# Massweight",massweight

  ! create file list
  line="\ls -1 "//trim(adjustl(path))//"hess_*.txt > .tmp"
  call system(line)
!   line="\ls -1 "//trim(adjustl(pathtest))//"hess_*.txt > .tmp_test"
!   call system(line)
  !line="\ls -1 "//trim(adjustl(path))//"grad_*.txt > .gtmp"
  !call system(line)
  npoint=0
  open(unit=20,file=".tmp")
  ! read header to allocate arrays
  read(20,fmt='(a)',end=1000,err=1000) fname

  ! get number of atoms from first file
  open(unit=30,file=fname)
  read(30,fmt='(a)',end=1100,err=1100) line
  read(30,fmt='(a)',end=1100,err=1100) line
  read(line,*) nat
  print*,"# Number of atoms ",nat
  npoint=1
  ncoord=3*nat
  close(30)

  ! return variable:
  if(nat==1) then
    ncoord_=ncoord
  else if (nat==2) then
    ncoord_=ncoord-5 ! =1
  else
    ncoord_=ncoord-6 ! =1
  end if

  ! find out the number of files/points with Hessian information
  do 
    read(20,fmt='(a)',iostat=ios) fname
    if(ios/=0) exit
    npoint=npoint+1
  end do
  npointh=npoint
  print*,"# Number of files with Hessian information ",npointh
  close(20)

!   open(unit=20,file=".tmp_test")
  npointtest=0
!   do 
!     read(20,fmt='(a)',iostat=ios) fname
!     if(ios/=0) exit
!     npointtest=npointtest+1
!   end do
  print*,"# Number of files in test set              ",npointtest
!   close(20)

!!$  ! find out the number of files/points with gradient information only
!!$  open(unit=21,file=".gtmp")
!!$  ! read header to allocate arrays
!!$  do 
!!$    read(21,fmt='(a)',iostat=ios) fname
!!$    if(ios/=0) exit
!!$    npoint=npoint+1
!!$  end do
!!$  print*,"# Number of files with gradient information ",npoint-npointh

  goto 1001
1000 print*,"Error reading from the file list"
  stop

1100 print*,"Error reading from file ",trim(adjustl(fname))
  stop
1001 continue

  tinit=.true.

end subroutine gpr_in_chemsh_pes_init
  
subroutine gpr_in_chemsh_pes_destroy
  use gpr_in_chemsh_pes_module
  implicit none
  if(.not.tinit) then
    print*,"Call gpr_in_chemsh_pes_init before calling gpr_in_chemsh_pes_destroy"
    stop "error"
  end if
  deallocate(align_refcoords)
  deallocate(align_modes)
  deallocate(refmass)
end subroutine gpr_in_chemsh_pes_destroy

subroutine gpr_in_chemsh_pes_read(ncoord_,npoint,ntest,refcoords,refene,refgrad,refhess,testcoords,testene,testgrad,testhess)
  use gpr_in_chemsh_pes_module
  implicit none
  integer, intent(in) :: ncoord_,npoint
!   integer, intent(inout) :: ntest
  integer, intent(inout) :: ntest
  real(8),intent(out) :: refcoords(ncoord_,npoint),refene(npoint),refgrad(ncoord_,npoint),refhess(ncoord_,ncoord_,npoint)
  real(8),intent(out) :: testcoords(ncoord_,ntest),testene(ntest),testgrad(ncoord_,ntest),testhess(ncoord_,ncoord_,ntest)
  character(256) :: line,fname
  integer :: ios,ipoint ! point refers to control point
  real(8), allocatable :: xcoords(:),xgradient(:),xhessian(:,:) ! coords directly read in from file
  real(8), allocatable :: xcoords_store(:,:) 
  real(8) :: trans(3),rotmat(3,3),svar
  real(8), allocatable :: distmat(:,:)
  real(8), allocatable :: modes_tmp(:,:)
  integer :: jpoint,ivar,ncoord,npointh,arr2(2)
  logical :: tok,superimpose=.true.
  real(8) :: mass(nat)
  real(8) :: mindist
  integer :: ntest_diff,iat,jat,icoord
  real(8), allocatable :: mxhessian(:,:) 

  if(.not.tinit) then
    print*,"Call gpr_in_chemsh_pes_init before calling gpr_in_chemsh_pes_read"
    stop "error"
  end if
  
  ncoord=3*nat
  npointh=npoint

  ! allocate local arrays
  allocate(xcoords(ncoord))
  allocate(xgradient(ncoord))
  allocate(xhessian(ncoord,ncoord))
  allocate(xcoords_store(ncoord,npoint))

  ! now read files
  open(unit=20,file=".tmp")
!  rewind(21)
  open(unit=52,file="training.xyz")
  open(unit=53,file="validation.xyz")
  do ipoint=1,npoint+ntest
    if(ipoint==npoint+1) then
      close(20)
      open(unit=20,file=".tmp_test")
    end if
!    if(ipoint<=npointh) then
    read(20,fmt='(a)',iostat=ios) fname
!    else
!      read(21,fmt='(a)',iostat=ios) fname
!    end if
    if(ios/=0) then
      print*,"Error getting file name"
      stop
    end if
    if(ipoint<=npoint) then
      write(*,"('Reference point',i3,' is ',a)") ipoint,trim(fname)
    else
      write(*,"('Test point',i3,' is ',a)") ipoint,trim(fname)
    end if
    open(unit=30,file=fname)

    read(30,fmt='(a)',end=1100,err=1100) line
    read(30,fmt='(a)',end=1100,err=1100) line
    read(30,fmt='(a)',end=1100,err=1100) line
    read(30,fmt='(a)',end=1100,err=1100) line

    if(ipoint<=npoint) then
      read(30,*) refene(ipoint)
    else
      read(30,*) testene(ipoint-npoint)
    end if
      
    read(30,fmt='(a)',end=1100,err=1100) line
    !print*,trim(adjustl(line))
    read(30,*) xcoords !refcoords(:,ipoint)

    read(30,fmt='(a)',end=1100,err=1100) line
    !print*,trim(adjustl(line))
    read(30,*) xgradient !refgrad(:,ipoint)

    read(30,fmt='(a)',end=1100,err=1100) line
    !print*,trim(adjustl(line))
    xhessian=0.D0
    read(30,*) xhessian 

    if(ipoint<=npoint) xcoords_store(:,ipoint)=xcoords

    ! now superimpose structure to first frame and transform derivatives

    ! scoords=superimposed to reference structure
    if(superimpose) then
      if(ipoint==1) then
        ! allocate array for module
        allocate(align_refcoords(3*nat))
        allocate(modes_tmp(3*nat,3*nat))
        mass=1.D0 ! no mass-weighting
        if(massweight) then
          allocate(refmass(nat))
          read(30,fmt='(a)',end=1100,err=1100) line
          refmass=0.D0
          read(30,*) refmass
          ! transform from a.u. to amu (does not matter, but leads to numerically more similar values to E and G)
          refmass=refmass/(1.66054D-27/9.10939D-31)
          mass=refmass
        end if
        align_refcoords=xcoords
        ! mass-weight the Hessian
        allocate(mxhessian(ncoord,ncoord))
        do iat=1,nat
          do jat=1,nat
            mxhessian(iat*3-2:iat*3,jat*3-2:jat*3)=xhessian(iat*3-2:iat*3,jat*3-2:jat*3)/sqrt(mass(iat)*mass(jat))
          end do
        end do
        call gpr_in_chemsh_dlf_thermal_project_readin(nat,mass,align_refcoords,mxhessian,ivar,modes_tmp,tok)
        deallocate(mxhessian)
        if(.not.tok) stop "Error in gpr_in_chemsh_dlf_thermal_project_readin"
        print*,"# Number of spatial variables ",ivar
        if(ivar/=ncoord_) then
          print*,"Error in number of coordinates:"
          print*,"ncoord_=",ncoord_
          print*,"number of coords required by projection=",ivar
          stop "error"
        end if
        
        allocate(align_modes(3*nat,ncoord_))
        align_modes=modes_tmp(:,1:ncoord_)
        deallocate(modes_tmp)
        ! now we have the modes. The coords are going to be transformed to mode elongations in gpr_in_chemsh_cgh_xtos.

        !check if align_modes is orthogonal:
        !write(*,'("orthogonal?",6f10.5)') matmul(transpose(align_modes),align_modes)

      end if ! ipoint==1

      if(ipoint<=npoint) then
        call gpr_in_chemsh_cgh_xtos(ncoord_,align_refcoords,xcoords,xgradient,xhessian,trans,rotmat,&
            refcoords(:,ipoint),refgrad(:,ipoint),refhess(:,:,ipoint),.true.)
      else
        call gpr_in_chemsh_cgh_xtos(ncoord_,align_refcoords,xcoords,xgradient,xhessian,trans,rotmat,&
            testcoords(:,ipoint-npoint),testgrad(:,ipoint-npoint),testhess(:,:,ipoint-npoint),.false.)
      end if
    end if

    close(30)

!    refcoords(:,ipoint)=xcoords
!    refgrad(:,ipoint)=xgradient
!    if(ipoint<=npointh) refhess(:,:,ipoint)=xhessian

  end do

  close(20) ! .tmp
  close(52)
  close(53)
!  close(21) ! .gtmp

  call system("rm -f .tmp .gtmp .tmp_test")

  goto 1001
1100 print*,"Error reading from file ",trim(adjustl(fname))
  stop
1001 continue
! 
!   deallocate(xcoords)
!   deallocate(xgradient)
!   deallocate(xhessian)
!   
!   ! now get information about mutal distances of control points. Maybe one can merge some?
!   if(npoint>1) then
!     allocate(distmat(npoint,npoint))
!     distmat=0.D0
!     do ipoint=1,npoint
!       do jpoint=ipoint+1,npoint
!         distmat(ipoint,jpoint)=sum((refcoords(:,ipoint)-refcoords(:,jpoint))**2)
!       end do
!     end do
!     ! set lower half and diagonal to averag value of distmat, so that it
!     ! does not disturb the rest
!     svar=sum(distmat)/(npoint*(npoint-1)/2)
!     do ipoint=1,npoint
!       do jpoint=1,ipoint
!         distmat(ipoint,jpoint)=svar
!       end do
!     end do
!     write(*,'(" # Minimum distance between two points: ",es12.4,", control points",2i6)') minval(distmat),minloc(distmat)
!     write(*,'(" # Maximum distance between two points: ",es12.4,", control points",2i6)') maxval(distmat),maxloc(distmat)
!     print*,"List of shortest dinstances:"
!     print*,"Number       Distance              Control Points"
!     do ipoint=1,min(10,npoint)
!       print*,ipoint, minval(distmat),minloc(distmat)
!       arr2=minloc(distmat)
!       distmat(arr2(1),arr2(2))=huge(1.D0)
!     end do
!     
!     deallocate(distmat)
!   end if
! 
!   !print*,"# to be done: make sure that test and trainig set are mutually exclusive"
!   ntest_diff=0
!   do ipoint=npoint+1,npoint+ntest ! test set
!     mindist=huge(1.D0)
!     !ntest_diff=0
!     do jpoint=1,npoint ! training set
!       svar=sum((testcoords(:,ipoint-npoint)-refcoords(:,jpoint))**2)
!       if(svar<mindist) then
!         mindist=svar
!         !ntest_diff=jpoint
!       end if
!     end do
!     !print*,"point",ipoint,"mindist",mindist,ntest_diff   
!     if(mindist>-1.D-6) then ! negative value: test deactivated!
!       ntest_diff=ntest_diff+1
!       !print*,"copying",ipoint," to ",ntest_diff
!       testcoords(:,ntest_diff)=testcoords(:,ipoint-npoint)
!       testene(ntest_diff)=testene(ipoint-npoint)
!       testgrad(:,ntest_diff)=testgrad(:,ipoint-npoint)
!       testhess(:,:,ntest_diff)=testhess(:,:,ipoint-npoint)
!     else
!       print*,"Test point",ipoint," excluded because it is too close to a training point"
!     end if
!     !print*,"point",ipoint,"mindist",mindist,ntest_diff
!   end do
!   ntest=ntest_diff
!   print*,"# Initialisation of neural network interpolation finished"
!   !tinit=.true.
! 
!   ! print the internal coordinates (refcoords, testcoords) as parallel coordintes
!   open(unit=52,file="trainingset.pc")
!   write(52,'(a)') "# internal coordinates of training set as parallel coordinates, one geometry after the next"
!   do ipoint=1,npoint
!     write(52,'(a,i4)') "# Reference point ",ipoint
!     do icoord=1,ncoord_
!       write(52,'(i4,f15.10)') icoord,refcoords(icoord,ipoint)
!     end do
!     write(52,*) 
!   end do
!   close(52)
! 
!   ! print the internal coordinates (refcoords, testcoords) as parallel coordintes
!   open(unit=52,file="testset.pc")
!   write(52,'(a)') "# internal coordinates of test/validation set as parallel coordinates, one geometry after the next"
!   do ipoint=npoint+1,npoint+ntest ! test set
!     write(52,'(a,i4)') "# Test point ",ipoint
!     do icoord=1,ncoord_
!       write(52,'(i4,f15.10)') icoord,testcoords(icoord,ipoint-npoint)
!     end do
!     write(52,*) 
!   end do
!   close(52)
! 
!   !call gpr_in_chemsh_print_coords(ncoord_,3*nat,npoint,refcoords,xcoords_store,refene)
! 
!   !call gpr_in_chemsh_move_coords(ncoord_,3*nat,npoint,refcoords,align_refcoords,align_modes,refmass)

end subroutine gpr_in_chemsh_pes_read

subroutine gpr_in_chemsh_move_coords(ncoord_internal,ncoord_cart,npoint,refcoords,&
    align_refcoords,align_modes,refmass)
  ! purpose: re-construct xcoords from normal mode coordinates and move them
  ! along specific normal modes. later: get energies from DFT and see how the
  ! energy depends on the elongations for GPR.
  implicit none
  integer,intent(in) :: ncoord_internal,ncoord_cart,npoint
  real(8),intent(in) :: refcoords(ncoord_internal,npoint)
  real(8),intent(in) :: align_refcoords(ncoord_cart)
  real(8),intent(in) :: align_modes(ncoord_cart,ncoord_internal)
  real(8),intent(in) :: refmass(ncoord_cart/3)
  integer :: ipoint,icoord,ielong
  real(8) :: xcoords(ncoord_cart),massvec(ncoord_cart)
  real(8) :: icoords(ncoord_internal)
  character(128) :: fname
  
  do icoord=1,ncoord_cart/3
    massvec(icoord*3)=1.D0/sqrt(refmass(icoord))
    massvec(icoord*3-1)=massvec(icoord*3)
    massvec(icoord*3-2)=massvec(icoord*3)
  end do
  print*,"constructing xcoords"
  ipoint=27
  do ielong=-50,50
    xcoords=align_refcoords
    icoords=refcoords(:,ipoint)
    icoords(7)=icoords(7)+dble(ielong)/5.D0
    write(fname,'(i3)') ielong
    fname='res_'//trim(adjustl(fname))//'.xyz'
    open(unit=100,file=fname)
    do icoord=1,ncoord_internal
      print*,"internal coord",icoord,refcoords(icoord,ipoint)
      xcoords=xcoords+icoords(icoord)*align_modes(:,icoord)*massvec
    end do
!!$  print*,"structure"
!!$  do icoord=1,ncoord_cart,3
!!$    write(*,'(3f10.5)') xcoords(icoord:icoord+2)
!!$  end do
    write(100,*) 6
    write(100,*) 
    do icoord=1,ncoord_cart-3,3 ! leave out the last atom and study
       ! methanol only
      if(icoord==1) then
        write(100,'(" C ",3f12.7)') xcoords(icoord:icoord+2)*5.2917720810086D-01
      else if(icoord==4) then
        write(100,'(" O ",3f12.7)') xcoords(icoord:icoord+2)*5.2917720810086D-01
      else
        write(100,'(" H ",3f12.7)') xcoords(icoord:icoord+2)*5.2917720810086D-01
      end if
    end do
    close(100)
  end do
end subroutine gpr_in_chemsh_move_coords

subroutine gpr_in_chemsh_print_coords(ncoord_internal,ncoord_cart,npoint,refcoords,xcoords,refene)
  implicit none
  integer,intent(in) :: ncoord_internal,ncoord_cart,npoint
  real(8),intent(in) :: refcoords(ncoord_internal,npoint)
  real(8),intent(in) :: xcoords(ncoord_cart,npoint)
  real(8),intent(in) :: refene(npoint)
  integer :: icoord,ipoint,ndist,iat,jat,nat,idist
  real(8) :: sortlist(npoint),intervall
  real(8),allocatable :: dist(:,:)

  intervall=(maxval(refene)-minval(refene))*0.5D0

  ! normal coordinates
  open(unit=103,file='trainingsset.normal_coords')
  write(103,*) "# Energy of control points of training set vs. normal coordinates"
  write(103,*) "# The energy is shifted for each coordinate for better visibility"
  do icoord=1,ncoord_internal
    write(103,'("#Coordinate ",i3," energies - ",f10.5)') &
        icoord,dble(icoord-1)*intervall
    sortlist=refcoords(icoord,:)
    do ipoint=1,npoint
      write(103,*) minval(sortlist),refene(minloc(sortlist,dim=1))-dble(icoord-1)*intervall
      sortlist(minloc(sortlist,dim=1))=huge(1.D0)
    end do
    write(103,*)
  end do
  close(103)

  ! cartesian coordinates
  open(unit=103,file='trainingsset.cartesian_coords')
  write(103,*) "# Energy of control points of training set vs. cartesian coordinates"
  write(103,*) "# The energy is shifted for each coordinate for better visibility"
  do icoord=1,ncoord_cart
    write(103,'("#Coordinate ",i3," energies - ",f10.5)') &
        icoord,dble(icoord-1)*intervall
    sortlist=xcoords(icoord,:)
    do ipoint=1,npoint
      write(103,*) minval(sortlist),refene(minloc(sortlist,dim=1))-dble(icoord-1)*intervall
      sortlist(minloc(sortlist,dim=1))=huge(1.D0)
    end do
    write(103,*)
  end do
  close(103)

  ! interatomic distances
  nat=ncoord_cart/3
  ndist=nat*(nat-1)/2
  allocate(dist(ndist,npoint))
  do ipoint=1,npoint
    idist=1
    do iat=1,nat
      do jat=iat+1,nat
        if(idist>ndist) stop "wrong number of distances"
        dist(idist,ipoint)=sqrt(sum((xcoords(3*iat-2:3*iat,ipoint)-xcoords(3*jat-2:3*jat,ipoint))**2))
        idist=idist+1
      end do
    end do
  end do
  open(unit=103,file='trainingsset.distances')
  write(103,*) "# Energy of control points of training set vs. interatomic dinstances"
  write(103,*) "# The energy is shifted for each coordinate for better visibility"
  do icoord=1,ndist
    write(103,'("#Coordinate ",i3," energies - ",f10.5)') &
        icoord,dble(icoord-1)*intervall
    sortlist=dist(icoord,:)
    do ipoint=1,npoint
      write(103,*) minval(sortlist),refene(minloc(sortlist,dim=1))-dble(icoord-1)*intervall
      sortlist(minloc(sortlist,dim=1))=huge(1.D0)
    end do
    write(103,*)
  end do
  close(103)
  open(unit=103,file='trainingsset.invdistances')
  write(103,*) "# Energy of control points of training set vs. inverse interatomic dinstances"
  write(103,*) "# The energy is shifted for each coordinate for better visibility"
  do icoord=1,ndist
    write(103,'("#Coordinate ",i3," energies - ",f10.5)') &
        icoord,dble(icoord-1)*intervall
    sortlist=dist(icoord,:)
    do ipoint=1,npoint
      write(103,*) 1.D0/minval(sortlist),refene(minloc(sortlist,dim=1))-dble(icoord-1)*intervall
      sortlist(minloc(sortlist,dim=1))=huge(1.D0)
    end do
    write(103,*)
  end do
  close(103)

end subroutine gpr_in_chemsh_print_coords


! superimpose coords to rcoords and transform gradient and hessian in
! an appropriate way as well. Return transformed coordinates, ...
! 
! Algorithm for superposition:
! !! See W. Kabsch, Acta Cryst. A 32, p 922 (1976).
!
subroutine gpr_in_chemsh_cgh_xtos(ncoord,rcoords,xcoords,xgradient,xhessian,trans,rotmat,dcoords,dgradient,dhessian,ttrain)
  use gpr_in_chemsh_pes_module
  implicit none
  integer,intent(in) :: ncoord
  real(8),intent(in) :: rcoords(3*nat) ! the set of coordinates the new ones should be fitted to
  real(8),intent(inout) :: xcoords(3*nat) ! could be made intent(in), it is easier that way
  real(8),intent(inout) :: xgradient(3*nat)
  real(8),intent(inout) :: xhessian(3*nat,3*nat)
  real(8),intent(out)   :: trans(3),rotmat(3,3)
  real(8),intent(out) :: dcoords(ncoord)
  real(8),intent(out) :: dgradient(ncoord)
  real(8),intent(out) :: dhessian(ncoord,ncoord)
  logical,intent(in) :: ttrain

  integer:: iat,ivar,jvar,jat,fid
  real(8) :: rmat(3,3),rsmat(3,3),eigvec(3,3),eigval(3)
  real(8) :: center(3)
  real(8) :: weight(3*nat)

  integer :: itry,i,j
  real(8) :: detrot
  
  if(massweight) then
    do iat=1,nat
      weight(iat*3-2:iat*3)=refmass(iat)
    end do
  else
    weight=1.D0
  end if


  ! as compared to dlf_cartesian_align: coords1=rcoords coords2=coords

  trans=0.D0
  rotmat=0.D0
  do ivar=1,3
    rotmat(ivar,ivar)=1.D0
  end do

  ! if there are other cases to ommit a transformation, add them here
  if(nat==1) return
  !if(.not.superimpose) return

  trans=0.D0
  center=0.D0
  do iat=1,nat
    center(:)=center(:)+rcoords(iat*3-2:iat*3)*weight(iat*3-2:iat*3)
    trans(:)=trans(:)+(xcoords(iat*3-2:iat*3)-rcoords(iat*3-2:iat*3))*weight(iat*3-2:iat*3)
  end do
  trans=trans/sum(weight)*3.D0
  center=center/sum(weight)*3.D0

  !print*,"# trans",trans

  ! translate them to common centre
  do iat=1,nat
    xcoords(iat*3-2:iat*3)=xcoords(iat*3-2:iat*3)-trans(:)
  end do

  rmat=0.D0
  do iat=1,nat
    do ivar=1,3
      do jvar=1,3
        rmat(ivar,jvar)=rmat(ivar,jvar)+weight(3*iat)*(rcoords(ivar+3*iat-3)-center(ivar))* &
            (xcoords(jvar+3*iat-3)-center(jvar))
      end do
    end do
  end do
  rmat=rmat/sum(weight)*3.D0
  !write(*,"('R   ',3f10.3)") rmat
  rsmat=transpose(rmat)
  eigvec=matmul(rsmat,rmat)
  rsmat=eigvec

  !write(stdout,"('RtR ',3f10.3)") rsmat
  call gpr_in_chemsh_matrix_diagonalise(3,rsmat,eigval,eigvec)

  ! It turns out that the rotation matrix may have a determinat of -1
  ! in the procedure used here, i.e. the system is mirrored - which is
  ! wrong chemically. This can be avoided by inserting a minus in the
  ! equation
  ! 1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))

  ! So, here we first calculate the rotation matrix, and if it is
  ! zero, the first eigenvalue is reversed

  do itry=1,2
    ! rsmat are the vectors b:
    j=-1
    do i=1,3
      if(eigval(i)<1.D-8) then
        if(i>1) then
          ! the system is linear - no rotation necessay.
          ! WHY ?! There should still be one necessary!
          return
          !print*,"Eigenval. zero",i,eigval(i)
          !call dlf_fail("Error in dlf_cartesian_align")
        end if
        j=1
      else
        if(i==1.and.itry==2) then
          rsmat(:,i)=-1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
        else
          rsmat(:,i)=1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
        end if
      end if
    end do
    if(j==1) then
      ! one eigenvalue was zero, the system is planar
      rsmat(1,1)=rsmat(2,2)*rsmat(3,3)-rsmat(3,2)*rsmat(2,3)
      rsmat(2,1)=rsmat(3,2)*rsmat(1,3)-rsmat(1,2)*rsmat(3,3)
      rsmat(3,1)=rsmat(1,2)*rsmat(2,3)-rsmat(2,2)*rsmat(1,3)
      ! deal with negative determinant
      if (itry==2) then
         rsmat(:,1) = -rsmat(:,1)
      end if
    end if

    do i=1,3
      do j=1,3
        rotmat(i,j)=sum(rsmat(i,:)*eigvec(j,:))
      end do
    end do
    !write(*,"('rotmat ',3f10.3)") rotmat
    detrot=   &
        rotmat(1,1)*(rotmat(2,2)*rotmat(3,3)-rotmat(2,3)*rotmat(3,2)) &
        -rotmat(2,1)*(rotmat(1,2)*rotmat(3,3)-rotmat(1,3)*rotmat(3,2)) &
        +rotmat(3,1)*(rotmat(1,2)*rotmat(2,3)-rotmat(1,3)*rotmat(2,2))
    !write(*,*) "Determinat of rotmat", detrot
    if(detrot > 0.D0) exit
    if(detrot < 0.D0 .and. itry==2) then
      stop "Error in dlf_cartesian_align, obtained a mirroring instead of rotation."
    end if

  end do


!!$  do ivar=1,3
!!$    rsmat(:,ivar)=1.d0/dsqrt(eigval(ivar)) * matmul(rmat,eigvec(:,ivar))
!!$  end do
!!$
!!$  do ivar=1,3
!!$    do jvar=1,3
!!$      rotmat(ivar,jvar)=sum(rsmat(ivar,:)*eigvec(jvar,:))
!!$    end do
!!$  end do

  ! transform coordinates
  do iat=1,nat
    xcoords(iat*3-2:iat*3)= xcoords(iat*3-2:iat*3)-center
    xcoords(iat*3-2:iat*3)=matmul(rotmat,xcoords(iat*3-2:iat*3))
    xcoords(iat*3-2:iat*3)= xcoords(iat*3-2:iat*3)+center
  end do

  ! write xyz
  if(ttrain) then
    fid=52
  else
    fid=53
  end if
  write(fid,*) nat
  write(fid,*) 
  do iat=1,nat
    write(fid,'(" H ",3f12.7)') xcoords(iat*3-2:iat*3)*5.2917720810086D-01
  end do
  
!  print*,"transformed coordinates"
!  write(*,'(3f15.5)') coords
  
  ! transform gradient
  do iat=1,nat
    xgradient(iat*3-2:iat*3)=matmul(rotmat,xgradient(iat*3-2:iat*3))
  end do
  !print*,"transformed gradient"
  !write(*,'(3f15.5)') gradient

  !print*,"rotation matrix"
  !write(*,'(3f15.5)') rotmat

  ! transform hessian
  do iat=1,nat
    do jat=1,nat
      xhessian(iat*3-2:iat*3,jat*3-2:jat*3)=matmul(matmul(rotmat,xhessian(iat*3-2:iat*3,jat*3-2:jat*3)),transpose(rotmat)) 
    end do
  end do

!!$  print*,"transformed hessian"
!!$  do ivar=1,6
!!$    write(*,'(3es13.5,2x,3es13.5,2x,3es13.5)') hessian(ivar,1:9)
!!$    if(ivar==3) print*
!!$  end do

  ! now all quantities have been transformed to c-coords (or relative to c-coordinates)

  ! now, the coordinates need to be mass-weighted!
  dcoords=matmul(transpose(align_modes),sqrt(weight)*(xcoords-align_refcoords))

  if(massweight) then
    do iat=1,nat
      xgradient(iat*3-2:iat*3)=xgradient(iat*3-2:iat*3)/sqrt(refmass(iat))
      do jat=1,nat
        xhessian(iat*3-2:iat*3,jat*3-2:jat*3)=&
            xhessian(iat*3-2:iat*3,jat*3-2:jat*3)/sqrt(refmass(iat))/sqrt(refmass(jat))
      end do
    end do
  end if
  print*, "aModes", SIZE(xgradient),SIZE(align_modes,1)
  dgradient=matmul(transpose(align_modes),xgradient)
  dhessian=matmul(transpose(align_modes),matmul(xhessian,align_modes))

end subroutine gpr_in_chemsh_cgh_xtos


!!$subroutine gpr_in_chemsh_pes_destroy
!!$  !use gpr_in_chemsh_pes_module
!!$  implicit none
!!$  ! deallocate arrays
!!$  deallocate(refcoords)
!!$  deallocate(refene)
!!$  deallocate(refgrad)
!!$  deallocate(refhess)
!!$
!!$  tinit=.false.
!!$end subroutine gpr_in_chemsh_pes_destroy


SUBROUTINE gpr_in_chemsh_matrix_diagonalise(N,H,E,U) 
  IMPLICIT NONE

  LOGICAL(4) ,PARAMETER :: TESSLERR=.FALSE.
  INTEGER   ,INTENT(IN) :: N
  REAL(8)   ,INTENT(IN) :: H(N,N)
  REAL(8)   ,INTENT(OUT):: E(N)
  REAL(8)   ,INTENT(OUT):: U(N,N)
  REAL(8)               :: WORK1((N*(N+1))/2)
  REAL(8)               :: WORK2(3*N)
  INTEGER               :: K,I,J
  !CHARACTER(8)          :: SAV2101
  INTEGER               :: I1,I2
  INTEGER               :: INFO
  INTEGER               :: INF1
  INTEGER               :: INF2
  
  K=0
  DO J=1,N
    DO I=J,N
      K=K+1
      WORK1(K)=0.5D0*(H(I,J)+H(J,I))
    ENDDO
  ENDDO
  
  CALL dspev('V','L',N,WORK1,E,U,N,WORK2,INFO) !->LAPACK intel
  IF(INFO.NE.0) THEN
    PRINT*,'DIAGONALIZATION NOT CONVERGED (gpr_in_chemsh_matrix_diagonalise)'
    STOP
  END IF
  
END SUBROUTINE gpr_in_chemsh_matrix_diagonalise

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hessian/gpr_in_chemsh_dlf_thermal_project_readin
!!
!! FUNCTION
!!
!! * project out non-zero translational and rotational modes from Hessian
!!
!! References: 
!!   1 "Vibrational Analysis in Gaussian", Joseph W. Ochterski
!!      http://www.gaussian.com/g_whitepap/vib/vib.pdf
!!   2 The code in OPTvibfrq in opt/vibfrq.f
!!
!! Notes
!!   - Both of the above start from non-mass-weighted Cartesians whereas
!!     in DL-FIND the Hessian is already mass-weighted 
!!   - There appears to be an error in Ref 1 eqn 5 as the expression 
!!     should be multiplied by sqrt(mass), not divided by it (cf ref 2)
!!     If the uncorrected eqn5 is used then the rotational modes are not 
!!     fully orthogonal to the translational ones.
!!
!! SYNOPSIS
subroutine gpr_in_chemsh_dlf_thermal_project_readin(nat,mass,coords,hessian,npmodes,pmodes,tok)
!! SOURCE
!  use dlf_parameter_module, only: rk
!  use dlf_global, only: glob,stdout,printl
  use gpr_in_chemsh_pes_module, only: massweight
  implicit none
  real(8), external :: ddot
  integer, intent(in) :: nat
  real(8),intent(in)  :: mass(nat)
  real(8),intent(in)  :: coords(3*nat)
  real(8),intent(in)  :: hessian(3*nat,3*nat)
  integer, intent(out)  :: npmodes ! number of vibrational modes
  real(8) :: peigval(3*nat) ! eigenvalues after projection
  real(8), intent(out)  :: pmodes(3*nat, 3*nat) ! vib modes after proj. (non-mass-weighted) 
  logical               :: tok
  real(8)              :: comcoords(3,nat) ! centre of mass coordinates
  real(8)              :: com(3) ! centre of mass
  real(8)              :: totmass ! total mass
  real(8)              :: moi(3,3) ! moment of inertia tensor
  real(8)              :: moivec(3,3) ! MOI eigenvectors
  real(8)              :: moival(3) ! MOI eigenvalues
  real(8)              :: transmat(3*nat,3*nat) ! transformation matrix
  real(8)              :: px(3), py(3), pz(3)
  real(8)              :: smass
  real(8), parameter   :: mcutoff = 1.0d-12
  integer               :: ntrro ! number of trans/rot modes
  real(8)              :: test, norm
  real(8)              :: trialv(3*nat)
  real(8)              :: phess(3*nat,3*nat) ! projected Hessian
  real(8)              :: peigvec(3*nat, 3*nat) ! eigenvectors after proj.
!  real(8)              :: pmodes(3*nat, 3*nat) ! vib modes after proj.
  integer               :: pstart
  integer               :: ival, jval, kval, lval, icount
  integer              :: printl=4
! **********************************************************************
  tok=.false.
  ! Do not continue if any coordinates are frozen
  if (nat==1) then
     write(*,*)
     write(*,"('Frozen atoms found: no modes will be projected out')")
     npmodes = 3*nat
     !peigval = eigval
     return
  end if

  write(*,*)
  write(*,"('Projecting out translational and rotational modes')")

  ! Calculate centre of mass and moment of inertia tensor

!  ! xcoords is not fully up to date so convert icoords instead
!  call dlf_cartesian_itox(nat, 3*nat, glob%nicore, &
!       glob%massweight, glob%icoords, comcoords)
  comcoords=reshape(coords,(/3,nat/))
  com(:) = 0.0d0
  totmass = 0.0d0
  do ival = 1, nat
     com(1:3) = com(1:3) + mass(ival) * comcoords(1:3, ival)
     totmass = totmass + mass(ival)
  end do
  com(1:3) = com(1:3) / totmass

  do ival = 1, nat
     comcoords(1:3, ival) = comcoords(1:3, ival) - com(1:3)
  end do

  moi(:,:) = 0.0d0
  do ival = 1, nat
     moi(1,1) = moi(1,1) + mass(ival) * &
          (comcoords(2,ival) * comcoords(2,ival) + comcoords(3,ival) * comcoords(3,ival))
     moi(2,2) = moi(2,2) + mass(ival) * &
          (comcoords(1,ival) * comcoords(1,ival) + comcoords(3,ival) * comcoords(3,ival))
     moi(3,3) = moi(3,3) + mass(ival) * &
          (comcoords(1,ival) * comcoords(1,ival) + comcoords(2,ival) * comcoords(2,ival))
     moi(1,2) = moi(1,2) - mass(ival) * comcoords(1, ival) * comcoords(2, ival)
     moi(1,3) = moi(1,3) - mass(ival) * comcoords(1, ival) * comcoords(3, ival)
     moi(2,3) = moi(2,3) - mass(ival) * comcoords(2, ival) * comcoords(3, ival)
  end do
  moi(2,1) = moi(1,2)
  moi(3,1) = moi(1,3)
  moi(3,2) = moi(2,3)

  call gpr_in_chemsh_matrix_diagonalise(3, moi, moival, moivec)

  if (printl >= 6) then
     write(*,"(/,'Centre of mass'/3f15.5)") com(1:3)
     write(*,"('Moment of inertia tensor')")
     write(*,"(3f15.5)") moi(1:3, 1:3)
     write(*,"('Principal moments of inertia')")
     write(*,"(3f15.5)") moival(1:3)
     write(*,"('Principal axes')")
     write(*,"(3f15.5)") moivec(1:3, 1:3)
  end if

  ! Construct transformation matrix to internal coordinates
  ntrro = 6
  transmat(:, :) = 0.0d0
  do ival = 1, nat
     smass = sqrt(mass(ival))
     kval = 3 * (ival - 1)
     ! Translational vectors
     transmat(kval+1, 1) = smass
     transmat(kval+2, 2) = smass
     transmat(kval+3, 3) = smass
     ! Rotational vectors
     px = sum(comcoords(1:3,ival) * moivec(1:3,1))
     py = sum(comcoords(1:3,ival) * moivec(1:3,2))
     pz = sum(comcoords(1:3,ival) * moivec(1:3,3))
     transmat(kval+1:kval+3, 4) = (py*moivec(1:3,3) - pz*moivec(1:3,2))*smass
     transmat(kval+1:kval+3, 5) = (pz*moivec(1:3,1) - px*moivec(1:3,3))*smass
     transmat(kval+1:kval+3, 6) = (px*moivec(1:3,2) - py*moivec(1:3,1))*smass
  end do
  ! Normalise vectors and check for linear molecules (one less mode)
  do ival = 1, 6
     test = sum(transmat(:,ival)**2)  !ddot(3*nat, transmat(1,ival), 1, transmat(1,ival), 1)
     if (test < mcutoff) then
        kval = ival
        ntrro = ntrro - 1
        if (ntrro < 5) then
           write(*,"('Error: too few rotational/translation modes')")
           npmodes = 3*nat
           !peigval = eigval
           return
        end if
     else
        norm = 1.0d0/sqrt(test)
        call dscal(3*nat, norm, transmat(1,ival), 1)
     end if
  end do
  if (ntrro == 5 .and. kval /= 6) then
     transmat(:, kval) = transmat(:, 6)
     transmat(:, 6) = 0.0d0
  end if
  write(*,"(/,'Number of translational/rotational modes:',i4)") ntrro

  ! Generate 3N-ntrro other orthogonal vectors 
  ! Following the method in OPTvibfrq
  icount = ntrro
  do ival = 1, 3*nat
     trialv(:) = 0.0d0
     trialv(ival) = 1.0d0
     do jval = 1, icount
        ! Test if trial vector is linearly independent of previous set
        test = -sum(transmat(:,jval)*trialv(:)) !-ddot(3*nat, transmat(1,jval), 1, trialv, 1)
        call daxpy(3*nat, test, transmat(1,jval), 1, trialv, 1)
     end do
     test = ddot(3*nat, trialv, 1, trialv, 1)
     if (test > mcutoff) then
        icount = icount + 1
        norm = 1.0d0/sqrt(test)
        transmat(1:3*nat, icount) = norm * trialv(1:3*nat)
     end if
     if (icount == 3*nat) exit
  end do
  if (icount /= 3*nat) then
     write(*,"('Error: unable to generate transformation matrix')")
     npmodes = 3*nat
     !peigval = eigval
     return
  end if
  if (printl >= 6) then
     write(*,"(/,'Transformation matrix')")
     !call dlf_matrix_print(3*nat, 3*nat, transmat)
  end if

  ! Apply transformation matrix: D(T) H D
  ! Use peigvec as scratch to store intermediate
  phess(:,:) = 0.0d0
  peigvec(:,:) = 0.0d0
  call dlf_matrix_multiply(3*nat, 3*nat, 3*nat, &
       1.0d0,hessian, transmat, 0.0d0, peigvec)
  ! Should alter dlf_matrix_multiply to allow transpose option to be set...
  transmat = transpose(transmat)
  call dlf_matrix_multiply(3*nat, 3*nat, 3*nat, &
       1.0d0, transmat, peigvec, 0.0d0, phess)
  transmat = transpose(transmat)

  if (printl >= 6) then
     write(*,"(/,'Hessian matrix after projection:')")
     !call dlf_matrix_print(3*nat, 3*nat, phess)
  end if

  ! Find eigenvalues of Nvib x Nvib submatrix
  peigval(:) = 0.0d0
  peigvec(:,:) = 0.0d0
  npmodes = 3*nat - ntrro
  pstart = ntrro + 1
  call gpr_in_chemsh_matrix_diagonalise(npmodes, phess(pstart:3*nat, pstart:3*nat), &
       peigval(1:npmodes), peigvec(1:npmodes,1:npmodes))

  if (printl >= 6) then
     write(*,"('Vibrational submatrix eigenvalues:')")
     write(*,"(12f9.5)") peigval(1:npmodes)
!     write(*,"('Vibrational submatrix eigenvectors:')")
     !call dlf_matrix_print(npmodes, npmodes, peigvec(1:npmodes, 1:npmodes))
  end if

  ! Print out normalised normal modes
  ! These are in non-mass-weighted Cartesians (division by smass)
  pmodes(:,:) = 0.0d0
  do kval = 1, 3*nat
     do ival = 1, npmodes
        do jval = 1, npmodes
           pmodes(kval, ival) = pmodes(kval, ival) + &
                transmat(kval, ntrro + jval) * peigvec(jval, ival)
        end do
        lval = (kval - 1) / 3 + 1
        smass = sqrt(mass(lval))
        ! the next line must be commented out if pmodes should be returned in mass-weighted cartesians
        !if(.not.massweight) pmodes(kval, ival) = pmodes(kval, ival) / smass
     end do
  end do
  do ival = 1, npmodes
     test = ddot(3*nat, pmodes(1,ival), 1, pmodes(1,ival), 1)
     norm = 1.0d0 / sqrt(test)
     call dscal(3*nat, norm, pmodes(1,ival), 1)
  end do

  if (printl >= 4) then
     write(*,"(/,'Normalised normal modes (Cartesian coordinates):')")
     !call dlf_matrix_print(3*nat, npmodes, pmodes(1:3*nat, 1:npmodes))
  end if
  
  tok=.true.

end subroutine gpr_in_chemsh_dlf_thermal_project_readin
!!****















! The following is copied from dl-finds dlf_pes.f90 file

! superimpose coords to rcoords and transform gradient and hessian in
! an appropriate way as well. Return transformed coordinates, ...
! this is the same routine as in readin.f90 of the NN fitting code, but with the gradient, hessian removed.
! (the name of the module is also changed: pes_module -> nn_module)
! nvar is added (3*nat replaced by nvar)
!
! Algorithm for superposition:
! !! See W. Kabsch, Acta Cryst. A 32, p 922 (1976).
!

module gpr_chemsh_fromdlf_nn_module
  use dlf_parameter_module, only: rk
  implicit none
  integer :: ni,nj,nk,remove_n_dims,interatomic_dof
  logical :: dimension_reduce,coords_interatomic,coords_inverse
  real(rk),allocatable,save :: wone(:,:),wtwo(:,:),wthree(:),bone(:),btwo(:) !wone(nj,ni),wtwo(nk,nj),wthree(nk),bone(nj),btwo(nk)
  real(rk),save :: bthree,alpha,pinv_tol
  real(rk),allocatable,save :: pca_eigvect(:,:)
  integer, allocatable,save :: radii_omit(:)
  real(rk),allocatable,save :: mu(:)
  real(rk) training_e_ave(3,2)
  logical :: temin,barycentre_switch
  real(rk) :: emin
end module gpr_chemsh_fromdlf_nn_module

module gpr_chemsh_fromdlf_nn_av_module
  use dlf_parameter_module, only: rk
  logical, parameter :: avnncalc=.false.
  integer, parameter :: numfiles=13
  integer, parameter :: numimg=100
  integer, save :: imgcount=0
  integer :: iii, iij, iik 
  character (len=10) :: infiles(numfiles)
  real(rk), allocatable, save :: woneav(:,:,:), wtwoav(:,:,:), wthreeav(:,:), boneav(:,:), btwoav(:,:), bthreeav(:)
  real(rk), allocatable, save :: align_refcoordsav(:,:), align_modesav(:,:,:)
  integer, save :: ifile=0
  real(rk), allocatable, save :: energyav(:),gradientav(:,:),hessianav(:,:,:)
end module gpr_chemsh_fromdlf_nn_av_module

subroutine gpr_chemsh_fromdlf_cgh_xtos(ncoord,nvar,rcoords,xcoords_,trans,rotmat,dcoords)
  use gpr_in_chemsh_pes_module, only: tinit, massweight, align_refcoords,&
                                      align_modes, refmass
  use gpr_chemsh_fromdlf_nn_module
  use gpr_chemsh_fromdlf_nn_av_module
  implicit none
  integer,intent(in) :: ncoord,nvar
  real(8),intent(in) :: rcoords(nvar) ! the set of coordinates the new ones should be fitted to
  real(8),intent(in) :: xcoords_(nvar) 
  real(8),intent(out) :: trans(3),rotmat(3,3)
  real(8),intent(out) :: dcoords(ncoord)

  integer:: iat,ivar,jvar,nat
  real(8) :: rmat(3,3),rsmat(3,3),eigvec(3,3),eigval(3)
  real(8) :: center(3)
  real(8) :: weight(nvar)
  real(8) :: xcoords(nvar)

  integer :: itry,i,j
  real(8) :: detrot

  xcoords=xcoords_

  nat=nvar/3

  if(massweight) then
    do iat=1,nat
      weight(iat*3-2:iat*3)=refmass(iat)
    end do
  else
    weight=1.D0
  end if


  ! as compared to dlf_cartesian_align: coords1=rcoords coords2=coords

  trans=0.D0
  rotmat=0.D0
  do ivar=1,3
    rotmat(ivar,ivar)=1.D0
  end do

  ! if there are other cases to ommit a transformation, add them here
  if(nat==1) return
  !if(.not.superimpose) return

  trans=0.D0
  center=0.D0
  do iat=1,nat
    center(:)=center(:)+rcoords(iat*3-2:iat*3)*weight(iat*3-2:iat*3)
    trans(:)=trans(:)+(xcoords(iat*3-2:iat*3)-rcoords(iat*3-2:iat*3))*weight(iat*3-2:iat*3)
  end do
  trans=trans/sum(weight)*3.D0
  center=center/sum(weight)*3.D0

  !print*,"# trans",trans

  ! translate them to common centre
  do iat=1,nat
    xcoords(iat*3-2:iat*3)=xcoords(iat*3-2:iat*3)-trans(:)
  end do

  rmat=0.D0
  do iat=1,nat
    do ivar=1,3
      do jvar=1,3
        rmat(ivar,jvar)=rmat(ivar,jvar)+weight(3*iat)*(rcoords(ivar+3*iat-3)-center(ivar))* &
            (xcoords(jvar+3*iat-3)-center(jvar))
      end do
    end do
  end do
  rmat=rmat/sum(weight)*3.D0
  !write(*,"('R   ',3f10.3)") rmat
  rsmat=transpose(rmat)
  eigvec=matmul(rsmat,rmat)
  rsmat=eigvec

  !write(stdout,"('RtR ',3f10.3)") rsmat
  call gpr_chemsh_fromdlf_mat_diag(3,rsmat,eigval,eigvec)

  ! It turns out that the rotation matrix may have a determinat of -1
  ! in the procedure used here, i.e. the system is mirrored - which is
  ! wrong chemically. This can be avoided by inserting a minus in the
  ! equation
  ! 1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))

  ! So, here we first calculate the rotation matrix, and if it is
  ! zero, the first eigenvalue is reversed

  do itry=1,2
    ! rsmat are the vectors b:
    j=-1
    do i=1,3
      if(eigval(i)<1.D-8) then
        if(i>1) then
          ! the system is linear - no rotation necessay.
          ! WHY ?! There should still be one necessary!
          return
          !print*,"Eigenval. zero",i,eigval(i)
          !call dlf_fail("Error in dlf_cartesian_align")
        end if
        j=1
      else
        if(i==1.and.itry==2) then
          rsmat(:,i)=-1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
        else
          rsmat(:,i)=1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
        end if
      end if
    end do
    if(j==1) then
      ! one eigenvalue was zero, the system is planar
      rsmat(1,1)=rsmat(2,2)*rsmat(3,3)-rsmat(3,2)*rsmat(2,3)
      rsmat(2,1)=rsmat(3,2)*rsmat(1,3)-rsmat(1,2)*rsmat(3,3)
      rsmat(3,1)=rsmat(1,2)*rsmat(2,3)-rsmat(2,2)*rsmat(1,3)
      ! deal with negative determinant
      if (itry==2) then
         rsmat(:,1) = -rsmat(:,1)
      end if
    end if

    do i=1,3
      do j=1,3
        rotmat(i,j)=sum(rsmat(i,:)*eigvec(j,:))
      end do
    end do
    !write(*,"('rotmat ',3f10.3)") rotmat
    detrot=   &
        rotmat(1,1)*(rotmat(2,2)*rotmat(3,3)-rotmat(2,3)*rotmat(3,2)) &
        -rotmat(2,1)*(rotmat(1,2)*rotmat(3,3)-rotmat(1,3)*rotmat(3,2)) &
        +rotmat(3,1)*(rotmat(1,2)*rotmat(2,3)-rotmat(1,3)*rotmat(2,2))
    !write(*,*) "Determinat of rotmat", detrot
    if(detrot > 0.D0) exit
    if(detrot < 0.D0 .and. itry==2) then
      stop "Error in dlf_cartesian_align, obtained a mirroring instead of rotation."
    end if

  end do


!!$  do ivar=1,3
!!$    rsmat(:,ivar)=1.d0/dsqrt(eigval(ivar)) * matmul(rmat,eigvec(:,ivar))
!!$  end do
!!$
!!$  do ivar=1,3
!!$    do jvar=1,3
!!$      rotmat(ivar,jvar)=sum(rsmat(ivar,:)*eigvec(jvar,:))
!!$    end do
!!$  end do

  ! transform coordinates
  do iat=1,nat
    xcoords(iat*3-2:iat*3)= xcoords(iat*3-2:iat*3)-center
    xcoords(iat*3-2:iat*3)=matmul(rotmat,xcoords(iat*3-2:iat*3))
    xcoords(iat*3-2:iat*3)= xcoords(iat*3-2:iat*3)+center
  end do

!!$  ! write xyz
!!$  if(ttrain) then
!!$    fid=52
!!$  else
!!$    fid=53
!!$  end if
!!$  write(fid,*) nat
!!$  write(fid,*) 
!!$  do iat=1,nat
!!$    write(fid,'(" H ",3f12.7)') xcoords(iat*3-2:iat*3)*5.2917720810086D-01
!!$  end do
  
!  print*,"transformed coordinates"
!  write(*,'(3f15.5)') coords
  
  ! now all quantities have been transformed to c-coords (or relative to c-coordinates)

  ! now, the coordinates need to be mass-weighted!
  dcoords=matmul(transpose(align_modes),sqrt(weight)*(xcoords-align_refcoords))

end subroutine gpr_chemsh_fromdlf_cgh_xtos

subroutine gpr_chemsh_fromdlf_get_drotmat(ncoord,nvar,rcoords,xcoords_,drotmat)
  use gpr_in_chemsh_pes_module
  use gpr_chemsh_fromdlf_nn_module
  use gpr_chemsh_fromdlf_nn_av_module
  implicit none
  integer,intent(in) :: ncoord,nvar
  real(8),intent(in) :: rcoords(nvar) ! the set of coordinates the new ones should be fitted to
  real(8),intent(in) :: xcoords_(nvar) 
  real(8),intent(out) :: drotmat(3,3,nvar)
  real(8) :: trans(3),dcoords(ncoord),tmpmat(3,3)
  integer :: ivar
  real(8) :: delta=1.D-5
  real(8) :: xcoords(nvar)
  !print*,"FD rotmat with delta",delta
  do ivar=1,nvar
    xcoords=xcoords_
    xcoords(ivar)=xcoords(ivar)+delta
    call gpr_chemsh_fromdlf_cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,drotmat(:,:,ivar),dcoords)
    xcoords(ivar)=xcoords(ivar)-2.D0*delta
    call gpr_chemsh_fromdlf_cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat,dcoords)
    drotmat(:,:,ivar)=(drotmat(:,:,ivar)-tmpmat)/2.D0/delta
  end do
end subroutine gpr_chemsh_fromdlf_get_drotmat

subroutine gpr_chemsh_fromdlf_get_ddrotmat(ncoord,nvar,rcoords,xcoords_,ddrotmat)
  use gpr_in_chemsh_pes_module
  use gpr_chemsh_fromdlf_nn_module
  use gpr_chemsh_fromdlf_nn_av_module
  implicit none
  integer,intent(in) :: ncoord,nvar
  real(8),intent(in) :: rcoords(nvar) 
  real(8),intent(in) :: xcoords_(nvar) 
  real(8),intent(out) :: ddrotmat(3,3,nvar,nvar)
  real(8) :: trans(3),dcoords(ncoord),tmpmat(3,3),tmpmat2(3,3)
  integer :: ivar,jvar
  real(8) :: delta=1.D-5
  real(8) :: xcoords(nvar)

  xcoords=xcoords_
  ddrotmat=0.D0
  do ivar=1,nvar
    ! first the off-diagonal elements
    do jvar=ivar+1,nvar

      tmpmat=0.D0

      xcoords=xcoords_
      xcoords(ivar)=xcoords(ivar)+delta
      xcoords(jvar)=xcoords(jvar)+delta
      call gpr_chemsh_fromdlf_cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
      tmpmat=tmpmat+tmpmat2

      xcoords=xcoords_
      xcoords(ivar)=xcoords(ivar)-delta
      xcoords(jvar)=xcoords(jvar)+delta
      call gpr_chemsh_fromdlf_cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
      tmpmat=tmpmat-tmpmat2

      xcoords=xcoords_
      xcoords(ivar)=xcoords(ivar)+delta
      xcoords(jvar)=xcoords(jvar)-delta
      call gpr_chemsh_fromdlf_cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
      tmpmat=tmpmat-tmpmat2

      xcoords=xcoords_
      xcoords(ivar)=xcoords(ivar)-delta
      xcoords(jvar)=xcoords(jvar)-delta
      call gpr_chemsh_fromdlf_cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
      tmpmat=tmpmat+tmpmat2


      ddrotmat(:,:,ivar,jvar)=tmpmat/4.D0/delta**2
      ddrotmat(:,:,jvar,ivar)=ddrotmat(:,:,ivar,jvar)
    end do

    ! now the diagonal element
    tmpmat=0.D0
    
    xcoords=xcoords_
    xcoords(ivar)=xcoords(ivar)+delta
    call gpr_chemsh_fromdlf_cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
    tmpmat=tmpmat+tmpmat2
    
    xcoords=xcoords_
    !xcoords(ivar)=xcoords(ivar)
    call gpr_chemsh_fromdlf_cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
    tmpmat=tmpmat-2.D0*tmpmat2
    
    xcoords=xcoords_
    xcoords(ivar)=xcoords(ivar)-delta
    call gpr_chemsh_fromdlf_cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
    tmpmat=tmpmat+tmpmat2

    ddrotmat(:,:,ivar,ivar)=tmpmat/delta**2

  end do
end subroutine gpr_chemsh_fromdlf_get_ddrotmat


subroutine gpr_chemsh_fromdlf_coords_to_interatomic(nat,interatomic_dof,x_unchanged,barycentre,x_out,xx,&
inv_tf,barycentre_switch)
  use gpr_in_chemsh_pes_module, only: massweight,refmass
  implicit none
  integer, intent(in) :: nat,interatomic_dof
  real(8), intent(in) :: x_unchanged(3*nat),barycentre(3)
  logical, intent(in) :: inv_tf,barycentre_switch
  real(8), intent(out) :: x_out(interatomic_dof),xx(3*nat)
  integer, allocatable :: mapping(:,:),back_mapping(:,:)
  integer i,j,counter,inv

  if(.not.massweight)then
    xx=x_unchanged
  else
    do i=1,nat
      xx(3*i-2:3*i)=x_unchanged(3*i-2:3*i)/sqrt(refmass(i))
    enddo
  endif

  if(inv_tf)inv=-1
  if(.not.inv_tf)inv=1

  allocate(mapping(3*nat,3*nat))
  if(barycentre_switch)then
    allocate(back_mapping(2,interatomic_dof-nat))
  else
    allocate(back_mapping(2,interatomic_dof))
  endif
  mapping=-1
  
  counter=0
  do i=1,nat
    do j=i+1,nat
      counter=counter+1
      back_mapping(1,counter)=i
      back_mapping(2,counter)=j
      mapping(i,j)=counter
      mapping(j,i)=counter
      x_out(counter)=sqrt(sum((xx(3*i-2:3*i)-&
      xx(3*j-2:3*j))**2,dim=1))**inv
    enddo
  enddo
  if(barycentre_switch)then
    do i=1,nat
      counter=counter+1
      x_out(counter)=sqrt(sum((xx(3*i-2:3*i)-&
      barycentre)**2,dim=1))**inv
    enddo
  endif

  deallocate(mapping)
  deallocate(back_mapping)

end subroutine gpr_chemsh_fromdlf_coords_to_interatomic

subroutine gpr_chemsh_fromdlf_redundant_DMAT(nat,interatomic_dof,DMAT,x_out,x_unchanged,barycentre,&
refmass,coords_inverse,barycentre_switch)
  use gpr_chemsh_fromdlf_nn_module, only: radii_omit
  implicit none
  integer, intent(in) :: nat,interatomic_dof
  real(8), intent(out):: DMAT(interatomic_dof,3*nat)
  real(8), intent(in) :: x_unchanged(3*nat),x_out(interatomic_dof),barycentre(3),&
  refmass(nat)
  logical, intent(in) :: coords_inverse,barycentre_switch
  real(8), allocatable :: vector_diff(:,:)
  integer i,j,m,n,l,kk,ata,counter
  integer, allocatable :: mapping(:,:)
  real(8) rp(3),gpr_chemsh_fromdlf_kronecker,total_mass

  total_mass=sum(refmass,dim=1)
  allocate(mapping(nat,nat))
  mapping=-1
  counter=0
  kk=0
  do i=1,nat
    do j=i+1,nat
      counter=counter+1
      if(.not.any(radii_omit==counter))then
        kk=kk+1
        mapping(i,j)=kk
        mapping(j,i)=kk
      endif
    enddo
  enddo

  if(barycentre_switch)then
    allocate(vector_diff(3*(nat+1),nat+1))
  else
    allocate(vector_diff(3*nat,nat))
  endif
  DMAT=0.d0
  vector_diff=0.d0
  do m=1,nat
    do l=1,nat
      vector_diff(3*l-2:3*l,m)=x_unchanged(3*m-2:3*m)-&
      x_unchanged(3*l-2:3*l)
    enddo
  enddo
  if(barycentre_switch)then
    do l=1,nat
      m=nat+1
      vector_diff(3*l-2:3*l,m)=x_unchanged(3*l-2:3*l)-&
      barycentre(:)
    enddo
    do m=1,nat
      l=nat+1
      vector_diff(3*l-2:3*l,m)=x_unchanged(3*m-2:3*m)-&
      barycentre(:)
    enddo
  endif

  do m=1,nat
    do l=m+1,nat
      if(mapping(l,m).ne.-1)then
        if(coords_inverse)then
          DMAT(mapping(l,m),3*m-2:3*m)=-vector_diff(3*l-2:3*l,m)*&
          x_out(mapping(l,m))**3
        else
          DMAT(mapping(l,m),3*m-2:3*m)=vector_diff(3*l-2:3*l,m)/&
          x_out(mapping(l,m))
        endif
        DMAT(mapping(l,m),3*l-2:3*l)=-DMAT(mapping(l,m),3*m-2:3*m)
      endif
    enddo
  enddo
  if(barycentre_switch)then
    do m=interatomic_dof-nat+1,interatomic_dof
      do kk=1,3*nat
        ata=(kk+2)/3
        rp=0.d0
        do j=1,3
          rp(2)=0.d0
          do n=1,nat
            rp(2)=rp(2)+refmass(n)*gpr_chemsh_fromdlf_kronecker(3*n-3+j,kk)
          enddo
          rp(2)=rp(2)/total_mass
          rp(3)=gpr_chemsh_fromdlf_kronecker(3*ata-3+j,kk)-rp(2)
          rp(1)=rp(1)+(x_unchanged(3*ata-3+j)-barycentre(j))*rp(3)
        enddo
        if(coords_inverse)then
          rp(1)=-rp(1)*x_out(m)**3
        else
          rp(1)=rp(1)*x_out(m)**(-1)
        endif
        DMAT(m,kk)=rp(1)
      enddo
    enddo
  endif
  deallocate(mapping)
  deallocate(vector_diff)

end subroutine gpr_chemsh_fromdlf_redundant_DMAT

subroutine gpr_chemsh_fromdlf_DDMAT(nat,ncoord,x_unchanged,x_out,DMAT,coords_inverse,DM2)
  implicit none
  integer, intent(in) :: nat,ncoord
  real(8), intent(in) :: x_unchanged(3*nat),x_out(ncoord),DMAT(ncoord,3*nat)
  real(8), intent(out):: DM2(3*nat,3*nat,ncoord)
  logical, intent(in) :: coords_inverse
  real(8) rp,rp2(2),gpr_chemsh_fromdlf_kronecker
  integer i,l,ata,kk,atb,j

  do i=1,ncoord
    do l=1,3*nat
      ata=(l+2)/3
      do kk=1,3*nat
        atb=(kk+2)/3
        rp=0.d0
        rp2=0.d0
        do j=1,3
          rp2(1)=rp2(1)+(x_unchanged(3*ata-3+j)-x_unchanged(3*atb-3+j))*&
          (gpr_chemsh_fromdlf_kronecker(3*ata-3+j,l)-gpr_chemsh_fromdlf_kronecker(3*atb-3+j,l))
          rp2(2)=rp2(2)+(x_unchanged(3*ata-3+j)-x_unchanged(3*atb-3+j))*&
          (gpr_chemsh_fromdlf_kronecker(3*ata-3+j,kk)-gpr_chemsh_fromdlf_kronecker(3*atb-3+j,kk))
          rp=rp+(gpr_chemsh_fromdlf_kronecker(3*ata-3+j,l)-gpr_chemsh_fromdlf_kronecker(3*atb-3+j,l))*&
          (gpr_chemsh_fromdlf_kronecker(3*ata-3+j,kk)-gpr_chemsh_fromdlf_kronecker(3*atb-3+j,kk))
        enddo
        if(coords_inverse)then
!            DM2(l,kk,i)=(3.d0*x_out(i)**5)*(rp2(1)*rp2(2))-(x_out(i)**3)*rp
            DM2(l,kk,i)=(3.d0/x_out(i))*DMAT(i,l)*DMAT(i,kk)-x_out(i)**3*rp
        else
!            DM2(l,kk,i)=(-1.d0/x_out(i,k)**2)*(rp(1)*rp(2))+(1.d0/x_out(i,k))*rp(3)
          if(abs(x_out(i)).gt.1.d-8)then
            DM2(l,kk,i)=(1.d0/x_out(i))*(-DMAT(i,l)*DMAT(i,kk)+rp)
          endif
        endif
      enddo
    enddo
  enddo

end subroutine gpr_chemsh_fromdlf_DDMAT

subroutine gpr_chemsh_fromdlf_non_redundant_evect(evect,n,NR_size,short_evect)
  implicit none
  integer, intent(in) :: n,NR_size
  real(8), intent(in) :: evect(n,n)
  real(8), intent(out):: short_evect(n,NR_size)

  short_evect=evect(:,1:NR_size)

end subroutine gpr_chemsh_fromdlf_non_redundant_evect


real(8) function gpr_chemsh_fromdlf_kronecker(i,j)
  implicit none
  integer, intent(in) :: i,j

  if(i.eq.j)then
    gpr_chemsh_fromdlf_kronecker=1.d0
  else
    gpr_chemsh_fromdlf_kronecker=0.d0
  endif
  return

end function gpr_chemsh_fromdlf_kronecker

SUBROUTINE gpr_chemsh_fromdlf_mat_diag(N,H,E,U) 
  IMPLICIT NONE

  LOGICAL(4) ,PARAMETER :: TESSLERR=.FALSE.
  INTEGER   ,INTENT(IN) :: N
  REAL(8)   ,INTENT(IN) :: H(N,N)
  REAL(8)   ,INTENT(OUT):: E(N)
  REAL(8)   ,INTENT(OUT):: U(N,N)
  REAL(8)               :: WORK1((N*(N+1))/2)
  REAL(8)               :: WORK2(3*N)
  INTEGER               :: K,I,J
  INTEGER               :: INFO

  K=0
  DO J=1,N
    DO I=J,N
      K=K+1
      WORK1(K)=0.5D0*(H(I,J)+H(J,I))
    ENDDO
  ENDDO

  CALL dspev('V','L',N,WORK1,E,U,N,WORK2,INFO) !->LAPACK intel
  IF(INFO.NE.0) THEN
    PRINT*,'DIAGONALIZATION NOT CONVERGED (gpr_chemsh_fromdlf_mat_diag)'
    STOP
  END IF

END SUBROUTINE gpr_chemsh_fromdlf_mat_diag

subroutine gpr_chemsh_fromdlf_sort_smallest_first(n,array_in,sortlist)
  integer n,sortlist(n),i,j
  real(8) array_in(n),array_hold(n),array_in_max

  array_in_max=maxval(dabs(array_in))*2.d0
  array_hold=array_in
  do i=1,n
    j=minloc(dabs(array_hold),dim=1)
    sortlist(i)=j
    array_in(i)=array_hold(j)
    array_hold(j)=array_in_max
  enddo

  return
end subroutine gpr_chemsh_fromdlf_sort_smallest_first

subroutine gpr_chemsh_fromdlf_sort_largest_first(n,array_in,sortlist)
  integer n,sortlist(n),i,j,k
  real(8) array_in(n),array_hold(n),array_in_min

  array_in_min=minval(dabs(array_in))*0.5d0
  array_hold=array_in
  k=0 
  do i=1,n
    j=maxloc(dabs(array_hold),dim=1)
    sortlist(i)=j
    array_in(i)=array_hold(j)
    array_hold(j)=array_in_min
  enddo

  return
end subroutine gpr_chemsh_fromdlf_sort_largest_first

subroutine gpr_chemsh_fromdlf_vary_vector(x_in,nat,delta_x,delta_r,i_red)
  implicit none
  integer, intent(in) :: nat,i_red
  real(8), intent(in) :: x_in(3*nat)
  real(8), intent(out):: delta_x(3*nat),delta_r(i_red)
  real(8) rand_val,radii(i_red),x_var(3*nat)
  integer i,j,k

  k=0
  do i=1,nat
    do j=i+1,nat
      k=k+1
      radii(k)=sqrt(sum((x_in(3*i-2:3*i)-x_in(3*j-2:3*j))**2))
    enddo
  enddo
  
  do i=1,3*nat
    call random_seed()
    call random_number(rand_val)
    rand_val=2.d0*rand_val-1.d0
    rand_val=rand_val/1.d3
    delta_x(i)=x_in(i)*rand_val
    x_var(i)=x_in(i)+x_in(i)*rand_val
  enddo
  
  k=0
  do i=1,nat
    do j=i+1,nat
      k=k+1
      delta_r(k)=sqrt(sum((x_var(3*i-2:3*i)-x_var(3*j-2:3*j))**2))-radii(k)
    enddo
  enddo

end subroutine gpr_chemsh_fromdlf_vary_vector

!!****

! Module and routines for implementing a distance-based potential
! function called IDPP (image dependent pair potential) to
! pre-optimize NEB paths. The method is based on DOI
! 10.1063/1.4878664, Improved initial guess for minimum energy path
! calculations by Soren Smidstrup, Andreas Pedersen, Kurt Stokbro, and
! Hannes Jonsson, J. Chem. Phys. 140, 214106 (2014)
!
! One could move these routines to a separate file, but for the time
! being it remains here.
!
! Everything related to idpp was written by Matthias Bohner
! (bohner@theochem.uni-stuttgart.de)

module gpr_idpp
  use dlf_parameter_module, only: rk
  !holds the target distances
  !one gets target distances between ith and jth atom
  !by distances(nat*(i-1)-(i*(i+1))/2+j,iimage)
  logical :: tdist=.false.
  real(rk), allocatable, save :: distances(:,:) !(number of pairs,nimage)
end module gpr_idpp

subroutine gpr_idpp_get_distances(nvar,nvar2,coords,coords2,nspec,spec,nimage)
  use gpr_idpp
  use dlf_parameter_module, only: rk
  integer, intent(in) :: nvar
  integer, intent(in) :: nvar2
  real(rk), intent(in) :: coords(nvar)
  real(rk), intent(in) :: coords2(nvar2)
  integer, intent(in) :: nspec
  integer, intent(in) :: spec(nspec)
  integer, intent(in) :: nimage

  integer ::nstructs !number of given intermediate structures
  integer :: i, j, jvat, iimage, istruct, offset, tmp(1), aindex, bindex
  !interdist holds pair distances for the intermediate structures
  real(rk), allocatable :: interdist(:,:)
  !pathatstruct holds pathlength at given structures
  real(rk) ::  pathatstruct((nvar2-nspec)/nvar)
  !path_per_image: distance between path
  !path: pathvariable path=(iimage-1)*path_per_image
  !lambda interpolation parameter
  real(rk) :: path_per_image, path, lambda

  !If we have nspec atoms we have nspec*(nspec-1)/2 possible pairs
  !count(-1.eq.spec) is number of frozen atoms
  !Ergo nspec*(nspec-1)/2-count(-1.eq.spec)*(count(-1.eq.spec)-1)/2
  !relevant pairs
  offset=count(-1.eq.spec) !abusing offset
  allocate(distances(((nspec)*(nspec-1))/2-((offset)*(offset-1))/2,nimage-1))
  allocate(interdist(((nspec)*(nspec-1))/2-((offset)*(offset-1))/2,(nvar2-nspec)/nvar))
  !last nspec entries are for the atom masses
  nstructs=(nvar2-nspec)/nvar-1

  pathatstruct(1)=sqrt(sum((coords2(1:nvar)-coords)**2))
  do i=2,nstructs+1,1
    pathatstruct(i)=sqrt(sum((coords2(nvar*(i-1)+1:nvar*i)-coords2(nvar*(i-2)+1:nvar*(i-1)))**2))&
        +pathatstruct(i-1)
  enddo

  path_per_image=pathatstruct(nstructs+1)/dble(nimage-2)
  !Calculate all pair distances for the first structure and write it
  !into distances(:,1)
  jvat=0
  do i=1, nspec, 1
    do j=i+1, nspec ,1
      if((spec(i).eq.-1).and.(spec(j).eq.-1)) cycle
      jvat=jvat+1 !jvat counts all relevant pairs
      !<magic type=voodo>
      !      distances(nspec*(i-1)-(i*(i+1))/2+jvat,1)=&
      !<\magic>
      distances(jvat,1)=&
          sqrt(sum((coords(3*(i-1)+1:3*i)-coords(3*(j-1)+1:3*j))**2))
    enddo
  enddo


  !Calculate all pair distances for the intermediate structures
  !into interdist(:,nimage)
  do istruct=1,nstructs,1
    offset=nvar*(istruct-1)+1
    jvat=0
    do i=1, nspec, 1
      do j=i+1, nspec ,1
        if((spec(i).eq.-1).and.(spec(j).eq.-1)) cycle
        jvat=jvat+1 !jvat counts relevant pairs
        interdist(jvat,istruct)=&
            sqrt(sum((coords2(offset+3*(i-1):3*i+offset-1)&
            -coords2(offset+3*(j-1):3*j+offset-1))**2))
      enddo
    enddo
  enddo

  !Calculate all pair distances for the last structure and write it
  !into distances(:,nimage-1)
  offset=nvar2-nspec-nvar+1
  jvat=0
  do i=1, nspec, 1
    do j=i+1, nspec ,1
      if((spec(i).eq.-1).and.(spec(j).eq.-1)) cycle
      jvat=jvat+1
      distances(jvat,nimage-1)=&
          sqrt(sum((coords2(offset+3*(i-1):3*i+offset-1)&
          -coords2(offset+3*(j-1):3*j+offset-1))**2))
    enddo
  enddo


  !linear interpolations of pair-distances between given structures
  if(nstructs==0) then
    do iimage=2, nimage-2,1
      lambda=dble(iimage-1)*path_per_image/pathatstruct(nstructs+1)
      distances(:,iimage)=(1D0-lambda)*distances(:,1)+lambda*distances(:,nimage-1)
    enddo
  else
    do iimage=2,nimage-2,1
      !Determining where image is located along the path
      path=dble(iimage-1)*path_per_image
      tmp=maxloc(pathatstruct, pathatstruct<path)
      aindex=tmp(1)
      tmp=minloc(pathatstruct, pathatstruct>path)
      bindex=tmp(1)

      if(0==aindex) then
        lambda=path/pathatstruct(1)
        distances(:,iimage)=(1D0-lambda)*distances(:,1)+lambda*interdist(:,1)
        cycle
      endif
      lambda=(path-pathatstruct(aindex))/(pathatstruct(bindex)-pathatstruct(aindex))

      if(bindex==nstructs+1) then
        distances(:,iimage)=(1D0-lambda)*interdist(:,aindex)+lambda*distances(:,nimage-1)
        cycle
      endif

      distances(:,iimage)=(1D0-lambda)*interdist(:,aindex)+lambda*interdist(:,bindex)

    end do
  endif
  tdist=.true.
end subroutine gpr_idpp_get_distances

subroutine gpr_idpp_destroy()
  use gpr_idpp
  if(.not.tdist) then
    STOP ("gpr_idpp_get_distances must be called before gpr_idpp_destroy")
  end if
  deallocate(distances)
end subroutine gpr_idpp_destroy

subroutine gpr_get_idpp_hessian(nvar,coords,hessian,status,iimage)
  !  get the IDPP hessian at a given geometry
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use gpr_idpp
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: hessian(nvar,nvar)
  integer   ,intent(out)   :: status
  integer   ,intent(in), optional :: iimage
  real(rk) :: svar,svar2
  integer  :: iat,jat,nat,jvat
  real(rk) :: dvec(3), dyadic(3,3), identy(3,3),aux(3,3)
  ! **********************************************************************

  if(.not.tdist) then
    STOP ("gpr_idpp_get_distances must be called before gpr_get_idpp_hessian")
  end if

  hessian(:,:)=0.D0
  status=1

  nat=nvar/3
  hessian=0D0
  identy=0D0
  forall(iat=1:3) identy(iat,iat)=1.D0
  jvat=0
  do iat=1,nat,1
    do jat=iat+1,nat,1
      if((glob%spec(iat).eq.-1).and.(glob%spec(jat).eq.-1)) cycle
      jvat=jvat+1
      dvec=coords(3*(iat-1)+1:3*iat)-coords(3*(jat-1)+1:3*jat)
      !svar holds the actual distance
      svar=sqrt(dot_product(dvec,dvec))
      dvec=dvec/svar
      !svar2 holds the target distance
      svar2=distances(jvat,iimage)
      !dyadic product of the distance unity vector
      dyadic=spread(dvec,dim=2,ncopies=3)*spread(dvec,dim=1,ncopies=3)

      aux=2D0*svar**(-4)*((12D0*svar**(-2)*(svar2-svar)**2+9D0/svar*(svar2-svar)+1D0&
          )*dyadic-(2D0*svar**(-2)*(svar2-svar)**2+(svar2-svar)*svar**(-1))*identy)


      ! aux=2D0*(svar**(-4))*(((8D0*(svar**(-2))*(svar2-svar)+7D0/svar)*(svar2-svar)+1&
      !      )*dyadic-(2D0*(svar**(-2))*(svar2-svar)+1D0/svar)*(svar2-svar)*identy)

      !+delta_l^k part
      hessian(3*(iat-1)+1:3*iat,3*(iat-1)+1:3*iat)=hessian(3*(iat-1)+1:3*iat,3*(iat-1)+1:3*iat)&
          +aux
      hessian(3*(jat-1)+1:3*jat,3*(jat-1)+1:3*jat)=hessian(3*(jat-1)+1:3*jat,3*(jat-1)+1:3*jat)&
          +aux
      !-delta_l^j part
      hessian(3*(iat-1)+1:3*iat,3*(jat-1)+1:3*jat)=hessian(3*(iat-1)+1:3*iat,3*(jat-1)+1:3*jat)&
          -aux
      hessian(3*(jat-1)+1:3*jat,3*(iat-1)+1:3*iat)=hessian(3*(jat-1)+1:3*jat,3*(iat-1)+1:3*iat)&
          -aux

    enddo
  enddo
  status=0

end subroutine gpr_get_idpp_hessian

subroutine gpr_get_idpp_gradient(nvar,coords,energy,gradient,iimage,status)
  ! calculate an idpp gradient for a given geometry
  use dlf_global, only: glob
  use dlf_parameter_module, only: rk
  use gpr_idpp
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: energy
  real(rk)  ,intent(out)   :: gradient(nvar)
  integer   ,intent(in)    :: iimage
  integer   ,intent(out)   :: status
  !
  real(rk) :: svar,svar2
  integer  :: nat,iat,jat,jvat
  ! additional variables non-cont. diff MEP potential
  !variables for the initial path construction
  ! The Journal of Chemical Physics 140, 214106 (2014); doi: 10.1063/1.4878664
  real(rk) :: aux
  ! **********************************************************************

  if(.not.tdist) then
    STOP ("gpr_idpp_get_distances must be called before gpr_get_idpp_gradient")
  end if

  !  call test_update
  status=1
  nat=nvar/3
  energy=0D0
  gradient=0D0
  jvat=0
  do iat=1,nat,1
    do jat=iat+1,nat,1
      if((glob%spec(iat).eq.-1).and.(glob%spec(jat).eq.-1)) cycle
      jvat=jvat+1
      !svar holds the actual distance
      svar=sqrt(sum((coords(3*(iat-1)+1:3*iat)-coords(3*(jat-1)+1:3*jat))**2))
      !svar2 holds desired value
      svar2=distances(jvat,iimage)
      energy=energy+(svar2-svar)**2*svar**(-4)

      aux=-2D0*svar**(-4)*(svar2-svar)*(2D0*svar**(-1)*(svar2-svar)+1D0)
      !atomnumbers
      gradient(3*(iat-1)+1:3*iat)=gradient(3*(iat-1)+1:3*iat)&
          +aux*(coords(3*(iat-1)+1:3*iat)-coords(3*(jat-1)+1:3*jat))/svar

      gradient(3*(jat-1)+1:3*jat)=gradient(3*(jat-1)+1:3*jat)&
          -aux*(coords(3*(iat-1)+1:3*iat)-coords(3*(jat-1)+1:3*jat))/svar
    end do
  end do
  !  print*, iimage, sqrt(sum(gradient**2))
  status=0

end subroutine gpr_get_idpp_gradient

subroutine gpr_read_qts_coords_init(nat,nimage,varperimage)
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout,printl,glob
  implicit none
  integer, intent(out)   :: nat
  integer, intent(out)   :: nimage
  integer, intent(out)   :: varperimage
  character(128)         :: filename
  logical                :: there
  filename='qts_coords.txt'
  print*, "Reading from file ",trim(filename)
  if (glob%ntasks > 1) filename="../"//trim(filename)

  inquire(file=filename,exist=there)
  if(.not.there) STOP "qts_coords.txt does not exist! Start structure&
      & for qts hessian is missing."

  open(unit=555,file=filename, action='read')
  read(555,FMT='(a)',end=201,err=200)
  read(555,*,end=201,err=200) nat,nimage,varperimage
  close(555)
  return
  
200 continue
  STOP "Error reading qts_coords.txt file"
  print*, "Error reading file"
  return
201 continue
  STOP "Error (EOF) reading qts_coords.txt file"
  print*, "Error (EOF) reading file"
  return
end subroutine gpr_read_qts_coords_init

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/gpr_read_qts_coords
!!
!! FUNCTION
!!
!! Read coordinates, dtau, etunnel, and dist from qts_coords.txt
!!
!! SYNOPSIS
subroutine gpr_read_qts_coords(nat,nimage,varperimage,temperature,&
    S_0,S_pot,S_ins,ene,xcoords,dtau,etunnel,dist)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout,printl,glob
  implicit none
  integer, intent(in)    :: nat
  integer, intent(in)    :: nimage
  integer, intent(in)    :: varperimage
  real(rk),intent(inout) :: temperature
  real(rk),intent(out)   :: S_0,S_pot,S_ins
  real(rk),intent(out)   :: ene(nimage)
  real(rk),intent(out)   :: xcoords(3*nat,nimage)
  real(rk),intent(out)   :: dtau(nimage+1)
  real(rk),intent(out)   :: etunnel
  real(rk),intent(out)   :: dist(nimage+1)
  !
  logical :: there
  integer :: nat_,nimage_,varperimage_,ios
  character(128) :: line,filename

  ene=0.D0
  xcoords=0.D0

  filename='qts_coords.txt'
!   print*, "Reading from file ",filename
  if (glob%ntasks > 1) filename="../"//trim(filename)

  inquire(file=filename,exist=there)
  if(.not.there) STOP "qts_coords.txt does not exist! Start structure&
      & for qts hessian is missing."

  open(unit=555,file=filename, action='read')
  read(555,FMT='(a)',end=201,err=200) 
  read(555,*,end=201,err=200) nat_,nimage_,varperimage_
  if(nat/=nat_) STOP "Error reading qts_coords.txt file: Number of &
      &atoms not consistent"
  
  ! test of varperimage commented out. I don't think it should be a problem if that changes
  !if(varperimage/=varperimage_) call dlf_fail("Error reading qts_coords.txt file: Variables &
  !    &per image not consistent")

  read(555,*,end=201,err=200) temperature
  !read(555,*,end=201,err=200) S_0 
  read(555,fmt="(a)") line
  read(line,*,iostat=ios) S_0,S_pot
  if(ios/=0) then
    read(line,*) S_0
  end if
  read(555,*,end=201,err=200) S_ins
  if(ios/=0) then
    S_pot=S_ins-0.5D0*S_0
    if(printl>=2) print*, "Warning: could not read S_pot from qts_coords.txt"
  end if
  read(555,*,end=201,err=200) ene(1:nimage)
  read(555,*,end=201,err=200) xcoords(1:3*nat,1:nimage)
  ! try and read dtau (not here in old version, and we have to stay consistent)
  read(555,fmt="(a)",iostat=ios) line
  if(ios==0) then
    read(555,*,end=201,err=200) dtau(1:1+nimage)
    read(555,*,end=201,err=200) etunnel
    read(555,*,end=201,err=200) dist(1:1+nimage)
  else
    if(printl>=2) print*, "Warning, dtau not read from qts_coords.txt, using constant dtau"
    dtau=-1.D0
    etunnel=-1.D0  
    dist(:)=-1.D0 ! set to useless value to flag that it was not read
  end if

  close(555)

  if(printl >= 4) print*, "('qts_coords.txt file successfully read')"

  return

  ! return on error
200 continue
  STOP "Error reading qts_coords.txt file"
  print*, "Error reading file"
  return
201 continue
  STOP "Error (EOF) reading qts_coords.txt file"
  print*, "Error (EOF) reading file"
  return

10 format("Checkpoint reading WARNING: ",a) 
end subroutine gpr_read_qts_coords
!!****


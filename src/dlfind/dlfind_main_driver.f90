
module driver_parameter_module
  use dlf_parameter_module
  ! variables for Mueller-Brown potential
  real(rk) :: acappar(4),apar(4),bpar(4),cpar(4),x0par(4),y0par(4)
  ! variables for Lennard-Jones potentials
  real(rk),parameter :: epsilon=1.D-1
  real(rk),parameter :: sigma=1.D0
!!$  ! variables for the Eckart potential (1D), taken from Andri's
!thesis
  real(rk),parameter :: Va= -0.191D0*0.0367493254D0 !2nd factor is conversion to E_h from eV
  real(rk),parameter :: Vb=  1.343D0*0.0367493254D0 !2nd factor is conversion to E_h from eV
  real(rk),parameter :: alpha= 5.762D0/1.889725989D0 !2nd factor is for conversion from Angstrom to Bohr (a.u.)

  ! modified Eckart potential to have a comparison
!!$  REAL(8), PARAMETER :: Va= -0.091D0*0.0367493254D0  !2nd factor is
!conversion to E_h from eV
!!$  REAL(8), PARAMETER :: Vb=  1.343D0*0.0367493254D0  !2nd factor is
!conversion to E_h from eV
!!$  REAL(8), PARAMETER :: alpha =20.762D0/1.889725989D0 
!!$  real(rk),parameter :: Va= -0.191D0*0.0367493254D0 !2nd factor is
!conversion to E_h from eV
!!$  real(rk),parameter :: Vb=  1.343D0*0.0367493254D0 !2nd factor is
!conversion to E_h from eV
!!$  real(rk),parameter :: alpha= 20.762D0/1.889725989D0 !2nd factor is
!for conversion from Angstrom to Bohr (a.u.)

  ! "fitted" to Glum SN-1-Glu, E->D
!  real(rk),parameter :: Va= 0.D0 ! symmetric (real: almost symmetric)
!  real(rk),parameter :: Vb= 4.D0*74.81D0/2625.5D0 ! 4*barrier for
!  symmetric
!  real(rk),parameter :: alpha= 2.38228D0 

!!$  ! Eckart for comparison with scattering
!!$  real(rk),parameter :: Va= -100.D0
!!$  real(rk),parameter :: Vb= 373.205080756888D0
!!$  real(rk),parameter :: alpha= 7.07106781186547D0

  real(rk) :: xvar,yvar
  ! quartic potential
  real(rk),parameter :: V0=1.D0
  real(rk),parameter :: X0=5.D0
  ! polymul
  integer ,parameter :: num_dim=3 ! should be a multiple of 3
  real(rk),parameter :: Ebarr=80.D0/2526.6D0  ! >0
  real(rk) :: dpar(num_dim)
!!$  real(rk),parameter :: vdamp=15.D0 ! change of frequencies towards
!TS. +inf -> no change
end module driver_parameter_module

! **********************************************************************
! subroutines that have to be provided to dl_find from outside
! **********************************************************************

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_params(nvar,nvar2,nspec,coords,coords2,spec,ierr, &
    tolerance,printl,maxcycle,maxene,tatoms,icoord, &
    iopt,iline,maxstep,scalestep,lbfgs_mem,nimage,nebk,dump,restart,&
    nz,ncons,nconn,update,maxupd,delta,soft,inithessian,carthessian,tsrel, &
    maxrot,tolrot,nframe,nmass,nweight,timestep,fric0,fricfac,fricp, &
    imultistate, state_i,state_j,pf_c1,pf_c2,gp_c3,gp_c4,ln_t1,ln_t2, &
    printf,tolerance_e,distort,massweight,minstep,maxdump,task,temperature, &
    po_pop_size,po_radius,po_contraction,po_tolerance_r,po_tolerance_g, &
    po_distribution,po_maxcycle,po_init_pop_size,po_reset,po_mutation_rate, &
    po_death_rate,po_scalefac,po_nsave,ntasks,tdlf_farm,n_po_scaling, &
    neb_climb_test, neb_freeze_test, &
    nzero, coupled_states, qtsflag,&
    imicroiter, maxmicrocycle, micro_esp_fit)
  use driver_parameter_module
  use dlf_parameter_module, only: rk
  use quick_molspec_module, only: natom, xyz, quick_molspec
  use quick_constants_module, only: EMASS
  use quick_method_module,only: quick_method

  !use vib_pot
  implicit none
  integer   ,intent(in)      :: nvar 
  integer   ,intent(in)      :: nvar2
  integer   ,intent(in)      :: nspec
  real(rk)  ,intent(inout)   :: coords(nvar) ! start coordinates
  real(rk)  ,intent(inout)   :: coords2(nvar2) ! a real array that can be used
                                               ! depending on the calculation
                                               ! e.g. a second set of coordinates
  integer   ,intent(inout)   :: spec(nspec)  ! specifications like fragment or frozen
  integer   ,intent(out)     :: ierr
  real(rk)  ,intent(inout)   :: tolerance
  real(rk)  ,intent(inout)   :: tolerance_e
  integer   ,intent(inout)   :: printl
  integer   ,intent(inout)   :: maxcycle
  integer   ,intent(inout)   :: maxene
  integer   ,intent(inout)   :: tatoms
  integer   ,intent(inout)   :: icoord
  integer   ,intent(inout)   :: iopt
  integer   ,intent(inout)   :: iline
  real(rk)  ,intent(inout)   :: maxstep
  real(rk)  ,intent(inout)   :: scalestep
  integer   ,intent(inout)   :: lbfgs_mem
  integer   ,intent(inout)   :: nimage
  real(rk)  ,intent(inout)   :: nebk
  integer   ,intent(inout)   :: dump
  integer   ,intent(inout)   :: restart
  integer   ,intent(inout)   :: nz
  integer   ,intent(inout)   :: ncons
  integer   ,intent(inout)   :: nconn
  integer   ,intent(inout)   :: update
  integer   ,intent(inout)   :: maxupd
  real(rk)  ,intent(inout)   :: delta
  real(rk)  ,intent(inout)   :: soft
  integer   ,intent(inout)   :: inithessian
  integer   ,intent(inout)   :: carthessian
  integer   ,intent(inout)   :: tsrel
  integer   ,intent(inout)   :: maxrot
  real(rk)  ,intent(inout)   :: tolrot
  integer   ,intent(inout)   :: nframe
  integer   ,intent(inout)   :: nmass
  integer   ,intent(inout)   :: nweight
  real(rk)  ,intent(inout)   :: timestep
  real(rk)  ,intent(inout)   :: fric0
  real(rk)  ,intent(inout)   :: fricfac
  real(rk)  ,intent(inout)   :: fricp
  integer   ,intent(inout)   :: imultistate
  integer   ,intent(inout)   :: state_i
  integer   ,intent(inout)   :: state_j
  real(rk)  ,intent(inout)   :: pf_c1  
  real(rk)  ,intent(inout)   :: pf_c2  
  real(rk)  ,intent(inout)   :: gp_c3  
  real(rk)  ,intent(inout)   :: gp_c4
  real(rk)  ,intent(inout)   :: ln_t1  
  real(rk)  ,intent(inout)   :: ln_t2  
  integer   ,intent(inout)   :: printf
  real(rk)  ,intent(inout)   :: distort
  integer   ,intent(inout)   :: massweight
  real(rk)  ,intent(inout)   :: minstep
  integer   ,intent(inout)   :: maxdump
  integer   ,intent(inout)   :: task
  real(rk)  ,intent(inout)   :: temperature
  integer   ,intent(inout)   :: po_pop_size
  real(rk)  ,intent(inout)   :: po_radius
  real(rk)  ,intent(inout)   :: po_contraction
  real(rk)  ,intent(inout)   :: po_tolerance_r
  real(rk)  ,intent(inout)   :: po_tolerance_g
  integer   ,intent(inout)   :: po_distribution
  integer   ,intent(inout)   :: po_maxcycle
  integer   ,intent(inout)   :: po_init_pop_size
  integer   ,intent(inout)   :: po_reset
  real(rk)  ,intent(inout)   :: po_mutation_rate
  real(rk)  ,intent(inout)   :: po_death_rate
  real(rk)  ,intent(inout)   :: po_scalefac
  integer   ,intent(inout)   :: po_nsave
  integer   ,intent(inout)   :: ntasks
  integer   ,intent(inout)   :: tdlf_farm
  integer   ,intent(inout)   :: n_po_scaling
  real(rk)  ,intent(inout)   :: neb_climb_test
  real(rk)  ,intent(inout)   :: neb_freeze_test
  integer   ,intent(inout)   :: nzero
  integer   ,intent(inout)   :: coupled_states
  integer   ,intent(inout)   :: qtsflag
  integer   ,intent(inout)   :: imicroiter
  integer   ,intent(inout)   :: maxmicrocycle
  integer   ,intent(inout)   :: micro_esp_fit
  ! local variables
  real(rk)                   :: svar
  integer                    :: i, iat,jat,mpierror
  character ::c
! **********************************************************************
  ierr=0
  tsrel=1

  nz         = quick_molspec%natom
  nmass      = quick_molspec%natom
  ncons      = 0
  nconn      = 0
  nzero      = 0 
  nframe     = 0
  coords(:)  =-1.D0
  spec(:)    =0
  coords2(:) =-1.D0

   do iat = 1, nvar2
     do jat = 1, 3
        coords((iat-1)*3 + jat) = xyz(jat, iat)
    enddo 
  enddo
 
  spec(1+nz:nz+nz) = quick_molspec%iattype(:)

  do iat = 1, quick_molspec%natom
    coords2(iat) = EMASS(quick_molspec%iattype(iat))
  enddo

!*************END case of isystem checking******************

  tolerance=quick_method%gradMaxCrt ! negative: default settings
  tolerance_e =quick_method%EChange
  printl=4
  printf=4
  maxcycle=100 !200
  maxene=100000

  tolrot=1.D2
!  tolrot=0.1D0
!  maxrot=100 !was 100

  task=0 !1011

  distort=0.D0 !0.4 
  tatoms=0
  icoord=quick_method%dlfind_icoord !0 cartesian coord !210 Dimer !190 qts search !120 NEB frozen endpoint
  massweight=0

! TO DO (urgent): better dtau for endpoints (interpolate energy) when reading dist

  imicroiter=0 ! means: use microiterative optimization

  iopt=quick_method%dlfind_iopt ! 20 !3 or 20 later change to 12
  temperature = 2206.66844626D0*0.99D0 ! K ! T_c for the MB-potential for hydrogen is 2206.668 K
  !temperature = 0.8D0*333.40829D0  ! K ! T_c 20102.83815 K
  !temperature = 500.D0
  ! Cubic potential (isystem=5: Crossover temperature for tunnelling  1846563.59447 K)
  temperature = 0.99D0 * 1846563.59447D0
  ! eckart potential (scatter) Tc: 3186172.17493753 K
  temperature = 0.9D0*3186172.17493753D0
  temperature= 0.7D0 * 275.13301D0 
! T_c for 162 DOF: 511.06618

  iline=0
  maxstep=0.1D0
  scalestep=1.0D0
  lbfgs_mem=100
  nimage=39 !*k-k+1
  nebk=1.0D0 ! for QTS calculations with variable tau, nebk transports the parameter alpha (0=equidist. in tau)
  qtsflag=0 ! 1: Tunneling splittings, 11/10: read image hessians

  !"accurate" etunnel for 0.4 Tc: 0.3117823831234522

  ! Hessian
  delta=1.D-2
  soft=-6.D-4
  update=2
  maxupd=0
  inithessian=0
  minstep=1.D0 **2 ! 1.D-5
  minstep=1.D-5

!!$  ! variables for exchange by sed
!!$  iopt=IOPT_VAR
!!$  nimage=NIMAGE_VAR
!!$  temperature = TFAC ! K ! T_c for the MB-potential for hydrogen is 2206.668 K
!!$  nebk= NEBK_VAR


  ! damped dynamics
  fric0=0.1D0
  fricfac=1.0D0
  fricp=0.1D0

  dump=0
  restart=0

  ! Parallel optimization

  po_pop_size=25
  po_radius=0.5D0
  po_init_pop_size=50
  po_contraction=0.95D0
  po_tolerance_r=1.0D-8
  po_tolerance_g=1.0D-6
  po_distribution=3
  po_maxcycle=100000
  po_reset=500
  po_mutation_rate=0.15D0
  po_death_rate=0.5D0
  po_scalefac=10.0D0
  po_nsave=10
  n_po_scaling=0 ! meaning that the base radii values for the sampling and tolerance 
                 ! are used for all components of the working coordinate vector.
                 ! Remember to change the second arg in the call to dl_find if a 
                 ! non-zero value for n_po_scaling is desired, and also to add the 
                 ! necessary values to the coords2 array...

  ! Taskfarming 
  ! (the two lines below, with any values assigned, 
  ! may be safely left in place for a serial build) 
  ntasks = 1
  tdlf_farm = 1

  tatoms=1
  
!  call test_ene

end subroutine dlf_get_params


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_gradient(nvar,coords,energy,gradient,iimage,kiter,status,ierr)
  !  Mueller-Brown Potential
  !  see K Mueller and L. D. Brown, Theor. Chem. Acta 53, 75 (1979)
  !  taken from JCP 111, 9475 (1999)
  use driver_parameter_module
  use dlf_parameter_module, only: rk
  use dlf_stat, only: stat
  use allmod
  use quick_gridpoints_module
  use quick_molspec_module, only: natom, xyz, quick_molspec
  use quick_cshell_gradient_module, only: scf_gradient
  use quick_cutoff_module, only: schwarzoff
  use quick_cshell_eri_module, only: getEriPrecomputables
  use quick_cshell_gradient_module, only: scf_gradient
  use quick_oshell_gradient_module, only: uscf_gradient
  use quick_method_module,only: quick_method
  use quick_exception_module, only: RaiseException 
#ifdef MPIV
  use quick_mpi_module, only: master
#endif
  !use vib_pot
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: energy
  real(rk)  ,intent(out)   :: gradient(nvar)
  integer   ,intent(in)    :: iimage
  integer   ,intent(in)    :: kiter
  integer   ,intent(out)   :: status
  integer, intent(inout) :: ierr

#ifdef MPIV
   include "mpif.h" 
#endif

#ifdef MPIV
#define CHECK_ERR if(master .and. ierr /= 0)
#else
#define CHECK_ERR if(ierr /= 0)
#endif
#define CHECK_ERROR(ierr) CHECK_ERR call RaiseException(ierr) 

  !
! **********************************************************************
!  call test_update
  status=1

#if defined CUDA || defined CUDA_MPIV                                                             
  call gpu_setup(natom,nbasis, quick_molspec%nElec, quick_molspec%imult, &                  
              quick_molspec%molchg, quick_molspec%iAtomType)                                      
  call gpu_upload_xyz(xyz)                                                                  
  call gpu_upload_atom_and_chg(quick_molspec%iattype, quick_molspec%chg)                    
#endif                                                                                            

  call getEriPrecomputables
  call schwarzoff

#if defined CUDA || defined CUDA_MPIV                                                             
  call gpu_upload_basis(nshell, nprim, jshell, jbasis, maxcontract, &                       
        ncontract, itype, aexp, dcoeff, &                                                   
        quick_basis%first_basis_function, quick_basis%last_basis_function,&                
        quick_basis%first_shell_basis_function,quick_basis%last_shell_basis_function,&     
        quick_basis%ncenter, quick_basis%kstart, quick_basis%katom, &                       
        quick_basis%ktype, quick_basis%kprim,quick_basis%kshell,quick_basis%Ksumtype, &    
        quick_basis%Qnumber, quick_basis%Qstart,quick_basis%Qfinal,quick_basis%Qsbasis, quick_basis%Qfbasis, &                                                                               
        quick_basis%gccoeff, quick_basis%cons, quick_basis%gcexpo, quick_basis%KLMN)        
                                                                                                  
  call gpu_upload_cutoff_matrix(Ycutoff, cutPrim)                                           

        call gpu_upload_oei(quick_molspec%nExtAtom, quick_molspec%extxyz, quick_molspec%extchg, ierr)
                                                                                                  
#ifdef CUDA_MPIV                                                                                  
  timer_begin%T2elb = timer_end%T2elb                                                         
  call mgpu_get_2elb_time(timer_end%T2elb)                                                    
  timer_cumer%T2elb = timer_cumer%T2elb+timer_end%T2elb-timer_begin%T2elb                     
#endif                                                                                            
                                                                                                  
#endif                                                                                            

  call getEnergy(.false., ierr)

  if (quick_method%analgrad) then
     if (quick_method%UNRST) then
        if (.not. quick_method%uscf_conv .and. .not. quick_method%allow_bad_scf) then
           ierr=33
           CHECK_ERROR(ierr)
        endif
        CALL uscf_gradient
     else
        if (.not. quick_method%scf_conv .and. .not. quick_method%allow_bad_scf) then 
           ierr=33
           CHECK_ERROR(ierr)
        endif
        CALL scf_gradient
     endif
  endif

#if defined CUDA || defined CUDA_MPIV
  if (quick_method%bCUDA) then
     call gpu_cleanup()
  endif
#endif  

  energy   = quick_qm_struct%Etot
  gradient = quick_qm_struct%gradient

if (quick_method%DFT) then
     if(stat%ccycle .le.quick_method%iopt) then
          call deform_dft_grid(quick_dft_grid)
     endif
endif

  status=0
end subroutine dlf_get_gradient

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_hessian(nvar,coords,hessian,status)
  !  get the hessian at a given geometry
  use driver_parameter_module
  use dlf_parameter_module
!  use allmod
!  use quick_cshell_gradient_module, only: cshell_gradient
  !use vib_pot
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: hessian(nvar,nvar)
  integer   ,intent(out)   :: status
  real(rk) :: acoords(3,nvar/3),r,svar,svar2
  integer  :: posi,posj,iat,jat,m,n
  ! variables for Mueller-Brown potential
  !real(rk) :: x,y
  integer  :: icount
  ! variables non-cont. diff. potential
  real(rk) :: t
  logical :: failed = .false. 
! **********************************************************************
  hessian(:,:)=0.D0
  status = 1

    acoords=reshape(coords,(/3,nvar/3/))
    do iat=1,nvar/3
      do jat=iat+1,nvar/3
        r=sum((acoords(:,iat)-acoords(:,jat))**2)
        ! Lennard-Jones Potential
        svar = 96.D0*epsilon * (7.D0*sigma**12/r**8-2.D0*sigma**6/r**5) ! coeffof x1x2
        svar2= epsilon * (-2.D0*sigma**12/r**7+sigma**6/r**4) ! for x1x1
        posi=(iat-1)*3+1
        posj=(jat-1)*3+1
        ! off-diag
        hessian(posi,posi+1)  =hessian(posi,posi+1)  + svar *(acoords(1,iat)-acoords(1,jat)) * (acoords(2,iat)-acoords(2,jat))
        hessian(posi,posi+2)  =hessian(posi,posi+2)  + svar *(acoords(1,iat)-acoords(1,jat)) * (acoords(3,iat)-acoords(3,jat))
        hessian(posi+1,posi)  =hessian(posi+1,posi)  + svar *(acoords(2,iat)-acoords(2,jat)) * (acoords(1,iat)-acoords(1,jat))
        hessian(posi+1,posi+2)=hessian(posi+1,posi+2)+ svar *(acoords(2,iat)-acoords(2,jat)) * (acoords(3,iat)-acoords(3,jat))
        hessian(posi+2,posi)  =hessian(posi+2,posi)  + svar *(acoords(3,iat)-acoords(3,jat)) * (acoords(1,iat)-acoords(1,jat))
        hessian(posi+2,posi+1)=hessian(posi+2,posi+1)+ svar *(acoords(3,iat)-acoords(3,jat)) * (acoords(2,iat)-acoords(2,jat))

        do m=0,2
          do n=0,2
            if(m==n) cycle
  hessian(posi+m,posj+n)=hessian(posi+m,posj+n)- svar *(acoords(M+1,iat)-acoords(M+1,jat)) * (acoords(N+1,iat)-acoords(N+1,jat))
  hessian(posj+m,posi+n)=hessian(posj+m,posi+n)- svar *(acoords(M+1,iat)-acoords(M+1,jat)) * (acoords(N+1,iat)-acoords(N+1,jat))
          end do
        end do
        ! Diag for different atoms ...
        do m=0,2
          hessian(posi+m,posj+m)=hessian(posi+m,posj+m) -24.D0*(svar2+1.D0/24.D0*svar* (acoords(m+1,iat)-acoords(M+1,jat))**2)
          hessian(posj+m,posi+m)=hessian(posj+m,posi+m) -24.D0*(svar2+1.D0/24.D0*svar* (acoords(m+1,iat)-acoords(M+1,jat))**2)
        end do

        hessian(posj,posj+1)  =hessian(posj,posj+1)  + svar * (acoords(1,iat)-acoords(1,jat)) * (acoords(2,iat)-acoords(2,jat))
        hessian(posj,posj+2)  =hessian(posj,posj+2)  + svar * (acoords(1,iat)-acoords(1,jat)) * (acoords(3,iat)-acoords(3,jat))
        hessian(posj+1,posj)  =hessian(posj+1,posj)  + svar * (acoords(2,iat)-acoords(2,jat)) * (acoords(1,iat)-acoords(1,jat))
        hessian(posj+1,posj+2)=hessian(posj+1,posj+2)+ svar * (acoords(2,iat)-acoords(2,jat)) * (acoords(3,iat)-acoords(3,jat))
        hessian(posj+2,posj)  =hessian(posj+2,posj)  + svar * (acoords(3,iat)-acoords(3,jat)) * (acoords(1,iat)-acoords(1,jat))
        hessian(posj+2,posj+1)=hessian(posj+2,posj+1)+ svar * (acoords(3,iat)-acoords(3,jat)) * (acoords(2,iat)-acoords(2,jat))
        ! diag
        hessian(posi,posi)    =hessian(posi,posi)    + 24.D0*(svar2+1.D0/24.D0*svar* (acoords(1,iat)-acoords(1,jat))**2)
        hessian(posi+1,posi+1)=hessian(posi+1,posi+1)+ 24.D0*(svar2+1.D0/24.D0*svar* (acoords(2,iat)-acoords(2,jat))**2)
        hessian(posi+2,posi+2)=hessian(posi+2,posi+2)+ 24.D0*(svar2+1.D0/24.D0*svar* (acoords(3,iat)-acoords(3,jat))**2)

        hessian(posj,posj)    =hessian(posj,posj)    + 24.D0*(svar2+1.D0/24.D0*svar* (acoords(1,iat)-acoords(1,jat))**2)
        hessian(posj+1,posj+1)=hessian(posj+1,posj+1)+ 24.D0*(svar2+1.D0/24.D0*svar* (acoords(2,iat)-acoords(2,jat))**2)
        hessian(posj+2,posj+2)=hessian(posj+2,posj+2)+ 24.D0*(svar2+1.D0/24.D0*svar* (acoords(3,iat)-acoords(3,jat))**2)
      end do
    end do
    status=0
  
!  call calcHessian(failed)
!  if (.not. failed) then
!    hessian = quick_qm_struct%hessian
!    status = 0
!  endif 
 
end subroutine dlf_get_hessian

! initialize parameters for the test potentials
subroutine driver_init
  use driver_parameter_module
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
end subroutine driver_init

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_put_coords(nvar,mode,energy,coords,iam)
  use dlf_parameter_module
  implicit none
  integer   ,intent(in)    :: nvar
  integer   ,intent(in)    :: mode
  integer   ,intent(in)    :: iam
  real(rk)  ,intent(in)    :: energy
  real(rk)  ,intent(in)    :: coords(nvar)
  integer                  :: iat
! **********************************************************************

! Only do this writing of files if I am the rank-zero processor
  if (iam /= 0) return

  if(mod(nvar,3)==0) then
    !assume coords are atoms
    if(mode==2) then
      open(unit=20,file="tsmode.xyz")
    else
      open(unit=20,file="coords.xyz")
    end if
    write(20,*) nvar/3
    write(20,*) 
    do iat=1,nvar/3
      write(20,'("H ",3f12.7)') coords((iat-1)*3+1:(iat-1)*3+3)
    end do
    close(20)
  else
  end if
end subroutine dlf_put_coords

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_error()
  implicit none
! **********************************************************************
  call dlf_mpi_abort() ! only necessary for a parallel build;
                       ! can be present for a serial build
  call exit(1)
end subroutine dlf_error

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_update()
  implicit none
! **********************************************************************
  ! only a dummy routine here.
end subroutine dlf_update


subroutine dlf_get_multistate_gradients(nvar,coords,energy,gradient,iimage,status)
  ! only a dummy routine up to now
  ! for conical intersection search
  use dlf_parameter_module
  implicit none
  integer   ,intent(in)    :: nvar
  integer   ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(in)    :: energy(2)
  real(rk)  ,intent(in)    :: gradient(nvar,2)
  integer   ,intent(in)    :: iimage
  integer   ,intent(in)    :: status
end subroutine dlf_get_multistate_gradients


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_put_procinfo(dlf_nprocs, dlf_iam, dlf_global_comm)

  implicit none

  integer, intent(in) :: dlf_nprocs ! total number of processors
  integer, intent(in) :: dlf_iam ! my rank, from 0, in mpi_comm_world
  integer, intent(in) :: dlf_global_comm ! world-wide communicator
! **********************************************************************

!!! variable in the calling program = corresponding dummy argument

end subroutine dlf_put_procinfo


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_procinfo(dlf_nprocs, dlf_iam, dlf_global_comm)

  implicit none

  integer :: dlf_nprocs ! total number of processors
  integer :: dlf_iam ! my rank, from 0, in mpi_comm_world
  integer :: dlf_global_comm ! world-wide communicator
! **********************************************************************

!!! dummy argument = corresponding variable in the calling program

end subroutine dlf_get_procinfo


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_put_taskfarm(dlf_ntasks, dlf_nprocs_per_task, dlf_iam_in_task, &
                        dlf_mytask, dlf_task_comm, dlf_ax_tasks_comm)

  implicit none

  integer, intent(in) :: dlf_ntasks          ! number of taskfarms
  integer, intent(in) :: dlf_nprocs_per_task ! no of procs per farm
  integer, intent(in) :: dlf_iam_in_task     ! my rank, from 0, in my farm
  integer, intent(in) :: dlf_mytask          ! rank of my farm, from 0
  integer, intent(in) :: dlf_task_comm       ! communicator within each farm
  integer, intent(in) :: dlf_ax_tasks_comm   ! communicator involving the 
                                             ! i-th proc from each farm
! **********************************************************************

!!! variable in the calling program = corresponding dummy argument 

end subroutine dlf_put_taskfarm


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_taskfarm(dlf_ntasks, dlf_nprocs_per_task, dlf_iam_in_task, &
                        dlf_mytask, dlf_task_comm, dlf_ax_tasks_comm)

  implicit none

  integer :: dlf_ntasks          ! number of taskfarms
  integer :: dlf_nprocs_per_task ! no of procs per farm
  integer :: dlf_iam_in_task     ! my rank, from 0, in my farm
  integer :: dlf_mytask          ! rank of my farm, from 0
  integer :: dlf_task_comm       ! communicator within each farm
  integer :: dlf_ax_tasks_comm   ! communicator involving the
                                 ! i-th proc from each farm
! **********************************************************************

!!! dummy argument = corresponding variable in the calling program

end subroutine dlf_get_taskfarm


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_output(dum_stdout, dum_stderr)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,keep_alloutput
  implicit none
  integer :: dum_stdout
  integer :: dum_stderr
  integer :: ierr
  logical :: topened
  character(len=10) :: suffix

! sort out output units; particularly important on multiple processors
 
! set unit numbers for main output and error messages
  if (dum_stdout >= 0) stdout = dum_stdout 
  if (dum_stderr >= 0) stderr = dum_stderr

  if (glob%iam /= 0) then
     inquire(unit=stdout, opened=topened, iostat=ierr)
     if (topened .and. ierr == 0) close(stdout)
     if (keep_alloutput) then ! hardwired in dlf_global_module.f90
        write(suffix,'(i10)') glob%iam
        open(unit=stdout,file='output.proc'//trim(adjustl(suffix)))
     else
        open(unit=stdout,file='/dev/null')
     end if
  endif

  if (glob%nprocs > 1) then
     ! write some info on the parallelization
     write(stdout,'(1x,a,i10,a)')"I have rank ",glob%iam," in mpi_comm_world"
     write(stdout,'(1x,a,i10)')"Total number of processors = ",glob%nprocs
     if (keep_alloutput) then
        write(stdout,'(1x,a)')"Keeping output from all processors"
     else
        write(stdout,'(1x,a)')"Not keeping output from processors /= 0"
     end if
  end if

end subroutine dlf_output


module driver_parameter_module
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
  logical :: use_nn=.true. ! use neural network instead of real potential?
end module driver_parameter_module

! **********************************************************************
! subroutines  that have to be provided to dl_find from outside
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
    imicroiter, maxmicrocycle, micro_esp_fit,tstore_int,initialpathneb,eonly,&
    ircstep,minircsteps,irot,&
    sct_in_IRC,bimol_int,usePath,minwavenumber, &
    useGeodesic,gprmep_mode,&
    calc_final_energies, initPathName, pathNameLength, &
    vpt2_resonance_treatment,&
    vpt2_resonance_criterion,vpt2_res_tol_deltae,&
    vpt2_res_tol_deltae_hard,vpt2_res_tol_martin,&
    vpt2_res_tol_isaacson,vpt2_hdcpt2_alpha,&
    vpt2_hdcpt2_beta,vpt2_deriv_deltaq,vpt2_asym_tol,&
    vpt2_Tmin,vpt2_Tmax,vpt2_nT,vpt2_do_part_func,&
    vpt2_grad_only,vpt2_force_doscf,vpt2_dq_factor_4hess,gpr_internal &
#ifdef CBINDINGS
    bind ( C, name="dlf_get_params" )&
#endif
        )
  use iso_c_binding, only: C_CHAR, c_null_char
  use dlf_parameter_module, only: rk
  use driver_parameter_module
  use quick_molspec_module, only: natom, xyz, quick_molspec
  use quick_constants_module, only: EMASS
  use quick_method_module,only: quick_method
  use dlf_constants
  use nn_module
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
  integer   ,intent(inout)   :: tstore_int  
  integer   ,intent(inout)   :: initialpathneb
  integer   ,intent(inout)   :: eonly
  real(rk)  ,intent(inout)   :: ircstep
  integer   ,intent(inout)   :: minircsteps
  integer   ,intent(inout)   :: irot
  integer   ,intent(inout)   :: sct_in_IRC
  integer   ,intent(inout)   :: bimol_int
  integer   ,intent(inout)   :: usePath
  real(rk)  ,intent(inout)   :: minwavenumber
  integer   ,intent(inout)   :: useGeodesic
  integer   ,intent(inout)   :: gprmep_mode
  integer   ,intent(inout)   :: calc_final_energies
  character(kind=c_char, len=1), dimension (4096),intent(inout) :: initPathName
  integer   ,intent(inout)   :: pathNameLength
  integer   ,intent(inout)   :: vpt2_resonance_treatment
  integer   ,intent(inout)   :: vpt2_resonance_criterion
  real(rk)  ,intent(inout)   :: vpt2_res_tol_deltae
  real(rk)  ,intent(inout)   :: vpt2_res_tol_deltae_hard
  real(rk)  ,intent(inout)   :: vpt2_res_tol_martin
  real(rk)  ,intent(inout)   :: vpt2_res_tol_isaacson
  real(rk)  ,intent(inout)   :: vpt2_hdcpt2_alpha
  real(rk)  ,intent(inout)   :: vpt2_hdcpt2_beta
  real(rk)  ,intent(inout)   :: vpt2_deriv_deltaq
  real(rk)  ,intent(inout)   :: vpt2_asym_tol
  real(rk)  ,intent(inout)   :: vpt2_Tmin
  real(rk)  ,intent(inout)   :: vpt2_Tmax
  integer   ,intent(inout)   :: vpt2_nT
  integer   ,intent(inout)   :: vpt2_do_part_func
  integer   ,intent(inout)   :: vpt2_grad_only
  integer   ,intent(inout)   :: vpt2_force_doscf
  real(rk)  ,intent(inout)   :: vpt2_dq_factor_4hess
  integer   ,intent(inout)   :: gpr_internal
  ! local variables
  real(rk)                   :: svar
  integer                    :: i,iat,jat,ivar,mpierror
  character(2)               :: str2
  integer                    :: nat_
  interface
    subroutine read_rand(arr)
      use dlf_parameter_module, only: rk
      use dlf_global, only : glob
      real(rk)  :: arr(:)
    end subroutine read_rand
  end interface
! **********************************************************************
  ierr=0
  tsrel=1
  usePath=0

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

  vpt2_resonance_treatment=4
  vpt2_deriv_deltaq=0.05D0

  tolerance=quick_method%gradMaxCrt ! negative: default settings
  tolerance_e =quick_method%EChange
  printl=5
  printf=4
  maxcycle=quick_method%iopt !200
  maxene=100000

  tolrot=1.D-2
!  tolrot=0.1D0
!  maxrot=100 !was 100

  task=0 !1011

  distort=0.D0 !0.4 
  tatoms=0
  icoord=quick_method%dlfind_icoord !0 cartesian coord !210 Dimer !190 qts search !120 NEB frozen endpoint
  massweight=0

  imicroiter=0 ! 1 means: use microiterative optimization

  iopt=quick_method%dlfind_iopt! 20 !3 or 20 later change to 12
  ! 2D-polynomial: Tc= 1658.21091 K - the same for low=0.1
  !temperature=0.9D0*30463.72961D0
  gpr_internal = 1
  ! crossover-T for 6th order: 50317.25742D0 (this was with linear 0.01 but it is kept for the time being)
  !temperature=1.47D0*50317.25742D0
  ! Tc for MB-Potential is 2206.66845 K
  temperature= 100.D0
  temperature=0.9D0

!  iline=0
  maxstep=0.5D0
  scalestep=1.0D0
  lbfgs_mem=100
  nimage=20 !*k-k+1
  nebk=0.0D0 ! for QTS calculations with variable tau, nebk transports the parameter alpha (0=equidist. in tau)
  qtsflag=-3 ! 1: Tunneling splittings, 11/10: read image hessians

  !"accurate" etunnel for 0.4 Tc: 0.3117823831234522

  ! Hessian
  delta= 1.D-2 ! warning: 1.D-5 is very small and will probably only
  !delta= REPL_DELTA!1.D-2 ! warning: 1.D-5 is very small and will probably only work for fitted PES
  eonly=0 ! 1 = true
  soft=-6.D-3
  update=1! 2
  maxupd=1
  !inithessian=2
  minstep=1.D0 **2 ! 1.D-5
  minstep=1.D-5

  ! IRC stuff
  minircsteps=10
!  ircstep=-0.04D0

!!$  ! T_c for poly3= 30463.27961
!!$  ! variables for exchange by sed
!!$  iopt=IOPT_VAR
!!$  nimage=NIMAGE_VAR
!!$  temperature = TFAC *  2872.10555D0 ! K ! T_c for the MB-potential for hydrogen is 2206.668 K
!!$  !temperature = TFAC !* 234.50644 ! T_c in K 
!!$  nebk= NEBK_VAR

  ! read parameters from file dlf.in
!  call read_dlf_in(nvar,coords(1:nvar),coords2(1:nvar),&
!      iopt,icoord,nimage,temperature,nebk,coords2(nvar+1:4*nvar/3),distort,tolerance)
  !temperature=temperature*234.90288D0 ! quartic

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
  
  !call test_ene

end subroutine dlf_get_params

subroutine test_update
  use dlf_parameter_module, only: rk
  use dlf_allocate, only: allocate,deallocate
  use dlf_global, only: glob,printl
  implicit none
  integer(4) :: varperimage,nimage,iimage,ivar4
  real(rk), allocatable:: coords(:,:),grad(:,:) ! varperimage,nimage
  real(rk), allocatable:: hess(:,:,:) ! varperimage,varperimage,nimage
  real(rk), allocatable:: fhess(:,:) ! 2*varperimage*nimage,2*varperimage*nimage
  real(rk), allocatable:: eigval(:),eigvec(:,:)
  real(rk), allocatable:: vec0(:)
  real(rk), allocatable:: determinant(:)
  real(rk), allocatable:: tmphess(:,:)
  real(rk) :: svar
  integer :: ivar,jvar,vp8,target_image,step,lastimage,jimage,turnimg
  logical :: havehessian,fracrecalc,was_updated

  open(unit=102,file="grad_coor.bin",form="unformatted")
  read(102) varperimage,nimage
  print*,"varperimage,nimage",varperimage,nimage
  vp8=varperimage

!!$  call allocate(coords,int(varperimage,kind=8),int(nimage,kind=8))
!!$  call allocate(grad,int(varperimage,kind=8),int(nimage,kind=8))
!!$  call allocate(determinant,int(nimage,kind=8))

  do iimage=1,nimage
    read(102) ivar4
    print*,"Reading coords/grad of image",ivar4
    read(102) grad(:,iimage)
    read(102) coords(:,iimage)
  end do
  close(102)
  print*,"Coords sucessfully read"

  ! print coords and grad
  iimage=2
  print*,"Coords and grad for image ",iimage
  do ivar=1,vp8
    write(6,"(i6,1x,2es18.9)") &
        ivar,coords(ivar,iimage),grad(ivar,iimage)
  end do

  open(unit=101,file="hessian.bin",form="unformatted")
  ! Hessian in mass-weighted coordinates on the diagnal blocks - everything else should be zero
  read(101) iimage,ivar4!neb%varperimage,neb%nimage
  if(iimage/=varperimage.or.ivar4/=nimage) then
    print*,"Dimensions read",iimage,ivar4
    call dlf_fail("ERROR: wrong dimensions in hessian.bin!")
  end if
  ivar=2*varperimage*nimage
  print*,"File Hessian size",ivar
  call allocate(fhess,ivar,ivar)
  read(101) fhess
  close(101)
  print*,"Hessian sucessfully read"

  ! map hessian to different array:
!!$  call allocate(hess,int(varperimage,kind=8),int(varperimage,kind=8),int(nimage,kind=8))
  do iimage=1,nimage
    print*,"Image",iimage,"Hessian positions",(iimage-1)*varperimage+1,iimage*varperimage
    hess(:,:,iimage)=fhess((iimage-1)*varperimage+1:iimage*varperimage,(iimage-1)*varperimage+1:iimage*varperimage)
  end do
  call deallocate(fhess)

  call allocate(eigval,vp8)
  call allocate(eigvec,vp8,vp8)
  ! now we have all we need

  print*,"# Distance from previous image"
  do iimage=2,nimage
    print*,iimage,sqrt(sum( (coords(:,iimage)-coords(:,iimage-1))**2))
  end do

  print*,"# Determinant of Hessian"
  do iimage=1,nimage
    do ivar=1,vp8
      do jvar=ivar+1,vp8
        if(abs(hess(ivar,jvar,iimage)-hess(jvar,ivar,iimage))>1.D-20) &
            print*,"Unsymmetric:",ivar,jvar,iimage,hess(ivar,jvar,iimage),hess(jvar,ivar,iimage)
      end do
    end do
    call dlf_matrix_diagonalise(vp8,hess(:,:,iimage),eigval,eigvec)
    !write(6,"(i6,1x,9es12.3)") iimage,product(eigval(8:vp8)),eigval(1:8)
    do ivar=1,6
      eigval(minloc(abs(eigval)))=1.D0
    end do
    determinant(iimage)=product(eigval)
    write(6,"(i6,1x,9es12.3)") iimage,product(eigval),eigval(1:8)
  end do
  
  print*,"maxval(hess(:,:,nimage-1)-hess(:,:,nimage))",maxval(hess(:,:,nimage-1)-hess(:,:,nimage))

!!$  ! Richtungsableitung
!!$  call allocate(vec0,vp8)!int(varperimage,kind=8))
!!$  do iimage=2,nimage-1
!!$    ! eigval is Vector along which the derivative is taken
!!$    eigval=coords(:,iimage+1)-coords(:,iimage-1)
!!$    svar=sqrt(sum(eigval**2))
!!$    eigval=eigval/svar
!!$    vec0=matmul(hess(:,:,iimage),eigval)
!!$    do ivar=1,vp8
!!$      write(6,"(2i6,1x,2es18.9,1x,f10.5)") &
!!$          iimage,ivar,(grad(ivar,iimage+1)-grad(ivar,iimage-1))/svar,vec0(ivar),&
!!$          vec0(ivar)/((grad(ivar,iimage+1)-grad(ivar,iimage-1))/svar)
!!$    end do
!!$  end do

  !
  ! now test updates
  !
  call allocate(tmphess,vp8,vp8)
  havehessian=.true.
  fracrecalc=.false.
  printl=2
  glob%maxupd=30000

  target_image=1 !nimage
  ! update hessians to the one of the first image
  print*,"Updating Hessians to that of image",target_image
  print*,"Sum-of-squares difference"
  do iimage=1,nimage
    tmphess(:,:)=hess(:,:,iimage)
    call dlf_hessian_update(vp8, &
        coords(:,target_image),coords(:,iimage),&
        grad(:,target_image),grad(:,iimage), &
        tmphess, havehessian, fracrecalc, was_updated)
    if(.not.havehessian) then
      print*,"Problem with hessian update, image",iimage
      havehessian=.true.
    end if
    print*,iimage,sum( (tmphess-hess(:,:,target_image))**2),&
        sum( (hess(:,:,iimage)-hess(:,:,target_image))**2)
  end do

  print*,"Minstep",glob%minstep
  print*,"Updating Hessians to that of image",target_image
  print*,"Determinant"
  open(file="determinant",unit=10)
  do iimage=1,nimage
    tmphess(:,:)=hess(:,:,iimage)
    call dlf_hessian_update(vp8, &
        coords(:,target_image),coords(:,iimage),&
        grad(:,target_image),grad(:,iimage), &
        tmphess, havehessian, fracrecalc, was_updated)
    if(.not.havehessian) then
      print*,"Problem with hessian update, image",iimage
      havehessian=.true.
    end if
    call dlf_matrix_diagonalise(vp8,tmphess,eigval,eigvec)
    !write(6,"(i6,1x,9es12.3)") iimage,product(eigval(8:vp8)),eigval(1:8)
    do ivar=1,6
      eigval(minloc(abs(eigval)))=1.D0
    end do

    print*,iimage,product(eigval),determinant(iimage),determinant(target_image)
    write(10,*) iimage,product(eigval),determinant(iimage),determinant(target_image)
  end do
  close(10)

  print*,"Updating Hessians to that of image",target_image
  print*,"Determinant - incremental"
  open(file="determinant_incr",unit=10)
  do iimage=1,nimage
    tmphess(:,:)=hess(:,:,iimage)
    step=1
    if(iimage>target_image) step=-1
    lastimage=iimage
    do jimage=iimage+step,target_image,step
      !print*,"updating",lastimage," to ",jimage
      call dlf_hessian_update(vp8, &
          coords(:,jimage),coords(:,lastimage),&
          grad(:,jimage),grad(:,lastimage), &
          tmphess, havehessian, fracrecalc, was_updated)

      if(.not.havehessian) then
        print*,"Problem with hessian update, image",iimage
        havehessian=.true.
      end if
      lastimage=jimage
    end do
    call dlf_matrix_diagonalise(vp8,tmphess,eigval,eigvec)
    !write(6,"(i6,1x,9es12.3)") iimage,product(eigval(8:vp8)),eigval(1:8)
    do ivar=1,6
      eigval(minloc(abs(eigval)))=1.D0
    end do

    print*,iimage,product(eigval),determinant(iimage),determinant(target_image)
    write(10,*) iimage,product(eigval),determinant(iimage),determinant(target_image)
  end do
  close(10)

  print*,"Updating Hessians to that of image",target_image
  print*,"Determinant - incremental turning around"
  turnimg=20
  open(file="determinant_turn",unit=10)
  do iimage=1,nimage
    tmphess(:,:)=hess(:,:,iimage)
    lastimage=iimage
    ! first upwards to turnimg
    do jimage=iimage+1,turnimg
      print*,"updating",lastimage," to ",jimage
      call dlf_hessian_update(vp8, &
          coords(:,jimage),coords(:,lastimage),&
          grad(:,jimage),grad(:,lastimage), &
          tmphess, havehessian, fracrecalc, was_updated)

      if(.not.havehessian) then
        print*,"Problem with hessian update, image",iimage
        havehessian=.true.
      end if
      lastimage=jimage
    end do
    step=1
    if(lastimage>target_image) step=-1
    do jimage=lastimage+step,target_image,step
      print*,"updating",lastimage," to ",jimage
      call dlf_hessian_update(vp8, &
          coords(:,jimage),coords(:,lastimage),&
          grad(:,jimage),grad(:,lastimage), &
          tmphess, havehessian, fracrecalc, was_updated)

      if(.not.havehessian) then
        print*,"Problem with hessian update, image",iimage
        havehessian=.true.
      end if
      lastimage=jimage
    end do
    call dlf_matrix_diagonalise(vp8,tmphess,eigval,eigvec)
    !write(6,"(i6,1x,9es12.3)") iimage,product(eigval(8:vp8)),eigval(1:8)
    do ivar=1,6
      eigval(minloc(abs(eigval)))=1.D0
    end do

    print*,iimage,product(eigval),determinant(iimage),determinant(target_image)
    write(10,*) iimage,product(eigval),determinant(iimage),determinant(target_image)
  end do
  close(10)

!!$  do ivar=1,vp8
!!$    WRITE(*,'(33f10.5)') hess(ivar,:,1)*1.D6
!!$  end do
!!$
!!$  do ivar=1,vp8
!!$    write(6,"(i6,1x,es18.9)") &
!!$            ivar,eigval(ivar)
!!$  end do

  call deallocate(coords)
  call deallocate(grad)
  
  call dlf_fail("stop in test_update")
end subroutine test_update

! just a test routine
subroutine test_ene
  use dlf_parameter_module, only: rk
  implicit none
  integer :: ivar,ivar2,status
  integer :: samples
  real(rk) :: halfSamples
  real(rk) :: coords(3),grad(3),hess(3,3),ene
  integer :: ierr
  coords(:)=0.D0
!  open(file="energy",unit=13)
  open(file="energy-2d.dat",unit=13)
  samples = 100
  halfSamples = dble(samples) * 0.25D0
  do ivar2=1,samples
!    do ivar=1,samples
    coords(1)=dble(ivar2)/dble(samples)+0.5D0
 !     coords(1)=dble(ivar-samples/2)/halfSamples
 !     coords(2)=dble(ivar2-samples/2)/halfSamples
      call dlf_get_gradient(3,coords,ene,grad,1,-1,status,ierr)
      call dlf_get_hessian(3,coords,hess,status)
!     write(13,*) coords(1),coords(2),ene,grad(1),grad(2),hess(1,1),hess(2,2),hess(2,1)
     write(13,'(4f20.10)') coords(1),ene,grad(1),hess(1,1)
!    end do
!    write(13,*) ""
  end do
  close(13)
  call dlf_fail("stop in test_ene")
end subroutine test_ene

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_gradient(nvar,coords,energy,gradient,iimage,kiter,status,ierr)
  !  Mueller-Brown Potential
  !  see K Mueller and L. D. Brown, Theor. Chem. Acta 53, 75 (1979)
  !  taken from JCP 111, 9475 (1999)
  use dlf_parameter_module, only: rk
  use driver_parameter_module
  use dlf_constants, only: dlf_constants_get
  use dlf_allocate, only: allocate,deallocate
  use nn_av_module
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
  integer, intent(inout)   :: ierr

#if defined DEBUG || defined DEBUGTIME
#define CHECK_ERROR(ierr)
#else
#ifdef MPIV
#define CHECK_ERR if(master .and. ierr /= 0)
#else
#define CHECK_ERR if(ierr /= 0)
#endif
#define CHECK_ERROR(ierr) CHECK_ERR call RaiseException(ierr)
#endif

  !
! **********************************************************************
!  call test_update
  status=1

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  call gpu_setup(natom,nbasis, quick_molspec%nElec, quick_molspec%imult, &
              quick_molspec%molchg, quick_molspec%iAtomType)
  call gpu_upload_xyz(xyz)
  call gpu_upload_atom_and_chg(quick_molspec%iattype, quick_molspec%chg)
#endif                                                                                            

  call getEriPrecomputables
  call schwarzoff

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
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

#if defined CUDA_MPIV || defined HIP_MPIV
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

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
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

! a rudimentary interface to xtb
subroutine xtb_get_gradient(nvar,coords,energy,gradient,status)
  use dlf_parameter_module, only: rk
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)                 :: znuc(nvar/3) 
  real(rk)  ,intent(out)   :: energy
  real(rk)  ,intent(out)   :: gradient(nvar)
  integer   ,intent(out)   :: status
  !
  logical  :: eof
  character(len=80) :: readString
  integer                   :: io,icount,i
  !
  status=1
  !print*,"coords in energy eval",coords
  OPEN(unit=1569,file='coords_xtbin.xyz')
  call write_xyz(1569, nvar/3, znuc, coords)
  CLOSE(1569)
  
  eof = .false.
  ! using the command line to calculate the energy of coords_xtbin.xyz with xtb
  CALL execute_command_line('xtb coords_xtbin.xyz --grad > xtb_eAndG.out')
  ! ********************************************
  ! the following is only valid for XTB of course... Different read-in functions
  ! have to be found here
  
  ! read energy
  OPEN(unit=1569,file='energy')
  READ(1569,*)
  do while (.not.eof)
    READ(1569,'(A)',iostat=io) readString !icount, energy
    if (io/=0.or.trim(readString)=="$end") then
      eof = .true.
      exit
    else
      READ(readString(1:),*) icount, energy
    end if
  end do
  CLOSE(1569)
  !print*,"Energy: ",energy
  
  ! read gradient
  eof=.false.
  OPEN(unit=1569,file='gradient')
  READ(1569,*)
  do while (.not.eof)
    READ(1569,'(A)',iostat=io) readString
    if (io/=0.or.trim(readString)=="$end") then
      eof = .true.
      exit
    else if (readString(1:7)=="  cycle") then
      ! nothing to do
    else
      READ(readString(1:),*) gradient(1:3)        
      do i = 2, nvar/3
        READ(1569,'(A)',iostat=io) readString
        READ(readString(1:),*) gradient((i-1)*3+1:i*3)
      end do
    end if
  end do
  CLOSE(1569)
  !print*,"gradient",gradient
  status=0
end subroutine xtb_get_gradient

! dummy routine - must be commented out when using a real potential
subroutine getpot (coords,svar)
  implicit none
  real(8) :: coords(*)
  real(8) :: svar
  integer, parameter :: nvar=21
  real(8) :: gradient(nvar)
  call pes_get_gradient(nvar,coords,svar,gradient)
  !call nn_get_energy(nvar,coords,svar)
  !print*,"Warning: link to a real getpot routine instead!"
end subroutine getpot

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_energy(nvar,coords,energy,iimage,kiter,status)
  use dlf_parameter_module, only: rk
  use driver_parameter_module
  use dlf_constants, only: dlf_constants_get
  use dlf_allocate, only: allocate,deallocate
  !use vib_pot
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: energy
  integer   ,intent(in)    :: iimage
  integer   ,intent(in)    :: kiter
  integer   ,intent(out)   :: status
  real(rk) :: gradient(nvar)
  integer  :: isystem
  status=1
  ! only isystem=10 for the time being
  isystem=3
  select case (isystem)
  case (10)
    !call getpot(coords,energy)
    call nn_get_gradient(nvar,coords,energy,gradient)
    if(isnan(energy)) call dlf_fail("Energy is NaN")
    status=0
  case default
    call dlf_fail("dlf_get_energy at present only implemented for isystem=10")
  end select

end subroutine dlf_get_energy

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_hessian(nvar,coords,hessian,status)
  !  get the hessian at a given geometry
  use dlf_parameter_module
  use driver_parameter_module
  use nn_av_module
  use dlf_allocate, only: allocate,deallocate
  !use vib_pot
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: hessian(nvar,nvar)
  integer   ,intent(out)   :: status
  real(rk) :: acoords(3,nvar/3),r,svar,svar2
  integer  :: posi,posj,iat,jat,m,n
  ! variables for Mueller-Brown potential
  real(rk) :: x,y
  integer  :: icount,j1,j2,isystem
  ! variables non-cont. diff. potential
  real(rk) :: t,pi
  real(rk) :: gradient(nvar),energy
  ! modified eckart (with dip)
  real(rk) :: ddv,alprime,yprime
  real(rk), allocatable :: eigenvalues(:,:),eigenvector(:,:,:),store(:,:,:)
  integer, allocatable :: sortlist(:)
! **********************************************************************
  pi=4.D0*atan(1.D0)
  hessian(:,:)=0.D0
  status=1
  isystem=2
  select case (isystem)
  case(1)
    ! mueller-brown
    
    if(use_nn) then
      call nn_get_hessian(nvar,coords,hessian)
      status=0 ! 1 means it did not work
    else
      x =  coords(1)
      y =  coords(2)
      
      hessian=0.D0
      do icount=1,4
        svar= apar(icount)*(x-x0par(icount))**2 + &
            bpar(icount)*(x-x0par(icount))*(y-y0par(icount)) + &
            cpar(icount)*(y-y0par(icount))**2 
        svar2= acappar(icount) * dexp(svar)
        !energy=energy+ svar2
!!$      gradient(1)=gradient(1) + svar2 * &
!!$          (2.D0* apar(icount)*(x-x0par(icount))+bpar(icount)*(y-y0par(icount)))
!!$      gradient(2)=gradient(2) + svar2 * &
!!$          (2.D0* cpar(icount)*(y-y0par(icount))+bpar(icount)*(x-x0par(icount)))
        hessian(1,1)=hessian(1,1)+svar2 * &
            (2.D0* apar(icount)*(x-x0par(icount))+bpar(icount)*(y-y0par(icount)))**2 + &
            svar2 * 2.D0 * apar(icount)
        hessian(2,2)=hessian(2,2)+svar2 * &
            (2.D0* cpar(icount)*(y-y0par(icount))+bpar(icount)*(x-x0par(icount)))**2 + &
            svar2 * 2.D0 * cpar(icount)
        hessian(1,2)=hessian(1,2) + svar2 * &
            (2.D0* apar(icount)*(x-x0par(icount))+bpar(icount)*(y-y0par(icount)))* &
            (2.D0* cpar(icount)*(y-y0par(icount))+bpar(icount)*(x-x0par(icount))) + &
            svar2 * bpar(icount)
      end do
      hessian(2,1)=hessian(1,2)
      status=0 ! 1 means it did not work
    end if

  case(2,3)
    acoords=reshape(coords,(/3,nvar/3/))
    do iat=1,nvar/3
      do jat=iat+1,nvar/3
        r=sum((acoords(:,iat)-acoords(:,jat))**2)
        ! Lennard-Jones Potential
        svar = 96.D0*epsilon * (7.D0*sigma**12/r**8-2.D0*sigma**6/r**5) ! coeff of x1x2
        svar2= epsilon * (-2.D0*sigma**12/r**7+sigma**6/r**4) ! for x1x1
        posi=(iat-1)*3+1
        posj=(jat-1)*3+1
        ! off-diag
        hessian(posi,posi+1)  =hessian(posi,posi+1)  + svar * (acoords(1,iat)-acoords(1,jat)) * (acoords(2,iat)-acoords(2,jat))
        hessian(posi,posi+2)  =hessian(posi,posi+2)  + svar * (acoords(1,iat)-acoords(1,jat)) * (acoords(3,iat)-acoords(3,jat))
        hessian(posi+1,posi)  =hessian(posi+1,posi)  + svar * (acoords(2,iat)-acoords(2,jat)) * (acoords(1,iat)-acoords(1,jat))
        hessian(posi+1,posi+2)=hessian(posi+1,posi+2)+ svar * (acoords(2,iat)-acoords(2,jat)) * (acoords(3,iat)-acoords(3,jat))
        hessian(posi+2,posi)  =hessian(posi+2,posi)  + svar * (acoords(3,iat)-acoords(3,jat)) * (acoords(1,iat)-acoords(1,jat))
        hessian(posi+2,posi+1)=hessian(posi+2,posi+1)+ svar * (acoords(3,iat)-acoords(3,jat)) * (acoords(2,iat)-acoords(2,jat))

        do m=0,2
          do n=0,2
            if(m==n) cycle
  hessian(posi+m,posj+n)=hessian(posi+m,posj+n)- svar * (acoords(M+1,iat)-acoords(M+1,jat)) * (acoords(N+1,iat)-acoords(N+1,jat))
  hessian(posj+m,posi+n)=hessian(posj+m,posi+n)- svar * (acoords(M+1,iat)-acoords(M+1,jat)) * (acoords(N+1,iat)-acoords(N+1,jat))
          end do
        end do
        ! Diag for different atoms ...
        do m=0,2
          hessian(posi+m,posj+m)=hessian(posi+m,posj+m) - 24.D0*(svar2+1.D0/24.D0*svar* (acoords(m+1,iat)-acoords(M+1,jat))**2)
          hessian(posj+m,posi+m)=hessian(posj+m,posi+m) - 24.D0*(svar2+1.D0/24.D0*svar* (acoords(m+1,iat)-acoords(M+1,jat))**2)
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
    !status=1 ! 1= not possible
  case(4)
    ! Eckart
    xvar= coords(1)
    yvar= (Vb+Va)/(Vb-Va)*exp(alpha*xvar)
    svar=1.D0+yvar
    hessian(1,1)=alpha**2*yvar * ( Va/svar + (Vb-Va*yvar)/svar**2 - 2.D0*Vb*yvar/svar**3) + &
        alpha**2 * yvar**2 / svar**2 * ( -2.D0*Va + (-4.D0*Vb+2.D0*Va*yvar)/svar +6.D0*Vb*yvar/svar**2)

    ! modified eckart (with a dip) - comment out these lines if you want normal Eckart:
    ddv=alpha**2*(va**2-vb**2)**2/8.D0/vb**3
    if(abs(dip)<1.D-50) then
      alprime=0.D0
    else
      alprime=sqrt(ddv*8.D0/(dip*4.D0))*244.784D0/1206.192D0
    end if
    yprime=exp(alprime*(xvar-dshift))

    svar=1.D0+yprime
    !gradient(1)=gradient(1) - alprime*yprime/svar**2 * ( &
    !     4.D0*dip - 2.D0*4.D0*dip*yprime/svar)
    hessian(1,1)=hessian(1,1) - alprime**2*yprime * ( (4.D0*Dip)/svar**2 - 2.D0*4.D0*Dip*yprime/svar**3) - &
        alprime**2 * yprime**2 / svar**2 * ( (-4.D0*4.D0*Dip)/svar +6.D0*4.D0*Dip*yprime/svar**2)
    !print*,"position",xvar," Energy",energy


!!$    ! debug output - test of hessian-at-maximum-equations.
!!$    print*,"Hessian calculated, 1,1:",hessian(1,1)+alpha**2*vb/8.D0,hessian(1,1),-alpha**2*vb/8.D0
!!$    print*,"Hessian short           ",hessian(1,1)+alpha**2*(Va+Vb)**2*(Vb-Va)/8.D0/Vb**3    ,&
!!$        hessian(1,1),-alpha**2*(Va+Vb)**3*(Vb-Va)/8.D0/Vb**3
!!$    print*,va,hessian(1,1)+0.08085432314993411D0,"leading"
!!$    print*,"should be zero",Va/svar + (Vb-Va*yvar)/svar**2 - 2.D0*Vb*yvar/svar**3
    status=0 ! 0=OK 1=not available 
  case(5)
!!$    ! Quartic
!!$    xvar= coords(1)
!!$    hessian(1,1)=V0 * 4.D0 / x0**2 * (3.D0*xvar**2/x0**2-1.D0)
!!$    status=0 
    
!!$    ! 3rd order polynomial as also used in the 1D-scattering code
!!$    xvar=coords(1)*1.5D0
!!$    !gradient(1)= 50.D0*(-6.D0*xvar**2+6.D0*xvar) * 1.5D0
!!$    hessian(1,1)=50.D0*(-12.D0*xvar+6.D0) * 1.5D0**2
!!$    status=0 

!!$    ! 3rd order polynomial 
!!$    xvar=coords(1)!*1.5D0
!!$    hessian(1,1)=(-12.D0*xvar-6.D0) 
!!$    status=0 

!!$    ! 5th order polynomial - should have an extended instanton for T > Tc
!!$    xvar=coords(1)
!!$    !gradient(1)= -10.D0*xvar**4 -10.D0*xvar**3 -xvar 
!!$    hessian(1,1)= -40.D0*xvar**3 -30.D0*xvar**2 -1.D0
!!$    status=0 

    ! 4th order polynomial with asymmetry
    xvar=coords(1)
    hessian(1,1)= 12.D0*V0*xvar**2 -4.D0*V0 + &
        delpot * steep**2 * 2.D0 / (1+((steep*(xvar-shift)))**2)**2 * &
        ((steep*(xvar-shift)))
    status=0 

!!$    ! 6th order polynomial - should have an extended instanton for T > Tc
!!$    xvar=coords(1)
!!$    !gradient(1)= 12.D0*xvar**5 -8.D0*xvar**3 -xvar -0.01
!!$    hessian(1,1)= 60.D0*xvar**4 -24.D0*xvar**2 -1.D0
!!$    status=0 

  case(6)
    !Vibrational potential
    !call vib_get_hessian(coords,hessian)
    !status=0 
  case(7)

    x =  coords(1)
    y =  coords(2)
    t =  0.0d0!smallbarvar 
    hessian=0.D0


!!$    energy = 10.D0*dexp(-2.D0*(x+1.D0)**2)*(dexp(-2.D0*(y-1.D0)**2)+(dexp(-2.D0*(y+1.D0)**2)))- &
!!$       dexp(-2.D0*(y**2+(x+2.D0)**2)) - 1.6D0*dexp(-2.D0*(x-1.D0)**2)* &
!!$       (dexp(-2.D0*(y+t)**2)+dexp(-2.D0*(y-t)**2))

    hessian(1,1)=-4.D0*(10.D0*(dexp(-2.D0*(y-1.D0)**2)+dexp(-2.D0*(y+1.D0)**2))*(dexp(-2.D0*(x+1.D0)**2)*&
        (-4.D0*(x+1.D0)**2+1.D0))-3.D0*dexp(-2.D0*(y**2+(x+2.D0)**2))*(-4.D0*(x+2.D0)**2+1.D0)-1.6D0* &
        (dexp(-2.D0*(x-1.D0)**2)*(-4.D0*(x-1.D0)**2+1.D0))*(dexp(-2.D0*(y+t)**2)+dexp(-2.D0*(y-t)**2)))
    
    hessian(2,2)=-4.D0*(10.D0*dexp(-2.D0*(x+1.D0)**2)*(dexp(-2.D0*(y-1.D0)**2)*(-4.D0*(y-1.D0)**2+1.D0))- &
        3.D0*dexp(-2.D0*(y**2+(x+2.D0)**2))*(-4.D0*y**2+1.D0)-1.6D0*dexp(-2.D0*(x-1.D0)**2)* &
        (dexp(-2.D0*(y+t)**2)*(-4.D0*(y+t)**2+1.D0)+dexp(-2.D0*(y-t)**2)*(-4.D0*(y-t)**2+1.D0)))
    
    hessian(1,2)= 8.D0*(10.D0*dexp(-2.D0*(x+1.D0)**2)*(x+1.D0)*(dexp(-2.D0*(y-1.D0)**2)*(y-1.D0)+ &
        (dexp(-2.D0*(y+1.D0)**2)*(y+1.D0)))-3.D0*dexp(-2.D0*(y**2+(x+2.D0)**2))*(x+2.D0)*y- &
        1.6D0*dexp(-2.D0*(x-1.D0)**2)*(dexp(-2.D0*(y+t)**2)*(y+t)+dexp(-2.D0*(y-t)**2)*(y-t))*(x-1.D0))

    hessian(2,1)=hessian(1,2)

    status=0

  case(8)
    ! polymul (Polynomial in multiple dimensions)
    ! E=E1+E2+...+En
    ! E1= (-2Ebarr) x1**3 + (3Ebarr) x1**2
    ! Ei=xi**2 * 1/2 * d_i (1-x1/5)  | i>1
    hessian=0.D0
    hessian(1,1)=-12.D0 * Ebarr * coords(1) + 6.D0 * Ebarr
!!$    ! finite vdamp
!!$    do iat=2,num_dim
!!$      hessian(1,iat)=-2.D0/vdamp * coords(iat)
!!$      hessian(iat,1)=-2.D0/vdamp * coords(iat)
!!$    end do
    do iat=2,num_dim
!!$      ! finite vdamp
!!$      hessian(iat,iat)=dpar(iat) * (1.D0-coords(1)/vdamp)
      ! vdamp = infinity
      hessian(iat,iat)=dpar(iat)
    end do
    
    status=0

  case(9)

    x =  coords(1)
    y =  coords(2)

    hessian=0.D0
    hessian(1,1)= 6.D0*par3*x + 2.D0*par2 + 2.D0*parb*y**2
    hessian(1,2)= 4.D0*parb*x*y
    hessian(2,1)=hessian(1,2)
    hessian(2,2)= 12.D0*y**2 - 2.D0*low + 2.D0*parb*x**2
    status=0

  case(10)

    ! H2CN or OH+H2 potential
    ! FD hessian from energy
!print*,"fd"
!    call fdhess4(nvar,coords,1.D-2,hessian)

!    call nn_get_hessian(nvar,coords,hessian)
    ifile=0
    if(avnncalc) then
      hessian=0.d0
      do iii=1,numfiles
        ifile=iii
        call pes_get_hessian(nvar,coords,hessianav(:,:,ifile))
        hessian=hessian+hessianav(:,:,ifile)
      end do
!      hessian=hessian/dble(numfiles)
      hessian=hessianav(:,:,1)
!      call allocate(eigenvalues,nvar,numfiles)
!      call allocate(eigenvector,nvar,nvar,numfiles)
!      call allocate(store,nvar,nvar,numfiles+1)
!      store=0.d0
!      do iii=1,numfiles
!        call r_diagonal(nvar,hessianav(:,:,iii),eigenvalues(:,iii),eigenvector(:,:,iii))
!        do j1=1,nvar
!          store(j1,j1,iii)=eigenvalues(j1,iii)
!        enddo
!      enddo
!      store(:,:,numfiles+1)=store(:,:,1)
!      hessian=matmul(matmul(eigenvector(:,:,2),store(:,:,numfiles+1)),transpose(eigenvector(:,:,2)))
!      call deallocate(eigenvalues)
!      call deallocate(eigenvector)
!      call deallocate(store)
    else
      call pes_get_hessian(nvar,coords,hessian)
    endif

!    enddo
    status=0

  case(11)
    ! shepard
  !  call pes_egradh(coords,energy,gradient,hessian)
    status=0
  case(12)
    call xtb_get_hessian(nvar,coords,hessian,status)
  end select

end subroutine dlf_get_hessian

subroutine xtb_get_hessian(nvar,coords,hessian,status)
  use dlf_parameter_module, only: rk
  implicit none
  integer, intent(in) :: nvar
  real(rk), intent(in) :: coords(nvar)
  real(rk)             ::znuc(nvar/3)
  real(rk), intent(out):: hessian(nvar,nvar)
  integer, intent(out) :: status

  real(rk) :: ahessian(nvar*nvar)
  logical :: eof
  character(len=80) :: readString
  integer                   :: io,icount,i,maxline,k,j,n,l,m
  status = 1
  OPEN(unit=1570,file='coords_hess.xyz')
  call write_xyz(1570, nvar/3, znuc, coords)
  CLOSE(1570)  
  eof = .false.
  maxline = ceiling(dble(nvar)/5.0)*nvar
  n = nvar/5
  l = modulo(nvar,5)
  CALL execute_command_line('xtb coords_hess.xyz --hess > xtb_hess.out')
  ahessian = reshape(hessian,(/nvar*nvar/))
  OPEN(unit=1569,file='hessian')
  READ(1569,*)
  do while (.not.eof)
    READ(1569,'(A)',iostat=io) readString
    if (io/=0.or.trim(readString)=="$end") then
      eof = .true.
      exit
    else if (readString(1:7)=="  cycle") then
      ! nothing to do
    else
      READ(readString(1:),*) ahessian(1:5)    
      k=10
      do i = 2, maxline
          READ(1569,'(A)',iostat=io) readString
          m = modulo(i,n+1)
          if(m /= 0) then
            READ(readString(1:),*) ahessian(k-4:k)
            k=k+5
          else
            READ(readString(1:),*) ahessian(k-4:k-5+l)
            k=k+l
          endif
      end do
    end if
  end do
  CLOSE(1569)
  hessian = reshape(ahessian,(/nvar,nvar/))
  status =0
end subroutine xtb_get_hessian

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

! classical flux of the Eckart potential 
! returns log(flux)
subroutine driver_clrate(beta_hbar,flux)
  use driver_parameter_module
  implicit none
  real(rk),intent(in) :: beta_hbar
  real(rk),intent(out):: flux
  real(rk) :: vmax,pi
  pi=4.D0*atan(1.D0)
  vmax=(Va+Vb)**2/(4.D0*Vb)
  print*,"V_max=",vmax
  flux=-beta_hbar*vmax-log(2.D0*pi*beta_hbar)
  
end subroutine driver_clrate

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
      open(unit=20,file="coords_out.xyz")
    end if
    write(20,*) nvar/3
    write(20,*) 
    do iat=1,nvar/3
      ! only H and no conversion to Ang!
      write(20,'("H ",3f12.7)') coords((iat-1)*3+1:(iat-1)*3+3)
    end do
    close(20)
  else
    !print*,"Coords in put_coords: ",coords
  end if
end subroutine dlf_put_coords

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_put_hessian(nvar,hessian,iam)
  use dlf_parameter_module
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: hessian(nvar,nvar)
  integer   ,intent(in)    :: iam
! **********************************************************************
  ! dummy routine, do nothing
end subroutine dlf_put_hessian

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


! **********************************************************************
! **********************************************************************
! The following routine either writes random numbers to a file, or reads
! them. This is to have equal starting conditions for different compilers
subroutine read_rand(arr)
  use dlf_parameter_module, only: rk
  use dlf_global, only : glob
  real(rk)  :: arr(:)
  integer, parameter :: si1=12
  integer, parameter :: si2=3000
  logical,parameter :: readf=.true.
  real(rk) :: ar(si2)
  integer :: l(1),length
  l=ubound(arr)
  length=l(1)
  if(readf) then
    if(length<=si1) then
      open(unit=201,file="random1.bin",form="unformatted")
    else if(length<=si2) then
      open(unit=201,file="random2.bin",form="unformatted")
    else
      call dlf_mpi_finalize() ! only necessary for a parallel build;
                              ! can be present for a serial build
      stop "Too many coordinates to be read from random.bin file"
    end if
    read(201) ar(1:length)
    close(201)
    arr=ar(1:length)
  else
    if (glob%iam == 0) then
       call random_number(ar)
       open(unit=201,file="random1.bin",form="unformatted")
       write(201) ar(1:si1)
       close(201)
       open(unit=201,file="random2.bin",form="unformatted")
       write(201) ar(1:si2)
       close(201)
    end if
  end if
end subroutine read_rand

subroutine driver_eck(vb_,alpha_)
  use dlf_parameter_module
  use driver_parameter_module
  real(rk),intent(out) :: vb_,alpha_
  vb_=vb
  alpha_=alpha
end subroutine driver_eck

subroutine fdgrad2(nvar,coords_in,delta,grad)
  use dlf_parameter_module
  implicit none
  !use driver_parameter_module
  integer,intent(in)  :: nvar
  real(rk),intent(in) :: coords_in(nvar)
  real(rk),intent(in) :: delta
  real(rk),intent(out) :: grad(nvar)
  real(rk) :: coords(nvar),ene
  integer :: ivar

  coords=coords_in
  grad=0.D0
  do ivar=1,nvar

    coords=coords_in
    coords(ivar)=coords(ivar)+delta
    call getpot(coords,grad(ivar))
    if(isnan(grad(ivar))) call dlf_fail("Energy is NaN")
    coords=coords_in
    coords(ivar)=coords(ivar)-delta
    call getpot(coords,ene)
    if(isnan(ene)) call dlf_fail("Energy is NaN")
    grad(ivar)=(grad(ivar)-ene)/(2.d0*delta)
  end do

end subroutine fdgrad2

! finite-difference gradient 4th order
subroutine fdgrad4(nvar,coords_in,delta,grad)
  use dlf_parameter_module
  implicit none
  !use driver_parameter_module
  integer,intent(in)  :: nvar
  real(rk),intent(in) :: coords_in(nvar)
  real(rk),intent(in) :: delta
  real(rk),intent(out) :: grad(nvar)
  real(rk) :: coords(nvar),svar,gradcomp
  integer :: ivar

  coords=coords_in
  grad=0.D0
  do ivar=1,nvar

    gradcomp=0.D0

    coords=coords_in
    coords(ivar)=coords(ivar)-2.D0*delta
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    gradcomp=gradcomp+svar/12.D0

    coords=coords_in
    coords(ivar)=coords(ivar)-1.D0*delta
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    gradcomp=gradcomp-2.D0*svar/3.D0

    coords=coords_in
    coords(ivar)=coords(ivar)+1.D0*delta
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    gradcomp=gradcomp+2.D0*svar/3.D0

    coords=coords_in
    coords(ivar)=coords(ivar)+2.D0*delta
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    gradcomp=gradcomp-svar/12.D0

    grad(ivar)=gradcomp/delta
  end do

end subroutine fdgrad4


! finite-difference hessian 2nd order
subroutine fdhess2(nvar,coords_in,delta,hess)
  use dlf_parameter_module
  implicit none
  !use driver_parameter_module
  integer,intent(in)  :: nvar
  real(rk),intent(in) :: coords_in(nvar)
  real(rk),intent(in) :: delta
  real(rk),intent(out) :: hess(nvar,nvar)
  real(rk) :: coords(nvar),svar,hesscomp
  integer :: ivar,jvar

  coords=coords_in
  hess=0.D0
  do ivar=1,nvar
    ! first the off-diagonal elements
    do jvar=ivar+1,nvar

      hesscomp=0.D0

      coords=coords_in
      coords(ivar)=coords(ivar)+delta
      coords(jvar)=coords(jvar)+delta
      call getpot(coords,svar)
      if(isnan(svar)) call dlf_fail("Energy is NaN")
      hesscomp=hesscomp+svar

      coords=coords_in
      coords(ivar)=coords(ivar)-delta
      coords(jvar)=coords(jvar)+delta
      call getpot(coords,svar)
      if(isnan(svar)) call dlf_fail("Energy is NaN")
      hesscomp=hesscomp-svar

      coords=coords_in
      coords(ivar)=coords(ivar)+delta
      coords(jvar)=coords(jvar)-delta
      call getpot(coords,svar)
      if(isnan(svar)) call dlf_fail("Energy is NaN")
      hesscomp=hesscomp-svar

      coords=coords_in
      coords(ivar)=coords(ivar)-delta
      coords(jvar)=coords(jvar)-delta
      call getpot(coords,svar)
      if(isnan(svar)) call dlf_fail("Energy is NaN")
      hesscomp=hesscomp+svar


      hess(ivar,jvar)=hesscomp/4.D0/delta**2
      hess(jvar,ivar)=hess(ivar,jvar)
    end do

    ! now the diagonal element
    hesscomp=0.D0
    
    coords=coords_in
    coords(ivar)=coords(ivar)+delta
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    hesscomp=hesscomp+svar
    
    coords=coords_in
    !coords(ivar)=coords(ivar)
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    hesscomp=hesscomp-2.D0*svar
    
    coords=coords_in
    coords(ivar)=coords(ivar)-delta
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    hesscomp=hesscomp+svar

    hess(ivar,ivar)=hesscomp/delta**2

  end do

end subroutine fdhess2

! finite-difference hessian 2nd order with only 2 energy evaluations
! per off-diagonal element
subroutine fdhess2_red(nvar,coords_in,delta,hess)
  use dlf_parameter_module
  implicit none
  !use driver_parameter_module
  integer,intent(in)  :: nvar
  real(rk),intent(in) :: coords_in(nvar)
  real(rk),intent(in) :: delta
  real(rk),intent(out) :: hess(nvar,nvar)
  real(rk) :: coords(nvar),svar,hesscomp
  real(rk) :: ecenter,eminus(nvar),eplus(nvar)
  integer :: ivar,jvar

  coords=coords_in
  hess=0.D0

  ! central point (non-disturbed)
  coords=coords_in
  call getpot(coords,svar)
  if(isnan(svar)) call dlf_fail("Energy is NaN")
  ecenter=svar
  
  ! calculate the diagonal elements and eminus, eplus
  eminus=0.D0
  eplus=0.D0
  do ivar=1,nvar
     
     coords=coords_in
     coords(ivar)=coords(ivar)+delta
     call getpot(coords,svar)
     if(isnan(svar)) call dlf_fail("Energy is NaN")
     eplus(ivar)=svar
     
     coords=coords_in
     coords(ivar)=coords(ivar)-delta
     call getpot(coords,svar)
     if(isnan(svar)) call dlf_fail("Energy is NaN")
     eminus(ivar)=svar

     hesscomp=(eminus(ivar)-2.D0*ecenter+eplus(ivar))/delta**2
     hess(ivar,ivar)=hesscomp
  end do

  ! now we could also calculate the gradient
  
  ! now the off-diagonal elements
  do ivar=1,nvar
    do jvar=ivar+1,nvar

      hesscomp=0.D0

      coords=coords_in
      coords(ivar)=coords(ivar)+delta
      coords(jvar)=coords(jvar)+delta
      call getpot(coords,svar)
      if(isnan(svar)) call dlf_fail("Energy is NaN")
      hesscomp=hesscomp+svar

      coords=coords_in
      coords(ivar)=coords(ivar)-delta
      coords(jvar)=coords(jvar)-delta
      call getpot(coords,svar)
      if(isnan(svar)) call dlf_fail("Energy is NaN")
      hesscomp=hesscomp+svar

      hesscomp=hesscomp-eplus(ivar)-eplus(jvar)-eminus(ivar)-eminus(jvar) &
           + 2.D0*ecenter
      hesscomp=hesscomp/2.D0/delta**2
      
      hess(ivar,jvar)=hesscomp
      hess(jvar,ivar)=hesscomp
    end do
  end do

end subroutine fdhess2_red

! finite-difference hessian 4th order
subroutine fdhess4(nvar,coords_in,delta,hess)
  use dlf_parameter_module
  implicit none
  !use driver_parameter_module
  integer,intent(in)  :: nvar
  real(rk),intent(in) :: coords_in(nvar)
  real(rk),intent(in) :: delta
  real(rk),intent(out) :: hess(nvar,nvar)
  real(rk) :: coords(nvar),svar,hesscomp
  integer :: ivar,jvar,divar,djvar
  real(rk) :: coeff(4),dfac(4)

  ! coefficients of delta
  dfac(1)=-2.D0
  dfac(2)=-1.D0
  dfac(3)= 1.D0
  dfac(4)= 2.D0

  ! coefficent of the function value
  coeff(1)= 1.D0/12.D0
  coeff(2)=-2.D0/3.D0
  coeff(3)= 2.D0/3.D0
  coeff(4)=-1.D0/12.D0

  hess=0.D0
  do ivar=1,nvar
    ! first the off-diagonal elements
    do jvar=ivar+1,nvar

      hesscomp=0.D0

      do divar=1,4
        do djvar=1,4

          coords=coords_in
          coords(ivar)=coords(ivar)+dfac(divar)*delta
          coords(jvar)=coords(jvar)+dfac(djvar)*delta
          call getpot(coords,svar)
          if(isnan(svar)) call dlf_fail("Energy is NaN")
          hesscomp= hesscomp + coeff(divar)*coeff(djvar) * svar
          
        end do
      end do

      hess(ivar,jvar)=hesscomp/delta**2
      hess(jvar,ivar)=hess(ivar,jvar)
    end do

    ! now the diagonal element
    hesscomp=0.D0
    
    coords=coords_in
    coords(ivar)=coords(ivar)+2.D0*delta
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    hesscomp=hesscomp-1.D0/12.D0*svar
    
    coords=coords_in
    coords(ivar)=coords(ivar)+1.D0*delta
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    hesscomp=hesscomp+4.D0/3.D0*svar
    
    coords=coords_in
    !coords(ivar)=coords(ivar)
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    hesscomp=hesscomp-2.5D0*svar
    
    coords=coords_in
    coords(ivar)=coords(ivar)-1.D0*delta
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    hesscomp=hesscomp+4.D0/3.D0*svar
    
    coords=coords_in
    coords(ivar)=coords(ivar)-2.D0*delta
    call getpot(coords,svar)
    if(isnan(svar)) call dlf_fail("Energy is NaN")
    hesscomp=hesscomp-1.D0/12.D0*svar
    

    hess(ivar,ivar)=hesscomp/delta**2

  end do

end subroutine fdhess4

! finite-difference hessian 4th order, Equation with only 4 energy
! calculations for diagonal and off-diagonal elements
subroutine fdhess4_red(nvar,coords_in,delta,hess)
  use dlf_parameter_module
  implicit none
  !use driver_parameter_module
  integer,intent(in)  :: nvar
  real(rk),intent(in) :: coords_in(nvar)
  real(rk),intent(out) :: hess(nvar,nvar)
  real(rk),intent(in) :: delta
  real(rk) :: coords(nvar),svar,hesscomp
  integer :: ivar,jvar,divar,djvar
  real(rk) :: coeff(4),dfac(4)
  real(rk) :: ecenter,emov(4,nvar),eoff(4)

  ! undisturbed energy (center point)
  coords=coords_in
  call getpot(coords,svar)
  if(isnan(svar)) call dlf_fail("Energy is NaN")
  ecenter=svar

 ! coefficients of delta
  dfac(1)=-2.D0
  dfac(2)=-1.D0
  dfac(3)= 1.D0
  dfac(4)= 2.D0

  ! coefficent of the function value
  coeff(1)=-1.D0/12.D0
  coeff(2)=4.D0/3.D0
  coeff(3)=4.D0/3.D0
  coeff(4)=-1.D0/12.D0

  hess=0.D0
  
  do ivar=1,nvar

     do divar=1,4
        coords=coords_in
        coords(ivar)=coords(ivar)+dfac(divar)*delta
        call getpot(coords,svar)
        if(isnan(svar)) call dlf_fail("Energy is NaN")
        emov(divar,ivar)=svar
     end do
     hesscomp=sum(coeff(:)*emov(:,ivar))-2.5D0*ecenter
     hess(ivar,ivar)=hesscomp/delta**2
  end do

  ! now we could calculate the gradient

  ! off-diagonal elements
  do ivar=1,nvar
    ! first the off-diagonal elements
    do jvar=ivar+1,nvar

       hesscomp=0.D0
       
       do divar=1,4
          coords=coords_in
          coords(ivar)=coords(ivar)+dfac(divar)*delta
          coords(jvar)=coords(jvar)+dfac(divar)*delta
          call getpot(coords,svar)
          if(isnan(svar)) call dlf_fail("Energy is NaN")
          eoff(divar)=svar
       end do
       hesscomp=sum(coeff(:)*eoff(:))-2.5D0*ecenter
       hesscomp=hesscomp*0.5D0/delta**2-(hess(ivar,ivar)+hess(jvar,jvar))*0.5D0
       hess(ivar,jvar)=hesscomp
       hess(jvar,ivar)=hesscomp
    end do
  end do

end subroutine fdhess4_red


! read parameters from file dlf.in
!
! dlf.in - a simplistic input file for the
! standalone-version of DL-FIND.
!
! Syntax:
! Any characters after a hash sign (#) are ignored
! Input lines in the following order:
!  1 name of the xyz file containing coords
!  2 name of the xyz file containing coords2
!  3 iopt
!  4 icoord
!  5 nimage
!  6 temperature
!  7 nebk
!  8 mass # as many entries as atoms!
! Any lines after that input are ignored as well
!
subroutine read_dlf_in(nvar,coords,coords2,&
      iopt,icoord,nimage,temperature,nebk,mass,dist,tol)
  use dlf_parameter_module
  use dlf_constants
  implicit none
  integer,intent(in)  :: nvar
  real(rk),intent(inout) :: coords(nvar)
  real(rk),intent(inout) :: coords2(nvar)
  integer,intent(inout)  :: iopt,icoord,nimage
  real(rk),intent(inout) :: temperature,nebk,mass(nvar/3),dist,tol
  !
  character(128) :: filename,line
  character(2) :: str2
  integer :: count,iat,nat,ios
  logical :: tchk
  real(rk) :: ang

  filename="dlf.in"
  inquire(file=filename,exist=tchk)

  if(.not.tchk) then
    print*,"read_dlf_in called but no file dlf.in present"
    print*,"leaving data as is!"
    return
  end if

  call dlf_constants_get("ANG_AU",ang)

  print*
  print*,"Reading dlf.in"
  open(unit=23,file=filename)
  count=0
  do
!    READ (23,'(a)',err=203,end=203) line
    READ (23,'(a)',end=203) line
    if(index(line,"#")>0) line=line(1:index(line,"#")-1)
!    if(trim(line)=="") cycle
    count=count+1
!    print*,count,"--",trim(line),"--"
    select case (count)
    case (1,2)
      read(line,'(a)') filename
      inquire(file=filename,exist=tchk)
      if(.not.tchk) then
        print*,"File ",trim(filename)," not found"
        print*,"Input coordinates unchanged"
      else
         ! decide between xyz file and .txt (which is interpreted as PES point)
         if(index(filename,"xyz")/=0) then
            !if(index(filename,"xyz")/=-1) then
            open(unit=24,file=filename)
            read(24,'(a)') line
            read(line,*) nat
            if(nat*3/=nvar) print*,"Warning: number of atoms not consistent!"
            read(24,'(a)')
            ! read coords
            if(count==1) then
               do iat=1,nvar/3
                  read(24,*) str2, coords(3*iat-2:3*iat)
               end do
               ! ang conversion commented out
               coords=coords/ang
               print*,"Coords sucessfully read from file ",trim(filename)
            else
               do iat=1,nvar/3
                  read(24,*) str2, coords2(3*iat-2:3*iat)
               end do
               ! ang conversion
               coords2=coords2/ang
               print*,"Coords2 sucessfully read from file ",trim(filename)
            end if
            close(24)
         else if(index(filename,"txt")/=0) then
            open(unit=24,file=filename)
            read(24,'(a)') line
            read(24,'(a)') line
            read(line,*) nat
            if(nat*3/=nvar) print*,"Warning: number of atoms not consistent!"
            read(24,'(a)') line
            read(24,'(a)') line 
            read(24,'(a)') line
            read(24,'(a)') line
            print*,"read",trim(line)
            if(count==1) then
               do iat=1,nvar/3
                  read(24,*) coords(3*iat-2:3*iat)
               end do
               print*,"Coords sucessfully read from file ",trim(filename)
            else
               do iat=1,nvar/3
                  read(24,*) coords2(3*iat-2:3*iat)
               end do
               print*,"Coords2 sucessfully read from file ",trim(filename)
            end if
            close(24)
         else
            call dlf_fail("Input file format can not be determined (neither xyz nor txt).")
         end if
      end if
    case (3)
      read(line,*,iostat=ios) iopt
      if(ios==0) then
        print*,"iopt is set to ",iopt
      else
        print*,"iopt NOT read, remains ",iopt
      end if
    case (4)
      read(line,*,iostat=ios) icoord
      if(ios==0) then
        print*,"icoord is set to ",icoord
      else
        print*,"icoord NOT read, remains ",icoord
      end if
    case (5)
      read(line,*,iostat=ios) nimage
      if(ios==0) then
        print*,"nimage is set to ",nimage
      else
        print*,"nimage NOT read, remains ",nimage
      end if
    case (6)
      read(line,*,iostat=ios) temperature
      if(ios==0) then
        print*,"temperature is set to ",temperature
      else
        print*,"temperature NOT read, remains ",temperature
      end if
    case (7)
      read(line,*,iostat=ios) nebk
      if(ios==0) then
        print*,"nebk is set to ",nebk
      else
        print*,"nebk NOT read, remains ",nebk
      end if
    case (8)
      read(line,*,iostat=ios) mass
      if(ios==0) then
        print*,"mass is set to ",mass
      else
        print*,"mass NOT read, remains ",mass
      end if
    case (9)
      read(line,*,iostat=ios) dist
      if(ios==0) then
        print*,"distort is set to ",dist
      else
        print*,"distort NOT read, remains ",dist
      end if
    case (10)
      read(line,*,iostat=ios) tol
      if(ios==0) then
        print*,"tolerance is set to ",tol
      else
        print*,"tolerance NOT read, remains ",tol
      end if

      count=count+1
    end select
  end do
203 continue
  close(23)

  print*,"Finished reading dlf.in"
  print*

end subroutine read_dlf_in

! for 1-D Potentials: print energy, gradient and Hessian for test
subroutine test_egh
  implicit none
  integer :: ix,nx,ivar
  real(8) :: xvar,coords(3),grad(3),hess(3,3),energy
  integer :: ierr

  open(unit=199,file="test_egh.dat")
  nx=500
  do ix=1,nx
    xvar=dble(ix-nx/2)/dble(nx)*2.D0
    coords=0.D0
    coords(1)=xvar
    ivar=1
    call dlf_get_gradient(3,coords,energy,grad,ivar,ivar,ivar,ierr)
    call dlf_get_hessian(3,coords,hess,ivar)
    write(199,'(4e22.14)') xvar,energy,grad(1),hess(1,1)
  end do
  close(199)
  
  call dlf_fail("Stop in test_egh")

end subroutine test_egh

! routines for using neural network interpolations

!module nn_module
!  use dlf_parameter_module, only: rk
!  implicit none
!  integer :: ni,nj,nk
!  logical :: nnmassweight
!  real(rk),allocatable,save :: wone(:,:),wtwo(:,:),wthree(:),bone(:),btwo(:) !wone(nj,ni),wtwo(nk,nj),wthree(nk),bone(nj),btwo(nk)
!  real(rk),save :: bthree,alpha
!  real(rk),allocatable,save :: align_refcoords(:)     ! (3*nat)
!  real(rk),allocatable,save :: align_modes(:,:) ! (3*nat,ni)
!  real(rk),allocatable,save :: refmass(:) ! (nat)
!  logical :: temin
!  real(rk) :: emin
!end module nn_module

!subroutine nn_init(infile,nvar)
!  use nn_module
!  implicit none
!  integer,intent(out) :: nvar
!  character(*),intent(in) :: infile
!  integer(4) :: ns,nat,ios ! not used, except for printout
!  integer(4) :: ni4,nj4,nk4,coord_system4
!  logical(4) :: nmw4
!  open(unit=40,file=infile,form='unformatted')
!  read(40,iostat=ios) ni4,nj4,nk4,ns,nat,nmw4,coord_system4
!  if(ios/=0) then
!    !read(40,iostat=ios) ni4,nj4,nk4,ns,nat,nmw4
!    print*,"Coordinate system can not be read in, assuming normal coordinates"
!    coord_system4=0
!  end if
!  nnmassweight=nmw4
!  ni=ni4
!  nj=nj4
!  nk=nk4
!  !  nnmassweight=.false.
!  print*,"Reading neural network data from file: ",trim(adjustl(infile))
!  print*,"Number of atoms",nat
!  print*,"Number of coordinates",ni
!  print*,"Number of variables in first hidden layer",nj
!  print*,"Number of variables in second hidden layer",nk
!  print*,"Number of input geometries",ns
!  print*,"Mass weighting",nnmassweight
!  if(coord_system4==0) print*,"Mass-weighted normal coordinates"

!  nvar=3*nat

!  allocate(wone(nj,ni))
!  allocate(wtwo(nk,nj))
!  allocate(wthree(nk))
!  allocate(bone(nj))
!  allocate(btwo(nk))
!  allocate(align_refcoords(nvar))
!  allocate(align_modes(nvar,ni))
!  allocate(refmass(nat))

!  read(40) wone
!  read(40) wtwo
!  read(40) wthree
!  read(40) bone
!  read(40) btwo
!  read(40) bthree
!  read(40) align_refcoords
!  read(40) align_modes
!  read(40) alpha
!  !read(40,iostat=ios) alpha
!  !if(ios/=0) then
!  !  alpha=0.D0 ! this can be removed once all .dat files contain alpha
!  !  print*,"Alpha not read from file."
!  !end if
!  !print*,"Alpha",alpha
!  if(nnmassweight) then
!    read(40) refmass
!  else
!    refmass=1.D0
!  end if
!  read(40,iostat=ios) temin
!  if(ios/=0) then
!    print*,"Information about minimum energy not read"
!    temin=.false.
!  end if
!  if(temin) then
!    read(40) emin
!    print*,"Minimum energy of NN: ",emin
!  end if
!  close(40)

!end subroutine nn_init

!subroutine nn_destroy
!  use nn_module
!  implicit none

!  deallocate(wone)
!  deallocate(wtwo)
!  deallocate(wthree)
!  deallocate(bone)
!  deallocate(btwo)
!  deallocate(align_refcoords)
!  deallocate(align_modes)
!  deallocate(refmass)

!end subroutine nn_destroy

!subroutine nn_get_gradient(nvar,xcoords,energy,xgradient)
!  use dlf_parameter_module, only: rk
!  use nn_module
!  implicit none
!  integer, intent(in) :: nvar
!  real(rk), intent(in) :: xcoords(nvar) 
!  real(rk), intent(out) :: energy, xgradient(nvar)
!  real(rk) :: yone(nj),ytwo(nk)
!  real(rk) :: dyone(nj),dytwo(nk) ! dyone = f_1'
!  integer :: ii,ij,ik,iat,jat
!  real(rk) :: dcoords(ni),dgradient(ni),trans(3),rotmat(3,3)
!  real(rk) :: energy_dir
!  real(rk) :: drotmat(3,3,nvar),com(3),tmpvec(3),dcj_dxi(nvar,nvar),svar

!  call cgh_xtos(ni,nvar,align_refcoords,xcoords,trans,rotmat,dcoords)

!  yone=tanh(bone+matmul(wone,dcoords))
!  dyone=1.D0-yone**2
!  ytwo=tanh(btwo+matmul(wtwo,yone))
!  dytwo=1.D0-ytwo**2

!  energy=bthree+sum(wthree*ytwo)+alpha*sum(dcoords**2)

!  do ii=1,ni
!    dgradient(ii)=2.D0*alpha*dcoords(ii)
!    do ik=1,nk
!      do ij=1,nj
!        dgradient(ii)=dgradient(ii)+(wthree(ik)*wtwo(ik,ij)*wone(ij,ii)*dytwo(ik)*dyone(ij))
!      end do
!    end do
!  end do

!  if(temin) then
!    dgradient=2.D0*energy*dgradient
!    energy=emin+energy**2
!  end if

!  ! transform from dgradient to gradient
!  xgradient=matmul(align_modes,dgradient)

!  ! re-mass-weight
!  if(nnmassweight) then
!    do iat=1,nvar/3
!      xgradient(iat*3-2:iat*3)=xgradient(iat*3-2:iat*3)*sqrt(refmass(iat))
!    end do
!  end if

!  ! now we have gradient=dE/dc_j (i.e. the gradient with respect to
!  ! aligned coordinates) 

!  !now transform to dE/dx_i
!  call get_drotmat(ni,nvar,align_refcoords,&
!       xcoords,drotmat)
!  
!  ! calculate center of mass of xcoords
!  com=0.D0
!  do iat=1,nvar/3
!     com(:)=com(:)+xcoords(iat*3-2:iat*3)*refmass(iat)
!  end do
!  com=com/sum(refmass)
!      
!  dcj_dxi=0.D0
!  svar=sum(refmass)
!  do iat=1,nvar/3
!     dcj_dxi(iat*3-2:iat*3,iat*3-2:iat*3)=transpose(rotmat)
!  end do

!  do iat=1,nvar/3
!     do jat=1,nvar
!        tmpvec=matmul(drotmat(:,:,jat),(xcoords(iat*3-2:iat*3)-com))
!        dcj_dxi(jat,iat*3-2:iat*3)= dcj_dxi(jat,iat*3-2:iat*3) + tmpvec
!     end do
!  end do
!  
!  xgradient=matmul(dcj_dxi,xgradient)

!  
!!  ! now transform gradient back:
!!  do iat=1,nvar/3
!!    xgradient(iat*3-2:iat*3)=matmul(transpose(rotmat),xgradient(iat*3-2:iat*3))
!!  end do
!end subroutine nn_get_gradient

!subroutine get_drotmat(ncoord,nvar,rcoords,xcoords_,drotmat)
!  use nn_module
!  implicit none
!  integer,intent(in) :: ncoord,nvar
!  real(8),intent(in) :: rcoords(nvar) ! the set of coordinates the new ones should be fitted to
!  real(8),intent(in) :: xcoords_(nvar) 
!  real(8),intent(out) :: drotmat(3,3,nvar)
!  real(8) :: trans(3),dcoords(ncoord),tmpmat(3,3)
!  integer :: ivar
!  real(8) :: delta=1.D-5
!  real(8) :: xcoords(nvar)
!  !print*,"FD rotmat with delta",delta
!  do ivar=1,nvar
!    xcoords=xcoords_
!    xcoords(ivar)=xcoords(ivar)+delta
!    call cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,drotmat(:,:,ivar),dcoords)
!    xcoords(ivar)=xcoords(ivar)-2.D0*delta
!    call cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat,dcoords)
!    drotmat(:,:,ivar)=(drotmat(:,:,ivar)-tmpmat)/2.D0/delta
!  end do
!end subroutine get_drotmat


!subroutine nn_get_hessian(nvar,xcoords,xhessian)
!  use dlf_parameter_module, only: rk
!  use nn_module
!  implicit none
!  integer, intent(in) :: nvar
!  real(rk), intent(in) :: xcoords(nvar)
!  real(rk), intent(out) :: xhessian(nvar,nvar)
!  real(rk) :: yone(nj),ytwo(nk)
!  real(rk) :: dyone(nj),dytwo(nk) ! dyone = f_1'
!  integer :: ii,ij,ik,iip,iat,jat
!  real(rk) :: amat(nk,ni)
!  real(rk) :: dcoords(ni),dhessian(ni,ni),trans(3),rotmat(3,3)
!  real(rk) :: energy,dgradient(ni)

!  call cgh_xtos(ni,nvar,align_refcoords,xcoords,trans,rotmat,dcoords)

!  print*,"Warning: rotation not treated properly here yet...!"
!  
!!print*,"dcoords to evaluate Hessian at",dcoords

!!print*,"Mass",glob%mass

!  yone=tanh(bone+matmul(wone,dcoords))
!  dyone=1.D0-yone**2
!  ytwo=tanh(btwo+matmul(wtwo,yone))
!  dytwo=1.D0-ytwo**2
!  !!print*,"ytwo",ytwo
!  amat=0.D0
!  ! order: ni*nk*nj
!  do ii=1,ni
!    do ik=1,nk
!      amat(ik,ii)=sum(wtwo(ik,:)*dyone(:)*wone(:,ii))
!    end do
!  end do
!!!$print*,"amat",amat
!!!$print*,"wthree",wthree
!!!$print*,"ytwo",ytwo
!!!$print*,"dytwo",dytwo
!!!$print*,"w2",wtwo
!!!$print*,"wone",wone
!!!$print*,"yone",yone
!!!$print*,"dyone",dyone
!  dhessian=0.D0
!  ! order: ni*ni*nk*nj = ni^2 * nj * nk
!  do ii=1,ni
!    do iip=ii,ni
!      ! sum
!      do ik=1,nk
!        dhessian(ii,iip)=dhessian(ii,iip)-2.D0*wthree(ik)* &
!            (ytwo(ik)*dytwo(ik)*amat(ik,ii)*amat(ik,iip) + &
!            dytwo(ik)*sum(wtwo(ik,:)*wone(:,ii)*wone(:,iip)*yone(:)*dyone(:)))
!        !if(ii==1.and.iip==1) print*,"JKK",ik,dhessian(ii,iip)
!      end do
!      dhessian(iip,ii)=dhessian(ii,iip)
!    end do
!    dhessian(ii,ii)=dhessian(ii,ii)+2.D0*alpha
!  end do

!  ! transform dhessian to emin if required
!  if(temin) then
!    ! have to re-calculate energy and gradient for conversion:
!    energy=bthree+sum(wthree*ytwo)+alpha*sum(dcoords**2)
!    do ii=1,ni
!      dgradient(ii)=2.D0*alpha*dcoords(ii)
!      do ik=1,nk
!        do ij=1,nj
!          dgradient(ii)=dgradient(ii)+(wthree(ik)*wtwo(ik,ij)*wone(ij,ii)*dytwo(ik)*dyone(ij))
!        end do
!      end do
!    end do

!    dhessian=energy*dhessian
!    do ik=1,ni
!      do ij=1,ni
!        dhessian(ik,ij)=dhessian(ik,ij)+dgradient(ik)*dgradient(ij)
!      end do
!    end do
!    dhessian=dhessian*2.D0
!  end if ! (temin)


!  ! transform from dgradient to gradient
!  xhessian=matmul(align_modes,matmul(dhessian,transpose(align_modes)))

!  ! re-mass-weight
!  if(nnmassweight) then
!    do iat=1,nvar/3
!      do jat=1,nvar/3
!        xhessian(iat*3-2:iat*3,jat*3-2:jat*3)=xhessian(iat*3-2:iat*3,jat*3-2:jat*3)&
!            *sqrt(refmass(iat)*refmass(jat))
!      end do
!    end do
!  end if

!  ! now transform hessian back:
!  do iat=1,nvar/3
!    do jat=1,nvar/3
!      xhessian(iat*3-2:iat*3,jat*3-2:jat*3)=matmul(matmul(transpose(rotmat),xhessian(iat*3-2:iat*3,jat*3-2:jat*3)),rotmat) 
!    end do
!  end do

!end subroutine nn_get_hessian
!!-------------------------------------------------------------------------

!! superimpose coords to rcoords and transform gradient and hessian in
!! an appropriate way as well. Return transformed coordinates, ...
!! this is the same routine as in readin.f90 of the NN fitting code, but with the gradient, hessian removed.
!! (the name of the module is also changed: pes_module -> nn_module)
!! nvar is added (3*nat replaced by nvar)
!!
!! Algorithm for superposition:
!! !! See W. Kabsch, Acta Cryst. A 32, p 922 (1976).
!!
!subroutine cgh_xtos(ncoord,nvar,rcoords,xcoords_,trans,rotmat,dcoords)
!  use nn_module
!  implicit none
!  integer,intent(in) :: ncoord,nvar
!  real(8),intent(in) :: rcoords(nvar) ! the set of coordinates the new ones should be fitted to
!  real(8),intent(in) :: xcoords_(nvar) 
!  real(8),intent(out) :: trans(3),rotmat(3,3)
!  real(8),intent(out) :: dcoords(ncoord)

!  integer:: iat,ivar,jvar,jat,fid,nat
!  real(8) :: rmat(3,3),rsmat(3,3),eigvec(3,3),eigval(3)
!  real(8) :: center(3)
!  real(8) :: weight(nvar)
!  real(8) :: xcoords(nvar)

!  integer :: itry,i,j
!  real(8) :: detrot

!  xcoords=xcoords_

!  nat=nvar/3

!  if(nnmassweight) then
!    do iat=1,nat
!      weight(iat*3-2:iat*3)=refmass(iat)
!    end do
!  else
!    weight=1.D0
!  end if


!  ! as compared to dlf_cartesian_align: coords1=rcoords coords2=coords

!  trans=0.D0
!  rotmat=0.D0
!  do ivar=1,3
!    rotmat(ivar,ivar)=1.D0
!  end do

!  ! if there are other cases to ommit a transformation, add them here
!  if(nat==1) return
!  !if(.not.superimpose) return

!  trans=0.D0
!  center=0.D0
!  do iat=1,nat
!    center(:)=center(:)+rcoords(iat*3-2:iat*3)*weight(iat*3-2:iat*3)
!    trans(:)=trans(:)+(xcoords(iat*3-2:iat*3)-rcoords(iat*3-2:iat*3))*weight(iat*3-2:iat*3)
!  end do
!  trans=trans/sum(weight)*3.D0
!  center=center/sum(weight)*3.D0

!  !print*,"# trans",trans

!  ! translate them to common centre
!  do iat=1,nat
!    xcoords(iat*3-2:iat*3)=xcoords(iat*3-2:iat*3)-trans(:)
!  end do

!  rmat=0.D0
!  do iat=1,nat
!    do ivar=1,3
!      do jvar=1,3
!        rmat(ivar,jvar)=rmat(ivar,jvar)+weight(3*iat)*(rcoords(ivar+3*iat-3)-center(ivar))* &
!            (xcoords(jvar+3*iat-3)-center(jvar))
!      end do
!    end do
!  end do
!  rmat=rmat/sum(weight)*3.D0
!  !write(*,"('R   ',3f10.3)") rmat
!  rsmat=transpose(rmat)
!  eigvec=matmul(rsmat,rmat)
!  rsmat=eigvec

!  !write(stdout,"('RtR ',3f10.3)") rsmat
!  call matrix_diagonalise(3,rsmat,eigval,eigvec)

!  ! It turns out that the rotation matrix may have a determinat of -1
!  ! in the procedure used here, i.e. the system is mirrored - which is
!  ! wrong chemically. This can be avoided by inserting a minus in the
!  ! equation
!  ! 1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))

!  ! So, here we first calculate the rotation matrix, and if it is
!  ! zero, the first eigenvalue is reversed

!  do itry=1,2
!    ! rsmat are the vectors b:
!    j=-1
!    do i=1,3
!      if(eigval(i)<1.D-8) then
!        if(i>1) then
!          ! the system is linear - no rotation necessay.
!          ! WHY ?! There should still be one necessary!
!          return
!          !print*,"Eigenval. zero",i,eigval(i)
!          !call dlf_fail("Error in dlf_cartesian_align")
!        end if
!        j=1
!      else
!        if(i==1.and.itry==2) then
!          rsmat(:,i)=-1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
!        else
!          rsmat(:,i)=1.d0/dsqrt(eigval(i)) * matmul(rmat,eigvec(:,i))
!        end if
!      end if
!    end do
!    if(j==1) then
!      ! one eigenvalue was zero, the system is planar
!      rsmat(1,1)=rsmat(2,2)*rsmat(3,3)-rsmat(3,2)*rsmat(2,3)
!      rsmat(2,1)=rsmat(3,2)*rsmat(1,3)-rsmat(1,2)*rsmat(3,3)
!      rsmat(3,1)=rsmat(1,2)*rsmat(2,3)-rsmat(2,2)*rsmat(1,3)
!      ! deal with negative determinant
!      if (itry==2) then
!         rsmat(:,1) = -rsmat(:,1)
!      end if
!    end if

!    do i=1,3
!      do j=1,3
!        rotmat(i,j)=sum(rsmat(i,:)*eigvec(j,:))
!      end do
!    end do
!    !write(*,"('rotmat ',3f10.3)") rotmat
!    detrot=   &
!        rotmat(1,1)*(rotmat(2,2)*rotmat(3,3)-rotmat(2,3)*rotmat(3,2)) &
!        -rotmat(2,1)*(rotmat(1,2)*rotmat(3,3)-rotmat(1,3)*rotmat(3,2)) &
!        +rotmat(3,1)*(rotmat(1,2)*rotmat(2,3)-rotmat(1,3)*rotmat(2,2))
!    !write(*,*) "Determinat of rotmat", detrot
!    if(detrot > 0.D0) exit
!    if(detrot < 0.D0 .and. itry==2) then
!      stop "Error in dlf_cartesian_align, obtained a mirroring instead of rotation."
!    end if

!  end do


!!!$  do ivar=1,3
!!!$    rsmat(:,ivar)=1.d0/dsqrt(eigval(ivar)) * matmul(rmat,eigvec(:,ivar))
!!!$  end do
!!!$
!!!$  do ivar=1,3
!!!$    do jvar=1,3
!!!$      rotmat(ivar,jvar)=sum(rsmat(ivar,:)*eigvec(jvar,:))
!!!$    end do
!!!$  end do

!  ! transform coordinates
!  do iat=1,nat
!    xcoords(iat*3-2:iat*3)= xcoords(iat*3-2:iat*3)-center
!    xcoords(iat*3-2:iat*3)=matmul(rotmat,xcoords(iat*3-2:iat*3))
!    xcoords(iat*3-2:iat*3)= xcoords(iat*3-2:iat*3)+center
!  end do

!!!$  ! write xyz
!!!$  if(ttrain) then
!!!$    fid=52
!!!$  else
!!!$    fid=53
!!!$  end if
!!!$  write(fid,*) nat
!!!$  write(fid,*) 
!!!$  do iat=1,nat
!!!$    write(fid,'(" H ",3f12.7)') xcoords(iat*3-2:iat*3)*5.2917720810086D-01
!!!$  end do
!  
!!  print*,"transformed coordinates"
!!  write(*,'(3f15.5)') coords
!  
!  ! now all quantities have been transformed to c-coords (or relative to c-coordinates)

!  ! now, the coordinates need to be mass-weighted!
!  dcoords=matmul(transpose(align_modes),sqrt(weight)*(xcoords-align_refcoords))

!end subroutine cgh_xtos


!! this needs to be removed when linked to dl-find! - why? It does not lead to clashes
!SUBROUTINE matrix_diagonalise(N,H,E,U) 
!  IMPLICIT NONE

!  LOGICAL(4) ,PARAMETER :: TESSLERR=.FALSE.
!  INTEGER   ,INTENT(IN) :: N
!  REAL(8)   ,INTENT(IN) :: H(N,N)
!  REAL(8)   ,INTENT(OUT):: E(N)
!  REAL(8)   ,INTENT(OUT):: U(N,N)
!  REAL(8)               :: WORK1((N*(N+1))/2)
!  REAL(8)               :: WORK2(3*N)
!  INTEGER               :: K,I,J
!  CHARACTER(8)          :: SAV2101
!  INTEGER               :: I1,I2
!  INTEGER               :: INFO
!  INTEGER               :: INF1
!  INTEGER               :: INF2

!  K=0
!  DO J=1,N
!    DO I=J,N
!      K=K+1
!      WORK1(K)=0.5D0*(H(I,J)+H(J,I))
!    ENDDO
!  ENDDO

!  CALL dspev('V','L',N,WORK1,E,U,N,WORK2,INFO) !->LAPACK intel
!  IF(INFO.NE.0) THEN
!    PRINT*,'DIAGONALIZATION NOT CONVERGED'
!    STOP
!  END IF

!END SUBROUTINE MATRIX_DIAGONALISE

#define VPT2_PRINT_FC 1
!!#define LJ_INTERFACE
!!#define LJ_INTERFACE_NOISE
!!#define ANALYTICAL_CUB_QUART
!!#define VPT2_HESS_DEBUG_OUT

! New Lennard-Jones code that implements 1st to 4th analytic
! derivatives. This can be used when DL-FIND is compiled in 
! stand-alone mode, for testing purposes. Additional definition 
! of the LJ_INTERFACE_NOISE macro adds artificial noise to the 
! gradients and Hessians.

#ifdef LJ_INTERFACE
include "dlf_vpt2_lj_pes_new.mod.fortincl"
#endif

! Module containing the main VPT2 driver routine, 
! called by dlf_vpt2_wrap() in dlf_formstep.f90
! Activated within DL_FIND with the option iopt=15

module dlf_vpt2_main
use dlf_parameter_module
#ifdef LJ_INTERFACE
#ifdef LJ_INTERFACE_NOISE
use dlf_vpt2_lj_pes_new, only: & 
       & dlf_get_energy_lj, dlf_get_noisy_gradient_lj, dlf_get_noisy_hessian_lj, & 
       & dlf_vpt2_lj_pes_new_init,dlf_get_cubic_quartic_lj
#else
use dlf_vpt2_lj_pes_new, only: & 
       & dlf_get_energy_lj, dlf_get_gradient_lj, dlf_get_hessian_lj, & 
       & dlf_vpt2_lj_pes_new_init,dlf_get_cubic_quartic_lj
#endif
#endif

implicit none

#ifdef LJ_INTERFACE
#ifdef LJ_INTERFACE_NOISE
interface dlf_get_energy
  procedure dlf_get_energy_lj
end interface
interface dlf_get_gradient
  procedure dlf_get_noisy_gradient_lj
end interface
interface dlf_get_hessian
  procedure dlf_get_noisy_hessian_lj
end interface
#else
interface dlf_get_energy
  procedure dlf_get_energy_lj
end interface
interface dlf_get_gradient
  procedure dlf_get_gradient_lj
end interface
interface dlf_get_hessian
  procedure dlf_get_hessian_lj
end interface
#endif
#endif

contains

! Auxiliar routine to check if molecule is linear 
! in order to set the parameter glob%nzero to 5. 
! This routine should be called before dlf_vpt2
! because there, a lot of automatic size arrays 
! are allocated that depend on the nzero parameter

subroutine dlf_vpt2_check_if_linear(is_linear)
  use dlf_parameter_module
  use dlf_global, only: glob
  use dlf_vpt2_project, only: shift_com, rotate_to_princ_axes_ordered, moi
  use dlf_constants
  implicit none
  logical, intent(out) :: is_linear
  
  integer :: i
  real(rk), dimension(glob%nvar) :: coords
  real(rk), dimension(glob%nat)  :: mv,xx,yy,zz,xnew,ynew,znew
  real(rk) :: Arot,Brot,Crot
  real(rk), dimension(3,3) :: moments_of_inertia
  real(rk) :: amu2au

  call dlf_constants_get('AMU',amu2au)
  
  coords=pack(glob%xcoords,.true.)
  mv(1:glob%nat)=glob%mass(1:glob%nat)*amu2au
  
  ! Coordinate shifting and reorientation
  
   do i=1,glob%nat
    xx(i)=coords(3*(i-1)+1)
    yy(i)=coords(3*(i-1)+2)
    zz(i)=coords(3*(i-1)+3)
  enddo
  
  call shift_com(glob%nat,xx,yy,zz,mv,xnew,ynew,znew)
  xx=xnew
  yy=ynew
  zz=znew
  moments_of_inertia=moi(glob%nat,xx,yy,zz,mv)
  call rotate_to_princ_axes_ordered(glob%nat,xx,yy,zz,mv,xnew,ynew,znew,Arot,Brot,Crot)
  
  is_linear=.false.
  
  if (abs(Arot+666._rk) < 1.e-2_rk) is_linear=.true.
  if (abs(Brot+666._rk) < 1.e-2_rk) is_linear=.true.
  if (abs(Crot+666._rk) < 1.e-2_rk) is_linear=.true.
  
  return
end subroutine dlf_vpt2_check_if_linear

!
! The main VPT2 driver code
!

subroutine dlf_vpt2()
use dlf_linalg_interface_mod, only: dlf_matrix_ortho_trans, dlf_matmul_simp
use dlf_vpt2_utility, only: matrix_output, symb2mass, vector_output, error_print, &
                            & dlf_gl_bcast, znuc2symb, redo_energy_option
use dlf_vpt2_freq, only: get_frequencies_mw_ts_or_min, print_frequencies
use dlf_vpt2_hess_deriv
use dlf_vpt2_terms, only: vpt2_driver, vpt2_get_zpve, vpt2_fundamentals, vpt2_init_fermi_res_params
use dlf_vpt2_part_func, only: qvib_harmonic, qvib_anharmonic, qvib_anharmonic_via_rho, &
                              & generate_bdens_input, generate_sctst_input
use dlf_vpt2_project, only: shift_com, rotate_to_princ_axes_ordered,moi
!use dlf_vpt2_test_routines
use dlf_allocate
use dlf_constants
use dlf_global, only: glob, stdout
implicit none
!interface
!  subroutine dlf_get_hessian(nvar,coords,hessian,status)
!    use dlf_parameter_module
!    implicit none
!    integer   ,intent(in)    :: nvar
!    real(rk)  ,intent(in)    :: coords(nvar)
!    real(rk)  ,intent(out)   :: hessian(nvar,nvar)
!    integer   ,intent(out)   :: status
!  end subroutine dlf_get_hessian
!end interface
!interface
!subroutine dlf_get_gradient(nvar,coords,energy,gradient,iimage,kiter,status)
!  use dlf_parameter_module, only: rk
!    implicit none
!    integer   ,intent(in)    :: nvar
!    real(rk)  ,intent(in)    :: coords(nvar)
!    real(rk)  ,intent(out)   :: energy
!    real(rk)  ,intent(out)   :: gradient(nvar)
!    integer   ,intent(in)    :: iimage
!    integer   ,intent(in)    :: kiter
!    integer   ,intent(out)   :: status
!  end subroutine dlf_get_gradient
!end interface

! Control parameters (will be obtained from dlf_global_module)
integer              :: vpt2_resonance_treatment
integer              :: vpt2_resonance_criterion
real(rk)             :: vpt2_res_tol_deltae
real(rk)             :: vpt2_res_tol_deltae_hard
real(rk)             :: vpt2_res_tol_martin
real(rk)             :: vpt2_res_tol_isaacson
real(rk)             :: vpt2_hdcpt2_alpha
real(rk)             :: vpt2_hdcpt2_beta
real(rk)             :: vpt2_deriv_deltaq
real(rk)             :: vpt2_asym_tol
real(rk)             :: vpt2_Tmin
real(rk)             :: vpt2_Tmax
integer              :: vpt2_nT
logical              :: vpt2_do_part_func
logical              :: vpt2_grad_only
logical              :: vpt2_force_doscf
real(rk)             :: vpt2_dq_factor_4hess
!
real(rk), dimension(:),allocatable :: Temps, qanh_via_rho
integer :: i,j,k
integer :: istat,nimag,nvareff,nreal
character(len=100) :: resonance_mode
character(len=100) :: fermi_crit
logical, parameter  :: project_hessian=.true.
real(rk), dimension(glob%nvar) :: coords
real(rk), dimension(glob%nat)  :: mv,xx,yy,zz,xnew,ynew,znew
real(rk), dimension(3*glob%nat)  :: mv3
real(rk), dimension(3*glob%nat,3*glob%nat)  :: invsqrtmass
real(rk) :: Arot,Brot,Crot,rdum,kB_cm
real(rk), dimension(3,3) :: moments_of_inertia
real(rk), dimension(glob%nvar,glob%nvar) :: hessian
real(rk), dimension(glob%nat) :: mass_loc
character(len=2), dimension(glob%nat) :: atsym
real(rk), dimension(glob%nvar,glob%nvar-glob%nzero) :: normal_mode_vectors, normal_mode_vectors_nonmw
real(rk), dimension(glob%nvar,glob%nvar-glob%nzero) :: fake_nm
real(rk), dimension(glob%nvar-glob%nzero) :: freqs,eigenvalues_au
real(rk), dimension(glob%nvar-glob%nzero,glob%nvar-glob%nzero,glob%nvar-glob%nzero) :: & 
                    & cubic_nm, quartic_nm_semidiag, cubic_reduced, quartic_reduced
logical :: saddle_point
character(len=1000) :: fn, cmdline
character(len=15) :: printch
real(rk) :: E0, zpve_harm, zpve_anh_direct, zpve_anh_explicit_sum, xFF, delT
real(rk) :: Temp,qhar,qanh_simp,qanh_direct
real(rk), dimension(glob%nvar-glob%nzero,glob%nvar-glob%nzero) :: X
real(rk), dimension(glob%nvar-glob%nzero) :: vpt2_funda,xiF
real(rk) :: amu2au
logical  :: new_punch
integer  :: nhess,nhessdone
integer, dimension(-(glob%nvar-glob%nzero):(glob%nvar-glob%nzero))  :: displacement_map
real(rk), dimension(glob%nvar,glob%nvar,2*(glob%nvar-glob%nzero)+1) :: hessians_cart
logical, dimension(2*(glob%nvar-glob%nzero)+1) :: hessdone
integer :: ngrad,ngraddone,ngrad4hess,ngraddone4hess,ngradhalf
integer, dimension(glob%nvar-glob%nzero,glob%nvar-glob%nzero,-1:1,-1:1) :: displacement_map_grad4hess
!integer, dimension(glob%nvar-glob%nzero,glob%nvar-glob%nzero,-1:1,-1:1) :: displacement_map_grad
integer, dimension(glob%nvar-glob%nzero,glob%nvar-glob%nzero,-2:2,-2:2) :: displacement_map_grad_dual
!real(rk), dimension(glob%nvar,2*(glob%nvar-glob%nzero)**2) :: grad_cart
real(rk), dimension(glob%nvar,2*(glob%nvar-glob%nzero))    :: grad_cart_4hess
real(rk), dimension(glob%nvar,4*(glob%nvar-glob%nzero)**2) :: grad_cart_dual
!integer, dimension(2*(glob%nvar-glob%nzero)**2,4) :: joblist_grad
integer, dimension(2*(glob%nvar-glob%nzero),4)    :: joblist_grad4hess
integer, dimension(4*(glob%nvar-glob%nzero)**2,4) :: joblist_grad_dual
!logical, dimension(2*(glob%nvar-glob%nzero)**2)   :: graddone
logical, dimension(2*(glob%nvar-glob%nzero))      :: graddone4hess
logical, dimension(4*(glob%nvar-glob%nzero)**2)   :: graddone_dual
integer :: io,ip,so,sp
character(len=23) :: timestamp
!!integer :: irank, isize
integer :: nbas
real(rk), allocatable, dimension(:) :: coefficients
real(rk), dimension(glob%nvar-glob%nzero) :: grad_nc_ref
real(rk), dimension(glob%nvar-glob%nzero,glob%nvar-glob%nzero) :: hessian_nc
logical  :: analytical_cub_quart, ex_custom_masses, standalone
real(rk) :: dq_4hess

analytical_cub_quart=.false.
#ifdef ANALYTICAL_CUB_QUART
analytical_cub_quart=.true.
#endif

! Default initialization of control parameters in stand-alone mode
! This is overriden when called within ChemShell (see further below)
vpt2_resonance_treatment=1
vpt2_resonance_criterion=1
vpt2_res_tol_deltae=100._rk
vpt2_res_tol_deltae_hard=200._rk
vpt2_res_tol_martin=1._rk
vpt2_res_tol_isaacson=0.4_rk
vpt2_hdcpt2_alpha=1.0_rk
vpt2_hdcpt2_beta =5.0e5_rk
vpt2_deriv_deltaq=0.5_rk
vpt2_asym_tol=1.e-5_rk
vpt2_Tmin=200._rk
vpt2_Tmax=600._rk
vpt2_nT=5
vpt2_do_part_func=.false.
vpt2_grad_only=.false.
vpt2_force_doscf=.true.
vpt2_dq_factor_4hess=10._rk

write(stdout,*) 'Number of processors:        ', glob%nprocs
write(stdout,*) 'Number of taskfarms:         ', glob%ntasks
write(stdout,*) 'Number of procs/taskfarm:    ', glob%nprocs_per_task
write(stdout,*) 'My rank(globally):           ', glob%iam
write(stdout,*) 'My rank(within task):        ', glob%iam_in_task
write(stdout,*) 'My task index:               ', glob%mytask

!!call dlf_mpi_get_size(isize)
!!call dlf_mpi_get_rank(irank)
!!write(stdout,*) 'Number of processors: (directly from MPI_COMM_SIZE call)      ', isize
!!write(stdout,*) 'Rank of current processor: (directly from MPI_COMM_RANK call) ', irank
!!glob%iam=irank
!!glob%nprocs=isize
!!call dlf_mpi_reset_global_comm()

call dlf_constants_get('AMU',amu2au)
nvareff=glob%nvar-glob%nzero

write(stdout,*) '----------------------------------------'
write(stdout,*) 'Number of atoms:                        ', glob%nat
write(stdout,*) 'Number of degrees of freedom: (int+ext) ', glob%nvar
write(stdout,*) 'Number of external degrees of freedom:  ', glob%nzero
write(stdout,*) 'Number of internal degrees of freedom:  ', nvareff
write(stdout,*) '----------------------------------------'

displacement_map(:)=0
displacement_map_grad_dual(:,:,:,:)=0
displacement_map_grad4hess(:,:,:,:)=0
hessians_cart(:,:,:)=0._rk
grad_cart_4hess(:,:)=0._rk
grad_cart_dual(:,:)=0._rk
nhessdone=0
hessdone(:)=.false.
new_punch=.true.
nhess=2*nvareff+1
do i=-nvareff,+nvareff,1
  displacement_map(i)=i+nvareff+1
enddo
ngraddone=0
ngraddone4hess=0
graddone_dual(:)=.false.
graddone4hess(:)=.false.
ngradhalf=2*nvareff*nvareff
ngrad=2*ngradhalf
ngrad4hess=2*nvareff
i=0
do j=1,nvareff
  i=i+1
  displacement_map_grad_dual(j,j,1,0)=i
  displacement_map_grad4hess(j,j,1,0)=i
  joblist_grad_dual(i,1)=j
  joblist_grad_dual(i,2)=j
  joblist_grad_dual(i,3)=1
  joblist_grad_dual(i,4)=0
  joblist_grad4hess(i,1)=j
  joblist_grad4hess(i,2)=j
  joblist_grad4hess(i,3)=1
  joblist_grad4hess(i,4)=0
  i=i+1
  displacement_map_grad_dual(j,j,-1,0)=i
  displacement_map_grad4hess(j,j,-1,0)=i
  joblist_grad_dual(i,1)=j
  joblist_grad_dual(i,2)=j
  joblist_grad_dual(i,3)=-1
  joblist_grad_dual(i,4)=0
  joblist_grad4hess(i,1)=j
  joblist_grad4hess(i,2)=j
  joblist_grad4hess(i,3)=-1
  joblist_grad4hess(i,4)=0
enddo
do j=1,nvareff
  do k=1,j-1
    i=i+1
    displacement_map_grad_dual(j,k,1,1)=i
    joblist_grad_dual(i,1)=j
    joblist_grad_dual(i,2)=k
    joblist_grad_dual(i,3)=1
    joblist_grad_dual(i,4)=1
    i=i+1
    displacement_map_grad_dual(j,k,1,-1)=i
    joblist_grad_dual(i,1)=j
    joblist_grad_dual(i,2)=k
    joblist_grad_dual(i,3)=1
    joblist_grad_dual(i,4)=-1
    i=i+1
    displacement_map_grad_dual(j,k,-1,1)=i
    joblist_grad_dual(i,1)=j
    joblist_grad_dual(i,2)=k
    joblist_grad_dual(i,3)=-1
    joblist_grad_dual(i,4)=1
    i=i+1
    displacement_map_grad_dual(j,k,-1,-1)=i
    joblist_grad_dual(i,1)=j
    joblist_grad_dual(i,2)=k
    joblist_grad_dual(i,3)=-1
    joblist_grad_dual(i,4)=-1
  enddo
enddo
joblist_grad_dual(ngradhalf+1:ngrad,:)  =joblist_grad_dual(1:ngradhalf,:)
joblist_grad_dual(ngradhalf+1:ngrad,3:4)=2*joblist_grad_dual(ngradhalf+1:ngrad,3:4)
do j=ngradhalf+1,ngrad
  io=joblist_grad_dual(j,1)
  ip=joblist_grad_dual(j,2)
  so=joblist_grad_dual(j,3)
  sp=joblist_grad_dual(j,4)
  displacement_map_grad_dual(io,ip,so,sp)=j
enddo

if (glob%iam == 0) then
  call get_command(cmdline)
  write(stdout,'(2A)') 'Calling command: ', trim(adjustl(cmdline))
  if (index(cmdline,'find.x') /= 0) then
    standalone=.true.
    ! When called from find.x instead of chemsh.x, get atom symbols and
    ! masses from coordinate input file specified in dlf.in instead of glob%mass !
    open(9911,file='dlf.in')
    read(9911,'(A)') fn
    close(9911)
    open(9912,file=trim(adjustl(fn)))
    read(9912,*)
    read(9912,*)
    do i=1,glob%nat
      read(9912,*) atsym(i), (rdum, j=1,3)
    enddo
    close(9912)
    mass_loc(:)=symb2mass(atsym(:),single_iso_in=.true.)
  else
    standalone=.false.
    ! Get masses (and atom symbols) from global module: 
    mass_loc(1:glob%nat)=glob%mass(1:glob%nat)
    atsym(1:glob%nat)=znuc2symb(glob%znuc(1:glob%nat))
    inquire(file='custom_masses.in',exist=ex_custom_masses)
    if (ex_custom_masses) then
      write(stdout,'(A)') '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
      write(stdout,'(A)') 'Reading atom masses from custom_masses.in... '
      write(stdout,'(A)') '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
      open(9913,file='custom_masses.in', action='read')
      do i=1,glob%nat
        read(9913,*) mass_loc(i)
        write(stdout,'(I6,1X,F15.5)') i, mass_loc(i)
      enddo
      close(9913)
    endif
  endif
endif

call dlf_gl_bcast(mass_loc,glob%nat,0)
do i=1,glob%nat
  call dlf_gl_bcast(atsym(i),0)
enddo

#ifdef LJ_INTERFACE
call dlf_vpt2_lj_pes_new_init(glob%nat)
#endif

if (standalone.and.1==2) then ! that part is commented out. vpt2.in is not necessary any more, the data are available via the interface
  if (glob%iam == 0) then
    ! This separate read-in of VPT2-specific configuration parameters
    ! from vpt2.in is only done in standalone mode
    open(9910,file='vpt2.in')
    read(9910,'(A)') resonance_mode
    read(9910,'(A)') fermi_crit
    read(9910,*) vpt2_deriv_deltaq
    read(9910,*) vpt2_asym_tol
    read(9910,*) vpt2_grad_only
    read(9910,*) vpt2_force_doscf
    close(9910)
  endif
  
  call dlf_gl_bcast(resonance_mode,0)
  call dlf_gl_bcast(fermi_crit,0)
  call dlf_gl_bcast(vpt2_deriv_deltaq,0)
  call dlf_gl_bcast(vpt2_asym_tol,0)
  call dlf_gl_bcast(vpt2_grad_only,0)
  call dlf_gl_bcast(vpt2_force_doscf,0)
  
else
  ! Grab local control variables from global module
  vpt2_resonance_treatment = glob%vpt2_resonance_treatment
  vpt2_resonance_criterion = glob%vpt2_resonance_criterion
  vpt2_res_tol_deltae      = glob%vpt2_res_tol_deltae
  vpt2_res_tol_deltae_hard = glob%vpt2_res_tol_deltae_hard
  vpt2_res_tol_martin      = glob%vpt2_res_tol_martin
  vpt2_res_tol_isaacson    = glob%vpt2_res_tol_isaacson
  vpt2_hdcpt2_alpha        = glob%vpt2_hdcpt2_alpha
  vpt2_hdcpt2_beta         = glob%vpt2_hdcpt2_beta
  vpt2_deriv_deltaq        = glob%vpt2_deriv_deltaq
  vpt2_asym_tol            = glob%vpt2_asym_tol
  vpt2_Tmin                = glob%vpt2_Tmin
  vpt2_Tmax                = glob%vpt2_Tmax
  vpt2_nT                  = glob%vpt2_nT
  vpt2_do_part_func        = glob%vpt2_do_part_func
  vpt2_grad_only           = glob%vpt2_grad_only
  vpt2_force_doscf         = glob%vpt2_force_doscf
  vpt2_dq_factor_4hess     = glob%vpt2_dq_factor_4hess
  
  select case(vpt2_resonance_treatment)
    case(0)
      resonance_mode='depert'
    case(1)
      resonance_mode='depert'
    case(2)
      resonance_mode='dcpt2'
    case(3)
      resonance_mode='hdcpt2'
    case(4)
      resonance_mode='none'
    case default
      call error_print('dlf_vpt2: Invalid choice of vpt2_resonance_treatment.')
  end select
  
  select case(vpt2_resonance_criterion)
    case(0)
      fermi_crit='martin'
    case(1)
      fermi_crit='martin'
    case(2)
      fermi_crit='deltaE_absolute'
    case(3)
      fermi_crit='isaacson'
    case(4)
      fermi_crit='all'
    case(5)
      fermi_crit='none'
    case(6)
      fermi_crit='manual_input'
    case default
      call error_print('dlf_vpt2: Invalid choice of vpt2_resonance_criterion.')
  end select
endif

redo_energy_option=vpt2_force_doscf
if (vpt2_dq_factor_4hess == 0._rk) then
  dq_4hess=vpt2_deriv_deltaq
else
  dq_4hess=vpt2_deriv_deltaq/vpt2_dq_factor_4hess
endif

call vpt2_init_fermi_res_params(vpt2_res_tol_deltae,vpt2_res_tol_deltae_hard, &
       & vpt2_res_tol_martin,vpt2_res_tol_isaacson,vpt2_hdcpt2_alpha,vpt2_hdcpt2_beta)

if (glob%iam == 0) then
  write(stdout,*) ' '
  write(stdout,*) 'Number of atoms:                ', glob%nat
  write(stdout,*) 'Number of variables:            ', glob%nvar
  write(stdout,*) 'Number of zero modes:           ', glob%nzero
  write(stdout,*) 'Number of internal coordinates: ', nvareff
  write(stdout,*) ' '
  write(stdout,*) ' >>>>> Summary of VPT2 Input Parameters <<<<< '
  write(stdout,*) ' '
  write(stdout,*) 'vpt2_resonance_treatment:  ', vpt2_resonance_treatment
  write(stdout,*) '  int code corresponds to: ', trim(adjustl(resonance_mode))
  write(stdout,*) 'vpt2_resonance_criterion:  ', vpt2_resonance_criterion
  write(stdout,*) '  int code corresponds to: ', trim(adjustl(fermi_crit))
  write(stdout,*) 'vpt2_res_tol_deltae:       ', vpt2_res_tol_deltae
  write(stdout,*) 'vpt2_res_tol_deltae_hard:  ', vpt2_res_tol_deltae_hard
  write(stdout,*) 'vpt2_res_tol_martin:       ', vpt2_res_tol_martin
  write(stdout,*) 'vpt2_res_tol_isaacson:     ', vpt2_res_tol_isaacson
  write(stdout,*) 'vpt2_hdcpt2_alpha:         ', vpt2_hdcpt2_alpha
  write(stdout,*) 'vpt2_hdcpt2_beta:          ', vpt2_hdcpt2_beta
  write(stdout,*) 'vpt2_deriv_deltaq:         ', vpt2_deriv_deltaq
  write(stdout,*) 'vpt2_asym_tol:             ', vpt2_asym_tol
  write(stdout,*) 'vpt2_Tmin:                 ', vpt2_Tmin
  write(stdout,*) 'vpt2_Tmax:                 ', vpt2_Tmax
  write(stdout,*) 'vpt2_nT:                   ', vpt2_nT
  write(stdout,*) 'vpt2_do_part_func:         ', vpt2_do_part_func
  write(stdout,*) 'vpt2_grad_only:            ', vpt2_grad_only
  write(stdout,*) 'vpt2_force_doscf:          ', vpt2_force_doscf
  write(stdout,*) 'vpt2_dq_factor_4hess:      ', vpt2_dq_factor_4hess
  write(stdout,*) ' '
  write(stdout,*) 'All values are printed in cm-1 unless noted otherwise! '
  write(stdout,*) ' '
  call matrix_output(transpose(glob%xcoords(:,:)),stdout,'F20.12','Cartesian coordinates (a.u.)')
  call vector_output(mass_loc(:),stdout,'F20.12','atomic masses (amu or Dalton)')
  write(stdout,*) 'Element list: '
  do i=1,glob%nat
    write(stdout,'(I0,1X,A)') i, atsym(i)
  enddo
  
  coords=pack(glob%xcoords,.true.)
  mv(:)=mass_loc(:)*amu2au
  do i=1,glob%nat
    mv3(3*(i-1)+1:3*(i-1)+3)=mv(i)
  enddo
  invsqrtmass(:,:)=0._rk
  do i=1,glob%nvar
    invsqrtmass(i,i)=1._rk/sqrt(mv3(i))
  enddo
  
  ! Coordinate shifting and reorientation
  
   do i=1,glob%nat
    xx(i)=coords(3*(i-1)+1)
    yy(i)=coords(3*(i-1)+2)
    zz(i)=coords(3*(i-1)+3)
  enddo
  
  call shift_com(glob%nat,xx,yy,zz,mv,xnew,ynew,znew)
  xx=xnew
  yy=ynew
  zz=znew
  moments_of_inertia=moi(glob%nat,xx,yy,zz,mv)
  call matrix_output(moments_of_inertia,stdout,'F20.12','Inertia tensor, before rotation')
  call rotate_to_princ_axes_ordered(glob%nat,xx,yy,zz,mv,xnew,ynew,znew,Arot,Brot,Crot)
  xx=xnew
  yy=ynew
  zz=znew
  moments_of_inertia=moi(glob%nat,xx,yy,zz,mv)
  call matrix_output(moments_of_inertia,stdout,'F20.12','Inertia tensor, after rotation')
  write(stdout,*) ''
  write(stdout,*) 'Rotational constants: (note: -666 means infinity, i.e. linear molecule)'
  write(stdout,*) 'A (cm^-1):  ', Arot
  write(stdout,*) 'B (cm^-1):  ', Brot
  write(stdout,*) 'C (cm^-1):  ', Crot
  write(stdout,*) ''
  ! JK: this translation/rotation leads to problems with some NN-PES (comment
  ! it out in that case)
  do i=1,glob%nat
    coords(3*(i-1)+1)=xx(i)
    coords(3*(i-1)+2)=yy(i)
    coords(3*(i-1)+3)=zz(i)
  enddo
  
  if (glob%nat>2 .and. (abs((Arot-Brot)/(Arot+Brot))<=vpt2_asym_tol &
                 & .or. abs((Brot-Crot)/(Brot+Crot))<=vpt2_asym_tol)  ) then
    call error_print('dlf_vpt2: Symmetric, linear(polyatomic) or spherical top detected! (not implemented yet)')
  endif
  
  if (abs(Arot+666._rk) < 1.e-2_rk) Arot=0._rk
  
endif

! Collect all punch files in MPI-rank 0 working directory, if applicable
call gather_punch_files_in_rank0_dir(glob%nvar)

if (glob%iam == 0) then
  ! Read punch files if applicable
  call read_punch_files(glob%nat,glob%nvar,nvareff,nhess,ngrad,ngrad4hess, &
          & atsym,coords,displacement_map,displacement_map_grad_dual, &
          & displacement_map_grad4hess,nhessdone,hessdone,ngraddone,graddone_dual, &
          & ngraddone4hess,graddone4hess,normal_mode_vectors,freqs,eigenvalues_au, &
          & hessians_cart,grad_cart_dual,grad_cart_4hess, fake_nm, vpt2_deriv_deltaq, &
          & dq_4hess,timestamp)
  !!nhessdone=0
  !!ngraddone=0
  !!ngraddone4hess=0
  if (nhessdone>0 .or. ngraddone>0 .or. ngraddone4hess>0) new_punch=.false.

  if (new_punch) then
    call write_master_punch_file_header(glob%nat,glob%nvar,nvareff,atsym,coords,timestamp)
  endif

  ! Get Hessian for normal coordinate analysis
  
  if (hessdone(displacement_map(0))) then
    nimag=count(freqs(:)<0._rk)
    call print_frequencies(nvareff,freqs)
    normal_mode_vectors_nonmw=dlf_matmul_simp(invsqrtmass,normal_mode_vectors)
  else
    write(stdout,'(A)') 'Getting Hessian for normal mode analysis...'
    if (vpt2_grad_only) then
      call get_hessian_finite_difference(dlf_get_gradient_with_updated_scf,glob%nat,glob%nvar, &
               & nvareff,ngrad4hess, &
               & graddone4hess,grad_cart_4hess,joblist_grad4hess,displacement_map_grad4hess, &
               & fake_nm,coords,mv,mv3,hessian,dq_4hess)
    else
#ifdef VPT2_HESS_DEBUG_OUT
      call dlf_get_hessian_with_updated_scf(glob%nvar,coords,hessian,istat,'+000')
#else
      call dlf_get_hessian_with_updated_scf(glob%nvar,coords,hessian,istat)
#endif
    endif
    write(stdout,'(A)') '...Hessian done.'
    hessdone(displacement_map(0))=.true.
    nhessdone=nhessdone+1
    call matrix_output(hessian,stdout,'F20.12','Hessian (a.u.)')
    ! Frequency analysis
    call get_frequencies_mw_ts_or_min(glob%nat,glob%nzero,coords,atsym,mv,hessian,freqs, &
              & eigenvalues_au,normal_mode_vectors,nimag,proj=project_hessian)
    normal_mode_vectors_nonmw=dlf_matmul_simp(invsqrtmass,normal_mode_vectors)
    hessians_cart(:,:,displacement_map(0)) = hessian
    if (.not. new_punch) then
      open(5446, file='dlf_vpt2_restart_proc_000000.dat', position='append')
      write(5446,'(A)') '$HESSIAN_CART'
      write(5446,'(I0,1X,I0,1X,ES15.6)') 0, 0, 0._rk
      call matrix_output(hessians_cart(:,:,displacement_map(0)),5446,'ES24.16','__BLANK__')
      close(5446)
    endif
  endif

  if (nimag==0) then
    saddle_point=.false.
    write(stdout,'(A)') ''
    write(stdout,'(A)') '##################################################'
    write(stdout,'(A)') ' No imaginary frequency -> VPT2 in minimum mode   '
    write(stdout,'(A)') '##################################################'
    write(stdout,'(A)') ''
    nreal=glob%nvar-glob%nzero
  elseif (nimag==1) then
    saddle_point=.true.
    write(stdout,'(A)') ''
    write(stdout,'(A)') '######################################################'
    write(stdout,'(A)') ' ONE imaginary frequency -> VPT2 in saddle point mode '
    write(stdout,'(A)') '######################################################'
    write(stdout,'(A)') ''
    nreal=glob%nvar-glob%nzero-1
  else
    write(stdout,*) "Number of imaginary frequencies: ",nimag
    call error_print('Error in anharmonic frequency module: No. of imaginary frequencies must be 0 or 1!')
  endif
endif

call dlf_gl_bcast(coords,glob%nvar,0)
call dlf_gl_bcast(mv,glob%nat,0)
call dlf_gl_bcast(mv3,3*glob%nat,0)
call dlf_gl_bcast(normal_mode_vectors,glob%nvar,glob%nvar-glob%nzero,0)
call dlf_gl_bcast(normal_mode_vectors_nonmw,glob%nvar,glob%nvar-glob%nzero,0)
call dlf_gl_bcast(nhessdone,0)
call dlf_gl_bcast(hessdone,nhess,0)
call dlf_gl_bcast(hessians_cart,glob%nvar,glob%nvar,nhess,0)
call dlf_gl_bcast(new_punch,0)
call dlf_gl_bcast(ngraddone,0)
call dlf_gl_bcast(graddone_dual,ngrad,0)
call dlf_gl_bcast(grad_cart_dual,glob%nvar,ngrad,0)
call dlf_gl_bcast(grad_cart_4hess,glob%nvar,ngrad4hess,0)

if (glob%iam==0) then 
  call write_hessian_to_master_punch_file(glob%nvar,nvareff,normal_mode_vectors, &
                               & freqs,eigenvalues_au,hessians_cart(:,:,displacement_map(0)),timestamp)
endif

call dlf_gl_bcast(timestamp,0)

if (glob%iam/=0) then
  call write_slave_punch_file_header(timestamp)
endif

if (vpt2_grad_only) then 
  call calculate_displaced_gradients_mpi(dlf_get_gradient_with_updated_scf,glob%nvar,nvareff,ngrad, &
               & coords,normal_mode_vectors,mv3, &
               & joblist_grad_dual,graddone_dual,grad_cart_dual,vpt2_deriv_deltaq)
else
  call calculate_displaced_hessians_mpi(dlf_get_hessian_with_updated_scf,glob%nvar,nvareff,nhess,coords, &
               & normal_mode_vectors,mv3,displacement_map,hessdone,hessians_cart,[vpt2_deriv_deltaq])
endif

call gather_punch_files_in_rank0_dir(glob%nvar)

if (glob%iam == 0) then
  call punch_file_collapse_range(0,glob%nprocs-1,500,nvareff,0)
  if (vpt2_grad_only) then
    call derivatives_from_gradient_list_dual_step(glob%nvar,nvareff,ngrad,mv3, &
                  & grad_cart_dual,normal_mode_vectors,normal_mode_vectors_nonmw,eigenvalues_au, &
                  & freqs, hessian, displacement_map_grad_dual,cubic_nm,quartic_nm_semidiag, &
                  & vpt2_deriv_deltaq,.true.,hessian_nc,grad_nc_ref)
  elseif (analytical_cub_quart) then
#ifdef ANALYTICAL_CUB_QUART
    call get_cubic_quartic_nm_via_analytical_routine(dlf_get_cubic_quartic_lj,glob%nvar,nvareff, &
                  &  coords,mv3,normal_mode_vectors,cubic_nm,quartic_nm_semidiag)
#else
    continue
#endif
  else
    call derivatives_from_hessian_list(glob%nvar,nvareff,nhess,hessians_cart,normal_mode_vectors_nonmw, &
                         & displacement_map,cubic_nm,quartic_nm_semidiag, [ vpt2_deriv_deltaq ])
  endif
endif


if (glob%iam==0) then
  ! Symmetric fill
  call symmetric_fill_cubic(nvareff,cubic_nm,cubic_nm)
  call symmetric_fill_quartic_semidiag(nvareff,quartic_nm_semidiag,quartic_nm_semidiag)
  
  ! Convert force constants to reduced form
  call convert_cubic_to_reduced_cubic(nvareff,cubic_nm,eigenvalues_au,cubic_reduced)
  call convert_quartic_to_reduced_quartic(nvareff,quartic_nm_semidiag,eigenvalues_au,quartic_reduced)
  
#ifdef VPT2_PRINT_FC
  do i=1,nvareff
    write(printch,'(A,I0)') "i = ", i
    call matrix_output(cubic_reduced(:,:,i), stdout, 'F20.12', &
                               & 'reduced cubic force constants (cm^-1) '//trim(adjustl(printch)))
  enddo
  
  do i=1,nvareff
    write(printch,'(A,I0)') "i = ", i
    call matrix_output(quartic_reduced(i,:,:), stdout, 'F20.12', & 
                    & 'reduced semi-diagonal quartic force constants (cm^-1) '//trim(adjustl(printch)))
  enddo
#endif
  
  xFF=0._rk
  xiF=0._rk
  
  call vpt2_driver(glob%nvar,nvareff,abs(freqs),normal_mode_vectors, & 
                 & cubic_reduced,quartic_reduced,Arot,Brot,Crot,E0,X,zpve_harm,zpve_anh_direct, &
                 & trim(adjustl(resonance_mode)),trim(adjustl(fermi_crit)),saddle_point,xFF,xiF)
  
  write(stdout,*) 'E0 = ', E0
  call matrix_output(X,stdout,'F20.12','X matrix')
  call matrix_output(X,stdout,'D14.6','X matrix')
  
  call vpt2_get_zpve(nvareff,freqs,E0,X,zpve_anh_explicit_sum,zpve_harm,saddle_point)
  
  write(stdout,*) ''
  write(stdout,*) '******************  ZPVE summary  ************************'
  write(stdout,*) ''
  write(stdout,*) 'ZPVE, harmonic (cm^-1) :   ', zpve_harm
  write(stdout,*) 'ZPVE, anharmonic (cm^-1), direct (resonance-free)  : ', zpve_anh_direct
  write(stdout,*) 'ZPVE, anharmonic (cm^-1), explicit sum (via E0, X) : ', zpve_anh_explicit_sum
  write(stdout,*) '------------------------------------------------'
  write(stdout,*) 'ZPVE anh. corr. (cm^-1), direct (resonance-free)  : ', zpve_anh_direct-zpve_harm
  write(stdout,*) 'ZPVE anh. corr. (cm^-1), explicit sum (via E0, X) : ', zpve_anh_explicit_sum-zpve_harm
  write(stdout,*) ''
  write(stdout,*) '**********************************************************'
  
  call vpt2_fundamentals(nvareff,freqs,X,vpt2_funda,saddle_point)
  
  call matrix_output(reshape([freqs(1:nreal),vpt2_funda(1:nreal)],[nreal,2]),stdout,'F10.2', & 
                     & 'Harmonic (left) and VPT2 (right) fundamentals')
  if (saddle_point) then
    write(stdout,'(A)') ''
    write(stdout,'(A,F10.2,A)') 'Harmonic imaginary frequency:  ',abs(freqs(nreal+1)),'i'
    write(stdout,'(A)') '(An imaginary fundamental would be physically meaningless)'
    write(stdout,'(A)') ''
  endif
  call generate_bdens_input(nreal,freqs(1:nreal),X(1:nreal,1:nreal),Arot,Brot,Crot,vpt2_Tmin,vpt2_Tmax)
  if (saddle_point) then
    call generate_sctst_input(nreal,freqs(1:nreal),X(1:nreal,1:nreal),Arot,Brot,Crot, &
                  & vpt2_Tmin,vpt2_Tmax,abs(freqs(nreal+1)),XFF,XiF(1:nreal))
  endif
endif

if (vpt2_do_part_func) then
  if (glob%iam==0) then
    call allocate(Temps, vpt2_nT)
    call allocate(qanh_via_rho, vpt2_nT)
    if (vpt2_nT > 1) then
      delT=(vpt2_Tmax-vpt2_Tmin)/real(vpt2_nT-1)
    else
      delT=0._rk
    endif
    
    call dlf_constants_get("KBOLTZ_CM_INV",kB_cm)
    
    do i=1,vpt2_nT
      Temps(i)=vpt2_Tmin+(i-1)*delT
    enddo
    write(stdout,*) ''
    write(stdout,*) 'Vibrational partition functions: '
    write(stdout,*) ''
    write(stdout,*) '**************************************************************'
    write(stdout,*) '************************ ATTENTION! **************************'
    write(stdout,*) ''
    write(stdout,*) 'The internal brute-force DOS counting algorithm for the'
    write(stdout,*) 'partition functions has been activated. This is a very'
    write(stdout,*) 'inefficient way to calculate q(T) and is only there for'
    write(stdout,*) 'debugging, testing, and comparison purposes.'
    write(stdout,*) 'Instead, use the files bdens.dat and/or sctst.dat'
    write(stdout,*) 'which are generated by this program, in combination with the'
    write(stdout,*) 'programs bdens and sctst from the MultiWell code of Barker '
    write(stdout,*) 'et al., see: '
    write(stdout,*) ''
    write(stdout,*) ' http://clasp-research.engin.umich.edu/multiwell '
    write(stdout,*) ''
    write(stdout,*) '**************************************************************'
    write(stdout,*) '**************************************************************'
    write(stdout,*) ''
    write(stdout,*) '**** Note ****'
    write(stdout,*) 'SPT means simple perturbation theory, see'
    write(stdout,*) 'D.G. Truhlar, A.D. Isaacson, J. Chem. Phys. 94, 357 (1991).'
    write(stdout,*) 'It is simply the analytical harmonic oscillator partition'
    write(stdout,*) 'function with anharmonic fundamentals instead of the'
    write(stdout,*) 'harmonic frequencies. qANH denotes proper anharmonic'
    write(stdout,*) 'partition functions.'
    write(stdout,*) '*******************'
    write(stdout,*) ''
    write(stdout,*) 'Anharmonic vibrational partition functions (via density of states): '
    write(stdout,*) 
    write(stdout,'(A10,3A20)') 'T/K', 'qHO(T)', 'qSPT(T)', 'qANH(T)'
    write(stdout,*) '-----------------------------------------------------'
  endif
  call dlf_gl_bcast(nvareff,0)
  call dlf_gl_bcast(saddle_point,0)
  call dlf_gl_bcast(freqs,glob%nvar-glob%nzero,0)
  call dlf_gl_bcast(Temps,vpt2_nT,0)
  call dlf_gl_bcast(X,glob%nvar-glob%nzero,glob%nvar-glob%nzero,0)
  qanh_via_rho=qvib_anharmonic_via_rho(nvareff,freqs,X,vpt2_nT,Temps,hybrid=.true.,saddle_point=saddle_point)
  if (glob%iam == 0) then
    do i=1,vpt2_nT
      Temp=Temps(i)
      qhar=qvib_harmonic(nvareff,freqs,Temp,saddle_point)
      qanh_simp=qvib_harmonic(nvareff,vpt2_funda,Temp,saddle_point)
      write(stdout,'(F10.1,3F20.12)') Temps, qhar, qanh_simp, qanh_via_rho(i)
    enddo
    !! This uses the old, canonical code:
    !write(stdout,*) ''
    !write(stdout,*) 'Vibrational partition functions: '
    !write(stdout,*) ''
    !write(stdout,*) '**** Note ****'
    !write(stdout,*) 'SPT means simple perturbation theory, see'
    !write(stdout,*) 'D.G. Truhlar, A.D. Isaacson, J. Chem. Phys. 94, 357 (1991).'
    !write(stdout,*) 'It is simply the analytical harmonic oscillator partition'
    !write(stdout,*) 'function with anharmonic fundamentals instead of the'
    !write(stdout,*) 'harmonic frequencies. qANH denotes proper anharmonic'
    !write(stdout,*) 'partition functions.'
    !write(stdout,*) '*******************'
    !write(stdout,*) ''
    !write(stdout,'(A10,3A20)') 'T/K', 'qHO(T)', 'qSPT(T)', 'qANH(T)'
    !write(stdout,*) '----------------------------------------------------------------------'
    !do i=1,vpt2_nT
    !  Temp=Temps(i)
    !  qhar=qvib_harmonic(nvareff,freqs,Temp,saddle_point)
    !  qanh_simp=qvib_harmonic(nvareff,vpt2_funda,Temp,saddle_point)
    !  qanh_direct=qvib_anharmonic(nvareff,freqs,X,Temp,hybrid=.true.,saddle_point=saddle_point)
    !  write(stdout,'(F10.1,3F20.12)') Temp, qhar, qanh_simp, qanh_direct
    !  !write(stdout,'(F10.1,2F20.12)') Temp, qhar, qanh_simp
    !enddo
    call deallocate(qanh_via_rho)
    call deallocate(Temps)
  endif
end if

!return

end subroutine dlf_vpt2

!!subroutine dlf_get_hessian_compare(nvar,coords,hessian,status)
!!  use get_egh, only: dlf_get_hessian_dd
!!  use dlf_vpt2_utility, only: matrix_output
!!  implicit none
!!  interface dlf_get_energy
!!    subroutine dlf_get_energy(nvar,coords,ener,iimage,iter,ierr)
!!      use dlf_parameter_module, only: rk
!!      implicit none
!!      integer   ,intent(in)    :: nvar
!!      real(rk)  ,intent(in)    :: coords(nvar)
!!      real(rk)  ,intent(out)   :: ener
!!      integer,intent(in) :: iimage
!!      integer,intent(in) :: iter
!!      integer,intent(out):: ierr
!!    end subroutine dlf_get_energy
!!  end interface dlf_get_energy
!!  interface
!!    subroutine dlf_get_hessian(nvar,coords,hessian,status)
!!      use dlf_parameter_module
!!      implicit none
!!      integer   ,intent(in)    :: nvar
!!      real(rk)  ,intent(in)    :: coords(nvar)
!!      real(rk)  ,intent(out)   :: hessian(nvar,nvar)
!!      integer   ,intent(out)   :: status
!!    end subroutine dlf_get_hessian
!!  end interface
!!  integer   ,intent(in)    :: nvar
!!  real(rk)  ,intent(in)    :: coords(nvar)
!!  real(rk)  ,intent(out)   :: hessian(nvar,nvar)
!!  integer   ,intent(out)   :: status
!!  
!!  real(rk)   :: hessian_dd(nvar,nvar)
!!  real(rk)   :: hessian_chemsh(nvar,nvar)
!!  real(rk)   :: hessdiff(nvar,nvar)
!!  real(rk)   :: energy_dummy
!!  integer, parameter :: iimage=1,kiter=-1
!!  integer :: energy_status
!!  
!!  call execute_command_line('cp control_dd_tmpl control')
!!  call dlf_get_hessian_dd(nvar,coords,hessian_dd,status)
!!  call execute_command_line('sync && cp nprhessian nprhessian_dd')
!!  call execute_command_line('sync && cp coord coord_dd')
!!  call execute_command_line('sync && cp control control_dd')
!!  call execute_command_line('rm -f control')
!!  !call execute_command_line('cp control_cs control')
!!  call dlf_get_energy(nvar,coords,energy_dummy,iimage,kiter,energy_status)
!!  call dlf_get_hessian(nvar,coords,hessian_chemsh,status)
!!  !call execute_command_line('cp control control_cs')
!!  call execute_command_line("sync && grep '$nprhessian' -A 10000000 control | "// &
!!                & "grep -E '^\$' --max-count=2 -B 10000000 >nprhessian_cs")
!!  call execute_command_line('sync && cp coord coord_cs')
!!  call execute_command_line('sync && cp control control_cs')
!!  
!!  hessdiff(:,:)=-hessian_dd(:,:)+hessian_chemsh(:,:)
!!  hessian(:,:)=hessian_dd(:,:)
!!  hessian(:,:)=hessian_chemsh(:,:)
!!  
!!  call matrix_output(hessdiff,6,'F20.12','hessdiff')
!!  read(*,*)
!!  
!!  return
!!end subroutine dlf_get_hessian_compare

! ****************************************
! ****************************************
! ****************************************

! A wrapper for dlf_get_hessian, which does a 
! dlf_get_energy call before each Hessian. 
! This behavior can be switched off by setting the 
! redo_energy_option in the dlf_vpt2_utility module 
! to .false.. There is also a user parameter in the 
! Chemshell interface called vpt2_force_doscf which 
! lets the user control this behavior.

#ifdef VPT2_HESS_DEBUG_OUT
subroutine dlf_get_hessian_with_updated_scf(nvar,coords,hessian,status,calctag)
#else
subroutine dlf_get_hessian_with_updated_scf(nvar,coords,hessian,status)
#endif
  use dlf_vpt2_utility, only: error_print, matrix_output, vector_output, redo_energy_option
  use dlf_global, only: stdout
  implicit none
#ifndef DLF_STANDALONE_INTERFACE
  interface dlf_get_energy
    subroutine dlf_get_energy(nvar,coords,ener,iimage,iter,ierr)
      use dlf_parameter_module, only: rk
      implicit none
      integer   ,intent(in)    :: nvar
      real(rk)  ,intent(in)    :: coords(nvar)
      real(rk)  ,intent(out)   :: ener
      integer,intent(in) :: iimage
      integer,intent(in) :: iter
      integer,intent(out):: ierr
    end subroutine dlf_get_energy
  end interface dlf_get_energy
#endif
  interface
    subroutine dlf_get_hessian(nvar,coords,hessian,status)
      use dlf_parameter_module
      implicit none
      integer   ,intent(in)    :: nvar
      real(rk)  ,intent(in)    :: coords(nvar)
      real(rk)  ,intent(out)   :: hessian(nvar,nvar)
      integer   ,intent(out)   :: status
    end subroutine dlf_get_hessian
  end interface
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: hessian(nvar,nvar)
  integer   ,intent(out)   :: status
#ifdef VPT2_HESS_DEBUG_OUT
  character(len=*), intent(in),optional :: calctag
#endif
  
  real(rk)   :: energy_dummy
  integer, parameter :: iimage=1,kiter=-1
  integer :: energy_status
  character(len=1000) :: debug_fn
  
#ifdef VPT2_HESS_DEBUG_OUT
  if (present(calctag)) then
    open(9002, file='dlf_get_hessian_debug_' // trim(adjustl(calctag)) // '.dat' )
    write (9002, '(A)') '$COORDS'
    call vector_output(coords,9002,'ES28.20','__BLANK__')
  endif
#endif
  
#ifndef DLF_STANDALONE_INTERFACE
  if (redo_energy_option) then
    call dlf_get_energy(nvar,coords,energy_dummy,iimage,kiter,energy_status)
    write(stdout,'(A,1X,F20.8)') 'Electronic energy (a.u.): ', energy_dummy
    if (energy_status/=0) call &
     & error_print('dlf_get_hessian_with_updated_scf: Error in dlf_get_energy call!')
  endif
#endif
  !print*,"JK coords:",coords
  !write(*,'(3f12.7)') coords
  call dlf_get_hessian(nvar,coords,hessian,status)
  if (status/=0) call &
   & error_print('dlf_get_hessian_with_updated_scf: Error in dlf_get_hessian call!')
  
#ifdef VPT2_HESS_DEBUG_OUT
  if (present(calctag)) then
    write (9002, '(A)') '$ENERGY'
    write (9002,'(ES24.16)') energy_dummy
    write (9002, '(A)') '$HESSIAN_CART'
    call matrix_output(hessian,9002,'ES24.16','__BLANK__')
    close(9002)
  endif
#endif
  
  return
end subroutine dlf_get_hessian_with_updated_scf

! ****************************************
! ****************************************
! ****************************************

! A wrapper for dlf_get_gradient, which does a 
! dlf_get_energy call before each gradient. 
! This behavior can be switched off by setting the 
! redo_energy_option in the dlf_vpt2_utility module 
! to .false.. There is also a user parameter in the 
! Chemshell interface called vpt2_force_doscf which 
! lets the user control this behavior.

#ifdef VPT2_GRAD_DEBUG_OUT
subroutine dlf_get_gradient_with_updated_scf & 
      & (nvar,coords,energy,gradient,iimage,kiter,status,calctag)
#else
subroutine dlf_get_gradient_with_updated_scf & 
      & (nvar,coords,energy,gradient,iimage,kiter,status)
#endif
  use dlf_vpt2_utility, only: error_print, vector_output, redo_energy_option
  implicit none
#ifndef DLF_STANDALONE_INTERFACE
  interface dlf_get_energy
    subroutine dlf_get_energy(nvar,coords,ener,iimage,iter,ierr)
      use dlf_parameter_module, only: rk
      implicit none
      integer   ,intent(in)    :: nvar
      real(rk)  ,intent(in)    :: coords(nvar)
      real(rk)  ,intent(out)   :: ener
      integer,intent(in) :: iimage
      integer,intent(in) :: iter
      integer,intent(out):: ierr
    end subroutine dlf_get_energy
  end interface dlf_get_energy
#endif
  interface
    subroutine dlf_get_gradient(nvar,coords,energy,gradient,iimage,kiter,status)
      use dlf_parameter_module, only: rk
      implicit none
      integer   ,intent(in)    :: nvar
      real(rk)  ,intent(in)    :: coords(nvar)
      real(rk)  ,intent(out)   :: energy
      real(rk)  ,intent(out)   :: gradient(nvar)
      integer   ,intent(in)    :: iimage
      integer   ,intent(in)    :: kiter
      integer   ,intent(out)   :: status
    end subroutine dlf_get_gradient
  end interface
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: energy
  real(rk)  ,intent(out)   :: gradient(nvar)
  integer   ,intent(in)    :: iimage
  integer   ,intent(in)    :: kiter
  integer   ,intent(out)   :: status
#ifdef VPT2_GRAD_DEBUG_OUT
  character(len=*), intent(in),optional :: calctag
#endif
  real(rk)   :: energy_dummy
  integer :: energy_status
  character(len=1000) :: debug_fn
  
#ifdef VPT2_GRAD_DEBUG_OUT
  if (present(calctag)) then
    open(9002, file='dlf_get_gradient_debug_' // trim(adjustl(calctag)) // '.dat' )
    write (9002, '(A)') '$COORDS'
    call vector_output(coords,9002,'ES28.20','__BLANK__')
  endif
#endif
  
#ifndef DLF_STANDALONE_INTERFACE
  if (redo_energy_option) then
    call dlf_get_energy(nvar,coords,energy_dummy,iimage,kiter,energy_status)
    if (energy_status/=0) call &
     & error_print('dlf_get_gradient_with_updated_scf: Error in dlf_get_energy call!')
  endif
#endif
  call dlf_get_gradient(nvar,coords,energy,gradient,iimage,kiter,status)
  if (status/=0) call &
   & error_print('dlf_get_gradient_with_updated_scf: Error in dlf_get_gradient call!')
  
#ifdef VPT2_GRAD_DEBUG_OUT
  if (present(calctag)) then
    write (9002, '(A)') '$ENERGY'
    write (9002,'(ES24.16)') energy_dummy
    write (9002, '(A)') '$GRADIENT_CART'
    call vector_output(gradient,9002,'ES24.16','__BLANK__')
    close(9002)
  endif
#endif
  
  return
end subroutine dlf_get_gradient_with_updated_scf

! ****************************************
! ****************************************
! ****************************************

end module dlf_vpt2_main



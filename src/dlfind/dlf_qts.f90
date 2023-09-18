! **********************************************************************
! **                         Reaction Rate Module                     **
! **   Instanton theory and classical transition state theory with    **
! **      several approximations for tunnelling rate constants        **
! **           additional related routines in dlf_rateaux.f90         **
! **********************************************************************
!!****h* neb/qts
!!
!! NAME
!! qts
!!
!! FUNCTION
!! Instanton (quantum transition state) optimisation
!!
!! Search for a quantum transition state: move a closed Feynman path
!! to a saddle point of the action
!!
!! Instanton rate calculations: calculate the Hessian of the potential
!! energy at each point along the instanton path and then calculate
!! the rate from that.
!!
!! Rates by classical harmonic transition state theory and a few
!! simple quantum extensions are available in the routine dlf_htst_rate.
!!
!! This module introduces an additional file format for storing Hessian
!! information. It is written by routines like write_qts_coords,
!! write_qts_hessian and so on.
!!
!! This is conceptually a part of the NEB module. It also uses that module
!! throughout. However, for coding, it has been taken to a different file.
!!
!! DATA
!! $Date:  $
!! $Rev:  $
!! $Author:  $
!! $URL:  $
!! $Id:  $
!!
!! COPYRIGHT
!!
!!  Copyright 2010-2020 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Judith B. Rommel (rommel@theochem.uni-stuttgart.de)
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
!! SYNOPSIS
module dlf_qts
  use dlf_parameter_module, only: rk
  type qts_type
    real(rk),allocatable :: tau_qts(:)   ! (nimage*varperimage) direction 
    ! midpoint-endpoint 
    real(rk),allocatable :: theta(:)     ! (nimage*varperimage) rotational force (dimer)
    real(rk), allocatable  :: d_actionS_alt(:), d_actionS(:) ! (nimage*varperimage) Derivative of action 
    real(rk), allocatable  :: tau_prime(:) ! (nimage*varperimage)
    real(rk), allocatable  :: ihessian_image(:,:) ! (varperimage,varperimage)
    real(rk), allocatable  :: dtau(:) ! (nimage+1) delta tau 
    integer   :: status, image_status
    real(rk)  :: C_Tau

    !S_ins (Instanton action) = 0.5*S_0 + S_pot
    real(rk)  :: S_ins ! Potential part of S_ins 
    real(rk)  :: S_0 ! 2*Kinetic part of S_ins 
!************************ADDED BY SEAN SEP 2015*************************
    real(rk)  :: tcross
    complex(rk)  :: S_sigma !Correction to the action: Kryvohuz
    complex(rk)  :: S_dsigma
    complex(rk)  :: S_ddsigma
    complex(rk)  :: S_dsigma_star
!************************ADDED BY SEAN SEP 2015*************************
    real(rk)  :: S_pot ! Potential part of S_ins 
    real(rk)  :: etunnel
    real(rk)  :: ers ! Energy of a minimum next to one end of the instanton path

    real(rk)  :: phi_1 !rotational angle 1 
    real(rk)  :: rate !quantum transition rate
    real(rk)  :: b_1 !derivative of PES curvature along start tau
    integer   :: hessian_mode ! Switch how the Hessian should be calculated (JK)
    logical   :: first_hessian
    logical   :: needhessian 
    real(rk), allocatable :: coords_midp(:) !midpoint coordinates
    ! required for Hessian update
    logical   :: tupdate
    real(rk), allocatable :: igradient(:) ! derivative of the potential
    logical   :: try_analytic_hessian
    real(rk), allocatable :: dist(:) ! (nimage+1) distance to previous image,
    ! Hessian updates
    real(rk), allocatable :: vhessian(:,:,:) ! (varperimage,varperimage,nimage)
    real(rk), allocatable :: igradhess(:,:)  ! (varperimage,nimage) gradient at the position of the Hessian in vhessian
    real(rk), allocatable :: icoordhess(:,:) ! (varperimage,nimage) coordinates of the position of the Hessian in vhessian
    logical   :: tsplit
    real(rk) :: dEbdbeta ! d E_b / d beta_hbar
    real(rk) :: dEbdbeta_prod ! d E_b / d beta_hbar from product of eigenvalues
    real(rk) :: sigma ! Kryvohuz sigma from product of eigenvalues
    real(rk), allocatable :: stabpar_diffeq(:)
    real(rk), allocatable :: stabpar_trace(:)
  end type qts_type
  type(qts_type),save :: qts
  logical, parameter:: hess_dtau=.false. ! if true, the Potential-Hessian is
  logical :: dbg_rot=.false.
  integer :: taskfarm_mode=1 ! default, especially if QTS is not used ...
  logical, parameter :: time=.false. ! print timing information
  ! there is a global variable, glob%qtsflag. Meaning:
  ! 0: normal calculation
  ! 1: calculate tunnelling splittings (only valid for symmetric cases)
  !
  ! a derived-type variable for partition functions:
  type pf_type
    logical   :: tused
    logical   :: tfrozen ! do frozen atoms exist in that fragment?
    real(rk)  :: mass
    real(rk)  :: ene
    integer   :: nat
    real(rk)  :: moi(3) ! moments of inertia (diagonal)
    integer   :: nmoi ! number of used moments of inertia
    integer   :: nvib ! number of vibrational modes
    real(rk)  :: omega_imag ! absolute value of imaginary frequency (TS)
    real(rk)  :: omega2zero(6) ! hessian-eigenvalues which are supposed to be zero
    real(rk), allocatable :: omega(:) ! (nvib)
    real(rk)  :: coeff(2) ! coefficients of even and odd angular momentum quantum numbers (ortho/para)  
  end type pf_type
  real(rk) :: minfreq
end module dlf_qts
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/dlf_qts_init
!!
!! FUNCTION
!!
!! Initialise QTS (instanton) calculations:
!! * allocate arrays 
!! * set default values for the calculation
!!
!! SYNOPSIS
subroutine dlf_qts_init()
  !! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_neb, only: neb,unitp,xyzall,beta_hbar
  use dlf_global, only: glob,stdout,printl
  use dlf_hessian, only: fd_hess_running
  use dlf_constants, only : dlf_constants_get
  !use dlf_hessian
  use dlf_allocate, only: allocate,deallocate
  use dlf_qts
  implicit none
  integer   :: nimage,iimage,ivar
  real(rk)  :: kboltz_au,svar
  real(rk)  :: deltay(neb%nimage)
  real(rk)  :: eigvals(neb%varperimage) ! only needed as dummy for read_qts_reactant
  logical   :: tok

  ! no need to do anything if just the classical rate is to be calculated
  if(glob%iopt==13) return

!!$  ! tmp>>>>
!!$
!!$  real(rk) :: mat(10,10)
!!$  real(rk) :: eigval(10)
!!$  real(rk) :: eigvec(10,10)
!!$  integer  :: ival
!!$  mat=0.D0
!!$  do ival=1,10
!!$    mat(ival,ival)=2.D0
!!$    if(ival<10) then
!!$      mat(ival,ival+1)=-1.D0
!!$      mat(ival+1,ival)=-1.D0
!!$    end if
!!$  end do
!!$  mat(1,10)=-1.D0
!!$  mat(10,1)=-1.D0
!!$  call dlf_matrix_diagonalise(10,mat,eigval,eigvec)
!!$  do ival=1,10
!!$    write(*,'(11f10.5)') eigval(ival),eigvec(:,ival)
!!$  end do
!!$  call dlf_fail("here")
!!$   ! <<<

  ! glob%nivar is varperimage * nimage

  call allocate (qts%tau_prime,glob%nivar) 
  call allocate (qts%tau_qts,glob%nivar) 
  call allocate (qts%theta,glob%nivar)
  call allocate (qts%coords_midp,neb%varperimage*neb%nimage)
  call allocate (qts%d_actionS_alt,neb%varperimage*neb%nimage)
  call allocate (qts%d_actionS,neb%varperimage*neb%nimage)
  call allocate (qts%dtau,neb%nimage+1)
  call allocate (qts%dist,neb%nimage+1)

  ! this decision should be made in formstep, but we leave it here for now...
  if(glob%iopt<10 .or. glob%iopt==30) then
    qts%needhessian=.false.
  else
    qts%needhessian=.true.
    glob%maxupd=huge(1) ! fix to make sure the hessian is always updated IMPROVE
  end if

!!$  ! check if hessian is required (call to formstep). However, this
!!$  ! does not work, as formstep is not initialised by the time this call
!!$  ! is made ... . Therefore left out for the time being
!!$  call dlf_formstep_get_logical("NEEDHESSIAN",qts%needhessian)
!!$  print*,"JK test needhessian",qts%needhessian

  ! Predict tau
  ! commented out: this is now called from neb_init
  !  call qts_tau_from_path()

  call dlf_lbfgs_select("dimer rotation",.true.)
  call dlf_lbfgs_init(glob%nivar,min(glob%maxrot,glob%nivar)) 
  call dlf_lbfgs_deselect
  qts%status=1

  ! print header with citation information
  if(printl>=2) then
    write(stdout,'(a)') '%%%%%%%%%%%'
    write(stdout,'(a)') 'Please include these references which describe the optimisation'
    write(stdout,'(a)') 'algorithms and the integration grid in published work using the instanton rate algorithms:'
    write(stdout,'(a)') 'J.B. Rommel, T.P.M. Goumans, J. Kastner, J. Chem. Theory Comput. 7, 690 (2011)'
    write(stdout,'(a)') 'J.B. Rommel, J. Kastner, J. Chem. Phys. 134, 184107 (2011)'
    write(stdout,'(a)') 'along with the original references to instanton theory.'
    write(stdout,'(a)') '%%%%%%%%%%%'
  end if

  if( qts%needhessian) then

    if(glob%iopt==12) then
      !-------------------------------------------------------------------------
      ! read in neb%xcoords, S_0 and S_ins 
      !-------------------------------------------------------------------------
      nimage=neb%nimage
      call read_qts_coords(glob%nat,nimage,neb%varperimage,glob%temperature,&
          qts%S_0,qts%S_pot,qts%S_ins,neb%ene,neb%xcoords,qts%dtau,qts%etunnel,qts%dist)
      if(nimage/=neb%nimage) call dlf_fail("Number of images in the file qts_coords.txt&
          & is inconsistent")

!!$      ! check if qts_reactant.txt is present - stop if absent.
!!$      ! commented out: throws up if mass-re-weighting is to be done - not needed with inithessian=5
!!$      call read_qts_reactant(glob%nat,neb%varperimage,&
!!$          svar,ivar,eigvals,svar,"",tok)
!!$      if(.not.tok) then
!!$        call dlf_fail("qts_reactant.txt not available for rate calculation")
!!$      end if

      ! beta_hbar has to be defined here
      call dlf_constants_get("KBOLTZ_AU",kboltz_au)
      beta_hbar=1.D0/(glob%temperature*kboltz_au)  ! beta * hbar = hbar / (kB*T)
      if(qts%dtau(1)<0.D0) then
        if(printl>=2) write(stdout,*) "Warning, dtau not read from file, setting as constant"
        qts%dtau(:)=beta_hbar/dble(2*neb%nimage)
      end if

      ! set glob%icoords (only needed at the moment for estimating vec0)
      do iimage=1,neb%nimage
        call dlf_direct_xtoi(glob%nvar,neb%varperimage,neb%coreperimage,&
            neb%xcoords(:,iimage),glob%xgradient,&
            glob%icoords(neb%cstart(iimage):neb%cend(iimage)), &
            glob%igradient(neb%cstart(iimage):neb%cend(iimage)))
      end do

      ! recalculate S_0 - in case the masses have changed
      deltay(:)=0.D0
      qts%S_0 = 0.D0
      do iimage=1,nimage-1
        deltay(iimage)=sum( (glob%icoords(neb%cstart(iimage+1):&
            neb%cend(iimage+1))-glob%icoords(neb%cstart(iimage):&
            neb%cend(iimage)))**2 )
        qts%S_0=qts%S_0 + 2.D0*deltay(iimage)/qts%dtau(iimage+1)
      end do

    end if

    call allocate (qts%vhessian, neb%varperimage,neb%varperimage,neb%nimage)
    qts%vhessian=0.D0
    
!    call allocate (qts%total_hessian,neb%varperimage*neb%nimage*2,neb%varperimage*neb%nimage*2)
!    qts%total_hessian(:,:)=0.D0

    call allocate (qts%ihessian_image,neb%varperimage,neb%varperimage)
    !call allocate (qts%xhessian_image,glob%nvar,glob%nvar)

  endif

  ! the temperature may change in read_qts_coords, so beta_hbar should be
  ! defined after that
  !beta_hbar=1.D0/(glob%temperature*3.166815208D-6)  ! beta * hbar = hbar / (kB*T)
  call dlf_constants_get("KBOLTZ_AU",kboltz_au)
  beta_hbar=1.D0/(glob%temperature*kboltz_au)  ! beta * hbar = hbar / (kB*T)


  fd_hess_running=.false.

  qts%image_status=1
!  qts%const_hess=(beta_hbar/dble(neb%nimage*2))**2
  glob%xcoords=reshape(neb%xcoords(:,1),(/3,glob%nat/))

  ! Explanation to qts%hessian_mode (JK):
  ! 0 recalculate in each step 
  ! 1 calculate in first step, but keep it fixed then
  ! 2 copy it from the middle image (non-middle images only)
  ! 3 update
  ! unit-place (mod(hessian_mode,10)): middle image
  ! 10-place (hessian_mode/10): non-middle images
  ! 00  recalculate Hessians of all images in each optimisation step
  ! 11  calculate Hessians of all images at the start, but keep them fixed during the optimisation
  ! 33  calculate Hessians of all images at the start, update them during the optimisation 
  ! 20  recalculate only the hessian of the middle image at each step and copy it to all other images - deactivated option
  ! 30  recalculate only the hessian of the middle image at each step, update the other images with it 
  ! 21  calculate only the hessian of the middle image at the start, copy it to all other images, keep it fixed - deactivated option
  ! 31  calculate only the hessian of the middle image at the start, update all others based on that 

  ! This may become an input parameter in the future
  qts%hessian_mode=33 ! generally: 33, hessian_mode=0 is broken. If
                      ! you want recalculation at each step, see below
                      ! at "hack to recalculate rather than update:"
  qts%first_hessian=.true.

  if(glob%iopt==12) qts%hessian_mode=0 ! make sure the whole Hessian is calculated for Rate calculations

  ! allocate storage for hessian update in case any update is requested
  qts%tupdate=qts%needhessian.and.(mod(qts%hessian_mode,10)==3 .or. qts%hessian_mode/10==3)
  call allocate (qts%igradient, neb%varperimage*neb%nimage) 
  qts%igradient=0.D0
  if(qts%tupdate) then
! call allocate (qts%igradient_old, neb%varperimage*neb%nimage) ! delete ...
! call allocate (qts%icoords_old, neb%varperimage*neb%nimage)
    call allocate (qts%igradhess, neb%varperimage,neb%nimage)
    call allocate (qts%icoordhess, neb%varperimage,neb%nimage)
  end if

  ! try for analytic hessian the first time
  qts%try_analytic_hessian=.true.

  ! read and interpret glob%qtsflag
  qts%tsplit= (mod(glob%qtsflag,10)==1) 

  ! here the method of task farming can be decided (based on the number of
  ! images) and should be reported
  taskfarm_mode=1 
  !1: FD-Hessians are parallelised within an image over the 3*nat components
  ! (no parallelisation of analytic Hessians possible

  ! 2: Hessians are parallelised over the images (only makes sense for ntasks
  ! <=nimage) multiplied by dtau in the Action-Hessian

  ! The energy calculations in an QTS optimisation are always parallelised
  ! over images. taskfarm_mode has to be 1 in this case
  if(glob%ntasks>1) then
    if(glob%iopt==12.and.glob%ntasks<=neb%nimage) taskfarm_mode=2
    if(printl>=2) then
      if(taskfarm_mode==1) write(stdout,"('Parallelisation of FD-Hessians within one image')")
      if(taskfarm_mode==2) write(stdout,"('Parallelisation of FD-Hessians over images')")
    end if
  end if
  
  minfreq=-1.D0

end subroutine dlf_qts_init
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/dlf_qts_destroy
!!
!! FUNCTION
!!
!! Deallocate arrays
!!
!! SYNOPSIS
subroutine dlf_qts_destroy()
  !! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_qts
  use dlf_global, only: glob
  use dlf_allocate, only: allocate, deallocate
  use dlf_neb, only: neb,unitp,xyzall
  implicit none

  ! no need to do anything if just the classical rate is to be calculated
  if(glob%iopt==13) return

  if (allocated(qts%tau_prime)) call deallocate(qts%tau_prime)
  if (allocated(qts%tau_qts)) call deallocate(qts%tau_qts)
  if (allocated(qts%theta)) call deallocate(qts%theta)
  if (allocated(qts%coords_midp)) call deallocate(qts%coords_midp)
  if (allocated(qts%d_actionS_alt)) call deallocate(qts%d_actionS_alt)
  if (allocated(qts%d_actionS)) call deallocate(qts%d_actionS)
  if (allocated(qts%dtau)) call deallocate(qts%dtau)
  if (allocated(qts%dist)) call deallocate(qts%dist)

  call dlf_lbfgs_select("dimer rotation",.false.)
  call dlf_lbfgs_destroy
  call dlf_lbfgs_deselect

  if(qts%needhessian) then
    if (allocated(qts%vhessian)) call deallocate (qts%vhessian)
!    call deallocate (qts%total_hessian)
    if (allocated(qts%ihessian_image)) call deallocate (qts%ihessian_image)
    !call deallocate (qts%xhessian_image)
  endif

  ! deallocate storage for hessian update in case any update is requested
  if (allocated(qts%igradient)) call deallocate (qts%igradient) 
  if(qts%tupdate) then
!      call deallocate (qts%igradient_old)
!      call deallocate (qts%icoords_old)
    if (allocated(qts%igradhess)) call deallocate (qts%igradhess)
    if (allocated(qts%icoordhess)) call deallocate (qts%icoordhess)
  end if

end subroutine dlf_qts_destroy
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/qts_tau_from_path
!!
!! FUNCTION
!!
!! Calculate qts%tau_qts as the tangent to the instanton path (normalised)
!!
!! INPUTS
!!
!! glob%icoords
!!
!! OUTPUTS
!! 
!! qts%tau_qts 
!!
!! SYNOPSIS
subroutine qts_tau_from_path()
  !! SOURCE
  use dlf_neb, only: neb,unitp,xyzall
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, pi
  use dlf_qts
  implicit none
  integer             :: iimage, nimage
  logical,parameter   :: sinscale=.false.

  ! no need to do anything if just the classical rate is to be calculated
  if(glob%iopt==13) return

  nimage=neb%nimage

  if (sinscale) then 

    ! sin(iimage/(nimage+1)*pi) for scaling
    ! if ( iimage==1)

    qts%tau_qts(neb%cstart(1):neb%cend(1))=0.5D0*(glob%icoords&
        (neb%cstart(2):neb%cend(2))-glob%icoords&
        (neb%cstart(1):neb%cend(1)))

    !Norm single image 
    qts%tau_qts(neb%cstart(1):neb%cend(1))=qts%tau_qts(neb%cstart(1):neb%cend(1))/&
        dsqrt(sum((qts%tau_qts(neb%cstart(1):neb%cend(1)))**2))

    !Scaling with sin(iimage/(nimage+1.D0)*pi)
    qts%tau_qts(neb%cstart(1):neb%cend(1))=&
        sin(1.D0/(dble(nimage)+1.D0)*pi)*qts%tau_qts(neb%cstart(1):neb%cend(1))


    do iimage=2,nimage-1
      qts%tau_qts(neb%cstart(iimage):neb%cend(iimage))=0.5D0*(glob%icoords&
          (neb%cstart(iimage+1):neb%cend(iimage+1))-glob%icoords&
          (neb%cstart(iimage-1):neb%cend(iimage-1)))

      !Norm single image
      qts%tau_qts(neb%cstart(iimage):neb%cend(iimage))=&
          qts%tau_qts(neb%cstart(iimage):neb%cend(iimage))/&
          dsqrt(sum((qts%tau_qts(neb%cstart(iimage):neb%cend(iimage)))**2))

      !Scaling with sin(iimage/(nimage+1.D0)*pi)
      qts%tau_qts(neb%cstart(iimage):neb%cend(iimage))=&
          sin(dble(iimage)/(dble(nimage)+1.D0)*pi)*&
          qts%tau_qts(neb%cstart(iimage):neb%cend(iimage))
    end do

    !   if ( iimage==nimage)
    qts%tau_qts(neb%cstart(nimage):neb%cend(nimage))=0.5D0*(glob%icoords&
        (neb%cstart(nimage):neb%cend(nimage))-glob%icoords&
        (neb%cstart(nimage-1):neb%cend(nimage-1)))

    !Norm single image
    qts%tau_qts(neb%cstart(nimage):neb%cend(nimage))=&
        qts%tau_qts(neb%cstart(nimage):neb%cend(nimage))/&
        dsqrt(sum((qts%tau_qts(neb%cstart(nimage):neb%cend(nimage)))**2))

    !Scaling with sin(iimage/(nimage+1.D0)*pi)
    qts%tau_qts(neb%cstart(nimage):neb%cend(nimage))=&
        sin(dble(nimage)/(dble(nimage)+1.D0)*pi)*qts%tau_qts(neb%cstart(nimage):neb%cend(nimage))

  else ! (sinscale)

    ! no scaling
    ! if ( iimage==1)

    qts%tau_qts(neb%cstart(1):neb%cend(1))=0.5D0*(glob%icoords&
        (neb%cstart(2):neb%cend(2))-glob%icoords&
        (neb%cstart(1):neb%cend(1)))

    do iimage=2,nimage-1
      qts%tau_qts(neb%cstart(iimage):neb%cend(iimage))=0.5D0*(glob%icoords&
          (neb%cstart(iimage+1):neb%cend(iimage+1))-glob%icoords&
          (neb%cstart(iimage-1):neb%cend(iimage-1)))
    end do

    !   if ( iimage==nimage)

    qts%tau_qts(neb%cstart(nimage):neb%cend(nimage))=0.5D0*(glob%icoords&
        (neb%cstart(nimage):neb%cend(nimage))-glob%icoords&
        (neb%cstart(nimage-1):neb%cend(nimage-1)))

!!$   do iimage=1,nimage
!!$     !Norm single image - only useful for variable dtau!
!!$     qts%tau_qts(neb%cstart(iimage):neb%cend(iimage))=&
!!$         qts%tau_qts(neb%cstart(iimage):neb%cend(iimage))/&
!!$         dsqrt(sum((qts%tau_qts(neb%cstart(iimage):neb%cend(iimage)))**2))
!!$   end do

  end if !sinscale

  ! normalise
  qts%tau_qts(1:glob%nivar)=qts%tau_qts(1:glob%nivar)/&
      dsqrt(sum(qts%tau_qts**2)) 
  
end subroutine qts_tau_from_path
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/dlf_qts_trans_force
!!
!! FUNCTION
!!
!! Calculate the gradient (glob%igradient) to be used in the
!! optimisation. This can be the direct conversion of the energy-gradient to
!! the action-gradient (for saddle-point search algorithms), or it can mean to
!! apply the dimer method or the tangent-mode method. Calls qts_gradient_etos.
!!
!! Writes status files and information (qtsinfo, qtspath.xyz)
!!
!! INPUTS
!!
!! glob%icoords, glob%igradient
!!
!! OUTPUTS
!! 
!! glob%igradient, qts%dist 
!!
!! SYNOPSIS
subroutine dlf_qts_trans_force(trerun,tok_back)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, pi,printl,printf
  use dlf_neb, only: neb,unitp,xyzall,beta_hbar
  use dlf_qts

  implicit none
  logical, intent(out)  :: trerun ! calculate all E&G once more
  logical, intent(inout):: tok_back ! return status. Input: true for real geometry, false for FD grad
  logical   :: stoprot
  real(rk)  :: tolrot,svar,etunnel
  integer   :: iimage
  !  real(rk)  :: dist(neb%nimage+1) ! distance to previous image (or to start of path)
  !                                  ! dist(nimage+1)=distance of the path end to the last image
  real(rk)  :: tmpvec(neb%varperimage)

  !********* Set variables ********************************************
  tolrot=glob%tolrot/180.D0*pi

  ! *********************************************************************
  ! Switch for calculating the hessian only at the start of a minimisation (JK):
  !JK if the following line is commented out, the Hessian is only calculated at the start
  if(mod(qts%hessian_mode,10)==0 .or. mod(qts%hessian_mode,10)==3) qts%image_status=1 

  ! set parameters
  !  delta=glob%delta
  !glob%delta=1


  ! real start:
  tok_back=.true.

  ! calculate the derivative of the action with respect to the coordinates:
  call qts_gradient_etos(qts%d_actionS)

  if(tolrot >= pi/2.D0 .or. (glob%iopt>=10.and.glob%iopt/=30) ) then !if tolrot_check

    ! in case of constant mode, get mode (tau) from the path geometry
    call qts_tau_from_path()

!    if(glob%iopt<10) print*,"JK tau:",shape(qts%tau_qts),qts%tau_qts(10:20)

    qts%d_actionS_alt=qts%d_actionS
    stoprot=.true.

  else

    ! Dimer Method

    if(qts%status == 1) then

      !compute action at midpoint, modify coordinates (endpoint x_1)
      !set qts%status=2

      call dlf_qts_phi_1(trerun,stoprot)

      !    call dlf_dimer_was_midpoint(trerun_energy,testconv)??
      !    if(.not.trerun_energy) return ! converged


    else if (qts%status == 2) then

      !Compute Theta, C_Tau, estimate rotation angle Phi_1 (convergence test) and modify coordinates (endpoint x_prime)
      call dlf_qts_min_phi_part1(trerun,stoprot)

    else if (qts%status == 3) then

      !Compute C_Tau2, phi_min, rotation of tau_qts, modify coordinates (endpoint x_min), convergence test
      call dlf_qts_min_phi_part2(trerun,stoprot)

    else
      call dlf_fail("Wrong qts%status")
    end if
  end if !tolrot_check
  ! ********************************************************************
  ! If the rotations are converged, transform the gradient 
  ! ********************************************************************

  !
  ! calculate path length
  !
  ! end points:
  ! get tangent vector:
  tmpvec(:)=glob%icoords(neb%cstart(2):neb%cend(2))-glob%icoords(neb%cstart(1):neb%cend(1))
  ! normalise
  svar=sqrt(sum(tmpvec**2))
  tmpvec=tmpvec/svar

  svar=sum(glob%igradient(neb%cstart(1):neb%cend(1))*tmpvec) ! should be >0
  qts%dist(1)=(neb%ene(1)-max(qts%etunnel,qts%ers))/svar

  ! get tangent vector:
  tmpvec(:)=glob%icoords(neb%cstart(neb%nimage-1):neb%cend(neb%nimage-1))-&
      glob%icoords(neb%cstart(neb%nimage):neb%cend(neb%nimage))
  ! normalise
  svar=sqrt(sum(tmpvec**2))
  tmpvec=tmpvec/svar

  svar=sum(glob%igradient(neb%cstart(neb%nimage):neb%cend(neb%nimage))*tmpvec) ! should be >0
  qts%dist(neb%nimage+1)=(neb%ene(neb%nimage)-max(qts%etunnel,qts%ers))/svar
  !print*,"Path extension:",qts%dist(1),qts%dist(neb%nimage+1)

  ! length of the rest of the path: just add up the components:
  do iimage=2,neb%nimage
    qts%dist(iimage)=qts%dist(iimage-1)+sqrt(sum((glob%icoords(neb%cstart(iimage-1):&
        neb%cend(iimage-1))-glob%icoords(neb%cstart(iimage):&
        neb%cend(iimage)))**2)) 
  end do
  qts%dist(neb%nimage+1)=qts%dist(neb%nimage+1)+qts%dist(neb%nimage)
  !qts%dist=dist

  ! transform the gradient in case of dimer. Store old gradient (updates)
  if(stoprot) then
    if(glob%iopt<10.or.glob%iopt==30) then
      ! we have a minimum-search, modify the gradient by reversing it along tau
      !F_trans (see Kae08, eq.2) (in glob%igradient)
      glob%igradient=qts%d_actionS_alt-2.D0*DOT_PRODUCT(qts%d_actionS_alt,&
          qts%tau_qts)*qts%tau_qts

      if(tolrot < pi/2.D0) then !if tolrot_check
        call dlf_lbfgs_select("dimer rotation",.false.)
        call dlf_lbfgs_restart
        call dlf_lbfgs_deselect
      end if

      qts%status=1
    else
      ! Newton-Raphson update
      if(qts%tupdate) qts%igradient=glob%igradient ! store for update
      glob%igradient=qts%d_actionS
    end if
  end if



  ! store coordinates to disk
  call write_qts_coords(glob%nat,neb%nimage,neb%varperimage,glob%temperature,&
      qts%S_0,qts%S_pot,qts%S_ins,neb%ene,neb%xcoords,qts%dtau,qts%etunnel,qts%dist)

  !
  ! write information files (qtsinfo)
  !
  if(qts%status==1.and.printl>=2.and.printf>=2) then
    ! list of energies 
    if (glob%ntasks > 1) then
      open(unit=501,file="../qtsinfo")
    else
      open(unit=501,file="qtsinfo")
    end if
    write(501,"('# S_0:   ',f20.10)") qts%S_0
    write(501,"('# S_pot: ',f20.10)") qts%S_pot
    write(501,"('# S_ins: ',f20.10)") qts%S_ins
    etunnel=(qts%S_pot - 0.5D0 * qts%S_0 )/beta_hbar
    write(501,"('# Etunnel from those:          ',f18.10)") etunnel
    write(501,"('# Etunnel energy conservation: ',f18.10)") qts%etunnel
    write(501,"('# Unit of path length: mass-weighted Cartesians (sqrt(m_e)*Bohr)')")
    write(501,*)
    write(501,"('# Path length  Energy-E_b         Energy          dtau            deltay')")
    do iimage=1,neb%nimage
      if(iimage==1) then
        svar=qts%dist(iimage)
      else
        svar=qts%dist(iimage)-qts%dist(iimage-1)
      end if
      write(501,"(f10.5,4f17.10)") qts%dist(iimage),neb%ene(iimage)-qts%etunnel,neb%ene(iimage),qts%dtau(iimage),svar
    end do
    ! report on last image
    iimage=neb%nimage+1
    write(501,"('#',f9.5,17x,17x,2f17.10)") qts%dist(iimage),qts%dtau(iimage),qts%dist(iimage)-qts%dist(iimage-1)
    write(501,*)
    write(501,"(f10.5,2f17.10)") 0.D0,0.D0,qts%etunnel
    write(501,"(f10.5,2f17.10)") qts%dist(neb%nimage+1),0.D0,qts%etunnel
    close(501)

    ! qtspath.xyz
    if (glob%ntasks > 1) then
      open(unit=501,file="../qtspath.xyz")
    else
      open(unit=501,file="qtspath.xyz")
    end if
    
    do iimage=1,neb%nimage
      ! here, always write the whole system
      !if(xyzall) then
      call write_xyz(501,glob%nat,glob%znuc,neb%xcoords(:,iimage))
      !else
      !  call write_xyz_active(501,glob%nat,glob%znuc,glob%spec,neb%xcoords(:,iimage))
      !end if
      ! write chemshell fragments as well (or any other format used by dlf_put_coords)
      svar=neb%ene(iimage)
      if(neb%maximage>0) svar=neb%ene(neb%nimage)
      call dlf_put_coords(glob%nat,-iimage,svar,neb%xcoords(:,iimage),glob%iam)
    end do
    close(501)

    if(printf>=4) then
      ! qtspath2.xyz
      ! qts path doubled, for visualisation of modes
      if (glob%ntasks > 1) then
        open(unit=501,file="../qtspath2.xyz")
      else
        open(unit=501,file="qtspath2.xyz")
      end if
      do iimage=1,neb%nimage
        call write_xyz(501,glob%nat,glob%znuc,neb%xcoords(:,iimage))
      end do
      do iimage=neb%nimage,1,-1
        call write_xyz(501,glob%nat,glob%znuc,neb%xcoords(:,iimage))
      end do
      close(501)
    end if

  end if

  tok_back=.true.

end subroutine dlf_qts_trans_force
!!****

! use gradient and coordinates (and, maybe, the energies) of the
! images to define a dtau which fits to the geometry provided.  This
! routine was written for the paper J. Chem. Phys. 134, 184107 (2011),
! but the method is not generally applicable. The routine should not
! be used.
subroutine qts_get_dtau()
!! SOURCE
  use dlf_neb, only: neb,unitp,xyzall,beta_hbar
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,printl,printf,stdout
  use dlf_constants, only: dlf_constants_get
  use dlf_stat, only: stat
  use dlf_qts
  implicit none
  real(rk)  :: gradleft(neb%nimage) ! is also grad-parallel
  real(rk)  :: gradright(neb%nimage)
  real(rk)  :: gradperp(neb%nimage)
  real(rk)  :: alpha(neb%nimage) ! 1 if vectors around iimage parallel
  real(rk)  :: dist(neb%nimage+1) ! (nimage+1) distance to previous image
  real(rk)  :: dtau(neb%nimage+1)
  real(rk)  :: sgrad(neb%nimage)
  real(rk)  :: sgrado(neb%nimage) ! orthogonal
  real(rk)  :: sgradp(neb%nimage) ! parallel
  real(rk)  :: dsgrad(neb%nimage) ! needed??
  real(rk)  :: vec(neb%varperimage) ! vector from one image to the next
  real(rk)  :: vecp(neb%varperimage) ! vector from one image to the next
  integer   :: iimage,iter,iiter,ivar
  real(rk)  :: svar,scalegrad,gamma,delta
!
  real(rk)  :: dfdd(neb%nimage+1)
  real(rk)  :: olddtau(neb%nimage+1)
  real(rk)  :: oldgrad(neb%nimage+1)
  real(rk)  :: fdgrad(neb%nimage+1)
  logical   :: tconv,tchk,nanchk
  real(8),parameter :: toler=1.D-14

  if(printl>=4) write(stdout,*) "Generating an optimal set of dtau to keep the path as it is!"

  ! distance to previous image
  dist=0.D0
  do iimage=2,neb%nimage
    dist(iimage)=sqrt(sum( ( &
        glob%icoords(neb%cstart(iimage-1):neb%cend(iimage-1)) - &
        glob%icoords(neb%cstart(iimage):neb%cend(iimage))  )**2 ) ) 
  end do

  ! gradient component along the spring force
  gradleft=0.D0
  gradright=0.D0
  do iimage=1,neb%nimage
    if(iimage<neb%nimage) then
      vec(:)=glob%icoords(neb%cstart(iimage+1):neb%cend(iimage+1))-&
          glob%icoords(neb%cstart(iimage):neb%cend(iimage))
      svar=sqrt(sum(vec**2))
      if(svar>0.D0) then
        vec=vec/svar
        gradright(iimage)=sum(vec*glob%igradient(neb%cstart(iimage):neb%cend(iimage)))
      end if
    end if
    if(iimage>1) then
      vec(:)=glob%icoords(neb%cstart(iimage):neb%cend(iimage))-&
          glob%icoords(neb%cstart(iimage-1):neb%cend(iimage-1))
      vecp=0.D0
      if(iimage<neb%nimage) &
          vecp(:)=glob%icoords(neb%cstart(iimage+1):neb%cend(iimage+1))-&
          glob%icoords(neb%cstart(iimage):neb%cend(iimage))
      svar=sqrt(sum(vec**2))
      if(svar>0.D0) then
        vec=vec/svar
        gradleft(iimage)=sum(vec*glob%igradient(neb%cstart(iimage):neb%cend(iimage)))
      end if
      ! calculate gradperp and alpha
      svar=sqrt(sum(vecp**2))
      if(svar>0.D0) then
        vecp=vecp/svar
        svar=sum(vec*vecp)
        alpha(iimage)=svar
        if(alpha(iimage)>1.D0) alpha(iimage)=1.D0
        vecp=vecp-vec*svar
        svar=sqrt(sum(vecp**2))
        if(svar>0.D0) then
          vecp=vecp/svar
          gradperp(iimage)=sum(vecp*glob%igradient(neb%cstart(iimage):neb%cend(iimage)))
        else
          alpha(iimage)=1.D0
          gradperp(iimage)=0.D0
        end if
      else
        alpha(iimage)=1.D0
        gradperp(iimage)=0.D0
      end if

    end if
    ! what to do with the outer gradients of the end points? They play
    ! a role as well if I understand it right? At the moment they are zero, though
  end do
  ! set outer gradients to inner gradients
  gradleft(1)=gradright(1)
  gradright(neb%nimage)=gradleft(neb%nimage)
  gradperp(1)=0.D0
  gradperp(neb%nimage)=0.D0
  alpha(1)=1.D0

  dtau=qts%dtau
!!$  ! read dtau
!!$  open(file="dtau",unit=10)
!!$  read(10,*) dtau
!!$  close(10)
!!$  qts%dtau=dtau

  ! use constant step size as starting guess:
  !dtau(:)=beta_hbar/dble(2*neb%nimage)

!!$  do iimage=1,neb%nimage+1
!!$    print*,"dtau",iimage,dtau(iimage)
!!$  end do

  ! calculate sgrad
  do iimage=1,neb%nimage
    sgrad(iimage)=2.D0*(dist(iimage)/dtau(iimage)-dist(iimage+1)/dtau(iimage+1)) + &
        dtau(iimage)*gradleft(iimage)+dtau(iimage+1)*gradright(iimage)
  end do

!!$  do iimage=1,neb%nimage
!!$    print*,"sgrad along path",iimage,sgrad(iimage),sgrad(iimage)/beta_hbar
!!$  end do

  scalegrad=1.D0
  gamma=0.D0
  olddtau=0.D0
  call dlf_lbfgs_init(neb%nimage+1,20)
  ! iterate to solve the nonlinear system of equations:
  ! IMPROVE: add convergence test, ...
  tconv=.false.
  do iter=1,1000 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!$    ! FD of the objective function
!!$    delta=1.D-5
!!$    do ivar=1,neb%nimage+1
!!$      dtau(ivar)=dtau(ivar)+delta
!!$
!!$      do iimage=1,neb%nimage
!!$        sgradp(iimage)=2.D0*(dist(iimage)/dtau(iimage)-alpha(iimage)*dist(iimage+1)/dtau(iimage+1)) + &
!!$            (dtau(iimage)+dtau(iimage+1))*gradleft(iimage)
!!$        sgrado(iimage)=-2.D0*sqrt(1.D0-alpha(iimage)**2)*dist(iimage+1)/dtau(iimage+1) + &
!!$            (dtau(iimage)+dtau(iimage+1))*gradperp(iimage)
!!$      end do
!!$      svar=sum(sgradp**2)+sum(sgrado**2)
!!$      fdgrad(ivar)=svar
!!$
!!$      dtau(ivar)=dtau(ivar)-2.D0*delta
!!$
!!$      do iimage=1,neb%nimage
!!$        sgradp(iimage)=2.D0*(dist(iimage)/dtau(iimage)-alpha(iimage)*dist(iimage+1)/dtau(iimage+1)) + &
!!$            (dtau(iimage)+dtau(iimage+1))*gradleft(iimage)
!!$        sgrado(iimage)=-2.D0*sqrt(1.D0-alpha(iimage)**2)*dist(iimage+1)/dtau(iimage+1) + &
!!$            (dtau(iimage)+dtau(iimage+1))*gradperp(iimage)
!!$      end do
!!$      svar=sum(sgradp**2)+sum(sgrado**2)
!!$      fdgrad(ivar)=(fdgrad(ivar)-svar)*0.5D0/delta
!!$
!!$      dtau(ivar)=dtau(ivar)+delta
!!$    end do
!!$    ! <<<<<<< END OF FD-GRAD

    ! sgradp (parallel) and sgrado (orthogonal)
    do iimage=1,neb%nimage
      sgradp(iimage)=2.D0*(dist(iimage)/dtau(iimage)-alpha(iimage)*dist(iimage+1)/dtau(iimage+1)) + &
          (dtau(iimage)+dtau(iimage+1))*gradleft(iimage)
      sgrado(iimage)=-2.D0*sqrt(1.D0-alpha(iimage)**2)*dist(iimage+1)/dtau(iimage+1) + &
          (dtau(iimage)+dtau(iimage+1))*gradperp(iimage)
    end do
!!$  do iimage=1,neb%nimage
!!$    print*,"sgrad parallel, orthogonal",iimage,sgradp(iimage),sgrado(iimage)
!!$  end do
    svar=sum(sgradp**2)+sum(sgrado**2)
    if(svar<toler) then
      tconv=.true.
      exit
    end if
!    if(mod(iter,100)==1) 
    ! NaN check (comparison of NaN with any number 
    ! returns .false. , pgf90 does not understand isnan() )
    nanchk = .false.
    if (abs(svar) > huge(1.D0)) then
       nanchk = .true.
    else
       if (.not. abs(svar) < huge(1.D0)) nanchk = .true. 
    end if
    if(nanchk) then
      if(printl>=2) then
        do iimage=1,neb%nimage
          write(stdout,*) iimage,dist(iimage),alpha(iimage),sqrt(1.D0-alpha(iimage)**2),dtau(iimage)
          write(stdout,*) iimage,gradleft(iimage),gradperp(iimage)
        end do
      end if
      call dlf_fail("Objective function is NaN")
    end if
    !print*,"Objective function",iter,svar!,sum(dtau)+sum(dtau(2:neb%nimage))-beta_hbar
    
    ! calculate gradient of the objective function
    do iimage=1,neb%nimage+1
      dfdd(iimage)=0.D0
      ! the part for l=k
      if(iimage<=neb%nimage) then
        dfdd(iimage)=2.D0*sgradp(iimage)* &
            (-2.D0/dtau(iimage)**2*dist(iimage) + gradleft(iimage) ) + &
            2.D0*sgrado(iimage)*gradperp(iimage)
      end if
      ! add the part for l+1=k
      if(iimage>1) then
        dfdd(iimage)=dfdd(iimage)+&
            2.D0*sgradp(iimage-1) * (2.D0*alpha(iimage-1)/dtau(iimage)**2*dist(iimage) + gradleft(iimage-1)) + &
            2.D0*sgrado(iimage-1) * (2.D0*sqrt(1.D0-alpha(iimage-1)**2)/dtau(iimage)**2*dist(iimage) + &
            gradperp(iimage-1) ) 
      end if
    end do
!!$  do iimage=1,neb%nimage+1
!!$    print*,"grad of the objective function,FD",iimage,dfdd(iimage),fdgrad(iimage)
!!$  end do

  ! steepest descent:
  !dtau=dtau-scalegrad*dfdd
  
!!$  ! CG step
!!$  if(mod(iter,50)==1) then
!!$    gamma=0.D0
!!$    olddtau=0.D0
!!$  else
!!$    gamma=sum((dfdd-oldgrad)*dfdd)/sum(oldgrad*oldgrad)
!!$    if(gamma<0.D0) then
!!$      gamma=0.D0
!!$      print*,"gamma < 0"
!!$    end if
!!$  end if
!!$  gamma=0.D0
!!$  dtau=dtau +scalegrad*(-dfdd + gamma * olddtau)
!!$
!!$  oldgrad=dfdd
!!$  olddtau=-dfdd + gamma * olddtau

    ! oldgrad is used as step
    cALL DLF_LBFGS_STEP(dtau,dfdd,oldgrad)

    !
    ! check for nan
    !
    tchk=.false.
    do iimage=1,neb%nimage+1
      ! NaN check (comparison of NaN with any number 
      ! returns .false. , pgf90 does not understand isnan() )
      nanchk = .false.
      if (abs(oldgrad(iimage)) > huge(1.D0)) then
         nanchk = .true.
      else
         if (.not. abs(oldgrad(iimage)) < huge(1.D0)) nanchk = .true. 
      end if
      if(nanchk) then
        write(stdout,*) "Step direction component",iimage," is NaN"
        tchk=.true.
      end if
    end do
    if(tchk) then
      do iimage=1,neb%nimage+1
        write(stdout,*) iimage,dtau(iimage),dfdd(iimage),oldgrad(iimage)
      end do
      exit
     ! call dlf_fail("Step size is nan")
    end if

    do iiter=1,50
      if(minval(dtau+oldgrad)<1.D-10) then 
        oldgrad=oldgrad*0.5D0
        write(stdout,*) "reducing step size in iteration",iter
      else
        exit
      end if
    end do
    
    ! check for nan
    do iimage=1,neb%nimage+1
      ! NaN check (comparison of NaN with any number 
      ! returns .false. , pgf90 does not understand isnan() )
      nanchk = .false.
      if (abs(oldgrad(iimage)) > huge(1.D0)) then
         nanchk = .true.
      else
         if (.not. abs(oldgrad(iimage)) < huge(1.D0)) nanchk = .true. 
      end if
      if(nanchk) write(stdout,*) "Step direction component",iimage," is NaN after shortening"
    end do

    dtau=dtau+oldgrad

!  print*,"norm",sum(dtau)+sum(dtau(2:neb%nimage))-beta_hbar
  !works

    ! renormalise dtau so that the sum matches beta_hbar
    ! Attention: the current procedure does not make sure dtau stays > 0 in each case!
    svar=sum(dtau)+sum(dtau(2:neb%nimage))-beta_hbar
    svar=svar/dble(neb%nimage+1)
    dtau(1)=dtau(1)-svar
    dtau(2:neb%nimage)=dtau(2:neb%nimage)-0.5D0*svar
    dtau(neb%nimage+1)=dtau(neb%nimage+1)-svar

!!$  !alternative:
!!$  svar=sum(dtau)+sum(dtau(2:neb%nimage))-beta_hbar
!!$  svar=svar/dble(2*neb%nimage)
!!$  dtau(1)=dtau(1)-svar
!!$  dtau(2:neb%nimage)=dtau(2:neb%nimage)-svar
!!$  dtau(neb%nimage+1)=dtau(neb%nimage+1)-svar
!  print*,"norm",sum(dtau)+sum(dtau(2:neb%nimage))-beta_hbar

  end do ! iter=1,1000
  call dlf_lbfgs_destroy
  ! and loop until here

!!$  print*,"Final dtau:"
!!$  do iimage=1,neb%nimage+1
!!$    print*,"dtau",dtau(iimage)
!!$  end do

  if(.not.tconv) then
    write(stdout,*) "Severe warning: Loop for finding dtau not converged"
!    call dlf_fail("Loop for finding dtau not converged")
  end if
  
  qts%dtau=dtau

!  call dlf_fail("End in qts_get_dtau")
  
end subroutine qts_get_dtau

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/qts_gradient_etos
!!
!! FUNCTION
!!
!! Converts a gradient of the potential energy to a gradient of Euclidean
!! action. Calculates the total action. Writes the file etunnel.
!!
!! INPUTS
!!
!! glob%igradient, qts%dtau
!!
!! OUTPUTS
!! 
!! qts%S_pot, qts%S_ins, qts%S_0, qts%etunnel
!!
!! SYNOPSIS
subroutine qts_gradient_etos(grad_action)
!! SOURCE
  use dlf_neb, only: neb,unitp,xyzall,beta_hbar
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,printl,printf,stdout
  use dlf_constants, only: dlf_constants_get
  use dlf_stat, only: stat
  use dlf_qts
  implicit none
  real(rk),intent(out)  :: grad_action(neb%varperimage*neb%nimage) ! dS_E / dy
  real(rk)  :: svar
  real(rk)  :: hartree,echarge,ev
  real(rk)  :: deltay(neb%nimage)
  integer   :: iimage, nimage,cstart,cend,ivar
  real(rk)  :: s_0var,spotvar
  real(rk)  :: etunnel,etunnelapprox(neb%nimage-2)
  !**********************************************************************************

  call dlf_constants_get("HARTREE",HARTREE)
  call dlf_constants_get("ECHARGE",echarge)
  ev=hartree/echarge

  nimage=neb%nimage
  deltay(:)=0.D0

  ! here is the place where dtau is used the first time in the first
  ! step. So, here we could modify dtau and already have all energies
  ! and gradients available
  
  ! adjust tau to keep image positions: (commented out normally)
  !if(stat%ccycle==1.and.glob%nebk>0.99D0) call qts_get_dtau()

  ! store gradient of first images for image updates
  if(stat%ccycle==1) then

    ! if all or some Hessians were read:
    if(qts%tupdate.and.qts%hessian_mode==33) then
      qts%igradhess(:,:)=reshape(glob%igradient,(/neb%varperimage,neb%nimage/))
    end if

    ! if only the classical Hessian was read:
    if(qts%tupdate.and.qts%hessian_mode==31) then
      ! provide gradient for the middle image at least
      if(mod(neb%nimage,2)==1) then
        ! odd number of images
        ivar=(neb%nimage+1)/2 ! middle image
        do iimage=1,neb%nimage
          qts%igradhess(:,iimage)=glob%igradient(neb%cstart(ivar):neb%cend(ivar))
        end do
!!$        qts%igradhess=0.D0
        ! if the gradient is zero, the coords and gradient of the middle image must be set after the first updates (line about 1270)
      else
        ! even number of images
        ivar=neb%nimage/2 ! left of middle
        do iimage=1,neb%nimage
          qts%igradhess(:,iimage)=0.5D0*(glob%igradient(neb%cstart(ivar):neb%cend(ivar))+&
              glob%igradient(neb%cstart(ivar+1):neb%cend(ivar+1)))
        end do
!!$        qts%igradhess=0.D0
      end if
    end if

  end if

  do iimage=1,nimage

    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage)
    ! this implements the version with all images doubled, also the end-images
    if ( iimage==1) then
      grad_action(cstart:cend)= &
          glob%igradient(cstart:cend)*(qts%dtau(iimage)+qts%dtau(iimage+1)) + &
          2.D0 * ( &
          +glob%icoords(cstart:cend)/(qts%dtau(iimage+1)) &
          -glob%icoords(neb%cstart(iimage+1):neb%cend(iimage+1))/qts%dtau(iimage+1))
    else if (iimage>1 .AND. iimage<nimage) then
      ! general image
      grad_action(cstart:cend)= &
          glob%igradient(cstart:cend)*(qts%dtau(iimage)+qts%dtau(iimage+1)) + &
          2.D0 * ( &
          -glob%icoords(neb%cstart(iimage-1):neb%cend(iimage-1))/qts%dtau(iimage) &
          +glob%icoords(cstart:cend)*(1.D0/qts%dtau(iimage)+1.D0/qts%dtau(iimage+1)) &
          -glob%icoords(neb%cstart(iimage+1):neb%cend(iimage+1))/qts%dtau(iimage+1))
    else
      grad_action(cstart:cend)= &
          glob%igradient(cstart:cend)*(qts%dtau(iimage)+qts%dtau(iimage+1)) + &
          2.D0 * ( &
          -glob%icoords(neb%cstart(iimage-1):neb%cend(iimage-1))/qts%dtau(iimage) &
          +glob%icoords(cstart:cend)/qts%dtau(iimage)) 
    end if
    !    grad_action(neb%cstart(iimage):neb%cend(iimage))=dSdy1(1:neb%varperimage)

    ! for calculating the action itself
    if ( iimage/=nimage) then
      deltay(iimage)=sum( (glob%icoords(neb%cstart(iimage+1):&
          neb%cend(iimage+1))-glob%icoords(neb%cstart(iimage):&
          neb%cend(iimage)))**2 )
    end if

  end do

  ! in the equidistant-tau code, the gradient was divided by beta_hbar to make
  ! it independent of beta_hbar (with respect to energy). This may be a good
  ! idea. I'll do that here as well
  grad_action(:)=grad_action/beta_hbar

  !make sure the action parts are only calculated for the dimer midpoint
  if(qts%status==1) then
    ! qts%S_0 is the correct S_0 in dim. of action
    qts%S_0 = sum(deltay)*4.D0*dble(nimage)/beta_hbar
    s_0var=0.D0
    spotvar=0.D0
    do iimage=1,nimage
      s_0var=s_0var+deltay(iimage)/qts%dtau(iimage+1)
      spotvar=spotvar+neb%ene(iimage)*(qts%dtau(iimage)+qts%dtau(iimage+1))
    end do
    s_0var=s_0var*2.D0
    !print*,"S_0",qts%S_0 ,  s_0var

    qts%S_pot = sum(neb%ene(1:nimage))*beta_hbar/dble(nimage)
    !print*,"Spot", qts%S_pot,spotvar

    !in inst. action:
    qts%S_ins= 0.5D0 * qts%S_0 + qts%S_pot
    !print*,"sins",qts%S_ins,0.5D0*s_0var+spotvar

    ! store correct values:
    qts%S_ins=0.5D0*s_0var+spotvar
    qts%S_0=s_0var
    qts%S_pot=spotvar

    if(printl>=4) then
      write(stdout,"('Euclidean action in  Hartree   and      eV')")
      write(stdout,"('Total action      ',2es15.7)") qts%S_ins, qts%S_ins*ev
      write(stdout,"('S_0               ',2es15.7)") qts%S_0, qts%S_0*ev
      write(stdout,"('S_pot             ',2es15.7)") qts%S_pot, qts%S_pot*ev
      svar=(qts%S_pot - 0.5D0 * qts%S_0 )/beta_hbar
      !write(stdout,"('Tunnelling energy ',2es15.7)") svar,svar*ev
      write(stdout,"('Total path length ',2es15.7)") sum(sqrt(deltay))
    end if
  end if
  if(sum(sqrt(deltay))<1.D-7) then
    if(printl>=0) write(stdout,"('Warning: Path is collapsed to a point, rate not well-defined.')")
  end if

  ! print status file
  if(printf>=4) then ! this if statement should be removed because etunnel is not set otherwise IMPROVE
    open(file="etunnel",unit=110)
    write(110,*) "#r(iimage), tunnelling energy"
    write(110,*) 0.D0, svar ! global value
    write(110,*) sum(sqrt(deltay(1:nimage-1))), svar
    write(110,*) 
    ! average energy
    write(110,*) "# Average energy"
    do iimage=1,nimage-1
      ! deltay(iimage) refers to the interval following iimage
      ! dtau(iimage) refers to the interval before iimage
      write(110,*) sum(sqrt(deltay(1:iimage)))-sqrt(deltay(iimage))*0.5D0, &
          0.5D0*(neb%ene(iimage)+neb%ene(iimage+1)) - &
          0.5D0*deltay(iimage)/(qts%dtau(iimage+1))**2
      !            2.D0*dble(nimage)**2/beta_hbar**2*deltay(iimage)
    end do
    write(110,*) 
    write(110,*) "# geometric mean"
    ! next approximation: geometric mean
    do iimage=2,nimage-1
      etunnelapprox(iimage-1)=neb%ene(iimage)- &
          0.5D0 * sqrt(deltay(iimage)*deltay(iimage-1)) / qts%dtau(iimage+1)/qts%dtau(iimage)
      write(110,*) sum(sqrt(deltay(1:iimage-1))),etunnelapprox(iimage-1)
      !            2.D0*dble(nimage)**2/beta_hbar**2 * sqrt(deltay(iimage)*deltay(iimage-1))
    end do
    ! we could use the median of this to define etunnel
    call median(nimage-2,etunnelapprox,etunnel)
    write(stdout,"('Tunnelling energy ',2es15.7, ' (average along the path)')") etunnel,etunnel*ev
    qts%etunnel=etunnel

    write(110,*) 
    write(110,*) "# arithmetic mean"
    ! next approximation: arithmetic mean
    do iimage=2,nimage-1
      write(110,*) sum(sqrt(deltay(1:iimage-1))), &
          neb%ene(iimage)- &
          0.25D0* (sqrt(deltay(iimage))+sqrt(deltay(iimage-1)))**2 / (qts%dtau(iimage+1)**2+qts%dtau(iimage)**2)
      !            0.5D0*dble(nimage)**2/beta_hbar**2 * (sqrt(deltay(iimage))+sqrt(deltay(iimage-1)))**2
    end do
    write(110,*) 
    write(110,*) "#r(iimage), tunnelling energy: median of geometric mean"
    write(110,*) 0.D0, qts%etunnel ! global value
    write(110,*) sum(sqrt(deltay(1:nimage-1))),qts%etunnel
    write(110,*) 
    close(110)
  end if
  !end if ! qts%status==1

end subroutine qts_gradient_etos
!!****


! ********************************************************************
! Dimer rotation: Compute action at midpoint and move endpoint
! ********************************************************************
subroutine dlf_qts_phi_1(trerun,stoprot)
  use dlf_qts
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_neb, only: neb,unitp,xyzall
  implicit none

  logical, intent(out) :: trerun,stoprot


  !save gradient of the action at x_1
  qts%d_actionS_alt=qts%d_actionS


  !modify coordinates
  qts%coords_midp(1:neb%varperimage*neb%nimage)=&
      glob%icoords(1:neb%varperimage*neb%nimage)

  glob%icoords=glob%icoords+glob%delta*qts%tau_qts

  stoprot=.false.
  qts%status=2
  trerun=.true.

end subroutine dlf_qts_phi_1

! ********************************************************************
! Dimer rotation: Compute  phi_min (part1)
! ********************************************************************
subroutine dlf_qts_min_phi_part1(trerun,stoprot)
  use dlf_qts
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, pi, stdout
  use dlf_neb, only: neb,unitp,xyzall
  implicit none

  logical, intent(out) :: trerun,stoprot
  real(rk)  :: F_rot(neb%varperimage*neb%nimage) !rotational force
  real(rk)  :: tolrot, svar


  !********* Set variables ********************************************
  tolrot=glob%tolrot/180.D0*pi
  !********************************************************************


  svar=DOT_PRODUCT((qts%d_actionS(:)-&
      qts%d_actionS_alt(:)),&
      qts%tau_qts(:))

!!$  print*,"Abs(grad_alt-grad)",sqrt(sum((qts%d_actionS(:)-&
!!$      qts%d_actionS_alt(:))**2))
!!$  print*,"Abs(grad_alt)",sqrt(sum((&
!!$      qts%d_actionS_alt(:))**2))
!!$  print*,"Abs(grad)",sqrt(sum((qts%d_actionS(:))**2))

  ! rotational force
  F_rot(:)=-2.D0*(qts%d_actionS(:)-&
      qts%d_actionS_alt(:))+&
      2.D0*svar*qts%tau_qts(:)

  write(*,*) 'svar', svar
  write(*,*)

  !Compute theta
  call dlf_lbfgs_select("dimer rotation",.false.)
  call dlf_lbfgs_step(qts%tau_qts,-F_rot,qts%theta)
  call dlf_lbfgs_deselect
  !normalise theta
  qts%theta=qts%theta/dsqrt(sum(qts%theta**2))

  !Curvature along tau
  qts%C_Tau=DOT_PRODUCT((qts%d_actionS(:)-&
      qts%d_actionS_alt(:)),&
      qts%tau_qts(:))/glob%delta

  write(*,*) 'qts%C_Tau',qts%C_Tau
  write(*,*)
  write(stdout,"('Curvature before dimer rotation           ',es12.5)") qts%C_Tau

  !b_1 dC_taudPhi=0
  qts%b_1=DOT_PRODUCT((qts%d_actionS(:)-&
      qts%d_actionS_alt(:)),&
      qts%theta(:))/glob%delta

  !angle of first rotation (rough estimate)
  !qts%phi_1=-datan(-qts%b_1/qts%C_Tau)*0.5D0
  !qts%phi_1=datan(-qts%b_1/(abs(qts%C_Tau)))*0.5D0 !(wie im Paper Hey05)
  qts%phi_1=-datan(qts%b_1/(abs(qts%C_Tau)))*0.5D0 !(wie im Paper Kae08)


  ! write(*,*) 'phi_1', qts%phi_1
  !qts%phi_1=0.5D0

  write(*,*) 'phi_1', qts%phi_1, qts%phi_1/pi*180.D0
  write(*,*) 'b_1', qts%b_1

  if(abs(qts%phi_1)>tolrot) then
    !   !Rotation of endpoint to x_prime
    !   glob%icoords=qts%coords_midp+qts%tau_qts*cos(qts%phi_1)+qts%C_Tau*sin(qts%phi_1)
    !modify tau
    qts%tau_prime=qts%tau_qts*cos(qts%phi_1)+qts%theta*sin(qts%phi_1)
    qts%tau_prime=qts%tau_prime/dsqrt(sum(qts%tau_prime**2))
    !Rotation of endpoint to x_prime
    glob%icoords=qts%coords_midp+glob%delta*qts%tau_prime
    qts%status=3
    stoprot=.false.
    trerun=.true.
    return

  else 
    !converged, do translation
    write(*,*) 'QTS rotation converged 1, translation'
    glob%icoords=qts%coords_midp
    trerun=.false.
    stoprot=.true.
  end if

end subroutine dlf_qts_min_phi_part1

! ********************************************************************
! Dimer rotation: Compute phi_min (part2)
! ********************************************************************
subroutine dlf_qts_min_phi_part2(trerun,stoprot)
  use dlf_qts
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, pi, stdout
  use dlf_neb, only: neb,unitp,xyzall
  implicit none

  logical, intent(out) :: trerun, stoprot
  real(rk)  :: a_1, a_0, C_Tau2, phi_min, tolrot

  !********* Set variables ********************************************
  tolrot=glob%tolrot/180.D0*pi

  !********************************************************************

  write(stdout,"('Curvature after dimer rotation            ',es12.5)") &
      DOT_PRODUCT((qts%d_actionS(:)-&
      qts%d_actionS_alt(:)),&
      qts%tau_prime(:))/glob%delta

  !compute action(endpoint x_prime) and Fourier coefficients

  a_1=(qts%C_Tau-DOT_PRODUCT((qts%d_actionS(:)-&
      qts%d_actionS_alt(:)),&
      qts%tau_prime(:))/glob%delta+qts%b_1*sin(2.D0*qts%phi_1))&
      /(1.D0-cos(2.D0*qts%phi_1))

  a_0=2.D0*(qts%C_Tau-a_1)

  phi_min=0.5D0*datan(qts%b_1/a_1)


  if(abs(phi_min-qts%phi_1) > pi*0.5D0) then
    if(phi_min<qts%phi_1) then
      phi_min=phi_min+pi
      !print*,"adding pi to phi_min"
    else
      phi_min=phi_min-pi
      !print*,"adding -pi to phi_min"
    end if
  end if

  write(*,*) 'phi_min', phi_min,phi_min/pi*180.D0 
  write(*,*) "phi_min, phi1 deg",phi_min/pi*180.D0 ,qts%phi_1/pi*180.D0 
  write(*,*) 
  write(*,*) 'a_1', a_1 
  write(*,*)
  write(*,*) 'a_0', a_0 
  write(*,*)

  C_Tau2=0.5D0*a_0+a_1*cos(2.D0*phi_min)+qts%b_1*sin(2.D0*phi_min)

  !  if(printl>=6) then
  write(stdout,"('Expected curvature at dimer minimum       ',es12.5)") C_Tau2
  !  end if

  if(C_Tau2  > qts%C_Tau ) then
    ! Obtained angle seems to be a maximum: 
    phi_min=phi_min-0.5d0 * pi
    ! new curvature
    C_Tau2=0.5D0*a_0+a_1*cos(2.D0*phi_min)+qts%b_1*sin(2.D0*phi_min)
    !PRINT*,"got a maximum in curv, adding -pi/2"
  end if
  !Rotation t_min ausrechnen
  qts%tau_qts=qts%tau_qts*cos(phi_min)+qts%theta*sin(phi_min)
  qts%tau_qts=qts%tau_qts/dsqrt(sum(qts%tau_qts**2))


  !Check for convergence
  if(abs(phi_min)>tolrot) then
    !open(unit=555,file="tau_qts",POSITION="APPEND")

    ! write(555,*) qts%tau_qts
    ! write(555,*)
    ! write(555,*)
    ! write(555,*)

    !close(555)



    !Rotation of endpoint to x_min
    glob%icoords=qts%coords_midp+qts%tau_qts*glob%delta

    write(*,*) 'C_Tau2',C_Tau2  
    write(*,*)

    qts%status=2
    stoprot=.false.
    trerun=.true.
    return
  else
    !converged, translation
    write(*,*) 'QTS rotation converged 2, translation'
    write(stdout,"('Curvature after rotations converged       ',es12.5)") C_Tau2
    !rotate tau
    ! qts%tau_qts=qts%tau_prime/dsqrt(sum(qts%tau_prime**2))
    !glob%icoords auf midpoint x_0 setzen
    glob%icoords=qts%coords_midp
    trerun=.false.
    stoprot=.true.
  end if
end subroutine dlf_qts_min_phi_part2

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/dlf_qts_get_hessian
!!
!! FUNCTION
!!
!! Calculate the Hessian of the potential energy (V-Hessian) at each single
!! image. Analytic Hessian is tried, otherwise a finite-difference scheme is
!! used.
!!
!! glob%havehessian is used for each image individually here.
!!
!! INPUTS
!!
!! OUTPUTS
!! 
!! qts%vhessian
!!
!! SYNOPSIS
subroutine dlf_qts_get_hessian(trerun_energy)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_neb, only: neb,unitp,xyzall
  use dlf_global, only: glob,stdout,printl,tstore
  use dlf_hessian, only: fd_hess_running
  use dlf_allocate, only: allocate,deallocate
  use dlf_constants, only : dlf_constants_get
  use dlf_qts
  implicit none
  logical, intent(inout) :: trerun_energy
  integer                :: status
  !real(rk)               :: xhessian(glob%nvar,glob%nvar)
  real(rk),allocatable   :: xhessian(:,:) ! (glob%nvar,glob%nvar)
  integer                :: ivar,jvar, cstart, cend, iimage,printl_store
  logical                :: havehessian,fracrecalc,tok
!  real(rk)               :: hess_image(neb%varperimage,neb%varperimage)
!  real(rk),allocatable   :: hess_image(:,:) ! (neb%varperimage,neb%varperimage)
  integer,save           :: count_calc
  integer                :: nimage_read,dir
  real(rk),allocatable   :: mass(:)
  logical                :: was_updated,tokmass
  integer                :: fromimg,startimg,endimg,iat
  real(rk)               :: svar,arr2(2)
  character(64)          :: label
  real(rk),allocatable   :: ene_store(:)
  real(rk)               :: tmp(1) !needed to compile with gfortran10
  !**********************************************************************
  fracrecalc=.false.

  ! catch the case of inithessian==5 (Hessian to be read from
  ! qts_hessian.txt)
  if (glob%inithessian==5.or.glob%inithessian==7) then
    ! read hessian from disk
    nimage_read=neb%nimage
    call allocate(mass,glob%nat)
    if(glob%inithessian==7) then
      call allocate(ene_store,neb%nimage)
      ene_store=neb%ene
    end if
    call read_qts_hessian(glob%nat,nimage_read,neb%varperimage,glob%temperature,&
         neb%ene,neb%xcoords,qts%igradient,qts%vhessian,qts%etunnel,qts%dist,mass,"",tok,arr2)
    if(glob%inithessian==7) then
      neb%ene=ene_store
      call deallocate(ene_store)
    end if
    ! vhessian was set here, but the corresponding coordinates will be set
    ! here we should do a mass-re-weighting if necessary.
    if(minval(mass) > 0.D0) then
      call dlf_constants_get("AMU",svar)
      !mass=mass/svar

      tokmass=.true.
      do iat=1,glob%nat
        if(abs(mass(iat)-glob%mass(iat))>1.D-7) then
          tokmass=.false.
          if(printl>=6) &
              write(stdout,*) "Mass of atom ",iat," differs from Hessian file. File:",mass(iat)," input",glob%mass(iat)
        end if
      end do
      
      ! Re-mass-weight
      if(.not.tokmass) then
        do iimage=1,nimage_read
          call dlf_re_mass_weight_hessian(glob%nat,neb%varperimage,mass/svar,glob%mass/svar,qts%vhessian(:,:,iimage))
        end do
      end if
    end if

    call deallocate(mass)

    if(.not.tok) call dlf_fail("Error reading qts_hessian.txt")
    if(nimage_read/=neb%nimage) call dlf_fail("Wrong number of images read from qts_hessian.txt")

    glob%havehessian=.true.
    ! 
    return
    !
  end if
  

  ! updates only
  if(qts%hessian_mode==33.or.qts%hessian_mode==31) then
    if(qts%hessian_mode==31.and.qts%first_hessian) then

      ! incrementally update the Hessian starting from the middle image(s)
      do dir=-1,1,2 ! fist down, then up
        if(mod(neb%nimage,2)==0) then
          if(dir==1) startimg=neb%nimage/2+1
          if(dir==-1) startimg=neb%nimage/2
        else
          startimg=(neb%nimage+1)/2
        end if
        if(dir==1) then
          endimg=neb%nimage
        else
          endimg=1
        end if
        fromimg=startimg
        do iimage=startimg,endimg,dir
          cstart=neb%cstart(iimage)
          cend=neb%cend(iimage)
          if(printl>=4) write(stdout,*) "Updating hessian",iimage," from image",fromimg," (classical)"
          ! if only one image is read, its coordinates are written into
          ! qts%icoords_old by definepath.
          havehessian=.true.

          ! incremental:
          qts%vhessian(:,:,iimage)=qts%vhessian(:,:,fromimg)
          call dlf_hessian_update(neb%varperimage, &
              glob%icoords(neb%cstart(iimage):neb%cend(iimage)),&
              qts%icoordhess(:,fromimg),&
              qts%igradient(neb%cstart(iimage):neb%cend(iimage)), &
              qts%igradhess(:,fromimg),&
              qts%vhessian(:,:,iimage), havehessian, fracrecalc,was_updated)
!!$
!!$          ! alternative - direct:
!!$          call dlf_hessian_update(neb%varperimage, &
!!$              glob%icoords(neb%cstart(iimage):neb%cend(iimage)),&
!!$              qts%icoordhess(:,iimage),&
!!$              qts%igradient(neb%cstart(iimage):neb%cend(iimage)), &
!!$              qts%igradhess(:,iimage),&
!!$              qts%vhessian(:,:,iimage), havehessian, fracrecalc,was_updated)

          if(was_updated) then
            qts%icoordhess(:,iimage)=glob%icoords(neb%cstart(iimage):neb%cend(iimage))
            qts%igradhess(:,iimage)=qts%igradient(neb%cstart(iimage):neb%cend(iimage))
            fromimg=iimage
          end if

        end do ! iimage=startimg,endimg

      end do ! dir=-1,1,2

!!$      ! only relevant if the initial gradient is zero:
!!$      if(mod(neb%nimage,2)==1) then
!!$        iimage=(neb%nimage+1)/2
!!$        qts%icoordhess(:,iimage)=glob%icoords(neb%cstart(iimage):neb%cend(iimage))
!!$        qts%igradhess(:,iimage)=qts%igradient(neb%cstart(iimage):neb%cend(iimage))
!!$      end if

      qts%first_hessian=.false.
    else

      ! update the V-Hessians (no effect in case of first step)
      call dlf_qts_update_hessian

    end if ! (qts%hessian_mode==31.and.qts%first_hessian)

    !
    return
    !
  end if ! (qts%hessian_mode==33.or.qts%hessian_mode==31) (updates only)

  ! recalculate the hessian?
  if (qts%first_hessian) then
    if( qts%hessian_mode==20.or.qts%hessian_mode==21.or.qts%hessian_mode==30.or.qts%hessian_mode==31) &
        qts%image_status=neb%nimage/2
  else
    if( qts%hessian_mode==20.or.qts%hessian_mode==30) qts%image_status=neb%nimage/2
  end if

  if(qts%image_status==1.and..not.fd_hess_running) then
    count_calc=1
  end if

  !
  ! try to read the current Hessian from file
  !
  if(.not.fd_hess_running .and. glob%inithessian == 6) then
    status=0
    qts%ihessian_image=0.D0
    if(glob%dotask) then
      write(label,'(i6)') qts%image_status
      label="image"//trim(adjustl(label))
      iat=1 ! number of images (inout)
      call allocate(mass,glob%nat)
      cstart=neb%cstart(qts%image_status)
      cend=neb%cend(qts%image_status)
      ! read the Hessian of one image:
      call read_qts_hessian(glob%nat,iat,neb%varperimage,glob%temperature,&
          neb%ene(qts%image_status),neb%xcoords(:,qts%image_status),qts%igradient(cstart:cend),qts%ihessian_image,&
          qts%etunnel,arr2,mass,label,tok,arr2)
      call deallocate(mass)
      if(tok) status=1
    end if
    if(glob%ntasks>1) then
      if(taskfarm_mode==1) call dlf_tasks_int_sum(status, 1)
      if(status > 0 ) then ! hessian was read by first task
        ! distribute it to others - only qts%ihessian_image
        if(taskfarm_mode==1) call dlf_tasks_real_sum(qts%ihessian_image,neb%varperimage*neb%varperimage)
        if(printl>=4) write(stdout,'("Hessian of image ",i4," read from file")') qts%image_status
        if(taskfarm_mode==2.and..not.glob%dotask) qts%ihessian_image=0.D0
        !qts%ihessian_image=0.D0
        glob%havehessian = .true.
        count_calc=count_calc+neb%varperimage*2+1
      end if
    end if
    if(glob%ntasks==1.and.tok) then
      if(printl>=4) write(stdout,'("Hessian of image ",i4," read from file")') qts%image_status
      glob%havehessian = .true.
      count_calc=count_calc+neb%varperimage*2+1
    end if
  end if

  !
  ! now get the hessian either analytically or by finite difference
  !

  if (glob%inithessian == 4) then
    ! Initial Hessian is the identity matrix
    glob%ihessian = 0.0d0
    do ivar = 1, neb%varperimage*neb%nimage
      glob%ihessian(ivar, ivar) = 1.0d0
    end do
    glob%havehessian = .true.
  end if

  !
  ! try to get an analytic hessian
  !
  if (.not. glob%havehessian .and. (glob%inithessian == 0 .or. glob%inithessian == 6 ).and. &
      (glob%ntasks==1.or.taskfarm_mode==2)) then
    ! don't even attempt analytic Hessians for parallelisation within each
    ! image - it would replicate analytic Hessian calculations

    ! call to an external routine ...
    if(qts%try_analytic_hessian.and.glob%dotask) then
      call allocate(xhessian,glob%nvar,glob%nvar)
      if(glob%eonly>0) then
        if(mod(glob%eonly,2)==1) then
          call dlf_fd_energy_hessian2(glob%nvar,neb%xcoords(:,qts%image_status),&
              xhessian,status)
        else 
          call dlf_fd_energy_hessian4(glob%nvar,neb%xcoords(:,qts%image_status),&
              xhessian,status)
        end if
      else
        if(printl>=4) write(stdout,"('Calculating analytic Hessian for image ',i4)") &
            qts%image_status 
        call dlf_get_hessian(glob%nvar,neb%xcoords(:,qts%image_status),xhessian,status)
      end if
    else
      status=1
    end if

    if(status==0) then
      
      ! quite likely, the gradient is wrong here. (does seem to be right ...)
      if (tstore) call dlf_store_egh(glob%nvar,xhessian) 
      ! store gradient after Hessian calculation
      iimage=qts%image_status 
      call dlf_direct_xtoi(glob%nvar,neb%varperimage,neb%coreperimage,&
            neb%xcoords(:,iimage),glob%xgradient,&
            glob%icoords(neb%cstart(iimage):neb%cend(iimage)), &
            qts%igradient(neb%cstart(iimage):neb%cend(iimage)))

      call dlf_cartesian_hessian_xtoi(glob%nat,glob%nvar,neb%varperimage,glob%massweight,&
          xhessian,glob%spec,glob%mass,&
          qts%ihessian_image)
      call deallocate(xhessian)
      !     write(*,"('iHESS',8F10.4)") glob%ihessian
      !        call clock_stop("COORDS")
      glob%havehessian=.true.

    else

      if(qts%try_analytic_hessian.and.printl>=2.and.glob%dotask) write(stdout,'(a)') &
          "External Hessian not available, using two point FD."
      glob%havehessian=.false.
      if(glob%dotask) qts%try_analytic_hessian=.false.

    end if
  end if ! (.not. glob%havehessian .and. glob%inithessian == 0...)

  if(allocated(xhessian)) call deallocate(xhessian)

  cstart=neb%cstart(qts%image_status)
  cend=neb%cend(qts%image_status)

  if(.not. glob%havehessian) then   

    if (glob%inithessian==0.or.glob%inithessian==2.or.glob%inithessian==6) then

      ! after first energy evaluation, the energy and gradient obtained have
      ! to be communicated to the tasks:
      if(.not.fd_hess_running.and.glob%ntasks>0.and.taskfarm_mode==1) then
        call dlf_tasks_real_sum(glob%xgradient, glob%nvar)
        tmp(1) = glob%energy
        call dlf_tasks_real_sum(tmp(1), 1)
      end if

      ! reduce print level in dlf_fdhessian
      printl_store=printl
      printl=min(printl,3)

      ! Hessian in internals
      ! Finite Difference Hessian calculation in internal coordinates
      call dlf_direct_xtoi(glob%nvar,neb%varperimage,neb%coreperimage,&
           glob%xcoords,glob%xgradient, &
          glob%icoords(cstart:cend),glob%igradient(cstart:cend))
      if(.not.fd_hess_running) qts%igradient(cstart:cend)=glob%igradient(cstart:cend)
      call dlf_fdhessian(neb%varperimage,.false.,glob%energy,&
          glob%icoords(cstart:cend), &
          glob%igradient(cstart:cend),qts%ihessian_image,glob%havehessian)

!!$      ! Alternative: finite-difference Hessian in cartesians
!!$      call dlf_fdhessian(glob%nvar,.false.,glob%energy,&
!!$          glob%xcoords, &
!!$          glob%xgradient,qts%xhessian_image,glob%havehessian)

      ! if Hessian is finished, store it
      if(tstore.and.(.not.fd_hess_running)) then
        ! hessian can only be written if it is calculated in the full
        ! coordinate set 
        ! we have to remove mass-weighting from hessian and have to use
        ! glob%igradient to get the midpoint gradient (from which
        ! mass-weighting has to be removed as well)
        call allocate(xhessian,glob%nvar,glob%nvar)
        call dlf_cartesian_hessian_itox(glob%nat,glob%nvar,neb%varperimage,glob%massweight,&
            qts%ihessian_image,glob%spec,glob%mass,xhessian)
        call dlf_cartesian_itox(glob%nat,neb%varperimage,neb%varperimage,glob%massweight,&
            glob%icoords(cstart:cend),glob%xcoords)
        call dlf_cartesian_gradient_itox(glob%nat,neb%varperimage,neb%varperimage,glob%massweight,&
            glob%igradient(cstart:cend),glob%xgradient)
        call dlf_store_egh(glob%nvar,xhessian)
        call deallocate(xhessian)
        
      end if


      ! print here rather than in dlf_fdhessian
      if(printl_store>=4.and.glob%dotask) then
        write(stdout,"('Finite-difference Hessian: calculation ',i6,&
            &' of ',i6,', image ',i4)") &
            count_calc,neb%nimage*(neb%varperimage*2+1),qts%image_status
      end if
      count_calc=count_calc+1

      printl=printl_store

!!$         ! Finite Difference Hessian calculation in internal coordinates
!!$         ! dlf_fdhessian writes fd_hess_running
!!$         call dlf_fdhessian(glob%nivar,fracrec,glob%energy,glob%icoords, &
!!$              glob%igradient,glob%ihessian,glob%havehessian)
    else
      call dlf_fail("Instanton rate calculation only possible for&
          & inithessian 0, 2, 6, or 7")
    end if ! (glob%inithessian==0.or.glob%inithessian==2) 


    ! check if FD-Hessian calculation currently running
    trerun_energy=(fd_hess_running) 
    if(trerun_energy) then
      !       call clock_start("COORDS")

      ! Hessian in internals
      call dlf_direct_itox(glob%nvar,neb%varperimage,neb%coreperimage, &
          glob%icoords(cstart:cend),glob%xcoords,tok)

      !       call dlf_coords_itox(qts%image_status)
      !       call clock_stop("COORDS")
!      call deallocate(hess_image)
      return

    end if


  end if

!!$  ! transform xhessian_image into internal coordinates
!!$  call dlf_cartesian_hessian_xtoi(glob%nvar,neb%varperimage,glob%massweight,&
!!$      qts%xhessian_image,qts%ihessian_image)

  !write(*,*) qts%image_status, 'HESS_GLOB PER IMAGE'!, qts%ihessian_image
  !do ivar=1,neb%varperimage
  !  write(*,"(i5,10f10.5)") ivar,qts%ihessian_image(1:min(10,neb%varperimage),ivar)
  !end do

  ! Write data of the hessian of the current image to the global hessian
  qts%vhessian(:,:,qts%image_status)=qts%ihessian_image

  ! Write Hessian of the current image to a file
  ! for taskfarm_mode=2 only the first task can write something at this stage ...
  if(taskfarm_mode==1.or.glob%dotask) then
    write(label,'(i6)') qts%image_status
    label="image"//trim(adjustl(label))
    cstart=neb%cstart(qts%image_status)
    cend=neb%cend(qts%image_status)
    call write_qts_hessian(glob%nat,1,neb%varperimage,glob%temperature,&
        neb%ene(qts%image_status),neb%xcoords(:,qts%image_status),qts%igradient(cstart:cend),qts%vhessian(:,:,qts%image_status),&
        qts%etunnel,qts%dist(1:2),label)
  end if

  ! prepare for next hessian
  glob%havehessian=.false.
  qts%image_status=qts%image_status+1
  if(qts%image_status<=neb%nimage) then
    glob%xcoords=reshape(neb%xcoords(:,qts%image_status),(/3,glob%nat/))
  else
    ! now all Hessians of all images should be calculated. For taskfarm_mode=2
    ! they have to be combined now.
    ! should I introduce some error-checking at this point?
    if(taskfarm_mode==2) call dlf_tasks_real_sum(qts%vhessian, &
        neb%varperimage*neb%varperimage*neb%nimage)
  end if


!  call deallocate(hess_image)

  if(qts%image_status<=neb%nimage) then
    ! recalculate the energy and gradient at the undistorted
    ! position of image qts%image_status. For the two-point formula
    ! this is strictly only needed for printout (differences to
    ! midpoint). The energy difference to midpoint could also be
    ! calculated as difference to the value obtained from
    ! qts_coords.txt

    ! for analytic Hessians this is required (depending on the interface)
    ! because a lot of QM programs require an energy calculation prior to a
    ! hessian calculation

    trerun_energy=.true.
    return
  end if

!!$  ! write binary Hessian - tmp for test of updates:
!!$  open(unit=101,file="hessian.bin",form="unformatted")
!!$  ! Hessian in mass-weighted coordinates on the diagonal blocks - everything else should be zero
!!$  write(101) neb%varperimage,neb%nimage
!!$  write(101) qts%total_hessian/qts%const_hess
!!$  close(101)
!!$  ! <<<<<<<<<<<<<<


end subroutine dlf_qts_get_hessian
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/dlf_qts_update_hessian
!!
!! FUNCTION
!!
!! Update the qTS hessian. On input qts%total_hessian contains in the diagonal
!! blocks the V-Hessian multiplied by qts%const_hess, qts%igradient contains
!! the gradient of the potential.  Update the hessian blocks individually
!!
!! INPUTS
!! 
!! qts%vhessian, qts%igradient
!!
!! OUTPUTS
!! 
!! qts%vhessian
!!
!! SYNOPSIS
subroutine dlf_qts_update_hessian
!! SOURCE
  use dlf_neb, only: neb
  use dlf_global, only: glob,printl,stdout
  use dlf_allocate, only: allocate,deallocate
  use dlf_qts
  implicit none
  integer   :: iimage,cstart,cend
  !real(rk)  :: hess_image(neb%varperimage,neb%varperimage)
!  real(rk),allocatable  :: hess_image(:,:) ! (neb%varperimage,neb%varperimage)
  logical   :: havehessian,fracrecalc,was_updated
  integer   :: nvar,status
  real(rk),allocatable :: xhessian(:,:) 
  fracrecalc=.false.

  nvar=neb%nimage*neb%varperimage*2

!  call allocate(hess_image,neb%varperimage,neb%varperimage)

  do iimage=1,neb%nimage
    ! update rather than copy
    if(printl>=4) write(stdout,*) "updating hessian ",iimage," from last step"

    !hess_image will be the updated hessian
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage)
!    hess_image=qts%total_hessian(cstart:cend,cstart:cend)/qts%const_hess
    havehessian=.true.
!!$    print*,"UPDATE"
!!$    print*,"coords    ",glob%icoords(     cstart:cend)
!!$    print*,"coords old",qts%icoords_old(  cstart:cend)
!!$    print*,"gradient  ",qts%igradient(    cstart:cend)
!!$    print*,"grad   old",qts%igradient_old(cstart:cend)
!!$    print*,"Hess before update",hess_image
!!$    call dlf_hessian_update(neb%varperimage, &
!!$        glob%icoords(     cstart:cend),&
!!$        qts%icoords_old(  cstart:cend),&
!!$        qts%igradient(    cstart:cend), &
!!$        qts%igradient_old(cstart:cend), &
!!$        hess_image, havehessian, fracrecalc)

!!$    if(iimage==2) then
!!$    print*,"image:",iimage
!!$      print*,"coords",glob%icoords(cstart:cend)
!!$      print*,"old coords",qts%icoordhess(:,iimage)
!!$      print*,"grad",qts%igradient(cstart:cend)
!!$      print*,"old grad",qts%igradhess(:,iimage)
!!$      print*,"hess before",qts%vhessian(:,:,iimage)
!!$    end if
    
    ! hack to recalculate rather than update:
    if(1==2) then ! 1==2 means: not not use, 1==1 means: always use
      if(printl>=4) write(stdout,"('Calculating analytic Hessian for image ',i4)") &
          iimage
      call allocate(xhessian,glob%nvar,glob%nvar)
      call dlf_get_hessian(glob%nvar,neb%xcoords(:,iimage),xhessian,status)
      if(status/=0) call dlf_fail("Analytic Hessian not available, but required here")
      call dlf_cartesian_hessian_xtoi(glob%nat,glob%nvar,neb%varperimage,glob%massweight,&
          xhessian,glob%spec,glob%mass,&
          qts%vhessian(:,:,iimage))
      call deallocate(xhessian)
    else
      ! this is the default
      call dlf_hessian_update(neb%varperimage, &
          glob%icoords(cstart:cend),&
          qts%icoordhess(:,iimage),&
          qts%igradient(cstart:cend), &
          qts%igradhess(:,iimage), &
          qts%vhessian(:,:,iimage), havehessian, fracrecalc,was_updated)
    end if

!!$    if(iimage==2) then
!!$      print*,"hess after",qts%vhessian(:,:,iimage)
!!$    end if

    if(was_updated) then
      qts%icoordhess(:,iimage)=glob%icoords(cstart:cend)
      qts%igradhess(:,iimage)=qts%igradient(cstart:cend)
    end if


!!$    print*,"Hess after  update",hess_image

    !print*,"Hessian update ok?",havehessian
!!$    if(havehessian) then
!!$      qts%total_hessian(cstart:cend,cstart:cend)=hess_image*qts%const_hess
!!$      ! duplication (short) of the Hessian 
!!$      qts%total_hessian(&
!!$          nvar-neb%cend(iimage)+1:nvar-neb%cstart(iimage)+1,&
!!$          nvar-neb%cend(iimage)+1:nvar-neb%cstart(iimage)+1)=hess_image*qts%const_hess
!!$    end if

  end do
!!$  print*,"qts%total_hessian after update"
!!$  write(*,'(20f8.4)') qts%total_hessian
!  call deallocate(hess_image)

end subroutine dlf_qts_update_hessian
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/qts_hessian_etos_halfpath
!!
!! FUNCTION
!!
!! Convert the V-Hessian to an S-Hessian of the path with images pairwise
!! collapsed. The output of this routine is needed in optimisation, but not in
!! rate calculations
!!
!! INPUTS
!! 
!! qts%vhessian
!!
!! OUTPUTS
!! 
!! glob%ihessian
!!
!! SYNOPSIS
subroutine qts_hessian_etos_halfpath
!! SOURCE
  use dlf_global, only: glob
  use dlf_neb, only: neb,beta_hbar
  use dlf_qts
  implicit none
  integer :: iimage,posi,posj,ivar,jvar,cstart,cend

  qts%first_hessian=.false.

  !qts%total_hessian contains the potential hessian multiplied by qts%const_hess
  !glob%ihessian=qts%total_hessian(1:glob%nivar,1:glob%nivar) / qts%const_hess
  do iimage=1,neb%nimage
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage)
    glob%ihessian(cstart:cend,cstart:cend)=qts%vhessian(:,:,iimage)
  end do
  ! now, glob%ihessian contains only the potential hessian

  ! convert the potential part
  do iimage=1,neb%nimage
    do ivar=1,neb%varperimage
      do jvar=1,neb%varperimage
        posi=(iimage-1)*neb%varperimage+ivar
        posj=(iimage-1)*neb%varperimage+jvar
        glob%ihessian(posi,posj)=glob%ihessian(posi,posj) * ( qts%dtau(iimage) + qts%dtau(iimage+1) )
      end do
    end do
  end do

  ! add the "kinetic" part
  do iimage=1,neb%nimage
    do ivar=1,neb%varperimage
      posi=(iimage-1)*neb%varperimage+ivar

      if(iimage==1) then
        glob%ihessian(posi,posi)=glob%ihessian(posi,posi)+2.D0*(1.D0/qts%dtau(iimage+1))
      else if(iimage==neb%nimage) then
        glob%ihessian(posi,posi)=glob%ihessian(posi,posi)+2.D0*(1.D0/qts%dtau(iimage))
      else
        glob%ihessian(posi,posi)=glob%ihessian(posi,posi)+2.D0*(1.D0/qts%dtau(iimage)+1.D0/qts%dtau(iimage+1))
      end if

      if(posi+neb%varperimage <=glob%nivar) then
        glob%ihessian(posi+neb%varperimage,posi)=-2.D0/qts%dtau(iimage+1)
        glob%ihessian(posi,posi+neb%varperimage)=-2.D0/qts%dtau(iimage+1)
      end if
    end do
  end do

!!$  ! test symmetry
!!$  svar=0.D0
!!$  do ivar=1,glob%nivar
!!$    do jvar=ivar+1,glob%nivar
!!$      svar=max(svar,abs(glob%ihessian(ivar,jvar)-glob%ihessian(jvar,ivar)))
!!$      if(abs(glob%ihessian(ivar,jvar)-glob%ihessian(jvar,ivar))>1.D-10) &
!!$          print*,ivar,jvar,abs(glob%ihessian(ivar,jvar)-glob%ihessian(jvar,ivar))
!!$    end do
!!$  end do
!!$  print*,"max non-symmetry:",svar
!!$  ! -> the thing is symmetric!

  glob%ihessian=glob%ihessian/beta_hbar ! same scaling as gradient

  glob%havehessian=.true.

  ! write that Hessian to a file
  call write_qts_hessian(glob%nat,neb%nimage,neb%varperimage,glob%temperature,&
      neb%ene,neb%xcoords,qts%igradient,qts%vhessian,qts%etunnel,qts%dist,"intermediate")

end subroutine qts_hessian_etos_halfpath
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/qts_hessian_etos
!!
!! FUNCTION
!!
!! Convert the V-Hessian to an S-Hessian (full path) for rate
!! calculations. The resulting S-Hessian is symmetric.
!!
!! INPUTS
!! 
!! total_hessian (as dummy variable)
!!
!! OUTPUTS
!! 
!! total_hessian (as dummy variable)
!!
!! SYNOPSIS
subroutine qts_hessian_etos(task,nimage,varperimage,dtau,total_hessian)
!! SOURCE
  use dlf_parameter_module, only: rk
  !  use dlf_neb, only: beta_hbar
  use dlf_qts, only: hess_dtau
  implicit none
  integer,intent(in) :: task
  ! task=1 return a Hessian with the potential part as d2E/dy2*dtau
  ! task=2 return a Hessian with the potential part as d2E/dy2
  ! task=3 just convert the d2E/dy2*dtau-Hessian to the d2E/dy2 one
  integer,intent(in) :: nimage
  integer,intent(in) :: varperimage
  real(rk),intent(in) :: dtau(nimage+1)
  real(rk),intent(inout) :: total_hessian(nimage*varperimage*2,nimage*varperimage*2)
  !
  integer    :: ivar,nvar,iimage,jvar,posi,posj

  nvar=varperimage*nimage*2

  if(task==1.or.task==2) then
    ! convert the potential part
    do iimage=1,nimage
      do ivar=1,varperimage
        do jvar=1,varperimage
          ! first half
          posi=(iimage-1)*varperimage+ivar
          posj=(iimage-1)*varperimage+jvar
          total_hessian(posi,posj)=total_hessian(posi,posj) * 0.5D0*( dtau(iimage) + dtau(iimage+1) )
          ! second half
          posi=nimage*varperimage + (nimage - iimage)*varperimage+ivar
          posj=nimage*varperimage + (nimage - iimage)*varperimage+jvar
          total_hessian(posi,posj)=total_hessian(posi,posj) * 0.5D0*( dtau(iimage) + dtau(iimage+1) )
        end do
      end do
    end do

    ! add the "kinetic" part
    do iimage=1,nimage
      do ivar=1,varperimage
        posi=(iimage-1)*varperimage+ivar
        ! posj here is the position of ivar transformed to the second half of the path
        posj=nimage*varperimage + (nimage - iimage)*varperimage+ivar

        ! general case - no need to handle the ends differently
        total_hessian(posi,posi)=total_hessian(posi,posi)+1.D0*(1.D0/dtau(iimage)+1.D0/dtau(iimage+1))
        total_hessian(posj,posj)=total_hessian(posj,posj)+1.D0*(1.D0/dtau(iimage)+1.D0/dtau(iimage+1))

        ! posi is always < nvar (=2*nimage*varperimage)
        total_hessian(posi+varperimage,posi)=-1.D0/dtau(iimage+1)
        total_hessian(posi,posi+varperimage)=-1.D0/dtau(iimage+1)
        if(posj+varperimage <=nvar) then
          total_hessian(posj+varperimage,posj)=-1.D0/dtau(iimage)
          total_hessian(posj,posj+varperimage)=-1.D0/dtau(iimage)
        else
          total_hessian(ivar,posj)=-1.D0/dtau(iimage)
          total_hessian(posj,ivar)=-1.D0/dtau(iimage)
        end if
      end do
    end do
  end if !(task==1.or.task==2)

  if(task==2.or.task==3) then
    if(.not.hess_dtau) then
      ! convert everything back to a non-converted potential part (symmetric)
      do iimage=1,nimage
        do ivar=1,varperimage
          ! first half
          posi=(iimage-1)*varperimage+ivar
          total_hessian(posi,:)=total_hessian(posi,:) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
          ! second half
          posi=nimage*varperimage + (nimage - iimage)*varperimage+ivar
          total_hessian(posi,:)=total_hessian(posi,:) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
        end do
      end do
      do iimage=1,nimage
        do ivar=1,varperimage
          ! first half
          posi=(iimage-1)*varperimage+ivar
          total_hessian(:,posi)=total_hessian(:,posi) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
          ! second half
          posi=nimage*varperimage + (nimage - iimage)*varperimage+ivar
          total_hessian(:,posi)=total_hessian(:,posi) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
        end do
      end do

!!$      ! now test if symmetric!
!!$      print*,"testing asym"
!!$      do iimage=1,nimage*2*varperimage
!!$        do ivar=1,nimage*2*varperimage
!!$          svar=abs(total_hessian(iimage,ivar)-total_hessian(ivar,iimage))
!!$          if(svar>1.D-10) print*,"asym",svar,iimage,ivar
!!$        end do
!!$      end do

    end if ! (.not.hess_dtau)
  end if

end subroutine qts_hessian_etos
!!****


subroutine qts_det_tsplitting(nimage,varperimage,dtau_,total_hessian,det_tsplit)
!! SOURCE
  use dlf_parameter_module, only: rk
  !  use dlf_neb, only: beta_hbar
  use dlf_qts, only: hess_dtau
  use dlf_neb, only: beta_hbar,neb
  use dlf_global, only: glob,stdout,printl
  use dlf_allocate, only: allocate,deallocate
  implicit none
  integer,intent(in) :: nimage
  integer,intent(in) :: varperimage
  real(rk),intent(in) :: dtau_(nimage+1)
  real(rk),intent(in) :: total_hessian(nimage*varperimage*2,nimage*varperimage*2)
  real(rk),intent(out) :: det_tsplit ! actually, the square root of the determinant with one eigenvalue ignored
  !
  integer    :: ivar,nvar,iimage,jvar,posi,posj,ival
  real(rk) :: dtau(nimage+1)
  real(rk) :: grad_rs(neb%varperimage)
  real(rk) :: hess_rs(neb%varperimage,neb%varperimage)
  real(rk) :: arr2(2),svar,mass(glob%nat), svar_tmp(1) ! scratch
!  real(rk) :: evals_rs( neb%varperimage * neb%nimage )
  logical :: tok

  ! local matrix
  real(rk),allocatable :: hessian(:,:) ! (nimage*varperimage,nimage*varperimage)
  real(rk),allocatable :: evals_hess(:)   ! (neb%varperimage*neb%nimage)
  real(rk),allocatable :: evecs_hess(:,:) ! (neb%varperimage*neb%nimage,neb%varperimage*neb%nimage)

  call allocate(hessian,nimage*varperimage,nimage*varperimage)
  hessian=0.D0
  dtau=dtau_
  nvar=varperimage*nimage
  dtau(:)=beta_hbar/dble(2*nimage) ! The factor 2 seems to belong here!

    ! convert the potential part
    do iimage=1,nimage
      do ivar=1,varperimage
        do jvar=1,varperimage
          ! first half
          posi=(iimage-1)*varperimage+ivar
          posj=(iimage-1)*varperimage+jvar
          hessian(posi,posj)=total_hessian(posi,posj) * 0.5D0*( dtau(iimage) + dtau(iimage+1) )
        end do
      end do
    end do

    ! add the "kinetic" part
    do iimage=1,nimage
      do ivar=1,varperimage
        posi=(iimage-1)*varperimage+ivar
        ! posj here is the position of ivar transformed to the second half of the path
        posj=nimage*varperimage + (nimage - iimage)*varperimage+ivar

        if(iimage==1) then
          hessian(posi,posi)=hessian(posi,posi)+1.D0*(1.D0/dtau(iimage+1))
        else if(iimage==nimage) then
          hessian(posi,posi)=hessian(posi,posi)+1.D0*(1.D0/dtau(iimage))
        else 
          ! general case - no need to handle the ends differently
          hessian(posi,posi)=hessian(posi,posi)+1.D0*(1.D0/dtau(iimage)+1.D0/dtau(iimage+1))
        end if

!        total_hessian(posj,posj)=total_hessian(posj,posj)+1.D0*(1.D0/dtau(iimage)+1.D0/dtau(iimage+1))


        if(posi+varperimage <=nvar) then
          hessian(posi+varperimage,posi)=-1.D0/dtau(iimage+1)
          hessian(posi,posi+varperimage)=-1.D0/dtau(iimage+1)
        end if

        ! no closure of the path

      end do
    end do

    if(.not.hess_dtau) then
      ! convert everything back to a non-converted potential part (symmetric)
      do iimage=1,nimage
        do ivar=1,varperimage
          ! first half
          posi=(iimage-1)*varperimage+ivar
          hessian(posi,:)=hessian(posi,:) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
!          ! second half
!          posi=nimage*varperimage + (nimage - iimage)*varperimage+ivar
!          total_hessian(posi,:)=total_hessian(posi,:) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
        end do
      end do
      do iimage=1,nimage
        do ivar=1,varperimage
          ! first half
          posi=(iimage-1)*varperimage+ivar
          hessian(:,posi)=hessian(:,posi) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
!          ! second half
!          posi=nimage*varperimage + (nimage - iimage)*varperimage+ivar
!          total_hessian(:,posi)=total_hessian(:,posi) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
        end do
      end do

!!$      ! now test if symmetric!
!!$      print*,"testing asym"
!!$      do iimage=1,nimage*2*varperimage
!!$        do ivar=1,nimage*2*varperimage
!!$          svar=abs(total_hessian(iimage,ivar)-total_hessian(ivar,iimage))
!!$          if(svar>1.D-10) print*,"asym",svar,iimage,ivar
!!$        end do
!!$      end do

    end if ! (.not.hess_dtau)

  ! now we have the Hessian of the linear chain between the two minima

  ! diagonalise it
  call allocate(evals_hess, nvar)
  call allocate(evecs_hess, nvar,nvar)

  call dlf_matrix_diagonalise(nvar,hessian,evals_hess,evecs_hess)

  if(printl>=4) then
    write(stdout,*) "Eigenvalues of the linear chain for T-splittings,", glob%nzero+1," are ignored"
    do ival=1,min(15,nvar)
      write(stdout,"(i6,1x,es18.9)") &
          ival,evals_hess(ival)
    end do
  end if

  det_tsplit=0.5D0* sum(log(abs(evals_hess(glob%nzero+2:nvar)))) 

! now we also have to calculate the same for the reactant!

  !
  ! calculate the partition function from a full diagonalisation of the whole Hessian
  !
  ! get the whole hessian of the reactant
  ivar=1 ! nimage
  call read_qts_hessian(glob%nat,ivar,neb%varperimage,svar,&
      svar_tmp,glob%xcoords,grad_rs,hess_rs,svar,arr2,mass,"rs",tok,arr2)
  svar=svar_tmp(1)
  if(.not.tok) then
    if(printl>=2) write(stdout,*) "Warning: full reactant state Hessian not available (qts_hessian_rs.txt)"
    det_tsplit=0.D0
    return
  end if
  ! write it repeatedly into total_hessian
  hessian=0.D0
  ivar=neb%nimage*neb%varperimage
  do iimage=1,neb%nimage
    hessian(neb%cstart(iimage):neb%cend(iimage),neb%cstart(iimage):neb%cend(iimage)) = &
        hess_rs
  end do


  ! convert the Hessian to Action - copied from above >>>>>>>>>>
    ! convert the potential part
    do iimage=1,nimage
      do ivar=1,varperimage
        do jvar=1,varperimage
          ! first half
          posi=(iimage-1)*varperimage+ivar
          posj=(iimage-1)*varperimage+jvar
          ! the following line is changed compared to above:
          hessian(posi,posj)=hessian(posi,posj) * 0.5D0*( dtau(iimage) + dtau(iimage+1) )
        end do
      end do
    end do

    ! add the "kinetic" part
    do iimage=1,nimage
      do ivar=1,varperimage
        posi=(iimage-1)*varperimage+ivar
        ! posj here is the position of ivar transformed to the second half of the path
        posj=nimage*varperimage + (nimage - iimage)*varperimage+ivar

        if(iimage==1) then
          hessian(posi,posi)=hessian(posi,posi)+1.D0*(1.D0/dtau(iimage+1))
        else if(iimage==nimage) then
          hessian(posi,posi)=hessian(posi,posi)+1.D0*(1.D0/dtau(iimage))
        else 
          ! general case - no need to handle the ends differently
          hessian(posi,posi)=hessian(posi,posi)+1.D0*(1.D0/dtau(iimage)+1.D0/dtau(iimage+1))
        end if


!!$        ! general case - no need to handle the ends differently
!!$        hessian(posi,posi)=hessian(posi,posi)+1.D0*(1.D0/dtau(iimage)+1.D0/dtau(iimage+1))
!        total_hessian(posj,posj)=total_hessian(posj,posj)+1.D0*(1.D0/dtau(iimage)+1.D0/dtau(iimage+1))


        if(posi+varperimage <=nvar) then
          hessian(posi+varperimage,posi)=-1.D0/dtau(iimage+1)
          hessian(posi,posi+varperimage)=-1.D0/dtau(iimage+1)
        end if

        ! no closure of the path

      end do
    end do

    if(.not.hess_dtau) then
      ! convert everything back to a non-converted potential part (symmetric)
      do iimage=1,nimage
        do ivar=1,varperimage
          ! first half
          posi=(iimage-1)*varperimage+ivar
          hessian(posi,:)=hessian(posi,:) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
!          ! second half
!          posi=nimage*varperimage + (nimage - iimage)*varperimage+ivar
!          total_hessian(posi,:)=total_hessian(posi,:) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
        end do
      end do
      do iimage=1,nimage
        do ivar=1,varperimage
          ! first half
          posi=(iimage-1)*varperimage+ivar
          hessian(:,posi)=hessian(:,posi) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
!          ! second half
!          posi=nimage*varperimage + (nimage - iimage)*varperimage+ivar
!          total_hessian(:,posi)=total_hessian(:,posi) /sqrt( 0.5D0*( dtau(iimage) + dtau(iimage+1) ))
        end do
      end do

!!$      ! now test if symmetric!
!!$      print*,"testing asym"
!!$      do iimage=1,nimage*2*varperimage
!!$        do ivar=1,nimage*2*varperimage
!!$          svar=abs(total_hessian(iimage,ivar)-total_hessian(ivar,iimage))
!!$          if(svar>1.D-10) print*,"asym",svar,iimage,ivar
!!$        end do
!!$      end do

    end if ! (.not.hess_dtau)
 ! <<<<<<<<< copied from above


  call dlf_matrix_diagonalise(nvar,hessian,evals_hess,evecs_hess)

  if(printl>=4) then
    write(stdout,*) "Eigenvalues of the RS linear chain for T-splittings,", glob%nzero," are ignored"
    do ival=1,min(15,nvar)
      write(stdout,"(i6,1x,es18.9)") &
          ival,evals_hess(ival)
    end do
  end if

  det_tsplit=det_tsplit-0.5D0* sum(log(abs(evals_hess(glob%nzero+1:nvar)))) 
  if(printl>=4) write(stdout,*) "Determinant for Tunnelling splitting:",det_tsplit,exp(det_tsplit)

  ! <<<< end of reactant

  call deallocate(evals_hess)
  call deallocate(evecs_hess)
  call deallocate(hessian)

end subroutine qts_det_tsplitting
!!****

subroutine report_qts_pathlength
  use dlf_parameter_module, only: rk
  use dlf_neb, only: neb,beta_hbar
  use dlf_global, only: glob,stdout,printl,printf,pi
  use dlf_constants, only : dlf_constants_get
  use dlf_qts
  implicit none
  real(rk) :: plength(glob%nat)
  integer  :: iat,nat,iimage
  real(rk) :: ang_au
  !
  if(printl<4) return

  nat=glob%nat
  plength(:)=0.D0
  do iat=1,nat
    do iimage=2,neb%nimage
      plength(iat)=plength(iat) + sqrt(sum((neb%xcoords(iat*3-2:iat*3,iimage)&
          -neb%xcoords(iat*3-2:iat*3,iimage-1))**2))
    end do
  end do
  call dlf_constants_get("ANG_AU",ang_au)
  write(stdout,'(a)') "Atom   length of instanton path (Bohr and Ang)"
  do iat=1,nat
    if(plength(iat)<1.D-6) cycle
    write(stdout,'(i6,2f10.5)') iat,plength(iat),plength(iat)*ang_au
  end do
end subroutine report_qts_pathlength

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/dlf_qts_rate
!!
!! FUNCTION
!!
!! Calculate the instanton rate constant. Calls
!! write_qts_hessian to write qts_hessian.txt.
!!
!! INPUTS
!! 
!! qts%vhessian, qts%S_0, qts%S_pot
!!
!! OUTPUTS
!! 
!! printout
!!
!! SYNOPSIS
subroutine dlf_qts_rate()
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_neb, only: neb,unitp,xyzall,beta_hbar
  use dlf_global, only: glob,stdout,printl,printf,pi
  use dlf_hessian
  use dlf_allocate, only: allocate,deallocate
  use dlf_constants, only : dlf_constants_get
  use dlf_qts
  implicit none
  real(rk),allocatable :: total_hessian(:,:) ! (neb%varperimage*neb%nimage*2,neb%varperimage*neb%nimage*2)
  real(rk),allocatable :: evals_hess(:)   ! (neb%varperimage*neb%nimage*2)
  real(rk),allocatable :: evecs_hess(:,:) ! (neb%varperimage*neb%nimage*2,neb%varperimage*neb%nimage*2)
  real(rk),allocatable :: xmode(:,:) ! (3,glob%nat)
  real(rk),allocatable :: vec0(:) ! (neb%varperimage*neb%nimage*2)
  real(rk),allocatable :: zpe(:) ! (neb%nimage)
  real(rk),allocatable :: evals_image(:) ! (neb%varperimage)
  real(rk),allocatable :: evecs_image(:,:) ! (neb%varperimage,neb%varperimage)
  real(rk)   :: prod_evals_hess_prime,ers,qrs
  real(rk)   :: prod_sigma
  integer    :: ivar,ival,nvar,iimage,nimage,jvar,cstart,cend
  logical    :: tok
  real(rk)   :: second_au
  character(2) :: chr2
  character(50) :: filename
  real(rk)   :: cm_inv_for_amu,amu,svar,svar2,svar3
  logical    :: tbimol
  integer    :: primage,neimage
  real(rk)   :: det_tsplit
  character(20) :: label
  real(rk)   :: vb,alpha,kappa_eck
  real(rk)   :: eff_mass,norm,norm_total,eff_mass_total
  integer    :: icomp,iat,nzero
  real(rk)   :: qrot,qrot_part,zpers,moi(3)
  logical    :: tflux=.false.
  real(rk)   :: corrfac,qrot_quant,qrot_quant_part,coeff(2)
  real(rk)  :: qrot_store,projtr
  real(rk),allocatable :: mass_all(:) ! nat*nimage
  real(rk),allocatable :: trmodes(:,:),trmodes2(:,:)
  integer :: jimage,info
  ! variables for Andreas' alternative way to calculate the rate
  real(rk) :: fluc_rs_ana,fluc_rs,mass_bimol,pe
  real(rk) :: fluc_inst,phi_rel,q_rot_trans,rate_const
  complex(rk), allocatable :: u_param(:), u_param2(:)
  integer :: varperimage_read,nvib
  character(5000) :: line
  real :: time2,time3
  real(rk),allocatable :: omega2(:) ! RS hessian eigenvalues
  !**********************************************************************
  !write(*,*) 'Total hessian', qts%total_hessian

  ! this should only be done by the master process. If the matrix
  ! diagonalisation can be parallelised (which is not the case at the moment),
  ! the parallel part should continue. But for the time being slaves running
  ! here would only waste CPU time.
  if(glob%iam > 0) then
    if(printl>=4) then
      write(stdout,'("Process ",i4," returning. Rate is calculated by master")') glob%iam
    end if
    return
  end if

  call clock_start("COORDS")

  nvar=neb%varperimage*neb%nimage*2
  nimage=neb%nimage
  prod_evals_hess_prime=1.D0
  ! write the hessian to disk
  if (glob%inithessian/=5) then
    if(glob%iopt==12) then
      label=""
    else
      label="upd"
    end if
    call write_qts_hessian(glob%nat,neb%nimage,neb%varperimage,glob%temperature,&
        neb%ene,neb%xcoords,qts%igradient,qts%vhessian,qts%etunnel,qts%dist,label)
  end if

  if(printl>=4) then
    call report_qts_pathlength()
  end if

  !print*,"jk total hessian",qts%total_hessian
  call allocate(omega2,neb%varperimage)

  call qts_reactant(qrs,ers,zpers,qrot_part,qrot_quant,tbimol,tok, phi_rel,&
      neb%varperimage,omega2,nvib)
  qrot=-qrot_part
  qrot_quant=-qrot_quant
  if(.not.tok) then
    ers=0.D0
  end if

!!$  !check if Hessian is duplicated correctly:
!!$        !print*,"Hessian update ok?",havehessian
!!$  do iimage=1,neb%nimage
!!$    do ivar = 1,neb%varperimage
!!$      do jvar =  1,neb%varperimage
!!$        svar=abs(qts%total_hessian(ivar+(iimage-1)*neb%varperimage,&
!!$            jvar+(iimage-1)*neb%varperimage) - (&
!!$            qts%total_hessian(neb%varperimage*neb%nimage*2+&
!!$                ivar-iimage*neb%varperimage,&
!!$                neb%nimage*neb%varperimage*2+jvar-iimage*neb%varperimage)&
!!$               ))
!!$        if(svar>1.D-10) print*,"Hessian not duplicated correctly",iimage,ivar,jvar,svar
!!$      end do
!!$    end do
!!$  end do
!!$  print*,"duplication checked"

  ! calculate the rotational partition function of the instanton 
  if(glob%irot==0) then
    qrot_store=qrot

!!$    ! calculate the rotational partition function as geometric average of those of the individual images:
!!$    svar=sum(qts%dtau(1:nimage+1))-0.5D0*(qts%dtau(1)+qts%dtau(nimage+1))
!!$    do iimage=1,nimage
!!$      nzero=glob%nzero
!!$      call rotational_partition_function(glob%nat,glob%mass,nzero,neb%xcoords(:,iimage),beta_hbar,qrot_part,moi)
!!$      if(nzero/=glob%nzero) print*,"Warning: nzero is likely to be wrong!"
!!$      qrot=qrot + (qts%dtau(iimage)+qts%dtau(iimage+1))*0.5D0/svar * log(qrot_part)
!!$    end do
!!$    if(dbg_rot) print*,"ln(Rotational partition function) instanton:",qrot 
!!$    print*,"ln(Rotational partition function) instanton: (prod of Q, not used)",qrot 

    ! calculate the rotational partition function from a supermolecule
    if(printl>=4) write(stdout,"('Rotational partition function of the instanton is calculated as supermolecule.')")
    call allocate(mass_all,glob%nat*nimage)
    do iimage=1,nimage
       mass_all((iimage-1)*glob%nat+1:iimage*glob%nat)=glob%mass/dble(nimage) ! this should include dtau for variable dtau
    end do
    nzero=glob%nzero
    coeff=1.D0 ! odd/even J in rotation (no possibility to set by user yet)
    call rotational_partition_function(glob%nat*nimage,mass_all,nzero,neb%xcoords(:,:),beta_hbar,coeff,&
         qrot_part,qrot_quant_part,moi)
    if(nzero/=glob%nzero.and.printl>=2) write(stdout,"('Warning: nzero is likely to be wrong!')")
    call deallocate(mass_all)
    qrot=qrot_store+log(qrot_part)
    qrot_quant=qrot_quant+log(qrot_quant_part)
    !print*,"ln(Rotational partition function) instanton: (supermolecule, used)",qrot 
  end if
  
  ! re-calculate S_pot in case of inithessian=7
  if(glob%inithessian==7.or.glob%inithessian==5) then
    ! JK: 5 was added along with 7: it should not hurt in general, but makes it possible to re-calculate dual-level
    ! rate constants with inithessian=5
    qts%S_pot=0.D0
    do iimage=1,nimage
      qts%S_pot=qts%S_pot+neb%ene(iimage)*(qts%dtau(iimage)+qts%dtau(iimage+1))
    end do
  end if

  call allocate(total_hessian,neb%varperimage*neb%nimage*2,neb%varperimage*neb%nimage*2)
  total_hessian(:,:)=0.D0

  ! build up duplicated total hessian (containing the V-Hessian at the moment)
  do iimage=1,nimage
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage)

    total_hessian(cstart:cend,cstart:cend)=qts%vhessian(:,:,iimage)

    ! duplication
    total_hessian(&
          nvar-neb%cend(iimage)+1:nvar-neb%cstart(iimage)+1,&
          nvar-neb%cend(iimage)+1:nvar-neb%cstart(iimage)+1)= &
          qts%vhessian(:,:,iimage)

  end do


  ! convert to d^2 E / d y^2 without any scaling factors 
!  qts%total_hessian=qts%total_hessian/beta_hbar**2*dble(2*nimage)**2

  if(qts%tsplit) then
    ! calculate Hessian of the open path - for tunnelling splittings
    det_tsplit=0.D0
    call qts_det_tsplitting(neb%nimage,neb%varperimage,qts%dtau,total_hessian,det_tsplit)
  end if

  ! convert the hessian of potential energies to one of action (modify
  ! diagonal, add spring contributions)
  call qts_hessian_etos(2,neb%nimage,neb%varperimage,qts%dtau,total_hessian)

  ! Print total Hessian
  !write(*,*) 'Total hessian_2', qts%total_hessian
  if(printl>=6) then
    write(stdout,"('Total Hessian')")
    do ival=1,nvar
      write(stdout,"(i5,1x,10f10.5)") ival,total_hessian(1:10,ival)
    end do
  end if

  ! estimate zero eigenvector:
  call allocate(vec0,neb%varperimage*neb%nimage*2)
  vec0(:)=0.D0
  do iimage=1,nimage
    primage=iimage-1
    if(iimage==1) primage=1
    neimage=iimage+1
    if(iimage==nimage) neimage=nimage
    vec0(neb%cstart(iimage):neb%cend(iimage))=( &
        glob%icoords(neb%cstart(neimage):neb%cend(neimage)) - &
        glob%icoords(neb%cstart(primage):neb%cend(primage)) ) /  &
        (qts%dtau(iimage)+qts%dtau(iimage+1))
    ! mirror images:
    vec0(nvar-neb%cend(iimage)+1:nvar-neb%cstart(iimage)+1)= &
        -vec0(neb%cstart(iimage):neb%cend(iimage)) 
  end do
  vec0=vec0/sqrt(sum(vec0**2))
  write(*,'("Expectation value of the image rotation:",es18.9)') sum(vec0*matmul(total_hessian,vec0))

  ! calculate d E_b / d beta_hba
  if(sum(abs(qts%igradient))>0.D0) then ! meaning: if it was set
    call calc_dEdbeta(neb%nimage, glob%icoords, qts%vhessian, qts%igradient, &
        neb%varperimage, beta_hbar, info, qts%dEbdbeta)
  else
    if(printl>=4) write(stdout,'("No gradient information, dEb/dbeta is calculated from product of eigenvalues.")')
    qts%dEbdbeta=1.D0 ! positive is flagged as useless
  end if ! gradient was set

  call allocate(qts%stabpar_diffeq,neb%varperimage-glob%nzero-1)
  call allocate(qts%stabpar_trace,neb%varperimage-glob%nzero-1)
  
  ! calculate stability parameters by solving the stability matrix
  ! differential equation numerically
  ! this call sets qts%stabpar_diffeq
  call stability_parameters_monodromy
  
  if(printl>=4) write(stdout,"('Diagonalising the Hessian matrix ...')")
  call allocate(evals_hess, neb%varperimage*neb%nimage*2)
  call allocate(evecs_hess, neb%varperimage*neb%nimage*2,neb%varperimage*neb%nimage*2)
  !if(hess_dtau) then
  if(time) call CPU_TIME (time2)
  call dlf_matrix_diagonalise(neb%varperimage*neb%nimage*2,total_hessian,evals_hess,evecs_hess)
  if(time) call CPU_TIME (time3)
  if(time) print*,"time full diagonalisation:", time3-time2
  !else
    ! Asymmetric:
    ! using MKL, the eigenvalues (singular values) are returned sorted. However,
    ! only their absolute value is returned in any case
    ! call dlf_matrix_asymm_diagonalise(neb%varperimage*neb%nimage*2,total_hessian,evals_hess)

    ! Symmetric
  !  call dlf_matrix_diagonalise(neb%varperimage*neb%nimage*2,total_hessian,evals_hess,evecs_hess)
  !end if

  call deallocate(total_hessian)

  !
  ! write information files (qtsene)
  !
  if(qts%status==1.and.printl>=2.and.printf>=2) then
    ! list of energies 
    if (glob%ntasks > 1) then
      open(unit=501,file="../qtsene")
    else
      open(unit=501,file="qtsene")
    end if
    write(501,"('# E_RS:   ',f20.10)") ers
    write(501,"('# ZPE(RS):',f20.10)") zpers
    write(501,"('# S_0:    ',f20.10)") qts%S_0
    write(501,"('# S_pot:  ',f20.10)") qts%S_pot - ers*beta_hbar
    write(501,"('# S_ins:  ',f20.10)") qts%S_ins- ers*beta_hbar
    svar=(qts%S_pot - 0.5D0 * qts%S_0 )/beta_hbar
    write(501,"('# Etunnel from those (rel to RS)',f18.10)") svar-ers
    write(501,"('# Etunnel energy conservation:  ',f18.10)") qts%etunnel-ers
    write(501,"('# Unit of path length: mass-weighted Cartesians (sqrt(m_e)*Bohr)')")
    write(501,*)
    write(501,"('# Path length   Energy-E_RS      VAE-RS           Energy           VAE              rel. ZPE')")
    call allocate(zpe,neb%nimage)
    zpe=0.D0
    call allocate(evals_image,neb%varperimage)
    call allocate(evecs_image,neb%varperimage,neb%varperimage)
    do iimage=1,neb%nimage
      ! calculate ZPE for that image
      call dlf_matrix_diagonalise(neb%varperimage,qts%vhessian(:,:,iimage),evals_image,evecs_image)
      icomp=glob%nzero
      if(abs(evals_image(1))>abs(evals_image(glob%nzero+1))) icomp=icomp+1
      do ivar=icomp+1,neb%varperimage
        ! it would be better to project out R&T
        ! now we have the total ZPE. Do we want it perpendicular to the path, or along the path?
        if(evals_image(ivar)>0.D0) then
          zpe(iimage)=zpe(iimage)+0.5D0*sqrt(evals_image(ivar))
        end if
      end do
      write(501,"(f10.5,5f17.10)") qts%dist(iimage),neb%ene(iimage)-ers,&
          neb%ene(iimage)+zpe(iimage)-ers-zpers,neb%ene(iimage),&
          neb%ene(iimage)+zpe(iimage),zpe(iimage)-zpers
    end do
    call deallocate(zpe)
    call deallocate(evals_image)
    call deallocate(evecs_image)
    
    write(501,*)
    write(501,"(f10.5,f17.10)") 0.D0,qts%etunnel-ers
    write(501,"(f10.5,f17.10)") qts%dist(neb%nimage+1),qts%etunnel-ers
    close(501)
 end if
  ! normalise each image of first eigenvector
  jvar=2
  do ivar=1,19
    if((sum(vec0*evecs_hess(:,ivar)))**2>0.8D0) then
      jvar=ivar
      exit
    end if
  end do
  !print*,"Rotation eigenvector",jvar
  call allocate(trmodes,2*neb%varperimage*glob%nimage,1)
  trmodes=reshape(evecs_hess(:,jvar),(/2*neb%varperimage*glob%nimage,1/)) ! image rotation mode

  ! now we could run the averaged eigenvalue method:
  call sigma_frequency_average(trmodes)

  ! Another way to calculate stability parameters:
  if(nzero==6) then
    ! this routine sets qts%stabpar_trace
    call stability_parameters_trace(trmodes)
  end if

  ! Hessian average
  !call stapar_avg_hessian(evecs_hess(:,jvar))

  ! analysis of eigenvalues of the full Hessian
  do iimage=1,2*neb%nimage
    cstart=iimage*neb%varperimage-neb%varperimage+1
    cend=iimage*neb%varperimage
    svar=sum(trmodes(cstart:cend,1)**2)
    trmodes(cstart:cend,1)=trmodes(cstart:cend,1)/sqrt(svar)
  end do
  icomp=0
  svar2=0.D0
  prod_evals_hess_prime=0.D0
  prod_sigma=0.D0
  do ival=1,2*neb%nimage*neb%varperimage
    svar=0.D0
    do iimage=1,2*neb%nimage
      cstart=iimage*neb%varperimage-neb%varperimage+1
      cend=iimage*neb%varperimage
      svar=svar+(sum(trmodes(cstart:cend,1)*evecs_hess(cstart:cend,ival)))**2
    end do
    if(ival/=jvar) svar2=svar2+svar
    ! use all eigenvalues, but only a fraction of them
    if(ival/=jvar.and.(ival==1.or.ival>glob%nzero+2)) then
      prod_evals_hess_prime=prod_evals_hess_prime&
          +svar*0.5D0*log(abs(evals_hess(ival)))
      prod_sigma=prod_sigma &
          +(1.D0-svar)*0.5D0*log(abs(evals_hess(ival)))
!!$      print*,"projection",ival,svar,evals_hess(ival), "using"
!!$    else
!!$      print*,"projection",ival,svar,evals_hess(ival), "NOT using"
    end if
    icomp=2*neb%nimage ! needed below
    !
  end do
  call deallocate(trmodes)
  !print*,"prod_evals_hess_prime",prod_evals_hess_prime
  !print*,"dtau^P               ",icomp,dble(icomp)*log(beta_hbar/dble(2*neb%nimage))
  !print*,"sum proj",svar2," number with proj > 0.5:",icomp
  if(printl>=4.and.abs(svar2+1.D0-dble(2*neb%nimage))>1.D-6) then
     write(stdout,'("Warning: projection of eigenvectors problematic! ",f10.5)') &
          svar2+1.D0-dble(2*neb%nimage)
  end if
  !print*,"test1=ln  ",prod_evals_hess_prime+dble(2*neb%nimage)*log(beta_hbar/dble(2*neb%nimage))
  !print*,"test2=108 ",exp(prod_evals_hess_prime+dble(2*neb%nimage)*log(beta_hbar/dble(2*neb%nimage)))
  svar=exp(prod_evals_hess_prime+dble(icomp)*log(beta_hbar/dble(2*neb%nimage)))
  svar=-qts%S_0/svar**2
  qts%dEbdbeta_prod=svar
  print*,"dE_b/dbh from product",qts%dEbdbeta_prod
  if(qts%dEbdbeta>0.99D0) qts%dEbdbeta=qts%dEbdbeta_prod

  ! now get sigma
  svar2=0.D0
  do iimage=1,2*neb%nimage-1
    svar2=svar2+0.5D0*log(4.D0/(beta_hbar/dble(2*neb%nimage))**2*(sin(pi*dble(iimage)/dble(2*neb%nimage)))**2)
  end do
  svar=prod_sigma-svar2*glob%nzero+dble(2*neb%nimage*(neb%varperimage-1-glob%nzero))*&
      log(beta_hbar/dble(2*neb%nimage))
  qts%sigma=svar
  if(printl>=2) write(stdout,'("Sigma from product of eigenvalues of full Hessian:",es15.7)') qts%sigma

  ! translation and rotation modes of the whole path
  call allocate(mass_all,glob%nat*nimage)
  do iimage=1,nimage
     mass_all((iimage-1)*glob%nat+1:iimage*glob%nat)=glob%mass/dble(nimage) ! this should include dtau for variable dtau
  end do
  !call allocate(longxcoords,3,glob%nat*glob%nimage)
  call allocate(trmodes,neb%varperimage*glob%nimage,6)
  call allocate(trmodes2,2*neb%varperimage*glob%nimage,6)
  if(neb%varperimage==3*glob%nat) then
     call dlf_trmodes(glob%nat*glob%nimage,mass_all,neb%xcoords(:,:),ival,trmodes)
     ! set useful entries in trmodes2
     ! double entries in trmodes
     trmodes2(1:neb%varperimage*glob%nimage,:)=trmodes
     ivar=neb%varperimage*glob%nimage
     do iimage=1,nimage
        jimage=nimage-iimage+1
        trmodes2(ivar+jimage*neb%varperimage-neb%varperimage+1:ivar+jimage*neb%varperimage,:)=&
             trmodes(iimage*neb%varperimage-neb%varperimage+1:iimage*neb%varperimage,:)
     end do
  else
     trmodes2=0.D0 ! must not be used anyway
  end if
  call deallocate(trmodes)
  call deallocate(mass_all)
  trmodes2=trmodes2/sqrt(2.D0)
  
  ! Print Eigenvalues (not all...)
  if(printl>=4) then
    call dlf_constants_get("CM_INV_FOR_AMU",CM_INV_FOR_AMu)
    call dlf_constants_get("AMU",amu)
    write(stdout,"('Eigenvalues of the qTS Hessian')")
   !write(stdout,"('Number    Eigenvalue      Wave Number       Projection onto tangent')")
    write(stdout,"('Number    Eigenvalue      Wave Number       <tan> <T&R>')")

    do ival=1,min(15,nvar) ! was 15
      projtr=0.D0
      do icomp=1,6
        projtr=projtr+(sum(evecs_hess(:,ival)*trmodes2(:,icomp)))**2
      end do
      
      if(hess_dtau) then
        svar=evals_hess(ival)*dble(2*neb%nimage)/beta_hbar*amu ! transformed EV - for E" multiplied by dtau in Hessian
      else
        svar=evals_hess(ival)*amu ! transformed EV - for E" without mult in Hessian
      end if
      svar=sqrt(abs(svar))*CM_INV_FOR_AMU
      if(evals_hess(ival)<0.D0) svar=-svar
      if(ival>1.and.ival<glob%nzero+3) then
        write(stdout,"(i6,1x,es18.9,2x,f10.3,' cm^-1',2f6.3,', treated as zero')") &
            ival,evals_hess(ival),svar,(sum(vec0*evecs_hess(:,ival)))**2,projtr
      else
        write(stdout,"(i6,1x,es18.9,2x,f10.3,' cm^-1',2f6.3)") ival,evals_hess(ival),&
            svar,(sum(vec0*evecs_hess(:,ival)))**2,projtr
      end if
    end do
  end if

  ! check if the correct modes were omitted (needs to be done outside
  ! of the above print loop, because that is only executed at high
  ! enough printl)
  projtr=0.D0
  do ival=2,glob%nzero+2
    do icomp=1,6
      projtr=projtr+(sum(evecs_hess(:,ival)*trmodes2(:,icomp)))**2
    end do
    projtr=(sum(vec0*evecs_hess(:,ival)))**2+projtr
  end do
  if(projtr<dble(glob%nzero+1)-0.1D0-10.D0) then ! -10: switch that check OFF
    if(printl>=4) then
      write(stdout,'("Part of the small eigenvalues include vibrations. &
          &Taking that into account in the product.")')
      write(stdout,'("sum of projections",f10.5)') projtr
    end if
    prod_evals_hess_prime=0.D0
    do ival=1,nvar
      projtr=0.D0
      do icomp=1,6
        projtr=projtr+(sum(evecs_hess(:,ival)*trmodes2(:,icomp)))**2
      end do
      svar=(sum(vec0*evecs_hess(:,ival)))**2
      if(projtr+svar<=0.1D0) prod_evals_hess_prime=prod_evals_hess_prime+&
          log(abs(evals_hess(ival)))
      ! include a fraction of the eigenvalue is projection is in between 0.1 and 0.9
      if(projtr+svar>0.1D0.and.projtr+svar<0.9D0) &
          prod_evals_hess_prime=prod_evals_hess_prime+&
          log(abs(evals_hess(ival)))*(1.D0-(projtr+svar))
    end do
    prod_evals_hess_prime=0.5D0*prod_evals_hess_prime
  else
    ! calculate the log of the product rather than the product - to avoid overflow
    prod_evals_hess_prime=0.5D0* (log(abs(evals_hess(1))) + sum(log(abs(evals_hess(glob%nzero+3:nvar)))) )
  end if
  
  call deallocate(trmodes2)
  ! print warnings
  if(printl>=2) then
    if(abs(evals_hess(1)/evals_hess(2)) < 10.D0) then
      write(stdout,"('Warning: negative eigenvalue and the lowest zero &
          &eigenvalue differ by less than a factor of 10')")
      write(stdout,"('  Factor by which they differ',f10.5)") evals_hess(1)/evals_hess(2)
    end if
    if(abs(evals_hess(glob%nzero+3)/evals_hess(glob%nzero+2)) < 10.D0) then
      write(stdout,"('Warning: first positive eigenvalue and highest zero &
          &eigenvalue differ by less than a factor of 10')")
      write(stdout,"('  Factor by which they differ',f10.5)") &
          evals_hess(glob%nzero+3)/evals_hess(glob%nzero+2)
    end if
  end if

  ! write a few eigenvectors to files
  if(printf>=4 .and. glob%iam == 0 ) then
    call allocate(xmode,3,glob%nat)
    xmode(:,:)=0.D0
    do ival=1,min(9,nvar)
      write(chr2,'("0",i1)') ival
      if (glob%ntasks > 1) then
        filename="../qtsmode"//chr2//".xyz"
      else
        filename="qtsmode"//chr2//".xyz"
      end if
      open(unit=501,file=filename)
      do iimage=1,2*neb%nimage
        call dlf_cartesian_itox(glob%nat,neb%varperimage,neb%varperimage,glob%massweight,&
            evecs_hess((iimage-1)*neb%varperimage+1:iimage*neb%varperimage,ival),&
            xmode)
        call write_xyz(501,glob%nat,glob%znuc,xmode)
      end do
      close(501)
    end do
    ! do also write vec0
    if (glob%ntasks > 1) then
      filename="../qtsmode_vec0.xyz"
    else
      filename="qtsmode_vec0.xyz"
    end if
    open(unit=501,file=filename)
    do iimage=1,2*neb%nimage
      call dlf_cartesian_itox(glob%nat,neb%varperimage,neb%varperimage,glob%massweight,&
          vec0((iimage-1)*neb%varperimage+1:iimage*neb%varperimage),xmode)
      call write_xyz(501,glob%nat,glob%znuc,xmode)
    end do
    close(501)
    call deallocate(xmode)
  end if

  !---------------------------------------------------------------------------------
  !Calculate Rate (hbar=1) (see Messina/Schenter, J. Chem. Phys. 1995)
  !---------------------------------------------------------------------------------


  ! check if mode of image rotation is within the zero modes
  svar=0.D0
  do ival=2,glob%nzero+2
    svar=max(svar,(sum(vec0*evecs_hess(:,ival)))**2)
  end do
  if(svar<0.1D0) then
    ! search for image rotation mode among all modes and exchange eigenvalues
    do ival=glob%nzero+3,nvar
      if( (sum(vec0*evecs_hess(:,ival)))**2 > 0.8D0 ) then
        if(printl>=2) write(stdout,"('Warning: Mode',i4,' appears to be &
            &the mode responsible for image rotation.', f10.5)") ival,(sum(vec0*evecs_hess(:,ival)))**2
        if(printl>=2) write(stdout,"('Treating that as zero instead of mode',i3)") &
            glob%nzero+2
        prod_evals_hess_prime= prod_evals_hess_prime - &
            0.5D0*log(abs(evals_hess(ival))) + &
            0.5D0*log(abs(evals_hess(glob%nzero+2)))
        exit
      end if
    end do
  end if

  ! calculate a reduced mass of the instanton
  ! transform the transition mode to xcoords
  if(printl>=6) then
    call allocate(xmode,3,glob%nat)
    xmode(:,:)=0.D0
    call dlf_constants_get("AMU",amu)
    eff_mass_total=0.D0
    norm_total=0.D0
    do iimage=1,neb%nimage
      call dlf_cartesian_itox(glob%nat,neb%varperimage,neb%varperimage,glob%massweight,&
          evecs_hess((iimage-1)*neb%varperimage+1:iimage*neb%varperimage,1),&
          xmode)
      eff_mass=0.D0
      norm=0.D0
      icomp=0
      do iat=1,glob%nat
        svar=sum(xmode(:,iat)**2)
        if(svar>1.D-20) then
          ! this atom is active
          icomp=icomp+3
          svar=sum(evecs_hess((iimage-1)*neb%varperimage+icomp-2:&
              (iimage-1)*neb%varperimage+icomp,1)**2)
          eff_mass=eff_mass+svar/glob%mass(iat)
          norm=norm+svar
        end if
      end do
      if(icomp==neb%varperimage) then
        eff_mass_total=eff_mass_total+eff_mass
        norm_total=norm_total+norm
        eff_mass=norm/eff_mass
        !print*,"Norm:",norm
        ! print
        write(stdout,"('Effective mass at image ',i5,':     ',es17.10,' amu')") &
            iimage,eff_mass/amu
        !else
        !  write(stdout,"('Error in calculating the effective mass of image',i5)") iimage
      end if
    end do
    !print*,"Total norm",norm_total
    if(eff_mass_total>0.D0) then
      eff_mass_total=norm_total/eff_mass_total
      write(stdout,"('Effective mass of whole path       ',es17.10,' amu')") &
          eff_mass_total/amu
    end if
    deallocate(xmode)
  end if
  ! end of effective mass
  
  call deallocate(evecs_hess)
!  call deallocate(evals_hess)
  call deallocate(vec0)

  qts%S_ins=0.5D0 * qts%S_0 + qts%S_pot - ers*beta_hbar

  call dlf_constants_get("SECOND_AU",second_au)
  if(printl>=2) then
    write(stdout,"('S_0                                ',es17.10,' hbar')") &
        qts%S_0
    write(stdout,"('S_pot                              ',es17.10,' hbar')") &
        qts%S_pot - ers*beta_hbar
    write(stdout,"('S_ins                              ',es17.10,' hbar')") &
        qts%S_ins 
    write(stdout,"('E_eff                              ',es17.10)") &
        qts%S_ins/beta_hbar
    write(stdout,"('beta*hbar                          ',es17.10)") &
        beta_hbar
    write(stdout,"('ln SQRT(S_0/2pi)                   ',es17.10)") &
        0.5D0*log(qts%S_0/(2.D0*pi))
    write(stdout,"('  SQRT(S_0/2pi)                    ',es17.10)") &
        sqrt(qts%S_0/(2.D0*pi))
    if(hess_dtau) then
      !write(stdout,"('-ln Prod( SQRT( eigvals))          ',es17.10)") &
      !    -prod_evals_hess_prime!-log(beta_hbar/dble(2*neb%nimage))*dble(glob%nzero+1)
      write(stdout,"('-ln dTau^7*Prod( SQRT( eigvals))   ',es17.10)") &
          -prod_evals_hess_prime-log(beta_hbar/dble(2*neb%nimage))*(dble(glob%nzero+1)*0.5D0+&
          dble(neb%varperimage*neb%nimage))
      write(stdout,"('ln dtau                            ',es17.10)") &
          log(beta_hbar/dble(2*neb%nimage))
    else
      write(stdout,"('-ln Prod( SQRT( eigvals))          ',es17.10)") &
          -prod_evals_hess_prime!-log(beta_hbar/dble(2*neb%nimage))*dble(glob%nzero+1)
      !print*,"test1=ln  ",prod_evals_hess_prime+dble(2*neb%nimage*neb%varperimage)*log(beta_hbar/dble(2*neb%nimage))
      !print*,"test2=108 ",exp(prod_evals_hess_prime+dble(2*neb%nimage*neb%varperimage)*log(beta_hbar/dble(2*neb%nimage)))
      !svar=exp(prod_evals_hess_prime+dble(2*neb%nimage*neb%varperimage)*log(beta_hbar/dble(2*neb%nimage)))
      !svar=-qts%S_0/svar**2
      !print*,"test dE_b/dbh",svar
      
      write(stdout,"('ln dtau                            ',es17.10)") &
          log(beta_hbar/dble(2*neb%nimage))
    end if
    
    write(stdout,"('-S_ins/hbar                        ',es17.10)") -qts%S_ins
  end if

  !qts%rate=sqrt(qts%S_0/(2.D0*pi*qts%const_hess))*exp(-qts%S_ins)/prod_evals_hess_prime
  if(hess_dtau) then
    ! was:
    !  qts%rate=0.5D0*log(qts%S_0/2.D0/pi) -svar &
    !      -prod_evals_hess_prime-log(beta_hbar/dble(2*neb%nimage))*&
    !      (dble(glob%nzero+1)*0.5D0+dble(neb%varperimage*neb%nimage))
    qts%rate=0.5D0*log(qts%S_0/2.D0/pi) -qts%S_ins &
        -prod_evals_hess_prime-0.5D0*log(beta_hbar/dble(2*neb%nimage))
  else
    qts%rate=0.5D0*log(qts%S_0/2.D0/pi) -qts%S_ins &
        -prod_evals_hess_prime!-log(beta_hbar/dble(2*neb%nimage))*dble(glob%nzero+1)
  end if

  if(printl>=2) write(stdout,"('ln(RATE * Q(RS)) = ln(Flux)        ',es17.10)") &
       qts%rate-log(beta_hbar/dble(2*neb%nimage))*dble(2*neb%nimage*neb%varperimage)
  if(printl>=2) write(stdout,"('   RATE * Q(RS) = Flux             ',es17.10)") &
       exp(qts%rate-log(beta_hbar/dble(2*neb%nimage))*dble(2*neb%nimage*neb%varperimage))
  if(tflux) then
    write(stdout,"('Calculating a flux rather than a rate constant!')")
    qrs=log(beta_hbar/dble(2*neb%nimage))*dble(2*neb%nimage)
  end if

!!$  ! Flux, tested for the Eckart potential
!!$  if(printl>=2) write(stdout,"('ln(FLUX)                           ',es18.9)") &
!!$      qts%rate-log(beta_hbar/dble(2*neb%nimage))*dble(2*neb%nimage)
!!$  if(printl>=2) write(stdout,"('log10(FLUX)                        ',es18.9)") &
!!$      (qts%rate-log(beta_hbar/dble(2*neb%nimage))*dble(2*neb%nimage))/log(1.D1)
!!$  if(printl>=2) write(stdout,"('log10(FLUX/sec)                    ',es18.9)") &
!!$      (qts%rate-log(beta_hbar/dble(2*neb%nimage))*dble(2*neb%nimage)+log(second_au))/log(1.D1)
!!$  ! only for eckart
!!$  call driver_clrate(beta_hbar,svar)
!!$  if(printl>=2) write(stdout,"('ln(FLUX_Classical)                 ',es18.9)") svar
!!$  if(printl>=2) write(stdout,"('ln(Gamma)                          ',es18.9)") &
!!$      qts%rate-log(beta_hbar/dble(2*neb%nimage))*dble(2*neb%nimage)-svar
!!$  if(printl>=2) write(stdout,"('Gamma                              ',es18.9)") &
!!$      exp(qts%rate-log(beta_hbar/dble(2*neb%nimage))*dble(2*neb%nimage)-svar)
!!$  !comparison to exact eckart (only symmetric implemented)
!!$  call driver_eck(vb,alpha)
!!$  call kappa_eckart(beta_hbar,Vb,alpha,kappa_eck)
!!$  if(printl>=2) write(stdout,"('kappa Eckart                       ',es18.9)") kappa_eck
  

!!$  ! tmp:
!!$  print*,"log10(rate * Q(RS) * exp(beta*DeltaE))",(0.5D0*log(qts%S_0/2.D0/pi)-prod_evals_hess_prime)/log(10.D0)-&
!!$      log(beta_hbar*0.5D0/dble(neb%nimage))*dble(2*neb%nimage)/log(10.D0)
!!$  print*,"rate * Q(RS) * exp(beta*DeltaE)",exp(0.5D0*log(qts%S_0/2.D0/pi)-prod_evals_hess_prime-&
!!$      log(beta_hbar*0.5D0/dble(neb%nimage))*dble(2*neb%nimage))
!!$  print*,"1/(rate * Q(RS) * exp(beta*DeltaE))",1.D0/exp(0.5D0*log(qts%S_0/2.D0/pi)-prod_evals_hess_prime-&
!!$      log(beta_hbar*0.5D0/dble(neb%nimage))*dble(2*neb%nimage))
!!$  print*,"beta hbar",beta_hbar
!!$  print*,"log10(dtau)*2P",log(beta_hbar*0.5D0/dble(neb%nimage))*dble(2*neb%nimage)/log(10.D0)
 
  if(tok) then
    ! make sure the tunneling energy is sensible (relevant mainly for dual-level):
    if(qts%etunnel-ers < 0.D0) then
      qts%etunnel=(qts%S_pot - 0.5D0 * qts%S_0 )/beta_hbar
      if(printl>=2) write(stdout,"('Tunneling energy re-calculated from S_pot')")
    end if
    if(printl>=2) then
      write(stdout,"('Energy of the RS                   ',es17.10,' Hartree')") &
          ers
      write(stdout,"('Tunneling energy (rel. to RS)      ',es17.10,' Hartree')") &
          qts%etunnel-ers
      write(stdout,"('d E_b / d beta_hbar                ',es17.10)") qts%dEbdbeta
      write(stdout,"('ln(Q(RS))                          ',es17.10)") qrs
      write(stdout,"('ln Qrot_rel                        ',es17.10)") qrot
      write(stdout,"('Quantum rigid rotors change k by a factor',es17.10)") exp(qrot_quant-qrot)
      write(stdout,"(' The following should be as independent of nimage as possible:')") 
      if(hess_dtau) then
         write(stdout,"('ln(Q_TS/Q_RS)_vib                  ',es17.10)") -prod_evals_hess_prime &
            -log(beta_hbar/dble(2*neb%nimage))*(dble(glob%nzero+1)*0.5D0+dble(2*neb%nimage))-qrs
      else
        write(stdout,"('ln(Q_TS/Q_RS)_vib                  ',es17.10)") -prod_evals_hess_prime -qrs
      end if
      ! commented out due to floating overflows:
!      write(stdout,"('Q(RS)                              ',es17.10)") exp(qrs)

      if(qts%tsplit.and.printl>=2) then
        ! Tunnelling splitting stuff
        write(stdout,*)
        write(stdout,"('Tunnelling splitting assuming a symmetric potential:')")
        !print*,"S_0/2",qts%S_0/2.D0
        !print*,"sqrt(qts%S_0/4.D0/pi)",sqrt(qts%S_0/4.D0/pi)
        !print*,"exp(-0.5D0*qts%S_0)",exp(-0.5D0*qts%S_0)

      ! Tunnelling splitting from closed path:
!!$      ! modifying prod_evals_hess_prime!
!!$      prod_evals_hess_prime= prod_evals_hess_prime - &
!!$            0.5D0*log(abs(evals_hess(1)))
!!$      ! should be the same:
!!$      prod_evals_hess_prime=0.5D0* sum(log(abs(evals_hess(glob%nzero+3:nvar)))) 
!!$      svar=sqrt(qts%S_0/4.D0/pi)* &
!!$          exp(-0.5D0*qts%S_0 -prod_evals_hess_prime - qrs) 
!!$      write(stdout,"('Tunnelling splitting closed path    ',es18.9,' Hartree')") svar
!!$      call dlf_constants_get("CM_INV_FROM_HARTREE",svar2)
!!$      write(stdout,"('Tunnelling splitting closed path    ',es18.9,' cm^-1')") svar*svar2

        ! Tunnelling splitting from linear (open) path:
        write(stdout,*) "Tunnelling splitting with S_0 calculated from the path length"
        write(stdout,"('S_0/2 from the path length         ',es17.10)") qts%S_0/2.D0
        svar=2.D0*sqrt(qts%S_0/4.D0/pi)* &
            exp(-0.5D0*qts%S_0 -det_tsplit) 
        write(stdout,"('Tunnelling splitting linear path S0 ',es17.10,' Hartree')") svar
        call dlf_constants_get("CM_INV_FROM_HARTREE",svar2)
        write(stdout,"('Tunnelling splitting linear path S0 ',es17.10,' cm^-1')") svar*svar2
        
        ! calculate S_0 from potential: 
        write(stdout,'(a)') " Tunnelling splitting with S_0 calculated from S_pot and E_RS (recommended)"
        ! S_0= 2*sqrt(2) * int sqrt(E-E_b) dy
        ! S_0= 2 ( S_pot - beta_hbar E_b)
        write(stdout,"('S_0/2 from S_pot and E_RS          ',es17.10)") qts%S_pot - ers*beta_hbar
        svar2=2.D0*(qts%S_pot - ers*beta_hbar)
        ! Tunnelling splitting from linear (open) path:
        svar=2.D0*sqrt(svar2/4.D0/pi)* &
            exp(-0.5D0*svar2 -det_tsplit) 
        write(stdout,"('Tunnelling splitting linear path p  ',es17.10,' Hartree')") svar
        call dlf_constants_get("CM_INV_FROM_HARTREE",svar2)
        write(stdout,"('Tunnelling splitting linear path p  ',es17.10,' cm^-1')") svar*svar2
        write(stdout,"('ln(resonant rate constant)          ',es17.10)") log(svar/2.D0/pi)
        write(stdout,*)
      end if
!!!!!!!! end of tunnelling splittings !!!!!!!!!!!!!!!!!!!!!!!!!!!

      ! add rotational partition function to ln rate
      qts%rate=qts%rate+qrot

      write(stdout,"('ln(rate constant)                  ',es17.10)") qts%rate-qrs
      write(stdout,"('Rate constant                      ',es17.10)") exp(qts%rate-qrs)
      write(stdout,"('Effective free-energy barrier      ',es17.10)") &
          -(log(beta_hbar*2.D0*pi)+qts%rate-qrs)/beta_hbar
      write(stdout,"('log10(second_au)                   ',es17.10)")log(second_au)/log(10.D0)
      if(tbimol) then
        call dlf_constants_get("ANG_AU",svar)
        svar=log(svar*1.D-10) ! ln of bohr in meters
        ! the factor of 1.D6 converts from m^3 to cm^3
        write(stdout,"('ln(rate constant)                  ',es17.10,' cm^3 per second')") &
            qts%rate-qrs+log(second_au)+log(1.D6)+3.D0*svar
        write(stdout,"('log10(rate constant in cm^3/s)     ',es17.10, ' at ', f10.5, ' K')") &
            (qts%rate-qrs+log(second_au)+log(1.D6)+3.D0*svar)/log(10.D0),&
            glob%temperature
      else
        write(stdout,"('ln(rate constant)                  ',es17.10,' per second')") qts%rate-qrs+log(second_au) 
        write(stdout,"('log10(rate constant per second)    ',es17.10, ' at ', f10.5, ' K')") &
            (qts%rate-qrs+log(second_au))/log(10.D0), glob%temperature
     end if

     ! Kryvohuz correction stuff
     svar=maxval(neb%ene)-qts%etunnel!-qts%sigma/beta_hbar
     !print*,"V_max-E_b ",maxval(neb%ene)-qts%etunnel
     !print*,"sigma'    ",qts%sigma/beta_hbar
     !if(svar<0.D0) then
     !  print*,"Warning, V_max-E_b-sigma' < 0, ignoring sigma'!"
     !  svar=maxval(neb%ene)-qts%etunnel
     !end if
     svar=svar/sqrt(abs(qts%dEbdbeta)) ! andreas' version
     !print*,"arg of erf",svar/sqrt(2.D0)
     svar=erf(svar/sqrt(2.d0))
     corrfac=0.5D0+0.5D0*svar
     write(stdout,"('Kryvohuz correction factor         ',es17.10)") corrfac
     if(tbimol) then
       call dlf_constants_get("ANG_AU",svar)
       svar=log(svar*1.D-10) ! ln of bohr in meters
       ! the factor of 1.D6 converts from m^3 to cm^3
       !write(stdout,"('log10(rate constant in cm^3/s) coAL',es17.10, ' at ', f10.5, ' K')") &
       write(stdout,"('log10(cor. rate constant in cm^3/s)',es17.10, ' at ', f10.5, ' K')") &
           (qts%rate-qrs+log(second_au)+log(1.D6)+3.D0*svar+log(corrfac))/log(10.D0),&
           glob%temperature
     else
       !write(stdout,"('log10(rate constant per second)coAL',es17.10, ' at ', f10.5, ' K')") &
       write(stdout,"('log10(corr. rate constant / second)',es17.10, ' at ', f10.5, ' K')") &
            (qts%rate-qrs+log(second_au)+log(corrfac))/log(10.D0), glob%temperature
     end if

     write(stdout,"('The rate constant still needs to be multiplied &
          &by the symmetry factor sigma')")

!!$     ! again Kryvohuz correction factor - this time my version:
!!$     svar=maxval(neb%ene)-qts%etunnel!-qts%sigma/beta_hbar
!!$     print*,"V_max-E_b ",maxval(neb%ene)-qts%etunnel
!!$     print*,"sigma'    ",qts%sigma/beta_hbar
!!$     if(svar<0.D0) then
!!$       print*,"Warning, V_max-E_b-sigma' < 0, ignoring sigma'!"
!!$       svar=maxval(neb%ene)-qts%etunnel
!!$     end if
!!$     svar=svar/sqrt(abs(qts%dEbdbeta_prod))
!!$     print*,"arg of erf",svar/sqrt(2.D0)
!!$     svar=erf(svar/sqrt(2.d0))
!!$     corrfac=0.5D0+0.5D0*svar
!!$     write(stdout,"('Kryvohuz correction factor JK      ',es17.10)") corrfac
!!$     if(tbimol) then
!!$       call dlf_constants_get("ANG_AU",svar)
!!$       svar=log(svar*1.D-10) ! ln of bohr in meters
!!$       ! the factor of 1.D6 converts from m^3 to cm^3
!!$       write(stdout,"('log10(rate constant in cm^3/s) coJK',es17.10, ' at ', f10.5, ' K')") &
!!$           (qts%rate-qrs+log(second_au)+log(1.D6)+3.D0*svar+log(corrfac))/log(10.D0),&
!!$           glob%temperature
!!$     else
!!$       write(stdout,"('log10(rate constant per second)coJK',es17.10, ' at ', f10.5, ' K')") &
!!$            (qts%rate-qrs+log(second_au)+log(corrfac))/log(10.D0), glob%temperature
!!$     end if


      !write(stdout,"('log10(Rate per second) ',es18.9)") (qts%rate-qrs+log(second_au))/log(10.D0)
     ! print Arrhenius info as log to the base 10
     !write(stdout,"('Arrhenius (log10)',2f12.6)") 1.D3/glob%temperature,(qts%rate-qrs)/log(10.D0)
!!$      write(stdout,"('Data as they were before fixing the reactant state bug:')")
!!$      qrs=qrs+dble(glob%nzero)*log(dble(2*neb%nimage))
!!$      print*,"log(dble(2*neb%nimage))",log(dble(2*neb%nimage))
!!$      write(stdout,"('  !ln(Q(RS)) ',es18.9)") qrs
!!$      write(stdout,"('  !Q(RS)     ',es18.9)") exp(qrs)
!!$      write(stdout,"('  !ln(Rate)  ',es18.9)") qts%rate-qrs
!!$      write(stdout,"('  !Rate      ',es18.9)") exp(qts%rate-qrs)
!!$      ! print Arrhenius info as log to the base 10
!!$      write(stdout,"('  !Arrhenius (log10)',2f12.6)") 1.D3/glob%temperature,(qts%rate-qrs)/log(10.D0)
    end if
  else
    if(printl>=2) write(stdout,"('No rate information is printed, as no &
        &file qts_reactant.txt is present')")
  end if

  ! moved here temporarily - should be between dealloc of evecs_hess and vec0
  call deallocate(evals_hess)

  ! Alternative approach to calculate the rate constant by Andreas Loehle
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! New Method!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  print*
  if(glob%nzero/=6.or.(abs(minval(qts%igradient))<1.D-15.and.abs(maxval(qts%igradient))<1.D-15)) then
     print*,"=== Gradient not available, Andreas' analysis not possible. ==="
     print*,"nzero=",glob%nzero
     print*,"abs(maxval(qts%igradient))",abs(maxval(qts%igradient))
  else

    print*,"Loehle method:"
    
    ! reactant
    call calc_QRS(neb%nimage, beta_hbar, omega2(1:nvib), &
        nvib, fluc_rs, fluc_rs_ana)  
    
    call deallocate(omega2)

    allocate(u_param(2*neb%varperimage))  

    tok=.false.
    if(qts%dist(2)-qts%dist(1) > qts%dist(neb%nimage)-qts%dist(neb%nimage-1)) then
       print*,"The end at nimage is the smoother end"
       tok=.true.
    end if
    
    ! Calculating stability parameters
    call calc_ui_param(neb%nimage, beta_hbar, qts%vhessian, &
        neb%varperimage, tok, u_param)

    ! print stability parameters:
    write(stdout,*) "U-parameters (real,imag, log(real), log(real)/beta) for first image:"
    do ivar=1,2*neb%varperimage
      if(real(u_param(ivar))>0.D0) then
        write(*,'(i4,2es15.6,2x,2es16.8)') ivar,real(u_param(ivar)),imag(u_param(ivar)), &
            log(real(u_param(ivar))),log(real(u_param(ivar)))/beta_hbar
      else
        write(*,'(i4,2es15.6)') ivar,real(u_param(ivar)),imag(u_param(ivar))
      end if
      if(ivar==neb%varperimage-glob%nzero-1) write(*,*)
      if(ivar==neb%varperimage+glob%nzero+1) write(*,*)
    end do
    
    IF (tbimol) THEN
      print*,"Bimolecular rate constant"
      q_rot_trans = exp(qrot)/phi_rel
      print*,"phi_rel",phi_rel
    ELSE
      print*,"Unimolecular rate constant"
      q_rot_trans = exp(qrot)
      print*,"q_rot_trans",q_rot_trans
    END IF
    
    ! calculate sigma (fluc_inst) from u_param
    ! do not sort, use the first neb%varperimage-glob%nzero-1 of u_param
    fluc_inst=0.D0
    do ivar=1,neb%varperimage-glob%nzero-1
      !u_i:
      svar=log(dble(u_param(ivar)))
      fluc_inst=fluc_inst+log(2.D0*sinh(0.5D0*svar))
    end do

    print*, "#############################Method Loehle################ "
    write(stdout,"('d E_b / d beta_hbar                ',es17.10,1x,a)") qts%dEbdbeta,"Loehle method"
    write(stdout,"('d E_b / d beta_hbar                ',es17.10,1x,a)") qts%dEbdbeta_prod,"From product"
    write(stdout,"('ln(rate constant) is the sum of:')")
    write(stdout,"('ln sqrt(-dE/beta /2pi)             ',es17.10)") 0.5D0*log(-qts%dEbdbeta/2.D0/pi)
    write(stdout,"('ln Qrot_trans                      ',es17.10)") log(q_rot_trans)
    write(stdout,"('-ln Q_RS                           ',es17.10)") -log(fluc_rs)
    write(stdout,"('ln Q_inst = -sigma                 ',es17.10)") -fluc_inst
    write(stdout,"('-S_inst+S_RS                       ',es17.10)") -qts%S_ins
    rate_const=0.5D0*log(-qts%dEbdbeta/2.D0/pi)+log(q_rot_trans)-log(fluc_rs)-fluc_inst-qts%S_ins
    write(stdout,"('ln(rate constant)                  ',es17.10)") rate_const
    write(stdout,"('rate constant                      ',es17.10)") exp(rate_const)
    write(stdout,*)
    write(stdout,"('Comparison:')")
    if(tbimol) then
      call dlf_constants_get("ANG_AU",svar)
      svar=log(svar*1.D-10)
      svar=log(second_au)+log(1.D6)+3.D0*svar
      write(stdout,"('      ln(k)     log10(k in cm^3 per second) description')")
    else
      svar=log(second_au)
      write(stdout,"('      ln(k)             log10(k per second) description')")
    end if
    write(stdout,'(es17.5,9x,es17.5,1x,a)') qts%rate-qrs,(qts%rate-qrs+svar)/log(10.D0),&
        "Determinant method"
    write(stdout,'(es17.5,9x,es17.5,1x,a)') rate_const,(rate_const+svar)/log(10.D0),"Loehle Method"

    rate_const=0.5D0*log(-qts%dEbdbeta_prod/2.D0/pi)+log(q_rot_trans)-log(fluc_rs)-fluc_inst-qts%S_ins
    write(stdout,'(es17.5,9x,es17.5,1x,a)') rate_const,(rate_const+svar)/log(10.D0),&
        "dE/db Product, sigma: Loehle Method"

    ! fluc_inst from qts%stabpar_diffeq
    fluc_inst=0.D0
    do ivar=1,neb%varperimage-glob%nzero-1
      !u_i:
      fluc_inst=fluc_inst+log(2.D0*sinh(0.5D0*beta_hbar*qts%stabpar_diffeq(ivar)))
    end do
    rate_const=0.5D0*log(-qts%dEbdbeta/2.D0/pi)+log(q_rot_trans)-log(fluc_rs)-fluc_inst-qts%S_ins
    write(stdout,'(es17.5,9x,es17.5,1x,a)') rate_const,(rate_const+svar)/log(10.D0),&
        "dE/db Loehle, sigma: stability matrix diff. eq."

    ! fluc_inst from qts%stabpar_trace
    fluc_inst=0.D0
    do ivar=1,neb%varperimage-glob%nzero-1
      !u_i:
      fluc_inst=fluc_inst+log(2.D0*sinh(0.5D0*beta_hbar*qts%stabpar_trace(ivar)))
    end do
    rate_const=0.5D0*log(-qts%dEbdbeta/2.D0/pi)+log(q_rot_trans)-log(fluc_rs)-fluc_inst-qts%S_ins
    write(stdout,'(es17.5,9x,es17.5,1x,a)') rate_const,(rate_const+svar)/log(10.D0),&
        "dE/db Loehle, sigma: eigenvalue tracing"

    write(stdout,*)
    
    !print*, dlog(dreal(u_param))
    line=""
    do ivar=neb%varperimage-glob%nzero-1,1,-1
      write(line,'(a,es15.8)') trim(line),log(real(u_param(ivar)))/beta_hbar
      !     print*,"out",trim(line)
    end do
    write(*,'("Stability parameters Loehle method: ",a)') trim(line)
    print*, "#################P(E)######################################'"
    deallocate(u_param)
  end if ! min/max grad

  call deallocate(qts%stabpar_diffeq)
  call deallocate(qts%stabpar_trace)
  
  call clock_stop("COORDS")

end subroutine dlf_qts_rate
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/qts_reactant
!!
!! FUNCTION
!!
!!
!! Read the Energy and hessian of the reactant from (qts_reactant.txt) and
!! calculate the reactant's partition function
!!
!! Take into account the possibility for bimolecular rates here (calculate the
!! translational and rotational partition functions of the incoming molecule)
!!
!! SYNOPSIS
subroutine qts_reactant(qrs,ers,zpers,qrot,qrot_quant,tbimol, tok, phi_rel,&
    ! variables for Andreas Loehle's routines
    varperimage,omega2,nvib    )
  use dlf_parameter_module, only: rk
  use dlf_neb, only: neb,unitp,xyzall,beta_hbar
  use dlf_global, only: printl,stdout,glob,pi
  use dlf_constants, only : dlf_constants_get
  use dlf_allocate, only: allocate,deallocate
  use dlf_qts
  implicit none
  real(rk), intent(out) :: qrs ! ln of the partition function (no potential part)
  real(rk), intent(out) :: ers ! energy of the reactant
  real(rk), intent(out) :: zpers ! vibrational zero point energy of the reactant
  real(rk), intent(out) :: qrot ! log relative rotational partition function of the reactant
  real(rk), intent(out) :: qrot_quant ! quantum rigid rotor
  real(rk), intent(out) :: phi_rel ! translational partition function
  logical , intent(out) :: tbimol
  logical , intent(out) :: tok
  integer , intent(in)  :: varperimage
  real(rk), intent(out) :: omega2(varperimage) ! Hessian eigenvalues
  integer , intent(out) :: nvib ! number of actual vibrational modes in omega2
  integer  :: ivar,iimage,varperimage_read
  real(rk) :: svar
  real(rk) :: qrsi,dtau

  ! get eigenvalues
  real(rk),allocatable :: hess_sp(:,:) !neb%nimage*2,neb%nimage*2
  real(rk),allocatable :: eval_sp(:) !neb%nimage*2
  real(rk),allocatable :: evec_sp(:,:) !neb%nimage*2,neb%nimage*2
  logical  :: tconst_dtau

  real(rk) :: mu_bim,qrot_part,qrot_quant_part
  type(pf_type) :: rs1,rs2
  logical :: toneat
  
  tok=.false.
  tbimol=.false.
  qrot=0.D0
  qrot_quant=0.D0
  dtau=beta_hbar/dble(2*neb%nimage) ! only for constant dtau

  ! define if dtau is constant or variable
  tconst_dtau=(abs(maxval(qts%dtau)-minval(qts%dtau)) < 1.D-14*abs(minval(qts%dtau)))

  ! initialise
  rs1%tused=.false.
  rs2%tused=.false.
  rs2%ene=0.D0
  rs2%nvib=0
  rs2%nat=0
  rs2%mass=0.D0

  toneat=.false.
  ! read first reactant
  call read_qts_txt(1,0,rs1,toneat)
  if(rs1%nat==-1) return ! no reactant files present.
  
  if(rs1%nat<glob%nat) then
    ! read second reactant
    call read_qts_txt(2,rs1%nat,rs2,toneat)
  end if

  ers=rs1%ene+rs2%ene
  
  if(rs2%tused) then
    call dlf_constants_get("AMU",svar)
    ! define the reduced mass
    mu_bim=rs1%mass*rs2%mass/(rs1%mass+rs2%mass)
    if(rs2%tfrozen) mu_bim=rs1%mass
    if(rs1%tfrozen) mu_bim=rs2%mass
    if(rs1%tfrozen.and.rs2%tfrozen) mu_bim=rs1%mass*rs2%mass/(rs1%mass+rs2%mass)
    !if(rs2%nat==1) mu_bim=rs2%mass ! mass (and not reduced mass) is read from file
    if(glob%irot==1) mu_bim=rs2%mass ! mimic a surface
    if(printl>=4) then
!!$       if(rs2%nat==1) then
!!$          write(stdout,'(a,f20.10,a)') " Using a reduced mass of ",&
!!$               mu_bim," amu read from file"
!!$       else
          write(stdout,'(a,f20.10,a)') " Using a reduced mass of ",&
               mu_bim," amu calculated from the two fragments"
!!$       end if
    end if
 end if

 if(rs1%nat+rs2%nat/=glob%nat) then
    write(stdout,'("Number of atoms: ",3i6)') rs1%nat,rs2%nat,glob%nat
    print*,"JK: there is a good chance that this check will fail for frozen atoms..., but we need some check."
    call dlf_fail("Number of atoms of reactants and instanton not equal!")
 end if
 
 !
 ! calculate the rotational partition function (same code as in dlf_htst_rate)
 !
 qrot=1.D0
 qrot_quant=1.D0
 ! do nothing for atoms (their rotational partition function is unity)
 if(glob%irot==0) then
   if(rs1%nmoi>1) then
     !print*,"rotpart rs1",rs1%coeff
     call rotational_partition_function_calc(rs1%moi,beta_hbar,rs1%coeff,qrot_part,qrot_quant_part)
     qrot=qrot*qrot_part
     qrot_quant=qrot_quant*qrot_quant_part
   end if
 else
   if(rs2%tused) then
     write(stdout,'("Mimicking a surface: only rotation of &
         &RS2 considered, reduced mass = mass of RS2")')
   else
     if(printl>=2) write(stdout,'("Mimicking a surface: rotational partition function kept constant.")')
   end if
 end if
 if(rs2%tused.and.rs2%nmoi>1) then
   !print*,"rotpart rs2",rs2%coeff
   call rotational_partition_function_calc(rs2%moi,beta_hbar,rs2%coeff,qrot_part,qrot_quant_part)
   qrot=qrot*qrot_part
   qrot_quant=qrot_quant*qrot_quant_part
 end if
 qrot= log(qrot) 
 qrot_quant= log(qrot_quant) 

 ! communicate omega^2 to outside
 nvib=rs1%nvib
 if(rs2%tused) nvib=nvib+rs2%nvib
 omega2(1:rs1%nvib)=rs1%omega(1:rs1%nvib)**2
 if(rs2%tused) omega2(rs1%nvib+1:nvib)=rs2%omega(1:rs2%nvib)**2

 ! partition function with an infinite number of images
  qrsi=0.D0
  zpers=0.D0
  do ivar=1,rs1%nvib
    ! the following is the limit for nimage->infinity
    ! for error compensation, however, we should use the same number of images as in the TS
    qrsi=qrsi-log(2.D0*sinh(rs1%omega(ivar)*beta_hbar*0.5D0))
    ! for the ZPE, we could do a projection rather than ignoring a few eigenvalues
    zpers=zpers+rs1%omega(ivar)*0.5D0
  end do
  ! now the same for rs2
  do ivar=1,rs2%nvib
    qrsi=qrsi-log(2.D0*sinh(rs2%omega(ivar)*beta_hbar*0.5D0))
    zpers=zpers+rs2%omega(ivar)*0.5D0
  end do
  if(printl>=4.and.hess_dtau) write(stdout,"('Ln Vibrational part of the RS partition &
      &function (infinite images) ',es18.9)") &
      qrsi+log(dtau)*dble(neb%nimage*neb%varperimage)

  ! partition function with nimage images (but equi-spaced tau)
  qrs=0.D0
  do iimage=1,2*neb%nimage
    do ivar=1,rs1%nvib
      qrs=qrs+log(4.D0*(sin(pi*dble(iimage)/dble(2*neb%nimage)))**2/dtau+&
          dtau*rs1%omega(ivar)**2)
    end do
    do ivar=1,rs2%nvib
      qrs=qrs+log(4.D0*(sin(pi*dble(iimage)/dble(2*neb%nimage)))**2/dtau+&
          dtau*rs2%omega(ivar)**2)
    end do
  end do

!!$  ! add zero modes of RS1 including their noise explicitly:
!!$  do iimage=1,2*neb%nimage-1 ! -1 because 6 eigenvalues must be left out!
!!$    do ivar=1,3+rs1%nmoi
!!$      qrs=qrs+log(4.D0*(sin(pi*dble(iimage)/dble(2*neb%nimage)))**2/dtau+&
!!$          dtau*rs1%omega2zero(ivar))
!!$    end do
!!$    ! RS2 is not included here because the instanton also has at most 6 zero modes.
!!$  end do
!!$  if(printl>=4) write(stdout,"('Zero-eigenvalues of RS1 are used in the product over zero modes')")
!!$
  ! alternatively: add contribution of zero modes
  qrs=qrs+2.D0*(log(dble(2*neb%nimage))*dble(glob%nzero) - &
      log(dtau)*0.5D0*dble(glob%nzero*(2*neb%nimage-1)))
  
  qrs=-0.5D0*qrs
  
  if(rs1%nmoi==2.and.glob%nzero==6.and..not.rs2%tused) then
    ! one more DOF in the TS than in the RS. The product of eigenvalues needs to account for that.
    ! product of all dtau
    svar=sum(log(qts%dtau(2:neb%nimage)))*2.D0
    svar=svar+log(qts%dtau(1))+log(qts%dtau(neb%nimage+1))
    qrs=qrs- svar
  end if
  
  if(printl>=4.and.hess_dtau) write(stdout,"('Ln Vibrational part of the RS partition &
      &function (nimage images)   ',es18.9)") qrs
  if(printl>=4.and.hess_dtau) write(stdout,"(' The latter is used')")

  if(.not.hess_dtau) then
    qrsi=qrsi+log(dtau)*dble(2*neb%nimage*(glob%nzero+rs1%nvib+rs2%nvib))
    if(printl>=4) write(stdout,"('Ln Vibrational part of the RS partition &
        &function (infinite images)                  ',es18.9)") qrsi

    ! transform the above value to what we would expect with the asymmetric dtau notation
    qrs=qrs+log(dtau)*dble(2*neb%nimage*(glob%nzero+rs1%nvib+rs2%nvib)-glob%nzero)*0.5D0

    if(printl>=4) write(stdout,"('Ln Vibrational part of the RS partition &
         &function (nimage images, equi-spaced tau)   ',es18.9)") qrs
    ! it turns out that the value obtained above is (slightly) different from
    ! the one obtained by diagonalising the Spring-matrix (for constant
    ! dtau). I guess, the analytic value is better. Maybe I should use that
    ! one?
    if(tconst_dtau) then
      if(printl>=4) write(stdout,"(' The latter is used')")
    else
      ! calculate the partition function for non-constant dtau
       ! construct hessian of only spring forces:
       call allocate(hess_sp,neb%nimage*2,neb%nimage*2)
      hess_sp(:,:)=0.D0 ! <-- todo: make that allocatable!
      ! convert the hessian of potential energies to one of action (modify
      ! diagonal, add spring contributions)
      call qts_hessian_etos(2,neb%nimage,1,qts%dtau,hess_sp)
      call allocate(eval_sp,neb%nimage*2)
      call allocate(evec_sp,neb%nimage*2,neb%nimage*2)
      call dlf_matrix_diagonalise(neb%nimage*2,hess_sp,eval_sp,evec_sp)
      call deallocate(evec_sp)
      call deallocate(hess_sp)
      !calculate qrs from eval_sp and eigvals
      qrs=0.D0
      do iimage=1,neb%nimage*2
        do ivar=1,rs1%nvib
          qrs=qrs+log(abs(eval_sp(iimage)+rs1%omega(ivar)**2))
       end do
        if(iimage>1) then
           do ivar=1,glob%nzero
              qrs=qrs+log(abs(eval_sp(iimage)))
           end do
        end if
        call deallocate(eval_sp)
        if(rs2%tused) call dlf_fail("Variable step size and bimolecular not implemented.")
        ! in that case one would have to deal withe the additional zero modes.
     end do
     qrs=-0.5D0*qrs

    end if ! (tconst_dtau) 

    ! correct for bimolecular rates (fewer degrees of freedom in the RS
    ! partition function than in the TS partition function)
    if(rs2%tused) then
      if(printl>=4) write(stdout,*) "Calculating a bimolecular rate."

      ! product of all dtau
      svar=sum(log(qts%dtau(2:neb%nimage)))*2.D0
      svar=svar+log(qts%dtau(1))+log(qts%dtau(neb%nimage+1))
      varperimage_read=glob%nzero+rs1%nvib+rs2%nvib
      qrs=qrs+ svar*(neb%varperimage-varperimage_read)  

      call dlf_constants_get("AMU",svar)
      phi_rel=mu_bim*svar/2.D0/pi/beta_hbar
      phi_rel=phi_rel*sqrt(phi_rel)
      if(printl>=4) write(stdout,"('ln translational partition function of incoming &
           &fragment                            ',es18.9)") -log(phi_rel)
      qrs=qrs+log(phi_rel)
      tbimol=.true.
   end if

   if(printl>=4) then
      if(tbimol) then
         write(stdout,"('ln vibrational and translational part of the RS partition &
              &function (nimage images)  ',es18.9)") qrs
      else
         write(stdout,"('ln vibrational part of the RS partition &
            &function (nimage images)                    ',es18.9)") qrs
      end if
    end if

 end if ! (.not.hess_dtau)
 tok=.true.

end subroutine qts_reactant
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/dlf_qts_convergence
!!
!! FUNCTION
!!
!! Test convergence of qts path in cartesian coordinates rather than internal
!! coordinates
!!
!! SYNOPSIS
subroutine dlf_qts_convergence!(testconv)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_neb, only: neb,unitp,xyzall,beta_hbar
  use dlf_global, only: glob
  use dlf_qts
  implicit none
  real(rk) :: xstep(3*glob%nat*neb%nimage)
  real(rk) :: xgradient(3*glob%nat*neb%nimage)
  integer :: iat,iimage

  ! testconv=.true.
  xstep=0.D0
  xgradient=0.D0

  do iimage=1,neb%nimage
    call dlf_cartesian_itox(glob%nat,neb%varperimage,neb%varperimage,glob%massweight,&
        glob%step(neb%cstart(iimage):neb%cend(iimage)),&
        xstep( (iimage-1)*3*glob%nat+1 : iimage*3*glob%nat ) )
    call dlf_cartesian_gradient_itox(glob%nat,neb%varperimage,0,glob%massweight,&
        glob%igradient(neb%cstart(iimage):neb%cend(iimage)),&
        xgradient( (iimage-1)*3*glob%nat+1 : iimage*3*glob%nat ) )
!!$    do iat=1,glob%nat
!!$      write(*,'("grad",i3,3f15.6)') iimage,xgradient((iimage-1)*3*glob%nat+1 + (iat-1)*3:&
!!$          (iimage-1)*3*glob%nat + (iat)*3) 
!!$    end do
!!$    do iat=1,glob%nat
!!$      write(*,'("step",i3,3f15.6)') iimage,xstep(    (iimage-1)*3*glob%nat+1 + (iat-1)*3:&
!!$          (iimage-1)*3*glob%nat + (iat)*3) 
!!$    end do

  end do

  ! to be written
  call convergence_set_info("in cartesian coordinates",3*glob%nat*neb%nimage,qts%S_ins/beta_hbar, &
      xgradient,xstep)

end subroutine dlf_qts_convergence
!!****

! now get-routines so that other files don't have to use the qts module.
subroutine dlf_qts_get_int(label,val)
  use dlf_qts
  implicit none
  character(*), intent(in)  :: label
  integer     , intent(out) :: val
  !
  if (label=="TASKFARM_MODE") then
    val=taskfarm_mode
  else if (label=="IMAGE_STATUS") then
    val=qts%image_status
  else
    call dlf_fail("Wrong label in dlf_qts_get_int")
  end if

end subroutine dlf_qts_get_int


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/dlf_htst_rate
!!
!! FUNCTION
!!
!! Calculate a reaction rate based on energies and Hessians at a
!! minimum and a first order saddle point. No energy calculations are
!! performed, only the files qts_reactant.txt (or qts_hessian_rs.txt)
!! and qts_ts.txt (or qts_hessian_ts.txt) are used.
!!
!! The reaction rate is calculated in the following ways:
!! * Classical
!! * Classical with quantised vibrations (includes ZPE)
!! *  The latter + Bell correction
!! * A symmetric Eckart barrier fitted to the hight of the barrier and
!!   the curvature. The analytic quantum mechanical solution for the
!!   flux is used in the reaction coordinate. The harmonic
!!   approximation (quantum) is used perpendicular to that.
!!
!! SYNOPSIS
subroutine dlf_htst_rate
  !! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl,pi
!  use dlf_allocate, only: allocate,deallocate
  use dlf_qts, only: qts,dbg_rot,pf_type,minfreq
  use dlf_allocate, only: allocate,deallocate
  use dlf_constants, only: dlf_constants_init,dlf_constants_get
  implicit none
  real(rk)  :: tstart ! lower end of the temperature range
  real(rk)  :: tend  ! upper end of the temperature range
  integer   :: tsteps ! number of temperatures at which to calculate the rates
  real(rk)  :: mu_bim  ! reduced mass (amu) for calculating phi_rel
  real(rk)  :: phi_rel  ! relative translational p.fu. per unit volume

  real(rk)  :: ZPE,VAD
  logical   :: tok
  real(rk)  :: lrate_cl ! log of rate
  real(rk)  :: lrate_qvib! log of qq-rate (HJ-notation) multiplied by (2pi)**3
                        ! to make it consistent with the classical rate
  real(rk)  :: kryv_lrate,qrsi
  real(rk)  :: temp,beta_hbar,KBOLTZ_AU,svar
  real(rk)  :: tdel
  integer   :: itemp,ivar,iter
  real(rk)  :: second_au
  real(rk)  :: log10,timeunit

  logical   :: tkie
  logical   :: chkfilepresent
  real(rk)  :: hcl,hqq,hbell
  character(256) :: sline,line
  integer   :: linlen
  real(rk)  :: avogadro,hartree,kjmol,echarge,ev,amc,planck,kboltz
  real(rk)  :: zpe_rs,zpe_ts
  integer   :: nimage_read,iat,ios
  real(rk)  :: temperature!,etunnel,dist(2)
  logical   :: teck=.true.
  real(rk)  :: Vb,alpha
  real(rk)  :: kappa_eck,heck,kappa_aeck,va
  real(rk)  :: wavenumber,frequency_factor,amu,wavenumberts
  real(rk)  :: bvar
  character(128) :: filename
  real(rk)  :: l_qrot,beta,alpha_bell,kappa0,wkbbell,kappa_bell,prod_sinh
  logical   :: taeck=.true.
  real(rk)  :: vb_aeck,alpha_aeck,vmax,xvar,yvar,eharm,esym,easym
  type(pf_type) :: rs1,rs2,ts
  logical   :: toneat
  real(rk)  :: qrot,qrot_quant,qrot_part,qrot_quant_part
  real(rk)  :: sigrs,dsigrs,beta_crit
  ! print one or more rates versions for T>Tc
  real(rk),allocatable :: rate_data(:,:)
  real(rk),allocatable :: SCT_kappas(:)

  if(glob%iam > 0 ) return ! only task zero should do this (task farming does
                           ! not make sense for such fast calculations. No
                           ! energies and gradients are ever calculated).

  if(glob%icoord==190) then ! for now: iopt=13 and icoord=190 -> micro, not classical
     call dlf_microcanonical_rate
     return
  end if
  
  ! some small header/Info
  if(printl>=2) then
    write(stdout,'(a)') "Calculating the reaction rate based on harmonic &
        &transition state theory and one-dimensional tunnelling corrections."
  end if

  ! initialise
  rs1%tused=.false.
  rs2%tused=.false.
  rs2%ene=0.D0
  rs2%nvib=0
  rs2%nat=0
  rs2%mass=0.D0
  ts%tused=.false.
  
  !
  ! read class.in or class.auto (if class.in does not exist)
  !

  ! the file is structured as
  ! line 1: minfreq in cm^-1
  ! line 2: dEb/dbeta at TS ! was: n0_r, n0_ts ! <-- not used any more
  ! line 3: tstart, tend, tsteps
  ! line 4: bimolecular (T/F) ! <-- not used any more
  ! line 5: exothermicity for asymmetric Eckart approximation (in Hartree)
  filename="class.in"
  if (glob%ntasks > 1) filename="../"//trim(filename)
  INQUIRE(FILE=filename,EXIST=chkfilepresent)
  if (.not.chkfilepresent) THEN
    filename="class.auto" ! if class.in exists, it is used, otherwise
                          ! class.auto, written by the ChemShell
                          ! interface, is used
    if (glob%ntasks > 1) filename="../"//trim(filename)
    INQUIRE(FILE=filename,EXIST=chkfilepresent)
  end if
  IF (chkfilepresent) THEN
    OPEN(28,FILE=filename,STATUS='old',ACTION='read')
    READ (28,'(a)') sline ! line 1
    READ (sline,iostat=ios,fmt=*) minfreq
    if(ios/=0) then
       minfreq=-1.D0 ! switch it off
    end if
!    varperimager=natr*3
!   varperimage=nat*3
    READ (28,'(a)') sline  ! line 2
    READ (sline,iostat=ios,fmt=*) qts%debdbeta
    if(ios/=0) then
       qts%debdbeta=1.D0 ! switch it off
    end if
    
!    READ (28,*) nzero_r, nzero_ts
    READ (28,*) tstart,tend,tsteps
    taeck=.false.
    READ (28,'(a)',err=201,end=201) sline
    READ (28,'(a)',err=201,end=201) sline
    read(sline,fmt=*,err=201,end=201) va
    va=-va
    taeck=.true.
201 continue
    REWIND(28)
    CLOSE(28)
  ELSE
    if(printl>=0) then
      write(stdout,*) " Read Error: file class.in not found"
      write(stdout,*) " The file provides input for rate calculations and should contain the lines:"
      write(stdout,*) " Line 1: Minimum wave number (in cm^-1, smaller wave numbers will be raised to that)"
      write(stdout,*) " Line 2: dEb/dbeta at TS"
      write(stdout,*) " Line 3: T_start T_end Number_of_steps ! Temperature"
      write(stdout,*) " Line 4: not used"
      write(stdout,*) " Line 5: the exothermicity in Hartree if asymmetric Eckart is requested"
    end if
    call dlf_fail("File class.in missing")
  END IF
  
  if(minfreq>0.D0) then
    call dlf_constants_get("CM_INV_FOR_AMU",svar)
    if(printl>=2) write(stdout,'("Vibrational frequencies are raised &
        &to a minimum of ",f8.3," cm^-1")') minfreq
    minfreq=minfreq/svar ! now it should be in atomic units
  end if
  
  toneat=.false.
  ! read first reactant
  call read_qts_txt(1,0,rs1,toneat)
  
  if(rs1%nat<glob%nat) then
    ! read second reactant
    call read_qts_txt(2,rs1%nat,rs2,toneat)
  end if
  
  ! read transition state
  toneat=.false.
  call read_qts_txt(3,0,ts,toneat)
  
  ! sanity checks
  if(abs(rs1%mass+rs2%mass-ts%mass)>1.D-4) then
    write(stdout,'("Masses: ",3f10.5)') rs1%mass,rs2%mass,ts%mass
    call dlf_fail("Masses of reactants and transition state not equal!")
  end if
  if(rs1%nat+rs2%nat/=ts%nat) then
    write(stdout,'("Number of atoms: ",3i6)') rs1%nat,rs2%nat,ts%nat
    call dlf_fail("Number of atoms of reactants and transition state not equal!")
  end if

  !
  ! now calculate rate constants for each temperature
  !

  call dlf_constants_get("SECOND_AU",second_au)
  call dlf_constants_get("HARTREE",HARTREE)
  call dlf_constants_get("AVOGADRO",avogadro)
  call dlf_constants_get("ECHARGE",echarge)
  call dlf_constants_get("CM_INV_FOR_AMU",frequency_factor)
  call dlf_constants_get("AMU",amu)
  call dlf_constants_get("KBOLTZ_AU",KBOLTZ_AU)
  kjmol=avogadro*hartree*1.D-3
  ev=hartree/echarge

  temp=tstart
  tdel=(tstart-tend)/(tend*tstart*dble(tsteps-1))
  !print*,"tdel",tdel
  
  ! calculate the transmission coefficients for SCT
  if (glob%sctRate) then
    allocate(SCT_kappas(tsteps))
    call dlf_sct_main(tstart,tdel,tsteps,SCT_kappas)
  end if
  
  call allocate(rate_data,tsteps,2)

  zpe_rs=0.D0
  zpe_ts=0.D0
  do ivar=1,rs1%nvib
    zpe_rs = zpe_rs + 0.5D0*rs1%omega(ivar)
  end do
  if(rs2%tused) then
    do ivar=1,rs2%nvib
      zpe_rs = zpe_rs + 0.5D0*rs2%omega(ivar)
    end do
  end if
  do ivar=1,ts%nvib
    zpe_ts = zpe_ts + 0.5D0*ts%omega(ivar)
  end do
  
  if(printl>=4) then
    write(stdout,"('Zero-point energy',3x,f10.5,15x,f10.5)") zpe_rs,zpe_ts
  end if

  
  ZPE=zpe_ts-zpe_rs 
  VAD=ts%ene-rs1%ene-rs2%ene+ZPE

  ! crossover temperature
  qts%tcross=ts%omega_imag*0.5D0/pi/KBOLTZ_AU

  ! Eckart stuff
  !Vb=4.D0*(ene_ts-ene_rs)
  Vb=4.D0*(vad)
  alpha=ts%omega_imag*sqrt(8.D0/vb)

  ! Asymmetric Eckart barrier
  vmax=vad

  if(taeck.and.printl>=4) then
    vb_aeck=2.D0*vmax-va+sqrt((2.D0*vmax-va)**2-va**2)
    alpha_aeck=ts%omega_imag*sqrt(8.D0*vb_aeck**3/(va**2-vb_aeck**2)**2)
    ! plot curve
    open(unit=125,file='ene_profile')
    write(125,'(a)') "# Reaction coordinate in mass-weighted coordinates."
    write(125,'(a)') "# Reaction coord.      E_harm       symm. Eckart   asymm. Eckart"
    do itemp=1,101
      xvar=10.D0*(-1.D0+dble(itemp-1)/50.D0)/alpha_aeck
      eharm=vmax-ts%omega_imag**2*0.5D0*xvar**2
      ! symm
      yvar=exp(alpha*xvar)
      esym=vb*yvar/(1.D0+yvar)**2
      ! asym
      yvar=(va+vb_aeck)/(vb_aeck-va)*exp(alpha_aeck*xvar)
      easym=va*yvar/(1.d0+yvar)+vb_aeck*yvar/(1.d0+yvar)**2
      write(125,'(4f15.7)') xvar,eharm,esym,easym
    end do
    close(125)
  end if
  
  
  if(printl>=3) then
    write(stdout,'(a)') "                                      Hartree           kJ/mol            eV                K"
    write(stdout,'(a,4f18.8)') "Potential energy Barrier     ", &
        ts%ene-rs1%ene-rs2%ene,(ts%ene-rs1%ene-rs2%ene)*kjmol,  &
        (ts%ene-rs1%ene-rs2%ene)*ev,(ts%ene-rs1%ene-rs2%ene)/KBOLTZ_AU
    write(stdout,'(a,4f18.8)') "ZPE Correction               ", &
        ZPE, ZPE*kjmol,ZPE*ev,ZPE/KBOLTZ_AU
    write(stdout,'(a,4f18.8)') "Vibrational adiabatic barrier", &
        VAD,VAD*kjmol,VAD*ev,VAD/KBOLTZ_AU
    ! qrot = qrot_red * beta_hbar**rot_beta_exp
    ! todo:
!    svar=-(KBOLTZ_AU*temp)* (log(qrot_red) - rot_beta_exp*log(KBOLTZ_AU*temp) )
!    write(stdout,'(a,4f18.8)') "Rotational contr. at start T ", &
!        svar,svar*kjmol,svar*ev,svar/KBOLTZ_AU


    write(stdout,'(a,f18.8,1x,a)') "Crossover Temperature        ",qts%tcross,"K"
    beta_crit=1.d0/qts%tcross/kboltz_au
    write(stdout,'(a,f18.8,1x,a)') "beta_c                       ",beta_crit,"1/Hartree"
  end if

  if(printl>=2.and.glob%irot==1) then
    if(rs2%tused) then
      write(stdout,'("Implicit surface approach: only rotation of &
          &RS2 considered, reduced mass = mass of RS2")')
    else
      if(printl>=2) write(stdout,'("Implicit surface approach: rotational partition function kept constant.")')
    end if
  end if

  if(rs2%tused) then
    call dlf_constants_get("AMU",svar)
    ! define the reduced mass
    mu_bim=rs1%mass*rs2%mass/(rs1%mass+rs2%mass)
    if(rs2%tfrozen) mu_bim=rs1%mass
    if(rs1%tfrozen) mu_bim=rs2%mass
    if(rs1%tfrozen.and.rs2%tfrozen) mu_bim=rs1%mass*rs2%mass/(rs1%mass+rs2%mass)
    !if(rs2%nat==1) mu_bim=rs2%mass ! needs to be removed: rs2%mass is the real mass of the additional atom
    if(glob%irot==1) mu_bim=rs2%mass ! mimic a surface
    svar=mu_bim*svar/2.D0/pi*(KBOLTZ_AU*temp)
    svar=-(KBOLTZ_AU*temp)* (-1.5D0*log(svar))
    if(printl>=3) write(stdout,'(a,4f18.8)') "Transl. contr. at start T    ", &
        svar,svar*kjmol,svar*ev,svar/KBOLTZ_AU
    if(printl>=4) write(stdout,'("Reduced mass =",f10.5," amu")') mu_bim
  end if
  
  if(taeck.and.printl>=2) write(stdout,'("Asymmetric Eckart Approximation is used &
      &with an exothermicity of ",f15.8," Hartree")') -va 
  ! Parameters for the Eckart approximation (commented out)
  !print*,"Eckart: V'',Emax",abs(eigvals_ts(1)),(ene_ts-ene_rs)
  !print*,"Eckart: Vb, alpha",Vb,alpha

  if(printl>=4) then
     if(qts%dEbdbeta>=0.D0) then
        call dlf_kryvohuz_supercrit_jk(ts%nvib,-1,rs1%ene+rs2%ene,ts%ene,&
             qrsi,ts%omega**2,beta_hbar,kryv_lrate,.true.)
        write(stdout,'("Using dE_b / dbeta from an inverted Morse potential= ",es10.3)') kryv_lrate
     else
        write(stdout,'("Using dE_b / dbeta from input= ",es10.3)') qts%debdbeta
     end if
  end if
  
  ! Set the unit and the base of the logarithm
  log10=log(10.D0)        ! for log to the base of 10
  timeunit=log(second_au) ! for printout in s^-1
  if(rs2%tused) then
    call dlf_constants_get("ANG_AU",svar)
    svar=log(svar*1.D-10) ! ln of bohr in meters
    ! the factor of 1.D6 converts from m^3 to cm^3
    timeunit=log(second_au)+log(1.D6)+3.D0*svar ! for printout in cm^3 s^-1
  end if
  !log10=1.D0              ! for natural log
  !timeunit=0.D0           ! for printout in at. u.

  ! Adjust the printout to the settings above!
  if(printl>=3) then
    if (rs2%tused) then
      write(stdout,'("The following list contains the log_10 of rate &
          &constants in cm^3 sec^-1.")')
      !write(stdout,*) "log_10 of rates in cm^3/at.u."
      !write(stdout,*) "ln of rates in cm^3 sec^-1"
      !write(stdout,*) "ln of rates in cm^3/at.u."
    else
      write(stdout,'("The following list contains the log_10 of rate &
          &constants in second^-1.")')
      !write(stdout,*) "ln of rates in second^-1"
      !write(stdout,*) "log_10 of rates in at.u."
      !write(stdout,*) "ln of rates in at.u."
    end if
    write(stdout,'("All rate constants still need to be multiplied &
        &by the symmetry factor sigma.")')
    write(stdout,'("Rotational partition functions were calculated for&
        & the classical rigid rotor.")')
    write(stdout,'("To obtain results for the quantum rigid rotor, add &
        & values from the column `Q_rot class/quant` in the file arrhenius.")')
    !write(stdout,'(a,f18.8)') "Change of log(rate) by the rotational partition function (start T)",(log(qrot_red)- rot_beta_exp*log(KBOLTZ_AU*temp))/log10

  end if
  
  filename="rate_H"
  if (glob%ntasks > 1) filename="../"//trim(filename)
  inquire(file=filename,exist=tkie)
  if(tkie) then
    open(unit=12,file=trim(filename))
    do
      read(12,FMT="(a)",ERR=1002,END=1002) sline
      if(index(sline,"#")==0) exit
    end do
    goto 1003
1002 tkie=.false.
1003 continue
    if(tkie) then
      if(printl>=4) write(stdout,'("File rate_H found, calculating KIE and writing to file kie.")')
      open(file="kie",unit=13)
      write(13,'(a)') "#       T[K]              KIE  classical   KIE w. quant. vib   KIE Bell            KIE Eckart"
    else
      if(printl>0) write(stdout,*) "File rate_H found, but not readable, no KIE calculated."
    end if
  end if

  call dlf_constants_get("AMC",amc)
  call dlf_constants_get("PLANCK",planck)
  call dlf_constants_get("KBOLTZ",kboltz)

  open(file="arrhenius",unit=15)
  !open(file="free_energy_barrier",unit=16)
  open(file="arrhenius_components",unit=17)
  line="       1000/T           rate classical       quantised vib.      Bell                Eckart           "
  linlen=100
  if (glob%sctRate) then
    line=line(1:linlen)//"     SCT rate     "
    linlen=linlen+20
  end if
  if (taeck) then
    line=line(1:linlen)//"     asymm. Eckart"
    linlen=linlen+20
  end if
  !line=line(1:linlen)//"   Q_rot class/quant"
  !linlen=linlen+20
  write(15,'(a)') "#"//line(2:linlen)//"   Q_rot class/quant"
  line=line(1:linlen)//"   reduced instanton"
  if(printl>=2) write(stdout,'(a)') trim(line)
  !write(16,'(a)') &
  !    "#     T           free-energy barrier from rate with quantised vib."

  if(.not.taeck) then
    if (glob%sctRate) then
      write(17,'(a)') &
        "#      1000/T         1/(2 pi beta hbar)   prod sinh(...)     rotation           translation       exp(-beta dE)   &
        &    kappa_bell         kappa_WKBbell      kappa_Eckart       kappa_SCT          timeunit"
    else
      write(17,'(a)') &
        "#      1000/T           1/(2 pi beta hbar)    prod sinh(...)       rotation            translation       exp(-beta dE)   &
        &     kappa_bell          kappa_WKBbell       kappa_Eckart        timeunit"
    end if
  else
    ! output with asymmetric Eckart rates
    if (glob%sctRate) then
      write(17,'(a)') &
        "#      1000/T         1/(2 pi beta hbar)   prod sinh(...)     rotation           translation       exp(-beta dE)   &
        &    kappa_bell         kappa_WKBbell      kappa_sEckart      kappa_aEckart      kappa_SCT          timeunit"
    else
      write(17,'(a)') &
        "#      1000/T           1/(2 pi beta hbar)    prod sinh(...)       rotation            translation       exp(-beta dE)   &
        &     kappa_bell          kappa_WKBbell       kappa_sEckart       kappa_aEckart       timeunit"
    end if
  end if
  do itemp=1,tsteps

    !print*,"Temperature",itemp,temp
    beta_hbar=1.D0/(temp*KBOLTZ_AU)  ! beta * hbar = hbar / (kB*T)
    !print*,"beta_hbar",beta_hbar

    !
    ! calculate the rotational partition function
    !
    qrot=1.D0
    qrot_quant=1.D0
    ! do nothing for atoms (their rotational partition function is unity)
    if(glob%irot==0) then
      if(rs1%nmoi>1) then
        !print*,"rotpart rs1",rs1%coeff
        call rotational_partition_function_calc(rs1%moi,beta_hbar,rs1%coeff,qrot_part,qrot_quant_part)
        qrot=qrot*qrot_part
        qrot_quant=qrot_quant*qrot_quant_part
      end if
      if(ts%nmoi>1) then
        !print*,"rotpart TS",ts%coeff
        call rotational_partition_function_calc(ts%moi,beta_hbar,ts%coeff,qrot_part,qrot_quant_part)
        qrot=qrot/qrot_part
        qrot_quant=qrot_quant/qrot_quant_part
      end if
      
!!$      if(rs1%nmoi==3) then
!!$        qrot=qrot*sqrt(rs1%moi(1)*rs1%moi(2)*rs1%moi(3))*sqrt(8.D0*pi/beta_hbar**3)
!!$      else if (rs1%nmoi==2) then
!!$        qrot=qrot*maxval(rs1%moi)*2.D0/beta_hbar ! the two moi must be equal
!!$      end if
!!$      if(ts%nmoi==3) then
!!$        qrot=qrot/(sqrt(ts%moi(1)*ts%moi(2)*ts%moi(3))*sqrt(8.D0*pi/beta_hbar**3))
!!$      else if (ts%nmoi==2) then
!!$        qrot=qrot/(maxval(ts%moi)*2.D0/beta_hbar) ! the two moi must be equal
!!$      end if
    end if
    !print*,"rotpart rs2"
    if(rs2%tused.and.rs2%nmoi>1) then
      !print*,"rotpart rs2",rs2%coeff
      call rotational_partition_function_calc(rs2%moi,beta_hbar,rs2%coeff,qrot_part,qrot_quant_part)
      qrot=qrot*qrot_part
      qrot_quant=qrot_quant*qrot_quant_part
    end if
!!$      if(rs2%nmoi==3) then
!!$        qrot=qrot*sqrt(rs2%moi(1)*rs2%moi(2)*rs2%moi(3))*sqrt(8.D0*pi/beta_hbar**3)
!!$      else if (rs2%nmoi==2) then
!!$        qrot=qrot*maxval(rs2%moi)*2.D0/beta_hbar ! the two moi must be equal
!!$      end if
!!$    end if
    l_qrot= -log(qrot) ! minus because of Q_TS/Q_RS

    ! translational partition function
    phi_rel=0.D0
    if(rs2%tused) then
      call dlf_constants_get("AMU",svar)
      phi_rel=mu_bim*svar/2.D0/pi/beta_hbar
      phi_rel=1.5D0 * log(phi_rel)
    end if

    lrate_cl=0.D0
    lrate_qvib=0.D0
    do ivar=1,rs1%nvib
      lrate_cl=lrate_cl+log(rs1%omega(ivar))
      lrate_qvib=lrate_qvib+log(2.D0*sinh(0.5D0*rs1%omega(ivar)*beta_hbar))
    end do
    if(rs2%tused) then
      do ivar=1,rs2%nvib
        lrate_cl=lrate_cl+log(rs2%omega(ivar))
        lrate_qvib=lrate_qvib+log(2.D0*sinh(0.5D0*rs2%omega(ivar)*beta_hbar))
      end do
    end if
    do ivar=1,ts%nvib
      lrate_cl=lrate_cl-log(ts%omega(ivar))
      lrate_qvib=lrate_qvib-log(2.D0*sinh(0.5D0*ts%omega(ivar)*beta_hbar))
    end do
    prod_sinh=lrate_qvib

    ! pre-factors for the rate
    lrate_cl= lrate_cl -log(2.D0*pi) + ( -beta_hbar * (ts%ene-rs1%ene-rs2%ene) )
    lrate_qvib= lrate_qvib -log(2.D0*pi*beta_hbar) + ( -beta_hbar * (ts%ene-rs1%ene-rs2%ene) )

    ! rotational part:
    lrate_cl=lrate_cl+l_qrot
    lrate_qvib=lrate_qvib+l_qrot
    if(dbg_rot) print*,"ln(Quotient of Rotational partition functions):",l_qrot

    ! translational part:
    if (rs2%tused) then
      lrate_cl=lrate_cl-phi_rel 
      lrate_qvib=lrate_qvib-phi_rel 
    end if
 
    !wigner_simple=1.D0+1.D0/24.D0*beta_hbar**2*abs(eigvals_ts(1))
    
    call kappa_eckart(beta_hbar,Vb,alpha,kappa_eck)
    if(taeck) call kappa_eckart_asymm(beta_hbar,vmax,va,alpha_aeck,kappa_aeck)
    
    !
    ! full Bell
    !
    bvar=ts%omega_imag*0.5D0/pi*beta_hbar ! mu
  !  alpha_bell=(ts%ene-rs1%ene-rs2%ene)*beta_hbar
  !  beta=(ts%ene-rs1%ene-rs2%ene)/ (sqrt(abs(eigvals_ts(1)))*0.5D0/pi)
    alpha_bell=(vad)*beta_hbar
    beta=(vad)/ (ts%omega_imag*0.5D0/pi)
    kappa0=pi*bvar/sin(pi*bvar)

    svar=0.D0
    do iter=0,20 ! 20 iterations should be enough, unless one goes to _very_ low T

      ! lift poles at T = T_c / n (with n integer > 0)
      if(abs(1.D0+dble(iter)-bvar)< 1.D-5) then
        kappa0=0.D0
        svar=svar + (-1.D0)**(1+iter)*(alpha_bell ) / (bvar*exp(alpha_bell-beta))
      else
        svar=svar + dble((-1)**iter) /(1.D0+dble(iter)-bvar) * exp(-dble(iter)*beta)
      end if
    end do
    kappa_bell=kappa0-bvar*exp(alpha_bell-beta)*svar
    ! calculate log(kappa_bell) for low temperature (to make it more stable)
    if( (alpha_bell-beta)>700.D0 ) then
      ! neglect kappa0
      kappa_bell=log(abs(bvar))+(alpha_bell-beta)+log(abs(svar))
    else
      kappa_bell=log(kappa_bell)
    end if

    ! calculate reduced instanton rate > Tc
    sigrs=0.D0
    dsigrs=0.D0
    do ivar=1,rs1%nvib
      sigrs=sigrs-log(2.D0*sinh(0.5D0*rs1%omega(ivar)*beta_hbar))
      dsigrs=dsigrs-rs1%omega(ivar)*0.5D0
    end do
    if(rs2%tused) then
      do ivar=1,rs2%nvib
        sigrs=sigrs-log(2.D0*sinh(0.5D0*rs2%omega(ivar)*beta_hbar))
        dsigrs=dsigrs-rs2%omega(ivar)*0.5D0
      end do
    end if
    dsigrs=dsigrs*beta_hbar
    beta_crit=1.d0/qts%tcross/kboltz_au
    qrsi=sigrs ! purely sinh-expression
    ! comment out the next expression if you want the standard partition
    ! function of the reactant!
!!$    qrsi=((beta_crit-beta_hbar)**0.5D0/beta_crit**0.5D0) * sigrs + &
!!$        (1.d0-(beta_crit-beta_hbar)**0.5D0/beta_crit**0.5D0) * dsigrs
    call dlf_kryvohuz_supercrit_jk(ts%nvib,-1,rs1%ene+rs2%ene,ts%ene,&
        qrsi,ts%omega**2,beta_hbar,kryv_lrate,.false.)
    kryv_lrate=kryv_lrate+l_qrot
    if (rs2%tused) then
       kryv_lrate=kryv_lrate-phi_rel
    end if
    
    ! print result
    line=""
    linlen=0
    ! this is written in all cases (file arrhenius):
    write(line,'(5f20.12)') 1000.D0/temp,(lrate_cl+timeunit)/log10,&
        (lrate_qvib+timeunit)/log10, &
        (lrate_qvib+kappa_bell+timeunit)/log10, (lrate_qvib+kappa_eck+timeunit)/log10
    linlen=100
    ! add asymmetric Eckart if available
    if(taeck) then
      write(line,'(a,f20.12)') line(1:linlen), (lrate_qvib+kappa_aeck+timeunit)/log10
      linlen=120
    end if
    
    ! Add SCT rate, if requested
    if (glob%sctRate) then
      write(line,'(a,f20.12)') line(1:linlen), (lrate_qvib+log(SCT_kappas(itemp))+timeunit)/log10
      linlen=linlen+20
    end if

    write(15,'(a,f20.12)') trim(line),log(qrot/qrot_quant)/log10 ! file arrhenius

    ! add reduced instanton above Tc
    if(temp>qts%tcross) then
      write(line,'(a,f20.12)') line(1:linlen), (kryv_lrate+timeunit)/log10
    end if

    if(printl>=2) write(stdout,'(a)') trim(line)
    rate_data(itemp,1)=1000.D0/temp
    rate_data(itemp,2)=(kryv_lrate+timeunit)/log10
!!$    if(.not.taeck) then
!!$      if(printl>=2) write(stdout,'(6f20.12)') &
!!$          1000.D0/temp,(lrate_cl+timeunit)/log10,(lrate_qvib+timeunit)/log10, &
!!$          (lrate_qvib+kappa_bell+timeunit)/log10, (lrate_qvib+kappa_eck+timeunit)/log10,&
!!$          (kryv_lrate+timeunit)/log10
!!$      !    write(15,'(5f20.12)') beta_hbar,(lrate_cl+timeunit)/log10,(lrate_qvib+timeunit)/log10, &
!!$      ! should be that:
!!$   !        write(15,'(6f20.12)') 1000.D0/temp,(lrate_cl+timeunit)/log10,(lrate_qvib+timeunit)/log10, &
!!$   !        (lrate_qvib+kappa_bell+timeunit)/log10, (lrate_qvib+kappa_eck+timeunit)/log10,&
!!$   !        (kryv_lrate+timeunit)/log10
!!$      write(15,'(2f20.12)') 1000.D0/temp,          (kryv_lrate+timeunit)/log10
!!$    else
!!$      ! printout with asymmetric Eckart
!!$      if(printl>=2) write(stdout,'(6f20.12)') &
!!$          1000.D0/temp,(lrate_cl+timeunit)/log10,(lrate_qvib+timeunit)/log10, &
!!$          (lrate_qvib+kappa_bell+timeunit)/log10, (lrate_qvib+kappa_eck+timeunit)/log10,&
!!$          (lrate_qvib+kappa_aeck+timeunit)/log10
!!$      !    write(15,'(5f20.12)') beta_hbar,(lrate_cl+timeunit)/log10,(lrate_qvib+timeunit)/log10, &
!!$      write(15,'(6f20.12)') 1000.D0/temp,(lrate_cl+timeunit)/log10,(lrate_qvib+timeunit)/log10, &
!!$          (lrate_qvib+kappa_bell+timeunit)/log10, (lrate_qvib+kappa_eck+timeunit)/log10,&
!!$          (lrate_qvib+kappa_aeck+timeunit)/log10
!!$    end if
    !write(16,'(2f20.12)') temp,-(lrate_qvib+log(beta_hbar*2.D0*pi))/beta_hbar ! was the file free_energy_barrier
    
    ! tcross=ts%omega_imag*0.5D0/pi/KBOLTZ_AU
    ! beta_hbar=1.D0/(temp*KBOLTZ_AU)  ! beta * hbar = hbar / (kB*T)
    !wkbbell=(temp-tcross * exp( (ts%ene-rs1%ene-rs2%ene)*(beta_hbar-2.D0*pi/ts%omega_imag )))/(temp-tcross)
    wkbbell=(temp-qts%tcross * exp( vad*(beta_hbar-2.D0*pi/ts%omega_imag )))/(temp-qts%tcross)

    ! write components of rate expressions (file arrhenius_components)
    if(.not.taeck) then
      if (glob%sctRate) then
        write(17,'(11f19.12)') 1000.D0/temp, &
          (-log(2.D0*pi*beta_hbar))/log10,&
          prod_sinh/log10,&
          (l_qrot)/log10,&
          (-phi_rel)/log10,&
          (-beta_hbar * (ts%ene-rs1%ene-rs2%ene))/log10,&
          (kappa_bell)/log10,&
          (log(wkbbell))/log10,&
          kappa_eck/log10,&
          log(SCT_kappas(itemp))/log10,&
          timeunit/log10
      else
        write(17,'(10f20.12)') 1000.D0/temp, &
          (-log(2.D0*pi*beta_hbar))/log10,&
          prod_sinh/log10,&
          (l_qrot)/log10,&
          (-phi_rel)/log10,&
          (-beta_hbar * (ts%ene-rs1%ene-rs2%ene))/log10,&
          (kappa_bell)/log10,&
          (log(wkbbell))/log10,&
          kappa_eck/log10,&
          timeunit/log10
      end if
    else
      if (glob%sctRate) then
        write(17,'(12f19.12)') 1000.D0/temp, &
          (-log(2.D0*pi*beta_hbar))/log10,&
          prod_sinh/log10,&
          (l_qrot)/log10,&
          (-phi_rel)/log10,&
          (-beta_hbar * (ts%ene-rs1%ene-rs2%ene))/log10,&
          (kappa_bell)/log10,&
          (log(wkbbell))/log10,&
          kappa_eck/log10,&
          kappa_aeck/log10,&
          log(SCT_kappas(itemp))/log10,&
          timeunit/log10
      else
        write(17,'(11f20.12)') 1000.D0/temp, &
          (-log(2.D0*pi*beta_hbar))/log10,&
          prod_sinh/log10,&
          (l_qrot)/log10,&
          (-phi_rel)/log10,&
          (-beta_hbar * (ts%ene-rs1%ene-rs2%ene))/log10,&
          (kappa_bell)/log10,&
          (log(wkbbell))/log10,&
          kappa_eck/log10,&
          kappa_aeck/log10,&
          timeunit/log10
      end if
    end if
        
    if(tkie) then
      read(sline,'(5f20.12)') svar,hcl,hqq,hbell,heck
      write(13,'(5g20.12)') temp,exp(hcl*log10-(lrate_cl+timeunit)), &
          exp(hqq*log10-(lrate_qvib+timeunit)),exp(hbell*log10-(lrate_qvib+kappa_bell+timeunit)),&
          exp(heck*log10-(lrate_qvib+kappa_eck+timeunit))
      if(abs(svar-1000.D0/temp) > 1.D-6) then
        write(13,*) "#Error: temperatures don't match. Here: ",temp,", rate_H:",1000.D0/svar
        close(13)
        close(12)
        tkie=.false.
      end if
      if(tkie.and.itemp<tsteps) read(12,FMT="(a)") sline
    end if

    temp=temp/(1.D0+temp*tdel)
  end do

  ! now write reduces instanton data above Tc
  write(15,*)
  write(15,'(a)') "#      1000/T          reduced instanton"
  do itemp=1,tsteps
    if(1000.D0/rate_data(itemp,1) > qts%tcross) then
      write(15,'(2f20.12)') rate_data(itemp,:)
    end if
  end do

  if(tkie) then
    close(13)
    close(12)
  end if

  close(15)
  !close(16)
  close(17)

  ! reset variables (just in case ...)
  if(rs1%tused) call deallocate(rs1%omega)
  if(rs2%tused) call deallocate(rs2%omega)
  if(ts%tused) call deallocate(ts%omega)
  rs1%tused=.false.
  rs2%tused=.false.
  ts%tused=.false.
  call deallocate (rate_data)
  if (glob%sctRate) then
    deallocate(SCT_kappas)
  end if

end subroutine dlf_htst_rate
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/rotational_partition_function
!!
!! FUNCTION
!!
!! Calculate the relative rotational partition function for one
!! geometry. Return 1 for any frozen atoms.
!!
!! INPUTS
!!
!! nzero, xcoords, mass, beta_hbar
!!
!! OUTPUTS
!! 
!! qrot,qrot_quant,moi 
!!
!! SYNOPSIS
! rotational partition function
subroutine rotational_partition_function(nat,mass,nzero,xcoords,beta_hbar,coeff,qrot,qrot_quant,eigval)
  !! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: pi,printl,stdout
  use dlf_qts, only: dbg_rot
  implicit none
  integer, intent(in)  :: nat
  real(rk), intent(in) :: mass(nat)
  integer, intent(inout) :: nzero
  real(rk), intent(in) :: xcoords(3*nat)
  real(rk), intent(in) :: beta_hbar
  real(rk), intent(in) :: coeff(2)
  real(rk), intent(out):: qrot,qrot_quant
  real(rk),intent(out) :: eigval(3) ! eigenvalues of the moment of inertia
  real(rk) :: com(3)
  real(rk) :: xcoords_rel(3*nat) ! relative to the centre of mass
  integer  :: iat,icor,jcor,jval
  real(rk) :: inert(3,3),eigvec(3,3),ene

  qrot=1.D0
  qrot_quant=1.D0 
  
  if(dbg_rot) then
    write(stdout,'("coords ",3f15.10)') xcoords
    write(stdout,'("masses ",3f15.7)') mass
  end if

!!$  print*,"rotpart"
!!$  print*,'nat,nzero',nat,nzero
!!$  print*,'mass',mass
  
  ! ignore frozen atoms
  if(nzero/=5.and.nzero/=6) return
  !if(nzero==5.and.nat>2) return

!!$  ! check for the correct unit of mass
!!$  ! had to comment that out if the rotational partition function of the instanton is calculated as supermolecule.
!!$  if(minval(mass)<100.D0) then
!!$    print*,mass
!!$    call dlf_fail("Too small masses in rotational_partition_function. Wrong Unit?")
!!$  end if
  if(minval(mass)>2.D5) call dlf_fail("Too large masses in rotational_partition_function. Wrong Unit?")
  
  ! get the centre of mass
  com=0.D0
  do iat=1,nat
    com = com + mass(iat)* xcoords(3*iat-2:3*iat)
  end do
  com=com/sum(mass)
  
  do iat=1,nat
    xcoords_rel(3*iat-2:3*iat)=xcoords(3*iat-2:3*iat) - com
  end do
  
  if(nat>2) then
  ! now a larger molecule
  inert(:,:)=0.D0
  do iat=1,nat
    ! off-diagonal part
    do icor=1,3
      do jcor=icor+1,3
        inert(icor,jcor)=inert(icor,jcor) - mass(iat) * &
            xcoords_rel(3*iat-3+icor)*xcoords_rel(3*iat-3+jcor)
      end do
    end do
    ! diagonal elements
    inert(1,1)=inert(1,1)+ mass(iat) * &
        (xcoords_rel(3*iat-1)**2+xcoords_rel(3*iat-0)**2)
    inert(2,2)=inert(2,2)+ mass(iat) * &
        (xcoords_rel(3*iat-2)**2+xcoords_rel(3*iat-0)**2)
    inert(3,3)=inert(3,3)+ mass(iat) * &
        (xcoords_rel(3*iat-2)**2+xcoords_rel(3*iat-1)**2)
  end do
  
  !symmetrise
  do icor=1,3
    do jcor=1,icor-1
      inert(icor,jcor)=inert(jcor,icor)
    end do
  end do
  
  call dlf_matrix_diagonalise(3,inert,eigval,eigvec)
  
  if(dbg_rot) print*,"Eigenvalues of inertia",eigval
  end if ! nat>2

  if(nat==2) then
     qrot=mass(1)*sum(xcoords_rel(1:3)**2) + mass(2)*sum(xcoords_rel(4:6)**2)
     eigval(1)=qrot
     eigval(2)=qrot
     eigval(3)=0.D0
     nzero=5
  end if

  if(minval(abs(eigval))<10.D0) then
    nzero=5
    ! make sure that the last eigenvalue is smallest
    if(minloc(abs(eigval),dim=1)==1) then
      eigval(1)=eigval(3)
      eigval(3)=0.D0
    end if
    if(minloc(abs(eigval),dim=1)==2) then
      eigval(2)=eigval(3)
      eigval(3)=0.D0
    end if
  end if
  
  call rotational_partition_function_calc(eigval,beta_hbar,coeff,qrot,qrot_quant)
  
end subroutine rotational_partition_function
!!****

subroutine rotational_partition_function_calc(moi,beta_hbar,coeff,qrot,qrot_quant)
  use dlf_parameter_module, only: rk
  use dlf_global, only: pi
  implicit none
  real(rk), intent(in) :: moi(3)
  real(rk), intent(in) :: beta_hbar
  real(rk), intent(in) :: coeff(2)
  real(rk), intent(out) :: qrot
  real(rk), intent(out) :: qrot_quant
  integer :: jval,kval,comp
  real(rk) :: ene
  
  ! linear molecule
  if(minval(abs(moi))<10.D0) then
     ! the two remaining eigenvalues must be equal
     qrot=maxval(abs(moi))*2.D0/beta_hbar

     !print*,"lin q/c",beta_hbar/(0.25D0*2.D0*maxval(moi))
     if(beta_hbar<0.1D0*2.D0*maxval(moi)) then
        qrot_quant=qrot
     else
        !print*,"JK: calculation qrot_quant via sum"
        qrot_quant=0.D0
        do jval=0,200 ! maximum of 200 levels
           ene=dble(jval*(jval+1))*0.5D0/maxval(moi)
           qrot_quant=qrot_quant+coeff(mod(jval,2)+1)*dble(2*jval+1)*exp(-beta_hbar*ene)
           if(dble(2*jval+1)*exp(-beta_hbar*ene)<1.D-7*qrot_quant) exit ! stop summing if contributions too small
        end do
        !print*,"iterated to",jval
     end if
     
     !print*,"lin Q/Q_class=",qrot_quant/qrot
     return
    
  end if ! linear molecule
  
  qrot=sqrt(moi(1)*moi(2)*moi(3))
  qrot=qrot*sqrt(8.D0*pi/beta_hbar**3)
  !print*,"non-lin q/c",beta_hbar/(0.25D0*2.D0*(pi*product(moi))**(1.D0/3.D0))
  if(beta_hbar <  0.1D0*2.D0*(pi*product(moi))**(1.D0/3.D0)) then
    qrot_quant=qrot
  else
    ! check for symmetric top:
    comp=0
    if(abs(moi(2)-moi(1))<10.D0) comp=1
    if(abs(moi(2)-moi(3))<10.D0) comp=3
    if(comp/=0) then
      print*,"JK: calculation qrot_quant for symmetric top"
      do jval=0,200
        do kval=-jval,jval
          ene=dble(jval*(jval+1))/2.D0/moi(2)+dble(kval**2)*&
              (0.5D0/moi(comp)-0.5D0/moi(2))
          qrot_quant=qrot_quant+coeff(mod(jval,2)+1)*dble(2*jval+1)*exp(-beta_hbar*ene)
        end do
        if(dble(2*jval+1)*exp(-beta_hbar*ene)<1.D-7*qrot_quant) exit 
      end do
    else
      !print*,"JK: calculation qrot_quant via sum. not yet ..."
      call qr_asym(moi,qrot_quant,1.D0/beta_hbar,coeff)
      !qrot_quant=qrot
    end if
    !print*,"nonlin Q/Q_class=",qrot_quant/qrot
  end if !beta<...


end subroutine rotational_partition_function_calc

! rotational partition function for an asymmetric top. Written by Sean
! R. McConnell
subroutine qr_asym(moi,pfout,kbt,coeff)
  use dlf_parameter_module, only: rk
  use dlf_allocate, only: allocate,deallocate
  implicit none
  real(rk), intent(in) :: moi(3),kbt,coeff(2)
  real(rk), intent(out):: pfout
  integer j,k,k_1,m_j,i
  real(8) en,pfstore,energy,convtarget
  real(8), allocatable :: Ham_mat(:,:),eignum_r(:),eignum_i(:),eigvectr(:,:),store(:,:)
  j=1
  m_j=2*j+1
  allocate(Ham_mat(-j:j,-j:j))
  Ham_mat=0.d0
  do k_1=-j,j
    Ham_mat(k_1,k_1)=(1.d0/4.d0/moi(1)+1.d0/4.d0/moi(2))*&
        dble(j*(j+1))+dble(k_1)**2*(1.d0/2.d0/moi(3)-1.d0/4.d0/moi(1)-&
        1.d0/4.d0/moi(2))
    if(k_1+2.le.j)then
      Ham_mat(k_1,k_1+2)=(1.d0/8.d0)*(1.d0/moi(1)-1.d0/moi(2))*&
          sqrt(dble((j-k_1)*(j-k_1-1)*(j+k_1+1)*(j+k_1+2)))
    endif
    if(k_1-2.ge.-j)then
      Ham_mat(k_1,k_1-2)=(1.d0/8.d0)*(1.d0/moi(1)-1.d0/moi(2))*&
          sqrt(dble((j+k_1)*(j+k_1-1)*(j-k_1+1)*(j-k_1+2)))
    endif
  enddo
  call allocate(store,m_j,m_j)
  store(:,:)=Ham_mat(-j:j,-j:j)
  deallocate(Ham_mat)
  call allocate(eignum_r,m_j)
  call allocate(eignum_i,m_j)
  call allocate(eigvectr,m_j,m_j)
  ! how can store be non-symmetric? If it is symmetric, dlf_matrix_diagonalise
  ! should be called instead.
  call dlf_matrix_diagonalise_general(m_j,store,eignum_r,eignum_i,eigvectr)
  call deallocate(eigvectr)
  call deallocate(eignum_i)
  call deallocate(store)
  convtarget=1.d-7
  pfout=1.d0*coeff(1) ! for J=0
  pfstore=pfout
  do i=1,m_j
    en=eignum_r(i)
    pfstore=pfstore+coeff(mod(j,2)+1)*exp(-en/kbt)*dble(m_j)
  enddo
  call deallocate(eignum_r)

  !print*,'          ITER ',' CONV. TARGET:',convtarget
  do while(abs(pfstore-pfout)/pfout.gt.convtarget)
    pfout=pfstore
    j=j+1
    m_j=2*j+1
    allocate(Ham_mat(-j:j,-j:j))
    Ham_mat=0.d0
    do k_1=-j,j
      Ham_mat(k_1,k_1)=(1.d0/4.d0/moi(1)+1.d0/4.d0/moi(2))*dble(j*(j+1))+&
          dble(k_1)**2*(1.d0/2.d0/moi(3)-1.d0/4.d0/moi(1)-1.d0/4.d0/moi(2))
      if(k_1+2.le.j)then
        Ham_mat(k_1,k_1+2)=(1.d0/8.d0)*(1.d0/moi(1)-1.d0/moi(2))*&
            sqrt(dble((j-k_1)*(j-k_1-1)*(j+k_1+1)*(j+k_1+2)))
      endif
      if(k_1-2.ge.-j)then
        Ham_mat(k_1,k_1-2)=(1.d0/8.d0)*(1.d0/moi(1)-1.d0/moi(2))*&
            sqrt(dble((j+k_1)*(j+k_1-1)*(j-k_1+1)*(j-k_1+2)))
      endif
    enddo
    call allocate(store,m_j,m_j)
    store(:,:)=Ham_mat(-j:j,-j:j)
    deallocate(Ham_mat)
    call allocate(eignum_r,m_j)
    call allocate(eignum_i,m_j)
    call allocate(eigvectr,m_j,m_j)
    call dlf_matrix_diagonalise_general(m_j,store,eignum_r,eignum_i,eigvectr)
    call deallocate(eigvectr)
    call deallocate(eignum_i)
    call deallocate(store)
    do i=1,m_j
      en=eignum_r(i)
      pfstore=pfstore+coeff(mod(j,2)+1)*exp(-en/kbt)*dble(m_j)
    enddo
    call deallocate(eignum_r)
    !print*,"asym loop",j,' ',abs(pfstore-pfout)
  enddo
  pfout=pfstore

end subroutine qr_asym


! kappa_eck is the tunnelling enhancement by the symmetric Eckart barrier. The
! mass is to be ignored, since we have mass-weighted coordinates
! we return the log of kappa for improved numerical stability
subroutine kappa_eckart(beta_hbar,Vb,alpha,kappa_eck)
  use dlf_parameter_module, only: rk
  implicit none
  real(rk), intent(in) :: beta_hbar
  real(rk), intent(in) :: vb,alpha
  real(rk), intent(out):: kappa_eck
  real(rk) :: ene,enemax,trans,apar,dpar,pi
  integer  :: npoint,ipoint
  real(rk) :: trans_tol=1.D-10 ! parameter

  ! for a very wide variety of parameters, this should be fine:
  enemax=max(1.D0/beta_hbar*30.D0,vb*0.25D0*10.D0) 
  npoint=100000
  pi=4.D0*atan(1.D0)
  ! Vmax = Vb/4

!  if(2.D0*Vb - alpha**2 * 0.25D0 > 0.D0) then
     dpar=2.D0*pi*sqrt( 2.D0*Vb - alpha**2 * 0.25D0 ) / alpha
!  else
!     ! cosh(ix)=cos(x) for real-valued x
!     dpar=2.D0*pi*sqrt( -2.D0*Vb + alpha**2 * 0.25D0 ) / alpha     
!     print*,"dpar",dpar,cos(dpar),cosh(dpar)
!  end if
  ! integrate
  kappa_eck=0.D0
  do ipoint=1,npoint
    ene=enemax*dble(ipoint-1)/dble(npoint-1)
    apar=2.D0*pi*sqrt(2.D0*ene)/alpha
    if(dpar>500.D0.and.2.D0*apar>500.D0) then
      ! catch a inf/inf case
      trans=(1.D0-exp(-2.D0*apar))/(1.D0+exp(dpar-2.D0*apar))
    else
      if(2.D0*Vb - alpha**2 * 0.25D0 > 0.D0) then
        trans=(cosh(2.D0*apar)-1.D0)/(cosh(2.D0*apar)+cosh(dpar))
      else
        ! cosh(ix)=cos(x) for real-valued x
        dpar=2.D0*pi*sqrt( -2.D0*Vb + alpha**2 * 0.25D0 ) / alpha     
        trans=(cosh(2.D0*apar)-1.D0)/(cosh(2.D0*apar)+cos(dpar))
      end if
    end if
    ! weights for endpoints (last is removed outside of loop)
    if(ipoint==1) trans=trans*0.5D0
    kappa_eck=kappa_eck+trans*exp(-beta_hbar*ene)
    !print*,ene,kappa_eck,trans*exp(-beta_hbar*ene)

    ! terminate if trans is very close to 1
    if(trans > 1.D0-trans_tol) then
      exit
    end if
  end do
  ! half weight for last point
  kappa_eck=kappa_eck-0.5D0*trans*exp(-beta_hbar*ene)
  npoint=ipoint-1
  ! step size
  kappa_eck=kappa_eck*ene/dble(npoint-1)

  ! add the rest of the integral to infty
  kappa_eck=kappa_eck + exp(-beta_hbar*ene)/beta_hbar

  ! now divide by classical flux (and transform to log for better stability)
  !kappa_eck=kappa_eck * beta_hbar * exp(beta_hbar*0.25D0*Vb)
  kappa_eck=log(kappa_eck) + log(beta_hbar) +beta_hbar*0.25D0*Vb
end subroutine kappa_eckart

!!****

! kappa_eck is the tunnelling enhancement by the asymmetric Eckart barrier. The
! mass is to be ignored, since we have mass-weighted coordinates
! we return the log of kappa for improved numerical stability
!
! For consistency with the routine for the symmetric barrier, vb in the input is 4*Vmax,
! alpha=sqrt(abs(eigvals_ts(1))*8.D0/vb)
! Va is the asymmetry, the energy of the product channel
subroutine kappa_eckart_asymm(beta_hbar,vmax,Va,alpha,kappa_eck)
  use dlf_parameter_module, only: rk
  implicit none
  real(rk), intent(in) :: beta_hbar
  real(rk), intent(in) :: vmax,va,alpha
  real(rk), intent(out):: kappa_eck
  real(rk) :: ene,enemax,trans,apar,dpar,pi,curv,vb,bpar
  integer  :: npoint,ipoint
  real(rk) :: trans_tol=1.D-10 ! parameter

  ! for a very wide variety of parameters, this should be fine:
  enemax=max(1.D0/beta_hbar*30.D0,vmax*10.D0) 
  npoint=100000
  pi=4.D0*atan(1.D0)
  ! Vmax = Vb/4

!!$! asymmetric Eckart
!!$    apar=2.D0*pi*sqrt(2.D0*mass*ene)/alpha
!!$    bpar=2.D0*pi*sqrt(2.D0*mass*(ene-va))/alpha
!!$    dpar=2.D0*pi*sqrt(2.D0*mass*(sqrt(barrier)+sqrt(barrier-va))**2 - alpha**2 * 0.25D0 ) / alpha
!!$    trans_eck=(cosh(apar+bpar)-cosh(apar-bpar))/(cosh(apar+bpar)+cosh(dpar))
  
  !dpar=2.D0*pi*sqrt( 2.D0*Vb - alpha**2 * 0.25D0 ) / alpha
!!$  vmax=vb_/4.D0
!!$  curv=-alpha_**2*vb_/8.D0
!!$  vb=2.D0*vmax-va+sqrt((2.D0*vmax-va)**2-va**2)
!!$  alpha=sqrt(-curv*8.D0*vb**3/(va**2-vb**2)**2)
  
  dpar=2.D0*pi*sqrt(2.D0*(sqrt(vmax)+sqrt(vmax-va))**2 - alpha**2 * 0.25D0 ) / alpha
  
  ! integrate
  kappa_eck=0.D0
  do ipoint=1,npoint
    ene=enemax*dble(ipoint-1)/dble(npoint-1)
    apar=2.D0*pi*sqrt(2.D0*ene)/alpha
    bpar=2.D0*pi*sqrt(2.D0*(ene-va))/alpha
    !if(dpar>500.D0.and.2.D0*apar>500.D0) then
    !  ! catch a inf/inf case
    !  trans=(1.D0-exp(-2.D0*apar))/(1.D0+exp(dpar-2.D0*apar))
       
    !else
    if(ene<va) then
      trans=0.D0
    else
      if(2.D0*(sqrt(vmax)+sqrt(vmax-va))**2 - alpha**2 * 0.25D0 > 0.D0) then
        trans=(cosh(apar+bpar)-cosh(apar-bpar))/(cosh(apar+bpar)+cosh(dpar))
      else
        dpar=2.D0*pi*sqrt(-2.D0*(sqrt(vmax)+sqrt(vmax-va))**2 + alpha**2 * 0.25D0 ) / alpha
        trans=(cosh(apar+bpar)-cosh(apar-bpar))/(cosh(apar+bpar)+cos(dpar))
      end if
    end if
    if(isnan(trans)) then
      print*,"Error: trans in Asymmetric Eckart is NaN"
      print*,"apar",apar
      print*,"bpar",bpar
      print*,"dpar",dpar
    end if
    !end if
    ! weights for endpoints (last is removed outside of loop)
    if(ipoint==1) trans=trans*0.5D0
    kappa_eck=kappa_eck+trans*exp(-beta_hbar*ene)
    !print*,ene,kappa_eck,trans*exp(-beta_hbar*ene)
    
    ! terminate if trans is very close to 1
    if(trans > 1.D0-trans_tol) then
      exit
    end if
  end do
  ! half weight for last point
  kappa_eck=kappa_eck-0.5D0*trans*exp(-beta_hbar*ene)
  npoint=ipoint-1
  ! step size
  kappa_eck=kappa_eck*ene/dble(npoint-1)

  ! add the rest of the integral to infty
  kappa_eck=kappa_eck + exp(-beta_hbar*ene)/beta_hbar

  ! now divide by classical flux (and transform to log for better stability)
  !kappa_eck=kappa_eck * beta_hbar * exp(beta_hbar*0.25D0*Vb)
  kappa_eck=log(kappa_eck) + log(beta_hbar) +beta_hbar*vmax
end subroutine kappa_eckart_asymm

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/dlf_qts_definepath
!!
!! FUNCTION
!!
!! Define a qTS path in internal coordinates using input guesses or old
!! structures in x-coordinates
!!
!! Read qts_hessian.txt (via call to read_qts_hessian)
!!
!! Calculates qts%dtau
!!
!! useimage(iimage)=0 -> this image is interpolated, x-information is 
!! not used (not implemented at the moment)
!!
!! images are not reordered, the first image has to be "used", the 
!! later ones are uniformly distributed
!!
!! at the end, we have icoords and consistent xcoords - or an error termination
!!
!! INPUTS
!!
!! neb%xcoords glob%spec
!!
!!
!! OUTPUTS
!! 
!! glob%icoords
!!
!! SYNOPSIS
subroutine dlf_qts_definepath(nimage,useimage)
  !! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl,pi
  use dlf_allocate, only: allocate,deallocate
  use dlf_neb, only: neb,unitp,beta_hbar
  use dlf_constants, only : dlf_constants_get
  use dlf_bspline, only: spline_init, spline_create, spline_get, &
      spline_destroy
  use dlf_qts
  implicit none
  integer ,intent(in)   :: nimage
  integer ,intent(inout):: useimage(nimage)
  integer               :: iimage,jimage,kimage,cstart,cend,ivar
  integer               :: nimage_read
  integer               :: maxuse,nuse,map(nimage),jvar,iat
  real(rk)              :: svar,dist(nimage+1),lastdist,temperature
  logical               :: tok
  logical               :: hess_read
  !real(rk)              :: midpoint(neb%varperimage),direction(neb%varperimage)
  real(rk)              :: midpoint(3*glob%nat),direction(3*glob%nat)
!  real(rk)              :: hess_local(neb%varperimage*neb%nimage*2,neb%varperimage*neb%nimage*2)
!  real(rk),allocatable  :: hess_local(:,:,:) ! (neb%varperimage,neb%varperimage,neb%nimage)
  real(rk)              :: s_0,etunnel,ene,target_dist
  real(rk)              :: alpha,deltay
  real(rk)              :: de1dy
  logical               :: distread
  real(rk),allocatable  :: eigval(:),eigvec(:,:) ! varperimage(*varperimage)
  real(rk),allocatable  :: xcoords(:) ! 3*nat
  integer               :: varperimage_read
  real(rk)              :: mass_bimol,dene,ddene,arr2(2)
  real(rk),allocatable  :: mass(:)
  real(rk)              :: newcoords(3*glob%nat,neb%nimage)
  logical,parameter     :: tkeep=.false. ! tau adjusted to keep image positions (exactly equally spaced in y)
  logical               :: tokmass
  ! **********************************************************************

  ! no need to do anything if just the classical rate is to be calculated
  if(glob%iopt==13) return

  glob%igradient=0.D0
  glob%icoords=0.D0
  if(qts%tupdate) then
    qts%icoordhess(:,:)=0.D0
    qts%igradhess(:,:)=0.D0
  end if
  dist(:)=-1.D0 ! dist is a cumulative distance from the first image
  distread=.false.
  maxuse=1
  nuse=0 ! number of images to use to construct the path

  hess_read=.false.
!  call allocate(hess_local,neb%varperimage,neb%varperimage,neb%nimage)
  if(glob%iopt /=12 ) then
    ! fist, try to read the file qts_hessian.txt, written by a previous rate
    ! calculation.

    ! do not read qts_hessian in case of rate calculation: use qts_coords
    ! instead

    nimage_read=neb%nimage
    call allocate(mass,glob%nat)
    ! need that array temporarily 
    if(.not.qts%needhessian) call allocate(qts%vhessian,neb%varperimage,neb%varperimage,neb%nimage)

    call read_qts_hessian(glob%nat,nimage_read,neb%varperimage,temperature,&
        neb%ene,neb%xcoords,qts%igradient,qts%vhessian,etunnel,dist,mass,"",tok,arr2)
    ! vhessian was set here, but the corresponding coordinates will be set
    ! here we should do a mass-re-weighting if necessary.
    if(minval(mass) > 0.D0) then
      call dlf_constants_get("AMU",svar)
      !mass=mass/svar

      tokmass=.true.
      do iat=1,glob%nat
        if(abs(mass(iat)-glob%mass(iat))>1.D-7) then
          tokmass=.false.
          if(printl>=6) &
              write(stdout,*) "Mass of atom ",iat," differs from Hessian file. File:",mass(iat)," input",glob%mass(iat)
        end if
      end do
      
      ! Re-mass-weight
      if(.not.tokmass) then
        do iimage=1,nimage_read
          call dlf_re_mass_weight_hessian(glob%nat,neb%varperimage,mass/svar,glob%mass/svar,qts%vhessian(:,:,iimage))
        end do
      end if
    end if

    ! below (x->i necessary)
    call deallocate(mass)
    if(.not.qts%needhessian) call deallocate(qts%vhessian)

    if(nimage_read>neb%nimage) then
      if(printl>=2) write(stdout,*) "Read ",nimage_read," images, using only ", neb%nimage
      if(printl>=2) write(stdout,*) "Sorry, reduction of the number of images not yet implemented"
      call dlf_fail("Reading more images from&
          & qts_hessian.txt than used is not implemented")
    end if
    if(tok) then
      ! some images were read
      if(printl>=4) write(stdout,'("Using the qTS hessian for          ",f10.5," K")') glob%temperature
      if(qts%needhessian) hess_read=.true.
      useimage(:)=0
      useimage(1:nimage_read)=1
      if(dist(1)>-0.5D0) then
        distread=.true.
        if(printl>=2) write(stdout,*) "Distances read from file"
      else
        if(printl>=2) write(stdout,*) "Distances not read from file"
      end if

      if(printl>=4.and.nimage_read==neb%nimage) then
        write(stdout,'("All images and their Hessians read from file qts_hessian.txt")')
      end if

      !print*,"JK qts%first_hessian set to false (read)"
      qts%first_hessian=.false. ! make sure the qTS hessian update accepts
      !                           this Hessian as existing
      if(qts%needhessian.and.nimage_read==1) then
        ! We start from a classical TS (using distort)
        ! The hessian and the corresponding coordinates are distributed onto the images
        ! The gradient is set to zero here (although it never will be exactly zero)
        cstart=neb%cstart(1)
        cend=neb%cend(1)
        call dlf_direct_xtoi(glob%nvar,neb%varperimage,neb%coreperimage,neb%xcoords(:,1), &
            glob%xgradient,qts%icoordhess(:,1),glob%igradient(cstart:cend))

        ! distribute this information over all Hessians
        do iimage=2,neb%nimage
          qts%vhessian(:,:,iimage)=qts%vhessian(:,:,1)
          qts%icoordhess(:,iimage)=qts%icoordhess(:,1)
        end do
        qts%igradhess(:,:)=0.D0

        qts%hessian_mode=31
        qts%first_hessian=.true.
        hess_read=.true.

      end if

    else
      nimage_read=0
      if(qts%needhessian) then
        call dlf_fail("File qts_hessian.txt is required but could not be found.")
      end if
    end if
  end if

  ! try and read qts%ers
  call allocate(eigval,neb%varperimage)
  call allocate(xcoords,3*glob%nat)
  call allocate(mass,glob%nat)
  call read_qts_reactant(glob%nat,neb%varperimage,&
      qts%ers,varperimage_read,xcoords,eigval,mass_bimol,"",mass,tok)
  call deallocate(mass)
  call deallocate(xcoords)
  call deallocate(eigval)
  if(qts%ers>=huge(1.D0)) then !.not.tok) then
    qts%ers=-huge(1.D0)
    if(printl>=2) write(stdout,"('No file qts_reactant.txt present, no reactant energy used')")
  else
    if(printl>=2) write(stdout,*) "Reactant energy read: ",qts%ers
  end if

  !  if(minval(useimage)/=1) call dlf_fail("all images have to be provided as input for &
  !       &qTS search at the moment")

  ! get internal coordinates of well-defined images
  lastdist=0.D0
  do iimage=1,nimage
    if(useimage(iimage)==0) cycle
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage)
    !print*," ((((((((((((((((((((( IMAGE",iimage," )))))))))))))))))))))))))))"
    call dlf_direct_xtoi(glob%nvar,neb%varperimage,neb%coreperimage,neb%xcoords(:,iimage), &
        glob%xgradient,glob%icoords(cstart:cend),glob%igradient(cstart:cend))
    ! store coordinates for hessian updates
    if(qts%tupdate.and.nimage_read>1) then
      qts%icoordhess(:,iimage)=glob%icoords(cstart:cend)
    end if

    ! distance to last used image
    if(iimage>1) then
      svar=sqrt(sum(( &
          glob%iweight(1:neb%varperimage)*( &
          glob%icoords(cstart:cend) - glob%icoords(neb%cstart(maxuse):neb%cend(maxuse))) &
          ) **2 ) ) 
      !svar=svar+sum(dist)
      if(.not.distread) dist(iimage)=svar+lastdist
      lastdist=dist(iimage)
    end if
    maxuse=iimage
    nuse=nuse+1
  end do
  if(.not.distread) dist(1)=0.D0

  !
  ! define variable dtau 
  !
  if(glob%nebk>1.D-20) then
    !
    ! switch off!
    call dlf_fail("Variable step size is not supported any more.")
    !
    if(printl>=2) write(stdout,*) "Calculating dtau"
    ! dist(iimage) is a cumulative distance from the first image

!!$    ! Calculate E_b
!!$    S_0=0.D0
!!$    if(abs(etunnel+1.D0)<1.D-7) then
!!$      if(printl>=2) write(stdout,*) "Calculating E_tunnel from S_0 and S_pot"
!!$      do iimage=2,nimage
!!$        s_0=s_0+(dist(iimage)-dist(iimage-1))**2
!!$      end do
!!$      s_0=s_0*4.D0*dble(nimage)/beta_hbar
!!$      etunnel=sum(neb%ene)/dble(neb%nimage)-S_0/(2.D0*beta_hbar)
!!$      if(printl>=2) write(stdout,*) "S_0",s_0
!!$    else
!!$      if(printl>=2) write(stdout,*) "E_tunnel read from file:",etunnel
!!$    end if
    ! estimate dtau(1) - test
    de1dy=(neb%ene(1)-etunnel)/dist(1) ! dE1/dy

    ! length of the total path
    if(distread) then
      lastdist=dist(nimage_read+1)
    else
      lastdist=dist(nimage_read)
    end if

    call spline_init(nimage_read,1)
    call spline_create(1,dist(1:nimage_read),neb%ene)

    if(printl>=2) write(stdout,*) "E_tunnel = e_RS"
    etunnel=qts%ers
    !  etunnel=.31101972580349507D0 ! MBetunnel hardcoded to E_RS - IMPROVE
    !  etunnel= -0.112425519854954D0 ! Malon
    if(printl>=2) write(stdout,*) "E_tunnel used            ",etunnel
    !    qts%ers=etunnel
    if(printl>=4) then
      do iimage=1,nimage_read
        write(stdout,*) "Ene-E_tunnel:",neb%ene(iimage)-etunnel,dist(iimage)
      end do
      write(stdout,*) "dist                          ",dist(iimage)
    end if
    if(distread) then
      ! |-1-2-3-4-| 
      deltay=lastdist/dble(neb%nimage) 
    else
      deltay=lastdist/dble(neb%nimage-1)
    end if
    !print*,"delta-y",deltay
    newcoords=neb%xcoords
    do iimage=1,nimage+1 ! this is dtau for each image doubled (also the end images)
!!$      if(iimage==1 -100) then  ! -100: switched off
!!$        if(neb%ene(iimage)-etunnel < 0.D0) then
!!$          if(printl>=2) write(stdout,*) "Error in defining dtau: ene-etunnel < 0 for image",iimage
!!$          call dlf_fail("Error in defining dtau")
!!$        end if
!!$        ene=neb%ene(iimage)
!!$        !qts%dtau(iimage)=1.D0/sqrt(2.D0*(neb%ene(iimage)-etunnel))   !*1.4D0
!!$      else if(iimage==nimage+1 +100) then  ! +100: switched off
!!$        if(neb%ene(iimage-1)-etunnel < 0.D0) then
!!$          if(printl>=2) write(stdout,*) "Error in defining dtau: ene-etunnel < 0 for image",iimage
!!$          call dlf_fail("Error in defining dtau")
!!$        end if
!!$        ene=neb%ene(iimage-1)
!!$        !qts%dtau(iimage)=1.D0/sqrt(2.D0*(neb%ene(iimage-1)-etunnel))   !*1.4D0
!!$      else ! 2 to nimage
        ! find "dist"
      if(distread) then
        target_dist=lastdist*dble(iimage-1)/dble(nimage)
        
        ! for fitted dtau:
        !target_dist=dist(1)+(dist(nimage)-dist(1))*dble(iimage-1)/dble(nimage-1)
      else
        target_dist=lastdist/dble(nimage-1)*(dble(iimage)-1.5D0)
      end if
      ! end points
      if(distread.and.iimage==nimage+1) &
          target_dist=lastdist*(dble(iimage)-1.5D0)/dble(nimage)
      !target_dist=dist(nimage+1)*(dble(iimage)-1.3D0)/dble(nimage)
      if(distread.and.iimage==1) &
          target_dist=lastdist*0.5D0/dble(nimage)
      !target_dist=dist(nimage+1)*0.8D0/dble(nimage)
      if(.not.distread) then
        if(iimage==nimage+1) target_dist=lastdist
        if(iimage==1) target_dist=0.D0
      end if


      ! find energy
      do jimage=1,nimage
        if(dist(jimage)>target_dist) exit
      end do
      if(jimage>nimage+1) call dlf_fail("inconsistency in defining dtau")
      !print*,"image",iimage,"jimage",jimage,svar
      if(jimage==nimage+1) then
        svar=(target_dist-dist(jimage-1))/(dist(jimage)-dist(jimage-1))
        ene=(1.D0-svar)*neb%ene(jimage-1) + (svar)*etunnel
      else if(jimage==1) then
        svar=(target_dist)/(dist(jimage))
        ene=(1.D0-svar)*etunnel + (svar)*neb%ene(jimage)
      else
        ! general case
        call spline_get(1,target_dist,ene,dene,ddene)
        !svar=(target_dist-dist(jimage-1))/(dist(jimage)-dist(jimage-1))
        !ene=(1.D0-svar)*neb%ene(jimage-1) + (svar)*neb%ene(jimage)
        
        if(tkeep) then
          if(iimage<=nimage.and.iimage>1) then
            ! interpolate coordinates - only applicable with tau adjusted to keep image positions.
            !print*,"iimage,jimage",iimage,jimage,svar
            newcoords(:,iimage) = &
                (1.D0-svar)*neb%xcoords(:,jimage-1) + svar * neb%xcoords(:,jimage)
          end if
        end if
        
      end if
      if(ene-etunnel < 0.D0) then
        if(printl>=2) then
          write(stdout,*) "Error in defining dtau: ene-etunnel < 0 for image",iimage
          write(stdout,*) "dist(nimage+1) (total path length)",dist(nimage+1)
          write(stdout,*) "dist(nimage)                      ",dist(nimage)
          write(stdout,*) "frac",svar
          write(stdout,*) "ene",ene
          write(stdout,*) "etunnel",etunnel
          write(stdout,*) "target dist",target_dist
        end if
        if(tkeep) then
          ! workaround - only applicable  with tau adjusted to keep image positions.
          ene=etunnel+0.01D0
        else
          call dlf_fail("Error in defining dtau")
        end if
      end if
      qts%dtau(iimage)=deltay/sqrt(2.D0*(ene-etunnel))
      if(printl>=6) write(stdout,*) "iimage",iimage,"dtau",qts%dtau(iimage)

    end do !iimage=1,nimage+1

    call spline_destroy
    
    if(tkeep) then
      ! convert xcoords to icoords - only applicable with tau adjusted to keep image positions.
      neb%xcoords=newcoords 
      ! call deallocate(newcoords)
      do iimage=1,nimage
        cstart=neb%cstart(iimage)
        cend=neb%cend(iimage)
        call dlf_direct_xtoi(glob%nvar,neb%varperimage,neb%coreperimage,&
            neb%xcoords(:,iimage),glob%xgradient,glob%icoords(cstart:cend),&
            glob%igradient(cstart:cend))
      end do
    end if

!!$    ! end points - this does not work well
!!$    svar=-sum(qts%dtau(2:nimage))+beta_hbar*0.5D0
!!$    !qts%dtau(1)=svar ! if this is commented out, only one side is set - which is a hard-coding for now IMPROVE
!!$    !qts%dtau(nimage+1)=svar 
!!$
!!$    ! test for dtau(1) - does not work well either. This should give a
!!$    ! force of zero on the first image if an exact gradient were used
!!$    svar=2.D0*deltay/qts%dtau(2)/de1dy-qts%dtau(2)
!!$    print*,"test dtau(1)",svar


    !qts%dtau=1.D0 ! constant dtau
    svar=2.D0*sum(qts%dtau)-qts%dtau(1)-qts%dtau(nimage+1) ! count end deltas only once
    qts%dtau=qts%dtau*beta_hbar/svar
    !!alpha=1: full equi-r, alpha=0: equidistant in tau
    !alpha=0.75D0 
    alpha=glob%nebk
    if(alpha<0.D0) alpha=0.D0
    if(alpha>1.D0) alpha=1.D0
    qts%dtau= alpha * qts%dtau + (1.D0-alpha) * beta_hbar/dble(2*nimage) 
    if(printl>=2) write(stdout,*) "Alpha for setting dtau",alpha
  else
    qts%dtau=beta_hbar/dble(2*nimage)
  end if !(nimage_read==neb%nimage) ! calculating dtau - experimental


  !print dtau
  if(printl>=4) then
    do iimage=1,nimage+1
      write(stdout,'("dtau (image ",i4,") ",es10.3)') iimage,qts%dtau(iimage)
    end do
  end if

  ! end of variable dtau

  ! build up a path using neb%xcoords(:,1) as midpoint and
  ! neb%xcoords(:,2) as (absolute) direction. Distribute them
  ! glob%distort in each direction
  if(glob%distort>1.D-10) then
    if(maxuse>2) then
      call dlf_fail("Distort can only be used in qTS if only two images are &
          &defined, coords being the midpoint, coords2 the direction")
    end if
    if(printl>=4) write(stdout,'("Constructing initial qTS path midpoint+distortion")')
    !if(printl>=4) write(stdout,'("Length of distortion: ",f10.5)') dist(maxuse)
    midpoint=neb%xcoords(:,1)

    if(nimage_read==1.and.qts%needhessian) then
      ! Hessian of the classical TS has been read. Use its lowest eigenvalue
      ! for spread rather than coords2:
      if(printl>=2) write(stdout,"('The lowest eigenvector of &
          &qts_hessian.txt is used rather than the coordinates in coords2')")
      call allocate(eigval,neb%varperimage)
      call allocate(eigvec,neb%varperimage,neb%varperimage)
      call dlf_matrix_diagonalise(neb%varperimage,qts%vhessian(:,:,1),eigval,eigvec)
      direction(:)=0.D0
      call dlf_cartesian_itox(glob%nat,neb%varperimage,neb%varperimage,glob%massweight,&
          eigvec(:,1),direction)
      call deallocate(eigval)
      call deallocate(eigvec)

    else
      direction=neb%xcoords(:,2)
      if(.not.glob%tsrelative) direction=direction-midpoint
    end if

    svar=sqrt(sum(direction**2))
    if(svar<1.D-8) call dlf_fail("Start and endpoint too close in qTS search")
    direction=direction/svar*glob%distort
    do iimage=1,nimage
      ! 1-0.5 = -pi/2
      ! nimage+0.5 = pi/2
      svar=(dble(iimage)-0.5D0)/dble(nimage) - 0.5D0
      if(printl>=4) write(stdout,"('Distributing image ',i4,2f10.5)") &
          iimage,glob%distort*sin(svar*pi),svar
      neb%xcoords(:,iimage)=midpoint+direction*sin(svar*pi)

      cstart=neb%cstart(iimage)
      cend=neb%cend(iimage)
      call dlf_direct_xtoi(glob%nvar,neb%varperimage,neb%coreperimage,neb%xcoords(:,iimage), &
          glob%xgradient,glob%icoords(cstart:cend),glob%igradient(cstart:cend))

    end do
    useimage(1:nimage)=1
    glob%xcoords=reshape(neb%xcoords(:,1),(/3,glob%nat/))
  else

    ! print alignment of images along the distance
    if(printl>=4) then
      write(stdout,'("Constructing initial qTS path from input images")')
      write(stdout,'("Length of the initial path: ",f10.5)') dist(maxuse)
      write(stdout,'("Using ",i3," input images to construct a path of ",&
          &i3," total images")') nuse,nimage
      lastdist=0.D0
      do iimage=1,maxuse
        if(useimage(iimage)==0) cycle
        write(stdout,'("Image ",i3," along path: ",f10.5," dist to prev. &
            &image:",f10.5)') iimage,dist(iimage),dist(iimage)-lastdist
        lastdist=dist(iimage)
      end do
    end if
    if(dist(maxuse)<1.D-5.and.glob%iopt/=12) call dlf_fail("Start and endpoint too close in qTS search")

  end if

  ! if all images are well-defined, return
  if(minval(useimage)==1) then
    glob%xcoords=reshape(neb%xcoords(:,1),(/3,glob%nat/))
    if (hess_read) then
      if(qts%tupdate.and.nimage_read>1) then
        ! all images (and their Hessians) are read in, set icoordhess and igradhess
        do iimage=1,neb%nimage
          cstart=neb%cstart(iimage)
          cend=neb%cend(iimage)
          qts%icoordhess(:,iimage)=glob%icoords(cstart:cend)
          qts%igradhess(:,iimage)=0.D0 ! will be set after the first energy & gradient calculations
        end do
      end if
!!$      ! duplication of the Hessian (not needed for optimisation, but does not hurt)
!!$      do iimage=1,neb%nimage
!!$        do ivar = 1,neb%varperimage
!!$          do jvar =  1,neb%varperimage
!!$            qts%total_hessian(&
!!$                neb%varperimage*neb%nimage*2+ivar-iimage*neb%varperimage , &
!!$                neb%nimage*neb%varperimage*2+jvar-iimage*neb%varperimage ) &
!!$                = qts%total_hessian(ivar+(iimage-1)*neb%varperimage,&
!!$                jvar+(iimage-1)*neb%varperimage)
!!$          end do
!!$        end do
!!$      end do

    end if
    return
  end if


  if(minval(useimage)==0) then ! meaning: if we have to interpolate

    ! if all requested images are provided by the user, do not interpolate
    if(nuse/=nimage) then
      ! distribute defined images as equally as possible according to dist
      map(:)=0 ! map(iimage) is the number, iimage should have after distribution
      do iimage=maxuse,2,-1

        if(useimage(iimage)==0) cycle

        !svar=1.D0+dist(iimage)/dist(maxuse)*dble(nimage-1)
        svar=1.D0+dble(iimage-1)/dble(maxuse-1)*dble(nimage-1)
        map(iimage)=nint(svar)
        if(printl>=2) write(stdout,*) "Image",iimage,"svar",svar,"map(iimage)",map(iimage)
        if(map(iimage)>nimage) then
          call dlf_fail("Problem with image distribution in qTS.")
        end if
      end do

      ! assign images
      do iimage=maxuse,2,-1

        tok=.true.
        if(map(iimage)==1) tok=.false.
        do jimage=maxuse,1,-1
          if(jimage==iimage) cycle
          if(map(jimage)==map(iimage)) tok=.false.
        end do
        ! commented out the next block as I don't see where this could help
!!$      if(.not.tok) then
!!$        ! see if map(iimage)=iimage would be OK
!!$        tok=.true.
!!$        do jimage=maxuse,2,-1
!!$          if(map(jimage)==iimage) tok=.false.
!!$        end do
!!$        if(tok) map(iimage)=iimage
!!$      end if
        if(.not.tok) then
          ! Drop current image
          map(iimage)=0
          useimage(iimage)=0
          if(printl>=2) write(stdout,'("Dropping input image ",i3," because &
              &there are too may input images in this area")') iimage
          cycle
        end if

        ! swap images
        if(printl>=4) then
          write(stdout,'("Using input image",i4," as image",i4)') iimage,&
              map(iimage)
        end if

        if(map(iimage)==iimage) cycle

        if(map(iimage)<iimage) then
          call dlf_fail("Moving image downwards, not implemented")
        end if

        ! move image
        ivar=map(iimage)
        ! move icoords
        if(printl>=2) write(stdout,*) "moving coords",iimage," to ",ivar
        glob%icoords(neb%cstart(ivar):neb%cend(ivar))= &
            glob%icoords(neb%cstart(iimage):neb%cend(iimage))
        ! move xcoords
        neb%xcoords(:,ivar)=neb%xcoords(:,iimage)
        ! move use info
        useimage(ivar)=1
        useimage(iimage)=0

        ! move hessian (and accompanying gradient info) if applicable
        if(hess_read) then
          qts%vhessian(:,:,ivar)=qts%vhessian(:,:,iimage)
          if(qts%tupdate.and.nimage_read>1) then
            qts%icoordhess(:,ivar)=qts%icoordhess(:,iimage)
          end if
!!$          qts%total_hessian(neb%cstart(ivar):neb%cend(ivar),&
!!$              neb%cstart(ivar):neb%cend(ivar))= &
!!$              qts%total_hessian(neb%cstart(iimage):neb%cend(iimage),&
!!$              neb%cstart(iimage):neb%cend(iimage))
        end if

      end do

    end if

    if(printl>=4) then
      do iimage=1,nimage
        if(useimage(iimage)==1) then
          write(stdout,'("Image",i3," is obtained from input")') iimage
        else
          write(stdout,'("Image",i3," is interpolated")') iimage
        end if
      end do
    end if

    ! linear transit between defined images
    if(useimage(1)==0) call dlf_fail("First image undefined in qTS")
    if(useimage(nimage)==0) call dlf_fail("Last image undefined in qTS")
    do iimage=1,nimage
      if(useimage(iimage)==0) cycle
      ! iimage is obtained from input
      do jimage=iimage+1,nimage
        if(useimage(jimage)/=0) exit
      end do
      ! jimage is also obtained from input
      if(jimage==iimage+1) cycle
      ! interpolate from iimage to jimage
      do kimage=iimage+1,jimage-1
        if(printl>=2) write(stdout,*) "interpolating image",kimage," between ",iimage," and ",jimage
        cstart=neb%cstart(kimage)
        cend=neb%cend(kimage)
        ! one can improve on this (at least when the energies are available, but
        ! probably otherwise as well):
        svar=dble(kimage-iimage)/dble(jimage-iimage) ! 0..1
        !print*,"svar",svar
        glob%icoords(cstart:cend)= &
            (1.D0-svar)*glob%icoords( neb%cstart(iimage) : neb%cend(iimage) ) + &
            svar*glob%icoords( neb%cstart(jimage) : neb%cend(jimage) )
        ! xcoords are interpolated as well, as qTS does only make sense in mass-weighted 
        ! cartesian coordinates, and these are linear with respect to cartesians
        neb%xcoords(:,kimage)= &
            (1.D0-svar)*neb%xcoords( :,iimage ) + &
            svar*neb%xcoords( :,jimage )
        ! interpolate hessian (maybe update is better, but I am not convinced...)
        if(hess_read) then
          qts%vhessian(:,:,kimage)=(1.D0-svar)*qts%vhessian(:,:,iimage) + &
              svar*qts%vhessian(:,:,jimage)
          if(qts%tupdate) then
            qts%icoordhess(:,kimage)=(1.D0-svar)*qts%icoordhess(:,iimage) + &
                svar*qts%icoordhess(:,jimage)
          end if
!!$          qts%total_hessian(cstart:cend,cstart:cend)=&
!!$              (1.D0-svar)*qts%total_hessian( neb%cstart(iimage):neb%cend(iimage) , &
!!$              neb%cstart(iimage):neb%cend(iimage)) + &
!!$              svar*qts%total_hessian( neb%cstart(jimage):neb%cend(jimage) , &
!!$              neb%cstart(jimage):neb%cend(jimage) )
        end if
      end do
    end do
  end if ! minval(useimage)==0)

  if (hess_read) then
    glob%xcoords=reshape(neb%xcoords(:,1),(/3,glob%nat/))
  end if

end subroutine dlf_qts_definepath
!!****


subroutine dlf_kryvohuz_supercrit_jk(nvpi,nimage,ers,ets,&
    qrsi2,expansion_freq,beta_hbar,lograte,tprint)
  use dlf_parameter_module, only: rk
  use dlf_allocate, only: allocate,deallocate
  use dlf_qts
  use dlf_constants, only : dlf_constants_get
  use dlf_global, only: pi!, stdout
  !      use dlf_neb, only: beta_hbar
  implicit none
  integer, intent(in) :: nvpi,nimage
  real(rk), intent(in):: ers,expansion_freq(nvpi),qrsi2,ets,&
      beta_hbar
  real(rk), intent(out) :: lograte
  logical, intent(in) :: tprint ! if set, only calculate ddS_wrt_beta
                                ! and return it as lograte
  real(rk) omega_crit,rs_to_inst_vec(nvpi),&
      V_tilde0,second_au,kboltz_au,beta_crit,A,g,c,omega_bar2
  real(rk) dE_tilde,lnrate,rpc,rpc2,&
      ddS_wrt_beta,delta
  real(rk) :: inp_hessian(nvpi,nvpi),expo

  inp_hessian=0.D0 ! apparently not needed

  call dlf_constants_get("SECOND_AU",second_au)
  call dlf_constants_get("KBOLTZ_AU",kboltz_au)
  beta_crit=1.d0/qts%tcross/kboltz_au

  omega_crit=2.d0*pi/beta_crit
  V_tilde0=ets-ers

  !     IN THE SUPERCRITICAL CASE, dE_b/dBeta DEPENDS ON A FOURTH ORDER 
  !     EXPANSION OF THE POTENTIAL FITTED IN A SMALL REGION NEAR THE TS. 
  !     I USE A MORSE POTENTIAL TO GET AN APPROXIMATION TO THE 3RD AND 4TH
  !     ORDER TERMS, THIS IS MORE THAN SUFFICIENT IN SUCH A SMALL REGION
  omega_bar2=16.d0*pi**2/beta_crit**2-omega_crit**2
  g=-(7.d0/2.d0)*omega_crit**4/(V_tilde0)
  c=(3.d0/dsqrt(2.d0))*omega_crit**3/dsqrt(V_tilde0)
  A=g/2.d0+c**2/omega_crit**2-c**2/2.d0/omega_bar2
  if(qts%dEbdbeta>=0.D0) then
     ddS_wrt_beta=(-4.d0/A/beta_crit)*(2.d0*pi/beta_crit)**4
  else
     ddS_wrt_beta=qts%dEbdbeta
  end if
  if(tprint) then
     lograte=ddS_wrt_beta
     !
     return
     !
  end if
  !      ddS_wrt_beta=-dabs((ets-ers)**3*omega_crit/&
  !      (omega_crit**3-2.d0*(ets-ers)**2))
  !      ddS_wrt_beta=-(ets-ers)*omega_crit/2.d0

  !      rp=minval(expansion_freq,dim=1)
  !print*,"nvpi",nvpi
  if(nvpi>=1)then
    rs_to_inst_vec=1.d0 ! not used
    call kryvohuz_sigma_high(nimage,nvpi,inp_hessian,expansion_freq,-1,&
        rs_to_inst_vec,beta_hbar)
  else
    qts%S_sigma=0.d0
    qts%S_dsigma=0.d0
    qts%S_ddsigma=0.d0
  endif

  !      write(stdout,"('LOG10 Rate (Kryvohuz)              ',es15.8)")&
  !      dble(-ddS_wrt_beta)
  !      write(stdout,"('LOG10 Rate (Kryvohuz)              ',es15.6)")&
  !      dble(qts%S_sigma)
  !      write(stdout,"('LOG10 Rate (Kryvohuz)              ',es15.6)")&
  !      qrsi2
  !      if(nvpi.eq.1)then
  !print*,"ddS_wrt_beta,qts%S_ddsigma",ddS_wrt_beta,qts%S_ddsigma
!  ddS_wrt_beta=-0.0002196734D0 !ddS_wrt_beta/2.0D0
   !ddS_wrt_beta=-6.D-3
  !  print*,"ddS_wrt_beta",ddS_wrt_beta
  dE_tilde=ddS_wrt_beta+real(qts%S_ddsigma)
  !      else
  !        dE_tilde=-min(abs(ddS_wrt_beta),abs(qts%S_ddsigma))
  !      endif      
  delta=(sqrt(beta_hbar*beta_crit)/2.d0)*((beta_crit/beta_hbar)**2-1.d0)*&
      sqrt(-dE_tilde) ! cdsqrt replace by modern sqrt
!print*,"delta",delta

 !   rpc=erf(-delta/dsqrt(2.d0))*0.5D0+0.5D0
!print*,"delta,rpc2",delta,rpc2,rpc,erf(-real(delta)/dsqrt(2.d0))*0.5D0+0.5D0
 !   print*,"appr beta",beta_hbar,beta_crit
 !   print*,"approach",-log(sin(pi*beta_hbar/beta_crit)),+0.5d0*delta**2+log(delta*sqrt(2.d0*pi))&
 !         +log(rpc),log(1.d0-1.d0/delta**2)
  if(beta_hbar>beta_crit) then
     lograte=0.D0
     !
     return
     !
  end if

  expo=0.5D0! 0.5D0
  
!!$  ! original expression by Sean
!!$  lnrate=-log(2.d0*beta_crit*sin(pi*beta_hbar/beta_crit)) &
!!$       -dble(beta_hbar*(V_tilde0+&
!!$       ((beta_crit-beta_hbar)**expo/beta_crit**expo) * real(qts%S_sigma)/beta_hbar+&
!!$       (1.d0-(beta_crit-beta_hbar)**expo/beta_crit**expo) * real(qts%S_dsigma))) &
!!$       -qrsi2
!!$  ! sin(x) replaced by x
!!$  lnrate=-log(2.d0*beta_crit*(pi*beta_hbar/beta_crit)) &
!!$       -dble(beta_hbar*(V_tilde0+&
!!$       ((beta_crit-beta_hbar)**expo/beta_crit**expo) * real(qts%S_sigma)/beta_hbar+&
!!$       (1.d0-(beta_crit-beta_hbar)**expo/beta_crit**expo) * real(qts%S_dsigma))) &
!!$       -qrsi2

  ! sigma/beta_hbar
  lnrate=-log(2.d0*beta_crit*sin(pi*beta_hbar/beta_crit)) &
       -beta_hbar*(V_tilde0+&
       dble(qts%S_sigma)/beta_hbar) &       
       -qrsi2
!!$  ! dsigma
!!$  lnrate=-log(2.d0*beta_crit*sin(pi*beta_hbar/beta_crit)) &
!!$       -beta_hbar*(V_tilde0+&
!!$       dble(qts%S_dsigma)) &       
!!$       -qrsi2

!  print*,"qts",real(qts%S_sigma),real(qts%S_sigma)/beta_hbar,real(qts%S_dsigma)
!  print*,"sig,sig'",((beta_crit-beta_hbar)**0.5d0/beta_crit**0.5d0) * real(qts%S_sigma)/beta_hbar,&
!       (1.d0-(beta_crit-beta_hbar)**0.5d0/beta_crit**0.5d0) * real(qts%S_dsigma)
!  print*,"delta",delta,log(1.d0-1.d0/delta**2)
  
  if(dble(delta).lt.7.d0)then!THE ERROR FUNCTION BECOMES VERY SMALL 
    !     AND INACCURATE WHEN DELTA BECOMES LARGE, HENCE THE APPROXIMATION
    !call cerror(-delta/dsqrt(2.d0),rpc2)
    !rpc=(0.5d0+0.5d0*rpc2)
    rpc=erf(-delta/dsqrt(2.d0))*0.5D0+0.5D0
!print*,"delta,rpc2",delta,rpc2,rpc,erf(-real(delta)/dsqrt(2.d0))*0.5D0+0.5D0
          !          -qts%S_sigma&
    lnrate=lnrate + 0.5d0*delta**2 + log(delta*sqrt(2.d0*pi)) + log(rpc)
          
    ! loprod X_i 
    !lograte=real(lnrate/log(10.d0))
  else
      lnrate=lnrate + log(1.d0-1.d0/delta**2)
  endif

  lograte=lnrate

end subroutine dlf_kryvohuz_supercrit_jk

subroutine kryvohuz_sigma_high(nimage,nvpi,inp_hessian,inp_eigvals,&
    instanton_pointer,rs_to_inst_vec,beta_hbar) 
  use dlf_parameter_module, only: rk     
  use dlf_qts
  use dlf_constants, only : dlf_constants_get
  use dlf_allocate, only: allocate,deallocate 
  use dlf_neb, only: neb!,beta_hbar
  use dlf_global, only: glob
  implicit none        
  integer, intent(in) :: nimage,nvpi,instanton_pointer
  real(rk), intent(in):: inp_eigvals(nvpi),inp_hessian(nvpi,nvpi),&
      beta_hbar
  integer i,j,k,l,iimage,im_start,im_end,im_direction,imagenum,&
      n_zeroevals,i_dim,tau_step,n_zerotest,imagenum_reverse
  real(rk) rp,kboltz_au,RS_proj,rs_to_inst_vec(nvpi),RS_proj2,&
      beta_crit
  complex(rk) rpc,rpc_reduce,sig(0:1),dsig(0:1),ddsig(0:1),rpc2,&
      rpc3,rpc4
  integer, dimension(:), allocatable :: projection
  real(rk), dimension(:), allocatable:: vec_inc,c_storage0,&
      c_storage,c_storage4,c_storage4_0,storage4
  real(rk), dimension(:,:), allocatable:: storage1,ev_perimage,&
      storage7,c_storage2,c_storage3,c_storage6,R_stabmat,ev_perimage2,&
      commutator,commutator2,commutator3,vec_array,storage3,storage2,&
      rotated_hess
  complex(rk), dimension(:,:), allocatable:: z_stor,z_stor2,z_stor3
  real(rk), dimension(:,:,:), allocatable :: F_stabmat,&
      GSbasis_array

  call dlf_constants_get("KBOLTZ_AU",kboltz_au)
  !     IF WE ARE IN THE SUPERCRITICAL REGION THEN ONLY THE NON-NEGATIVE
  !     EVALS OF THE TS HESSIAN ARE NEEDED TO CALCULATE TRANSVERSE CONTRI-
  !     BUTION TO THE ACTION
  if(instanton_pointer.eq.-1)then
    beta_crit=1.d0/qts%tcross/kboltz_au
    rp=0.d0
    do i=1,nvpi
      if(inp_eigvals(i).gt.1.d-16)then
        rp=rp+&
            log(2.d0*sinh(beta_hbar*0.5d0*&
            sqrt(inp_eigvals(i))))
      endif
    enddo
    qts%S_sigma=rp
    rp=0.d0
    do i=1,nvpi
      if(inp_eigvals(i).gt.1.d-16)then
        rp=rp+0.5d0*dsqrt(inp_eigvals(i))!/&
        !            dtanh(beta_crit*0.5d0*dsqrt(inp_eigvals(i)))
      endif
    enddo
    qts%S_dsigma=rp
    rp=0.d0
    do i=1,nvpi
      if(inp_eigvals(i).gt.1.d-16)then
        rp=rp-0.25d0*inp_eigvals(i)/&
            sinh(beta_crit*0.5d0*sqrt(inp_eigvals(i)))**2
      endif
    enddo
    qts%S_ddsigma=rp
    !
    return
    !
  endif

  ! routine truncated here, because we only need instanton_pointer==-1 for T>Tc

end subroutine kryvohuz_sigma_high


! **********************************************************************
! **                         Reaction Rate Module                     **
! **   Instanton theory and classical transition state theory with    **
! **           several approximations for tunneling rates             **
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
!!  Copyright 2010-2012 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
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
  end type qts_type
  type(qts_type),save :: qts
  logical, parameter:: hess_dtau=.false. ! if true, the Potential-Hessian is
  integer :: taskfarm_mode=1 ! default, especially if QTS is not used ...
  ! there is a global variable, glob%qtsflag. Meaning:
  ! 0: normal calculation
  ! 1: calculate tunnelling splittings (only valid for symmetric cases)
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
  qts%hessian_mode=33 ! generally: 33
  qts%first_hessian=.true.

  if(glob%iopt==12) qts%hessian_mode=0 ! make sure the whole Hessian is calculated for Rate calculations

  ! allocate storage for hessian update in case any update is requested
  qts%tupdate=qts%needhessian.and.(mod(qts%hessian_mode,10)==3 .or. qts%hessian_mode/10==3)
  if(qts%tupdate) then
    call allocate (qts%igradient, neb%varperimage*neb%nimage)
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
  if(qts%tupdate) then
    if (allocated(qts%igradient)) call deallocate (qts%igradient) 
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
  use dlf_global, only: glob,stdout,printl
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
  !**********************************************************************
  fracrecalc=.false.

  ! catch the case of inithessian==5 (Hessian to be read from
  ! qts_hessian.txt)
  if (glob%inithessian==5) then
    ! read hessian from disk
    nimage_read=neb%nimage
    call allocate(mass,glob%nat)
    call read_qts_hessian(glob%nat,nimage_read,neb%varperimage,glob%temperature,&
        neb%ene,neb%xcoords,qts%vhessian,qts%etunnel,qts%dist,mass,"",tok)

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
      ! read the Hessian of one image:
      call read_qts_hessian(glob%nat,iat,neb%varperimage,glob%temperature,&
          neb%ene(qts%image_status),neb%xcoords(:,qts%image_status),qts%ihessian_image,&
          qts%etunnel,arr2,mass,label,tok)
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
      if(printl>=4) write(stdout,"('Calculating analytic Hessian for image ',i4)") qts%image_status 
      call dlf_get_hessian(glob%nvar,neb%xcoords(:,qts%image_status),xhessian,status)
    else
      status=1
    end if

    if(status==0) then

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


  cstart=neb%cstart(qts%image_status)
  cend=neb%cend(qts%image_status)

  if(.not. glob%havehessian) then   

    if (glob%inithessian==0.or.glob%inithessian==2.or.glob%inithessian==6) then

      ! after first energy evaluation, the energy and gradient obtained have
      ! to be communicated to the tasks:
      if(.not.fd_hess_running.and.glob%ntasks>0.and.taskfarm_mode==1) then
        call dlf_tasks_real_sum(glob%xgradient, glob%nvar)
        call dlf_tasks_real_sum(glob%energy, 1)
      end if

      ! reduce print level in dlf_fdhessian
      printl_store=printl
      printl=min(printl,3)

      ! Hessian in internals
      ! Finite Difference Hessian calculation in internal coordinates
      call dlf_direct_xtoi(glob%nvar,neb%varperimage,neb%coreperimage,&
           glob%xcoords,glob%xgradient, &
          glob%icoords(cstart:cend),glob%igradient(cstart:cend))
      call dlf_fdhessian(neb%varperimage,.false.,glob%energy,&
          glob%icoords(cstart:cend), &
          glob%igradient(cstart:cend),qts%ihessian_image,glob%havehessian)

!!$      ! Alternative: finite-difference Hessian in cartesians
!!$      call dlf_fdhessian(glob%nvar,.false.,glob%energy,&
!!$          glob%xcoords, &
!!$          glob%xgradient,qts%xhessian_image,glob%havehessian)

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
      call dlf_fail("QTS rate calculation only possible for&
          & inithessian 0, 2, or 6")
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
    call write_qts_hessian(glob%nat,1,neb%varperimage,glob%temperature,&
        neb%ene(qts%image_status),neb%xcoords(:,qts%image_status),qts%vhessian(:,:,qts%image_status),&
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
  integer   :: nvar
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
    
    call dlf_hessian_update(neb%varperimage, &
        glob%icoords(cstart:cend),&
        qts%icoordhess(:,iimage),&
        qts%igradient(cstart:cend), &
        qts%igradhess(:,iimage), &
        qts%vhessian(:,:,iimage), havehessian, fracrecalc,was_updated)

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
      neb%ene,neb%xcoords,qts%vhessian,qts%etunnel,qts%dist,"intermediate")

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
  real(rk) :: hess_rs(neb%varperimage,neb%varperimage)
  real(rk) :: arr2(2),svar,mass(glob%nat) ! scratch
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
      svar,glob%xcoords,hess_rs,svar,arr2,mass,"rs",tok)
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
!! Calculate the tunnelling rate using the instanton method. Calls
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
  real(rk)   :: prod_evals_hess_prime,ers,qrs
  integer    :: ivar,ival,nvar,iimage,nimage,jvar,cstart,cend
  logical    :: tok
  real(rk)   :: second_au
  character(2) :: chr2
  character(50) :: filename
  real(rk)   :: cm_inv_for_amu,amu,svar,svar2
  logical    :: tbimol
  integer    :: primage,neimage
  real(rk)   :: det_tsplit
  character(20) :: label
  real(rk)   :: vb,alpha,kappa_eck
  real(rk)   :: eff_mass,norm,norm_total,eff_mass_total
  integer    :: icomp,iat
  real(rk)   :: qrot,qrot_part
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
        neb%ene,neb%xcoords,qts%vhessian,qts%etunnel,qts%dist,label)
  end if

  if(printl>=4) then
    call report_qts_pathlength()
  end if

  !print*,"jk total hessian",qts%total_hessian

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

  ! calculate the rotational partition function 
  qrot=0.D0 ! we calculate the logarithm here
  svar=sum(qts%dtau(1:nimage+1))-0.5D0*(qts%dtau(1)+qts%dtau(nimage+1))
  do iimage=1,nimage
    call rotational_partition_function(glob%nat,glob%nzero,neb%xcoords(:,iimage),qrot_part)
    qrot=qrot + (qts%dtau(iimage)+qts%dtau(iimage+1))*0.5D0/svar * log(qrot_part)
  end do

  

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
  write(*,'("Expectation value of vec0:",es18.9)') sum(vec0*matmul(total_hessian,vec0))

  if(printl>=4) write(stdout,"('Diagonalising the Hessian matrix ...')")
  call allocate(evals_hess, neb%varperimage*neb%nimage*2)
  call allocate(evecs_hess, neb%varperimage*neb%nimage*2,neb%varperimage*neb%nimage*2)
  if(hess_dtau) then
    call dlf_matrix_diagonalise(neb%varperimage*neb%nimage*2,total_hessian,evals_hess,evecs_hess)
  else
    ! Asymmetric:
    ! using MKL, the eigenvalues (singular values) are returned sorted. However,
    ! only their absolute value is returned in any case
    ! call dlf_matrix_asymm_diagonalise(neb%varperimage*neb%nimage*2,total_hessian,evals_hess)

    ! Symmetric
    call dlf_matrix_diagonalise(neb%varperimage*neb%nimage*2,total_hessian,evals_hess,evecs_hess)
  end if

  call deallocate(total_hessian)

  ! Print Eigenvalues (not all...)
  if(printl>=4) then
    call dlf_constants_get("CM_INV_FOR_AMU",CM_INV_FOR_AMu)
    call dlf_constants_get("AMU",amu)
    write(stdout,"('Eigenvalues of the qTS Hessian')")
    write(stdout,"('Number    Eigenvalue      Wave Number        Projection onto tangent')")

    do ival=1,min(15,nvar)
      if(hess_dtau) then
        svar=evals_hess(ival)*dble(2*neb%nimage)/beta_hbar*amu ! transformed EV - for E" multiplied by dtau in Hessian
      else
        svar=evals_hess(ival)*amu ! transformed EV - for E" without mult in Hessian
      end if
      svar=sqrt(abs(svar))*CM_INV_FOR_AMU
      if(evals_hess(ival)<0.D0) svar=-svar
      if(ival>1.and.ival<glob%nzero+3) then
        write(stdout,"(i6,1x,es18.9,2x,f10.3,' cm^-1 ',f5.3,', treated as zero')") &
            ival,evals_hess(ival),svar,(sum(vec0*evecs_hess(:,ival)))**2
      else
        write(stdout,"(i6,1x,es18.9,2x,f10.3,' cm^-1 ',f5.3)") ival,evals_hess(ival),&
            svar,(sum(vec0*evecs_hess(:,ival)))**2
      end if
    end do
  end if 
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

  ! calculate the log of the product rather than the product - to avoid overflow
  prod_evals_hess_prime=0.5D0* (log(abs(evals_hess(1))) + sum(log(abs(evals_hess(glob%nzero+3:nvar)))) )

  ! check if mode of image rotation is within the zero modes
  svar=0.D0
  do ival=2,glob%nzero+2
    svar=max(svar,(sum(vec0*evecs_hess(:,ival)))**2)
  end do
!!$PRINT*,"Warning: cyclic mode checker deactivated!"
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
  if(printl>=2) then
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

  call dlf_constants_get("SECOND_AU",second_au)
  call qts_reactant(qrs,ers,qrot_part,tbimol,tok)
  qrot=qrot-qrot_part
  if(.not.tok) then
    ers=0.D0
  end if
  if(printl>=2) then
    write(stdout,"('S_0                                ',es17.10,' hbar')") &
        qts%S_0
    write(stdout,"('S_pot                              ',es17.10,' hbar')") &
        qts%S_pot - ers*beta_hbar
    write(stdout,"('S_ins                              ',es17.10,' hbar')") &
        qts%S_ins - ers*beta_hbar
    write(stdout,"('E_eff                              ',es17.10)") &
        qts%S_ins/beta_hbar - ers
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
      write(stdout,"('ln dtau                            ',es17.10)") &
          log(beta_hbar/dble(2*neb%nimage))
    end if
    !write(stdout,"('  1 / (dTau*Prod( SQRT( eigvals))) ',es17.10)") &
    !    exp(-prod_evals_hess_prime)*dble(2*neb%nimage)/beta_hbar
    svar=0.5D0 * qts%S_0 + qts%S_pot
    if(tok) then
      svar=qts%S_pot - ers*beta_hbar
      svar=0.5D0 * qts%S_0 + svar
    end if
    write(stdout,"('-S_ins/hbar                        ',es17.10)") -svar !qts%S_ins
    !write(stdout,"('  exp(-S_ins/hbar)                 ',es17.10)") exp(-qts%S_ins)
  end if

  !qts%rate=sqrt(qts%S_0/(2.D0*pi*qts%const_hess))*exp(-qts%S_ins)/prod_evals_hess_prime
  if(hess_dtau) then
    ! was:
    !  qts%rate=0.5D0*log(qts%S_0/2.D0/pi) -svar &
    !      -prod_evals_hess_prime-log(beta_hbar/dble(2*neb%nimage))*&
    !      (dble(glob%nzero+1)*0.5D0+dble(neb%varperimage*neb%nimage))
    qts%rate=0.5D0*log(qts%S_0/2.D0/pi) -svar &
        -prod_evals_hess_prime-0.5D0*log(beta_hbar/dble(2*neb%nimage))
  else
    qts%rate=0.5D0*log(qts%S_0/2.D0/pi) -svar &
        -prod_evals_hess_prime!-log(beta_hbar/dble(2*neb%nimage))*dble(glob%nzero+1)
  end if

  if(printl>=2) write(stdout,"('ln(RATE * Q(RS))                   ',es18.9)") qts%rate
  if(printl>=2) write(stdout,"('   RATE * Q(RS)                    ',es18.9)") exp(qts%rate)

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
    if(printl>=2) then
      write(stdout,"('Energy of the Reactant             ',es17.10,' Hartree')") &
          ers
      write(stdout,"('ln(Q(RS))                          ',es18.9)") qrs
      write(stdout,"('ln Qrot_rel                        ',es18.9)") qrot
      write(stdout,"(' The following should be as independent of nimage as possible:')") 
      if(hess_dtau) then
        write(stdout,"('ln(Q_TS/Q_RS)_vib                  ',es18.9)") -prod_evals_hess_prime &
            -log(beta_hbar/dble(2*neb%nimage))*(dble(glob%nzero+1)*0.5D0+dble(2*neb%nimage))-qrs
      else
        write(stdout,"('ln(Q_TS/Q_RS)_vib                  ',es18.9)") -prod_evals_hess_prime -qrs
      end if
      write(stdout,"('Q(RS)                              ',es18.9)") exp(qrs)

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
        write(stdout,"('S_0/2 from the path length         ',es18.9)") qts%S_0/2.D0
        svar=2.D0*sqrt(qts%S_0/4.D0/pi)* &
            exp(-0.5D0*qts%S_0 -det_tsplit) 
        write(stdout,"('Tunnelling splitting linear path S0 ',es18.9,' Hartree')") svar
        call dlf_constants_get("CM_INV_FROM_HARTREE",svar2)
        write(stdout,"('Tunnelling splitting linear path S0 ',es18.9,' cm^-1')") svar*svar2
        
        ! calculate S_0 from potential: 
        write(stdout,'(a)') " Tunnelling splitting with S_0 calculated from S_pot and E_RS (recommended)"
        ! S_0= 2*sqrt(2) * int sqrt(E-E_b) dy
        ! S_0= 2 ( S_pot - beta_hbar E_b)
        write(stdout,"('S_0/2 from S_pot and E_RS          ',es18.9)") qts%S_pot - ers*beta_hbar
        svar2=2.D0*(qts%S_pot - ers*beta_hbar)
        ! Tunnelling splitting from linear (open) path:
        svar=2.D0*sqrt(svar2/4.D0/pi)* &
            exp(-0.5D0*svar2 -det_tsplit) 
        write(stdout,"('Tunnelling splitting linear path p  ',es18.9,' Hartree')") svar
        call dlf_constants_get("CM_INV_FROM_HARTREE",svar2)
        write(stdout,"('Tunnelling splitting linear path p  ',es18.9,' cm^-1')") svar*svar2


        write(stdout,*)
      end if
!!!!!!!! end of tunnelling splittings !!!!!!!!!!!!!!!!!!!!!!!!!!!

      ! add rotational partition function to ln rate
      qts%rate=qts%rate+qrot

      write(stdout,"('ln(Rate)                           ',es18.9)") qts%rate-qrs
      write(stdout,"('Rate                               ',es18.9)") exp(qts%rate-qrs)
      write(stdout,"('Effective free-energy barrier      ',es18.9)") &
          -(log(beta_hbar*2.D0*pi)+qts%rate-qrs)/beta_hbar
      write(stdout,"('log10(second_au)                   ',es18.9)")log(second_au)/log(10.D0)
      if(tbimol) then
        call dlf_constants_get("ANG_AU",svar)
        svar=log(svar*1.D-10) ! ln of bohr in meters
        ! the factor of 1.D6 converts from m^3 to cm^3
        write(stdout,"('ln(Rate)                           ',es18.9,' cm^3 per second')") &
            qts%rate-qrs+log(second_au)+log(1.D6)+3.D0*svar
        write(stdout,"('log10(Rate in cm^3 per second)     ',es18.9, ' at ', f10.5, ' K')") &
            (qts%rate-qrs+log(second_au)+log(1.D6)+3.D0*svar)/log(10.D0),&
            glob%temperature
      else
        write(stdout,"('ln(Rate)                           ',es18.9,' per second')") qts%rate-qrs+log(second_au) 
        write(stdout,"('log10(Rate per second)             ',es18.9, ' at ', f10.5, ' K')") &
            (qts%rate-qrs+log(second_au))/log(10.D0), glob%temperature
     end if
      !write(stdout,"('log10(Rate per second) ',es18.9)") (qts%rate-qrs+log(second_au))/log(10.D0)
      ! print Arrhenius info as log to the base 10
      write(stdout,"('Arrhenius (log10)',2f12.6)") 1.D3/glob%temperature,(qts%rate-qrs)/log(10.D0)
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
subroutine qts_reactant(qrs,ers,qrot,tbimol,tok)
  use dlf_parameter_module, only: rk
  use dlf_neb, only: neb,unitp,xyzall,beta_hbar
  use dlf_global, only: printl,stdout,glob,pi
  use dlf_constants, only : dlf_constants_get
  use dlf_allocate, only: allocate,deallocate
  use dlf_qts
  implicit none
  real(rk), intent(out) :: qrs ! ln of the partition function (no potential part)
  real(rk), intent(out) :: ers ! energy of the reactant
  real(rk), intent(out) :: qrot ! relative rotational partition function of the reactant
  logical , intent(out) :: tbimol
  logical , intent(out) :: tok
  real(rk) :: eigvals(neb%varperimage)
  integer :: ivar,iimage,varperimage_read,icount
  real(rk) :: svar,mass_bimol
  real(rk) :: qrsi,dtau
  !real(rk) :: arr2(2) ! scratch
  !real(rk) :: evals_rs( 2 * neb%varperimage * neb%nimage )
  !real(rk) :: hess_rs(neb%varperimage,neb%varperimage) 
  real(rk) :: phi_rel

  ! get eigenvalues
  real(rk) :: hess_sp(neb%nimage*2,neb%nimage*2)
  real(rk) :: eval_sp(neb%nimage*2)
  real(rk) :: evec_sp(neb%nimage*2,neb%nimage*2)
  logical  :: tconst_dtau

  ! for reweighting
  integer  :: nimage_read,iat
  real(rk) :: etunnel,temperature,dist(2)
  logical  :: tokmass
  real(rk), allocatable :: xcoords(:)
  real(rk), allocatable :: ihessian(:,:)
  real(rk), allocatable :: evecs(:,:)
  real(rk), allocatable :: mass_file(:)

  ! real bimolecular
  integer :: bimol,natr2,nimage_read2,varperimager2,natr,varperimager
  real(rk), allocatable :: eigvals2(:)
  real(rk) :: ers2
  

  tok=.false.
  tbimol=.false.
  bimol=0
  qrot=0.D0
  dtau=beta_hbar/dble(2*neb%nimage) ! only for constant dtau

  ! define if dtau is constant or variable
  tconst_dtau=(abs(maxval(qts%dtau)-minval(qts%dtau)) < 1.D-14*abs(minval(qts%dtau)))

  !tconst_dtau=.false.

  call allocate(xcoords,3*glob%nat)
  call allocate(mass_file,glob%nat)
  call read_qts_reactant(glob%nat,neb%varperimage,&
      ers,varperimage_read,xcoords,eigvals,mass_bimol,"",mass_file,tok)
  call deallocate(mass_file)
  !if(.not.tok) return

  if(.not.tok) then
    ! now we have to try and read qts_hessian_rs.txt
    if(printl>=6) write(stdout,*) "Attempting to read from file qts_hessian_rs.txt"
    
    varperimage_read=neb%varperimage ! bimolecular+reweighting not possible
    call allocate(ihessian,varperimage_read,varperimage_read)
    call allocate(evecs,varperimage_read,varperimage_read)
    call allocate(mass_file,glob%nat)
    
    nimage_read=1
    call read_qts_hessian(glob%nat,nimage_read,varperimage_read,temperature,&
        ers,xcoords,ihessian,etunnel,dist,mass_file,"rs",tok)
    
    if(.not.tok.and.nimage_read==1) then
      ! check for bimolecular case
      call head_qts_hessian(natr,nimage_read,varperimager,"rs",tok)
      if(tok) then
        call head_qts_hessian(natr2,nimage_read2,varperimager2,"rs2",tok)
        tok=(tok.and.nimage_read2==1)
      end if
      tok=(tok.and.nimage_read==1)
      if(tok) then
        ! we seem to have two molecules reacting
        if(printl>=4) write(stdout,*) "Bimolecular case with more than one atom in each molecule."
        
        bimol=natr2

        call deallocate(xcoords)
        call deallocate(ihessian)
        call deallocate(evecs)
        call deallocate(mass_file)

        call allocate(ihessian,varperimager,varperimager)
        call allocate(evecs,varperimager,varperimager)
        call allocate(mass_file,natr)
        call allocate(xcoords,3*natr)

        call read_qts_hessian(natr,nimage_read,varperimager,temperature,&
            ers,xcoords,ihessian,etunnel,dist,mass_file,"rs",tok)
        ! check for tok

        mass_bimol=sum(mass_file)
        call dlf_matrix_diagonalise(varperimager,ihessian,eigvals(1:varperimager),evecs)

        ! rotational partition function (relative)
        call rotational_partition_function(natr,glob%nzero,xcoords,qrot)

        call deallocate(xcoords)
        call deallocate(ihessian)
        call deallocate(evecs)
        call deallocate(mass_file)

        ! now deal with RS2

        call allocate(ihessian,varperimager2,varperimager2)
        call allocate(evecs,varperimager2,varperimager2)
        call allocate(eigvals2,varperimager2)
        call allocate(mass_file,natr2)
        call allocate(xcoords,3*natr2)

        call read_qts_hessian(natr2,nimage_read,varperimager2,temperature,&
            ers2,xcoords,ihessian,etunnel,dist,mass_file,"rs2",tok)
        ! check for tok

        svar=sum(mass_file)
        mass_bimol=(mass_bimol*svar)/(mass_bimol+svar)
        call dlf_constants_get("AMU",svar)
        mass_bimol=mass_bimol/svar
        if(printl>=4) write(stdout,'(a,f20.10,a)') " Using a reduced mass of ",&
            mass_bimol," amu calculated from the two fragments"

        call dlf_matrix_diagonalise(varperimager2,ihessian,eigvals2,evecs)

        if(natr2==2) then
          ivar=5
        else
          ivar=6
        end if
        
        !print*,"sizes",neb%varperimage,varperimager,varperimager2,ivar

        if(varperimager+varperimager2-ivar>neb%varperimage) then
          if(printl>=4) write(stdout,*) "Variables per image in bimolecular&
              & rates not consistent"
          tok=.false.
          return
        end if

        eigvals(varperimager+1:varperimager+varperimager2-ivar)=eigvals2(ivar+1:varperimager2)
        varperimage_read=varperimager+varperimager2-ivar
        if(printl>=4) write(stdout,'(a,i5)') "Number of vibrational modes from reactants ", &
            varperimage_read-glob%nzero

        ! rotational partition function (relative)
        call rotational_partition_function(natr2,ivar,xcoords,svar)
        qrot=qrot*svar
        if(bimol==2) qrot=qrot*2.D0/beta_hbar
        if(bimol>2) qrot=qrot*sqrt(8.D0*pi)/sqrt(beta_hbar**3)

        call deallocate(eigvals2)
        !call deallocate(xcoords)
        call deallocate(ihessian)
        call deallocate(evecs)
        !call deallocate(mass_file)
        mass_file=-1.D0 ! avoid re-weighting below

        !print*,"BIMOL STUFF FINISHED"

      end if ! real bimolecular (more than one atom)
    end if

    if(.not.tok.or.nimage_read/=1) then
      call deallocate(xcoords)
      call deallocate(ihessian)
      call deallocate(evecs)
      call deallocate(mass_file)
      return
    end if

    if(bimol==0) then

      ! If masses are not read in (returned negative), do no mass-weighting and
      ! assume glob%mass is correct
      if(minval(mass_file) > 0.D0) then
        call dlf_constants_get("AMU",svar)
        
        tokmass=.true.
        do iat=1,glob%nat
          if(abs(mass_file(iat)-glob%mass(iat))>1.D-7) then
            tokmass=.false.
            if(printl>=6) &
                write(stdout,*) "Mass of atom ",iat," differs from Hessian file. File:",&
                mass_file(iat)/svar," input",glob%mass(iat)/svar
          end if
        end do
        
        ! Re-mass-weight
        if(.not.tokmass) then
          call dlf_re_mass_weight_hessian(glob%nat,varperimage_read,mass_file/svar,glob%mass/svar,ihessian)
        end if
      end if
      
      call dlf_matrix_diagonalise(varperimage_read,ihessian,eigvals,evecs)
      
      call deallocate(ihessian)
      call deallocate(evecs)
    end if
    call deallocate(mass_file)

  end if ! if(.not.tok) after read_qts_reactant

  ! calculate rotational partition function (the logarithm is used in instanton rates)
  if (bimol==0) call rotational_partition_function(glob%nat,glob%nzero,xcoords,qrot)
  qrot=log(qrot)

  call deallocate(xcoords)

  ! partition function with an infinite number of images
  qrsi=0.D0
  do ivar=glob%nzero+1,varperimage_read
    ! the following is the limit for nimage->infinity
    ! for error compensation, however, we should use the same number of images as in the TS
    qrsi=qrsi-log(2.D0*sinh(sqrt(abs(eigvals(ivar)))*beta_hbar*0.5D0))
  end do
  if(printl>=4.and.hess_dtau) write(stdout,"('Ln Vibrational part of the RS partition &
      &function (infinite images) ',es18.9)") &
      qrsi+log(dtau)*dble(neb%nimage*neb%varperimage)!-dble(glob%nzero)*log(beta_hbar)

  !print*,"Rotational part of the RS partition function (infinite images)",qrs

  ! partition function with nimage images (but equi-spaced tau)
  qrs=0.D0
  do iimage=1,2*neb%nimage
    do ivar=glob%nzero+1,varperimage_read!neb%varperimage
      qrs=qrs+log(4.D0*(sin(pi*dble(iimage)/dble(2*neb%nimage)))**2/dtau+&
          dtau*eigvals(ivar))
    end do
  end do
  !qrs=1.D0/sqrt(qrs)
  qrs=-0.5D0*qrs-log(dble(2*neb%nimage))*dble(glob%nzero) + &
      log(dtau)*0.5D0*dble(glob%nzero*(2*neb%nimage-1))
  if(printl>=4.and.hess_dtau) write(stdout,"('Ln Vibrational part of the RS partition &
      &function (nimage images)   ',es18.9)") qrs
  if(printl>=4.and.hess_dtau) write(stdout,"(' The latter is used')")

  ! Reactant state partition function for non-equal spaced dtau
  if(.not.hess_dtau) then

    qrsi=qrsi+log(dtau)*dble(2*neb%nimage*varperimage_read)
    if(printl>=4) write(stdout,"('Ln Vibrational part of the RS partition &
        &function (infinite images)                  ',es18.9)") qrsi

    ! transform the above value to what we would expect with the asymmetric dtau notation
    qrs=qrs+log(dtau)*dble(2*neb%nimage*varperimage_read-glob%nzero)*0.5D0
    if(printl>=4) write(stdout,"('Ln Vibrational part of the RS partition &
        &function (nimage images, equi-spaced tau)   ',es18.9)") qrs
    ! it turns out that the value obtained above is (slightly) different from
    ! the one obtained by diagonalising the Spring-matrix (for constant
    ! dtau). I guess, the analytic value is better. Maybe I should use that
    ! one?

    if(tconst_dtau) then

      write(stdout,"(' The latter is used')")

    else

      ! calculate the partition function for non-constant dtau

!!$    !
!!$    ! calculate the partition function from a full diagonalisation of the whole Hessian
!!$    ! this is not needed, but the code stays in here for possible future tests
!!$    !
!!$    ! get the whole hessian of the reactant
!!$    ivar=1 ! nimage
!!$    call read_qts_hessian(glob%nat,ivar,neb%varperimage,svar,&
!!$        svar,glob%xcoords,hess_rs,svar,arr2,"rs",tok)
!!$    if(.not.tok) then
!!$      print*,"Warning: full reactant state Hessian not available (qts_hessian_rs.txt)"
!!$      return
!!$    end if
!!$    ! write it repeatedly into qts%total_hessian
!!$    qts%total_hessian=0.D0
!!$    ivar=neb%nimage*neb%varperimage
!!$    do iimage=1,neb%nimage
!!$      qts%total_hessian(neb%cstart(iimage):neb%cend(iimage),neb%cstart(iimage):neb%cend(iimage)) = &
!!$          hess_rs
!!$      qts%total_hessian(ivar+neb%cstart(iimage):ivar+neb%cend(iimage),ivar+neb%cstart(iimage):ivar+neb%cend(iimage)) = &
!!$          hess_rs
!!$    end do
!!$    
!!$    ! convert the hessian of potential energies to one of action (modify
!!$    ! diagonal, add spring contributions)
!!$    call qts_hessian_etos(2,neb%nimage,neb%varperimage,qts%dtau,qts%total_hessian)
!!$
!!$    ! diagonalise qts%total_hessian asymmetrically 
!!$    call dlf_matrix_asymm_diagonalise(neb%varperimage*neb%nimage*2,qts%total_hessian,evals_rs)
!!$
!!$    ! get the product of its (non-zero) eigenvalues (which are all for
!!$    ! nzero=0) qrs=0.D0
!!$    do ivar=1,2*neb%nimage*neb%varperimage
!!$      if(ivar<=glob%nzero) then
!!$        !print*,"RS-EV",ivar,evals_rs(ivar)," skipped"
!!$      else
!!$        !if(ivar<=12) print*,"RS-EV",ivar,evals_rs(ivar)
!!$        qrs=qrs+log(evals_rs(ivar))
!!$      end if
!!$    end do
!!$    qrs=-0.5D0*qrs
!!$
!!$    ! <<<< end of full diagonalisation of the whole Hessian


      ! construct hessian of only spring forces:
      hess_sp(:,:)=0.D0
      ! convert the hessian of potential energies to one of action (modify
      ! diagonal, add spring contributions)
      call qts_hessian_etos(2,neb%nimage,1,qts%dtau,hess_sp)
      call dlf_matrix_diagonalise(neb%nimage*2,hess_sp,eval_sp,evec_sp)

      ! tests
!!$    print*,"Evec_sp:",eval_sp
!!$    print*,"eigval",eigvals
!!$    
!!$    !now check if every eigenvalue has its counterpart:
!!$    do iimage=1,neb%nimage*2
!!$      do ivar=1,neb%varperimage
!!$        svar=huge(1.D0)
!!$        do icount=1,neb%varperimage*neb%nimage*2
!!$          svar=min(svar,abs(eval_sp(iimage)+eigvals(ivar)-evals_rs(icount)))
!!$        end do
!!$        print*,svar,iimage,ivar
!!$      end do
!!$    end do

!!$    svar=1.D0
!!$    do iimage=2,2*neb%nimage
!!$      svar=svar*eval_sp(iimage)
!!$    end do
!!$    print*,"prod",svar,1.D0/svar

      !calculate qrs from eval_sp and eigvals
      qrs=0.D0
      do iimage=1,neb%nimage*2
        do ivar=1,varperimage_read!neb%varperimage
          if(iimage==1.and.ivar<=glob%nzero) cycle
          qrs=qrs+log(abs(eval_sp(iimage)+eigvals(ivar)))
        end do
      end do
      qrs=-0.5D0*qrs

    end if ! (tconst_dtau) 

    ! correct for bimolecular rates (less degrees of freedom in the RS
    ! partition function than in the TS partition function)
    if(varperimage_read/=neb%varperimage) then
      if(printl>=4) write(stdout,*) "Calculating a bimolecular rate."

      ! product of all dtau
      svar=sum(log(qts%dtau(2:neb%nimage)))*2.D0
      svar=svar+log(qts%dtau(1))+log(qts%dtau(neb%nimage+1))
!      qrs=qrs+ log(beta_hbar/dble(2*neb%nimage)) * &
!          dble(2*neb%nimage*(neb%varperimage-varperimage_read))

!      print*,"correcting ln(Q(RS)) by 1",log(beta_hbar/dble(2*neb%nimage)) * &
!          dble(2*neb%nimage*(neb%varperimage-varperimage_read))
!      print*,"correcting ln(Q(RS)) by 2",svar*(neb%varperimage-varperimage_read)

      qrs=qrs+ svar*(neb%varperimage-varperimage_read)
      
      if(mass_bimol>0.D0) then
        ! trans partition function:
        call dlf_constants_get("AMU",svar)
        !print*,"mass_bimol",mass_bimol
        !print*,"mass_bimol in au",mass_bimol*svar,svar
        phi_rel=mass_bimol*svar/2.D0/pi/beta_hbar
        phi_rel=phi_rel*sqrt(phi_rel)
        !print*,"translational partition function of incoming atom",phi_rel
        if(printl>=4) write(stdout,*) "ln(translational partition function &
            &of incoming atom)",log(phi_rel)
        !print*,"log10(translational partition function of incoming atom)",log(phi_rel)/log(10.D0)
        
        qrs=qrs+log(phi_rel)

        tbimol=.true.

      else
        if(printl>=2) write(stdout,*) "Less DOF in the reactant than in&
            & the TS indicates a bimolecular reaction."        
        if(printl>=2) write(stdout,*) "However, no mass for the incoming &
            &particle is given, thus the translational partition function is ignored."        
      end if
    end if !(varperimage_read/=neb%varperimage)  = bimolecular

    if(printl>=4) write(stdout,"('Ln Vibrational part of the RS partition &
        &function (nimage images)                    ',es18.9)") qrs

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
!!****f* qts/write_qts_coords
!!
!! FUNCTION
!!
!! Write coordinates, dtau, etunnel, and dist to qts_coords.txt
!!
!! SYNOPSIS
subroutine write_qts_coords(nat,nimage,varperimage,temperature,&
    S_0,S_pot,S_ins,ene,xcoords,dtau,etunnel,dist)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none
  integer, intent(in) :: nat,nimage,varperimage
  real(rk),intent(in) :: temperature,S_0,S_pot,S_ins
  real(rk),intent(in) :: ene(nimage)
  real(rk),intent(in) :: xcoords(3*nat,nimage)
  real(rk),intent(in) :: dtau(nimage+1)
  real(rk),intent(in) :: etunnel
  real(rk),intent(in) :: dist(nimage+1)
  character(128) :: filename

  if(glob%iam > 0 ) return ! only task zero should write

  filename="qts_coords.txt"
  if (glob%ntasks > 1) filename="../"//trim(filename)

  open(unit=555,file=filename, action='write')
  write(555,*) "Coordinates of the qTS path written by dl-find"
  write(555,*) nat,nimage,varperimage
  write(555,*) temperature
  write(555,*) S_0,S_pot
  write(555,*) S_ins
  write(555,*) ene
  write(555,*) xcoords
  write(555,*) "Delta Tau"
  write(555,*) dtau
  write(555,*) etunnel
  write(555,*) dist
  close(555)

end subroutine write_qts_coords
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/read_qts_coords
!!
!! FUNCTION
!!
!! Read coordinates, dtau, etunnel, and dist from qts_coords.txt
!!
!! SYNOPSIS
subroutine read_qts_coords(nat,nimage,varperimage,temperature,&
    S_0,S_pot,S_ins,ene,xcoords,dtau,etunnel,dist)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout,printl,glob
  implicit none
  integer, intent(in)    :: nat
  integer, intent(inout) :: nimage
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
  if (glob%ntasks > 1) filename="../"//trim(filename)

  inquire(file=filename,exist=there)
  if(.not.there) call dlf_fail("qts_coords.txt does not exist! Start structure&
      & for qts hessian is missing.")

  open(unit=555,file=filename, action='read')
  read(555,FMT='(a)',end=201,err=200) 
  read(555,*,end=201,err=200) nat_,nimage_,varperimage_
  if(nat/=nat_) call dlf_fail("Error reading qts_coords.txt file: Number of &
      &atoms not consistent")
  
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
    if(printl>=2) write(stdout,*) "Warning: could not read S_pot from qts_coords.txt"
  end if
  read(555,*,end=201,err=200) ene(1:min(nimage,nimage_))
  read(555,*,end=201,err=200) xcoords(1:3*nat,1:min(nimage,nimage_))
  ! try and read dtau (not here in old version, and we have to stay consistent)
  read(555,fmt="(a)",iostat=ios) line
  if(ios==0) then
    read(555,*,end=201,err=200) dtau(1:1+min(nimage,nimage_))
    read(555,*,end=201,err=200) etunnel
    read(555,*,end=201,err=200) dist(1:1+min(nimage,nimage_))
  else
    if(printl>=2) write(stdout,*) "Warning, dtau not read from qts_coords.txt, using constant dtau"
    dtau=-1.D0
    etunnel=-1.D0  
    dist(:)=-1.D0 ! set to useless value to flag that it was not read
  end if

  close(555)
  nimage=nimage_

  if(printl >= 4) write(stdout,"('qts_coords.txt file successfully read')")

  return

  ! return on error
  close(100)
200 continue
  call dlf_fail("Error reading qts_coords.txt file")
  write(stdout,10) "Error reading file"
  return
201 continue
  call dlf_fail("Error (EOF) reading qts_coords.txt file")
  write(stdout,10) "Error (EOF) reading file"
  return

10 format("Checkpoint reading WARNING: ",a) 
end subroutine read_qts_coords
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/write_qts_hessian
!!
!! FUNCTION
!!
!! Write Hessians of individual images to disk (qts_hessian.txt)
!!
!! SYNOPSIS
subroutine write_qts_hessian(nat,nimage,varperimage,temperature,&
    ene,xcoords,hessian,etunnel,dist,label)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,printl,stdout
  use dlf_constants, only: dlf_constants_get
  use dlf_qts, only: taskfarm_mode
  implicit none
  integer, intent(in) :: nat,nimage,varperimage
  real(rk),intent(in) :: temperature
  real(rk),intent(in) :: ene(nimage)
  real(rk),intent(in) :: xcoords(3*nat,nimage)
  real(rk),intent(in) :: hessian(varperimage,varperimage,nimage)
  real(rk),intent(in) :: etunnel
  real(rk),intent(in) :: dist(nimage+1)
  character(*),intent(in):: label
  integer             :: iimage
  character(128)      :: filename
  real(rk)            :: svar

  if(glob%iam > 0 ) then
    ! only task zero should write
    if(taskfarm_mode==1) return
    ! image-Hessians should be written for taskfarm_mode=2 by their respective
    ! tasks. The caller must make sure in this case that the routine is only
    ! called if there are valid data in hessian.
    if(glob%iam_in_task>0) return
    if(label(1:5)/="image") return
  end if

  if(trim(label)/="") then
    filename='qts_hessian_'//trim(label)//'.txt'
    if(printl>=4) write(stdout,*) "Writing Hessian file ",trim(filename)
  else
    filename='qts_hessian.txt'
  end if
  if (glob%ntasks > 1) filename="../"//trim(filename)
  
  open(unit=555,file=filename, action='write')
  write(555,*) "Coordinates and Hessian of the qTS path written by dl-find"
  write(555,*) nat,nimage,varperimage
  write(555,*) temperature
  write(555,*) ene
  write(555,*) "Coordinates"
  write(555,*) xcoords
  write(555,*) "Hessian per image"
  do iimage=1,nimage
    write(555,*) hessian(:,:,iimage)
  end do
  write(555,*) "Etunnel"
  write(555,*) etunnel
  write(555,*) dist
  write(555,*) "Masses in au"
  if((glob%icoord==190.or.glob%icoord==390).and.glob%iopt/=11.and.glob%iopt/=13) then
    svar=1.D0
  else
    call dlf_constants_get("AMU",svar)
  end if
  write(555,*) glob%mass*svar
  close(555)

end subroutine write_qts_hessian
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/read_qts_hessian
!!
!! FUNCTION
!!
!! Read V-Hessians of individual images from disk (qts_hessian.txt)
!!
!! On output, the hessian may be only partially filled (depending of
!! nimage read). It is returned as read from the file.
!!
!! SYNOPSIS
subroutine read_qts_hessian(nat,nimage,varperimage,temperature_,&
    ene,xcoords,hessian,etunnel,dist,mass,label,tok)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout,glob
  implicit none
  integer, intent(in)    :: nat
  integer, intent(inout) :: nimage
  integer, intent(in)    :: varperimage
  real(rk),intent(in)    :: temperature_ ! not used at the moment
  real(rk),intent(out)   :: ene(nimage)
  real(rk),intent(out)   :: xcoords(3*nat,nimage)
  real(rk),intent(out)   :: hessian(varperimage,varperimage,nimage)
  real(rk),intent(out)   :: etunnel
  real(rk),intent(out)   :: dist(nimage+1)
  real(rk),intent(out)   :: mass(nat)
  character(*),intent(in):: label
  logical ,intent(out)   :: tok
  !
  integer :: iimage,ios
  logical :: there
  integer :: nat_,nimage_,varperimage_
  real(rk):: temperature
  character(128) :: filename

  tok=.false.
  if(trim(label)/="") then
    filename='qts_hessian_'//trim(label)//'.txt'
    if(printl>=4) write(stdout,*) "Searching for hessian file ",trim(filename)
  else
    filename='qts_hessian.txt'
  end if
  if (glob%ntasks > 1) filename="../"//trim(filename)

  inquire(file=filename,exist=there)
  if(.not.there) return

  open(unit=555,file=filename, action='read')
  read(555,FMT='(a)',end=201,err=200) 
  read(555,*,end=201,err=200) nat_,nimage_,varperimage_
  if(nat/=nat_) then
    if(printl>=4) write(*,*) "Error reading ",trim(filename)," file: Number of &
        &atoms not consistent"
    if(printl>=4) write(*,*) "Number of atoms expected",nat
    if(printl>=4) write(*,*) "Number of atoms got     ",nat_
    close(555)
    return
  end if
  if(varperimage/=varperimage_) then
    if(printl>=4) write(*,*) "Error reading ",trim(filename)," file: Variables &
        &per image not consistent"
    close(555)
    return
  end if

  read(555,*,end=201,err=200) temperature
  if(printl>=4) then
    if(temperature>0.D0) then
      write(stdout,'("Reading qTS hessian of temperature ",f10.5," K")') temperature
    else
      write(stdout,'("Reading classical Hessian")')
    end if
  end if
  read(555,*,end=201,err=200) ene(1:min(nimage,nimage_))

  read(555,FMT='(a)',end=201,err=200) 
  read(555,*,end=201,err=200) xcoords(1:3*nat,1:min(nimage,nimage_))

  read(555,FMT='(a)',end=201,err=200) 
  hessian=0.D0
  do iimage=1,min(nimage,nimage_)
    read(555,*,end=201,err=200) hessian(:,:,iimage)
  end do
  ! read etunnel
  read(555,FMT='(a)',iostat=ios)
  if(ios==0) then
    read(555,*,end=201,err=200) etunnel
    read(555,*,end=201,err=200) dist(1:1+min(nimage,nimage_))
  else
    etunnel=-1.D0 ! tag as unread...
    dist=-1.D0
  end if
  ! read mass
  read(555,FMT='(a)',iostat=ios)
  if(ios==0) then
    read(555,*,end=201,err=200) mass
  else
    mass(:)=-1.D0 ! tag as unread
  end if

  close(555)

  nimage=nimage_

  if(printl >= 6) write(stdout,"(a,' file successfully read')") trim(filename)

  tok=.true.
  return

  ! return on error
  close(500)
200 continue
  call dlf_fail("Error reading "//trim(filename)//" file")
  !write(stdout,10) "Error reading file"
  !return
201 continue
  call dlf_fail("Error (EOF) reading qts_hessian.txt file")
  !write(stdout,10) "Error (EOF) reading file"
  !return

end subroutine read_qts_hessian
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/head_qts_hessian
!!
!! FUNCTION
!!
!! Read V-Hessians of individual images from disk (qts_hessian.txt)
!!
!! On output, the hessian may be only partially filled (depending of
!! nimage read). It is returned as read from the file.
!!
!! SYNOPSIS
subroutine head_qts_hessian(nat,nimage,varperimage,label,tok)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout,glob
  implicit none
  integer, intent(out)    :: nat
  integer, intent(out) :: nimage
  integer, intent(out)    :: varperimage
  character(*),intent(in):: label
  logical ,intent(out)   :: tok
  !
  integer :: iimage,ios
  logical :: there
  real(rk):: temperature
  character(128) :: filename

  tok=.false.
  if(trim(label)/="") then
    filename='qts_hessian_'//trim(label)//'.txt'
    if(printl>=4) write(stdout,*) "Searching for hessian file ",trim(filename)
  else
    filename='qts_hessian.txt'
  end if
  if (glob%ntasks > 1) filename="../"//trim(filename)

  inquire(file=filename,exist=there)
  if(.not.there) return

  open(unit=555,file=filename, action='read')
  read(555,FMT='(a)',end=2010,err=2000) 
  read(555,*,end=2010,err=2000) nat,nimage,varperimage
  close(555)

  tok=.true.
  return

  ! return on error
2000 continue
  call dlf_fail("Error reading "//trim(filename)//" file in head_qts_hessian")
  return
2010 continue
  call dlf_fail("Error (EOF) reading qts_hessian.txt file")
  return

end subroutine head_qts_hessian
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/write_qts_reactant
!!
!! FUNCTION
!!
!! Write energy and Hessian Eigenvalues of the reactant to disk
!! (qts_reactant.txt). Called in dlf_thermo. Eigval are in mass-weighted
!! cartesians.
!!
!! SYNOPSIS
subroutine write_qts_reactant(nat,varperimage,&
    ene,xcoords,eigvals,label)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,printl,stdout ! for masses
!  use dlf_neb, only: neb ! cstart and cend
  implicit none
  integer, intent(in) :: nat,varperimage
  real(rk),intent(in) :: ene
  real(rk),intent(in) :: xcoords(3*nat)
  real(rk),intent(in) :: eigvals(varperimage)
  character(*),intent(in):: label
  character(128) :: filename

  if(glob%iam > 0 ) return ! only task zero should write

  if(trim(label)=="ts") then
    filename='qts_ts.txt'
    if(printl>=4) write(stdout,*) "Writing file qts_ts.txt"
  else if(trim(label)/="") then
    filename='qts_reactant_'//trim(label)//'.txt'
    if(printl>=4) write(stdout,*) "Writing file ",trim(filename)
  else
    filename='qts_reactant.txt'
  end if
  if (glob%ntasks > 1) filename="../"//trim(filename)

  open(unit=555,file=filename, action='write')
  write(555,'(a)') "Energy and Hessian eigenvalues of the reactant for qTS written by dl-find &
      &(for bimolecular reactions: add energy of incoming atom and include mass (in amu) after the energy)"
  write(555,*) nat,varperimage
  write(555,*) ene
  write(555,*) "Coordinates"
  write(555,*) xcoords
  write(555,*) "Hessian Eigenvalues"
  write(555,*) eigvals
  write(555,*) "Masses in amu (M(12C)=12)"
  write(555,*) glob%mass
  close(555)

end subroutine write_qts_reactant
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/read_qts_reactant
!!
!! FUNCTION
!!
!! Read energy and Hessian Eigenvalues of the reactant from disk
!! (qts_reactant.txt). Eigval are in mass-weighted cartesians.
!!
!! In case of bimolecular reactions, the file has to be hand-edited:
!! the (electronic) energy of the incoming atom should be added to the
!! energy (third line), and after the energy, the mass of the incoming
!! atom in amu should be put (with a space as separator)
!!
!! SYNOPSIS
subroutine read_qts_reactant(nat,varperimage,&
    ene,varperimage_read,xcoords,eigvals,mass_bimol,label,mass,tok)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_neb, only: neb ! cstart and cend
  use dlf_global, only: glob,printl,stdout
  use dlf_constants, only: dlf_constants_get
  implicit none
  integer, intent(in)  :: nat,varperimage
  real(rk),intent(out) :: ene
  integer ,intent(out) :: varperimage_read
  real(rk),intent(out) :: xcoords(3*nat)
  real(rk),intent(out) :: eigvals(varperimage)
  real(rk),intent(out) :: mass_bimol ! reduced mass of the incoming
                                     ! part of a bimolecular reaction
                                     ! set to -1 if not read
  character(*),intent(in):: label
  real(rk),intent(out) :: mass(nat)
  logical ,intent(out) :: tok
  !
  logical  :: there,tokmass
  integer  :: nat_,varperimage_,ios,iat
  character(128) :: line
  real(8)  :: svar
  character(128) :: filename

  ene=huge(1.D0)
  tok=.false.

  ! find file name
  if(trim(label)=="ts") then
    filename='qts_ts.txt'
    !if(printl>=4) write(stdout,*) "Reading file qts_ts.txt"
  else if(trim(label)/="") then
    filename='qts_reactant_'//trim(label)//'.txt'
    if(printl>=4) write(stdout,*) "Searching for file ",trim(filename)
  else
    filename='qts_reactant.txt'
  end if
  if (glob%ntasks > 1) filename="../"//trim(filename)

  inquire(file=filename,exist=there)
  if(.not.there) return

  open(unit=555,file=filename, action='read')
  read(555,FMT='(a)',end=201,err=200) 
  read(555,*,end=201,err=200) nat_,varperimage_
  ! allow for bimolecular reactions
  if(nat<nat_) then
    if(printl>=2) write(*,*) "Error reading ",trim(filename)," file: Number of &
        &atoms not consistent"
    return
  end if
  if(varperimage<varperimage_) then
    if(printl>=2) write(*,*) "Error reading qts_reactant.txt file: Variables &
        &per image not consistent"
    return
  end if

  !read(555,*,end=201,err=200) ene
  read(555,fmt="(a)",end=201,err=200) line
  read(line,*,iostat=ios) ene,mass_bimol
  if(ios/=0) then
    read(line,*) ene
    mass_bimol=-1.D0
  end if

  read(555,FMT='(a)',end=201,err=200) 
  read(555,*,end=201,err=200) xcoords(1:3*nat_)

  eigvals=0.D0
  read(555,FMT='(a)',end=201,err=200) 
  read(555,*,end=201,err=200) eigvals(1:varperimage_)

  ! read mass
  read(555,FMT='(a)',iostat=ios)
  if(ios==0) then
    read(555,*,end=201,err=200) mass(1:nat_)
    ! check consistency of masses
    if(nat==nat_) then
      tokmass=.true.
      if((glob%icoord==190.or.glob%icoord==390).and.&
          glob%iopt/=11.and.glob%iopt/=13) then
        call dlf_constants_get("AMU",svar)
      else
        svar=1.D0
      end if
      do iat=1,nat
        if(abs(mass(iat)-glob%mass(iat)/svar)>1.D-7) then
          tokmass=.false.
          if(printl>=2) &
              write(stdout,*) "Mass of atom ",iat," inconsistent. File:",mass(iat)," input",glob%mass(iat)/svar
        end if
      end do
      if(.not.tokmass) then
        if(printl>=2) &
            write(stdout,*) "Masses inconsistent, this file ",trim(filename)," can not be used"
        return
      end if
    else
      if(printl>=4) then
        write(stdout,*) "Masses can not be checked for bimolecular reactions. "
        write(stdout,*) " Make sure manually that the correct masses were used in ",trim(filename)
      end if
    end if
  else
    if(printl>=4) write(stdout,*) "Masses not read from ",trim(filename)
  end if
  
  close(555)

  if(printl >= 6) write(stdout,"(a,' file successfully read')") trim(filename)

  varperimage_read=varperimage_

  tok=.true.
  return

  ! return on error
  close(100)
200 continue
  !call dlf_fail("Error reading qts_reactant.txt file")
  write(stdout,10) "Error reading file"
  return
201 continue
  !call dlf_fail("Error (EOF) reading qts_reactant.txt file")
  write(stdout,10) "Error (EOF) reading file"
  return

10 format("Checkpoint reading WARNING: ",a) 

end subroutine read_qts_reactant
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/head_qts_reactant
!!
!! FUNCTION
!!
!! Read read only nat and varperimage from qts_reactant.txt for allocation
!!
!! SYNOPSIS
subroutine head_qts_reactant(nat,varperimage,label,tok)
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout,printl
  implicit none
  integer, intent(out) :: nat,varperimage
  character(*),intent(in):: label
  logical ,intent(out) :: tok
  !
  logical  :: there
  character(128) :: filename

  tok=.false.

  ! find file name
  if(trim(label)=="ts") then
    filename='qts_ts.txt'
    !if(printl>=4) write(stdout,*) "Reading file qts_ts.txt"
  else if(trim(label)/="") then
    filename='qts_reactant_'//trim(label)//'.txt'
    if(printl>=4) write(stdout,*) "Searching for file ",trim(filename)
  else
    filename='qts_reactant.txt'
  end if

  inquire(file=filename,exist=there)
  if(.not.there) return

  open(unit=555,file=filename, action='read')
  read(555,FMT='(a)',end=201,err=200) 
  read(555,*,end=201,err=200) nat,varperimage

  close(555)
  tok=.true.
  return

  ! return on error
  close(100)
200 continue
  call dlf_fail("Error reading qts_reactant.txt file")
  write(stdout,10) "Error reading file"
  close(555)
  return
201 continue
  call dlf_fail("Error (EOF) reading qts_reactant.txt file")
  write(stdout,10) "Error (EOF) reading file"
  close(555)
  return

10 format("Checkpoint reading WARNING: ",a) 

end subroutine head_qts_reactant
!!****

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
!! *  The latter + the simplified Wigner correction
!! *  The latter + the full Wigner correction (for T>Tc)
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
!  use dlf_neb, only: neb,unitp,beta_hbar
!  use dlf_qts
  use dlf_allocate, only: allocate,deallocate
  use dlf_constants, only: dlf_constants_init,dlf_constants_get
  implicit none
  ! user parameters
  integer   :: nat !  number of atoms TS
  integer   :: natr ! number of atoms R
  integer   :: varperimager ! number of degrees of freedom R
  integer   :: varperimage ! number of degrees of freedom TS
  integer   :: nzero_r  ! number of zeroes R
  integer   :: nzero_ts  ! number of zeroes TS
  logical   :: bimolat  ! bimolecular with only one atom for the second molecule
  integer   :: bimol    ! truely bimolecular: number of atoms for the second molecule (0 for unimol)
  real(rk)  :: tstart ! lower end of the temperature range
  real(rk)  :: tend  ! upper end of the temperature range
  integer   :: tsteps ! number of temperatures at which to calculate the rates
  real(rk)  :: mu_bim  ! reduced mass (amu) for calculating phi_rel
  real(rk)  :: twopimuktoverh2  ! part of relative translational p.fu. per unit volume
  real(rk)  :: phi_rel  ! relative translational p.fu. per unit volume

  real(rk)  :: zpene_rs
  real(rk)  :: zpene_ts
  real(rk)  :: ZPE,VAD
  real(rk)  :: ene_rs
  real(rk)  :: ene_ts
  real(rk),allocatable  :: eigvals_rs(:)
  real(rk),allocatable  :: eigvals_ts(:)
  logical   :: tok
  real(rk)  :: lrate_cl ! log of rate
  real(rk)  :: lrate_qvib! log of qq-rate (HJ-notation) multiplied by (2pi)**3
                        ! to make it consistent with the classical rate
  real(rk)  :: tfact,temp,beta_hbar,KBOLTZ_AU,svar
  real(rk)  :: tdel
  real(rk)  :: wigner,wigner_simple
  real(rk)  :: tcross
  integer   :: itemp,ivar
  integer   :: varperimage_read,varperimage_readr
  real(rk),allocatable  :: rate_wigner(:)
  real(rk)  :: second_au
  real(rk)  :: log10,timeunit

  logical   :: tkie
  logical   :: chkfilepresent
  real(rk)  :: hcl,hqq,hwig
  character(128) :: sline
  real(rk)  :: avogadro,hartree,kjmol,echarge,ev,amc,planck,kboltz
  real(rk)  :: zpe_rs,zpe_ts
  real(rk), allocatable :: xcoords(:) ! 3*nat
  real(rk), allocatable :: ihessian(:,:) ! varperimage,varperimage
  real(rk), allocatable :: evecs(:,:) ! varperimage,varperimage
  real(rk), allocatable :: mass_file(:) ! nat
  integer   :: nimage_read,iat
  real(rk)  :: temperature,etunnel,dist(2)
  logical   :: tokmass
  ! Rotational partition function
  real(rk)  :: qrot_red,qrot_rs,qrot_ts
  ! Eckart for the transition mode
  logical   :: teck=.true.
  real(rk)  :: Vb,alpha
  real(rk)  :: kappa_eck,heck
  real(rk)  :: wavenumber,frequency_factor,amu,wavenumberts
  real(rk)  :: bvar
  integer   :: varperimager2,natr2
  real(rk)  :: ene_rs2
  real(rk), allocatable :: eigvals_rs2(:)
  character(128) :: filename

  if(glob%iam > 0 ) return ! only task zero should do this (task farming does
                           ! not make sense for such fast calculations. No
                           ! energies and gradients are ever calculated).

  ! some small header/Info
  if(printl>=2) then
    write(stdout,*) "Calculating the reaction rate based on harmonic transition state theory"
  end if

! now read this from class.in
! this is structured as
! line 1: nat_r, nat_ts ! <-- not used any more
! line 2: n0_r, n0_ts
! line 3: tstart, tend, tsteps
! line 4: bimolecular (T/F)
  bimolat=.false.
  bimol=0
  filename="class.in"
  if (glob%ntasks > 1) filename="../"//trim(filename)
  INQUIRE(FILE=filename,EXIST=chkfilepresent)
  IF (chkfilepresent) THEN
    OPEN(28,FILE=filename,STATUS='old',ACTION='read')
    READ (28,'(a)') sline 
    !READ (28,*), natr, nat
!    varperimager=natr*3
!   varperimage=nat*3
    READ (28,*) nzero_r, nzero_ts
    READ (28,*) tstart,tend,tsteps
    bimolat=.false.
    READ (28,'(a)',err=201,end=201) sline
    bimolat=(sline(1:1)=="T")
201 continue
    !IF (bimolat) then 
     ! READ (28,*), mu_bim
     ! print *, '# Read a reduced mass of ', mu_bim, ' amu for calculating bimolecular rates'
    !end IF
    REWIND(28)
    CLOSE(28)
  ELSE
    if(printl>=0) then
      write(stdout,*) " Read Error: file class.in not found"
      write(stdout,*) " The file provides input for rate calculations and should contain the lines:"
      write(stdout,*) " Line 1: ignored"
      write(stdout,*) " Line 2: N_zero(RS) N_zero(TS) ! Number of zero modes in RS and TS"
      write(stdout,*) " Line 3: T_start T_end Number_of_steps ! Temperature"
      write(stdout,*) " Line 4: T ! if a bimolecular reaction should be considered"
    end if
    call dlf_fail("File class.in missing")
  END IF
  call allocate(rate_wigner,tsteps)
  
  if(printl>=4) write(stdout,'(" Number of zero modes in RS and TS:",2i5)') nzero_r,nzero_ts

  rate_wigner(:)=0.D0

  ! read data of reactant
  if(printl>=6) write(stdout,*) "reading qts_reactant.txt"

  ! first get natr and varperimager. This is done also if Hessian should be
  ! re-mass-weighted, because we need these from the file
  call head_qts_reactant(natr,varperimager,"",tok)
  if(.not.tok) call dlf_fail("Error reading reactant")
  call allocate(eigvals_rs,varperimager)
  call allocate(xcoords,3*natr)

  ! Now read the full reactant data
  call allocate(mass_file,natr)
  call read_qts_reactant(natr,varperimager,&
      ene_rs,varperimage_readr,xcoords,eigvals_rs,mu_bim,"",mass_file,tok)
  call deallocate(mass_file)

  if(bimolat.and.mu_bim<0.D0) then
    ! bimolecular case with more than one atom, read in complete RS data below
    tok=.true.
    print*,"Bimolecular case"
  end if

  if(.not.tok.and.(.not.bimolat)) then ! the same is done for the TS below ...
    ! now we have to try and read qts_hessian_rs.txt
    if(printl>=6) write(stdout,*) "Attempting to read from file qts_hessian_rs.txt"

    call allocate(ihessian,varperimager,varperimager)
    call allocate(evecs,varperimager,varperimager)
    call allocate(mass_file,natr)

    nimage_read=1
    call read_qts_hessian(natr,nimage_read,varperimager,temperature,&
        ene_rs,xcoords,ihessian,etunnel,dist,mass_file,"rs",tok)
    
    if(.not.tok) call dlf_fail("Error reading reactant from Hessian file.")
    if(nimage_read/=1) call dlf_fail("Wrong number of images for RS.")
    varperimage_readr=varperimager ! bimolecular+reweighting not possible

    ! If masses are not read in (returned negative), do no mass-weighting and
    ! assume glob%mass is correct
    if(minval(mass_file) > 0.D0) then
      call dlf_constants_get("AMU",svar)
      mass_file=mass_file/svar

      tokmass=.true.
      do iat=1,natr
        if(abs(mass_file(iat)-glob%mass(iat))>1.D-7) then
          tokmass=.false.
          if(printl>=6) &
              write(stdout,*) "Mass of atom ",iat," differs from Hessian file. File:",mass_file(iat)," input",glob%mass(iat)
        end if
      end do
      
      ! Re-mass-weight
      if(.not.tokmass) then
        call dlf_re_mass_weight_hessian(glob%nat,varperimager,mass_file,glob%mass,ihessian)
      end if
    end if

    call dlf_matrix_diagonalise(varperimager,ihessian,eigvals_rs,evecs)

    ! write hessian and qts_reactant file for later use?
    call write_qts_hessian(natr,nimage_read,varperimager,-1.D0,&
        ene_rs,xcoords,ihessian,etunnel,dist,"rs_mass")

    call deallocate(ihessian)
    call deallocate(evecs)
    call deallocate(mass_file)


  end if

  ! rotational partition function (relative)
  call rotational_partition_function(natr,glob%nzero,xcoords,qrot_rs)

  call deallocate(xcoords)

  ! handle bimolecular cases
  if(bimolat.and.mu_bim<0.D0) then
    if(printl>2) then
      write(stdout,*) "Bimolecular rate with each molecule larger than one atom"
      write(stdout,*) "Attempting to read qts_hessian_rs2.txt"
    end if

    ! RS (first one)
    call head_qts_hessian(natr,nimage_read,varperimager,"rs",tok)
    if(.not.tok) then
      if(printl>0) then
        write(stdout,*) "File qts_hessian_rs.txt required for bimolecular &
            &rates with each molecule larger than one atom"
      end if
      call dlf_fail("Error reading qts_hessian_rs.txt")
    end if
    if(varperimager/=3*natr)  call dlf_fail("Frozen atoms are not possible with bimolecular rates")

    call allocate(ihessian,varperimager,varperimager)
    call allocate(evecs,varperimager,varperimager)
    call deallocate(eigvals_rs)
    call allocate(eigvals_rs,varperimager)
    call allocate(mass_file,natr)
    call allocate(xcoords,3*natr)

    nimage_read=1
    call read_qts_hessian(natr,nimage_read,varperimager,temperature,&
        ene_rs,xcoords,ihessian,etunnel,dist,mass_file,"rs",tok)

    mu_bim=sum(mass_file)
    
    if(.not.tok) call dlf_fail("Error reading reactant from Hessian rs file.")
    if(nimage_read/=1) call dlf_fail("Wrong number of images for RS.")

    call dlf_matrix_diagonalise(varperimager,ihessian,eigvals_rs,evecs)

    !print*,"Eigenvalues RS", eigvals_rs

    ! rotational partition function (relative)
    call rotational_partition_function(natr,nzero_r,xcoords,qrot_rs)

    call deallocate(ihessian)
    call deallocate(evecs)
    call deallocate(mass_file)
    call deallocate(xcoords)

    ! RS2
    call head_qts_hessian(natr2,nimage_read,varperimager2,"rs2",tok)
    if(.not.tok) then
      if(printl>0) then
        write(stdout,*) "File qts_hessian_rs2.txt required for bimolecular &
            &rates with each molecule larger than one atom"
      end if
      call dlf_fail("Error reading qts_hessian_rs2.txt")
    end if
    bimol=natr2
    if(varperimager2/=3*natr2)  call dlf_fail("Frozen atoms are not possible with bimolecular rates")

    call allocate(ihessian,varperimager2,varperimager2)
    call allocate(evecs,varperimager2,varperimager2)
    call allocate(eigvals_rs2,varperimager2+varperimager)
    call allocate(mass_file,natr2)
    call allocate(xcoords,3*natr2)

    nimage_read=1
    call read_qts_hessian(natr2,nimage_read,varperimager2,temperature,&
        ene_rs2,xcoords,ihessian,etunnel,dist,mass_file,"rs2",tok)
    
    if(.not.tok) call dlf_fail("Error reading reactant from Hessian rs2 file.")
    if(nimage_read/=1) call dlf_fail("Wrong number of images for RS2.")

    ! calculate reduced mass
    svar=sum(mass_file)
    mu_bim=(mu_bim*svar)/(mu_bim+svar)
    call dlf_constants_get("AMU",svar)
    mu_bim=mu_bim/svar

    call dlf_matrix_diagonalise(varperimager2,ihessian,eigvals_rs2(1:varperimager2),evecs)

    !print*,"Eigenvalues RS2", eigvals_rs2(1:varperimager2)

    ! copy all eigenvalues to eigvals_rs2
    eigvals_rs2(varperimager2+1:)=eigvals_rs
    call deallocate(eigvals_rs)
    if(natr2==2) then
      ivar=5
    else
      ivar=6
    end if
    varperimage_readr=varperimager+varperimager2-ivar

    call allocate (eigvals_rs,varperimager2+varperimager)

    ! now the eigenvalues (vib. frequencies) have to be arranged to eigvals_rs:
    ! the first eigenvalues of eigvals_rs
    ! then all but the first 6 (5) eigenvalues of eigvals_rs2
    !print*,"varperimager,varperimager2,ivar,varperimage_readr",varperimager,varperimager2,ivar,varperimage_readr
    eigvals_rs(1:varperimager)=eigvals_rs2(varperimager2+1:)
    eigvals_rs(varperimager+1:)=eigvals_rs2(ivar+1:varperimager2)
    nzero_r=nzero_r
    varperimager=varperimage_readr

    ene_rs=ene_rs+ene_rs2

    ! rotational partition function (relative)
    call rotational_partition_function(natr2,ivar,xcoords,svar)
    qrot_rs=qrot_rs*svar

    if(printl>=2) write(stdout,'(a,f20.10,a)') " Using a reduced mass of ",&
        mu_bim, " amu calculated from the two fragments"

    call deallocate(xcoords)
    call deallocate(ihessian)
    call deallocate(evecs)
    call deallocate(mass_file)

  end if

  if(bimolat.and.bimol<2) then
    if(printl>=2) write(stdout,'(a,f20.10,a)') " Read a reduced mass of ",&
        mu_bim, " amu for calculating bimolecular rates"
  end if

  ! read data of TS
  if(printl>=6) write(stdout,*) "# reading qts_ts.txt"
  call head_qts_reactant(nat,varperimage,"ts",tok)
  if(.not.tok) then
    call dlf_fail("Error: need at least first two lines of qts_ts.txt")
  end if
  call allocate(eigvals_ts,varperimage)
  call allocate(xcoords,3*nat)
  call allocate(mass_file,nat)
  call read_qts_reactant(nat,varperimage,&
      ene_ts,ivar,xcoords,eigvals_ts,svar,"ts",mass_file,tok)
  call deallocate(mass_file)

  if(.not.tok) then ! the same is done for the RS above ...
    ! now we have to try and read qts_hessian_ts.txt
    if(printl>=6) write(stdout,*) "Attempting to read from file qts_hessian_ts.txt"

    call allocate(ihessian,varperimage,varperimage)
    call allocate(evecs,varperimage,varperimage)
    call allocate(mass_file,nat)

    nimage_read=1
    call read_qts_hessian(nat,nimage_read,varperimage,temperature,&
        ene_ts,xcoords,ihessian,etunnel,dist,mass_file,"ts",tok)
    
    if(.not.tok) call dlf_fail("Error reading TS from Hessian file.")
    if(nimage_read/=1) call dlf_fail("Wrong number of images for TS.")
    varperimage_read=varperimage ! bimolecular+reweighting not possible

    ! If masses are not read in (returned negative), do no mass-weighting and
    ! assume glob%mass is correct
    if(minval(mass_file) > 0.D0) then
      call dlf_constants_get("AMU",svar)
      mass_file=mass_file/svar
      
      tokmass=.true.
      do iat=1,natr
        if(abs(mass_file(iat)-glob%mass(iat))>1.D-7) then
          tokmass=.false.
          if(printl>=6) &
              write(stdout,*) "Mass of atom ",iat," differs from Hessian file. File:",mass_file(iat)," input",glob%mass(iat)
        end if
      end do
      
      ! Re-Mass-Weighting...
      if(.not.tokmass) then
        call dlf_re_mass_weight_hessian(glob%nat,varperimage,mass_file,glob%mass,ihessian)
      end if
    end if

    call dlf_matrix_diagonalise(varperimage,ihessian,eigvals_ts,evecs)

    ! write hessian and qts_reactant file for later use?
    call write_qts_hessian(nat,nimage_read,varperimage,-1.D0,&
        ene_ts,xcoords,ihessian,etunnel,dist,"ts_mass")

    call deallocate(ihessian)
    call deallocate(evecs)
    call deallocate(mass_file)

  end if

  ! rotational partition function (relative)
  call rotational_partition_function(nat,glob%nzero,xcoords,qrot_ts)
  qrot_red=qrot_ts/qrot_rs

  call deallocate(xcoords)

  if(printl>=3) then
    if(bimol==0) then
      write(stdout,*)          "                    Reactant        TS"
      write(stdout,'(a,2i10)') " Number of atoms   ",natr,nat
      write(stdout,'(a,2i10)') " Degrees of freedom",varperimager,varperimage
    else
      write(stdout,*)          "                   Reactant1 Reactant2        TS"
      write(stdout,'(a,3i10)') " Number of atoms   ",natr,natr2,nat
      write(stdout,'(a,i10,10x,i10)') " Degrees of freedom",varperimager,varperimage
    endif
  end if

  if(.not.tok) call dlf_fail("Error reading TS")

  ! print *,"eigvals_rs", eigvals_rs
  ! print *,"eigvals_ts", eigvals_ts
  call dlf_constants_get("KBOLTZ_AU",KBOLTZ_AU)
  call dlf_constants_get("SECOND_AU",second_au)
  call dlf_constants_get("HARTREE",HARTREE)
  call dlf_constants_get("AVOGADRO",avogadro)
  call dlf_constants_get("ECHARGE",echarge)
  call dlf_constants_get("CM_INV_FOR_AMU",frequency_factor)
  call dlf_constants_get("AMU",amu)
  kjmol=avogadro*hartree*1.D-3
  ev=hartree/echarge
  !print*,"KBOLTZ_AU",KBOLTZ_AU

  ! print all the frequencies (in case of high print level)
  if(printl>=6) then
    write(stdout,"(a)") "Vibrational Frequencies (wave numbers in cm^-1) of the reactant and the TS"
    write(stdout,"(' Mode     Eigenvalue Frequency     Eigenvalue Frequency ')")
    do ivar=1,max(varperimager,varperimage)
      wavenumber = sqrt(abs(amu*eigvals_rs(ivar))) * frequency_factor
      if(eigvals_rs(ivar)<0.D0) wavenumber=-wavenumber
      wavenumberts = sqrt(abs(amu*eigvals_ts(ivar))) * frequency_factor
      if(eigvals_ts(ivar)<0.D0) wavenumberts=-wavenumberts
      if(ivar<=varperimager.and.ivar<=varperimage) then
        write(stdout,"(i5,f15.10,f10.3,f15.10,f10.3)") ivar,eigvals_rs(ivar),wavenumber, &
            eigvals_ts(ivar),wavenumberts
      else
        if(ivar<=varperimager) then
          write(stdout,"(i5,f15.10,f10.3)") ivar,eigvals_rs(ivar),wavenumber
        end if
        if(ivar<=varperimage) then
          write(stdout,"(i5,25x,f15.10,f10.3)") ivar, &
              eigvals_ts(ivar),wavenumberts
        end if
      end if
    end do
  end if

  temp=tstart
  tfact=(tend/tstart)**(1.D0/dble(tsteps-1))
  tdel=(tstart-tend)/(tend*tstart*dble(tsteps-1))
  !print*,"tdel",tdel
  
  zpe_rs=0.D0
  zpe_ts=0.D0
  do ivar=1,varperimager
   if (ivar>nzero_r) zpe_rs = zpe_rs + (0.5D0*SQRT(abs(eigvals_rs(ivar))))
  end do
  do ivar=1,varperimage
   if (ivar>nzero_ts+1) zpe_ts = zpe_ts + (0.5D0*SQRT(abs(eigvals_ts(ivar))))
  end do

  ZPE=zpe_ts-zpe_rs 
  VAD=ene_ts-ene_rs+ZPE

  ! crossover temperature
  tcross=sqrt(abs(eigvals_ts(1)))*0.5D0/pi/KBOLTZ_AU

  if(printl>=3) then
    write(stdout,'(a)') "                                      Hartree           kJ/mol            eV                K"
    write(stdout,'(a,4f18.8)') "Potential energy Barrier     ", &
         ene_ts-ene_rs,(ene_ts-ene_rs)*kjmol,  (ene_ts-ene_rs)*ev,(ene_ts-ene_rs)/KBOLTZ_AU
    write(stdout,'(a,4f18.8)') "ZPE Correction               ", &
        ZPE, ZPE*kjmol,ZPE*ev,ZPE/KBOLTZ_AU
    write(stdout,'(a,4f18.8)') "Vibrational adiabatic barrier", &
        VAD,VAD*kjmol,VAD*ev,VAD/KBOLTZ_AU
    svar=KBOLTZ_AU*temp*log(qrot_red)
    write(stdout,'(a,4f18.8)') "Rotational contr. at start T ", &
        svar,svar*kjmol,svar*ev,svar/KBOLTZ_AU
    write(stdout,'(a,f18.8,1x,a)') "Crossover Temperature        ",tcross,"K"
  end if
  
  ! Eckart stuff
  !Vb=4.D0*(ene_ts-ene_rs)
  Vb=4.D0*(vad)
  alpha=sqrt(abs(eigvals_ts(1))*8.D0/vb)
  ! Parameters for the Eckart approximation (commented out)
  !print*,"Eckart: V'',Emax",abs(eigvals_ts(1)),(ene_ts-ene_rs)
  !print*,"Eckart: Vb, alpha",Vb,alpha


  ! Set the unit and the base of the logarithm
  log10=log(10.D0)        ! for log to the base of 10
  timeunit=log(second_au) ! for printout in s^-1
  !log10=1.D0              ! for natural log
  !timeunit=0.D0           ! for printout in at. u.

  ! Adjust the printout to the settings above!
  if(printl>=3) then
    if (bimolat) then
      write(stdout,*) "log_10 of rates in cm^3 sec^-1"
      !write(stdout,*) "log_10 of rates in cm^3/at.u."
      !write(stdout,*) "ln of rates in cm^3 sec^-1"
      !write(stdout,*) "ln of rates in cm^3/at.u."
    else
      write(stdout,*) "log_10 of rates in second^-1"
      !write(stdout,*) "ln of rates in second^-1"
      !write(stdout,*) "log_10 of rates in at.u."
      !write(stdout,*) "ln of rates in at.u."
    end if
    write(stdout,'(a,f18.8)') "Change of log(rate) by the rotational partition function",log(qrot_red)/log10

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
      if(printl>=4) write(stdout,*) "File rate_H found, calculating KIE and writing to file kie."
      open(file="kie",unit=13)
      write(13,'(a)') "#       T[K]              KIE  classical   KIE w. quant. vib   KIE simpl. Wigner   KIE Eckart"
    else
      if(printl>0) write(stdout,*) "File rate_H found, but not readable, no KIE calculated."
    end if
  end if

  call dlf_constants_get("AMC",amc)
  call dlf_constants_get("PLANCK",planck)
  call dlf_constants_get("KBOLTZ",kboltz)

  open(file="arrhenius",unit=15)
  open(file="arrhenius_polywigner",unit=17)
  open(file="free_energy_barrier",unit=16)
  if(printl>=2) write(stdout,'(a)') &
      "       1000/T           rate classical       quantised vib.      simpl. Wigner       Eckart"
  write(15,'(a)') &
      "#      1000/T           rate classical       quantised vib.      simpl. Wigner       Eckart"
  write(16,'(a)') &
      "#     T           free-energy barrier from rate with quantised vib."
  write(17,'(a)') &
      "#      1000/T              s-Wig 2             s-Wig 4             s-Wig 6             s-Wig 8"

  do itemp=1,tsteps

    !print*,"Temperature",itemp,temp
    beta_hbar=1.D0/(temp*KBOLTZ_AU)  ! beta * hbar = hbar / (kB*T)
    !print*,"beta_hbar",beta_hbar

    if(bimolat) then
      twopimuktoverh2= 2.D0 * pi * mu_bim * amc !amc=1.66053886E-027
      twopimuktoverh2= twopimuktoverh2 / planck !planck=6.626068E-034
      twopimuktoverh2= twopimuktoverh2 * kboltz * temp ! kboltz=1.3806503E-023
      twopimuktoverh2= twopimuktoverh2 / planck ! planck=6.626068E-034
      twopimuktoverh2= twopimuktoverh2 * SQRT(twopimuktoverh2)
      twopimuktoverh2= 1.D-6 * twopimuktoverh2
      phi_rel=log(twopimuktoverh2)
      !print *, 'log10(phi_rel)', phi_rel/log10, 'at T', temp
      if(bimol==2) qrot_rs=qrot_rs*2.D0/beta_hbar
      if(bimol>2) qrot_rs=qrot_rs*sqrt(8.D0*pi)/sqrt(beta_hbar**3)
    end if

    lrate_cl=0.D0
    lrate_qvib=0.D0
    do ivar=1,varperimage
      if(ivar<=varperimage_readr.and.ivar>nzero_r) then 
        lrate_cl=lrate_cl+log(abs(eigvals_rs(ivar)))
        lrate_qvib=lrate_qvib+log(2.D0*sinh(0.5D0*sqrt(abs(eigvals_rs(ivar)))*beta_hbar))
      end if
      if(ivar>1+nzero_ts) then
        lrate_cl=lrate_cl-log(abs(eigvals_ts(ivar)))
        lrate_qvib=lrate_qvib-log(2.D0*sinh(0.5D0*sqrt(abs(eigvals_ts(ivar)))*beta_hbar))
      end if

    end do
    ! orig:
    lrate_cl= 0.5D0*lrate_cl -log(2.D0*pi) + ( -beta_hbar * (ene_ts-ene_rs) )
!!$    ! modified for flux:
!!$    if(itemp==1) print*,"Classical rate modified for flux"
!!$    lrate_cl= 0.5D0*lrate_cl -log(2.D0*pi*beta_hbar) + ( -beta_hbar * (ene_ts-ene_rs) )
  
    ! rotational part:
    lrate_cl=lrate_cl+log(qrot_red)
    lrate_qvib=lrate_qvib+log(qrot_red)

    if (bimolat) lrate_cl=lrate_cl-phi_rel 
    lrate_qvib= lrate_qvib -log(2.D0*pi*beta_hbar) + ( -beta_hbar * (ene_ts-ene_rs) )
    if (bimolat) lrate_qvib=lrate_qvib-phi_rel 

    wigner_simple=1.D0+1.D0/24.D0*beta_hbar**2*abs(eigvals_ts(1))
    
    call kappa_eckart(beta_hbar,Vb,alpha,kappa_eck)

    if(temp>tcross) then
      svar=sqrt(abs(eigvals_ts(1)))*0.5D0*beta_hbar
      wigner=svar/sin(svar)
      rate_wigner(itemp)=(lrate_qvib+log(wigner)+timeunit)/log10
    else
      rate_wigner(itemp)=0.D0
    end if

    if(printl>=2) write(stdout,'(5f20.12)') &
        1000.D0/temp,(lrate_cl+timeunit)/log10,(lrate_qvib+timeunit)/log10, &
        (lrate_qvib+log(wigner_simple)+timeunit)/log10, (lrate_qvib+log(kappa_eck)+timeunit)/log10
!    write(15,'(5f20.12)') beta_hbar,(lrate_cl+timeunit)/log10,(lrate_qvib+timeunit)/log10, &
    write(15,'(5f20.12)') 1000.D0/temp,(lrate_cl+timeunit)/log10,(lrate_qvib+timeunit)/log10, &
        (lrate_qvib+log(wigner_simple)+timeunit)/log10, (lrate_qvib+log(kappa_eck)+timeunit)/log10
    write(16,'(2f20.12)') temp,-(lrate_qvib+log(beta_hbar*2.D0*pi))/beta_hbar!, &
        !-(lrate_qvib+log(wigner_simple)+log(beta_hbar*2.D0*pi))/beta_hbar

    !polywigner
    bvar=beta_hbar*sqrt(abs(eigvals_ts(1)))*0.5D0
!    write(17,'(5f20.12)') beta_hbar, &
    write(17,'(5f20.12)') 1000.D0/temp, &
        (lrate_qvib+log(1.D0+bvar**2/6.D0)+timeunit)/log10, &
        (lrate_qvib+log(1.D0+bvar**2/6.D0+bvar**4*7.D0/15.D0/24.D0)+timeunit)/log10, &
        (lrate_qvib+log(1.D0+bvar**2/6.D0+bvar**4*7.D0/15.D0/24.D0 + bvar**6*31.D0/21.D0/720.D0)+timeunit)/log10, &
        (lrate_qvib+log(1.D0+bvar**2/6.D0+bvar**4*7.D0/15.D0/24.D0 + bvar**6*31.D0/21.D0/720.D0 &
            + bvar**8*127.D0/15.D0/40320.D0)+timeunit)/log10
     !   (lrate_qvib+log(1.D0-1.d0/avar+cosh(bvar*sqrt(avar/3.D0))/avar)+timeunit)/log10
        
    if(tkie) then
      read(sline,'(5f20.12)') svar,hcl,hqq,hwig,heck
      write(13,'(5f20.12)') temp,exp(hcl*log10-(lrate_cl+timeunit)), &
          exp(hqq*log10-(lrate_qvib+timeunit)),exp(hwig*log10-(lrate_qvib+log(wigner_simple)+timeunit)),&
          exp(heck*log10-(lrate_qvib+log(kappa_eck)+timeunit))
      if(abs(svar-1000.D0/temp) > 1.D-6) then
        write(13,*) "#Error: temperatures don't match. Here: ",temp,", rate_H:",1000.D0/svar
        close(13)
        close(12)
        tkie=.false.
      end if
      if(tkie.and.itemp<tsteps) read(12,FMT="(a)") sline
    end if

    !temp=temp*tfact
    temp=temp/(1.D0+temp*tdel)
  end do

  if(tkie) then
    close(13)
    close(12)
  end if
 
  ! now print wigner results where defined
  temp=tstart
  if(tend>tcross.or.tstart>tcross) then
    if(printl>=2) write(stdout,*)
    if(printl>=2) write(stdout,*) "#       1000/T           full wigner"
    write(15,*)
    write(15,*) "#       1000/T           full wigner"
    do itemp=1,tsteps
      if(temp>tcross) then
        if(printl>=2) write(stdout,'(2f20.12)') 1000.D0/temp,rate_wigner(itemp)
         write(15,'(2f20.12)') 1000.D0/temp,rate_wigner(itemp)
      end if
      temp=temp/(1.D0+temp*tdel)
    end do
  end if
  close(15)
  close(16)
  close(17)
  call deallocate(rate_wigner)
  call deallocate(eigvals_rs)
  call deallocate(eigvals_ts)

end subroutine dlf_htst_rate
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* qts/rotational_partition_function
!!
!! FUNCTION
!!
!! Calculate the relative rotational partition function for one
!! geometry (actually, this is just the square root of the product of
!! eigenvalues of the moment of inertia). Return 1 for any frozen atoms.
!!
!! INPUTS
!!
!! nzero, xcoords, glob%mass
!!
!! OUTPUTS
!! 
!! qrot
!!
!! SYNOPSIS
! rotational partition function
subroutine rotational_partition_function(nat,nzero,xcoords,qrot)
  !! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  implicit none
  integer, intent(in)  :: nat
  integer, intent(in)  :: nzero
  real(rk), intent(in) :: xcoords(3*nat)
  real(rk), intent(out):: qrot
  real(rk) :: com(3)
  real(rk) :: xcoords_rel(3*nat) ! relative to the center of mass
  integer  :: iat,icor,jcor
  real(rk) :: inert(3,3),eigval(3),eigvec(3,3)

  qrot=1.D0

  ! ignore frozen atoms
  if(nzero/=5.and.nzero/=6) return
  if(nzero==5.and.nat>2) return

  ! get the center of mass
  com=0.D0
  do iat=1,nat
    com = com + glob%mass(iat)* xcoords(3*iat-2:3*iat)
  end do
  com=com/sum(glob%mass)

  do iat=1,nat
    xcoords_rel(3*iat-2:3*iat)=xcoords(3*iat-2:3*iat) - com
  end do

  if(nzero==5) then
    qrot=glob%mass(1)*sum(xcoords_rel(1:3)**2) + glob%mass(2)*sum(xcoords_rel(4:6)**2)
    !
    return
    !
  end if

  ! now a larger molecule
  inert(:,:)=0.D0
  do iat=1,nat
    ! off-diagonal part
    do icor=1,3
      do jcor=icor+1,3
        inert(icor,jcor)=inert(icor,jcor) - glob%mass(iat) * &
            xcoords_rel(3*iat-3+icor)*xcoords_rel(3*iat-3+jcor)
      end do
    end do
    ! diagonal elements
    inert(1,1)=inert(1,1)+ glob%mass(iat) * &
        (xcoords_rel(3*iat-1)**2+xcoords_rel(3*iat-0)**2)
    inert(2,2)=inert(2,2)+ glob%mass(iat) * &
        (xcoords_rel(3*iat-2)**2+xcoords_rel(3*iat-0)**2)
    inert(3,3)=inert(3,3)+ glob%mass(iat) * &
        (xcoords_rel(3*iat-2)**2+xcoords_rel(3*iat-1)**2)
  end do

  !symmetrize
  do icor=1,3
    do jcor=1,icor-1
      inert(icor,jcor)=inert(jcor,icor)
    end do
  end do

  call dlf_matrix_diagonalise(3,inert,eigval,eigvec)

  qrot=sqrt(eigval(1)*eigval(2)*eigval(3))

end subroutine rotational_partition_function
!!****

! kappa_eck is the tunnelling enhancement by the symmetric Eckart barrier. The
! mass is to be ignored, since we have mass-weighted coordinates
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
  
  dpar=2.D0*pi*sqrt( 2.D0*Vb - alpha**2 * 0.25D0 ) / alpha
  
  ! integrate
  kappa_eck=0.D0
  do ipoint=1,npoint
    ene=enemax*dble(ipoint-1)/dble(npoint-1)
    apar=2.D0*pi*sqrt(2.D0*ene)/alpha
    if(dpar>500.D0.and.2.D0*apar>500.D0) then
      ! catch a inf/inf case
      trans=(1.D0-exp(-2.D0*apar))/(1.D0+exp(dpar-2.D0*apar))
    else
      trans=(cosh(2.D0*apar)-1.D0)/(cosh(2.D0*apar)+cosh(dpar)) 
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

  ! now divide by classical flux
  kappa_eck=kappa_eck * beta_hbar * exp(beta_hbar*0.25D0*Vb)
end subroutine kappa_eckart

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
  real(rk)              :: mass_bimol,dene,ddene
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
        neb%ene,neb%xcoords,qts%vhessian,etunnel,dist,mass,"",tok)
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

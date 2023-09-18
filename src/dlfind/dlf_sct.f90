! **********************************************************************
! **                    Small Curvature Tunneling                     **
! **********************************************************************
!!****h* DL-FIND/sct
!!
!! NAME
!! sct
!!
!! FUNCTION
!! Calculates a transmission coefficient in the SCT-approximation
!! (Method by Truhlar, see for example DOI: 10.1002/9780470116449.ch3)
!! The first subroutine dlf_sct has to be called 
!! at the and of each main cycle while DL-FIND is performing 
!! an IRC path search (IOPT = 63). 
!! DL-FIND must run two times:
!! Once with a positive, once with a negative 
!! value of ircstep (see main.f90 file or chemshell variable).
!! The second subroutine dlf_sct_main has to be called when the
!! IRC search with dlf_sct in both directions is completed.
!! It calculates all necessary data for SCT.
!!
!! Inputs
!!    sct%icoords
!!    glob%ihessian
!!    glob%energy
!!     Initialisation routines may require additional parameters.
!!     sct%icoords are the actual coords at each step on the IRC, 
!!     written in dlf_formstep.f90
!! 
!! Outputs
!!    kappa to different temperatures for correcting
!!    the adiabatic theory or the microcanonical variation 
!!    theory. To apply to conventional TST
!!    multiply result by exp(sct%beta*(sct%VG_TS-sct%VAG_a)).
!!    The factor becomes exp(sct%beta*(sct%VG_(s*)-sct%VAG_a))
!!    for CVT in which s* is the minimum for recrossing
!!
!! COMMENTS
!! This SCT implementation assumes that the potential energy surface 
!! has 0 or 2 classical
!! turningpoints for every energy above the quantum threshold energy. 
!! The programm will fail or yield wrong results if that is not 
!! the case.
!! One of two quadrature rules can be used
!! You must use at least one of them (Gauss-Kronrod is recommended)
!! If both quadrature rules are active, the endresult
!! will only be calculated with Gauss-Kronrod.
!!
!! COPYRIGHT
!!
!!  Copyright 2018 , Alexander Denzel (denzel@theochem.uni-stuttgart.de)
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
!!
!!****

! #define DebugInformation

#define useGaussKronrod
!#define useSimpson

!If zero curvature tunneling is activated, SCT will not be calculated
!#define ZCT

module sct_module
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  implicit none
  type sct_type
    logical               :: dataopen = .false. ! Used for writing coords
                                                ! and Hessians
    logical               :: massweight ! Using mass weighted coordinates?
    real                  :: omit_steps = 2.0d0     
                             ! percent (relative to overall steps)
                             ! of steps around the TS 
                             ! that are ignored
                             ! 0 means that the TS itself is ignored
                             ! > 0 means the transition state and the
                             ! given number of images in both directions
                             ! is ignored,
                             ! < 0 means that nothing is ignored    
                             ! a value of approx. 2 is recommended
    integer               :: imagecounter  !counts the amount of images
    integer               :: nImgs                ! number of images
    integer               :: nivar                ! number of deg.freed.
    integer               :: nat                  ! number of atoms
    integer               :: nicore               ! number of internals 
                                                  ! in inner region
    integer               :: iupd  ! number of updates (Hessian) 
                                   ! since last reset
    real                  :: nBorderConst  = 0d0!5d0! ! Percent of images
                                          ! (starting from product 
                                          !  and reactant)
                                          ! that shall have effective
                                          ! masses of "1"
                                          ! (in percent of number of
                                          ! total images)
    real(rk), allocatable :: coord(:)
    real(rk), allocatable :: MuSpline(:,:)
    real(rk), allocatable :: VG_aSpline(:,:)
    real(rk), allocatable :: P_SAGSpline(:,:)
    real(rk), allocatable :: steplengths(:)
    real(rk), allocatable :: VG_a(:)              
                             ! Discrete version of the 
                             ! ground-state vib. adiab. Potential
    real(rk), allocatable :: Mu(:)  ! The effective mass along the path
    real(rk), allocatable :: Test(:)    
    real(rk)              :: beta
    real(rk)              :: E_0    ! Quantum Threshold energy
    real(rk)              :: VAG_a  ! Barrier Height of the 
                             ! ground-state vib. adiab. Pot. VG_a
    integer               :: posVAG_a_Int
    integer               :: posE_0_Int
    real(rk)              :: posE_0_exact
    real(rk)              :: posTS_exact
    real(rk)              :: posVAG_a_exact
    logical               :: splinesAlloc = .false.
    logical               :: stopper = .true.
    integer               :: ButterOrderMu = 2!2  ! 2 recommended   
                             ! Order of the Butterworth filter 
                             ! applied to mu (0 deactivates it)
    integer               :: ButterOrderVG_a = 0!2 ! 0 recommended 
                             ! Order of the Butterworth filter 
                             ! applied to VG_a (0 deactivates it)
    real(rk)              :: cutoffFrqzMu = 20D0      
                             ! Cutoff frequency of the 
                             ! Butterworth filter (mu)
                             ! In percent of Pathlength
    real(rk)              :: cutoffFrqzVG_a = 20D0    
                             ! Cutoff frequency of the 
                             ! Butterworth filter (VG_a)
                             ! In percent of Pathlength
      ! Both cutoff frequencies are given in units of
      ! ((numberOfImages-4)*2)
      ! Therefore, they are independent of the number of Images 
      ! The corresponding period length is given in a "fraction" of
      ! the whole pathlength
      ! (Cutoff frequency of "a" yields period length of
      ! "(numberOfImages-4)*2 / a"
    integer               :: nrP_SAG_Values = 129 ! Must not be 
                                                  ! divisible by 2 
                                                  ! Has an influence
                                                  ! on the precision of the 
                                                  ! P_SAG values
    real(rk),allocatable  :: P_SAG_Val(:)
    real(rk),allocatable  :: P_SAG_Dist(:)
#ifdef useGaussKronrod
    real(rk)              :: kron_eps_a = 1D-7 ! Absolute error est.
                                               ! (fine-course)
    integer               :: kron_order = 32767!4095 ! Maximum order of the
                                               ! kronrod integration
    integer               :: minKronrod = 63  ! start kronrod integ.
                                               ! at this order
                                               ! should be an element
                                               ! of the series below
    ! The Gauss-Kronrod order is always an element of the 
    ! recursive progression 
    ! a(n) = a(n-1)*2+1, with a(1) = 1
    ! first elements are: 3, 7, 15, 31, 63, 127, 255, ...
    ! But in principle any value can be chosen here
#endif
#ifdef useSimpson
    integer               :: nrInt_Values = 1001 
                             ! Number of points for the energy integral
                             ! if the Simpson sum is used
#endif
  end type sct_type
  type(sct_type) ,save :: sct
end module sct_module

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_init
!!
!! FUNCTION
!!
!! Initialization for dlf_sct
!!
!! SYNOPSIS
!! Writing default values
subroutine dlf_sct_init(nivar)
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  use sct_module
  use dlf_constants
  integer, intent(in)   ::  nivar
  allocate(sct%coord(nivar))
  sct%dataopen = .false.
end subroutine dlf_sct_init

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_init
!!
!! FUNCTION
!!
!! Deallocating etc.. for dlf_sct
!!
!! SYNOPSIS
!! Destroys everything not needed anymore
subroutine dlf_sct_destroy(nivar)
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  use sct_module
  use dlf_constants
  integer, intent(in)   ::  nivar
  if (allocated(sct%coord)) deallocate(sct%coord)
end subroutine dlf_sct_destroy

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct
!!
!! FUNCTION
!!
!! Writing all necessary information for the SCT calculations
!! while a IRC search is performed.
!!
!! SYNOPSIS
!! Writes the Hessian, Energies, and Coordinates at every point 
!! on the path. 
!! Note that the vector sct%coord must be allocated elsewhere 
!! during the initialization of DL-FIND
!! (it is written on in dlf_formstep to get the actual coordinates 
!! on the IRC path)
subroutine dlf_sct()
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  use sct_module
  use dlf_constants
  implicit none
  if (.not.sct%dataopen) then
    if (glob%ircstep>0.0d0) then !Forward
      open(unit = 728, file = "IRCposStep.bin", status = "replace", &
           action='write', access = 'stream', &
           position='append', form='unformatted')
    else !Backward
      open(unit = 728, file = "IRCnegStep.bin", status = "replace", &
           action='write', access = 'stream', &
           position='append', form='unformatted')
    end if
    write(728) glob%nivar, glob%nat, glob%nicore, glob%massweight
    sct%imagecounter = 0 
    sct%dataopen = .true.
    close(unit = 728)
  end if
  if (glob%ircstep>0.0d0) then !Forward
    open(unit = 728, file = "IRCposStep.bin", status = "old", &
         action='write', access = 'stream', position='append', &
         form='unformatted')
    open(unit = 729, file = "firstNrImages.tmpout", &
         status = "replace", action='write', position='rewind')
  else !Backward
    open(unit = 728, file = "IRCnegStep.bin", status = "old", &
         action='write', access = 'stream', position='append', &
         form='unformatted')
    open(unit = 729, file = "secondNrImages.tmpout", &
         status = "replace", action='write',position='rewind')
  end if

  sct%imagecounter = sct%imagecounter + 1
  rewind(729)
  write(729, fmt='(I10)') sct%imagecounter
  write(728)  glob%ihessian(:,:), glob%energy, sct%coord, sct%iupd
!   write(stdout,*) "step: ", glob%step
!   write(stdout,*) "steplength: ",dsqrt(dot_product(glob%step,glob%step))
!   write(stdout,*) "icoords: ", glob%icoords
  close(unit = 728)
  close(unit = 729)
end subroutine dlf_sct
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_main
!!
!! FUNCTION
!!
!! Calculates the SCT transmission coefficient kappa 
!! from the output files of dlf_sct
!!
!! INPUTS
!! output files of dlf_sct
!!
!! OUTPUTS
!! writes kappa through stdout
!!
!! SYNOPSIS
subroutine dlf_sct_main(tstart,tdel,tsteps,kappas)
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  use dlf_allocate, only: allocate,deallocate
  use sct_module
  use dlf_constants
  implicit none
  real(rk), intent(in)  :: tstart,& !start temperature
                           tdel ! delta temperature
  integer, intent(in)   :: tsteps ! number of temperature steps
  real(rk),intent(out)  :: kappas(tsteps)
  real(rk)              :: temperature
  integer               :: npmodes
  integer               :: i, j, k
  integer               :: IOstatus
  integer               :: overallImages
  integer               :: nrFirstImages
  integer               :: nrSecondImages
  integer               :: firstValidImage
  integer               :: lastValidImage
  integer               :: nrOmittedSteps
  integer               :: newPos
  integer               :: oldPos
  integer               :: posTS_Int
  integer               :: lInter
  integer               :: rInter
  integer               :: counter
  integer               :: nIntBorderConst
  integer               :: stat
  integer               :: nivar,nat
  logical               :: massweight
  real(rk)              :: findMaxForE_0(2)
  real(rk)              :: kappa 
  real(rk)              :: norm 
  real(rk)              :: norm2 
  real(rk)              :: energyShift
  real(rk)              :: tmpenergy
  real(rk)              :: tmp_dist
  real(rk)              :: tmp
  real(rk)				:: tmp_element(1),tmp_element2(1)
  logical               :: acceptForFilter
  integer, allocatable  :: iupds(:)
  real(rk),allocatable  :: tmpdistances(:)
  real(rk),allocatable  :: hessSpline(:,:)
  real(rk),allocatable  :: energies(:) 
  real(rk),allocatable  :: vibenergies(:) 
  real(rk),allocatable  :: stepderivatives(:,:) 
  real(rk),allocatable  :: vibenergies_step(:) 
  real(rk),allocatable  :: hess(:,:,:) 
  real(rk),allocatable  :: phess(:,:,:) 
  real(rk),allocatable  :: steps(:,:) 
  real(rk),allocatable  :: icoords(:,:) 
  real(rk),allocatable  :: peigvals(:,:) 
  real(rk),allocatable  :: peigvecs(:,:,:) 
  real(rk),allocatable  :: trafos(:,:,:) 
  real(rk),allocatable  :: tmpvec(:) 
  real(rk),allocatable  :: B_F(:,:)! curvature component describing the
                                   ! coupling between reaction
                                   ! coordinate and perpendicular mode m 
  real(rk),allocatable  :: curvatures_k(:) !ground state turningpoints 
  real(rk),allocatable  :: harmonicfrqzs(:) 
  real(rk),allocatable  :: tbars(:) 
  real(rk),allocatable  :: tbarderivatives(:) 
  real(rk),allocatable  :: abars(:) !temporary variable for calculation
                                    ! of the jacobian factor

#ifdef useSimpson
  if (Mod(INT(sct%nrP_SAG_Values),2).eq.0) then
    write(stderr,*) "Number of used integral Values in the ",&
                    "Simpson sum must be even!"
  end if
#endif
  ! ***************************************
  ! Read in Data for initialization
  ! Reading the number of images for every run
  open(unit = 729, file = "firstNrImages.tmpout", &
       status = "old", action='read',IOSTAT=stat,position='rewind')
  if (stat.ne.0) then
    write(stderr,*) "Error reading file 'firstNrImages.tmpout' for SCT rates",&
                    " (does it exist?)"
  end if
  read(729, fmt='(I10)') nrFirstImages
  close(unit = 729)
  open(unit = 729, file = "secondNrImages.tmpout", &
       status = "old", action='read',IOSTAT=stat,position='rewind')
  if (stat.ne.0) then
    write(stderr,*) "Error reading file 'secondNrImages.tmpout' for SCT rates",&
                    " (does it exist?)"
  end if
  read(729, fmt='(I10)') nrSecondImages
  close(unit = 729)
  sct%imagecounter = 0 ! counts number of elements
  overallImages = nrFirstImages + nrSecondImages
  ! Testcase: nrFirstImages, nrSecondImages, overallImages = 72, 56, 128

  open(unit = 728, file = "IRCnegStep.bin", status = "old",&
       action='read', access = 'stream', position='rewind', &
       form='unformatted',IOSTAT=stat)
  if (stat.ne.0) then
    write(stderr,*) "Error reading file 'IRCnegStep.bin' for SCT rates",&
                    " (does it exist? Did you run IRC with SCT=true?)"
  end if
  read(728) sct%nivar, sct%nat, sct%nicore, sct%massweight
  ! ***************************************
  ! Initialize: variables can be allocated and initialized
  call allocate(iupds,overallImages) 
  call allocate(energies,overallImages) 
  call allocate(sct%P_SAG_Dist, sct%nrP_SAG_Values)
  call allocate(sct%P_SAG_Val, sct%nrP_SAG_Values)
  call allocate(vibenergies,overallImages) 
  call allocate(sct%VG_a,overallImages) 
  call allocate(stepderivatives,sct%nivar,overallImages) 
  call allocate(vibenergies_step,overallImages)
  call allocate(hess,sct%nivar,sct%nivar,overallImages) 
  call allocate(phess,sct%nivar,sct%nivar,overallImages) 
  call allocate(steps,sct%nivar,overallImages) 
  call allocate(icoords,sct%nivar,overallImages) 
  call allocate(sct%steplengths,overallImages) 
  call allocate(peigvals,sct%nivar,overallImages) 
  call allocate(peigvecs,sct%nivar,sct%nivar,overallImages) 
  call allocate(trafos,sct%nivar,sct%nivar,overallImages) 
  call allocate(tmpvec,sct%nivar) 
  call allocate(B_F,sct%nivar,overallImages) 
  call allocate(curvatures_k,overallImages) 
  call allocate(harmonicfrqzs,overallImages) 
  call allocate(tbars,overallImages) 
  call allocate(tbarderivatives,overallImages) 
  call allocate(abars,overallImages) 
  call allocate(sct%Mu,overallImages) 
  call allocate(tmpdistances,overallImages)

  steps(:,:) = 0d0
  icoords(:,:) = 0d0
  trafos(:,:,:) = 0d0
  peigvals(:,:) = 0d0
  peigvecs(:,:,:) = 0d0
  energies(:) = 0d0
  sct%VG_a(:) = 0d0
  sct%Mu(:) = 0d0
! ======================================================================
! Read in all the data along the IRC
! ======================================================================
  do i = 0, nrSecondImages-1
!        write(stdout,*) "I read image nr", nrSecondImages-i
    read(728,IOSTAT=IOstatus)  hess(:,:,nrSecondImages-i), &
                               energies(nrSecondImages-i), &
                               icoords(:,nrSecondImages-i), &
                               iupds(nrSecondImages-i)
    if (IOstatus .gt. 0) then 
      write(stderr,*) "AN ERROR OCCURED IN THE READING of 'second' file"
    else if (IOstatus .lt. 0) then
      close (unit = 728)
      write(stdout,*) "End of 'second' file reached,",& 
                      "reading 'first' file"
      exit
    else
      !successfully read
    end if
  end do

  open(unit = 728, file = "IRCposStep.bin", status = "old", &
       action='read', access = 'stream', position='rewind', &
       form='unformatted',IOSTAT=stat)
  if (stat.ne.0) then
    write(stderr,*) "Error reading file 'IRCposStep.bin' for SCT rates",&
                    " (does it exist? Did you run IRC with SCT=true?)"
  end if
  read(728) nivar, nat, sct%nicore, massweight
  if (nivar.ne.sct%nivar .or. &
      nat.ne.sct%nat .or. &
      massweight.neqv.sct%massweight) &
      write(stderr,*) "'IRCposStep.bin' and ",&
        "'IRCnegStep.bin' are not constistent"

  ! Now ignore the last entry of the backward path, 
  ! since it is the transition state 
  ! (it will be overwritten with the first one of the next file, 
  !  which is also the transition state) 
  ! -> one image less (overallImages--)
!   write(stdout,*) "nr Images second run", nrSecondImages
  i = nrSecondImages
  read(728,IOSTAT=IOstatus)  hess(:,:,i), energies(i), icoords(:,i), &
                             iupds(i)
  overallImages = overallImages -1
  write(stdout,*) "nr images     ", overallImages
  ! Start reading the rest of the images
  do i = nrSecondImages+1, overallImages
!     write(stdout,*) "I read image nr", i
    read(728,IOSTAT=IOstatus)  hess(:,:,i), energies(i), icoords(:,i), &
                               iupds(i)
    if (IOstatus .gt. 0) then 
      write(stderr,*) "AN ERROR OCCURED IN THE READING OF 'first' file"
    else if (IOstatus .lt. 0) then
      close (unit = 728)
      write(stdout,*) "End of 'first' file reached"
      exit
    else
      ! successfully read
    end if
  end do
! ======================================================================
! Calculate the step vectors between the images
! ======================================================================
  do i = 1, overallImages-1 
    steps(:,i) = icoords(:,i+1)-icoords(:,i)
    ! check length of steps
!     write(stdout,*) steps(:,1)
    sct%steplengths (i) = sqrt(dot_product(steps(:,i),steps(:,i)))
!     write(stdout,*) "FirstStep sct%steplengthsi",i, &
!                     sct%steplengths(i), steps(:,i)
    steps(:,i) = steps(:,i)/sct%steplengths (i) 
  end do
!   sct%steplengths(nrSecondImages+sct%omit_steps-1) = sct%steplengths(1)*sct%omit_steps
!   sct%steplengths(nrSecondImages+sct%omit_steps)   = sct%steplengths(1)*sct%omit_steps
  !sct%steplengths(nrSecondImages-1) = sct%steplengths(nrSecondImages-1) * 0.095D0
  !sct%steplengths(nrSecondImages) = sct%steplengths(nrSecondImages) * 0.92D0 
  ! Da steps nur bis overallImages-1 ausgewertet werden kann 
  ! -> ein Image weniger betrachten
  overallImages = overallImages - 1 
! write(stdout,*) "OverallStepLength", sum(sct%steplengths(1:overallImages))
! ======================================================================
! Interpolate the calculated Hessians at the steps of updated Hessians
! ======================================================================
  do j = 1, sct%nivar
    do k = 1, sct%nivar
      counter = 0 ! counter counts the number of points used for interpolation
      tmp_dist = 0d0
      do i = 1, overallImages
        if (iupds(i).eq.0) then
          counter = counter + 1 
          sct%VG_a(counter) = hess(k,j,i)
          if (counter .gt. 1) then
            tmpdistances(counter-1) = tmp_dist ! used as temporary memory for steplengths
          else
            tmp = tmp_dist ! tmp is the zero x-coordinate of the spline
          end if
          tmp_dist = sct%steplengths(i)
        else
          tmp_dist = tmp_dist + sct%steplengths(i)
        end if
      end do
      if (k == 1 .and. j == 1) call allocate(hessSpline, 4, counter) ! Only for first k
      call dlf_sct_init_csplines(counter, sct%VG_a, tmpdistances, &
                                 hessSpline)
      ! get the rest of the values
      tmp_dist = 0d0
      do i = 1, overallImages
        if (iupds(i).ne.0) then
!           write(stdout,*) "Interpolated at", i
		  tmp_element(1)=tmp_dist-tmp
          call dlf_sct_eval_csplines_noCheck(counter, hessSpline,&
                                    tmpdistances, 1, &
                                    tmp_element, hess(k,j,i))       
          tmp_dist = tmp_dist + sct%steplengths(i)
        else
          tmp_dist = tmp_dist + sct%steplengths(i)
        end if
      end do
    end do
  end do
  call deallocate(hessSpline)
! ======================================================================
! Calculate vibrational modes perpendicular to the steps
! ======================================================================
  do i = 1, overallImages
!    if (iupds(i).eq.0) then ! only necessary for not updated Hessians
    ! Give vectors to span the subspace of perpendicular vibrational
    ! modes
    call giveTrRoStTrafo(npmodes, peigvecs(:,:,i), &
                         .true., icoords(:,i), steps(:,i))
    ! output: npmodes = Number of Modes that are projected out
    !         peigvecs(:,1:npmodes,i) Modes that are projected out 
    !         (not orthogonalized)
    !         peigvecs(:,1+npmodes:sct%nivar,i) Orthonormal vectors
    !         that are also orthonormal to the Modes that are 
    !         projected out
    trafos(:,:,i) = 0d0
    trafos(:,1:sct%nivar-npmodes,i) = &
                    peigvecs(:,1+npmodes:sct%nivar,i)
    ! Now the Trafo Matrix to the orthogonal subspace is in
    ! trafos(:,1:sct%nivar-npmodes,i)

    ! Construct the Hessian in the reduced subspace
    ! The following Method assumes orthonormalized vectors in
    ! peigvecs(:,npmodes+1:dimen,i)
    ! The vectors from which to construct the hessian must be in
    ! peigvecs(:,npmodes+1:dimen,i) hess is the original hessian
    call giveSubHessFromOrthogVec(sct%nivar, npmodes, &
                                  peigvecs(:,npmodes+1:sct%nivar,i), &
                                  hess(:,:,i), phess(:,:,i))
    ! projected (3N-7)x(3N-7) hessian is in phess(1:sct%nivar-npmodes,
    ! 1:sct%nivar-npmodes,i)


    ! Diagonalize the projected/reduced hessian
    call dlf_matrix_diagonalise(sct%nivar-npmodes, &
                                phess(1:sct%nivar-npmodes,&
                                      1:sct%nivar-npmodes,i),&
				peigvals(1:sct%nivar-npmodes,i),&
                                peigvecs(1:sct%nivar-npmodes,&
                                         1:sct%nivar-npmodes,i))
    ! peigvals(1:sct%nivar-npmodes,i) are now the Eigenvalues
    ! perpendicular to Trans/Rot/Step
    ! peigvecs(1:sct%nivar-npmodes, 1:sct%nivar-npmodes,i) 
    ! are now the Eigenvectors perpendicular to Trans/Rot/Step
!    end if    
  end do

! ======================================================================
! Calculate vibrational energies perpendicular to the steps
! ======================================================================
  do i = 1, overallImages 
!     write(stdout,* ) "pvecs", &
!                      peigvecs(1:sct%nivar-npmodes, 1:sct%nivar-npmodes,i)
    ! Gives vibrational energies from eigenvalues in peigval(1:nevals)
    call dlf_GSAdiabaticPot(peigvals(1:sct%nivar-npmodes,i), &
                            sct%nivar-npmodes, vibenergies(i))
    ! calculate Eigenvalue in direction of the reaction 
    ! (use last entry of peigvals as storage)
    peigvals(sct%nivar,i) = 0d0
    ! step^T * hess * step
    do j = 1, sct%nivar
      do k = 1, sct%nivar
        peigvals(sct%nivar,i) = peigvals(sct%nivar,i) &
                                 + steps(j,i)*hess(j,k,i)*steps(k,i)
      end do
    end do
    call dlf_GSAdiabaticPot(peigvals(sct%nivar,i), 1, &
                            vibenergies_step(i))
  end do
! ======================================================================
! Calculate tbars and abars
! ======================================================================
  firstValidImage = 1
  lastValidImage  = overallImages
  ! First and last step with first order finite differences
  stepderivatives(:,firstValidImage)= (steps(:,firstValidImage+1)&
                                      -steps(:,firstValidImage))/&
                                      sct%steplengths(firstValidImage)
  stepderivatives(:,lastValidImage) = (steps(:,lastValidImage)&
                                      -steps(:,lastValidImage-1))/&
                                      sct%steplengths(lastValidImage-1)
  ! All others with second order central differences
  do i = firstValidImage+1, lastValidImage-1 
    ! Steps are already normalized
    ! ircsteps can change for different paths? 
    ! -> Central differences for two different distances
    stepderivatives(:,i) = ((steps(:,i)-steps(:,i-1))/&
                               (sct%steplengths(i-1))&
                           +(steps(:,i+1)-steps(:,i))/&
                               (sct%steplengths(i))) / 2d0
  end do

  do i = firstValidImage, lastValidImage 
#ifdef DebugInformation  
    write (stdout, *) "Step_derivatives^2", i, &
                      dsqrt(dot_product(stepderivatives(:,i), &
                                        stepderivatives(:,i)))
#endif                                
    ! tmpvec shall be used as the (N-6/7) dimensional vector for the
    ! scalar product calculated for B_F 
    ! -> project the stepderivatives onto the space perpendicular to
    !    Trans/Rot/Step
    tmpvec(:) = 0d0
    do j = 1, sct%nivar-npmodes
      do k = 1, sct%nivar
	!project stepderivative in the subspace (B(T) * stepderivative)
	tmpvec(j) = tmpvec(j) + trafos(k,j,i)*stepderivatives(k,i)
      end do
    end do
    ! tmpvec(1:sct%nivar-npmodes) contains the projected components of
    ! the stepderivative
    ! Calculating the curvature components B_kF
    B_F = 0d0
    do k = 1, sct%nivar-npmodes
      B_F(k,i) = dot_product(peigvecs(1:sct%nivar-npmodes,k,i),&
                             tmpvec(1:sct%nivar-npmodes))
    end do
    ! Not considering the sign of B_F (harmonic approximation is made)
    ! the relevant curvature is given by
    curvatures_k(i) = dsqrt(dot_product(B_F(1:sct%nivar-npmodes,i),&
                                        B_F(1:sct%nivar-npmodes,i)))
    ! calculate the harmonicfrqzs \bar{omega}
    harmonicfrqzs(i) = 0d0   
    ! use GS_AdiabPot for the following??
    do j = 1, sct%nivar-npmodes
      if (peigvals(j,i).gt.0d0) then 
        harmonicfrqzs(i) = harmonicfrqzs(i) + &
                           (dsqrt(peigvals(j,i)) * &
                           (B_F(j,i)/curvatures_k(i)))**2d0 
      end if
    end do
    ! Scaling to a.u. and calculating tbars
    harmonicfrqzs(i) = dsqrt(harmonicfrqzs(i))/42.695d0
    tbars(i) = dsqrt(1.0d0/harmonicfrqzs(i))/42.695d0
!     write(stdout,*) "harmonicfrqzs ", i,  harmonicfrqzs(i)
    ! calculating the abbreviation variables abar
    abars(i) = abs(curvatures_k(i)*tbars(i))
  end do
  
! ======================================================================
! Calculate the derivative of tbar and the reduced masses
! ======================================================================
  ! First and last step with first order finite differences
  tbarderivatives(firstValidImage)= (tbars(firstValidImage+1)&
                                    -tbars(firstValidImage))/&
                                    sct%steplengths(firstValidImage)
  tbarderivatives(lastValidImage) = (tbars(lastValidImage)&
                                    -tbars(lastValidImage-1))/&
                                    sct%steplengths(lastValidImage-1)
  ! All others with central differences
  do i = firstValidImage+1, lastValidImage-1
    tbarderivatives(i) = ((tbars(i)-tbars(i-1))/sct%steplengths(i-1)+&
                         (tbars(i+1)-tbars(i))/&
                         (sct%steplengths(i)))/2d0
  end do

  do i = firstValidImage, lastValidImage
    ! The effective Masses µ_eff in units of µ 
    ! (arbitrary mass that is 1 in units of u) 
    ! -> Faktor 1.82289D3 for a.u.
!     write(stdout,*) "tbarderivatives ", i, tbarderivatives(i)
    ! SCT
   sct%Mu(i) = min(1d0, dexp(-2d0*abars(i)-(abars(i))**2d0&
                              +(tbarderivatives(i))**2d0))*1.82289D3
#ifdef ZCT
    sct%Mu(i) = 1.82289D3
#endif
#ifdef DebugInformation
    write(stdout,*) "EffMass_nonExp", i, &
                    (1d0-abars(i))**2d0+(tbarderivatives(i))**2d0
#endif                    
  end do
  nIntBorderConst = INT(REAL(lastValidImage-firstValidImage+1,rk)&
                             /1D2*sct%nBorderConst)
  do i = 1, nIntBorderConst
    sct%Mu(firstValidImage+i-1) = 1.82289D3
    sct%Mu(lastValidImage-i+1) = 1.82289D3
  end do
! ======================================================================
! Determine the Position of the transition state
! ======================================================================
  posTS_Int = sum(maxloc(energies(firstValidImage:lastValidImage)))
  posTS_Int = posTS_Int + firstValidImage - 1
  sct%posTS_exact = sum(sct%steplengths(1:posTS_Int-1))

! ======================================================================
! Overall vibrational adiabatic potential
! ======================================================================
  sct%VG_a(:) = energies(:) + vibenergies(:)


! ======================================================================
! Delete points around the Transition state
! ======================================================================
  ! Delete points around the Transition state at nrSecondImages
  ! icoords, sct%Mu, sc1t%VG_a, sct%steplengths have to be changed 
  ! (these are now different "steps")
  nrOmittedSteps = INT(REAL(overallImages-1,rk)/1D2*sct%omit_steps)
  write(stdout,*) "nrOmittedSteps", nrOmittedSteps
  ! Steplength that will be omitted
  tmp = sum(sct%steplengths(nrSecondImages-nrOmittedSteps-1: &
                            nrSecondImages+nrOmittedSteps))
  if (nrOmittedSteps.ge.0) then
    do i = nrsecondImages+1, overallImages-nrOmittedSteps
      oldPos = i+nrOmittedSteps
      newPos = i-nrOmittedSteps-1
      icoords(:,newPos) = icoords(:,oldPos)
      sct%Mu(newPos) = sct%Mu(oldPos)
      sct%VG_a(newPos) = sct%VG_a(oldPos)
!       write(stdout,*) i-nrsecondImages, " steps around TS deleted."
    end do
    lastValidImage = lastValidImage-nrOmittedSteps*2-1 !Also the TS was deleted
    ! recalculate steplengths (they are needed later on)
    sct%steplengths(nrsecondImages-nrOmittedSteps-1) = tmp
    sct%steplengths(nrsecondImages-nrOmittedSteps-1) = (tmp +&
      sqrt(dot_product(icoords(:,nrsecondImages-nrOmittedSteps)-&
                       icoords(:,nrsecondImages-nrOmittedSteps-1), &
                       icoords(:,nrsecondImages-nrOmittedSteps)-&
                       icoords(:,nrsecondImages-nrOmittedSteps-1))))/&
                       2D0
  end if  
  
! ======================================================================
! Determine the approximate Position of VAG_a
! ======================================================================
  sct%posVAG_a_Int = sum(maxloc(sct%VG_a(firstValidImage:lastValidImage)))
  sct%posVAG_a_Int = sct%posVAG_a_Int + firstValidImage - 1
! ======================================================================
! Calculate the Quantum threshold energy
! ======================================================================



  if (glob%bimol_sct) then
    print*, "Calculating a bimolecular SCT rate."
    ! bimolecular
    i = sum(minloc(sct%VG_a(firstValidImage:sct%posVAG_a_Int-1))) ! Reactant
                                                           ! minumum
    i = i + firstValidImage-1
    j = sum(minloc(sct%VG_a(sct%posVAG_a_Int+1:lastValidImage)))  ! Product
                                                           ! minumum
    j = sct%posVAG_a_Int+j
    findMaxForE_0(1) = sct%VG_a(i)
    findMaxForE_0(2) = sct%VG_a(j)
    if (findMaxForE_0(1).lt.findMaxForE_0(2)) then
      sct%posE_0_Int = j
    else
      sct%posE_0_Int = i
    end if
    sct%E_0 = sct%VG_a(sct%posE_0_Int)
  else
    print*, "Calculating a unimolecular SCT rate."
    ! unimolecular
    i = sum(minloc(sct%VG_a(firstValidImage:sct%posVAG_a_Int-1))) ! Reactant
                                                             ! minumum
    j = sum(minloc(sct%VG_a(sct%posVAG_a_Int+1:lastValidImage)))  ! Product
                                                             ! minumum
    i = i + firstValidImage-1
    j = sct%posVAG_a_Int+j
    findMaxForE_0(1) = sct%VG_a(i) + vibenergies_step(i)
    findMaxForE_0(2) = sct%VG_a(j) + vibenergies_step(j)
    if (findMaxForE_0(1).lt.findMaxForE_0(2)) then
      sct%posE_0_Int = j
    else
      sct%posE_0_Int = i
    end if
    ! sct%posE_0_Int now gives the position of the Maximum of product-minumum 
    ! and reactant-minumum  
      ! Quantum Threshold energy with vibrational energy in the direction
    ! of the reaction
    sct%E_0 = sct%VG_a(sct%posE_0_Int) + vibenergies_step(sct%posE_0_Int) 
  end if
  ! Note that the minimum is just approximated and not calculated in a
  ! continuous matter.

! ======================================================================
! Shift energies to the Quantum threshold energy as "Zero of Energy" 
! + energyShift to keep values relatively small
! ======================================================================
  energyShift = 0d0! Was necessary for numerical stability in some tests
                   ! Normally 0 is alright.
                   ! abs(sct%VG_a(sct%posVAG_a_Int) - sct%VG_a(sct%posE_0_Int))*5D-1
  do i = firstValidImage, lastValidImage
    sct%VG_a(i) = sct%VG_a(i)-sct%E_0+energyShift
!     energies(i) = energies(i)-sct%E_0+energyShift !just for output
!     write(stdout,*) "Shifted_Adiabatic_Potential", &
!       sum(sct%steplengths(1:i-1))-sct%posTS_exact, sct%VG_a(i)*1D3
!     write (stdout,*) "Vib_Energies ", &
!              sum(sct%steplengths(1:i-1))-sct%posTS_exact, &
!              vibenergies(i)*1D3
  end do
  sct%E_0 = energyShift
!   write(stdout,*) " Shift", energyShift
!   do i = firstValidImage, lastValidImage
!     write (stdout,*) "Vib_Energies ", &
!              sum(sct%steplengths(1:i-1))-sct%posTS_exact, &
!              vibenergies(i)*1D3
!   end do
!   do i = firstValidImage, lastValidImage
!     write (stdout,*) "globEnergies ", &
!       sum(sct%steplengths(1:i-1))-sct%posTS_exact, energies(i)*1D3
!   end do
!   do i = firstValidImage, lastValidImage
!     write(stdout,*) "curvatures_k", &
!       sum(sct%steplengths(1:i-1))-sct%posTS_exact, curvatures_k(i)
!   end do
!   do i = firstValidImage, lastValidImage
!     write(stdout,*) "tbars ", &
!       sum(sct%steplengths(1:i-1))-sct%posTS_exact, tbars(i)
!   end do
!   do i = firstValidImage, lastValidImage
!     write(stdout,*) "abars ", &
!       sum(sct%steplengths(1:i-1))-sct%posTS_exact, abars(i)
!   end do
!   do i = firstValidImage, lastValidImage
!     write(stdout,*) "Effective_Masses: ", &
!       sum(sct%steplengths(1:i-1))-sct%posTS_exact, sct%Mu(i)
!   end do   
 

! ======================================================================
! Shift valid array elements to the beginning of the arrays
! ======================================================================
  ! Shift the indices of all arrays so that the valid images 
  ! are in the first 1:n entries in which n is the number of valid
  ! images

  sct%steplengths = cshift(sct%steplengths, firstValidImage-1)
  sct%Mu = cshift(sct%Mu, firstValidImage-1)
  sct%VG_a = cshift(sct%VG_a, firstValidImage-1)
  sct%posVAG_a_Int = sct%posVAG_a_Int - firstValidImage + 1
  sct%posE_0_Int = sct%posE_0_Int - firstValidImage + 1
  sct%nImgs = lastValidImage-firstValidImage+1
  nrSecondImages = nrSecondImages - firstValidImage + 1

  ! The valid points for sct%steplengths is now in (1:n)
! ======================================================================
! Smoothing reduced Masses and GS_AdiabPot
! ======================================================================

! ! !   sct%nImgs = 1
! ! !   allocate (sct%Test(1:1048576), STAT=k)
! ! !   k = sct%nImgs
! ! !   j = 2
! ! !   do while (j<=1048576)
! ! !    !guarantee that j has no two prime factors >= 7 (k und i verwendbar)
! ! ! acceptForFilter = .false.
! ! ! do while (acceptForFilter==.false.)
! ! !    ! testing j
! ! !    i = j
! ! !    do while(MOD(i,2)==0)
! ! !      i = i/2
! ! !    end do
! ! !    ! possible other prime factors smaller than 7
! ! !    do k = 3, 5, 2
! ! !      if(i.eq.1) exit
! ! !      do while(MOD(i,k)==0)
! ! !        i = i/k
! ! !      end do
! ! !    end do
! ! !    counter = 0
! ! !    do k = 7, i, 2
! ! !      if(i.eq.1) exit
! ! !      do while(MOD(i,k)==0)
! ! !        i = i/k
! ! !        counter = counter + 1
! ! !      end do
! ! !    end do
! ! !    if (counter<2) then
! ! !      acceptForFilter = .true.
! ! !    else
! ! !      ! Überspringe diesen Wert
! ! !      j = j + 1
! ! !    end if
! ! ! end do
! ! !    
! ! !    do i = 1, j
! ! !       sct%Test(i) = sin(pi*i/j)
! ! !    end do
! ! !    write(stdout,*) "Start filtertest with nImgs = ", j
! ! !    call dlf_sct_filter(j, sct%Test(1:j), & 
! ! !                         sct%ButterOrderMu, &
! ! !                         real(sct%nImgs,rk)/sct%cutoffFrqzMu)
! ! !    write(stdout,*) "Succesfull filtertest with nImgs = ", j
! ! !    j = j +1
! ! !   end do 
! ! !   call exit(0) 
  ! The problem is that the used filtering methods from fftpack5.1d
  ! only seem to be capable of handling array sizes that are
  ! powers of 2-6, higher prime factors are problematic (runtime errors)
!   write(stdout,*) "NRIM", sct%nImgs, sum(sct%steplengths(1:sct%nImgs))
!   do i = 1, sct%nImgs
!     write(stdout,*) "unFiltered Mass", &
!               sum(sct%steplengths(1:i-1))-sct%posTS_exact, sct%Mu(i)
!   end do
  if (sct%ButterOrderMu>0.or.sct%ButterOrderVG_a>0) then
    ! In order for the filter (external code) to work we need to
    ! make sure that the integer factorization (consisting of primes)
    ! of sct%nImgs only contains 1 prime larger or equal to 7
    ! (then it seems to work)
    acceptForFilter = .false.
    do while (.not.acceptForFilter)
       ! testing sct%nImgs
       i = sct%nImgs
       do while(MOD(i,2)==0)
         i = i/2
       end do
       ! possible other prime factors smaller than 7
       do k = 3, 5, 2
         if(i.eq.1) exit
         do while(MOD(i,k)==0)
           i = i/k
         end do
       end do
       counter = 0
       do k = 7, i, 2
         if(i.eq.1) exit
         do while(MOD(i,k)==0)
           i = i/k
           counter = counter + 1
         end do
       end do
       if (counter<2) then
         acceptForFilter = .true.
       else
         ! Delete an image on the correct side (where the integration
         ! is not influenced, i.e. on the reactant/product
         ! side, depending which has lower energy)
         if (sct%VG_a(1).gt.sct%VG_a(sct%nImgs)) then
           ! right side
           sct%nImgs = sct%nImgs - 1
           ! shift the borders and with it the constant values
           if (nIntBorderConst>0) then
             sct%Mu(sct%nImgs+1-nIntBorderConst) = 1.82289D3
           end if
         else
           ! left side
           sct%Mu = cshift(sct%Mu, 1)
           sct%nImgs = sct%nImgs - 1
           if (nIntBorderConst>0) then
             sct%Mu(nIntBorderConst) = 1.82289D3
           end if 
         end if
       end if
    end do
  end if
  call dlf_sct_filter(sct%nImgs, sct%Mu(1:sct%nImgs), & 
                      sct%ButterOrderMu, &
                      real(sct%nImgs,rk)/1D2*sct%cutoffFrqzMu)
!   do i = 1, sct%nImgs
!     write(stdout,*) "Filtered_Mass", &
!                     sum(sct%steplengths(1:i-1))-sct%posTS_exact, &
!                     sct%Mu(i)
!   end do

  call dlf_sct_filter(sct%nImgs, sct%VG_a(1:sct%nImgs), & 
                     sct%ButterOrderVG_a, &
                     real(sct%nImgs,rk)/1D2*sct%cutoffFrqzVG_a)
!   do i = 1, overallImages
!     write (stdout,*) "GS_AdiabPot",   i, sct%VG_a(i)
!   end do
! ======================================================================
! Initialize splines for Eff Mass and GS Adiabatic Pot
! ======================================================================
  if (.not.sct%splinesAlloc) then
    call allocate(sct%MuSpline, 4, sct%nImgs-1)
    call allocate(sct%VG_aSpline, 4, sct%nImgs-1)
    call allocate(sct%P_SAGSpline,4, sct%nrP_SAG_Values-1)
  end if
  call dlf_sct_init_csplines(sct%nImgs, sct%Mu, sct%steplengths, &
                              sct%MuSpline)
  call dlf_sct_init_csplines(sct%nImgs, sct%VG_a, sct%steplengths, &
                              sct%VG_aSpline)

  tmpvec(1) = 0d0
  tmpvec(2) = sum(sct%steplengths(1:sct%nImgs-1))/10000D0

! ======================================================================
! Optimize TS and VAG_a pos exact and determine barrierheight V^AG
! ======================================================================
  call dlf_sct_opt_VAG_a_E_0()
  tmp_element(1)=sct%posVAG_a_exact
  call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                              sct%steplengths, 1, &
                              tmp_element, tmp_element2)
  sct%VAG_a=tmp_element2(1)
!   do i = 1, sct%nImgs
!     write(stdout,*) "justFiltered VG_a", &
!                     sum(sct%steplengths(1:i-1))-sct%posTS_exact, &
!                     (sct%VG_a(i)-sct%E_0)*630D0+(17.6154D0-sct%VAG_a*630D0)
!   end do
    call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                              sct%steplengths, 1, &
                              tmpvec(1), tmpvec(3))
!   do i = 1, 10000
!     call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
!                               sct%steplengths, 1, &
!                               tmpvec(1), tmpvec(3))
!     write(stdout,*) "VG_a Spline", tmpvec(1)-sct%posTS_exact, &
!                (tmpvec(3)-sct%E_0)*630D0 +(17.6154D0-sct%VAG_a*630D0)
!     call dlf_sct_eval_csplines(sct%nImgs, sct%MuSpline,&
!                               sct%steplengths, 1, &
!                               tmpvec(1), tmpvec(3))
!     write(stdout,*) "EffMass Spline", tmpvec(1)-sct%posTS_exact, &
!                     tmpvec(3)
!     tmpvec(1) = tmpvec(1)+tmpvec(2)
!   end do
  write(stdout,*) "posTS_exact", sct%posTS_exact
! ======================================================================
! Initialize a spline for P_SAG
! ======================================================================

! Distribution of points for P_SAG interpolation according to
! d/dx tan(x), to be exact 
! (looks approx. like a boltzmann distribution)
! This is recommended (alternitavely there is a uniform distribution
! below)

  norm2= 2D0/((sct%VAG_a-sct%E_0))
  norm = atan(norm2*(sct%VAG_a-sct%E_0))
  tmp_dist = 2D0/((REAL(sct%nrP_SAG_Values,rk)-1D0))
  ! First determine the positions then the distances from that
  tmp = 0D0
  do i = 1, (sct%nrP_SAG_Values-1)
    sct%P_SAG_Dist(i)                    =&
       1D0/norm2*tan(norm*(-1D0+tmp))+sct%VAG_a
    tmp = tmp + tmp_dist
  end do
  sct%P_SAG_Dist(sct%nrP_SAG_Values)=2D0*sct%VAG_a-sct%E_0
  sct%P_SAG_Dist((sct%nrP_SAG_Values-1)/2+1)=sct%VAG_a
!   write(stdout,*) " E_0", sct%E_0
  write(stdout,*) "VAG_a-E_0", sct%VAG_a
!   do i = 1, (sct%nrP_SAG_Values)
!     write(stdout,*) "TestP_SAG Positions", sct%P_SAG_Dist(i)
!   end do
  do i = 1, (sct%nrP_SAG_Values)-1
    sct%P_SAG_Dist(i)= &
      sct%P_SAG_Dist(i+1)-&
      sct%P_SAG_Dist(i)
  end do
  sct%P_SAG_Dist(sct%nrP_SAG_Values) = 0D0 !Not used
!   do i = 1, (sct%nrP_SAG_Values)
!     write(stdout,*) "TestP_SAG Distances", sct%P_SAG_Dist(i)
!   end do
!   write(stdout,*) "Overall", SUM(sct%P_SAG_Dist(:))

!   ! Uniformly distributed values seems also reasonable
!   sct%P_SAG_Dist(:) = (sct%VAG_a-sct%E_0)/&
!                       DBLE((sct%nrP_SAG_Values-1)/2)
!   write(stdout,*) "Overall", SUM(sct%P_SAG_Dist(:))

  tmpenergy = sct%E_0
  ! nrInt_Values is guaranteed to be odd here
  do i = 1, (sct%nrP_SAG_Values-1)/2+1
    call P_SAG_fct(tmpenergy, sct%P_SAG_Val(i))
!     write(stdout,*) "TestP_SAG", tmpenergy, sct%P_SAG_Val(i)
    tmpenergy = tmpenergy + sct%P_SAG_Dist(i) !The energy at point i+1
  end do
  ! Use symmetry of P_SAG
  j = (sct%nrP_SAG_Values-1)/2+1
  do i = 1,(sct%nrP_SAG_Values-1)/2
    sct%P_SAG_Val(j+i) = 1D0 - sct%P_SAG_Val(j-i)
  end do
!   tmpenergy = sct%E_0
!   do i = 1, sct%nrP_SAG_Values
!     write(stdout,*) "TestP_SAG disc", &
!     (tmpenergy-sct%E_0)*630D0+(17.6154D0-sct%VAG_a*630D0), sct%P_SAG_Val(i)
!     tmpenergy = tmpenergy + sct%P_SAG_Dist(i)
!   end do
  call dlf_sct_init_csplines(sct%nrP_SAG_Values, sct%P_SAG_Val, sct%P_SAG_Dist,&
                             sct%P_SAGSpline)
  tmpvec(1) = 0d0
  tmpvec(2) = sum(sct%P_SAG_Dist(1:sct%nrP_SAG_Values-1))/10000D0
  do i = 1, 10000
    call dlf_sct_eval_csplines(sct%nrP_SAG_Values, sct%P_SAGSpline,&
                              sct%P_SAG_Dist, 1, &
                              tmpvec(1), tmpvec(3))
    tmpvec(1) = tmpvec(1)+tmpvec(2)
!     write(stdout,*) "TestP_SAGSpline", i, tmpvec(3)
  end do
! ======================================================================
! Perform Integrations at different temperatures
! ======================================================================
  temperature=tstart
  do i =1, tsteps
    call dlf_sct_kappa_of_T(temperature, kappas(i))
    temperature=temperature/(1.D0+temperature*tdel)
  end do
  
!   temperature = 50D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
!   temperature = 55D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
!   temperature = 60D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
!   temperature = 65D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
!   temperature = 70D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
!   temperature = 75D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
!   temperature = 80D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
!   temperature = 90D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
!   temperature = 100D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
!   temperature = 110D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
!   temperature = 120D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
!   temperature = 130D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 140D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 150D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 160D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 180D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 200D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 220D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 240D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 250D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 298D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 300D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 400D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 600D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 800D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 1000D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 1250D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 1500D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa
! 
! temperature = 2000D0
!   call dlf_sct_kappa_of_T(temperature, kappa)
!   write(stdout,*) "T[K] kappa(adiab.theory/µVT)", temperature, kappa

! ======================================================================
! Deallocate all arrays that have been used
! ======================================================================
  call deallocate(energies) 
  call deallocate(vibenergies) 
  call deallocate(sct%VG_a) 
  call deallocate(stepderivatives) 
  call deallocate(vibenergies_step)
  call deallocate(hess) 
  call deallocate(phess) 
  call deallocate(steps) 
  call deallocate(icoords) 
  call deallocate(sct%steplengths) 
  call deallocate(peigvals) 
  call deallocate(peigvecs) 
  call deallocate(trafos) 
  call deallocate(tmpvec) 
  call deallocate(B_F) 
  call deallocate(curvatures_k) 
  call deallocate(harmonicfrqzs) 
  call deallocate(tbars) 
  call deallocate(tbarderivatives) 
  call deallocate(abars) 
  call deallocate(sct%Mu) 
  call deallocate(sct%MuSpline)
  call deallocate(sct%VG_aSpline)
end subroutine dlf_sct_main
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_filter
!!
!! FUNCTION
!!
!! Applies a Butterworth filter to uniformly distributed data
!!
!! INPUTS
!! n - number of datapoints
!! r(n) - the data that shall be smoothed
!! butter_order - order of the Butterworth filter
!! cutoff_Frqz - the cutoff frequency of the filter
!!
!! OUTPUTS
!! r(n) - filtered data
!!
!! COMMENTS
!! uses the external fftpack5-1d.f90
!! 
!! SYNOPSIS
subroutine dlf_sct_filter(n, r, butter_order, cutoff_Frqz)
  use dlf_global, only: stderr,stdout
  use dlf_parameter_module, only: rk
  use dlf_allocate, only: allocate,deallocate
  implicit none
  integer, intent(in)      :: n 
  real(rk), intent(inout)  :: r(n)
  integer, intent(in)      :: butter_order
  real(rk), intent(in)     :: cutoff_Frqz
  integer                  :: lensav, ier, inc, i           
  real(rk), allocatable    :: wsave(:)             
  real(rk), allocatable    :: work(:)                    
  lensav = n + int(log(real(n, rk))/log(real(2.0D+00, rk)))+4! Minimum length of wsave
                                                    ! array
  inc = 1
  call allocate (wsave, INT(lensav))
  call allocate(work, INT(n))
  ! Now apply RFFT1F 
  ! (onedimensional real double precision forward Fast Fourier Trafo)
  ! Then lowpass filter, then RFFT1B (backward transformation)
  ! Using Butterworth filter of order butter_order with cutoff Frequency
  ! cutoff_Frqz
  if (butter_order.ne.0) then
    ! Initialization for fft
    call rfft1i (n, wsave, lensav, ier)
    ! forward real fft
    call rfft1f (n, inc, r, n, wsave, lensav, work, n, ier )
    ! The first element represents the "zero" frequency and only has a
    ! real part that is now in r(0) 
    ! For an even n the "-pi" frequency is saved in and r(n)
    ! The other frequencies have real and imaginary parts that are now
    ! in r(i) and r(i+1) for even i
    ! The frequency of an element in r(i) is i/(n*2)
    ! The cutoffFrqzMu must also be given in units of (n*2)
    if (mod(INT(n), 2) .eq.0d0) then
      ! n even
      ! First element is "ZeroFrequency" and shall have gain = 1
      ! (nothing to do)
      do i = 2, n-2, 2        
        r(i)   = r(i)/dsqrt(1.0D0+(REAL(i, rk)&
                     /cutoff_Frqz)**(2*butter_order))!real Part
        r(i+1) = r(i+1)/dsqrt(1.0D0+(REAL(i, rk)&
                       /cutoff_Frqz)**(2*butter_order))!imaginary Part
      end do 
      r(n)  = r(n)/dsqrt(1.0D0+(REAL(n, rk) &
                  /cutoff_Frqz)**(2*butter_order))
    else
      ! n odd
      ! First element is "ZeroFrequency" and shall have gain = 1
      ! (nothing to do)
      do i = 2, n-1, 2
        r(i)   = r(i)/dsqrt(1.0D0+(REAL(i, rk)&
                     /cutoff_Frqz)**(2*butter_order))!real Part
        r(i+1) = r(i+1)/dsqrt(1.0D0+(REAL(i, rk)&
                       /cutoff_Frqz)**(2*butter_order))!imaginary Part
      end do
    end if
    ! backward real fft
    call rfft1b (n, inc, r, n, wsave, lensav, work, n, ier )
  end if
  call deallocate (wsave)
  call deallocate(work)
end subroutine dlf_sct_filter
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_kappa_of_T
!!
!! FUNCTION
!!
!! Calculates kappa for a given temperature
!!
!! INPUTS
!! T_in_k - temperature in Kelvin
!! sct%steplengths
!! sct%VG_a
!! sct%nImgs
!! sct%VG_aSpline
!! sct%posVAG_a_Int
!! sct%E_0
!! sct%nrInt_Values
!! sct%beta
!! sct%kron_eps_a
!! sct%kron_order
!! sct%nrP_SAG_Values
!! sct%P_SAGSpline
!!
!! OUTPUTS
!! kappa - the transmission coefficient
!!
!! SYNOPSIS
subroutine dlf_sct_kappa_of_T (T_in_K, kappa)
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  use sct_module
  use dlf_allocate
  use dlf_constants
  implicit none
  real(rk), intent(in)        :: T_in_K
  real(rk), intent(out)       :: kappa
  integer                     :: i
  real(rk)                    :: KBOLTZ_AU
  real(rk)                    :: tmpenergy
  real(rk)                    :: IntValEnergy ! Result of the quadrature
                                              ! over energy
  real(rk)                    :: k_Arrh
  real(rk)                    :: tmp
  real(rk)                    :: P_SAG(1), P_SAG2(1)
  real(rk)                    :: time
#ifdef useSimpson
  real(rk)                    :: energyInt_Dist
  real(rk)                    :: energyInt_Val(sct%nrInt_Values)
#endif
#ifdef useGaussKronrod
  integer                     :: n
  real(rk), allocatable       :: x(:)
  real(rk), allocatable       :: w1(:)
  real(rk), allocatable       :: w2(:)
  real(rk)                    :: i1, i2
  real(rk)                    :: eps
  real(rk)                    :: GaussKron_midpoint
#endif

! ======================================================================
! Thermal properties
! ======================================================================
  call dlf_constants_get("KBOLTZ_AU ",KBOLTZ_AU)
!   write(stdout,*) "KBOLTZ_AU", KBOLTZ_AU
  sct%beta = 1.0D0/(T_in_K*KBOLTZ_AU)  ! beta = 1 / (kB*T)

! ======================================================================
! Energy integration between sct%E_0 and 2*sct%VAG_a-sct%E_0
! ======================================================================
  ! Just called for timing purposes
 ! call P_SAG_fct(0d0, tmpenergy)

! ======================================================================
! Simpson
! ======================================================================
#ifdef useSimpson
  call get_wall_time(time)
  energyInt_Dist = (2.0D0 * (sct%VAG_a-sct%E_0))/&
                    REAL(sct%nrInt_Values-1,rk)
  tmpenergy = sct%E_0
  do i = 1, (sct%nrInt_Values-1)/2
    call dlf_sct_eval_csplines(sct%nrP_SAG_Values, sct%P_SAGSpline,&
                              sct%P_SAG_Dist, 1, &
                              tmpenergy, energyInt_Val(i))
    energyInt_Val(sct%nrInt_Values+1-i) = 1D0-energyInt_Val(i)
    energyInt_Val(i) = energyInt_Val(i)*dexp(-sct%beta*tmpenergy)
    energyInt_Val(sct%nrInt_Values+1-i) = &
                             energyInt_Val(sct%nrInt_Values+1-i)*&
                             dexp(-sct%beta*&
                             (2.0D0 * (sct%VAG_a-sct%E_0)-&
                              tmpenergy))
    !write(stdout,*) "TestP_SAG", tmpenergy, energyInt_Val(i)
    tmpenergy = tmpenergy + energyInt_Dist !The energy at point i+1
  end do
  call dlf_sct_eval_csplines(sct%nrP_SAG_Values, sct%P_SAGSpline,&
                             sct%P_SAG_Dist, 1, &
                             sct%VAG_a, &
                             energyInt_Val((sct%nrInt_Values-1)/2+1))
  energyInt_Val((sct%nrInt_Values-1)/2+1) = &
    energyInt_Val((sct%nrInt_Values-1)/2+1)*dexp(-sct%beta*sct%VAG_a)



  call dlf_sct_quad_equ(sct%nrInt_Values, energyInt_Val, &
                         energyInt_Dist, IntValEnergy)
  if(printl>=6) then
    write (stdout, *) " Temperature    ", T_in_K
  end if
#ifdef DebugInformation  
    write (stdout, *) " Beta           ", sct%beta
    write (stdout, *) " VAG_a          ", sct%VAG_a 
    write (stdout, *) "  first part int", IntValEnergy 
    write(stdout,*)   " second part int", &
                      1.0D0/sct%beta*&
                      dexp(-sct%beta*(2.0D0*sct%VAG_a-sct%E_0))
#endif
  IntValEnergy = IntValEnergy + &
                 1.0D0/sct%beta*&
                 dexp(-sct%beta*(2.0D0*sct%VAG_a-sct%E_0))
#ifdef DebugInformation  
    write (stdout, *) " GesamtIntegral ", IntValEnergy 
#endif
  kappa = IntValEnergy/dexp(-sct%beta*sct%VAG_a)
  kappa = sct%beta*kappa
  k_Arrh = dexp(-sct%beta*sct%VAG_a)
  if(printl>=6) then
    write (stdout, *) " kappa          ", kappa
    write (stdout, *) "                "
  end if
  call get_wall_time(tmp)
  time = tmp - time
  if(printl>=6) then
    write(stdout,*) " Time_Simpson ", time !Allocation is not included
  end if
!   write(stdout,*) " Error_Simpson ", kappa - 1.09736299341881D0 !Allocation is not included
#endif

! ======================================================================
! Gauss-Kronrod
! ======================================================================
#ifdef useGaussKronrod
  call get_wall_time(time)
  ! eps only determines how precise the abszisses are calculated
  ! and it sounds plausible to chose it of the same order of magnitude
  ! like the precision of the quadrature itself
  eps=sct%kron_eps_a
  n = sct%minKronrod
  ! (2*VAG_a-E_0 + E_0)/2
  GaussKron_midpoint = (sct%VAG_a)
  do
    call allocate(x,n+1)
    call allocate(w1,n+1) 
    call allocate(w2,n+1)  
    call kronrod ( n, eps, x, w1, w2 )
    ! kronrod_adjust adjusts a Gauss-Kronrod rule from [-1,+1] to [A,B]
    call kronrod_adjust(sct%E_0, 2.0D0*sct%VAG_a-sct%E_0, &
                        n, x, w1, w2)
    call dlf_sct_eval_csplines(sct%nrP_SAG_Values, sct%P_SAGSpline,&
                              sct%P_SAG_Dist, 1, &
                              x(n+1), P_SAG(1))
    P_SAG(1) = P_SAG(1)*dexp(-sct%beta*x(n+1))
    i1 = w1(n+1)*P_SAG(1)
    i2 = w2(n+1)*P_SAG(1)
    call dlf_sct_eval_csplines(sct%nrP_SAG_Values, sct%P_SAGSpline,&
                              sct%P_SAG_Dist, 1, &
                              x(1), P_SAG2(1))
    P_SAG(1) = 1d0 - P_SAG2(1)
    P_SAG(1) = P_SAG(1)*dexp(-sct%beta*(2D0*GaussKron_midpoint-x(1)))
    P_SAG2(1) = P_SAG2(1)*dexp(-sct%beta*x(1))
    i1 = i1 + w1(1) * (P_SAG(1) + P_SAG2(1))
    do i = 2, n, 2
      call dlf_sct_eval_csplines(sct%nrP_SAG_Values, sct%P_SAGSpline,&
                              sct%P_SAG_Dist, 1, &
                              x(i), P_SAG2(1))
      P_SAG(1) = 1d0 - P_SAG2(1)
      P_SAG(1) = P_SAG(1)*dexp(-sct%beta*(2D0*GaussKron_midpoint-x(i)))
      P_SAG2(1) = P_SAG2(1)*dexp(-sct%beta*x(i))
      tmp = P_SAG(1) + P_SAG2(1)
      i1 = i1 + w1(i) * tmp
      i2 = i2 + w2(i) * tmp
      call dlf_sct_eval_csplines(sct%nrP_SAG_Values, sct%P_SAGSpline,&
                              sct%P_SAG_Dist, 1, &
                              x(i+1), P_SAG2(1))
      P_SAG(1) = 1d0 - P_SAG2(1)
      P_SAG(1) = P_SAG(1)*dexp(-sct%beta*(2D0*GaussKron_midpoint-x(i+1)))
      P_SAG2(1) = P_SAG2(1)*dexp(-sct%beta*x(i+1))
      i1 = i1 + w1(i+1) * (P_SAG(1) + P_SAG2(1))
    end do
    
    if ( abs(i1 - i2 ) < sct%kron_eps_a) then 
      if(printl>=6) then
        write(stdout,*) " Kronrod: error in tolerance with n=", n
        write(stdout,*) " Kronrod: Coarse integral estimate  ", i1
        write(stdout,*) " Kronrod: Fine   integral estimate  ", i2
        write(stdout,*) " Kronrod: Error estimate            ", &
                        abs (i2 - i1)
      end if
      exit
    end if

    if ( sct%kron_order <= n ) then
      if(printl>=6) then
        write(stdout,*) &
        " Kronrod: error tolerance still not satisfied with n = ", n
        write(stdout,*) " Kronrod: Cancel kronrod integration,",& 
                        " use the finest estimate so far!"
        write(stdout,*) " Kronrod: Coarse integral estimate   ", i1
        write(stdout,*) " Kronrod: Fine   integral estimate   ", i2
        write(stdout,*) " Kronrod: Error estimate             ", &
                        abs ( i2 - i1 )
      end if
      exit
    end if

    deallocate ( x )
    deallocate ( w1 )
    deallocate ( w2 )

    n = 2 * n + 1
  end do

  IntValEnergy = i1
  if(printl>=6) then
    write (stdout, *) " Temperature         ", T_in_K
  end if
#ifdef DebugInformation  
    write (stdout, *) " Beta                ", sct%beta
    write (stdout, *) " VAG_a               ", sct%VAG_a 
    write (stdout, *) " Integral(quadrature)", IntValEnergy 
    write(stdout,*)   " Rest of the integral", &
                      1.0D0/sct%beta * &
                      dexp(-sct%beta * (2.0D0*sct%VAG_a-sct%E_0))
#endif
  IntValEnergy = IntValEnergy + &
                 1.0D0/sct%beta * &
                 dexp(-sct%beta * (2.0D0*sct%VAG_a-sct%E_0))
#ifdef DebugInformation                
  write (stdout, *) " Overall Integral    ", IntValEnergy 
#endif  
  kappa = IntValEnergy/dexp(-sct%beta*sct%VAG_a)
  kappa = sct%beta*kappa
  if(printl>=6) then
    write (stdout, *) " kappa               ", kappa
    write (stdout, *) "                     "
  end if
  call get_wall_time(tmp)
  time = tmp - time
  if(printl>=6) then
    write(stdout,*)   " Time [ms]   Kronrod ", time*1d3
  end if
#endif
end subroutine dlf_sct_kappa_of_T
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_quad_equ
!!
!! FUNCTION
!!
!! Simpson sum
!!
!! COMMENTS
!! linearly interpolates last step if the the number of evaluations
!! is even.
!!
!! INPUTS
!!
!! nvalues - number of data points
!! values(nvalues) - data points
!! distance - the distance between the data points 
!!
!! OUTPUTS
!! 
!! quadresult - result of the quadrature
!!
!! SYNOPSIS
subroutine dlf_sct_quad_equ(nvalues,values,distance, quadresult)
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  use dlf_allocate
  use sct_module
  use dlf_constants
  implicit none
  integer, intent(in)   :: nvalues
  real(rk), intent(in)  :: values(nvalues)
  real(rk), intent(in)  :: distance !distance between points
  real(rk), intent(out) :: quadresult
  integer               :: i
  quadresult = 0d0
  ! Simpson-Rule only applicable for mod(nvalues,2)-1 = 0 & nvalues >= 3
  ! Do linear interpolation at the last point, if that is not satisfied
  if (nvalues >= 3 .and. mod(nvalues,2)==1) then
    ! Apply Simpson rule
    quadresult = quadresult + values(1) + values(nvalues)
    do i = 2, nvalues-3, 2
      quadresult = quadresult + 4.0D0*values(i) + 2.0D0*values(i+1)    
    end do
    quadresult = quadresult + 4.0D0*values(nvalues-1)
    quadresult = quadresult * distance /3.0D0
  else if (nvalues >= 3) then
    ! Apply Simpson rule up to nvalues -1 and use linear interpolation
    ! for last step
    quadresult = quadresult + values(1) + values(nvalues-1)
    do i = 2, nvalues-4, 2
      quadresult = quadresult + 4.0D0*values(i) + 2.0D0*values(i+1)    
    end do
    quadresult = quadresult + 4.0D0*values(nvalues-2)
    quadresult = quadresult * distance /3.0D0
    ! last value:
    quadresult = quadresult + distance / 2.0D0 * &
                              (values(nvalues)-values(nvalues-1))
  end if
end subroutine dlf_sct_quad_equ
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_opt_VAG_a_E_0
!!
!! FUNCTION
!!
!! Uses the pre-calculated sct%posVAG_a_Int to optimize the exact turningpoint
!!
!! COMMENTS
!! Only call after sct%posVAG_a_Int is calculated correctly.
!!
!! INPUTS
!! sct%posVAG_a_Int - integer value determining the TS position roughly
!! sct%steplengths
!! sct%posVAG_a_Int
!! sct%nImgs
!! sct%VG_aSpline
!!
!! OUTPUTS
!! 
!! sct%posVAG_a_exact - the exact turningpoint
!!
!! SYNOPSIS
subroutine dlf_sct_opt_VAG_a_E_0()
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  use sct_module
  use dlf_constants
  implicit none
  real(rk)              :: tmp(1)
  real(rk)              :: limiter
  real(rk)              :: epsErr     ! The precision up to which the
                                      ! maximum point is calculated
  real(rk)              :: epsErrChange ! The precision up to which a
                                        ! change of the error 
                                        ! in the iteration to find the
                                        ! turningpoints stops
                                        ! should be smaller than epsErr
  real(rk)              :: l(1)
  real(rk)              :: r(1)
  real(rk)              :: l_limit
  real(rk)              :: r_limit
  real(rk)				:: tmp_element(1)

  tmp(1) = 0d0  
  epsErr = 1D-6
  epsErrChange = 1D-13
  ! Optimize VAG_a position
  sct%posVAG_a_exact = sum(sct%steplengths(1:sct%posVAG_a_Int-1)) 
  limiter = max(sct%steplengths(sct%posVAG_a_Int-1), &
                sct%steplengths(sct%posVAG_a_Int))/4D0
  tmp_element(1)=sct%posVAG_a_exact
  call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                              sct%steplengths, 1, &
                              tmp_element, tmp(1))

  do while(limiter > 1D-13)
    tmp_element(1) =sct%posVAG_a_exact+limiter
    call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                                sct%steplengths, 1, &
                                tmp_element, r(1))
	tmp_element(1)=sct%posVAG_a_exact-limiter
    call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                                sct%steplengths, 1, &
                                tmp_element, l(1))

    if (r(1)>tmp(1)) then
      sct%posVAG_a_exact = sct%posVAG_a_exact+limiter
      tmp(1) = r(1)
    else if (l(1)>tmp(1)) then
      sct%posVAG_a_exact = sct%posVAG_a_exact-limiter
      tmp(1) = l(1)
    else
      limiter = limiter/2D0
    end if
  end do
  if(printl>=4) then
    write(stdout,*) "VAG_a position", sct%posVAG_a_exact
  end if

  !Optimize E_0
  r_limit = sum(sct%steplengths(1:sct%nImgs-1))
  l_limit = 0d0
  tmp(1) = 0d0  
  epsErr = 1D-6
  epsErrChange = 1D-13
  ! Optimize E_0 position
  sct%posE_0_exact = sum(sct%steplengths(1:sct%posE_0_Int-1)) 
  if (sct%posE_0_Int-1.ge.1) then
      limiter = max(sct%steplengths(sct%posE_0_Int-1), &
                sct%steplengths(sct%posE_0_Int))/2D0
  else
    limiter = sct%steplengths(sct%posE_0_Int)/2D0
  end if
  tmp_element(1)=sct%posE_0_exact
  call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                              sct%steplengths, 1, &
                              tmp_element, tmp(1))

  do while(limiter > 1D-13)
    if (sct%posE_0_exact+limiter.le.r_limit) then
      tmp_element(1)=sct%posE_0_exact+limiter
      call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                                 sct%steplengths, 1, &
                                 tmp_element, r(1))
    end if
    if (sct%posE_0_exact-limiter.ge.l_limit) then
      tmp_element(1)=sct%posE_0_exact-limiter
      call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                                 sct%steplengths, 1, &
                                 tmp_element, l(1))
    end if     

    if (r(1)<tmp(1)) then
      sct%posE_0_exact = sct%posE_0_exact+limiter
      tmp(1) = r(1)
    else if (l(1)<tmp(1)) then
      sct%posE_0_exact = sct%posE_0_exact-limiter
      tmp(1) = l(1)
    else
      limiter = limiter/2D0
    end if
  end do
  if(printl>=4) then
    write(stdout,*) "E_0 position", sct%posE_0_exact
  end if
end subroutine dlf_sct_opt_VAG_a_E_0
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/P_SAG
!!
!! FUNCTION
!!
!! Calculates the semiclassical transmission probability for a given 
!! energy
!!
!! COMMENTS
!! The transmission probability is weighted with exp(-beta * energy)
!!
!! INPUTS
!! energy
!! sct%steplengths
!! sct%E_0
!! sct%VAG_a
!! sct%beta
!! sct%Mu(i)
!! sct%VG_a(i)
!!
!! OUTPUTS
!! 
!! P_SAG - semiclassical transmission probability
!!
!! SYNOPSIS
subroutine P_SAG_fct(energy, P_SAG)
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  use sct_module
  implicit none
  real(rk), intent(in)  :: energy
  real(rk), intent(out) :: P_SAG
  real(rk)              :: theta  
!   write(stdout,*) "Lengths", sum(sct%steplengths(1:sct%nImgs-1))
  if (energy.lt.sct%E_0) then
    P_SAG = 0.0D0
  else if (energy.le.sct%VAG_a) then
    call dlf_sct_theta(energy, theta)
!     write(stdout, *) "ThetaDB", energy, theta
    P_SAG = 1.0D0/(1.0D0+dexp(2.0D0*theta))
  else if (energy.le.(2.0D0*sct%VAG_a-sct%E_0)) then
    call dlf_sct_theta(2d0*sct%VAG_a-energy, theta) 
!     write(stdout,*) "Marker"
!     write(stdout, *) "ThetaDB", energy, theta
    P_SAG = (1.0D0 - 1.0D0/(1.0D0+dexp(2.0D0*theta)))
  else
    P_SAG = 1.0D0
  end if    
!   write(stdout,*) "Energy,P_SAG", (energy-sct%E_0)*630D0+11.6132D0, P_SAG/dexp(-sct%beta*energy)
end
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_theta
!!
!! FUNCTION
!!
!! Calculates the imaginary action/barrier penetration integral
!!
!! COMMENTS
!! Used by P_SAG
!!
!! INPUTS
!! energy
!! sct%VG_a
!! sct%posVAG_a_Int
!! sct%nImgs
!! sct%MuSpline
!! sct%VG_aSpline
!! sct%posVAG_a_exact
!!
!! OUTPUTS
!! 
!! theta - imaginary action integral
!!
!! SYNOPSIS
subroutine dlf_sct_theta(energy, theta)
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  use sct_module
  use dlf_constants
  implicit none
  real(rk), intent(in)  :: energy
  real(rk), intent(out) :: theta
  real(rk)              :: differences(sct%nImgs)
  real(rk)              :: integrand(sct%nImgs)
  real(rk)              :: tmp(1)
  real(rk)              :: tmpStepSize
  real(rk)              :: lerror
  real(rk)              :: rerror
  real(rk)              :: lerror_old
  real(rk)              :: rerror_old
  real(rk)              :: change
  real(rk)              :: limiter
  real(rk)              :: epsErr
  real(rk)              :: realIter(1) ! The precision up to 
                                       ! which the turningpoints are
                                       ! calculated
  real(rk)              :: epsErrChange ! The precision up to which a
                                        ! change of the error 
                                        ! in the iteration to find the
                                        ! turningpoints stops
                                        ! should be smaller than epsErr
  real(rk)              :: lTurnPoint(1)
  real(rk)              :: rTurnPoint(1)
  real(rk)              :: lMin
  real(rk)              :: rMax
  real(rk)              :: tmpMu(1)
  real(rk)              :: tmpVG_a(1)
  integer               :: lTurnPointInt
  integer               :: rTurnPointInt
  integer               :: lpos
  integer               :: rpos
  integer               :: i
  integer               :: nrSteps
! ======================================================================
! Coarse search for classical turningpoints (on discretization)
! ======================================================================
! Searching approximated Turningpoints 
! (limited to the stepdiscretization of the IRC path) 

  lpos = sum(minloc((sct%VG_a(1:sct%posVAG_a_Int))))
  rpos = sum(minloc((sct%VG_a(sct%posVAG_a_Int:sct%nImgs))))+sct%posVAG_a_Int-1
  do i = lpos, rpos
    differences(i) = abs(sct%VG_a(i) - energy)
  end do

  lTurnPointInt = sum(minloc(differences(lpos:sct%posVAG_a_Int)))&
                  +lpos-1
  rTurnPointInt = sum(minloc(differences(sct%posVAG_a_Int:rpos)))&
                  +sct%posVAG_a_Int-1

  ! Use these to estimate the real valued, preciseturningpoints
  if (lTurnPointInt-1.ge.1) then 
    lTurnPoint(1) = sum(sct%steplengths(1:lTurnPointInt-1))
  else
    lTurnPoint(1) = 0d0
  end if
  rTurnPoint(1) = sum(sct%steplengths(1:rTurnPointInt-1))
  ! Since all values are discrete in the coarse optimization 
  ! in the two lines above
  ! it is possible that one of the determined Turnpoints 
  ! is on the wrong side 
  ! of the true (exact) transition state. This must be excluded.
  if (lTurnPointInt.eq.sct%posVAG_a_Int) then
    ! is lTurnPoint(1) on the wrong side of the Potential?
    if (lTurnPoint(1)>sct%posVAG_a_exact) then
      lTurnPointInt = lTurnPointInt - 1 
      lTurnPoint(1) = lTurnPoint(1) - sct%steplengths(lTurnPointInt)
    end if
  end if
  if (rTurnPointInt.eq.sct%posVAG_a_Int) then
    ! is rTurnPoint(1) on the wrong side of the Potential?
    if (rTurnPoint(1)<sct%posVAG_a_exact) then
      rTurnPoint(1) = rTurnPoint(1) + sct%steplengths(rTurnPointInt)
      rTurnPointInt = rTurnPointInt + 1
    end if
  end if

!    write (stdout,*) "Turningpoints are at position (left, right, energy)", &
!                     lTurnPointInt, rTurnPoint(1), energy
  ! Now evaluate the integrands for integration afterward
  tmp(1) = 0d0
  tmpStepSize = sum(sct%steplengths(1:sct%nImgs-1))/1.0D4 
  do i = 1,10000
   if(sct%stopper) then
     call dlf_sct_eval_csplines(sct%nImgs, sct%MuSpline, &
                                 sct%steplengths, 1, tmp(1), tmpMu(1))
!      write(stdout,*) "Theta filteredMass  ", tmp(1), tmpMu(1)
     call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                                 sct%steplengths, 1, tmp(1), tmpVG_a(1))
!      write(stdout,*) "Theta filteredVG_a  ", tmp(1), tmpVG_a(1)
     tmp(1) = tmp(1) + tmpStepSize
   end if
  end do
#ifdef DebugInformation  
  if (abs(energy-sct%VAG_a)<1D-4) &
       write(stdout,*) "Near to TS theta", &
       sct%posVAG_a_Int, sum(sct%steplengths(1:sct%posVAG_a_Int-1)),&
       sct%posVAG_a_exact, lTurnPointInt, rTurnPointInt, &
       sct%VAG_a-energy, differences(sct%posVAG_a_Int-1),&
       differences(sct%posVAG_a_Int), differences(sct%posVAG_a_Int+1), &
       differences(sct%posVAG_a_Int+2),&
       sum(sct%steplengths(1:sct%posVAG_a_Int-1)), &
       sum(sct%steplengths(1:sct%posVAG_a_Int)), &
       sum(sct%steplengths(1:sct%posVAG_a_Int+1))
#endif
! ======================================================================
! Fine search for classical turningpoints
! ======================================================================
  ! Find the exact turningpoints with Newton's method
  ! The exact Turning-Points must be in the area around lTurnPointInt
  ! and rTurnPointInt
  ! Apply Newton's method until close enough
  ! It is advisible to use an additional termination criterion:
  !    If the improvement through the iterations becomes very small 
  !    -> stop iteration and take the absolut values for
  !       the square root (could still be negative)
  tmp(1) = 0d0 
  epsErr = 1D-6
  epsErrChange = 1D-13
! ======================================================================
! Optimize left turnpoint
! ======================================================================
  ! Do not jump out of the determined interval
  if (lTurnPointInt-2.ge.1) then 
    lMin = sum(sct%steplengths(1:lTurnPointInt-2)) 
  else
    lMin = 0d0
  end if
  rMax = min(sum(sct%steplengths(1:lTurnPointInt)), sct%posVAG_a_exact)
  call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                              sct%steplengths, 1, lTurnPoint(1), tmpVG_a(1))
  lerror = tmpVG_a(1)-energy
  if (lTurnPointInt-1.ge.1) then 
    limiter = min(sct%steplengths(lTurnPointInt-1),&
                  sct%steplengths(lTurnPointInt))/2D0
  else
    limiter = sct%steplengths(lTurnPointInt)/2D0
  end if
  change = -limiter

  do while((abs(lerror)>epsErr))
    if ((change)>0d0) then
      ! Algorithm seems to be worsening -> stronger limitation needed,
      ! do not abort
      limiter = limiter/2D0
    else if (abs(change)<epsErrChange) then
      ! Change is very small 
      ! -> The function may not have a root but only a minimum
      ! If the beginning limiter (initialization of change) is smaller
      ! than 1D-13 no optimization is required
      exit
    end if      
    ! d/ds (V(s)-E) = d/ds V(s)
    call dlf_sct_eval_csplines_d(sct%nImgs, sct%VG_aSpline, &
                                  sct%steplengths, 1, lTurnPoint(1),&
                                  tmpVG_a(1))
    tmp(1) = lerror/tmpVG_a(1)
    if (abs(tmp(1)).gt.limiter) tmp(1) = sign(limiter, tmp(1)) ! limit the jump
                                                      ! to the next 
                                                      ! by the value 
                                                      ! of limiter
    lTurnPoint(1) = lTurnPoint(1)-tmp(1)
    if (lTurnPoint(1) < lMin) lTurnPoint(1) = lMin
    if (lTurnPoint(1) > rMax) lTurnPoint(1) = rMax
    call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                                sct%steplengths, 1, lTurnPoint(1),&
                                tmpVG_a(1))
    lerror_old = lerror
    lerror = tmpVG_a(1)-energy
    change = abs(lerror)-abs(lerror_old)
  end do

! ======================================================================
! Optimize right turnpoint
! ======================================================================
  ! Do not jump out of the determined interval or left of lTurnPoint(1)
  lMin = max(sum(sct%steplengths(1:rTurnPointInt-2)), sct%posVAG_a_exact)
  rMax = sum(sct%steplengths(1:rTurnPointInt))
  call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                              sct%steplengths, 1, rTurnPoint(1), tmpVG_a(1))
  rerror = tmpVG_a(1)-energy
  limiter = min(sct%steplengths(rTurnPointInt-1),&
            sct%steplengths(rTurnPointInt))/2D0
  change = -limiter

  do while((abs(rerror)>epsErr))
    if ((change)>0d0) then
      ! Algorithm seems to be worsening
      ! -> stronger limitation needed, do not abort
      limiter = limiter/2D0
    else if (abs(change)<epsErrChange) then
      ! Change is very small 
      ! -> Abort: The function may not have a root but only a minimum
      ! If the beginning limiter (initialization of change) is smaller
      ! than 1D-13 no optimization is required
      exit
    end if
    ! d/ds (V(s)-E) = d/ds V(s)
    call dlf_sct_eval_csplines_d(sct%nImgs, sct%VG_aSpline, &
                                  sct%steplengths, 1, &
                                  rTurnPoint(1), tmpVG_a(1))
    tmp(1) = rerror/tmpVG_a(1)
    if (abs(tmp(1)).gt.limiter) tmp(1) = sign(limiter, tmp(1)) ! limit the jump
                                                      ! to the next
                                                      ! value by limiter
    rTurnPoint(1) = rTurnPoint(1)-tmp(1)
    if (rTurnPoint(1) < lMin) rTurnPoint(1) = lMin
    if (rTurnPoint(1) > rMax) rTurnPoint(1) = rMax

    call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                                sct%steplengths, 1, &
                                rTurnPoint(1), tmpVG_a(1))
    rerror_old = rerror
    rerror = tmpVG_a(1)-energy
    change = abs(rerror)-abs(rerror_old)
  end do
! ======================================================================
! Catch errors
! ======================================================================
  if (lTurnPoint(1)>rTurnPoint(1)) then
    write(stderr,*) "Turningpoints are changed! ",&
                    "(lTurnPoint(1) > rTurnPoint(1))"
  end if


!   write(stdout,*) "Tpoints at energy  ", energy
!   write(stdout,*) "Tpoints Position ca", lTurnPointInt, rTurnPointInt
!   write(stdout,*) "Tpoints are approx ", &
!                   sum(sct%steplengths(1:lTurnPointInt-1)), &
!                   sum(sct%steplengths(1:rTurnPointInt-1))
  if(printl>=4) then
    write(stdout,3794) "Turningpoints (left, right, energy)", lTurnPoint(1), &
                                         rTurnPoint(1), energy
  end if
  3794  FORMAT (A, E12.4,E12.4,E12.4)                                        
!   !write(stdout,*) "Tptse", sct%posVAG_a_exact, sct%posVAG_a_exact, sct%VAG_a
!   write(stdout,*) "Tpex",(energy-sct%E_0)*630D0+(17.6154D0-sct%VAG_a*630D0), &
!                                          lTurnPoint(1)-(sct%posVAG_a_exact+&
!                                                      0.034950D0*1.889725989D0), &
!                                          rTurnPoint(1)-(sct%posVAG_a_exact+&
!                                                      0.034950D0*1.889725989D0)
!   write(stdout,*) "Tpoints approx err ", lerror, rerror
!   write(stdout,*) "Tpoints            "
  if (abs(lerror)>1D-1.or.abs(rerror)>1D-1) then
    write(stderr,*) "Tpoints Problem at Energy ", energy
  end if

! ======================================================================
! Integrate with Simpson sum
! ======================================================================
  realIter(1) = lTurnPoint(1)
  nrSteps = sct%nImgs
!   write(stdout,*) "nrImages for Simpson (theta calc)", sct%nImgs
  tmpStepSize = ((rTurnPoint(1)-lTurnPoint(1)))/REAL(nrSteps-1, rk)
  do i = 1, nrSteps
!     write(stdout,*) "Positionen", realIter(1), sum(sct%steplengths(1:sct%nImgs-1))
    call dlf_sct_eval_csplines(sct%nImgs, sct%MuSpline, &
                                sct%steplengths, 1, realIter(1), tmpMu(1))
    call dlf_sct_eval_csplines(sct%nImgs, sct%VG_aSpline,&
                                sct%steplengths, 1, realIter(1), tmpVG_a(1))
    tmp(1) = 2D0*tmpMu(1)*(tmpVG_a(1)-energy)
    integrand(i) = dsqrt(abs(tmp(1)))
    realIter(1) = realIter(1) + tmpStepSize
  end do
  call dlf_sct_quad_equ(nrSteps, integrand,   tmpStepSize, theta)
  sct%stopper = .false.   
end subroutine dlf_sct_theta
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_init_csplines
!!
!! FUNCTION
!! Generates nInput-1 cubic splines from nInput datapoints
!!
!! COMMENTS
!! splines are of the form 
!! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
!! The function must start at x = 0
!! Uses external functino dgtsv to solve tridiagonal system
!!
!! INPUTS
!! nInput - number of datapoints to be used for interpolation
!! y(nInput) - datapoints for interpolation
!! h(nInput) - distances between those points (uses only h(1:nInput-1))
!!
!! OUTPUTS 
!! param(4,nInput-1) - array with spline parameters
!!
!! SYNOPSIS
subroutine dlf_sct_init_csplines(nInput, y, h, param)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout
  implicit none
  integer, intent(in)    :: nInput ! = (n+1)
  real(rk), intent(in)   :: y(nInput) ! y axis (values at the points)
  real(rk), intent(in)   :: h(nInput) ! distances between the points
  real(rk), intent(out)  :: param(4,nInput-1)
  real(rk)               :: fpp(nInput) ! temporary variables to
                                        ! simplify the calculation
                                        ! fpp stands for f''                                        
  real(rk)               :: diag(nInput-2)
  real(rk)               :: subdiag(nInput-3)
  real(rk)               :: superdiag(nInput-3)
  real(rk)               :: rhs(nInput-2)
  integer                :: n ! Number of Splines that will be created
  integer                :: i, info
  ! one spline less then datapoints
  n = nInput - 1
! ======================================================================
! Calculate temporary variables fpp
! ======================================================================
  ! Natural boundary conditions:
  fpp(:) = 0d0
  fpp(n+1) = 0d0
  fpp(1) = 0d0
  diag(:) = 0d0
  subdiag(:) = 0d0
  superdiag(:) = 0d0
  ! Now calculate all other fpp entries
  ! For this purpose, calculate the matrix that will be inverted

  ! diagonal elements
  do i = 1, n-1
    diag(i) =  2D0*(h(i)+h(i+1))
  end do
  ! off-diagonal elements
  do i = 1, n-2
    subdiag(i) = h(i+1)
    superdiag(i) = h(i+1)
  end do   
  ! Right hand side
  rhs(:) = 0d0
  do i = 1, n-1
    rhs (i) = 6D0*((y(i+2)-y(i+1))/h(i+1)-(y(i+1)-y(i))/h(i)) 
  end do  
  call dgtsv(n-1, 1, subdiag, diag, superdiag, rhs, n-1, info)
! ======================================================================
! Calculate the spline parameters
! ======================================================================
  ! Constants of the spline
  fpp(2:n) = rhs(1:n-1)
  param(1,1:n) = y(1:n)
  do i = 1, n
    param(2,i) = (y(i+1)-y(i))/h(i)-h(i)/6D0*(fpp(i+1)+2*fpp(i))
  end do
  do i = 1, n
    param(3,i) = fpp(i)/2D0
  end do
  do i = 1, n
    param(4,i) = (fpp(i+1)-fpp(i))/(6D0*h(i))
  end do  
end subroutine dlf_sct_init_csplines
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_eval_csplines
!!
!! FUNCTION
!! Evaluates splines
!!
!! COMMENTS
!! splines are of the form 
!! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
!! The function must start at x = 0
!! Must be given same value of nInput
!! and the output of dlf_sct_init_csplines (parameters of the splines)
!!
!! INPUTS
!! nInput - number of datapoints to be used for interpolation
!! param(4,nInput-1) - array with spline parameters
!! h(nInput) - distances between those points (uses only h(1:nInput-1))
!! nevals - number of points that shall be evaluated
!! x(nevals) - points on which the spline should be evaluated
!!
!! OUTPUTS 
!! eval(nevals) - Result of the evaluation
!!
!! SYNOPSIS
subroutine dlf_sct_eval_csplines(nInput, param, h, nevals, x, eval)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout
  implicit none
  integer, intent(in)    :: nInput, nevals ! = (n+1)
  real(rk), intent(in)   :: param(4,nInput-1)
  real(rk), intent(in)   :: h(nInput) ! distances between the points
                                      ! (last entry ignored)
  real(rk), intent(in)   :: x(nevals)
  real(rk), intent(out)  :: eval(nevals)
  real(rk)               :: pos
  integer                :: n ! Number of Splines
  integer                :: i,k
  ! splines are of the form 
  ! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
  n = nInput - 1
  do k = 1, nevals 
    ! evaluate all elements of x, d.h. evaluate at x(k)
    ! Throw error, if evaluation point is too small
    ! Area of definition begins with 0d0 but a tolerance
    ! of 1d-12 seems acceptable.
    if (x(k)<-1d-12) then
      write(stderr,*) "Error: evaluation of spline not ",&
                      "in the area of definition! (too small)"
      stop 42
    end if  
    ! evaluate at position x(k)
    pos = 0d0 ! position of search
    i = 1
    ! n is number of splines
    do while(i<=n-1)
      if(x(k)<=pos+h(i)) then 
        ! evaluate Spline_i at x(k)
        eval(k) = param(1,i) + param(2,i)*(x(k)-pos) + &
                  param(3,i)*(x(k)-pos)**2 + param(4,i)*(x(k)-pos)**3
        exit
      else
        pos = pos + h(i)
        i = i + 1
      end if
    end do 
    ! Tolerance of 1d-12 for the last spline (right border value)
    if (i==n) then
      if (x(k)<= pos+h(n)+1d-12) then
        eval(k) = param(1,n) + param(2,n)*(x(k)-pos) + &
                  param(3,n)*(x(k)-pos)**2 + param(4,n)*(x(k)-pos)**3  
      else
        ! Throw error, if evaluation point is too high
        write(stderr,*) "Error: evaluation of spline not ",&
                        "in the area of definition (too high)!"
        stop 42
      end if
    end if
  end do
end subroutine dlf_sct_eval_csplines
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_eval_csplines_noCheck
!!
!! FUNCTION
!! Evaluates splines
!!
!! COMMENTS
!! splines are of the form 
!! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
!! The function must start at x = 0
!! Must be given same value of nInput
!! and the output of dlf_sct_init_csplines (parameters of the splines)
!! This routine also allows evaluation outside the area of definition!
!!
!! INPUTS
!! nInput - number of datapoints to be used for interpolation
!! param(4,nInput-1) - array with spline parameters
!! h(nInput) - distances between those points (uses only h(1:nInput-1))
!! nevals - number of points that shall be evaluated
!! x(nevals) - points on which the spline should be evaluated
!!
!! OUTPUTS 
!! eval(nevals) - Result of the evaluation
!!
!! SYNOPSIS
subroutine dlf_sct_eval_csplines_noCheck(nInput, param, h, nevals, x, eval)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout
  implicit none
  integer, intent(in)    :: nInput, nevals ! = (n+1)
  real(rk), intent(in)   :: param(4,nInput-1)
  real(rk), intent(in)   :: h(nInput) ! distances between the points
                                      ! (last entry ignored)
  real(rk), intent(in)   :: x(nevals)
  real(rk), intent(out)  :: eval(nevals)
  real(rk)               :: pos
  integer                :: n ! Number of Splines
  integer                :: i,k
  ! splines are of the form 
  ! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
  n = nInput - 1
  do k = 1, nevals 
    ! evaluate all elements of x, d.h. evaluate at x(k)
    ! evaluate at position x(k)
    pos = 0d0 ! position of search
    i = 1
    ! n is number of splines
    do while(i<=n-1)
      if(x(k)<=pos+h(i)) then 
        ! evaluate Spline_i at x(k)
        eval(k) = param(1,i) + param(2,i)*(x(k)-pos) + &
                  param(3,i)*(x(k)-pos)**2 + param(4,i)*(x(k)-pos)**3
        exit
      else
        pos = pos + h(i)
        i = i + 1
      end if
    end do 
    if (i==n) then
        eval(k) = param(1,n) + param(2,n)*(x(k)-pos) + &
                  param(3,n)*(x(k)-pos)**2 + param(4,n)*(x(k)-pos)**3  
    end if
  end do
end subroutine dlf_sct_eval_csplines_noCheck
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_sct_eval_csplines_d
!!
!! FUNCTION
!! Evaluates the derivative of splines
!!
!! COMMENTS
!! splines are of the form 
!! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
!! The function must start at x = 0
!! Must be given same value of nInput
!! and the output of dlf_sct_init_csplines (parameters of the splines)
!!
!! INPUTS
!! nInput - number of datapoints to be used for interpolation
!! param(4,nInput-1) - array with spline parameters
!! h(nInput) - distances between those points (uses only h(1:nInput-1))
!! nevals - number of points that shall be evaluated
!! x(nevals) - points on which the spline should be evaluated
!!
!! OUTPUTS 
!! eval(nevals) - Result of the evaluation
!!
!! SYNOPSIS
subroutine dlf_sct_eval_csplines_d(nInput, param, h, nevals, x, eval)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout
  implicit none
  integer, intent(in)    :: nInput, nevals ! = (n+1)
  real(rk), intent(in)   :: param(4,nInput-1)
  real(rk), intent(in)   :: h(nInput) ! distances between the points
  real(rk), intent(in)   :: x(nevals)
  real(rk), intent(out)  :: eval(nevals)
  real(rk)               :: pos
  integer                :: n ! Number of Splines 
  integer                :: i,k
  ! splines are of the form 
  ! p(i) = z(i)*(x-x(i))+ (z(i+1)-z(i))/(2*h(i))*(x-x(i))^2+c(i)
  n = nInput - 1
  do k = 1, nevals 
    ! evaluate all elements of x, d.h. evaluate at x(k)
    ! Throw error, if evaluation point is too small
    if (x(k)<0d0) then
      write(stderr,*) "Error: evaluation of spline not ",&
                      "in the area of definition! (too small)"
      stop 42
    end if  
    ! evaluate at position x(k)
    pos = 0d0 ! position of search
    i = 1
    do while(i<=n)
      if(x(k)<=pos+h(i)) then 
        ! evaluate Spline_i at x(k)
        eval(k) = param(2,i) + 2D0*param(3,i)*(x(k)-pos) &
                             + 3D0*param(4,i)*(x(k)-pos)**2
        exit
      else
        pos = pos + h(i)
      end if
      i = i + 1
    end do    
    ! Throw error, if evaluation point is too high
    if(i==n+1) then
      write(stderr,*) "Error: evaluation of spline not ",&
                      "in the area of definition (too high)!"
      stop 42
    end if
  end do
end subroutine dlf_sct_eval_csplines_d
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/giveSubHessFromOrthogVec
!!
!! FUNCTION
!! Gives Hessian in a subspace
!!
!! COMMENTS
!! The vectors from which to construct the Hessian (vec) 
!! must be orthonormalized.
!! The vectors from which to construct the hessian 
!! must be in vec(:,nproj+1:dimen)
!!
!! INPUTS
!! dimen - dimension of the original hessian 
!! nproj - number of vectors that shall be projected out
!! vec(dimen,dimen) - vec(:,nproj+1:dimen) contains the vectors
!!                    from which to construct the new hessian
!! hess(dimen,dimen) - the original full dimensional hessian
!!
!! OUTPUTS 
!! phess(dimen,dimen) - phess(nproj+1:dimen, nproj+1:dimen) is the 
!!                      projected hessian
!!
!! SYNOPSIS
subroutine giveSubHessFromOrthogVec(dimen, nproj, vec, hess, phess)
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  use sct_module
  implicit none
  integer, intent(in)   :: dimen ! dimension of the original
                                 ! system/original matrix
  integer, intent(in)   :: nproj ! number of vectors to project out
  real(rk), intent(in)  :: vec(dimen,dimen)  ! vectors to project out
  real(rk), intent(in)  :: hess(dimen,dimen) ! original hessian
  real(rk), intent(out) :: phess(dimen,dimen)! projected hessian
  integer               :: i
  integer               :: j
  integer               :: k
  integer               :: l
  ! constuct the new hessian
  phess(:,:) = 0d0
  do i = 1, dimen-nproj
    do j = 1, dimen-nproj
      do k = 1, dimen
        do l = 1, dimen
          phess(i,j) = phess(i,j) + &
                       vec(k,i) * hess(k,l) * vec(l,j)
        end do
      end do
    end do
  end do
end subroutine giveSubHessFromOrthogVec
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/giveTrRoStTrafo
!!
!! FUNCTION
!! Calculates modes corresponding to translation/rotation/(step)
!!
!! COMMENTS
!! The first 5-7 entries of peigvec are the translational/rotational 
!! eigenmodes and a normalized step vector (if pstep=.true.).
!! The rest of the vectors in peigvec are vectors that
!! are orthonormal to these 5-7 first vectors.
!! Note that the first 5-7 entries are not necessarily orthonormal.
!!
!! INPUTS
!! pstep - should the step vector be projected out?
!! icoords(sct%nivar) - the coordinates in internals
!! step(sct%nivar) - the step vector in internals
!!
!! OUTPUTS 
!! npmodes - number of trans/rot modes + step (if pstep = .true.)
!! peigvec(sct%nivar,sct%nivar) - trans/rot vectors, step vector
!!                                  and arbitrary orthonormal vectors
!!
!! SYNOPSIS
subroutine giveTrRoStTrafo(npmodes, peigvec, pstep, icoords, step)
  use dlf_global, only: glob,stderr,stdout,printl, pi
  use dlf_parameter_module, only: rk
  use sct_module
  implicit none
  integer, intent(out)  :: npmodes
  real(rk), intent(out) :: peigvec(sct%nivar,sct%nivar)
  logical, intent(in)   :: pstep
  real(rk), intent(in)  :: icoords(sct%nivar)
  real(rk), intent(in)  :: step(sct%nivar)  
  logical               :: frozen = .false.
  real(rk)              :: peigval(sct%nivar)
  real(rk)              :: eigval(sct%nivar)
  real(rk), external    :: ddot
  real(rk)              :: moi(3,3)
  real(rk)              :: moitmp(3,3)
  real(rk)              :: moival(3)
  real(rk)              :: moivec(3,3)
  integer               :: ival, kval
  real(rk)              :: comcoords(3,glob%nat) ! centre of mass
                                                 ! coordinates
  real(rk)              :: com(3)
  real(rk)              :: totmass
  real(rk)              :: smass
  real(rk), parameter   :: mcutoff = 1.0d-12
  real(rk)              :: test, epsil
  integer               :: ntrro ! number of trans/rot modes
  integer               :: alpha, beta, i, j
  real(rk)              :: tmp
  real(rk)              :: locvec(sct%nivar,7) ! will be the
                                                ! orthonormalized
                                                ! vectors to project out
  real(rk)              :: tmpvec(sct%nivar)

  ! Do not continue if any coordinates are frozen
  if (sct%nivar /= glob%nat * 3 .or. glob%nat==1) then
    write(stdout,*)
    write(stdout,"('Frozen atoms found: no ',&
        &'modes will be projected out')")
    frozen = .true.
    ntrro = 0
  end if
if (.not.frozen) then
! ======================================================================
! Calculate center of mass and moment of inertia tensor
! ======================================================================
  ! xcoords from icoords
  call dlf_cartesian_itox(sct%nat, sct%nivar, sct%nicore, &
       sct%massweight, icoords(:), comcoords)
  com(:) = 0.0d0
  totmass = 0.0d0
  do ival = 1, glob%nat
     com(1:3) = com(1:3) + glob%mass(ival) * comcoords(1:3, ival)
     totmass = totmass + glob%mass(ival)
  end do
  ! Center of Mass coordinate (COM)
  com(1:3) = com(1:3) / totmass
  ! calculate everything in to COM coordinates
  do ival = 1, glob%nat
     comcoords(1:3, ival) = comcoords(1:3, ival) - com(1:3)
  end do
  ! Moment of inertia Tensor
  moi(:,:) = 0.0d0
  do ival = 1, glob%nat
     moi(1,1) = moi(1,1) + glob%mass(ival) * &
          (comcoords(2,ival) * comcoords(2,ival) &
           + comcoords(3,ival) * comcoords(3,ival))
     moi(2,2) = moi(2,2) + glob%mass(ival) * &
          (comcoords(1,ival) * comcoords(1,ival) &
           + comcoords(3,ival) * comcoords(3,ival))
     moi(3,3) = moi(3,3) + glob%mass(ival) * &
          (comcoords(1,ival) * comcoords(1,ival) &
           + comcoords(2,ival) * comcoords(2,ival))
     moi(1,2) = moi(1,2) - &
                glob%mass(ival) * comcoords(1, ival) * &
                comcoords(2, ival)
     moi(1,3) = moi(1,3) - &
                glob%mass(ival) * comcoords(1, ival) * &
                comcoords(3, ival)
     moi(2,3) = moi(2,3) - &
                glob%mass(ival) * comcoords(2, ival) * &
                comcoords(3, ival)
  end do
  moi(2,1) = moi(1,2)
  moi(3,1) = moi(1,3)
  moi(3,2) = moi(2,3)

  if (printl >= 6) then
     write(stdout,"(/,'Centre of mass'/3f15.5)") com(1:3)
     write(stdout,"('Moment of inertia tensor')")
     write(stdout,"(3f15.5)") moi(1:3, 1:3)
     write(stdout,"('Principal moments of inertia')")
     write(stdout,"(3f15.5)") moival(1:3)
     write(stdout,"('Principal axes')")
     write(stdout,"(3f15.5)") moivec(1:3, 1:3)
  end if
! ======================================================================
! Calculate the modes according to Miller, Handy, Adams 
! J.Chem.Phys., Vol 72, No 1, Januar 1980
! ======================================================================
  call dlf_matrix_invert(3,.false.,moi,tmp)
  ! Calculate moi^-1/2 = S sqrt(D) S^-1 (Diagonalization)
  call dlf_matrix_diagonalise(3, moi, moival, moivec)
  moival(:) = dsqrt(moival(:))
  !moivec = transpose(moivec)
  do ival = 1, 3
    do kval = 1, 3
      moitmp(ival,kval) = moival(ival)*moivec(kval,ival) !sqrt(D) S^{-1}
    end do
  end do
  ! moivec = transpose(moivec)
  call dlf_matrix_multiply(3, 3, 3, &
       1.0d0, moivec, moitmp, 0.0d0, moi) ! S sqrt(D) S^{-1}
  ! moi = moi^-1/2 -> Getestet und klappt
  peigvec(:,:) = 0d0
  ntrro = 6
  do ival = 1, glob%nat
     !S ee formulas of the paper
     smass = dsqrt(glob%mass(ival))
     kval = 3 * (ival - 1)
     ! Translational vectors
     peigvec(kval+1, 1) = smass/dsqrt(totmass)  !gamma=1
     peigvec(kval+2, 2) = smass/dsqrt(totmass)  !gamma=2
     peigvec(kval+3, 3) = smass/dsqrt(totmass)  !gamma=3
     ! Rotational vectors
     do alpha = 1, 3
       do beta = 1, 3
         ! gamma=1
         peigvec(kval+1, 4) = peigvec(kval+1, 4) + &
                              moi(1,alpha)*epsil(alpha,beta,1)*&
                              comcoords(beta,ival)*smass 
         ! gamma=2
         peigvec(kval+2, 4) = peigvec(kval+2, 4) + &
                              moi(1,alpha)*epsil(alpha,beta,2)*&
                              comcoords(beta,ival)*smass
         ! gamma=3 
         peigvec(kval+3, 4) = peigvec(kval+3, 4) + &
                              moi(1,alpha)*epsil(alpha,beta,3)*&
                              comcoords(beta,ival)*smass 

         peigvec(kval+1, 5) = peigvec(kval+1, 5) + &
                              moi(2,alpha)*epsil(alpha,beta,1)*&
                              comcoords(beta,ival)*smass
         peigvec(kval+2, 5) = peigvec(kval+2, 5) + &
                              moi(2,alpha)*epsil(alpha,beta,2)*&
                              comcoords(beta,ival)*smass
         peigvec(kval+3, 5) = peigvec(kval+3, 5) + &
                              moi(2,alpha)*epsil(alpha,beta,3)*&
                              comcoords(beta,ival)*smass

         peigvec(kval+1, 6) = peigvec(kval+1, 6) + &
                              moi(3,alpha)*epsil(alpha,beta,1)*&
                              comcoords(beta,ival)*smass
         peigvec(kval+2, 6) = peigvec(kval+2, 6) + &
                              moi(3,alpha)*epsil(alpha,beta,2)*&
                              comcoords(beta,ival)*smass
         peigvec(kval+3, 6) = peigvec(kval+3, 6) + &
                              moi(3,alpha)*epsil(alpha,beta,3)*&
                              comcoords(beta,ival)*smass
       end do
     end do
  end do

  ! Check for linear molecules (one less mode)
  do ival = 1, 6
     test = ddot(sct%nivar, peigvec(1,ival), 1, peigvec(1,ival), 1)
     if (test < mcutoff) then
        kval = ival
        ntrro = ntrro - 1
        if (ntrro < 5) then
           write(stdout,&
                 "('Error: too few rotational/translation modes')")
           peigval = eigval
           return
        end if
     end if
  end do
  ! Last one (Number 6) is the missing Mode and set to zero if one only
  ! has 5 rotational modes
  if (ntrro == 5 .and. kval /= 6) then
     peigvec(:, kval) = peigvec(:, 6)
     peigvec(:, 6) = 0.0d0
  end if

  ! Normalise all vectors
  do ival = 1, ntrro
    peigvec(:,ival) = peigvec(:,ival)/&
                      dsqrt(dot_product(peigvec(:,ival),&
                                        peigvec(:,ival)))
  end do  

  !write (stdout,*) "number of trans/rot modes: ", ntrro
end if
  ! Project out step?
  if (pstep) then
    peigvec(:,ntrro+1) = step(:)/&
                         dsqrt(dot_product(step(:),step(:)))
    npmodes = ntrro+1
  else
    ! Number of vectors to project out is equal to number of trans/rot
    ! vectors
    npmodes = ntrro
  end if

! ======================================================================
! Generate the vibrational modes using Gram-Schmidt
! ======================================================================
! The vibrational modes are put them in the 
! remaining peigvec components: peigvec(:,npmodes+1:sct%nivar)
  ! Initialize with random vector that is normalized
  call random_seed()
  do i = npmodes+1, sct%nivar
    do j = 1, sct%nivar
      CALL RANDOM_NUMBER(tmp)    
      peigvec(j,i) = tmp
    end do
    peigvec(:,i) = peigvec(:,i)/&
                   dsqrt(dot_product(peigvec(:,i),&
                                     peigvec(:,i)))
  end do    
  ! Move (already normalized) Vectors to local vectors that have to be
  ! projected out
  locvec(:,:) = 0d0
  do i = 1, npmodes
    locvec(:,i) = peigvec(:,i)
  end do 

  ! Until now, not all locvecs are orthogonal to each other. 
  ! Guarantee this now. Keep Normalization.
  do i = 1, npmodes
    tmpvec(:) = 0d0
    do j = 1, i-1
      tmpvec(:) = tmpvec(:) &
                  + dot_product(locvec(:,i), locvec(:,j)) &
                  * locvec(:,j)
    end do
    locvec(:,i) = locvec(:,i) - tmpvec(:)
    locvec(:,i) = locvec(:,i)/&
                  dsqrt(dot_product(locvec(:,i),&
                                    locvec(:,i)))
  end do

  ! Generate the vectors that are orthogonal to all locvecs
  ! (trans/rot/step)
  do i = npmodes+1, sct%nivar
    tmpvec(:) = 0d0
    do j = 1, npmodes    
      tmpvec(1:sct%nivar) = tmpvec(1:sct%nivar) + & 
                             dot_product(peigvec(1:sct%nivar,i), &
                                         locvec(1:sct%nivar,j)) &
                             * locvec(1:sct%nivar,j)
    end do
    do j = npmodes+1, i-1, 1
      tmpvec(1:sct%nivar) = tmpvec(1:sct%nivar) + &
                             dot_product(peigvec(1:sct%nivar,i), &
                                         peigvec(1:sct%nivar,j)) &
                             * peigvec(1:sct%nivar,j)
    end do
    peigvec(1:sct%nivar,i) = peigvec(1:sct%nivar,i) &
                              - tmpvec(1:sct%nivar)
    peigvec(1:sct%nivar,i) = peigvec(1:sct%nivar,i) &
                              /dsqrt(dot_product&
                                    (peigvec(1:sct%nivar,i),&
                                     peigvec(1:sct%nivar,i)))
  end do  

  ! Test if they are orthogonal
  tmp = 0d0
  do i = npmodes+1, sct%nivar
    do j = npmodes+1, i-1
      tmp = tmp + abs(dot_product(peigvec(:,i), peigvec(:,j)))
    end do
    do j = 1, npmodes
      tmp = tmp + abs(dot_product(peigvec(:,i), locvec(:,j)))
    end do
  end do
  if(tmp.ge.1D-5) then 
    write(stderr,*) "Problem with peigvec not beeing orthogonal",& 
                    " to vectors that are projected out or itself"
  end if
end subroutine giveTrRoStTrafo
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/dlf_GSAdiabaticPot
!!
!! FUNCTION
!! Calculates ground state adiabatic potential from vibrational
!! eigenvalues
!!
!! COMMENTS
!! It is assumed that the input eigenvalues are calculated from a 
!! Hessian in mass weighted coordinates with masses in units of u. 
!! The output is given in atomic units, i.e. Hartree.
!!
!! INPUTS
!! eigval(neigval) - eigenvalues from which to compute the vib. energy
!! neigval - number of eigenvalues
!!
!! OUTPUTS 
!! energy - vibrational energy
!!
!! SYNOPSIS
!! Used to calculate the ground state adiabatic potential 
!! in rigid-rotor-harmonic-oscillatory approximation 
!! from the eigenvalues of (projected) Hessian.
subroutine dlf_GSAdiabaticPot(eigval, neigval, energy)
  use dlf_global, only: glob,stdout,printl
  use dlf_parameter_module, only: rk
  implicit none
  real(rk), intent(out) :: energy ! eigenvalues after projection
  real(rk), intent(in)  :: eigval(neigval)
  integer, intent(in)   :: neigval
  integer               :: i, counter
  counter = 0
  energy = 0d0
  do i = 1, neigval
    if (eigval(i).GT.1d-10) then
      ! hbar = 1 & factor sqrt(u/m_e) Masse
      energy = energy + 0.5D0 * dsqrt((eigval(i))) / 42.695d0 
    else
      counter = counter + 1 
    end if
  end do
#ifdef DebugInformation  
  write(stdout,*) counter, &
                  " eigenvalues excluded/too small; vib energy:", &
                  energy, "overall adiabatic pot:", energy + glob%energy
#endif                  
end subroutine dlf_GSAdiabaticPot
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/epsil
!!
!! FUNCTION
!! Levi-Civita Symbol
!!
!! INPUTS
!! alpha - first index
!! beta - second index
!! gamma - third index
!!
!! OUTPUTS 
!! epsil - gets 1, -1 or 0 depending on the indices
!!
!! SYNOPSIS
function epsil(alpha,beta,gamma)
  use dlf_parameter_module, only: rk  
  integer, intent(in)  :: alpha  
  integer, intent(in)  :: beta
  integer, intent(in)  :: gamma
  real(rk)              :: epsil
  SELECT CASE (alpha)
    CASE(1)
      SELECT CASE (beta)   
        CASE(2)
          if(3.eq.gamma) then
            epsil = 1d0
          else
            epsil = 0d0
          end if
          return
        CASE(3)
          if(2.eq.gamma) then
            epsil = -1d0
          else
            epsil = 0d0
          end if
          return
        CASE DEFAULT
          epsil = 0d0
      END SELECT
    CASE(2)
      SELECT CASE (beta)
        CASE(1)
          if(3.eq.gamma) then
            epsil = -1d0
          else
            epsil = 0d0
          end if
          return
        CASE(3)
          if(1.eq.gamma) then
            epsil = 1d0
          else
            epsil = 0d0
          end if
          return
        CASE DEFAULT
          epsil = 0d0
      END SELECT
    CASE(3)
      SELECT CASE (beta)
        CASE(1)
          if(2.eq.gamma) then
            epsil = 1d0
          else
            epsil = 0d0
          end if
          return
        CASE(2)
          if(1.eq.gamma) then
            epsil = -1d0
          else
            epsil = 0d0
          end if
          return
        CASE DEFAULT
          epsil = 0d0
      END SELECT
    CASE DEFAULT
      epsil = 0d0
  END SELECT
  epsil = 0d0
end function epsil
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* sct/delt
!!
!! FUNCTION
!! Kronecker-Delta
!!
!! INPUTS
!! i - first index
!! j - second index
!!
!! OUTPUTS 
!! delt - gets 1 or 0 depending on the indices
!!
!! SYNOPSIS
function delt(i,j)
  use dlf_parameter_module, only: rk  
  integer, intent(in)  :: i  
  integer, intent(in)  :: j
  real(rk)             :: delt
  if (i.eq.j) then
    delt = 1d0
  else 
    delt = 0d0
  end if
end function delt

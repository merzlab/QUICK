! **********************************************************************
! **                   Nudged elastic band method                     **
! **********************************************************************
!!****h* coords/neb
!!
!! NAME
!! neb
!!
!! FUNCTION
!! Nudged Elastic band method
!!
!!
!! DATA
!! $Date$
!! $Rev$
!! $Author$
!! $URL$
!! $Id$
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
!! SYNOPSIS
module dlf_neb
  use dlf_parameter_module, only: rk
  type neb_type
    integer    :: nimage              ! total number of images
    integer    :: iimage              ! current image
    integer    :: varperimage         ! number of variables to be 
                                      !   optimised per image
    integer    :: coreperimage        ! number of variables in inner 
                                      !   microiterative region per image
    integer    :: step                ! number of iterations done
    integer    :: maximage            ! number of the image closest to 
                                      !   the climbing image
    integer    :: mode                ! 0: endpoints free, 
                                      ! 1: endpoints move perp. to tau
                                      ! 2: endpoints frozen
    real(rk)   :: k                   ! force constant
    logical    :: tclimb              ! a climbing images is used
    real(rk)   :: tolfreeze           ! gradient tolerance for freezing 
                                      !   images
    real(rk)   :: tolclimb            ! gradient tolerance for starting 
                                      !   climbing image process
    logical    :: allfrozen           ! all images but climbing image 
                                      !   are frozen
    real(rk),allocatable :: ene(:)    ! (nimage) energies of the images
    integer ,allocatable :: cstart(:) ! (nimage) start position of image
                                      !   coordinate array
    integer ,allocatable :: cend(:)   ! (nimage) end position of image 
                                      !   coordinate array

    ! needed for freezing windows
    logical              :: tfreeze   ! decide if windows to be frozen
    logical ,allocatable :: frozen(:) ! (nimage) image not recalculated 
                                      !   due to small gradient
    real(rk),allocatable :: gradt(:)  ! (nimage*varperimage) true 
                                      !   internal gradient
    real(rk),allocatable :: tau(:)    ! (nimage*varperimage) Tangent 
                                      !   vectors

    ! Cartesian coordinates of each image are required for HDLCs
    real(rk),allocatable :: xcoords(:,:) ! (3*glob%nat,nimage) Cartesian
                                      !  coordinates of each image
    logical              :: optcart   ! use internals only for initial 
                                      !   path, then Cartesians
  end type neb_type
  integer,parameter   :: unitp=1000 ! base unit for path xyz files
  logical,parameter   :: xyzall=.true. ! xyz files of all atoms or only active atoms?
  integer,parameter   :: maxxyzfile=300 ! larger than 9999 will require modification of file names
  type(neb_type),save :: neb
  real(rk)  :: beta_hbar,ene_ins_ext
  ! temp: using L-BFGS with a memory of only 2 converges reasonably well with the string method
  logical, parameter:: string=.false.
end module dlf_neb
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* neb/dlf_neb_init
!!
!! FUNCTION
!!
!! Initialise NEB calculations:
!! * allocate arrays concerning internal coordinates
!! * set default values for the calculation
!! * define the starting chain (call dlf_neb_definepath for that)
!!
!! SYNOPSIS
subroutine dlf_neb_init(nimage,icoord)
!! SOURCE
  use dlf_parameter_module, only: rk,ik
  use dlf_global, only: glob,stderr,stdout,printf,printl
  use dlf_neb, only: neb,unitp,xyzall,maxxyzfile
  use dlf_allocate, only: allocate,deallocate
  implicit none
  integer, intent(in)  :: nimage
  integer, intent(in)  :: icoord ! choice of NEB details
  integer              :: iimage,ivar,iat
  real(rk)             :: tolg
  character(20)        :: filename
  real(rk), allocatable :: tmpcoords(:,:)
  integer              :: useimage(nimage)
  integer              :: iarr3(3),nframe,iframe
! **********************************************************************
  neb%nimage=nimage
  neb%iimage=1
  neb%k=glob%nebk
  neb%step=0

  neb%tfreeze=.true.
  if(icoord==190) neb%tfreeze=.false.

  ! check nimage
  if(nimage<3) call dlf_fail("More than 2 images have to be used in NEB")

  ! check coords2
  if(.not.glob%tcoords2) then
    call dlf_fail("Second set of coordinates required for NEB!")
  end if

  iarr3=shape(glob%xcoords2)
  nframe=iarr3(3)
  if(nframe>nimage-2.and.icoord/=190) then
    nframe=nimage-2
    write(stdout,'("Warning: too many frames input in NEB. Only the &
        &first",i4," frames are used")') nframe
  end if
  if(nframe>nimage-1.and.icoord==190) then
    if(printl>=2) write(stdout,'("Warning: too may frames input in qTS. The middle &
        &",i4," frames will be removed")') nframe-nimage
    do ivar=nimage/2,nimage
      glob%xcoords2(:,:,ivar)=glob%xcoords2(:,:,ivar+nframe-nimage)
    end do
  end if

 ! set mode
  if( (icoord<100 .or. icoord>199))&
       call dlf_fail("Wrong icoord in NEB")
  neb%mode=(icoord-100)/10
  ! catch qTS
  if(neb%mode==9) neb%mode=0

  if(neb%mode>5) call dlf_fail("Mode in NEB has to be 0-5")
  neb%optcart=.false.
  if(neb%mode>2) then
    neb%optcart=.true.
    neb%mode=neb%mode-3
  end if

  ivar=mod(icoord,10)
  select case (ivar)
  ! Cartesian coordinates
  case (0)
    ! define the number of internal variables (covering all images)
    call dlf_direct_get_nivar(0, neb%varperimage)
    call dlf_direct_get_nivar(1, neb%coreperimage)

    ! calculate iweights
    call allocate( glob%iweight,neb%varperimage*nimage)
    ! first image
    ivar=1
    if(glob%tatoms) then
    do iat=1,glob%nat
      if(glob%spec(iat)>=0) then
        glob%iweight(ivar:ivar+2)=glob%weight(iat)
        ivar=ivar+3
      else if(glob%spec(iat)==-1) then
      else if(glob%spec(iat)>=-4) then
        glob%iweight(ivar:ivar+1)=glob%weight(iat)
        ivar=ivar+2
      else
        glob%iweight(ivar)=glob%weight(iat)
        ivar=ivar+1
      end if
    end do
    if(ivar-1/=neb%varperimage) then
      call dlf_fail("Error in Cartesian iweight calculation in NEB")
    end if
    ! copy weights to all other images
    do iimage=2,nimage
      ivar=neb%varperimage*(iimage-1)+1
      glob%iweight(ivar:ivar+neb%varperimage-1)=glob%iweight(1:neb%varperimage)
    end do
    else
      glob%iweight(:)=1.D0
    end if

  ! HDLC
  case(1:4)
    call dlf_hdlc_init(glob%nat,glob%spec,mod(glob%icoord,10),glob%ncons, &
        glob%icons,glob%nconn,glob%iconn)

    call dlf_hdlc_get_nivar(0, neb%varperimage)
    call dlf_hdlc_get_nivar(1, neb%coreperimage)

    ! create hdlc with start and endpoint as coordinates
    call allocate(tmpcoords,3,(1+nframe)*glob%nat)
    tmpcoords(:,1:glob%nat)=glob%xcoords(:,:)
    tmpcoords(:,glob%nat+1:(1+nframe)*glob%nat)= &
        reshape(glob%xcoords2(:,:,:),(/3,nframe*glob%nat/))
    call dlf_hdlc_create(glob%nat,neb%coreperimage,glob%spec,glob%micspec,&
        glob%znuc,1+nframe,tmpcoords,glob%weight,glob%mass)
    call deallocate(tmpcoords)

    ! calculate iweights
    call allocate( glob%iweight,neb%varperimage*nimage)
    ! first image
    call dlf_hdlc_getweight(glob%nat,neb%varperimage,neb%coreperimage,glob%micspec,&
         glob%weight,glob%iweight(1:neb%varperimage))
    ! copy weights to all other images
    do iimage=2,nimage
      ivar=neb%varperimage*(iimage-1)+1
      glob%iweight(ivar:ivar+neb%varperimage-1)=glob%iweight(1:neb%varperimage)
    end do


  case default
    write(stderr,'("Coordinate type ",i2," not supported in NEB")') icoord
  end select

  glob%nivar=neb%varperimage*nimage
  glob%nicore = neb%coreperimage * nimage

  call allocate( glob%icoords,glob%nivar)
  call allocate( glob%igradient,glob%nivar)
  call allocate( glob%step,glob%nivar) 
  call allocate( neb%ene,nimage)
  call allocate( neb%cstart,nimage)
  call allocate( neb%cend,nimage)
  ! strictly only necessary for HDLCs
  if(glob%tatoms) then
    call allocate( neb%xcoords,3*glob%nat,nimage)
  else
    call allocate( neb%xcoords,glob%nvar,nimage)
  end if

  call allocate(neb%frozen,nimage)
  neb%frozen(:)=.false.
  call allocate(neb%gradt,glob%nivar)
  call allocate(neb%tau,glob%nivar)

  ! set the positions of the images in the internal arrays
  do iimage=1,nimage
    neb%cstart(iimage)=(iimage-1)*neb%varperimage+1
    neb%cend(iimage)=neb%cstart(iimage)+neb%varperimage-1
  end do
  ! initialise energy array
  neb%ene(:) = 0.0d0

  ! ====================================================================
  ! Define the starting path: linear transit guess
  ! ====================================================================

  ! align the input coordinates to closest overlap
  useimage(:)=0
  useimage(1)=1
  if(glob%tatoms) then
    neb%xcoords(:,1)=reshape(glob%xcoords(:,:),(/3*glob%nat/))
    do iframe=1,min(nframe,nimage-1)
      if(glob%nat>1) call dlf_cartesian_align(glob%nat,glob%xcoords,glob%xcoords2(:,:,iframe))
      neb%xcoords(:,1+iframe)=reshape(glob%xcoords2(:,:,iframe),(/3*glob%nat/))
      useimage(1+iframe)=1
    end do
    
    if(icoord/=190) then
      ! future climbing image - coords for first image:
      neb%xcoords(:,nimage)=reshape(glob%xcoords(:,:),(/3*glob%nat/))
      ! set i- and x-coordinates of all images
      call dlf_neb_definepath(nimage,useimage)
      ! future climbing image again:
      neb%xcoords(:,nimage)=reshape(glob%xcoords(:,:),(/3*glob%nat/))
    else
      call dlf_qts_init() ! here only possible w/o alloc_tau
      if(glob%iopt/=12) call dlf_qts_definepath(nimage,useimage)
    end if
  else
    neb%xcoords(:,1)=reshape(glob%xcoords(:,:),(/glob%nvar/))
    do iframe=1,nframe
      neb%xcoords(:,1+iframe)=reshape(glob%xcoords2(:,:,iframe),(/glob%nvar/))
      useimage(1+iframe)=1
    end do
    
    neb%xcoords(:,nimage)=reshape(glob%xcoords(:,:),(/glob%nvar/))

    ! set i- and x-coordinates of all images
    call dlf_neb_definepath(nimage,useimage)

    neb%xcoords(:,nimage)=reshape(glob%xcoords(:,:),(/glob%nvar/))
  end if
  
  if(neb%optcart) then
    ! use the path guessed in internal coordinates, but optimise in
    ! Cartesians
    glob%icoord=glob%icoord-mod(glob%icoord,10)-30
    call dlf_direct_get_nivar(0, ivar)
    call dlf_hdlc_destroy
    write(stdout,"('Initial path set. Switching to Cartesian coordinates.')")
    if(ivar/=neb%varperimage) then
      neb%varperimage=ivar
      glob%nivar=neb%varperimage*nimage
      call deallocate( glob%icoords)
      call deallocate( glob%igradient)
      call deallocate( glob%step) 
      call deallocate( glob%iweight)
      call allocate( glob%icoords,glob%nivar)
      call allocate( glob%igradient,glob%nivar)
      call allocate( glob%step,glob%nivar) 
      call allocate( glob%iweight,glob%nivar)
      if(neb%tfreeze) then
        call deallocate(neb%gradt)
        call allocate(neb%gradt,glob%nivar)
        call deallocate(neb%tau)
        call allocate(neb%tau,glob%nivar)
      end if
      do iimage=1,nimage
        neb%cstart(iimage)=(iimage-1)*neb%varperimage+1
        neb%cend(iimage)=neb%cstart(iimage)+neb%varperimage-1
      end do

    end if ! (ivar/=neb%varperimage)

    ! calculate iweights
    ! first image
    ivar=1
    do iat=1,glob%nat
      if(glob%spec(iat)>=0) then
        glob%iweight(ivar:ivar+2)=glob%weight(iat)
        ivar=ivar+3
      else if(glob%spec(iat)==-1) then
      else if(glob%spec(iat)>=-4) then
        glob%iweight(ivar:ivar+1)=glob%weight(iat)
        ivar=ivar+2
      else
        glob%iweight(ivar)=glob%weight(iat)
        ivar=ivar+1
      end if
    end do
    if(ivar-1/=neb%varperimage) then
      call dlf_fail("Error in Cartesian iweight calculation in NEB")
    end if
    ! copy weights to all other images
    do iimage=2,nimage
      ivar=neb%varperimage*(iimage-1)+1
      glob%iweight(ivar:ivar+neb%varperimage-1)=glob%iweight(1:neb%varperimage)
    end do

  end if

  if(icoord==190) then
    ! set tau depending on the coordinates
    call qts_tau_from_path()
  end if

  ! set x-coordinates of the first image
  ! nothing to do ...
  neb%gradt(:)=0.D0
  glob%igradient(:)=0.D0

  ! open units for writing the path xyz files
  if(printf>=4) then
    do iimage=1,nimage
      if(iimage>maxxyzfile) exit
      if(iimage<10) then
        write(filename,'("000",i1)') iimage
      else if(iimage<100) then
        write(filename,'("00",i2)') iimage
      else if(iimage<1000) then
        write(filename,'("0",i3)') iimage
      else
        write(filename,'(i4)') iimage
      end if
      filename="neb_"//trim(adjustl(filename))//".xyz"
      if (glob%iam == 0) open(unit=unitp+iimage,file=filename)
    end do

    ! write initial xyz coordinates
    if (glob%iam == 0) then
      do iimage=1,nimage
        if(iimage>maxxyzfile) exit
        if(xyzall) then
          call write_xyz(unitp+iimage,glob%nat,glob%znuc,neb%xcoords(:,iimage))
        else
          call write_xyz_active(unitp+iimage,glob%nat,glob%znuc,glob%spec,neb%xcoords(:,iimage))
        end if
      end do
    end if
  end if

  ! initialise climbing image, freezing
  call convergence_get("TOLG",tolg)
  ! set freezing criterion
  neb%tolfreeze= glob%neb_freeze_test * tolg 
  neb%tolclimb=  glob%neb_climb_test * tolg 
  neb%tclimb=.false.
  if(icoord/=190) neb%frozen(neb%nimage)=.true.
  neb%allfrozen=.false.
  
  ! commented out so climbing image can be switched off altogether.
  ! TODO will need extra code to handle convergence/result in this case
  ! if(neb%tolclimb< neb%tolfreeze) neb%tolclimb=neb%tolfreeze

end subroutine dlf_neb_init
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* neb/dlf_neb_destroy
!!
!! FUNCTION
!!
!! deallocate arrays concerning internal coordinates
!!
!! SYNOPSIS
subroutine dlf_neb_destroy
!! SOURCE
  use dlf_parameter_module, only: rk,ik
  use dlf_global, only: glob,stderr,printf
  use dlf_neb, only: neb,unitp,maxxyzfile
  use dlf_allocate, only: deallocate
  implicit none
  integer :: iimage
! **********************************************************************

  if(glob%icoord==190) then
    call dlf_qts_destroy() 
  end if

 ! deallocate arrays
  if (allocated(glob%icoords)) call deallocate( glob%icoords)
  if (allocated(glob%igradient)) call deallocate( glob%igradient)
  if (allocated(glob%step)) call deallocate( glob%step)
  if (allocated(glob%iweight)) call deallocate( glob%iweight)
  if (allocated(neb%ene)) call deallocate(neb%ene)
  if (allocated(neb%cstart)) call deallocate(neb%cstart)
  if (allocated(neb%cend)) call deallocate(neb%cend)
  if (allocated(neb%xcoords)) call deallocate(neb%xcoords)

  if (allocated(neb%frozen)) call deallocate(neb%frozen)
  if (allocated(neb%gradt)) call deallocate(neb%gradt)
  if (allocated(neb%tau)) call deallocate(neb%tau)

  select case (mod(glob%icoord,10))
  ! HDLC
  case(1:4)
    call dlf_hdlc_destroy()
  end select

  ! close units for writing the path xyz files
  if(printf>=4 .and. glob%iam == 0) then
    do iimage=1,neb%nimage
      if(iimage>maxxyzfile) exit
      close(unit=unitp+iimage)
    end do
  end if

end subroutine dlf_neb_destroy
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* neb/dlf_neb_xtoi
!!
!! FUNCTION
!!
!! transform Cartesian coordinates into the NEB array
!!
!! This does not only SET the internal (NEB) coordinates, but also sets
!! the xcoords of the next image from neb%xcoords:
!! * First image : i-coordinates and gradient are set to x-coordinates 
!!   and gradient. x-coordinates are set to second image
!! * Nth image : i-coordinates and gradient are set to x-coordinates 
!!   and gradient. x-coordinates are set to N+1's image
!! * Last image : i-coordinates and gradient are set to x-coordinates 
!!   and gradient. Spring forces are added (calls 
!!   dlf_neb_improved_tangent_neb for that). trerun_energy=false
!!
!! SYNOPSIS
subroutine dlf_neb_xtoi(trerun_energy,external_iimage)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printf
  use dlf_neb, only: neb,unitp,xyzall,ene_ins_ext,maxxyzfile
  implicit none
  logical, intent(out)  :: trerun_energy
  integer, intent(out)  :: external_iimage ! Which image is the next to calculate
  integer               :: status ! For parallel NEB tested here
  integer               :: cstart,cend,jimage,iimage
  logical               :: tok
! **********************************************************************
  ! start and endpoint of current image in array icoords and igradient
  cstart=neb%cstart(neb%iimage)
  cend=neb%cend(neb%iimage)

  ! Parallel NEB: X->I transformation is potentially costly so parallelised
  if (glob%dotask) then
     call dlf_direct_xtoi(glob%nvar,neb%varperimage,neb%coreperimage,&
          glob%xcoords,glob%xgradient,glob%icoords(cstart:cend),&
          glob%igradient(cstart:cend))
     neb%ene(neb%iimage)=glob%energy

     ! store true gradient (to calculate the work even if this image is frozen)
     neb%gradt(cstart:cend)=glob%igradient(cstart:cend)
  else
     ! Explicitly set to zero to ensure that the allreduce at the end of the
     ! cycle gives the correct result
     glob%icoords(cstart:cend) = 0.d0
     glob%igradient(cstart:cend) = 0.d0
     neb%ene(neb%iimage) = 0.d0
     neb%gradt(cstart:cend) = 0.d0
  end if

  ! write xyz file
  if(printf>=4.and.glob%tatoms.and.neb%iimage<=maxxyzfile .and. glob%iam == 0) then
    if(xyzall) then
      call write_xyz(unitp+neb%iimage,glob%nat,glob%znuc,glob%xcoords(:,:))
    else
      call write_xyz_active(unitp+neb%iimage,glob%nat,glob%znuc,glob%spec,glob%xcoords(:,:))
    end if
  end if

  if(neb%iimage<neb%nimage) then

    neb%iimage=neb%iimage+1
    trerun_energy=.true.

    if(neb%tfreeze) then
      do jimage=neb%iimage,neb%nimage
        if(neb%frozen(jimage)) then
          ! icoords did not vary
          ! set gradient
          cstart=neb%cstart(jimage)
          cend=neb%cend(jimage)
          glob%igradient(cstart:cend)=0.D0
          neb%iimage=jimage

          ! Parallel NEB: this is to avoid multiple counting at the end
          if (.not. glob%dotask) then
             glob%icoords(cstart:cend) = 0.d0
             neb%ene(neb%iimage) = 0.d0
             neb%gradt(cstart:cend) = 0.d0           
          end if

          ! write xyz file
          ! coords i->x
          if(printf>=4.and.glob%tatoms.and.neb%iimage<=maxxyzfile) then
            glob%xcoords=reshape(neb%xcoords(:,jimage),(/3,glob%nat/))
            if (glob%iam == 0) then
               if(xyzall) then
                 call write_xyz(unitp+neb%iimage,glob%nat,glob%znuc,&
                                glob%xcoords(:,:))
               else
                 call write_xyz_active(unitp+neb%iimage,glob%nat,glob%znuc,&
                                       glob%spec,glob%xcoords(:,:))
               end if
            end if
          end if
          
        else
          neb%iimage=jimage
          exit
        end if
      end do
      if(neb%iimage==neb%nimage.and.neb%frozen(neb%iimage)) trerun_energy=.false.
    end if
    
    if(trerun_energy) then
      cstart=neb%cstart(neb%iimage)
      cend=neb%cend(neb%iimage)
      external_iimage=neb%iimage
      if(glob%tatoms) then
        glob%xcoords=reshape(neb%xcoords(:,neb%iimage),(/3,glob%nat/))
      else
        glob%xcoords=reshape(neb%xcoords(:,neb%iimage),(/1,glob%nvar/))
      end if
      !-------------
      return
      !-------------
    end if
  end if

  ! ====================================================================
  ! LAST IMAGE, ALL GRADIENTS ARE GATHERED
  ! ====================================================================

  ! Parallel NEB: check no gradient evaluations in other workgroups failed
  ! and make all coordinates and gradients available to all workgroups
  if (glob%ntasks > 1) then
    ! If it has reached here all gradients have succeeded in this workgroup
    status = 0
    call dlf_tasks_int_sum(status, 1)
    if (status > 0) then
      call dlf_fail("Task-farmed gradient evaluations failed")
    end if
    if (glob%serial_cycle == 0) then
      call dlf_tasks_real_sum(glob%icoords, glob%nivar)
      call dlf_tasks_real_sum(glob%igradient, glob%nivar)
      call dlf_tasks_real_sum(neb%ene, neb%nimage)
      call dlf_tasks_real_sum(neb%gradt, glob%nivar)
    end if
  end if
  
  if(glob%icoord==190) then

    ! freeze boundary points
    if(neb%tfreeze) then
      neb%frozen(1)=.true.
      glob%igradient(neb%cstart(1):neb%cend(1))=0.D0
      neb%frozen(neb%nimage)=.true.
      glob%igradient(neb%cstart(neb%nimage):neb%cend(neb%nimage))=0.D0
    end if

    if(glob%iopt/=12) then
       tok=.true.
       call dlf_qts_trans_force(trerun_energy,tok)
       ! tok always returns true for the time being ...

       if(.not.tok) then
          call  dlf_fail("No quantum transition state could be found")
       end if

!!$    tok=.true.
!!$    call dlf_qts_r(trerun_energy,tok)
!!$
!!$    if(.not.tok) then
!!$      call  dlf_fail("No quantum transition state could be found")

    end if
    
    if(trerun_energy) then
      ! run all E&G calculations once more
      ! after dlf_qts_trans_force, icoords are set, but not neb%xcoords
      ! this loop is not parallelized yet - IMPROVE (but dimer should not be used anyway...)
      do iimage=1,neb%nimage
        cstart=neb%cstart(iimage)
        cend=neb%cend(iimage)
        call dlf_direct_itox(glob%nvar,neb%varperimage,neb%coreperimage, &
            glob%icoords(cstart:cend),neb%xcoords(:,iimage),tok)

        if(.not.tok) then
          ! This should never happen, as qTS can only be used in mass-weighted
          ! cartesians
          call dlf_fail("Error in i->x coordinate conversion after qTS")
        end if

      end do

      ! set counter to first image
      neb%iimage=1
      glob%xcoords=reshape(neb%xcoords(:,neb%iimage),(/3,glob%nat/))
      external_iimage=neb%iimage
    end if

  else !(icoord==190)

     if (glob%imicroiter < 2) then
        ! Standard NEB or macroiterative step

        ! For convergence testing when there is no climbing image,
        ! set global energy to the energy of the highest image
        glob%energy = neb%ene(1)
        do jimage=2, neb%nimage-1
           if (neb%ene(jimage) > glob%energy) glob%energy = neb%ene(jimage)
        end do
    
        call dlf_neb_improved_tangent_neb
     else
        ! Microiterative step
        ! There are no spring forces in environment region, so 
        ! simply use the true gradient set above
        ! (with frozen images set to 0 as above).
        
        ! For convergence testing in the microiterative cycles,
        ! take the average of all energy images
        glob%energy = 0.0d0
        do jimage = 1, neb%nimage
           glob%energy = glob%energy + neb%ene(jimage)
        end do
        glob%energy = glob%energy / neb%nimage
     end if

     trerun_energy=.false.

  end if
  
  ! this is not used, as no image is calculated until neb_itox is called
  external_iimage=1 

end subroutine dlf_neb_xtoi
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* neb/dlf_neb_improved_tangent_neb
!!
!! FUNCTION
!!
!! transform the forces according to the 
!! IMPROVED-TANGENT NUDGED ELASTIC BAND by
!! G. Henkelman and H. Jonsson J. Chem. Phys. 113, 9978 (2000)
!! including a climbing image as an additional image once the maximum
!! perpendicular gradient is below tolclimb
!!
!! SYNOPSIS
subroutine dlf_neb_improved_tangent_neb
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,pi,printl,printf
  use dlf_neb, only: neb,unitp,xyzall,string
  implicit none
  integer               :: cstart,cend
  integer               :: iimage,jimage,ivar
  logical,parameter     :: dbg=.false.
  real(rk)              :: emin,emax,svar,svar2
  logical               :: tclimbnext,tchk
  ! bookkeeping
  character(20)         :: str
  real(rk)              :: ftan(neb%nimage) ! tangential force
  real(rk)              :: fper(neb%nimage) ! perpendicular force
  real(rk)              :: path(neb%varperimage,neb%nimage) ! path(:,2)
                                            ! is the vector img2->img3
                                            ! including weights
  real(rk)              :: dist(neb%nimage) ! distance between images
  real(rk)              :: ang(neb%nimage)  ! angles between image-connections
  real(rk)              :: ang13(neb%nimage)! angles between further image-connections
  real(rk)              :: dwork(neb%nimage)! Delta work (grad x path) from 
                                            ! last to present image
  real(rk)              :: delta(neb%nimage)
  logical               :: tmass
  integer               :: iat
! **********************************************************************
  
  neb%step=neb%step+1

  if(neb%tfreeze) then
    neb%frozen(1:neb%nimage-1)=.false.
  end if

  ftan(:)=0.D0
  fper(:)=0.D0
  ang(:)=0.D0
  ang13(:)=0.D0
  dwork(:)=0.D0
  dist(:)=0.D0
  path(:,:)=0.D0

  ! ====================================================================
  ! Calculate distances and angles
  ! ====================================================================
  ! path, dist, and work
  do iimage=1,neb%nimage-1

    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage) 

    if(iimage<neb%nimage-1) then
      path(:,iimage)= (glob%icoords(cstart+neb%varperimage:cend+neb%varperimage)- &
          glob%icoords(cstart:cend)) * glob%iweight(1:neb%varperimage)
      dist(iimage)=dsqrt(sum(path(:,iimage)**2))
      dwork(iimage+1)=sum( (neb%gradt(cstart:cend)+ &
            neb%gradt(cstart+neb%varperimage:cend+neb%varperimage)) * &
          path(:,iimage) ) * 0.5D0
    else
      dist(iimage)=0.D0
    end if
  end do

  !angles
  do iimage=2,neb%nimage-2
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage) 
    svar=sum(path(:,iimage-1) * path(:,iimage)) / (dist(iimage-1) * &
        dist(iimage))
    if(svar>=1.D0) then
      ang(iimage)=0.D0
    else
      ang(iimage)=acos(svar)
    end if
    
    if(iimage<neb%nimage-2) then
      svar=sum(path(:,iimage-1) * path(:,iimage+1)) / (dist(iimage-1) * &
          dist(iimage+1))
      if(svar>=1.D0) then
        ang13(iimage)=0.D0
      else
        ang13(iimage)=acos(svar)
      end if
    end if
  end do

  ! ==================================================================
  ! Calculate the tangent vector tau
  ! ==================================================================
  if(1==2.and.string) then
    call string_get_tau
  else
  do iimage=1,neb%nimage-1
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage) 

    if(iimage==1) then
      ! Tau+
      neb%tau(cstart:cend)=path(:,iimage)
    else if(iimage==neb%nimage-1) then
      ! Tau-
      neb%tau(cstart:cend)=path(:,iimage-1)
    else
      if(neb%ene(iimage+1)>neb%ene(iimage) .and. &
          neb%ene(iimage)>neb%ene(iimage-1)) then
        ! Tau+
        neb%tau(cstart:cend)=path(:,iimage)
        if(dbg) print*,"Image",iimage," Tau+"
      else if(neb%ene(iimage+1)<neb%ene(iimage) .and. &
          neb%ene(iimage)<neb%ene(iimage-1)) then
        ! Tau-
        neb%tau(cstart:cend)=path(:,iimage-1)
        if(dbg) print*,"Image",iimage," Tau-"
      else
        
        if(dbg) print*,"Image",iimage," Tau+-"
        ! iimage is either a minimum or a maximum
        emin=min(abs(neb%ene(iimage+1)-neb%ene(iimage)), &
            abs(neb%ene(iimage-1)-neb%ene(iimage)))
        emax=max(abs(neb%ene(iimage+1)-neb%ene(iimage)), &
            abs(neb%ene(iimage-1)-neb%ene(iimage)))
        if(neb%ene(iimage+1)<neb%ene(iimage-1)) then
          ! interchange them
          svar=emin
          emin=emax
          emax=svar
        end if

        ! catch the erratic case emin=emax=0
        if(emin==emax.and.emin==0.D0) then
          emin=1.D0
          emax=1.D0
        end if

        neb%tau(cstart:cend)=emax*path(:,iimage) + emin*path(:,iimage-1)

      end if
    end if
    ! improve - blas
    neb%tau(cstart:cend)=neb%tau(cstart:cend)/sqrt(sum(neb%tau(cstart:cend)**2))
  end do
  end if

  ! ====================================================================
  ! Calculate tangent vector of the climbing image
  ! ====================================================================
  if(neb%maximage>0) then
    cstart=neb%cstart(neb%nimage)
    cend=neb%cend(neb%nimage) 
    ! find image that is closest to the climbing image
    ivar=-1    ! in the end: number of closest image
    jimage=-1  ! in the end: number of 2nd-closest image
    emax=1.D90 ! in the end: distance CI -- closest
    svar=1.D90 ! in the end: distance CI -- 2nd-closest
    do iimage=1,neb%nimage-1
      ! should this be weighted ??
      svar2=sum((glob%icoords(cstart:cend)-glob%icoords( &
          neb%cstart(iimage):neb%cend(iimage)))**2)
      if(svar2<emax) then
        ! 2nd closest
        jimage=ivar
        svar=emax
        ! closest
        ivar=iimage
        emax=svar2
      else if (svar2<svar) then
        ! 2nd closest
        jimage=iimage
        svar=svar2        
      end if
    end do
    neb%maximage=ivar

    ! take weighted average of the two tau (weight is the distance of the other image)
    neb%tau(cstart:cend)= &
        svar*neb%tau(neb%cstart(ivar):neb%cend(ivar)) + &
        emax*neb%tau(neb%cstart(jimage):neb%cend(jimage))
    neb%tau(cstart:cend)=neb%tau(cstart:cend)/sqrt(sum(neb%tau(cstart:cend)**2))

    dist(neb%nimage)=dsqrt(emax)

    if(printl>=2) then
      write(stdout,'("Climbing image in cycle ",i7," is closest to &
          &image ",i4," (next: ",i4,")")') &
          neb%step,neb%maximage,jimage
    end if
    if(printl>=4) write(stdout,'("Distance to 2nd closest image is",f10.5)') sqrt(svar)

  else

    if(printl>=2) then
      write(stdout,'("No climbing image used in cycle ",i7)') neb%step
    end if

  end if

  ! ====================================================================
  ! Calculate the Forces
  ! ====================================================================
  tclimbnext=.true.
  do iimage=2, neb%nimage-2

    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage) 

    ! make sure frozen images can be unfrozen if required
    glob%igradient(cstart:cend)=neb%gradt(cstart:cend)

    ! ================================================================
    ! project out the parallel component of the true force
    ! ================================================================
    ! improve - blas
    if(dbg) print*,"Image",iimage,"F_t||=",sum(glob%igradient(cstart:cend)*neb%tau(cstart:cend))
    glob%igradient(cstart:cend)=glob%igradient(cstart:cend)- &
        sum(glob%igradient(cstart:cend)*neb%tau(cstart:cend))*neb%tau(cstart:cend)
    fper(iimage)=dsqrt(sum((glob%igradient(cstart:cend))**2)/dble(neb%varperimage))

    if((.not.neb%tclimb).and.maxval(abs(glob%igradient(cstart:cend))) > &
        neb%tolclimb) then
      tclimbnext=.false.
    end if

    ! ==================================================================
    ! decide on freezing
    ! ==================================================================
    ! this is now done on the unweighted gradient. Is this a good idea?
    if(neb%tfreeze.and.maxval(abs(glob%igradient(cstart:cend)))<=neb%tolfreeze) then
      neb%frozen(iimage)=.true.
      glob%igradient(cstart:cend)=0.D0
    else

      ! ================================================================
      ! add the parallel component of the spring force
      ! ================================================================

      if(.not.string) then
        glob%igradient(cstart:cend)=glob%igradient(cstart:cend)+ &
            neb%k*(dist(iimage-1)-dist(iimage))*neb%tau(cstart:cend)
      end if

    end if
    ftan(iimage)=neb%k*(dist(iimage-1)-dist(iimage))
    if(string) ftan(iimage)=0.D0

  end do
  
  ! ==================================================================
  ! string method: add step(s) for L-BFGS to learn
  ! ==================================================================
  !delta=0.00001D0
!!$  call random_number(delta)
!!$  delta=(delta-0.5D0)*0.01D0
!!$  print*,"delta",delta
!!$  if(string.and.1==1) then
!!$     do iimage=2, neb%nimage-2
!!$        cstart=neb%cstart(iimage)
!!$        cend=neb%cend(iimage) 
!!$        ! move this image
!!$        glob%icoords(cstart:cend)=glob%icoords(cstart:cend)    +delta*neb%tau(cstart:cend)
!!$        glob%igradient(cstart:cend)=glob%igradient(cstart:cend)+delta*2.D0*neb%k*neb%tau(cstart:cend)
!!$        jimage=iimage-1
!!$        cstart=neb%cstart(jimage)
!!$        cend=neb%cend(jimage) 
!!$        glob%igradient(cstart:cend)=glob%igradient(cstart:cend)-delta*neb%k*neb%tau(cstart:cend)
!!$        jimage=iimage+1
!!$        cstart=neb%cstart(jimage)
!!$        cend=neb%cend(jimage) 
!!$        glob%igradient(cstart:cend)=glob%igradient(cstart:cend)-delta*neb%k*neb%tau(cstart:cend)
!!$     end do
!!$     call dlf_lbfgs_step(glob%icoord,glob%igradient,glob%step)
!!$     do iimage=2, neb%nimage-2
!!$        cstart=neb%cstart(iimage)
!!$        cend=neb%cend(iimage) 
!!$        ! move this image back
!!$        glob%icoords(cstart:cend)=glob%icoords(cstart:cend)    -delta*neb%tau(cstart:cend)
!!$        glob%igradient(cstart:cend)=glob%igradient(cstart:cend)-delta*2.D0*neb%k*neb%tau(cstart:cend)
!!$        jimage=iimage-1
!!$        cstart=neb%cstart(jimage)
!!$        cend=neb%cend(jimage) 
!!$        glob%igradient(cstart:cend)=glob%igradient(cstart:cend)+delta*neb%k*neb%tau(cstart:cend)
!!$        jimage=iimage+1
!!$        cstart=neb%cstart(jimage)
!!$        cend=neb%cend(jimage) 
!!$        glob%igradient(cstart:cend)=glob%igradient(cstart:cend)+delta*neb%k*neb%tau(cstart:cend)
!!$     end do
!!$  end if

  ! spawn a climbing image next cycle if all gradients are below neb%tolclimb
  if(tclimbnext) neb%tclimb=.true.

  if(neb%maximage>0) then
    ! ==================================================================
    ! climbing image
    ! ==================================================================
    iimage=neb%nimage
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage) 

    ! send information on energy, gradient, and step to convergence tester
    call convergence_set_info("of NEB climbing image",neb%varperimage,&
        neb%ene(iimage),glob%igradient(cstart:cend),glob%step(cstart:cend))
    
    
    ! send information to set_tsmode
    call dlf_formstep_set_tsmode(1,-1,neb%ene(iimage)) ! send energy
    call dlf_formstep_set_tsmode(glob%nvar,0,neb%xcoords(:,iimage)) ! TS-geometry
    call dlf_formstep_set_tsmode(neb%varperimage,11,neb%tau(cstart:cend)) ! TS-mode
    
    ! modify gradient
    ! improve - blas
    ftan(iimage)=sum(glob%igradient(cstart:cend)*neb%tau(cstart:cend))
    glob%igradient(cstart:cend)=glob%igradient(cstart:cend) - &
        2.D0 * ftan(iimage) * neb%tau(cstart:cend)
    fper(iimage)=sqrt(sum(( glob%igradient(cstart:cend)- &
        1.D0*ftan(iimage)*neb%tau(cstart:cend))**2)/dble(neb%varperimage))
  else
     ! No climbing image, so set approximate tsmode info using the highest
     ! energy image.
     ! Note the convergence info will be set in the main routine as usual
     ! using information from ALL images
     iimage = 1
     do jimage = 2, neb%nimage - 1
        if (neb%ene(jimage) > neb%ene(iimage)) iimage = jimage
     end do
     cstart = neb%cstart(iimage)
     cend = neb%cend(iimage)
     call dlf_formstep_set_tsmode(1,-1,neb%ene(iimage)) ! send energy
     call dlf_formstep_set_tsmode(glob%nvar,0,neb%xcoords(:,iimage)) ! TS-geometry
     call dlf_formstep_set_tsmode(neb%varperimage,11,neb%tau(cstart:cend)) ! TS-mode     
  end if

  ! ====================================================================
  ! check first and last image
  ! ====================================================================
  if(neb%mode==2) then
    if(.not.neb%tfreeze) call dlf_fail("NEB: mode=2 and tfreeze=false is invalid")
    neb%frozen(1)=.true.
    neb%frozen(neb%nimage-1)=.true.
    glob%igradient(neb%cstart(1):neb%cend(1))=0.D0
    glob%igradient(neb%cstart(neb%nimage-1):neb%cend(neb%nimage-1))=0.D0
  else 
    if (neb%mode==1) then
      ! first image
      cstart=neb%cstart(1)
      cend=neb%cend(1) 
      glob%igradient(cstart:cend)=glob%igradient(cstart:cend)- &
          sum(glob%igradient(cstart:cend)*neb%tau(cstart:cend))*neb%tau(cstart:cend)
      fper(1)=sqrt(sum(glob%igradient(cstart:cend)**2)/dble(neb%varperimage))

      ! last image
      cstart=neb%cstart(neb%nimage-1)
      cend=neb%cend(neb%nimage-1) 
      glob%igradient(cstart:cend)=glob%igradient(cstart:cend)- &
          sum(glob%igradient(cstart:cend)*neb%tau(cstart:cend))*neb%tau(cstart:cend)
      fper(neb%nimage-1)=sqrt(sum(glob%igradient(cstart:cend)**2)/dble(neb%varperimage))
    end if
    if(neb%mode==0) then
      cstart=neb%cstart(1)
      cend=neb%cend(1) 
      fper(1)=sqrt(sum(glob%igradient(cstart:cend)**2)/dble(neb%varperimage))
      cstart=neb%cstart(neb%nimage-1 )
      cend=neb%cend(neb%nimage-1 ) 
      fper(neb%nimage-1 )=sqrt(sum(glob%igradient(cstart:cend)**2)/dble(neb%varperimage))
    end if
    if(neb%tfreeze) then
      iimage=1
      cstart=neb%cstart(iimage)
      cend=neb%cend(iimage)
      if(maxval(abs(glob%igradient(cstart:cend)))<=neb%tolfreeze) then
        neb%frozen(iimage)=.true.
        glob%igradient(cstart:cend)=0.D0
      end if
      iimage=neb%nimage-1
      cstart=neb%cstart(iimage)
      cend=neb%cend(iimage)
      if(maxval(abs(glob%igradient(cstart:cend)))<=neb%tolfreeze) then
        neb%frozen(iimage)=.true.
        glob%igradient(cstart:cend)=0.D0
      end if
    end if
  end if

  ! ====================================================================
  ! check if all images (but the climbing image) are frozen
  ! ====================================================================
  if(neb%allfrozen) then
    cstart=neb%cstart(1)
    cend=neb%cend(neb%nimage-1)
    ! set complete gradient but frozen image to 0
    glob%igradient(cstart:cend)=0.D0 
  else
    tchk=.true.
    do iimage=1,neb%nimage-1
      if(.not.neb%frozen(iimage)) tchk=.false.
    end do
    if(tchk) then
      ! first time that all are frozen
      neb%allfrozen=.true.

      if(neb%frozen(neb%nimage)) then

        if(printl>=2) then
          write(stdout,"('All images are frozen and no climbing image &
              &exists.')")
          write(stdout,"('Either the climbing image threshold &
              &was not reached or the path is monotonic.')")
          write(stdout,"('Maximum gradient component converged to the frozen &
              &image tolerance of ',es10.4)") neb%tolfreeze
        end if

        ! set complete gradient to 0
        glob%igradient(:)=0.D0 
        glob%step(:)=0.D0 

        ! send information on energy, gradient, and step to convergence tester - 
        ! all set to zero
        call convergence_set_info("",neb%varperimage,&
            glob%oldenergy_conv,glob%igradient(cstart:cend),glob%step(cstart:cend))

      else
        if(printl>=2) write(stdout,"('All images but the climbing image are frozen')")
        
        if(neb%maximage<0) call dlf_fail("All images frozen and no climbing image present")
      
        cstart=neb%cstart(1)
        cend=neb%cend(neb%nimage-1)
        ! set complete gradient but frozen image to 0
        glob%igradient(cstart:cend)=0.D0 
        
        call dlf_formstep_restart
      end if
    end if
      
  end if

  ! ====================================================================
  ! write some report
  ! ====================================================================
  if(printl>=4) then
    write(stdout,"('NEB Report')")
    write(stdout,"('             Energy       F_tang    F_perp     Dist&
        &     Angle 1-3 Ang 1-2 Sum ')")
    do iimage=1,neb%nimage-1
      str=""
      !if(iimage<neb%nimage-2.and.iimage>1.and.&
      !    ang(iimage)+ang(iimage+1)>ang13(iimage)*1.2D0) &
      !    str="!"
      if(neb%frozen(iimage)) str=trim(str)//" frozen"
      !if(iimage==neb%maximage) str=trim(str)//" Climbing image"
      write(stdout,"('Img ',i4,f15.7,3f10.5,3f8.2,2x,a)") &
          iimage,neb%ene(iimage),ftan(iimage),fper(iimage),dist(iimage), &
          ang(iimage)*180.D0/pi,ang13(iimage)*180.D0/pi,&
          (ang(iimage)+ang(min(iimage+1,neb%nimage-1)))*180.D0/pi,str
    end do
    if(neb%maximage>0) then
      iimage=neb%nimage
      write(stdout,"('Cimg',i4,f15.7,3f10.5,3f8.2)") &
          neb%maximage,neb%ene(iimage),ftan(iimage),fper(iimage),dist(iimage), &
          ang(iimage)*180.D0/pi,ang13(iimage)*180.D0/pi,&
          (ang(iimage)+ang(min(iimage+1,neb%nimage-1)))*180.D0/pi
    end if
    write(stdout,"('Total path length ',f10.5)") sum(dist(1:neb%nimage-2))
  end if

  ! write information files
  if(printl>=2.and.printf>=2) then
    tmass=(neb%varperimage==glob%nat*3)
    ! list of energies (and work?)
    if (glob%iam == 0) then
      open(unit=501,file="nebinfo")
      if(tmass) then
        write(501,"('# Path length     Energy      Work       Effective Mass')")
      else
        write(501,"('# Path length     Energy      Work')")
      end if
    end if
    do iimage=1,neb%nimage-1
      if(iimage==1) then
        svar=0.D0
      else
        svar=sum(dist(1:iimage-1))
      end if
      if (glob%iam == 0) then
        if(tmass) then
          ! neb%tau is normalized tangent
          svar2=0.D0 ! reduced mass
          do iat=1,glob%nat
            svar2=svar2+sum((neb%tau(neb%cstart(iimage)+(iat-1)*3:&
                neb%cstart(iimage)+(iat-1)*3+2))**2)/glob%mass(iat)
          end do
          svar2=1.D0/svar2
          write(501,"(f10.5,3f15.10)") svar,neb%ene(iimage)-neb%ene(1),sum(dwork(1:iimage)),svar2
        else
          write(501,"(f10.5,2f15.10)") svar,neb%ene(iimage)-neb%ene(1),sum(dwork(1:iimage))
        end if
      end if
    end do
    if (glob%iam == 0) close(501)
    
    ! nebpath.xyz
    if (glob%iam == 0) open(unit=501,file="nebpath.xyz")
    do iimage=1,neb%nimage-1
      ! here, always write the whole system
      !if(xyzall) then
      if (glob%iam == 0) call write_xyz(501,glob%nat,glob%znuc,&
                                        neb%xcoords(:,iimage))
      !else
      !  call write_xyz_active(501,glob%nat,glob%znuc,glob%spec,neb%xcoords(:,iimage))
      !end if
      ! write chemshell fragments as well (or any other format used by dlf_put_coords)
      svar=neb%ene(iimage)
      ! The commented out line ensured that the final energy given to chemshell
      ! was that of the climbing image. A better solution is not to save the energy
      ! in put_coords if the mode < 0
      ! if(neb%maximage>0) svar=neb%ene(neb%nimage)
      call dlf_put_coords(glob%nat,-iimage,svar,neb%xcoords(:,iimage),glob%iam)
    end do
    if (glob%iam == 0) close(501)
  end if

end subroutine dlf_neb_improved_tangent_neb
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* neb/dlf_neb_checkstep
!!
!! FUNCTION
!!
!! Set step of frozen images to zero
!!
!! In case of large angles between the images, it sets the images back 
!! to a line connecting the neighbouring images.
!!
!! SYNOPSIS
subroutine dlf_neb_checkstep
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,pi
  use dlf_neb, only: neb,string
  implicit none
  integer               :: iimage,jimage,mimage
  real(rk)              :: svar,angle12(neb%nimage),angle13(neb%nimage)
  logical               :: treduce(neb%nimage)
! **********************************************************************
  !if(.not.neb%tfreeze) return
  do iimage=1,neb%nimage
    if(neb%frozen(iimage)) then
      glob%step(neb%cstart(iimage):neb%cend(iimage))=0.D0
    end if
  end do
  
  if(glob%icoord==190) return ! do nothing in case of qTS search

  if (glob%imicroiter == 2) return ! do not correct paths during microiterations

  ! calculate the angles:
  angle12(:)=0.D0
  angle13(:)=0.D0
  do iimage=2,neb%nimage-2
    call angle(iimage-1,iimage,iimage,iimage+1,angle12(iimage))
    if(iimage<neb%nimage-2) &
        call angle(iimage-1,iimage,iimage+1,iimage+2,angle13(iimage))
  end do
  treduce(:)=.false.
  do iimage=2,neb%nimage-2
    ! do not consider angles of images next to frozen endpoints
    if(neb%mode==2.and.iimage==2) cycle
    if(neb%mode==2.and.iimage==neb%nimage-2) cycle

    if(angle12(iimage) > 90.D0/180.D0*pi ) treduce(iimage)=.true.
  end do

!  if(.not.string) then ! it turned out not to be a good idea to reduce the images in string ...
  do iimage=2,neb%nimage-2
    if(.not.treduce(iimage)) cycle
    if(iimage==neb%maximage) cycle
    PRINT*,"-------- Path corrections -----------"
    do jimage=iimage+1,neb%nimage-2
      if(.not.treduce(jimage)) exit
      if(jimage==neb%maximage) exit
      treduce(jimage)=.false. ! will be covered now
    end do
    print*,"Resetting images",iimage," to ",jimage-1
    ! This does not appear to cause problems for microiterative opts
    ! (at least, no more than the problems it causes for standard opts)
    ! but keeping an eye on it...
    do mimage=iimage,jimage-1
      svar=dble(mimage - (iimage-1))/dble(jimage - (iimage-1))
      glob%step(neb%cstart(mimage):neb%cend(mimage))= &
          (1.D0-svar) * (&
          glob%icoords(neb%cstart(iimage-1):neb%cend(iimage-1)) + &
          glob%step(neb%cstart(iimage-1):neb%cend(iimage-1)) ) + &
          svar * ( &
          glob%icoords(neb%cstart(jimage):neb%cend(jimage)) + &
          glob%step(neb%cstart(jimage):neb%cend(jimage))) &
          -glob%icoords(neb%cstart(mimage):neb%cend(mimage))
    end do
  END do
!  end if
  
  treduce(:)=.false.
  if(string.and.(.not.neb%allfrozen)) call string_reparametrise(treduce)

contains 

  subroutine angle(img1,img2,img3,img4,ang)
    implicit none
    integer,intent(in)     :: img1,img2,img3,img4
    real(rk),intent(out)   :: ang
    real(rk)               :: svar
    integer                :: img,iarr(4)
    real(rk)               :: path(neb%varperimage,2)
  ! ********************************************************************
    iarr(:)=(/img1,img2,img3,img4/)
    do img=2,4,2
      path(:,img/2)=(glob%icoords(neb%cstart(iarr(img)):neb%cend(iarr(img))) + &
                     glob%step   (neb%cstart(iarr(img)):neb%cend(iarr(img))) - &
                     glob%icoords(neb%cstart(iarr(img-1)):neb%cend(iarr(img-1))) - &
                     glob%step   (neb%cstart(iarr(img-1)):neb%cend(iarr(img-1))) ) * &
                     glob%iweight(1:neb%varperimage)
    end do
    svar=sum(path(:,1)*path(:,2))/dsqrt(sum(path(:,1)**2)) &
        / dsqrt(sum(path(:,2)**2))
    if(svar>=1.D0) then
      ang=0.D0
    else
      ang=acos(svar)
    end if
  end subroutine angle

end subroutine dlf_neb_checkstep
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* neb/dlf_neb_itox
!!
!! FUNCTION
!!
!! transform NEB array to Cartesian coordinates
!!
!! All images will be transformed to xyz coordinates here. If 
!! unsuccessful the algorithm will fall back to Cartesian coordinates.
!! This however, will end the usage of constraints!
!!
!! Initiate the climbing image if required
!!
!! SYNOPSIS
subroutine dlf_neb_itox(external_iimage)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printf,printl
  use dlf_allocate, only: allocate,deallocate
  use dlf_neb, only: neb,unitp,xyzall,maxxyzfile
  implicit none
  integer ,intent(out)  :: external_iimage
  integer               :: cstart,cend,iimage,mx(1),ivar
  logical               :: tok,tfail
  real(rk), allocatable :: xtmp(:,:)
  integer , allocatable :: useimage(:)
  real(rk)              :: svar,svar2
! **********************************************************************

  ! ====================================================================
  ! Determine the initial position of the climbing image
  ! ====================================================================
  if(neb%tclimb .and. neb%frozen(neb%nimage) .and. glob%imicroiter < 2) then
    ! initialise climbing image
    mx=maxloc(neb%ene(2:neb%nimage-2))
    ivar=mx(1)+1 ! maximum image

    cstart=neb%cstart(neb%nimage)
    cend=neb%cend(neb%nimage) 

    ! make sure no climbing image is spawned if there is no maximum on the path
    if(ivar==2 .and. neb%ene(1)>neb%ene(ivar)) neb%tclimb=.false.
    if(ivar==neb%nimage-2 .and. neb%ene(neb%nimage-1)>neb%ene(ivar)) neb%tclimb=.false.

    if(neb%tclimb) then

      if(printl>=4) write(stdout,'("Spawning climbing image")')

      !Comment: The climbing image is set according to the energies before the 
      !         step but the geometry after the step. This may lead to 
      !         difficulties if the step is large.
      
      ! interpolate maximum from nearby 3 images
      svar=(-(neb%ene(ivar-1)-neb%ene(ivar))-(neb%ene(ivar+1)-neb%ene(ivar)))/(-2.D0)
      if(svar>0.d0) then
        ! the parabola through the three points has a minimum, no maximum.
        ! In this case, use the geometry of ivar as initial climbing image
        glob%icoords(cstart:cend)=glob%icoords(neb%cstart(ivar):neb%cend(ivar))
        glob%xcoords(:,neb%nimage)=glob%xcoords(:,ivar)
        if(printl>=4) write(stdout,'("Initial climbing image is identical&
            & to image",i4)') ivar
      else
        svar2=(neb%ene(ivar+1)-neb%ene(ivar-1))/2.D0 - 2.D0*svar
        svar=-svar2/(2.D0*svar)
        
        if(svar>1.D0) then
          ! interpolate between images ivar and ivar+1
          glob%icoords(cstart:cend)= &
              (2.D0-svar) * glob%icoords(neb%cstart(ivar):neb%cend(ivar)) + &
              (svar-1.D0) * glob%icoords(neb%cstart(ivar+1):neb%cend(ivar+1))
          neb%xcoords(:,neb%nimage)= &
              (2.D0-svar) * neb%xcoords(:,ivar) + &
              (svar-1.D0) * neb%xcoords(:,ivar+1)
          if(printl>=4) write(stdout,'("Initial climbing image is ",f5.1,"% of&
              & image ",i4," and ",f5.1,"% of image ",i4)') (2.D0-svar)*100.D0,&
              ivar,(svar-1.D0)*100.D0,ivar+1
        else
          ! interpolate between images ivar-1 and ivar
          glob%icoords(cstart:cend)= &
              (1.D0-svar) * glob%icoords(neb%cstart(ivar-1):neb%cend(ivar-1)) + &
              svar * glob%icoords(neb%cstart(ivar):neb%cend(ivar))
          neb%xcoords(:,neb%nimage)= &
              (1.D0-svar) * neb%xcoords(:,ivar-1) + &
              svar * neb%xcoords(:,ivar)
          if(printl>=4) write(stdout,'("Initial climbing image is ",f5.1,"% of&
              & image ",i4," and ",f5.1,"% of image ",i4)') (1.D0-svar)*100.D0,&
              ivar-1,svar*100.D0,ivar
        end if
      end if ! (svar>0.d0)
      call dlf_formstep_restart
      neb%frozen(neb%nimage)=.false.
      neb%maximage=ivar
    else
      if(printl>=4) write(stdout,'("No climbing image spawned because the energy &
          &maximum is an endpoint of the path")')
    end if !(neb%tclimb) 
  end if ! (neb%tclimb .and. neb%frozen(neb%nimage))
  if(neb%iimage/=neb%nimage) print*,"Warning, NEB images have been tampered with!"

  if (glob%tatoms) then
     call allocate( xtmp,3*glob%nat,neb%nimage)
  else
     call allocate( xtmp,glob%nvar,neb%nimage)
  end if
  tfail=.false.
  xtmp(:,:)=neb%xcoords(:,:) ! starting guess and frozen structures
  do iimage=1,neb%nimage
    if(neb%tfreeze) then
      if(neb%frozen(iimage)) cycle
    end if
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage)
    call dlf_direct_itox(glob%nvar,neb%varperimage,neb%coreperimage, &
      glob%icoords(cstart:cend),xtmp(:,iimage),tok)
    if(.not.tok) then
      tfail=.true.
      exit
    end if
  end do
  if(tfail) then
    ! reset and restart
    call allocate(useimage,neb%nimage)
    useimage(:)=1
    if(printl>=2) write(stdout, &
        "('HDLC coordinate breakdown. Recalculating HDLCs and &
        &restarting optimiser and NEB.')")
    call dlf_hdlc_reset
    call dlf_hdlc_create(glob%nat,neb%coreperimage,glob%spec,glob%micspec,&
        glob%znuc,neb%nimage,neb%xcoords,glob%weight,glob%mass)

    ! calculate iweights
    ! first image
    call dlf_hdlc_getweight(glob%nat,glob%nivar,neb%coreperimage,glob%micspec,&
        glob%weight,glob%iweight(1:neb%varperimage))
    ! copy weights to all other images
    do iimage=2,neb%nimage
      ivar=neb%varperimage*(iimage-1)+1
      glob%iweight(ivar:ivar+neb%varperimage-1)=glob%iweight(1:neb%varperimage)
    end do

    call dlf_formstep_restart

    ! recalculate NEB images
    call dlf_neb_definepath(neb%nimage,useimage)    

    call deallocate(useimage)
  else
    ! all coordinate conversion were successful
    neb%xcoords(:,:)=xtmp
  end if
  call deallocate(xtmp)

  ! now all images are present in xcoords (neb%xcoords)

  ! find first non-frozen image to calculate the energy on
  if(neb%tfreeze) then
    do iimage=1,neb%nimage
      if(neb%frozen(iimage)) then
        cstart=neb%cstart(iimage)
        cend=neb%cend(iimage)
        glob%igradient(cstart:cend)=0.D0

        ! Parallel NEB: this is to avoid multiple counting at the end of xtoi
        if (glob%mytask /= 0) then
           glob%icoords(cstart:cend) = 0.d0
           neb%ene(iimage) = 0.d0
           neb%gradt(cstart:cend) = 0.d0           
        end if
        
        ! write xyz file
        if(printf>=4.and.glob%tatoms.and.neb%iimage<=maxxyzfile .and. glob%iam == 0) then
          if(xyzall) then
            call write_xyz(unitp+iimage,glob%nat,glob%znuc,neb%xcoords(:,iimage))
          else
            call write_xyz_active(unitp+iimage,glob%nat,glob%znuc,glob%spec,neb%xcoords(:,iimage))
          end if
        end if
      else
        neb%iimage=iimage
        exit
      end if
    end do
  else
    neb%iimage=1
  end if

  external_iimage=neb%iimage
  if(glob%tatoms) then
     glob%xcoords=reshape(neb%xcoords(:,neb%iimage),(/3,glob%nat/))
  else
     glob%xcoords=reshape(neb%xcoords(:,neb%iimage),(/1,glob%nvar/))
  end if

end subroutine dlf_neb_itox
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* neb/dlf_neb_definepath
!!
!! FUNCTION
!!
!! Define a NEB path in internal coordinates using input guesses or old
!! structures in x-coordinates
!!
!! useimage(iimage)=0 -> this image is interpolated, x-information is 
!! not used
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
subroutine dlf_neb_definepath(nimage,useimage)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_allocate, only: allocate,deallocate
  use dlf_neb, only: neb,unitp
  implicit none
  integer ,intent(in)   :: nimage
  integer ,intent(inout):: useimage(nimage)
  integer               :: xok(nimage)
  integer               :: iimage,jimage,kimage,cstart,cend,lowok,ivar
  integer               :: maxuse,iat,nuse,map(nimage)
  real(rk)              :: svar,dist(nimage),lastdist
  logical               :: tok,tchk
! **********************************************************************
  glob%igradient=0.D0
  glob%icoords=0.D0
  dist(:)=0.D0 ! dist is a cumulative distance from the first image
  maxuse=1
  nuse=0 ! number of images to use to construct the path

  ! get internal coordinates of well-defined images
  lastdist=0.D0
  do iimage=1,nimage
    if(useimage(iimage)==0) cycle
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage)
    !print*," ((((((((((((((((((((( IMAGE",iimage," )))))))))))))))))))))))))))"
    call dlf_direct_xtoi(glob%nvar,neb%varperimage,neb%coreperimage,neb%xcoords(:,iimage), &
        glob%xgradient,glob%icoords(cstart:cend),glob%igradient(cstart:cend))
    ! distance to last used image
    if(iimage>1) then
      svar=sqrt(sum(( &
          glob%iweight(1:neb%varperimage)*( &
          glob%icoords(cstart:cend) - glob%icoords(neb%cstart(maxuse):neb%cend(maxuse))) &
          ) **2 ) ) 
      !svar=svar+sum(dist)
      dist(iimage)=svar+lastdist
      lastdist=dist(iimage)
    end if
    maxuse=iimage
    if(maxuse<nimage) nuse=nuse+1
  end do

  ! make sure only images 1 to nimage-1 are used to interpolate
  if(maxuse>=nimage) maxuse=nimage-1

  ! if all images are well-defined, return
  if(minval(useimage)==1) return

  if(dist(maxuse)<1.D-3) call dlf_fail("Start and endpoint too close in NEB")

  ! print alignment of images along the distance
  if(printl>=4) then
    write(stdout,'("Constructing initial NEB path from input images")')
    write(stdout,'("Length of the initial path: ",f10.5)') dist(maxuse)
    write(stdout,'("Using ",i3," input images to construct a path of ",&
        &i3," total images")') nuse,nimage-1
    lastdist=0.D0
    do iimage=1,maxuse
      if(useimage(iimage)==0) cycle
      write(stdout,'("Image ",i3," along path: ",f10.5," dist to prev. &
          &image:",f10.5)') iimage,dist(iimage),dist(iimage)-lastdist
      lastdist=dist(iimage)
    end do
  end if

  ! if all requested images are provided by the user, do not interpolate
  if(nuse/=nimage-1) then
    ! distribute defined images as equally as possible according to dist
    map(:)=0 ! map(iimage) is the number, iimage should have after distribution
    do iimage=maxuse,2,-1

      if(useimage(iimage)==0) cycle

      svar=1.D0+dist(iimage)/dist(maxuse)*dble(nimage-2)
      map(iimage)=nint(svar)
      !print*,"Image",iimage,"svar",svar,"map(iimage)",map(iimage)
      if(map(iimage)>=nimage) then
        call dlf_fail("Problem with image distribution in NEB.")
      end if
    end do

    ! assign images
    do iimage=maxuse,2,-1

      tok=.true.
      do jimage=maxuse,2,-1
        if(jimage==iimage) cycle
        if(map(jimage)==map(iimage)) tok=.false.
      end do
      if(.not.tok) then
        ! see if map(iimage)=iimage would be OK
        tok=.true.
        do jimage=maxuse,2,-1
          if(map(jimage)==iimage) tok=.false.
        end do
        if(tok) map(iimage)=iimage
      end if
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

      ! move image
      ivar=map(iimage)
      ! move icoords
      glob%icoords(neb%cstart(ivar):neb%cend(ivar))= &
          glob%icoords(neb%cstart(iimage):neb%cend(iimage))
      ! move xcoords
      neb%xcoords(:,ivar)=neb%xcoords(:,iimage)
      ! move use info
      useimage(ivar)=1
      useimage(iimage)=0

    end do

  end if

  if(printl>=4) then
    do iimage=1,nimage-1
      if(useimage(iimage)==1) then
        write(stdout,'("Image",i3," is obtained from input")') iimage
      else
        write(stdout,'("Image",i3," is interpolated")') iimage
      end if
    end do
  end if

  ! linear transit between defined images
  if(useimage(1)==0) call dlf_fail("First image undefined in NEB")
  if(useimage(nimage-1)==0) call dlf_fail("Last image undefined in NEB")
  do iimage=1,nimage-1
    if(useimage(iimage)==0) cycle
    ! iimage is defined
    do jimage=iimage+1,nimage-1
      if(useimage(jimage)/=0) exit
    end do
    if(jimage==iimage+1) cycle
    ! interpolate from iimage to jimage
    do kimage=iimage+1,jimage-1
      cstart=neb%cstart(kimage)
      cend=neb%cend(kimage)
      svar=dble(kimage-iimage)/dble(jimage-iimage) ! 0..1
      glob%icoords(cstart:cend)= &
          (1.D0-svar)*glob%icoords( neb%cstart(iimage) : neb%cend(iimage) ) + &
          svar*glob%icoords( neb%cstart(jimage) : neb%cend(jimage) )
    end do
  end do
  ! now all icoords are defined

  ! find consistent xcoords
  xok=0
  xok(1)=1
  xok(nimage-1)=1
  !do iimage=2,nimage-2
  iimage=1
  do while (iimage<nimage-2)
    iimage=iimage+1
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage)
    ! try to convert internals back to Cartesians. These are rather large steps,
    ! thus the DLC converter may have difficulties. Catch them!
    ! First, guess the Cartesian coordinates from the previous image and 
    ! the next defined (used) image
    if(useimage(iimage)==0) then
      ! find the next "used" image
      do jimage=iimage+1,nimage-1
        if(useimage(jimage)/=0) exit
      end do
      svar=1.D0/dble(jimage-iimage+1)
      neb%xcoords(:,iimage)=(1.D0-svar)*neb%xcoords(:,iimage-1) &
          + svar*neb%xcoords(:,jimage)
    end if
    !print*," ((((((((((((((((((((( IMAGE",iimage," i->x )))))))))))))))))))))))))))"
    call dlf_direct_itox(glob%nvar,neb%varperimage,neb%coreperimage, &
        glob%icoords(cstart:cend),neb%xcoords(:,iimage),tok)
    if(tok) then
      xok(iimage)=1
    else
      !print*,"Failed at image ",iimage
      ! jump to next defined image
      iimage=jimage
    end if
  end do
  !print*,"JK Converted:",xok
  !print*,"iimage",iimage

  if(minval(xok(1:nimage-1))<1) then
    ! at least one image could not be converted. Try from last image back
!    lowok=iimage-1
    do iimage=nimage-2,2,-1
      if(xok(iimage)==1) cycle
      cstart=neb%cstart(iimage)
      cend=neb%cend(iimage)
      ! find the next defined image
      do jimage=iimage-1,2,-1
        if(xok(jimage)==1) exit
      end do
      svar=1.D0/dble(iimage-jimage+1)
      neb%xcoords(:,iimage)=(1.D0-svar)*neb%xcoords(:,iimage+1) &
          + svar*neb%xcoords(:,jimage)
      !print*," ((((((((((((((((((((( IMAGE",iimage," i->x back))))))))))))))))))))))))"
      call dlf_direct_itox(glob%nvar,neb%varperimage,neb%coreperimage, &
          glob%icoords(cstart:cend),neb%xcoords(:,iimage),tok)
      if(tok) then
        xok(iimage)=1
      else
        !print*,"Failed at image ",iimage
        exit
      end if
    end do
  end if

  if(minval(xok(1:nimage-1))<1) then
    write(stdout,"('Image conversion success: ',10i4)") xok
    call dlf_fail("Failed converting images to Cartesians")
  end if

  ! Set xcoords of frozen atoms of all images to those in image 1
  ! This may be improved by specifying a different image (TS?)
  IF(glob%tatoms) then
    do iimage=2,nimage-1
      do iat=1,glob%nat
        if(glob%spec(iat)/=-1) cycle
        ivar=3*(iat-1)+1
        neb%xcoords(ivar:ivar+2,iimage)=neb%xcoords(ivar:ivar+2,1)
      end do
    end do
  end IF

  ! Another test to see if the conversion works with the input created above
  do iimage=2,nimage-2
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage)
    call dlf_direct_itox(glob%nvar,neb%varperimage,neb%coreperimage, &
        glob%icoords(cstart:cend),neb%xcoords(:,iimage),tok)
    if(.not.tok) then
      !print*,"Failed at image ",iimage
      call dlf_fail("NEB internal image initialisation &
          &failed in final test.")
    end if
  end do

end subroutine dlf_neb_definepath
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* neb/string_get_tau
!!
!! FUNCTION
!!
!! Define the string path as bsplines and calculate the tangents
!!
!! INPUTS
!!
!! glob%icoords 
!!
!! OUTPUTS
!! 
!! neb%tau
!!
!! SYNOPSIS
subroutine string_get_tau
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_neb, only: neb,unitp
  use dlf_bspline, only: spline_init, spline_create, spline_get, &
      spline_destroy
  implicit none
  integer  :: ivar,iimage,cstart,cend
  real(rk) :: stringpos(neb%nimage-1) ! 0..1
  real(rk) :: stringval(neb%nimage-1), svar
! **********************************************************************
  call spline_init(neb%nimage-1,neb%varperimage)

  ! define x-values of the splines
  do iimage=1,neb%nimage-1
    stringpos(iimage)=dble(iimage-1)/dble(neb%nimage-2)
  end do

  do ivar=1,neb%varperimage

    ! define y-value of the spline
    do iimage=1,neb%nimage-1
      stringval(iimage)=glob%icoords(neb%cstart(iimage)+ivar-1)
    end do

    call spline_create(ivar,stringpos,stringval)

    ! get derivatives of the spline
    do iimage=1,neb%nimage-1
      call spline_get(ivar,stringpos(iimage),svar, &
          neb%tau(neb%cstart(iimage)+ivar-1),svar)
    end do
  end do

  call spline_destroy

  ! normalise tau
  do iimage=1,neb%nimage-1
    cstart=neb%cstart(iimage)
    cend=neb%cend(iimage) 
    neb%tau(cstart:cend)=neb%tau(cstart:cend)/sqrt(sum(neb%tau(cstart:cend)**2))
  end do

end subroutine string_get_tau
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* neb/string_reparametrise
!!
!! FUNCTION
!!
!! Reparametrise the string path: the input images (beads on the string)
!! are glob%icoords+glob%step. 
!! 
!! Modify glob%step to make them evenly distributed along the path
!!
!! The string is defined with equally-spaced alpha (ranging from 0 to 1)
!! but will have different spacing in i-coordinate space. Then, alpha 
!! values for equal i-coordinate spacing are calculated, and the step is
!! modified.
!!
!! INPUTS
!!
!! glob%icoords, glob%step
!!
!! OUTPUTS
!! 
!! glob%step
!!
!! SYNOPSIS
subroutine string_reparametrise(treduce)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_neb, only: neb,unitp
  use dlf_bspline, only: spline_init, spline_create, spline_get, &
      spline_destroy
  use dlf_stat ! REMOVE JK
  implicit none
  logical,intent(in) :: treduce(neb%nimage)
  integer  :: ivar,iimage,cstart,cend
  real(rk) :: stringpos(neb%nimage-1) ! 0..1
  real(rk) :: stringval(neb%nimage-1), svar
  integer, parameter :: npoint=300 ! number of points interpolated
  real(rk) :: length(npoint) ! length of the string up to that point
  real(rk) :: energy(npoint) ! energy of the string (interpolated)
  real(rk) :: length_img(neb%nimage-1) ! ength of the string up to that
                                       ! image
  integer  :: ipoint,low,high,jimage,count
  real(rk) :: step,alpha,yvar,yvar2,dyvar,new_length,ene
  integer :: step_print
  real(rk):: svar2
  real(rk) :: tmp_step(glob%nivar),tmp_grad(glob%nivar),tmp_tau(neb%varperimage)
! **********************************************************************
  step_print=40

  if(printl>=4) write(stdout,'(a)') "Reparametrising string"

  count=0
  do iimage=1,neb%nimage-1
    if(.not.treduce(iimage)) then
      count=count+1
    else
      write(*,'("Image",i4," is reset to string")') iimage
    end if
  end do
  print*,"Number of images considered for the string:",count

!  call spline_init(neb%nimage-1,neb%varperimage)
!  call spline_init(count,neb%varperimage)
  call spline_init(count,neb%varperimage+1)

  ! define x-values of the splines - equally spaced
  count=0
  do iimage=1,neb%nimage-1
    if(.not.treduce(iimage)) then
      count=count+1
      stringpos(count)=dble(iimage-1)/dble(neb%nimage-2)
    end if
  end do

  do ivar=1,neb%varperimage+1

    ! define y-value of the spline: coordinates and energy
    count=0
    do iimage=1,neb%nimage-1
      if(.not.treduce(iimage)) then
        count=count+1
        if(ivar<=neb%varperimage) then
          stringval(count)=glob%icoords(neb%cstart(iimage)+ivar-1) + &
              glob%step(neb%cstart(iimage)+ivar-1)
        else
          stringval(count)=neb%ene(iimage)
        end if
      end if
    end do

    call spline_create(ivar,stringpos,stringval)

  end do

  ! energy
  do ipoint=1,npoint
    alpha=dble(ipoint-1)/dble(npoint-1)
    call spline_get(neb%varperimage+1,alpha,energy(ipoint),dyvar,svar)
 !   print*,ipoint,energy(ipoint)
  end do


  ! calculate the length of the string (arc length)
  length(1)=0.D0
  step=1.D0/dble(npoint-1)
  do ipoint=2,npoint
    length(ipoint)=0.D0
    alpha=dble(ipoint-1)/dble(npoint-1)
    do ivar=1,neb%varperimage
      call spline_get(ivar,alpha,yvar,dyvar,svar)
      call spline_get(ivar,alpha-step,yvar2,dyvar,svar)
      length(ipoint)=length(ipoint)+(yvar2-yvar)**2
    end do
    ene=0.5D0*(energy(ipoint)+energy(ipoint-1))-minval(energy)
    ene=ene/(maxval(energy)-minval(energy)) ! 0..1
    svar=0.0D0 ! energy weighting parameter 0=no weighting
    length(ipoint)=length(ipoint-1)+dsqrt(length(ipoint)) * &
        ! weighting
        (1.D0-svar+ene*2.D0*svar)

!    length(ipoint)=length(ipoint-1)+dsqrt(length(ipoint))
  end do

  if(printl>=4) write(stdout,'("Length of the string ",f10.5)') &
      length(npoint)

  ! calculate current positions of the images along the string
  ! only important for frozen images - will be wrong for reduced ones
  length_img(:)=0.D0
  do iimage=1,neb%nimage-1
    ! int rounds to the lower value
    ipoint=int(dble(iimage-1)/dble(neb%nimage-2)*dble(npoint-1))+1  ! point left of the image
    svar=dble(iimage-1)/dble(neb%nimage-2)*dble(npoint-1)-dble(ipoint-1)
    if(ipoint==npoint) then
      length_img(iimage)=length(npoint)
    else
      length_img(iimage)=(1.d0-svar)*length(ipoint)+svar*length(ipoint+1)
    end if
  end do

  if(printl>=6) write(stdout,'("Positions of images along the string: ",6f10.5)') &
       length_img(:)

  ! set non-frozen images to the optimum position on the string
  low=1
  high=neb%nimage-1
  svar2=0.D0
  do iimage=1,neb%nimage-1
    if(neb%frozen(iimage)) then
      low=iimage
    else
      high=neb%nimage-1
      do jimage=iimage+1,neb%nimage-1
        if(neb%frozen(jimage)) then
          high=jimage
          exit
        end if
      end do
      ! position of iimage on the arc - equal arc lenth spacing
      svar=dble(iimage-low)/dble(high-low)
      new_length=(1.D0-svar)*length_img(low)+svar*length_img(high)
      ! new_length is now the lenght value at which the image should be placed

      ! find the string variable (0..1) that corresponds to this length value
      do ipoint=1,npoint
        if(length(ipoint)>=new_length) exit
      end do

      ! the new image should be located at string(alpha):
      if(ipoint>1) then
        svar=(new_length-length(ipoint-1))/(length(ipoint)-length(ipoint-1))
        alpha=(dble(ipoint-2)+svar)/dble(npoint-1) 
      else
        alpha=0.D0
      end if

      ! recalculate the step
      do ivar=1,neb%varperimage
        call spline_get(ivar,alpha,yvar,dyvar,svar)
        svar2=svar2+(yvar - &
            glob%icoords(neb%cstart(iimage)+ivar-1))**2
        glob%step(neb%cstart(iimage)+ivar-1) = yvar - &
            glob%icoords(neb%cstart(iimage)+ivar-1)
      end do
    end if
  end do
  
  do iimage=1,neb%nimage-1
    svar=sqrt(sum((glob%step(neb%cstart(iimage):neb%cend(iimage)))**2))
    write(*,"('Img ',i3,' Step lenth:',f10.5)") iimage,svar
  end do

  call spline_destroy

end subroutine string_reparametrise
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_neb_write
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr
  use dlf_neb, only: neb
  use dlf_checkpoint, only: tchkform,write_separator
  implicit none
! **********************************************************************
  if(tchkform) then

    open(unit=100,file="dlf_neb.chk",form="formatted")

    call write_separator(100,"NEB Sizes")
    write(100,*) neb%nimage,neb%varperimage
    call write_separator(100,"NEB Parameters")
    write(100,*) neb%iimage,neb%step,neb%maximage,neb%mode,neb%k &
        ,neb%optcart
    call write_separator(100,"NEB Arrays")
    write(100,*) neb%ene,neb%frozen,neb%gradt,neb%xcoords,neb%frozen
    call write_separator(100,"END")


  else

    open(unit=100,file="dlf_neb.chk",form="unformatted")

    call write_separator(100,"NEB Sizes")
    write(100) neb%nimage,neb%varperimage
    call write_separator(100,"NEB Parameters")
    write(100) neb%iimage,neb%step,neb%maximage,neb%mode,neb%k &
        ,neb%optcart
    call write_separator(100,"NEB Arrays")
    write(100) neb%ene,neb%frozen,neb%gradt,neb%xcoords,neb%frozen
    call write_separator(100,"END")

  end if

  close(100)
    
end subroutine dlf_checkpoint_neb_write

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_neb_read(tok)
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout,printl
  use dlf_neb, only: neb
  use dlf_checkpoint, only: tchkform, read_separator
  implicit none
  logical,intent(out) :: tok
  logical             :: tchk
  integer             :: nimage,varperimage
! **********************************************************************
  tok=.false.

  ! check if checkpoint file exists
  INQUIRE(FILE="dlf_neb.chk",EXIST=tchk)
  if(.not.tchk) then
    write(stdout,10) "File dlf_neb.chk not found"
    return
  end if

  if(tchkform) then
    open(unit=100,file="dlf_neb.chk",form="formatted")
  else
    open(unit=100,file="dlf_neb.chk",form="unformatted")
  end if

  call read_separator(100,"NEB Sizes",tchk)
  if(.not.tchk) return    

  if(tchkform) then
    read(100,*,end=201,err=200) nimage,varperimage
  else
    read(100,end=201,err=200) nimage,varperimage
  end if

  if(neb%nimage/=nimage) then
    write(stdout,10) "Different numbers of NEB images"
    close(100)
    return
  end if
  
  if(neb%varperimage/=varperimage) then
    write(stdout,10) "Different numbers of variables per NEB image"
    close(100)
    return
  end if
  
  call read_separator(100,"NEB Parameters",tchk)
  if(.not.tchk) return    

  if(tchkform) then
    read(100,*,end=201,err=200) neb%iimage,neb%step, &
        neb%maximage,neb%mode,neb%k,neb%optcart
  else
    read(100,end=201,err=200) neb%iimage,neb%step, &
        neb%maximage,neb%mode,neb%k,neb%optcart
  end if

  call read_separator(100,"NEB Arrays",tchk)
  if(.not.tchk) return    

  if(tchkform) then
    read(100,*,end=201,err=200) neb%ene,neb%frozen,neb%gradt &
        ,neb%xcoords,neb%frozen
  else
    read(100,end=201,err=200) neb%ene,neb%frozen,neb%gradt &
        ,neb%xcoords,neb%frozen
  end if

  call read_separator(100,"END",tchk)
  if(.not.tchk) return

  close(100)
  tok=.true.

  if(printl >= 6) write(stdout,"('NEB checkpoint file successfully read')")

  return

  ! return on error
  close(100)
200 continue
  write(stdout,10) "Error reading file"
  return
201 continue
  write(stdout,10) "Error (EOF) reading file"
  return

10 format("Checkpoint reading WARNING: ",a)
end subroutine dlf_checkpoint_neb_read

!!****


! **********************************************************************
! **                                                                  **
! **              Hybrid Delocalised Internal Coordinates             **
! **                                                                  **
! **                         HDLC Driver routines                     **
! **                                                                  **
! **                                                                  **
! **********************************************************************
!!****h* coords/hdlc
!!
!! NAME
!! hdlc
!!
!! FUNCTION
!! 
!! Hybrid Delocalised Internal Coordinates
!! 
!! HDLC Driver routines
!!
!! Usage:
!! *  call dlf_hdlc_init
!! *  call dlf_hdlc_create
!! *  use the hdlc with dlf_hdlc_xtoi and dlf_hdlc_itox. In case a 
!!    breakdown occurs (tok==.false.) use:
!! *  call dlf_hdlc_reset 
!! *  call dlf_hdlc_create
!! *  finally: call dlf_hdlc_destroy
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
!!  Tom Keal (thomas.keal@stfc.ac.uk), Alex Turner, Salomon Billeter,
!!  Stephan Thiel, Max-Planck Institut fuer Kohlenforshung, Muelheim, 
!!  Germany.
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

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hdlc/dlf_hdlc_init
!!
!! FUNCTION
!!
!! Initialise HDLC object for one hdlc set:
!! * allocate all arrays that are independent of the coordinates and thus
!! of the actual hdlc state
!!
!! SYNOPSIS
subroutine dlf_hdlc_init(nat,spec,icoord,nconstr,iconstr,nincon,incon)
!! SOURCE
  use dlf_parameter_module, only: ik
  use dlf_global, only: stdout,printl
  use dlfhdlc_hdlclib, only: hdlc
  use dlf_allocate, only: allocate,deallocate
  implicit none
  integer ,intent(in) :: nat
  integer ,intent(in) :: spec(nat)
  integer ,intent(in) :: icoord
  integer ,intent(in) :: nconstr
  integer ,intent(in) :: iconstr(5,max(nconstr,1)) 
  integer ,intent(in) :: nincon
  integer ,intent(in) :: incon(2,max(nincon,1)) 
  ! local vars
  integer             :: ngroups,i,j,first,length,ires,iat,ndel,ii,count
  logical, parameter :: dbg=.false.
  integer, allocatable :: rlength(:)
! **********************************************************************
  hdlc%tinit=.true.
  hdlc%lhdlc=.false. ! no hdlcs created yet

  if(nat<1) then
    call dlf_fail("Number of atoms in HDLC must be >0!")
  end if

  if(printl>=4) then
    write(stdout,"('Residue member list:')")
    do iat=1,nat,10
      write(stdout,"(i5,' : ',10i5)") iat-1,(spec(iat+i),i=0,min(9,nat-iat))
    end do
  end if

  ! allocate and set residue numbers
  call allocate(hdlc%resn,nat)

  !  hdlc%resn(:)=spec(:)
  if(dbg) print*,"spec",spec
  hdlc%resn=0
  ! attention: ngoups is calculated here as well as in dlf_hdlc_create.
  ! it may be smaller there, as residues may have been deleted
  ! ires makes sure that residues are sorted and consecutively numbered
  ngroups = 0
  j = 0
  first=1
  hdlc%nfrozen=0
  ires=0
  !determine maximum number of residues
  DO i = 1, nat
    IF (spec(i)<=0) THEN
      IF (spec(i)<0) then
        hdlc%nfrozen=hdlc%nfrozen+1
        hdlc%resn(i)=spec(i)
      end IF
    ELSE 
      ! see if this spec has been handled before
      j=0
      do ii=i-1,1,-1
        if(spec(ii)==spec(i)) then
          j=hdlc%resn(ii)
          exit
        end if
      end do
      if(j/=0) then
        hdlc%resn(i)=j
      else
        ! new residue starts here
        ires=ires+1
        hdlc%resn(i)=ires
      end if
    end IF
  end DO
  ! now hdlc%resn(i) contains residue numbers in ascending order, not necessarily 
  !continous
  ! Determine length of the residues
  ngroups=ires
  call allocate(rlength,ngroups)
  rlength(:)=0
  DO i = 1, nat
    ! this should never happen
    if(hdlc%resn(i)>ngroups) call dlf_fail("Inconsistent residue list")
    if(hdlc%resn(i)>0) rlength(hdlc%resn(i))=rlength(hdlc%resn(i))+1
  end DO
  !delete residues with only one atom
  ndel=0
  do ires=1,ngroups
    ! this should never happen
    if(rlength(ires)==0) call dlf_fail("Inconsistent residue size")
    if(rlength(ires)==1) ndel=ndel+1
  end do
  if(ndel>0 .and. printl>=4 ) then
    print*,ndel," groups will be deleted as they only contain one atom"
  end if
  ngroups=ngroups-ndel

  if(dbg) print*,"hdlc%resn",hdlc%resn
  if(dbg) PRINT*,"ngroups",ngroups

  IF(ngroups==0) then
    call dlf_fail("No residues present in dfl_hdlc_init")
  end IF

  call allocate(hdlc%err_cnt,ngroups)
  hdlc%err_cnt(:)=0

  !=====================================================================
  ! allocate residues
  !=====================================================================
  hdlc%first=1
  hdlc%last=ngroups
  allocate(hdlc%res(1:ngroups))
  ! storage is a bit difficut to monitor here ...
  ! initialise the name (number of residue)
  do i=hdlc%first,hdlc%last
    hdlc%res(i)%name=i
  end do
  ! get the list of atoms participating in the corresponding residue (res%at)
  ires=0
  do i=1,ngroups+ndel
    if(rlength(i)==1) then
      ! delete this residue
      DO iat = 1, nat
        if(hdlc%resn(iat)==i) hdlc%resn(iat)=0
      end DO
    else
      ires=ires+1
      hdlc%res(ires)%natom=rlength(i)
      allocate(hdlc%res(ires)%at(rlength(i))) ! pointer
      allocate(hdlc%res(ires)%xweight(rlength(i))) ! pointer
      count=1
      do iat = 1, nat
        if(hdlc%resn(iat)==i) then
          ! this works at resn is in ascending order
          hdlc%resn(iat)=ires ! may be different as to removed residues
          hdlc%res(ires)%at(count)=iat
          count=count+1
        end if
      end do
    end if
  end do
  call deallocate(rlength)
!!$  ! debug:
!!$  do ires=1,ngroups
!!$    print*,"JK Group",ires,":",hdlc%res(ires)%at(:)
!!$  end do

  !=====================================================================
  ! handle constraints
  !=====================================================================
  hdlc%nconstr=nconstr
  call allocate(hdlc%iconstr,5,max(nconstr,1))
  if(nconstr>0) hdlc%iconstr(:,:)=iconstr(:,:)

  !=====================================================================
  ! handle user input connections
  !=====================================================================
  hdlc%nincon=nincon
  call allocate(hdlc%incon,2,max(nincon,1))
  if(nincon>0) hdlc%incon(:,:)=incon(:,:)

  !=====================================================================
  ! handle contyp (0 internals, 1 Total connection)
  !=====================================================================
  if(icoord==1) then
    hdlc%contyp=0
  else if(icoord==2) then
    hdlc%contyp=1
  else if(icoord==3) then
    hdlc%contyp=0
  else if(icoord==4) then
    hdlc%contyp=1
  else
    Write(stdout,"('icoord=',i4,'is not supported in HDLC')")
    call dlf_fail("Wrong icoord setting in HDLC")
  end if
  hdlc%internal=(icoord>=3)
  if(hdlc%internal) then
    if(maxval(spec)/=minval(spec)) then
      call dlf_fail("All atoms have to belong to the same residue if icoord > 2")
    end if
    hdlc%nmin=6
  else
    hdlc%nmin=0
  end if

end subroutine dlf_hdlc_init
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hdlc/dlf_hdlc_get_nivar
!!
!! FUNCTION
!!
!! return number of internal coordinates
!!
!! note that global data is used in these calculations
!!
!! SYNOPSIS
subroutine dlf_hdlc_get_nivar(region, nivar)
!! SOURCE
  use dlf_global, only: glob, stdout
  use dlfhdlc_hdlclib, only: hdlc, get_cons_regions
  implicit none
  integer, intent(in) :: region
  integer  ,intent(out) :: nivar
  integer :: i, iat, ninner, nouter, nfull
  integer :: nincons, noutcons
! **********************************************************************
  if(.not.hdlc%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_get_nivar")
  ! nat should prbably be replaced ...
  nfull = glob%nat * 3 - hdlc%nconstr - hdlc%nmin - hdlc%nfrozen * 3

  ninner = 0
  nouter = 0
  ! Total inner and outer coordinates
  do i = 1, glob%nat
     if (glob%spec(i) < 0) cycle ! frozen
     if (glob%micspec(i) == 1) then
        ninner = ninner + 3
     else
        nouter = nouter + 3
     end if
  end do
  ! Subtract no. of constraints from the correct region 
  call get_cons_regions(hdlc%nconstr, hdlc%iconstr, glob%nat, glob%spec, &
       glob%micspec, nincons, noutcons)
  ninner = ninner - nincons
  nouter = nouter - noutcons

  ! nmin only applies to a single (inner) residue setup, i.e. DLC or TC
  ninner = ninner - hdlc%nmin

  if (nfull /= ninner + nouter) then
     write(stdout,*) "nfull, ninner, nouter = ", nfull, ninner, nouter
     call dlf_fail("Inconsistent nivar values in dlf_hdlc_get_nivar")
  end if

  select case (region)
  case (0)
     nivar = nfull
  case (1)
     nivar = ninner
  case (2)
     nivar = nouter
  case default
     call dlf_fail("Unknown region number in dlf_hdlc_get_nivar")
  end select
  
end subroutine dlf_hdlc_get_nivar
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hdlc/dlf_hdlc_create
!!
!! FUNCTION
!!
!! create all hdlcs (i.e. the Ut matrix) 
!!
!! more than one image may be provided. In this case, the union of the
!! connections of all image are used. The Ut matrix is calculated for 
!! the first image provided.
!!
!! SYNOPSIS
subroutine dlf_hdlc_create(nat,nicore,spec,micspec,attypes,nimage,xcoords,xweight,mass)
!! SOURCE
  use dlf_global, only: printl,stdout
  use dlf_allocate, only: allocate, deallocate
  use dlfhdlc_hdlclib, only: hdlc,matrix,int_matrix,matrix_create,matrix_set, &
      int_matrix_destroy,matrix_destroy, &
      ! subroutines
      connect_prim, ci_conn, assign_cons, hdlc_create, assign_cons, &
      hdlc_create
  use dlf_parameter_module, only: rk
  implicit none
  integer ,intent(in) :: nat
  integer, intent(in) :: nicore
  integer ,intent(in) :: spec(nat)
  integer, intent(in) :: micspec(nat)
  integer ,intent(in) :: attypes(nat)
  integer ,intent(in) :: nimage
  real(rk),intent(in) :: xcoords(3,nimage*nat)
  real(rk),intent(in) :: xweight(nat)
  real(rk),intent(in) :: mass(nat)
  !
  integer             :: ngroups
  real(rk),allocatable:: vconstr(:) !max(1,nconstr) ! value of the constraint??
  TYPE (matrix)       :: cns, xyz
  INTEGER, pointer    :: iconn(:,:)
  TYPE (int_matrix)   :: con
  integer             :: iatom, group, ifin,idum,length, istart,nconn
  integer             :: i,j,fail,ncart,maxcon,iimage,iphdlc,ipinner,ipouter
  integer             :: ires, iat,irun
  real(rk),allocatable:: atmp(:)
  integer, allocatable:: atmpi(:)
  integer             :: nconst_here,nconst_sum
  integer             :: iregion
! **********************************************************************

  if(.not.hdlc%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_create")

  hdlc%lhdlc=.true. ! this does not seem to have any effect ...

! ngroups: number of HDLC residues in hdlc%resn
! the variable hdlc%ngroups has to denote the number of actually allocated
!   residues
  j = 0
  ncart = 0
  ngroups = 0
  DO i = 1, nat
    IF (hdlc%resn(i)==0) THEN
      ncart = ncart + 1
    else if (hdlc%resn(i)<0) then
      cycle
    ELSE IF (hdlc%resn(i)>j) THEN
      ngroups = ngroups + 1
      j = hdlc%resn(i)
    END IF
  END DO

  call allocate(vconstr,max(1,hdlc%nconstr))
  vconstr(:)=0.D0
  
  IF (printl>=2) WRITE (stdout,'(/,a,i4,a)') &
      'Generating new HDLC for ', ngroups, ' residues'

! loop over all groups
! icoords is ordered with inner region coords followed by
! outer region coords. ip (icoords position) must be set 
! accordingly.
  ipinner = 1
  ipouter = nicore + 1
  hdlc%first=1
  nconst_sum=0
  do ires=1,ngroups
    ! Determine which region the residue is in by the 
    ! micspec value of the first atom in it
    ! NB: for standard (non-microiterative) optimisations, 
    ! all residues are considered to be 'inner'
    iregion = micspec(hdlc%res(ires)%at(1))
    if (iregion == 1) then
       iphdlc = ipinner
    else
       iphdlc = ipouter
    end if
    ! Residues must not contain a mix of inner and outer atoms
    do i = 1, hdlc%res(ires)%natom
       if (micspec(hdlc%res(ires)%at(i)) /= iregion) then
          write (stdout,'(3x,a,i4,a)') 'Residue number ', ires, &
               ' crosses inner/outer boundary.'
          write (stdout,'("   Atoms: ",10i6)') hdlc%res(ires)%at(:)
          call dlf_fail("HDLC residue crosses inner/outer boundary")
       end if
    end do
    hdlc%res(ires)%ip=iphdlc
    iphdlc=iphdlc+hdlc%res(ires)%natom * 3

    length=hdlc%res(ires)%natom
! now we have group, istart, ifin
    istart=1
    ifin=1 !dummies to be removed - IMPROVE!
    IF (printl>=2) THEN
      WRITE (stdout,'(/,a)') 'Located a new residue for HDLC'
      WRITE (stdout,'(3x,a,i4)') 'Residue number is ', ires
      WRITE (stdout,'(3x,a,i5)') 'Number of atoms it contains ',length
      WRITE (stdout,'("   Atoms: ",10i6)') hdlc%res(ires)%at(:)
    END IF
    IF (length<2) THEN
      CALL hdlc_errflag('No residue can contain less than two atoms', &
          'stop')
    END IF
    xyz = matrix_create(3*length,1,'XYZ')

    ! define res%xweight
    do iat=1,length
      hdlc%res(ires)%xweight(iat)=xweight(hdlc%res(ires)%at(iat))
    end do

    ! get connections for primitive internals
    IF (hdlc%contyp==0) THEN

      nconn=0
      maxcon=length
      allocate (iconn(2,maxcon)) ! pointer

      ! get connections from the images, using image 1 as last one,
      !  because these xyz coordinates should remain
      do iimage=nimage,1,-1
        !check in atom coordinates
        call allocate(atmp,3*length)
        do iat=1,length
          atmp((iat-1)*3+1:(iat-1)*3+3)= &
              xcoords(:, (iimage-1)*nat + hdlc%res(ires)%at(iat))
        end do
        !CALL dummyarg_checkin(reshape(xcoords(:,(iimage-1)*nat+1:iimage*nat),&
        !    (/3*nat/)),3*(istart-1)+1,3*length)
        idum = matrix_set(xyz,size(atmp),atmp)
        !CALL dummyarg_clear
        call deallocate(atmp)

        !CALL dummyarg_checkin(attypes,istart,length)
        call allocate(atmpi,length)
        do iat=1,length
          atmpi(iat)=attypes(hdlc%res(ires)%at(iat))
        end do
        CALL connect_prim(length,maxcon,atmpi,nconn,iconn,xyz)
        call deallocate(atmpi)
        !CALL dummyarg_clear
      end do

      ! check in user connections and create connections matrix
      CALL ci_conn(con,nconn,iconn,hdlc%nincon,hdlc%incon,length,&
          hdlc%res(ires)%at)
      DEALLOCATE (iconn) ! pointer

      ! check in constraints
      CALL assign_cons(cns,hdlc%iconstr,hdlc%nconstr,vconstr, &
          1,length,hdlc%res(ires)%at,hdlc%internal,nconst_here)
      iphdlc=iphdlc-nconst_here

      ! create the HDLC - primitives
      CALL hdlc_create(hdlc%res(ires),xyz,con,cns,ires)
      idum = int_matrix_destroy(con)
      idum = matrix_destroy(cns)

      ! create the HDLC - total connection scheme - no check if only stretch constr.
    ELSE ! (ctrl%contyp.eq.0 ...)

      !check in atom coordinates
      call allocate(atmp,3*length)
      do iat=1,length
        atmp((iat-1)*3+1:(iat-1)*3+3)=xcoords(:,hdlc%res(ires)%at(iat))
      end do
      idum = matrix_set(xyz,size(atmp),atmp)
      call deallocate(atmp)

      !CALL dummyarg_checkin(reshape(xcoords,(/3*nat/)),3*(istart-1)+1,3*length)
      !idum = matrix_set(xyz,dummyarg)
      !CALL dummyarg_clear

      CALL assign_cons(cns,hdlc%iconstr,hdlc%nconstr,vconstr, &
          1,length,hdlc%res(ires)%at,hdlc%internal,nconst_here)
      iphdlc=iphdlc-nconst_here

      CALL hdlc_create(hdlc%res(ires),xyz,con,cns,ires)
      idum = matrix_destroy(cns)
    END IF ! (ctrl%contyp.eq.0 ...)

! restore the error counter
    hdlc%res(ires)%err_cnt = hdlc%err_cnt(ires)
    idum = matrix_destroy(xyz)
    nconst_sum=nconst_sum+nconst_here

    ! Write out mapping of HDLCs to internal coordinates
    ! Useful for identifying which HDLC the max grad component 
    ! corresponds to during an optimisation
    if (printl >= 2) then
       write(stdout, '("   Residue ", i6, " has internal coords ", i6, &
            " to ", i6)') ires, hdlc%res(ires)%ip, iphdlc-1
    end if

    ! update inner or outer coords position marker
    if (iregion == 1) then
       ipinner = iphdlc
    else
       ipouter = iphdlc
    end if

  END DO ! ires=1,ngroups
  call deallocate(vconstr)

  hdlc%nconstr=nconst_sum
  
end subroutine dlf_hdlc_create
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hdlc/dlf_hdlc_getweight
!!
!! FUNCTION
!!
!! return weights on internal coordinates
!!
!! SYNOPSIS
subroutine dlf_hdlc_getweight(nat,nivar,nicore,micspec,xweight,iweight)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout  
  use dlfhdlc_hdlclib
  implicit none
  integer ,intent(in) :: nat
  integer ,intent(in) :: nivar
  integer ,intent(in) :: nicore
  integer ,intent(in) :: micspec(nat)
  real(rk),intent(in) :: xweight(nat)   ! only used for cartesians
  real(rk),intent(out):: iweight(nivar)
  integer             :: iphdlc,group,iat,length,ndfhdlc,inta2(2)
  integer             :: ipinner, ipouter
  TYPE (residue_type) :: residue
! **********************************************************************
  if(.not.hdlc%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_getweight")

  iweight(:)=0.D0
  ipinner = 1
  ipouter = nicore + 1
  DO group = 1, hdlc%ngroups
 
    ! ignore deleted residues
    if(hdlc%res(group)%name==-1) cycle
    residue=hdlc%res(group)
    iphdlc = residue%ip 
    length = residue%natom

    if(.not.allocated(hdlc%res(group)%iweight%data)) then
      call dlf_fail("No weights present in dlf_hdlc_getweight")
    end if

    !ndfhdlc = 3*length - residue%ncons ! orig
    inta2=shape(hdlc%res(group)%iweight%data)
    ndfhdlc=inta2(1)

    iweight(iphdlc:iphdlc+ndfhdlc-1)=reshape(  &
        hdlc%res(group)%iweight%data(:,:),(/ndfhdlc/))
    ! for correct start of cartesians
    if (iphdlc <= nicore) then
       ipinner = ipinner + ndfhdlc-hdlc%nmin
    else 
       ipouter = ipouter + ndfhdlc-hdlc%nmin
    end if

  end DO

  DO iat = 1, nat
    IF (hdlc%resn(iat)==0) THEN
       if (micspec(iat) == 1) then
          iweight(ipinner:ipinner+2) = xweight(iat)
          ipinner = ipinner + 3
       else
          iweight(ipouter:ipouter+2) = xweight(iat)
          ipouter = ipouter + 3
       end if
    end IF
  end DO

end subroutine dlf_hdlc_getweight
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hdlc/dlf_hdlc_xtoi
!!
!! FUNCTION
!!
!! convert Cartesian -> HDLC
!!
!! SYNOPSIS
subroutine dlf_hdlc_xtoi(nat,nivar,nicore,micspec,xcoords,xgradient,&
     icoords,igradient)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout  
  use dlf_allocate, only: allocate,deallocate
  use dlfhdlc_hdlclib, only: hdlc,matrix,residue_type,matrix_create, &
      matrix_set,matrix_get,matrix_destroy, matrix_print, &
      ! subroutines
      coord_cart_to_hdlc, grad_cart_to_hdlc, hdlc_split_cons, hdlc_split_cons
  implicit none
  integer ,intent(in) :: nat
  integer ,intent(in) :: nivar
  integer, intent(in) :: nicore
  integer ,intent(in) :: micspec(nat)  
  real(rk),intent(in) :: xcoords(3,nat)
  real(rk),intent(in) :: xgradient(3,nat)
  real(rk),intent(out):: icoords(nivar)
  real(rk),intent(out):: igradient(nivar)

  TYPE (residue_type) :: residue
  TYPE (matrix)       :: cns, xyz, cxyz, chdlc, gxyz, ghdlc
  integer             :: ndfcons,group,iphdlc,length,ndfhdlc
  integer             :: idum,j,iat,m,ndfopt
  integer             :: ipinner, ipouter
  real(rk),pointer    :: prim_tmp(:) 
  real(rk),allocatable:: atmp(:)
! **********************************************************************
  if(.not.hdlc%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_xtoi")

  IF(hdlc%NGROUPS<1) call dlf_fail("Number of fragements in HDLC must be >0")

  IF (printl>=4) WRITE (stdout,'(/,A)') &
      'Converting Cartesians to HDLC'
  ndfcons = 0

  ipinner = 1
  ipouter = nicore + 1

  DO group = 1, hdlc%ngroups
 
    ! ignore deleted residues
    if(hdlc%res(group)%name==-1) cycle
    residue=hdlc%res(group)

    iphdlc = residue%ip 
    ndfcons = ndfcons + residue%ncons
    length = residue%natom
    ndfhdlc = 3*length - residue%ncons

! now we have group, istart, ifin, iphdlc, ndfhdlc; convert coords to HDLC
    cxyz = matrix_create(3*length,1,'CXYZ')
    chdlc = matrix_create(3*length-hdlc%nmin,1,'CHDLC')
    call allocate(atmp,3*length)
    do iat=1,length
      atmp((iat-1)*3+1:(iat-1)*3+3)= xcoords(:,residue%at(iat))
    end do
    idum = matrix_set(cxyz,size(atmp),atmp)

    CALL coord_cart_to_hdlc(residue,cxyz,chdlc,prim_tmp,.FALSE.)

! convert gradient to HDLC
    gxyz = matrix_create(3*length,1,'GXYZ')
    ghdlc = matrix_create(3*length-hdlc%nmin,1,'GHDLC')
    do iat=1,length
      atmp((iat-1)*3+1:(iat-1)*3+3)= xgradient(:,residue%at(iat))
    end do
    !CALL dummyarg_checkin(reshape(xgradient,(/3*nat/)),3*(istart-1)+1,3*length)
    idum = matrix_set(gxyz,size(atmp),atmp)
    call deallocate(atmp)
    !CALL dummyarg_clear
    CALL grad_cart_to_hdlc(residue,cxyz,gxyz,ghdlc)

! separate between active space and constraints - resize CHDLC and GHDLC
    IF (residue%ncons/=0) THEN
      CALL hdlc_split_cons(residue,chdlc,.true.)
      CALL hdlc_split_cons(residue,ghdlc,.FALSE.)
    END IF

! set icoords and igradient, size of chdlc/ghdlc now: ndfhdlc
    idum = matrix_get(chdlc,ndfhdlc-hdlc%nmin,&
        icoords(iphdlc:iphdlc+ndfhdlc-hdlc%nmin-1))
    idum = matrix_get(ghdlc,ndfhdlc-hdlc%nmin,&
        igradient(iphdlc:iphdlc+ndfhdlc-hdlc%nmin-1))

! prepare for the next group
    idum = matrix_destroy(cxyz)
    idum = matrix_destroy(gxyz)
    idum = matrix_destroy(chdlc)
    idum = matrix_destroy(ghdlc)

    ! for correct start of cartesians
    if (iphdlc <= nicore) then
       ipinner = ipinner + ndfhdlc-hdlc%nmin
    else
       ipouter = ipouter + ndfhdlc-hdlc%nmin
    end if

  END DO ! group = 1,hdlc%ngroups

! check in Cartesians - no cartesian constraints implemented!
  j = 0
  DO iat = 1, nat
    IF (hdlc%resn(iat)==0) THEN
       if (micspec(iat) == 1) then
          iphdlc = ipinner
       else
          iphdlc = ipouter
       end if
       DO m = 0, 2
          icoords(iphdlc+m) =   xcoords(m+1,iat)
          igradient(iphdlc+m) = xgradient(m+1,iat)
       END DO
       iphdlc = iphdlc + 3
       if (micspec(iat) == 1) then
          ipinner = iphdlc
       else
          ipouter = iphdlc
       end if
    END IF
  END DO ! iat = 1,nat

  if (ipinner /= nicore + 1) then
     write(stdout,*) "ipinner, nicore=", ipinner, nicore
     call dlf_fail("Error in the transformation hdlc_xtoi (inner)")
  end if
  if (ipouter /= nivar + 1) then
     write(stdout,*) "ipouter, nivar=", ipouter, nivar
     call dlf_fail("Error in the transformation hdlc_xtoi (outer)")
  end if

end subroutine dlf_hdlc_xtoi
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hdlc/dlf_hdlc_hessian_xtoi
!!
!! FUNCTION
!!
!!  convert Hessian: Cartesian -> HDLC
!! 
!!  This routine misses a term within each residue: the drivative of the internals with 
!!  respect to both cartesians!
!!
!!  Additionally, all inter-residue terms are missing. This may be less 
!!  of a problem, as the whole core should be one residue anyway.
!!
!!  Also does not yet support microiterative calcs. In principle this 
!!  could be implemented with HDLCs, providing the inner region contains
!!  only one residue.
!!
!! SYNOPSIS
subroutine dlf_hdlc_hessian_xtoi(nat,nivar,xcoords,xhessian,ihessian)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: stdout,printl
  use dlfhdlc_hdlclib, only: hdlc,matrix,residue_type,matrix_create, &
      matrix_set,matrix_get,matrix_destroy, &
      ! subroutines
      hess_cart_to_hdlc
  implicit none
  integer ,intent(in) :: nat
  integer ,intent(in) :: nivar
  real(rk),intent(in) :: xcoords(3,nat)
  real(rk),intent(in) :: xhessian(3*nat,3*nat)
  real(rk),intent(out):: ihessian(nivar,nivar)
  integer             :: group,istart,length,idum,pos
  TYPE (matrix)       :: cxyz,hxyz,hhdlc
  TYPE (residue_type) :: residue
! **********************************************************************
  if(.not.hdlc%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_xtoi")
  
  IF(hdlc%NGROUPS<1) call dlf_fail("Number of fragemnts in HDLC must be >0")

  if(hdlc%ngroups>1) then
    call dlf_fail("Conversion of a Cartesian Hessian to HDLC only works &
        & for only one fragment (residue)")
  end if

  IF (printl>=4) WRITE (stdout,'(/,A)') &
      'Converting Cartesian Hessian to HDLC'

  pos=1
  istart=1
  DO group = 1, hdlc%ngroups

    ! ignore deleted residues
    if(hdlc%res(group)%name==-1) cycle
    residue=hdlc%res(group)

    length = residue%natom

    ! set coordinates
    cxyz = matrix_create(3*length,1,'CXYZ')
    !CALL dummyarg_checkin(reshape(xcoords,(/3*nat/)),3*(istart-1)+1,3*length)
    !idum = matrix_set(cxyz,dummyarg)
    idum = matrix_set(cxyz,3*length,xcoords(:,istart:istart+length-1))
    !CALL dummyarg_clear

    ! set cartesian hessian matrix
    hxyz = matrix_create(3*length,3*length,'HXYZ')
    !CALL dummyarg_checkin(hess,3*(istart-1)+1,3*length)
    ! This will be a block-diagonal Hessian with a block for each residue only.
    !    that's not what it is supposed to be in the end...
    ! position of the residue in icoords (residue%ip) should be considered!
    idum=(3*length-hdlc%nmin)**2
    idum = matrix_set(hxyz,idum,xhessian(3*(istart-1)+1:3*(istart-1)+3*length,&
        3*(istart-1)+1:3*(istart-1)+3*length))
    !CALL dummyarg_clear

    hhdlc = matrix_create(3*length-hdlc%nmin,3*length-hdlc%nmin,'HHDLC')
    ! convert cartesian to HDLC
    call hess_cart_to_hdlc(residue,cxyz,hxyz,hhdlc)

    ! here, constraints should be handeled

    ! read out the hessian to hess
    idum=(3*length-hdlc%nmin)**2
    idum = matrix_get(hhdlc,idum,ihessian(pos:pos+3*length-hdlc%nmin-1, &
        pos:pos+3*length-hdlc%nmin-1))

    idum = matrix_destroy(cxyz)
    idum = matrix_destroy(hxyz)
    idum = matrix_destroy(hhdlc)

    pos=pos+3*length-hdlc%nmin
    
  end DO

end subroutine dlf_hdlc_hessian_xtoi
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hdlc/dlf_hdlc_itox
!!
!! FUNCTION
!!
!!  convert HDLC -> Cartesian
!!
!! SYNOPSIS
subroutine dlf_hdlc_itox(nat,nivar,nicore,micspec,icoords,xcoords,tok)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout  
  use dlfhdlc_hdlclib, only: hdlc,matrix,residue_type,matrix_create,matrix_set, &
      matrix_destroy,matrix_get, matrix_print,&
      ! subroutines
      hdlc_rest_cons,coord_hdlc_to_cart,hdlc_destroy_all
  use dlf_allocate, only: allocate, deallocate
  implicit none
  integer ,intent(in)   :: nat
  integer ,intent(in)   :: nivar
  integer, intent(in)   :: nicore
  integer ,intent(in)   :: micspec(nat) 
  real(rk),intent(in)   :: icoords(nivar)
  real(rk),intent(inout):: xcoords(3,nat)
  logical ,intent(out)  :: tok
  !
  TYPE (residue_type)   :: residue
  TYPE (matrix)         :: cns, xyz, cxyz, chdlc, gxyz, ghdlc
  integer               :: ndfcons,group,iphdlc,length,ndfhdlc
  integer               :: idum,j,iatom,m,ndfopt,i,iat
  integer               :: ipinner, ipouter
  real(rk),allocatable  :: atmp(:)
! **********************************************************************

  if(.not.hdlc%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_itox")

  IF(hdlc%NGROUPS<1) call dlf_fail("Number of fragemnts in HDLC must be >0")

  tok=.true.
  hdlc%interror=.false.
  IF (printl>=4) WRITE (stdout,'(/,A)') &
      'Converting HDLC to Cartesians'

  ipinner = 1
  ipouter = nicore + 1

! loop over all residues
  hdlc%ngroupsdrop = 0
  ndfcons = 0
  DO group = 1, hdlc%ngroups
    residue = hdlc%res(group)
    ! ignore deleted residues
    if(hdlc%res(group)%name==-1) cycle

    iphdlc = residue%ip 
    ndfcons = ndfcons + residue%ncons
    length = residue%natom
    ndfhdlc = 3*length - residue%ncons

! now we have group, istart, ifin, iphdlc, ndfhdlc; check out HDLC coordinates
    chdlc = matrix_create(ndfhdlc-hdlc%nmin,1,'CHDLC')
    cxyz = matrix_create(3*length,1,'CXYZ')
    !CALL dummyarg_checkin(icoords,iphdlc,ndfhdlc-hdlc%nmin)
    idum = matrix_set(chdlc,ndfhdlc-hdlc%nmin,icoords(iphdlc:iphdlc+ndfhdlc-hdlc%nmin-1))
    call allocate(atmp,3*residue%natom)
    do iat=1,length
      atmp((iat-1)*3+1:(iat-1)*3+3)= xcoords(:,residue%at(iat))
    end do
    idum = matrix_set(cxyz,size(atmp),atmp) !xcoords(:,istart:istart+residue%natom-1))

! restore values of constrained variables if required
    IF (residue%ncons/=0) THEN
      CALL hdlc_rest_cons(residue,chdlc)
    END IF

! fit Cartesian coordinates to HDLC coordinates; size of chdlc now: 3*length
    CALL coord_hdlc_to_cart(residue,cxyz,chdlc)

! conversion HDLC -> Cartesian failed due to singular G matrix
    IF ( .NOT. residue%lgmatok) THEN
      tok=.false.
      residue%err_cnt = residue%err_cnt + 1000

      IF (printl>=2) WRITE (stdout,'(3X,A,I4,A,I4,/)') &
          'Conversion of residue ', residue%name, &
          ' failed , HDLC failure gauge: ', residue%err_cnt

! persistent HDLC failure - remove residue if there are no constraints
      IF (residue%err_cnt>=1990) THEN
        ! the removal of a residue is broken since the introduction of residue%at and
        ! residue %xweight. Thus stop.
        WRITE (stdout,'(A,I4,A)') &
              'Cyclic failure at residue ', residue%name, &
              ', stopping'
        call dlf_fail("Residue conversion error")

        IF (residue%ncons>0) THEN
          hdlc%interror = .TRUE.
          IF (printl>=1) WRITE (stdout,'(A,I4,A,I4)') &
              'Warning: could not remove residue', residue%name, &
              ', number of constraints: ', residue%ncons
        ELSE
          hdlc%ngroupsdrop = hdlc%ngroupsdrop + 1
          IF (printl>=2) WRITE (stdout,'(A,I4,A)') &
              'Cyclic failure - removing residue ', residue%name, &
              ' from list'
          DO i = 1,length
            hdlc%resn(residue%at(i)) = -10 ! changed from -2 to -10
          END DO
        END IF
      END IF

      ! leave xcoords for this residue unchanged if failes

! conversion HDLC -> Cartesian was successful
    ELSE
      residue%err_cnt = residue%err_cnt/2
      IF (printl>=6) WRITE (stdout,'(5X,A,I5,A,I3,/)') 'Residue ', &
          residue%name, ', HDLC failure gauge: ', residue%err_cnt

      ! check in coordinates
      idum=matrix_get(cxyz,3*residue%natom,atmp)
      do iat=1,length
        xcoords(:,residue%at(iat))=atmp((iat-1)*3+1:(iat-1)*3+3)
      end do
      !idum=matrix_get(cxyz,xcoords(:,istart:istart+length-1))

    END IF

! prepare for the next group
    ! for correct start of cartesians
    if (iphdlc <= nicore) then
       ipinner = ipinner + ndfhdlc-hdlc%nmin
    else
       ipouter = ipouter + ndfhdlc-hdlc%nmin
    end if
    idum = matrix_destroy(cxyz)
    idum = matrix_destroy(chdlc)
    call deallocate(atmp)
    hdlc%res(group) = residue 
  END DO ! (group = 1,ngroups)

! check in Cartesians
  DO iatom = 1, nat
    IF (hdlc%resn(iatom)==0) THEN
       if (micspec(iatom) == 1) then
          iphdlc = ipinner
       else
          iphdlc = ipouter
       end if
       DO m = 0, 2
          xcoords(m+1,iatom) = icoords(iphdlc+m)
       END DO
       iphdlc = iphdlc + 3
       if (micspec(iatom) == 1) then
          ipinner = iphdlc
       else
          ipouter = iphdlc
       end if
    END IF
  END DO 

  if (ipinner /= nicore + 1) then
     write(stdout,*) "ipinner, nicore=", ipinner, nicore
     call dlf_fail("Error in the transformation hdlc_itox (inner)")
  end if
  if (ipouter /= nivar + 1) then
     write(stdout,*) "ipouter, nivar=", ipouter, nivar
     call dlf_fail("Error in the transformation hdlc_itox (outer)")
  end if

! if geometry died, save err_cnt and destroy all HDLC residues and restart
! tok is returned. It is the responsibility of the calling routine to call
! dlf_hdlc_reset if the HDLCs should really be destroyed.
!!$  IF ( .NOT. tok) THEN
!!$    CALL hdlc_destroy_all(.TRUE.,hdlc%err_cnt)
!!$    hdlc%ngroups = hdlc%ngroups - ngroupsdrop
!!$
!!$! if a residue has been removed, its number is set to -10 to prevent checkin
!!$    DO iatom = 1, nat
!!$      IF (hdlc%resn(iatom)==-10) hdlc%resn(iatom) = -1 ! this has to be checked ...
!!$    END DO
!!$    IF (interror) THEN
!!$      call dlf_fail("Giving up on HDLC due to a residue that does not &
!!$          &converge and contains constraints")
!!$    END IF
!!$  END IF
end subroutine dlf_hdlc_itox
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hdlc/dlf_hdlc_reset
!!
!! FUNCTION
!!
!! Reset the internal coordinate definition. Possibly remove one or more
!! residues. Should be called after a hdlc coordinate breakdown.
!!
!! SYNOPSIS
subroutine dlf_hdlc_reset
!! SOURCE
  use dlf_global, only:glob
  use dlfhdlc_hdlclib, only: hdlc, hdlc_destroy_all
  implicit none
  integer       :: iatom
! **********************************************************************
  if(.not.hdlc%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_reset")
  CALL hdlc_destroy_all(.TRUE.,size(hdlc%err_cnt),hdlc%err_cnt)
! if a residue has been removed, its number is set to -10 to prevent checkin
  DO iatom = 1, glob%nat ! <--- this is dangerous - improve!
    IF (hdlc%resn(iatom)==-10) hdlc%resn(iatom) = 0 ! this has to be checked ...
  END DO
  hdlc%ngroups = hdlc%ngroups - hdlc%ngroupsdrop
  hdlc%ngroupsdrop=0

  ! commented out as it leads to termination after the first breakdown
  !if (hdlc%ngroups==0.and.hdlc%internal) then
  !  call dlf_fail("Internal coordinates can not be restored. Use cartesians instead")
  !end if

  IF (hdlc%interror) THEN
    call dlf_fail("Giving up on HDLC due to a residue that does not &
        &converge and contains constraints")
  END IF
end subroutine dlf_hdlc_reset
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* hdlc/dlf_hdlc_destroy
!!
!! FUNCTION
!!
!! Destroy the HDLC object and deallocate all arrays
!!
!! SYNOPSIS
subroutine dlf_hdlc_destroy
!! SOURCE
  use dlf_parameter_module, only: ik
  use dlf_global, only: stdout
  use dlf_allocate, only: deallocate
  use dlfhdlc_hdlclib, only: hdlc, hdlc_destroy_all
  implicit none
  integer             :: fail,igroup
! **********************************************************************
  if(.not.hdlc%tinit) return ! nothing to do 

  hdlc%tinit=.false.

  if(hdlc%lhdlc) then
    CALL hdlc_destroy_all(.TRUE.,size(hdlc%err_cnt),hdlc%err_cnt)
  end if

  ! deallocate arrays of individual residues
  do igroup=1,size(hdlc%err_cnt)
    deallocate(hdlc%res(igroup)%xweight) ! pointer
    deallocate(hdlc%res(igroup)%at)      ! pointer
  end do

  ! deallocate hdlc arrays
  call deallocate(hdlc%resn)
  call deallocate(hdlc%err_cnt)
  call deallocate(hdlc%iconstr)
  call deallocate(hdlc%incon)

  ! deallocate residues - no storage check available
  ! this was previously commented out because of a bug in pgf90
  ! however not deallocating it causes problems with ifort, giving an
  ! error of array already allocated if an (h)dlc job is run twice in the
  ! same chemshell script. This code doesn't seem to cause pgf90 any 
  ! problems any more anyway (as of v9.0)
  deallocate(hdlc%res,stat=fail)
  if(fail>0) then
    call dlf_fail("Deallocation error at residues")
  end if
  
end subroutine dlf_hdlc_destroy
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_hdlc_write
  ! Write HDLC data to checkpoint file
  use dlfhdlc_hdlclib, only: hdlc,hdlc_wr_hdlc
  use dlf_checkpoint, only: tchkform
  implicit none
  logical :: lerr
! **********************************************************************
  if(tchkform) then
    open(unit=100,file="dlf_hdlc.chk",form="formatted")
  else
    open(unit=100,file="dlf_hdlc.chk",form="unformatted")
  end if

  call hdlc_wr_hdlc(100,hdlc,tchkform,lerr)

  close(100)

end subroutine dlf_checkpoint_hdlc_write

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_hdlc_read(lerr)
  ! Read HDLC data from checkpoint file
  use dlfhdlc_hdlclib, only: hdlc,hdlc_rd_hdlc
  use dlf_checkpoint, only: tchkform
  implicit none
  logical ,intent(out) :: lerr
! **********************************************************************
  if(tchkform) then
    open(unit=100,file="dlf_hdlc.chk",form="formatted")
  else
    open(unit=100,file="dlf_hdlc.chk",form="unformatted")
  end if

  call hdlc_rd_hdlc(100,hdlc,tchkform,lerr)

  close(100)

end subroutine dlf_checkpoint_hdlc_read

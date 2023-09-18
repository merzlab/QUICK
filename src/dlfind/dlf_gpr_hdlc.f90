module oop_hdlc
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout!,pi
  use dlfhdlc_matrixlib
  use dlfhdlc_primitive
  use dlfhdlc_constraint
  implicit none
  TYPE residue_type
    ! scalars
    INTEGER :: name ! number of the this residue
    integer :: natom
    LOGICAL :: lgmatok
    INTEGER :: np, nconn, nbend, nrots, ncons
    logical :: tbias ! is bias set?
    ! replacing the linked list:
    integer :: next,prev
    integer :: ip ! start position of this residue in the icoords array
    INTEGER :: err_cnt
    ! arrays
    ! allocated in hdlc_create, deallocated in hdlc_destroy
    REAL (rk), POINTER :: x(:), y(:), z(:) ! (natom)
    INTEGER, POINTER   :: iconn(:,:) ! (2,max(1,residue%nconn))
    INTEGER, pointer   :: ibend(:,:) ! (4,max(1, > residue%nbend))
    INTEGER, POINTER   :: irots(:,:) ! (4,max(1, > residue%nrots))
    INTEGER, POINTER   :: icons(:,:) ! (6,max(1,residue%ncons))
    REAL (rk), POINTER :: vcons(:) ! (max(1,residue%ncons))
    REAL (rk), POINTER :: biasv(:) ! (residue%np) reference value for torsions and impropers
    ! at and xweight are allocated in dlf_hdlc_init, deallocated in dlf_hdlc_destroy
    ! at and xweight used to be allocatable arrays rather than pointers.
    ! JK changed it to pointers to make the code work with pgf90 7.1 and newer
    integer,  pointer  :: at(:) ! (natom) atom indices 
    real(rk), pointer  :: xweight(:) ! (natom)
    ! matrices
    TYPE (matrix) :: ut      ! (n6,nii,'Ut matrix')
    TYPE (matrix) :: oldxyz  ! (3*residue%natom,1,'old XYZ')
    TYPE (matrix) :: iweight ! (matrix_dimension(bhdlc,1)-residue%ncons ,1,'i weight')
  END TYPE residue_type
  
  TYPE hdlc_ctrl
    ! all allcatables are allocated in dlf_hdlc_init, and dallocated in dlf_hdlc_destroy
    LOGICAL              :: tinit ! is this instance initialised?
    LOGICAL              :: lhdlc, internal
    INTEGER              :: ngroups
    integer              :: contyp ! 0 internals, 1 total connection
    integer, allocatable :: resn(:) ! (nat)
    integer, allocatable :: err_cnt(:) ! (ngroups) error counter
    integer              :: nmin ! 0 for hdlc, 6 for DLC
    ! residues
    integer              :: first,last ! number of first and last active res
    TYPE (residue_type), allocatable :: res(:) ! (last-first+1)
    ! constraint data
    integer              :: nconstr ! global number of constraints
    integer, allocatable :: iconstr(:,:) ! (6,nconstr) constraint data
    ! connection data
    integer              :: nincon ! number of user provided connections
    integer, allocatable :: incon(:,:) ! (2,nincon) connections
    logical              :: interror ! internal permanent error ocurred
    integer              :: ngroupsdrop ! number of groups to drop when restarted
    integer              :: nfrozen ! number of frozen atoms
    real(rk),allocatable :: bhdlc_matrix(:,:)
    contains
      procedure, pass(this) ::  dlf_hdlc_init
      procedure, pass(this) ::  dlf_hdlc_create
      procedure, pass(this) ::  dlf_hdlc_destroy
      procedure, pass(this) ::  dlf_hdlc_get_nivar
      procedure, pass(this) ::  dlf_hdlc_getweight
      procedure, pass(this) ::  dlf_hdlc_itox
      procedure, pass(this) ::  dlf_hdlc_reset
      procedure, pass(this) ::  dlf_hdlc_xtoi
      procedure, pass(this) ::  hdlc_create
      procedure, pass(this) ::  coord_cart_to_hdlc
      procedure, pass(this) ::  grad_cart_to_hdlc
      procedure, pass(this) ::  hdlc_split_cons
      procedure, pass(this) ::  hdlc_rest_cons
      procedure, pass(this) ::  coord_hdlc_to_cart
      procedure, pass(this) ::  hdlc_destroy_all
      procedure, pass(this) ::  hdlc_make_bprim
      procedure, pass(this) ::  hdlc_make_ut
      procedure, pass(this) ::  hdlc_make_bhdlc
      procedure, pass(this) ::  hdlc_make_ighdlc
      procedure, pass(this) ::  hdlc_report_failure
      procedure, pass(this) ::  hdlc_destroy
  END TYPE hdlc_ctrl    
interface hdlc_ctrl
  module procedure hdlcConstructor
end interface 

contains 

type(hdlc_ctrl) function hdlcConstructor()
  implicit none
  hdlcConstructor%tinit = .false.
  hdlcConstructor%lhdlc = .false.
  hdlcConstructor%ngroups = 0
end function hdlcConstructor


subroutine dlf_hdlc_init(this,nat,spec,icoord,nconstr,iconstr,nincon,incon)
!! SOURCE
  use dlf_parameter_module, only: ik
  use dlf_global
  use dlf_allocate, only: allocate,deallocate
  implicit none
  class(hdlc_ctrl), intent(inout) :: this
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
  this%tinit=.true.
  this%lhdlc=.false. ! no hdlcs created yet

  if(nat<1) then
    call dlf_fail("Number of atoms in HDLC must be >0!")
  end if

  if(printl>=6) then
    write(stdout,"('Residue member list:')")
    do iat=1,nat,10
      write(stdout,"(i5,' : ',10i5)") iat-1,(spec(iat+i),i=0,min(9,nat-iat))
    end do
  end if

  ! allocate and set residue numbers
  call allocate(this%resn,nat)
  allocate(this%bhdlc_matrix(glob%nivar,3*glob%nat))
  !  hdlc%resn(:)=spec(:)
  if(dbg) print*,"spec",spec
  this%resn=0
  ! attention: ngoups is calculated here as well as in dlf_hdlc_create.
  ! it may be smaller there, as residues may have been deleted
  ! ires makes sure that residues are sorted and consecutively numbered
  ngroups = 0
  j = 0
  first=1
  this%nfrozen=0
  ires=0
  !determine maximum number of residues
  DO i = 1, nat
    IF (spec(i)<=0) THEN
      IF (spec(i)<0) then
        this%nfrozen=this%nfrozen+1
        this%resn(i)=spec(i)
      end IF
    ELSE 
      ! see if this spec has been handled before
      j=0
      do ii=i-1,1,-1
        if(spec(ii)==spec(i)) then
          j=this%resn(ii)
          exit
        end if
      end do
      if(j/=0) then
        this%resn(i)=j
      else
        ! new residue starts here
        ires=ires+1
        this%resn(i)=ires
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
    if(this%resn(i)>ngroups) call dlf_fail("Inconsistent residue list")
    if(this%resn(i)>0) rlength(this%resn(i))=rlength(this%resn(i))+1
  end DO
  !delete residues with only one atom
  ndel=0
  do ires=1,ngroups
    ! this should never happen
    if(rlength(ires)==0) call dlf_fail("Inconsistent residue size")
    if(rlength(ires)==1) ndel=ndel+1
  end do
  if(ndel>0 .and. printl>=6 ) then
    print*,ndel," groups will be deleted as they only contain one atom"
  end if
  ngroups=ngroups-ndel

  if(dbg) print*,"this%resn",this%resn
  if(dbg) PRINT*,"ngroups",ngroups

  IF(ngroups==0) then
    call dlf_fail("No residues present in dfl_hdlc_init")
  end IF

  call allocate(this%err_cnt,ngroups)
  this%err_cnt(:)=0

  !=====================================================================
  ! allocate residues
  !=====================================================================
  this%first=1
  this%last=ngroups
  allocate(this%res(1:ngroups))
  ! storage is a bit difficut to monitor here ...
  ! initialise the name (number of residue)
  do i=this%first,this%last
    this%res(i)%name=i
  end do
  ! get the list of atoms participating in the corresponding residue (res%at)
  ires=0
  do i=1,ngroups+ndel
    if(rlength(i)==1) then
      ! delete this residue
      DO iat = 1, nat
        if(this%resn(iat)==i) this%resn(iat)=0
      end DO
    else
      ires=ires+1
      this%res(ires)%natom=rlength(i)
      allocate(this%res(ires)%at(rlength(i))) ! pointer
      allocate(this%res(ires)%xweight(rlength(i))) ! pointer
      count=1
      do iat = 1, nat
        if(this%resn(iat)==i) then
          ! this works at resn is in ascending order
          this%resn(iat)=ires ! may be different as to removed residues
          this%res(ires)%at(count)=iat
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
  this%nconstr=nconstr
  call allocate(this%iconstr,5,max(nconstr,1))
  if(nconstr>0) this%iconstr(:,:)=iconstr(:,:)

  !=====================================================================
  ! handle user input connections
  !=====================================================================
  this%nincon=nincon
  call allocate(this%incon,2,max(nincon,1))
  if(nincon>0) this%incon(:,:)=incon(:,:)

  !=====================================================================
  ! handle contyp (0 internals, 1 Total connection)
  !=====================================================================
  if(icoord==1) then
    this%contyp=0
  else if(icoord==2) then
    this%contyp=1
  else if(icoord==3) then
    this%contyp=0
  else if(icoord==4) then
    this%contyp=1
  else
    Write(stdout,"('icoord=',i4,'is not supported in HDLC')")
    call dlf_fail("Wrong icoord setting in HDLC")
  end if
  this%internal=(icoord>=3)
  if(this%internal) then
    if(maxval(spec)/=minval(spec)) then
      call dlf_fail("All atoms have to belong to the same residue if icoord > 2")
    end if
    this%nmin=6
  else
    this%nmin=0
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
subroutine dlf_hdlc_get_nivar(this,region, nivar)
!! SOURCE
  use dlf_global, only: glob, stdout
  implicit none
  class(hdlc_ctrl), intent(inout) :: this
  integer, intent(in) :: region
  integer  ,intent(out) :: nivar
  integer :: i, iat, ninner, nouter, nfull
  integer :: nincons, noutcons
! **********************************************************************
  if(.not.this%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_get_nivar")
  ! nat should prbably be replaced ...
  nfull = glob%nat * 3 - this%nconstr - this%nmin - this%nfrozen * 3

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
  call get_cons_regions(this%nconstr, this%iconstr, glob%nat, glob%spec, &
       glob%micspec, nincons, noutcons)
  ninner = ninner - nincons
  nouter = nouter - noutcons

  ! nmin only applies to a single (inner) residue setup, i.e. DLC or TC
  ninner = ninner - this%nmin

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
subroutine dlf_hdlc_create(this,nat,nicore,spec,micspec,attypes,nimage,xcoords,xweight,mass,spur)
!! SOURCE
  use dlf_global, only: printl,stdout
  use dlf_allocate, only: allocate, deallocate
  use dlf_parameter_module, only: rk
  implicit none
  class(hdlc_ctrl), intent(inout) :: this
  integer ,intent(in) :: nat
  integer, intent(in) :: nicore
  integer ,intent(in) :: spec(nat)
  integer, intent(in) :: micspec(nat)
  integer ,intent(in) :: attypes(nat)
  integer ,intent(in) :: nimage
  real(rk),intent(in) :: xcoords(3,nimage*nat)
  real(rk),intent(in) :: xweight(nat)
  real(rk),intent(in) :: mass(nat)
  real(rk),intent(out),optional :: spur
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

  if(.not.this%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_create")

  this%lhdlc=.true. ! this does not seem to have any effect ...

! ngroups: number of HDLC residues in hdlc%resn
! the variable hdlc%ngroups has to denote the number of actually allocated
!   residues
  j = 0
  ncart = 0
  ngroups = 0
  DO i = 1, nat
    IF (this%resn(i)==0) THEN
      ncart = ncart + 1
    else if (this%resn(i)<0) then
      cycle
    ELSE IF (this%resn(i)>j) THEN
      ngroups = ngroups + 1
      j = this%resn(i)
    END IF
  END DO

  call allocate(vconstr,max(1,this%nconstr))
  vconstr(:)=0.D0
  
  IF (printl>=6) WRITE (stdout,'(/,a,i4,a)') &
      'Generating new HDLC for ', ngroups, ' residues'

! loop over all groups
! icoords is ordered with inner region coords followed by
! outer region coords. ip (icoords position) must be set 
! accordingly.
  ipinner = 1
  ipouter = nicore + 1
  this%first=1
  nconst_sum=0
  do ires=1,ngroups
    ! Determine which region the residue is in by the 
    ! micspec value of the first atom in it
    ! NB: for standard (non-microiterative) optimisations, 
    ! all residues are considered to be 'inner'
    iregion = micspec(this%res(ires)%at(1))
    if (iregion == 1) then
       iphdlc = ipinner
    else
       iphdlc = ipouter
    end if
    ! Residues must not contain a mix of inner and outer atoms
    do i = 1, this%res(ires)%natom
       if (micspec(this%res(ires)%at(i)) /= iregion) then
          write (stdout,'(3x,a,i4,a)') 'Residue number ', ires, &
               ' crosses inner/outer boundary.'
          write (stdout,'("   Atoms: ",10i6)') this%res(ires)%at(:)
          call dlf_fail("HDLC residue crosses inner/outer boundary")
       end if
    end do
    this%res(ires)%ip=iphdlc
    iphdlc=iphdlc+this%res(ires)%natom * 3

    length=this%res(ires)%natom
! now we have group, istart, ifin
    istart=1
    ifin=1 !dummies to be removed - IMPROVE!
    IF (printl>=6) THEN
      WRITE (stdout,'(/,a)') 'Located a new residue for HDLC'
      WRITE (stdout,'(3x,a,i4)') 'Residue number is ', ires
      WRITE (stdout,'(3x,a,i5)') 'Number of atoms it contains ',length
      WRITE (stdout,'("   Atoms: ",10i6)') this%res(ires)%at(:)
    END IF
    IF (length<2) THEN
      CALL hdlc_errflag('No residue can contain less than two atoms', &
          'stop')
    END IF
    xyz = matrix_create(3*length,1,'XYZ')

    ! define res%xweight
    do iat=1,length
      this%res(ires)%xweight(iat)=xweight(this%res(ires)%at(iat))
    end do

    ! get connections for primitive internals
    IF (this%contyp==0) THEN

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
              xcoords(:, (iimage-1)*nat + this%res(ires)%at(iat))
        end do
        !CALL dummyarg_checkin(reshape(xcoords(:,(iimage-1)*nat+1:iimage*nat),&
        !    (/3*nat/)),3*(istart-1)+1,3*length)
        idum = matrix_set(xyz,size(atmp),atmp)
        !CALL dummyarg_clear
        call deallocate(atmp)

        !CALL dummyarg_checkin(attypes,istart,length)
        call allocate(atmpi,length)
        do iat=1,length
          atmpi(iat)=attypes(this%res(ires)%at(iat))
        end do
        CALL connect_prim(length,maxcon,atmpi,nconn,iconn,xyz)
        call deallocate(atmpi)
        !CALL dummyarg_clear
      end do

      ! check in user connections and create connections matrix
      CALL ci_conn(con,nconn,iconn,this%nincon,this%incon,length,&
          this%res(ires)%at)
      DEALLOCATE (iconn) ! pointer

      ! check in constraints
      CALL assign_cons(cns,this%iconstr,this%nconstr,vconstr, &
          1,length,this%res(ires)%at,this%internal,nconst_here)
      iphdlc=iphdlc-nconst_here

      ! create the HDLC - primitives
      if(present(spur))then
        CALL this%hdlc_create(this%res(ires),xyz,con,cns,ires,spur)
      else
        CALL this%hdlc_create(this%res(ires),xyz,con,cns,ires)
      endif
      idum = int_matrix_destroy(con)
      idum = matrix_destroy(cns)

      ! create the HDLC - total connection scheme - no check if only stretch constr.
    ELSE ! (ctrl%contyp.eq.0 ...)

      !check in atom coordinates
      call allocate(atmp,3*length)
      do iat=1,length
        atmp((iat-1)*3+1:(iat-1)*3+3)=xcoords(:,this%res(ires)%at(iat))
      end do
      idum = matrix_set(xyz,size(atmp),atmp)
      call deallocate(atmp)

      !CALL dummyarg_checkin(reshape(xcoords,(/3*nat/)),3*(istart-1)+1,3*length)
      !idum = matrix_set(xyz,dummyarg)
      !CALL dummyarg_clear

      CALL assign_cons(cns,this%iconstr,this%nconstr,vconstr, &
          1,length,this%res(ires)%at,this%internal,nconst_here)
      iphdlc=iphdlc-nconst_here
      if(present(spur)) then
        CALL this%hdlc_create(this%res(ires),xyz,con,cns,ires,spur)
      else
        CALL this%hdlc_create(this%res(ires),xyz,con,cns,ires)
      endif
      idum = matrix_destroy(cns)
    END IF ! (ctrl%contyp.eq.0 ...)

! restore the error counter
    this%res(ires)%err_cnt = this%err_cnt(ires)
    idum = matrix_destroy(xyz)
    nconst_sum=nconst_sum+nconst_here

    ! Write out mapping of HDLCs to internal coordinates
    ! Useful for identifying which HDLC the max grad component 
    ! corresponds to during an optimisation
    if (printl >= 6) then
       write(stdout, '("   Residue ", i6, " has internal coords ", i6, &
            " to ", i6)') ires, this%res(ires)%ip, iphdlc-1
    end if

    ! update inner or outer coords position marker
    if (iregion == 1) then
       ipinner = iphdlc
    else
       ipouter = iphdlc
    end if

  END DO ! ires=1,ngroups
  call deallocate(vconstr)

  this%nconstr=nconst_sum
  
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
subroutine dlf_hdlc_getweight(this,nat,nivar,nicore,micspec,xweight,iweight)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout  
  implicit none
  class(hdlc_ctrl), intent(inout) :: this
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
  if(.not.this%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_getweight")

  iweight(:)=0.D0
  ipinner = 1
  ipouter = nicore + 1
  DO group = 1, this%ngroups
 
    ! ignore deleted residues
    if(this%res(group)%name==-1) cycle
    residue=this%res(group)
    iphdlc = residue%ip 
    length = residue%natom

    if(.not.allocated(this%res(group)%iweight%data)) then
      call dlf_fail("No weights present in dlf_hdlc_getweight")
    end if

    !ndfhdlc = 3*length - residue%ncons ! orig
    inta2=shape(this%res(group)%iweight%data)
    ndfhdlc=inta2(1)

    iweight(iphdlc:iphdlc+ndfhdlc-1)=reshape(  &
        this%res(group)%iweight%data(:,:),(/ndfhdlc/))
    ! for correct start of cartesians
    if (iphdlc <= nicore) then
       ipinner = ipinner + ndfhdlc-this%nmin
    else 
       ipouter = ipouter + ndfhdlc-this%nmin
    end if

  end DO

  DO iat = 1, nat
    IF (this%resn(iat)==0) THEN
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
subroutine dlf_hdlc_xtoi(this,nat,nivar,nicore,micspec,xcoords,xgradient,&
     icoords,igradient,spur)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout  
  use dlf_allocate, only: allocate,deallocate
  implicit none
  class(hdlc_ctrl), intent(inout) :: this
  integer ,intent(in) :: nat
  integer ,intent(in) :: nivar
  integer, intent(in) :: nicore
  integer ,intent(in) :: micspec(nat)  
  real(rk),intent(in) :: xcoords(3,nat)
  real(rk),intent(in) :: xgradient(3,nat)
  real(rk),intent(out):: icoords(nivar)
  real(rk),intent(out):: igradient(nivar)
  real(rk),optional,intent(out) :: spur

  TYPE (residue_type) :: residue
  TYPE (matrix)       :: cns, xyz, cxyz, chdlc, gxyz, ghdlc
  integer             :: ndfcons,group,iphdlc,length,ndfhdlc
  integer             :: idum,j,iat,m,ndfopt
  integer             :: ipinner, ipouter
  real(rk),pointer    :: prim_tmp(:) 
  real(rk),allocatable:: atmp(:)
! **********************************************************************
  if(.not.this%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_xtoi")

  IF(this%NGROUPS<1) call dlf_fail("Number of fragements in HDLC must be >0")

  IF (printl>=6) WRITE (stdout,'(/,A)') &
      'Converting Cartesians to HDLC'
  ndfcons = 0

  ipinner = 1
  ipouter = nicore + 1

  DO group = 1, this%ngroups
 
    ! ignore deleted residues
    if(this%res(group)%name==-1) cycle
    residue=this%res(group)

    iphdlc = residue%ip 
    ndfcons = ndfcons + residue%ncons
    length = residue%natom
    ndfhdlc = 3*length - residue%ncons

! now we have group, istart, ifin, iphdlc, ndfhdlc; convert coords to HDLC
    cxyz = matrix_create(3*length,1,'CXYZ')
    chdlc = matrix_create(3*length-this%nmin,1,'CHDLC')
    call allocate(atmp,3*length)
    do iat=1,length
      atmp((iat-1)*3+1:(iat-1)*3+3)= xcoords(:,residue%at(iat))
    end do
    idum = matrix_set(cxyz,size(atmp),atmp)

    CALL this%coord_cart_to_hdlc(residue,cxyz,chdlc,prim_tmp,.FALSE.)

! convert gradient to HDLC
    gxyz = matrix_create(3*length,1,'GXYZ')
    ghdlc = matrix_create(3*length-this%nmin,1,'GHDLC')
    do iat=1,length
      atmp((iat-1)*3+1:(iat-1)*3+3)= xgradient(:,residue%at(iat))
    end do
    !CALL dummyarg_checkin(reshape(xgradient,(/3*nat/)),3*(istart-1)+1,3*length)
    idum = matrix_set(gxyz,size(atmp),atmp)
    call deallocate(atmp)
    !CALL dummyarg_clear
    if(present(spur)) then
      CALL this%grad_cart_to_hdlc(residue,cxyz,gxyz,ghdlc,spur)
    else
      CALL this%grad_cart_to_hdlc(residue,cxyz,gxyz,ghdlc)
    endif

! separate between active space and constraints - resize CHDLC and GHDLC
    IF (residue%ncons/=0) THEN
      CALL this%hdlc_split_cons(residue,chdlc,.true.)
      CALL this%hdlc_split_cons(residue,ghdlc,.FALSE.)
    END IF

! set icoords and igradient, size of chdlc/ghdlc now: ndfhdlc
    idum = matrix_get(chdlc,ndfhdlc-this%nmin,&
        icoords(iphdlc:iphdlc+ndfhdlc-this%nmin-1))
    idum = matrix_get(ghdlc,ndfhdlc-this%nmin,&
        igradient(iphdlc:iphdlc+ndfhdlc-this%nmin-1))

! prepare for the next group
    idum = matrix_destroy(cxyz)
    idum = matrix_destroy(gxyz)
    idum = matrix_destroy(chdlc)
    idum = matrix_destroy(ghdlc)

    ! for correct start of cartesians
    if (iphdlc <= nicore) then
       ipinner = ipinner + ndfhdlc-this%nmin
    else
       ipouter = ipouter + ndfhdlc-this%nmin
    end if

  END DO ! group = 1,hdlc%ngroups

! check in Cartesians - no cartesian constraints implemented!
  j = 0
  DO iat = 1, nat
    IF (this%resn(iat)==0) THEN
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
!!****f* hdlc/dlf_hdlc_itox
!!
!! FUNCTION
!!
!!  convert HDLC -> Cartesian
!!
!! SYNOPSIS
subroutine dlf_hdlc_itox(this,nat,nivar,nicore,micspec,icoords,xcoords,tok)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout  
  use dlf_allocate, only: allocate, deallocate
  implicit none
  class(hdlc_ctrl), intent(inout) :: this
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

  if(.not.this%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_itox")

  IF(this%NGROUPS<1) call dlf_fail("Number of fragemnts in HDLC must be >0")

  tok=.true.
  this%interror=.false.
  IF (printl>=6) WRITE (stdout,'(/,A)') &
      'Converting HDLC to Cartesians'

  ipinner = 1
  ipouter = nicore + 1

! loop over all residues
  this%ngroupsdrop = 0
  ndfcons = 0
  DO group = 1, this%ngroups
    residue = this%res(group)
    ! ignore deleted residues
    if(this%res(group)%name==-1) cycle

    iphdlc = residue%ip 
    ndfcons = ndfcons + residue%ncons
    length = residue%natom
    ndfhdlc = 3*length - residue%ncons

! now we have group, istart, ifin, iphdlc, ndfhdlc; check out HDLC coordinates
    chdlc = matrix_create(ndfhdlc-this%nmin,1,'CHDLC')
    cxyz = matrix_create(3*length,1,'CXYZ')
    !CALL dummyarg_checkin(icoords,iphdlc,ndfhdlc-hdlc%nmin)
    idum = matrix_set(chdlc,ndfhdlc-this%nmin,icoords(iphdlc:iphdlc+ndfhdlc-this%nmin-1))
    call allocate(atmp,3*residue%natom)
    do iat=1,length
      atmp((iat-1)*3+1:(iat-1)*3+3)= xcoords(:,residue%at(iat))
    end do
    idum = matrix_set(cxyz,size(atmp),atmp) !xcoords(:,istart:istart+residue%natom-1))

! restore values of constrained variables if required
    IF (residue%ncons/=0) THEN
      CALL this%hdlc_rest_cons(residue,chdlc)
    END IF

! fit Cartesian coordinates to HDLC coordinates; size of chdlc now: 3*length
    CALL this%coord_hdlc_to_cart(residue,cxyz,chdlc)

! conversion HDLC -> Cartesian failed due to singular G matrix
    IF ( .NOT. residue%lgmatok) THEN
      tok=.false.
      residue%err_cnt = residue%err_cnt + 1000

      IF (printl>=6) WRITE (stdout,'(3X,A,I4,A,I4,/)') &
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
          this%interror = .TRUE.
          IF (printl>=6) WRITE (stdout,'(A,I4,A,I4)') &
              'Warning: could not remove residue', residue%name, &
              ', number of constraints: ', residue%ncons
        ELSE
          this%ngroupsdrop = this%ngroupsdrop + 1
          IF (printl>=6) WRITE (stdout,'(A,I4,A)') &
              'Cyclic failure - removing residue ', residue%name, &
              ' from list'
          DO i = 1,length
            this%resn(residue%at(i)) = -10 ! changed from -2 to -10
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
       ipinner = ipinner + ndfhdlc-this%nmin
    else
       ipouter = ipouter + ndfhdlc-this%nmin
    end if
    idum = matrix_destroy(cxyz)
    idum = matrix_destroy(chdlc)
    call deallocate(atmp)
    this%res(group) = residue 
  END DO ! (group = 1,ngroups)

! check in Cartesians
  DO iatom = 1, nat
    IF (this%resn(iatom)==0) THEN
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
subroutine dlf_hdlc_reset(this)
!! SOURCE
  use dlf_global, only:glob

  implicit none
  class(hdlc_ctrl), intent(inout) :: this
  integer       :: iatom
! **********************************************************************
  if(.not.this%tinit) &
      call dlf_fail("HDLC not initialised in dlf_hdlc_reset")
  CALL this%hdlc_destroy_all(.TRUE.,size(this%err_cnt),this%err_cnt)
! if a residue has been removed, its number is set to -10 to prevent checkin
  DO iatom = 1, glob%nat ! <--- this is dangerous - improve!
    IF (this%resn(iatom)==-10) this%resn(iatom) = 0 ! this has to be checked ...
  END DO
  this%ngroups = this%ngroups - this%ngroupsdrop
  this%ngroupsdrop=0

  ! commented out as it leads to termination after the first breakdown
  !if (hdlc%ngroups==0.and.hdlc%internal) then
  !  call dlf_fail("Internal coordinates can not be restored. Use cartesians instead")
  !end if

  IF (this%interror) THEN
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
subroutine dlf_hdlc_destroy(this)
!! SOURCE
  use dlf_parameter_module, only: ik
  use dlf_global, only: stdout
  use dlf_allocate, only: deallocate
  implicit none
  class(hdlc_ctrl), intent(inout) :: this
  integer             :: fail,igroup
! **********************************************************************
  if(.not.this%tinit) return ! nothing to do 

  this%tinit=.false.

  if(this%lhdlc) then
    CALL this%hdlc_destroy_all(.TRUE.,size(this%err_cnt),this%err_cnt)
  end if

  ! deallocate arrays of individual residues
  do igroup=1,size(this%err_cnt)
    deallocate(this%res(igroup)%xweight) ! pointer
    deallocate(this%res(igroup)%at)      ! pointer
  end do

  ! deallocate hdlc arrays
  call deallocate(this%resn)
  call deallocate(this%err_cnt)
  call deallocate(this%iconstr)
  call deallocate(this%incon)
  call deallocate(this%bhdlc_matrix)
  ! deallocate residues - no storage check available
  ! this was previously commented out because of a bug in pgf90
  ! however not deallocating it causes problems with ifort, giving an
  ! error of array already allocated if an (h)dlc job is run twice in the
  ! same chemshell script. This code doesn't seem to cause pgf90 any 
  ! problems any more anyway (as of v9.0)
  deallocate(this%res,stat=fail)
  if(fail>0) then
    call dlf_fail("Deallocation error at residues")
  end if
  
end subroutine dlf_hdlc_destroy
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_hdlc_write(this)
  ! Write HDLC data to checkpoint file
  use dlf_checkpoint, only: tchkform
  implicit none
  class(hdlc_ctrl), intent(inout) :: this
  logical :: lerr
! **********************************************************************
  if(tchkform) then
    open(unit=100,file="dlf_hdlc.chk",form="formatted")
  else
    open(unit=100,file="dlf_hdlc.chk",form="unformatted")
  end if

  call hdlc_wr_hdlc(100,this,tchkform,lerr)

  close(100)

end subroutine dlf_checkpoint_hdlc_write

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_hdlc_read(this,lerr)
  ! Read HDLC data from checkpoint file

  use dlf_checkpoint, only: tchkform
  implicit none
  class(hdlc_ctrl), intent(inout) :: this
  logical ,intent(out) :: lerr
! **********************************************************************
  if(tchkform) then
    open(unit=100,file="dlf_hdlc.chk",form="formatted")
  else
    open(unit=100,file="dlf_hdlc.chk",form="unformatted")
  end if

  call hdlc_rd_hdlc(100,this,tchkform,lerr)

  close(100)

end subroutine dlf_checkpoint_hdlc_read
  
  

  SUBROUTINE hdlc_create(this,residue,xyz,con,cns,name,spur)
    use dlf_global
    ! create residue hdlc%res(name) 

! args
    class(hdlc_ctrl), intent(inout) :: this
    INTEGER name
    TYPE (residue_type), intent(inout) :: residue
    TYPE (matrix) :: xyz, cns
    TYPE (int_matrix) :: con
    real(rk),intent(out),optional :: spur

! local vars
    LOGICAL failed
    INTEGER idum, n6, ni,jdum
    TYPE (matrix) :: bprim, bhdlc, ighdlc
    ! for setting the bias
    real(rk),pointer    :: prim_tmp(:)
    TYPE (matrix) :: chdlc, primweight

! begin, do linked list stuff
    IF (printl>=6) THEN
      WRITE (stdout,'(3x,a,i4)') 'Generating HDLC for residue ', name
    END IF

    residue%name=name
    residue%next=-1
    IF (this%ngroups==0) THEN
      ! this is the first residue
      residue%prev=-1
    ELSE
      residue%prev=name-1
      this%res(name-1)%next=name
    END IF
    this%ngroups = this%ngroups + 1

! clear pointers of the new residue and other values
    NULLIFY (residue%x)
    NULLIFY (residue%y)
    NULLIFY (residue%z)
    NULLIFY (residue%iconn)
    NULLIFY (residue%ibend)
    NULLIFY (residue%irots)
    NULLIFY (residue%icons)
    NULLIFY (residue%vcons)
    NULLIFY (residue%biasv)
    idum=matrix_destroy(residue%ut)
    idum=matrix_destroy(residue%oldxyz)
    idum=matrix_destroy(residue%iweight)
    residue%ncons = 0
    residue%nconn = 0
    residue%nbend = 0
    residue%nrots = 0

! store residue information
    residue%name = name

! store the coordinates as components
    residue%natom = matrix_dimension(xyz,1)/3
    allocate (residue%x(residue%natom))
    allocate (residue%y(residue%natom))
    allocate (residue%z(residue%natom))
    CALL hdlc_linear_checkin(xyz,residue%natom,residue%x,residue%y,residue%z)

! if there are connections - fill in connections
    IF (allocated(con%data)) THEN
      residue%nconn = int_matrix_dimension(con,2)
      allocate (residue%iconn(2,max(1,residue%nconn)))
      CALL hdlc_con_checkin(con,residue%nconn,residue%iconn)

! fill in connections required for constraints
      IF (allocated(cns%data)) THEN
        residue%ncons = matrix_dimension(cns,2)
        allocate (residue%icons(6,max(1,residue%ncons)))
        allocate (residue%vcons(max(1,residue%ncons)))
        CALL ci_cons(cns,residue%ncons,residue%icons,residue%vcons, &
          residue%nconn,residue%iconn)
      else
        ! allocate constraint data nontheless
        allocate (residue%icons(6,max(1,residue%ncons)))
        allocate (residue%vcons(max(1,residue%ncons)))
      END IF


! generate angular primitives from connectivity
      CALL valcoor(residue%nconn,residue%iconn,residue%natom,residue%x, &
        residue%y,residue%z,residue%nbend,residue%ibend,residue%nrots, &
        residue%irots)

! if there are no connections yet - apply total connection scheme
    ELSE
      NULLIFY (residue%iconn)
      CALL connect_all(residue%natom,residue%nconn,residue%iconn)

      ! allocate constraint data nontheless
      allocate (residue%icons(6,max(1,residue%ncons)))
      allocate (residue%vcons(max(1,residue%ncons)))
    END IF ! (associated(con))

! done connections, now do angular constraints
    IF (residue%ncons>0) THEN
      CALL ck_angle_cons(residue%ncons,residue%icons,residue%nconn, &
        residue%ibend,residue%nbend,residue%irots,residue%nrots)
    END IF

! print out primitives - ugly patch for SGI machines
    IF ( .NOT. associated(residue%ibend)) allocate (residue%ibend(4,1))
    IF ( .NOT. associated(residue%irots)) allocate (residue%irots(4,1))
    CALL valcoor_print(residue%nconn,residue%iconn,residue%nbend, &
      residue%ibend,residue%nrots,residue%irots,residue%x,residue%y,residue%z, &
      residue%natom)

! generate the primitive B matrix
    CALL this%hdlc_make_bprim(residue%natom,residue%x,residue%y,residue%z, &
      residue%nconn,residue%iconn,residue%nbend,residue%ibend,residue%nrots, &
      residue%irots,residue%np,bprim,ni,residue%xweight,primweight)

! generate the Ut matrix
    CALL this%hdlc_make_ut(residue%ut,bprim)

! orthogonalise against constrained space
    IF (residue%ncons>0) THEN
      CALL ortho_cons(residue%ut,residue%ncons,residue%icons)
    END IF

! generate HDLC B matrix
    CALL this%hdlc_make_bhdlc(bprim,bhdlc,residue%ut)
    !this%bhdlc_matrix(:,:) = bhdlc%data(:,:)

    ! iweight_i = (%ut_ij)**2 * primweight_j
    residue%iweight = matrix_create( &
        matrix_dimension(bhdlc,1)-residue%ncons ,1,'i weight')
    !idum = matrix_multiply(1.0D0,residue%ut,primweight,0.0D0,residue%iweight)
    residue%iweight%data(:,:)=0.D0
    ! constrained hdlcs are at the end, and need not to be addressed here
    do idum=1,matrix_dimension(residue%ut,1)-residue%ncons ! number of hdlcs
      do jdum=1,matrix_dimension(residue%ut,2) ! number of primitives
        residue%iweight%data(idum,1)=residue%iweight%data(idum,1) + &
            residue%ut%data(idum,jdum)**2 * primweight%data(jdum,1)
      end do
    end do
    idum=matrix_destroy(primweight)

! generate HDLC inverse G matrix - only as check - set residue%lgmatok
    if(present(spur)) then
      CALL this%hdlc_make_ighdlc(bhdlc,ighdlc,failed,spur)
    else
      CALL this%hdlc_make_ighdlc(bhdlc,ighdlc,failed)
    endif
    CALL this%hdlc_report_failure(residue,failed,'create')

! here would be the initialisation of the internal biasing vector
    allocate (residue%biasv(residue%np))
    CALL vector_initialise(residue%biasv,residue%np,0.0D0)

! generate matrix holding the Cartesians of the previous step
    residue%oldxyz = matrix_create(3*residue%natom,1,'old XYZ')
    idum = matrix_copy(xyz,residue%oldxyz)

! initialise cyclic failure counter
    residue%err_cnt = 0

    residue%tbias=.false.
    ! transform coordinates to hdlc once to set bias
    chdlc = matrix_create(3*residue%natom-this%nmin,1,'CHDLC tmp')
    call this%coord_cart_to_hdlc(residue,xyz,chdlc,prim_tmp,.false.)
    idum = matrix_destroy(chdlc)

! clean up
    idum = matrix_destroy(bprim)
    idum = matrix_destroy(bhdlc)
    idum = matrix_destroy(ighdlc)

! end
  END SUBROUTINE hdlc_create

!------------------------------------------------------------------------------
! subroutine hdlc_destroy
!
! method: destry
!
! Destroys a HDLC residue
!------------------------------------------------------------------------------

  SUBROUTINE hdlc_destroy(this,residue)

! args
    class(hdlc_ctrl), intent(inout) :: this
    TYPE (residue_type),intent(inout) :: residue !, first, last

! local vars
    INTEGER idum

! begin
    IF (printl>=5) THEN
      WRITE (stdout,'(a,i4,/)') 'Destroying HDLC residue ', residue%name
    END IF

    ! care about linked list
    IF (residue%next/=-1) THEN
      IF (residue%prev/=-1) THEN ! not first, not last
        this%res(residue%prev)%next=residue%next
        this%res(residue%next)%prev=residue%prev
      ELSE ! first, not last
        this%first=residue%next
        this%res(residue%next)%prev=-1
      END IF
    ELSE ! not first, last
      IF (residue%prev/=-1) THEN
        this%last = residue%prev
        this%res(residue%prev)%next=-1
      ELSE ! first, last
        this%first=-1
        this%last=-1
      END IF
    END IF
    ! destroy storage
    IF (associated(residue%x)) deallocate (residue%x)
    IF (associated(residue%y)) deallocate (residue%y)
    IF (associated(residue%z)) deallocate (residue%z)
    IF (associated(residue%iconn)) deallocate (residue%iconn)
    IF (associated(residue%ibend)) deallocate (residue%ibend)
    IF (associated(residue%irots)) deallocate (residue%irots)
    IF (associated(residue%icons)) deallocate (residue%icons)
    IF (associated(residue%vcons)) deallocate (residue%vcons)
    IF (associated(residue%biasv)) deallocate (residue%biasv)

    ! To me this seems like a compiler bug: if i destroy residue%ut with g95,
    ! hdlc%res(residue%name)%ut is still allocated. Thus we destroy that directly
    ! this was the case, when hdlc_destroy was called with a residue that was not
    ! mapped back to hldc%res(i)
    idum = matrix_destroy(residue%ut)
    !idum = matrix_destroy(hdlc%res(residue%name)%ut)
    idum=matrix_destroy(residue%oldxyz)
    !idum=matrix_destroy(hdlc%res(residue%name)%oldxyz)
    idum=matrix_destroy(residue%iweight)
    !idum=matrix_destroy(hdlc%res(residue%name)%iweight)

    this%ngroups = this%ngroups - 1
    residue%name=-1

    ! residue%at and residue%xweight are deallocated in dlf_hdlc_destroy

  END SUBROUTINE hdlc_destroy

!------------------------------------------------------------------------------
! subroutine coord_cart_to_hdlc
!
! method: getint
!
! Convert Cartesian coordinates to HDLC
!
! Note: if this subroutine is called by coord_hdlc_to_cart,
! - set lback to .true.
! - prim is preallocated on call, prim is set with primitives on return
! - the Cartesians are not stored to res%oldxyz
!------------------------------------------------------------------------------

  SUBROUTINE coord_cart_to_hdlc(this,res,xyz,chdlc,prim,lback)

! args
    class(hdlc_ctrl), intent(inout) :: this
    LOGICAL lback
    REAL (rk), DIMENSION (:), POINTER :: prim
    TYPE (residue_type) :: res
    TYPE (matrix) :: xyz, chdlc

! local params
    REAL (rk) one
    PARAMETER (one=1.0D0)

! local vars
    INTEGER i, idum, j, n6, nip
    REAL (rk) a1, a2, dx, dy, dz, fact, r
    REAL (rk), DIMENSION (:), ALLOCATABLE :: x, y, z, p, iut

! begin, the following exception should never occur
    IF (res%natom/=matrix_dimension(xyz,1)/3) THEN
      WRITE (stdout,'(A,I4,A,I4,A,I4)') 'Residue ', res%name, ', natom: ', &
        res%natom, '; coordinates, natom: ', matrix_dimension(xyz,1)/3
      CALL hdlc_errflag('Size mismatch','abort')
    END IF

! allocate temporary space and separate Cartesian components
    call allocate (x,res%natom)
    call allocate (y,res%natom)
    call allocate (z,res%natom)
    call allocate (p,3*res%natom)
    CALL hdlc_linear_checkin(xyz,res%natom,x,y,z)

! init memory for primitive internals and rows of UT
    IF ( .NOT. lback) allocate (prim(res%np))
    call allocate (iut,res%np)
    nip = 0
    IF (this%internal) THEN
      n6 = 3*res%natom - 6
    ELSE
      n6 = 3*res%natom
    END IF

! set up scaling factor for Cartesians
!!$    IF (ctrl%cfact==0.0D0) THEN
      fact = 1.0D0/real(res%natom,rk)
      fact=1.d0 ! hardcoded at the moment
!!$    ELSE IF (ctrl%cfact<0.0D0) THEN
!!$      fact = -ctrl%cfact/real(res%natom,rk)
!!$    ELSE
!!$      fact = ctrl%cfact
!!$    END IF

! compute all stretches
    DO i = 1, res%nconn
      nip = nip + 1
      dx = x(res%iconn(1,i)) - x(res%iconn(2,i))
      dy = y(res%iconn(1,i)) - y(res%iconn(2,i))
      dz = z(res%iconn(1,i)) - z(res%iconn(2,i))
      r = sqrt(dx*dx+dy*dy+dz*dz)
      prim(nip) = r
    END DO

! compute all bends
    DO i = 1, res%nbend
      IF (res%ibend(1,i)>0) THEN
        nip = nip + 1

! bends: simple bend
        IF (res%ibend(4,i)==0) THEN
          prim(nip) = vangle(x(res%ibend(1,i)),x(res%ibend(2, &
            i)),x(res%ibend(3,i)),y(res%ibend(1,i)),y(res%ibend(2, &
            i)),y(res%ibend(3,i)),z(res%ibend(1,i)),z(res%ibend(2, &
            i)),z(res%ibend(3,i)))

! bends: dihedrals - impropers??
        ELSE IF (res%ibend(4,i)>0) THEN
          prim(nip) = vdihedral(x(res%ibend(1,i)),y(res%ibend(1, &
            i)),z(res%ibend(1,i)),x(res%ibend(2,i)),y(res%ibend(2, &
            i)),z(res%ibend(2,i)),x(res%ibend(3,i)),y(res%ibend(3, &
            i)),z(res%ibend(3,i)),x(res%ibend(4,i)),y(res%ibend(4, &
            i)),z(res%ibend(4,i)))
!          prim(nip) = prim(nip) + res%biasv(nip) ! JK was
          if(.not.res%tbias) then
            res%biasv(nip)=prim(nip)
            !hdlc%res(res%name)%biasv(nip)=prim(nip)
          else
            ! set as close as possible to bias ...
            if(prim(nip)-res%biasv(nip) > pi )  prim(nip)=prim(nip)- 2.D0 * pi
            if(prim(nip)-res%biasv(nip) < -pi ) prim(nip)=prim(nip)+ 2.D0 * pi
            ! now bprim should be within pi to bias, if not terminate
            if(prim(nip)-res%biasv(nip) > pi ) then
              print*,res%biasv(nip)
              print*,prim(nip)
              call dlf_fail("HDLC Bias problem")
            end if
            if(prim(nip)-res%biasv(nip) < -pi) then
              print*,res%biasv(nip)
              print*,prim(nip)
              call dlf_fail("HDLC Bias problem")
            end if
            if(printl >= 6 .and. abs(prim(nip)-res%biasv(nip)) > 3.D0) then
              write(stdout,"('Warning, Torsion or improper number ',i4,&
                  &' differs by more than 3.0 from bias. Troubles likely')") nip
              write(stdout,"('Improper=',es15.7,' Bias=',es15.7)") &
                  prim(nip),res%biasv(nip)
            end if
          end if

! bends: linear bends (two bends in each of two planes, see hdlc_make_bprim)
        ELSE IF (res%ibend(4,i)<=-1) THEN
          a1 = vangle(x(res%ibend(1,i)),x(res%ibend(2,i)),x(res%ibend(2, &
            i)),y(res%ibend(1,i)),y(res%ibend(2,i)),y(res%ibend(2, &
            i))+one,z(res%ibend(1,i)),z(res%ibend(2,i)),z(res%ibend(2,i)))
          a2 = vangle(x(res%ibend(2,i)),x(res%ibend(2,i)),x(res%ibend(3, &
            i)),y(res%ibend(2,i))+one,y(res%ibend(2,i)),y(res%ibend(3, &
            i)),z(res%ibend(2,i)),z(res%ibend(2,i)),z(res%ibend(3,i)))
!            prim(nip) = a1 + a2
          prim(nip) = y(res%ibend(2,i))
          nip = nip + 1
          IF (res%ibend(4,i)==-1) THEN
            a1 = vangle(x(res%ibend(1,i)),x(res%ibend(2,i)),x(res%ibend(2, &
              i)),y(res%ibend(1,i)),y(res%ibend(2,i)),y(res%ibend(2, &
              i)),z(res%ibend(1,i)),z(res%ibend(2,i)),z(res%ibend(2,i))+one)
            a2 = vangle(x(res%ibend(2,i)),x(res%ibend(2,i)),x(res%ibend(3, &
              i)),y(res%ibend(2,i)),y(res%ibend(2,i)),y(res%ibend(3, &
              i)),z(res%ibend(2,i))+one,z(res%ibend(2,i)),z(res%ibend(3,i)))
          ELSE
            a1 = vangle(x(res%ibend(1,i)),x(res%ibend(2,i)),x(res%ibend(2, &
              i))+one,y(res%ibend(1,i)),y(res%ibend(2,i)),y(res%ibend(2, &
              i)),z(res%ibend(1,i)),z(res%ibend(2,i)),z(res%ibend(2,i)))
            a2 = vangle(x(res%ibend(2,i))+one,x(res%ibend(2,i)),x(res%ibend(3, &
              i)),y(res%ibend(2,i)),y(res%ibend(2,i)),y(res%ibend(3, &
              i)),z(res%ibend(2,i)),z(res%ibend(2,i)),z(res%ibend(3,i)))
          END IF
!            prim(nip) = a1 + a2
          IF (res%ibend(4,i)==-1) THEN
            prim(nip) = z(res%ibend(2,i))
          ELSE
            prim(nip) = x(res%ibend(2,i))
          END IF
        END IF ! (res%ibend(4,i) .eq. 0) ... else if ...
      END IF ! (res%ibend(4,i) .eq. 0)
    END DO ! (i = 1,res%nbend)

! dihedrals
    DO i = 1, res%nrots
      nip = nip + 1
      prim(nip) = vdihedral(x(res%irots(1,i)),y(res%irots(1,i)),z(res%irots(1, &
        i)),x(res%irots(2,i)),y(res%irots(2,i)),z(res%irots(2, &
        i)),x(res%irots(3,i)),y(res%irots(3,i)),z(res%irots(3, &
        i)),x(res%irots(4,i)),y(res%irots(4,i)),z(res%irots(4,i)))
!      prim(nip) = prim(nip) + res%biasv(nip) ! JK was
      if(.not.res%tbias) then
        res%biasv(nip)=prim(nip)
        !hdlc%res(res%name)%biasv(nip)=prim(nip)
      else
        ! set as close as possible to bias ...
        if(prim(nip)-res%biasv(nip) > pi )  prim(nip)=prim(nip)- 2.D0 * pi
        if(prim(nip)-res%biasv(nip) < -pi ) prim(nip)=prim(nip)+ 2.D0 * pi
        ! now bprim should be within pi to bias, if not terminate
        if(prim(nip)-res%biasv(nip) > pi ) then
          print*,res%biasv(nip)
          print*,prim(nip)
          call dlf_fail("HDLC Bias problem")
        end if
        if(prim(nip)-res%biasv(nip) < -pi) then
          print*,res%biasv(nip)
          print*,prim(nip)
          call dlf_fail("HDLC Bias problem")
        end if
        if(printl >= 6 .and. abs(prim(nip)-res%biasv(nip)) > 3.D0) then
          write(stdout,"('Warning, Torsion or improper number ',i4,&
              &' differs by more than 3.0 from bias. Troubles likely')") nip
              write(stdout,"('Improper=',es15.7,' Bias=',es15.7)") &
                  prim(nip),res%biasv(nip)
        end if
      end if
    END DO

    res%tbias=.true. ! now, bias is set - this does not seem to work!
    ! to set a residue variable, we have to adress it via hdlc%res(res%name)
    this%res(res%name)%tbias=.true. 

! Cartesians if required
    IF ( .NOT. this%internal) THEN
      DO i = 1, res%natom
        nip = nip + 1
        prim(nip) = x(i)*fact
        nip = nip + 1
        prim(nip) = y(i)*fact
        nip = nip + 1
        prim(nip) = z(i)*fact
      END DO
    END IF

! if nip is not res%np, something went really wrong
    IF (nip/=res%np) THEN
      WRITE (stdout,'(/,A,I4,A,I4)') 'Error, nip: ', nip, ', np: ', res%np
      CALL hdlc_errflag('Error converting Cartesians to HDLC','stop')
    END IF

!//////////////////////////////////////////////////////////////////////////////
! the primitives are available in prim(1..nip) at this point
!//////////////////////////////////////////////////////////////////////////////
    ! print primitives
    IF ( .NOT. lback .and. printl >= 6) then
      DO i = 1,res%np
        write(stdout,'("Primitive ",i4," = ",es15.6," Bias = ",es15.6)') &
            i,prim(i),res%biasv(i)
      end DO
    end IF

! to find HDLC, multiply each row of UT by the primitive internals and sum up
    DO i = 1, n6
      p(i) = 0.0D0
      idum = matrix_get_row(res%ut,size(iut),iut,i)
      DO j = 1, res%np
        p(i) = p(i) + iut(j)*prim(j)
      END DO
!      write (stdout,'(a,i4,a,f20.14)') 'HDLC coordinate ', i, ': ', p(i)
    END DO

! if in internals, set last 6 to zero
    IF (this%internal) THEN
      DO i = n6 + 1, n6 + 6
        p(i) = 0.0D0
      END DO
    END IF

! free memory for primitive internals and rows of UT
    call deallocate (iut)
    IF ( .NOT. lback) deallocate (prim)

! store HDLC to the returned matrix and hold Cartesians in the residue
    idum = matrix_set(chdlc,size(p),p)
    IF ( .NOT. lback) idum = matrix_copy(xyz,res%oldxyz)

! clean up
    call deallocate (p)
    call deallocate (z)
    call deallocate (y)
    call deallocate (x)

  END SUBROUTINE coord_cart_to_hdlc

!------------------------------------------------------------------------------
! subroutine grad_cart_to_hdlc
!
! method: intgrd
!
! Convert Cartesian gradient to HDLC
!------------------------------------------------------------------------------

  SUBROUTINE grad_cart_to_hdlc(this,res,xyz,gxyz,ghdlc,spur)
	use dlf_global
! args
    class(hdlc_ctrl), intent(inout) :: this
    TYPE (residue_type) :: res
    TYPE (matrix) :: xyz, gxyz, ghdlc
    real(rk),optional,intent(out) :: spur

! local vars
    LOGICAL failed
    INTEGER i, idum, n, n6, ni
    REAL (rk), DIMENSION (:), ALLOCATABLE :: x, y, z
    REAL (rk), DIMENSION (:,:), POINTER :: ghdlc_dat
    TYPE (matrix) :: bprim, bhdlc, bthdlc, ighdlc, primweight

! begin, the following exception should never occur
    IF (res%natom/=matrix_dimension(gxyz,1)/3) THEN
      WRITE (stdout,'(A,I4,A,I4,A,I4)') 'Residue ', res%name, ', natom: ', &
        res%natom, '; coordinates, natom: ', matrix_dimension(gxyz,1)/3
      CALL hdlc_errflag('Size mismatch','abort')
    END IF

! allocate temporary space and separate Cartesian components
    call allocate (x,res%natom)
    call allocate (y,res%natom)
    call allocate (z,res%natom)
    CALL hdlc_linear_checkin(xyz,res%natom,x,y,z)

! number of HDLC internals: n6, number of Cartesians: n 
    n = res%natom*3
    IF (this%internal) THEN
      n6 = n - 6
    ELSE
      n6 = n
    END IF

! generate a primitive B matrix (force generation of a new matrix)
    idum = matrix_destroy(bprim)
    CALL this%hdlc_make_bprim(res%natom,x,y,z,res%nconn,res%iconn,res%nbend, &
      res%ibend,res%nrots,res%irots,res%np,bprim,ni,res%xweight,primweight)

    ! we don't need the weights here, so destroy primweight
    idum=matrix_destroy(primweight)

! generate delocalised B matrix (force again)
    idum = matrix_destroy(bhdlc)
    CALL this%hdlc_make_bhdlc(bprim,bhdlc,res%ut)
    this%bhdlc_matrix(:,:) = bhdlc%data(:,:)    
	!glob%b_hdlc = bhdlc%data
! generate HDLC inverse G matrix
    idum = matrix_destroy(ighdlc)
    if(present(spur)) then
      CALL this%hdlc_make_ighdlc(bhdlc,ighdlc,failed,spur)
    else
      CALL this%hdlc_make_ighdlc(bhdlc,ighdlc,failed)
    endif
    CALL this%hdlc_report_failure(res,failed,'intgrd')

! make HDLC Bt**-1
    bthdlc = matrix_create(n6,n,'HDLC_BT')
    idum = matrix_multiply(1.0D0,ighdlc,bhdlc,0.0D0,bthdlc)

! set HDLC gradient = (Bt**-1) * Cartesian gradient
    idum = matrix_multiply(1.0D0,bthdlc,gxyz,0.0D0,ghdlc)

! clean up
    idum = matrix_destroy(bprim)
    idum = matrix_destroy(bhdlc)
    idum = matrix_destroy(bthdlc)
    idum = matrix_destroy(ighdlc)
    call deallocate (z)
    call deallocate (y)
    call deallocate (x)
!   call timePrint('END-INTGRAD')

  END SUBROUTINE grad_cart_to_hdlc


!------------------------------------------------------------------------------
! subroutine coord_hdlc_to_cart
!
! method: getcrt
!
! Convert HDLC to Cartesian coordinates
!
! Arrays/matrices of Cartesians:
! xyz:                 Cartesians returned by this routine
! res%oldxyz:          Cartesians stored by coords_cart_to_hdlc
! res%x, res%y, res%z: Cartesians used to generate UT (by hdlc_create)
!------------------------------------------------------------------------------

  SUBROUTINE coord_hdlc_to_cart(this,res,xyz,chdlc)

! args
    class(hdlc_ctrl), intent(inout) :: this
    TYPE (residue_type) :: res
    TYPE (matrix) :: xyz, chdlc

! local params
    REAL (rk) poor
    PARAMETER (poor=1.0D-6)

! local vars
    LOGICAL failed, flipped
    INTEGER failc, i, idum, iter, j, k, n, n6, ni
    REAL (rk) absm, cexit, dx, dy, mag, trust
    REAL (rk), DIMENSION (:), ALLOCATABLE :: tmp_dif, x, y, z
    REAL (rk), DIMENSION (:), POINTER :: prim, oldprim, tmp_prim
    TYPE (matrix) :: dif, xdif, bthdlc, xyzbak, xyzback, olhdlc, bhdlc, &
      ighdlc, bprim, scrdif
    ! for printing redudant internals
    TYPE (matrix) :: umat,qmat,primweight

! begin, the following exception should never occur
    IF (res%natom/=matrix_dimension(xyz,1)/3) THEN
      WRITE (stdout,'(A,I4,A,I4,A,I4)') 'Residue ', res%name, ', natom: ', &
        res%natom, '; coordinates, natom: ', matrix_dimension(xyz,1)/3
      CALL hdlc_errflag('Size mismatch','abort')
    END IF

! this parameter needs to be set upon every call
    cexit = 1.0D-12

! say what we are doing
    IF (printl>=6) WRITE (stdout,'(7X,A,/)') 'Entering fitting algorithm'
    IF (printl>=6) THEN
      WRITE (stdout,'(5X,A,E8.1)') 'Initial maximum exit error in HDLC is ', &
        cexit
      WRITE (stdout,'(5X,A,E8.1)') &
        'Convergence considered poor if error greater than ', poor
    END IF

! n6 is the number of HDLC in the residue
    n = 3*res%natom
    IF (this%internal) THEN
      n6 = n - 6
    ELSE
      n6 = n
    END IF

! generate some work matrices
    dif = matrix_create(n6,1,'HDLC diff')
    xdif = matrix_create(n,1,'Cart diff')
    bthdlc = matrix_create(n6,n,'HDLC_BT')
    xyzbak = matrix_create(n,1,'Cart backup')
    ! xyzback is used in case of breakdown to copy back the original cartesians
    xyzback = matrix_create(n,1,'Cart store')
    olhdlc = matrix_create(n6,1,'old HDLC')

! allocate memory for primitives
    allocate (prim(res%np))
    allocate (oldprim(res%np))

! allocate arrays holding the Cartesian components
    call allocate (x,res%natom)
    call allocate (y,res%natom)
    call allocate (z,res%natom)

! intial guess of Cartesians are the input Cartesians - changed by JK,
    ! was the other way round
    idum = matrix_copy(xyz,res%oldxyz)

! backup old Carts in case the iterative B matrix method fails and least square
! SD fitting is used
    idum = matrix_copy(res%oldxyz,xyzbak)
    idum = matrix_copy(res%oldxyz,xyzback)

! set currect Carts to old Carts
    idum = matrix_copy(res%oldxyz,xyz)

! the following deallocations to force generation should never be necessary
    idum = matrix_destroy(bhdlc)
    idum = matrix_destroy(ighdlc)
    idum = matrix_destroy(bprim)

    ! This is to start from zero bias. I am not sure if this is good!
    !CALL vector_initialise(res%biasv,res%np,0.0D0)

    if(printl>=6.and..not.this%internal) then
      write(stdout,"('Fitting to non-redundant internals:')")
      idum=matrix_print(chdlc)
      write(stdout,"('Starting fit from cartesians:')")
      idum=matrix_print(xyz)
      write(stdout,"('Possible redundant internals:')")
      idum=matrix_copy(res%ut,umat)
      idum=matrix_transpose(umat)
      qmat = matrix_create(res%np,1,'Redudant internals')
      idum=matrix_multiply(1.D0,umat,chdlc,0.D0,qmat)
      idum=matrix_print(qmat)
      idum=matrix_destroy(qmat)
      idum=matrix_destroy(umat)
    end if

!//////////////////////////////////////////////////////////////////////////////
! Initial trust multiplier
!
! The algorithm makes non-linear fitting steps based on the B matrix.
! Often the step direction is OK, but the magnitude two large.
! The trust multiplier is used to reduce the step size upon step failure.
!//////////////////////////////////////////////////////////////////////////////

    trust = 1.0D0

! set iteration counter
    iter = 0

! set up fail counter, give up if ten consecutive steps fail to improve the fit
    failc = 0

!//////////////////////////////////////////////////////////////////////////////
! Initialise absolute error memory
!
! The absolute error is the measure by which the algorithm tests convergence,
! but the maximum element of the error vector is the exit criterion
!
! dx:   maximum of the absolute of the elements of the error vector
! absm: dx of the previous iteration
!
! dy:   absolute of the error vector (root of scalar product by itself)
! mag:  dy of the previous iteration
!//////////////////////////////////////////////////////////////////////////////

    absm = 0.0D0
    mag = 0.0D0

!//////////////////////////////////////////////////////////////////////////////
! start of main loop
!//////////////////////////////////////////////////////////////////////////////

100 CONTINUE
    CALL hdlc_linear_checkin(xyz,res%natom,x,y,z)
    k = 0
    flipped = .FALSE.

!//////////////////////////////////////////////////////////////////////////////
! Test if torsion flipped through 180 degrees:
!
! Torsion flip of 360 degrees due to identification of -179 degrees and 181
! degrees can cause failure of the non-linear fit
! trust radius would be > 180 degrees
! 
! The bias vector corrects for this
!//////////////////////////////////////////////////////////////////////////////

    DO WHILE (flipped .OR. k==0)
      k = k + 1
      IF (k>2) CALL hdlc_errflag('Code error in hdlclib$hdlc_cart_to_hdlc', &
        'abort')
      flipped = .FALSE.

! get HDLC from current Cartesians xyz
      CALL this%coord_cart_to_hdlc(res,xyz,olhdlc,prim,.TRUE.)
      IF (iter>0) THEN
        IF ( .NOT. this%internal) THEN
          j = res%np - res%natom*3
        ELSE
          j = res%np
        END IF
        ! commented out as bias is now the reference value of torsions
        ! and impropers at the first geometry. coord_cart_to_hdlc will
        ! set these primitives as close as possible to bias.
!!$        DO j = res%nconn + 1, j
!!$          IF (prim(j)>oldprim(j)+pi) THEN
!!$            res%biasv(j) = res%biasv(j) - pi*2.0D0
!!$            flipped = .TRUE.
!!$          ELSE IF (prim(j)<oldprim(j)-pi) THEN
!!$            res%biasv(j) = res%biasv(j) + pi*2.0D0
!!$            flipped = .TRUE.
!!$          END IF
!!$        END DO
      END IF ! (iter.gt.0)
    END DO ! (flipped .or. k.eq.0)
    tmp_prim => oldprim
    oldprim => prim
    prim => tmp_prim

! generate a primitive B matrix from current cartesians
    CALL this%hdlc_make_bprim(res%natom,x,y,z,res%nconn,res%iconn,res%nbend, &
      res%ibend,res%nrots,res%irots,res%np,bprim,ni,res%xweight,primweight)

    ! we don't need the weights here, so destroy primweight
    idum=matrix_destroy(primweight)

! generate HDLC B matrix
    CALL this%hdlc_make_bhdlc(bprim,bhdlc,res%ut)

! generate HDLC inverse G matrix
    CALL this%hdlc_make_ighdlc(bhdlc,ighdlc,failed)
    CALL this%hdlc_report_failure(res,failed,'getcrt')

! make HDLC Bt**-1
    idum = matrix_multiply(1.0D0,ighdlc,bhdlc,0.0D0,bthdlc)

! make [Bt**-1]t
    idum = matrix_transpose(bthdlc)

! get difference between input HDLC and HDLC of the current iteration (olhdlc)

    idum = matrix_copy(olhdlc,dif)
    idum = matrix_scale(dif,-1.0D0)
    idum = matrix_add(dif,dif,chdlc)
    dx = matrix_absmax(dif)
    dy = matrix_length(dif)

! relax limit after four and ten cycles
    IF (iter==4 .OR. iter==10) THEN
      cexit = cexit*1.0D2
      IF (printl>=6) WRITE (stdout,'(5x,a,e8.1)') &
        'Relaxing maximum exit error to ', cexit
    END IF

! report convergence info if requested
    IF (printl>=6) WRITE (stdout,'(5x,a,i3,a,g10.3,a,g10.3,a,f6.3)') &
      'Step ', iter, ', max error= ', dx, ', length of error= ', dy, &
      ', step scaling factor= ', trust
    CALL dlf_flushout

! set up on first iteration
    IF (iter==0) THEN
      absm = dx + 1.0D0
      mag = dy + 1.0D0
    END IF

! test maximum absolute element of difference vector for convergence
    IF (dx<cexit) THEN
      IF (mag<dy) THEN
        idum = matrix_copy(xyzbak,xyz)
        IF (printl>=6) WRITE (stdout,'(5x,a)') 'Recalling best step'
      END IF
      GO TO 200
    END IF

!//////////////////////////////////////////////////////////////////////////////
! *** NOT IMPLEMENTED BUT PLANNED ***
!
! if there are 10 unreasonably small steps 
! or a step that increases the error
! go over to a much more stable least squares fit method
!//////////////////////////////////////////////////////////////////////////////

! reject step if error increases
    IF (mag<dy) THEN
      IF (printl>=6) WRITE (stdout,'(/,5x,a,/)') 'Rejecting step'
      trust = trust*0.5D0
      idum = matrix_copy(xyzbak,xyz)
      IF (iter>2*n6) THEN
        IF (printl>=6) WRITE (stdout,'(/,5x,a,/)') &
          'Cannot make step - guessing'
        IF (dx>poor) res%lgmatok = .FALSE.
        GO TO 200
      END IF

! error is reduced - too slow convergence detects linear dependence
    ELSE
      IF (iter>2*n6) THEN
        IF (printl>=6) WRITE (stdout,'(/,5x,a,/)') &
          'Step converged too slowly'
        IF (dx>poor) res%lgmatok = .FALSE.
        GO TO 200
      END IF

! step accepted
      absm = dx
      mag = dy
      failc = 0
      trust = min(1.0D0,trust*1.25D0)

! record best position yet
      idum = matrix_copy(xyz,xyzbak)

! set xyz = oldxyz + [Bt**-1] * trust * [HDLC-HDLC(xyz(i))]
! if internal, create a scratch copy of dif with the last six elements missing
      IF (this%internal) THEN
        scrdif = matrix_create(n6,1,'scrdif')
        call allocate (tmp_dif,n)
        idum = matrix_get(dif,n,tmp_dif)
        idum = matrix_set(scrdif,n,tmp_dif)
        idum = matrix_multiply(1.0D0,bthdlc,scrdif,0.0D0,xdif)
        idum = matrix_scale(xdif,trust)
        idum = matrix_add(xyz,xdif,res%oldxyz)
        idum = matrix_destroy(scrdif)
        call deallocate (tmp_dif)
      ELSE
        idum = matrix_multiply(1.0D0,bthdlc,dif,0.0D0,xdif)
        idum = matrix_scale(xdif,trust)
        idum = matrix_add(xyz,xdif,res%oldxyz)
      END IF

! now xyz contains a guess - recompute the HDLC
    END IF ! (mag.lt.dy)
    idum = matrix_copy(xyz,res%oldxyz)
    idum = matrix_transpose(bthdlc)
    iter = iter + 1
    GO TO 100

!//////////////////////////////////////////////////////////////////////////////
! end of main loop - prepare for return 
!//////////////////////////////////////////////////////////////////////////////

200 CONTINUE

    ! back up cartesians in case of conversion failure
    if(.not.res%lgmatok) idum = matrix_copy(xyzback,xyz)

! if print level>=5, convergence has already been reported
    IF (printl==5) WRITE (stdout,'(5x,a,i3,a,/,5x,g10.3,a,/)') &
      'Converged Cartesians in ', iter, ' steps to ', dx, &
      ' maximum component of error vector'

    ! set bias to zero again.
    !CALL vector_initialise(res%biasv,res%np,0.0D0)

! clean up
    call deallocate (x)
    call deallocate (y)
    call deallocate (z)
    NULLIFY (tmp_prim)
    IF (associated(prim)) deallocate (prim)
    IF (associated(oldprim)) deallocate (oldprim)
    idum = matrix_destroy(dif)
    idum = matrix_destroy(xdif)
    idum = matrix_destroy(ighdlc)
    idum = matrix_destroy(bthdlc)
    idum = matrix_destroy(bhdlc)
    idum = matrix_destroy(bprim)
    idum = matrix_destroy(xyzbak)
    idum = matrix_destroy(xyzback)
    idum = matrix_destroy(olhdlc)

  END SUBROUTINE coord_hdlc_to_cart

!------------------------------------------------------------------------------
! subroutine hdlc_split_cons
!
! Wrapper to split_cons in library constraint: unwraps variables from residue.
! Separates the values of the constrained degrees of freedom and stores them
! to residue%vcons if required
!
! Arguments:
! residue:  residue (in)
! hdlc_mat: matrix of full HDLC coordinates (in)
!           matrix of active HDLC coordinates (out)
! lstore: constrained HDLC coordinates are stored to vcons if (lstore) (in)
!------------------------------------------------------------------------------

  SUBROUTINE hdlc_split_cons(this,residue,hdlc_mat,lstore)
    class(hdlc_ctrl), intent(inout) :: this
    LOGICAL lstore
    TYPE (residue_type) :: residue
    TYPE (matrix) :: hdlc_mat

    CALL split_cons(hdlc_mat,lstore,this%internal,residue%natom, &
        residue%ncons,residue%vcons)
  END SUBROUTINE hdlc_split_cons

!------------------------------------------------------------------------------
! subroutine hdlc_rest_cons
!
! Wrapper to rest_cons in library constraint: unwraps variables from residue.
! Restores the values of the constrained degrees of freedom from residue%vcons
!
! Arguments:
! residue:  residue (in)
! hdlc_mat: matrix of active HDLC coordinates (in)
!           matrix of all HDLC coordinates (out)
!------------------------------------------------------------------------------

  SUBROUTINE hdlc_rest_cons(this,residue,hdlc_mat)
    class(hdlc_ctrl), intent(inout) :: this
    TYPE (residue_type) :: residue
    TYPE (matrix) :: hdlc_mat

    CALL rest_cons(hdlc_mat,this%internal,residue%natom,residue%ncons,residue%vcons)
  END SUBROUTINE hdlc_rest_cons

!==============================================================================
! routines doing the calculations for the delocalisations
!==============================================================================

!//////////////////////////////////////////////////////////////////////////////
! subroutine hdlc_make_bprim
!
! Makes the B matrix for primitive internal coordinates
!
! On input
! ========
! n  (int)           = number of atoms
! x,y,z (dble)       = cartesian co-ordiantes
! nbend (int)        = number of bends (this is linears and wags as well)
! nrots (int)        = number of rotations
! nconn (int)        = number of connections
! bprim              = pointer to a matrix
!
! if bprim is not associated, a new matrix is made
! if bprim is associated, it is expected to point to a previous matrix
!
! iconn(nconn) (int) = connections array
!                      a,b a-b connection
!
! ibend(nbend) (int) = array from valcoor
!                      a,b,c,0  bend a-b-c
!                      a,b,c,-1 line relative to yz plane
!                      a,b,c,-2 line relative to xy plane
!                      a,b,c,d  dihedral a-b-c-d
!                      (the dihedral is no longer supported by valcoor
!                      but is here for user defined coords)
!
! irots(nrots) (int) = array from valcoor
!                      a,b,c,d  dihedral about b-c
!
! On output
! =========
! ni (int)     = number of primitive internal coordinates
! bprim        = pointer to primitive redundant B matrix
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE hdlc_make_bprim(this,natom,x,y,z,nconn,iconn,nbend,ibend,nrots,irots, &
      np,bprim,ni,xweight,primweight)

! args
    class(hdlc_ctrl), intent(inout) :: this
    INTEGER,intent(in) ::  natom
    INTEGER            :: nconn, nbend, nrots, np, ni
    INTEGER            :: iconn(2,nconn), ibend(4,nbend), irots(4,nrots)
    REAL (rk)          :: x(natom), y(natom), z(natom)
    TYPE (matrix)      :: bprim,primweight
    real (rk)          :: xweight(natom)

! local params
    INTEGER ib(4), noint, k, idum
    REAL (rk) :: cutoff
    PARAMETER (cutoff=1.0D-30)

! local vars
    LOGICAL lbumout
    INTEGER i, j
    REAL (rk), DIMENSION (:,:), ALLOCATABLE :: brow
    REAL (rk) :: b(3,4), d_zero, d_one, fact, cz(12)

! begin
    DATA lbumout/ .FALSE./
    DATA d_zero, d_one/0.0D0, 1.0D0/

    IF (printl>=6) WRITE (stdout,'(5X,A)') 'Entering B matrix generator'

! count the number of primitives
    IF (this%internal) THEN
      ni = nconn + nbend + nrots
    ELSE
      ni = 3*natom + nconn + nbend + nrots
    END IF

! count linear bends as two primitives
    DO i = 1, nbend
      IF (ibend(4,i)<0 .AND. ibend(1,i)/=0) THEN
        ni = ni + 1
        IF (printl>=6) WRITE (stdout,'(5X,A,I5,A)') 'Linear bend ', i, &
          ' counts as two primitives'
      END IF
    END DO

! report
    IF (printl>=6) THEN
      IF (this%internal) THEN
        WRITE (stdout,'(5X,A,I5,/)') 'Number of internal coordinates = ', ni
      ELSE
        WRITE (stdout,'(5X,A,I5)') 'Number of primitives = ', ni
        WRITE (stdout,'(5X,A,I5,A,/)') 'Of which are ', 3*natom, ' Cartesians'
      END IF
    END IF

! allocate memory for one row and the B matrix if required
    call allocate (brow,3,natom)

    IF ( .NOT. allocated(bprim%data)) THEN
      IF (printl>=6) WRITE (stdout,'(7X,A)') 'Allocating new B matrix'
      bprim = matrix_create(ni,3*natom,'B matrix')
    END IF

! allocate primweight matrix
    IF ( .NOT. allocated(primweight%data)) THEN
      IF (printl>=6) WRITE (stdout,'(7X,A)') 'Allocating new Prim Weight matrix'
      primweight = matrix_create(ni,1,'Prim Weight')
    END IF

! adjust the scaling factors for the cartesians (old code: start of ...bprim1)
!!$    IF (ctrl%cfact==d_zero) THEN
      fact = d_one/real(natom,rk)
      fact=1.D0
!!$    ELSE IF (ctrl%cfact<d_zero) THEN
!!$      fact = -ctrl%cfact/real(natom,rk)
!!$    ELSE
!!$      fact = ctrl%cfact
!!$    END IF
    noint = 0

! B matrix is made one row at a time by making brow
    brow(1:3,1:natom) = 0.D0

! loop over all connections making b matrix elements
    DO i = 1, nconn
      noint = i
      cz(1) = x(iconn(1,i))
      cz(2) = y(iconn(1,i))
      cz(3) = z(iconn(1,i))
      cz(4) = x(iconn(2,i))
      cz(5) = y(iconn(2,i))
      cz(6) = z(iconn(2,i))
      CALL str_dlc(1,1,2,b,ib,cz)

      primweight%data(noint,1)=(xweight(iconn(1,i)) + xweight(iconn(2,i))) * 0.5D0

! construct row of bmatrix
      DO j = 1, 2
        DO k = 1, 3
          brow(k,iconn(j,i)) = b(k,j)
        END DO
      END DO
      idum = matrix_set_row(bprim,size(brow),brow,noint)

! re-zero brow
      DO j = 1, 2
        DO k = 1, 3
          brow(k,iconn(j,i)) = d_zero
        END DO
      END DO
! end of loop over connections
    END DO

! loop over all bends making bends, wags and linears
    DO i = 1, nbend
      IF (ibend(1,i)>0) THEN
        noint = noint + 1

! 'normal' bend
        IF (ibend(4,i)==0) THEN
          cz(1) = x(ibend(1,i))
          cz(2) = y(ibend(1,i))
          cz(3) = z(ibend(1,i))
          cz(4) = x(ibend(2,i))
          cz(5) = y(ibend(2,i))
          cz(6) = z(ibend(2,i))
          cz(7) = x(ibend(3,i))
          cz(8) = y(ibend(3,i))
          cz(9) = z(ibend(3,i))
          CALL bend_dlc(1,1,2,3,b,ib,cz)

          primweight%data(noint,1)=(xweight(ibend(1,i)) + xweight(ibend(2,i)) &
              + xweight(ibend(3,i))) / 3.D0

! construct row of bmatrix
          DO j = 1, 3
            DO k = 1, 3
              brow(k,ibend(j,i)) = b(k,j)
            END DO
          END DO
          idum = matrix_set_row(bprim,size(brow),brow,noint)

! re-zero brow
          DO j = 1, 3
            DO k = 1, 3
              brow(k,ibend(j,i)) = d_zero
            END DO
          END DO

! improper dihedral
!
! Impropers can be constructed thus:
!
!        1         1
!        |         |
!        2    =>   2
!       / \         \  <- axis of rotation
!      4   3     4---3
!
! This is not a true wag but it should not make much
! difference in delocalised internals and requires less
! code!
        ELSE IF (ibend(4,i)>0) THEN
          cz(1) = x(ibend(1,i))
          cz(2) = y(ibend(1,i))
          cz(3) = z(ibend(1,i))
          cz(4) = x(ibend(2,i))
          cz(5) = y(ibend(2,i))
          cz(6) = z(ibend(2,i))
          cz(7) = x(ibend(3,i))
          cz(8) = y(ibend(3,i))
          cz(9) = z(ibend(3,i))
          cz(10) = x(ibend(4,i))
          cz(11) = y(ibend(4,i))
          cz(12) = z(ibend(4,i))
          CALL tors_dlc(1,1,2,3,4,b,ib,cz)

          primweight%data(noint,1)=(xweight(ibend(1,i)) + xweight(ibend(2,i)) &
              + xweight(ibend(3,i)) + xweight(ibend(4,i))) *0.25D0

! construct row of bmatrix
          DO j = 1, 4
            DO k = 1, 3
              brow(k,ibend(j,i)) = b(k,j)
            END DO
          END DO
          idum = matrix_set_row(bprim,size(brow),brow,noint)

! re-zero brow
          DO j = 1, 4
            DO k = 1, 3
              brow(k,ibend(j,i)) = d_zero
            END DO
          END DO

! 'linear' bend w.r.t. yz plane
!
! make a point translated through 1A from atom atom 2 in y, find both angles
! formed and contruct B matrix from contributions from real atoms, then
! repeat with y translation - two coordinates will be formed this way
!
! A   C    A Y    Y C
!  \ /  =   \| +  |/  
!   B        B    B
!
! first set up ib
        ELSE IF (ibend(4,i)<=-1) THEN
          cz(1) = x(ibend(1,i))
          cz(2) = y(ibend(1,i))
          cz(3) = z(ibend(1,i))
          cz(4) = x(ibend(2,i))
          cz(5) = y(ibend(2,i))
          cz(6) = z(ibend(2,i))
          cz(7) = x(ibend(2,i))
          cz(8) = y(ibend(2,i)) + 1.0D0
          cz(9) = z(ibend(2,i))
          CALL bend_dlc(1,1,2,3,b,ib,cz)

          primweight%data(noint,1)=(xweight(ibend(1,i)) + xweight(ibend(2,i))) *0.5D0

! copy atom 1 into real bmat
          DO j = 1, 3
            brow(j,ibend(1,i)) = b(j,1)
          END DO

! perform oposite angle
          cz(1) = x(ibend(2,i))
          cz(2) = y(ibend(2,i)) + 1.0D0
          cz(3) = z(ibend(2,i))
          cz(7) = x(ibend(3,i))
          cz(8) = y(ibend(3,i))
          cz(9) = z(ibend(3,i))
          CALL bend_dlc(1,1,2,3,b,ib,cz)

          primweight%data(noint,1)=(xweight(ibend(2,i)) + xweight(ibend(3,i))) *0.5D0
          
! copy atom 3 into bmat and add atoms 3 & 1 to make 2 
!            do j = 1,3
!               brow(j,ibend(3,i)) = b(j,3) 
!               brow(j,ibend(2,i)) = -1.0D0*(brow(j,ibend(1,i))+b(j,3))
!            end do
          brow(2,ibend(2,i)) = 1.0D0

! construct row of bmatrix
          idum = matrix_set_row(bprim,size(brow),brow,noint)

! re-zero brow
          DO j = 1, 3
            DO k = 1, 3
              brow(k,ibend(j,i)) = d_zero
            END DO
          END DO

! increment ic counter
          noint = noint + 1

! do other plane
!            in z if -1 or x if -2
          cz(1) = x(ibend(1,i))
          cz(2) = y(ibend(1,i))
          cz(3) = z(ibend(1,i))
          cz(4) = x(ibend(2,i))
          cz(5) = y(ibend(2,i))
          cz(6) = z(ibend(2,i))
          cz(8) = y(ibend(2,i))
          IF (ibend(4,i)==-1) THEN
            cz(7) = x(ibend(2,i))
            cz(9) = z(ibend(2,i)) + 1.0D0
          ELSE
            cz(7) = x(ibend(2,i)) + 1.0D0
            cz(9) = z(ibend(2,i))
          END IF
          CALL bend_dlc(1,1,2,3,b,ib,cz)

          primweight%data(noint,1)=(xweight(ibend(1,i)) + xweight(ibend(2,i))) *0.5D0

! copy atom 1 into real bmatrix
          DO j = 1, 3
            brow(j,ibend(1,i)) = b(j,1)
          END DO

! do opposite angle
          cz(2) = y(ibend(2,i))
          IF (ibend(4,i)==-1) THEN
            cz(1) = x(ibend(2,i))
            cz(3) = z(ibend(2,i)) + 1.0D0
          ELSE
            cz(1) = x(ibend(2,i)) + 1.0D0
            cz(3) = z(ibend(2,i))
          END IF
          cz(7) = x(ibend(3,i))
          cz(8) = y(ibend(3,i))
          cz(9) = z(ibend(3,i))
          CALL bend_dlc(1,1,2,3,b,ib,cz)

! copy atom 3 into bmatrix and add atoms 3 & 1 to make 2
!            do j = 1,3
!               brow(j,ibend(3,i))=b(j,3)
!               brow(j,ibend(2,i))=brow(j,ibend(1,i))+b(j,3)
!            end do
          IF (ibend(4,i)==-1) THEN
            brow(3,ibend(2,i)) = 1.0D0
          ELSE
            brow(1,ibend(2,i)) = 1.0D0
          END IF

! construct row of bmatrix
          idum = matrix_set_row(bprim,size(brow),brow,noint)

! re-zero brow
          DO j = 1, 3
            DO k = 1, 3
              brow(k,ibend(j,i)) = d_zero
            END DO
          END DO
        END IF ! if (ibend(4,i).eq.0) then ... elseif ...
      END IF ! if (ibend(1,i).gt.0) then

! end of loop over all bends etc.
    END DO ! do i = 1,nbend

! put in dihedrals
    DO i = 1, nrots
      noint = noint + 1
      cz(1) = x(irots(1,i))
      cz(2) = y(irots(1,i))
      cz(3) = z(irots(1,i))
      cz(4) = x(irots(2,i))
      cz(5) = y(irots(2,i))
      cz(6) = z(irots(2,i))
      cz(7) = x(irots(3,i))
      cz(8) = y(irots(3,i))
      cz(9) = z(irots(3,i))
      cz(10) = x(irots(4,i))
      cz(11) = y(irots(4,i))
      cz(12) = z(irots(4,i))
      CALL tors_dlc(1,1,2,3,4,b,ib,cz)

      primweight%data(noint,1)=(xweight(irots(1,i)) + xweight(irots(2,i)) &
          + xweight(irots(3,i)) + xweight(irots(4,i))) *0.25D0

! construct row of bmatrix
      DO j = 1, 4
        DO k = 1, 3
          brow(k,irots(j,i)) = b(k,j)
        END DO
      END DO
      idum = matrix_set_row(bprim,size(brow),brow,noint)

! re-zero brow
      DO j = 1, 4
        DO k = 1, 3
          brow(k,irots(j,i)) = d_zero
        END DO
      END DO

! end of loop over dihedrals
    END DO

! put in cartesians
    IF ( .NOT. this%internal) THEN
      DO i = 1, natom
        DO j = 1, 3
          noint = noint + 1
! weight all cartesians the same way
          brow(j,i) = fact
          primweight%data(noint,1)=xweight(i)
          
! weight them slightly different (to avoid degeneration in
! the spectrum of the primitive G matrix) JK
! if  the added value is too large, the Gram-Schmidt orthogonalization 
! in constraints.f90:ortho_mat produces too many non-zero eigenvalues
!brow(j,i) = fact + dble(i-1)* 3.D-5 + dble(j-1)* 1.D-5 
          idum = matrix_set_row(bprim,size(brow),brow,noint)
          brow(j,i) = 0.0D0
        END DO
      END DO
    END IF

! set the number of primitives and print the matrix if requested
    np = noint
    IF (printl>=6) i = matrix_print(bprim)

    call deallocate(brow)

!//////////////////////////////////////////////////////////////////////////////
! Helper routines for hdlc_make_bprim
!
! The following three subroutines are adapted from the normal coordinate
! analysis program of Schachtschneider, Shell development
!//////////////////////////////////////////////////////////////////////////////

  CONTAINS
    SUBROUTINE str_dlc(noint,i,j,b,ib,c)
      INTEGER i, j, ib(4,*), noint
      REAL (rk) c(*), b(3,4,1)
!
      INTEGER iaind, jaind, m
      REAL (rk) dzero, rij(3), dijsq
!
      DATA dzero/0.0D0/
!
      iaind = 0
      jaind = 3
      ib(1,noint) = i
      ib(2,noint) = j
      dijsq = dzero
      DO m = 1, 3
        rij(m) = c(m+jaind) - c(m+iaind)
        dijsq = dijsq + rij(m)**2
      END DO
      if (dijsq <= 1d-23) then
        call dlf_error("verkackt")
      endif
      DO m = 1, 3
        b(m,1,noint) = -rij(m)/sqrt(dijsq)
        b(m,2,noint) = -b(m,1,noint)
      END DO
      RETURN
!
    END SUBROUTINE str_dlc
!
    SUBROUTINE bend_dlc(noint,i,j,k,b,ib,c)
      INTEGER i, j, k, ib(4,1), noint
      REAL (rk) b(3,4,1), c(*)
!
      INTEGER iaind, jaind, kaind, m
      REAL (rk) rji(3), rjk(3), eji(3), ejk(3), djisq, djksq, dzero, done, &
        sinj, dotj, dji, djk
!
      DATA dzero/0.0D0/, done/1.0D0/
!
      iaind = 0
      jaind = 3
      kaind = 6
      ib(1,noint) = i
      ib(2,noint) = j
      ib(3,noint) = k
      djisq = dzero
      djksq = dzero
      DO m = 1, 3
        rji(m) = c(m+iaind) - c(m+jaind)
        rjk(m) = c(m+kaind) - c(m+jaind)
        djisq = djisq + rji(m)**2
        djksq = djksq + rjk(m)**2
      END DO
      dji = sqrt(djisq)
      djk = sqrt(djksq)
      dotj = dzero
      DO m = 1, 3
        eji(m) = rji(m)/dji
        ejk(m) = rjk(m)/djk
        dotj = dotj + eji(m)*ejk(m)
      END DO
      sinj = sqrt(done-dotj**2)
      IF (sinj<1.0D-20) GO TO 145
      DO m = 1, 3
        b(m,3,noint) = ((dotj*ejk(m)-eji(m)))/(djk*sinj)
        b(m,1,noint) = ((dotj*eji(m)-ejk(m)))/(dji*sinj)
        b(m,2,noint) = -b(m,1,noint) - b(m,3,noint)
      END DO
!
      RETURN
145   CONTINUE
      DO m = 1, 3
        b(m,3,noint) = 0.0D0
        b(m,1,noint) = 0.0D0
        b(m,2,noint) = 0.0D0
      END DO
      RETURN
!
    END SUBROUTINE bend_dlc
!
    SUBROUTINE tors_dlc(noint,i,j,k,l,b,ib,c)
      INTEGER i, j, k, l, ib(4,*), noint
      REAL (rk) b(3,4,1), c(*)
!
      INTEGER iaind, jaind, kaind, laind, m
      REAL (rk) dzero, done, dij, dijsq, djk, djksq, dkl, dklsq, dotpj, dotpk, &
        sinpj, sinpk, smi, smj, sml, f1, f2
      REAL (rk) rij(3), rjk(3), rkl(3), eij(3), ejk(3), ekl(3), cr1(3), cr2(3)
!
      DATA dzero/0.0D0/, done/1.0D0/
!
      iaind = 0
      jaind = 3
      kaind = 6
      laind = 9
      ib(1,noint) = i
      ib(2,noint) = j
      ib(3,noint) = k
      ib(4,noint) = l
      dijsq = dzero
      djksq = dzero
      dklsq = dzero
      DO m = 1, 3
        rij(m) = c(m+jaind) - c(m+iaind)
        dijsq = dijsq + rij(m)**2
        rjk(m) = c(m+kaind) - c(m+jaind)
        djksq = djksq + rjk(m)**2
        rkl(m) = c(m+laind) - c(m+kaind)
        dklsq = dklsq + rkl(m)**2
      END DO
      dij = sqrt(dijsq)
      djk = sqrt(djksq)
      dkl = sqrt(dklsq)
      DO m = 1, 3
        eij(m) = rij(m)/dij
        ejk(m) = rjk(m)/djk
        ekl(m) = rkl(m)/dkl
      END DO
      cr1(1) = eij(2)*ejk(3) - eij(3)*ejk(2)
      cr1(2) = eij(3)*ejk(1) - eij(1)*ejk(3)
      cr1(3) = eij(1)*ejk(2) - eij(2)*ejk(1)
      cr2(1) = ejk(2)*ekl(3) - ejk(3)*ekl(2)
      cr2(2) = ejk(3)*ekl(1) - ejk(1)*ekl(3)
      cr2(3) = ejk(1)*ekl(2) - ejk(2)*ekl(1)
      dotpj = -(eij(1)*ejk(1)+eij(2)*ejk(2)+eij(3)*ejk(3))
      dotpk = -(ejk(1)*ekl(1)+ejk(2)*ekl(2)+ejk(3)*ekl(3))
      sinpj = sqrt(done-dotpj**2)
      sinpk = sqrt(done-dotpk**2)
      DO m = 1, 3
        smi = -cr1(m)/(dij*sinpj*sinpj)
        b(m,1,noint) = smi
        f1 = 0.0D0
        f2 = 0.0D0
        IF (sinpj>1.0D-20) f1 = (cr1(m)*(djk-dij*dotpj))/(djk*dij*sinpj*sinpj)
        IF (sinpk>1.0D-20) f2 = (dotpk*cr2(m))/(djk*sinpk*sinpk)
        smj = f1 - f2
        b(m,2,noint) = smj
        sml = 0.0D0
        IF (sinpk>1.0D-20) sml = cr2(m)/(dkl*sinpk*sinpk)
        b(m,4,noint) = sml
        b(m,3,noint) = (-smi-smj-sml)
      END DO
      RETURN
    END SUBROUTINE tors_dlc

!//////////////////////////////////////////////////////////////////////////////
! end of helper routines for hdlc_make_bprim
!//////////////////////////////////////////////////////////////////////////////

  END SUBROUTINE hdlc_make_bprim
  

!//////////////////////////////////////////////////////////////////////////////
! subroutine hdlc_make_ut
!
! Constructs Ut, the transpose of the non-redundant eigenvectors of the
! redundant G matrix
!
! Alexander J Turner Dec 1997
!
!     See:
! J_Baker, A_Kessi and B_Delley
! J.Chem.Phys 105,(1),1 July 1996
!
! Input
! =====
!
! bprim:  primitive B matrix
!
! Output
! ======
!
! ut: transposed U matrix
!
! Paramters
! =========
!
! Cutoff decides if an eigenvalue of G is zero
!
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE hdlc_make_ut(this,ut,bprim)
    IMPLICIT NONE
! args
    class(hdlc_ctrl), intent(inout) :: this
    TYPE (matrix) :: bprim, ut

! local params
    REAL (rk), PARAMETER :: cutoff = 1.0D-8

! local vars
    INTEGER :: i, idum, j, n, n6, nii
    REAL (rk), DIMENSION (:), ALLOCATABLE :: temp
    TYPE (matrix) :: g_mat, btprim, r_mat, v_mat

! begin, nii: number of primitives
    IF (printl>=6) WRITE (stdout,'(5X,A)') 'Entering UT matrix generator'
    nii = matrix_dimension(bprim,1)
! n: number of cartesians
    n = matrix_dimension(bprim,2)

! n6: number of delocalised internal coordinates
    IF (this%internal) THEN
      n6 = n - 6
    ELSE
      n6 = n
    END IF

! test for insufficient primitives
    IF (nii<n6) THEN
      WRITE (stdout,'(A,I5,A,I5,A)') 'There are ', nii, ' primitives and ', &
        n6, ' required!'
      CALL hdlc_errflag('Insufficient primitive coordinates','abort')
    END IF

! allocate space
    g_mat = matrix_create(nii,nii,'G matrix')
    btprim = matrix_create(nii,n,'Bt matrix')

! form G
    idum = matrix_copy(bprim,btprim)
    idum = matrix_transpose(btprim)
    idum = matrix_multiply(1.0D0,bprim,btprim,0.0D0,g_mat)

! allocate work matrices for the diagonalisation; Bt is not used any longer
    idum = matrix_destroy(btprim)

! Next two lines change due to problems with LAPACK routine dsyevx
!   r_mat = matrix_create (nii, n6, 'R vectors')
!   v_mat = matrix_create (n6, 1,   'R roots ')
    r_mat = matrix_create(nii,nii,'R vectors')
    v_mat = matrix_create(nii,1,'R roots ')

! diagonalise and form U 
    idum = matrix_diagonalise(g_mat,r_mat,v_mat,.FALSE.)
    IF (idum/=0) THEN
      CALL hdlc_errflag('G matrix could not be diagonalised','stop')
    END IF

! allocate Ut matrix; the G matrix not used any longer
    idum = matrix_destroy(g_mat)
    ut = matrix_create(n6,nii,'Ut matrix')

! check for faulty eigenvalue structure, use temp as scratch
    call allocate (temp,nii)
    idum = matrix_get(v_mat,nii,temp)
    IF (printl>=6) WRITE (stdout,'(5x,a,/)') &
      'Eigenvalue structure of primitive G matrix'
    j = 0
    DO i = 1, nii
      IF (printl>=6) WRITE (stdout,'(5x,a,i5,a,f13.8)') 'Eigenvalue ', i, &
        ' = ', temp(i)
      IF (abs(temp(i))>cutoff) j = j + 1
    END DO
    IF (printl>=5) WRITE (stdout,'(/,5x,a,i5,a,i5,a,/)') &
      'The system has ', n6, ' degrees of freedom, and ', j, &
      ' non-zero eigenvalues'
    IF (j<n6) THEN
      WRITE (stdout,'(/,5x,a,i5,a,i5,a,/)') &
          'The system has ', n6, ' degrees of freedom, and ', j, &
          ' non-zero eigenvalues'
      IF (this%ngroups==1 .AND. this%internal .AND. this%contyp==1) THEN
        CALL hdlc_errflag( &
          'DLC based on total connection cannot be used for a planar system', &
          'stop')
      ELSE
        CALL hdlc_errflag( &
          'Too few delocalised coordinates with non-zero values','stop')
      END IF
    END IF

    IF (printl>=6) idum = matrix_print(r_mat)

! generate Ut (the eigenvectors come out of matrix_diagonalise in columns!)
    DO i = 1, n6
      j = matrix_get_column(r_mat,size(temp),temp,i)
      j = matrix_set_row(ut,size(temp),temp,i)
    END DO

! clean up some memory
    idum = matrix_destroy(v_mat)
    idum = matrix_destroy(r_mat)
    call deallocate (temp)

    IF (printl>=6) idum = matrix_print(ut)

! end
    RETURN
  END SUBROUTINE hdlc_make_ut

!//////////////////////////////////////////////////////////////////////////////
! subroutine hdlc_make_bhdlc
!
! Construct delocalised B matrix
!
! Input
! =====
! ut    : Ut matrix
! bprim : primitive B matrix
! bhdlc : unassociated => generate B matrix
!         associated   => pre-existing B matrix
!
! Output
! ======
! bhdlc : HDLC B matrix
!
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE hdlc_make_bhdlc(this,bprim,bhdlc,ut)

! args
    class(hdlc_ctrl), intent(inout) :: this
    TYPE (matrix) :: bhdlc, bprim, ut

! local vars
    INTEGER :: idum, n, np, n6

! begin, get dimensions of B matrix
    np = matrix_dimension(bprim,1)
    n = matrix_dimension(bprim,2)
    IF (this%internal) THEN
      n6 = n - 6
    ELSE
      n6 = n
    END IF

! create B matrix if needed
    IF ( .NOT. allocated(bhdlc%data)) THEN
      bhdlc = matrix_create(n6,n,'B HDLC')
      IF (printl>=6) WRITE (stdout,'(7X,A,/)') 'New HDLC B matrix'
    END IF

! do maths
    idum = matrix_multiply(1.0D0,ut,bprim,0.0D0,bhdlc)
    IF (printl>=6) idum = matrix_print(bhdlc)

  END SUBROUTINE hdlc_make_bhdlc
  

!//////////////////////////////////////////////////////////////////////////////
! subroutine hdlc_make_ighdlc
!
! Construct inverse G matrix from delocalised B matrix
!
! Input
! =====
! bhdlc  : HDLC B matrix
! ighdlc : unassociated => generate iG matrix
!          associated   => pre-existing iG matrix
!
! Output
! ======
! ighdlc : inverse HDLC G matrix
! failed : true if det|iG| < cutoff
!
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE hdlc_make_ighdlc(this,bhdlc,ighdlc,failed,spur)

! args
    class(hdlc_ctrl), intent(inout) :: this
    LOGICAL failed
    TYPE (matrix) :: bhdlc, ighdlc
    real(rk),intent(out),optional :: spur

! local params
    REAL (rk) :: cutoff
    PARAMETER (cutoff=1.0D-8)

! local vars
    INTEGER i, idum, n, n6
    REAL (rk) :: det
    TYPE (matrix) :: bthdlc
    integer       ::  j

! begin, get number of HDLC (n6) and number of cartesians (n)
    n6 = matrix_dimension(bhdlc,1)
    n = matrix_dimension(bhdlc,2)
! generate iG matrix if needed
    IF ( .NOT. allocated(ighdlc%data)) THEN
      ighdlc = matrix_create(n6,n6,'HDLC iG')
      IF (printl>=6) WRITE (stdout,'(7X,A,/)') 'New HDLC iG matrix'
    END IF

! make B transposed
    bthdlc = matrix_create(n6,n,'HDLC BT')
    idum = matrix_copy(bhdlc,bthdlc)
    idum = matrix_transpose(bthdlc)

! do maths
    idum = matrix_multiply(1.0D0,bhdlc,bthdlc,0.0D0,ighdlc)
    IF (printl>=6) THEN
      WRITE (stdout,'(7X,A,/)') &
        'Matrix printed is G HDLC (= B HDLC * BT HDLC) before inversion'
      idum = matrix_print(ighdlc)
    END IF
    if(present(spur)) then
      spur = 0d0
      do j=1,n6
        spur = spur + ighdlc%data(j,j)
      enddo
    endif
    idum = matrix_invert(ighdlc,det,.TRUE.)
    

! clean up
    idum = matrix_destroy(bthdlc)

! check for linear dependency in iG, scale det to be size-independent
    i = matrix_dimension(ighdlc,1)
   ! print *,det,det**(1.0D0/real(i,rk)),1d0/det,1d0/(det**(1.0D0/real(i,rk)))
    IF (printl>=6) THEN
      WRITE (stdout,'(7X,A,E13.5,/)') 'The HDLC G matrix determinant is ', det
    END IF
    if(det<0d0) then
      write(stdout,'(7X,A,E13.5,/)') 'HDLC determinant was',det
      call dlf_fail("deteminate of HDLC G- matrix is negative!")
    endif
    det = det**(1.0D0/real(i,rk))
    failed = (det<cutoff)    
  END SUBROUTINE hdlc_make_ighdlc

!==============================================================================
! checkpointing
!==============================================================================

!//////////////////////////////////////////////////////////////////////////////
! subroutine hdlc_wr_hdlc
!
! Write all HDLC objects to iunit
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE hdlc_wr_hdlc(iunit,s_hdlc,lform,lerr)
    use dlf_checkpoint, only: write_separator
    implicit none
! args
    LOGICAL lform, lerr
    INTEGER iunit
    CHARACTER*25 secthead
    TYPE (hdlc_ctrl) :: s_hdlc

! local vars
    INTEGER i, j, igroup
    TYPE (residue_type) :: res

! begin
    lerr = .FALSE.

    call write_separator(iunit,"HDLC sizes")

! write header, return if no HDLC residues
    IF (lform) THEN
      WRITE (iunit,*,err=98) s_hdlc%lhdlc, s_hdlc%internal &
          , s_hdlc%ngroups, s_hdlc%contyp
    ELSE
      WRITE (iunit,err=98) s_hdlc%lhdlc, s_hdlc%internal &
          , s_hdlc%ngroups, s_hdlc%contyp
    END IF
    IF ( .NOT. s_hdlc%lhdlc) RETURN

    ! new block for dl-find:
    IF (lform) THEN
      WRITE (iunit,*,err=98) s_hdlc%first, s_hdlc%last, &
          s_hdlc%nmin, s_hdlc%ngroupsdrop, s_hdlc%nfrozen, s_hdlc%err_cnt(:)
    ELSE
      write (iunit,err=98) s_hdlc%first, s_hdlc%last, &
          s_hdlc%nmin, s_hdlc%ngroupsdrop, s_hdlc%nfrozen, s_hdlc%err_cnt(:)
    END IF

! loop over all HDLC objects
    DO igroup = 1, s_hdlc%ngroups

      call write_separator(iunit,"HDLC residue")

      res = s_hdlc%res(igroup)

! write scalars
      IF (lform) THEN
        WRITE (iunit,'(8I8)',err=98) res%name, res%natom, res%np, &
          res%nconn, res%nbend, res%nrots, res%ncons, res%ip
        WRITE (iunit,*,err=98) res%err_cnt, res%lgmatok, res%next, &
            res%prev, res%tbias
      ELSE
        WRITE (iunit,err=98) res%name, res%natom, res%np, &
          res%nconn, res%nbend, res%nrots, res%ncons, res%ip
        WRITE (iunit,err=98) res%err_cnt, res%lgmatok, res%next, &
            res%prev, res%tbias
      END IF

    call write_separator(iunit,"HDLC res arrays")
! write arrays
      IF (lform) THEN
        WRITE (iunit,'(5F16.10)',err=98) (res%x(i),i=1,res%natom)
        WRITE (iunit,'(5F16.10)',err=98) (res%y(i),i=1,res%natom)
        WRITE (iunit,'(5F16.10)',err=98) (res%z(i),i=1,res%natom)
        WRITE (iunit,'(2I8)',err=98) ((res%iconn(i,j),i=1,2),j=1,res%nconn)
        WRITE (iunit,'(4I8)',err=98) ((res%ibend(i,j),i=1,4),j=1,res%nbend)
        WRITE (iunit,'(4I8)',err=98) ((res%irots(i,j),i=1,4),j=1,res%nrots)
        WRITE (iunit,'(6I8)',err=98) ((res%icons(i,j),i=1,6),j=1,res%ncons)
        WRITE (iunit,'(5F16.10)',err=98) (res%vcons(i),i=1,res%ncons)
        WRITE (iunit,'(6I8)',err=98) (res%at(i),i=1,res%natom)
        WRITE (iunit,'(5F16.10)',err=98) (res%biasv(i),i=1,res%np )
      ELSE
        WRITE (iunit,err=98) (res%x(i),i=1,res%natom)
        WRITE (iunit,err=98) (res%y(i),i=1,res%natom)
        WRITE (iunit,err=98) (res%z(i),i=1,res%natom)
        WRITE (iunit,err=98) ((res%iconn(i,j),i=1,2),j=1,res%nconn)
        WRITE (iunit,err=98) ((res%ibend(i,j),i=1,4),j=1,res%nbend)
        WRITE (iunit,err=98) ((res%irots(i,j),i=1,4),j=1,res%nrots)
        WRITE (iunit,err=98) ((res%icons(i,j),i=1,6),j=1,res%ncons)
        WRITE (iunit,err=98) (res%vcons(i),i=1,res%ncons)
        WRITE (iunit,err=98) (res%at(i),i=1,res%natom)
        WRITE (iunit,err=98) (res%biasv(i),i=1,res%np)
      END IF

! write matrices - see matrixlib
      CALL hdlc_wr_matrix(iunit,res%ut,lform,lerr)
      CALL hdlc_wr_matrix(iunit,res%iweight,lform,lerr)
      CALL hdlc_wr_matrix(iunit,res%oldxyz,lform,lerr)

! end loop over all HDLC residues
    END DO

    call write_separator(iunit,"END")

! I/O failure jump point
    RETURN
98  lerr = .TRUE.
    IF (printl>=-1) WRITE (stdout,'(/,a,i5,a,/)') &
      '*** Problem writing residue ', igroup, ' to checkpoint file ***'
  END SUBROUTINE hdlc_wr_hdlc

!//////////////////////////////////////////////////////////////////////////////
! subroutine hdlc_rd_hdlc
!
! Read all HDLC objects from iunit
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE hdlc_rd_hdlc(iunit,s_hdlc,lform,tok)
    use dlf_checkpoint, only: read_separator
    implicit none
! args
    LOGICAL lform, tok
    INTEGER iunit
    CHARACTER*25 secthead
    TYPE (hdlc_ctrl) :: s_hdlc

! local vars
    INTEGER i, j, igroup, contyp
    INTEGER, DIMENSION (1) :: idum
    TYPE (residue_type) :: res
    logical :: tchk,lerr,recoverable
    integer :: natom, nconn, nbend, nrots, ncons
    integer :: ar2(2),ar1(1)
! **********************************************************************

    tok = .false.
    igroup = 0

    call read_separator(iunit,"HDLC sizes",tchk)
    if(.not.tchk) return

! read header, return if no HDLC residues
    IF (lform) THEN
      READ (iunit,*,err=98) s_hdlc%lhdlc, s_hdlc%internal, igroup, contyp
    ELSE
      READ (iunit,err=98) s_hdlc%lhdlc, s_hdlc%internal, igroup, contyp
    END IF
    IF ( .NOT. s_hdlc%lhdlc) then
      tok=.true.
      RETURN
    end IF

    if(igroup/=s_hdlc%ngroups) then
      if(printl>=6) &
          write(stdout,"('Number of residues in checkpoint file inconsistent')")
      return
    end if

    if(contyp/=s_hdlc%contyp) then
      if(printl>=6) &
          write(stdout,"('Connection type in checkpoint file inconsistent')")
      return
    end if

    ! new block for dl-find (no constraints and input connections yet ...):
    IF (lform) THEN
      READ (iunit,*,err=98) s_hdlc%first, s_hdlc%last, &
          s_hdlc%nmin, s_hdlc%ngroupsdrop, s_hdlc%nfrozen, s_hdlc%err_cnt(:)
    ELSE
      READ (iunit,err=98) s_hdlc%first, s_hdlc%last, &
          s_hdlc%nmin, s_hdlc%ngroupsdrop, s_hdlc%nfrozen, s_hdlc%err_cnt(:)
    END IF

! loop over all HDLC objects
    DO igroup = 1, s_hdlc%ngroups

      call read_separator(iunit,"HDLC residue",tchk)
      if(.not.tchk) return

      res=s_hdlc%res(igroup)
! allocate new residue and read scalars
!      allocate (res)
      IF (lform) THEN
        READ (iunit,'(8I8)',err=98) res%name, natom, res%np, &
          nconn, nbend, nrots, ncons, res%ip
        READ (iunit,*,err=98) res%err_cnt, res%lgmatok, res%next, &
            res%prev, res%tbias
      ELSE
        READ (iunit,err=98) res%name, natom, res%np, &
          nconn, nbend, nrots, ncons, res%ip
        READ (iunit,err=98) res%err_cnt, res%lgmatok, res%next, &
            res%prev, res%tbias
      END IF

      !Check for correct array sizes
      tchk=.false.
      recoverable=.false.
      ! Somtimes, a bond moves to a rotation or vice versa. This may be recoverable
      if(nbend/=res%nbend) then
        if(printl>=6) then
          write(stdout,"('Number of bends in residue',i4,' inconsistent&
              & in checkpoint file')") igroup
          write(stdout,"('Number of bends created',i4)") res%nbend
          write(stdout,"('Number of bends read   ',i4)") nbend
          write(stdout,"('This will allow recovery')")
        endif
        tchk=.true.
        recoverable=.true.
      end if
      if(nrots/=res%nrots) then
        if(printl>=6) then
          write(stdout,"('Number of rotations in residue',i4,' inconsistent&
              & in checkpoint file')") igroup
          write(stdout,"('Number of rotations created',i4)") res%nrots
          write(stdout,"('Number of rotations read   ',i4)") nrots
          write(stdout,"('This will allow recovery')")
        end if
        tchk=.true.
        recoverable=.true.
      end if
      if(natom/=res%natom) then
        if(printl>=6) then
          write(stdout,"('Number of atoms in residue',i4,' inconsistent&
              & in checkpoint file')") igroup
          write(stdout,"('This will NOT allow recovery')")
        end if
        tchk=.true.
        recoverable=.false.
      end if
      if(nconn/=res%nconn) then
        if(printl>=6) then
          write(stdout,"('Number of connections in residue',i4,' inconsistent&
              & in checkpoint file')") igroup
          write(stdout,"('Number of connections created',i4)") res%nconn
          write(stdout,"('Number of connections read   ',i4)") nconn
          write(stdout,"('This will allow recovery')")
        end if
        tchk=.true.
        recoverable=.true.
      end if
      if(ncons/=res%ncons) then
        if(printl>=6) then
          write(stdout,"('Number of constraints in residue',i4,' inconsistent&
              & in checkpoint file')") igroup
          write(stdout,"('This will NOT allow recovery')")
        end if
        tchk=.true.
        recoverable=.false.
      end if

      ! error check
      if(tchk) then
        if(printl>=6) write(stdout,"('Problem in reading checkpoint')")
        if(recoverable) then
          if(printl>=6) write(stdout,"('Trying to recover')")
          ! only recover if original arrays are large enough, do 
          ! not reallocate
          ar2=shape(res%ibend)
          if(ar2(2)>= nbend) then
            ar2=shape(res%irots)
            if(ar2(2)>= nrots) then
              ar2=shape(res%iconn)
              if(ar2(2)>= nconn) then
                res%nbend= nbend
                res%nrots= nrots
                res%nconn= nconn
                if(printl>=6) write(stdout,"('Aiming at recovery')")
              else
                if(printl>=6) write(stdout,"('Failed to recover due to number of connections')")
                return
              end if
            else
              if(printl>=6) write(stdout,"('Failed to recover due to number of rotations')")
              return
            end if
          else
            if(printl>=6) write(stdout,"('Failed to recover due to number of bends')")
            return
          end if
        else
          if(printl>=6) write(stdout,"('Not trying to recover')")
          return
        end if
      end if

! allocate contained arrays
!!$      allocate (res%x(res%natom))
!!$      allocate (res%y(res%natom))
!!$      allocate (res%z(res%natom))
!!$      allocate (res%iconn(2,max(1,res%nconn)))
!!$      allocate (res%ibend(4,max(1,res%nbend)))
!!$      allocate (res%irots(4,max(1,res%nrots)))
!!$      allocate (res%icons(6,max(1,res%ncons)))
!!$      allocate (res%vcons(max(1,res%ncons)))

! here would be the initialisation of the internal biasing vector
!!$       allocate (res%biasv(res%np))
       call vector_initialise (res%biasv, res%np, 0.0D0)
!       nullify(res%oldxyz)
!       idum=matrix_destroy(res%oldxyz)

       call read_separator(iunit,"HDLC res arrays",tchk)
       if(.not.tchk) return

! read contained arrays
      IF (lform) THEN
        READ (iunit,'(5F16.10)',err=98) (res%x(i),i=1,res%natom)
        READ (iunit,'(5F16.10)',err=98) (res%y(i),i=1,res%natom)
        READ (iunit,'(5F16.10)',err=98) (res%z(i),i=1,res%natom)
        READ (iunit,'(2I8)',err=98) ((res%iconn(i,j),i=1,2),j=1,res%nconn)
        READ (iunit,'(4I8)',err=98) ((res%ibend(i,j),i=1,4),j=1,res%nbend)
        READ (iunit,'(4I8)',err=98) ((res%irots(i,j),i=1,4),j=1,res%nrots)
        READ (iunit,'(6I8)',err=98) ((res%icons(i,j),i=1,6),j=1,res%ncons)
        READ (iunit,'(5F16.10)',err=98) (res%vcons(i),i=1,res%ncons)
        READ (iunit,'(6I8)',err=98) (res%at(i),i=1,res%natom)
        READ (iunit,'(5F16.10)',err=98) (res%biasv(i),i=1,res%np )
      ELSE
        READ (iunit,err=98) (res%x(i),i=1,res%natom)
        READ (iunit,err=98) (res%y(i),i=1,res%natom)
        READ (iunit,err=98) (res%z(i),i=1,res%natom)
        READ (iunit,err=98) ((res%iconn(i,j),i=1,2),j=1,res%nconn)
        READ (iunit,err=98) ((res%ibend(i,j),i=1,4),j=1,res%nbend)
        READ (iunit,err=98) ((res%irots(i,j),i=1,4),j=1,res%nrots)
        READ (iunit,err=98) ((res%icons(i,j),i=1,6),j=1,res%ncons)
        READ (iunit,err=98) (res%vcons(i),i=1,res%ncons)
        READ (iunit,err=98) (res%at(i),i=1,res%natom)
        READ (iunit,err=98) (res%biasv(i),i=1,res%np )
      END IF

! read UT matrix - see matrixlib
      CALL hdlc_rd_matrix(iunit,res%ut,lform,lerr)
      if(lerr) then
        tok=.false.
        print*,"Error reading UT matrix from checkpoint"
        return
      end if
      CALL hdlc_rd_matrix(iunit,res%iweight,lform,lerr)
      if(lerr) then
        tok=.false.
        print*,"Error reading iweight matrix from checkpoint"
        return
      end if
      CALL hdlc_rd_matrix(iunit,res%oldxyz,lform,lerr)
      if(lerr) then
        tok=.false.
        print*,"Error reading OldXYZ matrix from checkpoint"
        return
      end if

      s_hdlc%res(igroup) = res
!!$! linked list stuff
!!$      IF (igroup==1) THEN
!!$!        s_hdlc%first => res
!!$      ELSE
!!$        res%next=-1
!!$        res%prev = s_hdlc%last
!!$      END IF
    END DO

    call read_separator(iunit,"END",tchk)
    if(.not.tchk) return

    tok=.true.
    RETURN

! I/O failure jump point - destroy already read objects if necessary
98  tok = .false.
!    s_hdlc%lhdlc = .FALSE.
!    IF (igroup>0) CALL hdlc_destroy_all(.FALSE.,idum)
    IF (printl>=6) WRITE (stdout,'(/,a,i5,a,/)') &
      '*** Problem reading residue ', igroup, ' ***'
  END SUBROUTINE hdlc_rd_hdlc

!==============================================================================
! general helper routines follow
!==============================================================================

!//////////////////////////////////////////////////////////////////////////////
! subroutine hdlc_destroy_all
!
! Destroys all HDLC residues and saves the fields residue%err_cnt to the array
! err_cnt if lsave is set
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE hdlc_destroy_all(this,lsave,ngroups,err_cnt)

! args
    class(hdlc_ctrl), intent(inout) :: this
    LOGICAL lsave
    integer, intent(in)  :: ngroups
    INTEGER, intent(out) :: err_cnt(ngroups)

! local vars
    INTEGER i,j

    IF (printl>=6) THEN
      WRITE (stdout,'(a,/)') 'Destroying all HDLC residues'
    END IF
    i = 1
    DO WHILE (this%ngroups/=0)
      j=this%first
      IF (lsave .AND. (this%res(j)%err_cnt<2000 .OR. this%res(j)%ncons>0)) THEN
        err_cnt(i) = this%res(j)%err_cnt
        i = i + 1
      END IF
      CALL this%hdlc_destroy(this%res(j))
      
    END DO
  END SUBROUTINE hdlc_destroy_all

!//////////////////////////////////////////////////////////////////////////////
! subroutine hdlc_linear_checkin
!
! Fill in cartesian components into separate arrays
! On input: mat is a (3,natom) matrix: ((x1,y1,z1),(x2,..),...)
!           or an (3*natom,1) matrix: (x1,y1,z1,x2,...)
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE hdlc_linear_checkin(mat,natom,x,y,z)

! args
    TYPE (matrix)          :: mat
    integer, intent(in)    :: natom
    REAL (rk), intent(out) :: x(natom), y(natom), z(natom)

! local vars
    INTEGER i, j, size
    REAL (rk), allocatable :: xyz_data(:,:)

! begin
    !xyz_data => matrix_data_array(mat)
    call allocate(xyz_data,matrix_dimension(mat,1),matrix_dimension(mat,2))
    xyz_data(:,:)=mat%data(:,:)

    IF (matrix_dimension(mat,2)==1) THEN
      size = matrix_dimension(mat,1)/3
      if(size>natom) CALL hdlc_errflag('Error in hdlc_linear_checkin A','stop')
      j = 0
      DO i = 1, size
        x(i) = xyz_data(j+1,1)
        y(i) = xyz_data(j+2,1)
        z(i) = xyz_data(j+3,1)
        j = j + 3
      END DO
    ELSE
      size = matrix_dimension(mat,2)
      if(size>natom) CALL hdlc_errflag('Error in hdlc_linear_checkin B','stop')
      DO i = 1, size
        x(i) = xyz_data(1,i)
        y(i) = xyz_data(2,i)
        z(i) = xyz_data(3,i)
      END DO
    END IF
    call deallocate(xyz_data)
  END SUBROUTINE hdlc_linear_checkin

!//////////////////////////////////////////////////////////////////////////////
! subroutine hdlc_con_checkin
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE hdlc_con_checkin(imat,nconn_in,iconn)

! args
    TYPE (int_matrix) :: imat
    integer, intent(in) :: nconn_in
    INTEGER, intent(out) :: iconn(2,nconn_in)

! local vars
    INTEGER i, nconn

! begin
    nconn = int_matrix_dimension(imat,2)
    if(nconn/=nconn_in) then
      CALL hdlc_errflag('Error in hdlc_con_checkin','stop')
    end if
    DO i = 1, nconn
      iconn(1,i) = imat%data(1,i)
      iconn(2,i) = imat%data(2,i)
    END DO
  END SUBROUTINE hdlc_con_checkin

!//////////////////////////////////////////////////////////////////////////////
! subroutine hdlc_report_failure
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE hdlc_report_failure(this,residue,failed,location)

! args
    class(hdlc_ctrl), intent(inout) :: this
    LOGICAL failed
    CHARACTER(*) location
    TYPE (residue_type) :: residue

! begin
    residue%lgmatok = ( .NOT. failed)
    IF (failed) THEN
      IF (printl>=6) WRITE (stdout,'(/,A,I3,A,A,/)') &
        'G matrix linear dependent in residue ', residue%name, ', routine: ', &
        location
    END IF
  END SUBROUTINE hdlc_report_failure

!//////////////////////////////////////////////////////////////////////////////
! subroutine vector_initialise
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE vector_initialise(vector,number,value)

! args
    INTEGER number
    REAL (rk) value
    REAL (rk), DIMENSION (number) :: vector

! local vars
    INTEGER i

! begin
    DO i = 1, number
      vector(i) = value
    END DO
  END SUBROUTINE vector_initialise




end module oop_hdlc

subroutine hdlc_gpr_errflag(msg,action)
  implicit none
  character(*),intent(in) :: msg,action
  write(*,'("HDLC-errflag, action: ",a)') action
  call dlf_fail(msg)
end subroutine hdlc_gpr_errflag

#ifndef GAMESS
!------------------------------------------------------------------------------
! subroutine dlf_flushout
!
! Flush the I/O buffers of standard output
!------------------------------------------------------------------------------
SUBROUTINE dlf_gpr_flushout()
  use dlf_global, only: stdout
  CALL flush(stdout)
END SUBROUTINE dlf_gpr_flushout
#else
SUBROUTINE dlf_gpr_flushout()
  CALL flushout
END SUBROUTINE dlf_gpr_flushout
#endif  
  

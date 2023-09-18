! routines for interfacting to neural network and other PES interpolations
! GPR interpolation is implemented in gpr.f90

! This module is used by all interpolation methods (nn, gpr, ...)
! Some of the elements in this module came out of the original nn_module.
! Since they have to be used by multiple machine learning techniques they are 
! moved to this module.
module cross
  implicit none
  public
  contains

  function cross_product(a, b)
    implicit none
    real(8), dimension(3) :: cross_product
    real(8), dimension(3), intent(in) :: a, b
    real(8), dimension(3) :: norm_a,norm_b

    norm_a=a/sqrt(sum(a**2))
    norm_b=b/sqrt(sum(b**2))

    cross_product(1) = norm_a(2)*norm_b(3) - norm_a(3)*norm_b(2)
    cross_product(2) = norm_a(3)*norm_b(1) - norm_a(1)*norm_b(3)
    cross_product(3) = norm_a(1)*norm_b(2) - norm_b(1)*norm_a(2)
    return
  end function cross_product

  function bad_w(a, b)
    implicit none
    real(8), dimension(3) :: bad_w
    real(8), dimension(3), intent(in) :: a, b
    real(8), dimension(3) :: norm_a,norm_b,v1,v2,norm_v1

    v1=(/1.d0,-1.d0,1.d0/)
    v2=(/-1.d0,1.d0,1.d0/)
    norm_v1=v1/sqrt(sum(v1**2))

    norm_a=a/sqrt(sum(a**2))
    norm_b=b/sqrt(sum(b**2))

    if(abs(dot_product(norm_a,norm_b)).lt.1.d0-1.d-12)then
      bad_w=cross_product(norm_a,norm_b)
    else
      if(abs(dot_product(norm_a,norm_v1)).lt.1.d0-1.d-12)then
        bad_w=cross_product(norm_a,v1)
      else
        bad_w=cross_product(norm_a,v2)
      endif
    endif

    return

  end function bad_w

end module cross

module pes_ml_module
  use gpr_module
  implicit none
  integer :: pes_type ! set to  0 for nn
                      !         1 for GPR
  type(gpr_type),save :: gprPES   
  logical :: pes_massweight  
  real(rk),allocatable,save :: align_refcoords(:)     ! (3*nat)
  real(rk),allocatable,save :: align_modes(:,:) ! (3*nat,ni)
  real(rk),allocatable,save :: refmass(:) ! (nat)
end module pes_ml_module

module nn_module
  use dlf_parameter_module, only: rk
  implicit none
  integer :: ni,nj,nk,np,remove_n_dims,interatomic_dof,dim_renorm
  logical :: dimension_reduce,coords_interatomic,coords_inverse,coords_b_a_d
  real(rk),allocatable,save :: wzero_one(:,:),wzero_two(:,:),wone(:,:),wtwo(:,:),wthree(:,:),bone(:),btwo(:),bzero_one(:)
  real(rk),save :: bthree,alpha,pinv_tol,pinv_tol_back
  real(rk),allocatable,save :: pca_eigvect(:,:)
  integer, allocatable,save :: radii_omit(:),bad_buildlist(:),bad_modes(:),atom_index(:)
  real(rk),allocatable,save :: mu(:),mass_list(:)
  real(rk) training_e_ave(3,2),largest_TR_eval,ave_seventh_eval
  logical :: temin
  real(rk) :: emin
end module nn_module

module nn_av_module
  use dlf_parameter_module, only: rk
  logical, parameter :: avnncalc=.false.
  integer, parameter :: numfiles=2
  integer, parameter :: numimg=100
  integer, save :: imgcount=0
  integer :: iii, iij, iik 
  character (len=32) :: infiles(numfiles)
  real(rk), allocatable, save :: wzero_oneav(:,:,:),wzero_twoav(:,:,:),woneav(:,:,:),wtwoav(:,:,:),&
  wthreeav(:,:,:),boneav(:,:),btwoav(:,:),bthreeav(:),bzero_oneav(:,:)
  integer, allocatable,save :: radii_omitav(:,:)
  real(rk), allocatable, save :: align_refcoordsav(:,:), align_modesav(:,:,:)
  integer, save :: ifile=0,npav(numfiles)=0,nkav(numfiles)=0,njav(numfiles)=0
  real(rk), allocatable, save :: energyav(:),gradientav(:,:),hessianav(:,:,:)
end module nn_av_module

module layers
  implicit none
  real(8), parameter :: neuron_scaler=2.d0
  public
  contains

  function yzero_one_(x,np,ni)
    use nn_module, only : bzero_one,wzero_one
    implicit none
    integer, intent(in) :: ni,np
    real(8) yzero_one_(np)
    real(8), intent(in) :: x(ni)

    yzero_one_=tanh((bzero_one+matmul(wzero_one(:,1:ni),x**(-1))+matmul(wzero_one(:,1+ni:2*ni),x**2))/neuron_scaler)
!    yzero_one_=tanh((bzero_one+matmul(wzero_one,x))/neuron_scaler)
!    yzero_one_=tanh((bzero_one+matmul(wzero_one,x**2))/neuron_scaler)
!    yzero_one_=tanh((bzero_one+matmul(wzero_one,x**(-1)))/neuron_scaler)
!    yzero_one_=2.d0/(1.d0+exp(-(bzero_one+matmul(wzero_one(:,1:ni),x**2)+matmul(wzero_one(:,1+ni:2*ni),x))))-1.d0

    return

  end function yzero_one_

  function dyzero_one_(x,np,ni)
    implicit none
    integer, intent(in) :: ni,np
    real(8) dyzero_one_(np)
    real(8), intent(in) :: x(ni)

    dyzero_one_=(1.d0-yzero_one_(x,np,ni)**2)/neuron_scaler
  !    dyzero_one_=(1.d0-yzero_one_(x,np,ni))*(1.d0+yzero_one_(x,np,ni))/2.d0

    return

  end function dyzero_one_

  function ddyzero_one_(x,np,ni)
    implicit none
    integer, intent(in) :: ni,np
    real(8) ddyzero_one_(np)
    real(8), intent(in) :: x(ni)

  !    ddyzero_one_=-2.d0*yzero_one_(x,np,ni)*dyzero_one_(x,np,ni)
    ddyzero_one_=-2.d0*yzero_one_(x,np,ni)*dyzero_one_(x,np,ni)/neuron_scaler

    return

  end function ddyzero_one_

  function dddyzero_one_(x,np,ni)
    implicit none
    integer, intent(in) :: ni,np
    real(8) dddyzero_one_(np)
    real(8), intent(in) :: x(ni)

  !    dddyzero_one_=-2.d0*dyzero_one_(x,np,ni)**2-2.d0*yzero_one_(x,np,ni)*ddyzero_one_(x,np,ni)
    dddyzero_one_=-2.d0*dyzero_one_(x,np,ni)**2/neuron_scaler-2.d0*yzero_one_(x,np,ni)*ddyzero_one_(x,np,ni)/neuron_scaler

    return

  end function dddyzero_one_

  function yzero_one_ave_(x,np,ni,filenum)
    use nn_av_module, only : bzero_oneav,wzero_oneav
    implicit none
    integer, intent(in) :: ni,np,filenum
    real(8) yzero_one_ave_(np)
    real(8), intent(in) :: x(ni)

    yzero_one_ave_=tanh((bzero_oneav(1:np,filenum)+matmul(wzero_oneav(1:np,1:ni,filenum),x**(-1))+&
    matmul(wzero_oneav(1:np,1+ni:2*ni,filenum),x**2))/neuron_scaler)
!    yzero_one_ave_=tanh((bzero_oneav(1:np,filenum)+matmul(wzero_oneav(1:np,1:ni,filenum),x**2))/neuron_scaler)
!    yzero_one_ave_=tanh((bzero_oneav(1:np,filenum)+matmul(wzero_oneav(1:np,1:ni,filenum),x**(-1)))/neuron_scaler)
!    yzero_one_=2.d0/(1.d0+exp(-(bzero_one+matmul(wzero_one(:,1:ni),x**2)+matmul(wzero_one(:,1+ni:2*ni),x))))-1.d0

    return

  end function yzero_one_ave_

  function dyzero_one_ave_(x,np,ni,filenum)
    implicit none
    integer, intent(in) :: ni,np,filenum
    real(8) dyzero_one_ave_(np)
    real(8), intent(in) :: x(ni)

    dyzero_one_ave_=(1.d0-yzero_one_ave_(x,np,ni,filenum)**2)/neuron_scaler
  !    dyzero_one_=(1.d0-yzero_one_(x,np,ni))*(1.d0+yzero_one_(x,np,ni))/2.d0

    return

  end function dyzero_one_ave_

  function ddyzero_one_ave_(x,np,ni,filenum)
    implicit none
    integer, intent(in) :: ni,np,filenum
    real(8) ddyzero_one_ave_(np)
    real(8), intent(in) :: x(ni)

  !    ddyzero_one_=-2.d0*yzero_one_(x,np,ni)*dyzero_one_(x,np,ni)
    ddyzero_one_ave_=-2.d0*yzero_one_ave_(x,np,ni,filenum)*dyzero_one_ave_(x,np,ni,filenum)/neuron_scaler

    return

  end function ddyzero_one_ave_

  function dddyzero_one_ave_(x,np,ni,filenum)
    implicit none
    integer, intent(in) :: ni,np,filenum
    real(8) dddyzero_one_ave_(np)
    real(8), intent(in) :: x(ni)

  !    dddyzero_one_=-2.d0*dyzero_one_(x,np,ni)**2-2.d0*yzero_one_(x,np,ni)*ddyzero_one_(x,np,ni)
    dddyzero_one_ave_=-2.d0*dyzero_one_ave_(x,np,ni,filenum)**2/neuron_scaler-&
    2.d0*yzero_one_ave_(x,np,ni,filenum)*ddyzero_one_ave_(x,np,ni,filenum)/neuron_scaler

    return

  end function dddyzero_one_ave_

  function yone_(x,nj,ni)
    use nn_module, only : bone,wone
    implicit none
    integer, intent(in) :: ni,nj
    real(8) yone_(nj)
    real(8), intent(in) :: x(ni)

    yone_=tanh((bone+matmul(wone,x)))

    return

  end function yone_

  function yone_ave_(x,nj,ni,filenum)
    use nn_av_module, only : boneav,woneav
    implicit none
    integer, intent(in) :: ni,nj,filenum
    real(8) yone_ave_(nj)
    real(8), intent(in) :: x(ni)

    yone_ave_=tanh((boneav(:,filenum)+matmul(woneav(:,:,filenum),x)))

    return

  end function yone_ave_

  function ytwo_(x,nk,nj,ni)
    use nn_module, only : btwo,wtwo
    implicit none
    integer, intent(in) :: nk,nj,ni
    real(8), intent(in) :: x(ni)
    real(8) ytwo_(nk)

    ytwo_=tanh((btwo+matmul(wtwo,yone_(x,nj,ni))))

    return

  end function ytwo_

  function ytwo_ave_(x,nk,nj,ni,filenum)
    use nn_av_module, only : btwoav,wtwoav
    implicit none
    integer, intent(in) :: nk,nj,ni,filenum
    real(8), intent(in) :: x(ni)
    real(8) ytwo_ave_(nk)

    ytwo_ave_=tanh((btwoav(:,filenum)+matmul(wtwoav(:,:,filenum),yone_ave_(x,nj,ni,filenum))))

    return

  end function ytwo_ave_

  function dyone_(x,nj,ni)
    implicit none
    integer, intent(in) :: ni,nj
    real(8) dyone_(nj)
    real(8), intent(in) :: x(ni)

    dyone_=(1.d0-yone_(x,nj,ni)**2)

    return

  end function dyone_

  function dyone_ave_(x,nj,ni,filenum)
    implicit none
    integer, intent(in) :: ni,nj,filenum
    real(8) dyone_ave_(nj)
    real(8), intent(in) :: x(ni)

    dyone_ave_=(1.d0-yone_ave_(x,nj,ni,filenum)**2)

    return

  end function dyone_ave_

  function dytwo_(x,nk,nj,ni)
    implicit none
    integer, intent(in) :: nk,nj,ni
    real(8), intent(in) :: x(ni)
    real(8) dytwo_(nk)

    dytwo_=(1.d0-ytwo_(x,nk,nj,ni)**2)

    return

  end function dytwo_

  function dytwo_ave_(x,nk,nj,ni,filenum)
    implicit none
    integer, intent(in) :: nk,nj,ni,filenum
    real(8), intent(in) :: x(ni)
    real(8) dytwo_ave_(nk)

    dytwo_ave_=(1.d0-ytwo_ave_(x,nk,nj,ni,filenum)**2)

    return

  end function dytwo_ave_

  function ddyone_(x,nj,ni)
    implicit none
    integer, intent(in) :: ni,nj
    real(8) ddyone_(nj)
    real(8), intent(in) :: x(ni)

    ddyone_=-2.d0*yone_(x,nj,ni)*dyone_(x,nj,ni)

    return

  end function ddyone_

  function ddyone_ave_(x,nj,ni,filenum)
    implicit none
    integer, intent(in) :: ni,nj,filenum
    real(8) ddyone_ave_(nj)
    real(8), intent(in) :: x(ni)

    ddyone_ave_=-2.d0*yone_ave_(x,nj,ni,filenum)*dyone_ave_(x,nj,ni,filenum)

    return

  end function ddyone_ave_

  function ddytwo_(x,nk,nj,ni)
    implicit none
    integer, intent(in) :: nk,nj,ni
    real(8), intent(in) :: x(ni)
    real(8) ddytwo_(nk)

    ddytwo_=-2.d0*ytwo_(x,nk,nj,ni)*dytwo_(x,nk,nj,ni)

    return

  end function ddytwo_

  function ddytwo_ave_(x,nk,nj,ni,filenum)
    implicit none
    integer, intent(in) :: nk,nj,ni,filenum
    real(8), intent(in) :: x(ni)
    real(8) ddytwo_ave_(nk)

    ddytwo_ave_=-2.d0*ytwo_ave_(x,nk,nj,ni,filenum)*dytwo_ave_(x,nk,nj,ni,filenum)

    return

  end function ddytwo_ave_

  function dddyone_(x,nj,ni)
    implicit none
    integer, intent(in) :: ni,nj
    real(8) dddyone_(nj)
    real(8), intent(in) :: x(ni)

    dddyone_=-2.d0*dyone_(x,nj,ni)**2-2.d0*yone_(x,nj,ni)*ddyone_(x,nj,ni)

    return

  end function dddyone_

  function dddyone_ave_(x,nj,ni,filenum)
    implicit none
    integer, intent(in) :: ni,nj,filenum
    real(8) dddyone_ave_(nj)
    real(8), intent(in) :: x(ni)

    dddyone_ave_=-2.d0*dyone_ave_(x,nj,ni,filenum)**2-2.d0*yone_ave_(x,nj,ni,filenum)*ddyone_ave_(x,nj,ni,filenum)

    return

  end function dddyone_ave_

  function dddytwo_(x,nk,nj,ni)
    implicit none
    integer, intent(in) :: nk,nj,ni
    real(8), intent(in) :: x(ni)
    real(8) dddytwo_(nk)

    dddytwo_=-2.d0*dytwo_(x,nk,nj,ni)**2-2.d0*ytwo_(x,nk,nj,ni)*ddytwo_(x,nk,nj,ni)

    return

  end function dddytwo_

  function dddytwo_ave_(x,nk,nj,ni,filenum)
    implicit none
    integer, intent(in) :: nk,nj,ni,filenum
    real(8), intent(in) :: x(ni)
    real(8) dddytwo_ave_(nk)

    dddytwo_ave_=-2.d0*dytwo_ave_(x,nk,nj,ni,filenum)**2-2.d0*ytwo_ave_(x,nk,nj,ni,filenum)*ddytwo_ave_(x,nk,nj,ni,filenum)

    return


  end function dddytwo_ave_

end module layers

! Initializes PES interpolation (either nn, gpr, ...)
! Some initialization procedure of gpr must be done here to use 
! the commonly (nn and gpr) used data in the pes_ml_module
subroutine pes_init(infile,nvar)
  use dlf_global, only: stderr
  use pes_ml_module
  use gpr_module
  use nn_module
  implicit none
  integer,intent(out) :: nvar
  character(*),intent(in) :: infile
  character(8)                :: pes_type_marker ! string that shows that a PES
                                                 ! type is given 
                                                 ! (if not chose 0 -> nn)
  integer(4) :: pes_type4 
  integer :: IOstatus
  logical :: file_exists

  INQUIRE(file=infile,EXIST=file_exists)
  print*, "infile:", infile
  if (.not.file_exists) STOP "PES file does not exist!"
  open(unit=40,file=infile,form='unformatted')
  read(40,IOSTAT=IOstatus) pes_type_marker, pes_type4
  if (IOstatus .gt. 0) then
    STOP "An error occured when reading the PES file"
  else if (IOstatus .lt. 0) then
    STOP "End of file reached earlier than expected."
  else
    !successfully read
  end if
  if (pes_type_marker=="PESType#") then
    pes_type = INT(pes_type4)
    SELECT CASE (pes_type)
      CASE (0)
        call nn_init(infile,nvar)
      CASE (1)
        print*,"Input is a GPR file: "
        ! Reads all data necessary for GPR interpolation
        call GPR_read(gprPES,infile,IOstatus)
        ! Data that is also used by neural networks must be initialized
        ! in here as well, not only in the gpr module itself:
        nvar=3*gprPES%nat
        allocate(align_refcoords(nvar))
        allocate(align_modes(nvar,gprPES%sdgf))
        allocate(refmass(gprPES%nat))
        pes_massweight=   gprPES%massweight
        align_refcoords=gprPES%align_refcoords
        align_modes =   gprPES%align_modes
        refmass     =   gprPES%refmass
      CASE DEFAULT
        write(stderr,*)"No valid number for PES type is given in the PES file",& 
                        infile
    END SELECT
  else
    REWIND(UNIT=40)
    call nn_init(infile,nvar)
  end if
end subroutine pes_init

subroutine nn_init(infile,nvar)
  use pes_ml_module
  use nn_module
  use nn_av_module
  use gpr_module
  implicit none
  integer,intent(out) :: nvar
  character(*),intent(in) :: infile
  integer(4) :: ns,nat ! not used, except for printout
  integer(4) :: ni4,nj4,nk4,np4,rem_n_dims!,coord_system4
  integer(4), allocatable :: rad_omit(:),badbuildlist(:),badmodes(:),atomindex(:)
  logical(4) :: nmw4,dim_red,coords_ia,coords_inv,coords_bad,&
  tmin
  integer :: ios
  logical ::file_opened
  real(8), allocatable :: storage(:,:)
  INQUIRE(40,OPENED=file_opened) 
  if(.not.file_opened) then
    open(unit=40,file=infile,form='unformatted')
  end if
  read(40,iostat=ios) ni4,nj4,nk4,np4,ns,nat,nmw4,rem_n_dims!coord_system4
  if(ios/=0) then
    !read(40,iostat=ios) ni4,nj4,nk4,ns,nat,nmw4
    print*,"Coordinate system can not be read in, assuming normal coordinates"
!    coord_system4=0
  end if
  pes_massweight=nmw4
  ni=ni4
  nj=nj4
  nk=nk4
  np=np4
  remove_n_dims=rem_n_dims
!tmp:
  if(nat==0) nat=7

  !  pes_massweight=.false.
  print*,"Reading neural network data from file: ",trim(adjustl(infile))
  print*,"Number of atoms",nat
  print*,"Number of coordinates",ni
  print*,"Number of variables in first hidden layer",nj
  print*,"Number of variables in second hidden layer",nk
  print*,"Number of variables in extra hidden layer",np
  print*,"Number of input geometries",ns
  print*,"Mass weighting",pes_massweight
  print*,"Dimensions removed",remove_n_dims

  nvar=3*nat
  if(nvar/3==1)then
    dim_renorm=nvar
  else if(nvar/3==2)then
    dim_renorm=nvar-5
  else
    dim_renorm=nvar-6
  endif
  interatomic_dof=(nvar/3)*(nvar/3-1)/2

  if(avnncalc .and. ifile == 1) then
    npav(1)=np
    nkav(1)=nk
    njav(1)=nj
    if(np.gt.0)then
      allocate(wzero_one(np,2*ni))
      allocate(wzero_two(1,np))
      allocate(bzero_one(np))
      allocate(wzero_oneav(np,2*ni,numfiles))
      allocate(wzero_twoav(1,np,numfiles))
      allocate(bzero_oneav(np,numfiles))
    endif
    allocate(wone(nj,ni))
    allocate(wtwo(nk,nj))
    allocate(wthree(1,nk))
    allocate(bone(nj))
    allocate(btwo(nk))
    allocate(align_refcoords(nvar))
    allocate(align_modes(nvar,dim_renorm))
    allocate(refmass(nat))

    allocate(woneav(nj,ni,numfiles))
    allocate(wtwoav(nk,nj,numfiles))
    allocate(wthreeav(1,nk,numfiles))
    allocate(boneav(nj,numfiles))
    allocate(btwoav(nk,numfiles))
    allocate(bthreeav(numfiles))
    allocate(align_refcoordsav(nvar,numfiles))
    allocate(align_modesav(nvar,dim_renorm,numfiles))
    
    allocate(energyav(numfiles))
    allocate(gradientav(nvar,numfiles))
    allocate(hessianav(nvar,nvar,numfiles))
    allocate(mass_list(nat))
  elseif(avnncalc .and. ifile .gt. 1) then
    if(np.ge.npav(ifile-1))then
      deallocate(wzero_one)
      deallocate(wzero_two)
      deallocate(bzero_one)
      if(allocated(wzero_oneav))then
        allocate(storage(npav(ifile-1),ni))
        storage=wzero_oneav(:,:,ifile-1)
        deallocate(wzero_oneav)
        allocate(wzero_oneav(np,2*ni,numfiles))
        wzero_oneav(1:npav(ifile-1),:,ifile-1)=storage
        deallocate(storage)
        allocate(storage(1,npav(ifile-1)))
        storage=wzero_twoav(:,:,ifile-1)
        deallocate(wzero_twoav)
        allocate(wzero_twoav(1,np,numfiles))
        wzero_twoav(:,1:npav(ifile-1),ifile-1)=storage
        deallocate(storage)
        allocate(storage(1,npav(ifile-1)))
        storage(1,:)=bzero_oneav(:,ifile-1)
        deallocate(bzero_oneav)
        allocate(bzero_oneav(np,numfiles))
        bzero_oneav(1:npav(ifile-1),ifile-1)=storage(1,:)
        deallocate(storage)
      endif
    endif
    npav(ifile)=np
    if(np.gt.0)then
      allocate(wzero_one(np,2*ni))
      allocate(wzero_two(1,np))
      allocate(bzero_one(np))
      if(.not.allocated(wzero_oneav))then
        allocate(wzero_oneav(np,2*ni,numfiles))
        allocate(wzero_twoav(1,np,numfiles))
        allocate(bzero_oneav(np,numfiles))
      endif
    endif
  end if

  if(.not. avnncalc) then
    if(np.gt.0)then
      allocate(wzero_one(np,2*ni))
      allocate(wzero_two(1,np))
      allocate(bzero_one(np))
    endif
    allocate(wone(nj,ni))
    allocate(wtwo(nk,nj))
    allocate(wthree(1,nk))
    allocate(bone(nj))
    allocate(btwo(nk))
    allocate(align_refcoords(nvar))
    allocate(align_modes(nvar,dim_renorm))
    allocate(refmass(nat))
    allocate(mass_list(nat))
  end if

  if(np.gt.0)then
    read(40) wzero_one
    read(40) bzero_one
    read(40) wzero_two
  endif
  read(40) wone
  read(40) wtwo
  read(40) wthree
  read(40) bone
  read(40) btwo
  read(40) bthree
  read(40) align_refcoords
  read(40) align_modes
  read(40) alpha
  !read(40,iostat=ios) alpha
  !if(ios/=0) then
  !  alpha=0.D0 ! this can be removed once all .dat files contain alpha
  !  print*,"Alpha not read from file."
  !end if
  !print*,"Alpha",alpha
  if(pes_massweight) then
    read(40) refmass
    mass_list=refmass
  else
    read(40) refmass
    mass_list=refmass
    refmass=1.D0
  end if
  read(40,iostat=ios) tmin
  temin=tmin
  if(ios/=0) then
    print*,"Information about minimum energy not read"
    temin=.false.
  end if
  if(temin) then
    read(40) emin
    print*,"Minimum energy of NN: ",emin
  end if
  read(40) dim_red,coords_ia,coords_inv,coords_bad
  dimension_reduce=dim_red
  coords_interatomic=coords_ia
  coords_inverse=coords_inv
  coords_b_a_d=coords_bad
  if(dimension_reduce)then
    allocate(pca_eigvect(remove_n_dims+ni,remove_n_dims+ni))
    allocate(mu(remove_n_dims+ni))
    read(40) pca_eigvect,mu
  endif
  if(coords_interatomic.or.coords_inverse)then
    if(.not.avnncalc)then
      read(40) pinv_tol,pinv_tol_back
      allocate(rad_omit(3*nat+1))
      read(40) rad_omit
      allocate(radii_omit(rad_omit(1)))
      radii_omit=rad_omit(2:1+rad_omit(1))
      deallocate(rad_omit)
    else
      if(ifile==1)then
        read(40) pinv_tol,pinv_tol_back
        allocate(rad_omit(3*nat+1))
        read(40) rad_omit
        allocate(radii_omitav(rad_omit(1),numfiles))
        radii_omitav(:,ifile)=rad_omit(2:1+rad_omit(1))
        deallocate(rad_omit)
      else
        read(40) pinv_tol,pinv_tol_back
        allocate(rad_omit(3*nat+1))
        read(40) rad_omit
        radii_omitav(:,ifile)=rad_omit(2:1+rad_omit(1))
        deallocate(rad_omit)
      endif
    endif
  endif
  if(coords_b_a_d)then
    read(40) pinv_tol,pinv_tol_back
    allocate(badbuildlist(2*ni - dim_renorm))
    allocate(atomindex(nat))
    allocate(badmodes(4))
    read(40) badbuildlist
    read(40) atomindex
    read(40) badmodes
    allocate(bad_buildlist(2*ni - dim_renorm))
    allocate(atom_index(nat))
    allocate(bad_modes(4))
    bad_buildlist=badbuildlist
    atom_index=atomindex
    bad_modes=badmodes
    deallocate(badbuildlist)
    deallocate(atomindex)
    deallocate(badmodes)
  endif
  read(40) training_e_ave,largest_TR_eval,ave_seventh_eval
  close(40)

  if(coords_interatomic) print*,"Interatomic coordinates"
  if(coords_inverse) print*,"Inverse interatomic coordinates"
  if(coords_b_a_d) print*,"Bonds/Angles/Dihedrals coordinates"
  if(.not.coords_inverse .and. .not.coords_interatomic .and. .not.coords_b_a_d) print*,"normal coordinates"

  if(avnncalc) then
    if(np.gt.0)then
      wzero_oneav(:,:,ifile)=wzero_one
      bzero_oneav(:,ifile)=bzero_one
      wzero_twoav(:,:,ifile)=wzero_two
    endif
    woneav(:,:,ifile)=wone
    wtwoav(:,:,ifile)=wtwo
    wthreeav(:,:,ifile)=wthree
    boneav(:,ifile)=bone
    btwoav(:,ifile)=btwo
    bthreeav(ifile)=bthree
    align_refcoordsav(:,ifile)=align_refcoords
    align_modesav(:,:,ifile)=align_modes
  end if

end subroutine nn_init

subroutine pes_destroy
  use dlf_global, only: stderr
  use pes_ml_module
  use gpr_module
  SELECT CASE (pes_type)
    CASE (0)  
      call nn_destroy
    CASE (1)
      call gpr_destroy(gprPES)
      deallocate(align_refcoords)
      deallocate(align_modes)
      deallocate(refmass)
    CASE DEFAULT
      write(stderr,*) "No valid PES type is given in the PES file" 
  END SELECT
end subroutine pes_destroy

subroutine nn_destroy
  use pes_ml_module
  use nn_module
  use nn_av_module
  use gpr_module
  implicit none

  deallocate(wone)
  deallocate(wtwo)
  deallocate(wthree)
  deallocate(bone)
  deallocate(btwo)
  deallocate(align_refcoords)
  deallocate(align_modes)
  deallocate(refmass)
  deallocate(mass_list)
  if(np.gt.0)then
    deallocate(wzero_one)
    deallocate(wzero_two)
    deallocate(bzero_one)
  endif

  if(avnncalc)then
    if(np.gt.0)then
      deallocate(wzero_oneav)
      deallocate(wzero_twoav)
      deallocate(bzero_oneav)
    endif
    deallocate(woneav)
    deallocate(wtwoav)
    deallocate(wthreeav)
    deallocate(boneav)
    deallocate(btwoav)
    deallocate(bthreeav)
    deallocate(align_refcoordsav)
    deallocate(align_modesav)
    deallocate(gradientav)
    deallocate(energyav)
  end if

  if(dimension_reduce)then
    deallocate(pca_eigvect)
    deallocate(mu)
  endif

  if(coords_interatomic.or.coords_inverse)then
    if(.not.avnncalc)then
      deallocate(radii_omit)
    else
      deallocate(radii_omitav)
    endif
  endif

  if(coords_b_a_d)then
    deallocate(bad_buildlist)
    deallocate(atom_index)
    deallocate(bad_modes)
  endif

end subroutine nn_destroy

subroutine pes_get_energy(nvar,xcoords,energy) 
 use dlf_global, only: stderr
  use pes_ml_module
  use dlf_parameter_module, only: rk  
  implicit none
  integer, intent(in) :: nvar
  real(rk), intent(in) :: xcoords(nvar) 
  real(rk), intent(out) :: energy
  ! The following variable declarations are copied from 
  ! nn_get_gradient (therefore they are not used in CASE(1) (neural network)
  ! but still necessary for other interpolation methods (coordinate trafo)
  real(rk) :: dcoords(gprPES%sdgf),trans(3),rotmat(3,3)  
  SELECT CASE (pes_type)
    CASE (1)    
      ! coordinate transformation is not done by the gpr module!
      ! Therefore it must be done in here.
      call cgh_xtos(gprPES%sdgf,nvar,gprPES%align_refcoords,&
                    xcoords,trans,rotmat,dcoords)
      call GPR_eval(gprPES, dcoords, energy)
    CASE DEFAULT
      write(stderr,*) "No valid number for PES type is given in the PES file"       
  END SELECT
end subroutine pes_get_energy

! Evaluates energies and gradients on PES surfaces (using either nn, gpr, ...)
subroutine pes_get_gradient(nvar,xcoords,energy,xgradient)
  use dlf_global, only: stderr
  use pes_ml_module
  use nn_module
  use nn_av_module
  use dlf_parameter_module, only: rk  
  implicit none
  integer, intent(in) :: nvar
  real(rk), intent(in) :: xcoords(nvar) 
  real(rk), intent(out) :: energy, xgradient(nvar)
  ! The following variable declarations are copied from 
  ! nn_get_gradient (therefore they are not used in CASE(1) (neural network)
  ! but still necessary for other interpolation methods (coordinate trafo)
  integer :: iat,jat,ij,ii,id,id2
  real(rk) :: trans(3),rotmat(3,3)
  real(rk) :: dcj_dxi(nvar,nvar),svar
  real(rk) :: drotmat(3,3,nvar),com(3),tmpvec(3)
  real(rk), allocatable :: dgradient(:),dcoords(:),dcoords_store(:),&
  x_out(:),xcoords_mw(:),x_out_store(:),DMAT_PINV2(:,:),DMAT_PINV(:,:),DMAT(:,:),KMAT(:,:),xgradient_store(:),&
  align_store(:),mode_store(:,:),rmass(:),projection(:,:)
  logical mweight

  allocate(align_store(nvar))
  allocate(rmass(nvar/3))
  select case(pes_type)
    case(0)
      rmass=refmass
      mweight=pes_massweight
      id=ni+remove_n_dims
      id2=ni
      if(avnncalc) then
        align_store=align_refcoordsav(:,ifile)
        allocate(mode_store(size(align_modesav(:,:,ifile),dim=1),size(align_modesav(:,:,ifile),dim=2)))
        mode_store=align_modesav(:,:,ifile)
      else
        align_store=align_refcoords
        allocate(mode_store(size(align_modes,dim=1),size(align_modes,dim=2)))
        mode_store=align_modes
      endif
    case(1)
      rmass=gprPES%refmass
      mweight=gprPES%massweight
      id=gprPES%sdgf
      id2=gprPES%sdgf
      align_store=gprPES%align_refcoords
      allocate(mode_store(size(gprPES%align_modes,dim=1),size(gprPES%align_modes,dim=2)))
      mode_store=gprPES%align_modes
    case DEFAULT
      write(stderr,*) "No valid number for PES type is given in the PES file"
  end select
  com=0.D0
  do iat=1,nvar/3
    com(:)=com(:)+xcoords(iat*3-2:iat*3)*rmass(iat)
  end do
  com=com/sum(rmass)
  allocate(dcoords(id))
  allocate(dgradient(id2))
  dcoords=0.d0
  if(.not.(coords_interatomic .or. coords_inverse .or. coords_b_a_d))then
    call cgh_xtos(id,nvar,align_store,xcoords,trans,rotmat,dcoords)
    if(dimension_reduce)then
      allocate(dcoords_store(id2))
      dcoords_store=matmul(transpose(pca_eigvect(:,1:id2)),(dcoords-mu))
      dcoords(1:id2)=dcoords_store
      deallocate(dcoords_store)
    endif
  else
    allocate(xcoords_mw(nvar))
    if(.not.coords_b_a_d)then
      allocate(x_out(interatomic_dof))
      allocate(x_out_store(interatomic_dof))
      call coords_to_interatomic(nvar/3,interatomic_dof,xcoords,&
      x_out,xcoords_mw,coords_inverse)
      x_out_store=x_out
      deallocate(x_out)
      allocate(x_out(id))
      ij=0
      do ii=1,interatomic_dof
        if(.not.avnncalc)then
          if(.not.any( radii_omit==ii ))then
            ij=ij+1
            x_out(ij)=x_out_store(ii)
          endif
        else
          if(.not.any( radii_omitav(:,ifile)==ii ))then
            ij=ij+1
            x_out(ij)=x_out_store(ii)
          endif
        endif
      enddo
      deallocate(x_out_store)
    else
      allocate(x_out(id))
      call coords_to_bad(nvar/3,id,xcoords,x_out,xcoords_mw)
    endif
    if(dimension_reduce)then
      dcoords(1:id2)=matmul(transpose(pca_eigvect(:,1:id2)),x_out-mu)
    else
      dcoords=x_out
    endif
  endif

  SELECT CASE (pes_type)
    CASE (0)

      call nn_get_gradient(id2,dcoords,energy,dgradient)

!      print*,dgradient

      if(temin) then
        dgradient=2.D0*energy*dgradient
        energy=emin+energy**2
      end if

  !    energy=energy*training_e_ave(1,2)
!      energy=energy+training_e_ave(1,1)
  !    xgradient=(xgradient)*training_e_ave(2,2)
  !    xgradient=xgradient+training_e_ave(2,1)
      ! re-mass-weight

    CASE (1)
      ! coordinate transformation is not done by the gpr module!
      ! Therefore it must be done in here.

      call GPR_eval(gprPES, dcoords, energy)
      call GPR_eval_grad(gprPES, dcoords, dgradient)


    CASE DEFAULT
      write(stderr,*) "No valid number for PES type is given in the PES file"
  END SELECT
  deallocate(dcoords)


  if(.not.(coords_interatomic .or. coords_inverse .or. coords_b_a_d))then
    if(.not.dimension_reduce)then
      xgradient=matmul(mode_store,dgradient)
    else
      allocate(xgradient_store(id))
      xgradient_store=matmul(pca_eigvect(:,1:id2),dgradient)
      xgradient=matmul(mode_store,xgradient_store)
      deallocate(xgradient_store)
    endif
!  elseif((coords_interatomic .NEQV. coords_inverse))then
  else
    allocate(DMAT(id,nvar))
    allocate(DMAT_PINV(nvar,id))
    allocate(DMAT_PINV2(id,nvar))
    allocate(KMAT(nvar,nvar))
    allocate(projection(id,id))
    if(dimension_reduce)then
      allocate(xgradient_store(id))
      xgradient_store=matmul(pca_eigvect(:,1:id2),dgradient)
      call DKMAT_interatomic(nvar/3,id,xcoords_mw,xgradient_store,&
      x_out,DMAT,DMAT_PINV,DMAT_PINV2,projection,KMAT,coords_inverse,coords_b_a_d)
      xgradient=matmul(transpose(DMAT),xgradient_store)
      deallocate(xgradient_store)
    else
      call DKMAT_interatomic(nvar/3,id,xcoords_mw,dgradient,&
      x_out,DMAT,DMAT_PINV,DMAT_PINV2,projection,KMAT,coords_inverse,coords_b_a_d)
      xgradient=matmul(transpose(DMAT),dgradient)
    endif
    deallocate(projection)
    deallocate(KMAT)
    deallocate(DMAT)
    deallocate(DMAT_PINV)
    deallocate(DMAT_PINV2)
    deallocate(x_out)
    deallocate(xcoords_mw)
  endif
  deallocate(mode_store)
  deallocate(dgradient)

  if(mweight) then
    do iat=1,nvar/3
      xgradient(iat*3-2:iat*3)=xgradient(iat*3-2:iat*3)*sqrt(rmass(iat))
    end do
  end if

!  print*,'GRADIENT'
!  print*,xgradient

  if(.not.(coords_interatomic .or. coords_inverse .or. coords_b_a_d))then
  ! now transform gradient back:
    call get_drotmat(id,nvar,align_store,xcoords,drotmat)
    
    dcj_dxi=0.D0
    svar=sum(rmass)
    do iat=1,nvar/3
      dcj_dxi(iat*3-2:iat*3,iat*3-2:iat*3)=transpose(rotmat)
    end do

    do iat=1,nvar/3
      do jat=1,nvar
        tmpvec=matmul(drotmat(:,:,jat),(xcoords(iat*3-2:iat*3)-com))
        dcj_dxi(jat,iat*3-2:iat*3)= dcj_dxi(jat,iat*3-2:iat*3) + tmpvec
      end do
    end do
    
    xgradient=matmul(dcj_dxi,xgradient)
  endif
  deallocate(align_store)
  deallocate(rmass)

end subroutine pes_get_gradient

subroutine nn_get_gradient(ni_in,dcoords,energy,dgradient)
  use dlf_global, only: glob
  use dlf_parameter_module, only: rk
  use pes_ml_module
  use nn_module
  use nn_av_module
  use gpr_module
  use layers
  implicit none
  integer, intent(in)   :: ni_in
  real(rk), intent(in) :: dcoords(ni+remove_n_dims)
  real(rk), intent(out) :: energy, dgradient(ni)
  real(rk) :: yone(nj),ytwo(nk),store(1)
  real(rk) :: dyone(nj),dytwo(nk)
  real(rk), allocatable :: yzero_one(:),dyzero_one(:)
  integer :: ii,ij,ik,p

  if (ni_in/=ni) call dlf_fail("Some inconsistency of 'ni' when calling nn_get_gradient")
  !Average NNs
  if (avnncalc) then

    yone=yone_ave_(dcoords,nj,ni,ifile)
    dyone=dyone_ave_(dcoords,nj,ni,ifile)
    ytwo=ytwo_ave_(dcoords,nk,nj,ni,ifile)
    dytwo=dytwo_ave_(dcoords,nk,nj,ni,ifile)
    if(npav(ifile).gt.0)then
      allocate(yzero_one(npav(ifile)))
      allocate(dyzero_one(npav(ifile)))
      yzero_one=yzero_one_ave_(dcoords,npav(ifile),ni,ifile)
      dyzero_one=dyzero_one_ave_(dcoords,npav(ifile),ni,ifile)
      store=bthreeav(ifile)+matmul(wthreeav(:,:,ifile),ytwo)+matmul(wzero_twoav(:,1:npav(ifile),ifile),yzero_one)+&
      alpha*(sum(dcoords**2,dim=1))
      deallocate(yzero_one)
    else
      store=bthreeav(ifile)+matmul(wthreeav(:,:,ifile),ytwo)+alpha*sum(dcoords**2,dim=1)
    endif
    energy=store(1)

    do ii=1,ni
      if(npav(ifile).gt.0)then
        dgradient(ii)=2.d0*alpha*dcoords(ii)
      else
        dgradient(ii)=2.D0*alpha*dcoords(ii)
      endif
      do ik=1,nk
        do ij=1,nj
          dgradient(ii)=dgradient(ii)+(wthreeav(1,ik,ifile)*wtwoav(ik,ij,ifile)*woneav(ij,ii,ifile)*dytwo(ik)*dyone(ij))
        end do
      end do
      if(npav(ifile).gt.0)then
        do p=1,npav(ifile)
          dgradient(ii)=dgradient(ii)+(-dcoords(ii)**(-2)*wzero_oneav(p,ii,ifile)+2.d0*dcoords(ii)*wzero_oneav(p,ii+ni,ifile))*&
          wzero_twoav(1,p,ifile)*dyzero_one(p)
!          gradientvalue(i)=gradientvalue(i)+wzero_one(p,i)*wzero_two(1,p)*dyzero_one(p)
!          dgradient(ii)=dgradient(ii)+(-dcoords(ii)**(-2)*wzero_oneav(p,ii,ifile))*&
!          wzero_twoav(1,p,ifile)*dyzero_one(p)

        enddo
      endif
    end do
    if(npav(ifile).gt.0)then
      deallocate(dyzero_one)
    endif

  else

    yone=yone_(dcoords,nj,ni)
    dyone=dyone_(dcoords,nj,ni)
    ytwo=ytwo_(dcoords,nk,nj,ni)
    dytwo=dytwo_(dcoords,nk,nj,ni)
    if(np.gt.0)then
      allocate(yzero_one(np))
      allocate(dyzero_one(np))
      yzero_one=yzero_one_(dcoords,np,ni)
      dyzero_one=dyzero_one_(dcoords,np,ni)
!      store=bthree+matmul(wthree,ytwo)+matmul(wzero_two,yzero_one)+alpha*(sum(dcoords**2,dim=1)+&
!      sum(dcoords**(4),dim=1))
      store=bthree+matmul(wthree,ytwo)+matmul(wzero_two,yzero_one)+alpha*(sum(dcoords**2,dim=1))
      deallocate(yzero_one)
    else
      store=bthree+matmul(wthree,ytwo)+alpha*sum(dcoords**2,dim=1)
    endif
    energy=store(1)

    do ii=1,ni
      if(np.gt.0)then
        dgradient(ii)=2.d0*alpha*dcoords(ii)
!        dgradient(ii)=4.d0*alpha*dcoords(ii)
      else
        dgradient(ii)=2.D0*alpha*dcoords(ii)
      endif
      do ik=1,nk
        do ij=1,nj
          dgradient(ii)=dgradient(ii)+(wthree(1,ik)*wtwo(ik,ij)*wone(ij,ii)*dytwo(ik)*dyone(ij))
        end do
      end do
      if(np.gt.0)then
        do p=1,np
          dgradient(ii)=dgradient(ii)+(-dcoords(ii)**(-2)*wzero_one(p,ii)+2.d0*dcoords(ii)*wzero_one(p,ii+ni))*&
          wzero_two(1,p)*dyzero_one(p)
!          dgradient(ii)=dgradient(ii)+wzero_one(p,ii)*wzero_two(1,p)*dyzero_one(p)
!          dgradient(ii)=dgradient(ii)+(-dcoords(ii)**(-2)*wzero_one(p,ii))*&
!          wzero_two(1,p)*dyzero_one(p)
!          dgradient(ii)=dgradient(ii)+(2.d0*dcoords(ii)*wzero_one(p,ii))*&
!          wzero_two(1,p)*dyzero_one(p)
        enddo
      endif
    end do
    if(np.gt.0)then
      deallocate(dyzero_one)
    endif

!! SHOULD PREVENT ATOMS FUSING
!    if(coords_interatomic)then
!      do ii=1,ni
!        if(dcoords(ii).lt.1.d0)then
!          energy=energy+1.d0/dcoords(ii)-1.d0
!          dgradient(ii)=dgradient(ii)-1.d0/dcoords(ii)**2
!        endif
!      enddo
!    elseif(coords_inverse)then
!      do ii=1,ni
!        if(dcoords(ii)**(-1).lt.1.d0)then
!          energy=energy+dcoords(ii)-1.d0
!          dgradient(ii)=dgradient(ii)-dcoords(ii)**2
!        endif
!      enddo
!    endif

  end if !avnncalc

end subroutine nn_get_gradient

! Evaluates Hessians on PES surfaces (using either nn, gpr, ...)
subroutine pes_get_hessian(nvar, xcoords, xhessian)
  use dlf_global, only: stderr
  use pes_ml_module
  use nn_module
  use nn_av_module
  use dlf_parameter_module, only: rk
  implicit none  
  integer, intent(in) :: nvar
  real(rk), intent(in) :: xcoords(nvar)
  real(rk), intent(out) :: xhessian(nvar,nvar)
  ! The following variable declarations are copied from 
  ! nn_get_hessian (therefore they are not used in CASE(1) (neural network)
  ! but still necessary for other interpolation methods (coordinate trafo)
  integer  ::   iat,jat,kat,ij,ii,ik,id,id2,nzero
  real(rk) ::   trans(3),rotmat(3,3)
  real(rk) :: dcj_dxi(nvar,nvar),svar
  real(rk) :: drotmat(3,3,nvar),com(3),tmpvec(3)
  real(rk) :: ddrotmat(3,3,nvar,nvar),xgradient(nvar)
  real(rk) :: dcjdxidxj(nvar,nvar,nvar),energy
  real(rk), allocatable :: dhessian(:,:),dgradient(:),dcoords(:),dcoords_store(:),&
  x_out(:),x_out_store(:),DMAT_PINV(:,:),DMAT_PINV2(:,:),DMAT(:,:),KMAT(:,:),KMAT_Q(:,:),xgradient_store(:),xcoords_mw(:),&
  xhessian_store(:,:),align_store(:),mode_store(:,:),rmass(:),projection(:,:),&
  eval(:),evec(:,:,:),evalm(:,:),dhessian_nokm(:,:),mode_coords(:),trproj(:,:)
  logical mweight

  nzero=nvar-dim_renorm
  allocate(align_store(nvar))
  allocate(rmass(nvar/3))
  select case(pes_type)
    case(0)
      rmass=refmass
      mweight=pes_massweight
      id=ni+remove_n_dims
      id2=ni
      if(avnncalc) then
        align_store=align_refcoordsav(:,ifile)
        allocate(mode_store(size(align_modesav(:,:,ifile),dim=1),size(align_modesav(:,:,ifile),dim=2)))
        mode_store=align_modesav(:,:,ifile)
      else
        align_store=align_refcoords
        allocate(mode_store(size(align_modes,dim=1),size(align_modes,dim=2)))
        mode_store=align_modes
      endif
    case(1)
      rmass=gprPES%refmass
      mweight=gprPES%massweight
      id=gprPES%sdgf
      id2=gprPES%sdgf
      align_store=gprPES%align_refcoords
      allocate(mode_store(size(gprPES%align_modes,dim=1),size(gprPES%align_modes,dim=2)))
      mode_store=gprPES%align_modes
    case DEFAULT
      write(stderr,*) "No valid number for PES type is given in the PES file"
  end select
  com=0.D0
  do iat=1,nvar/3
    com(:)=com(:)+xcoords(iat*3-2:iat*3)*rmass(iat)
  end do
  com=com/sum(rmass)
  allocate(dcoords(id))
  allocate(dgradient(id2))
  allocate(dhessian(id2,id2))
  dcoords=0.d0
  if(.not.(coords_interatomic .or. coords_inverse .or. coords_b_a_d))then
    call cgh_xtos(id,nvar,align_store,xcoords,trans,rotmat,dcoords)
    if(dimension_reduce)then
      allocate(dcoords_store(id2))
      dcoords_store=matmul(transpose(pca_eigvect(:,1:id2)),(dcoords-mu))
      dcoords(1:id2)=dcoords_store
      deallocate(dcoords_store)
    endif
  else
    allocate(mode_coords(nvar-6))
    call cgh_xtos(nvar-6,nvar,align_store,xcoords,trans,rotmat,mode_coords)
    deallocate(mode_coords)
    allocate(xcoords_mw(nvar))
    if(.not.coords_b_a_d)then
      allocate(x_out(interatomic_dof))
      allocate(x_out_store(interatomic_dof))
      call coords_to_interatomic(nvar/3,interatomic_dof,xcoords,&
      x_out,xcoords_mw,coords_inverse)
      x_out_store=x_out
      deallocate(x_out)
      allocate(x_out(id))
      ij=0
      do ii=1,interatomic_dof
        if(.not.avnncalc)then
          if(.not.any( radii_omit==ii ))then
            ij=ij+1
            x_out(ij)=x_out_store(ii)
          endif
        else
          if(.not.any( radii_omitav(:,ifile)==ii ))then
            ij=ij+1
            x_out(ij)=x_out_store(ii)
          endif
        endif
      enddo
      deallocate(x_out_store)
    else
      allocate(x_out(id))
      call coords_to_bad(nvar/3,id,xcoords,x_out,xcoords_mw)
    endif
    if(dimension_reduce)then
      dcoords(1:id2)=matmul(transpose(pca_eigvect(:,1:id2)),x_out-mu)
    else
      dcoords=x_out
    endif
  endif

  SELECT CASE (pes_type)
    CASE (0)

      call nn_get_hessian(dcoords,energy,dgradient,dhessian)
      ! transform dhessian to emin if required
      if(temin) then
        ! have to re-calculate energy and gradient for conversion:
        dhessian=energy*dhessian
        do ik=1,id2
          do ij=1,id2
            dhessian(ik,ij)=dhessian(ik,ij)+dgradient(ik)*dgradient(ij)
          end do
        end do
        dhessian=dhessian*2.D0
        dgradient=2.D0*energy*dgradient
      end if !

    CASE (1)

      call GPR_eval_hess(gprPES,dcoords,dhessian)
      call GPR_eval_grad(gprPES, dcoords, dgradient)

    CASE DEFAULT
      write(stderr,*) "No valid number for PES type is given in the PES file" 
  END SELECT
  deallocate(dcoords)

  ! transform from dgradient to gradient
  if(.not.(coords_interatomic .or. coords_inverse .or. coords_b_a_d))then
    if(.not.dimension_reduce)then
      xhessian=matmul(mode_store,matmul(dhessian,transpose(mode_store)))
    else
      allocate(xhessian_store(id,id))
      xhessian_store=matmul(pca_eigvect(:,1:id2),matmul(dhessian,transpose(pca_eigvect(:,1:id2))))
      xhessian=matmul(mode_store,matmul(xhessian_store,transpose(mode_store)))
      deallocate(xhessian_store)
    endif
  else
    allocate(DMAT(id,nvar))
    allocate(DMAT_PINV(nvar,id))
    allocate(DMAT_PINV2(id,nvar))
    allocate(KMAT(nvar,nvar))
    allocate(KMAT_q(id,id))
    allocate(evalm(id,id))
    allocate(projection(id,id))
    if(dimension_reduce)then
      allocate(xgradient_store(id))
      xgradient_store=matmul(pca_eigvect(:,1:id2),dgradient(1:id2))
      call DKMAT_interatomic(nvar/3,id,xcoords_mw,xgradient_store,&
      x_out,DMAT,DMAT_PINV,DMAT_PINV2,projection,KMAT,coords_inverse,coords_b_a_d)
      KMAT_Q=matmul(matmul(DMAT_PINV2,KMAT),DMAT_PINV)
      evalm=KMAT_Q
      do ij=1,id
        do ii=ij,id
          KMAT_Q(ii,ij)=0.5d0*(evalm(ii,ij)+evalm(ij,ii))
          KMAT_Q(ij,ii)=KMAT_Q(ii,ij)
        enddo
      enddo
      xgradient=matmul(transpose(DMAT),xgradient_store)
      deallocate(xgradient_store)
      allocate(xhessian_store(id,id))
      allocate(dhessian_nokm(id,id))
      dhessian_nokm=matmul(matmul(transpose(projection),matmul(pca_eigvect(:,1:id2),&
      matmul(dhessian,transpose(pca_eigvect(:,1:id2))))),transpose(projection))
      evalm=dhessian_nokm
      do ij=1,id
        do ii=ij,id
          dhessian_nokm(ii,ij)=0.5d0*(evalm(ii,ij)+evalm(ij,ii))
          dhessian_nokm(ij,ii)=dhessian_nokm(ii,ij)
        enddo
      enddo
      xhessian_store=dhessian_nokm+KMAT_Q
      deallocate(xgradient_store)
      xhessian=matmul(transpose(DMAT),matmul(xhessian_store,DMAT))
      deallocate(xhessian_store)
      allocate(xhessian_store(nvar,nvar))
      xhessian_store=matmul(transpose(DMAT),matmul(dhessian_nokm,DMAT))+KMAT
      deallocate(dhessian_nokm)
    else
      call DKMAT_interatomic(nvar/3,id,xcoords_mw,dgradient,&
      x_out,DMAT,DMAT_PINV,DMAT_PINV2,projection,KMAT,coords_inverse,coords_b_a_d)
      KMAT_Q=matmul(matmul(DMAT_PINV2,KMAT),DMAT_PINV)
      evalm=KMAT_Q
      do ij=1,id
        do ii=ij,id
          KMAT_Q(ii,ij)=0.5d0*(evalm(ii,ij)+evalm(ij,ii))
          KMAT_Q(ij,ii)=KMAT_Q(ii,ij)
        enddo
      enddo
      xgradient=matmul(transpose(DMAT),dgradient)
      allocate(dhessian_nokm(id,id))
      dhessian_nokm=matmul(matmul(transpose(projection),dhessian),transpose(projection))
      evalm=dhessian_nokm
      do ij=1,id
        do ii=ij,id
          dhessian_nokm(ii,ij)=0.5d0*(evalm(ii,ij)+evalm(ij,ii))
          dhessian_nokm(ij,ii)=dhessian_nokm(ii,ij)
        enddo
      enddo
      xhessian=matmul(transpose(DMAT),matmul(dhessian_nokm+KMAT_Q,DMAT))
      allocate(xhessian_store(nvar,nvar))
      xhessian_store=matmul(transpose(DMAT),matmul(dhessian_nokm,DMAT))+KMAT
      deallocate(dhessian_nokm)
    endif
    deallocate(evalm)
    allocate(evalm(nvar,nvar))
    allocate(evec(nvar,nvar,3))
    call dlf_matrix_diagonalise(nvar,xhessian,evalm(:,1),evec(:,:,1))
    call dlf_matrix_diagonalise(nvar,xhessian_store,evalm(:,2),evec(:,:,2))
    evec(:,:,3)=matmul(transpose(evec(:,:,1)),evec(:,:,2))
    xhessian_store=0.d0
    do ij=1,nvar
      ii=maxloc(abs(evec(:,ij,3)),dim=1)
      xhessian_store(ij,ij)=evalm(ii,1)
    enddo
    deallocate(evalm)
!    xhessian=matmul(matmul(evec(:,:,2),xhessian_store),transpose(evec(:,:,2)))
    deallocate(evec)
    xhessian_store=xhessian
    do ij=1,nvar
      do ii=ij+1,nvar
        xhessian(ij,ii)=0.5d0*(xhessian_store(ij,ii)+xhessian_store(ii,ij))
        xhessian(ii,ij)=xhessian(ij,ii)
      enddo
    enddo
    do ij=1,nvar
      xhessian(ij,ij)=xhessian_store(ij,ij)
    enddo
    deallocate(xhessian_store)

!    print*,'GRADIENT'
!    print*,xgradient
!    print*,'HESSIAN'
!    print*,xhessian(:,1)
!    print*,'KMAT'
!    print*,KMAT_Q
!    stop
!    allocate(eval(nvar))
!    allocate(evec(nvar,nvar,2))
!    allocate(evalm(nvar,nvar))
!    evalm=0.d0
!    call r_diagonal(nvar,xhessian,eval,evec(:,:,1))
!    do ii=1,nvar
!      if(abs(eval(ii)).gt.largest_TR_eval/100.d0)then
!        evalm(ii,ii)=eval(ii)
!      else
!        if(eval(ii).lt.0.d0)then
!          evalm(ii,ii)=-eval(ii)
!        else
!          evalm(ii,ii)=eval(ii)
!        endif
!      endif
!    enddo
!!    if(eval(1).gt.-largest_TR_eval)then
!!      evalm(1:6,1:6)=0.d0
!!      evalm(7,7)=max(eval(7),ave_seventh_eval)
!!    else
!!      evalm(2:7,2:7)=0.d0
!!      evalm(8,8)=max(eval(8),ave_seventh_eval)
!!    endif
!    xhessian=matmul(matmul(evec(:,:,1),evalm),transpose(evec(:,:,1)))
!    deallocate(eval)
!    deallocate(evec)
!    deallocate(evalm)

    deallocate(projection)
    deallocate(KMAT)
    deallocate(KMAT_q)
    deallocate(DMAT)
    deallocate(DMAT_PINV)
    deallocate(DMAT_PINV2)
    deallocate(x_out)
    deallocate(xcoords_mw)
  endif
  deallocate(dhessian)

  ! re-mass-weight
  if(mweight) then
    do iat=1,nvar/3
      do jat=1,nvar/3
        xhessian(iat*3-2:iat*3,jat*3-2:jat*3)=xhessian(iat*3-2:iat*3,jat*3-2:jat*3)&
          *sqrt(rmass(iat)*rmass(jat))
      end do
    end do
  end if

  ! now transform hessian back:
!  if(.not.(coords_interatomic .or. coords_inverse .or. coords_b_a_d))then
!    call get_drotmat(id,nvar,align_store,xcoords,drotmat)
  call get_drotmat(nvar-6,nvar,align_store,xcoords,drotmat)

  dcj_dxi=0.D0
  do iat=1,nvar/3
    dcj_dxi(iat*3-2:iat*3,iat*3-2:iat*3)=transpose(rotmat)
  end do

  do iat=1,nvar/3
    do jat=1,nvar
      tmpvec=matmul(drotmat(:,:,jat),(xcoords(iat*3-2:iat*3)-com))
      dcj_dxi(jat,iat*3-2:iat*3)= dcj_dxi(jat,iat*3-2:iat*3) + tmpvec
    end do
  end do
!  call r_diagonal(nvar,matmul(matmul(dcj_dxi,xhessian),transpose(dcj_dxi)),eval,evec(:,:,2))
!  call r_diagonal(nvar,xhessian,eval,evec(:,:,1))
  xhessian=matmul(matmul(dcj_dxi,xhessian),transpose(dcj_dxi))

  if(.not.(coords_interatomic .or. coords_inverse .or. coords_b_a_d))then
!    call get_ddrotmat(id,nvar,align_store,xcoords,ddrotmat)
    call get_ddrotmat(nvar-6,nvar,align_store,xcoords,ddrotmat)

    ! transform from dgradient to xgradient
    xgradient=matmul(mode_store,dgradient)
    ! re-mass-weight
    if(mweight) then
      do iat=1,nvar/3
        xgradient(iat*3-2:iat*3)=xgradient(iat*3-2:iat*3)*&
            sqrt(rmass(iat))
      end do
    end if

    dcjdxidxj(:,:,:)=0.D0
    ! fist index: c_k
    ! sedond,third index: xi, xj
    do iat=1,nvar/3 ! atoms a=c
      do jat=1,nvar
        dcjdxidxj(iat*3-2:iat*3,iat*3-2:iat*3,jat)= &
        dcjdxidxj(iat*3-2:iat*3,iat*3-2:iat*3,jat) + &
            drotmat(:,:,jat)
      end do
    end do
    do iat=1,nvar/3 ! atoms a=b
      do jat=1,nvar
        dcjdxidxj(iat*3-2:iat*3,jat,iat*3-2:iat*3)= &
        dcjdxidxj(iat*3-2:iat*3,jat,iat*3-2:iat*3) + &
            drotmat(:,:,jat)
      end do
    end do

    do iat=1,nvar
      do jat=1,nvar
        do kat=1,nvar/3
          dcjdxidxj(kat*3-2:kat*3,iat,jat)=dcjdxidxj(kat*3-2:kat*3,iat,jat) + &
              matmul(ddrotmat(:,:,iat,jat),(xcoords(kat*3-2:kat*3)-com))
        end do
      end do
    end do

    !hessian = hessian + sum_k dcjdxidxj_k:: * xgradient(k)
    do iat=1,nvar
      do jat=1,nvar
        xhessian(iat,jat)=xhessian(iat,jat)+&
            sum(dcjdxidxj(:,iat,jat)*xgradient(:))
      end do
    end do
  endif
  deallocate(rmass)
  deallocate(dgradient)
  deallocate(align_store)
  deallocate(mode_store)

end subroutine pes_get_hessian

subroutine nn_get_hessian(dcoords,energy,dgradient,dhessian)
  use dlf_parameter_module, only: rk
  use pes_ml_module
  use nn_module
  use nn_av_module
  use gpr_module
  use layers
  implicit none
  real(rk), intent(in) :: dcoords(ni+remove_n_dims)
  real(rk), intent(out) :: dhessian(ni,ni),dgradient(ni),energy
  real(rk) :: yone(nj),ytwo(nk),yzero
  real(rk) :: dyone(nj),dytwo(nk),svar ! dyone = f_1'
  real(rk) :: ddytwo(nk),ddyone(nj)
  real(rk), allocatable :: dyzero_one(:),ddyzero_one(:)
  integer :: ii,ij,ik,iip,ip
  real(rk) :: amat(nk,ni),kronecker

  !print*,"dcoords to evaluate Hessian at",dcoords

  !print*,"Mass",glob%mass
  if(avnncalc) then
    !print*,"Mass",glob%mass  
    if(npav(ifile).gt.0)then
      allocate(dyzero_one(npav(ifile)))
      allocate(ddyzero_one(npav(ifile)))
      dyzero_one=dyzero_one_ave_(dcoords,npav(ifile),ni,ifile)
      ddyzero_one=ddyzero_one_ave_(dcoords,npav(ifile),ni,ifile)
    endif
    dyone=dyone_ave_(dcoords,nj,ni,ifile)
    ddyone=ddyone_ave_(dcoords,nj,ni,ifile)
    ddytwo=ddytwo_ave_(dcoords,nk,nj,ni,ifile)
    dytwo=dytwo_ave_(dcoords,nk,nj,ni,ifile)

    call nn_get_gradient(ni,dcoords,energy,dgradient)

    !!print*,"ytwo",ytwo
    amat=0.D0
    ! order: ni*nk*nj
    do ii=1,ni
      do ik=1,nk
        amat(ik,ii)=sum(wtwoav(ik,:,ifile)*dyone(:)*woneav(:,ii,ifile))
      end do
    end do
    dhessian=0.D0
  ! order: ni*ni*nk*nj = ni^2 * nj * nk
    do ii=1,ni
      do iip=ii,ni
        svar=0.D0
        do ik=1,nk
          svar=svar+wthreeav(1,ik,ifile)*ddytwo(ik)*amat(ik,ii)*amat(ik,iip) + &
               wthreeav(1,ik,ifile)*dytwo(ik)*&
               sum(wtwoav(ik,:,ifile)*woneav(:,ii,ifile)*woneav(:,iip,ifile)*ddyone(:))
        end do
        if(npav(ifile).gt.0)then
          do ip=1,npav(ifile)
!LINEAR TERMS ONLY
!            svar=svar+wzero_oneav(ip,ii,ifile)*wzero_oneav(ip,iip,ifile)*wzero_two(1,ip)*ddyzero_one(ip)
!INVERSE ONLY
!            svar=svar+(-dcoords(ii)**(-2)*wzero_oneav(ip,ii,ifile))*&
!            (-dcoords(iip)**(-2)*wzero_oneav(ip,iip,ifile))*&
!            wzero_twoav(1,ip,ifile)*ddyzero_one(ip)+&
!            kronecker(iip,ii)*(2.d0*dcoords(ii)**(-3)*wzero_oneav(ip,iip,ifile))*&
!            wzero_twoav(1,ip,ifile)*dyzero_one(ip)
!INVERSE AND QUADRATIC
            svar=svar+(-dcoords(ii)**(-2)*wzero_oneav(ip,ii,ifile)+2.d0*wzero_oneav(ip,ii+ni,ifile)*dcoords(ii))*&
            (-dcoords(iip)**(-2)*wzero_oneav(ip,iip,ifile)+2.d0*wzero_oneav(ip,iip+ni,ifile)*dcoords(iip))*&
            wzero_twoav(1,ip,ifile)*ddyzero_one(ip)+&
            kronecker(iip,ii)*(2.d0*dcoords(ii)**(-3)*wzero_oneav(ip,iip,ifile)+2.d0*wzero_oneav(ip,iip+ni,ifile))*&
            wzero_twoav(1,ip,ifile)*dyzero_one(ip)
          end do
        endif

        dhessian(iip,ii)=svar
        dhessian(ii,iip)=svar
      end do
      if(npav(ifile).gt.0)then
        dhessian(ii,ii)=dhessian(ii,ii)+2.D0*alpha
      else
        dhessian(ii,ii)=dhessian(ii,ii)+2.D0*alpha
      endif
    end do
    if(npav(ifile).gt.0)then
      deallocate(dyzero_one)
      deallocate(ddyzero_one)
    endif

!Single NN
  else
    if(np.gt.0)then
      allocate(dyzero_one(np))
      allocate(ddyzero_one(np))
      dyzero_one=dyzero_one_(dcoords,np,ni)
      ddyzero_one=ddyzero_one_(dcoords,np,ni)
    endif
    dyone=dyone_(dcoords,nj,ni)
    ddyone=ddyone_(dcoords,nj,ni)
    ddytwo=ddytwo_(dcoords,nk,nj,ni)
    dytwo=dytwo_(dcoords,nk,nj,ni)

    call nn_get_gradient(ni,dcoords,energy,dgradient)

    !!print*,"ytwo",ytwo
    amat=0.D0
    ! order: ni*nk*nj
    do ii=1,ni
      do ik=1,nk
        amat(ik,ii)=sum(wtwo(ik,:)*dyone(:)*wone(:,ii))
      end do
    end do
    dhessian=0.D0
  ! order: ni*ni*nk*nj = ni^2 * nj * nk
    do ii=1,ni
      do iip=ii,ni
        svar=0.D0
        do ik=1,nk
          svar=svar+wthree(1,ik)*ddytwo(ik)*amat(ik,ii)*amat(ik,iip) + &
               wthree(1,ik)*dytwo(ik)*&
               sum(wtwo(ik,:)*wone(:,ii)*wone(:,iip)*ddyone(:))
        end do
        if(np.gt.0)then
          do ip=1,np
!LINEAR TERMS ONLY
!            svar=svar+wzero_one(ip,ii)*wzero_one(ip,iip)*wzero_two(1,ip)*ddyzero_one(ip)
!INVERSE AND QUADRATIC TERMS
            svar=svar+(-dcoords(ii)**(-2)*wzero_one(ip,ii)+2.d0*wzero_one(ip,ii+ni)*dcoords(ii))*&
            (-dcoords(iip)**(-2)*wzero_one(ip,iip)+2.d0*wzero_one(ip,iip+ni)*dcoords(iip))*wzero_two(1,ip)*ddyzero_one(ip)+&
            kronecker(iip,ii)*(2.d0*dcoords(ii)**(-3)*wzero_one(ip,iip)+2.d0*wzero_one(ip,iip+ni))*wzero_two(1,ip)*dyzero_one(ip)
!INVERSE ONLY
!            svar=svar+(-dcoords(ii)**(-2)*wzero_one(ip,ii))*&
!            (-dcoords(iip)**(-2)*wzero_one(ip,iip))*wzero_two(1,ip)*ddyzero_one(ip)+&
!            kronecker(iip,ii)*(2.d0*dcoords(ii)**(-3)*wzero_one(ip,iip))*wzero_two(1,ip)*dyzero_one(ip)
!QUADRATIC ONLY
!            svar=svar+(2.d0*dcoords(ii)*wzero_one(ip,ii))*&
!            (2.d0*dcoords(iip)*wzero_one(ip,iip))*wzero_two(1,ip)*ddyzero_one(ip)+&
!            kronecker(iip,ii)*(2.d0*wzero_one(ip,iip))*wzero_two(1,ip)*dyzero_one(ip)
          end do
        endif
        dhessian(iip,ii)=svar
        dhessian(ii,iip)=svar
      end do
      if(np.gt.0)then
        dhessian(ii,ii)=dhessian(ii,ii)+2.D0*alpha
!        dhessian(ii,ii)=dhessian(ii,ii)+2.D0*alpha+12.d0*alpha*dcoords(ii)**(2)
      else
        dhessian(ii,ii)=dhessian(ii,ii)+2.D0*alpha
      endif
    end do
    if(np.gt.0)then
      deallocate(dyzero_one)
      deallocate(ddyzero_one)
    endif

!! SHOULD PREVENT ATOMS FUSING
!    if(coords_interatomic)then
!      do ii=1,ni
!        if(dcoords(ii).lt.1.d0)then
!          dhessian(ii,ii)=dhessian(ii,ii)+2.d0/dcoords(ii)**3
!        endif
!      enddo
!    elseif(coords_inverse)then
!      do ii=1,ni
!        if(dcoords(ii)**(-1).lt.1.d0)then
!          dhessian(ii,ii)=dhessian(ii,ii)+2.d0*dcoords(ii)**3
!        endif
!      enddo
!    endif

  end if !avnncalc
!  do iat=1,nvar
!    write(*,*)xhessian(iat,1)
!  enddo
!  pause
!  deallocate(dcoords)
!  deallocate(xcoords_store)

end subroutine nn_get_hessian
!-------------------------------------------------------------------------

! superimpose coords to rcoords and transform gradient and hessian in
! an appropriate way as well. Return transformed coordinates, ...
! this is the same routine as in readin.f90 of the NN fitting code, but with the gradient, hessian removed.
! (the name of the module is also changed: pes_module -> nn_module)
! nvar is added (3*nat replaced by nvar)
!
! Algorithm for superposition:
! !! See W. Kabsch, Acta Cryst. A 32, p 922 (1976).
!
subroutine cgh_xtos(ncoord,nvar,rcoords,xcoords_,trans,rotmat,dcoords)
  use pes_ml_module
  use nn_module
  use nn_av_module
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

  if(pes_massweight) then
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
  call matrix_diagonalise(3,rsmat,eigval,eigvec)

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

end subroutine cgh_xtos

subroutine get_drotmat(ncoord,nvar,rcoords,xcoords_,drotmat)
  use pes_ml_module
  use nn_module
  use nn_av_module
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
    call cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,drotmat(:,:,ivar),dcoords)
    xcoords(ivar)=xcoords(ivar)-2.D0*delta
    call cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat,dcoords)
    drotmat(:,:,ivar)=(drotmat(:,:,ivar)-tmpmat)/2.D0/delta
  end do
end subroutine get_drotmat

subroutine get_ddrotmat(ncoord,nvar,rcoords,xcoords_,ddrotmat)
  use pes_ml_module
  use nn_module
  use nn_av_module
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
      call cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
      tmpmat=tmpmat+tmpmat2

      xcoords=xcoords_
      xcoords(ivar)=xcoords(ivar)-delta
      xcoords(jvar)=xcoords(jvar)+delta
      call cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
      tmpmat=tmpmat-tmpmat2

      xcoords=xcoords_
      xcoords(ivar)=xcoords(ivar)+delta
      xcoords(jvar)=xcoords(jvar)-delta
      call cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
      tmpmat=tmpmat-tmpmat2

      xcoords=xcoords_
      xcoords(ivar)=xcoords(ivar)-delta
      xcoords(jvar)=xcoords(jvar)-delta
      call cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
      tmpmat=tmpmat+tmpmat2


      ddrotmat(:,:,ivar,jvar)=tmpmat/4.D0/delta**2
      ddrotmat(:,:,jvar,ivar)=ddrotmat(:,:,ivar,jvar)
    end do

    ! now the diagonal element
    tmpmat=0.D0
    
    xcoords=xcoords_
    xcoords(ivar)=xcoords(ivar)+delta
    call cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
    tmpmat=tmpmat+tmpmat2
    
    xcoords=xcoords_
    !xcoords(ivar)=xcoords(ivar)
    call cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
    tmpmat=tmpmat-2.D0*tmpmat2
    
    xcoords=xcoords_
    xcoords(ivar)=xcoords(ivar)-delta
    call cgh_xtos(ncoord,nvar,rcoords,xcoords,trans,tmpmat2,dcoords)
    tmpmat=tmpmat+tmpmat2

    ddrotmat(:,:,ivar,ivar)=tmpmat/delta**2

  end do
end subroutine get_ddrotmat

subroutine coords_to_bad(nat,id,x_unchanged,x_out,xx)
  use pes_ml_module, only: pes_massweight
  use nn_module, only: bad_buildlist,bad_modes,dim_renorm,mass_list,atom_index
  use dlf_parameter_module, only: rk
  implicit none
  integer, intent(in) :: nat,id
  real(rk), intent(in) :: x_unchanged(3*nat)
  real(rk), intent(out) :: x_out(id),xx(3*nat)
  integer i,j,k,l
  real(rk), allocatable :: buildlist(:,:),bad_coords(:),mass(:)
  character(256) :: fname,line
  character(2) dummy(4)
  real(rk) pi

  pi=4.d0*atan(1.d0)
  allocate(mass(nat))
  xx=x_unchanged
  mass=1.d0
  if(pes_massweight)then
    mass=mass_list
    do i=1,nat
      xx(3*i-2:3*i)=x_unchanged(3*i-2:3*i)/sqrt(mass(i))
    enddo
  endif
  deallocate(mass)
  allocate(buildlist(nat,3))
  buildlist=0.d0
  j=0
  k=1
  l=0
  do i=1,sum(bad_modes(1:3))
    j=j+1
    l=l+1
    buildlist(j+k,k)=bad_buildlist(l)
    if(j.eq.bad_modes(k))then
      k=k+1
      j=0
    endif
  enddo
  fname='tmp_geom.zmt'
  open(unit=6500,file='tmp_geom.xyz',status='replace')
  call write_xyz(6500,nat,(nint(mass_list)+1)/2,xx)
  close(6500)

!  open(unit=6500,file='buildlist.dat',status='replace')
!  write(6500,'(A3,I3.1,A1,I1,A1)')'# (',nat,',',4,')'
!  do i=1,nat
!    write(6500,'(E24.17)')dble(i)
!    do j=1,min(3,i-1)
!      write(6500,'(E24.17)')buildlist(i,j)
!    enddo
!    do j=4,i+1,-1
!      write(6500,'(E24.17)')0.d0
!    enddo
!  enddo
!  close(6500)
!  deallocate(buildlist)
  open(unit=6500,file='construction_table.csv',status='replace')
  write(6500,'(A1,2A2,A1)')',','b,','a,','d'
  write(line,'(I3.1,A8,A4,A3)')atom_index(1),',origin,','e_z,','e_x'
  call StripSpaces(line)
  write(6500,'(A)')trim(adjustl(line))
  write(line,'(I3.1,A1,I3.1,A5,A3)')atom_index(2),',',nint(buildlist(2,1)),',e_z,','e_x'
  call StripSpaces(line)
  write(6500,'(A)')trim(adjustl(line))
  write(line,'(I3.1,A1,I3.1,A1,I3.1,A4)')atom_index(3),',',nint(buildlist(3,1)),',',nint(buildlist(3,2)),',e_x'
  call StripSpaces(line)
  write(6500,'(A)')trim(adjustl(line))
  do i=4,nat
    write(line,'(I3.1,A1,I3.1,A1,I3.1,A1,I3.1)')atom_index(i),',',nint(buildlist(i,1)),',',&
    nint(buildlist(i,2)),',',nint(buildlist(i,3))
    call StripSpaces(line)
    write(6500,'(A)')trim(adjustl(line))
  enddo
  close(6500)
  deallocate(buildlist)

!  call execute_command_line('python3.4 chemcoord/xyz_to_zmt.py tmp_geom.xyz 0 > '//trim(fname))
!  call system('python2.7 -W ignore chemcoord/xyz_to_zmt_cc_2_0_3.py tmp_geom.xyz 0 > '//trim(fname))
! THERE'S NO NEED TO USE THE PYTHON PROGRAM HERE, BUILDLIST AND ATOM_INDEX ARE READ IN, JUST CALCULATE THE COORDS

  allocate(bad_coords(dim_renorm))
  call bonds_angles_dihedrals(nat,bad_coords,xx)!,fname)
  x_out(1:bad_modes(1))=bad_coords(1:bad_modes(1))**(-1)
  x_out(bad_modes(1)+1:sum(bad_modes(1:3)))=bad_coords(bad_modes(1)+1:sum(bad_modes(1:3)))
  deallocate(bad_coords)
  do k=1,id-dim_renorm
    x_out(sum(bad_modes(1:3))+k)=sqrt(sum((&
    xx(3*bad_buildlist(sum(bad_modes(1:3))+2*k-1)-2:3*bad_buildlist(sum(bad_modes(1:3))+2*k-1))-&
    xx(3*bad_buildlist(sum(bad_modes(1:3))+2*k)-2:3*bad_buildlist(sum(bad_modes(1:3))+2*k)))**2))**(-1)
  enddo

end subroutine coords_to_bad

subroutine bonds_angles_dihedrals(nat,bad_coords,xx)
  use nn_module, only: dim_renorm,atom_index,bad_buildlist,bad_modes
  use dlf_parameter_module, only: rk
  use cross
  implicit none
  integer, intent(in) :: nat
  real(rk), intent(in) :: xx(3*nat)
  real(rk), intent(out) :: bad_coords(dim_renorm)
  integer at1,at2,at3,at4,i,j,k
  real(rk), allocatable :: w(:),u(:),v(:)
  real(rk) v_norm1,v_norm2,v_norm3,pi,cross_prod(3,2)

  pi=4.d0*atan(1.d0)
  j=1
  do i=1,bad_modes(1)
    j=j+1
    at1=atom_index(j)
    at2=bad_buildlist(j-1)
    bad_coords(i)=sqrt(sum((xx(3*at1-2:3*at1)-xx(3*at2-2:3*at2))**2))
  enddo
  j=2
  allocate(u(3))
  allocate(v(3))
  do i=bad_modes(1)+1,sum(bad_modes(1:2))
    j=j+1
    at1=atom_index(j)
    at2=bad_buildlist(j-1)
    at3=bad_buildlist(i)
    u=xx(3*at1-2:3*at1) - xx(3*at2-2:3*at2)
    v=xx(3*at3-2:3*at3) - xx(3*at2-2:3*at2)
    v_norm1=sqrt(sum(u**2))
    v_norm2=sqrt(sum(v**2))
    u=u/v_norm1
    v=v/v_norm2
    bad_coords(i)=acos(dot_product(u,v))
  enddo
  j=3
  k=bad_modes(1)+1
  allocate(w(3))
  do i=sum(bad_modes(1:2))+1,sum(bad_modes(1:3))
    j=j+1
    k=k+1
    at1=atom_index(j)
    at2=bad_buildlist(j-1)
    at3=bad_buildlist(k)
    at4=bad_buildlist(i)
    u=xx(3*at1-2:3*at1) - xx(3*at2-2:3*at2)
    w=xx(3*at2-2:3*at2) - xx(3*at3-2:3*at3)
    v=xx(3*at3-2:3*at3) - xx(3*at4-2:3*at4)
    v_norm1=sqrt(sum(u**2))
    v_norm2=sqrt(sum(w**2))
    v_norm3=sqrt(sum(v**2))
    u=u/v_norm1
    w=w/v_norm2
    v=v/v_norm3
    cross_prod(:,1)=cross_product(u,w)
    cross_prod(:,2)=cross_product(v,w)
    bad_coords(i)=dot_product(cross_prod(:,1),cross_prod(:,2))/&
    sqrt(1.d0-dot_product(u,w)**2)/sqrt(1.d0-dot_product(v,w)**2)
    bad_coords(i)=acos(bad_coords(i))
  enddo
  deallocate(u)
  deallocate(v)
  deallocate(w)

end subroutine bonds_angles_dihedrals

subroutine coords_to_interatomic(nat,interatomic_dof,x_unchanged,x_out,xx,&
inv_tf)
  use pes_ml_module, only: pes_massweight,refmass
  implicit none
  integer, intent(in) :: nat,interatomic_dof
  real(8), intent(in) :: x_unchanged(3*nat)
  logical, intent(in) :: inv_tf
  real(8), intent(out) :: x_out(interatomic_dof),xx(3*nat)
  integer, allocatable :: mapping(:,:),back_mapping(:,:)
  integer i,j,counter,inv

  if(.not.pes_massweight)then
    xx=x_unchanged
  else
    do i=1,nat
      xx(3*i-2:3*i)=x_unchanged(3*i-2:3*i)/sqrt(refmass(i))
    enddo
  endif

  if(inv_tf)inv=-1
  if(.not.inv_tf)inv=1

  allocate(mapping(3*nat,3*nat))
  allocate(back_mapping(2,interatomic_dof))
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

  deallocate(mapping)
  deallocate(back_mapping)

end subroutine coords_to_interatomic

subroutine redundant_DMAT(nat,interatomic_dof,DMAT,x_out,x_unchanged,&
coords_inverse)
  use nn_module, only: radii_omit
  use nn_av_module, only: radii_omitav,avnncalc,ifile
  implicit none
  integer, intent(in) :: nat,interatomic_dof
  real(8), intent(out):: DMAT(interatomic_dof,3*nat)
  real(8), intent(in) :: x_unchanged(3*nat),x_out(interatomic_dof)
  logical, intent(in) :: coords_inverse
  real(8), allocatable :: vector_diff(:,:)
  integer i,j,m,l,kk,counter
  integer, allocatable :: mapping(:,:)

  allocate(mapping(nat,nat))
  mapping=-1
  counter=0
  kk=0
  do i=1,nat
    do j=i+1,nat
      counter=counter+1
      if(.not.avnncalc)then
        if(.not.any(radii_omit==counter))then
          kk=kk+1
          mapping(i,j)=kk
          mapping(j,i)=kk
        endif
      else
        if(.not.any(radii_omitav(:,ifile)==counter))then
          kk=kk+1
          mapping(i,j)=kk
          mapping(j,i)=kk
        endif
      endif
    enddo
  enddo

  allocate(vector_diff(3*nat,nat))
  DMAT=0.d0
  vector_diff=0.d0
  do m=1,nat
    do l=1,nat
      vector_diff(3*l-2:3*l,m)=x_unchanged(3*m-2:3*m)-&
      x_unchanged(3*l-2:3*l)
    enddo
  enddo

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
  deallocate(mapping)
  deallocate(vector_diff)

end subroutine redundant_DMAT

subroutine bad_DMAT(nat,id,DMAT,x_out,x_unchanged)
  use nn_module, only: bad_buildlist,bad_modes,dim_renorm,atom_index
  use cross
  implicit none
  integer, intent(in) :: nat,id
  real(8), intent(out):: DMAT(id,3*nat)
  real(8), allocatable :: vector_diff(:,:),cross_prod(:,:),w(:),u(:),v(:)
  integer j,m,l,k1,counter,at1,at2,at3,at4,la,lb,ld
  real(8), intent(in) :: x_unchanged(3*nat),x_out(id)
  integer, allocatable :: nucleus_loop(:)
  real(8) v_norm1,v_norm2,v_norm3,zeta_amn

  DMAT=0.d0
  allocate(vector_diff(3*nat,nat))
  allocate(cross_prod(3,2))
  allocate(w(3))
  allocate(u(3))
  allocate(v(3))
  allocate(nucleus_loop(4))
  vector_diff=0.d0
  do m=1,nat
    do l=1,nat
      vector_diff(3*l-2:3*l,m)=x_unchanged(3*m-2:3*m)-&
      x_unchanged(3*l-2:3*l)
    enddo
  enddo
!     BONDS
  do m=1,bad_modes(1)
    l=bad_buildlist(m)
!    DMAT(m,3*(m+1)-2:3*(m+1))=vector_diff(3*l-2:3*l,m+1)/&
!    (x_out(m))
    DMAT(m,3*atom_index(m+1)-2:3*atom_index(m+1))=-vector_diff(3*l-2:3*l,atom_index(m+1))*&
    x_out(m)**3
    DMAT(m,3*l-2:3*l)=-DMAT(m,3*atom_index(m+1)-2:3*atom_index(m+1))
  enddo
!ADDITION OF REDUNDANT MODE
  do j=1,id-dim_renorm
    m=sum(bad_modes(1:3))+j
    l=bad_buildlist(m)
    k1=bad_buildlist(m+1)
!    DMAT(m,3*l-2:3*l)=vector_diff(3*l-2:3*l,k1)/&
!    (x_out(m))
    DMAT(m,3*l-2:3*l)=-vector_diff(3*l-2:3*l,k1)*x_out(m)**3
    DMAT(m,3*k1-2:3*k1)=-DMAT(m,3*l-2:3*l)
  enddo

!     ANGLES
  l=2
  lb=1
  la=0
  do m=bad_modes(1)+1,sum(bad_modes(1:2))
    l=l+1
    lb=lb+1
    la=la+1
!        at1=nat-bad_modes(2)+l!NEW NUCLEUS
    at1=atom_index(l)
!        at2=bad_buildlist(at1-1)!BONDED TO
    at2=bad_buildlist(lb)!BONDED TO
!        at3=bad_buildlist(m)!MAKES ANGLE WITH
    at3=bad_buildlist(bad_modes(1)+la)!MAKES ANGLE WITH
    nucleus_loop=(/at1,at2,at3,-1/)
    u=x_unchanged(3*at1-2:3*at1) - x_unchanged(3*at2-2:3*at2)
    v=x_unchanged(3*at3-2:3*at3) - x_unchanged(3*at2-2:3*at2)
    w=bad_w(u,v)
    v_norm1=sqrt(sum(u**2))
    v_norm2=sqrt(sum(v**2))
    v_norm3=sqrt(sum(w**2))
    u=u/v_norm1
    v=v/v_norm2
    w=w/v_norm3
    cross_prod(:,1)=cross_product(u,w)
    cross_prod(:,2)=cross_product(w,v)
    do counter=1,3
      k1=nucleus_loop(counter)
      DMAT(m,3*k1-2:3*k1)=zeta_amn(k1,at1,at2)*cross_prod(:,1)/v_norm1+&
      zeta_amn(k1,at3,at2)*cross_prod(:,2)/v_norm2
    enddo
  enddo

!     DIHEDRALS
  l=3
  lb=2
  la=1
  ld=0
  do m=sum(bad_modes(1:2))+1,sum(bad_modes(1:3))
    l=l+1
    lb=lb+1
    la=la+1
    ld=ld+1
!        at1=nat-bad_modes(3)+l!NEW NUCLEUS
    at1=atom_index(l)!NEW NUCLEUS
!        at2=bad_buildlist(at1-1)!BONDED TO
    at2=bad_buildlist(lb)!BONDED TO
!        at3=bad_buildlist(sum(bad_modes(1:2))-(nat-at1))!MAKES ANGLE WITH
    at3=bad_buildlist(bad_modes(1)+la)!MAKES ANGLE WITH
!        at4=bad_buildlist(m)!MAKES DIHEDRAL WITH
    at4=bad_buildlist(sum(bad_modes(1:2))+ld)!MAKES DIHEDRAL WITH
    nucleus_loop=(/at1,at2,at3,at4/)
    u=x_unchanged(3*at1-2:3*at1) - x_unchanged(3*at2-2:3*at2)
    w=x_unchanged(3*at2-2:3*at2) - x_unchanged(3*at3-2:3*at3)
    v=x_unchanged(3*at3-2:3*at3) - x_unchanged(3*at4-2:3*at4)
    v_norm1=sqrt(sum(u**2))
    v_norm2=sqrt(sum(w**2))
    v_norm3=sqrt(sum(v**2))
    u=u/v_norm1
    w=w/v_norm2
    v=v/v_norm3
    cross_prod(:,1)=cross_product(u,w)
    cross_prod(:,2)=cross_product(v,w)
    if(dot_product(u,w)**2.lt.0.99d0 .and. dot_product(w,v)**2.lt.0.99d0)then
      do counter=1,4
        k1=nucleus_loop(counter)
        DMAT(m,3*k1-2:3*k1)=&
        zeta_amn(k1,at1,at2)*cross_prod(:,1)/v_norm1/(1.d0-dot_product(u,w)**2)+&
        zeta_amn(k1,at3,at4)*cross_prod(:,2)/v_norm3/(1.d0-dot_product(v,w)**2)+&
        zeta_amn(k1,at2,at3)*(&
        cross_prod(:,1)*dot_product(u,w)/v_norm2/(1.d0-dot_product(u,w)**2)+&
        cross_prod(:,2)*dot_product(v,w)/v_norm2/(1.d0-dot_product(v,w)**2))
      enddo
    endif
  enddo

  deallocate(cross_prod)
  deallocate(w)
  deallocate(u)
  deallocate(v)
  deallocate(nucleus_loop)
  deallocate(vector_diff)

end subroutine bad_DMAT

subroutine DDMAT(nat,ncoord,x_out,DMAT,coords_inverse,DM2)
  implicit none
  integer, intent(in) :: nat,ncoord
  real(8), intent(in) :: x_out(ncoord),DMAT(ncoord,3*nat)
  real(8), intent(out):: DM2(3*nat,3*nat,ncoord)
  logical, intent(in) :: coords_inverse
  real(8) rp,kronecker
  integer i,l,ata,kk,atb,j

  do i=1,ncoord
    do l=1,3*nat
      ata=(l+2)/3
      do kk=1,3*nat
        atb=(kk+2)/3
        rp=0.d0
        do j=1,3
          rp=rp+(kronecker(3*ata-3+j,l)-kronecker(3*atb-3+j,l))*&
          (kronecker(3*ata-3+j,kk)-kronecker(3*atb-3+j,kk))
        enddo
        if(abs(x_out(i)).gt.1.d-9 .and. abs(x_out(i)) .lt. 1.d3)then
          if(coords_inverse)then
            DM2(l,kk,i)=(3.d0/x_out(i))*DMAT(i,l)*DMAT(i,kk)-x_out(i)**3*rp
          else
            DM2(l,kk,i)=(1.d0/x_out(i))*(-DMAT(i,l)*DMAT(i,kk)+rp)
          endif
        endif
      enddo
    enddo
  enddo

end subroutine DDMAT

subroutine bad_DDMAT(nat,ncoord,x_unchanged,x_out,DMAT,DM2)
  use nn_module, only: bad_buildlist,bad_modes,dim_renorm,atom_index
  use cross
  implicit none
  integer, intent(in) :: nat,ncoord
  real(8), intent(in) :: x_unchanged(3*nat),x_out(ncoord),DMAT(ncoord,3*nat)
  real(8), intent(out):: DM2(3*nat,3*nat,ncoord)
  real(8) rp,kronecker
  integer i,l,m,ata,kk,atb,j,counter,at1,at2,at3,at4,nucleus_loop(4),k1,il,ik,counter2,ki,&
  lb,ld,la
  real(8) v_norm1,v_norm2,v_norm3,u(3),v(3),w(3),cross_prod(3,2),&
  cosu,cosv,sinu,sinv,zeta_amn

  !BONDS
  do i=1,bad_modes(1)
    do l=1,3*nat
      ata=(l+2)/3
      do kk=1,3*nat
        atb=(kk+2)/3
        rp=0.d0
        do j=1,3
          rp=rp+(kronecker(3*ata-3+j,l)-kronecker(3*atb-3+j,l))*&
          (kronecker(3*ata-3+j,kk)-kronecker(3*atb-3+j,kk))
        enddo
        if(abs(x_out(i)).gt.1.d-9 .and. abs(x_out(i)) .lt. 1.d3)then
!          DM2(l,kk,i)=(1.d0/(x_out(i)))*(-DMAT(i,l)*DMAT(i,kk)+rp)
          DM2(l,kk,i)=(3.d0/x_out(i))*(DMAT(i,l)*DMAT(i,kk)-x_out(i)**3*rp)
        endif
      enddo
    enddo
  enddo
!ADDITION OF REDUNDANT MODE
  do m=1,ncoord-dim_renorm
    i=sum(bad_modes(1:3))+m
    do l=1,3*nat
      ata=(l+2)/3
      do kk=1,3*nat
        atb=(kk+2)/3
        rp=0.d0
        do j=1,3
          rp=rp+(kronecker(3*ata-3+j,l)-kronecker(3*atb-3+j,l))*&
          (kronecker(3*ata-3+j,kk)-kronecker(3*atb-3+j,kk))
        enddo
        if(abs(x_out(i)).gt.1.d-9 .and. abs(x_out(i)) .lt. 1.d3)then
          DM2(l,kk,i)=(3.d0/x_out(i))*(DMAT(i,l)*DMAT(i,kk)-x_out(i)**3*rp)
        endif
      enddo
    enddo
  enddo

  !ANGLES
  l=2
  lb=1
  la=0
  do m=bad_modes(1)+1,sum(bad_modes(1:2))
    l=l+1
    lb=lb+1
    la=la+1
!        at1=nat-bad_modes(2)+l!NEW NUCLEUS
    at1=atom_index(l)
!        at2=bad_buildlist(at1-1)!BONDED TO
    at2=bad_buildlist(lb)!BONDED TO
!        at3=bad_buildlist(m)!MAKES ANGLE WITH
    at3=bad_buildlist(bad_modes(1)+la)!MAKES ANGLE WITH
    nucleus_loop=(/at1,at2,at3,-1/)
    u=x_unchanged(3*at1-2:3*at1) - x_unchanged(3*at2-2:3*at2)
    v=x_unchanged(3*at3-2:3*at3) - x_unchanged(3*at2-2:3*at2)
    v_norm1=sqrt(sum(u**2))
    v_norm2=sqrt(sum(v**2))
    u=u/v_norm1
    v=v/v_norm2
    if(dot_product(u,v)**2.lt.0.99d0)then
      do counter=1,3
        k1=nucleus_loop(counter)
        do il=1,3
          do counter2=1,3
            kk=nucleus_loop(counter2)
            do ik=1,3
                DM2(3*k1-3+il,3*kk-3+ik,m)=&
                zeta_amn(k1,at1,at2)*zeta_amn(kk,at1,at2)*&
                (u(il)*v(ik)+u(ik)*v(il)-3.d0*u(il)*u(ik)*dot_product(u,v)+kronecker(il,ik)*dot_product(u,v))/&
                v_norm1**2/sqrt(1.d0-dot_product(u,v)**2)+&
                zeta_amn(k1,at3,at2)*zeta_amn(kk,at3,at2)*&
                (v(il)*u(ik)+v(ik)*u(il)-3.d0*v(il)*v(ik)*dot_product(u,v)+kronecker(il,ik)*dot_product(u,v))/&
                v_norm2**2/sqrt(1.d0-dot_product(u,v)**2)+&
                zeta_amn(k1,at1,at2)*zeta_amn(kk,at3,at2)*&
                (u(il)*u(ik)+v(ik)*v(il)-u(il)*v(ik)*dot_product(u,v)-kronecker(il,ik))/&
                v_norm1/v_norm2/sqrt(1.d0-dot_product(u,v)**2)+&
                zeta_amn(k1,at3,at2)*zeta_amn(kk,at1,at2)*&
                (v(il)*v(ik)+u(ik)*u(il)-v(il)*u(ik)*dot_product(u,v)-kronecker(il,ik))/&
                v_norm1/v_norm2/sqrt(1.d0-dot_product(u,v)**2)-&
                DMAT(m,3*k1-3+il)*DMAT(m,3*kk-3+ik)*dot_product(u,v)/sqrt(1.d0-dot_product(u,v)**2)
            enddo
          enddo
        enddo
      enddo
    endif
  enddo
  !DIHEDRALS
  l=3
  lb=2
  la=1
  ld=0
  do m=sum(bad_modes(1:2))+1,sum(bad_modes(1:3))
    l=l+1
    lb=lb+1
    la=la+1
    ld=ld+1
!        at1=nat-bad_modes(3)+l!NEW NUCLEUS
    at1=atom_index(l)!NEW NUCLEUS
!        at2=bad_buildlist(at1-1)!BONDED TO
    at2=bad_buildlist(lb)!BONDED TO
!        at3=bad_buildlist(sum(bad_modes(1:2))-(nat-at1))!MAKES ANGLE WITH
    at3=bad_buildlist(bad_modes(1)+la)!MAKES ANGLE WITH
!        at4=bad_buildlist(m)!MAKES DIHEDRAL WITH
    at4=bad_buildlist(sum(bad_modes(1:2))+ld)!MAKES DIHEDRAL WITH
    nucleus_loop=(/at1,at2,at3,at4/)
    u=x_unchanged(3*at1-2:3*at1) - x_unchanged(3*at2-2:3*at2)
    w=x_unchanged(3*at2-2:3*at2) - x_unchanged(3*at3-2:3*at3)
    v=x_unchanged(3*at3-2:3*at3) - x_unchanged(3*at4-2:3*at4)
    v_norm1=sqrt(sum(u**2))
    v_norm2=sqrt(sum(w**2))
    v_norm3=sqrt(sum(v**2))
    u=u/v_norm1
    w=w/v_norm2
    v=v/v_norm3
    cosu=dot_product(u,w)
    cosv=-dot_product(v,w)
    sinu=sqrt(1.d0-cosu**2)
    sinv=sqrt(1.d0-cosv**2)
    cross_prod(:,1)=cross_product(u,w)
    cross_prod(:,2)=cross_product(v,w)
    if(cosu**2.lt.0.99d0 .and. cosv**2.lt.0.99d0)then
      do counter=1,4
        k1=nucleus_loop(counter)
        do il=1,3
          do counter2=1,4
            kk=nucleus_loop(counter)
            if((kk.eq.at1 .and. k1.ne.at4) .or. (kk.eq.at4 .and. k1.ne.at1))then
              do ik=1,3
                DM2(3*k1-3+il,3*kk-3+ik,m)=&
                zeta_amn(k1,at1,at2)*zeta_amn(kk,at1,at2)*&
                (cross_prod(il,1)*w(ik)*cosu-u(ik)+cross_prod(ik,1)*w(il)*cosu-u(il))/&
                (v_norm1**2*sinu**4)+&
                zeta_amn(k1,at4,at3)*zeta_amn(kk,at4,at3)*&
                (cross_prod(il,2)*w(ik)*cosv-v(ik)+cross_prod(ik,2)*w(il)*cosv-v(il))/&
                (v_norm2**2*sinv**4)+&
                (zeta_amn(k1,at1,at2)*zeta_amn(kk,at2,at3)+&
                zeta_amn(k1,at3,at2)*zeta_amn(kk,at2,at1))*&
                (cross_prod(il,1)*(w(ik)-2.d0*u(ik)*cosu+w(ik)*cosu**2)+&
                cross_prod(ik,1)*(w(il)-2.d0*u(il)*cosu+w(il)*cosu**2))/&
                (2.d0*v_norm1*v_norm2*sinu**4)+&
                (zeta_amn(k1,at4,at3)*zeta_amn(kk,at3,at2)+&
                zeta_amn(k1,at3,at2)*zeta_amn(kk,at4,at3))*&
                (cross_prod(il,2)*(w(ik)+2.d0*u(ik)*cosv+w(ik)*cosv**2)+&
                cross_prod(ik,2)*(w(il)+2.d0*u(il)*cosv+w(il)*cosv**2))/&
                (2.d0*v_norm1*v_norm2*sinv**4)+&
                zeta_amn(k1,at2,at3)*zeta_amn(kk,at3,at2)*&
                ((cross_prod(il,1)*(u(ik)+u(ik)*cosu**2-3.d0*w(ik)*cosu+w(ik)*cosu**3))+&
                (cross_prod(ik,1)*(u(il)+u(il)*cosu**2-3.d0*w(il)*cosu+w(il)*cosu**3)))/&
                (2.d0*v_norm2**2*sinu**4)+&
                zeta_amn(k1,at2,at3)*zeta_amn(kk,at2,at3)*&
                ((cross_prod(il,2)*(v(ik)+v(ik)*cosv**2+3.d0*w(ik)*cosv-w(ik)*cosv**3))+&
                (cross_prod(ik,2)*(v(il)+v(il)*cosv**2+3.d0*w(il)*cosv-w(il)*cosv**3)))/&
                (2.d0*v_norm2**2*sinv**4)
                if(ik.ne.il)then
                  ki=6-ik-il
                  DM2(3*k1-3+il,3*kk-3+ik,m)=DM2(3*k1-3+il,3*kk-3+ik,m)+&
                  (1-kronecker(k1,kk))*(zeta_amn(k1,at1,at2)*zeta_amn(kk,at2,at3)+&
                  zeta_amn(k1,at3,at2)*zeta_amn(kk,at2,at1))*&
                  (ik-il)*(0.5d0**abs(ik-il))*(w(ki)*cosu-u(ki))/(v_norm1*v_norm2*sinu)+&
                  (1-kronecker(k1,kk))*(zeta_amn(k1,at4,at2)*zeta_amn(kk,at2,at3)+&
                  zeta_amn(k1,at3,at2)*zeta_amn(kk,at2,at1))*&
                  (ik-il)*(0.5d0**abs(ik-il))*(w(ki)*cosv-v(ki))/(v_norm3*v_norm2*sinv)
                endif
              enddo
            endif
          enddo
        enddo
      enddo
    endif
  enddo

end subroutine bad_DDMAT

subroutine non_redundant_evect(evect,n,NR_size,short_evect)
  implicit none
  integer, intent(in) :: n,NR_size
  real(8), intent(in) :: evect(n,n)
  real(8), intent(out):: short_evect(n,NR_size)

  short_evect=evect(:,1:NR_size)

end subroutine non_redundant_evect

subroutine DKMAT_interatomic(nat,id,x_unchanged,xgradient_store,&
x_out,DMAT,DMAT_PINV,DMAT_PINV2,projection,KMAT,coords_inverse,coords_b_a_d)
  use nn_module, only: pinv_tol,pinv_tol_back
  implicit none
  integer, intent(in) :: nat,id
  real(8), intent(inout)::xgradient_store(id)
  real(8), intent(in) :: x_unchanged(3*nat)
  real(8), intent(in) :: x_out(id)
  logical, intent(in) :: coords_inverse,coords_b_a_d
  real(8), intent(out):: DMAT(id,3*nat),KMAT(3*nat,3*nat),&
  projection(id,id),DMAT_PINV2(id,3*nat),DMAT_PINV(3*nat,id)
  real(8), allocatable :: R_DM2(:,:,:),gg(:),xgtmp(:),proj_inv(:,:)
  integer i,l,kk
  real(8) tol_save_res,tol_save_x,tol_save_resx,tol_save_graphical,&
  DM_minval

  allocate(gg(3*nat))
  gg=0.d0
  if(.not.coords_b_a_d)then
    call redundant_DMAT(nat,id,DMAT,x_out,x_unchanged,&
    coords_inverse)
  else
    call bad_DMAT(nat,id,DMAT,x_out,x_unchanged)
  endif
  DM_minval=minval(abs(DMAT))

  call tolerance_simple(DMAT,gg,xgradient_store,id,3*nat,tol_save_res,tol_save_x,&
  tol_save_resx,tol_save_graphical,'M',pinv_tol_back,0)

!  call Pseudoinverse(id,3*nat,DMAT,DMAT_pinv,2.33d-6,'M')
!  call Pseudoinverse(id,3*nat,DMAT,DMAT_pinv,pinv_tol_back,'M')
!  call Pseudoinverse(id,3*nat,DMAT,DMAT_pinv,max(tol_save_res,DM_minval),'M')
  call Pseudoinverse(id,3*nat,DMAT,DMAT_pinv,tol_save_res,'M')
!  projection=matmul(DMAT,DMAT_pinv)
  projection=0.d0
  do l=1,id
    projection(l,l)=1.d0
  enddo
  allocate(proj_inv(id,id))
  proj_inv=transpose(projection)
  allocate(xgtmp(id))
  xgtmp=matmul(proj_inv,xgradient_store)
  deallocate(proj_inv)
  gg=matmul(DMAT_pinv,xgtmp)
  call tolerance_simple(transpose(DMAT),xgtmp,gg,3*nat,id,tol_save_res,tol_save_x,&
  tol_save_resx,tol_save_graphical,'M',pinv_tol,0)
  deallocate(xgtmp)
  deallocate(gg)
!  call Pseudoinverse(3*nat,id,transpose(DMAT),DMAT_pinv2,2.48d-2,'M')
!  call Pseudoinverse(3*nat,id,transpose(DMAT),DMAT_pinv2,pinv_tol,'M')
!  call Pseudoinverse(3*nat,id,transpose(DMAT),DMAT_pinv2,max(tol_save_res,DM_minval),'M')
  call Pseudoinverse(3*nat,id,transpose(DMAT),DMAT_pinv2,tol_save_res,'M')

  allocate(R_DM2(3*nat,3*nat,id))
  if(.not.coords_b_a_d)then
    call DDMAT(nat,id,x_out,DMAT,coords_inverse,R_DM2)
  else
    call bad_DDMAT(nat,id,x_unchanged,x_out,DMAT,R_DM2)
  endif
  KMAT=0.d0
  xgradient_store=matmul(transpose(projection),xgradient_store)
  do l=1,3*nat
    do kk=1,3*nat
      do i=1,id
        KMAT(l,kk)=KMAT(l,kk)+xgradient_store(i)*R_DM2(l,kk,i)
      enddo
    enddo
  enddo
  deallocate(R_DM2)

end subroutine DKMAT_interatomic

subroutine KMAT_reweight(KMAT,nat,evals_in,nzero)
  implicit none
  real(8), intent(inout) :: KMAT(3*nat,3*nat)
  real(8), intent(in) :: evals_in(3*nat)
  integer, intent(in)  :: nat
  real(8) KMAT_COMP(3*nat,3*nat,2),evals(3*nat,2),evals_sort(3*nat)
  integer l,sortlist(3*nat),nzero

  evals_sort=evals_in
  call sort_smallest_first(3*nat,evals_sort,sortlist)
  KMAT_COMP=0.d0
  call dlf_matrix_diagonalise_general(3*nat,KMAT,evals(:,1),evals(:,2),KMAT_COMP(:,:,1))
!   KMAT's negative eigenvalues are very nearly degenerate.
!    if(evals(1,2).lt.rp(5)/100.d0 .and. evals(1,2) .gt. 2.d0*rp(5))then
!      KMAT_COMP(1,1,3)=evals(1,1)!max(rp(5),evals(1,1))
!      KMAT_COMP(:,2:2+nzero,2)=0.d0
!      do l=2+nzero,3*nat
!!        if(abs(evals(l,1)).lt. 0.1d0*abs(evals(l,2)))then
!          KMAT_COMP(l,l,3)=evals(l,1)
!!        else
!!          KMAT_COMP(:,l,2)=0.d0
!!        endif
!      enddo
!    else
!      KMAT_COMP(:,1:1+nzero,2)=0.d0
    do l=1,3*nat
!      do l=1+nzero,3*nat
      if(evals(l,1).gt.0.d0)then
        KMAT_COMP(l,l,2)=evals(l,1)
      else
        KMAT_COMP(:,l,1)=0.d0
      endif
    enddo
!    endif
  KMAT=matmul(matmul(KMAT_COMP(:,:,1),KMAT_COMP(:,:,2)),transpose(KMAT_COMP(:,:,1)))

end subroutine KMAT_reweight

subroutine DKMAT_interatomic_inv(nat,interatomic_dof,x_unchanged,xgradient_store,&
x_out,DMAT,DMAT_PINV,KMAT,pinv_tol)
  implicit none
  integer, intent(in) :: nat,interatomic_dof
  real(8), intent(in) :: x_unchanged(3*nat),xgradient_store(interatomic_dof),pinv_tol
  real(8), intent(in) :: x_out(interatomic_dof)
  real(8), intent(out):: DMAT(3*nat,interatomic_dof),DMAT_PINV(interatomic_dof,3*nat),&
  KMAT(3*nat,3*nat)
  real(8), allocatable :: vector_diff(:,:),DM2(:,:,:)
  integer, allocatable :: mapping(:,:),back_mapping(:,:)
  integer i,j,l,m,counter,ata,atb,kk
  real(8) rp(3),kronecker!,tol_estimate

  allocate(mapping(3*nat,3*nat))
  allocate(back_mapping(2,interatomic_dof))
  mapping=-1
  
  counter=0
  do i=1,nat
    do j=i+1,nat
      counter=counter+1
      back_mapping(1,counter)=i
      back_mapping(2,counter)=j
      mapping(i,j)=counter
      mapping(j,i)=counter
    enddo
  enddo

  allocate(vector_diff(3*nat,nat))
  DMAT=0.d0
  DMAT_PINV=0.d0

  vector_diff=0.d0
  do m=1,nat
    do l=1,nat
      vector_diff(3*l-2:3*l,m)=x_unchanged(3*m-2:3*m)-&
      x_unchanged(3*l-2:3*l)
    enddo
  enddo

  do m=1,nat
    do l=m+1,nat
      DMAT(3*m-2:3*m,mapping(l,m))=-vector_diff(3*l-2:3*l,m)*&
      x_out(mapping(l,m))**3
      DMAT(3*l-2:3*l,mapping(l,m))=-DMAT(3*m-2:3*m,mapping(l,m))
    enddo
  enddo
!  do m=1,3*nat
!    rp=sqrt(sum(DMAT(:,m)**2))
!    DMAT(:,m)=DMAT(:,m)/rp
!  enddo

!  call tolerance_estimate(DMAT,x_unchanged,3*nat,interatomic_dof,tol_estimate,x_out)
!  if(tol_estimate.eq.0.d0)then
!    tol_estimate=pinv_tol
!    CALL Pseudoinverse(3*nat,interatomic_dof,DMAT,DMAT_PINV,tol_estimate,'M')
    CALL Pseudoinverse(3*nat,interatomic_dof,DMAT,DMAT_PINV,pinv_tol,'M')
!  else
!    CALL Pseudoinverse(3*nat,interatomic_dof,DMAT,DMAT_PINV,tol_estimate,'T')
!  endif

  allocate(DM2(interatomic_dof,3*nat,3*nat))
  do i=1,interatomic_dof
    do l=1,3*nat
      ata=(l+2)/3
      do kk=1,3*nat
        atb=(kk+2)/3
        rp=0.d0
        do j=1,3
          rp(1)=rp(1)+(x_unchanged(3*ata-3+j)-x_unchanged(3*atb-3+j))*&
          (kronecker(3*ata-3+j,l)-kronecker(3*atb-3+j,l))
          rp(2)=rp(2)+(x_unchanged(3*ata-3+j)-x_unchanged(3*atb-3+j))*&
          (kronecker(3*ata-3+j,kk)-kronecker(3*atb-3+j,kk))
          rp(3)=rp(3)+(kronecker(3*ata-3+j,l)-kronecker(3*atb-3+j,l))*&
          (kronecker(3*ata-3+j,kk)-kronecker(3*atb-3+j,kk))
        enddo
        DM2(i,l,kk)=(3.d0*x_out(i)**5)*(rp(1)*rp(2))-(x_out(i)**3)*rp(3)
      enddo
    enddo
  enddo
  KMAT=0.d0
  do l=1,3*nat
    do kk=1,3*nat
      do i=1,interatomic_dof
        KMAT(l,kk)=KMAT(l,kk)+xgradient_store(i)*DM2(i,l,kk)
      enddo
    enddo
  enddo
  deallocate(DM2)
  deallocate(back_mapping)
  deallocate(mapping)
  deallocate(vector_diff)

end subroutine DKMAT_interatomic_inv

subroutine tolerance_simple(A,x_0,b,m,n,tol_save_res,tol_save_x,tol_save_resx,tol_save_graphical,opt,pinv,TID)
  implicit none
  integer, intent(in) :: m,n,TID
  real(8), intent(in) :: A(m,n),b(m),pinv
  real(8), intent(inout):: x_0(n,1)
  real(8), intent(out):: tol_save_res,tol_save_x,tol_save_resx,tol_save_graphical
  character(1), intent(in) :: opt
  integer K,i,j,l
  real(8), allocatable :: a_in(:,:),u(:,:),s(:),vt(:,:),s_store(:,:),&
  L_curve_dot_product(:),tols(:),L_curve(:,:)
  real(8) mod_xlambda(2),mod_residue(2),tol_init,x_lambda(n,1),bb(m,1),&
  range_factor,range_factor_init,residue(m,1),tol_guess,mod_resx(2),&
  l_curve_dist,oldxy(2),newxy(2),oldvec(2,2),newvec(2,3),svar,mod_dotprod!,stddev
  logical file_exists,check_vec_zero
  character(1) CTID
  character(2) KSVD

  check_vec_zero=.false.
  if(sum(x_0).eq.0.d0)then
    check_vec_zero=.true.
  endif

  write(CTID,'(I1.1)')TID

  inquire(file='L_curve_'//CTID//'.dat',exist=file_exists)
  if(file_exists)then
    open(file='L_curve_'//CTID//'.dat',unit=91+TID)
  else
    open(file='L_curve_'//CTID//'.dat',unit=91+TID,status='new')
  endif
  inquire(file='L_curve_D_'//CTID//'.dat',exist=file_exists)
  if(file_exists)then
    open(file='L_curve_D_'//CTID//'.dat',unit=192+TID)
  else
    open(file='L_curve_D_'//CTID//'.dat',unit=192+TID,status='new')
  endif

  range_factor_init=1.4d0
  allocate(a_in(m,n))
  a_in=a
  K = MIN(M,N)
  bb(:,1)=b

  allocate(u(m,K))
  allocate(s(K))
  write(KSVD,'(I2.2)')K
  allocate(vt(K,n))
  allocate(s_store(K,K))
  s_store=0.d0
  call svd(m,n,K,a_in,s,u,vt)
!  print*,'SINGULAR VALUES'
!  write(*,'('//KSVD//'E11.4)')s
  tol_init=1.d-1*pinv!max(1.d-2*minval(abs(s)),1.d-8*maxval(abs(s)))
  tol_guess=tol_init
  tol_save_res=tol_guess
  tol_save_resx=tol_guess
  tol_save_x=tol_guess
  tol_save_graphical=tol_guess
  mod_dotprod=1.d0
  mod_residue(1)=huge(1.d0)
  mod_xlambda(1)=huge(1.d0)
  mod_resx(1)=huge(1.d0)
  if(opt.eq.'T')then
    do i=1,K
      s_store(i,i)=s(i)/(s(i)**2+tol_guess**2)
    enddo
  !  do i=1,k
  !    if(dabs(s(i)).gt.tol_guess)then
  !      s_store(i,i)=1.d0/s(i)
  !    endif
  !  enddo
    j=0
    l=0
  !  do j=2,K-1
!    do while(tol_guess.lt.100.d0*maxval(abs(s)))
    do while(tol_guess.lt.100.d0*pinv)
      j=j+1
      range_factor=range_factor_init**sqrt(dble(j))
      x_lambda=matmul(matmul(matmul(transpose(vt),s_store),transpose(u)),bb)
      residue=matmul(a,x_lambda)-bb
      if(check_vec_zero)x_0=sum(x_lambda)/dble(n)
      mod_xlambda(2)=log(sum((x_lambda-x_0)**2))
      mod_residue(2)=log(sum(residue(:,1)**2))
      mod_resx(2)=mod_xlambda(2)+mod_residue(2)
      newxy=(/mod_xlambda(2),mod_residue(2)/)

      write(91+TID,'(E15.8,X,E15.8,X,E15.8)')tol_guess,mod_residue(2),mod_xlambda(2)
!      if(l.ge.2)then
!        write(92,'(E15.8,X,E15.8)')mod_xlambda(2),dot_product(oldvec(:,1),newvec(:,1))!,newvec(:,2),newvec(:,3),oldvec(:,2)
!      endif
!      if(l.ge.1)then
!        oldvec=newvec(:,1:2)
!      endif
      if(j.eq.1)then
        oldxy=newxy
      endif
      if(j.gt.1)then
        l_curve_dist=sqrt(sum((newxy-oldxy)**2))
        if(l_curve_dist.gt.1.d-2)then
          l=l+1
          newvec(:,1)=newxy-oldxy
          newvec(:,2)=newxy
          newvec(:,3)=oldxy
          if(l.ge.2)then
            write(192+TID,'(E15.8,X,E15.8)')tol_guess,dot_product(oldvec(:,1),newvec(:,1))/&
            sqrt(sum(oldvec(:,1)**2)*sum(newvec(:,1)**2))
          endif
          oldvec=newvec(:,1:2)
          oldxy=newxy
        endif
      endif
!      print*,tol_guess,mod_residue
      if(mod_xlambda(2).lt.mod_xlambda(1))then
        mod_xlambda(1)=mod_xlambda(2)
        tol_save_x=tol_guess
      endif
      if(mod_residue(2).lt.mod_residue(1))then
        mod_residue(1)=mod_residue(2)
        tol_save_res=tol_guess
      endif
      if(mod_resx(2).lt.mod_resx(1))then
        mod_resx(1)=mod_resx(2)
        tol_save_resx=tol_guess
      endif
      tol_guess=tol_guess+range_factor*tol_init!/dble(steps)
  !    tol_guess=dabs(s(j))
      s_store=0.d0
      do i=1,K
        s_store(i,i)=s(i)/(s(i)**2+tol_guess**2)
      enddo
  !    do i=1,k
  !      if(dabs(s(i)).gt.tol_guess)then
  !        s_store(i,i)=1.d0/s(i)
  !      endif
  !    enddo
    enddo
    close(91+TID)
    close(192+TID)
    open(file='L_curve_D_'//CTID//'.dat',unit=192+TID, status='old')
    open(file='L_curve_'//CTID//'.dat',unit=91+TID, status='old')
    allocate(L_curve(j-1,4))
    do i=1,j-1
      read(91+TID,'(E15.8,X,E15.8,X,E15.8)')L_curve(i,1),L_curve(i,3),L_curve(i,2)
    enddo
!!   FIND AVERAGE AND OF 2*LOG|X|
!    svar=sum(L_curve(:,2))/dble(j-1)
!    stddev=sqrt(sum((L_curve(:,2)-svar)**2)/dble(j-1))
!!   SHIFT 2*LOG|X| TO POSITIVE VALUES
!    L_curve(:,2)=L_curve(:,2)-svar+2.d0*stddev
!!   FIND AVERAGE AND OF 2*LOG|A*X-B|
!    svar=sum(L_curve(:,3))/dble(j-1)
!    stddev=sqrt(sum((L_curve(:,3)-svar)**2)/dble(j-1))
!!   SHIFT 2*LOG|A*X-B| TO POSITIVE VALUES
!    L_curve(:,3)=L_curve(:,3)-svar+2.d0*stddev

    svar=minval(L_curve(:,2))
    L_curve(:,2)=L_curve(:,2)-svar
    svar=minval(L_curve(:,3))
    L_curve(:,3)=L_curve(:,3)-svar

!    open(file='L_curve.dat',unit=91, status='replace')
!    do i=1,j-1
!      write(91,'(E15.8,X,E15.8,X,E15.8)')L_curve(i,1),L_curve(i,3),L_curve(i,2)
!    enddo
!    close(91)
!   FIND DISTANCE OF L_CURVE TO THE ORIGIN, LOCALISE SMALLEST DISTANCE (SHOULD BE THE CORNER)
    L_curve(:,4)=sqrt(l_curve(:,2)**2+l_curve(:,3)**2)
    i=minloc(L_curve(:,4),dim=1)
    tol_save_graphical=l_curve(i,1)
    write(91+TID,'(A,X,4E15.8)')'#',tol_save_res,tol_save_x,tol_save_resx,tol_save_graphical
    close(91+TID)
    deallocate(L_curve)
    allocate(L_curve_dot_product(l-1))
    allocate(tols(l-1))
    do i=1,l-1
      read(192+TID,'(E15.8,X,E15.8)')tols(i),L_curve_dot_product(i)
    enddo
    close(192+TID)!,status='delete')
    do i=11,l-1
      if(any(L_curve_dot_product(i-10:i-1).lt.0.9999d0))then
        continue
      else
        if(L_curve_dot_product(i).lt.0.9d0*mod_dotprod)then
          mod_dotprod=L_curve_dot_product(i)
!          tol_save_graphical=tols(i)
          L_curve_dot_product(i)=1.d0
          if(all(L_curve_dot_product(i+1:i+10).gt.1.d0-1.d-4))then
            exit
          endif
        endif
      endif
    enddo
  elseif(opt.eq.'M')then
!    do i=1,K
!      s_store(i,i)=s(i)/(s(i)**2+tol_guess**2)
!    enddo
    do i=1,k
      if(dabs(s(i)).gt.tol_guess)then
        s_store(i,i)=1.d0/s(i)
      else
        s_store(i,i)=1.d0/(maxval(abs(s))*dble(k))
      endif
    enddo
!    j=0
    do j=K-1,1,-1
!    do while(tol_guess.lt.5.d0*maxval(abs(s)))
!      j=j+1
!      range_factor=range_factor_init**sqrt(dble(j))

      x_lambda=matmul(matmul(matmul(transpose(vt),s_store),transpose(u)),bb)
      residue=matmul(a,x_lambda)-bb
      if(check_vec_zero)x_0=sum(x_lambda)/dble(n)
      mod_xlambda(2)=log(sum((x_lambda-x_0)**2))
      mod_residue(2)=log(sum(residue(:,1)**2))
      mod_resx(2)=mod_xlambda(2)+mod_residue(2)
      write(91+TID,'(E15.8,X,E15.8,X,E15.8)')tol_guess,mod_residue(2),mod_xlambda(2)
      if(mod_xlambda(2).lt.mod_xlambda(1))then
        mod_xlambda(1)=mod_xlambda(2)
        tol_save_x=tol_guess
      endif
      if(mod_residue(2).lt.mod_residue(1))then
        mod_residue(1)=mod_residue(2)
        tol_save_res=tol_guess
      endif
      if(mod_resx(2).lt.mod_resx(1))then
        mod_resx(1)=mod_resx(2)
        tol_save_resx=tol_guess
      endif
!      tol_guess=tol_guess+range_factor*tol_init!/dble(steps)
      tol_guess=dabs(s(j))
      s_store=0.d0
!      do i=1,K
!        s_store(i,i)=s(i)/(s(i)**2+tol_guess**2)
!      enddo
      do i=1,k
        if(dabs(s(i)).gt.tol_guess)then
          s_store(i,i)=1.d0/s(i)
        else
          s_store(i,i)=1.d0/maxval(abs(s)*dble(k))
        endif
      enddo
    enddo
    tol_save_res=min(max(tol_save_res,0.1*pinv),10.d0*pinv)
    tol_save_resx=min(max(tol_save_resx,0.1*pinv),10.d0*pinv)
    tol_save_x=min(max(tol_save_x,0.1*pinv),10.d0*pinv)
    close(91+TID)
    close(192+TID)
    open(file='L_curve_'//CTID//'.dat',unit=91+TID, status='old')
    allocate(L_curve(K-2,4))
    do i=1,K-2
      read(91+TID,'(E15.8,X,E15.8,X,E15.8)')L_curve(i,1),L_curve(i,3),L_curve(i,2)
    enddo
    svar=minval(L_curve(:,2))
    L_curve(:,2)=L_curve(:,2)-svar
    svar=minval(L_curve(:,3))
    L_curve(:,3)=L_curve(:,3)-svar
    L_curve(:,4)=sqrt(l_curve(:,2)**2+l_curve(:,3)**2)
    i=minloc(L_curve(:,4),dim=1)
    tol_save_graphical=l_curve(i,1)
    tol_save_graphical=min(max(tol_save_graphical,0.1*pinv),10.d0*pinv)
    close(91+TID)
  else
    print*, 'NO VALID PSEUDOINVERSE OPTION GIVEN'
    stop
  endif
  if(check_vec_zero)x_0=0.d0
  deallocate(u)
  deallocate(s)
  deallocate(vt)

end subroutine tolerance_simple

subroutine tolerance_estimate(A,b,m,n,tol_guess,x_guess)
  implicit none
  integer, intent(in) :: m,n
  real(8), intent(in) :: A(m,n),b(m),x_guess(n)
  real(8), intent(out):: tol_guess
  integer K,i,j,steps,max_at_step
  real(8), allocatable :: a_in(:,:),u(:,:),s(:),vt(:,:),s_store(:,:),&
  weights(:)
  real(8) mod_xlambda(3),mod_residue(3),hessian,gradient,tol_save_h,&
  tol_save_g,hessian_max,grad_max,tol_init,x_lambda(n,1),bb(m,1),&
  gradient_xl,gradient_rl,hessian_xl,hessian_rl,kappa,kappa_max,&
  tol_save_kappa

  steps=100
  allocate(a_in(m,n))
  a_in=a
  K = MIN(M,N)
  bb(:,1)=b

  allocate(u(m,K))
  allocate(s(K))
  allocate(weights(K))
  allocate(vt(K,n))
  allocate(s_store(K,K))
  s_store=0.d0
  call svd(m,n,K,a_in,s,u,vt)
  tol_init=s(1)
  tol_guess=tol_init
  tol_save_h=tol_guess
  tol_save_g=tol_guess
  hessian_max=0.d0
  grad_max=0.d0
  kappa_max=-huge(1.d0)
  tol_save_kappa=-huge(1.d0)
  weights=s**2/(s**2+tol_guess**2)
  do i=1,K
    s_store(i,i)=s(i)/(s(i)**2+tol_guess**2)
  enddo
!  open(unit=92,file='kurvature.dat',status='replace')
!  open(unit=91,file='L_curve.dat',status='replace')
  do j=1,steps
    if(j.gt.2)then
      mod_xlambda(3)=mod_xlambda(2)
      mod_residue(3)=mod_residue(2)
    endif
    if(j.gt.1)then
      mod_xlambda(2)=mod_xlambda(1)
      mod_residue(2)=mod_residue(1)
    endif
    mod_xlambda(1)=0.d0
    mod_residue(1)=0.d0
    do i=1,K
!      mod_xlambda(1)=mod_xlambda(1)+((weights(i)/s(i))*sum(u(:,i)*b))**2
      mod_residue(1)=mod_residue(1)+((1.d0-weights(i))*sum(u(:,i)*b))**2
    enddo
    x_lambda=matmul(matmul(matmul(transpose(vt),s_store),transpose(u)),bb)
    mod_xlambda(1)=sum(dabs(x_lambda(:,1)-x_guess)**2,dim=1)
    mod_xlambda(1)=dlog(mod_xlambda(1))
    mod_residue(1)=dlog(mod_residue(1))
!    write(91,*)mod_residue(1)/2.d0,mod_xlambda(1)/2.d0,tol_guess
    if(j.gt.1)then
      gradient=(mod_xlambda(1)-mod_xlambda(2))/(mod_residue(1)-mod_residue(2))
      gradient_xl=(mod_xlambda(1)-mod_xlambda(2))/(-tol_init/dble(steps))
      gradient_rl=(mod_residue(1)-mod_residue(2))/(-tol_init/dble(steps))
      if(gradient.gt.grad_max)then
        grad_max=dabs(gradient)
        tol_save_g=tol_guess
      endif
    endif
    if(j.gt.2)then
      hessian=(mod_xlambda(1)-mod_xlambda(2))/(mod_residue(1)-mod_residue(2))-&
      (mod_xlambda(2)-mod_xlambda(3))/(mod_residue(2)-mod_residue(3))
      hessian=hessian/(mod_residue(1)-mod_residue(2))/(mod_residue(2)-mod_residue(3))
      hessian_xl=((mod_xlambda(1)-mod_xlambda(2))/(-tol_init/dble(steps)) - &
      (mod_xlambda(2)-mod_xlambda(3))/(-tol_init/dble(steps)))/(tol_init/dble(steps))**2
      hessian_rl=((mod_residue(1)-mod_residue(2))/(-tol_init/dble(steps)) - &
      (mod_residue(2)-mod_residue(3))/(-tol_init/dble(steps)))/(tol_init/dble(steps))**2
      if(dabs(hessian).gt.hessian_max)then
        hessian_max=dabs(hessian)
        tol_save_h=tol_guess
      endif
      kappa=2.d0*(gradient_rl*hessian_xl - hessian_rl*gradient_xl)/&
      ((gradient_rl**2 + gradient_xl**2)**(1.5d0))
      if(kappa.gt.kappa_max)then
        max_at_step=j
        kappa_max=kappa
        tol_save_kappa=tol_guess
      endif
!      write(92,*)tol_guess,kappa
    endif
    tol_guess=tol_guess-tol_init/dble(steps)
    weights=s**2/(s**2+tol_guess**2)
    do i=1,K
      s_store(i,i)=s(i)/(s(i)**2+tol_guess**2)
    enddo
  enddo
!  close(92)
!  close(91)
!  pause
  if(max_at_step.lt.steps .and. tol_save_kappa.gt.0.d0)then
    tol_guess=tol_save_kappa
  else
    tol_guess=0.d0
  endif
  deallocate(u)
  deallocate(s)
  deallocate(vt)

end subroutine tolerance_estimate

real(8) function kronecker(i,j)
  implicit none
  integer, intent(in) :: i,j

  if(i.eq.j)then
    kronecker=1.d0
  else
    kronecker=0.d0
  endif
  return

end function kronecker

SUBROUTINE matrix_diagonalise(N,H,E,U) 
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
    PRINT*,'DIAGONALIZATION NOT CONVERGED'
    STOP
  END IF

END SUBROUTINE MATRIX_DIAGONALISE

subroutine sort_smallest_first(n,array_in,sortlist)
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
end subroutine sort_smallest_first

subroutine sort_largest_first(n,array_in,sortlist)
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
end subroutine sort_largest_first

subroutine vary_vector(x_in,nat,delta_x,delta_r,i_red)
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

end subroutine vary_vector

subroutine StripSpaces(string)
  character(len=*) :: string
  integer :: stringLen 
  integer :: last, actual

  stringLen = len (string)
  last = 1
  actual = 1

  do while (actual < stringLen)
      if (string(last:last) == ' ') then
          actual = actual + 1
          string(last:last) = string(actual:actual)
          string(actual:actual) = ' '
      else
          last = last + 1
          if (actual < last) &
              actual = last
      endif
  end do

end subroutine StripSpaces

real(8) FUNCTION zeta_amn(a,m,n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: a,m,n
  real(8) kronecker

  zeta_amn=kronecker(a,m)-kronecker(a,n)
  return

END FUNCTION zeta_amn

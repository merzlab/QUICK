! read pot files from Guntram Rauhut in their MolPro format
! Task: a get_gradient routine and a get_hessian routine
! 
! Read the file and store data in arrays
!
! Convert cartesian coordinates into their normal coordinates. Calculate the gradients
!
! I don't understand the scaling of their vectors for the time being...
!
module vib_pot
  use dlf_parameter_module, only: rk
  use dlf_allocate, only: allocate,deallocate
  use dlf_global, only: printl,stdout
  implicit none

  public :: read_pot_init, read_pot_destroy, vib_get_gradient, vib_get_hessian

  private
  integer,save :: nvar ! number of xcoords
  integer,save :: nivar ! number of icoords (generally nvar-6)
  real(rk), allocatable,save :: refcoords(:) ! (nvar)
  real(rk), allocatable,save :: mode(:,:) ! (nvar,nivar) mode(:,ivar) is \vec Q_i in the doc
                         ! coeff1d(ivar,ico) = c_{i,n} in the docu
  real(rk), allocatable,save :: coeff1d(:,:) ! (nivar,poly_order(1))
  integer,save :: poly_order(4)


contains

! read data from pot file and store them in arrays
subroutine read_pot_init(nat)
  use dlf_constants, only: dlf_constants_get
  implicit none
  integer, intent(in):: nat
  character(*),parameter :: pot_file="read_pot/nh3-4D-f12-big3.pot_NEW_1d"
  character(*),parameter :: molpro_file="read_pot/nh3-poly.out"
  !
  integer :: ivar,ios,ivar2,iat,icoord,ico,jvar
  character(256) :: line
  real(8) :: cminv,svar,svar2
  real(8),allocatable :: qqmat(:,:)

  ! find the position to start reading
  open(file=pot_file,unit=20,action="read")
  do ivar=1,100
    read(20,'(a)') line
    read(line,*,iostat=ios) ivar2
    if(ios==0) then
      if(ivar2==nat) exit
    end if
  end do
  if(ivar2/=nat) call dlf_fail("Failed to find the number of atoms in pot file")

  nvar=3*nat
  nivar=3*nat-6

  ! allocate arrays
  call allocate(refcoords,nvar)
  call allocate(mode,nvar,nivar)

  ! read reference coordinates
  do iat=1,nat
    read(20,'(a)') line
    read(line(3:),*) refcoords(iat*3-2:iat*3)
  end do

  ! read modes (\vec Q_i in the doc)
  do ivar=1,nivar
    read(20,'(a)') line
    read(20,'(a)') line ! this is the frequency and the scaling factor
                        ! - ignore them for now
    do icoord=1,nvar
      read(20,*) mode(icoord,ivar)
    end do
  end do

  ! everything we need from the pot file is read in now.
  close(20)

  ! print for debug purposes
  print*,"Refernce Coords"
  do iat=1,nat
    write(*,'(i3,3f20.10)') iat,refcoords(iat*3-2:iat*3)
  end do
  print*,"Modes"
  ! this only looks good for 6 modes:
  write(*,'(6i15)') 1,2,3,4,5,6
  do icoord=1,nvar
    write(*,'(6f15.10)') mode(icoord,:)
  end do
  print*,"Norm:"
  write(*,'(6f15.10)') (sqrt(sum(mode(:,ivar)**2)),ivar=1,nivar)

  ! transform modes to "inverse modes"
  call allocate(qqmat,nivar,nivar)
  qqmat=0.D0
  do ivar=1,nivar
    do jvar=1,nivar
      qqmat(ivar,jvar)=sum(mode(:,ivar)*mode(:,jvar))
    end do
  end do
  call dlf_matrix_invert(nivar,.true.,qqmat,svar)
  print*,"Determinat:",svar
  mode=transpose(matmul(qqmat,transpose(mode)))
  call deallocate(qqmat)
  print*,"Modes transformed"
  ! this only looks good for 6 modes:
  write(*,'(6i15)') 1,2,3,4,5,6
  do icoord=1,nvar
    write(*,'(6f15.10)') mode(icoord,:)
  end do
  print*,"Norm:"
  write(*,'(6f15.10)') (sqrt(sum(mode(:,ivar)**2)),ivar=1,nivar)

  ! now read the molpro output for the coefficients of the polynomials
  open(file=molpro_file,unit=21,action="read")
  ! fist find the position to start the reading:

  ! find out order of polynomial fit:
  poly_order=0
  do
    read(21,'(a)',iostat=ios) line
    if(ios/=0) exit
    if(index(line,"Number of fit functions for fit of 1D grid:")>0) then
      read(line(77:),*,iostat=ios) ivar
      if(ios==0) poly_order(1)=ivar
    end if
    if(index(line,"Number of fit functions for fit of 2D grid:")>0) then
      read(line(77:),*,iostat=ios) ivar
      if(ios==0) poly_order(2)=ivar
    end if
    if(index(line,"Number of fit functions for fit of 3D grid:")>0) then
      read(line(77:),*,iostat=ios) ivar
      if(ios==0) poly_order(3)=ivar
    end if
    if(index(line,"Number of fit functions for fit of 4D grid:")>0) then
      read(line(77:),*,iostat=ios) ivar
      if(ios==0) poly_order(4)=ivar
      exit
    end if
  end do

  if(printl>=4) then
    do ivar=1,4
      write(stdout,"('Order of polynomial for ',i1,'D terms:',i3)") ivar,poly_order(ivar) 
    end do
  end if
  if(maxval(poly_order)==0) call dlf_fail("No polynomial order found")

  do
    read(21,'(a)',END=1101) line
    !print*,index(line,"PROGRAM * POLY (Transformation of PES grid representation to polynomials)"),trim(line)
    if(index(line,"PROGRAM * POLY (Transformation of PES grid representation to polynomials)")>0) goto 1102
  end do
1101 call dlf_fail("No coefficients found in molpro file")
1102 continue

  ! search for string "Mode"
  do
    read(21,'(a)',END=1103) line
    if(index(line,"Mode")>0) goto 1104
  end do
1103 call dlf_fail("No coefficients found in molpro file")
1104 continue

  do ivar=1,nivar+2
    read(21,'(a)') line
    !print*,trim(line)
  end do
    
  ! now, we are in the first line to read coefficients. We have to find out what oder of polynomial is used.

  ! for the time being, we only use 1D terms:
  call allocate(coeff1D,nivar,poly_order(1))
  coeff1D=0.D0

  do ivar=1,nivar
    read(21,'(a)') line
    read(line(6:),*,iostat=ios) coeff1D(nivar-ivar+1,1:min(10,poly_order(1)))
    if(ios/=0) call dlf_fail("Error reading 1D coefficients")
    ! we ignore everything in the next line - until the number 13 vs. 12 is solved
    if(poly_order(1)>10) read(21,'(a)') line
  end do

  ! print out coefficients for debug purposes
  if(printl>=4) then
    write(stdout,*) "Coefficients of 1D expansion"
    line=""
    do ico=1,poly_order(1)
      write(line,'(a,i12)') trim(line),ico
    end do
    write(stdout,'(3x,a)') trim(line)
    do ivar=1,nivar
      line=""
      do ico=1,poly_order(1)
        write(line,'(a,e12.4)') trim(line),coeff1D(ivar,ico)
      end do
      write(stdout,'(i3,a)') ivar,trim(line)
    end do
  end if

  ! test: convert from wave numbers into hartree:
  call dlf_constants_get("CM_INV_FROM_HARTREE",cminv)
  print*,"cminv",cminv
  !coeff1D=coeff1D*cminv
  
!!$  ! test 
!!$  open(file="mode6",unit=100)
!!$  do ivar=0,1000
!!$    svar=ivar/10.D0
!!$    svar2=0.D0
!!$    do ico=1,poly_order(1)
!!$      svar2=svar2+svar**ico*coeff1D(6,ico)
!!$    end do
!!$    write(100,*) svar,svar2
!!$  end do
!!$  close(100)
!!$  open(file="min6",unit=100)
!!$  write(100,'(3f20.10)') refcoords+35.5D0*mode(:,6)
!!$  close(100)
!!$  ! test ortogonality:
!!$  print*,"Test orthogonality"
!!$  do ivar=1,nivar
!!$    write(*,'(6f10.5)') (sum(mode(:,ivar)*mode(:,ico)),ico=1,nivar)
!!$  end do

  close(21)

end subroutine read_pot_init

! deallocate arrays
subroutine read_pot_destroy
  call deallocate(refcoords)
  call deallocate(mode)
  call deallocate(coeff1D)
end subroutine read_pot_destroy

! calculate energy and gradient
! for the time being only 1D terms
subroutine vib_get_gradient(xcoords,energy,xgradient)
  real(rk), intent(in)  :: xcoords(nvar)
  real(rk), intent(out) :: energy
  real(rk), intent(out) :: xgradient(nvar)
  !
  real(rk) :: modals(nivar)
  integer  :: ico,ivar

  call coords_to_modals(xcoords,modals)

  ! 1D energy
  energy=0.D0
  do ico=1,poly_order(1)
    energy=energy+sum(coeff1D(:,ico)*(modals(:))**ico)
  end do

  ! 1D gradient
  xgradient=0.D0
  do ico=1,poly_order(1)
    do ivar=1,nivar 
      xgradient(:)=xgradient(:)+ &
          modals(ivar)**(ico-1)*dble(ico)*coeff1D(ivar,ico)*mode(:,ivar)
    end do
  end do
end subroutine vib_get_gradient

subroutine vib_get_hessian(xcoords,xhessian)
  real(rk), intent(in)  :: xcoords(nvar)
  real(rk), intent(out) :: xhessian(nvar,nvar)
  !
  real(rk) :: modals(nivar)
  real(rk) :: qiqi(nvar,nvar) ! \vec Q_i \otimes \vec Q_i
  integer  :: ico,ivar,ivarq,jvarq

  call coords_to_modals(xcoords,modals)

  ! 1D Hessian
  xhessian=0.D0
  do ivar=1,nivar
    
    ! calculate qiqi
    qiqi=0.D0
    do ivarq=1,nvar
      do jvarq=1,nvar
        qiqi(ivarq,jvarq)=mode(ivarq,ivar)*mode(jvarq,ivar)
      end do
    end do

    do ico=2,poly_order(1) ! linear term vanishes
      xhessian(:,:)=xhessian(:,:)+ &
          modals(ivar)**(ico-2)*dble(ico)*dble(ico-1)*coeff1D(ivar,ico)*qiqi(:,:)
    end do

  end do

end subroutine vib_get_hessian

subroutine coords_to_modals(xcoords_,modals)
  real(rk), intent(in) :: xcoords_(nvar)
  real(rk), intent(out):: modals(nivar)
  real(rk) :: xcoords(nvar)
  integer  :: ivar
  !
  xcoords=xcoords_-refcoords
  do ivar=1,nivar
    modals(ivar)=sum(xcoords*mode(:,ivar))
  end do
end subroutine coords_to_modals
end module vib_pot



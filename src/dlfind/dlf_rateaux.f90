! **********************************************************************
! **           Auxiliary files for the Reaction Rate Module           **
! **             here are the routines for:                           **
! **             Microcanonical rate constants                        **
! **             File I/O specific to rate constants                  **
! **********************************************************************
!!****h* neb/rateaux
!!
!! NAME
!! rateaux
!!
!! FUNCTION
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
!!  Copyright 2010-2016 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
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

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! % Routines related to microcanonical rate constants
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

! calculate the stability parameters by integrating the monodromy matrix
! (stability matrix differential equation)
subroutine stability_parameters_monodromy
  use dlf_neb, only: neb,unitp,beta_hbar
  use dlf_allocate, only: allocate,deallocate
  use dlf_qts
  use dlf_global, only: glob,printl,printf,stdout
  use dlf_constants, only : dlf_constants_get
  implicit none
  integer :: varperimage,nimage,iimage,ivar,jvar
  real(rk), allocatable :: rmat0(:,:),rmatp(:,:),fmat(:,:)
  real(rk), allocatable :: eigvalr(:),eigvali(:),eigvecr(:,:)
  real(rk), allocatable :: sortlist(:),evoverlap(:,:)
  character(1024) :: line
  integer, allocatable :: mapimage(:) ! 2*nimage
  integer, allocatable :: mapimage2(:) ! 2*nimage
  ! RK-stuff
  real(rk), allocatable :: k1(:,:),k2(:,:),k3(:,:),k4(:,:),fmatp(:,:),fmatpp(:,:)
  real(rk) :: delta
  logical :: tok
  
  if(printl>=4) write(stdout,*) "Integrating the monodromy matrix"

  varperimage=neb%varperimage
  nimage=neb%nimage

  call allocate(rmat0,2*varperimage,2*varperimage)
  call allocate(rmatp,2*varperimage,2*varperimage)
  call allocate(fmat,2*varperimage,2*varperimage)
  call allocate(sortlist,2*varperimage)
  call allocate(mapimage,2*nimage)
  !allocate(mapimage2(2*nimage)) ! can be removed later
  call allocate(k1,2*varperimage,2*varperimage)
  call allocate(k2,2*varperimage,2*varperimage)
  call allocate(k3,2*varperimage,2*varperimage)
  call allocate(k4,2*varperimage,2*varperimage)
  call allocate(fmatp,2*varperimage,2*varperimage)
  call allocate(fmatpp,2*varperimage,2*varperimage)
  

  rmat0=0.D0
  do ivar=1,2*varperimage
    rmat0(ivar,ivar)=1.D0
  end do

  ! one variable to proceed along the path back and forth
  do iimage=1,nimage
    mapimage(iimage)=iimage
  end do
  do iimage=nimage,1,-1
    mapimage(2*nimage-iimage+1)=iimage
  end do
!!$  ! shift images - can be removed later
!!$  ivar=0
!!$  do iimage=1,2*nimage
!!$     if(iimage+ivar <= 2*nimage) then
!!$        mapimage2(iimage)=mapimage(iimage+ivar)
!!$     else
!!$        mapimage2(iimage)=mapimage(iimage+ivar-2*nimage)
!!$     end if
!!$  end do
!!$  mapimage=mapimage2


  ! integrate along the whole path
  do iimage=1,2*nimage,2 ! the step 2 must be removed for impl. Euler!
    !print*,iimage,mapimage(iimage)
    ! construct fmat
    fmat=0.D0
    do ivar=1,varperimage
      fmat(ivar+varperimage,ivar)=-1.D0
    end do
    do ivar=1,varperimage
      do jvar=1,varperimage
        ! minus sign because we propagate in real time
        fmat(ivar,jvar+varperimage)=-qts%vhessian(ivar,jvar,mapimage(iimage))
      end do
    end do
    if(iimage<2*nimage-1) then ! only needed for RK
      fmatp=0.D0
      do ivar=1,varperimage
        fmatp(ivar+varperimage,ivar)=-1.D0
      end do
      do ivar=1,varperimage
        do jvar=1,varperimage
          ! minus sign because we propagate in real time
          fmatp(ivar,jvar+varperimage)=-qts%vhessian(ivar,jvar,mapimage(iimage+1))
        end do
      end do
      fmatpp=0.D0
      do ivar=1,varperimage
        fmatpp(ivar+varperimage,ivar)=-1.D0
      end do
      do ivar=1,varperimage
        do jvar=1,varperimage
          ! minus sign because we propagate in real time
          fmatpp(ivar,jvar+varperimage)=-qts%vhessian(ivar,jvar,mapimage(iimage+2))
        end do
      end do
    end if

!!$     ! print one fmat for test
!!$     if(iimage==1) then
!!$        print*,"F(1):"
!!$        do ivar=1,2*varperimage
!!$           write(*,'(24f9.5)') real(fmat(:,ivar))
!!$        end do
!!$     end if

!!$    rmatp=rmat0
!!$    !R+ = R0 - delta * F(1/2) * R(1/2)
!!$    ! implicit Euler
!!$    rmatp = rmat0 - beta_hbar/dble(2*nimage)*matmul(fmat,0.5D0*(rmatp+rmat0)) ! repeat that line!
!!$    rmatp = rmat0 - beta_hbar/dble(2*nimage)*matmul(fmat,0.5D0*(rmatp+rmat0)) ! repeat that line!
!!$    rmatp = rmat0 - beta_hbar/dble(2*nimage)*matmul(fmat,0.5D0*(rmatp+rmat0)) ! repeat that line!
!!$    rmatp = rmat0 - beta_hbar/dble(2*nimage)*matmul(fmat,0.5D0*(rmatp+rmat0)) ! repeat that line!
!!$    rmatp = rmat0 - beta_hbar/dble(2*nimage)*matmul(fmat,0.5D0*(rmatp+rmat0)) ! repeat that line!
!!$    rmatp = rmat0 - beta_hbar/dble(2*nimage)*matmul(fmat,0.5D0*(rmatp+rmat0)) ! repeat that line!
!!$    rmatp = rmat0 - beta_hbar/dble(2*nimage)*matmul(fmat,0.5D0*(rmatp+rmat0)) ! repeat that line!
!!$    rmatp = rmat0 - beta_hbar/dble(2*nimage)*matmul(fmat,0.5D0*(rmatp+rmat0)) ! repeat that line!
!!$    rmatp = rmat0 - beta_hbar/dble(2*nimage)*matmul(fmat,0.5D0*(rmatp+rmat0)) ! repeat that line!
!!$    rmatp = rmat0 - beta_hbar/dble(2*nimage)*matmul(fmat,0.5D0*(rmatp+rmat0)) ! repeat that line!
!!$
    ! 4th order Runge-Kutta
    delta=beta_hbar/dble(2*nimage)
    k1=-matmul(fmat,rmat0)
    k2=-matmul(fmatp,rmat0+k1*delta)
    k3=-matmul(fmatp,rmat0+k2*delta)
    k4=-matmul(fmatpp,rmat0+k3*delta*2.D0)
    rmatp=rmat0+2.D0/3.D0*delta*(0.5D0*k1+k2+k3+0.5D0*k4)
    if(iimage==2*nimage-2) exit

    ! propagate
    rmat0=rmatp 
  end do ! iimage=1,2*nimage

  call deallocate(rmat0)
  call deallocate(fmat)
  call deallocate(mapimage)

  ! now analyse and diagonalise rmatp

!!$  print*,"real(R(T)):"
!!$  do ivar=1,2*varperimage
!!$     write(*,'(24f9.5)') real(rmatp(:,ivar))
!!$  end do
!!$  print*,"imag(R(T)):"
!!$  do ivar=1,2*varperimage
!!$     write(*,'(24f9.5)') imag(rmatp(:,ivar))
!!$  end do

  call allocate(eigvalr,2*varperimage)
  call allocate(eigvali,2*varperimage)
  call allocate(eigvecr,2*varperimage,2*varperimage)

  !call dlf_matrix_diagonalise(2*varperimage,rmatp,eigval,eigvec)
  ! c_diagonal_general(nm,a,eignum,eigvectr)
  ! rmatp is a real matrix. A different routine could be used?
  !  call c_diagonal_general(2*varperimage,rmatp,eigval,eigvec)
  call dlf_matrix_diagonalise_general(2*varperimage,rmatp,eigvalr,eigvali,eigvecr)

  ! sort the eigenvalues by their real part:
  sortlist=eigvalr
  rmatp(:,1)=eigvali ! abusing rmatp here, it is not needed any more
  do ivar=1,2*varperimage
    jvar=maxloc(sortlist,dim=1)
    eigvalr(ivar)=sortlist(jvar)
    eigvali(ivar)=rmatp(jvar,1)
    sortlist(jvar)=-huge(1.D0)
  end do

  call deallocate(rmatp)
  call deallocate(sortlist)

  if(printl>=4) then
    write(stdout,*) "Eigenvalues of the monodromy matrix (real,imag, log(real), log(real)/beta):"
    do ivar=1,2*varperimage
      if(eigvalr(ivar)>0.D0) then
        write(*,'(i4,2es15.6,2x,2es16.8)') ivar,eigvalr(ivar),eigvali(ivar), &
            log(eigvalr(ivar)),log(eigvalr(ivar))/beta_hbar
      else
        write(*,'(i4,2es15.6)') ivar,eigvalr(ivar),eigvali(ivar)
      end if
      ! blank lines to distinguish zero and non-zero values
      if(ivar==varperimage-glob%nzero-1) write(*,*)
      if(ivar==varperimage+glob%nzero+1) write(*,*)
    end do
  end if

  ! try SVD instead of full diagonalisation
  !call dlf_matrix_svd(2*varperimage,2*varperimage,real(rmatp),eigvali)
  ! does not work well, even for temperatures where dlf_matrix_diagonalise_general is still stable
  ! singular values are typically too large (too far away from 1)

!!$  print*,"Eigenvectors - real parts (with their eigenvalues in the fist line)"
!!$  write(*,'(4x,2es15.6)') eigvalr(1),eigvalr(9)
!!$  do ivar=1,2*varperimage
!!$     write(*,'(i4,2es15.6)') ivar,eigvecr(ivar,1),eigvecr(ivar,9)
!!$  end do

!!$  ! analyse eigenvector overlap - maybe one can learn something from that?
!!$  call allocate(evoverlap,2*varperimage,2*varperimage))
!!$  do ivar=1,2*varperimage
!!$    do jvar=1,2*varperimage
!!$      evoverlap(ivar,jvar)=sum(eigvecr(1:varperimage,ivar)*eigvecr(1:varperimage,jvar))-&
!!$          sum(eigvecr(varperimage+1:,ivar)*eigvecr(varperimage+1:,jvar))
!!$    end do
!!$    print*,'overl i i ',sum(eigvecr(1:varperimage,ivar)*eigvecr(1:varperimage,ivar)), &
!!$        sum(eigvecr(varperimage+1:,ivar)*eigvecr(varperimage+1:,ivar))
!!$    evoverlap(ivar,ivar)=8.88D0
!!$  end do
!!$  ! can one print that overlap?
!!$  do ivar=1,2*varperimage
!!$    write(*,'(i2,24f5.2)') ivar,abs(evoverlap(ivar,:))
!!$  end do
!!$  call deallocate(evoverlap)
  call deallocate(eigvecr)

  ! print the stability parameters in increasing order: (here version with signed sorting)
  if(printl>=2) then
    tok=.true.
    line=""
    do ivar=varperimage-glob%nzero-1,1,-1
      if(abs(eigvali(ivar))>1.D-8) then
        tok=.false.
        exit
      end if
      write(line,'(a,es15.8)') trim(line),log(eigvalr(ivar))/beta_hbar
    end do
    qts%stabpar_diffeq=-1.D0
    if(tok) then
      qts%stabpar_diffeq=log(eigvalr(1:varperimage-glob%nzero-1))/beta_hbar
      write(*,'("Stability matrix differential equation / monodromy matrix:")')
      write(*,'("Stability parameters from monodromy matrix: ",a)') trim(line)
    else
      write(*,'("No reliable results from the stability matrix differential equation / monodromy matrix!")')
    end if
  end if

  call deallocate(eigvalr)
  call deallocate(eigvali)

end subroutine stability_parameters_monodromy

! calculate stability parameters by following a co-moving coordinate
! system along the instanton path
subroutine stability_parameters_trace(rotmode)
  use dlf_neb, only: neb,unitp,beta_hbar
  use dlf_allocate, only: allocate,deallocate
  use dlf_qts
  use dlf_global, only: glob,printl,printf,stdout
  use dlf_constants, only : dlf_constants_get
  implicit none
  real(rk), intent(in) :: rotmode(neb%varperimage*neb%nimage*2) ! tangent of the path
  integer :: varperimage,nimage,iimage,ivar,jvar
  real(rk), allocatable :: trans_rot_vectors(:,:)
  real(rk) :: avgeigval,parallelcont,svar
  real(rk), allocatable :: basis(:,:)
  real(rk), allocatable :: phessian(:,:),peigval(:),peigvec(:,:),peigvec_old(:,:) ! pvar
  real(rk), allocatable :: avg_ev(:) ! pvar
  integer :: imode,jmode,pickmode,pvar,iter
  logical :: tstartimage
  integer, allocatable :: sortlist(:) ! pvar
  character(1024) :: line
  ! TS-Hessian
  logical :: tok,havetshess
  integer :: nat,nimage_read_,varperimage_,iimage_
  real(rk) :: ene(1),etunnel,arr2(2)
  integer, allocatable :: imagelist(:)
  real(rk), allocatable :: xcoords(:),igradient(:),ts_hessian(:,:),dist(:),mass(:)
  real(rk), allocatable :: basis_init(:,:),peigvec_init(:,:)

  varperimage=neb%varperimage
  nimage=neb%nimage
  call allocate(trans_rot_vectors,varperimage,glob%nzero)
  call allocate(basis,varperimage,varperimage)
  pvar=varperimage-1-glob%nzero
  call allocate(phessian,pvar,pvar)
  call allocate(peigval,pvar)
  call allocate(peigvec,pvar,pvar)
  call allocate(peigvec_old,pvar,pvar)
  call allocate(sortlist,pvar)
  call allocate(imagelist,nimage)
  call allocate(avg_ev,pvar)
  call allocate(basis_init,varperimage,varperimage)
  call allocate(peigvec_init,pvar,pvar)

  ! initialise imagelist
  do iimage=1,nimage
    imagelist(iimage)=iimage
  end do

  ! try to get the TS-Hessian as a reference for sorting:
  havetshess=.false.
  call head_qts_hessian(nat,nimage_read_,varperimage_,"ts",tok) ! everything except label is out
  if(tok) then
    if(nimage_read_/=1) tok=.false.
    if(varperimage_/=varperimage) tok=.false.
  end if
  if(tok) then
    call allocate(xcoords,3*nat)
    call allocate(igradient,varperimage)
    call allocate(ts_hessian,varperimage,varperimage)
    call allocate(dist,2)
    call allocate(mass,nat)
    call read_qts_hessian(nat,nimage_read_,varperimage_,svar,&
        ene,xcoords,igradient,ts_hessian,etunnel,dist,mass,"ts",tok,arr2)
    call deallocate(xcoords)
    call deallocate(igradient)
    call deallocate(dist)
    call deallocate(mass)
    havetshess=tok
    tok=.true. ! signalling that ts_hessian is allocated

    ! define the list over which to cycle over the images:
    ! start with the image with highest energy, which presumably is 
    ! closest to the TS. Sort the eigenvalues for that image according to 
    ! the overlap with the TS. Then proceed to nimage. Then continue at the image
    ! previous to the highest one (re-use the basis and sortlist from the start)
    ! and proceed to image 1.
    ivar=maxloc(neb%ene,dim=1) ! image index with highest energy, presumably
                               ! closest to TS
    !print*,"Image with highest energy:",ivar
    do iimage=1,nimage
      if(iimage-1+ivar<=nimage) then
        imagelist(iimage)=iimage-1+ivar
      else
        !exit
        imagelist(iimage)=nimage+1-iimage
      end if
    end do
    !print*,"imagelist",imagelist
  end if

  !print*,"Contributions of images:
  !open(unit=337,file='eigenvalues.phessian')
  if(printl>=4) open(unit=338,file='eigenvalues.phessian_sorted')
  !open(unit=335,file='eigenvalue.1')
  avgeigval=0.D0
  tstartimage=.true.
  call random_number(basis) ! use random basis vectors for everything except translation/rotation
  avg_ev=0.D0
!  do iimage=nimage,1,-1 ! we still need to think where to start and in which direction to propagate.
  do iimage_=1,nimage
    iimage=imagelist(iimage_)
    basis(:,1:glob%nzero+1)=0.D0
    trans_rot_vectors=0.D0
    if(glob%nzero>0.and.varperimage==3*glob%nat) then
      call dlf_trans_rot(glob%nzero,neb%xcoords(:,iimage),trans_rot_vectors)
    end if
    ! basis(:,1) is the tangential mode
    basis(:,1)=rotmode(neb%cstart(iimage):neb%cend(iimage))
    svar=sum(basis(:,1)*basis(:,1))
    basis(:,1)=basis(:,1)/sqrt(svar) ! normalise tangent vector
    ! basis(:,2-7) are the rotation and translation modes
    basis(:,2:glob%nzero+1)=trans_rot_vectors

    ! now check if the modes are orthogonal:
!!$    print*,"image",iimage
!!$    do imode=1,7
!!$      write(*,'(12f10.5)') basis(:,imode)
!!$    end do
!!$    do imode=1,7
!!$      do jmode=imode,7
!!$        print*,"<",imode,",",jmode,">=",sum(basis(:,imode)*basis(:,jmode))
!!$      end do
!!$    end do

    ! now care about other modes

    ! orthogonalise all modes to each other (Gram-Schmidt)
    ! assume that the rot/trans and tangential modes are already orthogonal
    do iter=1,5 ! apparently Gram-Schmidt needs a few iterations...
      do imode=glob%nzero+2,varperimage
        ! orthogonalise that to all lower modes
        do jmode=imode-1,1,-1
          basis(:,imode)=basis(:,imode)-sum(basis(:,imode)*basis(:,jmode))/&
              sum(basis(:,jmode)**2)*basis(:,jmode)
        end do
        svar=sqrt(sum(basis(:,imode)**2))
        basis(:,imode)=basis(:,imode)/svar
      end do
    end do
!!$    print*,"Basis after orthogonalisation:"
!!$    do imode=1,varperimage
!!$      write(*,'(i3,12f10.5)') imode,basis(:,imode)
!!$    end do

    ! project the Hessian
    phessian=matmul(transpose(basis(:,glob%nzero+2:varperimage)),matmul(qts%vhessian(:,:,iimage),&
        basis(:,glob%nzero+2:varperimage)))
    call dlf_matrix_diagonalise(pvar,phessian,peigval,peigvec)

    ! re-sort them / trace them
    if(.not.tstartimage) then
      phessian=matmul(transpose(peigvec),peigvec_old) ! projection matrix

!!$      print*,"Projection matrix:",iimage
!!$      do imode=1,pvar
!!$        write(*,'(i3,6f10.5)') imode,phessian(:,imode)
!!$      end do
      
      sortlist=maxloc(abs(phessian),dim=1) ! this causes a SIGSEGV with g95
    else
      if(havetshess) then
        print*,"using TS hessian to sort stability parameters"    
        phessian=matmul(transpose(basis(:,glob%nzero+2:varperimage)),matmul(ts_hessian,&
            basis(:,glob%nzero+2:varperimage)))
        call dlf_matrix_diagonalise(pvar,phessian,peigval,peigvec_old)
        phessian=matmul(transpose(peigvec),peigvec_old) ! projection matrix
        sortlist=maxloc(abs(phessian),dim=1)
        !print*,"initial sortlist",sortlist
        basis_init=basis
        do imode=1,pvar
          peigvec_init(:,imode)=peigvec(:,sortlist(imode))
        end do
      else
        ! no sorting necessary
        print*,"ignoring hessian"
        do imode=1,pvar
          sortlist(imode)=imode
        end do
      end if
      tstartimage=.false.
   end if
   
   do imode=1,pvar
     peigvec_old(:,imode)=peigvec(:,sortlist(imode))
   end do

   if(iimage==nimage) then
     basis=basis_init
     peigvec_old=peigvec_init
   end if

   ! print sorted eigenvalues
   line=''
   do imode=1,pvar
     write(line,'(a,f20.15)') trim(line),peigval(sortlist(imode))
     if(peigval(sortlist(imode))>0.D0) then
       avg_ev(imode)=avg_ev(imode)+sqrt(peigval(sortlist(imode))) ! we average the frequencies
       !rather than the eigenvalue. That agrees better with the stability matrix
     else
       ! Do nothing if that eigenvalue is < 0. Is that justified?
     end if
   end do
   
   if(printl>=4) write(338,'(i4,a)') iimage,trim(line)
   
 end do ! iimage=nimage,1,-1

  !close(337)
  if(printl>=4) close(338)

  avg_ev=avg_ev/dble(nimage)
  if(printl>=2) then
    line=''
    do imode=1,pvar
      write(line,'(a,es15.8)') trim(line),avg_ev(imode) !*beta_hbar ! new: print omega_i rather than u_i
      qts%stabpar_trace(imode)=avg_ev(imode)
    end do
    write(stdout,'("Stability parameters from tracing: ",a)') trim(line)
  end if
!  print*,"Sigma from average eigenvalue (JK):",beta_hbar*avgeigval/2.D0/dble(nimage)

  call deallocate(trans_rot_vectors)
  call deallocate(basis)
  call deallocate(phessian)
  call deallocate(peigval)
  call deallocate(peigvec)
  call deallocate(peigvec_old)
  call deallocate(sortlist)
  call deallocate(imagelist)
  call deallocate(avg_ev)
  if(tok) call deallocate(ts_hessian)

end subroutine stability_parameters_trace

! calculate the "sum" of stability parameters from averaging the
! Hessian eigenvalues along the instanton path and subtracting the
! contribution along the path.
!
!this was average eigenvalue (avev)
subroutine sigma_frequency_average(rotmode)
  use dlf_neb, only: neb,unitp,beta_hbar
  use dlf_allocate, only: allocate,deallocate
  use dlf_qts
  use dlf_global, only: glob,printl,printf,stdout
  use dlf_constants, only : dlf_constants_get
  implicit none
  real(rk), intent(in) :: rotmode(neb%varperimage*neb%nimage*2)
  integer :: varperimage,nimage,iimage,ivar,jvar
  real(rk), allocatable :: eigval(:),eigvec(:,:),lasteigvec(:,:),tang(:)
  real(rk) :: avgeigval,parallelcont
  integer, allocatable :: pos(:)
  character(1024):: line
  
  varperimage=neb%varperimage
  nimage=neb%nimage
  call allocate(tang,varperimage)
  call allocate(eigval,varperimage)
  call allocate(eigvec,varperimage,varperimage)
  call allocate(lasteigvec,varperimage,varperimage)
  call allocate(pos,glob%nzero)

  !print*,"Contributions of images:"
  if(printf>=4) open(unit=334,file='eigenvalues.vhessian')
  !open(unit=335,file='eigenvalue.1')
  avgeigval=0.D0
  do iimage=nimage,1,-1
    eigval=0.D0
    call dlf_matrix_diagonalise(varperimage,qts%vhessian(:,:,iimage),eigval,eigvec)

    ! analyse eigenvalues
    if(printf>=4) then
      line=''
      do ivar=1,varperimage
        write(line,'(a,f20.15)') trim(line),eigval(ivar)
      end do
      write(334,'(i4,a)') iimage,trim(line)
    end if
       
    tang=rotmode(neb%cstart(iimage):neb%cend(iimage))
    tang=tang/sqrt(sum(tang**2))
    ! contribution parallel to the path:
    parallelcont=dot_product(tang,matmul(qts%vhessian(:,:,iimage),tang))
    if(parallelcont<0.D0) parallelcont=0.D0
    !print*,iimage,sum(sqrt(abs(eigval))),parallelcont
    ! the 6 smallest eigenvalues should be ignored in that sum
    !print*,"nimageA",sqrt(abs(eigval))
    do ivar=1,glob%nzero
      pos(ivar)=minloc(abs(eigval),dim=1)
      eigval(pos(ivar))=huge(1.D0)
    end do
    do ivar=1,glob%nzero
      eigval(pos(ivar))=0.D0
    end do

    ! sum of frequencies rather than eigenvalues. Negative eigenvalues are
    ! ignored rather than taking the absolute value (as done previously).
    do ivar=1,varperimage
      if(eigval(ivar)>0.D0) then
         !print*,"debug-print" ! why-ever, in OH-H2 the code fails without that print statement for ifort...
        eigval(ivar)=sqrt(eigval(ivar))
      else
        eigval(ivar)=0.D0
      end if
    end do

    !print*,"nimage0",sqrt(abs(eigval))
    avgeigval=avgeigval+sum(eigval)-sqrt(parallelcont)
  end do
  if(printf>=4) close(334)

  ! print result
  if(printl>=2) write(stdout,'("Sigma from frequency averaging: ",es15.7)') beta_hbar*avgeigval/2.D0/dble(nimage)

  call deallocate(pos)
  
  !print*,"End of averaged Eigenvalue by JK"

end subroutine sigma_frequency_average

! Gelfand-Yaglom according to Sean. Later called average Hessian method
! was Gelfand_Yaglom_avg
subroutine stapar_avg_hessian(rotmode)
  use dlf_neb, only: neb,unitp,beta_hbar
  use dlf_allocate, only: allocate,deallocate
  use dlf_qts
  use dlf_global, only: glob,printl,printf,stdout
  use dlf_constants, only : dlf_constants_get
  implicit none
  real(rk), intent(in) :: rotmode(neb%varperimage*neb%nimage*2)
  integer :: varperimage,nimage,iimage,ivar,jvar
  real(rk), allocatable :: hess_avg(:,:),eigval(:),eigvec(:,:),tang(:)
  complex(rk), allocatable :: fmat(:,:)
  complex(rk), allocatable :: ceigval(:),ceigvec(:,:)
  character(2256) :: line
  real(rk) :: inst_vec(neb%varperimage)
  real(rk) :: sigma,proj,amu,CM_INV_FOR_AMu,svar,sigmaprime,parallelcont
  integer :: foundzero,foundrot

  varperimage=neb%varperimage
  nimage=neb%nimage
  call allocate(hess_avg,varperimage,varperimage)
  call allocate(eigval,varperimage)
  call allocate(eigvec,varperimage,varperimage)
  call allocate(tang,varperimage)
  ! find tangent to instanton path
!!$  ! Version A: first to last image
!!$  inst_vec=glob%icoords(neb%cstart(nimage):neb%cend(nimage))-&
!!$      glob%icoords(neb%cstart(1):neb%cend(1))

  ! Version B: first to second image on the side with image accumulation
  !print*,"qts%dist",qts%dist
  !print*,qts%dist(2)-qts%dist(1)
  !print*,qts%dist(nimage)-qts%dist(nimage-1)
  if( abs(qts%dist(2)-qts%dist(1)) < abs(qts%dist(nimage)-qts%dist(nimage-1)) ) then
    ! image 1 is at the end of the path with smaller slope (the side with
    ! image accumulation)
    inst_vec=glob%icoords(neb%cstart(2):neb%cend(2))-&
        glob%icoords(neb%cstart(1):neb%cend(1))
  else
    inst_vec=glob%icoords(neb%cstart(nimage):neb%cend(nimage))-&
        glob%icoords(neb%cstart(nimage-1):neb%cend(nimage-1))
  end if
!!$
!!$  ! version C: normalise vector between each pair of images and add:
!!$  inst_vec=0.D0
!!$  do iimage=2,nimage
!!$     ! abuse eigval in the next 3 lines...
!!$     eigval=glob%icoords(neb%cstart(iimage):neb%cend(iimage))-&
!!$        glob%icoords(neb%cstart(iimage-1):neb%cend(iimage-1))
!!$     eigval=eigval/sqrt(sum(eigval**2))
!!$     inst_vec=inst_vec+eigval
!!$  end do

  ! normalise
  inst_vec=inst_vec/sqrt(sum(inst_vec**2))
  


  hess_avg=0.D0
  do iimage=1,nimage
    tang=rotmode(neb%cstart(iimage):neb%cend(iimage))
    tang=tang/sqrt(sum(tang**2))
    parallelcont=dot_product(tang,matmul(qts%vhessian(:,:,iimage),tang))
    hess_avg=hess_avg+qts%vhessian(:,:,iimage)
    ! subtract parallel component, i.e. set the respective eigenvalue to zero:
    do ivar=1,neb%varperimage
       do jvar=1,neb%varperimage
          hess_avg(ivar,jvar)=hess_avg(ivar,jvar)+(-parallelcont)*tang(ivar)*tang(jvar)
       end do
    end do
  end do
  hess_avg=hess_avg/dble(nimage)
  call dlf_matrix_diagonalise(varperimage,hess_avg,eigval,eigvec)
  print*,"Square of the stability parameters:"
  ! first a list of stability parameters:
  line=""
  do ivar=1,varperimage
    write(line,'(a,es12.4)') trim(line),eigval(ivar)
  end do
  write(*,'("eigvals: ",a)') trim(line)
!!$  do ivar=1,varperimage
!!$    ! print line with eigenvector:
!!$    line=""
!!$    do jvar=1,varperimage
!!$      write(line,'(a,f8.4)') trim(line),eigvec(jvar,ivar)
!!$    end do
!!$    write(*,'("EV:",i3,es12.4,3x,a,3x,f8.6)') ivar,eigval(ivar),trim(line),(sum(inst_vec*eigvec(:,ivar)))**2
!!$  end do

  sigma=0.D0
  sigmaprime=0.D0
  foundzero=0
  foundrot=0
  call dlf_constants_get("CM_INV_FOR_AMU",CM_INV_FOR_AMu)
  call dlf_constants_get("AMU",amu)
  write(stdout,'("Mode Stab.par.    Wave Number   Projection")')
  line=""
  do ivar=1,varperimage
    proj=(sum(inst_vec*eigvec(:,ivar)))**2
    if(proj>0.5D0) then ! 0.5 is quite critical!
      foundrot=foundrot+1
      svar=1.D0
      if(eigval(ivar)<0.d0) svar=-1.D0 
      write(stdout,'(i3,1x,es11.4,1x,f10.3," cm^-1 ",f6.3," parallel to instanton path")') &
           ivar,svar*sqrt(abs(eigval(ivar)))*beta_hbar,&
           svar*sqrt(amu*abs(eigval(ivar)))*CM_INV_FOR_AMU,proj
      !print*,"Eigval ",ivar," parallel to instanton path"
    else
      if(foundzero<glob%nzero) then
        svar=1.D0
        if(eigval(ivar)<0.d0) svar=-1.D0 
        write(stdout,'(i3,1x,es11.4,1x,f10.3," cm^-1 ",f6.3," ignored (rot/trans)")') &
              ivar,svar*sqrt(abs(eigval(ivar)))*beta_hbar,&
              svar*sqrt(amu*abs(eigval(ivar)))*CM_INV_FOR_AMU,proj
        !print*,"Eigval ",ivar," is ignored (rot/trans)"
        foundzero=foundzero+1
      else
        ! we have a stability parameter
        if(eigval(ivar)<0.D0) then
          !print*,"Eigval ",ivar," is negative, not small, and not parallel to instanton path - SUSPICIOUS!"
          write(stdout,'(i3,1x,es11.4,1x,f10.3," cm^-1 ",f6.3," negative, not small, and &
              &not parallel to instanton path - SUSPICIOUS!")') &
               ivar,-sqrt(-eigval(ivar))*beta_hbar,-sqrt(-amu*eigval(ivar))*CM_INV_FOR_AMU,proj
        else
          !print*,"stability parameter=",sqrt(eigval(ivar))*beta_hbar
          write(stdout,'(i3,1x,es11.4,1x,f10.3," cm^-1 ",f6.3," used")') &
               ivar,sqrt(eigval(ivar))*beta_hbar,sqrt(amu*eigval(ivar))*CM_INV_FOR_AMU,proj
          sigma=sigma+log(2.D0*sinh(sqrt(eigval(ivar))*beta_hbar*0.5D0))
          sigmaprime=sigmaprime+0.5D0*sqrt(eigval(ivar))/tanh(sqrt(eigval(ivar))*beta_hbar)
          write(line,'(a,1x,es15.8)') trim(line), sqrt(eigval(ivar))*beta_hbar
        end if
      end if
    end if
  end do
  if(printl>=2) then
    write(stdout,'("Stability parameters from averaged Hessians: ",a)') trim(line)
    if(foundrot/=1.or.foundzero/=glob%nzero) then
      print*,"Warning: inappropriate number of zero modes found"
    end if
    print*,"Sigma from GJ (JK)=",sigma
    print*,"Sigma/beta",sigma/beta_hbar
    print*,"Sigma'    ",sigmaprime
  end if

  !call dlf_fail("Gelfand_Yaglom_avg")
  call deallocate(eigval)
  call deallocate(eigvec)
  call deallocate(tang)


!!$  allocate(fmat(2*varperimage,2*varperimage))
!!$  allocate(ceigval(2*varperimage))
!!$  allocate(ceigvec(2*varperimage,2*varperimage))
!!$  fmat=0.D0
!!$  fmat(1:varperimage,varperimage+1:2*varperimage)=hess_avg
!!$  do ivar=1,varperimage
!!$    fmat(varperimage+ivar,ivar)=-1.D0
!!$  end do
!!$!  do ivar=1,2*varperimage
!!$!    write(*,'(8f10.5)') fmat(:,ivar)
!!$!  end do
!!$  call c_diagonal_general(2*varperimage,fmat,ceigval,ceigvec)
!!$  do ivar=1,2*varperimage
!!$    print*,"|eval|^2",ivar,abs(ceigval(ivar))**2
!!$    write(*,'(" eval",2f10.5)') ceigval(ivar)
!!$    do jvar=1,2*varperimage
!!$      !print*,"   evec",ceigvec(jvar,ivar)
!!$      write(*,'("   evec",2f10.5)') ceigvec(jvar,ivar)
!!$    end do
!!$  end do

end subroutine stapar_avg_hessian



! read in data from previous instanton runs and calculate microcanonical rate
! constants from them. 
subroutine dlf_microcanonical_rate
  !! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl,printf,pi
  !  use dlf_allocate, only: allocate,deallocate
  use dlf_qts, only: qts,dbg_rot,pf_type,minfreq
  use dlf_allocate, only: allocate,deallocate
  use dlf_constants, only: dlf_constants_init,dlf_constants_get
  implicit none
  real(rk)  :: tstart ! lower end of the temperature range
  real(rk)  :: tend  ! upper end of the temperature range
  integer   :: nbeta ! number of temperatures at which to calculate the rates
  real(rk)  :: mu_bim  ! reduced mass (amu) for calculating phi_rel
  real(rk)  :: phi_rel  ! relative translational p.fu. per unit volume

  real(rk)  :: KBOLTZ_AU,svar
  integer   :: ivar,itemp
  real(rk)  :: second_au
  real(rk)  :: timeunit

  logical   :: chkfilepresent
  character(2048) :: sline!,line
  real(rk)  :: avogadro,hartree,kjmol,echarge,ev,amc,planck,kboltz
  real(rk)  :: zpe_rs,zpe_ts
  integer   :: ios
  real(rk)  :: frequency_factor,amu
  character(128) :: filename
  type(pf_type) :: rs1,rs2,ts
  logical   :: toneat
  ! new variables for micro
  integer :: ntemp,iene,nene,ibeta
  real(rk),allocatable :: beta(:),s_0(:),e_b(:),sigma(:),lnk(:),rotfac(:) ! (0:ntemp)
  real(rk),allocatable :: sigmaprime(:),ene(:) ! (ntemp)
  real(rk) :: zpers,trans_prop,vbeta,betahi,betalow,rate(4),delta
  real(rk) :: qrot,apar,bpar,enemax,weight
  real(rk) :: qrot_quant,qrot_part,qrot_quant_part
  real(rk),allocatable :: trans(:),tene(:) ! (nene)
  real(rk) :: ene_eval,trans_one
  real(rk) :: trans_one_1d(3),trans_prop_1d(3) ! no tunnelling, Bell, Eckart
  real(rk),allocatable :: trans_1D(:,:) ! (3,nene)
  real(rk),allocatable :: trans4(:,:) ! (4,nene)
  ! trans_one_class,trans_class,trans_prop_class can be removed - NO
  real(rk) :: integral
  ! integer 4 or 8:
  integer(8) :: vibex,maxex
  integer(8) :: vibex_
  integer(8), allocatable :: maxexmode(:)
  integer, allocatable :: iexmode(:) ! iexmode must be the same kind in find_omega
!!$  integer :: vibex,maxex
!!$  integer :: vibex_
!!$  integer, allocatable :: maxexmode(:),iexmode(:)
  !
  integer :: max_nene,iene_,ivib,calcmodes,ntemp_stb,inumeric
  logical :: tnumeric(4)
  real(rk) :: factorial,numeric_factor(4)
  real(rk), allocatable :: maxexreal(:)
  real(rk) :: attempt_freq,zpets,eshift,emargin
  real(rk),allocatable :: stability_parameter(:,:) ! (0:nene,ts%nvib)
  logical :: tstab
  integer :: nclass,nclassold
  real(rk) :: vb,alpha,dpar,pclass
  ! Switches
  !logical :: tpofe_one=.true. ! deal with degeneracies outside of p_of_e
  logical :: tstabpar=.false. ! use individual stability parameters, set in mic.in
  logical :: tclass=.false. ! no tunnelling, but quantum vibrations

  if(glob%iam > 0 ) return ! only task zero should do this (task farming does
  ! not make sense for such fast calculations. No
  ! energies and gradients are ever calculated).

  ! some small header/Info
  if(printl>=2) then
    write(stdout,'(a)') "Calculating canonical rate constants from microcanonical ones read as input."

    ! print header with citation information
    write(stdout,'(a)') '%%%%%%%%%%%'
    write(stdout,'(a)') 'Please include this reference which describe the algorithms'
    write(stdout,'(a)') 'to compute microcanonical rate constants and use them to obtain canonical rate constants:'
    write(stdout,'(a)') 'S.R. McConnell, A. Lohle, J. Kastner, J. Chem. Phys. 146, 074105 (2017)'
    write(stdout,'(a)') '    DOI: 10.1063/1.4976129'
    write(stdout,'(a)') 'along with the original references to instanton theory.'
    write(stdout,'(a)') '%%%%%%%%%%%'

    write(stdout,'(a)') "Degeneracies are dealt with outside of the routine p_of_e."
    if(tclass) write(stdout,*) "WARNING: tunnelling is switched off, only classical tranmission probability!"
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
  ! read mic.in 
  !

  ! the file is structured as
  ! line 1: minfreq in cm^-1
  ! line 2: Attempt frequency (as wave number)
  ! line 3: tstart, tend, tsteps
  ! line 4: pclass, maxex, nene, emargin (default if negative)
  filename="mic.in"
  if (glob%ntasks > 1) filename="../"//trim(filename)
  INQUIRE(FILE=filename,EXIST=chkfilepresent)
  if (.not.chkfilepresent) THEN
    filename="mic.auto"   ! if mic.in exists, it is used, otherwise
                          ! mic.auto, written by the ChemShell
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
    READ (sline,iostat=ios,fmt=*) attempt_freq
    if(ios/=0) then
      attempt_freq=-1.D0 ! switch it off
    end if

    READ (28,iostat=ios,fmt=*) tstart,tend,nbeta
    if(ios/=0) then
      print*,"tstart:",tstart
      print*,"tend:",tend
      print*,"nbeta:",nbeta
      call dlf_fail("Error reading T_start T_end Number_of_steps from mic.in")
    end if
    READ (28,iostat=ios,fmt='(a)') sline  ! line 4
    !print*,"sline is",sline
    READ (sline,iostat=ios,fmt=*) pclass, maxex, nene, emargin
    if(ios/=0) then
      emargin=-1.D0
      READ (sline,iostat=ios,fmt=*) pclass, maxex, nene
      if(ios/=0) then
        nene=-1
        READ (sline,iostat=ios,fmt=*) pclass, maxex
        if(ios/=0) then
          maxex=-1
          READ (sline,iostat=ios,fmt=*) pclass
          if(ios/=0) pclass=-1
        end if
      end if
    end if
    !print*,"read:", pclass, maxex, nene, emargin ! remove that line
    CLOSE(28)
  ELSE
    if(printl>=0) then
      write(stdout,*) " Read Error: file mic.in not found"
      write(stdout,*) " The file provides input for rate calculations and should contain the lines:"
      write(stdout,*) " Line 1: minimum wave number"
      write(stdout,*) " Line 2: wave number of attempt frequency"
      write(stdout,*) " Line 3: T_start T_end Number_of_steps ! Temperature"
      write(stdout,*) " Line 4: pclass, maxex, nene, emargin (default if negative)"
    end if
    call dlf_fail("File mic.in missing")
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

  call dlf_constants_get("SECOND_AU",second_au)
  call dlf_constants_get("HARTREE",HARTREE)
  call dlf_constants_get("AVOGADRO",avogadro)
  call dlf_constants_get("ECHARGE",echarge)
  call dlf_constants_get("CM_INV_FOR_AMU",frequency_factor)
  call dlf_constants_get("AMU",amu)
  call dlf_constants_get("KBOLTZ_AU",KBOLTZ_AU)
  kjmol=avogadro*hartree*1.D-3
  ev=hartree/echarge

  ! crossover temperature
  qts%tcross=ts%omega_imag*0.5D0/pi/KBOLTZ_AU

  if(rs2%tused) then
    call dlf_constants_get("AMU",svar)
    ! define the reduced mass
    mu_bim=rs1%mass*rs2%mass/(rs1%mass+rs2%mass)
    if(rs2%tfrozen) mu_bim=rs1%mass
    if(rs1%tfrozen) mu_bim=rs2%mass
    if(rs1%tfrozen.and.rs2%tfrozen) mu_bim=rs1%mass*rs2%mass/(rs1%mass+rs2%mass)
    !if(rs2%nat==1) mu_bim=rs2%mass
    if(glob%irot==1) mu_bim=rs2%mass ! mimic a surface (implicit surface)
    if(printl>=4) write(stdout,'("Reduced mass =",f10.5," amu")') mu_bim
  end if

  if(printl>=2.and.glob%irot==1) then
    if(rs2%tused) then
      write(stdout,'("Implicit surface approach: only rotation of &
          &RS2 considered, reduced mass = mass of RS2")')
    else
      if(printl>=2) write(stdout,'("Implicit surface approach: rotational partition function kept constant.")')
    end if
  end if

  if(.not.rs2%tused) then
    if(attempt_freq>0.D0) then
      if(printl>=2) write(stdout,'("Attempt frequency of ",f8.3," cm^-1 is read from input and used")') &
          attempt_freq
      attempt_freq=attempt_freq/frequency_factor/sqrt(amu) ! now it should be in atomic units
    else
      attempt_freq=rs1%omega(1)
      if(printl>=2) write(stdout,'("Attempt frequency of ",f8.3," cm^-1 is used (first RS mode)")') &
          attempt_freq*frequency_factor*sqrt(amu)
    end if
  else
    attempt_freq=-1.D0 ! should never be used
  end if

  ! write some header
  if(printl>=2) then
    write(stdout,*)
    write(stdout,*) "Details of the calculation of P(E)"
  end if

  ! read in the file with different data

  filename="microcanonic.dat"
  if (glob%ntasks > 1) filename="../"//trim(filename)
  INQUIRE(FILE=filename,EXIST=chkfilepresent)
  if(.not.chkfilepresent) THEN
    write(stdout,*) "File ",trim(filename)," not found."
    call dlf_fail("Input file for microcanonical rate calculations missing")
  end if
  OPEN(28,FILE=filename,STATUS='old',ACTION='read')
  ! find out how many lines with rate data there are
  ntemp=0
  do
    READ (28,FMT='(a)',iostat=ios) sline
    if(ios/=0) exit
    !print*,index(sline,'#')
    if(index(sline,'#') > 0) sline=sline(1:index(sline,'#')-1)
    if(trim(sline)/="") ntemp=ntemp+1 ! lines with any non-commented information
    !print*,trim(sline)
  end do
  ntemp=ntemp-1 ! one line for T/F for stability parameters
  !print*,"ntemp",ntemp
  allocate(beta(0:ntemp))
  allocate(s_0(0:ntemp))
  allocate(e_b(0:ntemp))
  allocate(sigma(0:ntemp))
  sigma=0.D0
  allocate(lnk(0:ntemp))
  allocate(rotfac(0:ntemp))
  allocate(stability_parameter(0:ntemp,ts%nvib))
  stability_parameter=0.D0
  ! read in the data
  rewind(28)
  itemp=0
  ntemp_stb=ntemp
  do
    READ (28,FMT='(a)',iostat=ios) sline
    if(ios/=0) exit
    if(index(sline,'#') > 0) sline=sline(1:index(sline,'#')-1)
    if(trim(sline)/="") then
      if(itemp==0) then
        ! read T/F for tstabpar
        read(sline,fmt=*,iostat=ios) tstabpar
        if(ios/=0) then
          write(stdout,*) "Error reading file ",trim(filename)
          write(stdout,'("The fist line should contain T or F to indicate if stability parameters are to be read.")')
          call dlf_fail("Error reading input file for microcanonical rate constants.")
        end if
        if(printl>=2) then
          if(tstabpar) then
            write(stdout,'(a)') "Individual stability parameters are used."
          else
            write(stdout,'(a)') "Stability is estimated via sigma."
          end if
        end if
        itemp=1
        cycle
      end if
      !
      ! THIS IS THE LIST OF INPUT DATA AS CURRENTLY USED
      !
      if(tstabpar) then
        read(sline,fmt=*,iostat=ios) &
             e_b(itemp),beta(itemp),s_0(itemp),lnk(itemp),rotfac(itemp),stability_parameter(itemp,:)
        ! stability parameters are expected as omega_i, not as u_i!
        if(ios/=0) then  
          read(sline,fmt=*,iostat=ios) &
             e_b(itemp),beta(itemp),s_0(itemp),lnk(itemp),rotfac(itemp)
          stability_parameter(itemp,:)=0.D0
          ntemp_stb=min(ntemp_stb,itemp-1)
        end if
      else 
        read(sline,fmt=*,iostat=ios) &
             e_b(itemp),beta(itemp),s_0(itemp),lnk(itemp),rotfac(itemp),sigma(itemp)
      end if 
      ! e_b is expected relative to the RS!
      itemp=itemp+1
      if(ios/=0) then
        write(stdout,*) "Error reading file ",trim(filename)
        write(stdout,'("Line in which the error occurred:")')
        write(stdout,'(a)') trim(sline)
        if(tstabpar) then
          write(stdout,'("Individual stability parameters are used - should that be the case?")')
        end if
        call dlf_fail("Error reading input file for microcanonical rate constants.")
      end if
    end if
  end do
  close(28)

  if(itemp/=ntemp+1) print*,"Something is wrong when reading file ",trim(filename)

!!$  do itemp=1,ntemp
!!$     print*,itemp,beta(itemp),sigma(itemp),s_0(itemp),e_b(itemp),lnk(itemp)
!!$  end do

  if(beta(2)<beta(1)) then
    print*,"Error in input file: temperatures are supposed to decrease."
    call dlf_fail("Error")
  end if

  ! calculate properties at the transition state (itemp=0)
  beta(0)=1.D0/(qts%tcross*kboltz_au)
  sigma(0)=0.D0
  zpets=0.D0
  do ivar=1,ts%nvib
    sigma(0) = sigma(0) + log(2.D0*sinh(0.5D0*ts%omega(ivar)*beta(0))) 
    zpets=zpets+0.5D0*ts%omega(ivar)
  end do
  s_0(0)=0.D0
  e_b(0)=ts%ene-rs1%ene-rs2%ene
  stability_parameter(0,:)=ts%omega(:)!*beta(0)
  lnk(0)=0.D0 ! should never be used!
  rotfac(0)=rotfac(1) ! might be turned into input parameter
  
  ! if classical
  if(tclass) then
     do itemp=1,ntemp
        sigma(itemp)=0.D0
        do ivar=1,ts%nvib
           sigma(itemp) = sigma(itemp) + log(2.D0*sinh(0.5D0*ts%omega(ivar)*beta(itemp))) 
        end do
        stability_parameter(itemp,:)=ts%omega(:)
     end do
  end if
  
  ! derived variables
  allocate(sigmaprime(0:ntemp))
  allocate(ene(0:ntemp))

  do itemp=0,ntemp
    sigmaprime(itemp)=sigma(itemp)/beta(itemp)
    ene(itemp)=e_b(itemp)+sigmaprime(itemp) ! ene() is only used for stabpar=.false.
  end do

  if(tstabpar) then
    ! calculate sigma from stability parameters - even though it should not be used anywhere
    sigma=0.D0
    do itemp=0,ntemp_stb
      sigma(itemp)=sum(log(2.D0*sinh(stability_parameter(itemp,:)*0.5D0*beta(itemp))))
      ene(itemp)=e_b(itemp)+sigma(itemp)/beta(itemp) ! ene() is only used for stabpar=.false.
    end do
    ! print stability parameters
    if(printf>=4) then
      open(unit=31,file='stabpar.out')
      write(31,*) "# E-E_n=E_b   omega_i"
      do itemp=0,ntemp_stb
        sline=""
        do ivib=1,ts%nvib
          write(sline,'(a,es11.3)') trim(sline),stability_parameter(itemp,ivib)
        end do
        write(31,'(f10.5,a)') e_b(itemp),trim(sline)
      end do
      close(31)
   end if
   if(printl>=4.and.ntemp_stb<ntemp) then
     write(stdout,'("Stability parameters are read in for the first ",i4," entries")') ntemp_stb
   end if
  end if

  ! general output of read-in data
  if(printl>=4) then
    write(stdout,'(a)') "Input data:"
    write(stdout,'(a)') "  temperature  beta      energy     E_b        sigma      S_0         ln(k)     rotation factor"
    do itemp=0,ntemp
      !print*,itemp,beta(itemp),sigma(itemp),s_0(itemp),e_b(itemp),lnk(itemp)
       write(stdout,'(i3,f10.2,f10.3,4es11.3,2f10.3)') itemp,1.D0/beta(itemp)/kboltz_au,&
            beta(itemp),ene(itemp),e_b(itemp),&
          sigma(itemp),s_0(itemp),lnk(itemp),rotfac(itemp)
    end do
  end if

!!$  ! catch unimolecular cases - what to do in these cases??
!!$  if(.not.rs2%tused) then
!!$    print*,"Turning a unimolecular case into a pseudo-bimolecular one."
!!$    do itemp=0,ntemp
!!$      lnk(itemp)=lnk(itemp)-log(2.D0*sinh(beta(itemp)*rs1%omega(1)*0.5D0))+0.5D0*log(beta(itemp))
!!$      !print*,itemp,beta(itemp),-log(2.D0*sinh(beta(itemp)*rs1%omega(1)*0.5D0)),-beta(itemp)*rs1%omega(1)*0.5D0
!!$    end do
!!$    rs1%omega(1)=0.D0
!!$  end if

  ! that must be below "catch unimolecular ..."
  zpers=0.D0
  do ivar=1,rs1%nvib
    zpers=zpers + 0.5D0*rs1%omega(ivar)
  end do
  do ivar=1,rs2%nvib
    zpers=zpers + 0.5D0*rs2%omega(ivar)
  end do

  ! make sure the rotational zero point energy is added if the
  ! coefficients for even and odd J completely forbid J=0.
  if(rs1%coeff(1)==0.D0) then
    svar=0.D0
    if(minval(abs(rs1%moi))>10.D0) then
      write(*,"('Warning: Molecule non-linear, but J=0 forbidden - not implemented!')")
    end if
    svar=dble(2)*0.5D0/maxval(rs1%moi) ! rotational level for J=0
    if(printl>=2) then
      write(*,"('Zero point rotational energy RS1             ',es11.3)") svar
    end if
    zpers=zpers+svar
  end if
  if(rs2%coeff(1)==0.D0) then
    svar=0.D0
    if(minval(abs(rs2%moi))>10.D0) then
      write(*,"('Warning: Molecule non-linear, but J=0 forbidden - not implemented!')")
    end if
    svar=dble(2)*0.5D0/maxval(rs2%moi) ! rotational level for J=0
    if(printl>=2) then
      write(*,"('Zero point rotational energy RS2             ',es11.3)") svar
    end if
    zpers=zpers+svar
  end if
  
  if(printl>=2) then
    write(*,"('Zero point energy of the reactant(s):        ',es11.3)") zpers
    write(*,"('Zero point energy of the TS:                 ',es11.3)") zpets
    write(*,"('sigmaprime(TS)                               ',es11.3)") sigmaprime(0)
    write(*,"('Potential energy barrier                     ',es11.3)") e_b(0)
    write(*,"('Vibrationally adiabatic barrier              ',es11.3)") e_b(0)+zpets-zpers
    !print*,"Vibrationally adiabatic barrier",ene(0)-zpers
    !print*,"Zero point energy of the reactant:",zpers
    !print*,"emax=E_TS+ZPE",ene(0)
    !print*,"E_TS",e_b(0)
  end if

  if(ene(0)<zpers) then
    call dlf_fail("Vibrational adiabatic barrier is negative. No rate calculation possible")
  end if
  
  ! set default parameters for the calculation of P(E)
  if(maxex<0) maxex=10
  if(pclass<0.D0) pclass=30.D0 ! for P(E)>pclass, individual modes may be approximated classically
  if(emargin<0.D0) emargin= 10.D0/beta(ntemp) !makes sure that enough excited states are
                             !included even for the lowest energies

  ! enemax: upper bound of numeric integral
  if(rs2%tused) then
    ! bimolecular
    !enemax=ene(0)+5.D0*min(rs1%omega(1),rs2%omega(1)) 
    enemax=dble(maxex-1)*ts%omega(ts%nvib)+zpers-emargin ! that one should be used with stabpar
  else
    enemax=ene(0)+5.D0*rs1%omega(1)
  end if
  !print*,"enemax",enemax


!!$  ! print TS omega
!!$  do ivar=1,ts%nvib
!!$    write(*,'("omega(",i3,")=",es10.4)') ivar,ts%omega(ivar)
!!$  end do
!!$  print*,"Barrier frequency",ts%omega_imag

  ! first print interpolated P(E) to a file
  if(nene<1) nene=1001 ! please use odd number (Simpson rule)
  if(mod(nene,2)==0) nene=nene+1
  max_nene=nene
  call allocate(trans,nene)
  call allocate(trans_1d,3,nene)
  call allocate(tene,nene)
  if(printl>=2) then
    write(*,"('Number of vibrational modes without TS mode  ',i11)") ts%nvib
    write(*,"('Number of energies to evaluate P(E)          ',i11,' (nene)')") nene
    write(*,"('Lowest energy to evaluate P(E)               ',es11.3,' (zpers)')") &
        zpers
    write(*,"('Highest energy to evaluate P(E)              ',es11.3,' (enemax)')") &
        enemax
    write(*,"('Contributions taken into account for          ',es11.3,' Hartree (emargin)')") &
        emargin
    write(*,"('Maximum number of vibrational excitations per mode',i6,' (maxex)')") maxex
    write(*,"('P above which modes may be treated classically  ',f8.2,' (pclass)')") pclass
  end if

  ! unimolecular reactions
  if(.not.rs2%tused) then
    call allocate(maxexreal,rs1%nvib)
    call allocate(maxexmode,rs1%nvib)
    call allocate(iexmode,rs1%nvib)
    do ivar=1,rs1%nvib
      maxexreal(ivar)=1.D0/rs1%omega(ivar)
    end do
    print*,"maxexmode",maxexreal
    svar=dble(nene)/product(maxexreal)
    print*,"svar vorher",svar,rs1%nvib
    svar=svar**(1.D0/dble(rs1%nvib))
    print*,"svar",svar
    maxexreal=maxexreal*svar
    print*,"maxexreal",maxexreal
    print*,"maxexreal*omega",maxexreal*rs1%omega
    print*,"product(maxexreal)",product(maxexreal)
    maxexmode=max(nint(maxexreal),1)
    print*,"maxexmode",maxexmode
    print*,"Maximum excitations of RS:",maxexmode
    max_nene=int(min(product(maxexmode),nene),kind=kind(max_nene))
    print*,"total number of excitations covered",max_nene
!!$    do iene=1,nene
!!$      iene_=iene-1
!!$      tene(iene)=0.D0
!!$      do ivar=1,rs1%nvib
!!$        iexmode(ivar)=mod(iene_,maxexmode(ivar))
!!$        iene_=iene_/maxexmode(ivar)
!!$        tene(iene)=tene(iene)+rs1%omega(ivar)*(0.5D0+dble(iexmode(ivar)))
!!$      end do
!!$      !print*,iexmode
!!$      if(iene==max_nene) exit
!!$    end do
    ! excitations only along the reaction coordinate:
    do iene=1,nene
      tene(iene)=zpers+attempt_freq*(dble(iene-1))
    end do
    call deallocate(maxexreal)
    call deallocate(maxexmode)
    call deallocate(iexmode)
  end if

  open(unit=13,file='p_of_e')
  write(13,'(a)') "# P(E) for different approaches."
  write(13,'(a)') "# E is given as potential energy relative to the reactant."
  write(13,'(a)') "# E              Instanton      no tunnelling  Bell           Eckart"
  call allocate(maxexmode,ts%nvib)
  call allocate(iexmode,ts%nvib)
  nclassold=0
  nclass=0

  !
  ! Loop over energies
  !
  do iene=1,nene
    if(iene>max_nene) exit ! unimolecular

    trans(iene)=trans_prop

    tene(iene)=zpers+dble(iene-1)/dble(nene-1)*(enemax-zpers) 

    ! use analytic classical solution?
    ! that should be done for the last ~2 values of iene for validation purposes
    if(iene>=nene-2) then 
      ! analytic classical solution, P=1
      factorial=1.D0
      do ivar=1,ts%nvib
        factorial=factorial*dble(ivar)
      end do
      trans_prop=(tene(iene)-e_b(0))**ts%nvib/factorial/product(ts%omega(1:ts%nvib))*rotfac(0)
      trans_prop_1d(:)=(tene(iene)-e_b(0))**ts%nvib/factorial/product(ts%omega(1:ts%nvib))*rotfac(0)

      trans(iene)=trans_prop
      trans_1d(:,iene)=trans_prop_1d
      write(13,'(5es15.7)') tene(iene),max(trans_prop,1.D-20),max(trans_prop_1d,1.D-20)

      cycle
    end if

    ! find out how many vibrational excitations at the TS perpendicular
    ! to the instanton have to be taken into account
    !
    ! sum over states for TS and below
    ! excited states are included even for the lowest energies
    if(tene(iene)-zpers>e_b(ntemp/2).or.(.not.tstabpar)) then 
      !print*,"Using TS omega"
      do ivib=1,ts%nvib
        maxexmode(ivib)=1+int(-(zpers-tene(iene)-emargin)/ts%omega(ivib))
      end do
    else
      !print*,"Using low-E omega"
      do ivib=1,ts%nvib
        maxexmode(ivib)=1+int(-(zpers-tene(iene)-emargin)/&
            stability_parameter(ntemp_stb,ivib))
      end do
    end if

    ! find out which modes to treat classically
    if(iene>1) then
      if(trans(iene-1)>pclass.and.minval(trans_1d(:,iene-1))>pclass) then
        do ivib=max(nclass,1),ts%nvib
          if(maxexmode(ivib)>maxex) nclass=ivib 
        end do
      end if
    end if
    if(nclass/=nclassold.and.printl>=4) then
      if(nclassold==0.and.maxval(maxexmode)>maxex) then
        write(stdout,'("Maximum desirable excitation before classical &
            &treatment:",i4)') maxval(maxexmode)
      end if
      write(stdout,'(i3," mode(s) treated classically &
          &from E=",es10.3, " upwards")') nclass,tene(iene)
    end if
    nclassold=nclass
    maxexmode(1:nclass)=1

    ! The following should not have an effect, since those modes are
    ! treated classically (it has an effect below P=30 or below threshold):
    do ivib=1,ts%nvib
      if(maxexmode(ivib)>maxex) maxexmode(ivib)=maxex 
    end do

    ! report on the number of vibrational excitations used
    if(printl>=4) then
      if(iene==1) then
        sline="Lowest energy: number of excitations for each mode:  "
        do ivib=1,ts%nvib
          write(sline,'(a,i3)') trim(sline),maxexmode(ivib)
        end do
        write(stdout,'(a)') trim(sline)
      end if
      if(mod(iene,100)==0) then
        sline="Energy:"
        write(sline,'(a,es10.3," excitations for each mode: ")') &
            trim(sline),tene(iene)
        do ivib=1,nclass
          sline=trim(sline)//" --"
        end do
        do ivib=nclass+1,ts%nvib
          write(sline,'(a,i3)') trim(sline),maxexmode(ivib)
        end do
        write(sline,'(a," = ",i14)') trim(sline),product(maxexmode(nclass+1:))
      end if
    end if

    trans_prop=0.D0
    trans_prop_1d=0.D0
    vibex=0
    calcmodes=0

    !
    ! loop over vibrational excitations
    !
    do !vibex=0,product(maxexmode)-1 !vibex=0,maxex**ts%nvib-1
      ene_eval=tene(iene)
      ! get the vibrational excitation modes
      vibex_=vibex
      do ivar=1,ts%nvib
        iexmode(ivar)=int(mod(vibex_,maxexmode(ivar)),kind=kind(ivar))
        vibex_=vibex_/maxexmode(ivar)
      end do

      ! hardcode one excitation level:
      !iexmode(:)=0
      !iexmode(5)=1

      ! make sure no excitations of the classical modes are summed over
      do ivib=1,nclass
        if(iexmode(ivib)/=0) then
          ! actually, that can't happen any more, since
          ! maxexmode(1:nclass)=1
          print*,"Error: iexmode()/=0!, it is",iexmode(ivib),ivib
          print*,"iene",iene,vibex
          print*,"iexmode",iexmode
          print*,"maxexmode",maxexmode
          call dlf_fail("Error in classical/quantum separation of modes")
        end if
      end do

      ! check for very large E_n at high energies
      if(iene>1) then
        if(trans(iene-1)>10.D0) then ! that value of 10 could be changed ...
          ! only contributions with trans_one close to 1 will matter:
          ene_eval=tene(iene)-sum(ts%omega(nclass+1:)*(0.5D0+dble(iexmode(nclass+1:))))
          if(ene_eval<e_b(ntemp/2)) then
            vibex=vibex+(maxexmode(nclass+1)-iexmode(nclass+1))&
                *product(maxexmode(1:nclass))
            if(vibex>product(maxexmode)-1) exit
            cycle
          end if
        end if
      end if

      ! Bell, parabolic barrier
      ! for Bell, V1 and V2 are equal
      ene_eval=tene(iene)-sum(ts%omega(nclass+1:)*(0.5D0+dble(iexmode(nclass+1:))))
      trans_one_1d(2)=1.D0/(1.D0+exp(2.D0*pi*(e_b(0)-ene_eval)/ts%omega_imag))

      ! symmetric Eckart barrier:
      !ene_eval=tene(iene)-sum(ts%omega(:)*(0.5D0+dble(iexmode(:)))) ! full TS
      ene_eval=tene(iene)-zpers-sum(ts%omega(:)*(dble(iexmode(:)))) ! correct: V2 and V3
      if(ene_eval<0.D0) then
        trans_one_1d(3)=0.D0
      else
        !vb=4.D0*e_b(0) ! potential energy barrier, V1 and V3
        vb=4.D0*(e_b(0)+zpets-zpers) ! VAE,correct: V2 and V4
        alpha=ts%omega_imag*sqrt(8.D0/vb)
        apar=2.D0*pi*sqrt(2.D0*ene_eval)/alpha
        dpar=2.D0*pi*sqrt( 2.D0*vb - alpha**2 * 0.25D0 ) / alpha 
        if(dpar>500.D0.and.2.D0*apar>500.D0) then
          ! catch a inf/inf case
          trans_one_1d(3)=(1.D0-exp(-2.D0*apar))/(1.D0+exp(dpar-2.D0*apar))
        else
          if(2.D0*Vb - alpha**2 * 0.25D0 > 0.D0) then
            trans_one_1d(3)=(cosh(2.D0*apar)-1.D0)/(cosh(2.D0*apar)+cosh(dpar))
          else
            ! cosh(ix)=cos(x) for real-valued x
            dpar=2.D0*pi*sqrt( -2.D0*Vb + alpha**2 * 0.25D0 ) / alpha     
            trans_one_1d(3)=(cosh(2.D0*apar)-1.D0)/(cosh(2.D0*apar)+cos(dpar))
          end if
        end if
      end if

      ! classical P(E)
      ene_eval=tene(iene)-sum(ts%omega(:)*(0.5D0+dble(iexmode(:)))) ! full TS
      if(ene_eval<e_b(0)) then 
        trans_one_1d(1)=0.D0
      else
        trans_one_1d(1)=1.D0
      end if

      ! instanton
      if(tstabpar) then
        ! get the energy and the transmission coefficient 
        call find_omega(ntemp,ts%nvib,stability_parameter,e_b,s_0,ts,&
            tene(iene),iexmode,tstab,svar,tclass,ntemp_stb,trans_one)
        ! that find_omega could be replaced by Eckart or Bell
      else
        ! sigma rather than stability parameters:
        ene_eval=tene(iene)-sum(ts%omega(nclass+1:)*(dble(iexmode(nclass+1:))))
        call p_of_e_one(ene_eval,ntemp,ene,s_0,zpers-emargin,e_b(0),&
            ts%omega_imag,sigma,beta,rotfac,ts,tclass,trans_one)
      end if

      calcmodes=calcmodes+1

      ! now add the classical contribution
      if(nclass>0) then
        ene_eval=tene(iene)-e_b(0)-sum(ts%omega(nclass+1:)*&
            (0.5D0+dble(iexmode(nclass+1:))))
!!$            ! mod:
!!$            ene_eval=tene(iene)-e_b(0)-sum(ts%omega(1:nclass))*0.5D0&
!!$                 -sum(ts%omega(nclass+1:)*(0.5D0+dble(iexmode(nclass+1:))))
        if(ene_eval > 0.D0) then
          factorial=1.D0
          do ivar=1,nclass
            factorial=factorial*dble(ivar)
          end do
          trans_one=trans_one*ene_eval**nclass/product(ts%omega(1:nclass))/factorial
          trans_one_1d=trans_one_1d*ene_eval**nclass/product(ts%omega(1:nclass))/factorial
        else
          trans_one=0.D0
          trans_one_1d=0.D0
        end if
      end if

      trans_prop=trans_prop+trans_one
      trans_prop_1d=trans_prop_1d+trans_one_1d

      ! is trans_one<0.9D0 appropriate here, especially for nclass>0?
      if(trans_one/trans_prop<1.D-5.and.trans_one<0.9D0) then 
        ! we can skip at least until the next iex(1)
        vibex=vibex+(maxexmode(nclass+1)-1-iexmode(nclass+1))&
            *product(maxexmode(1:nclass))
      end if

      ! one can probably skip more for sigma
      !if((.not.tstabpar).and.

      vibex=vibex+product(maxexmode(1:nclass))
      if(vibex>product(maxexmode)-1) exit

      ! hardcode for excitation level
      !exit

    end do ! vibex

    if(printl>=4.and.mod(iene,100)==0) then
      write(stdout,'(a," vs. ",i10)') trim(sline),calcmodes
      !write(stdout,'(10x,"Number of actually calculated modes: ",i10)') calcmodes
    end if

    trans(iene)=trans_prop
    trans_1d(:,iene)=trans_prop_1d

    write(13,'(5es15.7)') tene(iene),max(trans_prop,1.D-20),max(trans_prop_1d,1.D-20)
  end do ! iene=1,nene

  ! write 1/(1+exp(S_0)) into the file p_of_e as well
  write(13,*)
  do itemp=0,ntemp
    if(tstabpar) then
      if(itemp>ntemp_stb) cycle
      write(13,*) e_b(itemp)+sigma(itemp)/beta(itemp),1.D0/(1.D0+exp(s_0(itemp)))
    else
      write(13,*) ene(itemp),1.D0/(1.D0+exp(s_0(itemp)))
    end if
  end do

  close(13) ! p_of_e

  call deallocate(maxexmode)
  call deallocate(iexmode)
  call allocate(trans4,4,nene)
  trans4(1,:)=trans ! instanton
  trans4(2:4,:)=trans_1d ! no-tunnel, Bell, Eckart
  call deallocate(trans)
  call deallocate(trans_1d)

  !
  ! now integrate over E (equation 3) to arrive at k(T)
  !
  if(printl>=2) then
    write(stdout,*)
    write(stdout,'(a)') "Printing k(T) to file rate_from_microcan.dat."
    write(stdout,'("Rotational partition functions were calculated for&
        & the TS structure using the classical rigid rotor.")')
    write(stdout,'("To obtain results for the quantum rigid rotor, add &
        & values from the column `Q_rot class/quant`.")')

  end if
  open(unit=13,file='rate_from_microcan.dat')
  write(13,*) "# log_10 of rate constants in cm^3/s"
  write(13,'(a)') " #1000/T [K]  instanton      no tunnelling  Bell        &
      &   Eckart        Q_rot class/quant"
  !open(unit=130,file='integrand.dat')
  betahi=1.D0/(tstart*kboltz_au)
  betalow=1.D0/(tend*kboltz_au)
  delta=1.D0/dble(nene-1)*(enemax-zpers)

  !  delta=1.D0/dble(nene-1)*2.D0*(ene(0)-zpers) ! WRONG
  ! loop over beta
  do ibeta=1,nbeta
    vbeta=betalow+dble(ibeta-1)/dble(nbeta-1)*(betahi-betalow)
    !print*,ibeta,vbeta

    !write(130,*)
    !write(130,'("#beta=",f10.3)') vbeta

    if(rs2%tused) then
      ! integrate over E
      rate=0.D0
      tnumeric=.false.
      numeric_factor=1.D0
      do iene=1,nene
        if(iene==1.or.iene==nene) then ! weights of Simpson's rule
          weight=1.D0/3.D0
        else
          if(mod(iene,2)==1) then
            weight=2.D0/3.D0
          else
            weight=4.D0/3.D0
          end if
        end if
        if(iene==1) then ! was +1
          ! tnumeric:
          ! -) improves stability of adding contributions to the integral
          ! -) 2*sinh(beta*omega/2) -> exp(beta*omega/2) if beta*omega/2 > 5
          do inumeric=1,4
            if(abs(delta * exp(-vbeta*tene(iene)) * trans4(inumeric,iene) * weight)<1.D-250) then
              tnumeric(inumeric)=.true.
              numeric_factor(inumeric)=vbeta*tene(iene)
              !  print*,ibeta,inumeric,"Numeric=TRUE, factor=",numeric_factor(inumeric)
              !else
              !  print*,ibeta,inumeric,"Numeric=false. nonzero=",delta * exp(-vbeta*tene(iene)) * &
              !        trans4(inumeric,iene) * weight
            end if
          end do
        end if
        do inumeric=1,4
          if(trans4(inumeric,iene)>0.D0) then ! trans may be exactly 0 for classical, Eckart, ...
            if(tnumeric(inumeric)) then
              rate(inumeric)=rate(inumeric) + exp(log(delta) -vbeta*tene(iene)+&
                  log(trans4(inumeric,iene)*weight)+numeric_factor(inumeric))
              !write(130,*) ene_in,exp(log(delta) -vbeta*ene_in+log(trans(iene))+numeric_factor)
            else
              rate(inumeric)=rate(inumeric) + delta * exp(-vbeta*tene(iene)) * &
                  trans4(inumeric,iene) * weight
            end if
          end if
        end do
        !write(130,*) ene_in,delta * exp(-vbeta*ene_in) * trans_prop
      end do !iene

      !close(130)
      !call dlf_fail("print integrand")

      ! add the rest of the integral (from enemax to infinity)
      if(ts%nvib==0) then
        rate=rate+exp(-vbeta*enemax)/vbeta
      else
        integral=0.D0
        factorial=1.D0
        do ivar=ts%nvib+1,1,-1
          if(ivar<ts%nvib) factorial=factorial*dble(ts%nvib+1-ivar)
          !print*,"D=",nvib+1," i=",ivar," (D-i)!=",factorial
          integral=integral+(enemax-e_b(0))**(ts%nvib+1-ivar)/vbeta**ivar/factorial
        end do
        integral=integral*exp(-vbeta*enemax)/product(ts%omega(1:ts%nvib))
        rate=rate+integral
      end if
      rate=rate/2.D0/pi

      !print*,"flux",rate
      do inumeric=1,4
        if(.not.tnumeric(inumeric)) then
          ! now divide by the RS partition function:
          do ivar=1,rs1%nvib
            ! the if-statement is only to catch pseudo-bimolecular cases
            if(rs1%omega(ivar)>0.D0) rate(inumeric)=rate(inumeric)*2.D0*sinh(0.5D0*rs1%omega(ivar)*vbeta)
          end do
          do ivar=1,rs2%nvib
            rate(inumeric)=rate(inumeric)*2.D0*sinh(0.5D0*rs2%omega(ivar)*vbeta)
          end do
        else
          rate(inumeric)=log(rate(inumeric))-numeric_factor(inumeric)
          do ivar=1,rs1%nvib
            ! the if-statement is only to catch pseudo-bimolecular cases
            if(rs1%omega(ivar)>0.D0) then
              if(0.5D0*rs1%omega(ivar)*vbeta>5.D0) then
                rate(inumeric)=rate(inumeric)+(0.5D0*rs1%omega(ivar)*vbeta)
              else
                rate(inumeric)=rate(inumeric)+log(2.D0*sinh(0.5D0*rs1%omega(ivar)*vbeta))
              end if
            end if
          end do
          do ivar=1,rs2%nvib
            if(0.5D0*rs2%omega(ivar)*vbeta>5.D0) then
              rate(inumeric)=rate(inumeric)+(0.5D0*rs2%omega(ivar)*vbeta)
            else
              rate(inumeric)=rate(inumeric)+log(2.D0*sinh(0.5D0*rs2%omega(ivar)*vbeta))
            end if
          end do
          rate(inumeric)=exp(rate(inumeric))
        end if ! (not.tnumeric(inumeric))
      END do ! inumeric
    else ! above: bimolecular, below unimolecular
      rate=0.D0
      do iene=1,max_nene
        rate(1)=rate(1) + exp(-vbeta*tene(iene)) * trans4(1,iene)
      end do
      svar=0.D0
      do iene=1,max_nene
        svar=svar + exp(-vbeta*tene(iene))
      end do
      rate=rate*attempt_freq/2.D0/pi/svar ! assuming this is the attempt mode!
    end if ! rs2%tused
    !print*,"rate",rate
    ! translational
    if(rs2%tused) then
      call dlf_constants_get("AMU",svar)
      phi_rel=sqrt((mu_bim*svar/2.D0/pi/vbeta)**3)
      rate=rate/phi_rel
    else
      ! pseudo-bimolecular case for the time being
      !rate=rate*sqrt(vbeta)
    end if
    !print*,"flux * Qvib*Qtrans",rate

    !
    ! calculate the rotational partition function
    !
    ! use the one from the TS for the instanton at present. That
    ! may be too approximative!
    qrot=1.D0
    qrot_quant=1.D0
    ! do nothing for atoms (their rotational partition function is unity)
    if(glob%irot==0) then
      if(rs1%nmoi>1) then
        !print*,"rotpart rs1",rs1%coeff
        call rotational_partition_function_calc(rs1%moi,vbeta,rs1%coeff,&
            qrot_part,qrot_quant_part)
        qrot=qrot*qrot_part
        qrot_quant=qrot_quant*qrot_quant_part
      end if
      if(ts%nmoi>1) then
        call rotational_partition_function_calc(ts%moi,vbeta,ts%coeff,&
            qrot_part,qrot_quant_part)
        qrot=qrot/qrot_part
        qrot_quant=qrot_quant/qrot_quant_part
      end if
    end if
    if(rs2%tused.and.rs2%nmoi>1) then
      call rotational_partition_function_calc(rs2%moi,vbeta,rs2%coeff,&
          qrot_part,qrot_quant_part)
      qrot=qrot*qrot_part
      qrot_quant=qrot_quant*qrot_quant_part
    end if

    rate=rate/qrot ! divide because of Q_TS/Q_RS
    !print*,"flux * Qvib*Qtrans*Qrot",rate

    !write(13,*) vbeta,rate ! atomic units
    ! "SI" units (well, rate in cm3/sec)
    call dlf_constants_get("ANG_AU",svar)
    svar=log(svar*1.D-10) ! ln of Bohr in meters
    if(rs2%tused) then
      ! 1000/K vs. log_10(rate in cm3/sec)
      write(13,'(f12.8,5es15.7)') vbeta*kboltz_au*1000.D0,&
          (log(rate)+log(second_au)+log(1.D6)+3.D0*svar)/log(10.D0), &
          log(qrot/qrot_quant)/log(10.D0)
    else
      write(13,*) vbeta*kboltz_au*1000.D0,&
          (log(rate(1))+log(second_au))/log(10.D0) 
    end if
  end do ! ibeta
  !close(130)

  ! print lnk into that file as well:
  write(13,*)
  call dlf_constants_get("ANG_AU",svar)
  svar=log(svar*1.D-10) ! ln of Bohr in meters
  do itemp=1,ntemp
    !write(13,*) beta(itemp),exp(lnk(itemp)) ! atomic units
    ! "SI" units (well, rate in cm3/sec)
    if(rs2%tused) then
      ! 1000/K vs. log_10(rate in cm3/sec)
      write(13,*) beta(itemp)*kboltz_au*1000.D0,&
          (lnk(itemp)+log(second_au)+log(1.D6)+3.D0*svar)/log(10.D0)
    else
      write(13,*) beta(itemp)*kboltz_au*1000.D0,&
          (lnk(itemp)+log(second_au))/log(10.D0)
    end if
  end do

  close(13)  

  call deallocate(trans4)
  call deallocate(tene)
  deallocate(sigmaprime)
  deallocate(ene)

  deallocate(beta)
  deallocate(s_0)
  deallocate(e_b)
  deallocate(sigma)
  deallocate(lnk)

end subroutine dlf_microcanonical_rate

! calculate/interpolate the energy dependent transmission probability
! for non-degenerate input. Deal with degeneracy outside.
subroutine p_of_e_one(ene_in,ntemp,ene,s_0,zpers,ets_,omegats,sigma,beta,rotfac,ts,tclass,trans_prop)
  use dlf_parameter_module, only: rk
  use dlf_qts, only: pf_type
  use dlf_global, only: pi
  implicit none
  real(rk), intent(in) :: ene_in ! =E-E_{vib,n} from Eq. 20 (ene_eval in calling routine)
  integer, intent(in) :: ntemp
  real(rk), intent(in) :: ene(0:ntemp) ! = E_b+sigmaprime
  real(rk), intent(in) :: s_0(0:ntemp),zpers
  real(rk), intent(in) :: ets_,omegats ! ets_ is e_b(0)
  real(rk), intent(in) :: sigma(0:ntemp),beta(0:ntemp)
  real(rk), intent(in) :: rotfac(0:ntemp)
  type(pf_type), intent(in) :: ts
  logical, intent(in) :: tclass
  real(rk), intent(out) :: trans_prop ! transmission probability
  integer :: itemp
  real(rk) :: frac,s_0_interp,sigma_interp,beta_interp
  ! vibrational excitations
  integer :: vibex,ivar,factorial,maxex,iex
  real(rk) :: ene_eval,ets,rotfac_interp,vb,alpha,apar,dpar

  trans_prop=1.D0

  if(ene_in<zpers) then
    trans_prop=0.D0
    return
  end if
  ! non-classical reflection
  !ets=ene(0)-0.5D0*sum(ts%omega(1:ts%nvib)) ! is that good?
  ets=ets_+0.5D0*sum(ts%omega(1:ts%nvib)) ! this version agrees better with classical k(T)
  if(ene_in>ene(0)) then 
!  if(ene_in>ets) then 
    if(ene_in>ene(0)+10.D0*ts%omega(1)) then 
      trans_prop=1.D0
      return
    else
      ! non-classical reflection at parabolic barrier
      !print*,"maximal excitation level needed",(3.d0*ene(0)-ets)/ts%omega(1)-0.5D0
      trans_prop=1.D0/(1.D0+exp(2.D0*pi*(ene(0)-ene_in)/omegats)) ! all vibrational modes in ground state
!!$       ! symmetric Eckart barrier:
!!$       vb=4.D0*(ene(0)) ! ene(0) should be replace by e_b(0)
!!$       alpha=ts%omega_imag*sqrt(8.D0/vb)
!!$       apar=2.D0*pi*sqrt(2.D0*ene_in)/alpha
!!$       dpar=2.D0*pi*sqrt( 2.D0*vb - alpha**2 * 0.25D0 ) / alpha 
!!$       if(dpar>500.D0.and.2.D0*apar>500.D0) then
!!$          ! catch a inf/inf case
!!$          trans_prop=(1.D0-exp(-2.D0*apar))/(1.D0+exp(dpar-2.D0*apar))
!!$       else
!!$          if(2.D0*Vb - alpha**2 * 0.25D0 > 0.D0) then
!!$             trans_prop=(cosh(2.D0*apar)-1.D0)/(cosh(2.D0*apar)+cosh(dpar))
!!$          else
!!$             ! cosh(ix)=cos(x) for real-valued x
!!$             dpar=2.D0*pi*sqrt( -2.D0*Vb + alpha**2 * 0.25D0 ) / alpha     
!!$             trans_prop=(cosh(2.D0*apar)-1.D0)/(cosh(2.D0*apar)+cos(dpar))
!!$          end if
!!$       end if

       if(tclass) trans_prop=1.D0
      return
    end if
  end if

  if(tclass) then
    trans_prop=0.D0
    return
  end if
  
!!$  if(ene_in<ene(ntemp)) then
!!$    trans_prop=0.D0 !exp(-s_0(ntemp))
!!$    print*,"Warning for energy:",ene_in
!!$    print*,"Ene(ntemp)",ene(ntemp)
!!$    ! linear extrapolation in S_0
!!$    s_0_interp=s_0(ntemp)-(s_0(ntemp-1)-s_0(ntemp))/(ene(ntemp)-ene(ntemp-1))*(ene_in-ene(ntemp))
!!$    trans_prop=exp(-s_0_interp) ! for the low end, this should always be fine
!!$    print*,"Requested energy is lower than available data, extrapolating"
!!$    return
!!$  end if

  if(ene_in>ene(0)) then
    trans_prop=0.5D0
    !print*,"Warning for energy:",ene_in
    !print*,"Requested energy is higher than available data, extrapolating"
    return
  end if

  do itemp=1,ntemp
    if(ene_in>ene(itemp)) exit
  end do
  !print*,"ene_in",ene_in
  !print*,"iene,ene(iene)",itemp,ene(itemp)
  if(itemp>ntemp) itemp=ntemp
  frac=(ene_in-ene(itemp))/(ene(itemp-1)-ene(itemp))
  !print*,"frac",frac
  s_0_interp=frac*s_0(itemp-1) + (1.D0-frac) * s_0(itemp)
  !sigma_interp=frac*sigma(itemp-1) + (1.D0-frac) * sigma(itemp)
  !beta_interp=frac*beta(itemp-1) + (1.D0-frac) * beta(itemp)
  rotfac_interp=frac*rotfac(itemp-1) + (1.D0-frac) * rotfac(itemp)

  trans_prop=exp(-s_0_interp)/(1.D0+exp(-S_0_interp)) * rotfac_interp
end subroutine p_of_e_one


! for a given excitation pattern iexmode, find the energy-dependent
! stability parameters omega iteratively, which also sets the energy
! ene_n. Then determine the transmission coefficient P(E) for that
! energy.
subroutine find_omega(ntemp,nvib,omega,e_b,s_0,ts,ene,iexmode,tstab,ene_n,tclass,ntemp_stb,trans)
  ! find ene_n for a given input energy, calculate P(E) at that input energy
  use dlf_parameter_module, only: rk
  use dlf_qts, only: pf_type
  use dlf_global, only: pi
  implicit none
  integer,intent(in) :: ntemp,nvib ! nvib = D-1
  real(rk), intent(in) :: omega(0:ntemp,nvib)
  real(rk), intent(in) :: e_b(0:ntemp)
  real(rk), intent(in) :: s_0(0:ntemp)
  type(pf_type), intent(in) :: ts
  integer, intent(in) :: iexmode(nvib)
  real(rk),intent(in) :: ene
  logical,intent(out) :: tstab
  real(rk),intent(out) :: ene_n
  logical,intent(in) :: tclass
  integer,intent(in) :: ntemp_stb ! max ntemp for which stability parameters are available
  real(rk),intent(out) :: trans
  !
  integer :: ivib,iter,niter,itemp
  real(rk) :: omega_int(nvib),frac,last_ene_n,s_0_int
  !integer :: ntemp_stb 
  !ntemp_stb=12!ntemp

  !print*,"finding E_n for vibrational excitation",iexmode

  ! first attempt on energy
  ene_n=0.d0
  do ivib=1,nvib
    ene_n=ene_n+(0.5D0+dble(iexmode(ivib)))*omega(0,ivib)
  end do
  last_ene_n=ene_n

  niter=20
  do iter=1,niter
    ! interpolate omega
    if(ene-ene_n>e_b(0)) then
      omega_int=omega(0,:)
      !print*,"ene_in big",iter,ene_n,ene-ene_n
    else if (ene-ene_n<e_b(ntemp_stb)) then
      omega_int=omega(ntemp_stb,:)
      !print*,"ene_in sma",iter,ene_n,ene-ene_n
    else
      !print*,"ene_in int",iter,ene_n,ene-ene_n
      ! interpolate
      do itemp=1,ntemp
        if(ene-ene_n>e_b(itemp)) exit
      end do
      if(itemp>ntemp) then
        call dlf_fail("itemp too large in find_omega")
      end if
      frac=(ene-ene_n-e_b(itemp))/(e_b(itemp-1)-e_b(itemp))
      omega_int=frac*omega(itemp-1,:)+ (1.D0-frac)*omega(itemp,:)
    end if
    ! re-calculate energy
    ene_n=0.d0
    do ivib=1,nvib
      ene_n=ene_n+(0.5D0+dble(iexmode(ivib)))*omega_int(ivib)
    end do
    if(abs(ene_n-last_ene_n)<1.D-6) exit
    last_ene_n=ene_n
  end do
  !  if(iter<niter+1) print*,"Iterations for E_b converged after",iter,"iterations."
  !print*,"ene_out",iter,ene_n

  tstab=.true.
  !if(ene-ene_n<e_b(ntemp_stb)) tstab=.false. ! now we should use sigma' - not implemented

  ! now interpolate S_0 for E-E_n
  if(ene-ene_n>e_b(0)) then
    ! parabolic barrier
    trans=1.D0/(1.D0+exp(2.D0*pi*(e_b(0)-(ene-ene_n))/ts%omega_imag)) ! all vibrational modes in ground state
    if(tclass) trans=1.D0
  else
    do itemp=1,ntemp
      if(ene-ene_n>e_b(itemp)) exit
    end do
    if(ene-ene_n<e_b(ntemp)) itemp=ntemp
    frac=(ene-ene_n-e_b(itemp))/(e_b(itemp-1)-e_b(itemp))
    s_0_int=frac*s_0(itemp-1)+ (1.D0-frac)*s_0(itemp)
    trans=1.D0/(1.D0+exp(s_0_int))
    if(tclass) trans=0.D0
  end if
  !print*,ene-ene_n,trans
end subroutine find_omega


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! % Below are auxiliary routines for reading and writing files
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
    ene,xcoords,igradient,hessian,etunnel,dist,label)
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
  real(rk),intent(in) :: igradient(varperimage,nimage)
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
  write(555,*) "Gradient per image"
  write(555,*) igradient
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
    ene,xcoords,igradient,hessian,etunnel,dist,mass,label,tok,coeff)
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
  real(rk),intent(out)   :: igradient(varperimage,nimage)
  real(rk),intent(out)   :: hessian(varperimage,varperimage,nimage)
  real(rk),intent(out)   :: etunnel
  real(rk),intent(out)   :: dist(nimage+1)
  real(rk),intent(out)   :: mass(nat)
  character(*),intent(in):: label
  logical ,intent(out)   :: tok
  real(rk),intent(out)   :: coeff(2) ! coefficients for odd and even rotational states
  !
  integer :: iimage,ios
  logical :: there
  integer :: nat_,nimage_,varperimage_
  real(rk):: temperature
  character(128) :: filename,line

  coeff=1.D0
  
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
  read(555,FMT='(a)',end=201,err=200) line ! nat nimage varperimage, maybe coeff
  read(line,*,iostat=ios) nat_,nimage_,varperimage_,coeff
  if(ios/=0) then
    coeff=1.D0
    read(line,*) nat_,nimage_,varperimage_
  end if
  !read(555,*,end=201,err=200) nat_,nimage_,varperimage_
  if(nat/=nat_) then
    if(printl>=4) write(*,*) "Error reading ",trim(filename)," file: Number of &
        &atoms not consistent"
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
  ! read gradient
  read(555,FMT='(a)',iostat=ios) !Gradient per image
  if(ios==0) then
    read(555,*,end=201,err=200) igradient(:,1:min(nimage,nimage_))
  else
    igradient(:,:)=0.D0 ! tag as unread
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
!! Cartesians.
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
!! (qts_reactant.txt). Eigval are in mass-weighted Cartesians.
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
  varperimage_read=0
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
!!****f* qts/read_qts_txt
!!
!! FUNCTION
!!
!! Read Hessian files (qts_reactant.txt, qts_hessian_rs.txt, ...) and
!! Calculate return the mass, the moments of inertia, and the
!! vibrational frequencies.
!!
!! Care about frozen atoms and the re-mass-weighting of the hessian.
!!
!! INPUTS
!!
!! imode (info on RS/RS2/TS), glob%mass
!!
!! OUTPUTS
!! 
!! the derived-type partition function "this"
!!
!! SYNOPSIS
subroutine read_qts_txt(imode,atoffset,this,toneat)
  !! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl,pi
  use dlf_qts, only: dbg_rot,pf_type,minfreq
  use dlf_allocate, only: allocate,deallocate
  use dlf_constants, only: dlf_constants_get
  implicit none
  integer,intent(in) :: imode ! 1: RS1, 2: RS2, 3: TS
  integer,intent(in) :: atoffset ! offset of atom numbers
  type(pf_type), intent(inout) :: this
  logical, intent(inout) :: toneat
  ! toneat (t_one_at): if true at input and imode=2: RS2 is only one atom,
  ! read from qts\_reactant.txt. On output at imode=1: provide information if
  ! RS2 is only one atom
  integer :: nat, varperimage,nimage_read,varperimage_read
  logical :: tok,tokmass
  real(rk), allocatable :: eigvals(:) ! varperimage
  real(rk), allocatable :: xcoords(:) ! 3*nat
  real(rk), allocatable :: mass_file(:) ! nat
  real(rk), allocatable :: ihessian(:,:) ! varperimage,varperimage
  real(rk), allocatable :: igradient(:) ! varperimage
  real(rk), allocatable :: evecs(:,:) ! varperimage,varperimage
  character(3) :: label
  real(rk) :: mu_bim,dist(2),etunnel,temperature,svar,beta_hbar,arr2(2),arr1(1)
  real(rk) :: qrot,qrot_quant,frequency_factor,wavenumber,amu,trot,kboltz_au
  integer :: iat,nzero,iomega,ivib
  logical :: ignoremass=.false. ! could be made an input parameter

  if(this%tused) call dlf_fail("read_qts_txt must act on a non-used fragment!")
  
  if(toneat) then
    if(imode/=2) then
      call dlf_fail("toneat can only be true for imode=2")
    end if
  end if
  
  ! handle imode
  if(imode==1) then
    if(printl>=4) write(stdout,*) "Reading reactant state."
    label=""
    if(atoffset/=0.and.printl>=2) then
      write(stdout,*) "WARNING: finite atom offset for first reactant:",atoffset
    end if
  else if (imode==2) then
    if(printl>=4) write(stdout,*) "Reading reactant state 2."
    if(printl>=6) write(stdout,*) "Atom offset is",atoffset
    if(atoffset==0.and.printl>=2) then
      write(stdout,*) "WARNING: zero atom offset for second reactant:",atoffset
    end if
    if(.not.toneat) then
      label="rs2"
    else
      label=""
    end if
  else if (imode==3) then
    if(printl>=4) write(stdout,*) "Reading transition state."
    label="ts"
    if(atoffset/=0.and.printl>=2) then
      write(stdout,*) "WARNING: finite atom offset for TS:",atoffset
    end if
  else
    call dlf_fail("wrong imode")
  end if

  if(imode/=2.or.toneat) then
    call head_qts_reactant(nat,varperimage,trim(label),tok)
  else ! meaning: imode=2
    call head_qts_hessian(nat,nimage_read,varperimage,"rs2",tok)
    if(.not.tok) then
      write(stdout,'("File qts_hessian_rs2.txt required for bimolecular &
          &rates with each molecule larger than one atom")')
      call dlf_fail("Error reading qts_hessian_rs2.txt")
    end if
  end if

  if(.not.tok) then
    if(imode==3) then
      call dlf_fail("Error reading qts_ts.txt")
    end if
  end if

  if(.not.tok.and.imode==1) then
     call head_qts_hessian(nat,nimage_read,varperimage,"rs",tok)
     if(.not.tok) then
        ! no reactant state information is available. Return with nat=-1
        this%nat=-1
        return
     end if
  end if

  this%nat=nat
  this%coeff=1.D0

  call allocate(eigvals,varperimage)
  call allocate(xcoords,3*nat)
  call allocate(mass_file,nat)

  ! read reactant frequencies
  if(imode/=2.or.toneat) then
    call read_qts_reactant(nat,varperimage,&
        svar,varperimage_read,xcoords,eigvals,mu_bim,&
        trim(label),mass_file,tok)
    this%ene=svar
    !print*,"Mu-bim",mu_bim,tok
    if(tok.and.mu_bim==0.D0) ignoremass=.true.
    if(ignoremass.and.varperimage_read>0) tok=.true.
    if(toneat) then ! meaning it was true at readin, fragment 2 is just one atom
      this%ene=0.D0 ! because the energy is included in the first fragment
      this%mass=mu_bim
      this%tused=.true.
      this%tfrozen=.false.
      this%nat=1
      this%moi=0.D0
      this%nmoi=0
      this%nvib=0
      this%omega_imag=0.D0
      call allocate(this%omega,1) ! to make sure...
      if(printl>=4) then
        write(stdout,'(/,"Information on fragment RS2 (one atom)")')
        write(stdout,'("Mass: ",f14.5," amu")') this%mass
      end if

      return ! nothing more to do for that fragment

    end if
    if(tok.and.mu_bim>0.D0) toneat=.true. ! for next call of this routine
  else
    tok=.false. ! to force reading qts_hessian_rs2.txt
  end if

  if(label=="") label="rs"

  if(.not.tok) then
    ! read full Hessian data
    if(printl>=6) write(stdout,*) "Attempting to read from file &
        &qts_hessian_"//trim(label)//".txt"
    call allocate(ihessian,varperimage,varperimage)
    call allocate(igradient,varperimage)
    call allocate(evecs,varperimage,varperimage)
    nimage_read=1
    call read_qts_hessian(nat,nimage_read,varperimage,temperature,&
         arr1,xcoords,igradient,ihessian,etunnel,dist,mass_file,label,tok,this%coeff)
    this%ene=arr1(1)
 ! here, there is no chance to read mu_bim
    if(.not.tok) call dlf_fail("Error reading Hessian file.")
    if(nimage_read/=1) call dlf_fail("Wrong number of images for"//label)

    ! If masses are not read in (returned negative), do no mass-weighting and
    ! assume glob%mass is correct
    if(minval(mass_file) > 0.D0.and..not.ignoremass) then
      call dlf_constants_get("AMU",svar)
      if(glob%icoord/=190.or.glob%iopt==13) mass_file=mass_file/svar

      tokmass=.true.
      do iat=1,nat
        if(abs(mass_file(iat)-glob%mass(iat+atoffset))>1.D-3) then
          tokmass=.false.
          if(printl>=4) &
              write(stdout,*) "Mass of atom ",iat+atoffset," differs from &
              &Hessian file. File:",mass_file(iat)," input",&
              glob%mass(iat+atoffset)
        end if
      end do

      ! Re-mass-weight
      if(.not.tokmass) then
        print*,"JK this is probably not going to work...! for frozen or bimolecular"
        call dlf_re_mass_weight_hessian(glob%nat,varperimage,mass_file,glob%mass,ihessian)
      end if
    end if

    call dlf_matrix_diagonalise(varperimage,ihessian,eigvals,evecs)

    ! write hessian and qts_reactant file for later use?
    if(.not.ignoremass) then
      if(.not.tokmass) then
        arr1=this%ene
        call write_qts_hessian(nat,nimage_read,varperimage,-1.D0,&
            arr1,xcoords,igradient,ihessian,etunnel,dist,trim(label)//"_mass")
      end if
    else
      ! must use the mass from qts_hessian*.txt, transform:
      call dlf_constants_get("AMU",svar)
      mass_file=mass_file/svar
    end if

    call deallocate(ihessian)
    call deallocate(igradient)
    call deallocate(evecs)
  end if

  if(printl>=4) then
    write(stdout,'(/,"Information on fragment ",a)') label
    if(ignoremass) write(stdout,'("Input masses ignored: no frozen atoms, &
        &no mass-reweighting!")')
  end if

  ! determine tfrozen
  tok=.false.
  call dlf_constants_get("AMU",svar)
  if(glob%icoord/=190.or.glob%iopt==13) svar=1.D0  
  do iat=1,nat
    if(glob%spec(iat+atoffset)<0) then
      if(ignoremass) call dlf_fail("ignoremass and frozen atoms not compatible!")
      tok=.true.
      if(printl>=6) then
        write(stdout,'("Atom ",i4," is frozen (atom ",i4," in this &
            &fragment) mass=",f14.5," amu")')&
            iat+atoffset,iat,glob%mass(iat+atoffset)/svar
      end if
    else
      if(printl>=6.and..not.ignoremass) then
        write(stdout,'("Atom ",i4," is active (atom ",i4," in this &
            &fragment) mass=",f14.5," amu")')&
            iat+atoffset,iat,glob%mass(iat+atoffset)/svar
      end if
    end if
  end do
  this%tfrozen=tok

  if(printl>=4) write(stdout,'("Number of atoms: ",i6)') this%nat

  ! determine mass
  this%mass=0.D0
  do iat=1,nat
    this%mass=this%mass+glob%mass(iat+atoffset)
  end do
  call dlf_constants_get("AMU",svar)
  if(glob%icoord==190.and.glob%iopt/=13) this%mass=this%mass/svar
  if(ignoremass) this%mass=sum(mass_file)
  if(printl>=4) write(stdout,'("Mass:   ",f14.5," amu")') this%mass
  if(printl>=4) write(stdout,'("Energy: ",es16.9," Hartree")') this%ene

  if(.not.this%tfrozen) then
    if(nat==1) then
      nzero=3
      this%nmoi=0
    else
      ! rotational partition function is used for this fragment
      beta_hbar=1.D0 ! a dummy here, has to be handled below
      call dlf_constants_get("AMU",svar)
      ! rotational partition function (relative) of reactant
      nzero=6 ! dummy here
      if(ignoremass) then
        call rotational_partition_function(nat,mass_file*svar,nzero,xcoords,&
            beta_hbar,this%coeff,qrot,qrot_quant,this%moi)
      else
         if(glob%icoord==190.and.glob%iopt/=13) svar=1.D0
        call rotational_partition_function(nat,glob%mass(1+atoffset:nat+atoffset)*svar,&
            nzero,xcoords,beta_hbar,this%coeff,qrot,qrot_quant,this%moi)
      end if
      this%nmoi=3
      if(nzero==5) this%nmoi=2
      if(printl>=4) then
         write(stdout,'("Moments of inertia: ",3es10.3)') this%moi
         call dlf_constants_get("KBOLTZ_AU",KBOLTZ_AU)
         if(nzero==5) then
            trot=0.5D0/kboltz_au/this%moi(1)
         else
            trot=0.5D0/(pi*product(this%moi))**(1.D0/3.D0)/kboltz_au
         end if
         write(stdout,'("Rotational temperature: ",f10.3," K")') trot
      end if
    end if
  else
    ! no rotation for this fragment
    this%nmoi=0
    nzero=0
  end if
  if(printl>=4) then
    if(abs(this%coeff(1)-1.D0)>1.D-3.or.abs(this%coeff(2)-1.D0)>1.D-3) then
      write(stdout,'("Even-J rotational states with weighting factor ",f10.5)') this%coeff(1)
      write(stdout,'("Odd-J rotational states with weighting factor  ",f10.5)') this%coeff(2)
    end if
    write(stdout,'(i1," modes in this fragment correspond to &
        &translation and rotation.")') nzero
    write(stdout,'(i1," modes used in the rotational partition function.")') this%nmoi
  end if
  call deallocate(xcoords)
  call deallocate(mass_file)

  ! now deal with vibration
  this%nvib=varperimage-nzero
  if(imode==3) this%nvib=this%nvib-1
  call allocate(this%omega,this%nvib)
  if(printl>=4) then
    write(stdout,"('Vibrations:')")
    write(stdout,"('Mode      Eigenvalue Frequency (cm^-1)')")
  end if
  iomega=0
  this%omega_imag=0.D0
  ! ignore the smallest nzero or nzero+1 eigenvalues
  if(imode==3) then
    nzero=nzero+1
    this%omega_imag=sqrt(abs(eigvals(1)))
  end if
  call dlf_constants_get("CM_INV_FOR_AMU",frequency_factor)
  call dlf_constants_get("AMU",amu)
  this%omega2zero=0.D0
  do ivib=1,varperimage
    wavenumber = sqrt(abs(amu*eigvals(ivib))) * frequency_factor
    if(eigvals(ivib)<0.D0) wavenumber=-wavenumber
    if(ivib<=nzero) then
      if(printl>=4) write(stdout,"(i5,f15.10,f10.3,' not used')") ivib,eigvals(ivib),wavenumber
      if(imode==3) then
        ! TS
        if(ivib>1.and.ivib<=7) then
          this%omega2zero(ivib-1)=eigvals(ivib)
        end if
      else
        if(ivib<=6) then
          this%omega2zero(ivib)=eigvals(ivib)
        end if
      end if
    else
      if(minfreq>0.D0.and.sqrt(abs(amu*eigvals(ivib)))<minfreq) then
        if(printl>=4) write(stdout,"(i5,f15.10,f10.3,' used, raised to ',f10.3)") &
            ivib,eigvals(ivib),wavenumber,minfreq*frequency_factor
        eigvals(ivib)=minfreq**2/amu
      else
        if(printl>=4) write(stdout,"(i5,f15.10,f10.3,' used')") ivib,eigvals(ivib),&
            wavenumber
      end if
      iomega=iomega+1
      if(iomega>this%nvib) call dlf_fail("Wrong assignment of iomega")
      this%omega(iomega)=sqrt(abs(eigvals(ivib)))
      !print*,"wavenumber used",this%omega(iomega)*frequency_factor*sqrt(amu)
    end if
  end do
  if(printl>=4) write(stdout,'("Vibrational zero point energy: ",es16.9," Hartree")') &
      sum(this%omega)*0.5D0

  call deallocate(eigvals)

  this%tused=.true.
  
end subroutine read_qts_txt

! d E_b / d beta_hbar written by Andreas Loehle
SUBROUTINE calc_dEdbeta(nimage, icoords_dlf, hessians, gradients_dlf, &
    varperimage, beta_hbar, info, dEdbeta)
  use dlf_parameter_module, only: rk
  use dlf_allocate, only: allocate,deallocate
  use dlf_global, only: printl,stdout
  use dlf_qts, only: time
  IMPLICIT NONE
  ! nimage       :: Number of images of the instanton solution
  ! icoords_dlf  :: Contains coordinates of the instanton in the shape( varperimage* nr of image)
  ! hessians     :: Contains hessian matrix of V for all points of the instanton (varperimage, varperimage, nr_image)
  ! gradients    :: Contains the gradient of V for all points of the instanton (varperimage*nr_image)
  ! varperimage  :: Number of degrees of freedom
  ! beta_hbar    :: period of the orbit beta*hbar
  ! info         :: Feedback if solving A*x = b was successful. If successful info = 0
  ! dEdbeta      :: Final result for dEdbeta
  integer, intent(in) :: nimage, varperimage
  real(rk), dimension(varperimage, varperimage, nimage), intent(in) :: hessians
  real(rk), dimension(varperimage*nimage), intent(in) :: gradients_dlf
  real(rk), dimension(varperimage*nimage), intent(in) :: icoords_dlf
  real(rk), intent(in) :: beta_hbar
  integer, intent(out) :: info
  real(rk), intent(out) :: dEdbeta
  real(rk) :: d1, d2 ! Measuring distance between images
  
  !!!!!!!!!!!!!!!!!!!!!!!!! Reshaped arrays for for further calculation
  real(rk), dimension(varperimage, nimage):: gradients
  real(rk), dimension(varperimage, nimage):: icoords
  
  !!!!!!!! Declaring dummy variables for the calculation A*q= b
  real(rk), dimension(varperimage*2*nimage, varperimage*2*nimage) :: Amat, Amat_c ! Hessian of the action
  real(rk), dimension(varperimage*2*nimage, 1) :: rhs !right hand side of the system of equations
  real(rk), dimension(2*varperimage*nimage) :: IPIV
  real(rk), dimension(varperimage, varperimage) :: unity_neg, unity
  real(rk), dimension(varperimage, 2*nimage) :: q_matrix
  real(rk), allocatable :: work(:)
  integer :: i, j
  real(rk) :: dummy
  real :: time1,time2,time3,time4

  if(time) call CPU_TIME (time1)
  
  !!! Reshaping instanton and gradients array for further calculations
  icoords = reshape(icoords_dlf, shape(icoords), order =  (/ 1, 2 /))
  gradients = reshape(gradients_dlf, shape(gradients), order =  (/ 1, 2 /))

  
  Amat = 0.0d0
  unity_neg = 0.0d0
  unity = 0.0d0
  
  DO i = 1, varperimage
    unity_neg(i,i) = -1.0d0
    unity(i, i) = 2.0d0
  END DO
 
  !!!!!!!!!! Writing hessians*(beta_hbar/P)**2 + 2  from x' to x '' in A
  DO i = 1, nimage
    Amat((i-1)*varperimage+1:(i-1)*varperimage+varperimage, &
        (i-1)*varperimage+1:(i-1)*varperimage+varperimage) = &
        unity + hessians(:, :, i)*(beta_hbar/dble(2*nimage))**2
  END DO
  !  Writing hessians back from x'' to x ' in A
  DO i = nimage+1, 2*nimage
    Amat((i-1)*varperimage+1:(i-1)*varperimage+varperimage, &
        (i-1)*varperimage+1:(i-1)*varperimage+varperimage) = &
         unity + hessians(:, :, 2*nimage+1-i)*(beta_hbar/dble(2*nimage))**2
  END DO
  !!!!!!!!!!!! Writing in unities in the off diagonal elements
  !First upper diagonal 
  DO i = 1, 2*nimage-1
     Amat((i-1)*varperimage+1:(i-1)*varperimage+varperimage, &
         (i-1)*varperimage+1+varperimage:(i-1)*varperimage+varperimage+varperimage) &
         = unity_neg(1:varperimage, 1:varperimage)
  END DO
  ! Lower diagonal 
  DO i = 1, 2*nimage-1
     Amat((i-1)*varperimage+1+varperimage:(i-1)*varperimage+varperimage+varperimage, &
          (i-1)*varperimage+1:(i-1)*varperimage+varperimage) = &
          unity_neg(1:varperimage, 1:varperimage)
  END DO
  !!! Finally the Edges
  Amat(2*nimage*varperimage-varperimage+1:2*nimage*varperimage,1:varperimage)&
      = unity_neg(1:varperimage, 1:varperimage)
  Amat(1:varperimage, 2*nimage*varperimage-varperimage+1:2*nimage*varperimage)&
      = unity_neg(1:varperimage, 1:varperimage)
  
  !!!!!!!!!!!!!!!!!!! Determine q in Aq = b
  ! Creating b vector 
  DO i = 1, nimage
     rhs((i-1)*varperimage+1:(i-1)*varperimage+varperimage, 1) = &
          -2.0d0*beta_hbar/dble((2*nimage)**2)*gradients(1:varperimage, i)  
  END DO
  
  DO i = 1+nimage, 2*nimage
     rhs((i-1)*varperimage+1:(i-1)*varperimage+varperimage, 1) = &
          -2.0d0*beta_hbar/dble((2*nimage)**2)*gradients(1:varperimage, 2*nimage+1-i)  
  END DO
 
  ! Solving linear equation Aq = b
  ! TODO: shift this to dlf_linalg
  if(time) call CPU_TIME (time2)
!!$  ! general matrix:
!!$  call DGESV(varperimage*2*nimage, 1, Amat, varperimage*2*nimage, IPIV, rhs, &
!!$      varperimage*2*nimage, INFO)

  ! symmetric matrix:
  call allocate(work,1)
  call dsysv('U',varperimage*2*nimage,1,amat, varperimage*2*nimage, IPIV, rhs, &
      varperimage*2*nimage, work, -1,info)
  i=int(work(1))
  call deallocate(work)
  call allocate(work,i)
  call dsysv('U',varperimage*2*nimage,1,amat, varperimage*2*nimage, IPIV, rhs, &
      varperimage*2*nimage, work, i,info)
  call deallocate(work)
  if(info/=0) then
    if(printl>=2) write(stdout,*) "Warning, system solver for dE/dbeta failed, info=",info
    dedbeta=1.D0 ! mark it as useless
  end if

  if(time) call CPU_TIME (time3)
  
  ! Calculating dE/dbeta
  
  ! Reshaping q_i matrix
  q_matrix = reshape(rhs(:,1), shape(q_matrix), order =  (/ 1, 2 /))
  
  dEdbeta = 0.0d0

  !In Principle every point can be chosen, yet for numerical reasons we check 
  !which points are closed together
  d1 = sqrt(  dot_product(icoords(:, nimage) - icoords(:, nimage-1) , &
      icoords(:, nimage) - icoords(:, nimage-1)))
  d2 = sqrt(dot_product(icoords(:, 2) - icoords(:, 1), icoords(:, 2) - icoords(:, 1)))
  
  IF (d1 <= d2) THEN
    j = nimage-1
    print*,"using last image for dE/dbeta."
  ELSE
    j = 2
    print*,"using first image for dE/dbeta."
  END IF
  
  
  dEdbeta = dble(2*nimage)**2/beta_hbar**3 * 1.0d0/4.0d0*&
      dot_product(icoords(:,j+1) - icoords(:,j-1), icoords(:,j+1)-icoords(:,j-1)) &
      - dble(2*nimage)**2/beta_hbar**2 *1.0d0/4.0d0 * &
      dot_product( icoords(:,j+1) - icoords(:,j-1) , q_matrix(:,j+1) - q_matrix(:,j-1)) &
      + dot_product(gradients(:,j),q_matrix(:,j))
  if(time) call CPU_TIME (time4)
  
  if(time) print*,"time dEdbeta:", time3-time2,time4-time1
  
END SUBROUTINE calc_dEdbeta

! Calculate Stability parameters using the Loehle method
SUBROUTINE calc_ui_param(nimage, beta_hbar, hessians, varperimage, tswap, u_param)
  use dlf_parameter_module, only: rk 
  use dlf_qts, only: time
  IMPLICIT NONE
  integer, intent(in) :: nimage, varperimage
  real(rk), intent(in) :: beta_hbar
  real(rk), dimension(varperimage, varperimage, nimage), intent(in) :: hessians
  logical, intent(in) :: tswap
  complex(rk), dimension(2*varperimage), intent(out) :: u_param
  !
  real(rk), dimension(varperimage, varperimage, 2*nimage+1) :: full_hessians
  real(rk), dimension(varperimage, varperimage) :: S_p_p, S_pp_pp, S_p_pp, A, B, C, D, b_inverse,store
  real(rk) :: dsedxdx(varperimage,varperimage,3)
  real(rk), dimension(varperimage, varperimage) :: S_alt
  real(rk), dimension(2*varperimage, 2*varperimage) :: Stab
  real(rk), dimension(varperimage) :: a_vec, b_vec
  complex(rk), dimension(varperimage) :: dummy_vec
  real(rk) :: u_param_r(2*varperimage),u_param_i(2*varperimage)
  real(rk) :: eigvectr(2*varperimage,2*varperimage)
  integer :: INFO, i, j 
  !complex(rk), dimension(2*varperimage) :: u_tmp
  !real(rk) :: sortlist(2*varperimage)
  real :: time1,time2,time3,time4
  complex(rk) :: dummy
  real(rk) :: svar

  if(time) call CPU_TIME (time1)
  ! duplicate the hessian entries (and swap ends of instanton if required)
  DO i = 1, nimage
     if(tswap) then
        ! swap is worse! correct?
        full_hessians(:, :, i) = hessians(:, :, nimage+1-i)
        full_hessians(:, :, nimage+i) = hessians(:, :, i)
     else
        full_hessians(:, :, i) = hessians(:, :, i)
        full_hessians(:, :, nimage+i) = hessians(:, :, nimage+1-i)
     end if
  END DO
  full_hessians(:, :, 2*nimage+1) = full_hessians(:, :, 2*nimage)

  if(time) call CPU_TIME (time2)
  call calc_dSE_dxdx(2*nimage+1, full_hessians, varperimage, beta_hbar, dsedxdx)
  if(time) call CPU_TIME (time3)
  if(time) print*,"time ui dSE_dxdx:",time3-time2

  S_p_p=dsedxdx(:,:,1)
  S_p_pp=dsedxdx(:,:,2)
  S_pp_pp=dsedxdx(:,:,3)

  ! invert B
  !call calc_inverse(S_p_pp, varperimage , b_inverse ,INFO)
  b_inverse=dsedxdx(:,:,2)
  if(time) call CPU_TIME (time2)
  call dlf_matrix_invert(varperimage,.false.,b_inverse,svar)
  if(time) call CPU_TIME (time3)
  if(time) print*,"time ui invert:",time3-time2

  A = matmul(-b_inverse, S_p_p)
  B = -b_inverse
  C = S_p_pp - matmul(S_pp_pp, matmul( b_inverse, S_p_p))
  D = matmul(-S_pp_pp, b_inverse)

  Stab(1:varperimage      ,       1:varperimage) = A
  Stab(1:varperimage      , varperimage+1:2*varperimage) = B
  Stab(varperimage+1:2*varperimage,       1:varperimage) = C
  Stab(varperimage+1:2*varperimage, varperimage+1:2*varperimage) = D

  ! which routine used here?
  !call calc_EV_values(Stab, 2*varperimage, u_param)
  if(time) call CPU_TIME (time2)
  call dlf_matrix_diagonalise_general(2*varperimage,stab,u_param_r,u_param_i,eigvectr)
  if(time) call CPU_TIME (time3)
  if(time) print*,"time ui diagonal:",time3-time2
  ! combine real and imaginary to complex u_param
  u_param=dcmplx(u_param_r,u_param_i)

  if(time) call CPU_TIME (time4)
  if(time) print*,"time ui:",time4-time1

END SUBROUTINE calc_ui_param

!
! calculate d2S_E / dx'dx' (and dx'dx", dx"dx")
!
! This routine is currently not yet optimised for memory
! requirements. I am sure some savings are possible
SUBROUTINE calc_dSE_dxdx(nimage2, hessians, varperimage, beta_hbar, ds)
  use dlf_parameter_module, only: rk 
  use dlf_global, only: printl,stdout
  use dlf_qts, only: time
  IMPLICIT NONE
  ! Notice d0f = degrees of freedom, in the general case varperimage-1 as only orthogonal contributions
  !are taken into account
  integer, intent(in) :: nimage2, varperimage
  real(rk), dimension(varperimage, varperimage, nimage2), intent(in) :: hessians
  real(rk), intent(in) :: beta_hbar
  real(rk), dimension(varperimage, varperimage,3), intent(out) :: ds
  integer :: P 
  integer :: INFO, l, s
  integer :: iimage, ivar, jvar

  real(rk), dimension(2*varperimage**2+1, (nimage2-2)*varperimage**2) :: A_band
  real(rk), dimension(2*varperimage**2 + varperimage**2 + 1, (nimage2-2)*varperimage**2) :: AB

  real(rk), dimension(varperimage**2, varperimage**2 ) :: unity, K_matrix
  real(rk), dimension(varperimage, varperimage) :: unity_small, K_dummy
  !real(rk), dimension(varperimage, nimage2-2) :: instanton_full, gradient_full

  real(rk), dimension(varperimage**2 *(nimage2-2), 2) :: rhs ! -> rhs
  real(rk), dimension(varperimage**2 *(nimage2-2)) :: IPIV
  real(rk), dimension(varperimage**2) :: b_start

  real :: time1,time2,time3,time4,time5,time6

  if(time) call CPU_TIME (time1)
  P = (nimage2-1)

  unity = 0.0d0
  DO ivar = 1, varperimage**2
    unity(ivar, ivar) = 1.0d0
  END DO
  unity_small = 0.0d0
  DO ivar = 1, varperimage
    unity_small(ivar, ivar) = 1.0d0
  END DO

  ! Filling A_band
  A_band = 0.0d0
  ! Filling  -1 Diagonal Elements
  A_band(1,        varperimage**2+1: (nimage2-2)*varperimage**2) = -1.0d0
  A_band(2*varperimage**2+1, 1:(nimage2-2)*varperimage**2-varperimage**2) = -1.0d0

  if(time) call CPU_TIME (time5)
  if(time) print*,"time u_i A_band",time5-time1

  ! Filling in K Matrix
  DO iimage = 1, nimage2-2
    K_matrix = 0.0d0
    K_dummy = 2.0d0*unity_small + hessians(:, :, iimage+1)*(beta_hbar**2/P**2)
    DO ivar = 1, varperimage 
      K_matrix(ivar*varperimage-varperimage+1:ivar*varperimage, &
          ivar*varperimage-varperimage+1:ivar*varperimage)= K_dummy
    END DO
    DO ivar = 1, varperimage**2
      DO jvar = ivar, varperimage**2
        s = ivar - jvar + varperimage**2 +1 
        l = jvar + (iimage-1)*varperimage**2 
        A_band(s, l) = K_matrix(ivar, jvar) 
      END DO
      DO jvar = 1, ivar-1
        s = ivar - jvar + varperimage**2 +1 
        l = ivar + (iimage-1)*varperimage**2 - s + varperimage**2 + 1
        A_band(s, l) = K_matrix(ivar, jvar) 
      END DO
    END DO
  END DO

  if(time) call CPU_TIME (time6)
  if(time) print*,"time u_i K matrix",time6-time5

  ! Preparing b-vector
  rhs = 0.0d0
  b_start = reshape(unity_small, shape(b_start))

  rhs(1:varperimage**2, 1) = b_start ! for xp_xp
  rhs(varperimage**2*(nimage2-2)-varperimage**2+1 : , 2) = b_start ! xp_xpp and xp_xpp

  ! Preparing AB matrix
  AB = 0.0d0
  AB(varperimage**2+1 : , :) = A_band(: , :)

  ! solving band Matrix
  if(time) then
    call CPU_TIME (time2)
    print*,"time u_i b and AB",time2-time6
    print*,"time u_i before band matrix",time2-time1
    call CPU_TIME (time2)
  end if

  ! this is for a generalised banded matrix, ours is symmetric as well. I
  ! don't know of any specialised routines making use of that.
  call dgbsv((nimage2-2)*varperimage**2, varperimage**2, varperimage**2, 2, AB,  &
      3*varperimage**2 + 1, IPIV, rhs, (nimage2-2)*varperimage**2, INFO) 		

  if(time) call CPU_TIME (time3)
  if(info/=0) then
    if(printl>=2) write(stdout,*) "Warning: system solver failed in d2S_E/dxdx"
    ds=0.D0
    return
  end if
  if(time) print*,"time u_i band matrix:",time3-time2
  !print*,"Info after solving band matrix",info

  !print*,"xp_xp",sum(abs(rhs(1:varperimage**2,1)))
  ! xp_xp
  ds(:,:,1) = (P/beta_hbar)*(unity_small-reshape(rhs(1:varperimage**2,1),(/varperimage,varperimage/))) &
      + 0.5d0*(beta_hbar/P)*hessians(:,:,1)

  !print*,"xp_xpp orig",sum(abs(rhs(1:varperimage**2,2)))
  !print*,"q1",rhs(1:varperimage**2,2)
  !print*,"xp_xpp sym?",sum(abs(rhs(varperimage**2*(nimage2-2)-varperimage**2+1:,1)))
  !print*,"q2",rhs(varperimage**2*(nimage2-2)-varperimage**2+1:,1)
  ! xp_xpp
  ! original:
 ! ds(:,:,2) = -(P/beta_hbar)*reshape( rhs(1:varperimage**2,2), (/varperimage,varperimage/))
  ! symmetric
  ds(:,:,2) = -0.5D0*(P/beta_hbar)*reshape( rhs(1:varperimage**2,2), (/varperimage,varperimage/)) &
      -0.5D0*(P/beta_hbar)*transpose(reshape( rhs(varperimage**2*(nimage2-2)-varperimage**2+1:,1), (/varperimage,varperimage/)))
    
  !print*,"xpp_xpp",sum(abs(rhs(varperimage**2*(nimage2-2)-varperimage**2+1:,2)))
  ! xpp_xpp
  ds(:,:,3) = (P/beta_hbar)*(unity_small-reshape(rhs(varperimage**2*(nimage2-2)-varperimage**2+1:,2),&
      (/varperimage,varperimage/))) + 0.5d0*(beta_hbar/P)*hessians(:,:,nimage2)
  if(time) then 
    call CPU_TIME (time4)
    print*,"time u_i after band matrix",time4-time3
    print*,"time u_i in calc_dSE_dxdx:",time4-time1
  end if

END SUBROUTINE calc_dSE_dxdx

SUBROUTINE calc_QRS(nimage, beta_hbar, hess_eigen_val, varperimage, fluc_fac, ana_fluc)
  use dlf_parameter_module, only: rk 
  IMPLICIT NONE
  integer, intent(in) :: varperimage, nimage
  real(rk), intent(in) :: beta_hbar
  real(rk), dimension(varperimage), intent(in) :: hess_eigen_val
  real(rk) :: dSEdxdx(varperimage,varperimage,3)
  real(rk), intent(out) :: fluc_fac, ana_fluc
  real(rk), dimension(varperimage, varperimage) :: A, B, hess_matrix
  real(rk), dimension(varperimage, varperimage, 2*nimage) :: hessians_full
  real(rk) :: detA, detB
  integer :: i, j
  
  hess_matrix = 0.0d0
  ana_fluc = 1.0d0
  DO j = 1, varperimage
    hess_matrix(j, j) = hess_eigen_val(j)
    ana_fluc = ana_fluc/(2.0d0*sinh(0.5d0*beta_hbar*dsqrt(hess_eigen_val(j))))
  END DO
  
  DO i = 1, 2*nimage
    hessians_full(:, :, i) = hess_matrix
  END DO
  
  call calc_dSE_dxdx(2*nimage, hessians_full, varperimage, beta_hbar, dsedxdx)
  
  A = -dsedxdx(:,:,2) !-S_p_pp
! B = S_p_p + 2.0d0 * S_p_pp + S_pp_pp
  B = dsedxdx(:,:,1) + 2.0d0 * dsedxdx(:,:,2) + dsedxdx(:,:,3)
  
  !call calc_Determinant(A, varperimage, detA)
  !call calc_Determinant(B, varperimage, detB)
  call dlf_matrix_determinant(varperimage,a,deta)
  call dlf_matrix_determinant(varperimage,b,detb)
  print*,"RS: detA, detB",detA,detB
  fluc_fac = sqrt(abs(detA) /abs(detB))
END SUBROUTINE calc_QRS

! sort is currently only used for the RS in the Loehle method - which
! has to be rewritten anyway. Then it can be removed
SUBROUTINE SORT(vec_unsorted, N, vec_sorted)
  use dlf_parameter_module, only: rk 
  IMPLICIT NONE
  integer, intent(in) :: N
  real(rk), dimension(N), intent(in) :: vec_unsorted
  real(rk), dimension(N), intent(out) :: vec_sorted
  integer :: max_pos, min_pos
  real(rk) :: max_value
  real(rk), dimension(N) :: dummy
  integer :: i 
  max_pos = maxloc(vec_unsorted, 1)
  max_value = vec_unsorted(max_pos)
  max_value = 10.0d0*max_value
  dummy = vec_unsorted
  DO i = 1, N
    min_pos = minloc(dummy, 1)
    vec_sorted(i) = dummy(min_pos)
    dummy(min_pos) = max_value
  END DO
  RETURN  
END SUBROUTINE SORT

!!****
! routines to store coordinates, energies and gradients (and maybe Hessians)
! for use in interpolation assumes a directory store_pes to be present

subroutine dlf_store_eandg
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,printl,stdout
  use dlf_constants, only: dlf_constants_get
  !use dlf_qts, only: taskfarm_mode
  implicit none
  integer             :: counter
  character(128)      :: filename,number
  real(rk)            :: svar
  logical             :: there

  if(glob%iam > 0 ) then
    ! only task zero should write
    return
  end if

  do counter=1,10000
    write(number,'(i6)') counter
    filename="store_pes/grad_"//trim(adjustl(number))//'.txt'
    inquire(file=filename,exist=there)
    if(.not.there) exit
  end do

  open(unit=555,file=filename, action='write')
  write(555,*) "Coordinates and Gradient of a Structure written by dl-find"
  write(555,*) glob%nat,1,1
  write(555,*) 0.D0
  write(555,*) "Energy"
  write(555,*) glob%energy
  write(555,*) "Coordinates"
  write(555,*) glob%xcoords
  write(555,*) "Gradient"
  write(555,*) glob%xgradient
  write(555,*) "Masses in au"
  if((glob%icoord==190.or.glob%icoord==390).and.glob%iopt/=11.and.glob%iopt/=13) then
    svar=1.D0
  else
    call dlf_constants_get("AMU",svar)
  end if
  write(555,*) glob%mass*svar
  close(555)

end subroutine dlf_store_eandg

subroutine dlf_store_egh(nvar,hessian)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,printl,stdout
  use dlf_constants, only: dlf_constants_get
  !use dlf_qts, only: taskfarm_mode
  implicit none
  integer ,intent(in) :: nvar
  real(rk),intent(in) :: hessian(nvar,nvar)
  integer             :: counter
  character(128)      :: filename,number
  real(rk)            :: svar
  logical             :: there

  if(glob%iam > 0 ) then
    ! only task zero should write
    return
  end if

  do counter=1,10000
    write(number,'(i6)') counter
    filename="store_pes/hess_"//trim(adjustl(number))//'.txt'
    inquire(file=filename,exist=there)
    if(.not.there) exit
  end do

  open(unit=555,file=filename, action='write')
  write(555,*) "Coordinates, Gradient and Hessian of a Structure written by dl-find"
  write(555,*) glob%nat,1,1
  write(555,*) 0.D0
  write(555,*) "Energy"
  write(555,*) glob%energy
  write(555,*) "Coordinates"
  write(555,*) glob%xcoords
  write(555,*) "Gradient"
  write(555,*) glob%xgradient
  write(555,*) "Hessian"
  write(555,*) hessian
  write(555,*) "Masses in au"
  if((glob%icoord==190.or.glob%icoord==390).and.glob%iopt/=11.and.glob%iopt/=13) then
    svar=1.D0
  else
    call dlf_constants_get("AMU",svar)
  end if
  write(555,*) glob%mass*svar
  close(555)

  ! communicate Hessian back to calling code (needed at least for the
  ! ChemShell-polyrate interface)
  ! Actually, at least for icoord=190 we could still write the Hessian...
  if(glob%icoord==0.and.minval(glob%spec)==0.and.maxval(glob%spec)==0) then
    ! make sure we have Cartesian coordinates and no frozen atoms
    if(printl>=4) write(stdout,'(a)') "Hessian is written"
    call dlf_put_hessian(nvar,hessian,glob%iam)
  else
    if(printl>=2) write(stdout,'(a)') "Hessian is not written due to wrong&
        & coordinate or frozen atom setting"
  end if

end subroutine dlf_store_egh

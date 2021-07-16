! dummy module to draw potential energy surfaces ...
! uses an analytic function and writes a povray surface consisting of
! smooth triangles
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dl_find(nvar)
  ! called dl_find even though it does not find anything ...
  use dlf_parameter_module
  implicit none
  integer   ,intent(in)    :: nvar
  !
  real(rk)  :: coords(nvar),gradient(nvar),energy
  real(8)             :: xstart,xend,ystart,yend,del
  integer(4)          :: nx,ny,ix,iy,status
  real(8)   ,allocatable :: func(:,:,:),nvec(:,:,:)
  real(8)             :: xv,yv,svar,svar2,point3(9),nvec3(9),e1,e2
  integer(4)          :: iunit
! **********************************************************************
  if(nvar.ne.2) then
    print*,"Erorr: drawing is only possible in 2 dimensions!"
    call dlf_error
  end if

!!$  coords(1)=-0.822D0
!!$  coords(2)=0.624D0
!!$  call dlf_get_gradient(nvar,coords,energy,gradient,status)
!!$  print*,"energy",energy
!!$  return
  

  xstart=-1.5D0
  xend=   1.5D0
  ystart=-0.5D0
  yend=   2.0D0
  nx=30
  ny=30

  allocate(func(3,nx,ny))
  allocate(nvec(3,nx,ny))

  ! DEfine function grid
  func(:,:,:)=0.D0
  do ix=1,nx
    xv=xstart+dble(ix-2)/dble(nx-3)*(xend-xstart)
    do iy=1,ny
      yv=ystart+dble(iy-2)/dble(ny-3)*(yend-ystart)
      func(1,ix,iy)=xv
      func(2,ix,iy)=yv
    end do
  end do

  ! define function
  do ix=1,nx
    do iy=1,ny
      coords=func(1:2,ix,iy)
      call dlf_get_gradient(nvar,coords,energy,gradient,status)
      if(status/=0) call dlf_error()
      func(3,ix,iy)=min(energy,2.D0)
      !write(*,'("x,y,func",2f10.5,3es15.7)') func(1:3,ix,iy),gradient
    end do
  end do

  print*,"Maximum:",maxval(func(3,:,:))
  print*,"Minimum:",minval(func(3,:,:))
  
  call normalvecotrs(nx,ny,func,nvec)

  iunit=12
  open(unit=iunit,file="surface.inc")
  do ix=2,nx-2
    do iy=2,ny-2
      point3(1:3)=func(:,ix,iy)
      nvec3(1:3) =nvec(:,ix,iy)
      point3(4:6)=func(:,ix,iy+1)
      nvec3(4:6) =nvec(:,ix,iy+1)
      point3(7:9)=func(:,ix+1,iy+1)
      nvec3(7:9) =nvec(:,ix+1,iy+1)
      call WRITE_POV_SMOOTHTRIANGLE(Iunit,POINT3,NVEC3)
      point3(1:3)=func(:,ix,iy)
      nvec3(1:3) =nvec(:,ix,iy)
      point3(4:6)=func(:,ix+1,iy)
      nvec3(4:6) =nvec(:,ix+1,iy)
      point3(7:9)=func(:,ix+1,iy+1)
      nvec3(7:9) =nvec(:,ix+1,iy+1)
      call WRITE_POV_SMOOTHTRIANGLE(Iunit,POINT3,NVEC3)
    end do
  end do
  close(iunit)

  deallocate(func)
  deallocate(nvec)

end subroutine dl_find

! HELPER ROUTINES TO DRAW FUNCTIONS ...

subroutine normalvecotrs(nx,ny,func,nvec)
  implicit none
  integer(4)      ,intent(in) :: nx,ny
  real(8)         ,intent(in) :: func(3,nx,ny)
  real(8)         ,intent(out):: nvec(3,nx,ny)
  integer(4)                  :: ix,iy
  real(8)                     :: v1(3),v2(3),v3(3),v4(3),v5(3),v6(3),v7(3),v8(3),vp(3)
  real(8)                     :: svar
! **********************************************************************
  do ix=2,nx-1
    do iy=2,ny-1
      v1=func(:,ix+1,iy+1)-func(:,ix,iy)
      v2=func(:,ix+1,iy  )-func(:,ix,iy)
      v3=func(:,ix+1,iy-1)-func(:,ix,iy)
      v4=func(:,ix  ,iy-1)-func(:,ix,iy)
      v5=func(:,ix-1,iy-1)-func(:,ix,iy)
      v6=func(:,ix-1,iy  )-func(:,ix,iy)
      v7=func(:,ix-1,iy+1)-func(:,ix,iy)
      v8=func(:,ix  ,iy+1)-func(:,ix,iy)

      call crossproduct(v1,v2,vp)
      nvec(:,ix,iy)=nvec(:,ix,iy)+vp
      call crossproduct(v2,v3,vp)
      nvec(:,ix,iy)=nvec(:,ix,iy)+vp
      call crossproduct(v3,v4,vp)
      nvec(:,ix,iy)=nvec(:,ix,iy)+vp
      call crossproduct(v4,v5,vp)
      nvec(:,ix,iy)=nvec(:,ix,iy)+vp
      call crossproduct(v5,v6,vp)
      nvec(:,ix,iy)=nvec(:,ix,iy)+vp
      call crossproduct(v6,v7,vp)
      nvec(:,ix,iy)=nvec(:,ix,iy)+vp
      call crossproduct(v7,v8,vp)
      nvec(:,ix,iy)=nvec(:,ix,iy)+vp
      call crossproduct(v8,v1,vp)
      nvec(:,ix,iy)=nvec(:,ix,iy)+vp

      svar=sqrt(sum(nvec(:,ix,iy)**2))
      nvec(:,ix,iy)=nvec(:,ix,iy)/svar
    end do
  end do
end subroutine normalvecotrs


! ......................................................................
SUBROUTINE WRITE_POV_SMOOTHTRIANGLE(IFILE,POINT,NVEC)
! USE THE XYZ POINTS AND NORMAL VECTORS TO WRITE TRIANGLES   
  IMPLICIT NONE
  INTEGER(4)  ,INTENT(IN)   :: IFILE
  REAL(8)     ,INTENT(IN)   :: POINT(9)
  REAL(8)     ,INTENT(IN)   :: NVEC(9)
  REAL(8)                   :: NL(3),V1(3),V2(3),svar
!  INTEGER(4)                :: I
! **********************************************************************
! CALCULATE REAL NORMAL VECTOR OF THE TRIANGLE

  V1=POINT(1:3)-POINT(7:9)
  V2=POINT(1:3)-POINT(4:6)
  CALL CROSSPRODUCT(V2,V1,NL)

! write the energy in Y-direction  
  write(ifile,"('smooth_triangle{<',f9.4,',',f9.4,',',f9.4,'>, ')") &
      point(1),point(3),point(2)
  svar=sum(nvec(1:3)*nl)
  if(svar.gt.0.D0) then
    write(ifile,"('                <',f9.4,',',f9.4,',',f9.4,'>, ')") &
        nvec(1),nvec(3),nvec(2)
  else
    write(ifile,"('                <',f9.4,',',f9.4,',',f9.4,'>, ')") &
        -nvec(1),-nvec(3),-nvec(2)
  end if

  write(ifile,"('                <',f9.4,',',f9.4,',',f9.4,'>, ')") &
      point(4),point(6),point(5)
  svar=sum(nvec(4:6)*nl)
  if(svar.gt.0.D0) then
    write(ifile,"('                <',f9.4,',',f9.4,',',f9.4,'>, ')") &
        nvec(4),nvec(6),nvec(5)
  else
    write(ifile,"('                <',f9.4,',',f9.4,',',f9.4,'>, ')") &
        -nvec(4),-nvec(6),-nvec(5)
  end if

  write(ifile,"('                <',f9.4,',',f9.4,',',f9.4,'>, ')") &
      point(7),point(9),point(8)
  svar=sum(nvec(7:9)*nl)
  if(svar.gt.0.D0) then
    write(ifile,"('                <',f9.4,',',f9.4,',',f9.4,'>')") &
        nvec(7),nvec(9),nvec(8)
  else
    write(ifile,"('                <',f9.4,',',f9.4,',',f9.4,'>')") &
        -nvec(7),-nvec(9),-nvec(8)
  end if
  write(ifile,"('}')") 
  
END SUBROUTINE WRITE_POV_SMOOTHTRIANGLE

SUBROUTINE CROSSPRODUCT(A,B,C)
! CALC C = A X B
  IMPLICIT NONE
  REAL(8),INTENT(IN) :: A(3),B(3)
  REAL(8),INTENT(OUT):: C(3)
! **********************************************************************
  C(1)=A(2)*B(3)-A(3)*B(2)
  C(2)=A(3)*B(1)-A(1)*B(3)
  C(3)=A(1)*B(2)-A(2)*B(1)
END SUBROUTINE CROSSPRODUCT


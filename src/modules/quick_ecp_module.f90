!
!	quick_ecp_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!  ECP module written by Alessandro GENONI 03/12/2006
module quick_ecp_module
   implicit none

   integer, parameter :: mxproj=5,mxang=3
   !
   ! Derived Parameters
   !
   integer, parameter :: mxnnn=max(2*mxang+1,mxang+mxproj+1),&
         mxprim=(mxang+1)*(mxang+2)/2,&
         mxgout=mxprim*mxprim,&
         lmax1=max(1,mxang+max(mxang,mxproj)),&
         lfdim=lmax1+1,&
         lmfdim=lfdim**2,&
         lmxdim=(lmax1*(lmax1+2)*(lmax1+4)/3 *  (lmax1+3) +&
         (lmax1+2)**2 * (lmax1+4))/16,&
         mc1dim=2*mxproj-1,&
         len_dfac=3*lmax1+3,&
         len_fac=mxproj*mxproj
   !
   integer :: necprim,nbf12,itolecp
   double precision :: tolecp,thrshecp

   integer, dimension(:), allocatable   :: nelecp,lmaxecp,nlp,kvett
   double precision, dimension (:), allocatable :: clp,zlp,ecp_int,gout

   integer, dimension(:,:), allocatable :: kfirst,klast
   !
   integer, dimension(:), allocatable   :: lf,lmf,lml,lmx,lmy,lmz
   integer, dimension(:,:), allocatable :: mc,mr

   double precision, dimension(:), allocatable   :: zlm,dfac,dfaci,factorial
   double precision, dimension(:,:), allocatable :: flmtx,fprod

   double precision, allocatable, dimension(:) :: eta
   integer, allocatable, dimension(:) :: kmin,kmax,ktypecp

end module quick_ecp_module

!
!	quick_ecp_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

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

contains

   !---------------------------------------------------
   ! Deallocate all ECP module arrays (call at cleanup)
   !---------------------------------------------------
   subroutine deallocate_quick_ecp()
      implicit none

      ! ECP atom/primitive arrays (allocated in ecp.f90:allocateatoms_ecp)
      if (allocated(nelecp))    deallocate(nelecp)
      if (allocated(lmaxecp))   deallocate(lmaxecp)
      if (allocated(clp))       deallocate(clp)
      if (allocated(zlp))       deallocate(zlp)
      if (allocated(nlp))       deallocate(nlp)
      if (allocated(kfirst))    deallocate(kfirst)
      if (allocated(klast))     deallocate(klast)

      ! ECP integral/scratch arrays (allocated in basis.f90 when ecp=.true.)
      if (allocated(kmin))      deallocate(kmin)
      if (allocated(kmax))      deallocate(kmax)
      if (allocated(eta))       deallocate(eta)
      if (allocated(ecp_int))   deallocate(ecp_int)
      if (allocated(kvett))     deallocate(kvett)
      if (allocated(gout))      deallocate(gout)
      if (allocated(ktypecp))   deallocate(ktypecp)
      if (allocated(zlm))       deallocate(zlm)
      if (allocated(flmtx))     deallocate(flmtx)
      if (allocated(lf))        deallocate(lf)
      if (allocated(lmf))       deallocate(lmf)
      if (allocated(lml))       deallocate(lml)
      if (allocated(lmx))       deallocate(lmx)
      if (allocated(lmy))       deallocate(lmy)
      if (allocated(lmz))       deallocate(lmz)
      if (allocated(mc))        deallocate(mc)
      if (allocated(mr))        deallocate(mr)
      if (allocated(dfac))      deallocate(dfac)
      if (allocated(dfaci))     deallocate(dfaci)
      if (allocated(factorial))  deallocate(factorial)
      if (allocated(fprod))     deallocate(fprod)

   end subroutine deallocate_quick_ecp

end module quick_ecp_module

!
!	quick_scratch_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

! SCRATCH module.
module quick_scratch_module

!------------------------------------------------------------------------
!  ATTRIBUTES  : HOLD,HOLD2
!  SUBROUTINES : allocate_quick_scratch
!                deallocate_quick_scratch
!  FUNCTIONS   : none
!  DESCRIPTION : This module is set for scratch varibles
!  AUTHOR      : Yipu Miao
!------------------------------------------------------------------------

    implicit none
    
    ! store some scratch arrays
    
    type quick_scratch_type
        double precision, dimension(:,:), allocatable :: hold,hold2
        ! variables required for fullx subroutines
        double precision, dimension(:,:), allocatable :: tmpx, tmphold, tmpco, V
        double precision, dimension(:), allocatable :: Sminhalf, IDEGEN1
        ! magic variables required for classopt subroutine
        double precision, dimension(:), allocatable :: X44,X44aa,X44bb,X44cc,X44dd  
#ifdef MPIV
        ! to store the result of operator reduction
        double precision, dimension(:,:), allocatable :: osum, obsum
#endif
    end type quick_scratch_type
    
    type (quick_scratch_type) quick_scratch
    
!    double precision, dimension(:,:), allocatable :: V2  !,hold,hold2
    
    interface alloc
        module procedure allocate_quick_scratch
    end interface alloc

    interface allocshellopt
        module procedure allocate_shellopt_scratch
    end interface allocshellopt

    interface allocfullx
        module procedure allocate_fullx_scratch
    end interface allocfullx

    interface dealloc
        module procedure deallocate_quick_scratch
    end interface dealloc 

    interface deallocshellopt
        module procedure deallocate_shellopt_scratch
    end interface deallocshellopt

    interface deallocfullx
        module procedure deallocate_fullx_scratch
    end interface deallocfullx
    
    contains
        subroutine allocate_quick_scratch(self,nbasis)
            implicit none
            integer nbasis
            type (quick_scratch_type) self
            
            if(.not. allocated(self%hold)) allocate(self%hold(nbasis,nbasis))
            if(.not. allocated(self%hold2)) allocate(self%hold2(nbasis,nbasis))
#ifdef MPIV
            if(.not. allocated(self%osum)) allocate(self%osum(nbasis,nbasis))
            if(.not. allocated(self%obsum)) allocate(self%obsum(nbasis,nbasis))
#endif
            return
            
        end subroutine allocate_quick_scratch
        
        
        subroutine deallocate_quick_scratch(self)
            implicit none
            type (quick_scratch_type) self
            
            if (allocated(self%hold)) deallocate(self%hold)
            if (allocated(self%hold2)) deallocate(self%hold2)
#ifdef MPIV
            if(allocated(self%osum)) deallocate(self%osum)
            if(allocated(self%obsum)) deallocate(self%obsum)
#endif
            return
            
        end subroutine deallocate_quick_scratch

        subroutine allocate_fullx_scratch(self,nbasis)
            implicit none
            integer :: nbasis, ii, jj
            type (quick_scratch_type) self

            if(.not. allocated(self%tmpx)) allocate(self%tmpx(nbasis,nbasis))
            if(.not. allocated(self%tmphold)) allocate(self%tmphold(nbasis,nbasis))
            if(.not. allocated(self%tmpco)) allocate(self%tmpco(nbasis,nbasis))
            if(.not. allocated(self%V)) allocate(self%V(3,nbasis))
            if(.not. allocated(self%Sminhalf)) allocate(self%Sminhalf(nbasis))
            if(.not. allocated(self%IDEGEN1)) allocate(self%IDEGEN1(nbasis))

            self%tmpx=0.0d0
            self%tmphold=0.0d0
            self%tmpco=0.0d0
            self%V=0.0d0
            self%Sminhalf=0.0d0
            self%IDEGEN1=0.0d0

            do ii=1,nbasis
              do jj=1, nbasis
                if(ii .eq. jj) then
                  self%tmpx(jj,ii)=1.0d0
                  self%tmphold(jj,ii)=1.0d0
                endif
              enddo
            enddo

            return

        end subroutine allocate_fullx_scratch

        subroutine deallocate_fullx_scratch(self)
            implicit none
            type (quick_scratch_type) self

            if (allocated(self%tmpx)) deallocate(self%tmpx)
            if (allocated(self%tmphold)) deallocate(self%tmphold)
            if (allocated(self%tmpco)) deallocate(self%tmpco)
            if (allocated(self%V)) deallocate(self%V)
            if (allocated(self%Sminhalf)) deallocate(self%Sminhalf)
            if (allocated(self%IDEGEN1)) deallocate(self%IDEGEN1)

            return

        end subroutine deallocate_fullx_scratch

        subroutine allocate_shellopt_scratch(self,maxcontract)
            implicit none
            integer :: maxcontract, arraysize
            type (quick_scratch_type) self

            arraysize=maxcontract**4
            if(.not. allocated(self%X44)) allocate(self%X44(arraysize))
            if(.not. allocated(self%X44aa)) allocate(self%X44aa(arraysize))
            if(.not. allocated(self%X44bb)) allocate(self%X44bb(arraysize))
            if(.not. allocated(self%X44cc)) allocate(self%X44cc(arraysize))
            if(.not. allocated(self%X44dd)) allocate(self%X44dd(arraysize))

            return

        end subroutine allocate_shellopt_scratch        

        subroutine deallocate_shellopt_scratch(self)
            implicit none
            type (quick_scratch_type) self

            if (allocated(self%X44)) deallocate(self%X44)
            if (allocated(self%X44aa)) deallocate(self%X44aa)
            if (allocated(self%X44bb)) deallocate(self%X44bb)
            if (allocated(self%X44cc)) deallocate(self%X44cc)
            if (allocated(self%X44dd)) deallocate(self%X44dd)

            return

        end subroutine deallocate_shellopt_scratch

end module quick_scratch_module

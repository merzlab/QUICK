!
!	quick_scratch_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

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
        ! magic variables required for classopt subroutine
        double precision, dimension(:), allocatable :: X44,X44aa,X44bb,X44cc,X44dd  
    end type quick_scratch_type
    
    type (quick_scratch_type) quick_scratch
    
!    double precision, dimension(:,:), allocatable :: V2  !,hold,hold2
    
    interface alloc
        module procedure allocate_quick_scratch
    end interface alloc

    interface allocshellopt
        module procedure allocate_shellopt_scratch
    end interface allocshellopt

    interface dealloc
        module procedure deallocate_quick_scratch
    end interface dealloc 

    interface deallocshellopt
        module procedure deallocate_shellopt_scratch
    end interface deallocshellopt
    
    contains
        subroutine allocate_quick_scratch(self,nbasis)
            implicit none
            integer nbasis
            type (quick_scratch_type) self
            
            if(.not. allocated(self%hold)) allocate(self%hold(nbasis,nbasis))
            if(.not. allocated(self%hold2)) allocate(self%hold2(nbasis,nbasis))

            return
            
        end subroutine allocate_quick_scratch
        
        
        subroutine deallocate_quick_scratch(self)
            implicit none
            type (quick_scratch_type) self
            
            if (allocated(self%hold)) deallocate(self%hold)
            if (allocated(self%hold2)) deallocate(self%hold2)
            return
            
        end subroutine deallocate_quick_scratch

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

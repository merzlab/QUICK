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
    end type quick_scratch_type
    
    type (quick_scratch_type) quick_scratch
    
!    double precision, dimension(:,:), allocatable :: V2  !,hold,hold2
    
    interface alloc
        module procedure allocate_quick_scratch
    end interface alloc
    
    interface dealloc
        module procedure deallocate_quick_scratch
    end interface dealloc
    
    contains
        subroutine allocate_quick_scratch(self,nbasis)
            implicit none
            integer nbasis
            type (quick_scratch_type) self
            
            allocate(self%hold(nbasis,nbasis))
            allocate(self%hold2(nbasis,nbasis))

            return
            
        end subroutine allocate_quick_scratch
        
        
        subroutine deallocate_quick_scratch(self)
            implicit none
            type (quick_scratch_type) self
            
            deallocate(self%hold,self%hold2)

            return
            
        end subroutine deallocate_quick_scratch
        

end module quick_scratch_module

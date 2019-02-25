!
!	quick_gridpoints_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!  Grid Points Module
module quick_gridpoints_module

! The gridpoints arrays are fairly simple: XANG, YANG, and ZANG hold
! the angular grid points location for a unit sphere, and WTANG
! is the weights of those points.  RGRID and RWEIGHT are the positions
! and weights of the radial grid points, which in use are sclaed by the
! radii and radii^3 of the atoms.

    use quick_size_module
    implicit none

    double precision ::  XANG(MAXANGGRID),YANG(MAXANGGRID), &
    ZANG(MAXANGGRID),WTANG(MAXANGGRID),RGRID(MAXRADGRID), &
    RWT(MAXRADGRID)
    double precision,  dimension(:), allocatable :: sigrad2
    integer :: iradial(0:10), iangular(10),iregion
    
    contains
    
    ! allocate gridpoints
    subroutine allocate_quick_gridpoints(nbasis)
        implicit none
        integer nbasis
        allocate(sigrad2(nbasis))
    end subroutine allocate_quick_gridpoints

    ! deallocate
    subroutine deallocate_quick_gridpoints
        implicit none
        deallocate(sigrad2)
    end subroutine deallocate_quick_gridpoints
    
end module quick_gridpoints_module
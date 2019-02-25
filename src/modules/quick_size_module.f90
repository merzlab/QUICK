!
!	quick_size_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/17/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!  SIZE Module
module quick_size_module
    implicit none

    ! Maximum number of atoms:
    integer, parameter :: MAXATM = 1000

    ! Maximum number of basis functions (total):
    ! integer, parameter :: MAXBASIS = 110
    ! integer, parameter :: MAXBASIS3 = 3*MAXBASIS

    ! Maximum contraction of each basis function:
    ! integer, parameter :: MAXCONTRACT = 3

    ! Maximum number of angular and radial grid points for quadrature:
    integer, parameter :: MAXANGGRID = 6000
    integer, parameter :: MAXRADGRID = 400

    ! M value of the lbfgs optimizer.
    integer, parameter :: MLBFGS = 200
    
    ! Minimal iteration for SCF
    integer, parameter :: MIN_SCF = 3
    
    ! MAX DIIS CYCLE= MAX_DII_CYCLE_TIME* MAXDIICYC
    ! notice the difference between this and ISCF
    integer,parameter :: MAX_DII_CYCLE_TIME = 30

    integer,parameter :: MAXPRIM = 10
end module quick_size_module

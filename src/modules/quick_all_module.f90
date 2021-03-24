!
!	quick_all_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/17/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

! This file list and define varibles used as modules. 
! Modules List:
!    mfccmod        : MFCC Calculation
!    sizes          : Size Parameters
!    method         : Define Method Used
!    basis          : Basis Set Parameters
!    params         : Parameters for SEDFT
!    molspec        : Molecule Specifiction
!    gridpoints     : Grid points
!    calculated     : Overlap, Transformation matrix and energy
!    geocnverg      : Geometry Optimization parameters
!    SCRATCH        
!    files          : I/O file Specifiction
!    constants      : Constant
!    ecpmod         : ECP parameters
!    quickdc        : D&C Module
!    electrondensity: Electron Density
!    divpb_interface: Div_PB modules 
!    divpb_private  :
!    MPI_module     : MPI module
!    timer          : timer module
!    AMBER_interface_module
!                   : AMBER interface module

!********************************************************
!  Use all modules (not for DivPB)
!--------------------------------------------------------
! Stupid way of dealing with modules, should be replaced and each file
! should have individual module lists

    module allmod
    use quick_mfcc_module
    use quick_size_module
    use quick_method_module
    use quick_basis_module
    use quick_params_module
    use quick_molspec_module
    use quick_gridpoints_module
    use quick_calculated_module
    use quick_gaussian_class_module
    use quick_SCRATCH_module
    use quick_files_module
    use quick_constants_module
    use quick_ecp_module
    use quick_divcon_module
    use quick_electrondensity_module
    use quick_MPI_module
    use quick_timer_module
    implicit none
    end module allmod
!********************************************************

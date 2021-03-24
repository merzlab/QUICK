!
!	quick_divcon_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

!  D&C module,  Div & Con varibles
module quick_divcon_module

    implicit none

    integer, dimension(:,:), allocatable :: DCCore,DCBuffer1,DCBuffer2,DCSub,wtospoint
    integer, dimension(:), allocatable :: DCCoren,DCBuffer1n,DCBuffer2n,DCSubn,nBasisDC, &
                                          nElecDCSub,selectNN,nElecMP2Sub
    integer, dimension(:,:), allocatable :: DCOverlap,DCConnect
    integer, dimension(:), allocatable :: kShellS,kShellF
    integer, dimension(:,:,:), allocatable :: DCLogic
    double precision, dimension(:,:), allocatable :: invDCOverlap
    double precision, dimension(:,:,:), allocatable :: ODCSub,PDCSub,XDCSub,SMatrixDCSub, &
                                                       coDCSub,PDCSubtran,coDCSubtran
    double precision, dimension(:,:), allocatable :: ODCSubtemp,VECtemp
    double precision, dimension(:,:), allocatable :: Vtemp,EVEC1temp,eValDCSub
    double precision, dimension(:), allocatable :: EVAL1temp,IDEGEN1temp
    logical, dimension(:,:), allocatable :: disDivMFCC
    logical, dimension(:), allocatable :: mp2Shell
    integer np,NNMax,npsaved

    integer,allocatable::selectC(:),charge(:),spin(:)
    integer,allocatable::selectN(:),selectCA(:),ccharge(:)
    character*200 cmdstr
    
    ! MPI
    integer,allocatable:: mpi_dc_fragn(:)   ! frag no. calculated this node
    integer,allocatable:: mpi_dc_frag(:,:)  ! frag series no. calcluated in this node
    integer,allocatable:: mpi_dc_nbasis(:)  ! basis set this node has
    
    
end module quick_divcon_module

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
                                          nElecDCSub,selectNN,nElecMP2Sub,DCSubn1

    ! Reverse lookup, basis index -> (small, buffer-overlap-bounded) list of
    ! fragments containing it, with the LOCAL index within each such
    ! fragment's own ODCSub/PDCSub numbering. O(nbasis x BasisFragMaxOverlap):
    ! BasisFragMaxOverlap is bounded by how many fragments' buffer regions can
    ! overlap at one atom, not by np, so this does NOT grow quadratically with
    ! system size the way wtospoint (np x nbasis, used only by the DC-MP2
    ! path) would. Built once per geometry by BuildBasisFragMap (inidivcon.f90).
    integer, dimension(:,:), allocatable :: BasisFragList,BasisFragLocalIdx
    integer, dimension(:), allocatable :: BasisFragCount
    integer :: BasisFragMaxOverlap

    ! Prefix-sum offsets into the GPU's compact per-fragment-block O buffer
    ! (0-based: GPUFragOffset(0)=0, GPUFragOffset(itt)=GPUFragOffset(itt-1)+
    ! nBasisDC(itt)^2), dimensioned (0:np). Built once per geometry in
    ! inidivcon.f90 alongside BuildBasisFragMap, reused every SCF iteration
    ! by GatherCompactO (dnc_scf_operator.f90) without recomputing. Only
    ! meaningful for GPU, single-point DnC+DENSDIIS jobs -- see 9.14.
    integer, dimension(:), allocatable :: GPUFragOffset

    ! Whether the GPU build actually honored the compact-O request (false on
    ! a legacy-fixed-point-atomics or open-shell build -- see
    ! gpu_upload_fragmap_'s USE_LEGACY_ATOMICS/OSHELL guard in gpu.cu, which
    ! falls back to dense O on those builds). Read back once per geometry in
    ! inidivcon.f90 right after gpu_upload_fragmap; dnc_scf_operator.f90
    ! gates its GatherCompactO call on this, not on the upload request alone.
    logical :: GPUUseCompactO = .false.

    ! Host-side landing buffer for gpu_download_compact_o, sized
    ! GPUFragOffset(np) elements. Allocated once per geometry in
    ! inidivcon.f90 alongside GPUFragOffset, reused every SCF iteration by
    ! dnc_scf_operator.f90's gpu_download_compact_o + GatherCompactO calls.
    double precision, dimension(:), allocatable :: GPUCompactOBuf
    integer, dimension(:,:), allocatable :: DCOverlap,DCConnect
    integer, dimension(:), allocatable :: kShellS,kShellF
    integer, dimension(:,:,:), allocatable :: DCLogic
    double precision, dimension(:,:), allocatable :: invDCOverlap
    double precision, dimension(:,:,:), allocatable :: ODCSub,PDCSub,XDCSub,SMatrixDCSub, &
                                                       coDCSub,PDCSubtran,coDCSubtran,OneDCSub
    double precision, dimension(:,:), allocatable :: ODCSubtemp,VECtemp
    double precision, dimension(:,:,:), allocatable :: OBDCSub,PBDCSub,cobDCSub,PBDCSubtran, &
                                                       cobDCSubtran,OBDCSubtemp
    double precision, dimension(:,:), allocatable :: Vtemp,EVEC1temp,eValDCSub,eValbDCSub
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

!
!	quick_pb_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
! Following modules contains all global variables used in divpb.              
! -- Ning Liao 05/15/2004
! div PB modules includes:
! divpb_interface: 
! divpb_private:
!    contains: initialize_divpbVars()
!              deallocate_divpbVars(ierror)

#include "util.fh"

! PB Interface modules
  module divpb_interface
! 
!comment
!
! Interface for DIVPB, any subroute outside of DIVPB need to use this  
! module: "use divpb_interface"                                        
!
!comment_end

  implicit none
  save

  ! original divcon variables, different name
  integer:: gnIscr, gnIout
  logical:: gbWater, gbOctanol
  logical:: gbScreen
  integer:: iiixiao

  ! the return of PB calculation - Reaction field energy & nonpolar energy
  real:: grGrf, grGnp,grGnpxiao

  ! the return of PB calculation - suface chargs & their coordinates
  ! will be used in other divcon modules (mostly pbfock)
  real, dimension(:,:), allocatable:: grpSchrgPos
  real, dimension(:), allocatable:: grpSchrg
  ! real, dimension(:,:), pointer:: grpSchrgPos
  ! real, dimension(:), pointer:: grpSchrg
  integer:: gnSchrgNum
  ! common/divpbSchrg/ grpSchrgPos, grpSchrg, gnSchrgNum

  ! keywords read in by divcon's rdkeys.F
  real :: grDielecIn, grDielecOut
  real :: grProbeRadius, grGridPerAng, grPerFill, grIonStrength
  logical :: gbUseDivpb
  logical :: gbFineSurf
  integer :: gnPBDump
  real :: grUsrCentX, grUsrCentY, grUsrCentZ
  integer :: gnUsrGridNum
  real :: grESPCentX, grESPCentY, grESPCentZ, grESPEdge
  logical:: gbPBInstant

  ! these are used to determine whether assigned value in divcon input or need
  ! to set default values in setdef.F
  data gbUseDivpb /.true./
  data gbFineSurf /.false./
  data gbPBInstant /.false./

  data grDielecIn /-1.0/
  data grDielecOut /-1.0/
  data grProbeRadius /-1.0/
  data grGridPerAng /-1.0/
  data grPerFill /-1.0/
  data grIonStrength /-1.0/
  data gnPBDump /0/

  data grUsrCentX/12.345/
  data grUsrCentY/12.345/
  data grUsrCentZ/12.345/
  data gnUsrGridNum/12345/

  data grESPCentX/12.345/
  data grESPCentY/12.345/
  data grESPCentZ/12.345/
  data grESPEdge/12.345/
end module divpb_interface
!
!----------------------------------------------------------------------
! Private modules
!----------------------------------------------------------------------

module divpb_private

  !comment
  !
  ! Private global data for DIVPB use only          
  !
  !comment_end


  implicit none
  save


! math parameters
!      double precision, parameter:: PI= 3.1415926535897932384626434d0

! total allowable grid points = MAX_GRID**3
      integer, parameter:: MAX_GRID=500
! file name to read radii of different atoms in
      character(len=20), parameter:: RADII_FILE='divcon.siz'
! file name to get parameters for divpb
      character(len=20), parameter:: PARAM_FILE='divpb.prm'
! file name to output relaxed electronic potential
      character(len=20), parameter:: POTENTIAL_FILE_TXT='dcqtp_pot.txt'
      character(len=20), parameter:: POTENTIAL_FILE_DX='dcqtp_pot.dx'

! paramter for dpwritepot
      integer, parameter:: PBDUMP_TXT=1, PBDUMP_DX=2

! paramter for dpmakesurface
      integer, parameter:: VDW_SURF=1, MS_SURF=2

! paramter for dpinterpolate, not used in this version of divpb module
!      integer, parameter:: INTPOL_START=1, INTPOL_doINT=0, INTPOL_END=-1

! Normally the maxium allowed iteration used in dploop
      integer, parameter:: MAX_FDM_LOOP=2000

! Max different radii
      integer, parameter:: MAX_RADII=300



  ! Molecular information
  character(len=15), dimension(:), pointer::gspMolInfo
  real, dimension(:,:), pointer:: grpMolPos
  real, dimension(:), pointer:: grpMolChrg, grpMolRadii
  real:: graMolSpanMid(3)
  integer :: gnMolAtomNum
  ! common/divpbMol/ gspMolInfo,grpMolPos,grpMolChrg,grpMolRadii,
  ! .              graMolSpanMid,gnMolAtomNum

  ! Grid information
  ! grpGridPhi(1:gnGridNum, 1:gnGridNum, 1:gnGridNum)
  ! gnpGridkappa(1:gnGridNum, 1:gnGridNum, 1:gnGridNum)
  ! grpGridEpsilon(1:gnGridNum, 1:gnGridNum, 1:gnGridNum, 1:3)
  ! grpGridMol(1:3, 1:gnMolAtomNum)
  ! gnpGridBndry(1:3, 1:gnGridBndryNum)
  ! grpGridChrg(1:gnGridChrgNum)
  ! gnpGridChrgXYZ(1:3, 1:gnGridChrgNum)
  real, dimension(:,:,:), pointer:: grpGridPhi
  integer, dimension(:,:,:), pointer:: gnpGridKappa
  real, dimension(:,:,:,:), pointer:: grpGridEpsilon
  real, dimension(:,:), pointer:: grpGridMol
  integer, dimension(:,:), pointer:: gnpGridBndry
  integer, dimension(:,:), pointer:: gnpGridChrgXYZ
  real, dimension(:), pointer:: grpGridChrg
  integer:: gnGridNum, gnGridBndryNum, gnGridChrgNum
  ! common/divpbGrid/ grpGridPhi,gnpGridKappa,
  ! .               grpGridEpsilon,grpGridMol,gnpGridBndry,
  ! .               grpGridChrg, gnpGridChrgXYZ,
  ! .               gnGridNum,gnGridBndryNum, gnGridChrgNum

  ! Surface information
  ! grpSurfExtVdw(1:3, 1:gnSurfExtVdwNum) coordinates of extended vdw surface pnts
  ! grpSurfMS(1:3, 1:gnSurfMSNum) coordinates of molecular surface pnts
  real, dimension(:,:), pointer:: grpSurfExtVdw, grpSurfMS
  real, dimension(:), pointer:: grpSurfMSArea
  integer:: gnSurfExtVdwNum, gnSurfMSNum, gnSurfType
  logical:: gbSurfFirstSpt
  ! common/divpbSurf/ grpSurfExtVdw, grpSurfMS, grpSurfMSArea,
  ! .                  gnSurfExtVdwNum, gnSurfMSNum, gnSurfType,
  ! .                  gbSurfFirstSpt


  ! arrays to accelerate relaxation of PB equation.
  ! grpAccEpsRes(0:6, 1:gnGridBndryNum) the residue (due to the un-unified epsilon)
  ! add up to the laplace version of PB.
  ! grpAccChrgRes(1:gnGridChrgNum) the residue due to the charge
  ! grpAccKappaLap(1:gnGridNum,1:gnGridNum,1:gnGridNum) the laplace equation counting
  ! in the kappa item in the salt solution.
  real, dimension(:,:,:), pointer:: grpAccEpsRes
  real, dimension(:,:), pointer:: grpAccChrgRes
  real, dimension(:), pointer:: grpAccPhiOdd,grpAccKappaOdd
  real, dimension(:), pointer:: grpAccPhiEven,grpAccKappaEven
  real, dimension(:,:), pointer:: grpAccBoxBndry
  integer, dimension(:,:), pointer:: gnpAccEpsIndx
  integer, dimension(:,:), pointer:: gnpAccChrgIndx
  integer, dimension(:,:), pointer:: gnpAccBoxBndryIndx
  integer, dimension(:,:), pointer:: gnpAccBndryChrgIndx
  integer:: gnAccBndryChrgNum
  ! common/divpbAcc/ grpAccEpsRes, grpAccChrgRes,
  ! .                grpAccKappaOdd, grpAccKappaEven,
  ! .                grpAccPhiOdd, grpAccPhiEven,grpAccBoxBndry,
  ! .                gnpAccEpsIndx, gnpAccChrgIndx,
  ! .                gnpAccBoxBndryIndx, gnpAccBndryChrgIndx,
  ! .                gnAccBndryChrgNum

  ! arrays to use finer grid at the dielectric boundary
  logical :: gbFSNow
  integer :: gnDblGridChrgNum
  real, dimension(:,:), pointer:: grpDblGridMol
  real, dimension(:), pointer:: grpDblGridChrg
  integer, dimension(:,:), pointer:: gnpDblGridChrgXYZ
  real, dimension(:,:,:), pointer:: grpDblPhi
  integer, dimension(:,:), pointer:: gnpFSLocalPnts
  integer, dimension(:,:), pointer:: gnpFSLocalChrgs

  real, dimension(:,:,:), pointer:: grpFSAccEpsRes
  integer, dimension(:,:), pointer:: gnpFSAccEpsIndx
  real, dimension(:,:), pointer:: grpFSAccChrgRes
  integer, dimension(:,:), pointer:: gnpFSAccChrgIndx
  real, dimension(:), pointer:: grpFSAccKappaOdd, grpFSAccKappaEven
  integer, dimension(:,:), pointer:: gnpFSAccBoxBndryIndx
  real, dimension(:,:), pointer:: grpFSAccBoxBndry
  integer :: gnFSAccBndryChrgNum
  integer, dimension(:,:), pointer:: gnpFSAccBndryChrgIndx
  real, dimension(:), pointer:: grpFSAccPhiOdd, grpFSAccPhiEven

  ! arrays for radii information
  character(len=2):: gsaRadAtomName(MAX_RADII)
  real:: graRadAtomRadii(MAX_RADII)
  integer:: gnRadAtomNum

  ! radii information for atoms used in SCRF caculation
  ! hhk : moved to the routine, initialize_divpbVars()

  ! hhk Below is added to control variables which have 'save' attribute and
  ! need to be cleaned for each Divcon run job.

  ! Used only in pbsolver.F
  logical :: bFirst
  integer :: count

  ! Used only in divpb.F
  logical :: bFirstRun_divpb
  real :: rSpectral_divpb

  ! Used only in dpchargegrid.F
  logical :: bFirstRun_dpchrg
  real :: rLastChrg

  ! Used only in dpsetepsilon.F
  integer, dimension(:,:), pointer :: npBndry1, npBndry2
  logical :: bUseBndry1
  integer :: nAllocated

  ! Used only in pb_spt.F
  real, dimension(:,:), pointer :: rpSurfVdw1, rpSurfVdw2
  real, dimension(:,:), pointer :: rpSurfMS1, rpSurfMS2
  real, dimension(:), pointer :: rpSurfMSArea1, rpSurfMSArea2
  logical :: bUseVdw1, bUseMS1
  integer :: nVdwAllocated, nMSAllocated

  ! Used only in pb_putpnt.F
  logical :: first_pb_putpnt


CONTAINS

  !========================================================================

  subroutine initialize_divpbVars()

  !comment
  !
  ! Initialize variables (used in DivPB calculation) which need to be
  ! fresh ones for multiple runs of divcon jobs (NOT divpb jobs).
  !
  !comment_end


    gnMolAtomNum = 0

    ! Used only in pbsolver.F
    bFirst = .TRUE.
    count = 0

    ! Used only in divpb.F
    bFirstRun_divpb = .TRUE.
    rSpectral_divpb = 0.0

    ! Used only in dpchargegrid.F
    bFirstRun_dpchrg = .TRUE.
    rLastChrg = 0.0

    ! Used only in dpsetepsilon.F
    bUseBndry1 = .TRUE.
    nAllocated = 0

    ! Used only in pb_spt.F
    bUseVdw1 = .FALSE.
    bUseMS1 = .FALSE.
    nVdwAllocated = 0
    nMSAllocated = 0

    ! Used only in pb_putpnt.F
    first_pb_putpnt = .TRUE.


    ! radii information for atoms used in SCRF caculation
    gnRadAtomNum = 83

    gsaRadAtomName(1) = 'H '
    graRadAtomRadii(1) = 1.150

    gsaRadAtomName(2) = 'HE'
    graRadAtomRadii(2) = 1.181

    gsaRadAtomName(3) = 'LI'
    graRadAtomRadii(3) = 1.226

    gsaRadAtomName(4) = 'BE'
    graRadAtomRadii(4) = 1.373

    gsaRadAtomName(5) = 'B '
    graRadAtomRadii(5) = 2.042

    gsaRadAtomName(6) = 'C '
    graRadAtomRadii(6) = 1.900

    gsaRadAtomName(7) = 'N '
    graRadAtomRadii(7) = 1.600

    gsaRadAtomName(8) = 'O '
    graRadAtomRadii(8) = 1.600

    gsaRadAtomName(9) = 'F '
    graRadAtomRadii(9) = 1.682

    gsaRadAtomName(10) = 'NE'
    graRadAtomRadii(10) = 1.621

    gsaRadAtomName(11) = 'NA'
    graRadAtomRadii(11) = 1.491

    gsaRadAtomName(12) = 'MG'
    graRadAtomRadii(12) = 1.510

    gsaRadAtomName(13) = 'AL'
    graRadAtomRadii(13) = 2.249

    gsaRadAtomName(14) = 'SI'
    graRadAtomRadii(14) = 2.147

    gsaRadAtomName(15) = 'P '
    graRadAtomRadii(15) = 2.074

    gsaRadAtomName(16) = 'S '
    graRadAtomRadii(16) = 1.900

    gsaRadAtomName(17) = 'CL'
    graRadAtomRadii(17) = 1.974

    gsaRadAtomName(18) = 'AR'
    graRadAtomRadii(18) = 1.934

    gsaRadAtomName(19) = 'K '
    graRadAtomRadii(19) = 1.906

    gsaRadAtomName(20) = 'CA'
    graRadAtomRadii(20) = 1.700

    gsaRadAtomName(21) = 'SC'
    graRadAtomRadii(21) = 1.647

    gsaRadAtomName(22) = 'TI'
    graRadAtomRadii(22) = 1.587

    gsaRadAtomName(23) = 'V '
    graRadAtomRadii(23) = 1.572

    gsaRadAtomName(24) = 'CR'
    graRadAtomRadii(24) = 1.511

    gsaRadAtomName(25) = 'MN'
    graRadAtomRadii(25) = 1.480

    gsaRadAtomName(26) = 'FE'
    graRadAtomRadii(26) = 1.456

    gsaRadAtomName(27) = 'CO'
    graRadAtomRadii(27) = 1.436

    gsaRadAtomName(28) = 'NI'
    graRadAtomRadii(28) = 1.417

    gsaRadAtomName(29) = 'CU'
    graRadAtomRadii(29) = 1.748

    gsaRadAtomName(30) = 'ZN'
    graRadAtomRadii(30) = 1.381

    gsaRadAtomName(31) = 'GA'
    graRadAtomRadii(31) = 2.192

    gsaRadAtomName(32) = 'GE'
    graRadAtomRadii(32) = 2.140

    gsaRadAtomName(33) = 'AS'
    graRadAtomRadii(33) = 2.115

    gsaRadAtomName(34) = 'SE'
    graRadAtomRadii(34) = 2.103

    gsaRadAtomName(35) = 'BR'
    graRadAtomRadii(35) = 2.095

    gsaRadAtomName(36) = 'KR'
    graRadAtomRadii(36) = 2.071

    gsaRadAtomName(37) = 'RB'
    graRadAtomRadii(37) = 2.057

    gsaRadAtomName(38) = 'SR'
    graRadAtomRadii(38) = 1.821

    gsaRadAtomName(39) = 'Y '
    graRadAtomRadii(39) = 1.673

    gsaRadAtomName(40) = 'ZR'
    graRadAtomRadii(40) = 1.562

    gsaRadAtomName(41) = 'NB'
    graRadAtomRadii(41) = 1.583

    gsaRadAtomName(42) = 'MO'
    graRadAtomRadii(42) = 1.526

    gsaRadAtomName(43) = 'TC'
    graRadAtomRadii(43) = 1.499

    gsaRadAtomName(44) = 'RU'
    graRadAtomRadii(44) = 1.481

    gsaRadAtomName(45) = 'RH'
    graRadAtomRadii(45) = 1.464

    gsaRadAtomName(46) = 'PD'
    graRadAtomRadii(46) = 1.450

    gsaRadAtomName(47) = 'AG'
    graRadAtomRadii(47) = 1.574

    gsaRadAtomName(48) = 'CD'
    graRadAtomRadii(48) = 1.424

    gsaRadAtomName(49) = 'IN'
    graRadAtomRadii(49) = 2.232

    gsaRadAtomName(50) = 'SN'
    graRadAtomRadii(50) = 2.196

    gsaRadAtomName(51) = 'SB'
    graRadAtomRadii(51) = 2.210

    gsaRadAtomName(52) = 'TE'
    graRadAtomRadii(52) = 2.235

    gsaRadAtomName(53) = 'I '
    graRadAtomRadii(53) = 2.250

    gsaRadAtomName(54) = 'XE'
    graRadAtomRadii(54) = 2.202

    gsaRadAtomName(55) = 'CS'
    graRadAtomRadii(55) = 2.259

    gsaRadAtomName(56) = 'BA'
    graRadAtomRadii(56) = 1.851

    gsaRadAtomName(57) = 'LA'
    graRadAtomRadii(57) = 1.761

    gsaRadAtomName(72) = 'HF'
    graRadAtomRadii(72) = 1.570

    gsaRadAtomName(73) = 'TA'
    graRadAtomRadii(73) = 1.585

    gsaRadAtomName(74) = 'W '
    graRadAtomRadii(74) = 1.534

    gsaRadAtomName(75) = 'RE'
    graRadAtomRadii(75) = 1.477

    gsaRadAtomName(76) = 'OS'
    graRadAtomRadii(76) = 1.560

    gsaRadAtomName(77) = 'IR'
    graRadAtomRadii(77) = 1.420

    gsaRadAtomName(78) = 'PT'
    graRadAtomRadii(78) = 1.377

    gsaRadAtomName(79) = 'AU'
    graRadAtomRadii(79) = 1.647

    gsaRadAtomName(80) = 'HG'
    graRadAtomRadii(80) = 1.353

    gsaRadAtomName(81) = 'TL'
    graRadAtomRadii(81) = 2.174

    gsaRadAtomName(82) = 'PB'
    graRadAtomRadii(82) = 2.148

    gsaRadAtomName(83) = 'BI'
    graRadAtomRadii(83) = 2.185

  end subroutine initialize_divpbVars

  !========================================================================

  subroutine deallocate_divpbVars(ierror)

  !comment
  !
  ! Deallocate dynamic arrays used in DivPB calculation.
  !
  !comment_end

    use divpb_interface

    IMPLICIT NONE

    INTEGER :: iDeallocateErr, ierror

    ierror = 0

    if (allocated(grpSchrgPos)) then
       deallocate(grpSchrgPos, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (allocated(grpSchrg)) then
       deallocate(grpSchrg, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(gspMolInfo)) then
       deallocate(gspMolInfo, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpMolPos)) then
       deallocate(grpMolPos, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpMolChrg)) then
       deallocate(grpMolChrg, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpMolRadii)) then
       deallocate(grpMolRadii, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpGridPhi)) then
       deallocate(grpGridPhi, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(gnpGridKappa)) then
       deallocate(gnpGridKappa, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpGridEpsilon)) then
       deallocate(grpGridEpsilon, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpGridMol)) then
       deallocate(grpGridMol, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(gnpGridChrgXYZ)) then
       deallocate(gnpGridChrgXYZ, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpGridChrg)) then
       deallocate(grpGridChrg, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif

    ! These were not allocated but pointed to other allocated pointers.
    if (associated(gnpGridBndry)) then
       nullify(gnpGridBndry)
    endif
    if (associated(grpSurfExtVdw)) then
       nullify(grpSurfExtVdw)
    endif
    if (associated(grpSurfMS)) then
       nullify(grpSurfMS)
    endif
    if (associated(grpSurfMSArea)) then
       nullify(grpSurfMSArea)
    endif

    if (associated(grpAccEpsRes)) then
       deallocate(grpAccEpsRes, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpAccChrgRes)) then
       deallocate(grpAccChrgRes, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpAccPhiOdd)) then
       deallocate(grpAccPhiOdd, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpAccKappaOdd)) then
       deallocate(grpAccKappaOdd, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpAccPhiEven)) then
       deallocate(grpAccPhiEven, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpAccKappaEven)) then
       deallocate(grpAccKappaEven, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(grpAccBoxBndry)) then
       deallocate(grpAccBoxBndry, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(gnpAccEpsIndx)) then
       deallocate(gnpAccEpsIndx, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(gnpAccChrgIndx)) then
       deallocate(gnpAccChrgIndx, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(gnpAccBoxBndryIndx)) then
       deallocate(gnpAccBoxBndryIndx, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(gnpAccBndryChrgIndx)) then
       deallocate(gnpAccBndryChrgIndx, stat=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif

    ! Used only in dpsetepsilon.F

    if (associated(npBndry1)) then
       DEALLOCATE(npBndry1, STAT=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(npBndry2)) then
       DEALLOCATE(npBndry2, STAT=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif

    ! Used only in pb_spt.F
    if (associated(rpSurfVdw1)) then
       DEALLOCATE(rpSurfVdw1, STAT=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(rpSurfVdw2)) then
       DEALLOCATE(rpSurfVdw2, STAT=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(rpSurfMS1)) then
       DEALLOCATE(rpSurfMS1, STAT=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(rpSurfMS2)) then
       DEALLOCATE(rpSurfMS2, STAT=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(rpSurfMSArea1)) then
       DEALLOCATE(rpSurfMSArea1, STAT=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif
    if (associated(rpSurfMSArea2)) then
       DEALLOCATE(rpSurfMSArea2, STAT=iDeallocateErr)
       if (iDeallocateErr /= 0) then
          ierror = 1
          return
       endif
    endif

    return

  end subroutine deallocate_divpbVars

end module divpb_private
!********************************************************

#include "util.fh"
!---------------------------------------------------------------------!
! Created by Madu Manathunga on 04/23/2021                            !
!                                                                     ! 
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This module contains source code required for computing long range  !
! interactions in QM/MM.                                              !
!_____________________________________________________________________!

module quick_long_range_module

  implicit none
  private

contains

  subroutine compute_long_range

    !----------------------------------------------------------------------!
    ! This is the main driver for computing long range interactions. The   !
    ! goal is to compute (ij|c) three center integral and add a potential  !
    ! into Fock matrix. Here i,j are two basis functions, c is a gaussian  !
    ! located at a certain distance. To make use of the existing ERI code, !
    ! we approximate above integral with (ij|c0) four center integral where!
    ! 0 is another gaussian with zero exponent.                            !
    !______________________________________________________________________!

    use quick_basis_module

    implicit none
    integer :: II, JJ ! shell pairs
    integer :: nc, Cc ! number of external gaussian to loop through, index     
    double precision :: Zc    ! charge 
    double precision :: Rc(3) ! cartesian coordinates of the center of external gaussian

    ! set number of external gaussians and charge for testing
    nc=1 
    Zc= 1.0d0
    Rc=0.0d0    

    do II = 1, jshell
      do JJ = II, jshell  
        do Cc = 1, nc 
          call compute_lngr_int(II,JJ,Cc,Zc,Rc)
        enddo
      enddo
    enddo

  end subroutine compute_long_range


  subroutine compute_lngr_int(II,JJ,KK,Zc,RC)

    use quick_basis_module
    integer, intent(in) :: II, JJ, KK
    double precision, intent(in) :: Zc, RC(3)

    integer :: M, LL, NII1, NII2, NJJ1, NJJ2, NKK1, NKK2, NLL1, NLL2, NNAB, NNCD, NABCDTYPE, &
               NNA, NNC, NABCD, ITT, JJJ, III
    double precision :: RA(3),RB(3),RD(3)

    ! set the coordinates of shell centers
    do M=1,3
      RA(M)=xyz(M,quick_basis%katom(II))
      RB(M)=xyz(M,quick_basis%katom(JJ))
    enddo

    ! set the center of null gaussian
    RD=0.0d0

    ! Get angular momentum quantum number for each shell
    ! s=0~0, p=1~1, sp=0~1, d=2~2, f=3~3
    NII1=quick_basis%Qstart(II)
    NII2=quick_basis%Qfinal(II)
    NJJ1=quick_basis%Qstart(JJ)
    NJJ2=quick_basis%Qfinal(JJ)
    NKK1=0
    NKK2=0
    NLL1=0
    NLL2=0

    NNAB=(NII2+NJJ2)
    NNCD=(NKK2+NLL2)

    NABCDTYPE=NNAB*10+NNCD 

    NNAB=sumindex(NNAB)
    NNCD=sumindex(NNCD)
    NNA=sumindex(NII1-1)+1
    NNC=sumindex(NKK1-1)+1    

    !The summation of the highest angular momentum number of each shell
    NABCD=NII2+NJJ2+NKK2+NLL2
    ITT=0

    

  end subroutine compute_lngr_int

end module quick_long_range_module


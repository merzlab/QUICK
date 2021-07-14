!---------------------------------------------------------------------!
! Created by Madu Manathunga on 04/27/2021                            !
!                                                                     !
! Previous contributors: Xiao He                                      !
!                                                                     ! 
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This header file contains horizontal recurrence relations code      !
! necessary for computing 3 center integrals found in long range      !
! corrected QM/MM.                                                    !
!_____________________________________________________________________!

subroutine hrr_tci

  use quick_basis_module
  use quick_params_module

  implicit none

  integer :: NA(3),NB(3)
  integer :: M,M1,M2,M3,MA,MB,MAB,i,itemp

  double precision :: coefangL(20)
  integer :: angL(20),numangularL

  select case (IJKLtype)
    case (0,1000)   ! (ss|ss), (ps|ss) integrals
       M1=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
       M3=1        
       Y=store(M1,M3)

    case (2000)     ! (ps|ss) integrals
       M1=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
       M3=1         
       Y=store(M1,M3)
 
       Y=Y*quick_basis%cons(III)*quick_basis%cons(JJJ)

       !write(*,*) "Case 2000: ", Y, quick_basis%cons(III),quick_basis%cons(JJJ),&
       !Y/(quick_basis%cons(III)*quick_basis%cons(JJJ))

    case(100)       ! (sp|ss) integrals
       do M=1,3
          NB(M)=quick_basis%KLMN(M,JJJ)
       enddo
       M1=trans(NB(1),NB(2),NB(3))
       do itemp=1,3
          if(NB(itemp).ne.0)then
             Y=store(M1,1)+(RA(itemp)-RB(itemp))*store(1,1)
             goto 111
          endif
       enddo

    case(1100)      ! (pp|ss) integrals
       do M=1,3
          NA(M)=quick_basis%KLMN(M,III)
          NB(M)=quick_basis%KLMN(M,JJJ)
       enddo
       MA=trans(NA(1),NA(2),NA(3))
       MAB=trans(NA(1)+NB(1),NA(2)+NB(2),NA(3)+NB(3))
       do itemp=1,3
          if(NB(itemp).ne.0)then
             Y=store(MAB,1)+(RA(itemp)-RB(itemp))*store(MA,1)
             goto 111
          endif
       enddo

    case(999)       ! remaining (ij|ss) integrals
 
       call lefthrr(RA,RB,quick_basis%KLMN(1:3,III),quick_basis%KLMN(1:3,JJJ),IJtype,coefangL,angL,numangularL)
 
       Y=0.0d0
       do i=1,numangularL
         Y=Y+coefangL(i)*store(angL(i),1)
       enddo

       Y=Y*quick_basis%cons(III)*quick_basis%cons(JJJ)

       !write(*,*) "Case 999: ", Y,quick_basis%cons(III),quick_basis%cons(JJJ),&
       !Y/(quick_basis%cons(III)*quick_basis%cons(JJJ))

  end select
  111 continue
end subroutine hrr_tci


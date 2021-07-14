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
! necessary for computing 3 center integral gradients found in long   !
! range corrected QM/MM.                                              !
!_____________________________________________________________________!

subroutine hrr_tci_grad

  use quick_basis_module
  use quick_params_module   

  implicit none

  integer :: NA(3),NB(3),NC(3),ND(3),angL(20),angR(20),angLnew(20),angRnew(20)
  integer :: M1,M2,M3,M4,numangularL,numangularR,numangularLnew,numangularRnew,itemp,itempxiao, &
             i,jxiao     

  double precision :: coefangL(20),coefangR(20),coefangLnew(20),coefangRnew(20)
  double precision :: tempconstant

  tempconstant=1.0d0
  tempconstant=quick_basis%cons(III)*quick_basis%cons(JJJ)

  call lefthrr(RA,RB,quick_basis%KLMN(1:3,III),quick_basis%KLMN(1:3,JJJ),IJtype,coefangL,angL,numangularL)

  ! Set values for the ket. Since external gaussian is always a s function, we
  ! set the following values.
  numangularR=1
  angR(1)=1
  coefangR(1)=1.0d0

  ! Gradients with respect to first center (atom A)
  do itemp=1,3
     do itempxiao=1,3
        NA(itempxiao)=quick_basis%KLMN(itempxiao,III)
     enddo

     NA(itemp)=quick_basis%KLMN(itemp,III)+1

     call lefthrr(RA,RB,NA(1:3),quick_basis%KLMN(1:3,JJJ),IJtype+10,coefangLnew,angLnew,numangularLnew)
     Yaa(itemp)=0.0d0
     do i=1,numangularLnew
        do jxiao=1,numangularR
           Yaa(itemp)=Yaa(itemp)+coefangLnew(i)*coefangR(jxiao)* &
                 storeAA(angLnew(i),angR(jxiao))
        enddo
     enddo


     if(quick_basis%KLMN(itemp,III).ge.1)then

        do itempxiao=1,3
           NA(itempxiao)=quick_basis%KLMN(itempxiao,III)
        enddo

        NA(itemp)=quick_basis%KLMN(itemp,III)-1

        call lefthrr(RA,RB,NA(1:3),quick_basis%KLMN(1:3,JJJ),IJtype-10,coefangLnew,angLnew,numangularLnew)
        do i=1,numangularLnew
           do jxiao=1,numangularR
              Yaa(itemp)=Yaa(itemp)-quick_basis%KLMN(itemp,III)*coefangLnew(i)* &
                    coefangR(jxiao)*store(angLnew(i),angR(jxiao))
           enddo
        enddo
     endif

     Yaa(itemp)=Yaa(itemp)*tempconstant
  enddo

  ! Gradients with respect to second center (atom B)
  do itemp=1,3
     do itempxiao=1,3
        NA(itempxiao)=quick_basis%KLMN(itempxiao,III)
        NB(itempxiao)=quick_basis%KLMN(itempxiao,JJJ)
     enddo
     NB(itemp)=quick_basis%KLMN(itemp,JJJ)+1

     call lefthrr(RA,RB,NA(1:3),NB(1:3),IJtype+1,coefangLnew,angLnew,numangularLnew)

     Ybb(itemp)=0.0d0
     do i=1,numangularLnew
        do jxiao=1,numangularR
           Ybb(itemp)=Ybb(itemp)+coefangLnew(i)*coefangR(jxiao)* &
                 storeBB(angLnew(i),angR(jxiao))
        enddo
     enddo

     if(quick_basis%KLMN(itemp,JJJ).ge.1)then

        do itempxiao=1,3
           NA(itempxiao)=quick_basis%KLMN(itempxiao,III)
           NB(itempxiao)=quick_basis%KLMN(itempxiao,JJJ)
        enddo

        NB(itemp)=quick_basis%KLMN(itemp,JJJ)-1

        call lefthrr(RA,RB,NA(1:3),NB(1:3),IJtype-1,coefangLnew,angLnew,numangularLnew)

        do i=1,numangularLnew
           do jxiao=1,numangularR
              Ybb(itemp)=Ybb(itemp)-quick_basis%KLMN(itemp,JJJ)*coefangLnew(i)* &
                    coefangR(jxiao)*store(angLnew(i),angR(jxiao))
           enddo
        enddo
     endif

     Ybb(itemp)=Ybb(itemp)*tempconstant
  enddo

  ! Gradients with respect to third center (atom C) can be obtained from
  ! translational invariance. 
  do itemp=1,3
    Ycc(itemp) = -1.0d0*(Yaa(itemp)+Ybb(itemp))
  enddo

100 continue

end subroutine hrr_tci_grad


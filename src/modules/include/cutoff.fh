!---------------------------------------------------------------------!
! Created by Madu Manathunga on 01/25/2021                            !
!                                                                     !
! Previous contributors: Yipu Miao, Xio He, Alessandro Genoni,        !
!                         Ken Ayers & Ed Brothers                     !
!                                                                     !
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

#ifdef OSHELL
subroutine oshell_dnscreen(II,JJ,DNmax1)
#else
subroutine cshell_dnscreen(II,JJ,DNmax1)
#endif

  use allmod

  Implicit double precision(a-h,o-z)

  NII1=quick_basis%Qstart(II)
  NII2=quick_basis%Qfinal(II)
  NJJ1=quick_basis%Qstart(JJ)
  NJJ2=quick_basis%Qfinal(JJ)

  NBI1=quick_basis%Qsbasis(II,NII1)
  NBI2=quick_basis%Qfbasis(II,NII2)
  NBJ1=quick_basis%Qsbasis(JJ,NJJ1)
  NBJ2=quick_basis%Qfbasis(JJ,NJJ2)

  II111=quick_basis%ksumtype(II)+NBI1
  II112=quick_basis%ksumtype(II)+NBI2
  JJ111=quick_basis%ksumtype(JJ)+NBJ1
  JJ112=quick_basis%ksumtype(JJ)+NBJ2

  do III=II111,II112
     do JJJ=JJ111,JJ112
#ifdef OSHELL
        DENSEJI=dabs(quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III))
        DENSEJIA=dabs(quick_qm_struct%dense(JJJ,III))
        DENSEJIB=dabs(quick_qm_struct%denseb(JJJ,III))
        DENSEMAX=max(DENSEJI,DENSEJIA,DENSEJIB)
        if(DENSEMAX.gt.DNmax1)DNmax1=DENSEMAX
#else
        DENSEJI=dabs(quick_qm_struct%dense(JJJ,III))
        if(DENSEJI.gt.DNmax1)DNmax1=DENSEJI
#endif
     enddo
  enddo

#ifdef OSHELL
end subroutine oshell_dnscreen
#else
end subroutine cshell_dnscreen
#endif


#ifdef OSHELL
subroutine oshell_density_cutoff
#else
subroutine cshell_density_cutoff
#endif

   !------------------------------------------------
   ! This subroutine is to cutoff delta density
   !------------------------------------------------
   use allmod
   implicit double precision(a-h,o-z)

   ! Cutmatrix(II,JJ) indicated for ii shell and jj shell, the max dense
   do II=1,jshell
      do JJ=II,jshell
         DNtemp=0.0d0
#ifdef OSHELL
         call oshell_dnscreen(II,JJ,DNtemp)
#else
         call cshell_dnscreen(II,JJ,DNtemp)
#endif
         Cutmatrix(II,JJ)=DNtemp
         Cutmatrix(JJ,II)=DNtemp
      enddo
   enddo

#ifdef OSHELL
end subroutine oshell_density_cutoff
#else
end subroutine cshell_density_cutoff
#endif


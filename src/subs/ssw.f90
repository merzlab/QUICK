#include "util.fh"
!
!	ssw.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!   Written by Ed Brothers. January 22, 2002

double precision function ssw(gridx,gridy,gridz,iparent)
  use allmod
  implicit double precision(a-h,o-z)

  ! This subroutie calculates the Scuseria-Stratmann wieghts.  There are
  ! two conditions that cause the weights to be unity: If there is only
  ! one atom:

  if (natom == 1)  then
     ssw=1.d0
     return
  endif

  ! Another time the weight is unity is r(iparent,g)<.5*(1-a)*R(i,n)
  ! where r(iparent,g) is the distance from the parent atom to the grid
  ! point, a is a parameter (=.64) and R(i,n) is the distance from the
  ! parent atom to it's nearest neighbor.

  xparent=xyz(1,iparent)
  yparent=xyz(2,iparent)
  zparent=xyz(3,iparent)

  rig=(gridx-xparent)*(gridx-xparent)
  rig=rig+(gridy-yparent)*(gridy-yparent)
  rig=rig+(gridz-zparent)*(gridz-zparent)
  rig=Dsqrt(rig)

  if (rig < 0.18d0*quick_molspec%distnbor(iparent)) then
     ssw=1.d0
     return
  endif

  ! If neither of those are the case, we have to actually calculate the
  ! weight.  First we must calculate the unnormalized wieght of the grid point
  ! with respect to the parent atom.

  ! Step one of calculating the unnormalized weight is finding the confocal
  ! elliptical coordinate between each cell.  This it the mu with subscripted
  ! i and j in the paper:
  ! Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
  ! 1996, pg 213-223.

  wofparent=1.d0
  Jatm=1
  do while (Jatm.ne.iparent.and.wofparent.ne.0.d0)
     xJatm=xyz(1,Jatm)
     yJatm=xyz(2,Jatm)
     zJatm=xyz(3,Jatm)
     rjg=(gridx-xJatm)*(gridx-xJatm)
     rjg=rjg+(gridy-yJatm)*(gridy-yJatm)
     rjg=rjg+(gridz-zJatm)*(gridz-zJatm)
     rjg=Dsqrt(rjg)
     Rij=(xparent-xJatm)*(xparent-xJatm)
     Rij=Rij+(yparent-yJatm)*(yparent-yJatm)
     Rij=Rij+(zparent-zJatm)*(zparent-zJatm)
     Rij=Dsqrt(Rij)
     confocal=(rig-rjg)/Rij
     if (confocal >= 0.64d0) then
        ! gofconfocal=1.d0
        ! wofparent=wofparent*.5d0*(1.d0-1.d0)
        wofparent=0.d0
     elseif(confocal >= -0.64d0) then
        frctn=confocal/0.64d0
        frctnto3=frctn*frctn*frctn
        frctnto5=frctnto3*frctn*frctn
        frctnto7=frctnto5*frctn*frctn
        gofconfocal=(35.d0*frctn-35.d0*frctnto3+21.d0*frctnto5 &
             -5.d0*frctnto7)/16.d0
        wofparent=wofparent*.5d0*(1.d0-gofconfocal)
     else
        ! gofconfocal=-1.d0
        ! wofparent=wofparent*.5d0*(1.d0-(-1.d0))
        ! wofparent=wofparent
        continue
     endif
     Jatm=Jatm+1
  enddo

  Jatm=iparent+1
  do while (Jatm.le.natom.and.wofparent.ne.0.d0)
     xJatm=xyz(1,Jatm)
     yJatm=xyz(2,Jatm)
     zJatm=xyz(3,Jatm)
     rjg=(gridx-xJatm)*(gridx-xJatm)
     rjg=rjg+(gridy-yJatm)*(gridy-yJatm)
     rjg=rjg+(gridz-zJatm)*(gridz-zJatm)
     rjg=Dsqrt(rjg)
     Rij=(xparent-xJatm)*(xparent-xJatm)
     Rij=Rij+(yparent-yJatm)*(yparent-yJatm)
     Rij=Rij+(zparent-zJatm)*(zparent-zJatm)
     Rij=Dsqrt(Rij)
     confocal=(rig-rjg)/Rij
     if (confocal >= 0.64d0) then
        ! gofconfocal=1.d0
        ! wofparent=wofparent*.5d0*(1.d0-1.d0)
        wofparent=0.d0
     elseif(confocal >= -0.64d0) then
        frctn=confocal/0.64d0
        frctnto3=frctn*frctn*frctn
        frctnto5=frctnto3*frctn*frctn
        frctnto7=frctnto5*frctn*frctn
        gofconfocal=(35.d0*frctn-35.d0*frctnto3+21.d0*frctnto5 &
             -5.d0*frctnto7)/16.d0
        wofparent=wofparent*.5d0*(1.d0-gofconfocal)
     else
        ! gofconfocal=-1.d0
        ! wofparent=wofparent*.5d0*(1.d0-(-1.d0))
        ! wofparent=wofparent
        continue
     endif
     Jatm=Jatm+1
  enddo

  totalw=wofparent
  if (wofparent == 0.d0) then
     ssw=0.d0
     return
  endif

  ! Now we have the unnormalized weight of the grid point with regard to the
  ! parent atom.  Now we have to do this for all other atom pairs to
  ! normalize the grid weight.


  do Iatm=1,natom
     if (iatm == iparent) goto 50
     xIatm=xyz(1,Iatm)
     yIatm=xyz(2,Iatm)
     zIatm=xyz(3,Iatm)
     wofiatm=1.d0
     rig=(gridx-xIatm)*(gridx-xIatm)
     rig=rig+(gridy-yIatm)*(gridy-yIatm)
     rig=rig+(gridz-zIatm)*(gridz-zIatm)
     rig=Dsqrt(rig)
     Jatm=1
     do while (Jatm.ne.Iatm.and.wofiatm.ne.0.d0)
        rjg=(gridx-xyz(1,Jatm))*(gridx-xyz(1,Jatm))
        rjg=rjg+(gridy-xyz(2,Jatm))*(gridy-xyz(2,Jatm))
        rjg=rjg+(gridz-xyz(3,Jatm))*(gridz-xyz(3,Jatm))
        rjg=Dsqrt(rjg)
        Rij=(xIatm-xyz(1,Jatm))*(xIatm-xyz(1,Jatm))
        Rij=Rij+(yIatm-xyz(2,Jatm))*(yIatm-xyz(2,Jatm))
        Rij=Rij+(zIatm-xyz(3,Jatm))*(zIatm-xyz(3,Jatm))
        Rij=Dsqrt(Rij)
        confocal=(rig-rjg)/Rij
        if (confocal >= 0.64d0) then
           ! gofconfocal=1.d0
           ! wofiatm=wofiatm*.5d0*(1.d0-1.d0)
           wofiatm=0.d0
        elseif(confocal >= -0.64d0) then
           frctn=confocal/0.64d0
           frctnto3=frctn*frctn*frctn
           frctnto5=frctnto3*frctn*frctn
           frctnto7=frctnto5*frctn*frctn
           gofconfocal=(35.d0*frctn-35.d0*frctnto3+21.d0*frctnto5 &
                -5.d0*frctnto7)/16.d0
           wofiatm=wofiatm*.5d0*(1.d0-gofconfocal)
        else
           ! gofconfocal=-1.d0
           ! wofiatm=wofiatm*.5d0*(1.d0-(-1.d0))
           ! wofiatm=wofiatm
           continue
        endif
        Jatm=Jatm+1
     enddo

     Jatm=Iatm+1
     do while (Jatm.le.natom.and.wofiatm.ne.0.d0)
        rjg=(gridx-xyz(1,Jatm))*(gridx-xyz(1,Jatm))
        rjg=rjg+(gridy-xyz(2,Jatm))*(gridy-xyz(2,Jatm))
        rjg=rjg+(gridz-xyz(3,Jatm))*(gridz-xyz(3,Jatm))
        rjg=Dsqrt(rjg)
        Rij=(xIatm-xyz(1,Jatm))*(xIatm-xyz(1,Jatm))
        Rij=Rij+(yIatm-xyz(2,Jatm))*(yIatm-xyz(2,Jatm))
        Rij=Rij+(zIatm-xyz(3,Jatm))*(zIatm-xyz(3,Jatm))
        Rij=Dsqrt(Rij)
        confocal=(rig-rjg)/Rij
        if (confocal >= 0.64d0) then
           ! gofconfocal=1.d0
           ! wofiatm=wofiatm*.5d0*(1.d0-1.d0)
           wofiatm=0.d0
        elseif(confocal >= -0.64d0) then
           frctn=confocal/0.64d0
           frctnto3=frctn*frctn*frctn
           frctnto5=frctnto3*frctn*frctn
           frctnto7=frctnto5*frctn*frctn
           gofconfocal=(35.d0*frctn-35.d0*frctnto3+21.d0*frctnto5 &
                -5.d0*frctnto7)/16.d0
           wofiatm=wofiatm*.5d0*(1.d0-gofconfocal)
        else
           ! gofconfocal=-1.d0
           ! wofiatm=wofiatm*.5d0*(1.d0-(-1.d0))
           ! wofiatm=wofiatm
           continue
        endif
        Jatm=Jatm+1
     enddo

     totalw = totalw+wofiatm
50   continue
  enddo

  ssw=wofparent/totalw

end function ssw

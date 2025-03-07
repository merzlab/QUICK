#include "util.fh"
! Ed Brothers. July 11, 2002
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    subroutine sswder(gridx,gridy,gridz,Exc,quadwt,Iparent)
    use allmod
    implicit double precision(a-h,o-z)

! dimension UW(maxatm),wtgrad(3*maxatm)
    dimension uw(natom),wtgrad(3*natom)


    DO I=1,natom
       UW(I) = 1.d0
    ENDDO
! Beofre we look at anything else, zero out wtgrad.

    DO I=1,natom*3
        wtgrad(I) = 0.d0
    ENDDO

! This subroutine calculates the derivatives of weight found in
! Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
! 1996, pg 213-223.  The mathematical development is similar to
! the developement found in the papers of Gill and Pople, but uses
! SSW method rather than Becke's weights.

! The derivative of the weight with repsect to the displacement of the
! parent atom is mathematically quite complicated, thus rotational
! invariance is used.  Rotational invariance simply is the condition that
! the sum of all derivatives must be zero.

! Certain things will be needed again and again in this subroutine.  We
! are therefore goint to store them.  They are the unnormalized weights.
! The array is called UW for obvious reasons.

    sumUW = 0.d0
    DO Iatm=1,natom
        UW(Iatm)=1.d0
        xIatm=xyz(1,Iatm)
        yIatm=xyz(2,Iatm)
        zIatm=xyz(3,Iatm)
        rig=(gridx-xIatm)*(gridx-xIatm)
        rig=rig+(gridy-yIatm)*(gridy-yIatm)
        rig=rig+(gridz-zIatm)*(gridz-zIatm)
        rig=Dsqrt(rig)
        DO Jatm=1,natom
            IF (Jatm == Iatm) THEN
                continue
            ELSE
                rjg=(gridx-xyz(1,Jatm))*(gridx-xyz(1,Jatm))
                rjg=rjg+(gridy-xyz(2,Jatm))*(gridy-xyz(2,Jatm))
                rjg=rjg+(gridz-xyz(3,Jatm))*(gridz-xyz(3,Jatm))
                rjg=Dsqrt(rjg)
                Rij=(xIatm-xyz(1,Jatm))*(xIatm-xyz(1,Jatm))
                Rij=Rij+(yIatm-xyz(2,Jatm))*(yIatm-xyz(2,Jatm))
                Rij=Rij+(zIatm-xyz(3,Jatm))*(zIatm-xyz(3,Jatm))
                Rij=Dsqrt(Rij)
                confocal=(rig-rjg)/Rij
                IF (confocal >= 0.64d0) THEN
                    UW(Iatm)=0.d0
                ELSEIF(confocal >= -0.64d0) THEN
                    frctn=confocal/0.64d0
                    frctnto3=frctn*frctn*frctn
                    frctnto5=frctnto3*frctn*frctn
                    frctnto7=frctnto5*frctn*frctn
                    gofconfocal=(35.d0*frctn-35.d0*frctnto3+21.d0*frctnto5 &
                    -5.d0*frctnto7)/16.d0
                    UW(Iatm)=UW(Iatm)*.5d0*(1.d0-gofconfocal)
                ELSE
                   continue
                ENDIF
            ENDIF
        ENDDO
        sumUW = sumUW+UW(Iatm)
    ENDDO

! At this point we now have the unnormalized weight and the sum of same.
! Calculate the parent atom - grid point distance, and then start the loop.

    rig=(gridx-xyz(1,Iparent))**2.d0
    rig=rig+(gridy-xyz(2,Iparent))**2.d0
    rig=rig+(gridz-xyz(3,Iparent))**2.d0
    rig=Dsqrt(rig)

    a = .64d0

    DO Jatm=1,natom
        jstart=(Jatm-1)*3
        IF (Jatm /= Iparent) THEN

            rjg=(gridx-xyz(1,Jatm))*(gridx-xyz(1,Jatm))
            rjg=rjg+(gridy-xyz(2,Jatm))*(gridy-xyz(2,Jatm))
            rjg=rjg+(gridz-xyz(3,Jatm))*(gridz-xyz(3,Jatm))
            rjg=Dsqrt(rjg)
            Rij=(xyz(1,Iparent)-xyz(1,Jatm))**2.d0
            Rij=Rij+(xyz(2,Iparent)-xyz(2,Jatm))**2.d0
            Rij=Rij+(xyz(3,Iparent)-xyz(3,Jatm))**2.d0
            Rij=Dsqrt(Rij)

            dmudx = (-1.d0/Rij)*(1/rjg)*(xyz(1,Jatm)-gridx) &
            +((rig-rjg)/(Rij**3.d0))*(xyz(1,Iparent)-xyz(1,Jatm))
            dmudy = (-1.d0/Rij)*(1/rjg)*(xyz(2,Jatm)-gridy) &
            +((rig-rjg)/(Rij**3.d0))*(xyz(2,Iparent)-xyz(2,Jatm))
            dmudz = (-1.d0/Rij)*(1/rjg)*(xyz(3,Jatm)-gridz) &
            +((rig-rjg)/(Rij**3.d0))*(xyz(3,Iparent)-xyz(3,Jatm))

            u = (rig-rjg)/Rij
            T =(-35.d0*(a + u)**3.d0)/((a - u)*(16.d0*a**3.d0 &
            +29.d0*a**2.d0*u + 20.d0*a*u**2.d0 + 5.d0*u**3.d0))

            wtgrad(jstart+1) = wtgrad(jstart+1) + UW(Iparent)*dmudx*T/sumUW
            wtgrad(jstart+2) = wtgrad(jstart+2) + UW(Iparent)*dmudy*T/sumUW
            wtgrad(jstart+3) = wtgrad(jstart+3) + UW(Iparent)*dmudz*T/sumUW

            DO  Latm=1,natom
                IF (Latm /= Jatm) THEN
                    rlg=(gridx-xyz(1,Latm))*(gridx-xyz(1,Latm))
                    rlg=rlg+(gridy-xyz(2,Latm))*(gridy-xyz(2,Latm))
                    rlg=rlg+(gridz-xyz(3,Latm))*(gridz-xyz(3,Latm))
                    rlg=Dsqrt(rlg)
                    Rjl=(xyz(1,Jatm)-xyz(1,Latm))**2.d0
                    Rjl=Rjl+(xyz(2,Jatm)-xyz(2,Latm))**2.d0
                    Rjl=Rjl+(xyz(3,Jatm)-xyz(3,Latm))**2.d0
                    Rjl=Dsqrt(Rjl)

                    dmudx = (-1.d0/Rjl)*(1/rjg)*(xyz(1,Jatm)-gridx) &
                    +((rlg-rjg)/(Rjl**3.d0))*(xyz(1,Latm)-xyz(1,Jatm))
                    dmudy = (-1.d0/Rjl)*(1/rjg)*(xyz(2,Jatm)-gridy) &
                    +((rlg-rjg)/(Rjl**3.d0))*(xyz(2,Latm)-xyz(2,Jatm))
                    dmudz = (-1.d0/Rjl)*(1/rjg)*(xyz(3,Jatm)-gridz) &
                    +((rlg-rjg)/(Rjl**3.d0))*(xyz(3,Latm)-xyz(3,Jatm))

                    u = (rlg-rjg)/Rjl

                    T =(-35.d0*(a + u)**3.d0)/((a - u)*(16.d0*a**3.d0 &
                    +29.d0*a**2.d0*u + 20.d0*a*u**2.d0 + 5.d0*u**3.d0))

                    wtgrad(jstart+1) = wtgrad(jstart+1) &
                    -UW(Latm)*UW(Iparent)*dmudx*T/sumUW**2.d0
                    wtgrad(jstart+2) = wtgrad(jstart+2) &
                    -UW(Latm)*UW(Iparent)*dmudy*T/sumUW**2.d0
                    wtgrad(jstart+3) = wtgrad(jstart+3) &
                    -UW(Latm)*UW(Iparent)*dmudz*T/sumUW**2.d0
                ENDIF
            ENDDO

            DO  Latm=1,natom
                IF (Latm /= Jatm) THEN
                    rlg=(gridx-xyz(1,Latm))*(gridx-xyz(1,Latm))
                    rlg=rlg+(gridy-xyz(2,Latm))*(gridy-xyz(2,Latm))
                    rlg=rlg+(gridz-xyz(3,Latm))*(gridz-xyz(3,Latm))
                    rlg=Dsqrt(rlg)
                    Rjl=(xyz(1,Jatm)-xyz(1,Latm))**2.d0
                    Rjl=Rjl+(xyz(2,Jatm)-xyz(2,Latm))**2.d0
                    Rjl=Rjl+(xyz(3,Jatm)-xyz(3,Latm))**2.d0
                    Rjl=Dsqrt(Rjl)

                    dmudx = (-1.d0/Rjl)*(1/rlg)*(xyz(1,Latm)-gridx) &
                    +((rjg-rlg)/(Rjl**3.d0))*(xyz(1,Jatm)-xyz(1,Latm))
                    dmudy = (-1.d0/Rjl)*(1/rlg)*(xyz(2,Latm)-gridy) &
                    +((rjg-rlg)/(Rjl**3.d0))*(xyz(2,Jatm)-xyz(2,Latm))
                    dmudz = (-1.d0/Rjl)*(1/rlg)*(xyz(3,Latm)-gridz) &
                    +((rjg-rlg)/(Rjl**3.d0))*(xyz(3,Jatm)-xyz(3,Latm))

                    u = (rjg-rlg)/Rjl

                    T =(-35.d0*(a + u)**3.d0)/((a - u)*(16.d0*a**3.d0 &
                    +29.d0*a**2.d0*u + 20.d0*a*u**2.d0 + 5.d0*u**3.d0))

                    wtgrad(jstart+1) = wtgrad(jstart+1) &
                    +UW(Jatm)*UW(Iparent)*dmudx*T/sumUW**2.d0
                    wtgrad(jstart+2) = wtgrad(jstart+2) &
                    +UW(Jatm)*UW(Iparent)*dmudy*T/sumUW**2.d0
                    wtgrad(jstart+3) = wtgrad(jstart+3) &
                    +UW(Jatm)*UW(Iparent)*dmudz*T/sumUW**2.d0
                ENDIF
            ENDDO

        ENDIF
    ENDDO

! Now do the rotational invariance part of the derivatives.

    istart=(Iparent-1)*3
    DO Jatm=1,natom
        IF (Jatm /= Iparent) THEN
            jstart=(Jatm-1)*3
            DO I=1,3
                wtgrad(istart+I)=wtgrad(istart+I)-wtgrad(jstart+I)
            ENDDO
        ENDIF
    ENDDO

! We should now have the derivatives of the SS weights.  Now just add it into
! the gradient

    DO I=1,natom*3
    quick_qm_struct%gradient(I)=quick_qm_struct%gradient(I)+wtgrad(I)*Exc*quadwt
    ENDDO
    return
    end subroutine sswder

! Following subroutine was implemented by Tim Giese based on Stratmann,
! Scuseria, and Frisch, Chem. Phys. Lett., v257 1996, pg 213-223 paper.

#define SSW_POLYFAC1 (3.4179687500d0)
#define SSW_POLYFAC2 (8.344650268554688d0)
#define SSW_POLYFAC3 (12.223608791828156d0)
#define SSW_POLYFAC4 (7.105427357601002d0)

  subroutine getssw(gridx,gridy,gridz,Iparent,natom,xyz,p)
    implicit none
    double precision,intent(in) :: gridx,gridy,gridz
    integer,intent(in) :: Iparent,natom
    double precision,intent(in) :: xyz(3,natom)
    double precision,intent(out) :: p
    
    double precision,parameter :: a = 0.64
    integer :: iat,jat
    double precision :: uw(natom), uw_local
    double precision :: mu,muoa,g,s,z,muoa3 
    double precision :: rxg(4,natom)
    double precision :: rigv(3),rig,rig2
    double precision :: rjgv(3),rjg,rjg2
    double precision :: rijv(3),rij,rij2
    double precision :: sumw
   

    uw = 0.d0
    
    do iat=1,natom
       rigv(1) = xyz(1,iat)-gridx
       rigv(2) = xyz(2,iat)-gridy
       rigv(3) = xyz(3,iat)-gridz
       rig2 = rigv(1)*rigv(1)+rigv(2)*rigv(2)+rigv(3)*rigv(3)
       rig = sqrt(rig2)
       rxg(1:3,iat) = rigv(1:3)
       rxg(4,iat) = rig
    end do


    ! Calculate wi(rg)
    do iat=1,natom
       uw_local = 1.0d0

       rigv(1:3) = rxg(1:3,iat)
       rig = rxg(4,iat)

       ! wi(rg) = \prod_{j /= i} s(mu_{ij})
       do jat=1,natom
          if ( jat == iat ) then
             cycle
          end if
          rjgv(1:3) = rxg(1:3,jat)
          rjg = rxg(4,jat)
          
          rijv(1) = xyz(1,iat)-xyz(1,jat)
          rijv(2) = xyz(2,iat)-xyz(2,jat)
          rijv(3) = xyz(3,iat)-xyz(3,jat)
          rij2 = rijv(1)*rijv(1)+rijv(2)*rijv(2)+rijv(3)*rijv(3)
          rij = sqrt(rij2)

          mu = (rig-rjg) * (1/rij)

          
          if ( mu <= -a ) then
             g = -1.d0
          else if ( mu >= a ) then
             g = 1.d0
          else

             !muoa = mu/a
             !z = (35.d0*muoa - 35.d0*muoa**3 &
             !     & + 21.d0*muoa**5 - 5.d0*muoa**7)/16.d0
             !g = z

             ! MM optimized above statements as follows. 

             !We can reduce the MUL operations by precomputing polynomial constants in eqn14. 
             !constant of the first term, 3.4179687500 = 35.0 * (1/0.64) * (1/16) 
             !constant of the second term, 8.344650268554688 = 35.0 * (1/0.64)^3 * (1/16) 
             !constant of the third term, 12.223608791828156 = 21.0 * (1/0.64)^5 * (1/16) 
             !constant of the fourth term, 7.105427357601002 = 5.0 * (1/0.64)^7 * (1/16)

             muoa = mu
             muoa3 = muoa * muoa * muoa

             g=SSW_POLYFAC1 * muoa - SSW_POLYFAC2 * muoa3 +&
               SSW_POLYFAC3 * muoa * muoa * muoa3 - &
               SSW_POLYFAC4 * muoa * muoa3 * muoa3

          end if
          !if ( iat == 1 .and. jat == 3 ) write(6,'(es20.10)')mu
          
          s = 0.50d0 * (1.d0-g)
          !if ( iat == 1 .and. jat == 3 ) write(6,'(2es20.10)')mu,s

          uw_local=uw_local*s
       end do
       uw(iat)=uw_local
    end do

    
    sumw = 0.d0
    do iat=1,natom
       sumw = sumw + uw(iat)
    end do
    p = uw(Iparent) / sumw

    !write(6,'(es20.10)')sumw

    
  end subroutine getssw



  subroutine getsswnumder(gridx,gridy,gridz,Iparent,natom,xyz,dp)
    implicit none
    double precision,intent(in) :: gridx,gridy,gridz
    integer,intent(in) :: Iparent,natom
    double precision,intent(in) :: xyz(3,natom)
    double precision,intent(out) :: dp(3,natom)
    double precision,parameter :: delta = 2.5d-5
    double precision :: phi,plo
    double precision :: tmpxyz(3,natom)
    double precision :: tx,ty,tz
    integer :: iat,k

    tmpxyz=xyz
    tx = gridx - xyz(1,Iparent)
    ty = gridy - xyz(2,Iparent)
    tz = gridz - xyz(3,Iparent)

    
    do iat=1,natom
       do k=1,3
          
          tmpxyz(k,iat) = tmpxyz(k,iat) + delta
          
          call getssw( tx+tmpxyz(1,Iparent), &
               & ty+tmpxyz(2,Iparent), &
               & tz+tmpxyz(3,Iparent), &
               & Iparent,natom,tmpxyz,phi)

          
          tmpxyz(k,iat) = tmpxyz(k,iat) - 2.d0*delta
          
          call getssw( tx+tmpxyz(1,Iparent), &
               & ty+tmpxyz(2,Iparent), &
               & tz+tmpxyz(3,Iparent), &
               & Iparent,natom,tmpxyz,plo)

          tmpxyz(k,iat) = tmpxyz(k,iat) + delta

          dp(k,iat) = (phi-plo)/(2.d0*delta)
       end do
    end do
    
  end subroutine getsswnumder


  
  
  subroutine getsswanader(gridx,gridy,gridz,Iparent,natom,xyz,grd)

    use quick_method_module, only: quick_method
    implicit none
    double precision,intent(in) :: gridx,gridy,gridz
    integer,intent(in) :: Iparent,natom
    double precision,intent(in) :: xyz(3,natom)
    double precision,intent(out) :: grd(3,natom)
    double precision :: p

    double precision,parameter :: a = 0.64
    integer :: iat,jat
    double precision :: uw, uw_parent
    double precision :: duw(3,natom), duw_parent(3,natom)
    double precision :: mu,g,s,z,mu2,mu3,mu4,mu5,mu6,mu7
    double precision :: rxg(4,natom)
    double precision :: rigv(3),rig,rig2
    double precision :: rjgv(3),rjg,rjg2
    double precision :: rijv(3),rij,rij2
    double precision :: sumw,dsdmu
    double precision :: dmudi(3),dmudj(3),dmudg(3)


    !double precision,parameter :: SSW_POLYFAC1 = 3.4179687500d0
    !double precision,parameter :: SSW_POLYFAC2 = 8.344650268554688d0
    !double precision,parameter :: SSW_POLYFAC3 = 12.223608791828156d0
    !double precision,parameter :: SSW_POLYFAC4 = 7.105427357601002d0

    !write(6,*)gridx,gridy,gridz
    grd = 0.d0

    sumw = 0.d0
    uw = 0.d0
    uw_parent = 0.d0
    duw = 0.d0
    duw_parent = 0.d0
    !write(6,*)xyz

    do iat=1,natom
       rigv(1) = xyz(1,iat)-gridx
       rigv(2) = xyz(2,iat)-gridy
       rigv(3) = xyz(3,iat)-gridz
       rig2 = rigv(1)*rigv(1)+rigv(2)*rigv(2)+rigv(3)*rigv(3)
       rig = sqrt(rig2)
       rxg(1:3,iat) = rigv(1:3)
       rxg(4,iat) = rig
    end do


    ! Calculate wi(rg)
    do iat=1,natom
       uw = 1.0d0
       duw = 0.d0

       rigv(1:3) = rxg(1:3,iat)
       rig = rxg(4,iat)

       ! wi(rg) = \prod_{j /= i} s(mu_{ij})
       do jat=1,natom
          if ( jat == iat ) then
             cycle
          end if
          rjgv(1:3) = rxg(1:3,jat)
          rjg = rxg(4,jat)

          rijv(1) = xyz(1,iat)-xyz(1,jat)
          rijv(2) = xyz(2,iat)-xyz(2,jat)
          rijv(3) = xyz(3,iat)-xyz(3,jat)
          rij2 = rijv(1)*rijv(1)+rijv(2)*rijv(2)+rijv(3)*rijv(3)
          rij = sqrt(rij2)


          mu = (rig-rjg) * (1.d0/rij)
          dmudi =  (rigv/rig)/rij - (rig-rjg) * (1.d0/rij**3) * rijv
          dmudj = -(rjgv/rjg)/rij + (rig-rjg) * (1.d0/rij**3) * rijv
          dmudg =  (-rigv/rig + rjgv/rjg)/rij



          if ( mu <= -a ) then
             !g = -1.d0
             !s = 0.50d0 * (1.d0-g)
             !uw(iat) = uw(iat)*s
             ! s = 1 and uw is unchanged
          else if ( mu >= a ) then
             !g = 1.d0
             !s = 0.50d0 * (1.d0-g)
             !uw(iat) = uw(iat)*s
             ! s = 0 and uw = 0
             uw = 0.d0
             duw = 0.d0
          else

             !muoa = mu/a
             !z = (35.d0*muoa - 35.d0*muoa**3 &
             !     & + 21.d0*muoa**5 - 5.d0*muoa**7)/16.d0
             !g = z

             ! MM optimized above statements as follows. 

             !We can reduce the MUL operations by precomputing polynomial constants in eqn14. 
             !constant of the first term, 3.4179687500 = 35.0 * (1/0.64) * (1/16) 
             !constant of the second term, 8.344650268554688 = 35.0 * (1/0.64)^3 * (1/16) 
             !constant of the third term, 12.223608791828156 = 21.0 * (1/0.64)^5 * (1/16) 
             !constant of the fourth term, 7.105427357601002 = 5.0 * (1/0.64)^7 * (1/16)

             mu2 = mu*mu
             mu3 = mu*mu2
             mu4 = mu2*mu2
             mu5 = mu2*mu3
             mu6 = mu3*mu3
             mu7 = mu3*mu4

             g=SSW_POLYFAC1 * mu - SSW_POLYFAC2 * mu3 +&
                  SSW_POLYFAC3 * mu5 - &
                  SSW_POLYFAC4 * mu7

             s = 0.50d0 * (1.d0-g)

             dsdmu = -0.50d0 * ( SSW_POLYFAC1 - SSW_POLYFAC2 * 3.d0*mu2 +&
                  SSW_POLYFAC3 * 5.0d0 * mu4 - &
                  SSW_POLYFAC4 * 7.0d0 * mu6 )

             uw=uw*s
             ! We will later multiply these by uw; the division by s
             ! seen here will remove this term from the product
             duw(:,iat) = duw(:,iat) + dsdmu*dmudi/s
             duw(:,jat) = duw(:,jat) + dsdmu*dmudj/s
             duw(:,Iparent) = duw(:,Iparent) + dsdmu*dmudg/s

             
          end if

       end do

       if ( abs(uw) > quick_method%DMCutoff ) then
          duw(:,:) = uw * duw(:,:)
       else
          duw(:,:) = 0.d0
       end if

       sumw = sumw + uw
       grd = grd + duw
       if ( iat == Iparent ) then
          duw_parent = duw
          uw_parent = uw
       end if
    end do

    p = uw_parent / sumw
    grd = (-p/sumw) * grd + duw_parent/sumw

  end subroutine getsswanader

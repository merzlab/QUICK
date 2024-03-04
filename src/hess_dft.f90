#include "util.fh"

#define SSW_POLYFAC1 (3.4179687500d0)
#define SSW_POLYFAC2 (8.344650268554688d0)
#define SSW_POLYFAC3 (12.223608791828156d0)
#define SSW_POLYFAC4 (7.105427357601002d0)

  subroutine ssw2der(gridx,gridy,gridz,Iparent,natom,xyz,wght,evec,gwt,hwt)

    use quick_method_module, only: quick_method
    implicit double precision(a-h,o-z) 
    double precision,intent(in) :: gridx,gridy,gridz
    integer,intent(in) :: Iparent,natom
    double precision,intent(in) :: xyz(3,natom)
    double precision,intent(in) :: evec,wght
    double precision,intent(out) :: gwt(3,natom)
    double precision,intent(out) :: hwt(3,natom,3,natom)

    double precision :: p(natom)
    double precision :: gmuab(3,natom,natom),hmuab(15,natom,natom)
    double precision :: gzb(3),gzc(3),hz(9)

    double precision,parameter :: a = 0.64
    integer :: iat,jat
    double precision :: uw, uw_parent
    double precision :: duw(3,natom), duw_parent(3,natom)
    double precision :: mu,g,s,z,mu2,mu3,mu4,mu5,mu6,mu7
    double precision :: rxg(4,natom)
    double precision :: rigv(3),rig,rig2
    double precision :: rjgv(3),rjg,rjg2
    double precision :: rijv(3),rij,rij2
    double precision :: sumw
    double precision :: dsdmu(natom,natom), d2sdmu2(natom,natom)
    PARAMETER (Zero=0.0d0,Half=0.5d0,One=1.0d0,ThreeHalf=3.0d0/2.0d0)
    PARAMETER (two=2.0d0)

    integer i12(3,3)
    data i12/1,2,4,2,3,5,4,5,6/


    !double precision,parameter :: SSW_POLYFAC1 = 3.4179687500d0
    !double precision,parameter :: SSW_POLYFAC2 = 8.344650268554688d0
    !double precision,parameter :: SSW_POLYFAC3 = 12.223608791828156d0
    !double precision,parameter :: SSW_POLYFAC4 = 7.105427357601002d0

!    grd = 0.d0

!    sumw = 0.d0
    uw = 0.d0
!    uw_parent = 0.d0
    duw = 0.d0
!    duw_parent = 0.d0
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

          call GHmu(gridx,gridy,gridz,xyz(1,jat),xyz(2,jat),xyz(3,jat), &
                   xyz(1,iat),xyz(2,iat),xyz(3,iat),1.0d0/rij,mu, &
                   gmuab(1:3,iat,jat),hmuab(1:15,iat,jat))

!          dmudi =  (rigv/rig)/rij - (rig-rjg) * (1.d0/rij**3) * rijv
!          dmudj = -(rjgv/rjg)/rij + (rig-rjg) * (1.d0/rij**3) * rijv
!          dmudg =  (-rigv/rig + rjgv/rjg)/rij

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

             dsdmu(iat,jat) = (-0.50d0 * ( SSW_POLYFAC1 - SSW_POLYFAC2 * 3.d0*mu2 +&
                  SSW_POLYFAC3 * 5.0d0 * mu4 - &
                  SSW_POLYFAC4 * 7.0d0 * mu6 ))/s

             d2sdmu2(iat,jat) = (-0.50d0 * (-SSW_POLYFAC2 * 6.d0 * mu + &
                  SSW_POLYFAC3 * 20.0d0 * mu3 - &
                  SSW_POLYFAC4 * 42.0d0 * mu5 ))/s 

             uw=uw*s
             ! We will later multiply these by uw; the division by s
             ! seen here will remove this term from the product
!             duw(:,iat) = duw(:,iat) + dsdmu*dmudi/s
!             duw(:,jat) = duw(:,jat) + dsdmu*dmudj/s
!             duw(:,Iparent) = duw(:,Iparent) + dsdmu*dmudg/s

             
          end if

       END DO

!       if ( abs(uw) > quick_method%DMCutoff ) then
!          duw(:,:) = uw * duw(:,:)
!       else
!          duw(:,:) = 0.d0
!       end if

       p(iat) = uw
       sumw = sumw + uw
!       grd = grd + duw
!       if ( iat == Iparent ) then
!          duw_parent = duw
!          uw_parent = uw
!       end if
    END DO

    if (sumw .lt. quick_method%DMCutoff) then
       Z = Zero
    else
       Z = one/sumw
    endif

    oma = wght*p(Iparent)*Z
    omae = oma*evec
    omap = oma
    DO Iat=1,NAtom
       If (Iat == Iparent ) then
             cycle
       End If
       qai=d2sdmu2(Iparent,iat)
       tai=dsdmu(Iparent,iat)
       call gradz(p,dsdmu,gmuab,gzb,iat,natom)
       call hbbz(p,dsdmu,d2sdmu2,gmuab,hmuab,hz,iat,natom)

!...
!...  gradient
!...
       gwt(1,iat)=-omap*(tai*gmuab(1,iat,iparent)+z*gzb(1))
       gwt(2,iat)=-omap*(tai*gmuab(2,iat,iparent)+z*gzb(2))
       gwt(3,iat)=-omap*(tai*gmuab(3,iat,iparent)+z*gzb(3))
    
!...
!...  hessian with respect center iat two times
!...
        do ic1=1,3
          do ic2=1,3
             hwt(ic1,iat,ic2,iat)= &
                +omae*(qai*gmuab(ic1,iat,iparent)*gmuab(ic2,iat,iparent) &
                -tai*hmuab(i12(ic1,ic2),iat,iparent) &
                +z*tai*(gmuab(ic1,iat,iparent)*gzb(ic2) &
                +gmuab(ic2,iat,iparent)*gzb(ic1)) &
                +two*z*z*gzb(ic1)*gzb(ic2) &
                -z*hz(i12(ic1,ic2))) 
          enddo
        enddo
        DO jat=iat+1,NATOM
          If (jat == Iparent ) then
             cycle
          End If
          call gradz(p,dsdmu,gmuab,gzc,jat,natom)
          call hbcz(p,dsdmu,d2sdmu2,gmuab,hmuab,hz,iat,jat,natom)
          taj=dsdmu(Iparent,jat)
!...
!...  hessian with respect centers iat and jat
!...
          index=0
          do ic1=1,3
            do ic2=1,3
            index=index+1
              hwt(ic1,iat,ic2,jat)= & 
                 +omae*(tai*taj*gmuab(ic1,iat,iparent)*gmuab(ic2,jat,iparent) &
                 +z*taj*gmuab(ic2,jat,iparent)*gzb(ic1)- &
                 z*hz(index)+ &
                 z*tai*gmuab(ic1,iat,iparent)*gzc(ic2)+ &
                 two*z*z*gzb(ic1)*gzc(ic2)) 
            enddo
          enddo
        ENDDO
    ENDDO

!    p = uw_parent / sumw
!    grd = (-p/sumw) * grd + duw_parent/sumw

  end subroutine ssw2der

! =================================================================
!
      SUBROUTINE GHMu(X,Y,Z,XB,YB,ZB,XA,YA,ZA,rAB,xmuAB, &
                       Gradx,Hess)
      IMPLICIT REAL*8(A-H,O-Z)
!
!    MM (01/09/2003), most of this subroutine has been generated
!    with maxima
!
!     computes the gradient annd hessian of the hyperbolic coordinates
!     mu(A,B) used in the calculations of Becke's quadrature weights,
!     taking into account the modifications due to size adjustment
!
!     for the hyperbolic coordinates definition and their gradient
!     see: Johnson, Gill and Pople, J.Chem.Phys.  98 (1993) 5612,
!     eqs. (B6) and (B10).
!
!     For the modifications due to size adjustment, see:
!     A.D. Becke, J. Chem. Phys. 88, 2547 (1988);
!     O. Treutler and R. Ahlrichs, J. Chem. Phys. 102, 346 (1995).
!
!  ARGUMENTS
!
!  X       -  x coordinate of grid point
!  Y       -  y coordinate of grid point
!  Z       -  z coordinate of grid point
!  XA      -  x coordinate of atom A
!  YA      -  y coordinate of atom A
!  ZA      -  z coordinate of atom A
!  XB      -  x coordinate of atom B
!  YB      -  y coordinate of atom B
!  ZB      -  z coordinate of atom B
!  rAB     -  inverse distance between atoms A and B
!  xaAB    -  atomic size adjustment factor
!  xmuAB   -  hyperbolic coordinate
!  Gradx   -  on exit contains the 3 components of gradient
!             adjusted for size (only the gradient with respect
!             atom a is computed)
!  Hess    -  on exit contains the 15 components of the hessian
!             adjusted for size (the second derivatives involving
!             two times atom B are not computed)
!
!     Ordering of array Hess:
!
!                         1     XA  XA
!                         2     YA  XA
!                         3     YA  YA
!                         4     ZA  XA
!                         5     ZA  YA
!                         6     ZA  ZA
!                         7     XB  XA
!                         8     XB  YA
!                         9     XB  ZA
!                        10     YB  XA
!                        11     YB  YA
!                        12     YB  ZA
!                        13     ZB  XA
!                        14     ZB  YA
!                        15     ZB  ZA
!
      DIMENSION Gradx(3),Hess(15)
      Real*8  Grad(6)
      PARAMETER (One=1.0d0,Two=2.0d0,Three=3.0d0)
!
!  vectors from grid point to atoms A and B
!
      ax = (XA-X)
      ay = (YA-Y)
      az = (ZA-Z)
      bx = (XB-X)
      by = (YB-Y)
      bz = (ZB-Z)
!...
!...  vector lengths
!...
      A = SQRT(az*az+ay*ay+ax*ax)
      B = SQRT(bz*bz+by*by+bx*bx)
!...
!...  precomputes common quantities
!...
      abx = bx-ax
      aby = by-ay
      abz = bz-az
      arab= rab/A
      brab= rab/B
      ab3 = rab*rab*rab
      ambab3 = ab3*(A-B)
!...
!...  gradient components
!...
      GRAD(1) = ax*arab-ambab3*abx
      GRAD(2) = ay*arab-ambab3*aby
      GRAD(3) = az*arab-ambab3*abz
      GRAD(4) = -bx*brab+ambab3*abx
      GRAD(5) = -by*brab+ambab3*aby
      GRAD(6) = -bz*brab+ambab3*abz
!...
!...  modified for size adjustment
!...
      coef = (One) !- Two*xaAB*xmuAB)
      GRADX(1) = grad(1)*coef
      GRADX(2) = grad(2)*coef
      GRADX(3) = grad(3)*coef
!...
!...  additional common quantities for hessian
!...
      aab3=ab3/A
      bab3=ab3/B
      a3ab=rab/A**3
      tambab5=Three*ambab3*rab*rab
!...
!...  hessian components with size adjustment factors
!...
      TxaAB = Two*xaAB
      hess(1)= (abx*abx*tambab5-two*ax*abx*aab3-ambab3-ax*ax*a3ab+arab)
     !   *coef-TxaAB*grad(1)*grad(1)
      hess(2)= (abx*aby*tambab5-(ax*aby+ay*abx)*aab3-ax*ay*a3ab)
     !   *coef-TxaAB*grad(2)*grad(1)
      hess(3)=(aby*aby*tambab5-two*ay*aby*aab3-ambab3-ay*ay*a3ab+arab)
     !   *coef-TxaAB*grad(2)*grad(2)
      hess(4)= (abx*abz*tambab5-(ax*abz+az*abx)*aab3-ax*az*a3ab)
     !   *coef-TxaAB*grad(3)*grad(1)
      hess(5)= (aby*abz*tambab5-(ay*abz+az*aby)*aab3-ay*az*a3ab)
     !   *coef-TxaAB*grad(3)*grad(2)
      hess(6)= (abz*abz*tambab5-two*az*abz*aab3-ambab3-az*az*a3ab+arab)
     !   *coef-TxaAB*grad(3)*grad(3)
      hess(7) = (abx*bx*bab3-tambab5*abx*abx+ax*abx*aab3+ambab3)
     !   *coef-TxaAB*grad(4)*grad(1)
      hess(8) = (aby*bx*bab3-tambab5*abx*aby+ay*abx*aab3)
     !   *coef-TxaAB*grad(4)*grad(2)
      hess(9) = (abz*bx*bab3-tambab5*abx*abz+az*abx*aab3)
     !   *coef-TxaAB*grad(4)*grad(3)
      hess(10) = (abx*by*bab3-tambab5*abx*aby+ax*aby*aab3)
     !   *coef-TxaAB*grad(5)*grad(1)
      hess(11) = (aby*by*bab3-tambab5*aby*aby+ay*aby*aab3+ambab3)
     !   *coef-TxaAB*grad(5)*grad(2)
      hess(12) = (abz*by*bab3-tambab5*abz*aby+az*aby*aab3)
     !   *coef-TxaAB*grad(5)*grad(3)
      hess(13) = (abx*bz*bab3-tambab5*abx*abz+ax*abz*aab3)
     !   *coef-TxaAB*grad(6)*grad(1)
      hess(14) = (aby*bz*bab3-tambab5*aby*abz+ay*abz*aab3)
     !   *coef-TxaAB*grad(6)*grad(2)
      hess(15) = (abz*bz*bab3-tambab5*abz*abz+az*abz*aab3+ambab3)
     !   *coef-TxaAB*grad(6)*grad(3)
      RETURN
      END
! =====================================================================
!======================================================================
      subroutine gradz(p,t,gmuab,grzet,ib,natoms)
      IMPLICIT REAL*8(A-H,O-Z)
!...
!...  MM (01/09/2003)
!...
!...  computes the gradient of Z, defined as the sum of the
!...  cell functions P, see eq B2 of Johnson, Gill and Pople,
!...  J. Chem Phys, 98, 5612 (1993),
!...  with respect center ib:
!...
!...  gz(ic)=sum_(i.ne.ib) (p(ib)*t(ib,i)-p(i)*t(i,ib))*gmuab(ic,ib,i)
!...
!...  where the summation is over all centers except ib.
!...
!...   p(i)       cell function of center i  (eq. B3)
!...   t(i,ib)    auxiliary function         (eq  B9)
!...   gmuab      gradient of adjusted hyperbolic coordinates
!...              (see subroutine ghmu)
!...
!...
      real*8 grzet(3)
      real*8 P(natoms),T(natoms,natoms),gmuab(3,natoms,natoms)
      parameter (zero=0.0d0)
      grzet(1)=zero
      grzet(2)=zero
      grzet(3)=zero
      do 10 i=1,natoms
        if(i.eq.ib) goto 10
        piti=p(ib)*t(ib,i)-p(i)*t(i,ib)
        grzet(1)=grzet(1)+piti*gmuab(1,ib,i)
        grzet(2)=grzet(2)+piti*gmuab(2,ib,i)
        grzet(3)=grzet(3)+piti*gmuab(3,ib,i)
10    continue
      return
      END
!======================================================================

!======================================================================
      subroutine hbbz(p,t,q,gmuab,hmuab,hz,ib,natoms)
      IMPLICIT REAL*8(A-H,O-Z)
!...
!...  MM (01/09/2003)
!...
!...  computes the second derivatives (with respect atom ib
!...  two times), of Z, defined as the sum of the
!...  cell functions P, see eq B2 of Johnson, Gill and Pople,
!...  J. Chem Phys, 98, 5612 (1993).
!...
!...    d**2 Z     d**2 P(ib)                 d**2 P(i)
!...  ---------  = ---------- + sum_(i.ne.ib) ---------
!...  (d_ib)**2    (d_ib)**2                  (d_ib)**2
!...
!...  where the sum is over all centers except ib.
!...
!...  P(i) is the cell function of center i (eq. B3).
!...
!...  Developing further, one obtains:
!...
!...  d**2 P(ib)
!...  ---------- = P(ib) sum_(i.ne.ib) [ q(ib,i) gmuab(ib,i)**2 +
!...  (d_ib)**2
!...                  t(ib,i) hmuab(ib,i)-(t(ib,i) gmuab(ib,i)**2 ] +
!...
!...             + P(ib) [sum_(i.ne.ib) t(ib,i) gmuab(ib,i)] **2
!...
!...  and
!...
!...  d**2 P(i)
!...  --------- (i.ne.ib) = P(i) [ q(i,ib) gmuab(ib,i)**2
!...  (d_ib)**2
!...                              - t(i,ib) hmuab(ib,i) ]
!...
!...   t(i,ib)    auxiliary function         (eq  B9)
!...   q(i,ib)    auxiliary function (the same as t, but involving
!...              the second derivative of s)
!...   gmuab      gradient of adjusted hyperbolic coordinates
!...              (see subroutine ghmu)
!...   hmuab      hessian of adjusted hyperbolic coordinates
!...              (see subroutine ghmu)
!...
      real*8 hz(6)
      real*8 P(natoms),T(natoms,natoms),Q(natoms,natoms), &
            gmuab(3,natoms,natoms),hmuab(15,natoms,natoms)
      real*8 psum(3)
      parameter (zero=0.0d0)
!...
      psum(1)=zero
      psum(2)=zero
      psum(3)=zero
      do ic=1,6
      hz(ic)=zero
      enddo
!...
!...              d**2 P(ib)
!... first part:  ----------
!...              (d_ib)**2
!...
      do 10 i=1,natoms
        if(i.eq.ib) goto 10
        qbi=q(ib,i)
        tbi=t(ib,i)
        index=0
        do ic1=1,3
          do ic2=1,ic1
            index=index+1
            hz(index)=hz(index)+(qbi-tbi*tbi)*gmuab(ic1,ib,i) &
                   *gmuab(ic2,ib,i)+tbi*hmuab(index,ib,i)
          enddo
          psum(ic1)=psum(ic1)+tbi*gmuab(ic1,ib,i)
        enddo
10    continue
      index=0
      do ic1=1,3
        do ic2=1,ic1
           index=index+1
           hz(index)=p(ib)*(hz(index)+psum(ic1)*psum(ic2))
        enddo
      enddo
!...
!...                            d**2 P(i)
!... second part: sum_(i.ne.ib) ---------
!...                            (d_ib)**2
!...
      do 20 i=1,natoms
        if(i.eq.ib) goto 20
        qib=q(i,ib)
        tib=t(i,ib)
        index=0
        do ic1=1,3
          do ic2=1,ic1
            index=index+1
           hz(index)=hz(index)+p(i)*(qib*gmuab(ic1,ib,i)*gmuab(ic2,ib,i) &
                                  -tib*hmuab(index,ib,i))
          enddo
        enddo
20    continue
      return
      END
!======================================================================

!======================================================================
      subroutine hbcz(p,t,q,gmuab,hmuab,hz,ib,ic,natoms)
      IMPLICIT REAL*8(A-H,O-Z)
!...
!...  MM (01/09/2003)
!...
!...  computes the second derivatives (with respect atoms ib
!...  and ic), of Z, defined as the sum of the
!...  cell functions P, see eq B2 of Johnson, Gill and Pople,
!...  J. Chem Phys, 98, 5612 (1993).
!...
!...   d**2 Z     d**2 P(ib)   d**2 P(ic)
!...  --------- = ---------- + ---------- +
!...  d_ib d_ic   d_ib d_ic    d_ib d_ic
!...
!...                                             d**2 P(i)
!...                 + sum_(i.ne.ib.and.i.ne.ic) ---------
!...                                             d_ib d_ic
!...
!...  where the sum is over all centers except ib and ic.
!...
!...  P(i) is the cell function of center i (eq. B3).
!...
!...  Developing further, one obtains:
!...
!...  d**2 P(ib)
!...  ---------- = P(ib) [ t(ib,ic) hmuab(ib,ic) -
!...  d_ib d_ic
!...                    q(ib,ic) gmuab(ib,ic) gmuab(ic,ib) -
!...
!...       t(ib) gmuab(ic,ib) sum_(i.ne.ib.and.i.ne.ic) t(ib,i) gmuab(ib,i) ]
!...
!...  and
!...
!...  d**2 P(i)
!...  --------- (i.ne.ib.and.i.ne.ic) =
!...  d_ib d_ic
!...
!...                       = P(i) t(i,ib) t(i,ic) gmuab(ib,i) gmuab(ic,i)
!...
!...   t(i,ib)    auxiliary function         (eq  B9)
!...   q(i,ib)    auxiliary function (the same as t, but involving
!...              the second derivative of s)
!...   gmuab      gradient of adjusted hyperbolic coordinates
!...              (see subroutine ghmu)
!...   hmuab      hessian of adjusted hyperbolic coordinates
!...              (see subroutine ghmu)
!...
      real*8 hz(9),dcbb(9),dbcc(9)
      real*8 P(natoms),T(natoms,natoms),Q(natoms,natoms), &
            gmuab(3,natoms,natoms),hmuab(15,natoms,natoms)
      parameter (zero=0.0d0)
!...
!...  d**2 P(ib)
!...  ----------
!...  d_ib d_ic
!...
!...  actually, this call to cpartd returns the derivative
!...  with respect ic as first index and ib as second. Thus
!...  the actual contribution is given by the transposed of
!...  the dcbb array
!...
      call cpartd(p,t,q,gmuab,hmuab,dcbb,ib,ic,natoms)
!...
!...  d**2 P(ic)
!...  ----------
!...  d_ib d_ic
!...
      call cpartd(p,t,q,gmuab,hmuab,dbcc,ic,ib,natoms)
!...
      hz(1)=dcbb(1)+dbcc(1)
      hz(2)=dcbb(4)+dbcc(2)
      hz(3)=dcbb(7)+dbcc(3)
      hz(4)=dcbb(2)+dbcc(4)
      hz(5)=dcbb(5)+dbcc(5)
      hz(6)=dcbb(8)+dbcc(6)
      hz(7)=dcbb(3)+dbcc(7)
      hz(8)=dcbb(6)+dbcc(8)
      hz(9)=dcbb(9)+dbcc(9)
!...
!...                           d**2 P(i)
!... sum_(i.ne.ib.and.i.ne.ic) ---------
!...                           d_ib d_ic
!...
      do 10 i=1,natoms
        if(i.eq.ib) goto 10
        if(i.eq.ic) goto 10
        ptt=p(i)*t(i,ib)*t(i,ic)
        index=0
        do ic1=1,3
          do ic2=1,3
            index=index+1
            hz(index)=hz(index)+ptt*gmuab(ic1,ib,i)*gmuab(ic2,ic,i)
          enddo
        enddo
10    continue
      return
      end
!======================================================================

!======================================================================
      subroutine cpartd(p,t,q,gmuab,hmuab,d,ib,ic,natoms)
      IMPLICIT REAL*8(A-H,O-Z)
!...
!...  MM (01/09/2003)
!...
!...  computes a partial sum needed in the calculation
!...  of the weight hessian. See the comments in subroutine
!...  hbcz for details
!...
      real*8 d(9)
      real*8 P(natoms),T(natoms,natoms),Q(natoms,natoms), &
            gmuab(3,natoms,natoms),hmuab(15,natoms,natoms)
      real*8 psum(3)
      parameter (zero=0.0d0)
      tbc=t(ib,ic)
      qbc=q(ib,ic)
      psum(1)=zero
      psum(2)=zero
      psum(3)=zero
      do 10 i=1,natoms
        if(i.eq.ib) goto 10
        if(i.eq.ic) goto 10
        tbi=t(ib,i)
        psum(1)=psum(1)+tbi*gmuab(1,ib,i)
        psum(2)=psum(2)+tbi*gmuab(2,ib,i)
        psum(3)=psum(3)+tbi*gmuab(3,ib,i)
10    continue
!...
!...  this is done with ic2 in the external loop,
!...  so the hmuab array is properly addressed
!...
      index=0
      do ic2=1,3
        do ic1=1,3
          index=index+1
          d(index)=p(ib)*(tbc*hmuab(index+6,ib,ic)-qbc*gmuab(ic1,ib,ic)* &
                  gmuab(ic2,ic,ib)-tbc*gmuab(ic2,ic,ib)*psum(ic1))
        enddo
      enddo
      return
      END
!======================================================================

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE FormMaxDen(NBas,Dens,DA,DM)
      IMPLICIT REAL*8(A-H,O-Z)
!
!  Forms vector of maximum density matrix element per column.
!  Used for density threshold in DFT code
!  (NOTE: Used for density AND difference-density matrices)
!
!  ARGUMENTS
!
!  NBas    -  number of basis functions
!  DA      -  density matrix (lower triangle)
!  DM      -  on exit contains maximum density element per column
!
!
      DIMENSION Dens(NBas,NBas),DA(NBas*(NBas+1)/2),DM(NBas)
!

      DO I=1,NBas
         II = (I*(I-1))/2
         DO J=1,NBas
            IJ = II+J
            DA(IJ)=Dens(I,J)
         ENDDO
      ENDDO

      DO 20 I=1,NBas
      DMx = 0.0d0
      II = (I*(I-1))/2
      DO 10 J=1,I
      IJ = II+J
      If(Abs(DA(IJ)).GT.DMx) DMx = Abs(DA(IJ))
 10   CONTINUE
      DO 11 J=I+1,NBas
      IJ = (J*(J-1))/2 + I
      If(Abs(DA(IJ)).GT.DMx) DMx = Abs(DA(IJ))
 11   CONTINUE
      DM(I) = DMx
 20   CONTINUE
!
      RETURN
      END
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

! =====================================================================
      subroutine formFdHess(nbas,    natoms, &
                       thrsh,  da,     dm,     gden, &
                       wght,   pra,    prara,  prarb,  pga, &
                       pgc,    praga,  pragb,  pragc,  pgaga, &
                       pgagb, pgagc, pgcgc, ibin, vao, vaox, &
                       vaoxx,  vaoxxx,    vm, &
                       icntr,   gwt, hwt,    fda,    hess)
      use quick_gridpoints_module
      use quick_basis_module
!     implicit real*8 (a-h,o-z)
      implicit none             ! trying to avoid typing errors
!
!
!  carries out numerical integration and accumulates contribution
!  from current grid point into derivative fock matrices and hessian
!  taking into account the weight derivatives.
!
!  All components of the derivative fock matrices are computed in
!  one pass. Note that the icntr components of derivative fock and
!  hessian are not computed by this subroutine, as they can be
!  obtained by translational invariance.
!
!  ** closed shell **
!
!  arguments
!
!  dft     -  method flag (note: all methods include slater exchange)
!              1 - 3 - local correlation
!             >3 - nonlocal
!  npp     -  number of contributing (non-zero) grid points this batch
!  nbas    -  total number of basis functions
!  nbf     -  indexing array to location of "non-zero" basis functions
!  natoms  -  number of atoms
!  nbatm   -  basis functions --> atom index
!  thrsh   -  threshold for neglect of contribution
!  da      -  closed-shell density matrix (lower triangle)
!  dm      -  maximum density matrix element per column
!  gden    -  density gradient at grid points (non local only, dft > 3)
!  wght    -  grid quadrature weights
!  pra   -  Functional derivative w.r.t. alpha density at grid points
!  prara -  Functional 2nd deriv. w.r.t. alpha density at grid points
!  prarb -  Funct. 2nd deriv. w.r.t. alpha and beta density (dft > 3)
!  pga   -  Funct. deriv. w.r.t. alpha gradient (dft > 3)
!  pgc   -  Funct. deriv. w.r.t. alpha beta gradient (dft > 3)
!  praga -  Funct. 2nd. deriv. w.r.t. alpha dens. and  grad. (dft > 3)
!  pragb -  Funct. 2nd. deriv. w.r.t. alpha dens. and  beta grad.
!           (dft > 3)
!  pragc -  Funct. 2nd. deriv. w.r.t. alpha dens. and  alpha beta grad.
!           (dft > 3)
!  pgaga -  Funct. 2nd. deriv. w.r.t. alpha grad. (dft > 3)
!  pgagb -  Funct. 2nd. deriv. w.r.t. alpha and beta grad. (dft > 3)
!  pgagc -  Funct. 2nd. deriv. w.r.t. alpha and alpha beta grad.
!           (dft > 3)
!  pgcgc -  Funct. 2nd. deriv. w.r.t. alpha beta grad. (dft > 3)
!  vao     -  "non-zero" basis function values at grid points
!  potx    -  gradient functional derivative (dft > 3 only)
!  potxx   -  gradient functional second derivative (dft > 3)
!  potxd   -  density-gradient mixed second derivative (dft > 3)
!  vaox    -  basis function 1st derivatives at grid points
!  vaoxx   -  basis function 2nd derivatives at grid points
!  vaoxxx  -  basis function 3rd derivatives at grid points (dft > 3)
!  inb     -  indexing array for non-zero entries to vao
!  vm      -  array containing maximum magnitude ao per grid point
!  indx    -  index into contributing columns of vao
!  denx    -  scratch storage for density gradient per grid point
!  denxx   -    ditto for density hessian per grid point
!  gdx     -  ditto for atomic gradient of density gradient (dft > 3)
!  gdxx    -  ditto for atomic hessian of density gradient (dft > 3)
!  gx      -  ditto for atomic gradient of gradient invariant (dft > 3)
!  gxx     -  ditto for atomic hessian of gradient invariant (dft > 3)
!  sv      -  storage for coefficient of vao(i)*vao(j) in quadrature
!          -  formula (dft > 3)
!  sw      -  storage for coefficient of
!               (vaox(i)*vao(j)+vao(i)*vaox(j))  (dft > 3)
!  icntr   -  current atomic center
!  gwt     -  gradient of quadrature weight
!  hwt     -  hessian of quadrature weight
!
!  on exit
!
!  fda     -  contribution to derivative fock matrices
!  hess    -  contribution to hessian matrix
!
      integer ibin,nbas,natoms,icntr,icount
      real*8 wght,gwt(3,natoms),hwt(3,natoms,3,natoms)
      real*8 vao(nbas),vaox(3,nbas),vaoxx(6,nbas),vaoxxx(10,nbas)
      real*8 vm,da(nbas*(nbas+1)/2),dm(nbas),gden(3)
      real*8 pra,pga,pgc,prara
      real*8 prarb,praga,pragb,pragc
      real*8 pgaga,pgagb,pgagc,pgcgc
      real*8 denx(3,natoms),denxx(3,natoms,3,natoms)
      real*8 gdx(3,3,natoms),gdxx(3,3,natoms,3,natoms)
      real*8 gx(3,natoms),gxx(3,natoms,3,natoms)
      real*8 sv(3,natoms),sw(3,3,natoms)
!      integer dft,nbf(*),inb(*),indx(npp),nbatm(nbas)
      real*8 fda(3,natoms,nbas*(nbas+1)/2),hess(3,natoms,3,natoms)
      real*8 thrsh
!      real*8 td1,tg1,tsw,tqf,tqh

      real*8 zero,two,three,four,six,twelve,half
      parameter (zero=0.0d0,two=2.0d0,three=3.0d0,four=4.0d0,six=6.0d0)
      parameter (twelve=12.0d0,half=0.5d0)
      real*8 epsi,alot
      parameter (epsi=1.0d-15,alot=1.0d17)
      logical  dodenx,dodenxx,dogradxx,dox,doxx,doval
      integer nat3,ip,ipp,i,j,ii,jj,ij,it
      integer ia,ja,iatm,jatm,iixx,katm,k1,isv,isw
      real*8 ra,ra2,ga,gc,rara,rara2,raga,ragb,ragc,prap
      real*8 gaga,gagb,gagc,gcgc,vmx,vmx2,wg,prg,prg2,pgg,pgg2,pg,pg2
      real*8 abra,abrara,thrx,thrx1,thrxx,valt,abdaij,abdaijm,abval
      real*8 valmm,valmm1,valmm2,valxx,valxy,valyy,valxz,valyz,valzz
      real*8 abvj,abvvj,valj,valjx,valjy,valjz,abijt,abij
      real*8 thrsh1,thtest
      real*8 t1,t2,t3,t4,t5,t6
      real*8 gwtmax,dmx,valx,valy,valz,daij,xyc,xzc,yzc
      real*8 dmaxyz,vmax,val,vali,vdxc,valm,vdjx,xc
      real*8 hvalx,hvaly,hvalz,dx,dy,dz,sxx,sxy,sxz
      real*8 vd1,vd2,vd3,svmax,swmax,tswmax
      real*8 vmax1,vmax2,vmax3
      real*8 valix,valiy,valiz,xvi,xxvx,xxvy,xxvz,valtest,valmxj
      real*8 valmx,valmx1,xvj,valm1,val1,smaxmx,val2,val3,val4
      real*8 vij,vijx,vijy,vijz,gij,hdx,hdy,hdz,hgx,hgy,hgz
      real*8 daijt,smax,dmax,potp,potxp,potp2,potxp2
      real*8 sxxi,sxyi,sxzi,dpmax2
      real*8 valt1,valt2
!
!
      nat3 = 3*natoms
!
        wg = wght
        vmx = vm
        ra = pra*wg
        ga = pga*wg
        gc = pgc*wg
        rara = (prara+prarb)*wg
        raga = praga*wg
        ragb = pragb*wg
        ragc = pragc*wg
        gaga = pgaga*wg
        gagb = pgagb*wg
        gagc = pgagc*wg
        gcgc = pgcgc*wg
!
!  some sums of the potentials that will be used later
!
        prg=raga+ragb+ragc
        pgg=gaga+gagb+two*gagc+half*gcgc
        pg=ga+half*gc
!
!  potential combinations for weight derivatives contributions
!
        potp = pra
        potxp = pga+half*pgc
        potp2 = potp+potp
        potxp2 = potxp+potxp
!
!  density gradient at current point
!
        dx=gden(1)
        dy=gden(2)
        dz=gden(3)
!
!  zero out derivatives of densities and gradients
!
        denx = 0.0d0
        denxx = 0.0d0
        gdx = 0.0d0
        gdxx = 0.0d0
        gx = 0.0d0
        gxx = 0.0d0
!
!   initializations for weight derivatives
!
!   unlike the local case above, we do not multiply
!   the weight gradient by the potential at this stage
!
        gwtmax=zero
        do ia=1,natoms
          if(ia.ne.icntr)then
            gwtmax=max(gwtmax,abs(gwt(1,ia)),abs(gwt(2,ia)),abs(gwt(3,ia)))
          endif
        enddo
!
!    form the atomic gradient and hessian  of density
!    and density gradient at current grid point
!
!    note:
!    for the closed shell case, the array da contains the
!    total density matrix, thus the factor two in the definition
!    of the gradient and hessian densities is omitted here, so
!    denx and denxx will contain half the atomic gradient and hessian
!    of the total closed shell density. likevise, gdx and gdxx will
!    contain half the atomic gradient and hessian of the total closed
!    shell density gradient
!
!
!    here it is too complex to compute cutoffs ad hoc for
!    the contribution to the various  densities, thus we
!    will only check the contributions against the global threshold.
!    in addition, we just check whether the second derivatives
!    of density and density gradient have to be computed at all.
!
        abra=abs(ra)
        dodenxx=abra.gt.epsi
        icount=quick_dft_grid%basf_counter(Ibin)+1
        DO 220 i=icount,quick_dft_grid%basf_counter(Ibin+1)
          ii=quick_dft_grid%basf(i)+1
          dmx = dm(ii)       ! max. element of first order density
          if(dmx.gt.epsi)then
             thtest=thrsh/(vmx*dmx)
          else
            goto 220
          endif
          it = (ii*(ii-1))/2
          iatm = quick_basis%ncenter(ii)
          if(iatm.eq.icntr) goto 220
          valx = vaox(1,ii)
          valy = vaox(2,ii)
          valz = vaox(3,ii)
          valm = max(abs(valx),abs(valy),abs(valz))
          valxx = vaoxx(1,ii)
          valxy = vaoxx(2,ii)
          valyy = vaoxx(3,ii)
          valxz = vaoxx(4,ii)
          valyz = vaoxx(5,ii)
          valzz = vaoxx(6,ii)
          valmm1 =max(abs(valxx),abs(valxy),abs(valyy))
          valmm2 =max(abs(valxz),abs(valyz),abs(valzz))
          valmm=max(valmm1,valmm2)
          if(vmx+valmm.lt.thtest)goto 220
          do 218 j=icount,i
            jj = quick_dft_grid%basf(j)+1
            ij = it + jj
            valj=vao(jj)
            jatm = quick_basis%ncenter(jj)
            abvj=abs(valj)
            valjx=vaox(1,jj)
            valjy=vaox(2,jj)
            valjz=vaox(3,jj)
            abvvj=max(abs(valjx),abs(valjy),abs(valjz))
            daijt=da(ij)
            abijt=abs(daijt)
            daij = daijt*valj
            abij=abs(daij)
!
! -- atomic gradient of density
!
              if(abij*valm.gt.thrsh)then
                denx(1,iatm) = denx(1,iatm) - daij*valx
                denx(2,iatm) = denx(2,iatm) - daij*valy
                denx(3,iatm) = denx(3,iatm) - daij*valz
              endif
!
! -- atomic gradient of density gradient
!
             if(abijt*(abvj*valmm+abvvj*valm).gt.thrsh)then
              gdx(1,1,iatm)=gdx(1,1,iatm)-daijt*(valj*valxx+valx*valjx)
              gdx(1,2,iatm)=gdx(1,2,iatm)-daijt*(valj*valxy+valy*valjx)
              gdx(1,3,iatm)=gdx(1,3,iatm)-daijt*(valj*valxz+valz*valjx)
              gdx(2,1,iatm)=gdx(2,1,iatm)-daijt*(valj*valxy+valx*valjy)
              gdx(2,2,iatm)=gdx(2,2,iatm)-daijt*(valj*valyy+valy*valjy)
              gdx(2,3,iatm)=gdx(2,3,iatm)-daijt*(valj*valyz+valz*valjy)
              gdx(3,1,iatm)=gdx(3,1,iatm)-daijt*(valj*valxz+valx*valjz)
              gdx(3,2,iatm)=gdx(3,2,iatm)-daijt*(valj*valyz+valy*valjz)
              gdx(3,3,iatm)=gdx(3,3,iatm)-daijt*(valj*valzz+valz*valjz)
             endif
!
! -- (a) one center terms: iatm with iatm
!
!
! -- atomic hessian of density
!
              if(dodenxx.and.abij*valmm.gt.thrsh)then
                denxx(1,iatm,1,iatm)=denxx(1,iatm,1,iatm)+daij*valxx
                xyc=daij*valxy
                denxx(1,iatm,2,iatm)=denxx(1,iatm,2,iatm)+xyc
                denxx(2,iatm,1,iatm)=denxx(2,iatm,1,iatm)+xyc
                denxx(2,iatm,2,iatm)=denxx(2,iatm,2,iatm)+daij*valyy
                xzc=daij*valxz
                denxx(1,iatm,3,iatm)=denxx(1,iatm,3,iatm)+xzc
                denxx(3,iatm,1,iatm)=denxx(3,iatm,1,iatm)+xzc
                yzc=daij*valyz
                denxx(2,iatm,3,iatm)=denxx(2,iatm,3,iatm)+yzc
                denxx(3,iatm,2,iatm)=denxx(3,iatm,2,iatm)+yzc
                denxx(3,iatm,3,iatm)=denxx(3,iatm,3,iatm)+daij*valzz
              endif
!
! -- atomic hessian of density gradient
!
              if(abijt*(abvj*vmx+abvvj*valmm).gt.thrsh)then
                gdxx(1,1,iatm,1,iatm)=gdxx(1,1,iatm,1,iatm)+ &
                         daijt * (vaoxxx(1,ii)*valj+valxx*valjx)
                gdxx(1,1,iatm,2,iatm)=gdxx(1,1,iatm,2,iatm)+ &
                         daijt * (vaoxxx(2,ii)*valj+valxy*valjx)
                gdxx(1,2,iatm,1,iatm)=gdxx(1,2,iatm,1,iatm)+ &
                         daijt * (vaoxxx(2,ii)*valj+valxy*valjx)
                gdxx(1,2,iatm,2,iatm)=gdxx(1,2,iatm,2,iatm)+ &
                         daijt * (vaoxxx(3,ii)*valj+valyy*valjx)
                gdxx(1,1,iatm,3,iatm)=gdxx(1,1,iatm,3,iatm)+ &
                         daijt * (vaoxxx(5,ii)*valj+valxz*valjx)
                gdxx(1,3,iatm,1,iatm)=gdxx(1,3,iatm,1,iatm)+ &
                         daijt * (vaoxxx(5,ii)*valj+valxz*valjx)
                gdxx(1,2,iatm,3,iatm)=gdxx(1,2,iatm,3,iatm)+ &
                         daijt * (vaoxxx(6,ii)*valj+valyz*valjx)
                gdxx(1,3,iatm,2,iatm)=gdxx(1,3,iatm,2,iatm)+ &
                         daijt * (vaoxxx(6,ii)*valj+valyz*valjx)
                gdxx(1,3,iatm,3,iatm)=gdxx(1,3,iatm,3,iatm)+ &
                         daijt * (vaoxxx(8,ii)*valj+valzz*valjx)
!
                gdxx(2,1,iatm,1,iatm)=gdxx(2,1,iatm,1,iatm)+ &
                         daijt * (vaoxxx(2,ii)*valj+valxx*valjy)
                gdxx(2,1,iatm,2,iatm)=gdxx(2,1,iatm,2,iatm)+ &
                         daijt * (vaoxxx(3,ii)*valj+valxy*valjy)
                gdxx(2,2,iatm,1,iatm)=gdxx(2,2,iatm,1,iatm)+ &
                         daijt * (vaoxxx(3,ii)*valj+valxy*valjy)
                gdxx(2,2,iatm,2,iatm)=gdxx(2,2,iatm,2,iatm)+ &
                         daijt * (vaoxxx(4,ii)*valj+valyy*valjy)
                gdxx(2,1,iatm,3,iatm)=gdxx(2,1,iatm,3,iatm)+ &
                         daijt * (vaoxxx(6,ii)*valj+valxz*valjy)
                gdxx(2,3,iatm,1,iatm)=gdxx(2,3,iatm,1,iatm)+ &
                         daijt * (vaoxxx(6,ii)*valj+valxz*valjy)
                gdxx(2,2,iatm,3,iatm)=gdxx(2,2,iatm,3,iatm)+ &
                         daijt * (vaoxxx(7,ii)*valj+valyz*valjy)
                gdxx(2,3,iatm,2,iatm)=gdxx(2,3,iatm,2,iatm)+ &
                         daijt * (vaoxxx(7,ii)*valj+valyz*valjy)
                gdxx(2,3,iatm,3,iatm)=gdxx(2,3,iatm,3,iatm)+ &
                         daijt * (vaoxxx(9,ii)*valj+valzz*valjy)

                gdxx(3,1,iatm,1,iatm)=gdxx(3,1,iatm,1,iatm)+ &
                         daijt * (vaoxxx(5,ii)*valj+valxx*valjz)
                gdxx(3,1,iatm,2,iatm)=gdxx(3,1,iatm,2,iatm)+ &
                         daijt * (vaoxxx(6,ii)*valj+valxy*valjz)
                gdxx(3,2,iatm,1,iatm)=gdxx(3,2,iatm,1,iatm)+ &
                         daijt * (vaoxxx(6,ii)*valj+valxy*valjz)
                gdxx(3,2,iatm,2,iatm)=gdxx(3,2,iatm,2,iatm)+ &
                         daijt * (vaoxxx(7,ii)*valj+valyy*valjz)
                gdxx(3,1,iatm,3,iatm)=gdxx(3,1,iatm,3,iatm)+ &
                         daijt * (vaoxxx(8,ii)*valj+valxz*valjz)
                gdxx(3,3,iatm,1,iatm)=gdxx(3,3,iatm,1,iatm)+ &
                         daijt * (vaoxxx(8,ii)*valj+valxz*valjz)
                gdxx(3,2,iatm,3,iatm)=gdxx(3,2,iatm,3,iatm)+ &
                         daijt * (vaoxxx(9,ii)*valj+valyz*valjz)
                gdxx(3,3,iatm,2,iatm)=gdxx(3,3,iatm,2,iatm)+ &
                         daijt * (vaoxxx(9,ii)*valj+valyz*valjz)
                gdxx(3,3,iatm,3,iatm)=gdxx(3,3,iatm,3,iatm)+ &
                         daijt * (vaoxxx(10,ii)*valj+valzz*valjz)
              endif
!
! -- (b) two center terms: iatm with jatm
            if(jatm.lt.iatm) go to 218
            if(jatm.eq.icntr)goto 218
!
! -- atomic hessian of density
!
              if(dodenxx.and.abijt*abvvj*valm.gt.thrsh)then
                daij = daijt*valjx
                denxx(1,iatm,1,jatm)=denxx(1,iatm,1,jatm)+daij*valx
                denxx(2,iatm,1,jatm)=denxx(2,iatm,1,jatm)+daij*valy
                denxx(3,iatm,1,jatm)=denxx(3,iatm,1,jatm)+daij*valz
                daij = daijt*valjy
                denxx(1,iatm,2,jatm)=denxx(1,iatm,2,jatm)+daij*valx
                denxx(2,iatm,2,jatm)=denxx(2,iatm,2,jatm)+daij*valy
                denxx(3,iatm,2,jatm)=denxx(3,iatm,2,jatm)+daij*valz
                daij = daijt*valjz
                denxx(1,iatm,3,jatm)=denxx(1,iatm,3,jatm)+daij*valx
                denxx(2,iatm,3,jatm)=denxx(2,iatm,3,jatm)+daij*valy
                denxx(3,iatm,3,jatm)=denxx(3,iatm,3,jatm)+daij*valz
              endif
!
! -- atomic hessian of density gradient
!
              if(abijt*(abvvj*valmm+vmx*valm).gt.thrsh)then
                gdxx(1,1,iatm,1,jatm)=gdxx(1,1,iatm,1,jatm)+ &
                            daijt * (valxx*valjx+valx*vaoxx(1,jj))
                gdxx(1,1,iatm,2,jatm)=gdxx(1,1,iatm,2,jatm)+ &
                            daijt * (valxx*valjy+valx*vaoxx(2,jj))
                gdxx(1,2,iatm,1,jatm)=gdxx(1,2,iatm,1,jatm)+ &
                            daijt * (valxy*valjx+valy*vaoxx(1,jj))
                gdxx(1,2,iatm,2,jatm)=gdxx(1,2,iatm,2,jatm)+ &
                            daijt * (valxy*valjy+valy*vaoxx(2,jj))
                gdxx(1,1,iatm,3,jatm)=gdxx(1,1,iatm,3,jatm)+ &
                            daijt * (valxx*valjz+valx*vaoxx(4,jj))
                gdxx(1,3,iatm,1,jatm)=gdxx(1,3,iatm,1,jatm)+ &
                            daijt * (valxz*valjx+valz*vaoxx(1,jj))
                gdxx(1,2,iatm,3,jatm)=gdxx(1,2,iatm,3,jatm)+ &
                            daijt * (valxy*valjz+valy*vaoxx(4,jj))
                gdxx(1,3,iatm,2,jatm)=gdxx(1,3,iatm,2,jatm)+ &
                            daijt * (valxz*valjy+valz*vaoxx(2,jj))
                gdxx(1,3,iatm,3,jatm)=gdxx(1,3,iatm,3,jatm)+ &
                            daijt * (valxz*valjz+valz*vaoxx(4,jj))
!
                gdxx(2,1,iatm,1,jatm)=gdxx(2,1,iatm,1,jatm)+ &
                            daijt * (valxy*valjx+valx*vaoxx(2,jj))
                gdxx(2,1,iatm,2,jatm)=gdxx(2,1,iatm,2,jatm)+ &
                            daijt * (valxy*valjy+valx*vaoxx(3,jj))
                gdxx(2,2,iatm,1,jatm)=gdxx(2,2,iatm,1,jatm)+ &
                            daijt * (valyy*valjx+valy*vaoxx(2,jj))
                gdxx(2,2,iatm,2,jatm)=gdxx(2,2,iatm,2,jatm)+ &
                            daijt * (valyy*valjy+valy*vaoxx(3,jj))
                gdxx(2,1,iatm,3,jatm)=gdxx(2,1,iatm,3,jatm)+ &
                            daijt * (valxy*valjz+valx*vaoxx(5,jj))
                gdxx(2,3,iatm,1,jatm)=gdxx(2,3,iatm,1,jatm)+ &
                            daijt * (valyz*valjx+valz*vaoxx(2,jj))
                gdxx(2,2,iatm,3,jatm)=gdxx(2,2,iatm,3,jatm)+ &
                            daijt * (valyy*valjz+valy*vaoxx(5,jj))
                gdxx(2,3,iatm,2,jatm)=gdxx(2,3,iatm,2,jatm)+ &
                            daijt * (valyz*valjy+valz*vaoxx(3,jj))
                gdxx(2,3,iatm,3,jatm)=gdxx(2,3,iatm,3,jatm)+ &
                            daijt * (valyz*valjz+valz*vaoxx(5,jj))
! 
                gdxx(3,1,iatm,1,jatm)=gdxx(3,1,iatm,1,jatm)+ &
                            daijt * (valxz*valjx+valx*vaoxx(4,jj))
                gdxx(3,1,iatm,2,jatm)=gdxx(3,1,iatm,2,jatm)+ &
                            daijt * (valxz*valjy+valx*vaoxx(5,jj))
                gdxx(3,2,iatm,1,jatm)=gdxx(3,2,iatm,1,jatm)+ &
                            daijt * (valyz*valjx+valy*vaoxx(4,jj))
                gdxx(3,2,iatm,2,jatm)=gdxx(3,2,iatm,2,jatm)+ &
                            daijt * (valyz*valjy+valy*vaoxx(5,jj))
                gdxx(3,1,iatm,3,jatm)=gdxx(3,1,iatm,3,jatm)+ &
                            daijt * (valxz*valjz+valx*vaoxx(6,jj))
                gdxx(3,3,iatm,1,jatm)=gdxx(3,3,iatm,1,jatm)+ &
                            daijt * (valzz*valjx+valz*vaoxx(4,jj))
                gdxx(3,2,iatm,3,jatm)=gdxx(3,2,iatm,3,jatm)+ &
                            daijt * (valyz*valjz+valy*vaoxx(6,jj))
                gdxx(3,3,iatm,2,jatm)=gdxx(3,3,iatm,2,jatm)+ &
                            daijt * (valzz*valjy+valz*vaoxx(5,jj))
                gdxx(3,3,iatm,3,jatm)=gdxx(3,3,iatm,3,jatm)+ &
                            daijt * (valzz*valjz+valz*vaoxx(6,jj))
              endif
 218      continue
          do 219 j=i+1,quick_dft_grid%basf_counter(Ibin+1)
            jj = quick_dft_grid%basf(j)+1
            ij = (jj*(jj-1))/2 + ii
            jatm = quick_basis%ncenter(jj)
            valj=vao(jj)
            abvj=abs(valj)
            valjx=vaox(1,jj)
            valjy=vaox(2,jj)
            valjz=vaox(3,jj)
            abvvj=max(abs(valjx),abs(valjy),abs(valjz))
            daijt= da(ij)
            abijt=abs(daijt)
            daij = daijt*valj
            abij=abs(daij)
!
! -- atomic gradient of density
!
              if(abij*valm.gt.thrsh)then
                denx(1,iatm) = denx(1,iatm) - daij*valx
                denx(2,iatm) = denx(2,iatm) - daij*valy
                denx(3,iatm) = denx(3,iatm) - daij*valz
              endif
!
! -- atomic gradient of density gradient
!
             if(abijt*(abvj*valmm+abvvj*valm).gt.thrsh)then
              gdx(1,1,iatm)=gdx(1,1,iatm)-daijt*(valj*valxx+valx*valjx)
              gdx(1,2,iatm)=gdx(1,2,iatm)-daijt*(valj*valxy+valy*valjx)
              gdx(1,3,iatm)=gdx(1,3,iatm)-daijt*(valj*valxz+valz*valjx)
              gdx(2,1,iatm)=gdx(2,1,iatm)-daijt*(valj*valxy+valx*valjy)
              gdx(2,2,iatm)=gdx(2,2,iatm)-daijt*(valj*valyy+valy*valjy)
              gdx(2,3,iatm)=gdx(2,3,iatm)-daijt*(valj*valyz+valz*valjy)
              gdx(3,1,iatm)=gdx(3,1,iatm)-daijt*(valj*valxz+valx*valjz)
              gdx(3,2,iatm)=gdx(3,2,iatm)-daijt*(valj*valyz+valy*valjz)
              gdx(3,3,iatm)=gdx(3,3,iatm)-daijt*(valj*valzz+valz*valjz)
             endif
!
! -- (a) one center terms: iatm with iatm
!
!
! -- atomic hessian of density
!
            if(dodenxx.and.abij*valmm.gt.thrsh)then
              denxx(1,iatm,1,iatm)=denxx(1,iatm,1,iatm)+daij*valxx
              xyc=daij*valxy
              denxx(1,iatm,2,iatm)=denxx(1,iatm,2,iatm)+xyc
              denxx(2,iatm,1,iatm)=denxx(2,iatm,1,iatm)+xyc
              denxx(2,iatm,2,iatm)=denxx(2,iatm,2,iatm)+daij*valyy
              xzc=daij*valxz
              denxx(1,iatm,3,iatm)=denxx(1,iatm,3,iatm)+xzc
              denxx(3,iatm,1,iatm)=denxx(3,iatm,1,iatm)+xzc
              yzc=daij*valyz
              denxx(2,iatm,3,iatm)=denxx(2,iatm,3,iatm)+yzc
              denxx(3,iatm,2,iatm)=denxx(3,iatm,2,iatm)+yzc
              denxx(3,iatm,3,iatm)=denxx(3,iatm,3,iatm)+daij*valzz
            endif
!
! -- atomic hessian of density gradient
!
              if(abijt*(abvj*vmx+abvvj*valmm).gt.thrsh)then
                gdxx(1,1,iatm,1,iatm)=gdxx(1,1,iatm,1,iatm)+ &
                         daijt * (vaoxxx(1,ii)*valj+valxx*valjx)
                gdxx(1,1,iatm,2,iatm)=gdxx(1,1,iatm,2,iatm)+ &
                         daijt * (vaoxxx(2,ii)*valj+valxy*valjx)
                gdxx(1,2,iatm,1,iatm)=gdxx(1,2,iatm,1,iatm)+ &
                         daijt * (vaoxxx(2,ii)*valj+valxy*valjx)
                gdxx(1,2,iatm,2,iatm)=gdxx(1,2,iatm,2,iatm)+ &
                         daijt * (vaoxxx(3,ii)*valj+valyy*valjx)
                gdxx(1,1,iatm,3,iatm)=gdxx(1,1,iatm,3,iatm)+ &
                         daijt * (vaoxxx(5,ii)*valj+valxz*valjx)
                gdxx(1,3,iatm,1,iatm)=gdxx(1,3,iatm,1,iatm)+ &
                         daijt * (vaoxxx(5,ii)*valj+valxz*valjx)
                gdxx(1,2,iatm,3,iatm)=gdxx(1,2,iatm,3,iatm)+ &
                         daijt * (vaoxxx(6,ii)*valj+valyz*valjx)
                gdxx(1,3,iatm,2,iatm)=gdxx(1,3,iatm,2,iatm)+ &
                         daijt * (vaoxxx(6,ii)*valj+valyz*valjx)
                gdxx(1,3,iatm,3,iatm)=gdxx(1,3,iatm,3,iatm)+ &
                         daijt * (vaoxxx(8,ii)*valj+valzz*valjx)
!  
                gdxx(2,1,iatm,1,iatm)=gdxx(2,1,iatm,1,iatm)+ &
                         daijt * (vaoxxx(2,ii)*valj+valxx*valjy)
                gdxx(2,1,iatm,2,iatm)=gdxx(2,1,iatm,2,iatm)+ &
                         daijt * (vaoxxx(3,ii)*valj+valxy*valjy)
                gdxx(2,2,iatm,1,iatm)=gdxx(2,2,iatm,1,iatm)+ &
                         daijt * (vaoxxx(3,ii)*valj+valxy*valjy)
                gdxx(2,2,iatm,2,iatm)=gdxx(2,2,iatm,2,iatm)+ &
                         daijt * (vaoxxx(4,ii)*valj+valyy*valjy)
                gdxx(2,1,iatm,3,iatm)=gdxx(2,1,iatm,3,iatm)+ &
                         daijt * (vaoxxx(6,ii)*valj+valxz*valjy)
                gdxx(2,3,iatm,1,iatm)=gdxx(2,3,iatm,1,iatm)+ &
                       daijt * (vaoxxx(6,ii)*valj+valxz*valjy)
                gdxx(2,2,iatm,3,iatm)=gdxx(2,2,iatm,3,iatm)+ &
                         daijt * (vaoxxx(7,ii)*valj+valyz*valjy)
                gdxx(2,3,iatm,2,iatm)=gdxx(2,3,iatm,2,iatm)+ &
                         daijt * (vaoxxx(7,ii)*valj+valyz*valjy)
                gdxx(2,3,iatm,3,iatm)=gdxx(2,3,iatm,3,iatm)+ &
                         daijt * (vaoxxx(9,ii)*valj+valzz*valjy)
!  
                gdxx(3,1,iatm,1,iatm)=gdxx(3,1,iatm,1,iatm)+ &
                         daijt * (vaoxxx(5,ii)*valj+valxx*valjz)
                gdxx(3,1,iatm,2,iatm)=gdxx(3,1,iatm,2,iatm)+ &
                         daijt * (vaoxxx(6,ii)*valj+valxy*valjz)
                gdxx(3,2,iatm,1,iatm)=gdxx(3,2,iatm,1,iatm)+ &
                         daijt * (vaoxxx(6,ii)*valj+valxy*valjz)
                gdxx(3,2,iatm,2,iatm)=gdxx(3,2,iatm,2,iatm)+ &
                         daijt * (vaoxxx(7,ii)*valj+valyy*valjz)
                gdxx(3,1,iatm,3,iatm)=gdxx(3,1,iatm,3,iatm)+ &
                         daijt * (vaoxxx(8,ii)*valj+valxz*valjz)
                gdxx(3,3,iatm,1,iatm)=gdxx(3,3,iatm,1,iatm)+ &
                         daijt * (vaoxxx(8,ii)*valj+valxz*valjz)
                gdxx(3,2,iatm,3,iatm)=gdxx(3,2,iatm,3,iatm)+ &
                         daijt * (vaoxxx(9,ii)*valj+valyz*valjz)
                gdxx(3,3,iatm,2,iatm)=gdxx(3,3,iatm,2,iatm)+ &
                         daijt * (vaoxxx(9,ii)*valj+valyz*valjz)
                gdxx(3,3,iatm,3,iatm)=gdxx(3,3,iatm,3,iatm)+ &
                         daijt * (vaoxxx(10,ii)*valj+valzz*valjz)
              endif
!
! -- (b) two center terms: iatm with jatm
!
            if(jatm.lt.iatm) go to 219
            if(jatm.eq.icntr)goto 219
!
! -- atomic hessian of density
!
              if(dodenxx.and.abijt*abvvj*valm.gt.thrsh)then
                daij = daijt*valjx
                denxx(1,iatm,1,jatm)=denxx(1,iatm,1,jatm)+daij*valx
                denxx(2,iatm,1,jatm)=denxx(2,iatm,1,jatm)+daij*valy
                denxx(3,iatm,1,jatm)=denxx(3,iatm,1,jatm)+daij*valz
                daij = daijt*valjy
                denxx(1,iatm,2,jatm)=denxx(1,iatm,2,jatm)+daij*valx
                denxx(2,iatm,2,jatm)=denxx(2,iatm,2,jatm)+daij*valy
                denxx(3,iatm,2,jatm)=denxx(3,iatm,2,jatm)+daij*valz
                daij = daijt*valjz
                denxx(1,iatm,3,jatm)=denxx(1,iatm,3,jatm)+daij*valx
                denxx(2,iatm,3,jatm)=denxx(2,iatm,3,jatm)+daij*valy
                denxx(3,iatm,3,jatm)=denxx(3,iatm,3,jatm)+daij*valz
              endif
!
! -- atomic hessian of density gradient
!
              if(abijt*(abvvj*valmm+vmx*valm).gt.thrsh)then
                gdxx(1,1,iatm,1,jatm)=gdxx(1,1,iatm,1,jatm)+ &
                            daijt * (valxx*valjx+valx*vaoxx(1,jj))
                gdxx(1,1,iatm,2,jatm)=gdxx(1,1,iatm,2,jatm)+ &
                            daijt * (valxx*valjy+valx*vaoxx(2,jj))
                gdxx(1,2,iatm,1,jatm)=gdxx(1,2,iatm,1,jatm)+ &
                            daijt * (valxy*valjx+valy*vaoxx(1,jj))
                gdxx(1,2,iatm,2,jatm)=gdxx(1,2,iatm,2,jatm)+ &
                            daijt * (valxy*valjy+valy*vaoxx(2,jj))
                gdxx(1,1,iatm,3,jatm)=gdxx(1,1,iatm,3,jatm)+ &
                            daijt * (valxx*valjz+valx*vaoxx(4,jj))
                gdxx(1,3,iatm,1,jatm)=gdxx(1,3,iatm,1,jatm)+ &
                            daijt * (valxz*valjx+valz*vaoxx(1,jj))
                gdxx(1,2,iatm,3,jatm)=gdxx(1,2,iatm,3,jatm)+ &
                            daijt * (valxy*valjz+valy*vaoxx(4,jj))
                gdxx(1,3,iatm,2,jatm)=gdxx(1,3,iatm,2,jatm)+ &
                            daijt * (valxz*valjy+valz*vaoxx(2,jj))
                gdxx(1,3,iatm,3,jatm)=gdxx(1,3,iatm,3,jatm)+ &
                            daijt * (valxz*valjz+valz*vaoxx(4,jj))
!    
                gdxx(2,1,iatm,1,jatm)=gdxx(2,1,iatm,1,jatm)+ &
                            daijt * (valxy*valjx+valx*vaoxx(2,jj))
                gdxx(2,1,iatm,2,jatm)=gdxx(2,1,iatm,2,jatm)+ &
                            daijt * (valxy*valjy+valx*vaoxx(3,jj))
                gdxx(2,2,iatm,1,jatm)=gdxx(2,2,iatm,1,jatm)+ &
                            daijt * (valyy*valjx+valy*vaoxx(2,jj))
                gdxx(2,2,iatm,2,jatm)=gdxx(2,2,iatm,2,jatm)+ &
                            daijt * (valyy*valjy+valy*vaoxx(3,jj))
                gdxx(2,1,iatm,3,jatm)=gdxx(2,1,iatm,3,jatm)+ &
                            daijt * (valxy*valjz+valx*vaoxx(5,jj))
                gdxx(2,3,iatm,1,jatm)=gdxx(2,3,iatm,1,jatm)+ &
                            daijt * (valyz*valjx+valz*vaoxx(2,jj))
                gdxx(2,2,iatm,3,jatm)=gdxx(2,2,iatm,3,jatm)+ &
                            daijt * (valyy*valjz+valy*vaoxx(5,jj))
                gdxx(2,3,iatm,2,jatm)=gdxx(2,3,iatm,2,jatm)+ &
                            daijt * (valyz*valjy+valz*vaoxx(3,jj))
                gdxx(2,3,iatm,3,jatm)=gdxx(2,3,iatm,3,jatm)+ &
                            daijt * (valyz*valjz+valz*vaoxx(5,jj))
!   
                gdxx(3,1,iatm,1,jatm)=gdxx(3,1,iatm,1,jatm)+ &
                            daijt * (valxz*valjx+valx*vaoxx(4,jj))
                gdxx(3,1,iatm,2,jatm)=gdxx(3,1,iatm,2,jatm)+ &
                            daijt * (valxz*valjy+valx*vaoxx(5,jj))
                gdxx(3,2,iatm,1,jatm)=gdxx(3,2,iatm,1,jatm)+ &
                            daijt * (valyz*valjx+valy*vaoxx(4,jj))
                gdxx(3,2,iatm,2,jatm)=gdxx(3,2,iatm,2,jatm)+ &
                            daijt * (valyz*valjy+valy*vaoxx(5,jj))
                gdxx(3,1,iatm,3,jatm)=gdxx(3,1,iatm,3,jatm)+ &
                            daijt * (valxz*valjz+valx*vaoxx(6,jj))
                gdxx(3,3,iatm,1,jatm)=gdxx(3,3,iatm,1,jatm)+ &
                            daijt * (valzz*valjx+valz*vaoxx(4,jj))
                gdxx(3,2,iatm,3,jatm)=gdxx(3,2,iatm,3,jatm)+ &
                            daijt * (valyz*valjz+valy*vaoxx(6,jj))
                gdxx(3,3,iatm,2,jatm)=gdxx(3,3,iatm,2,jatm)+ &
                            daijt * (valzz*valjy+valz*vaoxx(5,jj))
                gdxx(3,3,iatm,3,jatm)=gdxx(3,3,iatm,3,jatm)+ &
                            daijt * (valzz*valjz+valz*vaoxx(6,jj))
              endif 
 219      continue
 220    continue
!
!  form atomic gradient and hessian of density gradient invariant
!
        do iatm=1,natoms
          gx(1,iatm)=two*(dx*gdx(1,1,iatm)+ &
                        dy*gdx(2,1,iatm)+dz*gdx(3,1,iatm))
          gx(2,iatm)=two*(dx*gdx(1,2,iatm)+ &
                        dy*gdx(2,2,iatm)+dz*gdx(3,2,iatm))
          gx(3,iatm)=two*(dx*gdx(1,3,iatm)+ &
                        dy*gdx(2,3,iatm)+dz*gdx(3,3,iatm))
        enddo
!
        do iatm=1,natoms
          do jatm=iatm,natoms
            gxx(1,iatm,1,jatm)=two*( &
              dx*gdxx(1,1,iatm,1,jatm)+gdx(1,1,iatm)*gdx(1,1,jatm)+ &
              dy*gdxx(2,1,iatm,1,jatm)+gdx(2,1,iatm)*gdx(2,1,jatm)+ &
              dz*gdxx(3,1,iatm,1,jatm)+gdx(3,1,iatm)*gdx(3,1,jatm))
            gxx(1,iatm,2,jatm)=two*( &
              dx*gdxx(1,1,iatm,2,jatm)+gdx(1,1,iatm)*gdx(1,2,jatm)+ &
              dy*gdxx(2,1,iatm,2,jatm)+gdx(2,1,iatm)*gdx(2,2,jatm)+ &
              dz*gdxx(3,1,iatm,2,jatm)+gdx(3,1,iatm)*gdx(3,2,jatm))
            gxx(2,iatm,1,jatm)=two*( &
              dx*gdxx(1,2,iatm,1,jatm)+gdx(1,2,iatm)*gdx(1,1,jatm)+ &
              dy*gdxx(2,2,iatm,1,jatm)+gdx(2,2,iatm)*gdx(2,1,jatm)+ &
              dz*gdxx(3,2,iatm,1,jatm)+gdx(3,2,iatm)*gdx(3,1,jatm))
            gxx(2,iatm,2,jatm)=two*( &
              dx*gdxx(1,2,iatm,2,jatm)+gdx(1,2,iatm)*gdx(1,2,jatm)+ &
              dy*gdxx(2,2,iatm,2,jatm)+gdx(2,2,iatm)*gdx(2,2,jatm)+ &
              dz*gdxx(3,2,iatm,2,jatm)+gdx(3,2,iatm)*gdx(3,2,jatm))
            gxx(1,iatm,3,jatm)=two*( &
              dx*gdxx(1,1,iatm,3,jatm)+gdx(1,1,iatm)*gdx(1,3,jatm)+ &
              dy*gdxx(2,1,iatm,3,jatm)+gdx(2,1,iatm)*gdx(2,3,jatm)+ &
              dz*gdxx(3,1,iatm,3,jatm)+gdx(3,1,iatm)*gdx(3,3,jatm))
            gxx(3,iatm,1,jatm)=two*( &
              dx*gdxx(1,3,iatm,1,jatm)+gdx(1,3,iatm)*gdx(1,1,jatm)+ &
              dy*gdxx(2,3,iatm,1,jatm)+gdx(2,3,iatm)*gdx(2,1,jatm)+ &
              dz*gdxx(3,3,iatm,1,jatm)+gdx(3,3,iatm)*gdx(3,1,jatm))
            gxx(2,iatm,3,jatm)=two*( &
              dx*gdxx(1,2,iatm,3,jatm)+gdx(1,2,iatm)*gdx(1,3,jatm)+ &
              dy*gdxx(2,2,iatm,3,jatm)+gdx(2,2,iatm)*gdx(2,3,jatm)+ &
              dz*gdxx(3,2,iatm,3,jatm)+gdx(3,2,iatm)*gdx(3,3,jatm))
            gxx(3,iatm,2,jatm)=two*( &
              dx*gdxx(1,3,iatm,2,jatm)+gdx(1,3,iatm)*gdx(1,2,jatm)+ &
              dy*gdxx(2,3,iatm,2,jatm)+gdx(2,3,iatm)*gdx(2,2,jatm)+ &
              dz*gdxx(3,3,iatm,2,jatm)+gdx(3,3,iatm)*gdx(3,2,jatm))
            gxx(3,iatm,3,jatm)=two*( &
              dx*gdxx(1,3,iatm,3,jatm)+gdx(1,3,iatm)*gdx(1,3,jatm)+ &
              dy*gdxx(2,3,iatm,3,jatm)+gdx(2,3,iatm)*gdx(2,3,jatm)+ &
              dz*gdxx(3,3,iatm,3,jatm)+gdx(3,3,iatm)*gdx(3,3,jatm))
          enddo
        enddo
!
! -- now form the coefficients that multiply the basis functions and
!    their derivatives in the expression for the fock matrix
!    (i.e., quantities v, w and x at page 7436 of johnson and frisch).
!
        prg2=two*prg
        pgg2=two*pgg
        pg2=two*pg
        sxx=pg2*dx
        sxy=pg2*dy
        sxz=pg2*dz
        do iatm=1,natoms
          vd1=prg2*denx(1,iatm)+pgg2*gx(1,iatm)
          vd2=prg2*denx(2,iatm)+pgg2*gx(2,iatm)
          vd3=prg2*denx(3,iatm)+pgg2*gx(3,iatm)
          sv(1,iatm)=rara*denx(1,iatm)+prg*gx(1,iatm)
          sv(2,iatm)=rara*denx(2,iatm)+prg*gx(2,iatm)
          sv(3,iatm)=rara*denx(3,iatm)+prg*gx(3,iatm)
          sw(1,1,iatm)=pg2*gdx(1,1,iatm)+dx*vd1
          sw(2,1,iatm)=pg2*gdx(2,1,iatm)+dy*vd1
          sw(3,1,iatm)=pg2*gdx(3,1,iatm)+dz*vd1
          sw(1,2,iatm)=pg2*gdx(1,2,iatm)+dx*vd2
          sw(2,2,iatm)=pg2*gdx(2,2,iatm)+dy*vd2
          sw(3,2,iatm)=pg2*gdx(3,2,iatm)+dz*vd2
          sw(1,3,iatm)=pg2*gdx(1,3,iatm)+dx*vd3
          sw(2,3,iatm)=pg2*gdx(2,3,iatm)+dy*vd3
          sw(3,3,iatm)=pg2*gdx(3,3,iatm)+dz*vd3
        enddo
!
!    we are ready to perform the quadrature for the derivative
!    fock matrix.
!
! -- get the maximum absolute value of coefficients v and w
!
        call absmax(nat3,sv,isv,svmax)
        call absmax(3*nat3,sw,isw,swmax)
        tswmax=three*swmax
! -- global threshold testing
        vmx2=vmx*vmx
        smax = abs(sxx+sxy+sxz)
        dmax = abs(dx+dy+dz)
        vmax3 = (svmax+tswmax+tswmax)*vmx2
        vmax1 = (abra+two*smax)*vmx2
        dpmax2 = potxp2*dmax*(vmx2+vmx2)
        vmax2 = gwtmax*(potp*vmx2+dpmax2)
        vmax=max(vmax1,vmax2,vmax3)
        if(vmax.lt.thrsh)  go to 245
!
! -- numerical quadrature  for derivative fock matrix
!
        do 240 i=icount,quick_dft_grid%basf_counter(Ibin+1)
          ii=quick_dft_grid%basf(i)+1
          vali = vao(ii)
          valix = vaox(1,ii)
          valiy = vaox(2,ii)
          valiz = vaox(3,ii)
          xvi=sxx*valix+sxy*valiy+sxz*valiz+ra*vali
          sxxi=sxx*vali
          sxyi=sxy*vali
          sxzi=sxz*vali
          xxvx=sxx*vaoxx(1,ii)+sxy*vaoxx(2,ii)+sxz*vaoxx(4,ii)+ra*valix
          xxvy=sxx*vaoxx(2,ii)+sxy*vaoxx(3,ii)+sxz*vaoxx(5,ii)+ra*valiy
          xxvz=sxx*vaoxx(4,ii)+sxy*vaoxx(5,ii)+sxz*vaoxx(6,ii)+ra*valiz
          valm=abs(vali)
          valm1=abs(xvi)
          valmx=max(abs(valix),abs(valiy),abs(valiz))
          valmx1=max(abs(xxvx),abs(xxvy),abs(xxvz))
          val1=(valmx1+smax*valmx)*vmx
          smaxmx=smax*valm*vmx
          val2=valm1*vmx+smaxmx
          val3=(svmax*valm+tswmax*(valmx+valm))*vmx
          val4=gwtmax*(potp*valm*vmx+dpmax2)
          valt1=max(val1,val3)
          valt2=max(val3,val4)
          valtest=max(valt1,valt2)
          if(valtest.gt.thrsh)then
            it = (ii*(ii-1))/2
            iatm = quick_basis%ncenter(ii) 
            do 230 j=icount,i
              jj = quick_dft_grid%basf(j)+1
              ij = it + jj
              jatm = quick_basis%ncenter(jj)
              valj = vao(jj)
              valjx = vaox(1,jj)
              valjy = vaox(2,jj)
              valjz = vaox(3,jj)
              valmxj=max(abs(valjx),abs(valjy),abs(valjz))
              xvj=sxx*valjx+sxy*valjy+sxz*valjz
              if(valmx1*abs(valj)+valmx*abs(xvj).gt.thrsh)then
              if(iatm.ne.icntr) then
                fda(1,iatm,ij) = fda(1,iatm,ij) - xxvx*valj - valix*xvj
                fda(2,iatm,ij) = fda(2,iatm,ij) - xxvy*valj - valiy*xvj
                fda(3,iatm,ij) = fda(3,iatm,ij) - xxvz*valj - valiz*xvj
              endif
              endif
              if(valm1*valmxj+smaxmx.gt.thrsh)then
              if(jatm.ne.icntr) then
                fda(1,jatm,ij) = fda(1,jatm,ij) - xvi*valjx - &
                vaoxx(1,jj)*sxxi - vaoxx(2,jj)*sxyi - vaoxx(4,jj)*sxzi
                fda(2,jatm,ij) = fda(2,jatm,ij) - xvi*valjy - &
                vaoxx(2,jj)*sxxi - vaoxx(3,jj)*sxyi - vaoxx(5,jj)*sxzi
                fda(3,jatm,ij) = fda(3,jatm,ij) - xvi*valjz - &
                vaoxx(4,jj)*sxxi - vaoxx(5,jj)*sxyi - vaoxx(6,jj)*sxzi
              endif
              endif
              vij=vali*valj
              vijx=valix*valj+valjx*vali
              vijy=valiy*valj+valjy*vali
              vijz=valiz*valj+valjz*vali
              if(svmax*abs(vij)+swmax*abs(vijx+vijy+vijz).gt.thrsh)then
                do katm=1,natoms
                  fda(1,katm,ij) = fda(1,katm,ij) + (sv(1,katm)*vij+ &
                  sw(1,1,katm)*vijx+sw(2,1,katm)*vijy+sw(3,1,katm)*vijz)
                  fda(2,katm,ij) = fda(2,katm,ij) + (sv(2,katm)*vij+ &
                  sw(1,2,katm)*vijx+sw(2,2,katm)*vijy+sw(3,2,katm)*vijz)
                  fda(3,katm,ij) = fda(3,katm,ij) + (sv(3,katm)*vij+ &
                  sw(1,3,katm)*vijx+sw(2,3,katm)*vijy+sw(3,3,katm)*vijz)
                enddo
              endif
!
! -- weight derivatives contribution
!
              gij=potp*vij+potxp2*(dx*vijx+dy*vijy+dz*vijz)
              if(gwtmax*abs(gij).gt.thrsh)then
                do ia=1,natoms
                  if(ia.ne.icntr)then
                    fda(1,ia,ij)=fda(1,ia,ij)+gwt(1,ia)*gij
                    fda(2,ia,ij)=fda(2,ia,ij)+gwt(2,ia)*gij
                    fda(3,ia,ij)=fda(3,ia,ij)+gwt(3,ia)*gij
                  endif
                enddo
              endif
 230        continue
          endif
 240    continue
!
!   numerical quadrature for direct contribution to hessian matrix
!
 245    continue
!
! -- direct contribution to hessian matrix.
!    a factor two is applied
!
        ra2=two*ra
        rara2=two*rara
        do iatm=1,natoms
          hdx=rara2*denx(1,iatm)+prg2*gx(1,iatm)
          hdy=rara2*denx(2,iatm)+prg2*gx(2,iatm)
          hdz=rara2*denx(3,iatm)+prg2*gx(3,iatm)
          hgx=prg2*denx(1,iatm)+pgg2*gx(1,iatm)
          hgy=prg2*denx(2,iatm)+pgg2*gx(2,iatm)
          hgz=prg2*denx(3,iatm)+pgg2*gx(3,iatm)
          do jatm=iatm,natoms
       hess(1,iatm,1,jatm)=hess(1,iatm,1,jatm)+hdx*denx(1,jatm)+ &
       hgx*gx(1,jatm)+ra2*denxx(1,iatm,1,jatm)+pg2*gxx(1,iatm,1,jatm)
       hess(2,iatm,1,jatm)=hess(2,iatm,1,jatm)+hdy*denx(1,jatm)+ &
       hgy*gx(1,jatm)+ra2*denxx(2,iatm,1,jatm)+pg2*gxx(2,iatm,1,jatm)
       hess(3,iatm,1,jatm)=hess(3,iatm,1,jatm)+hdz*denx(1,jatm)+ &
       hgz*gx(1,jatm)+ra2*denxx(3,iatm,1,jatm)+pg2*gxx(3,iatm,1,jatm)
       hess(1,iatm,2,jatm)=hess(1,iatm,2,jatm)+hdx*denx(2,jatm)+ &
       hgx*gx(2,jatm)+ra2*denxx(1,iatm,2,jatm)+pg2*gxx(1,iatm,2,jatm)
       hess(2,iatm,2,jatm)=hess(2,iatm,2,jatm)+hdy*denx(2,jatm)+ &
       hgy*gx(2,jatm)+ra2*denxx(2,iatm,2,jatm)+pg2*gxx(2,iatm,2,jatm)
       hess(3,iatm,2,jatm)=hess(3,iatm,2,jatm)+hdz*denx(2,jatm)+ &
       hgz*gx(2,jatm)+ra2*denxx(3,iatm,2,jatm)+pg2*gxx(3,iatm,2,jatm)
       hess(1,iatm,3,jatm)=hess(1,iatm,3,jatm)+hdx*denx(3,jatm)+ &
       hgx*gx(3,jatm)+ra2*denxx(1,iatm,3,jatm)+pg2*gxx(1,iatm,3,jatm)
       hess(2,iatm,3,jatm)=hess(2,iatm,3,jatm)+hdy*denx(3,jatm)+ &
       hgy*gx(3,jatm)+ra2*denxx(2,iatm,3,jatm)+pg2*gxx(2,iatm,3,jatm)
       hess(3,iatm,3,jatm)=hess(3,iatm,3,jatm)+hdz*denx(3,jatm)+ &
       hgz*gx(3,jatm)+ra2*denxx(3,iatm,3,jatm)+pg2*gxx(3,iatm,3,jatm)
          enddo
        enddo
!
!  --  add weight derivatives contribution.
!      the sv array is used to store a partial sum
!
        do ia=1,natoms
          sv(1,ia)=potp2*denx(1,ia)+potxp2*gx(1,ia)
          sv(2,ia)=potp2*denx(2,ia)+potxp2*gx(2,ia)
          sv(3,ia)=potp2*denx(3,ia)+potxp2*gx(3,ia)
        enddo
        do ia=1,natoms
        if(ia.ne.icntr)then
          do ja=ia,natoms
          if(ja.ne.icntr)then
            hess(1,ia,1,ja)=hess(1,ia,1,ja)+gwt(1,ia)*sv(1,ja) &
                     +gwt(1,ja)*sv(1,ia)+hwt(1,ia,1,ja)
            hess(2,ia,1,ja)=hess(2,ia,1,ja)+gwt(2,ia)*sv(1,ja) &
                     +gwt(1,ja)*sv(2,ia)+hwt(2,ia,1,ja)
            hess(3,ia,1,ja)=hess(3,ia,1,ja)+gwt(3,ia)*sv(1,ja) &
                     +gwt(1,ja)*sv(3,ia)+hwt(3,ia,1,ja)
            hess(1,ia,2,ja)=hess(1,ia,2,ja)+gwt(1,ia)*sv(2,ja) &
                     +gwt(2,ja)*sv(1,ia)+hwt(1,ia,2,ja)
            hess(2,ia,2,ja)=hess(2,ia,2,ja)+gwt(2,ia)*sv(2,ja) &
                     +gwt(2,ja)*sv(2,ia)+hwt(2,ia,2,ja)
            hess(3,ia,2,ja)=hess(3,ia,2,ja)+gwt(3,ia)*sv(2,ja) &
                     +gwt(2,ja)*sv(3,ia)+hwt(3,ia,2,ja)
            hess(1,ia,3,ja)=hess(1,ia,3,ja)+gwt(1,ia)*sv(3,ja) &
                     +gwt(3,ja)*sv(1,ia)+hwt(1,ia,3,ja)
            hess(2,ia,3,ja)=hess(2,ia,3,ja)+gwt(2,ia)*sv(3,ja) &
                     +gwt(3,ja)*sv(2,ia)+hwt(2,ia,3,ja)
            hess(3,ia,3,ja)=hess(3,ia,3,ja)+gwt(3,ia)*sv(3,ja) &
                     +gwt(3,ja)*sv(3,ia)+hwt(3,ia,3,ja)
          endif
          enddo
        endif
        enddo
!
      return
      end subroutine formFdHess
! =====================================================================

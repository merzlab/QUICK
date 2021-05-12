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

  public :: computeLongRange

  ! module specific data structures
  double precision :: store(120,120)      ! store primitive integrals from vrr
  double precision :: RA(3), RB(3), RC(3) ! cartesian coordinates of the 3 centers
  double precision :: Zc, Cc              ! charge and a magic number of the external gaussian

  interface computeLongRange
    module procedure compute_long_range
  end interface computeLongRange

contains

  subroutine compute_long_range(c_coords, c_zeta, c_chg)

    !----------------------------------------------------------------------!
    ! This is the main driver for computing long range potential. The      !
    ! goal is to compute (ij|c) three center integral and add a potential  !
    ! into Fock matrix. Here i,j are two basis functions, c is a gaussian  !
    ! located at a certain distance. To make use of the existing ERI code, !
    ! we approximate above integral with (ij|c0) four center integral where!
    ! 0 is another gaussian with zero exponent.                            !
    !______________________________________________________________________!

    use quick_basis_module

    implicit none
    double precision, intent(in) :: c_coords(3), c_zeta, c_chg
    integer :: II, JJ         ! shell pairs

    ! set number of external gaussians and charge for testing
    !Cc=2.0000000000D+00
    !Zc=7.5000000000D-01
    !Rc(1)=1.5000000000D+00 
    !Rc(2)=2.5000000000D+00
    !RC(3)=3.5000000000D+00

    RC=c_coords
    Zc=c_zeta
    Cc=c_chg

    do II = 1, jshell
      do JJ = II, jshell  
        !II=2
        !JJ=1
        call compute_lngr_int(II,JJ)
      enddo
    enddo

  end subroutine compute_long_range


  subroutine compute_lngr_int(II,JJ)

    use quick_basis_module
    use quick_method_module
    use quick_molspec_module
    use quick_params_module

    integer, intent(in) :: II, JJ

    integer :: M, LL, NII1, NII2, NJJ1, NJJ2, NKK1, NKK2, NLL1, NLL2, NNAB, NNCD, NABCDTYPE, &
               NNA, NNC, NABCD, ITT, Nprij, Nprii, iitemp, I1, I2, I, J, K, L 
    double precision :: P(3), AAtemp(3), Ptemp(3), Q(3), W(3), WQtemp(3), &
                        Qtemp(3), WPtemp(3), FM(0:13)
    double precision :: AA, AB, ABtemp, cutoffprim1, cutoffprim, CD, ABCD, ROU, RPQ, ABCDsqrt, &
                        ABcom, CDtemp, ABCDtemp, CDcom, XXXtemp, T

    ! put the variables used in VRR & HRR in common blocks for the sake of consistency
    common /VRRcom/ Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp
    !common /hrrstore/ NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

    ! also put the following variables in a common block
    ! common /COM1/ RA, RB, NII1, NII2, NJJ1, NJJ2, NKK1, NKK2, NLL1, NLL2, NABCDTYPE, NABCD

    ! set the coordinates of shell centers
    do M=1,3
      RA(M)=xyz(M,quick_basis%katom(II))
      RB(M)=xyz(M,quick_basis%katom(JJ))
    enddo

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

    write(*,*) "II JJ NII1 NJJ1:",II, JJ, NII1, NJJ1

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

    do JJJ=1,quick_basis%kprim(JJ)
      Nprij=quick_basis%kstart(JJ)+JJJ-1

      ! the second cycle is for i prim
      ! II and NpriI are the tracking indices
      do III=1,quick_basis%kprim(II)
        Nprii=quick_basis%kstart(II)+III-1

        !For NpriI and NpriJ primitives, we calculate the following quantities
        AB=Apri(Nprii,Nprij)    ! AB = Apri = expo(NpriI)+expo(NpriJ). Eqn 8 of HGP.
        ABtemp=0.5d0/AB         ! ABtemp = 1/(2Apri) = 1/2(expo(NpriI)+expo(NpriJ))

        ! compute this for screening integrals further 
        cutoffprim1=dnmax*cutprim(Nprii,Nprij)

        do M=1,3
           !Eqn 9 of HGP
           ! P' is the weighting center of NpriI and NpriJ
           !                           --->           --->
           ! ->  ------>       expo(I)*xyz(I)+expo(J)*xyz(J)
           ! P = P'(I,J)  = ------------------------------
           !                       expo(I) + expo(J)
           P(M)=Ppri(M,Nprii,Nprij)

           !Multiplication of Eqns 9  by Eqn 8 of HGP.. 
           !                        -->            -->
           ! ----->         expo(I)*xyz(I)+expo(J)*xyz(J)                                 -->            -->
           ! AAtemp = ----------------------------------- * (expo(I) + expo(J)) = expo(I)*xyz(I)+expo(J)*xyz(J)
           !                  expo(I) + expo(J)
           AAtemp(M)=P(M)*AB

           !Requires for HGP Eqn 6. 
           ! ----->   ->  ->
           ! Ptemp  = P - A
           Ptemp(M)=P(M)-RA(M)
        enddo
        
        ! Since the Ket vector is composed of a single and a null gaussian, we do the following.
        LLL=1
        KKK=1
        
        ! We no longer have Nprik, Npril. Omit the primitive integral screening for now.
        !cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
        !if(cutoffprim.gt.quick_method%primLimit)then
          
          ! Nita quantity of HGP Eqn 10. This is same as zita (AB) above. 
          ! CD=expo(C)+expo(D) where the first term on R.H.S. will come from York group, the second is zero.
          CD=Zc

          !First term of HGP Eqn 12 without sqrt.
          ABCD=AB+CD            ! ABCD = expo(NpriI)+expo(NpriJ)+expo(NpriK)+expo(NpriL)

          !First term of HGP Eqn 13.
          !         AB * CD      (expo(I)+expo(J))*(expo(K)+expo(L))
          ! Rou = ----------- = ------------------------------------
          !         AB + CD         expo(I)+expo(J)+expo(K)+expo(L)
          ROU=AB*CD/ABCD

          RPQ=0.0d0

          !First term of HGP Eqn 12 with sqrt. 
          !              _______________________________
          ! ABCDsqrt = \/expo(I)+expo(J)+expo(K)+expo(L)
          ABCDsqrt=dsqrt(ABCD)

          CDtemp=0.5d0/CD       ! CDtemp =  1/2(expo(NpriK)+expo(NpriL))

          !These terms are required for HGP Eqn 6.
          !                expo(I)+expo(J)                        expo(K)+expo(L)
          ! ABcom = --------------------------------  CDcom = --------------------------------
          !          expo(I)+expo(J)+expo(K)+expo(L)           expo(I)+expo(J)+expo(K)+expo(L)          
          ABcom=AB/ABCD
          CDcom=CD/ABCD

          ! ABCDtemp = 1/2(expo(I)+expo(J)+expo(K)+expo(L))
          ABCDtemp=0.5d0/ABCD

          do M=1,3
            ! Calculate Q of HGP 10, which is same as P above. 
            ! Q' is the weighting center of NpriK and NpriL
            !                           --->           --->
            ! ->  ------>       expo(K)*xyz(K)+expo(L)*xyz(L)
            ! Q = P'(K,L)  = ------------------------------
            !                       expo(K) + expo(L)
            ! But since expo(L) is zero, this reduces to the following.
            Q(M)=RC(M)         

            ! HGP Eqn 10. 
            ! W' is the weight center for NpriI,NpriJ,NpriK and NpriL
            !                --->             --->             --->            --->
            ! ->     expo(I)*xyz(I) + expo(J)*xyz(J) + expo(K)*xyz(K) +expo(L)*xyz(L)
            ! W = -------------------------------------------------------------------
            !                    expo(I) + expo(J) + expo(K) + expo(L)
            W(M)=(AAtemp(M)+Q(M)*CD)/ABCD

            !Required for HGP Eqn 13.
            !        ->  ->  2
            ! RPQ =| P - Q |
            XXXtemp=P(M)-Q(M)
            RPQ=RPQ+XXXtemp*XXXtemp            

            ! ---->   ->  ->
            ! Qtemp = Q - K
            Qtemp(M)=Q(M)-RC(M)

            ! ----->   ->  ->
            ! WQtemp = W - Q
            ! ----->   ->  ->
            ! WPtemp = W - P
            WQtemp(M)=W(M)-Q(M)

            !Required for HGP Eqns 6 and 16.
            WPtemp(M)=W(M)-P(M)

          enddo

          !HGP Eqn 13. 
          !             ->  -> 2
          ! T = ROU * | P - Q|
          T=RPQ*ROU

          write(*,*) "HGP 13 T=",T,NABCD

          ! Compute boys function values, HGP eqn 14.
          !                         2m        2
          ! Fm(T) = integral(1,0) {t   exp(-Tt )dt}
          ! NABCD is the m value, and FM returns the FmT value
#ifdef MIRP
          call mirp_fmt(NABCD,T,FM)
#else
          call FmT(NABCD,T,FM)
#endif

          !Go through all m values, obtain Fm values from FM array we
          !just computed and calculate quantities required for HGP Eqn
          !12. 
          do iitemp=0,NABCD
             ! Yxiaotemp(1,1,iitemp) is the starting point of recurrsion
             Yxiaotemp(1,1,iitemp)=FM(iitemp)/ABCDsqrt
             write(*,*) "FM",FM(iitemp)
          enddo

          ITT=ITT+1

          write(*,*) "NABCDTYPE",NABCDTYPE

          ! now we will do vrr 
          call vertical(NABCDTYPE)

          do I2=NNC,NNCD
             do I1=NNA,NNAB
                Yxiao(ITT,I1,I2)=Yxiaotemp(I1,I2,0)
             enddo
          enddo          

        !endif
      enddo
    enddo

    do I=NII1,NII2
       NNA=Sumindex(I-1)+1
       do J=NJJ1,NJJ2
          NNAB=SumINDEX(I+J)
          do K=NKK1,NKK2
             NNC=Sumindex(k-1)+1
             do L=NLL1,NLL2
                NNCD=SumIndex(K+L)
                write(*,*) "Calling iclass_lngr_int: I J K L", I, J, K, L
                call iclass_lngr_int(I,J,K,L,NNA,NNC,NNAB,NNCD,II,JJ)
             enddo
          enddo
       enddo
    enddo

    return

  end subroutine compute_lngr_int

  subroutine iclass_lngr_int(I,J,K,L,NNA,NNC,NNAB,NNCD,II,JJ)

    use quick_basis_module
    use quick_constants_module
    use quick_method_module
    use quick_molspec_module
    use quick_calculated_module
    use quick_scratch_module

    implicit none

    integer, intent(in) :: I, J, K, L, NNA, NNC, NNAB, NNCD, II, JJ

    ! variables in common blocks
    !integer :: NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2, NII1, NII2, NJJ1, &
    !           NJJ2, NKK1, NKK2, NLL1, NLL2, NABCDTYPE, NABCD
    !double precision :: Ptemp(3), Qtemp(3), WPtemp(3), WQtemp(3), RA(3), RB(3), RC(3), &
    !                    ABtemp, CDtemp, ABCDtemp, ABcom, CDcom

    integer :: ITT, Nprii, Nprij, MM1, MM2, itemp, III1, III2, JJJ1, JJJ2, KKK1, KKK2, &
               LLL1, LLL2, NBI1, NBI2, NBJ1, NBJ2, NBK1, NBK2, NBL1, NBL2 
    double precision :: X44(129600)
    double precision :: X2, Ytemp

    ! put the variables used in VRR & HRR in common blocks for the sake of consistency
    !common /VRRcom/ Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp
    !common /hrrstore/ NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
    !common /COM1/ RA, RB, NII1, NII2, NJJ1, NJJ2, NKK1, NKK2, NLL1, NLL2, NABCDTYPE, NABCD

    store=0.0d0

    ITT=0
    do JJJ=1,quick_basis%kprim(JJ)
       Nprij=quick_basis%kstart(JJ)+JJJ-1
 
       do III=1,quick_basis%kprim(II)
          Nprii=quick_basis%kstart(II)+III-1
 
          ! X0 = 2.0d0*(PI)**(2.5d0), constants for HGP 15, X0 comes from constants module 
          ! multiplied twice for KAB and KCD
 
          X2=X0*quick_basis%Xcoeff(Nprii,Nprij,I,J)
          write(*,*) "X0,Xcoeff",X0,quick_basis%Xcoeff(Nprii,Nprij,I,J)
          !cutoffprim1=dnmax*cutprim(Nprii,Nprij)

          ! We no longer have Nprik, Npril. Omit the primitive integral screening for now.
          !cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
          !if(cutoffprim.gt.quick_method%primLimit)then
            
            ! Compute HGP eqn 15 for ket vector.
            !                    expo(C)*expo(D)*(xyz(C)-xyz(D))^2              1
            ! K'(C,D) =  exp[ - ------------------------------------]* -------------------
            !                            expo(C)+expo(D)                  expo(C)+expo(D)
            !
            ! Since we have a null gaussian (ie. expo(D) is zero), this equations becomes the following.
            !
            ! K'(A,B) =  1/expo(A) 

            ITT = ITT+1
            !This is the KAB x KCD value reqired for HGP 12.
            !itt is the m value. 
            X44(ITT) = X2*(1/Zc)*Cc*(Zc/PI)**1.5

write(*,*) "lngr itt, x0,xcoeff1,x2,xcoeff2,x44: ",itt,x0,quick_basis%Xcoeff(Nprii,Nprij,I,J),x2,&
(1/Zc)*Cc*(0.75/PI)**1.5,X44(ITT)
          !endif
       enddo
    enddo

    ! Compute HGP eqn 12.
    do MM2=NNC,NNCD
      do MM1=NNA,NNAB
        Ytemp=0.0d0
        do itemp=1,ITT
          Ytemp=Ytemp+X44(itemp)*Yxiao(itemp,MM1,MM2)
            write(*,*) "lngr X44, Yxio, Ytemp: ", X44(itemp),Yxiao(itemp,MM1,MM2),Ytemp
        enddo
        store(MM1,MM2)=Ytemp
write(*,*) "lngr store", MM1,MM2,store(MM1,MM2)
      enddo
    enddo

    !Get the start and end basis numbers for each angular momentum. 
    !For eg. Qsbasis and Qfbasis are 1 and 3 for P basis. 
    NBI1=quick_basis%Qsbasis(II,I)
    NBI2=quick_basis%Qfbasis(II,I)
    NBJ1=quick_basis%Qsbasis(JJ,J)
    NBJ2=quick_basis%Qfbasis(JJ,J)
    NBK1=0
    NBK2=0
    NBL1=0
    NBL2=0

    ! This is a whacky way of specifying integrals. We are interested in (IJ|00)
    ! type integrals. This means, IJKLtype will have 0, 100, 200, 300, 1000, 
    ! 1100, 1200, 1300, 2000, 2100, 2200, 2300, 3000, 3100, 3200, 3300 values
    ! for (00|00), (01|00), (02|00) and so on.
    IJtype=10*I+J
    KLtype=10*K+L
    IJKLtype=100*IJtype+KLtype

    write(*,*) "IJKLtype", IJKLtype

    if((max(I,J,K,L).eq.2.and.(J.ne.0.or.L.ne.0)).or.(max(I,J,K,L).ge.3))IJKLtype=999

    !quick_basis%ksumtype array has a cumulative sum of number of components of all
    !shells 
    III1=quick_basis%ksumtype(II)+NBI1
    III2=quick_basis%ksumtype(II)+NBI2
    JJJ1=quick_basis%ksumtype(JJ)+NBJ1
    JJJ2=quick_basis%ksumtype(JJ)+NBJ2
    KKK1=1+NBK1
    KKK2=1+NBK2
    LLL1=1+NBL1
    LLL2=1+NBL2    

    KKK=1
    LLL=1

    do III=III1,III2
      do JJJ=JJJ1,JJJ2
        call hrr_lngr
        quick_qm_struct%o(JJJ,III)=quick_qm_struct%o(JJJ,III)+Y
        write(*,*) JJJ,III,"lngr Y:", Y
      enddo
    enddo

  end subroutine iclass_lngr_int

#include "./include/hrr_lngr.f90"

end module quick_long_range_module


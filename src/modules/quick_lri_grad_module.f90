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
! gradients in QM/MM.                                                 !
!_____________________________________________________________________!

module quick_lri_grad_module

  implicit none
  private

  public :: computeLRIGrad
  public :: computeLRINumGrad

  ! module specific data structures
  double precision :: store(120,120)      ! store primitive integrals from vrr
  double precision :: storeaa(120,120)
  double precision :: storebb(120,120)
  double precision :: storecc(120,120)
  double precision :: RA(3), RB(3), RC(3) ! cartesian coordinates of the 3 centers
  double precision :: Zc, Cc              ! charge and a magic number of the external gaussian
  integer :: iC ! index of the third center required for gradient addition

  interface computeLRIGrad
    module procedure compute_lri_grad
  end interface computeLRIGrad

  interface computeLRINumGrad
    module procedure compute_lri_numgrad
  end interface computeLRINumGrad

contains




  
  subroutine compute_lri_numgrad( c_coords, c_zeta, c_chg, c_idx )
    use quick_lri_module, only : computeLRI
    
    use quick_basis_module
    use quick_method_module
    use quick_molspec_module
    use quick_params_module
    use quick_scratch_module
    use quick_calculated_module
    use quick_cshell_eri_module
    
    implicit none
    
    double precision, intent(in) :: c_coords(3), c_zeta, c_chg
    integer, intent(in) :: c_idx
    
    double precision,parameter :: delta = 2.e-5
    double precision :: elo, ehi
    integer :: a,k,o
    double precision :: ccrd(3)

    
    do a=1,natom
       o = 3*(a-1)
       do k=1,3
          xyz(k,a) = xyz(k,a) + delta
          call getEriPrecomputables
          quick_qm_struct%o = 0.d0
          call computeLRI( c_coords, c_zeta, c_chg )
          call copySym(quick_qm_struct%o,nbasis)
          ehi = sum(quick_qm_struct%dense*quick_qm_struct%o)
          
          xyz(k,a) = xyz(k,a) - 2.d0*delta
          call getEriPrecomputables
          quick_qm_struct%o = 0.d0
          call computeLRI( c_coords, c_zeta, c_chg )
          call copySym(quick_qm_struct%o,nbasis)
          elo = sum(quick_qm_struct%dense*quick_qm_struct%o)
          
          xyz(k,a) = xyz(k,a) + delta
          
          quick_qm_struct%gradient(o+k) = &
               & quick_qm_struct%gradient(o+k) + (ehi-elo)/(2.d0*delta)
       end do
    end do
    call getEriPrecomputables

    if ( c_idx <= natom ) then
       o = 3*(c_idx-1)
    else
       o = 3*(c_idx-natom-1)
    end if
    do k=1,3
       ccrd(k) = c_coords(k) + delta
       quick_qm_struct%o = 0.d0
       call computeLRI( ccrd, c_zeta, c_chg )
       call copySym(quick_qm_struct%o,nbasis)
       ehi = sum(quick_qm_struct%dense*quick_qm_struct%o)
       
       ccrd(k) = c_coords(k) - delta
       quick_qm_struct%o = 0.d0
       call computeLRI( ccrd, c_zeta, c_chg )
       call copySym(quick_qm_struct%o,nbasis)
       elo = sum(quick_qm_struct%dense*quick_qm_struct%o)

       if ( c_idx <= natom ) then
          quick_qm_struct%gradient(o+k) = &
               & quick_qm_struct%gradient(o+k) + (ehi-elo)/(2.d0*delta)
       else
          quick_qm_struct%ptchg_gradient(o+k) = &
               & quick_qm_struct%ptchg_gradient(o+k) + (ehi-elo)/(2.d0*delta)
       end if
    end do
    

  end subroutine compute_lri_numgrad




  
  subroutine compute_lri_grad(c_coords, c_zeta, c_chg, c_idx )

    !----------------------------------------------------------------------!
    ! The goal of this subroutine is to compute (ij|c) three center        !
    ! integral gradients. (i.e. d(ij|c)/dA_i)                              !
    ! Here i,j are two basis functions, c is a gaussian                    !     
    ! located at a certain distance. To make use of the existing ERI code, !
    ! we approximate above integral with (ij|c0) four center integral where!
    ! 0 is another gaussian with zero exponent.                            !
    !______________________________________________________________________!

    use quick_basis_module
    !use quick_lri_module, only : has_angrenorm
    !use quick_lri_module, only : angrenorm
    !use quick_lri_module, only : CalcAngRenorm
    
    implicit none
    double precision, intent(in) :: c_coords(3), c_zeta, c_chg
    integer, intent(in) :: c_idx
    integer :: II, JJ         ! shell pairs

    RC=c_coords
    Zc=c_zeta
    Cc=c_chg
    iC=c_idx


    !if ( .not. has_angrenorm ) then
    !   has_angrenorm = .true.
    !   if ( associated( angrenorm ) ) deallocate( angrenorm )
    !   allocate( angrenorm( nbasis ) )
    !   call CalcAngRenorm( nbasis,angrenorm )
    !end if

    
    do II = 1, jshell
      do JJ = II, jshell  
        !II=1
         !JJ=1
        call compute_tci_grad(II,JJ)
      enddo
    enddo

  end subroutine compute_lri_grad


  subroutine compute_tci_grad(II,JJ)

    !----------------------------------------------------------------------!
    ! This subroutine computes quantities required for OSHGP algorithm,    !
    ! values of Boys function and calls appropriate subroutines to that    !
    ! performs VRR and HRR.                                                !
    !______________________________________________________________________!

    use quick_basis_module
    use quick_method_module
    use quick_molspec_module
    use quick_params_module
    use quick_scratch_module

    integer, intent(in) :: II, JJ

    integer :: M, LL, NII1, NII2, NJJ1, NJJ2, NKK1, NKK2, NLL1, NLL2, NNAB, NNABfirst, NNCD, NNCDfirst, &
               NABCDTYPE, NNA, NNC, NABCD, ITT, Nprij, Nprii, iitemp, I1, I2, I, J, K, L 
    double precision :: P(3), AAtemp(3), Ptemp(3), Q(3), W(3), WQtemp(3), &
                        Qtemp(3), WPtemp(3), FM(0:14)
    double precision :: AA, AB, ABtemp, cutoffprim1, cutoffprim, CD, ABCD, ROU, RPQ, ABCDsqrt, &
                        ABcom, CDtemp, ABCDtemp, CDcom, XXXtemp, T

    ! put the variables used in VRR in common a block (legacy code needs this)
    common /VRRcom/ Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp

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

    !write(*,*) "II JJ NII1 NJJ1:",II, JJ, NII1, NJJ1

    NNAB=(NII2+NJJ2)
    NNCD=(NKK2+NLL2)

    NABCDTYPE=NNAB*10+NNCD 

    NNAB=sumindex(NNAB)
    NNCD=sumindex(NNCD)

    NNABfirst=sumindex(NII2+NJJ2+1)
    NNCDfirst=sumindex(NKK2+NLL2+1)

    NNA=sumindex(NII1-2)+1
    NNC=sumindex(NKK1-2)+1    

    ! The summation of the highest angular momentum number of each shell
    ! For first derivative of nuclui motion, the total angular momentum is raised by 1
    NABCD=NII2+NJJ2+NKK2+NLL2+1+1
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

          !write(*,*) "HGP 13 T=",T,NABCD

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
          !   write(*,*) "FM",FM(iitemp)
          enddo

          ITT=ITT+1

          !write(*,*) "NABCDTYPE",NABCDTYPE

          ! now we will do vrr 
          call vertical(NABCDTYPE+11)

          do I2=NNC,NNCDfirst
             do I1=NNA,NNABfirst
                Yxiao(ITT,I1,I2)=Yxiaotemp(I1,I2,0)
             enddo
          enddo          

        !endif
      enddo
    enddo

   ! allocate scratch memory for X arrays
   call allocshellopt(quick_scratch,maxcontract)

    do I=NII1,NII2
       NNA=Sumindex(I-2)+1
       do J=NJJ1,NJJ2
          NNAB=SumINDEX(I+J)
          NNABfirst=SumINDEX(I+J+1)
          do K=NKK1,NKK2
             NNC=Sumindex(k-2)+1
             do L=NLL1,NLL2
                NNCD=SumIndex(K+L)
                NNCDfirst=SumIndex(K+L+1)
                call iclass_tci_grad(I,J,K,L,II,JJ,NNA,NNC,NNAB,NNCD,NNABfirst,NNCDfirst)
             enddo
          enddo
       enddo
    enddo

   ! deallocate scratch memory for X arrays
   call deallocshellopt(quick_scratch)

    return

  end subroutine compute_tci_grad

  subroutine iclass_tci_grad(I,J,K,L,II,JJ,NNA,NNC,NNAB,NNCD,NNABfirst,NNCDfirst)

    !----------------------------------------------------------------------!
    ! This subroutine computes contracted 3 center integrals by calling    !
    ! the appropriate subroutine and adds the integral contributions into  !
    ! gradient vector.                                                     !
    !______________________________________________________________________!

    use quick_basis_module
    use quick_constants_module
    use quick_method_module
    use quick_molspec_module
    use quick_calculated_module
    use quick_scratch_module
!    use quick_lri_module, only : angrenorm

    implicit none

    integer, intent(in) :: I, J, K, L, NNA, NNC, NNAB, NNCD, II, JJ, NNABfirst, NNCDfirst

    integer :: ITT, Nprii, Nprij, MM1, MM2, itemp, III1, III2, JJJ1, JJJ2, KKK1, KKK2, &
               LLL1, LLL2, NBI1, NBI2, NBJ1, NBJ2, NBK1, NBK2, NBL1, NBL2, iA, iB, &
               iAstart, iBstart, iCstart

    double precision :: AA, BB, X2, Ytemp, YtempAA, YtempBB, YtempCC, Agrad1, Agrad2, Agrad3, &
                        Bgrad1, Bgrad2, Bgrad3, Cgrad1, Cgrad2, Cgrad3

    double precision :: afact
    double precision :: grda(3),grdb(3)
    
    store=0.0d0
    storeaa=0.0d0
    storebb=0.0d0
    storecc=0.0d0

    ITT=0
    do JJJ=1,quick_basis%kprim(JJ)
       Nprij=quick_basis%kstart(JJ)+JJJ-1
       BB=quick_basis%gcexpo(JJJ,quick_basis%ksumtype(JJ)) 

       do III=1,quick_basis%kprim(II)
          Nprii=quick_basis%kstart(II)+III-1
          AA=quick_basis%gcexpo(III,quick_basis%ksumtype(II))
 
          ! X0 = 2.0d0*(PI)**(2.5d0), constants for HGP 15, X0 comes from constants module 
          ! multiplied twice for KAB and KCD
 
          X2=X0*quick_basis%Xcoeff(Nprii,Nprij,I,J)
          !cutoffprim1=dnmax*cutprim(Nprii,Nprij)

          ! We no longer have Nprik, Npril. Omit the primitive integral screening for now.
          !cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
          !if(cutoffprim.gt.quick_method%primLimit)then
            
            ! Compute HGP eqn 15 for ket vector.
            !                    expo(A)*expo(B)*(xyz(A)-xyz(B))^2              1
            ! K'(A,B) =  exp[ - ------------------------------------]* -------------------
            !                            expo(A)+expo(B)                  expo(A)+expo(B)
            !
            ! Since we have a null gaussian (ie. expo(B) is zero), this equations becomes the following.
            !
            ! K'(A,B) =  1/expo(A) 

            ITT = ITT+1
            !This is the KAB x KCD value reqired for HGP 12.
            !itt is the m value. Note that the gaussian contraction coefficient (gccoeff) is treated as 1.0. 
            quick_scratch%X44(ITT) = X2*(1/Zc)*Cc*(Zc/PI)**1.5d0

            !write(*,*) "lngr grad itt, x0,xcoeff1,x2,xcoeff2,x44:",itt,x0,quick_basis%Xcoeff(Nprii,Nprij,I,J),x2,&
            !(1/Zc)*Cc*(Zc/PI)**1.5,quick_scratch%X44(ITT)

            !compute the first term of eqn 20 for 3 centers. 
            quick_scratch%X44AA(ITT)=quick_scratch%X44(ITT)*AA*2.0d0
            quick_scratch%X44BB(ITT)=quick_scratch%X44(ITT)*BB*2.0d0
            quick_scratch%X44CC(ITT)=quick_scratch%X44(ITT)*Zc*2.0d0

          !endif
       enddo
    enddo

    ! Compute HGP eqn 12.
    do MM2=NNC,NNCD
      do MM1=NNA,NNAB
        Ytemp=0.0d0
        do itemp=1,ITT
          Ytemp=Ytemp+quick_scratch%X44(itemp)*Yxiao(itemp,MM1,MM2)

            !write(*,*) "lngr grad X44, Yxio, Ytemp: ",&
            !quick_scratch%X44(itemp),Yxiao(itemp,MM1,MM2),Ytemp

        enddo
        store(MM1,MM2)=Ytemp
        !write(*,*) "lngr grad: MM1, MM2, storeAA", MM1, MM2, storeAA(MM1,MM2)
      enddo
    enddo

    do MM2=NNC,NNCDfirst
       do MM1=NNA,NNABfirst
          YtempAA=0.0d0
          YtempBB=0.0d0
          YtempCC=0.0d0
          do itemp=1,ITT
             YtempAA=YtempAA+quick_scratch%X44AA(itemp)*Yxiao(itemp,MM1,MM2)
             YtempBB=YtempBB+quick_scratch%X44BB(itemp)*Yxiao(itemp,MM1,MM2)
             YtempCC=YtempCC+quick_scratch%X44CC(itemp)*Yxiao(itemp,MM1,MM2)

             !write(*,*) "lngr grad X44AA, Yxio, YtempAA: ",&
             !quick_scratch%X44AA(itemp),Yxiao(itemp,MM1,MM2),YtempAA
          enddo
          storeAA(MM1,MM2)=YtempAA
          storeBB(MM1,MM2)=YtempBB
          storeCC(MM1,MM2)=YtempCC

          !write(*,*) "lngr grad: MM1, MM2, storeAA", MM1, MM2, storeAA(MM1,MM2)
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

    Agrad1=0.d0
    Bgrad1=0.d0
    Cgrad1=0.d0
    Agrad2=0.d0
    Bgrad2=0.d0
    Cgrad2=0.d0
    Agrad3=0.d0
    Bgrad3=0.d0
    Cgrad3=0.d0

    ! This is a whacky way of specifying integrals. We are interested in (IJ|00)
    ! type integrals. This means, IJKLtype will have 0, 100, 200, 300, 1000, 
    ! 1100, 1200, 1300, 2000, 2100, 2200, 2300, 3000, 3100, 3200, 3300 values
    ! for (00|00), (01|00), (02|00) and so on.
    IJtype=10*I+J
    KLtype=10*K+L
    IJKLtype=100*IJtype+KLtype

    !write(*,*) "IJKLtype", IJKLtype

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

    iA = quick_basis%ncenter(III2)
    iB = quick_basis%ncenter(JJJ2)

    iAstart = (iA-1)*3
    iBstart = (iB-1)*3
    if ( iC <= natom ) then
       iCstart = (iC-1)*3
    else
       iCstart = (iC-natom-1)*3
    end if

    KKK=1
    LLL=1

    ! if ( abs(RC(1)-28.900) < 0.01 ) then
       

    !    write(6,*)
    !    write(6,*)

    !    write(6,'(A,3ES18.9)')"RC ",RC(1),  RC(2),  RC(3)
    !    write(6,'(A,I3)'),"LA",I
    !    write(6,'(A,3ES18.9)')"RA ",xyz(1:3,iA)
    !    write(6,'(A)',advance='no')"ZA "
    !    do III=1,quick_basis%kprim(II)
    !       Nprii=quick_basis%kstart(II)+III-1
    !       AA=quick_basis%gcexpo(III,quick_basis%ksumtype(II))
    !       write(6,'(ES18.9)',advance='no')AA
    !    end do
    !    write(6,*)
    !    write(6,'(A,I3)'),"LB",J
    !    write(6,'(A,3ES18.9)')"RB ",xyz(1:3,iB)
    !    write(6,'(A)',advance='no')"ZB "
    !    do JJJ=1,quick_basis%kprim(JJ)
    !       Nprij=quick_basis%kstart(JJ)+JJJ-1
    !       AA=quick_basis%gcexpo(JJJ,quick_basis%ksumtype(JJ))
    !       write(6,'(ES18.9)',advance='no')AA
    !    end do
    !    write(6,*)
    ! end if

    

    grda = 0.d0
    grdb = 0.d0
    
    do III=III1,III2
      do JJJ=JJJ1,JJJ2
        call hrr_tci_grad
        !write(*,*) "lngr Y:",IJKLtype,RA(1),RB(1),RC(1),Yaa(1),Yaa(2),Yaa(3),&
        !Ybb(1),Ybb(2),Ybb(3),Ycc(1),Ycc(2),Ycc(3)

        afact = 1.d0 ! angrenorm(JJJ) * angrenorm(III)
        ! The off-diagonal blocks are only computed once, so we need to
        ! scale by 2 to consider the full density matrix for these blocks
        if ( II /= JJ ) afact = afact * 2.d0

        !Ycc(1) = -Yaa(1)-Ybb(1)
        !Ycc(2) = -Yaa(2)-Ybb(2)
        !Ycc(3) = -Yaa(3)-Ybb(3)

        
        ! if ( abs(RC(1)-28.900) < 0.01 ) then
        !    write(6,'(2I4,8es13.4)')III,JJJ,  &
        !         & quick_qm_struct%dense(JJJ,III),quick_qm_struct%dense(III,JJJ), &
        !         & Yaa(1),Yaa(2),Yaa(3),Ybb(1),Ybb(2),Ybb(3)
        ! end if


        grda(1:3) = grda(1:3) + quick_qm_struct%dense(JJJ,III)*Yaa(1:3)*afact
        grdb(1:3) = grdb(1:3) + quick_qm_struct%dense(JJJ,III)*Ybb(1:3)*afact
        
        

        ! quick_qm_struct%gradient(iASTART+1) = quick_qm_struct%gradient(iASTART+1) &
        !      & +quick_qm_struct%dense(JJJ,III)*Yaa(1)*afact
        ! quick_qm_struct%gradient(iASTART+2) = quick_qm_struct%gradient(iASTART+2) &
        !      & +quick_qm_struct%dense(JJJ,III)*Yaa(2)*afact
        ! quick_qm_struct%gradient(iASTART+3) = quick_qm_struct%gradient(iASTART+3) &
        !      & +quick_qm_struct%dense(JJJ,III)*Yaa(3)*afact

        ! quick_qm_struct%gradient(iBSTART+1) = quick_qm_struct%gradient(iBSTART+1) &
        !      & +quick_qm_struct%dense(JJJ,III)*Ybb(1)*afact
        ! quick_qm_struct%gradient(iBSTART+2) = quick_qm_struct%gradient(iBSTART+2) &
        !      & +quick_qm_struct%dense(JJJ,III)*Ybb(2)*afact
        ! quick_qm_struct%gradient(iBSTART+3) = quick_qm_struct%gradient(iBSTART+3) &
        !      & +quick_qm_struct%dense(JJJ,III)*Ybb(3)*afact
 
        !  if(iC <= natom) then     
        !     quick_qm_struct%gradient(iCSTART+1) = quick_qm_struct%gradient(iCSTART+1) &
        !          & +quick_qm_struct%dense(JJJ,III)*Ycc(1)*afact
        !     quick_qm_struct%gradient(iCSTART+2) = quick_qm_struct%gradient(iCSTART+2) &
        !          & +quick_qm_struct%dense(JJJ,III)*Ycc(2)*afact
        !     quick_qm_struct%gradient(iCSTART+3) = quick_qm_struct%gradient(iCSTART+3) &
        !          & +quick_qm_struct%dense(JJJ,III)*Ycc(3)*afact
        !  else
        !     quick_qm_struct%ptchg_gradient(iCSTART+1) = quick_qm_struct%ptchg_gradient(iCSTART+1) &
        !          & +quick_qm_struct%dense(JJJ,III)*Ycc(1)*afact
        !     quick_qm_struct%ptchg_gradient(iCSTART+2) = quick_qm_struct%ptchg_gradient(iCSTART+2) &
        !          & +quick_qm_struct%dense(JJJ,III)*Ycc(2)*afact
        !     quick_qm_struct%ptchg_gradient(iCSTART+3) = quick_qm_struct%ptchg_gradient(iCSTART+3) &
        !          & +quick_qm_struct%dense(JJJ,III)*Ycc(3)*afact
        !  endif

        
      enddo
    enddo

    !write(6,'(A,i5,3es13.4)')"grda",iASTART,grda
    !write(6,'(A,i5,3es13.4)')"grdb",iBSTART,grdb
    !write(6,*)"iCSTART",iC,iCSTART
    do III=1,3
       quick_qm_struct%gradient(iASTART+III) = quick_qm_struct%gradient(iASTART+III)+grda(III)
       quick_qm_struct%gradient(iBSTART+III) = quick_qm_struct%gradient(iBSTART+III)+grdb(III)
    end do
    if(iC <= natom) then
       do III=1,3
          quick_qm_struct%gradient(iCSTART+III) = quick_qm_struct%gradient(iCSTART+III) &
               & - (grda(III)+grdb(III))
       end do
    else
       do III=1,3
          quick_qm_struct%ptchg_gradient(iCSTART+III) = quick_qm_struct%ptchg_gradient(iCSTART+III) &
               &  - (grda(III)+grdb(III))
       end do
    end if

    
  end subroutine iclass_tci_grad

#include "./include/hrr_lri_grad.fh"

end module quick_lri_grad_module


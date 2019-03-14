#include "config.h"
! Ed Brothers. May 23, 2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine hfgrad
  use allmod
  implicit double precision(a-h,o-z)

  integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
  common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

  ! The purpose of this subroutine is to calculate the gradient of
  ! the total energy with respect to nuclear displacement.  The results
  ! of this are stored in Gradient, which is organized by atom and then
  ! by direction of displacement, i.e. element 1 is the gradient of the
  ! x diplacement of atom 1, element 5 is the y displacement of atom 2.

  ! Not that this is the RHF version of the code.  It is simplest of
  ! the gradient codes in this program.

  !Xiao HE modified 09/12/2007
  call cpu_time(timer_begin%TGrad)
  do Iatm=1,natom*3
     quick_qm_struct%gradient(iatm)=0.d0
  enddo

  ! The gradient at this level of theory is the sum of five terms.
  ! 1)  The derivative of the nuclear repulsion.  Quick derivation:
  ! Vnn = (Sum over A) (Sum over B>A) ZA ZB / RAB
  ! where A and B are atoms, Z are charges, and RAB is the interatomic
  ! seperation.  If we take the derivative, all terms not involving
  ! A fall out. Thus:
  ! Vnn/dXA = ZA (Sum over B) d/dXA (ZB /RAB)
  ! Vnn/dXA = ZA (Sum over B) ZB d/dXA (RAB^-1)
  ! Vnn/dXA = ZA (Sum over B) ZB d/dXA(((XA-XB)^2+(YA-YB)^2+(ZA-ZB)^2)^-.5)
  ! Vnn/dXA = ZA (Sum over B) ZB*-.5*((XA-XB)^2+(YA-YB)^2+(ZA-ZB)^2)^-1.5
  ! *2*(XA-XB)^1
  ! Vnn/dXA = ZA (Sum over B) ZB*-((XA-XB)^2+(YA-YB)^2+(ZA-ZB)^2)^-1.5
  ! *(XA-XB)^1
  ! Vnn/dXA = ZA (Sum over B) ZB*((XA-XB)^2+(YA-YB)^2+(ZA-ZB)^2)^-1.5
  ! *(XB-XA)
  ! Vnn/dXA = ZA (Sum over B) ZB*(XB-XA) RAB^-3

  ! Thus this term is trivial, and can calculate it here. Note also
  ! that that atom A can never equal atom B, and A-B part of the derivative
  ! for A is the negative of the BA derivative for atom B.


  do Iatm = 1,natom*3
     do Jatm = Iatm+1,natom
        RIJ  = (xyz(1,Iatm)-xyz(1,Jatm))*(xyz(1,Iatm)-xyz(1,Jatm)) &
             +(xyz(2,Iatm)-xyz(2,Jatm))*(xyz(2,Iatm)-xyz(2,Jatm)) &
             +(xyz(3,Iatm)-xyz(3,Jatm))*(xyz(3,Iatm)-xyz(3,Jatm))
        ZAZBdivRIJ3 = quick_molspec%chg(Iatm)*quick_molspec%chg(Jatm)*(RIJ**(-1.5d0))
        XBminXA = xyz(1,Jatm)-xyz(1,Iatm)
        YBminYA = xyz(2,Jatm)-xyz(2,Iatm)
        ZBminZA = xyz(3,Jatm)-xyz(3,Iatm)
        ISTART = (Iatm-1)*3
        JSTART = (Jatm-1)*3
        quick_qm_struct%gradient(ISTART+1) = quick_qm_struct%gradient(ISTART+1)+XBminXA*ZAZBdivRIJ3
        quick_qm_struct%gradient(ISTART+2) = quick_qm_struct%gradient(ISTART+2)+YBminYA*ZAZBdivRIJ3
        quick_qm_struct%gradient(ISTART+3) = quick_qm_struct%gradient(ISTART+3)+ZBminZA*ZAZBdivRIJ3
        quick_qm_struct%gradient(JSTART+1) = quick_qm_struct%gradient(JSTART+1)-XBminXA*ZAZBdivRIJ3
        quick_qm_struct%gradient(JSTART+2) = quick_qm_struct%gradient(JSTART+2)-YBminYA*ZAZBdivRIJ3
        quick_qm_struct%gradient(JSTART+3) = quick_qm_struct%gradient(JSTART+3)-ZBminZA*ZAZBdivRIJ3
     enddo
  enddo

  ! 2)  The negative of the energy weighted density matrix element i j
  ! with the derivative of the ij overlap.

  ! 3)  The derivative of the 1 electron kinetic energy term ij times
  ! the density matrix element ij.

  ! These terms are grouped together since we loop over the same terms.
  ! Also note that these are the 2-center terms.

  ! The energy weighted denisty matrix is:
  ! Q(i,j) =2*(Sum over alpha electrons a)  E(a) C(I,a) C(J,a)
  ! Where C is the alpha or beta molecular orbital coefficients, and
  ! E is the alpha or beta molecular orbital energies.
  ! We'll store this in HOLD as we don't really need it (except for hessian
  ! calculations later).

  do I=1,nbasis
     do J=1,nbasis
        HOLDJI = 0.d0
        do K=1,quick_molspec%nelec/2
           HOLDJI = HOLDJI + (quick_qm_struct%E(K)*quick_qm_struct%co(J,K)*quick_qm_struct%co(I,K))
!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!!!
                write (ioutfile,'(A4,I5,I5,I5,7x,F20.10,7x,F20.10,7x,F20.10)')"Vars",I,J,K, &
                quick_qm_struct%E(K),quick_qm_struct%co(J,K),quick_qm_struct%co(I,K)
!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!!!

        enddo
        quick_scratch%hold(J,I) = 2.d0*HOLDJI
     enddo
  enddo

  if (quick_method%debug) then
     write(ioutfile,'(/"THE ENERGY WEIGHTED DENSITY MATRIX")')
     do I=1,nbasis
        do J=1,nbasis
           write (ioutfile,'("W[",I4,",",I4,"]=",F18.10)') &
                J,I,quick_scratch%hold(J,I)
        enddo
     enddo
  endif

  ! The contribution to the derivative of energy with respect to nuclear
  ! postion for this term is: -(Sum over i,j) Q(i,j) dS(ij)/dXA
  ! Now Q is symmetric, and dS(ij)/dXA = dS(ji)/dXA.  Furthermore, if
  ! i and j are on the same center, the term is zero. Thus we need to find
  ! the i j pairs for i and j not on the same atom.
  ! Also:  The derivative of a cartesian gtf is:

  ! d/dXA ((x-XA)^i (y-YA)^j (z-ZA)^k e^(-ar^2))
  ! = 2a((x-XA)^(i+1) (y-YA)^j (z-ZA)^k e^(-ar^2))
  ! - i ((x-XA)^(i-1) (y-YA)^j (z-ZA)^k e^(-ar^2))

  ! Note that the negative on the final term comes from the form of (x-XA).

  do Ibas=1,nbasis
     ISTART = (quick_basis%ncenter(Ibas)-1) *3
     do Jbas=quick_basis%last_basis_function(quick_basis%ncenter(IBAS))+1,nbasis
        JSTART = (quick_basis%ncenter(Jbas)-1) *3
        DENSEJI = quick_qm_struct%dense(Jbas,Ibas)

        ! We have selected our two basis functions, now loop over angular momentum.

        do Imomentum=1,3
           dSI = 0.d0
           dSJ =0.d0
           dKEI = 0.d0
           dKEJ = 0.d0

           ! do the Ibas derivatives first.

           itype(Imomentum,Ibas) = itype(Imomentum,Ibas)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 dSI = dSI + 2.d0*aexp(Icon,Ibas)* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 dKEI = dKEI + 2.d0*aexp(Icon,Ibas)* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
              enddo
           enddo
           itype(Imomentum,Ibas) = itype(Imomentum,Ibas)-1
           if (itype(Imomentum,Ibas) /= 0) then
              itype(Imomentum,Ibas) = itype(Imomentum,Ibas)-1
              do Icon=1,ncontract(Ibas)
                 do Jcon=1,ncontract(Jbas)
                    dSI = dSI - dble(itype(Imomentum,Ibas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                    dKEI = dKEI - dble(itype(Imomentum,Ibas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 enddo
              enddo
              itype(Imomentum,Ibas) = itype(Imomentum,Ibas)+1
           endif
           quick_qm_struct%gradient(ISTART+Imomentum) = quick_qm_struct%gradient(ISTART+Imomentum) &
                -dSI*quick_scratch%hold(Jbas,Ibas)*2.d0 &
                +dKeI*DENSEJI*2.d0

           ! Now do the Jbas derivatives.

           itype(Imomentum,Jbas) = itype(Imomentum,Jbas)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 dSJ = dSJ + 2.d0*aexp(Jcon,Jbas)* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 dKEJ = dKEJ + 2.d0*aexp(Jcon,Jbas)* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
              enddo
           enddo
           itype(Imomentum,Jbas) = itype(Imomentum,Jbas)-1
           if (itype(Imomentum,Jbas) /= 0) then
              itype(Imomentum,Jbas) = itype(Imomentum,Jbas)-1
              do Icon=1,ncontract(Ibas)
                 do Jcon=1,ncontract(Jbas)
                    dSJ = dSJ - dble(itype(Imomentum,Jbas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                    dKEJ = dKEJ - dble(itype(Imomentum,Jbas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 enddo
              enddo
              itype(Imomentum,Jbas) = itype(Imomentum,Jbas)+1
           endif
           quick_qm_struct%gradient(JSTART+Imomentum) = quick_qm_struct%gradient(JSTART+Imomentum) &
                -dSJ*quick_scratch%hold(Jbas,Ibas)*2.d0 &
                +dKEJ*DENSEJI*2.d0
        enddo
     enddo
  enddo


  ! 4)  The derivative of the 1 electron nuclear attraction term ij times
  ! the density matrix element ij.

  ! Please note that these are the three center terms.

  do IIsh=1,jshell
     do JJsh=IIsh,jshell
        call attrashellopt(IIsh,JJsh)
     enddo
  enddo

  !        write (ioutfile,'(/," ANALYTICAL GRADIENT first: ")')
  !        do Iatm=1,natom
  !            do Imomentum=1,3
  !                write (ioutfile,'(I5,7x,F20.10)')Iatm, &
  !                GRADIENT((Iatm-1)*3+Imomentum)
  !            enddo
  !        enddo



  ! 5)  The derivative of the 4center 2e- terms with respect to X times
  ! the coefficient found in the energy. (i.e. the multiplicative
  ! constants from the density matrix that arise as these are both
  ! the exchange and correlation integrals.

  !    call g2eshell

  ! Delta density matrix cutoff

  do II=1,jshell
     do JJ=II,jshell
        DNtemp=0.0d0
        call DNscreen(II,JJ,DNtemp)
        Cutmatrix(II,JJ)=DNtemp
        Cutmatrix(JJ,II)=DNtemp
     enddo
  enddo

#ifdef CUDA
    if (quick_method%bCUDA) then
        call gpu_upload_calculated(quick_qm_struct%o,quick_qm_struct%co, &
                                   quick_qm_struct%vec,quick_qm_struct%dense)
        call gpu_upload_cutoff(cutmatrix, quick_method%integralCutoff,quick_method%primLimit)
        call gpu_upload_grad(quick_qm_struct%gradient, quick_method%gradCutoff)
        call gpu_grad(quick_qm_struct%gradient)
    else
#endif


  ! ntempxiao1=0
  ! ntempxiao2=0

  do II=1,jshell
     do JJ=II,jshell
        Testtmp=Ycutoff(II,JJ)
        do KK=II,jshell
           do LL=KK,jshell
              if(quick_basis%katom(II).eq.quick_basis%katom(JJ).and.quick_basis%katom(II).eq. &
                quick_basis%katom(KK).and.quick_basis%katom(II).eq.quick_basis%katom(LL))then
                 continue
                 !          ntempxiao1=ntempxiao1+1
              else
                 testCutoff = TESTtmp*Ycutoff(KK,LL)
                 if(testCutoff.gt.quick_method%gradCutoff)then
                    DNmax=max(4.0d0*cutmatrix(II,JJ),4.0d0*cutmatrix(KK,LL), &
                         cutmatrix(II,LL),cutmatrix(II,KK),cutmatrix(JJ,KK),cutmatrix(JJ,LL))
                    cutoffTest=testCutoff*DNmax
                    if(cutoffTest.gt.quick_method%gradCutoff)then
                       call shellopt
                    endif
                 endif
              endif
           enddo
        enddo
     enddo
  enddo

#ifdef CUDA
endif
#endif

  call cpu_time(timer_end%TGrad)
  write(ioutfile, '(2x,"GRADIENT CALCULATION TIME",F15.9, " S")') timer_end%TGrad-timer_begin%TGrad
  timer_cumer%TGrad=timer_end%TGrad-timer_begin%TGrad+timer_cumer%TGrad

  return
end subroutine hfgrad

#ifdef MPI

! Yipu Miao  11/21/2010 Add MPI option
! Ed Brothers. May 23, 2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine mpi_hfgrad
  use allmod
  implicit double precision(a-h,o-z)

  integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
  common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
  double precision:: temp_grad(3*natom)
  include "mpif.h"

  ! The purpose of this subroutine is to calculate the gradient of
  ! the total energy with respect to nuclear displacement.  The results
  ! of this are stored in Gradient, which is organized by atom and then
  ! by direction of displacement, i.e. element 1 is the gradient of the
  ! x diplacement of atom 1, element 5 is the y displacement of atom 2.

  ! Not that this is the RHF version of the code.  It is simplest of
  ! the gradient codes in this program.

  !Xiao HE modified 09/12/2007
  call cpu_time(timer_begin%TGrad)

  do Iatm=1,natom*3
     quick_qm_struct%gradient(iatm)=0.d0
  enddo

  ! The gradient at this level of theory is the sum of five terms.
  ! 1)  The derivative of the nuclear repulsion.  Quick derivation:
  ! Vnn = (Sum over A) (Sum over B>A) ZA ZB / RAB
  ! where A and B are atoms, Z are charges, and RAB is the interatomic
  ! seperation.  If we take the derivative, all terms not involving
  ! A fall out. Thus:
  ! Vnn/dXA = ZA (Sum over B) d/dXA (ZB /RAB)
  ! Vnn/dXA = ZA (Sum over B) ZB d/dXA (RAB^-1)
  ! Vnn/dXA = ZA (Sum over B) ZB d/dXA(((XA-XB)^2+(YA-YB)^2+(ZA-ZB)^2)^-.5)
  ! Vnn/dXA = ZA (Sum over B) ZB*-.5*((XA-XB)^2+(YA-YB)^2+(ZA-ZB)^2)^-1.5
  ! *2*(XA-XB)^1
  ! Vnn/dXA = ZA (Sum over B) ZB*-((XA-XB)^2+(YA-YB)^2+(ZA-ZB)^2)^-1.5
  ! *(XA-XB)^1
  ! Vnn/dXA = ZA (Sum over B) ZB*((XA-XB)^2+(YA-YB)^2+(ZA-ZB)^2)^-1.5
  ! *(XB-XA)
  ! Vnn/dXA = ZA (Sum over B) ZB*(XB-XA) RAB^-3

  ! Thus this term is trivial, and can calculate it here. Note also
  ! that that atom A can never equal atom B, and A-B part of the derivative
  ! for A is the negative of the BA derivative for atom B.

  do Iatm = 1,natom*3
     do Jatm = Iatm+1,natom
        RIJ  = (xyz(1,Iatm)-xyz(1,Jatm))*(xyz(1,Iatm)-xyz(1,Jatm)) &
             +(xyz(2,Iatm)-xyz(2,Jatm))*(xyz(2,Iatm)-xyz(2,Jatm)) &
             +(xyz(3,Iatm)-xyz(3,Jatm))*(xyz(3,Iatm)-xyz(3,Jatm))
        ZAZBdivRIJ3 = quick_molspec%chg(Iatm)*quick_molspec%chg(Jatm)*(RIJ**(-1.5d0))
        XBminXA = xyz(1,Jatm)-xyz(1,Iatm)
        YBminYA = xyz(2,Jatm)-xyz(2,Iatm)
        ZBminZA = xyz(3,Jatm)-xyz(3,Iatm)
        ISTART = (Iatm-1)*3
        JSTART = (Jatm-1)*3
        if (master) then
           quick_qm_struct%gradient(ISTART+1) = quick_qm_struct%gradient(ISTART+1)+XBminXA*ZAZBdivRIJ3
           quick_qm_struct%gradient(ISTART+2) = quick_qm_struct%gradient(ISTART+2)+YBminYA*ZAZBdivRIJ3
           quick_qm_struct%gradient(ISTART+3) = quick_qm_struct%gradient(ISTART+3)+ZBminZA*ZAZBdivRIJ3
           quick_qm_struct%gradient(JSTART+1) = quick_qm_struct%gradient(JSTART+1)-XBminXA*ZAZBdivRIJ3
           quick_qm_struct%gradient(JSTART+2) = quick_qm_struct%gradient(JSTART+2)-YBminYA*ZAZBdivRIJ3
           quick_qm_struct%gradient(JSTART+3) = quick_qm_struct%gradient(JSTART+3)-ZBminZA*ZAZBdivRIJ3
        endif
     enddo
  enddo



  ! 2)  The negative of the energy weighted density matrix element i j
  ! with the derivative of the ij overlap.

  ! 3)  The derivative of the 1 electron kinetic energy term ij times
  ! the density matrix element ij.

  ! These terms are grouped together since we loop over the same terms.
  ! Also note that these are the 2-center terms.

  ! The energy weighted denisty matrix is:
  ! Q(i,j) =2*(Sum over alpha electrons a)  E(a) C(I,a) C(J,a)
  ! Where C is the alpha or beta molecular orbital coefficients, and
  ! E is the alpha or beta molecular orbital energies.
  ! We'll store this in HOLD as we don't really need it (except for hessian
  ! calculations later).
  
  if (master) then
  do I=1,nbasis
     do J=1,nbasis
        HOLDJI = 0.d0
        do K=1,quick_molspec%nelec/2
           HOLDJI = HOLDJI + (quick_qm_struct%E(K)*quick_qm_struct%co(J,K)*quick_qm_struct%co(I,K))
        enddo
        quick_scratch%hold(J,I) = 2.d0*HOLDJI
     enddo
  enddo
  endif
  
!  call MPI_BCAST(HOLD,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
!  call MPI_BCAST(quick_method%gradCutoff,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  
  if (quick_method%debug) then
     write(ioutfile,'(/"THE ENERGY WEIGHTED DENSITY MATRIX")')
     do I=1,nbasis
        do J=1,nbasis
           write (ioutfile,'("W[",I4,",",I4,"]=",F18.10)') &
                J,I,quick_scratch%hold(J,I)
        enddo
     enddo
  endif

  ! The contribution to the derivative of energy with respect to nuclear
  ! postion for this term is: -(Sum over i,j) Q(i,j) dS(ij)/dXA
  ! Now Q is symmetric, and dS(ij)/dXA = dS(ji)/dXA.  Furthermore, if
  ! i and j are on the same center, the term is zero. Thus we need to find
  ! the i j pairs for i and j not on the same atom.
  ! Also:  The derivative of a cartesian gtf is:

  ! d/dXA ((x-XA)^i (y-YA)^j (z-ZA)^k e^(-ar^2))
  ! = 2a((x-XA)^(i+1) (y-YA)^j (z-ZA)^k e^(-ar^2))
  ! - i ((x-XA)^(i-1) (y-YA)^j (z-ZA)^k e^(-ar^2))

  ! Note that the negative on the final term comes from the form of (x-XA).

  do i=1,mpi_nbasisn(mpirank)
     Ibas=mpi_nbasis(mpirank,i)
     ISTART = (quick_basis%ncenter(Ibas)-1) *3
     do Jbas=quick_basis%last_basis_function(quick_basis%ncenter(IBAS))+1,nbasis
        JSTART = (quick_basis%ncenter(Jbas)-1) *3
        DENSEJI = quick_qm_struct%dense(Jbas,Ibas)

        ! We have selected our two basis functions, now loop over angular momentum.

        do Imomentum=1,3
           dSI = 0.d0
           dSJ =0.d0
           dKEI = 0.d0
           dKEJ = 0.d0

           ! do the Ibas derivatives first.

           itype(Imomentum,Ibas) = itype(Imomentum,Ibas)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 dSI = dSI + 2.d0*aexp(Icon,Ibas)* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 dKEI = dKEI + 2.d0*aexp(Icon,Ibas)* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
              enddo
           enddo
           itype(Imomentum,Ibas) = itype(Imomentum,Ibas)-1
           if (itype(Imomentum,Ibas) /= 0) then
              itype(Imomentum,Ibas) = itype(Imomentum,Ibas)-1
              do Icon=1,ncontract(Ibas)
                 do Jcon=1,ncontract(Jbas)
                    dSI = dSI - dble(itype(Imomentum,Ibas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                    dKEI = dKEI - dble(itype(Imomentum,Ibas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 enddo
              enddo
              itype(Imomentum,Ibas) = itype(Imomentum,Ibas)+1
           endif
           quick_qm_struct%gradient(ISTART+Imomentum) = quick_qm_struct%gradient(ISTART+Imomentum) &
                -dSI*quick_scratch%hold(Jbas,Ibas)*2.d0 &
                +dKeI*DENSEJI*2.d0

           ! Now do the Jbas derivatives.

           itype(Imomentum,Jbas) = itype(Imomentum,Jbas)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 dSJ = dSJ + 2.d0*aexp(Jcon,Jbas)* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 dKEJ = dKEJ + 2.d0*aexp(Jcon,Jbas)* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
              enddo
           enddo
           itype(Imomentum,Jbas) = itype(Imomentum,Jbas)-1
           if (itype(Imomentum,Jbas) /= 0) then
              itype(Imomentum,Jbas) = itype(Imomentum,Jbas)-1
              do Icon=1,ncontract(Ibas)
                 do Jcon=1,ncontract(Jbas)
                    dSJ = dSJ - dble(itype(Imomentum,Jbas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                    dKEJ = dKEJ - dble(itype(Imomentum,Jbas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 enddo
              enddo
              itype(Imomentum,Jbas) = itype(Imomentum,Jbas)+1
           endif
           quick_qm_struct%gradient(JSTART+Imomentum) = quick_qm_struct%gradient(JSTART+Imomentum) &
                -dSJ*quick_scratch%hold(Jbas,Ibas)*2.d0 &
                +dKEJ*DENSEJI*2.d0
        enddo
     enddo
  enddo
  !endif

  ! 4)  The derivative of the 1 electron nuclear attraction term ij times
  ! the density matrix element ij.

  ! Please note that these are the three center terms.


  do i=1,mpi_jshelln(mpirank)
     IIsh=mpi_jshell(mpirank,i)
     do JJsh=IIsh,jshell
        call attrashellopt(IIsh,JJsh)
     enddo
  enddo


  do II=1,jshell
     do JJ=II,jshell
        DNtemp=0.0d0
        call DNscreen(II,JJ,DNtemp)
        Cutmatrix(II,JJ)=DNtemp
        Cutmatrix(JJ,II)=DNtemp
     enddo
  enddo



  do i=1,mpi_jshelln(mpirank)
     II=mpi_jshell(mpirank,i)
     do JJ=II,jshell
        Testtmp=Ycutoff(II,JJ)
        do KK=II,jshell
           do LL=KK,jshell
              if(quick_basis%katom(II).eq.quick_basis%katom(JJ).and.quick_basis%katom(II).eq.quick_basis%katom(KK) &
                 .and.quick_basis%katom(II).eq.quick_basis%katom(LL))then
                 continue
              else
                 testCutoff = TESTtmp*Ycutoff(KK,LL)
                 if(testCutoff.gt.quick_method%gradCutoff)then
                    DNmax=max(4.0d0*cutmatrix(II,JJ),4.0d0*cutmatrix(KK,LL), &
                         cutmatrix(II,LL),cutmatrix(II,KK),cutmatrix(JJ,KK),cutmatrix(JJ,LL))
                    cutoffTest=testCutoff*DNmax
                    if(cutoffTest.gt.quick_method%gradCutoff)then
                       call shellopt
                    endif
                 endif
              endif
           enddo
        enddo
     enddo
  enddo


  ! stop

  call cpu_time(timer_end%TGrad)

  write(ioutfile, '(2x,"GRADIENT CALCULATION TIME",F15.9, " S")'), timer_end%TGrad-timer_begin%TGrad
  timer_cumer%TGrad=timer_end%TGrad-timer_begin%TGrad+timer_cumer%TGrad

  ! slave node will send infos
  if(.not.master) then

     do i=1,natom*3
        temp_grad(i)=quick_qm_struct%gradient(i)
     enddo
     ! send operator to master node
     call MPI_SEND(temp_grad,3*natom,mpi_double_precision,0,mpirank,MPI_COMM_WORLD,IERROR)
  else
     ! master node will receive infos from every nodes
     do i=1,mpisize-1
        ! receive opertors from slave nodes
        call MPI_RECV(temp_grad,3*natom,mpi_double_precision,i,i,MPI_COMM_WORLD,MPI_STATUS,IERROR)
        ! and sum them into operator
        do ii=1,natom*3
           quick_qm_struct%gradient(ii)=quick_qm_struct%gradient(ii)+temp_grad(ii)
        enddo
     enddo
  endif

  return
end subroutine mpi_hfgrad
#endif

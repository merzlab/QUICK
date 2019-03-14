! Ed Brothers. May 23, 2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    subroutine dftgrad
    use allmod
    implicit double precision(a-h,o-z)

! The purpose of this subroutine is to calculate the gradient of
! the total energy with respect to nuclear displacement.  The results
! of this are stored in Gradient, which is organized by atom and then
! by direction of displacement, i.e. element 1 is the gradient of the
! x diplacement of atom 1, element 5 is the y displacement of atom 2.

! This is the restricted DFT code.  Notes on how this differs from the
! restricted HF gradient code will be found as the differences pop up.

    character(len=1) cartsym(3)

    cartsym(1) = 'X'
    cartsym(2) = 'Y'
    cartsym(3) = 'Z'
    call cpu_time(timer_begin%TGrad)
    do Iatm=1,natom*3
        quick_qm_struct%gradient(iatm)=0.d0
    enddo

! The gradient at this level of theory is the sum of seven terms.
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

!!!!!!!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!!
        write (ioutfile,'(/," Madu STEP1 :  ANALYTICAL GRADIENT: ")')
        do Iatm=1,natom
            do Imomentum=1,3
                write (ioutfile,'(I5,A1,7x,F20.10)')Iatm,cartsym(imomentum), &
                quick_qm_struct%gradient((Iatm-1)*3+Imomentum)
            enddo
        enddo
!!!!!!!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!

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

!!!!!!!!!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        write (ioutfile,'(/," Madu STEP 2-3: HOLD variables: ")')
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    do I=1,nbasis
        do J=1,nbasis
            HOLDJI = 0.d0
            do K=1,quick_molspec%nelec/2
                HOLDJI = HOLDJI + (quick_qm_struct%E(K)*quick_qm_struct%co(J,K)*quick_qm_struct%co(I,K))

!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!!!
                write (ioutfile,'(A4,I5,I5,I5,7x,F20.10,7x,F20.10,7x,F20.10)') "Vars",I,J,K, &
                quick_qm_struct%Eel(K),quick_qm_struct%co(J,K),quick_qm_struct%co(I,K)
!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!!!

            enddo
            quick_scratch%hold(J,I) = 2.d0*HOLDJI

!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!!!
            write (ioutfile,'(A1,I5,I5,7x,F20.10)') "H",I,J,quick_scratch%hold(J,I)
!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!!!

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

            ! Do the Ibas derivatives first.

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

!!!!!!!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!!
        write (ioutfile,'(/," Madu STEP 2-3 :  ANALYTICAL GRADIENT: ")')
        do Iatm=1,natom
            do Imomentum=1,3
                write (ioutfile,'(I5,A1,7x,F20.10)')Iatm,cartsym(imomentum), &
                quick_qm_struct%gradient((Iatm-1)*3+Imomentum)
            enddo
        enddo
!!!!!!!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!

! 4)  The derivative of the 1 electron nuclear attraction term ij times
! the density matrix element ij.

! Please note that these are the three center terms.

    do IIsh=1,jshell
      do JJsh=IIsh,jshell
        call attrashellopt(IIsh,JJsh)
      enddo
    enddo

! 5)  The derivative of the 4center 2e- terms with respect to X times
! the coefficient found in the energy. (i.e. the multiplicative
! constants from the density matrix that arise as these are both
! the exchange and correlation integrals.

! The previous two terms are the one electron part of the Fock matrix.
! The next term defines the electron repulsion.

! Delta density matrix cutoff

 do II=1,jshell
   do JJ=II,jshell
     DNtemp=0.0d0
     call DNscreen(II,JJ,DNtemp)
     Cutmatrix(II,JJ)=DNtemp
     Cutmatrix(JJ,II)=DNtemp
   enddo
 enddo

! Madu: Insert GPU code here        

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
          If(testCutoff.gt.quick_method%gradCutoff)then
            DNmax=max(cutmatrix(II,JJ),cutmatrix(KK,LL))
            cutoffTest=testCutoff*DNmax
            If(cutoffTest.gt.quick_method%gradCutoff)then
!              print*,II,JJ,KK,LL
!              call shelloptdft
              IIxiao=II
              JJxiao=JJ
              KKxiao=KK
              LLxiao=LL
              !Madu: Located in 2eshelloptdft.f90. 
              call shelloptdft(IIxiao,JJxiao,KKxiao,LLxiao)
!              ntempxiao2=ntempxiao2+1
            endif
          endif
         endif
       enddo
     enddo
   enddo
 enddo

! stop

        call cpu_time(timer_end%TGrad)

        write (ioutfile,'(" TIME of evaluation gradients of DFT 2e-integral= ",F12.2)') &
        timer_end%TGrad-timer_begin%TGrad
!        write (ioutfile,*)'ntempxiao1,ntempxiao2=',ntempxiao1,ntempxiao2
        timer_cumer%TGrad=timer_end%TGrad-timer_begin%TGrad+timer_cumer%TGrad

!!!!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        write (ioutfile,'(/," Madu STEP5: ANALYTICAL GRADIENT: ")')
        do Iatm=1,natom
            do Imomentum=1,3
                write (ioutfile,'(I5,A1,7x,F20.10)')Iatm,cartsym(imomentum), &
                quick_qm_struct%gradient((Iatm-1)*3+Imomentum)
            enddo
        enddo


! 6) The derivative of the exchange/correlation functional energy
! with respect to nuclear displacement.

! 7) The derivative of the weight of the quadrature points with respect
! to nuclear displacement.


! These two terms arise because of the quadrature used to calculate the
! XC terms.

! Exc = (Sum over grid points) W(g) f(g)
! dExc/dXA = (Sum over grid points) dW(g)/dXA f(g) + W(g) df(g)/dXA

! For the W(g) df(g)/dXA term, the derivation was done by Ed Brothers and
! is a varient of the method found in the Johnson-Gill-Pople paper.  It can
! be found in Ed's thesis, assuming he ever writes it.

! One of the actuals element is:
! dExc/dXa =2*Dense(Mu,nu)*(Sum over mu centered on A)(Sumover all nu)
! Integral((df/drhoa dPhimu/dXA Phinu)-
! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b))
! DOT Grad(dPhimu/dXa Phinu))

! where F alpha mu nu is the the alpha spin portion of the operator matrix
! element mu, nu,
! df/drhoa is the derivative of the functional by the alpha density,
! df/dgaa is the derivative of the functional by the alpha gradient
! invariant, i.e. the dot product of the gradient of the alpha
! density with itself.
! df/dgab is the derivative of the functional by the dot product of
! the gradient of the alpha density with the beta density.
! Grad(Phimu Phinu) is the gradient of Phimu times Phinu.
! First, find the grid point.


!    DO Ireg=1,iregion
!        call gridform(iangular(Ireg))
!        DO Irad=iradial(ireg-1)+1,iradial(ireg)
        call cpu_time(timer_begin%TGrad)

         do Iatm=1,natom
           if(quick_method%iSG.eq.1)then
             Iradtemp=50
           else
             if(quick_molspec%iattype(iatm).le.10)then
               Iradtemp=23
             else
               Iradtemp=26
             endif
            endif

           do Irad=1,Iradtemp
             if(quick_method%iSG.eq.1)then
               call gridformnew(iatm,RGRID(Irad),iiangt)
               rad = radii(quick_molspec%iattype(iatm))
             else
               call gridformSG0(iatm,Iradtemp+1-Irad,iiangt,RGRID,RWT)
               rad = radii2(quick_molspec%iattype(iatm))
             endif

!            DO Iatm=1,natom
!                rad = radii(iattype(iatm))
                rad3 = rad*rad*rad
               do Iang=1,iiangt
!                DO Iang=1,iangular(Ireg)
                    gridx=xyz(1,Iatm)+rad*RGRID(Irad)*XANG(Iang)
                    gridy=xyz(2,Iatm)+rad*RGRID(Irad)*YANG(Iang)
                    gridz=xyz(3,Iatm)+rad*RGRID(Irad)*ZANG(Iang)

                ! Next, calculate the weight of the grid point in the SSW scheme.  If
                ! the grid point has a zero weight, we can skip it.

                    sswt=SSW(gridx,gridy,gridz,Iatm)
                    weight=sswt*WTANG(Iang)*RWT(Irad)*rad3

                    if (weight < tol ) then
                        continue
                    else

                            do Ibas=1,nbasis
                                call pteval(gridx,gridy,gridz,phi,dphidx,dphidy, &
                                dphidz,Ibas)
                                phixiao(Ibas)=phi
                                dphidxxiao(Ibas)=dphidx
                                dphidyxiao(Ibas)=dphidy
                                dphidzxiao(Ibas)=dphidz
                            enddo

                    ! Next, evaluate the densities at the grid point and the gradient
                    ! at that grid point.

                        call denspt(gridx,gridy,gridz,density,densityb,gax,gay,gaz, &
                        gbx,gby,gbz)

                        if (density < tol ) then
                            continue
                        else

                        ! This allows the calculation of the derivative of the functional
                        ! with regard to the density (dfdr), with regard to the alpha-alpha
                        ! density invariant (df/dgaa), and the alpha-beta density invariant.

                            call becke(density,gax,gay,gaz,gbx,gby,gbz, &
                            dfdr,dfdgaa,dfdgab)

                            call lyp(density,densityb,gax,gay,gaz,gbx,gby,gbz, &
                            dfdr2,dfdgaa2,dfdgab2)

!                            dfdr = param7*dfdr+param8*dfdr2
!                            dfdgaa = param7*dfdgaa + param8*dfdgaa2
!                            dfdgab = param7*dfdgab + param8*dfdgab2

                            dfdr = dfdr+dfdr2
                            dfdgaa = dfdgaa + dfdgaa2
                            dfdgab = dfdgab + dfdgab2

                        ! Calculate the first term in the dot product shown above,i.e.:
                        ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b))

                            xdot=2.d0*dfdgaa*gax+dfdgab*gbx
                            ydot=2.d0*dfdgaa*gay+dfdgab*gby
                            zdot=2.d0*dfdgaa*gaz+dfdgab*gbz

                        ! Now loop over basis functions and compute the addition to the matrix
                        ! element.

                            do Ibas=1,nbasis
!                                call pteval(gridx,gridy,gridz,phi,dphidx,dphidy, &
!                                dphidz,Ibas)
                                phi=phixiao(Ibas)
                                dphidx=dphidxxiao(Ibas)
                                dphidy=dphidyxiao(Ibas)
                                dphidz=dphidzxiao(Ibas)

!                                quicktest = DABS(dphidx)+DABS(dphidy)+DABS(dphidz) &
!                                +DABS(phi)
                                quicktest = DABS(dphidx+dphidy+dphidz+ &
                                phi)
                                if (quicktest < tol ) then
                                    continue
                                else
                                    call pt2der(gridx,gridy,gridz,dxdx,dxdy,dxdz, &
                                    dydy,dydz,dzdz,Ibas)
                                    Ibasstart=(quick_basis%ncenter(Ibas)-1)*3

                                    do Jbas=1,nbasis
!                                        call pteval(gridx,gridy,gridz,phi2,dphi2dx,dphi2dy, &
!                                        dphi2dz,Jbas)
                                        phi2=phixiao(Jbas)
                                        dphi2dx=dphidxxiao(Jbas)
                                        dphi2dy=dphidyxiao(Jbas)
                                        dphi2dz=dphidzxiao(Jbas)

!                                        quicktest = DABS(dphi2dx)+DABS(dphi2dy)+DABS(dphi2dz) &
!                                        +DABS(phi2)
!                                        IF (quicktest < tol ) THEN
!                                            continue
!                                        ELSE
                                        quick_qm_struct%gradient(Ibasstart+1) = quick_qm_struct%gradient(Ibasstart+1) - &
                                            2.d0*quick_qm_struct%dense(Ibas,Jbas)*weight* &
                                            (dfdr*dphidx*phi2 &
                                            + xdot*(dxdx*phi2+dphidx*dphi2dx) &
                                            + ydot*(dxdy*phi2+dphidx*dphi2dy) &
                                            + zdot*(dxdz*phi2+dphidx*dphi2dz))
                                            quick_qm_struct%gradient(Ibasstart+2) = quick_qm_struct%gradient(Ibasstart+2) - &
                                            2.d0*quick_qm_struct%dense(Ibas,Jbas)*weight* &
                                            (dfdr*dphidy*phi2 &
                                            + xdot*(dxdy*phi2+dphidy*dphi2dx) &
                                            + ydot*(dydy*phi2+dphidy*dphi2dy) &
                                            + zdot*(dydz*phi2+dphidy*dphi2dz))
                                            quick_qm_struct%gradient(Ibasstart+3) = quick_qm_struct%gradient(Ibasstart+3) - &
                                            2.d0*quick_qm_struct%dense(Ibas,Jbas)*weight* &
                                            (dfdr*dphidz*phi2 &
                                            + xdot*(dxdz*phi2+dphidz*dphi2dx) &
                                            + ydot*(dydz*phi2+dphidz*dphi2dy) &
                                            + zdot*(dzdz*phi2+dphidz*dphi2dz))

!                                        ENDIF
                                    enddo

                                endif
                            enddo
        

                        ! We are now completely done with the derivative of the
                        ! exchange correlation energy with nuclear displacement at this point.
                        ! Now we need to do the quadrature weight derivatives.  At this point
                        ! in the loop, we know that the density and the weight are not zero.  Now
                        ! check to see fi the weight is one.  If it isn't, we need to
                        ! actually calculate the energy and the derivatives of the
                        ! quadrature at this point. Due to the volume of code, this is done in
                        ! sswder.  Note that if a new weighting scheme is ever added, this needs
                        ! to be modified with a second subprogram.

                            if (sswt == 1.d0) then
                                continue
                            else
                                call becke_e(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ex)
                                call lyp_e(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ec)
                                Exc=Ex+Ec
                                call sswder(gridx,gridy,gridz,Exc,weight/sswt,Iatm)
                            endif
                        endif
                    endif
                enddo
            enddo
        enddo
!    ENDDO

        call cpu_time(timer_end%TGrad)

        write (ioutfile,'(" TIME of evaluation numerical integral gradient of DFT= ",F12.2)') &
        timer_end%TGrad-timer_begin%TGrad
        timer_cumer%TGrad=timer_end%TGrad-timer_begin%TGrad+timer_cumer%TGrad

!!!!!!!!!!!!!!!!!!!!!!!!!!Madu!!!!!!!!!!!!!!!!!!!!!!!!
        write (ioutfile,'(/," Madu STEP7 :  ANALYTICAL GRADIENT: ")')
        do Iatm=1,natom
            do Imomentum=1,3
                write (ioutfile,'(I5,A1,7x,F20.10)')Iatm,cartsym(imomentum), &
                quick_qm_struct%gradient((Iatm-1)*3+Imomentum)
            enddo
        enddo


    end subroutine dftgrad





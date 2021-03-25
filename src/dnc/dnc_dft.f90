#include "util.fh"
! Ed Brothers. February 5, 2003
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
!
! Ed Brothers. November 27, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine dftoperatordelta
   use allmod
   use quick_cutoff_module, only: cshell_dnscreen
   implicit double precision(a-h,o-z)
   double precision g_table(200)
   integer i,j,k,ii,jj,kk,g_count

   ! The purpose of this subroutine is to form the operator matrix
   ! for a full Density Functional calculation, i.e. the KS matrix.  The
   ! KS  matrix is as follows:

   ! O(I,J) =  F(I,J) = KE(I,J) + IJ attraction to each atom + repulsion_prim
   ! with each possible basis  + Exchange/Correlation functional.

   ! Note that the KS operator matrix is symmetric.

   call cpu_time(t1)

   Eelxc=0.0d0

   if(quick_method%printEnergy)then
      quick_qm_struct%Eel=0.d0
      do Ibas=1,nbasis

         Bx = xyz(1,quick_basis%ncenter(Ibas))
         By = xyz(2,quick_basis%ncenter(Ibas))
         Bz = xyz(3,quick_basis%ncenter(Ibas))
         ii = itype(1,Ibas)
         jj = itype(2,Ibas)
         kk = itype(3,Ibas)
         g_count = ii+ii+jj+jj+kk+kk+2
         
         do Icon=1,ncontract(Ibas)
            do Jcon=1,ncontract(Ibas)

               b = aexp(Icon,Ibas)
               a = aexp(Jcon,Ibas)
               call gpt(a,b,Bx,By,Bz,Bx,By,Bz,Px,Py,Pz,g_count,g_table)      

               ! Kinetic energy.

               quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%denseSave(Ibas,Ibas)* &
                     dcoeff(Jcon,Ibas)*dcoeff(Icon,Ibas)* &
                      ekinetic(a,b,ii ,jj,kk,ii,jj,kk,Bx,By,Bz,Bx,By,Bz,Px,Py,Pz,g_table) 
!                     ekinetic(aexp(Jcon,Ibas),aexp(Icon,Ibas), &
!                     itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                     itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                     xyz(1,quick_basis%ncenter(Ibas)),xyz(2,quick_basis%ncenter(Ibas)), &
!                     xyz(3,quick_basis%ncenter(Ibas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                     xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))

               ! Nuclear attraction.

               !                do iatom = 1,natom
               !                    Eel=Eel+DENSESAVE(Ibas,Ibas)* &
                     !                    dcoeff(Jcon,Ibas)*dcoeff(Icon,Ibas)* &
                     !                    attraction(aexp(Jcon,Ibas),aexp(Icon,Ibas), &
                     !                    itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                     !                    itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                     !                    xyz(1,quick_basis%ncenter(Ibas)),xyz(2,quick_basis%ncenter(Ibas)), &
                     !                    xyz(3,quick_basis%ncenter(Ibas)),xyz(1,quick_basis%ncenter(Ibas)), &
                     !                    xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)), &
                     !                    xyz(1,iatom),xyz(2,iatom),xyz(3,iatom), &
                     !                    chg(iatom))
               !                enddo
            enddo
         enddo
      enddo

      do Ibas=1,nbasis
         do Jbas=Ibas+1,nbasis

            Ax = xyz(1,quick_basis%ncenter(Jbas))
            Bx = xyz(1,quick_basis%ncenter(Ibas))
            Ay = xyz(2,quick_basis%ncenter(Jbas))
            By = xyz(2,quick_basis%ncenter(Ibas))
            Az = xyz(3,quick_basis%ncenter(Jbas))
            Bz = xyz(3,quick_basis%ncenter(Ibas))
            
            ii = itype(1,Ibas)
            jj = itype(2,Ibas)
            kk = itype(3,Ibas)
            i = itype(1,Jbas)
            j = itype(2,Jbas)
            k = itype(3,Jbas)
            g_count = i+ii+j+jj+k+kk+2

            do Icon=1,ncontract(ibas)
               do Jcon=1,ncontract(jbas)

                 b = aexp(Icon,Ibas)
                 a = aexp(Jcon,Jbas)
                 call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)      

                  ! Kinetic energy.

                  quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%denseSave(Jbas,Ibas)* &
                        dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)* &
                        2.d0* &
                      ekinetic(a,b,i ,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) 

!                  ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                        itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                        itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                        xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                        xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                        xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))

                  ! Nuclear attraction.

                  !                    do iatom = 1,natom
                  !                        Eel=Eel+DENSESAVE(Jbas,Ibas)* &
                        !                        dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)* &
                        !                        2.d0*attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                        !                        itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                        !                        itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                        !                        xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                        !                        xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                        !                        xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)), &
                        !                        xyz(1,iatom),xyz(2,iatom),xyz(3,iatom), &
                        !                        chg(iatom))
                  !                    enddo
               enddo
            enddo
         enddo
      enddo

      do itemp1=1,nbasis
         do jtemp2=1,nbasis
            quick_scratch%hold(jtemp2,itemp1)=quick_qm_struct%dense(jtemp2,itemp1)
            quick_qm_struct%dense(jtemp2,itemp1)=quick_qm_struct%denseSave(jtemp2,itemp1)
         enddo
      enddo

      do IIsh=1,jshell
         do JJsh=IIsh,jshell
            call attrashellenergy(IIsh,JJsh)
         enddo
      enddo

      do itemp1=1,nbasis
         do jtemp2=1,nbasis
            quick_qm_struct%dense(jtemp2,itemp1)=quick_scratch%hold(jtemp2,itemp1)
         enddo
      enddo

   endif

   call cpu_time(t2)


   do I=1,nbasis
      do J=1,nbasis
         quick_qm_struct%o(i,j)=quick_qm_struct%Osavedft(i,j)
      enddo
   enddo

   !
   ! Alessandro GENONI 03/21/2007
   ! Sum the ECP integrals to the partial Fock matrix
   !
   if (quick_method%ecp) then
      call ecpoperator
   end if

   ! The previous two terms are the one electron part of the Fock matrix.
   ! The next term defines the electron repulsion_prim.

   call cpu_time(T1)

   ! Delta density matrix cutoff

   do II=1,jshell
      do JJ=II,jshell
         DNtemp=0.0d0
         call cshell_dnscreen(II,JJ,DNtemp)
         Cutmatrix(II,JJ)=DNtemp
         Cutmatrix(JJ,II)=DNtemp
      enddo
   enddo

   ! Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
   ! Reference: Strout DL and Scuseria JCP 102(1995),8448.

   ! print*,"before 2e"
   if(quick_method%B3LYP)then
      do II=1,jshell
         do JJ=II,jshell
            Testtmp=Ycutoff(II,JJ)
            do KK=II,jshell
               do LL=KK,jshell
                  !          Nxiao1=Nxiao1+1
                  testCutoff = TESTtmp*Ycutoff(KK,LL)
                  if(testCutoff.gt.quick_method%integralCutoff)then
                     DNmax=max(4.0d0*cutmatrix(II,JJ),4.0d0*cutmatrix(KK,LL), &
                           cutmatrix(II,LL),cutmatrix(II,KK),cutmatrix(JJ,KK),cutmatrix(JJ,LL))
                     !            DNmax=max(cutmatrix(II,JJ),cutmatrix(KK,LL) &
                           !                  )
                     cutoffTest=testCutoff*DNmax
                     if(cutoffTest.gt.quick_method%integralCutoff)then
                        IIxiao=II
                        JJxiao=JJ
                        KKxiao=KK
                        LLxiao=LL
                        call shelldftb3lyp(IIxiao,JJxiao,KKxiao,LLxiao)
                        !            Nxiao2=Nxiao2+1
                     endif
                     !            else
                     !             print*,II,JJ,KK,LL,cutoffTest,testCutoff,DNmax
                     !            print*,'***',O(1,1)
                  endif
               enddo
            enddo
         enddo
      enddo
   else
      do II=1,jshell
         do JJ=II,jshell
            Testtmp=Ycutoff(II,JJ)
            do KK=II,jshell
               do LL=KK,jshell
                  !          Nxiao1=Nxiao1+1
                  testCutoff = TESTtmp*Ycutoff(KK,LL)
                  if(testCutoff.gt.quick_method%integralCutoff)then
                     !            DNmax=max(4.0d0*cutmatrix(II,JJ),4.0d0*cutmatrix(KK,LL), &
                           !                  cutmatrix(II,LL),cutmatrix(II,KK),cutmatrix(JJ,KK),cutmatrix(JJ,LL))
                     DNmax=max(cutmatrix(II,JJ),cutmatrix(KK,LL) &
                           )
                     cutoffTest=testCutoff*DNmax
                     if(cutoffTest.gt.quick_method%integralCutoff)then
                        IIxiao=II
                        JJxiao=JJ
                        KKxiao=KK
                        LLxiao=LL
                        call shelldft(IIxiao,JJxiao,KKxiao,LLxiao)
                        !            Nxiao2=Nxiao2+1
                     endif
                     !            else
                     !             print*,II,JJ,KK,LL,cutoffTest,testCutoff,DNmax
                     !            print*,'***',O(1,1)
                  endif
               enddo
            enddo
         enddo
      enddo
   endif


   do I=1,nbasis
      do J=1,nbasis
         quick_qm_struct%Osavedft(i,j)=quick_qm_struct%o(i,j)
      enddo
   enddo

   do I=1,nbasis
      do J=1,nbasis
         quick_qm_struct%dense(I,J)=quick_qm_struct%denseSave(I,J)
      enddo
   enddo

   call cpu_time(t2)

   write (ioutfile,'(" TIME of evaluation integral = ",F12.2)') &
         T2-T1

   ! print*,'Nxiao1=',Nxiao1,'Nxiao2=',Nxiao2,integralCutOff

   do Ibas=1,nbasis
      do Jbas=Ibas+1,nbasis
         quick_qm_struct%o(Ibas,Jbas) = quick_qm_struct%o(Jbas,Ibas)
      enddo
   enddo

   if(quick_method%printEnergy)then
      do Ibas=1,nbasis
         do Jbas=1,nbasis
            quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%denseSave(Ibas,Jbas)*quick_qm_struct%o(Jbas,Ibas)
         enddo
      enddo

      quick_qm_struct%Eel=quick_qm_struct%Eel/2.0d0
   endif

   !           write (ioutfile,'("TOTAL ENERGY OF CURRENT CYCLE=",F16.9)') Eel

   ! stop

   ! The next portion is the exchange/correlation functional.
   ! The angular grid code came from CCL.net.  The radial grid
   ! formulas (position and wieghts) is from Gill, Johnson and Pople,
   ! Chem. Phys. Lett. v209, n 5+6, 1993, pg 506-512.  The weighting scheme
   ! is from Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
   ! 1996, pg 213-223.

   ! The actual element is:
   ! F alpha mu nu = Integral((df/drhoa Phimu Phinu)+
   ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))

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

   quick_qm_struct%aelec=0.d0
   quick_qm_struct%belec=0.d0

   ! Xiao HE 02/09/1007 SG0 SG1 grids
   !    do Ireg=1,iregion
   !        call gridform(iangular(Ireg))
   !        do Irad=iradial(ireg-1)+1,iradial(ireg)
   !            do Iatm=1,natom
   !                rad = radii(quick_molspec%iattype(iatm))
   !                rad3 = rad*rad*rad
   !                do Iang=1,iangular(Ireg)

   if(quick_method%B3LYP)then
      do Iatm=1,natom
         if(quick_method%ISG.eq.1)then
            Iradtemp=50
         else
            if(quick_molspec%iattype(iatm).le.10)then
               Iradtemp=23
            else
               Iradtemp=26
            endif
         endif

         do Irad=1,Iradtemp
            if(quick_method%ISG.eq.1)then
               call gridformnew(iatm,RGRID(Irad),iiangt)
               rad = radii(quick_molspec%iattype(iatm))
            else
               call gridformSG0(iatm,Iradtemp+1-Irad,iiangt,RGRID,RWT)
               rad = radii2(quick_molspec%iattype(iatm))
            endif

            rad3 = rad*rad*rad
            do Iang=1,iiangt
               gridx=xyz(1,Iatm)+rad*RGRID(Irad)*XANG(Iang)
               gridy=xyz(2,Iatm)+rad*RGRID(Irad)*YANG(Iang)
               gridz=xyz(3,Iatm)+rad*RGRID(Irad)*ZANG(Iang)

               ! Next, calculate the weight of the grid point in the SSW scheme.  if
               ! the grid point has a zero weight, we can skip it.

               weight=SSW(gridx,gridy,gridz,Iatm) &
                     *WTANG(Iang)*RWT(Irad)*rad3

               if (weight < quick_method%DMCutoff ) then
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

                  if (density < quick_method%DMCutoff ) then
                     continue
                  else

                     densitysum=2.0d0*density
                     sigma=4.0d0*(gax*gax+gay*gay+gaz*gaz)
                     call b3lyp_e(densitysum,sigma,zkec)
                     !
                     !                    Eel = Eel + (param7*Ex+param8*Ec) &
                           Eelxc = Eelxc + zkec*weight

                     quick_qm_struct%aelec = weight*density+quick_qm_struct%aelec
                     quick_qm_struct%belec = weight*densityb+quick_qm_struct%belec

                     ! This allows the calculation of the derivative of the functional
                     ! with regard to the density (dfdr), with regard to the alpha-alpha
                     ! density invariant (df/dgaa), and the alpha-beta density invariant.

                     !                            call becke(density,gax,gay,gaz,gbx,gby,gbz, &
                           !                            dfdr,dfdgaa,dfdgab)
                     !
                     !                            call lyp(density,densityb,gax,gay,gaz,gbx,gby,gbz, &
                           !                            dfdr2,dfdgaa2,dfdgab2)

                     densitysum=2.0d0*density
                     sigma=4.0d0*(gax*gax+gay*gay+gaz*gaz)

                     call b3lypf(densitysum,sigma,dfdr,xiaodot)

                     !                            dfdr=dfdr*2.0d0

                     !                            dfdr = dfdr+dfdr2
                     !                            dfdgaa = dfdgaa + dfdgaa2
                     !                            dfdgab = dfdgab + dfdgab2

                     ! Calculate the first term in the dot product shown above,i.e.:
                     ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))

                     xdot=xiaodot*gax
                     ydot=xiaodot*gay
                     zdot=xiaodot*gaz

                     !                            xdot=2.d0*dfdgaa*gax+dfdgab*gbx
                     !                            ydot=2.d0*dfdgaa*gay+dfdgab*gby
                     !                            zdot=2.d0*dfdgaa*gaz+dfdgab*gbz

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
                        quicktest = DABS(dphidx+dphidy+dphidz &
                              +phi)
                        if (quicktest < quick_method%DMCutoff ) then
                           continue
                        else
                           do Jbas=Ibas,nbasis
                              !                                        call pteval(gridx,gridy,gridz,phi2,dphi2dx,dphi2dy, &
                                    !                                        dphi2dz,Jbas)
                              phi2=phixiao(Jbas)
                              dphi2dx=dphidxxiao(Jbas)
                              dphi2dy=dphidyxiao(Jbas)
                              dphi2dz=dphidzxiao(Jbas)
                              !                                        quicktest = DABS(dphi2dx)+DABS(dphi2dy)+DABS(dphi2dz) &
                                    !                                        +DABS(phi2)
                              !                                        if (quicktest < quick_method%DMCutoff ) then
                              !                                            continue
                              !                                        else
                              temp = phi*phi2
                              tempgx = phi*dphi2dx + phi2*dphidx
                              tempgy = phi*dphi2dy + phi2*dphidy
                              tempgz = phi*dphi2dz + phi2*dphidz
                              quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+ &
                                    xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                              !                                        endif
                           enddo
                        endif
                     enddo
                  endif
               endif
            enddo
         enddo
      enddo
      !    enddo
   endif

   if(quick_method%BLYP)then
      do Iatm=1,natom
         if(quick_method%ISG.eq.1)then
            Iradtemp=50
         else
            if(quick_molspec%iattype(iatm).le.10)then
               Iradtemp=23
            else
               Iradtemp=26
            endif
         endif

         do Irad=1,Iradtemp
            if(quick_method%ISG.eq.1)then
               call gridformnew(iatm,RGRID(Irad),iiangt)
               rad = radii(quick_molspec%iattype(iatm))
            else
               call gridformSG0(iatm,Iradtemp+1-Irad,iiangt,RGRID,RWT)
               rad = radii2(quick_molspec%iattype(iatm))
            endif

            rad3 = rad*rad*rad
            !             print*,iiangt
            do Iang=1,iiangt
               gridx=xyz(1,Iatm)+rad*RGRID(Irad)*XANG(Iang)
               gridy=xyz(2,Iatm)+rad*RGRID(Irad)*YANG(Iang)
               gridz=xyz(3,Iatm)+rad*RGRID(Irad)*ZANG(Iang)

               ! Next, calculate the weight of the grid point in the SSW scheme.  if
               ! the grid point has a zero weight, we can skip it.

               weight=SSW(gridx,gridy,gridz,Iatm) &
                     *WTANG(Iang)*RWT(Irad)*rad3

               if (weight < quick_method%DMCutoff ) then
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

                  if (density < quick_method%DMCutoff ) then
                     continue
                  else

                     call becke_E(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ex)

                     call lyp_e(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ec)

                     Eelxc = Eelxc + (param7*Ex+param8*Ec) &
                           *weight


                     quick_qm_struct%aelec = weight*density+quick_qm_struct%aelec
                     quick_qm_struct%belec = weight*densityb+quick_qm_struct%belec

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
                     ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))

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
                        if (quicktest < quick_method%DMCutoff ) then
                           continue
                        else
                           do Jbas=Ibas,nbasis
                              !                                        call pteval(gridx,gridy,gridz,phi2,dphi2dx,dphi2dy, &
                                    !                                        dphi2dz,Jbas)
                              phi2=phixiao(Jbas)
                              dphi2dx=dphidxxiao(Jbas)
                              dphi2dy=dphidyxiao(Jbas)
                              dphi2dz=dphidzxiao(Jbas)
                              !                                        quicktest = DABS(dphi2dx)+DABS(dphi2dy)+DABS(dphi2dz) &
                                    !                                        +DABS(phi2)
                              !                                        if (quicktest < quick_method%DMCutoff ) then
                              !                                            continue
                              !                                        else
                              temp = phi*phi2
                              tempgx = phi*dphi2dx + phi2*dphidx
                              tempgy = phi*dphi2dy + phi2*dphidy
                              tempgz = phi*dphi2dz + phi2*dphidz
                              quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+ &
                                    xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                              !                                        endif
                           enddo
                        endif
                     enddo
                  endif
               endif
            enddo
         enddo
      enddo
   endif

   if(quick_method%MPW91LYP)then
      do Iatm=1,natom
         if(quick_method%ISG.eq.1)then
            Iradtemp=50
         else
            if(quick_molspec%iattype(iatm).le.10)then
               Iradtemp=23
            else
               Iradtemp=26
            endif
         endif

         do Irad=1,Iradtemp
            if(quick_method%ISG.eq.1)then
               call gridformnew(iatm,RGRID(Irad),iiangt)
               rad = radii(quick_molspec%iattype(iatm))
            else
               call gridformSG0(iatm,Iradtemp+1-Irad,iiangt,RGRID,RWT)
               rad = radii2(quick_molspec%iattype(iatm))
            endif

            rad3 = rad*rad*rad
            !             print*,iiangt
            do Iang=1,iiangt
               gridx=xyz(1,Iatm)+rad*RGRID(Irad)*XANG(Iang)
               gridy=xyz(2,Iatm)+rad*RGRID(Irad)*YANG(Iang)
               gridz=xyz(3,Iatm)+rad*RGRID(Irad)*ZANG(Iang)

               ! Next, calculate the weight of the grid point in the SSW scheme.  if
               ! the grid point has a zero weight, we can skip it.

               weight=SSW(gridx,gridy,gridz,Iatm) &
                     *WTANG(Iang)*RWT(Irad)*rad3

               if (weight < quick_method%DMCutoff ) then
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

                  if (density < quick_method%DMCutoff ) then
                     continue
                  else

                     call mpw91_E(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ex)

                     call lyp_e(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ec)

                     Eelxc = Eelxc + (param7*Ex+param8*Ec) &
                           *weight


                     quick_qm_struct%aelec = weight*density+quick_qm_struct%aelec
                     quick_qm_struct%belec = weight*densityb+quick_qm_struct%belec

                     ! This allows the calculation of the derivative of the functional
                     ! with regard to the density (dfdr), with regard to the alpha-alpha
                     ! density invariant (df/dgaa), and the alpha-beta density invariant.

                     call mpw91(density,gax,gay,gaz,gbx,gby,gbz, &
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
                     ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))

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
                        if (quicktest < quick_method%DMCutoff ) then
                           continue
                        else
                           do Jbas=Ibas,nbasis
                              !                                        call pteval(gridx,gridy,gridz,phi2,dphi2dx,dphi2dy, &
                                    !                                        dphi2dz,Jbas)
                              phi2=phixiao(Jbas)
                              dphi2dx=dphidxxiao(Jbas)
                              dphi2dy=dphidyxiao(Jbas)
                              dphi2dz=dphidzxiao(Jbas)
                              !                                        quicktest = DABS(dphi2dx)+DABS(dphi2dy)+DABS(dphi2dz) &
                                    !                                        +DABS(phi2)
                              !                                        if (quicktest < quick_method%DMCutoff ) then
                              !                                            continue
                              !                                        else
                              temp = phi*phi2
                              tempgx = phi*dphi2dx + phi2*dphidx
                              tempgy = phi*dphi2dy + phi2*dphidy
                              tempgz = phi*dphi2dz + phi2*dphidz
                              quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+ &
                                    xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                              !                                        endif
                           enddo
                        endif
                     enddo
                  endif
               endif
            enddo
         enddo
      enddo
   endif

   if(quick_method%BPW91)then
      do Iatm=1,natom
         if(quick_method%ISG.eq.1)then
            Iradtemp=50
         else
            if(quick_molspec%iattype(iatm).le.10)then
               Iradtemp=23
            else
               Iradtemp=26
            endif
         endif

         do Irad=1,Iradtemp
            if(quick_method%ISG.eq.1)then
               call gridformnew(iatm,RGRID(Irad),iiangt)
               rad = radii(quick_molspec%iattype(iatm))
            else
               call gridformSG0(iatm,Iradtemp+1-Irad,iiangt,RGRID,RWT)
               rad = radii2(quick_molspec%iattype(iatm))
            endif

            rad3 = rad*rad*rad
            !             print*,iiangt
            do Iang=1,iiangt
               gridx=xyz(1,Iatm)+rad*RGRID(Irad)*XANG(Iang)
               gridy=xyz(2,Iatm)+rad*RGRID(Irad)*YANG(Iang)
               gridz=xyz(3,Iatm)+rad*RGRID(Irad)*ZANG(Iang)

               ! Next, calculate the weight of the grid point in the SSW scheme.  if
               ! the grid point has a zero weight, we can skip it.

               weight=SSW(gridx,gridy,gridz,Iatm) &
                     *WTANG(Iang)*RWT(Irad)*rad3

               if (weight < quick_method%DMCutoff ) then
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

                  if (density < quick_method%DMCutoff ) then
                     continue
                  else

                     call becke_E(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ex)

                     call lyp_e(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ec)

                     Eelxc = Eelxc + (param7*Ex+param8*Ec) &
                           *weight


                     quick_qm_struct%aelec = weight*density+quick_qm_struct%aelec
                     quick_qm_struct%belec = weight*densityb+quick_qm_struct%belec

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
                     ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))

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
                        if (quicktest < quick_method%DMCutoff ) then
                           continue
                        else
                           do Jbas=Ibas,nbasis
                              !                                        call pteval(gridx,gridy,gridz,phi2,dphi2dx,dphi2dy, &
                                    !                                        dphi2dz,Jbas)
                              phi2=phixiao(Jbas)
                              dphi2dx=dphidxxiao(Jbas)
                              dphi2dy=dphidyxiao(Jbas)
                              dphi2dz=dphidzxiao(Jbas)
                              !                                        quicktest = DABS(dphi2dx)+DABS(dphi2dy)+DABS(dphi2dz) &
                                    !                                        +DABS(phi2)
                              !                                        if (quicktest < quick_method%DMCutoff ) then
                              !                                            continue
                              !                                        else
                              temp = phi*phi2
                              tempgx = phi*dphi2dx + phi2*dphidx
                              tempgy = phi*dphi2dy + phi2*dphidy
                              tempgz = phi*dphi2dz + phi2*dphidz
                              quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+ &
                                    xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                              !                                        endif
                           enddo
                        endif
                     enddo
                  endif
               endif
            enddo
         enddo
      enddo
   endif

   if(quick_method%MPW91PW91)then
      do Iatm=1,natom
         if(quick_method%ISG.eq.1)then
            Iradtemp=50
         else
            if(quick_molspec%iattype(iatm).le.10)then
               Iradtemp=23
            else
               Iradtemp=26
            endif
         endif

         do Irad=1,Iradtemp
            if(quick_method%ISG.eq.1)then
               call gridformnew(iatm,RGRID(Irad),iiangt)
               rad = radii(quick_molspec%iattype(iatm))
            else
               call gridformSG0(iatm,Iradtemp+1-Irad,iiangt,RGRID,RWT)
               rad = radii2(quick_molspec%iattype(iatm))
            endif

            rad3 = rad*rad*rad
            !             print*,iiangt
            do Iang=1,iiangt
               gridx=xyz(1,Iatm)+rad*RGRID(Irad)*XANG(Iang)
               gridy=xyz(2,Iatm)+rad*RGRID(Irad)*YANG(Iang)
               gridz=xyz(3,Iatm)+rad*RGRID(Irad)*ZANG(Iang)

               ! Next, calculate the weight of the grid point in the SSW scheme.  if
               ! the grid point has a zero weight, we can skip it.

               weight=SSW(gridx,gridy,gridz,Iatm) &
                     *WTANG(Iang)*RWT(Irad)*rad3

               if (weight < quick_method%DMCutoff ) then
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

                  if (density < quick_method%DMCutoff ) then
                     continue
                  else

                     call becke_E(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ex)

                     call lyp_e(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ec)

                     Eelxc = Eelxc + (param7*Ex+param8*Ec) &
                           *weight


                     quick_qm_struct%aelec = weight*density+quick_qm_struct%aelec
                     quick_qm_struct%belec = weight*densityb+quick_qm_struct%belec

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
                     ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))

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
                        if (quicktest < quick_method%DMCutoff ) then
                           continue
                        else
                           do Jbas=Ibas,nbasis
                              !                                        call pteval(gridx,gridy,gridz,phi2,dphi2dx,dphi2dy, &
                                    !                                        dphi2dz,Jbas)
                              phi2=phixiao(Jbas)
                              dphi2dx=dphidxxiao(Jbas)
                              dphi2dy=dphidyxiao(Jbas)
                              dphi2dz=dphidzxiao(Jbas)
                              !                                        quicktest = DABS(dphi2dx)+DABS(dphi2dy)+DABS(dphi2dz) &
                                    !                                        +DABS(phi2)
                              !                                        if (quicktest < quick_method%DMCutoff ) then
                              !                                            continue
                              !                                        else
                              temp = phi*phi2
                              tempgx = phi*dphi2dx + phi2*dphidx
                              tempgy = phi*dphi2dy + phi2*dphidy
                              tempgz = phi*dphi2dz + phi2*dphidz
                              quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+ &
                                    xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                              !                                        endif
                           enddo
                        endif
                     enddo
                  endif
               endif
            enddo
         enddo
      enddo
   endif



   ! Finally, copy lower diagonal to upper diagonal.


   do Ibas=1,nbasis
      do Jbas=Ibas+1,nbasis
         quick_qm_struct%o(Ibas,Jbas) = quick_qm_struct%o(Jbas,Ibas)
      enddo
   enddo

   call cpu_time(t3)

   write (ioutfile,'(" TIME of evaluation numerical integral = ",F12.2)') &
         T3-T2

   quick_qm_struct%Eel=quick_qm_struct%Eel+Eelxc

   !           write (ioutfile,'("TOTAL ENERGY OF CURRENT CYCLE=",F16.9)') Eel
end subroutine dftoperatordelta


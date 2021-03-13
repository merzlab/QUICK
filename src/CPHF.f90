#include "util.fh"

! Ed Brothers. November 14, 2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine formCPHFA(Ibas,Jbas,IIbas,JJbas)
  use allmod
  implicit double precision(a-h,o-z)

  dimension isame(4,8)
  logical :: same

  ! The purpose of the subroutine is to calculate an AO repulsion
  ! integral, determine it's contribution to an MO repulsion integral,
  ! and put it in the correct location in the CPHF A matrix.

  ! First, calculate the integral.

  iA = quick_basis%ncenter(Ibas)
  iB = quick_basis%ncenter(Jbas)
  iC = quick_basis%ncenter(IIbas)
  iD = quick_basis%ncenter(JJbas)
  AOint=0.d0
  do Icon=1,ncontract(Ibas)
     do Jcon=1,ncontract(Jbas)
        do IIcon=1,ncontract(IIbas)
           do JJcon=1,ncontract(JJbas)
              AOint = AOint + &
                   dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                   *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                   *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                   aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                   itype(1,Ibas), itype(2,Ibas), itype(3,Ibas), &
                   itype(1,Jbas), itype(2,Jbas), itype(3,Jbas), &
                   itype(1,IIbas),itype(2,IIbas),itype(3,IIbas), &
                   itype(1,JJbas),itype(2,JJbas),itype(3,JJbas), &
                   xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                   xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                   xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                   xyz(1,iD), xyz(2,iD),xyz(3,iD))

           enddo
        enddo
     enddo
  enddo
  
  ! Now we need to find how many times the AO integral appears by examining
  ! it's symmetry.  For an integral (ij|kl):
  ! (ij|kl)=(ji|kl)=(ij|lk)=(ji|lk)=(kl|ij)=(lk|ij)=(kl|ji)=(lk|ji)

  ! set up (ij|kl)
  isame(1,1)=Ibas
  isame(2,1)=Jbas
  isame(3,1)=IIbas
  isame(4,1)=JJbas
  ! set up (ji|kl)
  isame(1,2)=Jbas
  isame(2,2)=Ibas
  isame(3,2)=IIbas
  isame(4,2)=JJbas
  ! set up (ij|lk)
  isame(1,3)=Ibas
  isame(2,3)=Jbas
  isame(3,3)=JJbas
  isame(4,3)=IIbas
  ! set up (ji|lk)
  isame(1,4)=Jbas
  isame(2,4)=Ibas
  isame(3,4)=JJbas
  isame(4,4)=IIbas
  ! set up (kl|ij)
  isame(1,5)=IIbas
  isame(2,5)=JJbas
  isame(3,5)=Ibas
  isame(4,5)=Jbas
  ! set up (lk|ij)
  isame(1,6)=JJbas
  isame(2,6)=IIbas
  isame(3,6)=Ibas
  isame(4,6)=Jbas
  ! set up (kl|ji)
  isame(1,7)=IIbas
  isame(2,7)=JJbas
  isame(3,7)=Jbas
  isame(4,7)=Ibas
  ! set up (lk|ji)
  isame(1,8)=JJbas
  isame(2,8)=IIbas
  isame(3,8)=Jbas
  isame(4,8)=Ibas

  ! Now we check for redundancy.

  do Icheck=1,8
     if (isame(1,Icheck) /= 0) then
        do Jcheck=Icheck+1,8
           if (isame(1,Jcheck) /= 0) then
              same = isame(1,Icheck).eq.isame(1,Jcheck)
              same = same.and.isame(2,Icheck).eq.isame(2,Jcheck)
              same = same.and.isame(3,Icheck).eq.isame(3,Jcheck)
              same = same.and.isame(4,Icheck).eq.isame(4,Jcheck)
              if (same) then
                 do Iblank=1,4
                    isame(Iblank,Jcheck)=0
                 enddo
              endif
           endif
        enddo
     endif
  enddo

  ! Now we need to find out where the alpha and beta occupied/virtual
  ! lines are.

  if (quick_method%unrst) then
     lastAocc = quick_molspec%nelec
     lastBocc = quick_molspec%nelecb
  else
     lastAocc = quick_molspec%nelec/2
     lastBocc = lastAocc
  endif
  iBetastart = lastAocc*(nbasis-lastAocc)

  ! Now we can start filling up the CPHFA array.


  do Iunique=1,8
     if (isame(1,Iunique) /= 0) then

        ! Set up some dummy variables.

        Ival = isame(1,Iunique)
        Jval = isame(2,Iunique)
        IIval = isame(3,Iunique)
        JJval = isame(4,Iunique)

        ! Loop over alpha pairs.

        do iAvirt = lastAocc+1,nbasis
           do iAocc = 1,lastAocc

              ! iAvirt and iAocc form an ai pair.  Find it's location.

              iaCPHFA = (iAvirt-lastAocc-1)*lastAocc + iAocc

              ! Loop over alpha again.  This means we are filling the alpha-alpha
              ! portion of the matrix.

              do iAvirt2 = lastAocc+1,nbasis
                 do iAocc2 = 1,lastAocc

                    ! iAvirt2 and iAocc2 form an bj pair.  Find it's location.

                    jbCPHFA = (iAvirt2-lastAocc-1)*lastAocc + iAocc2

                    ! CPHFA(ai,jb) = 2(ai|jb) - (aj|bi) - (ab|ji)
                    ! Since all these are alpha, no elements drop out.
                    ! Calculate the value of the AO repulsions contribution to the MO repulsion.

                    quick_qm_struct%cphfa(iaCPHFA,jbCPHFA) = quick_qm_struct%cphfa(iaCPHFA,jbCPHFA) + &
                         2.d0*quick_qm_struct%co(Ival,IAvirt)*quick_qm_struct%co(Jval,IAocc)* &
                         quick_qm_struct%co(IIval,IAvirt2)*quick_qm_struct%co(JJval,iAocc2)*AOint &
                         -quick_qm_struct%co(Ival,IAvirt)*quick_qm_struct%co(Jval,iAocc2)* &
                         quick_qm_struct%co(IIval,IAvirt2)*quick_qm_struct%co(JJval,iAocc)*AOint &
                         -quick_qm_struct%co(Ival,IAvirt)*quick_qm_struct%co(Jval,iAvirt2)* &
                         quick_qm_struct%co(IIval,IAocc2)*quick_qm_struct%co(JJval,iAocc)*AOint

                 enddo
              enddo

              ! Now loop over beta for the bj pairs.

              do iBvirt2 = lastBocc+1,nbasis
                 do iBocc2 = 1,lastBocc

                    ! iBvirt2 and iBocc2 form an bj pair.  Find it's location.

                    jbCPHFA = (iBvirt2-lastBocc-1)*lastBocc +iBocc2 +iBetastart

                    ! CPHFA(ai,jb) = 2(ai|jb) - (aj|bi) - (ab|ji)
                    ! j and b are beta, thus it becomes:
                    ! CPHFA(ai,jb) = 2(ai|jb)
                    ! Calculate the value of the AO repulsions contribution to the MO repulsion.

                    quick_qm_struct%cphfa(iaCPHFA,jbCPHFA) = quick_qm_struct%cphfa(iaCPHFA,jbCPHFA) + &
                         2.d0*quick_qm_struct%co(Ival,IAvirt)*quick_qm_struct%co(Jval,IAocc)* &
                         quick_qm_struct%cob(IIval,IBvirt2)*quick_qm_struct%cob(JJval,iBocc2)*AOint

                 enddo
              enddo
           enddo
        enddo

        ! Loop over beta pairs.

        do iBvirt = lastBocc+1,nbasis
           do iBocc = 1,lastBocc

              ! iBvirt and iBocc form an ai pair.  Find it's location.

              iaCPHFA = (iBvirt-lastBocc-1)*lastBocc +iBocc +iBetastart

              ! Loop over beta again.  This means we are filling the beta-beta
              ! portion of the matrix.

              do iBvirt2 = lastBocc+1,nbasis
                 do iBocc2 = 1,lastBocc

                    ! iBvirt2 and iBocc2 form an bj pair.  Find it's location.

                    jbCPHFA = (iBvirt2-lastBocc-1)*lastBocc +iBocc2 +iBetastart

                    ! CPHFA(ai,jb) = 2(ai|jb) - (aj|bi) - (ab|ji)
                    ! Since all these are beta, no elements drop out.
                    ! Calculate the value of the AO repulsions contribution to the MO repulsion.

                    quick_qm_struct%cphfa(iaCPHFA,jbCPHFA) = quick_qm_struct%cphfa(iaCPHFA,jbCPHFA) + &
                         2.d0*quick_qm_struct%cob(Ival,IBvirt)*quick_qm_struct%cob(Jval,IBocc)* &
                         quick_qm_struct%cob(IIval,IBvirt2)*quick_qm_struct%cob(JJval,iBocc2)*AOint &
                         -quick_qm_struct%cob(Ival,IBvirt)*quick_qm_struct%cob(Jval,iBocc2)* &
                         quick_qm_struct%cob(IIval,IBvirt2)*quick_qm_struct%cob(JJval,iBocc)*AOint &
                         -quick_qm_struct%cob(Ival,IBvirt)*quick_qm_struct%cob(Jval,iBvirt2)* &
                         quick_qm_struct%cob(IIval,IBocc2)*quick_qm_struct%cob(JJval,iBocc)*AOint

                 enddo
              enddo

              ! Now loop over alpha for the bj pairs.

              do iAvirt2 = lastAocc+1,nbasis
                 do iAocc2 = 1,lastAocc

                    ! iAvirt2 and iAocc2 form an bj pair.  Find it's location.

                    jbCPHFA = (iAvirt2-lastAocc-1)*lastAocc + iAocc2

                    ! CPHFA(ai,jb) = 2(ai|jb) - (aj|bi) - (ab|ji)
                    ! j and b are alpha, thus it becomes:
                    ! CPHFA(ai,jb) = 2(ai|jb)
                    ! Calculate the value of the AO repulsions contribution to the MO repulsion.

                    quick_qm_struct%cphfa(iaCPHFA,jbCPHFA) = quick_qm_struct%cphfa(iaCPHFA,jbCPHFA) + &
                         2.d0*quick_qm_struct%cob(Ival,IBvirt)*quick_qm_struct%cob(Jval,IBocc)* &
                         quick_qm_struct%co(IIval,IAvirt2)*quick_qm_struct%co(JJval,iAocc2)*AOint

                 enddo
              enddo
           enddo
        enddo

     endif
  enddo



  return
end subroutine formCPHFA



! Ed Brothers. November 18, 2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine formCPHFB
  use allmod
  implicit double precision(a-h,o-z)
   double precision g_table(200)
   integer i,j,k,ii,jj,kk,g_count

  ! First some quick setup.

  if (quick_method%unrst) then
     lastAocc = quick_molspec%nelec
     lastBocc = quick_molspec%nelecb
  else
     lastAocc = quick_molspec%nelec/2
     lastBocc = lastAocc
  endif
  iBetastart = lastAocc*(nbasis-lastAocc)

  ! The purpose of this subroutine is to form B0 for the CPHF. There is
  ! one B0 for each nuclear perturbation, and this subroutine forms all
  ! of them.  The elements of B0(ai) = Qai(1)/(E(i)-E(a))

  ! Now some math for the B0 where the prreturbation is the y component
  ! of atom1.  (Y1)  This is similar for any perturbation.

  ! Qai(1) = Hai(1) - Sai(1) E(i) - (Sum over k and l) Skl(1) [(ai|lk)-(ak|li)]
  ! + (Sum over mu,nu,lambda,sigma) C(mu,a) C(nu,i) DENSE(lamba,sigma)
  ! * d/dY1 [(mu nu|lamba sigma) - (mu sigma| lamba nu)]

  ! Now Hai(1) = (Sum over mu,nu) C(mu,a) C(nu,i) Hmunu(1)
  ! = (Sum over mu,nu) C(mu,a) C(nu,i) d/dY1 H(mu,nu)

  ! And Sai(1) = (Sum over mu,nu) C(mu,a) C(nu,i) Smunu(1)
  ! = (Sum over mu,nu) C(mu,a) C(nu,i) d/dY1 S(mu,nu)

  ! We are going to calculate the first two terms: Hai(1) - Sai(1) E(i)

  do Ibas=1,nbasis
     ISTART = (quick_basis%ncenter(Ibas)-1) *3
     do Jbas=quick_basis%last_basis_function(quick_basis%ncenter(IBAS))+1,nbasis
        JSTART = (quick_basis%ncenter(Jbas)-1) *3

        ! We have selected our two basis functions, now loop over angular momentum.

        Ax = xyz(1,quick_basis%ncenter(Jbas))
        Bx = xyz(1,quick_basis%ncenter(Ibas))
        Ay = xyz(2,quick_basis%ncenter(Jbas))
        By = xyz(2,quick_basis%ncenter(Ibas))
        Az = xyz(3,quick_basis%ncenter(Jbas))
        Bz = xyz(3,quick_basis%ncenter(Ibas))
        
        do Imomentum=1,3
           dSI = 0.d0
           dSJ =0.d0
           dKEI = 0.d0
           dKEJ = 0.d0

           ! do the Ibas derivatives first.

           itype(Imomentum,Ibas) = itype(Imomentum,Ibas)+1
           ii = itype(1,Ibas)
           jj = itype(2,Ibas)
           kk = itype(3,Ibas)
           i = itype(1,Jbas)
           j = itype(2,Jbas)
           k = itype(3,Jbas)
           g_count = i+ii+j+jj+k+kk+2

           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)

                 b = aexp(Icon,Ibas)
                 a = aexp(Jcon,Jbas)
                 call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)      

                 dSI = dSI + 2.d0*b* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
            *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!                      *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 dKEI = dKEI + 2.d0*b* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *ekinetic(a,b,i ,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) 
!                      *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
              enddo
           enddo
           itype(Imomentum,Ibas) = itype(Imomentum,Ibas)-1
           if (itype(Imomentum,Ibas) /= 0) then
              itype(Imomentum,Ibas) = itype(Imomentum,Ibas)-1
              ii = itype(1,Ibas)
              jj = itype(2,Ibas)
              kk = itype(3,Ibas)
              i = itype(1,Jbas)
              j = itype(2,Jbas)
              k = itype(3,Jbas)
              g_count = i+ii+j+jj+k+kk+2

              do Icon=1,ncontract(Ibas)
                 do Jcon=1,ncontract(Jbas)
                    b = aexp(Icon,Ibas)
                    a = aexp(Jcon,Jbas)
                    call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)      
                    dSI = dSI - dble(itype(Imomentum,Ibas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
            *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!                         *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                    dKEI = dKEI - dble(itype(Imomentum,Ibas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *ekinetic(a,b,i ,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) 
!                         *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 enddo
              enddo
              itype(Imomentum,Ibas) = itype(Imomentum,Ibas)+1
           endif

           ! Now do the Jbas derivatives.

           itype(Imomentum,Jbas) = itype(Imomentum,Jbas)+1
           ii = itype(1,Ibas)
           jj = itype(2,Ibas)
           kk = itype(3,Ibas)
           i = itype(1,Jbas)
           j = itype(2,Jbas)
           k = itype(3,Jbas)
           g_count = i+ii+j+jj+k+kk+2

           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 b = aexp(Icon,Ibas)
                 a = aexp(Jcon,Jbas)
                 call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)      

                 dSJ = dSJ + 2.d0*a* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
            *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!                      *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 dKEJ = dKEJ + 2.d0*a* &
                      dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *ekinetic(a,b,i ,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) 
!                      *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                      itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                      itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                      xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                      xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                      xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
              enddo
           enddo
           itype(Imomentum,Jbas) = itype(Imomentum,Jbas)-1
           if (itype(Imomentum,Jbas) /= 0) then
              itype(Imomentum,Jbas) = itype(Imomentum,Jbas)-1
              ii = itype(1,Ibas)
              jj = itype(2,Ibas)
              kk = itype(3,Ibas)
              i = itype(1,Jbas)
              j = itype(2,Jbas)
              k = itype(3,Jbas)
              g_count = i+ii+j+jj+k+kk+2

              do Icon=1,ncontract(Ibas)
                 do Jcon=1,ncontract(Jbas)
                    b = aexp(Icon,Ibas)
                    a = aexp(Jcon,Jbas)
                    call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)      
                    dSJ = dSJ - dble(itype(Imomentum,Jbas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
            *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!                         *overlap(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                    dKEJ = dKEJ - dble(itype(Imomentum,Jbas)+1)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *ekinetic(a,b,i ,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) 
!                         *ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                         xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                         xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                         xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                 enddo
              enddo
              itype(Imomentum,Jbas) = itype(Imomentum,Jbas)+1
           endif

           ! At this point we have the derivatives.  Now we need to put them in
           ! the CPHFB array.
           ! ALPHA first.

           do iAvirt = lastAocc+1,nbasis
              do iAocc = 1,lastAocc
                 iaCPHF = (iAvirt-lastAocc-1)*lastAocc + iAocc
                 quick_qm_struct%cphfb(iaCPHF,ISTART+Imomentum) = &
                      quick_qm_struct%cphfb(iaCPHF,ISTART+Imomentum) &
                      +dKEI*(quick_qm_struct%co(Ibas,iAvirt)*quick_qm_struct%co(Jbas,iAocc) &
                      +quick_qm_struct%co(Ibas,iAocc)*quick_qm_struct%co(Jbas,iAvirt)) &
                      -dSI* (quick_qm_struct%co(Ibas,iAvirt)*quick_qm_struct%co(Jbas,iAocc) &
                      +quick_qm_struct%co(Ibas,iAocc)*quick_qm_struct%co(Jbas,iAvirt))*quick_qm_struct%E(iAocc)
                 quick_qm_struct%cphfb(iaCPHF,JSTART+Imomentum) = &
                      quick_qm_struct%cphfb(iaCPHF,JSTART+Imomentum) &
                      +dKEJ*(quick_qm_struct%co(Jbas,iAvirt)*quick_qm_struct%co(Ibas,iAocc) &
                      +quick_qm_struct%co(Jbas,iAocc)*quick_qm_struct%co(Ibas,iAvirt)) &
                      -dSJ* (quick_qm_struct%co(Jbas,iAvirt)*quick_qm_struct%co(Ibas,iAocc) &
                      +quick_qm_struct%co(Jbas,iAocc)*quick_qm_struct%co(Ibas,iAvirt))*quick_qm_struct%E(iAocc)
              enddo
           enddo
           ! BETA
           do iBvirt = lastBocc+1,nbasis
              do iBocc = 1,lastBocc
                 iaCPHF = (iBvirt-lastBocc-1)*lastBocc + iBocc +iBetaStart
                 quick_qm_struct%cphfb(iaCPHF,ISTART+Imomentum) = &
                      quick_qm_struct%cphfb(iaCPHF,ISTART+Imomentum) &
                      +dKEI*(quick_qm_struct%cob(Ibas,iBvirt)*quick_qm_struct%cob(Jbas,iBocc) &
                      +quick_qm_struct%cob(Ibas,iBocc)*quick_qm_struct%cob(Jbas,iBvirt)) &
                      -dSI* (quick_qm_struct%cob(Ibas,iBvirt)*quick_qm_struct%cob(Jbas,iBocc) &
                      +quick_qm_struct%cob(Ibas,iBocc)*quick_qm_struct%cob(Jbas,iBvirt))*quick_qm_struct%EB(iBocc)
                 quick_qm_struct%cphfb(iaCPHF,JSTART+Imomentum) = &
                      quick_qm_struct%cphfb(iaCPHF,JSTART+Imomentum) &
                      +dKEJ*(quick_qm_struct%cob(Jbas,iBvirt)*quick_qm_struct%cob(Ibas,iBocc) &
                      +quick_qm_struct%cob(Jbas,iBocc)*quick_qm_struct%cob(Ibas,iBvirt)) &
                      -dSJ* (quick_qm_struct%cob(Jbas,iBvirt)*quick_qm_struct%cob(Ibas,iBocc) &
                      +quick_qm_struct%cob(Jbas,iBocc)*quick_qm_struct%cob(Ibas,iBvirt))*quick_qm_struct%EB(iBocc)
              enddo
           enddo
        enddo
     enddo
  enddo


  ! We still have to calculate the three center term that arises in the
  ! core Hamiltonian.

  do Ibas=1,nbasis
     iA=quick_basis%ncenter(Ibas)
     ISTART = (iA-1)*3

     do Jbas=Ibas,nbasis
        iB = quick_basis%ncenter(Jbas)
        JSTART = (iB-1)*3

        do iC = 1,natom
           iCSTART = (iC-1)*3

           ! As before, if all terms are on the same atom, they move with the
           ! atom and the derivative is zero.

           if (iA == iC .and. iB == iC) then
              continue
           else


              ! If Ibas=Jbas, the term only shows up once in the energy, otherwise
              ! it shows up twice. This is not an issue with the 2-center terms above.


              dNAIX = 0.d0
              dNAIY = 0.d0
              dNAIZ = 0.d0
              dNAJX = 0.d0
              dNAJY = 0.d0
              dNAJZ = 0.d0
              dNACX = 0.d0
              dNACY = 0.d0
              dNACZ = 0.d0

              ! do the Ibas derivatives.

              do Icon=1,ncontract(Ibas)
                 do Jcon=1,ncontract(Jbas)
                    dNAIX = dNAIX + 2.d0*aexp(Icon,Ibas)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas)+1,itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                         xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                         xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    dNAIY = dNAIY + 2.d0*aexp(Icon,Ibas)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas)+1,itype(3,Ibas), &
                         xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                         xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                         xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    dNAIZ = dNAIZ + 2.d0*aexp(Icon,Ibas)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas)+1, &
                         xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                         xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                         xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))

                 enddo
              enddo


              if (itype(1,Ibas) /= 0) then
                 do Icon=1,ncontract(Ibas)
                    do Jcon=1,ncontract(Jbas)
                       dNAIX= dNAIX - dble(itype(1,Ibas))* &
                            dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                            itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                            itype(1,Ibas)-1,itype(2,Ibas),itype(3,Ibas), &
                            xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                            xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                            xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    enddo
                 enddo
              endif
              if (itype(2,Ibas) /= 0) then
                 do Icon=1,ncontract(Ibas)
                    do Jcon=1,ncontract(Jbas)
                       dNAIY= dNAIY - dble(itype(2,Ibas))* &
                            dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                            itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                            itype(1,Ibas),itype(2,Ibas)-1,itype(3,Ibas), &
                            xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                            xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                            xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    enddo
                 enddo
              endif
              if (itype(3,Ibas) /= 0) then
                 do Icon=1,ncontract(Ibas)
                    do Jcon=1,ncontract(Jbas)
                       dNAIZ= dNAIZ - dble(itype(3,Ibas))* &
                            dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                            itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                            itype(1,Ibas),itype(2,Ibas),itype(3,Ibas)-1, &
                            xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                            xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                            xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    enddo
                 enddo
              endif

              ! do the Jbas derivatives.

              do Icon=1,ncontract(Ibas)
                 do Jcon=1,ncontract(Jbas)
                    dNAJX = dNAJX + 2.d0*aexp(Jcon,Jbas)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas)+1,itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                         xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                         xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    dNAJY = dNAJY + 2.d0*aexp(Jcon,Jbas)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas)+1,itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                         xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                         xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    dNAJZ = dNAJZ + 2.d0*aexp(Jcon,Jbas)* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas)+1, &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                         xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                         xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                 enddo
              enddo

              if (itype(1,Jbas) /= 0) then
                 do Icon=1,ncontract(Ibas)
                    do Jcon=1,ncontract(Jbas)
                       dNAJX= dNAJX - dble(itype(1,Jbas))* &
                            dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                            itype(1,Jbas)-1,itype(2,Jbas),itype(3,Jbas), &
                            itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                            xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                            xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                            xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    enddo
                 enddo
              endif
              if (itype(2,Jbas) /= 0) then
                 do Icon=1,ncontract(Ibas)
                    do Jcon=1,ncontract(Jbas)
                       dNAJY= dNAJY - dble(itype(2,Jbas))* &
                            dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                            itype(1,Jbas),itype(2,Jbas)-1,itype(3,Jbas), &
                            itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                            xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                            xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                            xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    enddo
                 enddo
              endif
              if (itype(3,Jbas) /= 0) then
                 do Icon=1,ncontract(Ibas)
                    do Jcon=1,ncontract(Jbas)
                       dNAJZ= dNAJZ - dble(itype(3,Jbas))* &
                            dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                            itype(1,Jbas),itype(2,Jbas),itype(3,Jbas)-1, &
                            itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                            xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                            xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                            xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    enddo
                 enddo
              endif

              ! Now do the derivative with respect to the atom the basis functions
              ! are attracted to.

              do Icon=1,ncontract(Ibas)
                 do Jcon=1,ncontract(Jbas)
                    dNACX= dNACX + dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *electricfld(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         1,0,0, &
                         xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                         xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                         xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    dNACY= dNACY + dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *electricfld(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         0,1,0, &
                         xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                         xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                         xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                    dNACZ= dNACZ + dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *electricfld(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                         itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                         itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                         0,0,1, &
                         xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                         xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                         xyz(1,iC),xyz(2,iC),xyz(3,iC), quick_molspec%chg(iC))
                 enddo
              enddo

              ! Now add these into the CPHFB array.

              do iAvirt = lastAocc+1,nbasis
                 do iAocc = 1,lastAocc
                    iaCPHF = (iAvirt-lastAocc-1)*lastAocc + iAocc
                    quick_qm_struct%cphfb(iaCPHF,ISTART+1) = quick_qm_struct%cphfb(iaCPHF,ISTART+1) &
                         +dNAIX*(quick_qm_struct%co(Ibas,iAvirt)*quick_qm_struct%co(Jbas,iAocc))
                    quick_qm_struct%cphfb(iaCPHF,ISTART+2) = quick_qm_struct%cphfb(iaCPHF,ISTART+2) &
                         +dNAIY*(quick_qm_struct%co(Ibas,iAvirt)*quick_qm_struct%co(Jbas,iAocc))
                    quick_qm_struct%cphfb(iaCPHF,ISTART+3) = quick_qm_struct%cphfb(iaCPHF,ISTART+3) &
                         +dNAIZ*(quick_qm_struct%co(Ibas,iAvirt)*quick_qm_struct%co(Jbas,iAocc))
                    quick_qm_struct%cphfb(iaCPHF,JSTART+1) = quick_qm_struct%cphfb(iaCPHF,JSTART+1) &
                         +dNAJX*(quick_qm_struct%co(Ibas,iAvirt)*quick_qm_struct%co(Jbas,iAocc))
                    quick_qm_struct%cphfb(iaCPHF,JSTART+2) = quick_qm_struct%cphfb(iaCPHF,JSTART+2) &
                         +dNAJY*(quick_qm_struct%co(Ibas,iAvirt)*quick_qm_struct%co(Jbas,iAocc))
                    quick_qm_struct%cphfb(iaCPHF,JSTART+3) = quick_qm_struct%cphfb(iaCPHF,JSTART+3) &
                         +dNAJZ*(quick_qm_struct%co(Ibas,iAvirt)*quick_qm_struct%co(Jbas,iAocc))
                    quick_qm_struct%cphfb(iaCPHF,ICSTART+1) = quick_qm_struct%cphfb(iaCPHF,ICSTART+1) &
                         +dNACX*(quick_qm_struct%co(Ibas,iAvirt)*quick_qm_struct%co(Jbas,iAocc))
                    quick_qm_struct%cphfb(iaCPHF,ICSTART+2) = quick_qm_struct%cphfb(iaCPHF,ICSTART+2) &
                         +dNACY*(quick_qm_struct%co(Ibas,iAvirt)*quick_qm_struct%co(Jbas,iAocc))
                    quick_qm_struct%cphfb(iaCPHF,ICSTART+3) = quick_qm_struct%cphfb(iaCPHF,ICSTART+3) &
                         +dNACZ*(quick_qm_struct%co(Ibas,iAvirt)*quick_qm_struct%co(Jbas,iAocc))
                    if (Ibas /= Jbas) then
                       quick_qm_struct%cphfb(iaCPHF,ISTART+1) = quick_qm_struct%cphfb(iaCPHF,ISTART+1) &
                            +dNAIX*(quick_qm_struct%co(Jbas,iAvirt)*quick_qm_struct%co(Ibas,iAocc))
                       quick_qm_struct%cphfb(iaCPHF,ISTART+2) = quick_qm_struct%cphfb(iaCPHF,ISTART+2) &
                            +dNAIY*(quick_qm_struct%co(Jbas,iAvirt)*quick_qm_struct%co(Ibas,iAocc))
                       quick_qm_struct%cphfb(iaCPHF,ISTART+3) = quick_qm_struct%cphfb(iaCPHF,ISTART+3) &
                            +dNAIZ*(quick_qm_struct%co(Jbas,iAvirt)*quick_qm_struct%co(Ibas,iAocc))
                       quick_qm_struct%cphfb(iaCPHF,JSTART+1) = quick_qm_struct%cphfb(iaCPHF,JSTART+1) &
                            +dNAJX*(quick_qm_struct%co(Jbas,iAvirt)*quick_qm_struct%co(Ibas,iAocc))
                       quick_qm_struct%cphfb(iaCPHF,JSTART+2) = quick_qm_struct%cphfb(iaCPHF,JSTART+2) &
                            +dNAJY*(quick_qm_struct%co(Jbas,iAvirt)*quick_qm_struct%co(Ibas,iAocc))
                       quick_qm_struct%cphfb(iaCPHF,JSTART+3) = quick_qm_struct%cphfb(iaCPHF,JSTART+3) &
                            +dNAJZ*(quick_qm_struct%co(Jbas,iAvirt)*quick_qm_struct%co(Ibas,iAocc))
                       quick_qm_struct%cphfb(iaCPHF,ICSTART+1) = quick_qm_struct%cphfb(iaCPHF,ICSTART+1) &
                            +dNACX*(quick_qm_struct%co(Jbas,iAvirt)*quick_qm_struct%co(Ibas,iAocc))
                       quick_qm_struct%cphfb(iaCPHF,ICSTART+2) = quick_qm_struct%cphfb(iaCPHF,ICSTART+2) &
                            +dNACY*(quick_qm_struct%co(Jbas,iAvirt)*quick_qm_struct%co(Ibas,iAocc))
                       quick_qm_struct%cphfb(iaCPHF,ICSTART+3) = quick_qm_struct%cphfb(iaCPHF,ICSTART+3) &
                            +dNACZ*(quick_qm_struct%co(Jbas,iAvirt)*quick_qm_struct%co(Ibas,iAocc))
                    endif
                 enddo
              enddo
              ! BETA
              do iBvirt = lastBocc+1,nbasis
                 do iBocc = 1,lastBocc
                    iaCPHF = (iBvirt-lastBocc-1)*lastBocc + iBocc + IbetaSTART
                    quick_qm_struct%cphfb(iaCPHF,ISTART+1) = quick_qm_struct%cphfb(iaCPHF,ISTART+1) &
                         +dNAIX*(quick_qm_struct%cob(Ibas,iBvirt)*quick_qm_struct%cob(Jbas,iBocc))
                    quick_qm_struct%cphfb(iaCPHF,ISTART+2) = quick_qm_struct%cphfb(iaCPHF,ISTART+2) &
                         +dNAIY*(quick_qm_struct%cob(Ibas,iBvirt)*quick_qm_struct%cob(Jbas,iBocc))
                    quick_qm_struct%cphfb(iaCPHF,ISTART+3) = quick_qm_struct%cphfb(iaCPHF,ISTART+3) &
                         +dNAIZ*(quick_qm_struct%cob(Ibas,iBvirt)*quick_qm_struct%cob(Jbas,iBocc))
                    quick_qm_struct%cphfb(iaCPHF,JSTART+1) = quick_qm_struct%cphfb(iaCPHF,JSTART+1) &
                         +dNAJX*(quick_qm_struct%cob(Ibas,iBvirt)*quick_qm_struct%cob(Jbas,iBocc))
                    quick_qm_struct%cphfb(iaCPHF,JSTART+2) = quick_qm_struct%cphfb(iaCPHF,JSTART+2) &
                         +dNAJY*(quick_qm_struct%cob(Ibas,iBvirt)*quick_qm_struct%cob(Jbas,iBocc))
                    quick_qm_struct%cphfb(iaCPHF,JSTART+3) = quick_qm_struct%cphfb(iaCPHF,JSTART+3) &
                         +dNAJZ*(quick_qm_struct%cob(Ibas,iBvirt)*quick_qm_struct%cob(Jbas,iBocc))
                    quick_qm_struct%cphfb(iaCPHF,ICSTART+1) = quick_qm_struct%cphfb(iaCPHF,ICSTART+1) &
                         +dNACX*(quick_qm_struct%cob(Ibas,iBvirt)*quick_qm_struct%cob(Jbas,iBocc))
                    quick_qm_struct%cphfb(iaCPHF,ICSTART+2) = quick_qm_struct%cphfb(iaCPHF,ICSTART+2) &
                         +dNACY*(quick_qm_struct%cob(Ibas,iBvirt)*quick_qm_struct%cob(Jbas,iBocc))
                    quick_qm_struct%cphfb(iaCPHF,ICSTART+3) = quick_qm_struct%cphfb(iaCPHF,ICSTART+3) &
                         +dNACZ*(quick_qm_struct%cob(Ibas,iBvirt)*quick_qm_struct%cob(Jbas,iBocc))
                    if (Ibas /= Jbas) then
                       quick_qm_struct%cphfb(iaCPHF,ISTART+1) = quick_qm_struct%cphfb(iaCPHF,ISTART+1) &
                            +dNAIX*(quick_qm_struct%cob(Jbas,iBvirt)*quick_qm_struct%cob(Ibas,iBocc))
                       quick_qm_struct%cphfb(iaCPHF,ISTART+2) = quick_qm_struct%cphfb(iaCPHF,ISTART+2) &
                            +dNAIY*(quick_qm_struct%cob(Jbas,iBvirt)*quick_qm_struct%cob(Ibas,iBocc))
                       quick_qm_struct%cphfb(iaCPHF,ISTART+3) = quick_qm_struct%cphfb(iaCPHF,ISTART+3) &
                            +dNAIZ*(quick_qm_struct%cob(Jbas,iBvirt)*quick_qm_struct%cob(Ibas,iBocc))
                       quick_qm_struct%cphfb(iaCPHF,JSTART+1) = quick_qm_struct%cphfb(iaCPHF,JSTART+1) &
                            +dNAJX*(quick_qm_struct%cob(Jbas,iBvirt)*quick_qm_struct%cob(Ibas,iBocc))
                       quick_qm_struct%cphfb(iaCPHF,JSTART+2) = quick_qm_struct%cphfb(iaCPHF,JSTART+2) &
                            +dNAJY*(quick_qm_struct%cob(Jbas,iBvirt)*quick_qm_struct%cob(Ibas,iBocc))
                       quick_qm_struct%cphfb(iaCPHF,JSTART+3) = quick_qm_struct%cphfb(iaCPHF,JSTART+3) &
                            +dNAJZ*(quick_qm_struct%cob(Jbas,iBvirt)*quick_qm_struct%cob(Ibas,iBocc))
                       quick_qm_struct%cphfb(iaCPHF,ICSTART+1) = quick_qm_struct%cphfb(iaCPHF,ICSTART+1) &
                            +dNACX*(quick_qm_struct%cob(Jbas,iBvirt)*quick_qm_struct%cob(Ibas,iBocc))
                       quick_qm_struct%cphfb(iaCPHF,ICSTART+2) = quick_qm_struct%cphfb(iaCPHF,ICSTART+2) &
                            +dNACY*(quick_qm_struct%cob(Jbas,iBvirt)*quick_qm_struct%cob(Ibas,iBocc))
                       quick_qm_struct%cphfb(iaCPHF,ICSTART+3) = quick_qm_struct%cphfb(iaCPHF,ICSTART+3) &
                            +dNACZ*(quick_qm_struct%cob(Jbas,iBvirt)*quick_qm_struct%cob(Ibas,iBocc))
                    endif
                 enddo
              enddo

           endif
        enddo
     enddo
  enddo


  ! We've now done all the two and three center terms.  Now we need to
  ! add the 4 center terms into the B0 array.  These are:
  ! - (Sum over k and l) Skl(1) [(ai|lk)-(ak|li)]
  ! + (Sum over mu,nu,lambda,sigma) C(mu,a) C(nu,i) DENSE(lamba,sigma)
  ! * d/dY1 [(mu nu|lamba sigma) - (mu sigma| lamba nu)]
  ! We'll do this in a subprogram which is called for each unique integral.


  do I=1,nbasis
     call CPHFB4cnt(I,I,I,I)
     do J=I+1,nbasis
        call CPHFB4cnt(I,I,J,J)
        call CPHFB4cnt(I,J,J,J)
        call CPHFB4cnt(I,I,I,J)
        call CPHFB4cnt(I,J,I,J)
        do K=J+1,nbasis
           call CPHFB4cnt(I,J,I,K)
           call CPHFB4cnt(I,J,K,K)
           call CPHFB4cnt(I,K,J,J)
           call CPHFB4cnt(I,I,J,K)
        enddo
        do K=I+1,nbasis-1
           do L=K+1,nbasis
              call CPHFB4cnt(I,J,K,L)
           enddo
        enddo
     enddo
  enddo

  ! Now we need to go through and divide by (Ei-Ea).


  do iAvirt = lastAocc+1,nbasis
     do iAocc = 1,lastAocc
        iaCPHF = (iAvirt-lastAocc-1)*lastAocc + iAocc
        denom = quick_qm_struct%E(iAocc)-quick_qm_struct%E(iAvirt)
        do IDX = 1,natom*3
           quick_qm_struct%cphfb(iaCPHF,IDX) = quick_qm_struct%cphfb(iaCPHF,idX)/denom
        enddo
     enddo
  enddo

  do iBvirt = lastBocc+1,nbasis
     do iBocc = 1,lastBocc
        iaCPHF = (iBvirt-lastBocc-1)*lastBocc + iBocc+iBetastart
        denom = quick_qm_struct%EB(iBocc)-quick_qm_struct%EB(iBvirt)
        do IDX = 1,natom*3
           quick_qm_struct%cphfb(iaCPHF,IDX) = quick_qm_struct%cphfb(iaCPHF,IDX)/denom
        enddo
     enddo
  enddo

  ! Now the B0 array is complete.

end subroutine formcphfb

! Ed Brothers. November 18, 2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine CPHFB4cnt(Ibas,Jbas,IIbas,JJbas)
  use allmod
  implicit double precision(a-h,o-z)

  dimension isame(4,8),deriv(4,3),icenter(4)
  logical :: same
   double precision g_table(200)
   integer i,j,k,ii,jj,kk,g_count

  ! The purpose of the subroutine is to calculate an AO repulsion
  ! integral, determine it's contribution to an MO repulsion integral,
  ! and put it in the correct location in the CPHFB array, and then do the
  ! same with the integrals derivatives.

  ! First, calculate the integral.

  iA = quick_basis%ncenter(Ibas)
  iB = quick_basis%ncenter(Jbas)
  iC = quick_basis%ncenter(IIbas)
  iD = quick_basis%ncenter(JJbas)
  icenter(1)=iA
  icenter(2)=iB
  icenter(3)=iC
  icenter(4)=iD

  AOint=0.d0
  do Icon=1,ncontract(Ibas)
     do Jcon=1,ncontract(Jbas)
        do IIcon=1,ncontract(IIbas)
           do JJcon=1,ncontract(JJbas)
              AOint = AOint + &
                   dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                   *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                   *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                   aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                   itype(1,Ibas), itype(2,Ibas), itype(3,Ibas), &
                   itype(1,Jbas), itype(2,Jbas), itype(3,Jbas), &
                   itype(1,IIbas),itype(2,IIbas),itype(3,IIbas), &
                   itype(1,JJbas),itype(2,JJbas),itype(3,JJbas), &
                   xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                   xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                   xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                   xyz(1,iD), xyz(2,iD),xyz(3,iD))
           enddo
        enddo
     enddo
  enddo

  ! Now we need to find how many times the AO integral appears by examining
  ! it's symmetry.  For an integral (ij|kl):
  ! (ij|kl)=(ji|kl)=(ij|lk)=(ji|lk)=(kl|ij)=(lk|ij)=(kl|ji)=(lk|ji)

  ! set up (ij|kl)
  isame(1,1)=Ibas
  isame(2,1)=Jbas
  isame(3,1)=IIbas
  isame(4,1)=JJbas
  ! set up (ji|kl)
  isame(1,2)=Jbas
  isame(2,2)=Ibas
  isame(3,2)=IIbas
  isame(4,2)=JJbas
  ! set up (ij|lk)
  isame(1,3)=Ibas
  isame(2,3)=Jbas
  isame(3,3)=JJbas
  isame(4,3)=IIbas
  ! set up (ji|lk)
  isame(1,4)=Jbas
  isame(2,4)=Ibas
  isame(3,4)=JJbas
  isame(4,4)=IIbas
  ! set up (kl|ij)
  isame(1,5)=IIbas
  isame(2,5)=JJbas
  isame(3,5)=Ibas
  isame(4,5)=Jbas
  ! set up (lk|ij)
  isame(1,6)=JJbas
  isame(2,6)=IIbas
  isame(3,6)=Ibas
  isame(4,6)=Jbas
  ! set up (kl|ji)
  isame(1,7)=IIbas
  isame(2,7)=JJbas
  isame(3,7)=Jbas
  isame(4,7)=Ibas
  ! set up (lk|ji)
  isame(1,8)=JJbas
  isame(2,8)=IIbas
  isame(3,8)=Jbas
  isame(4,8)=Ibas

  ! Now we check for redundancy.

  do Icheck=1,8
     if (isame(1,Icheck) /= 0) then
        do Jcheck=Icheck+1,8
           if (isame(1,Jcheck) /= 0) then
              same = isame(1,Icheck).eq.isame(1,Jcheck)
              same = same.and.isame(2,Icheck).eq.isame(2,Jcheck)
              same = same.and.isame(3,Icheck).eq.isame(3,Jcheck)
              same = same.and.isame(4,Icheck).eq.isame(4,Jcheck)
              if (same) then
                 do Iblank=1,4
                    isame(Iblank,Jcheck)=0
                 enddo
              endif
           endif
        enddo
     endif
  enddo

  ! Now we need to find out where the alpha and beta occupied/virtual
  ! lines are.

  if (quick_method%unrst) then
     lastAocc = quick_molspec%nelec
     lastBocc = quick_molspec%nelecb
  else
     lastAocc = quick_molspec%nelec/2
     lastBocc = lastAocc
  endif
  iBetastart = lastAocc*(nbasis-lastAocc)

  ! Now we can start filling up the CPHFB array.
  ! Note we are first doing the term:
  ! - (Sum over k and l) Skl(1) [(ai|lk)-(ak|li)]

  ! Note k and l are both occupied occupied molecular orbitals.

  ! do the alpha first.

  do iAocc = 1,lastAocc
     do iAocc2 = 1,lastAocc

        ! K and L are selected.  Find atom and direction of perturbation.


        do Iatom = 1,natom
           do Imomentum=1,3

              ! Now we loop over basis functions.  Note that Kbas functions are always
              ! on center Katom, and the Lbas functions are always not on that center.
              ! This actually calculates the Skl

              Skl  = 0.d0
              do Kbas = quick_basis%first_basis_function(Iatom),quick_basis%last_basis_function(Iatom)
                 do Lbas = 1,nbasis
                    if (Lbas < quick_basis%first_basis_function(Iatom) .OR. Lbas > quick_basis%last_basis_function(Iatom)) then
                       dSK=0.d0

                       Ax = xyz(1,quick_basis%ncenter(Lbas))
                       Bx = xyz(1,quick_basis%ncenter(Kbas))
                       Ay = xyz(2,quick_basis%ncenter(Lbas))
                       By = xyz(2,quick_basis%ncenter(Kbas))
                       Az = xyz(3,quick_basis%ncenter(Lbas))
                       Bz = xyz(3,quick_basis%ncenter(Kbas))

                       itype(Imomentum,Kbas) = itype(Imomentum,Kbas)+1

                       ii = itype(1,Kbas)
                       jj = itype(2,Kbas)
                       kk = itype(3,Kbas)
                       i = itype(1,Lbas)
                       j = itype(2,Lbas)
                       k = itype(3,Lbas)
                       g_count = i+ii+j+jj+k+kk

                       do Kcon=1,ncontract(Kbas)
                          do Lcon=1,ncontract(Lbas)
                             b = aexp(Kcon,Kbas)
                             a = aexp(Lcon,Lbas)
                             call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)
                             dSK = dSK + 2.d0*b* &
                                  dcoeff(Lcon,Lbas)*dcoeff(Kcon,Kbas) &
            *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!                                  *overlap(aexp(Lcon,Lbas),aexp(Kcon,Kbas), &
!                                  itype(1,Lbas),itype(2,Lbas),itype(3,Lbas), &
!                                  itype(1,Kbas),itype(2,Kbas),itype(3,Kbas), &
!                                  xyz(1,quick_basis%ncenter(Lbas)),xyz(2,quick_basis%ncenter(Lbas)), &

!                                  xyz(3,quick_basis%ncenter(Lbas)),xyz(1,quick_basis%ncenter(Kbas)), &
!                                  xyz(2,quick_basis%ncenter(Kbas)),xyz(3,quick_basis%ncenter(Kbas)))
                          enddo
                       enddo
                       itype(Imomentum,Kbas) = itype(Imomentum,Kbas)-1
                       if (itype(Imomentum,Kbas) /= 0) then
                          itype(Imomentum,Kbas) = itype(Imomentum,Kbas)-1
                          ii = itype(1,Kbas)
                          jj = itype(2,Kbas)
                          kk = itype(3,Kbas)
                          i = itype(1,Lbas)
                          j = itype(2,Lbas)
                          k = itype(3,Lbas)
                          g_count = i+ii+j+jj+k+kk

                          do Kcon=1,ncontract(Kbas)
                             do Lcon=1,ncontract(Lbas)
                                b = aexp(Kcon,Kbas)
                                a = aexp(Lcon,Lbas)
                                call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)
                                dSK = dSK - dble(itype(Imomentum,Kbas)+1)* &
                                     dcoeff(Lcon,Lbas)*dcoeff(Kcon,Kbas) &
            *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!                                     *overlap(aexp(Lcon,Lbas),aexp(Kcon,Kbas), &
!                                     itype(1,Lbas),itype(2,Lbas),itype(3,Lbas), &
!                                     itype(1,Kbas),itype(2,Kbas),itype(3,Kbas), &
!                                     xyz(1,quick_basis%ncenter(Lbas)),xyz(2,quick_basis%ncenter(Lbas)), &
!                                     xyz(3,quick_basis%ncenter(Lbas)),xyz(1,quick_basis%ncenter(Kbas)), &
!                                     xyz(2,quick_basis%ncenter(Kbas)),xyz(3,quick_basis%ncenter(Kbas)))
                             enddo
                          enddo
                          itype(Imomentum,Kbas) = itype(Imomentum,Kbas)+1
                       endif
                       Skl=Skl+dSK*(quick_qm_struct%co(Kbas,iAocc)*quick_qm_struct%co(Lbas,iAocc2) + &
                            quick_qm_struct%co(Lbas,iAocc)*quick_qm_struct%co(Kbas,iAocc2))
                    endif
                 enddo
              enddo

              ! At this point we have the SKl value.  Now we need to loop over
              ! unique AO repulsions and and add the values into the array.
              ! we also need to find the location in the array defined by the Iatom
              ! Imomentum pair.

              ISTART=(Iatom-1)*3+Imomentum
              do Iunique=1,8
                 if (isame(1,Iunique) /= 0) then
                    ! Set up some dummy variables.

                    Ival = isame(1,Iunique)
                    Jval = isame(2,Iunique)
                    IIval = isame(3,Iunique)
                    JJval = isame(4,Iunique)

                    ! Loop over alpha pairs.

                    do iAvirt = lastAocc+1,nbasis
                       do iAocc3 = 1,lastAocc
                          iaCPHF = (iAvirt-lastAocc-1)*lastAocc + iAocc3
                          quick_qm_struct%cphfb(iaCPHF,Istart) = quick_qm_struct%cphfb(iaCPHF,Istart) &
                               -Skl*AOint*(quick_qm_struct%co(Ival,iAvirt)*quick_qm_struct%co(Jval,iAocc3)* &
                               quick_qm_struct%co(IIval,iAocc2)*quick_qm_struct%co(JJval,iAocc)-quick_qm_struct%co(Ival,iAvirt)* &
                               quick_qm_struct%co(Jval,iAocc)*quick_qm_struct%co(IIval,iAocc2)*quick_qm_struct%co(JJval,iAocc3))

                       enddo
                    enddo

                    ! Loop over beta pairs.

                    do iBvirt = lastBocc+1,nbasis
                       do iBocc3 = 1,lastBocc
                          iaCPHF=(iBvirt-lastBocc-1)*lastBocc + iBocc3+iBetastart
                          quick_qm_struct%cphfb(iaCPHF,Istart) = quick_qm_struct%cphfb(iaCPHF,Istart) &
                               -Skl*AOint*(quick_qm_struct%cob(Ival,iBvirt)*quick_qm_struct%cob(Jval,iBocc3)* &
                               quick_qm_struct%co(IIval,iAocc2)*quick_qm_struct%co(JJval,iAocc))
                       enddo
                    enddo
                 endif
              enddo
           enddo
        enddo
     enddo
  enddo

  ! Thats a lot of loop closures.  From top to bottow we are closing
  ! iBocc3,iBvirt,The iunique if, iunique, Imomentum,Iatom,IAocc2,IAocc.

  ! Now we need to repeat the whole process for the beta kl pairs.

  do iBocc = 1,lastBocc
     do iBocc2 = 1,lastBocc

        ! K and L are selected.  Find atom and direction of perturbation.


        do Iatom = 1,natom
           do Imomentum=1,3

              ! Now we loop over basis functions.  Note that Ibas functions are always
              ! on center Iatom, and the Jbas functions are always not on that center.
              ! This actually calculates the Skl

              Skl  = 0.d0
              do Kbas = quick_basis%first_basis_function(Iatom),quick_basis%last_basis_function(Iatom)
                 do Lbas = 1,nbasis
                    if (Lbas < quick_basis%first_basis_function(Iatom) .OR. Lbas > quick_basis%last_basis_function(Iatom)) then
                       Ax = xyz(1,quick_basis%ncenter(Lbas))
                       Bx = xyz(1,quick_basis%ncenter(Kbas))
                       Ay = xyz(2,quick_basis%ncenter(Lbas))
                       By = xyz(2,quick_basis%ncenter(Kbas))
                       Az = xyz(3,quick_basis%ncenter(Lbas))
                       Bz = xyz(3,quick_basis%ncenter(Kbas))

                       dSK=0.d0
                       itype(Imomentum,Kbas) = itype(Imomentum,Kbas)+1
                       ii = itype(1,Kbas)
                       jj = itype(2,Kbas)
                       kk = itype(3,Kbas)
                       i = itype(1,Lbas)
                       j = itype(2,Lbas)
                       k = itype(3,Lbas)
                       g_count = i+ii+j+jj+k+kk

                       do Kcon=1,ncontract(Kbas)
                          do Lcon=1,ncontract(Lbas)
                             b = aexp(Kcon,Kbas)
                             a = aexp(Lcon,Lbas)
                             call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)
                             dSK = dSK + 2.d0*b* &
                                  dcoeff(Lcon,Lbas)*dcoeff(Kcon,Kbas) &
            *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!                                  *overlap(aexp(Lcon,Lbas),aexp(Kcon,Kbas), &
!                                  itype(1,Lbas),itype(2,Lbas),itype(3,Lbas), &
!                                  itype(1,Kbas),itype(2,Kbas),itype(3,Kbas), &
!                                  xyz(1,quick_basis%ncenter(Lbas)),xyz(2,quick_basis%ncenter(Lbas)), &
!                                  xyz(3,quick_basis%ncenter(Lbas)),xyz(1,quick_basis%ncenter(Kbas)), &
!                                  xyz(2,quick_basis%ncenter(Kbas)),xyz(3,quick_basis%ncenter(Kbas)))
                          enddo
                       enddo
                       itype(Imomentum,Kbas) = itype(Imomentum,Kbas)-1
                       if (itype(Imomentum,Kbas) /= 0) then
                          itype(Imomentum,Kbas) = itype(Imomentum,Kbas)-1
                          ii = itype(1,Kbas)
                          jj = itype(2,Kbas)
                          kk = itype(3,Kbas)
                          i = itype(1,Lbas)
                          j = itype(2,Lbas)
                          k = itype(3,Lbas)
                          g_count = i+ii+j+jj+k+kk

                          do Kcon=1,ncontract(Kbas)
                             do Lcon=1,ncontract(Lbas)
                                b = aexp(Kcon,Kbas)
                                a = aexp(Lcon,Lbas)
                                call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)
                                dSK = dSK - dble(itype(Imomentum,Kbas)+1)* &
                                     dcoeff(Lcon,Lbas)*dcoeff(Kcon,Kbas) &
            *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
!                                     *overlap(aexp(Lcon,Lbas),aexp(Kcon,Kbas),! &
!                                     itype(1,Lbas),itype(2,Lbas),itype(3,Lbas), &
!                                     itype(1,Kbas),itype(2,Kbas),itype(3,Kbas), &
!                                     xyz(1,quick_basis%ncenter(Lbas)),xyz(2,quick_basis%ncenter(Lbas)), &
!                                     xyz(3,quick_basis%ncenter(Lbas)),xyz(1,quick_basis%ncenter(Kbas)), &
!                                     xyz(2,quick_basis%ncenter(Kbas)),xyz(3,quick_basis%ncenter(Kbas)))
                             enddo
                          enddo
                          itype(Imomentum,Kbas) = itype(Imomentum,Kbas)+1
                       endif
                       Skl=Skl+dSK*(quick_qm_struct%cob(Kbas,iBocc)*quick_qm_struct%cob(Lbas,iBocc2) + &
                            quick_qm_struct%cob(Lbas,iBocc)*quick_qm_struct%cob(Kbas,iBocc2))
                    endif
                 enddo
              enddo

              ! At this point we have the SKl value.  Now we need to loop over
              ! unique AO repulsions and and add the values into the array.
              ! we also need to find the location in the array defined by the Iatom
              ! Imomentum pair.

              ISTART=(Iatom-1)*3+Imomentum
              do Iunique=1,8
                 if (isame(1,Iunique) /= 0) then
                    ! Set up some dummy variables.

                    Ival = isame(1,Iunique)
                    Jval = isame(2,Iunique)
                    IIval = isame(3,Iunique)
                    JJval = isame(4,Iunique)

                    ! Loop over beta pairs.

                    do iBvirt = lastBocc+1,nbasis
                       do iBocc3 = 1,lastBocc
                          iaCPHF =(iBvirt-lastBocc-1)*lastBocc + iBocc3+ibetastart
                          quick_qm_struct%cphfb(iaCPHF,Istart) = quick_qm_struct%cphfb(iaCPHF,Istart) &
                               -Skl*AOint*(quick_qm_struct%cob(Ival,iBvirt)*quick_qm_struct%cob(Jval,iBocc3)* &
                               quick_qm_struct%cob(IIval,iBocc2)*quick_qm_struct%cob(JJval,iBocc)- &
                               quick_qm_struct%cob(Ival,iBvirt)*quick_qm_struct%cob(Jval,iBocc)* &
                               quick_qm_struct%cob(IIval,iBocc2)*quick_qm_struct%cob(Jval,iBocc3))
                       enddo
                    enddo

                    ! Loop over alpha pairs.

                    do iAvirt = lastAocc+1,nbasis
                       do iAocc3 = 1,lastAocc
                          iaCPHF = (iAvirt-lastAocc-1)*lastAocc + iAocc3
                          quick_qm_struct%cphfb(iaCPHF,Istart) = quick_qm_struct%cphfb(iaCPHF,Istart) &
                               -Skl*AOint*(quick_qm_struct%co(Ival,iAvirt)*quick_qm_struct%co(Jval,iAocc3)* &
                               quick_qm_struct%cob(IIval,iBocc2)*quick_qm_struct%cob(JJval,iBocc))
                       enddo
                    enddo
                 endif
              enddo
           enddo
        enddo
     enddo
  enddo

  ! Thats a lot of loop closures.  From top to bottow we are closing
  ! iAocc3,iAvirt,The iunique if, iunique, Imomentum,Iatom,IBocc2,IBocc.

  ! Now we calculate the final term:  (For the case where a and i are alpha)

  ! quick_qm_struct%co(Ibas,A)*quick_qm_struct%co(Jbas,I)*DENSE(TOTAL)(IIbas,JJbas)*d/dy(IbasJbas|IIbasJJbas)
  ! - quick_qm_struct%co(Ibas,A)*quick_qm_struct%co(JJbas,I)*DENSE(A)(IIbas,Jbas)*d/dy(IbasJbas|IIbasJJbas)

  ! Now, if all the gaussians are on the same center, ther derivative is zero.
  ! Check this.

  same=icenter(1).eq.icenter(2)
  same=same.and.icenter(1).eq.icenter(3)
  same=same.and.icenter(1).eq.icenter(4)
  if (same) return

  ! Otherwise, calculate the derivative.  This returns two arrays, one filled
  ! with derivatives and one filled with the center identities.  Note that
  ! this removes redundant centers, i.e. if this is a two center repulsion,
  ! only two slots are filled in the icenter and 6 slots in the deriv array.

  call CPHFB2Egrad(Ibas,Jbas,IIbas,JJbas,deriv,icenter)

  ! Now loop over atoms in icenter and momentums.  This will give us our
  ! Istart for the array.

  do Iatom = 1,4
     if (Icenter(Iatom) /= 0) then
        do Imomentum=1,3
           currderiv=deriv(Iatom,Imomentum)
           Istart = (Icenter(Iatom)-1)*3 + Imomentum
           do Iunique=1,8
              if (isame(1,Iunique) /= 0) then

                 ! Set up some dummy variables.

                 Ival = isame(1,Iunique)
                 Jval = isame(2,Iunique)
                 IIval = isame(3,Iunique)
                 JJval = isame(4,Iunique)

                 ! A quick note about the density used below.  If the integral was (ij|kl),
                 ! we need the total density matrix element k,l and the alpha and beta
                 ! density elements k,j.

                 ! Why is this?  The proof is left to the reader.

                 ! Seriously, (ij|kl) is the exchange integral that occurrs with the
                 ! coulombic integral (il|kj), and the indices on the exchange elements
                 ! density matrix are always the last two of the coloumbic matrix.

                 if (quick_method%unrst) then
                    DENSEKL = quick_qm_struct%dense(IIval,JJval)+quick_qm_struct%denseb(IIval,JJval)
                    DENSEAKJ = quick_qm_struct%dense(IIval,Jval)
                    DENSEBKJ = quick_qm_struct%denseb(IIval,Jval)
                 else
                    DENSEKL = quick_qm_struct%dense(IIval,JJval)
                    DENSEAKJ = quick_qm_struct%dense(IIval,Jval)*.5d0
                    DENSEBKJ = quick_qm_struct%dense(IIval,Jval)*.5d0
                 endif

                 ! Loop over alpha pairs.

                 do iAvirt = lastAocc+1,nbasis
                    do iAocc = 1,lastAocc
                       iaCPHF =(iAvirt-lastAocc-1)*lastAocc + iAocc
                       quick_qm_struct%cphfb(iaCPHF,Istart) = quick_qm_struct%cphfb(iaCPHF,Istart) &
                            +currderiv*quick_qm_struct%co(Ival,iAvirt)* &
                            (quick_qm_struct%co(Jval,iAocc)*DENSEKL-quick_qm_struct%co(JJval,iAocc)*DENSEAKJ)
                    enddo
                 enddo

                 ! Loop over beta pairs.

                 do iBvirt = lastBocc+1,nbasis
                    do iBocc = 1,lastBocc
                       iaCPHF =(iBvirt-lastBocc-1)*lastBocc + iBocc+Ibetastart
                       quick_qm_struct%cphfb(iaCPHF,Istart) = quick_qm_struct%cphfb(iaCPHF,Istart) &
                            +currderiv*quick_qm_struct%cob(Ival,iBvirt)* &
                            (quick_qm_struct%cob(Jval,iBocc)*DENSEKL-quick_qm_struct%cob(JJval,iBocc)*DENSEBKJ)
                    enddo
                 enddo

              endif
           enddo
        enddo
     endif
  enddo

  return
end subroutine CPHFB4cnt



! Ed Brothers. November 27,2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine CPHFB2Egrad(Ibas,Jbas,IIbas,JJbas,deriv,icenter)
  use allmod
  implicit double precision(a-h,o-z)

  dimension deriv(4,3),icenter(4)
  dimension itype2(3,4)

  ! The purpose of this subroutine is to calculate the gradient of
  ! the 2-electron 4-center integrals used in forming the B array of the
  ! CPHF.

  ! Note that this is basically grad2elec, and could be used to replace
  ! it at a later time.

  iA = quick_basis%ncenter(Ibas)
  iB = quick_basis%ncenter(Jbas)
  iC = quick_basis%ncenter(IIbas)
  iD = quick_basis%ncenter(JJbas)
  icenter(1)=iA
  icenter(2)=iB
  icenter(3)=iC
  icenter(4)=iD

  ! The itype2 array was added because if Ibas=Jbas, the code raises two
  ! angular momentums instead of one.

  do Imomentum=1,3
     itype2(Imomentum,1) = itype(Imomentum,Ibas)
     itype2(Imomentum,2) = itype(Imomentum,Jbas)
     itype2(Imomentum,3) = itype(Imomentum,IIbas)
     itype2(Imomentum,4) = itype(Imomentum,JJbas)
  enddo

  ! We have to calculate 12 quantities in this subprogram: the gradient in the
  ! X,Y, and Z directions for the 4 atom A,B,C,D.

  do Imomentum=1,3
     Agrad=0.d0
     Bgrad=0.d0
     Cgrad=0.d0
     Dgrad=0.d0

     do Icon = 1, ncontract(Ibas)
        do Jcon = 1, ncontract(Jbas)
           do IIcon = 1, ncontract(IIbas)
              do JJcon = 1, ncontract(JJbas)
                 cntrctcoeff = dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas)

                 itype2(Imomentum,1) = itype2(Imomentum,1)+1
                 Agrad = Agrad+2.d0*aexp(Icon,Ibas)*cntrctcoeff* &
                      repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 itype2(Imomentum,1) = itype2(Imomentum,1)-1

                 if (itype2(Imomentum,1) /= 0) then
                    itype2(Imomentum,1) = itype2(Imomentum,1)-1
                    temp = cntrctcoeff* &
                         repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    itype2(Imomentum,1) = itype2(Imomentum,1)+1
                    Agrad = Agrad-dble(itype2(Imomentum,1))*temp
                 endif

                 itype2(Imomentum,2) = itype2(Imomentum,2)+1
                 Bgrad = Bgrad+2.d0*aexp(Jcon,Jbas)*cntrctcoeff* &
                      repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 itype2(Imomentum,2) = itype2(Imomentum,2)-1
                 if (itype2(Imomentum,2) /= 0) then
                    itype2(Imomentum,2) = itype2(Imomentum,2)-1
                    temp = cntrctcoeff* &
                         repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    itype2(Imomentum,2) = itype2(Imomentum,2)+1
                    Bgrad = Bgrad-dble(itype2(Imomentum,2))*temp
                 endif


                 itype2(Imomentum,3) = itype2(Imomentum,3)+1
                 Cgrad = Cgrad+2.d0*aexp(IIcon,IIbas)*cntrctcoeff* &
                      repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 itype2(Imomentum,3) = itype2(Imomentum,3)-1
                 if (itype2(Imomentum,3) /= 0) then
                    itype2(Imomentum,3) = itype2(Imomentum,3)-1
                    temp = cntrctcoeff* &
                         repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    itype2(Imomentum,3) = itype2(Imomentum,3)+1
                    Cgrad = Cgrad-dble(itype2(Imomentum,3))*temp
                 endif

                 itype2(Imomentum,4) = itype2(Imomentum,4)+1
                 Dgrad = Dgrad+2.d0*aexp(JJcon,JJbas)*cntrctcoeff* &
                      repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 itype2(Imomentum,4) = itype2(Imomentum,4)-1
                 if (itype2(Imomentum,4) /= 0) then
                    itype2(Imomentum,4) = itype2(Imomentum,4)-1
                    temp = cntrctcoeff* &
                         repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    itype2(Imomentum,4) = itype2(Imomentum,4)+1
                    Dgrad = Dgrad-dble(itype2(Imomentum,4))*temp
                 endif
              enddo
           enddo
        enddo
     enddo

     ! Now we have the 4 gradients in a direction, e.g. the X gradient for
     ! atom A,B,C, and D.  Now add it into the gradient time the passed
     ! coefficient.

     deriv(1,imomentum) = Agrad
     deriv(2,imomentum) = Bgrad
     deriv(3,imomentum) = Cgrad
     deriv(4,imomentum) = Dgrad
  enddo

  ! Now check to see if any of the centers are redundant, starting from
  ! the end of the array.

  do J=4,2,-1
     do I=J-1,1,-1
        if (icenter(I) == icenter(J) .and. icenter(J) > 0) then
           do K=1,3
              deriv(I,K) = deriv(J,K) + deriv(I,K)
           enddo
           icenter(J) = 0
        endif
     enddo
  enddo

end subroutine CPHFB2Egrad


#include "util.fh"

! Ed Brothers. November 14, 2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine form_D1W1
  use allmod
  use quick_overlap_module, only: gpt, overlap
  use quick_oei_module, only: ekinetic, attrashellfock1
  use quick_cutoff_module, only: cshell_density_cutoff
  use quick_cshell_eri_module, only: getCshellEri

  implicit double precision(a-h,o-z)

    call form_d1const
 
    call form_CPHF

    call form_wdens1

end subroutine form_D1W1

subroutine form_d1const
  use allmod
  use quick_overlap_module, only: gpt, overlap
  use quick_oei_module, only: ekinetic, attrashellfock1
  use quick_cutoff_module, only: cshell_density_cutoff
  use quick_cshell_eri_module, only: getCshellEri

  implicit double precision(a-h,o-z)
  dimension itype2(3,2),ielecfld(3)
  logical :: ijcon, delta0
  double precision g_table(200),a,b, d1c((nbasis*(nbasis+1))/2,natom*3), fds(nbasis,nbasis,natom*3)
  integer i,j,k,ii,jj,kk,g_count, IIsh, JJsh, nocc

     do Ibas=1,nbasis
        ISTART = (quick_basis%ncenter(Ibas)-1) *3
           do Jbas=quick_basis%last_basis_function(quick_basis%ncenter(IBAS))+1,nbasis

              JSTART = (quick_basis%ncenter(Jbas)-1) *3

              DENSEJI = quick_qm_struct%dense(Jbas,Ibas)

  !  We have selecte two basis functions, now loop over angular momentum.
              do Imomentum=1,3

  !  do the Ibas derivatives first. In order to prevent code duplication,
  !  this has been implemented in a seperate subroutine. 
                 ijcon = .true.
                 call get_ijbas_fockderiv(Imomentum, Ibas, Jbas, Ibas, ISTART, ijcon, DENSEJI)

  !  do the Jbas derivatives.
                 ijcon = .false.
                 call get_ijbas_fockderiv(Imomentum, Ibas, Jbas, Jbas, JSTART, ijcon, DENSEJI)

              enddo
           enddo
     enddo

     ntri =  (nbasis*(nbasis+1))/2

     write(ioutfile,*)"  Derivative Fock after 1st-order Kinetic  "
     write(ioutfile,*)"     X             Y             Z "
     do I = 1, natom*3, 3
        write(ioutfile,*)" Atom no : ",(I+2)/3
        do J = 1, ntri
!           do K= 1, nbasis
           write(ioutfile,'(i3,2X,3(F9.6,7X))')J,quick_qm_struct%fd(I,J), &
           quick_qm_struct%fd(I+1,J),quick_qm_struct%fd(I+2,J)
!           enddo
        enddo
     enddo

!     call cshell_density_cutoff

     do IIsh = 1, jshell
        do JJsh = IIsh, jshell
           call attrashellfock1(IIsh,JJsh)
        enddo
     enddo

     write(ioutfile,*)"  Derivative Fock after 1st-order Ele-Nuc  "
     write(ioutfile,*)"     X             Y             Z "
     do I = 1, natom*3, 3
        write(ioutfile,*)" Atom no : ", (I+2)/3
        do J = 1, ntri
!           do K=1, nbasis
           write(ioutfile,'(i3,2X,3(F9.6,7X))')J,quick_qm_struct%fd(I,J), &
           quick_qm_struct%fd(I+1,J),quick_qm_struct%fd(I+2,J)
!           enddo
        enddo
     enddo

     write(ioutfile,*)"  1st-order Overlap  "
     write(ioutfile,*)"     X             Y             Z "
     do I = 1, natom*3, 3
        write(ioutfile,*)" Atom no : ", (I+2)/3
        do J = 1, ntri
!           do K= 1, nbasis
           write(ioutfile,'(i3,2X,3(F9.6,7X))')J,quick_qm_struct%od(I,J), &
           quick_qm_struct%od(I+1,J),quick_qm_struct%od(I+2,J)
!           enddo
        enddo
     enddo
    
     call calcFD(nbasis, ntri, quick_qm_struct%o, &
                 quick_qm_struct%dense,quick_qm_struct%fxd)

     nocc = quick_molspec%nelec/2
     nat3 = natom*3


     call d1_const(nat3, nbasis, ntri, nocc, quick_qm_struct%co, quick_qm_struct%E, &
                  quick_qm_struct%fxd, quick_qm_struct%fd, quick_qm_struct%od, &
                  quick_qm_struct%dense, d1c, quick_qm_struct%fds1)

     call trspmo(d1c,ntri,quick_qm_struct%d1con,nat3) 

end subroutine form_d1const

subroutine form_CPHF
  use allmod
  use quick_overlap_module, only: gpt, overlap
  use quick_oei_module, only: ekinetic, attrashellfock1
  use quick_cutoff_module, only: cshell_density_cutoff
  use quick_cshell_eri_module, only: getCshellEri

  implicit double precision(a-h,o-z)
  dimension natend(natom)
  dimension g1((nbasis*(nbasis+1))/2,3,natom) ! storage for G(D1,g0)
  dimension r1((nbasis*(nbasis+1))/2,3,natom) ! final 1st-order density
  dimension d1_curr((nbasis*(nbasis+1))/2,3,natom) ! current 1st-order density
  dimension d1con((nbasis*(nbasis+1))/2,3) ! storage for constant part of D1
  dimension work1((nbasis*(nbasis+1))/2,3)
  dimension work2((nbasis*(nbasis+1))/2,3)
  dimension w1(nbasis**2)
  dimension w2(nbasis*(quick_molspec%nelec/2))
  integer nocc,ntri,icycle,nvirt
  dimension x1((nbasis*(nbasis+1))/2,3,natom)
  dimension y1((nbasis*(nbasis+1))/2,3,natom)
  character*4 fext1, fext2
  dimension hess(3,natom,3,natom)
  dimension resx(natom),resy(natom),resz(natom)
  dimension hesx(natom),hesy(natom),hesz(natom)
  dimension deltx(natom),delty(natom),deltz(natom)
  dimension heltx(natom),helty(natom),heltz(natom)
  logical oscylates
  
!-------------------------------------------------------------------
      do iat=1,natom
         resx(iat)=1.d+5
         resy(iat)=1.d+5
         resz(iat)=1.d+5
         natend(iat)=0       ! cphf not converged for this atom yet
         hesx(iat)=1.d+5
         hesy(iat)=1.d+5
         hesz(iat)=1.d+5
      enddo
!-------------------------------------------------------------------
! for dens1_part1 :
!
      factor=2.0d0     ! for RHF , like orbital's occupancy
!-------------------------------------------------------------------

     ntri=(nbasis*(nbasis+1))/2
     do iat=1,natom 
        natend(iat) = 0
        do J=1,3
           r1(:,J,IAT)=quick_qm_struct%d1con(3*(iat-1)+J,:)
           quick_qm_struct%dense1(3*(iat-1)+J,:)=r1(:,J,IAT)
           d1_curr(:,J,IAT)=r1(:,J,IAT)
        enddo
     enddo
 
     g1=0.d0
     MXITER=30
     icycle=0
     lend=0

!     ITER=1

     DO ITER=1,MXITER
        icycle=icycle+1

        call get_fext(iter,fext1,fext2)
!                           name.fext1  - files for rl=l(r)
!                           name.fext2  - files for ro=O(r)

     ! Calculate G(D1,g0)

         do I=1, natom
            do J=1,3
               call quad(r1(:,J,I),quick_qm_struct%psdense,1.0d0,nbasis)
               quick_method%CalcFock_d1g0=.true.
               quick_qm_struct%o=0.d0
               call cshell_density_cutoff
               do II=1,jshell
                  call getCshellEri(II)
               enddo
               KL=0
               do K=1,nbasis
                  do L=1,K
                     KL=KL+1
                     g1(KL,J,I)=quick_qm_struct%o(K,L)
                  enddo
               enddo               
            enddo
         enddo

         do iat=1,natom
            natend(iat) = 0
         enddo

         nocc=quick_molspec%nelec/2

         DO IAT=1,NATOM
         IF(natend(iat).eq.0) THEN

            do icr=1,3
               call dens1_1dir1n(factor, nbasis, nocc, 0.0d0, g1(1,icr,iat), &
                    quick_qm_struct%co, quick_qm_struct%E, r1(1,icr,iat), w1, w2)
            enddo
!            call drumh(r1(1,1,iat),nbasis, 6  ,'PD1co -x ')
!            call drumh(r1(1,2,iat),nbasis, 6  ,'PD1co -y ')
!            call drumh(r1(1,3,iat),nbasis, 6  ,'PD1co -z ')
           do J=1,3
              d1con(:,J)=quick_qm_struct%d1con(3*(IAT-1)+J,:)
           enddo

            call file4cphf_o(ntri*3,iat,fext1,r1(1,1,iat),'write') !rl1=L(r1)
            call make_r_orto(iter,ntri,iat,d1con, r1(1,1,iat))
            call file4cphf_o(ntri*3,iat,fext2,r1(1,1,iat),'write') ! ro1=O(rl1)

            if(iter.ge.2) then
              call calc_d1r(iter,ntri,iat,work2,work1,g1(1,1,iat))
            endif

            if(iter.eq.1) then
            do J=1,3
               G1(:,J,IAT)=d1_curr(:,J,IAT)
            enddo
              call file4cphf_o(ntri*3,iat,fext1,r1(1,1,iat),'read')
              call daxpy(ntri*3,1.0d0,r1(1,1,iat),1,g1(1,1,iat),1)
            endif

            call file4cphf_o(ntri*3,iat,fext2,r1(1,1,iat),'read') !ro dft=0
            do J=1,3
               D1_CURR(:,J,IAT)=G1(:,J,IAT)
            enddo

         ENDIF ! if(natend(iat).eq.0) then
         ENDDO ! over atoms

         ! For convergence test calculate contributions to hessian :
         !                FDS1*D1 and S1DF*D1

         call HessCont2Diag(natom,natend,nbasis,ntri,r1(1,1,1),hess)
         call whessContDiag(hess,natom)

         thrs=1.0D-08

         call cphf_enHR(r1,hess,thrs,ntri,nbasis,iter,&
                       lend,1,natend,errmax,natom, & 
                       resx,resy,resz,hesx,hesy,hesz, &
                       deltx,delty,deltz, heltx,helty,heltz, &
                       oscylates)

!--------------------------------------------------------------------
!
         IF(LEND.EQ.1 .or. ICYCLE.EQ.MXITER) EXIT
!
!....................................................................


!
         IF(ICYCLE.EQ.MXITER) EXIT
!

    ENDDO    ! end of iterations

!--------------------------------------------------------------------
      call file4cphf_c(ntri*3,iter)
!--------------------------------------------------------------------
!
! read in last D1 soluiton (in R1) :
!
      do IAT=1,NATOM
         do J=1,3
            R1(:,J,IAT)=D1_CURR(:,J,IAT) 
            quick_qm_struct%dense1(3*(iat-1)+J,:)= D1_CURR(:,J,IAT)
         enddo
      enddo

!--------------------------------------------------------------------
! DO ONE MORE CALCULATION OF D1 USING FULL D1 & full integral thresh
!

!           this call is only to get G(D1,g0) with current (final) D1 
!           needed for Ist-order Weighted Density
!
!               W1 = D1*F*D + D*F1*D + D*F*D1
!
!           with full fock :  F=h+G(D,g)  and  F1=h1 + G(D1,g)+G(D,g1)
!
!           W1 MUST be consistent with D1 thus, G(D1,g0) in W1 must
!           be calculated with CURRENT D1
!
!           Thus, after this call D1 should NOT be changed
!

     ! Calculate G(D1,g0)

         do I=1, natom
            do J=1,3
               call quad(r1(:,J,I),quick_qm_struct%psdense,1.0d0,nbasis)
               quick_method%CalcFock_d1g0=.true.
               quick_qm_struct%o=0.d0
               call cshell_density_cutoff
               do II=1,jshell
                  call getCshellEri(II)
               enddo
               KL=0
               do K=1,nbasis
                  do L=1,K
                     KL=KL+1
                     g1(KL,J,I)=quick_qm_struct%o(K,L)
                  enddo
               enddo
               quick_qm_struct%fd1g0(3*(I-1)+J,:)=G1(:,J,I)
            enddo
         enddo

!--------------------------------------------------------------------
end subroutine form_CPHF

subroutine form_wdens1
      use allmod
      implicit real*8 (a-h,o-z)
!--------------------------------------------------------------
! This routine calculates the  Ist-order weighted density(x,y,z)
! one atom, 3 directions at once
!--------------------------------------------------------------
! At this point we have 1st-order density D1 and G(D1,g0).
! We need to calculate 1-st order "weighted density" W1 which
! contributes to the hessian as -2*Tr W1*S1 . This contribution
! to the final hessian can be expressed in terms of ordinary D1
! and 0th- and 1st-order FULL Fock as follows :
!

! -2 TrSa Wb =  -Tr Sa (Db F D + D Fb  D + D F Db )
!
! where F=h+G(D,g)  and  Fb=hb + G(Db,g)+G(D,gb)
!
! Comparing Tr Sa*Wb with Tr Sa*(Db*F*D+D*Fb*D+D*F*Db)
! one may named the (Db F D + D Fb D + D F Db) matrix
! the 1st-order "weighted density" . Thus we define Wb as :
!
!   Wb = Db*F*D + D*Fb*D + D*F*Db
!
! with full fock :  F=h+G(D,g)  and  Fb=hb + G(Db,g)+G(D,gb)
!
!  G(D1,g0) already done in chfsol_nat
!--------------------------------------------------------------
! INPUT :
! rhf      - logical flag for rhf/uhf
! ncf      - number of basis function
! ntri     -  ncf*(ncf+1)/2
! lind     - diagonals of i*(i-1)/2
! d0       - unpreturbed density
! fd       - FD matrix
! f1       - part of Ist-order fock matrix f1=h1 + G(D0,g1)
! g1       - G(D1,g0)
! d1       - Ist-order Density
! work     - scratch for D*F1
!
! OUTPUT :
! w1       - resulting Ist-order weighted density
!--------------------------------------------------------------
      dimension work(nbasis,nbasis,3)
      dimension d0(nbasis,nbasis)
      dimension fd(nbasis,nbasis)
!--------------------------------------------------------------
      dimension f1(nbasis*(nbasis+1)/2,3)
      dimension g1(nbasis*(nbasis+1)/2,3)

      dimension d1(nbasis*(nbasis+1)/2,3)
      dimension w1(nbasis*(nbasis+1)/2,3)
!--------------------------------------------------------------
   ncf=nbasis
   ntri=ncf*(ncf+1)/2

   do i=1,ncf
      do j=1,ncf
         d0(i,j)=quick_qm_struct%dense(i,j)
         fd(i,j)=quick_qm_struct%fxd(i,j)
      enddo
   enddo

   DO IAT=1, NATOM

      do J=1,3
         f1(:,J)=quick_qm_struct%fd(3*(IAT-1)+J,:)
         g1(:,J)=quick_qm_struct%fd1g0(3*(IAT-1)+J,:)
         d1(:,J)=quick_qm_struct%dense1(3*(IAT-1)+J,:)
      enddo

      do i=1,ncf
         ii=i*(i-1)/2
         do j=1,ncf
            jj=j*(j-1)/2
            dxfd=0.d0
            dyfd=0.d0
            dzfd=0.d0
            do k=1,ncf
               kk=k*(k-1)/2
               ik=ii+k
               kj=jj+k
               if(k.gt.i) ik=kk+i
               if(k.gt.j) kj=kk+j
               dxfd=dxfd+d1(ik,1)*fd(k,j)
               dyfd=dyfd+d1(ik,2)*fd(k,j)
               dzfd=dzfd+d1(ik,3)*fd(k,j)       !   D1*FD
            enddo
            work(i,j,1)=dxfd
            work(i,j,2)=dyfd
            work(i,j,3)=dzfd                    !  D1*FD
         enddo !   j=1,ncf
      enddo    !   i=1,ncf
!
      do i=1,ncf
         ii=i*(i-1)/2
         do j=1,ncf
            ij=ii+j
            if(j.gt.i) ij=(j*(j-1)/2)+i
            w1(ij,1)=work(i,j,1)+work(j,i,1)
            w1(ij,2)=work(i,j,2)+work(j,i,2)
            w1(ij,3)=work(i,j,3)+work(j,i,3)
         enddo
      enddo
!--------------------------------------------------------------
      do i=1,ncf
         ii=i*(i-1)/2
         do j=1,ncf
            jj=j*(j-1)/2
            dfx=0.d0
            dfy=0.d0
            dfz=0.d0
            do k=1,ncf
               kk=k*(k-1)/2
               ik=ii+k
               kj=jj+k
               if(k.gt.i) ik=kk+i
               if(k.gt.j) kj=kk+j
               dfx=dfx+d0(i,k)*( f1(kj,1)+g1(kj,1) )
               dfy=dfy+d0(i,k)*( f1(kj,2)+g1(kj,2) )
               dfz=dfz+d0(i,k)*( f1(kj,3)+g1(kj,3) )
            enddo
            work(i,j,1)=dfx
            work(i,j,2)=dfy
            work(i,j,3)=dfz                !  D0*F1
         enddo !   j=1,ncf
      enddo    !   i=1,ncf
!--------------------------------------------------------------
      ij=0
      do i=1,ncf
         ii=i*(i-1)/2
         do j=1,i
            jj=j*(j-1)/2
            ij=ij+1
            dfxd=0.d0
            dfyd=0.d0
            dfzd=0.d0
            do k=1,ncf
               kk=k*(k-1)/2
               kj=jj+k
               if(k.gt.j) kj=kk+j
!cccccc         ik=ii+k
!cccccc         if(k.gt.i) ik=kk+i
               dfxd=dfxd + work(i,k,1)*d0(k,j)      ! (D0*F1)*D0
               dfyd=dfyd + work(i,k,2)*d0(k,j)
               dfzd=dfzd + work(i,k,3)*d0(k,j)
            enddo
            w1(ij,1)=w1(ij,1)+dfxd
            w1(ij,2)=w1(ij,2)+dfyd
            w1(ij,3)=w1(ij,3)+dfzd
         enddo !   j=1,ncf
      enddo    !   i=1,ncf
!--------------------------------------------------------------
!
!     for rhf we need to divide by two
!
      call VScal(ntri*3,0.5d0,w1)
!
!--------------------------------------------------------------
!       Ist-order weighted density in W1 on return
!--------------------------------------------------------------
!
      do J=1,3
         quick_qm_struct%wdens1(3*(IAT-1)+J,:)=w1(:,J)
      enddo
  
   ENDDO
end subroutine form_wdens1

  subroutine get_ijbas_fockderiv(Imomentum, Ibas, Jbas, mbas, mstart, ijcon, DENSEJI)

  !-------------------------------------------------------------------------
  !  The purpose of this subroutine is to compute the I and J basis function
  !  derivatives required for get_kinetic_grad subroutine. The input variables
  !  mbas, mstart and ijcon are used to differentiate between I and J
  !  basis functions. For I basis functions, ijcon should be  true and should
  !  be false for J.  
  !-------------------------------------------------------------------------   
     use allmod
     use quick_overlap_module, only: gpt, opf, overlap
     use quick_oei_module, only: ekinetic
     implicit double precision(a-h,o-z)
     logical :: ijcon
     double precision g_table(200), valopf
     integer i,j,k,ii,jj,kk,g_count

     dSM = 0.0d0
     dKEM = 0.0d0

     Ax = xyz(1,quick_basis%ncenter(Jbas))
     Bx = xyz(1,quick_basis%ncenter(Ibas))
     Ay = xyz(2,quick_basis%ncenter(Jbas))
     By = xyz(2,quick_basis%ncenter(Ibas))
     Az = xyz(3,quick_basis%ncenter(Jbas))
     Bz = xyz(3,quick_basis%ncenter(Ibas))

     itype(Imomentum,mbas) = itype(Imomentum,mbas)+1

     ii = itype(1,Ibas)
     jj = itype(2,Ibas)
     kk = itype(3,Ibas)
     i = itype(1,Jbas)
     j = itype(2,Jbas)
     k = itype(3,Jbas)
     g_count = i+ii+j+jj+k+kk+2

     do Icon=1,ncontract(Ibas)
        b = aexp(Icon,Ibas)
        do Jcon=1,ncontract(Jbas)
           a = aexp(Jcon,Jbas)

           valopf = opf(a, b, dcoeff(Jcon,Jbas), dcoeff(Icon,Ibas), Ax,&
                    Ay, Az, Bx, By, Bz)

           if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then

             call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

             if(ijcon) then
                mcon = Icon
             else
                mcon = Jcon
             endif
             dSM= dSM + 2.d0*aexp(mcon,mbas)* &
             dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
             *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
             dKEM = dKEM + 2.d0*aexp(mcon,mbas)* &
             dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
             *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
           endif
        enddo
     enddo

     itype(Imomentum,mbas) = itype(Imomentum,mbas)-1

     if (itype(Imomentum,mbas) /= 0) then
        itype(Imomentum,mbas) = itype(Imomentum,mbas)-1

        ii = itype(1,Ibas)
        jj = itype(2,Ibas)
        kk = itype(3,Ibas)
        i = itype(1,Jbas)
        j = itype(2,Jbas)
        k = itype(3,Jbas)
        g_count = i+ii+j+jj+k+kk+2

        do Icon=1,ncontract(Ibas)
           b = aexp(Icon,Ibas)
           do Jcon=1,ncontract(Jbas)
             a = aexp(Jcon,Jbas)

             valopf = opf(a, b, dcoeff(Jcon,Jbas), dcoeff(Icon,Ibas), Ax,&
                      Ay, Az, Bx, By, Bz)

             if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then

               call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)
               dSM = dSM - dble(itype(Imomentum,mbas)+1)* &
               dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
               *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
               dKEM = dKEM - dble(itype(Imomentum,mbas)+1)* &
               dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
               *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
             endif
           enddo
        enddo

        itype(Imomentum,mbas) = itype(Imomentum,mbas)+1
     endif

     JID = quick_qm_struct%iarray(Jbas,Ibas)

     quick_qm_struct%od(mstart+Imomentum,JID) = quick_qm_struct%od(mstart+Imomentum,JID) &
     +dSM

     quick_qm_struct%fd(mstart+Imomentum,JID) = quick_qm_struct%fd(mstart+Imomentum,JID) &
     +dKEM!*DENSEJI*2.d0

     return

  end subroutine get_ijbas_fockderiv

      subroutine calcFD(ncf,ntri,f0,d0,fxd)
      use quick_scf_operator_module, only: scf_operator
      implicit real*8 (a-h,o-z)
      double precision f0(ncf,ncf),d0(ncf,ncf),fxd(ncf,ncf)
      logical :: deltaO   = .false.  ! delta Operator

      call scf_operator(deltaO)

      do i=1,ncf
         ii=i*(i-1)/2
         do j=1,ncf
            jj=j*(j-1)/2
            sum1=0.d0
            do k=1,ncf
               kk=k*(k-1)/2
               if(i.ge.k) then
                  ik=ii+k
               else
                  ik=kk+i
               endif
               if(k.ge.j) then
                  kj=kk+j
               else
                  kj=jj+k
               endif
               sum1=sum1+f0(i,k)*d0(k,j)
            enddo
            fxd(i,j)=sum1
         enddo ! j=1,ncf
      enddo    ! i=1,ncf

!   write fd and df on a disk ; unit=60

!      ncf2=ncf*ncf
!      call save1mat(60,lrec,ncf2,fd)

      end

      subroutine d1_const(nat3, ncf, ntri, nocc, vec, val, fxd, f1, s1, dense0, d1c, fds1)
      implicit double precision(a-h,o-z)
      double precision fxd(ncf,ncf), f1(nat3,ntri), s1(nat3,ntri), dense0(ncf,ncf)
      double precision fock1(3,ntri), overlap1(3,ntri), vec(ncf,ncf), val(ncf)
      double precision d1c(ntri,nat3), fds1(ncf,ncf,nat3)

      do i=1, nat3/3
         overlap1=s1(3*i-2:i*3,:)
         fock1=f1(3*i-2:i*3,:)
         call calcF1FDS1(ncf, ntri, i, fxd, overlap1, fock1, fds1(:,:,3*i-2:i*3))
         call d1const_xyz(ncf, ntri, nocc, i, vec, val, dense0, fock1, overlap1, d1c(:,3*i-2:i*3))
      enddo

      end subroutine d1_const

      subroutine calcF1FDS1(ncf, ntri, atm, fxd, s1, f1, fds1)
      implicit real*8 (a-h,o-z)
      dimension s1(3,ntri),f1(3,ntri)
      dimension fxd(ncf,ncf)
      dimension fds1(ncf,ncf,3)
!------------------------------------------------------------------------
! Input :
! ncf     - number of basis functions
! ntri    - ncf*(ncf+1)/2
! fxd      - Fock*Density matirx 
! s1      - 1st-order overlap
! f1      - 1st-order Fock
!
! Output:
!
! FDS1    - Fock1- Fock*Density*Overlap1 
!------------------------------------------------------------------------
!
! in order to save memory calculate first FDS1
!
      do i=1,ncf
         ii=i*(i-1)/2
         do j=1,ncf
            jj=j*(j-1)/2
            if(i.ge.j) then
               ij=ii+j
            else
               ij=jj+i
            endif
            sum1=0.d0
            sum2=0.d0
            sum3=0.d0
            do k=1,ncf
               kk=k*(k-1)/2
               if(k.ge.j) then
                  kj=kk+j
               else
                  kj=jj+k
               endif
               sum1=sum1+fxd(i,k)*s1(1,kj)
               sum2=sum2+fxd(i,k)*s1(2,kj)
               sum3=sum3+fxd(i,k)*s1(3,kj)
            enddo !  k=1,ncf

            fds1(i,j,1)=f1(1,ij)-sum1
            fds1(i,j,2)=f1(2,ij)-sum2
            fds1(i,j,3)=f1(3,ij)-sum3

         enddo !  j=1,ncf
      enddo !  i=1,ncf

!  and then S1DF and add it to FDS1

      do i=1,ncf
         ii=i*(i-1)/2
         do j=1,ncf
            jj=j*(j-1)/2
            sum1=0.d0
            sum2=0.d0
            sum3=0.d0
            do k=1,ncf
               kk=k*(k-1)/2
               if(k.ge.j) then
                  kj=kk+j
               else
                  kj=jj+k
               endif
               sum1=sum1+fxd(i,k)*s1(1,kj)
               sum2=sum2+fxd(i,k)*s1(2,kj)
               sum3=sum3+fxd(i,k)*s1(3,kj)
            enddo !  k=1,ncf

            fds1(j,i,1)=-sum1+fds1(j,i,1)
            fds1(j,i,2)=-sum2+fds1(j,i,2)
            fds1(j,i,3)=-sum3+fds1(j,i,3)
      
         enddo !  j=1,ncf
      enddo !  i=1,ncf

      end subroutine calcF1FDS1

      subroutine d1const_xyz(ncf, ntri, nocc, atm, vec, val, dense0, f1, s1, d1c)
!-----------------------------------------------------------------
! Calculates the constant part of the D1 matrix for :
!
!  (1) nuclear displacement perturbation (analytical hessian)
!
! It does it for three directions at once (x,y,z)
!-----------------------------------------------------------------
! INPUT :
!
! ncf      - number of basis function
! ntri     -  ncf*(ncf+1)/2
! nocc     - number of occupied orbitals
! vec      - eigen vectors
! val      - eigen values
! dense0   - unpreturbed density
! f1       - Ist-order fock matrix
! s1       - Ist-order overlap matrix
!
! OUTPUT :
!
! d1c      - constant part of D1 :
!            D1const=-1/2*DS1D+
!            2*Sum(o,v)(eo-ev)^-1*Co+(h1+G(d0,g1)-eo*S1)Cv*[CoCv+ + CvCo+]
!-----------------------------------------------------------------

      implicit real*8 (a-h,o-z)
      dimension dense0(ncf,ncf), vec(ncf,ncf), val(ncf)
      dimension s1(3,ntri), f1(3,ntri),s1t(ntri,3),f1t(ntri,3)
      dimension w0(ncf,ncf),w1(ncf,ncf), w2(ncf,ncf), w3(nocc,ncf)
      dimension d1c(ntri,3)          ! output

!--------------------------------------------------------------
!    Calculate the constant part of D10
!   ----------------------------------------
! 1. contributions from [ F1(d0,g1) - ei S1 ]
! 2. contributions from the 0.5 D0*S1*D0
!
!  S1 is not zero only for field dependent basis set
!
! The total constant part of the D1 stored in d1con(*)
!
!--------------------------------------------------------------
      xlvsh = 0.0d0
!--------------------------------------------------------------
      fact1=2.0d0     ! for dens1_part1 like occupancy number
      fact2=0.5d0     ! for dS1d
!      if(.not.rhf) then
!        fact1=1.0d0
!        fact2=1.0d0
!      endif
!--------------------------------------------------------------
! 1. calculate contributions from [ F1(d0,g1) - ei S1 ]
!
! one direction at the time :
!
      nvirt=ncf-nocc

      call trspmo(s1,3,s1t,ntri)
      call trspmo(f1,3,f1t,ntri)

      do icr=1,3
        call dens1_1dir2n(fact1,ncf, nocc, xlvsh, f1t(1,icr), &
                          s1t(1,icr),vec, val, d1c(1,icr), w1, &
                          w2,w3)
      enddo

!
!  2. calculate -fact2*D0*S1*D0 and add it to the constant part of D10.
!
      do icr=1,3
        call ds1d_1m(fact2,ncf, &
                     dense0,s1t(1,icr),d1c(1,icr))
      enddo

      end subroutine d1const_xyz

!==============================================================
!
      subroutine drumh (d,ncf,ifi,txt)
      implicit real*8 (a-h,o-z)
      character*8 txt
      dimension d(1)
!
      write (ifi,20) txt
!----------------------------------------------------------------------c
! do not print anything if it is too long :
      if(ncf.gt.200) then
          write(ifi,*) ' it is not printed out for NCF=',ncf,' > 200 '
         return
      endif
!----------------------------------------------------------------------c
      n=ncf
      ja=1
      do 10 i=1,n
         je=ja+i-1
         write (ifi,30) i,(d(j),j=ja,je)
         ja=je+1
   10 continue
      return
!
   20 format (30x,3h***,a8,3h***)
!  30 format (1x,i4,2x,7e12.3,/,(7x,7e12.3))
   30 format (1x,i4,2x,7f12.7,/,(7x,7f12.7))
!
      end
!=======================================================================

      subroutine dens1_1dir2n(fact ,ncf,nocc, xlv,  fock, smat, cvec ,eval ,d1, w1, w2,   w3)
!-----------------------------------------------------------------------
! one direction at the time
!-----------------------------------------------------------------------
! This routine calculates the constant part of the first-order density matrix :
!
! D1 = SUM(jocc,avir){Cj(T)*[F(D0,g1)-ej*S]*Ca/(ej-ea-xlv)*[Cj*Ca(T)+Ca*Cj(T)]}
!
!  It is calculated here as
!
! D1(p,q)=SUM(j,a)[C(T)*F-Eocc*C(T)S]*Cvirt](j,a)/(ej-ea-xlv)*[Cpj*Cqa+Cpa*Cqj]
!
!  (T) = transpose
!-----------------------------------------------------------------------
!  INTENT(IN)
!  fact    = occupancy : 2 for RHF, 1 for UHF
!  ncf     = number of contracted basis functions
!  nocc    = number of occupied orbitals
!  xlv     = level shift  (should not be used here, set it to zero)
!  fock    = first-order Fock matrix ,F(D0,g1) above, in triangular form
!  smat    = first-order AO overlap matrix in triangular form
!  cvec    = MO coefficients, C, the first nocc columns are occupied
!  eval    = orbital energies e above
!  INTENT(OUT)
!  d1      = the above part of the first-order density
!  STORAGE
!  w1      = a working array ncf**2 long
!  w2      = a working array ncf*nvirt long
!  w3      = a working array ncf*nocc long
!---------------------------------------------------
      implicit real*8 (a-h,o-z)
      parameter (zero=0.d0,one=1.0d0,two=2.d0)
      dimension fock(*),cvec(ncf,ncf),eval(ncf),d1(*)
      dimension w1(ncf,ncf),w2(ncf*(ncf-nocc)),w3(nocc,ncf)
      dimension w222(ncf,nocc),w22(nocc,ncf),w11(nocc,ncf-nocc)
!
      nvirt=ncf-nocc
!      ivirtst=nocc*ncf+1
!  expand the Fock matrix to quadratic
      call quad(fock,w1,one,ncf)
!  W2=Cocc(T)*F
      call dgemm('t','n',nocc,ncf,ncf, &
                 one, cvec, ncf, w1, ncf, &
                 zero, w2, nocc)
!  expand the overlap matrix to quadratic
      call quad(smat,w1,one,ncf)
!  W3=Cocc(T)*S
      call dgemm('t','n',nocc,ncf,ncf, &
                 one, cvec, ncf, w1, ncf, &
                 zero, w3, nocc)
!  Multiply the rows of W3 by the occupied orbital energies and subtract
!  them from W2
      call RowMultiply(nocc,ncf,eval,w3,w2)
!  Build W1=[Cocc(T)*F-Eocc*Cocc(T)*S]Cvirt=W2*Cvirt
      call dgemm('n','n', nocc, nvirt,ncf, &
                  one, w2, nocc, cvec(1,nocc+1), ncf, &
                  zero,w1, nocc)
!  Now scale W1cc(T)FCvirt with 1.0/(e(i)-e(a)-xlv)
      call scalebydenom(nocc,nvirt,eval,eval(nocc+1),xlv,w1)
!  Calculate Cvirt*W1(T). Multiply with the factor (occupancy)
!  Result is the perturbed coeff mx      
      call dgemm('n','t', ncf, nocc, nvirt, &
                  one, cvec(1,nocc+1), ncf, w1, nocc, &
                  zero,w2, ncf)
! Calculate W2*Cocc(T). Factor moved to this call by GM
      call dgemm('n','t',ncf,ncf,nocc, &
                  fact,w2,ncf,cvec,ncf, &
                  zero, w1,ncf)
!  Result, in quadratic form, is in W1.
!  Add transpose to W1 and transfer it to the triangular array d1
      call symtrian(ncf,w1,d1)
      end subroutine dens1_1dir2n
!=======================================================

      subroutine ds1d_1m(fact,ncf,d0,s1,d1c)
      implicit real*8 (a-h,o-z)
!
      parameter (zero=0.d0,one=1.0d0,half=0.5d0)
      dimension d0(ncf,ncf),s1(*)
      dimension w0(ncf,ncf),w1(ncf,ncf),w2(ncf,ncf)
      dimension d1c(*)         ! inp/out
!
!
!  expand the dens and overlap1 matrices to quadratic
!

      call quad(s1,w1, one,ncf)
!  calculate W2=D0*S1
      call dgemm('n','n',ncf,ncf,ncf, &
                 one,d0 , ncf, w1, ncf, &
                 zero, w2, ncf)
!  calculate w1=0.5*W2*D0=0.5*D0*S1*D0   RHF
!  calculate w1=1.0*W2*D0=1.0*D0*S1*D0   UHF
      call dgemm('n','n',ncf,ncf,ncf, &
                 fact,w2,ncf,d0,ncf, &
!     1           half,w2,ncf,w0,ncf,
                 zero, w1, ncf)
      ij=0
      do i=1,ncf
         do j=1,i
            ij=ij+1
            d1c(ij)=d1c(ij)-w1(j,i)
         enddo
      enddo

      end subroutine ds1d_1m

!=======================================================
      subroutine dens1_1dir1n(fact ,ncf,nocc, xlv,  fock, cvec ,eval ,d1, w1,   w2)
!-----------------------------------------------------------------------
! one direction at the time
!-----------------------------------------------------------------------
! This routine calculates a part of the first-order density matrix :
!
! D1 = SUM(jocc,avir){Cj(T)*F(D1,g0)*Ca/(ej-ea-xlv)*[Cj*Ca(T)+Ca*Cj(T)]}
!
!  (T) = transpose
!-----------------------------------------------------------------------
!  INTENT(IN)
!  fact    = occupancy : 2 for RHF, 1 for UHF
!  ncf     = number of contracted basis functions
!  nocc    = number of occupied orbitals
!  xlv     = level shift  (see below)
!  fock    = first-order Fock matrix ,F(D1,g0) above, in triangular form
!  cvec    = MO coefficients, C, the first nocc columns are occupied
!  eval    = orbital energies e above
!  INTENT(OUT)
!  d1      = the above part of the first-order density
!  STORAGE
!  w1      = a working array ncf**2 long
!  w2      = a working array ncf*nvirt long  (nvirt=ncf-nocc)
!---------------------------------------------------
      implicit real*8 (a-h,o-z)
      parameter (zero=0.d0,one=1.0d0,two=2.d0)
      dimension fock(*),cvec(ncf,ncf),eval(*),d1(*)
      dimension w1(*),w2(*)
!
      nvirt=ncf-nocc
!  expand the Fock matrix to quadratic
      call quad(fock,w1,one,ncf)
!  W2=Cocc(T)*F
      call dgemm('t','n',nocc,ncf,ncf, &
                 one, cvec, ncf, w1, ncf, &
                 zero, w2, nocc)
!  W1=Cocc(T)*F*Cvirt=W2*Cvirt  nocc x nvirt matrix
      call dgemm('n','n', nocc, nvirt,ncf, &
                  one, w2, nocc, cvec(1,nocc+1), ncf, &
                  zero,w1, nocc)
!  Now scale W1=Cocc(T)FCvirt with 1.0/(e(i)-e(a)-xlv)
      call scalebydenom(nocc,nvirt,eval,eval(nocc+1),xlv,w1)
!  Calculate Cvirt*W1(T). Multiply with the factor(occupancy)
!  Result is the perturbed coeff mx in W2!      
      call dgemm('n','t', ncf, nocc, nvirt, &
                  one, cvec(1,nocc+1), ncf, w1, nocc, &
                  zero,w2, ncf)
! Calculate W2*Cocc(T). Factor moved to this call by GM
      call dgemm('n','t',ncf,ncf,nocc, &
                  fact,w2,ncf,cvec,ncf, &
                  zero, w1,ncf)
!  Result, in quadratic form, is in W1.
!  Add transpose to W1 and transfer it to the triangular array d1
      call symtrian(ncf,w1,d1)
      end
!=======================================================
      subroutine trspmo(amat,m,bmat,n)
! This is matrix transpose A(T)--> b  (blas replacement)
      implicit real*8 (a-h,o-z)
      dimension amat(m,n), bmat(n,m)
!
      do 100 i=1,m
      do 100 j=1,n
      bmat(j,i)=amat(i,j)
  100 continue
      end
!==============================================================
      subroutine quad(a,b,c,m)
      implicit real*8 (a-h,o-z)
      dimension a(*),b(m,m)
!
!     make a quadratic symmetric or antisymmetric matrix b from
!     triangular matrix a
!
      c1=c
      con=abs(c1)
      ij=0
      do 10 i=1,m
         do 20 j=1,i
            ij=ij+1
            b(i,j)=c1*a(ij)
            b(j,i)=con*a(ij)
   20 continue
      b(i,i)=b(i,i)*(c1+con)/2
   10 continue
      return

      end
!======================================================================
      subroutine scalebydenom(nocc,nvirt,eocc,evirt,xlv,f)
!  This routine divides element (i,a) of the matrix F by
!  (eocc(i)-eocc(a)-xlv); F(i,a)=F(i,a)/(eocc(i)-eocc(a)-xlv)
!
!  Arguments:
!  INTENT(IN)
!  nocc     = number of occupied orbitals, number of rows of F
!  nvirt    = number of virtual orbitals, number of columns of F
!  eocc     = occupied orbital energies
!  evirt    = virtual orbital energies
!  INTENT(INOUT)
!  f        = Fock matrix (occupied x virtual part in MO basis)
      implicit real*8 (a-h,o-z)
      integer a
      dimension f(nocc,nvirt),eocc(nocc),evirt(nvirt)
      do a=1,nvirt
        xx=evirt(a)+xlv
        do i=1,nocc
          yy=eocc(i)-xx
          f(i,a)=f(i,a)/yy
        end do
      end do
      end subroutine scalebydenom
!======================================================================
      subroutine symtrian(n,a,b)
!  This routine adds A+A(T) to the symmetrical matrix B stored as
!  the upper triangle row-wise.
!  Arguments:
!  INTENT(IN)
!  n - dimension of the square matrix A
!  A(n,n) - square matrix
!  INTENT(OUT)
!  B(1:n*(n+1)/2): triangular matrix, it i,j element is A(i,j)+A(j,i)
      implicit real*8 (a-h,o-z)
      dimension a(n,n),b(*)
      ij=0
      do i=1,n
        do j=1,i
          ij=ij+1
          b(ij)=a(i,j)+a(j,i)
        end do
      end do
      end subroutine symtrian
!======================================================================
      subroutine RowMultiply(nocc,ncf,eocc,a,b)
!  B=B-Eocc*A
!  A,B= nocc x ncf matrices
!  Eocc is diagonal (orb. energies)
!  Arguments:
!  INTENT(IN)
!  nocc   = number of rows of A,B, Eocc  (number of occupied orbitals)
!  ncf    = number of columns of A,B (number of AOs)
!  Eocc   = occupied orbital energies (vector, diagonal matrix)
!  A      = nocc x ncf
!  INTENT(INOUT)
!  B      = nocc x ncf
      implicit real*8 (a-h,o-z)
      dimension eocc(nocc),a(nocc,ncf),b(nocc,ncf)
      do k=1,ncf
        do i=1,nocc
           b(i,k)=b(i,k)-eocc(i)*a(i,k)
        end do
      end do
      end subroutine RowMultiply
!======================================================================
      subroutine get_fext(liter,fext1,fext2)
      character*4 fext1,fext2      ! name extention for files
      character*4 name1(30),name2(30),name3(30)      ! name extention for files
      data name1 /'rl1 ','rl2 ','rl3 ','rl4 ','rl5 ', &
                  'rl6 ','rl7 ','rl8 ','rl9 ','rl10', &
                  'rl11','rl12','rl13','rl14','rl15', &
                  'rl16','rl17','rl18','rl19','rl20', &
                  'rl21','rl22','rl23','rl24','rl25', &
                  'rl26','rl27','rl28','rl29','rl30'/
      data name2 /'ro1 ','ro2 ','ro3 ','ro4 ','ro5 ', &
                  'ro6 ','ro7 ','ro8 ','ro9 ','ro10', &
                  'ro11','ro12','ro13','ro14','ro15', &
                  'ro16','ro17','ro18','ro19','ro20', &
                  'ro21','ro22','ro23','ro24','ro25', &
                  'ro26','ro27','ro28','ro29','ro30'/

         fext1=name1(liter)
         fext2=name2(liter)

      end
!======================================================================
      subroutine file4cphf_o(ndim,iat,fext,xmat,action)
      implicit real*8(a-h,o-z)
      character*256 jobname,scrf,filename
      Character*5 action
      Character*4 fext
      dimension xmat(ndim)
!----------------------------------------------------
! ndim - record length
!----------------------------------------------------
      if(action(1:3).eq.'wri' .or. action(1:3).eq.'rea') then
      else
!        call nerror(1,'file4cphf_o','wrong action: '//action,0,0)
      endif
!----------------------------------------------------
      lrec = ndim*8          ! record length in bytes
      nfile =97

      scrf='CPHF'
      len1=4
      lent = len1 + 6
      filename = scrf(1:len1)//'.'//fext
      open (unit=nfile,file=filename(1:lent), &
            form='unformatted',access='direct',recl=lrec)

      if(action(1:3).eq.'wri') then
         write(unit=nfile,rec=iat) xmat
      endif
      if(action(1:3).eq.'rea') then
         read(unit=nfile,rec=iat) xmat
      endif

      close (nfile,status='keep')

      end
!======================================================================
      subroutine make_r_orto(liter,ntri,iat,r,r_curr)

      use allmod
      implicit real*8 (a-h,o-z)
      character*4 fext1,fext2      ! name extention for files
      dimension  r_curr(ntri,3)    ! current    r
      dimension  r(ntri,3)         ! previous   r
      data acc /1.d-15 /

!
!       dn+1 = L(dn) - SUM(i=1,n)[ <di|L(dn)>/<di|di> * di ]
!
! e.g.       d2 = L(d1) - <d1|L(d1)>/<d1|d1> * d1
!

      do istep=0,liter-1
         if(istep.eq.0) then
            do J=1,3
               r(:,J)=quick_qm_struct%dense1(3*(iat-1)+J,:)
            enddo
         else
            call get_fext(istep,fext1,fext2)
            call file4cphf_o(ntri*3,iat,fext2,r ,'read') ! previous ro
         endif
!
!        calculate scalars <r|l(r_curr)> & <r|r>
!
         dxldx=ddot(ntri,r(1,1),1,r_curr(1,1),1)
         dyldy=ddot(ntri,r(1,2),1,r_curr(1,2),1)
         dzldz=ddot(ntri,r(1,3),1,r_curr(1,3),1)

         dx_dx=ddot(ntri,r(1,1),1,r(1,1),1)
         dy_dy=ddot(ntri,r(1,2),1,r(1,2),1)
         dz_dz=ddot(ntri,r(1,3),1,r(1,3),1)

         if(dx_dx.gt.acc) then
            scx=dxldx/dx_dx
         else
            scx=zero
         endif
         if(dy_dy.gt.acc) then
            scy=dyldy/dy_dy
         else
            scy=zero
         endif
         if(dz_dz.gt.acc) then
            scz=dzldz/dz_dz
         else
            scz=zero
         endif

         call daxpy(ntri,-scx,r(1,1),1,r_curr(1,1),1)
         call daxpy(ntri,-scy,r(1,2),1,r_curr(1,2),1)
         call daxpy(ntri,-scz,r(1,3),1,r_curr(1,3),1)
      enddo

      end
!======================================================================
      subroutine HessCont2Diag(natoms,natend,ncf,ntri,d1,hess)
      use quick_calculated_module, only : quick_qm_struct 
      implicit real*8 (a-h,o-z)
      dimension natend(natoms)
      dimension d1(ntri,3,natoms)
!
!     for 1 atom
      dimension fds1(ncf,ncf,3)
      dimension f1(ntri,3)
      dimension hess(3,natoms,3,natoms)
!---------------------------------------------------------------------
! calculates contributions to the hessian that involve D1
! for atom-diagonal part of the hessian
!---------------------------------------------------------------------
!     fds1 read from file 60 contains : f1 -(fds1+s1df)
!---------------------------------------------------------------------
!     contributions to Hess :
!
!     -0.5*Tr S1*[ D1FD + DF1D  +DFD1]=
!    =-0.5*Tr [D1*FDS1 + S1DF*D1]
!     -0.5*Tr DS1D*F1                 ! this one is not included
!
!                                      (I do not have G(D1,g0) only G(R1,g0)
!     and
!
!    +0.5*Tr [ h1 + G(D,g1) ]*D1     ;  f1=h1+G(d0,g)
!
!---------------------------------------------------------------------
!
      hess=0.0d0
!
      ncf2=ncf*ncf
!---------------------------------------------------------------------
!
!  Iat=Jat :
!
      DO IAT=1,NATOMS
        if(natend(iat).eq.0) then

         fds1(:,:,1)=quick_qm_struct%fds1(:,:,iat*3-2)
         fds1(:,:,2)=quick_qm_struct%fds1(:,:,iat*3-1)
         fds1(:,:,3)=quick_qm_struct%fds1(:,:,iat*3)

         do ixyz=1,3
            call spuX(d1(1,ixyz,iat),fds1(1,1,ixyz),ncf,ntri,dewe)
            hess(ixyz,iat,ixyz,iat)=hess(ixyz,iat,ixyz,iat)+dewe
            do jxyz=ixyz+1,3
                call spuX(d1(1,ixyz,iat),fds1(1,1,jxyz),ncf,ntri,ws1)
                call spuX(d1(1,jxyz,iat),fds1(1,1,ixyz),ncf,ntri,sw1)
                dewe= ws1+sw1
                dewe= dewe*0.5d0
                hess(ixyz,iat,jxyz,iat)=hess(ixyz,iat,jxyz,iat)+dewe
            enddo
         enddo
        endif    !    (natend(iat).eq.0) then
      ENDDO       !   DO IAT=1,NATOMS
!---------------------------------------------------------------------
! atom-diagonal only
!---------------------------------------------------------------------
! make full hessian out of its upper triangle part :
!
      call hess_full_up(hess,natoms)
!
!---------------------------------------------------------------------
!
!       do iat=1,natonce
!          do jat=1,natonce
!             write(6,*) ' Atoms ',iat,jat
!        write(6,66) ((hess(i,iat,j,jat),i=1,3),j=1,3)
!          enddo
!       enddo
!
!
! 66  format(1x,3(e12.6,1x))
! 66  format(1x,9(f9.5,1x))
!---------------------------------------------------------------------
      end
!======================================================================
      subroutine spuX(xmat1,xmat2,ncf,ntri,trace)
      implicit real*8 (a-h,o-z)
      parameter (zero=0.0d0)
      dimension xmat1(ntri) , xmat2(ncf,ncf)
!
!  calculates trace of X1(ntri)*X2(ncf,ncf)
!  Arguments
!  INTENT(IN)
!  Xmat1    = symmetric matrix stored in the columnwise upper triangle
!  form, Xmat(ij)=X1(i,j) where (ij)=i*(i-1)/2+j, i>=j, X is symmetrical
!  Xmat2    = square matrix
!  ncf      = dimension of the (square) matrices X1 (Xmat1) and Xmat2
!  ntri     = should be ncf*(ncf+1)/2
!  INTENT(OUT)
!  trace    = Trace(X1*Xmat2)

      trace=zero
      ii=0
      do i=1,ncf
        do k=1,i
          ik=ii+k
          trace=trace+(xmat2(i,k)+xmat2(k,i))*xmat1(ik)
        enddo
        ii=ii+i
      enddo
!  Subtract the overcounted diagonal
      ii=0
      do i=1,ncf
        ii=ii+i
        trace=trace-xmat2(i,i)*xmat1(ii)
      end do
      end
!======================================================================
      subroutine hess_full_up(hess,na)
      implicit real*8 (a-h,o-z)
      dimension hess(3,na,3,na)
!
      do i=1,na
         do ixyz=1,3
            do j=i,na
               if(j.eq.i) then
                  do jxyz=ixyz,3
                     hess(jxyz,j,ixyz,i)=hess(ixyz,i,jxyz,j)
                  enddo
               else
                  do jxyz=1,3
                     hess(jxyz,j,ixyz,i)=hess(ixyz,i,jxyz,j)
                  enddo
               endif
            enddo
         enddo
      enddo

      end
!======================================================================
      subroutine whessContDiag(hess,natoms)
      use allmod
      implicit real*8 (a-h,o-z)
      dimension hess(3,natoms,3,natoms)

      do iat=1,natoms
         xmasi=1.0d0/sqrt(emass(quick_molspec%iattype(iat)))
         factor=xmasi*xmasi
         do icr=1,3
            do jcr=1,3
               hess(icr,iat,jcr,iat)=hess(icr,iat,jcr,iat)*factor
            enddo
         enddo
      enddo

      end
!======================================================================
      subroutine cphf_enHR(r1,dhess,thrs,ntri,ncf,liter, &
                           lend, mgo, natend,errmax,natonce, &
                           resx,resy,resz,    hesx,hesy,hesz, &
                           deltx,delty,deltz, heltx,helty,heltz, &
                           oscylates)
!----------------------------------------------------------------------
! This routine checks the convergence in the CPHF procedure
!
! INPUT :
!
! natonce            -  number of atoms treated at once in CPHF
! dhess              -  changes in the hessain (delta hessian)
! thrs               -  cphf threshold
! ntri               -  ncf*(ncf+1)/2
!
! INPUT :
! liter              -  number of current CPHF iteration
! cpuit,elait        -  cpu & elapsed time of iteration
!
! OUTPUT
! lend               - shows end of CPHF : 1=converged, 0=not conv.
!
! INPUT/OUTPUT :
! natend             - natend(iatom)=1 or 0 : cophf converged or not for Iatom
!
! OUTPUT
! errmax             - maximum element in delta density
! INPUT/OUTPUT :
! resx,resy,resz     - maximu residuals for each atom
! oscylates          - logical showing smootness of CPHF
!----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      logical oscylates
      integer mgo
      dimension r1(ntri,3,natonce)     ! last residiuum
      dimension dhess(3,natonce,3,natonce)

      dimension resx(natonce),resy(natonce),resz(natonce)
      dimension hesx(natonce),hesy(natonce),hesz(natonce)

      dimension heltx(natonce),helty(natonce),heltz(natonce)
      dimension deltx(natonce),delty(natonce),deltz(natonce)

      dimension natprt(10000)    ! for local print only
      dimension natend(*)

      data zero /0.d0/
      data natconv /0/
!
!..............................................
! natend(iat) = 0 (not done) or 1 (done)
!..............................................
       thrs10= thrs*10.d0
!..............................................
!
      iout=ioutfile !call getival('iout',iout)
      iprint=1 !call getival('printh',iprint)
!..............................................
      DO IAT=1,NATONCE
        if(natend(iat).eq.0) then
           hx_max=zero
           hy_max=zero
           hz_max=zero
!try       do jat=1,natonce
           do jcr=1,3
              hx= abs( dhess(1,iat, jcr,Iat) )
              hy= abs( dhess(2,iat, jcr,Iat) )
              hz= abs( dhess(3,iat, jcr,Iat) )
              hx_max=max(hx_max,hx)
              hy_max=max(hy_max,hy)
              hz_max=max(hz_max,hz)
           enddo
!try       enddo
           heltx(iat)=hx_max
           helty(iat)=hy_max
           heltz(iat)=hz_max
        endif
      ENDDO
!..............................................
      DO IAT=1,NATONCE
        if(natend(iat).eq.0) then
           deltx(iat)=zero
           delty(iat)=zero
           deltz(iat)=zero
           call absmax(ntri,r1(1,1,iat),ix,deltx(iat))
           call absmax(ntri,r1(1,2,iat),iy,delty(iat))
           call absmax(ntri,r1(1,3,iat),iz,deltz(iat))
        endif
      ENDDO
!..............................................

      oscylates=.false.
      noscx=0
      noscy=0
      noscz=0

      errmax=0.d0
      pelRx=0.d0
      pelRy=0.d0
      pelRz=0.d0
      pelHx=0.d0
      pelHy=0.d0
      pelHz=0.d0
      do iat=1,natonce
        if(natend(iat).eq.0) then
!..............
         if(liter.gt.3) then
           if(resx(iat).gt.thrs10 .and. hesx(iat).gt.thrs10) then
            if(deltx(iat).gt.resx(iat).and.heltx(iat).gt.hesx(iat)) then
               noscx=noscx+1
               oscylates=.true.
            endif
           endif
           if(resy(iat).gt.thrs10 .and. hesy(iat).gt.thrs10) then
            if(delty(iat).gt.resy(iat).and.helty(iat).gt.hesy(iat)) then
               noscy=noscy+1
               oscylates=.true.
            endif
           endif
           if(resz(iat).gt.thrs10 .and. hesz(iat).gt.thrs10) then
            if(deltz(iat).gt.resz(iat).and.heltz(iat).gt.hesz(iat)) then
               noscz=noscz+1
               oscylates=.true.
            endif
           endif
         endif
!..............
           erRiat=max(deltx(iat),delty(iat),deltz(iat))
           erHiat=max(heltx(iat),helty(iat),heltz(iat))
           errorIat=MIN(erRiat,erHiat)
!CCCCCCCC   if(errorIat.le.thrs) natend(iat)=1
           if(errorIat.le.thrs .and. mgo.eq.3) natend(iat)=1
           errmax=max(errorIat,errmax)

           pelRx=max(     deltx(iat),pelRx)
           pelRy=max(     delty(iat),pelRy)
           pelRz=max(     deltz(iat),pelRz)

           pelHx=max(     Heltx(iat),pelHx)
           pelHy=max(     Helty(iat),pelHy)
           pelHz=max(     Heltz(iat),pelHz)
!..............
           resx(iat)=deltx(iat)
           resy(iat)=delty(iat)
           resz(iat)=deltz(iat)

           hesx(iat)=heltx(iat)
           hesy(iat)=helty(iat)
           hesz(iat)=heltz(iat)
!..............
        endif      !   (natend(iat).eq.0) then
      enddo

      lend=1
      if(errmax.gt.thrs) lend=0

      iatconv=0
      do iat=1,natonce
        if(natend(iat).eq.1) then
           iatconv=iatconv+1
           natprt(iatconv)=iat
        endif
      enddo

      if(oscylates) then
!         write(iout,421) liter,pelRx,pelRy,pelRz,oscylates, &
!                         noscx,noscy,noscz
!         write(iout,4200) pelHx,pelHy,pelHz
      else
!         write(iout,420) liter,pelRx,pelRy,pelRz,oscylates
!         write(iout,4200) pelHx,pelHy,pelHz
!
!        write(iout,520) liter,pelRx,pelRy,pelRz,
!    *                         pelHx,pelHy,pelHz,
!    *                   cput,elat,oscylates
! 520 format(i3,3x,3(1x,e10.3),1x,3(1x,e10.3),2(f8.2,1x),l5)
      endif

      if(iatconv.gt.natconv) then
         if(iprint.ge.1) then
            write(iout,422) iatconv,(natprt(k),k=1,iatconv)
         else
            write(iout,423) iatconv
         endif
      endif
      natconv=iatconv

 4200 format(  6x ,3(1x,e11.4) )
  420 format(i3,3x,3(1x,e11.4),1x,l5)
  421 format(i3,3x,3(1x,e11.4),1x,l5, &
              ' xyz=',3(i2,1x))
  422 format(3x,' cphf converged for ',i4,' unique atoms no :',6(i3,1x)/ &
           ,6(3x,'                    ',2x,'                  ', &
                                                             6(i3,1x)/))
  423 format(3x,' cphf converged for ',i4,' unique atoms ')

      end
!======================================================================
      subroutine absmax(n,a,i,xmax)
!  this routine returns the highest absolute value in the
! array a, from a(1) to a(n)
      implicit real*8 (a-h,o-z)
      dimension a(n)

      i=idamax(n,a,1)
      xmax=abs(a(i))
      end
!==============================================================
      subroutine file4cphf_c(ndim,liter)
      implicit real*8(a-h,o-z)
      character*256 jobname,scrf,filename
      common /job/jobname,lenJ
      character*4 fext1,fext2      ! name extention for files
!----------------------------------------------------
! delete files open for cphf :
!----------------------------------------------------
      lrec = ndim*8          ! record length in bytes
      nfile =97

      scrf='CPHF'
      len1=4
      len = len1 + 6

      do iter=1,liter
         call get_fext(iter,fext1,fext2)
         filename = scrf(1:len1)//'.'//fext1
         open (unit=nfile,file=filename(1:len+1), &
               form='unformatted',access='direct',recl=lrec)
         close (nfile,status='delete')

         filename = scrf(1:len1)//'.'//fext2
         open (unit=nfile,file=filename(1:len), &
               form='unformatted',access='direct',recl=lrec)
         close (nfile,status='delete')
      enddo

      end
!======================================================================
      subroutine calc_d1r(liter,ntri,iat,ri,rj,d)
      use quick_calculated_module, only : quick_qm_struct
      implicit real*8 (a-h,o-z)
      character*4 fexi1,fexi2      ! extention name for files
      character*4 fexj1,fexj2      ! extention name for files
      dimension ri(ntri,3),rj(ntri,3)
      dimension  d(ntri,3)             ! output
!----------------------------------------------------------------------
      dimension a(liter*liter),b(liter,liter),w(liter+1),c(liter)
      dimension lv(liter), mv(liter)
      dimension liter_dir(3)
!----------------------------------------------------------------------
! input :
!
! liter   - current iteration
! ntri    - ncf*(ncf+1)/2  ; ncf=no of b.f.
! a,b     - arrays (liter x liter) for linear sys. of eq.
!   w       vector (liter)
! lv,mv   - aux. vctors needed for: osinv (b,n,det ,acc,lv,mv)
!
!           a c = w
!
! ri,rj   - storage for residuals from previous iterations
!
! output :
!
! d       - resulting 1st-order density matrix (solution at iter=liter)
! there is no r among the parameters!!???      
! r       - "predicted" residuum , needed ONLY for RESET
!
!
!....................................................................
! Note : the algorithm works like this :
!
!    equation to solve :  D = D0 + L(D) -> Res R= D0 + L(D) -D
!
!  iter
!
!    0      d0=dconst=r0=r0o
!    1      calc: r1=L(r0) ,
!           make r1 orthog. to r0o  r1o=Ort(r1)
!    2      calc: r2=l(r1o),
!           make r2 orthog. to r0o,r1o      r2o=Ort(r2)
!           calc d2=c0*r0o + c1*r1o
!    3      calc: r3=l(r2o),
!           make r3 orthog. to r0o,r1o,r2o  r3o=Ort(r3)
!           calc d3=c0*r0o + c1*r1o + c2*r2o
!    4      calc: r4=l(r3o),
!           make r4 orthog. to r0o,...,r3o,  r4o=Ort(r4)
!           calc d4=c0*r0o + c1*r1o + c2*r2o + c3*r3o
!    5      calc: r5=l(r4o),
!           make r5 orthog. to r0o,...,r4o,  r5o=Ort(r5)
!           calc d5=c0*r0o + c1*r1o + c2*r2o + c3*r3o+c4*r4o
!
!   k+1     calc: r(k+1)=l(rko),
!           make r(k+1) orthog. to r0o,...,rko,  r(k+1)o=Ort(r(k+1))
!           calc d(k+1)=c0*r0o + c1*r1o + ... + c(k)*r(k)o
!....................................................................
! Coefficients { c(0).....c(k) } determined from projections of the
! " predicted " residuum R(K+1) on all previous ORTHOG. resid. {r(k)o}
!
!  <r0o | R(k+1)> =0
!  <r1o | R(k+1)> =0
!  <r2o | R(k+1)> =0
!    .........
!  <rko | R(k+1)> =0
!
! where R(K+1) = r0o + c0*(r1 - r0o)
!                    + c1*(r2 - r1o)
!                    + c2*(r3 - r2o)
!                      ............
!                    + ck*(r(k+1) - rko)
!
! R(K+1) needed to be calculated ONLY because of potential RESET:
! if we want to reset (restart) CPHF after, let say, iter=3
! then using :
!
!    d3=c0*r0o + c1*r1o + c2*r2o and R3=r0o+ c0*(r1 - r0o)
!                                          + c1*(r2 - r1o)
!                                          + c2*(r3 - r2o)
!
! we express :
!
!   d5 = d3  +  c3*r3o+c4*r4o
!
!   R5 = R3  +  c3*(r4-r3o) + c4*(r5-r4o)
!----------------------------------------------------------------------
! read in data :
      do J=1,3
         d(:,J)=quick_qm_struct%dense1(3*(iat-1)+J,:)
      enddo
!----------------------------------------------------------------------
      liter_dir(1)=liter
      liter_dir(2)=liter
      liter_dir(3)=liter
      do istep=1,liter
         call get_fext(istep,fexi1,fexi2)
         call file4cphf_o(ntri*3,iat,fexi1,ri,'read')   !  rl
         do icr=1,3
            call absmax(ntri,ri(1,icr),ix,rmax)
            if(rmax.le.1.0d-7) liter_dir(icr)=liter_dir(icr)-1
         enddo
      enddo
!----------------------------------------------------------------------
      do icr=1,3
         do istep=1,liter_dir(icr)
            if(istep.eq.1) then
               do J=1,3
                  ri(:,J)=quick_qm_struct%dense1(3*(iat-1)+J,:)
               enddo
            else
               call get_fext(istep-1,fexi1,fexi2)
               call file4cphf_o(ntri*3,iat,fexi2,ri,'read') !  ro
            endif

            w(istep)=ddot(ntri,ri(1,icr),1, d(1,icr),1) !<ro|dcon>
            c(istep)=ddot(ntri,ri(1,icr),1,ri(1,icr),1)

            do jstep=1,liter_dir(icr)
               call get_fext(jstep,fexj1,fexj2)
               call file4cphf_o(ntri*3,iat,fexj1,rj,'read')        !  rl
               b(istep,jstep)=ddot(ntri,ri(1,icr),1,rj(1,icr),1) ! <ro | rl>
            enddo
         enddo
!----------------------------------------------------------
         do ii=1,liter_dir(icr)
            b(ii,ii)=b(ii,ii)-c(ii)
            w(ii)=-w(ii)
         enddo

         call make_smallA(b,liter,a,liter_dir(icr))
!
!        ready to solve linear system of equations liter x liter :
!
         call lineqsys(liter_dir(icr),A,w,lv,mv,c)
!
!        calculate density (residuum not needed) :
!
!         d= c1*ro_0 + c2*ro_1 + c3*ro_2 + ...+ c_l*ro_l-1
!
!         r=d0 + c1*(rl_1-ro_0)
!              + c2*(rl_2-ro_1)
!              + c3*(rl_3-ro_2)
!              +...
!              + cl*(rl_l -ro_l-1)
!
! for density
              d(:,icr)=0.0d0

         do istep=1,liter_dir(icr)
            if(istep.eq.1) then
               do J=1,3
                  ri(:,J)=quick_qm_struct%dense1(3*(iat-1)+J,:)
               enddo
            else
               call get_fext(istep-1,fexi1,fexi2)
               call file4cphf_o(ntri*3,iat,fexi2,ri,'read') !  ro
            endif
            call daxpy(ntri,c(istep), ri(1,icr),1, d(1,icr),1 ) ! final d
         enddo
      enddo
!----------------------------------------------------------------------
      end
!======================================================================
      subroutine make_smallA(b,liter,a,liter_n0)
      implicit real*8 (a-h,o-z)
      dimension b(liter,liter)
      dimension a(liter_n0,liter_n0)

      do j=1,liter_n0
         do i=1,liter_n0
            a(i,j)=b(i,j)
         enddo
      enddo

      end
!======================================================================
      subroutine lineqsys(n,b,w,lv,mv,c)
      implicit real*8 (a-h,o-z)
      dimension b(n,n),w(n)
      dimension lv(n),mv(n)
      dimension c(n)            ! coefficients output
      data acc,zero,one /1.d-15 , 0.d0 , 1.d0/
!----------------------------------------------------------
! re-normalize rows :
!
      do 110 i=1,n
      bii1=one
      if(abs(b(i,i)).gt.acc) bii1=one/b(i,i)
      w(i)=w(i)*bii1
      do 110 j=1,n
      b(i,j)=b(i,j)*bii1
  110 continue
!----------------------------------------------------------
!     write(6,*)' b-matrix nxn w(n) : n=',n
!     call f_lush(6)
!
!     if(n.eq.2) then
!        do ii=1,n
!           write(6,62)(b(ii,jj),jj=1,n),w(ii)
!        enddo
!     endif
! 62  format(2(f12.6,1x),2x,f12.6)
!     if(n.eq.3) then
!        do ii=1,n
!           write(6,63)(b(ii,jj),jj=1,n),w(ii)
!        enddo
!     endif
! 63  format(3(f12.6,1x),2x,f12.6)
!----------------------------------------------------------
      call osinv (b,n,det ,acc,lv,mv)
!-------------------------------------------------------------
!     write(6,*)'det(n)=',det,' n=',n
!-------------------------------------------------------------
      if( abs(det).gt.acc)then
         do 111 i=1,n
            sx=zero
            do 222 j=1,n
            sx=sx+b(i,j)*w(j)
  222       continue
            c(i)=sx
  111    continue
      else
         do i=1,n
            c(i)=one
         enddo
      endif

!     write(6,*)'    n=',n,' det=',det,' coefficients :'
!     write(6,67) n,c
! 67  format('n=',i2,2x,6(f10.6,1x))
!-------------------------------------------------------------
      end
!======================================================================
      subroutine osinv (a,n,d,tol,l,m)
      implicit real*8 (a-h,o-z)
!
!     parameters:  a - input matrix , destroyed in computation and repla
!                      by resultant inverse (must be a general matrix)
!                  n - order of matrix a
!                  d - resultant determinant
!            l and m - work vectors of length n
!                tol - if pivot element is less than this parameter the
!                      matrix is taken for singular (usually = 1.0e-8)
!     a determinant of zero indicates that the matrix is singular
!
      dimension a(1), m(1), l(1)
      d=1.0d0
      nk=-n
      do 180 k=1,n
         nk=nk+n
         l(k)=k
         m(k)=k
         kk=nk+k
         biga=a(kk)
         do 20 j=k,n
            iz=n*(j-1)
         do 20 i=k,n
            ij=iz+i
!
!     10 follows
!
            if (abs(biga)-abs(a(ij))) 10,20,20
   10       biga=a(ij)
            l(k)=i
            m(k)=j
   20    continue
         j=l(k)
         if (j-k) 50,50,30
   30    ki=k-n
         do 40 i=1,n
            ki=ki+n
            holo=-a(ki)
            ji=ki-k+j
            a(ki)=a(ji)
   40    a(ji)=holo
   50    i=m(k)
         if (i-k) 80,80,60
   60    jp=n*(i-1)
         do 70 j=1,n
            jk=nk+j
            ji=jp+j
            holo=-a(jk)
            a(jk)=a(ji)
   70    a(ji)=holo
   80    if (abs(biga)-tol) 90,100,100
   90    d=0.0d0
         return
  100    do 120 i=1,n
            if (i-k) 110,120,110
  110       ik=nk+i
            a(ik)=a(ik)/(-biga)
  120    continue
         do 150 i=1,n
            ik=nk+i
            ij=i-n
         do 150 j=1,n
            ij=ij+n
            if (i-k) 130,150,130
  130       if (j-k) 140,150,140
  140       kj=ij-i+k
            a(ij)=a(ik)*a(kj)+a(ij)
  150    continue
         kj=k-n
         do 170 j=1,n
            kj=kj+n
            if (j-k) 160,170,160
  160       a(kj)=a(kj)/biga
  170    continue
         d=d*biga
         a(kk)=1.0d0/biga
  180 continue
      k=n
  190 k=k-1
      if (k) 260,260,200
  200 i=l(k)
      if (i-k) 230,230,210
  210 jq=n*(k-1)
      jr=n*(i-1)
      do 220 j=1,n
         jk=jq+j
         holo=a(jk)
         ji=jr+j
         a(jk)=-a(ji)
  220 a(ji)=holo
  230 j=m(k)
      if (j-k) 190,190,240
  240 ki=k-n
      do 250 i=1,n
         ki=ki+n
         holo=a(ki)
         ji=ki+j-k
         a(ki)=-a(ji)
  250 a(ji)=holo
      go to 190
  260 return
      end
!==============================================================
!      SUBROUTINE VScal(N,skal,V)
!      IMPLICIT REAL*8(A-H,O-Z)
!
!  Scales all elements of a vector by skal
!    V = V * skal
!
!      REAL*8 V(N)
!
!      DO 10 I=1,N
!      V(I) = V(I)*skal
! 10   CONTINUE
!
!      RETURN
!      END
! =================================================================


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
  use quick_overlap_module, only: gpt, overlap
  use quick_oei_module, only: ekinetic
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
  use quick_overlap_module, only: gpt, overlap
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


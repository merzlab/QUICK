#include "util.fh"
!

! hfoperatordeltadc
!-------------------------------------------------------
! Xiao HE, Delta density matrix increase is implemented here. 07/07/07 version
subroutine hfoperatordeltadc
   use allmod
   use quick_gaussian_class_module
   use quick_cutoff_module, only: cshell_dnscreen
   implicit double precision(a-h,o-z)

   double precision cutoffTest,testtmp
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   double precision fmmonearrayfirst(0:2,0:2,1:2,1:6,1:6,1:6,1:6)
   double precision fmmtwoarrayfirst(0:2,0:2,1:2,1:6,1:6,1:6,1:6)
   double precision g_table(200)
   integer i,j,k,g_count

   ! The purpose of this subroutine is to form the operator matrix
   ! for a full Hartree-Fock calculation, i.e. the Fock matrix.  The
   ! Fock matrix is as follows:

   ! O(I,J) =  F(I,J) = KE(I,J) + IJ attraction to each atom + repulsion_prim
   ! with each possible basis  - 1/2 exchange with each
   ! possible basis.

   ! Note that the Fock matrix is symmetric.

   ! May 15,2002-This code now also does all the HF energy calculation. Ed.

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

               do iatom = 1,natom
                  quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%denseSave(Ibas,Ibas)* &
                        dcoeff(Jcon,Ibas)*dcoeff(Icon,Ibas)* &
                        attraction(aexp(Jcon,Ibas),aexp(Icon,Ibas), &
                        itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                        itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                        xyz(1,quick_basis%ncenter(Ibas)),xyz(2,quick_basis%ncenter(Ibas)), &
                        xyz(3,quick_basis%ncenter(Ibas)),xyz(1,quick_basis%ncenter(Ibas)), &
                        xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)), &
                        xyz(1,iatom),xyz(2,iatom),xyz(3,iatom), &
                        quick_molspec%chg(iatom))
               enddo
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
!                        ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
!                        itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
!                        itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
!                        xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
!                        xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
!                        xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))

                  ! Nuclear attraction.

                  do iatom = 1,natom
                     quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%denseSave(Jbas,Ibas)* &
                           dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)* &
                           2.d0*attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                           itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                           itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                           xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                           xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                           xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)), &
                           xyz(1,iatom),xyz(2,iatom),xyz(3,iatom), &
                           quick_molspec%chg(iatom))
                  enddo
               enddo
            enddo
         enddo
      enddo

   endif
   !
   ! Alessandro GENONI 03/21/2007
   ! Sum the ECP integrals to the partial Fock matrix
   !
   if (quick_method%ecp) then
      call ecpoperator
   end if
   !
   ! The previous two terms are the one electron part of the Fock matrix.
   ! The next two terms define the two electron part.
   !

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

   do II=1,jshell
      do JJ=II,jshell
         Testtmp=Ycutoff(II,JJ)
         !         tbd1=quick_basis%gcexpomin(II)+quick_basis%gcexpomin(JJ)
         do KK=II,jshell
            do LL=KK,jshell
               !               tbd2=quick_basis%gcexpomin(KK)+quick_basis%gcexpomin(LL)
               testCutoff = TESTtmp*Ycutoff(KK,LL)
               if(testCutoff.gt.quick_method%integralCutoff)then
                  DNmax=max(4.0d0*cutmatrix(II,JJ),4.0d0*cutmatrix(KK,LL), &
                        cutmatrix(II,LL),cutmatrix(II,KK),cutmatrix(JJ,KK),cutmatrix(JJ,LL))
                  if((dcconnect(II,JJ).eq.1.and.(4.0d0*cutmatrix(KK,LL)*testCutoff).gt.quick_method%integralCutoff) &
                        .or.(dcconnect(KK,LL).eq.1.and.(4.0d0*cutmatrix(II,JJ)*testCutoff).gt.quick_method%integralCutoff) &
                        .or.(dcconnect(II,KK).eq.1.and.(cutmatrix(JJ,LL)*testCutoff).gt.quick_method%integralCutoff) &
                        .or.(dcconnect(LL,II).eq.1.and.(cutmatrix(JJ,KK)*testCutoff).gt.quick_method%integralCutoff) &
                        .or.(dcconnect(JJ,KK).eq.1.and.(cutmatrix(II,LL)*testCutoff).gt.quick_method%integralCutoff) &
                        .or.(dcconnect(JJ,LL).eq.1.and.(cutmatrix(II,KK)*testCutoff).gt.quick_method%integralCutoff))then

                     call shell
                  endif
               endif

            enddo
         enddo
      enddo
   enddo


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

   return
end subroutine hfoperatordeltadc




! hfoperatordc
!-------------------------------------------------------
! 11/14/2010 Yipu Miao: Clean up code with the integration of
! some subroutines
! Ed Brothers. November 27, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
subroutine hfoperatordc(oneElecO)
   use allmod
   use quick_gaussian_class_module
    use quick_cutoff_module, only: cshell_density_cutoff
   use quick_cshell_eri_module, only: getCshellEriEnergy
   implicit double precision(a-h,o-z)

   double precision cutoffTest,testtmp,oneElecO(nbasis,nbasis)
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   double precision fmmonearrayfirst(0:2,0:2,1:2,1:6,1:6,1:6,1:6)
   double precision fmmtwoarrayfirst(0:2,0:2,1:2,1:6,1:6,1:6,1:6)

   !---------------------------------------------------------------------
   ! This subroutine is to form hf operator with div-and-con
   ! The original purpose of this subroutine is to form the operator matrix
   ! for a full Hartree-Fock calculation, i.e. the Fock matrix. But after
   ! a very simple modification, it becomes hf operator generator for div-and-con
   ! Calculation. Since the Fock matrix is as follows:
   ! O(I,J) =  F(I,J) = KE(I,J) + IJ attraction to each atom + repulsion_prim
   ! with each possible basis  - 1/2 exchange with each possible basis. Note that
   ! the Fock matrix is symmetric. The 2e integral will be ommited if two basis is
   ! appears in the subsystems at the same time
   ! May 15,2002-This code now also does all the HF energy calculation. Ed.
   !---------------------------------------------------------------------

   !=================================================================
   ! Step 1. evaluate 1e integrals
   ! This job is only done on master node since it won't cost much resource
   ! and parallel will even waste more than it saves
   !-----------------------------------------------------------------
   ! The first part is kinetic part
   ! O(I,J) =  F(I,J) = "KE(I,J)" + IJ
   !-----------------------------------------------------------------
   call copyDMat(oneElecO,quick_qm_struct%o,nbasis)
   if (quick_method%printEnergy) call get1eEnergy

   !-----------------------------------------------------------------
   ! Alessandro GENONI 03/21/2007
   ! Sum the ECP integrals to the partial Fock matrix
   !-----------------------------------------------------------------
   if (quick_method%ecp) then
      call ecpoperator
   end if

   !--------------------------------------------
   ! The previous two terms are the one electron part of the Fock matrix.
   ! The next two terms define the two electron part.
   !--------------------------------------------
   call cshell_density_cutoff

   !--------------------------------------------
   ! Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
   ! Reference: Strout DL and Scuseria JCP 102(1995),8448.
   !--------------------------------------------
   do II=1,jshell
      call get2edc
   enddo

   call copySym(quick_qm_struct%o,nbasis)

   if(quick_method%printEnergy) call getCshellEriEnergy

   return
end subroutine hfoperatordc


#ifdef MPIV
!*******************************************************
! mpi_hfoperatordc
!-------------------------------------------------------
! Yipu Miao.    11/02/2010: Transfer it to MPI version
! Ed Brothers.  05/15/2002: This code now also does all the HF energy calculation
! Ed Brothers. created at November 27, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
!*******************************************************
subroutine mpi_hfoperatordc(oneElecO)
   use allmod
   use quick_gaussian_class_module
    use quick_cutoff_module, only: cshell_density_cutoff
   use quick_cshell_eri_module, only: getCshellEriEnergy
   use mpi
   implicit double precision(a-h,o-z)

   double precision testtmp,cutoffTest,oneElecO(nbasis,nbasis)
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   double precision,allocatable:: temp2d(:,:)

   double precision fmmonearrayfirst(0:2,0:2,1:2,1:6,1:6,1:6,1:6)
   double precision fmmtwoarrayfirst(0:2,0:2,1:2,1:6,1:6,1:6,1:6)
   !-------------------------------------------------------
   ! The purpose of this subroutine is to form the operator matrix
   ! for a full Hartree-Fock calculation, i.e. the Fock matrix.  The
   ! Fock matrix is as follows:

   ! O(I,J) =  F(I,J) = KE(I,J) + IJ attraction to each atom + repulsion_prim
   ! with each possible basis  - 1/2 exchange with each
   ! possible basis.

   ! Note that the Fock matrix is symmetric.
   !-------------------------------------------------------


   allocate(temp2d(nbasis,nbasis))

   !------- MPI/ ALL NODES -------------------

   !=================================================================
   ! Step 1. evaluate 1e integrals
   ! This job is only done on master node since it won't cost much resource
   ! and parallel will even waste more than it saves
   !-----------------------------------------------------------------
   ! The first part is kinetic part
   ! O(I,J) =  F(I,J) = "KE(I,J)" + IJ
   !-----------------------------------------------------------------

   !------- MPI/MASTER -------------------
   if(MASTER) then

      call copyDMat(oneElecO,quick_qm_struct%o,nbasis)
      !-----------------------------------------------------------------
      ! Now calculate 1e-Energy
      !-----------------------------------------------------------------
      if(quick_method%printEnergy) call get1eEnergy

      !-----------------------------------------------------------------
      ! Alessandro GENONI 03/21/2007
      ! Sum the ECP integrals to the partial Fock matrix
      !-----------------------------------------------------------------
      if (quick_method%ecp) then
         call ecpoperator
      end if
   endif

   !------- END MPI/MASTER ----------------


   !------- MPI/ ALL NODES ----------------

   !=================================================================
   ! Step 2. evaluate 2e integrals
   ! The 2e integrals are evenly distibuted to every nodes.(not absolutely even)
   ! And every node will work one some kind of shell
   ! since the integral will be summed into opeartor such as Fock Operator, together
   ! with 1e integrals, we reset Operator value for slave nodes
   ! and summation of the operator of slave nodes(only 2e integrals) and
   ! master node(with 1e integral and 2e integrals), is the anticipated operator
   !-----------------------------------------------------------------

   ! The previous two terms are the one electron part of the Fock matrix.
   ! The next two terms define the two electron part.
   call cshell_density_cutoff


   ! We reset the operator value for slave nodes. Actually, in most situation,
   ! they were zero before reset, but to make things safe
   if (.not.master) then
      do i=1,nbasis
         do j=1,nbasis
            quick_qm_struct%o(i,j)=0
         enddo
      enddo
   endif

!-----------------Madu----------------
   if(master) then
   write (*,*) "Madu: Before 2e "
   do i=1,nbasis
      do j=1,nbasis
         write (*,*) "Madu: O = ",quick_qm_struct%o(i,j) !Madu
      enddo
   enddo
   endif
!-----------------Madu----------------

   ! sync every nodes
   call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

   !------------------------------------------------------------------
   ! Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
   ! Reference: Strout DL and Scuseria JCP 102(1995),8448.
   !------------------------------------------------------------------

   ! every nodes will take about jshell/nodes shells integrals
   ! such as 1 water, which has 4 jshell, and 2 nodes will take 2 jshell respectively
   do i=1,mpi_jshelln(mpirank)
      ii=mpi_jshell(mpirank,i)
      call get2edc
   enddo

   ! After evaluation of 2e integrals, we can communicate every node so
   ! that we can sum all integrals

   ! slave node will send infos


   if(.not.master) then
      do i=1,nbasis
         do j=1,nbasis
            temp2d(i,j)=quick_qm_struct%o(i,j)
         enddo
      enddo
      ! send operator to master node
      call MPI_SEND(temp2d,nbasis*nbasis,mpi_double_precision,0,mpirank,MPI_COMM_WORLD,IERROR)

      ! master node will receive infos from every nodes
   else
      do i=1,mpisize-1
         ! receive opertors from slave nodes
         call MPI_RECV(temp2d,nbasis*nbasis,mpi_double_precision,i,i,MPI_COMM_WORLD,MPI_STATUS,IERROR)
         ! and sum them into operator
         do ii=1,nbasis
            do jj=1,nbasis
               quick_qm_struct%o(ii,jj)=quick_qm_struct%o(ii,jj)+temp2d(ii,jj)
            enddo
         enddo
      enddo
   endif

   ! sync all nodes
   call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

!-----------------Madu----------------
   if(master) then
   write (*,*) "Madu: after 2e "
   do i=1,nbasis
      do j=1,nbasis
         write (*,*) "Madu: O = ",quick_qm_struct%o(i,j) !Madu
      enddo
   enddo
   endif
!-----------------Madu----------------
   stop

   !--------- MPI/ MASTER NODE -------------------
   if (master) then
      ! remeber the operator is symmetry, which can save many resource
      call copySym(quick_qm_struct%o,nbasis)

      ! E=sigma[i,j] (Pij*Fji)
      if(quick_method%printEnergy) call getCshellEriEnergy
   endif
   !-------- END MPI/MASTER NODE -----------------

   return
end subroutine mpi_hfoperatordc

#endif

subroutine get2edc
   !------------------------------------------------
   ! This subroutine is to get 2e integral for d&c
   !------------------------------------------------
   use allmod
   implicit double precision(a-h,o-z)
   double precision testtmp,cutoffTest
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   RECORD_TIME(timer_begin%t2e) !Trigger the timer for 2e-integrals

   do JJ=II,jshell
      Testtmp=Ycutoff(II,JJ)
      !      tbd1=quick_basis%gcexpomin(II)+quick_basis%gcexpomin(JJ)
      do KK=II,jshell
         do LL=KK,jshell
            !            tbd2=quick_basis%gcexpomin(KK)+quick_basis%gcexpomin(LL)
            testCutoff = TESTtmp*Ycutoff(KK,LL)
            if(testCutoff.gt.quick_method%integralCutoff)then
               DNmax=max(4.0d0*cutmatrix(II,JJ),4.0d0*cutmatrix(KK,LL), &
                     cutmatrix(II,LL),cutmatrix(II,KK),cutmatrix(JJ,KK),cutmatrix(JJ,LL))
               if((dcconnect(II,JJ).eq.1.and.(4.0d0*cutmatrix(KK,LL)*testCutoff).gt.quick_method%integralCutoff) &
                     .or.(dcconnect(KK,LL).eq.1.and.(4.0d0*cutmatrix(II,JJ)*testCutoff).gt.quick_method%integralCutoff) &
                     .or.(dcconnect(II,KK).eq.1.and.(cutmatrix(JJ,LL)*testCutoff).gt.quick_method%integralCutoff) &
                     .or.(dcconnect(LL,II).eq.1.and.(cutmatrix(JJ,KK)*testCutoff).gt.quick_method%integralCutoff) &
                     .or.(dcconnect(JJ,KK).eq.1.and.(cutmatrix(II,LL)*testCutoff).gt.quick_method%integralCutoff) &
                     .or.(dcconnect(JJ,LL).eq.1.and.(cutmatrix(II,KK)*testCutoff).gt.quick_method%integralCutoff))then

                  call shell
               endif
            endif

         enddo
      enddo
   enddo


   RECORD_TIME(timer_end%T2e)  ! Terminate the timer for 2e-integrals
   timer_cumer%T2e=timer_cumer%T2e+timer_end%T2e-timer_begin%T2e ! add the time to cumer
end subroutine get2edc

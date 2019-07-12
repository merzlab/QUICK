#include "config.h"
!This file was created by Madu Manathunga on 04/11/2019

subroutine MPI_dftoperator(oneElecO,deltaO)
   use allmod
   implicit double precision(a-h,o-z)
   include "mpif.h"
   double precision oneElecO(nbasis,nbasis)
   double precision,allocatable:: temp2d(:,:) !Madu
   !This array holds the total number of grid points for each node
   integer, dimension(0:mpisize-1) :: itotgridspn !Madu
   !This array holds the grid pt upper limit for mpi
   integer, dimension(0:mpisize-1) :: igridptul !Madu 
   !This array holds the grid pt lower limit for mpi
   integer, dimension(0:mpisize-1) :: igridptll !Madu 

   allocate(temp2d(nbasis,nbasis)) !Madu

   ! The purpose of this subroutine is to form the operator matrix
   ! for a full Density Functional calculation, i.e. the KS matrix.  The
   ! KS  matrix is as follows:

   ! O(I,J) =  F(I,J) = KE(I,J) + IJ attraction to each atom + repulsion_prim
   ! with each possible basis  + Exchange/Correlation functional.

   ! Note that the KS operator matrix is symmetric.

   ! The first part is the one elctron code.

   !------- MPI/MASTER ------------------



if(MASTER) then  !Madu

! write(*,*) "E0=",quick_qm_struct%Eel
   call cpu_time(timer_begin%T1e)
   do Ibas=1,nbasis
      do Jbas=Ibas,nbasis
         quick_qm_struct%o(Jbas,Ibas) = 0.d0
         do Icon=1,ncontract(ibas)
            do Jcon=1,ncontract(jbas)

               quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+ &
                     dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)* &
                     ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                     itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                     itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                     xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                     xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                     xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
                     
            enddo
         enddo
      enddo
   enddo


   do IIsh=1,jshell
      do JJsh=IIsh,jshell
         call attrashell(IIsh,JJsh)
      enddo
   enddo

   Eelxc=0.0d0

  if(quick_method%printEnergy)then
     quick_qm_struct%Eel=0.d0
     do Ibas=1,nbasis
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Ibas)
 
              ! Kinetic energy.
 
              quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%dense(Ibas,Ibas)* &
                    dcoeff(Jcon,Ibas)*dcoeff(Icon,Ibas)* &
                    ekinetic(aexp(Jcon,Ibas),aexp(Icon,Ibas), &
                    itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                    itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                    xyz(1,quick_basis%ncenter(Ibas)),xyz(2,quick_basis%ncenter(Ibas)), &
                    xyz(3,quick_basis%ncenter(Ibas)),xyz(1,quick_basis%ncenter(Ibas)), &
                    xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
           enddo
        enddo
     enddo
 
     do Ibas=1,nbasis
        do Jbas=Ibas+1,nbasis
           do Icon=1,ncontract(ibas)
              do Jcon=1,ncontract(jbas)
 
                 ! Kinetic energy.
 
                 quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%dense(Jbas,Ibas)* &
                       dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)* &
                       2.d0*ekinetic(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                       itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                       itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                       xyz(1,quick_basis%ncenter(Jbas)),xyz(2,quick_basis%ncenter(Jbas)), &
                       xyz(3,quick_basis%ncenter(Jbas)),xyz(1,quick_basis%ncenter(Ibas)), &
                       xyz(2,quick_basis%ncenter(Ibas)),xyz(3,quick_basis%ncenter(Ibas)))
             enddo
           enddo
        enddo
     enddo
 
     do IIsh=1,jshell
        do JJsh=IIsh,jshell
           call attrashellenergy(IIsh,JJsh)
         enddo
     enddo

  endif


!   write(*,*) "E1=",quick_qm_struct%Eel
   call cpu_time(timer_end%T1e)
   timer_cumer%T1e=timer_cumer%T1e+timer_end%T1e-timer_begin%T1e
   if(quick_method%printEnergy)then
   !   write (ioutfile,'("Time for one-electron energy evaluation=",F16.9)') t2-t1
   endif

   !
   ! Alessandro GENONI 03/21/2007
   ! Sum the ECP integrals to the partial Fock matrix
   !
   if (quick_method%ecp) then
      call ecpoperator
   end if

 endif !Madu

 !------- END MPI/MASTER ----------------

 !------- MPI/ ALL NODES ----------------

   ! The previous two terms are the one electron part of the Fock matrix.
   ! The next term defines the electron repulsion_prim.

   ! Delta density matrix cutoff
   call cpu_time(timer_begin%T2e)
   
   call densityCutoff

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
!   if(master) then
!   write (*,*) "Madu: Before 2e "
!   do i=1,nbasis
!      do j=1,nbasis
!         write (*,*) "Madu: O = ",quick_qm_struct%o(i,j) !Madu
!      enddo
!   enddo
!   endif
!-----------------Madu----------------

   !stop !Madu
   ! sync every nodes
   call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

   ! Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
   ! Reference: Strout DL and Scuseria JCP 102(1995),8448.


   if(quick_method%B3LYP)then
      do imps=1, mpi_jshelln(mpirank)
        II=mpi_jshell(mpirank,imps)
         do JJ=II,jshell
            Testtmp=Ycutoff(II,JJ)
            do KK=II,jshell
               do LL=KK,jshell
                  cutoffTest = TESTtmp*Ycutoff(KK,LL)
                  if(testCutoff.gt.quick_method%integralCutoff)then
                     DNmax=max(4.0d0*cutmatrix(II,JJ),4.0d0*cutmatrix(KK,LL), &
                  cutmatrix(II,LL), &
                  cutmatrix(II,KK), &
                  cutmatrix(JJ,KK), &
                  cutmatrix(JJ,LL))
                     if(cutoffTest*DNmax.gt.quick_method%integralCutoff)then
                        IIxiao=II
                        JJxiao=JJ
                        KKxiao=KK
                        LLxiao=LL
                        call shelldftb3lyp(IIxiao,JJxiao,KKxiao,LLxiao)
                     endif
                  endif
               enddo
            enddo
         enddo
      enddo
   else
    do imps=1, mpi_jshelln(mpirank)
        II=mpi_jshell(mpirank,imps)
         do JJ=II,jshell
            Testtmp=Ycutoff(II,JJ)
            do KK=II,jshell
               do LL=KK,jshell

                  cutoffTest = TESTtmp*Ycutoff(KK,LL)
                  if(cutoffTest.gt.quick_method%integralCutoff)then
                     DNmax=max(4.0d0*cutmatrix(II,JJ),4.0d0*cutmatrix(KK,LL), &
                  cutmatrix(II,LL), &
                  cutmatrix(II,KK), &
                  cutmatrix(JJ,KK), &
                  cutmatrix(JJ,LL))
                     if(cutoffTest * DNmax .gt.quick_method%integralCutoff)then
                        IIxiao=II
                        JJxiao=JJ
                        KKxiao=KK
                        LLxiao=LL
                        call shelldft(IIxiao,JJxiao,KKxiao,LLxiao)
                     endif
                  endif
               enddo
            enddo
         enddo
       enddo
   endif


 if(.not. master) then
 !write (*,*) "I am a slave "
      ! Copy Opertor to a temp array and then send it to master
      call copyDMat(quick_qm_struct%o,temp2d,nbasis)
      ! send operator to master node
   call MPI_SEND(temp2d,nbasis*nbasis,mpi_double_precision,0,mpirank,MPI_COMM_WORLD,IERROR)
!-----------------Madu----------------
! write (*,*) "Madu: This is a slave "
   do i=1,nbasis
      do j=1,nbasis
!         write (*,*) "Madu: temp2d = ",temp2d(i,j) !Madu
      enddo
   enddo
!-----------------Madu----------------
  else
!-----------------Madu----------------
! write (*,*) "Madu: This is master "
   do i=1,nbasis
      do j=1,nbasis
!         write (*,*) "Madu: temp2d = ",quick_qm_struct%o(i,j) !Madu
      enddo
   enddo
!-----------------Madu----------------
         ! master node will receive infos from every nodes
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

 call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

!-----------------Madu----------------
!   if(master) then
!   write (*,*) "Madu: MPI version after 2e "
!   do i=1,nbasis
!      do j=1,nbasis
!         write (*,*) "Madu: O = ",quick_qm_struct%o(i,j) !Madu
!      enddo
!   enddo
!   endif
!-----------------Madu----------------
!  stop

   do I=1,nbasis
      do J=1,nbasis
         quick_qm_struct%Osavedft(i,j)=quick_qm_struct%o(i,j)
!        write (*,*) "Madu: O = ",quick_qm_struct%o(i,j)
      enddo
   enddo
   
   call cpu_time(timer_end%T2e)
   timer_cumer%T2e=timer_cumer%T2e+timer_end%T2e-timer_begin%T2e
  
   do Ibas=1,nbasis
      do Jbas=Ibas+1,nbasis
         quick_qm_struct%o(Ibas,Jbas) = quick_qm_struct%o(Jbas,Ibas)
      enddo
   enddo


   call cpu_time(timer_begin%TE)
   if(quick_method%printEnergy)then
      do Ibas=1,nbasis
         do Jbas=1,nbasis
            quick_qm_struct%Eel=quick_qm_struct%Eel+quick_qm_struct%dense(Ibas,Jbas)*quick_qm_struct%o(Jbas,Ibas)
         enddo
      enddo

      quick_qm_struct%Eel=quick_qm_struct%Eel/2.0d0
   endif

   call cpu_time(timer_end%TE)
   timer_cumer%TE=timer_cumer%TE+timer_end%TE-timer_begin%TE
   
!   write(*,*) "E2=",quick_qm_struct%Eel
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
   call cpu_time(timer_begin%TEx)


   if(quick_method%B3LYP)then

		 write(*,*)  Eelxc
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

               weight=SSW(gridx,gridy,gridz,Iatm)*WTANG(Iang)*RWT(Irad)*rad3
               if (weight < quick_method%DMCutoff ) then
                  continue
               else

                  do Ibas=1,nbasis
                  
                    !write(*,*) "c",gridx, gridy, gridz
                     call pteval(gridx,gridy,gridz,phi,dphidx,dphidy, &
                           dphidz,Ibas)
                     phixiao(Ibas)=phi
                     dphidxxiao(Ibas)=dphidx
                     dphidyxiao(Ibas)=dphidy
                     dphidzxiao(Ibas)=dphidz
                     
                    ! write(*,*) "b",phi, dphidx, dphidy, dphidz
                  enddo

                  ! Next, evaluate the densities at the grid point and the gradient
                  ! at that grid point.

                  call denspt(gridx,gridy,gridz,density,densityb,gax,gay,gaz, &
                        gbx,gby,gbz)
				  !write(*,*) density, gridx, gridy, gridz
                    !write(*,*) "gridx",gridx
                    !write(*,*) "gax",gax
                    !write(*,*) "gbx",gbx
                    !write(*,*) "DENSITY=",density
                  if (density < quick_method%DMCutoff ) then
                     continue
                  else

                     densitysum=2.0d0*density
                     sigma=4.0d0*(gax*gax+gay*gay+gaz*gaz)
                     call b3lyp_e(densitysum,sigma,zkec)
                           Eelxc = Eelxc + zkec*weight

                     quick_qm_struct%aelec = weight*density+quick_qm_struct%aelec
                     quick_qm_struct%belec = weight*densityb+quick_qm_struct%belec

                     ! This allows the calculation of the derivative of the functional
                     ! with regard to the density (dfdr), with regard to the alpha-alpha
                     ! density invariant (df/dgaa), and the alpha-beta density invariant.
                     densitysum=2.0d0*density
                     sigma=4.0d0*(gax*gax+gay*gay+gaz*gaz)

                     call b3lypf(densitysum,sigma,dfdr,xiaodot)
                     ! Calculate the first term in the dot product shown above,i.e.:
                     ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
                     xdot=xiaodot*gax
                     ydot=xiaodot*gay
                     zdot=xiaodot*gaz

                     ! Now loop over basis functions and compute the addition to the matrix
                     ! element.

                     do Ibas=1,nbasis
                        phi=phixiao(Ibas)
                        dphidx=dphidxxiao(Ibas)
                        dphidy=dphidyxiao(Ibas)
                        dphidz=dphidzxiao(Ibas)
                        quicktest = DABS(dphidx+dphidy+dphidz+ &
                              phi)
                        if (quicktest < quick_method%DMCutoff ) then
                           continue
                        else
                           do Jbas=Ibas,nbasis
                              phi2=phixiao(Jbas)
                              dphi2dx=dphidxxiao(Jbas)
                              dphi2dy=dphidyxiao(Jbas)
                              dphi2dz=dphidzxiao(Jbas)
                              temp = phi*phi2
                              tempgx = phi*dphi2dx + phi2*dphidx
                              tempgy = phi*dphi2dy + phi2*dphidy
                              tempgz = phi*dphi2dz + phi2*dphidz
                              quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+ &
                                    xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                           enddo
                        endif
                     enddo
                  endif
               endif
            enddo
         enddo
      enddo
   endif


!Madu: BLYP case

   ! We reset the operator value for slave nodes. Actually, in most situation,
   ! they were zero before reset, but to make things safe

   if (.not.master) then
      do i=1,nbasis
         do j=1,nbasis
            quick_qm_struct%o(i,j)=0

!        write(*,*) "Slave O:",quick_qm_struct%o(i,j)
         enddo
      enddo

      Eelxc=0
   endif

   do i=1,nbasis
      do j=1,nbasis
         temp2d(i,j)=0;
      enddo
   enddo

   if(quick_method%BLYP)then
!Madu: go all over the atoms
      do Iatm=1,natom
!Madu: If the grid is SG1, set the Itadtemp to 50 otherwise set the value
      ! on the atomic number
         if(quick_method%ISG.eq.1)then
            Iradtemp=50
         else
            if(quick_molspec%iattype(iatm).le.10)then
               Iradtemp=23
            else
               Iradtemp=26
            endif
         endif

!Madu: Now loop over Irad

!-------------Madu-------------

    if(master) then
        do impi=0, mpisize-1
         itotgridspn(impi)=0
         igridptul(impi)=0
         igridptll(impi)=0
        enddo
               
        itmpgriddist=Iradtemp
        do while(itmpgriddist .gt. 1)
          do impi=0, mpisize-1
            itotgridspn(impi)=itotgridspn(impi)+1            
            itmpgriddist=itmpgriddist-1
            if (itmpgriddist .lt. 1) exit
          enddo
        enddo
        
        itmpgridptul=0
        do impi=0, mpisize-1
                itmpgridptul=itmpgridptul+itotgridspn(impi)
                igridptul(impi)=itmpgridptul
                if(impi .eq. 0) then
                        igridptll(impi)=1
                else
                        igridptll(impi)=igridptul(impi-1)+1
                endif                                
        enddo

        do impi=0, mpisize-1
!          write(*,*) "Total gpts:",itotgridspn(impi)
!           write(*,*) "Grid lower limit:",igridptll(impi)
!           write(*,*) "Grid upper limit:",igridptul(impi)
        enddo

    endif

    if (bMPI) then
        call MPI_BCAST(igridptll,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
        call MPI_BCAST(igridptul,mpisize,mpi_integer,0,MPI_COMM_WORLD,mpierror)
        call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
    endif

!-------------Madu-------------
!            write(*,*) "mpirank, igridptll,igridptul",mpirank, &
!            igridptll(mpirank),igridptul(mpirank)


           do Irad=igridptll(mpirank),igridptul(mpirank)
!          do Irad=1,Iradtemp
!            write(*,*) "mpirank, igridptll,igridptul,irad",mpirank, &
!            igridptll(mpirank),igridptul(mpirank), irad

!Madu: Now, form the grid, Currently located in dft.f90
            if(quick_method%ISG.eq.1)then
               call gridformnew(iatm,RGRID(Irad),iiangt)
               rad = radii(quick_molspec%iattype(iatm))
            else
               call gridformSG0(iatm,Iradtemp+1-Irad,iiangt,RGRID,RWT)
               rad = radii2(quick_molspec%iattype(iatm))
            endif

            if(master) then
!                write(*,*)"MPI-Pretest:",iatm,natom,irad,Iradtemp,iiangt
             endif

            rad3 = rad*rad*rad
            do Iang=1,iiangt
               gridx=xyz(1,Iatm)+rad*RGRID(Irad)*XANG(Iang)
               gridy=xyz(2,Iatm)+rad*RGRID(Irad)*YANG(Iang)
               gridz=xyz(3,Iatm)+rad*RGRID(Irad)*ZANG(Iang)

!if(master) then
!    write(*,*) "Master: var", RGRID(Irad), XANG(Iang), YANG(Iang), ZANG(Iang)
!else
!    write(*,*) "Slave: var", RGRID(Irad), XANG(Iang), YANG(Iang), ZANG(Iang)
!endif

               ! Next, calculate the weight of the grid point in the SSW scheme.  if
               ! the grid point has a zero weight, we can skip it.

               weight=SSW(gridx,gridy,gridz,Iatm) &
                     *WTANG(Iang)*RWT(Irad)*rad3

                  if(master) then
                     !write(*,*)"MPI-Pretest:",iatm,irad,iang,iiangt
                  endif

               if (weight < quick_method%DMCutoff ) then
                  continue
               else

                  do Ibas=1,nbasis
!Madu: pteval is located in subs folder. 
! It calculates the value of basis function I and the value of its
! cartesian derivatives in all three derivatives.

                  if (.not.master) then
!                     write(*,*) "Madu: Pteval test", gridx,gridy,gridz
                  endif

                     call pteval(gridx,gridy,gridz,phi,dphidx,dphidy, &
                           dphidz,Ibas)
                     phixiao(Ibas)=phi
                     dphidxxiao(Ibas)=dphidx
                     dphidyxiao(Ibas)=dphidy
                     dphidzxiao(Ibas)=dphidz

                  if (.not.master) then
!                     write(*,*) "Madu: Pteval test",phi, dphidx, dphidy, dphidz
                  endif
                  enddo

                  ! Next, evaluate the densities at the grid point and the gradient
                  ! at that grid point.
!Madu: pteval is located in subs folder.

                  if(.not.master) then
                       !write(*,*) "quicktest before denspt: ", gridx,gridy,gridz,density,&
                       !densityb,gax,gay,gaz,gbx,gby,gbz
                  
                  endif


                  call denspt(gridx,gridy,gridz,density,densityb,gax,gay,gaz, &
                        gbx,gby,gbz)

                  if(.not.master) then
!                       write(*,*) "quicktest after denspt:", gridx,gridy,gridz,density,&
!                       densityb,gax,gay,gaz,gbx,gby,gbz

                  endif

                  if(master) then
!                     write(*,*) "MPI-Pre test:",iatm,irad,ibas,density,quick_method%DMCutoff 
                  endif

                  if (density < quick_method%DMCutoff ) then
                     continue
                  else

                  if(master) then
                     !write(*,*) "Pre becke:",density,densityb,gax,gay,gaz,gbx,gby,gbz,Ex
                  endif
                     call becke_E(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ex)

                  if(.not.master) then
!                     write(*,*) "Mpi-Post becke:",density,densityb,gax,gay,gaz,gbx,gby,gbz,Ex
                  endif

                     call lyp_e(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ec)

                  if(master) then
                     !write(*,*) "MPI-Post becke:",iatm,irad,ibas,density,densityb,gax,gay,&
                     !gaz,gbx,gby,gbz,Ex,Ec
                  endif


                     Eelxc = Eelxc + (param7*Ex+param8*Ec) &
                           *weight
                
                  if(.not.master) then
                       !write(*,*) "Eelxc:",Eelxc,param7,Ex, param8,weight
                       
                  endif                    

                     quick_qm_struct%aelec = weight*density+quick_qm_struct%aelec
                     quick_qm_struct%belec = weight*densityb+quick_qm_struct%belec

                     ! This allows the calculation of the derivative of the functional
                     ! with regard to the density (dfdr), with regard to the alpha-alpha
                     ! density invariant (df/dgaa), and the alpha-beta density invariant.

                  if(.not.master) then
                       !write(*,*) "quicktest before becke:",&
                       !density,gax,gay,gaz,gbx,gby,gbz,dfdr,dfdgaa,dfdgab
                  endif

                     call becke(density,gax,gay,gaz,gbx,gby,gbz, &
                           dfdr,dfdgaa,dfdgab)

                  if(.not.master) then
                       !write(*,*) "quicktest after becke:",&
                       !density,gax,gay,gaz,gbx,gby,gbz,dfdr,dfdgaa,dfdgab
                  endif


                     call lyp(density,densityb,gax,gay,gaz,gbx,gby,gbz, &
                           dfdr2,dfdgaa2,dfdgab2)

                  if(.not.master) then
                       !write(*,*) "quicktest after lyp:",&
                       !density,gax,gay,gaz,gbx,gby,gbz,dfdr2,dfdgaa2,dfdgab2
                  endif

                     dfdr = dfdr+dfdr2
                     dfdgaa = dfdgaa + dfdgaa2
                     dfdgab = dfdgab + dfdgab2

                  if(.not.master) then
                       !write(*,*) "quicktest afte lyp:",&
                       !dfdr,dfdgaa,dfdgab
                  endif


                     ! Calculate the first term in the dot product shown above,i.e.:
                     ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))

                     xdot=2.d0*dfdgaa*gax+dfdgab*gbx
                     ydot=2.d0*dfdgaa*gay+dfdgab*gby
                     zdot=2.d0*dfdgaa*gaz+dfdgab*gbz

                     ! Now loop over basis functions and compute the addition to the matrix
                     ! element.

                     do Ibas=1,nbasis
                        phi=phixiao(Ibas)
                        dphidx=dphidxxiao(Ibas)
                        dphidy=dphidyxiao(Ibas)
                        dphidz=dphidzxiao(Ibas)
                        quicktest = DABS(dphidx+dphidy+dphidz+ &
                              phi)
                              if(.not.master) then
                        !        write(*,*) "quicktest:", &
                        !quicktest,quick_method%DMCutoff
                              endif
                        
                        if (quicktest < quick_method%DMCutoff ) then
                           continue
                        else
                           do Jbas=Ibas,nbasis
                              phi2=phixiao(Jbas)
                              dphi2dx=dphidxxiao(Jbas)
                              dphi2dy=dphidyxiao(Jbas)
                              dphi2dz=dphidzxiao(Jbas)
                              temp = phi*phi2
                              tempgx = phi*dphi2dx + phi2*dphidx
                              tempgy = phi*dphi2dy + phi2*dphidy
                              tempgz = phi*dphi2dz + phi2*dphidz

                              if(.not.master) then
!                              write(*,*) "MPI master O:",quick_qm_struct%o(Jbas,Ibas)
                              endif

                              quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+ &
                                    xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                           enddo
                        endif
                     enddo
                  endif
               endif
            enddo
         enddo
      enddo

 call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
!stop
!----------------------Madu-------------------
 if(.not. master) then

!   write (*,*) "XC Energy from slave: ",Eelxc

   !Send the Exc energy value
   Eelxcslave=Eelxc
   call MPI_SEND(Eelxcslave,1,mpi_double_precision,0,mpirank,MPI_COMM_WORLD,IERROR)

   call copyDMat(quick_qm_struct%o,temp2d,nbasis)
   call MPI_SEND(temp2d,nbasis*nbasis,mpi_double_precision,0,mpirank,MPI_COMM_WORLD,IERROR)
! write (*,*) "Madu: This is a slave "
!   do i=1,nbasis
!      do j=1,nbasis
!         write (*,*) "Madu: temp2d = ",temp2d(i,j) !Madu
!         write (*,*) "Madu: quick_qm_struct%o = ",quick_qm_struct%o(i,j) !Madu
!      enddo
!   enddo
  else

!  write (*,*) "XC Energy from master: ",Eelxc
! write (*,*) "Madu: This is master "
!   do i=1,nbasis
!      do j=1,nbasis
!         write (*,*) "Madu: temp2d = ",quick_qm_struct%o(i,j) !Madu
!      enddo
!   enddo
         ! master node will receive infos from every nodes
      do i=1,mpisize-1
         ! receive exchange correlation energy from slaves
         call MPI_RECV(Eelxcslave,1,mpi_double_precision,i,i,MPI_COMM_WORLD,MPI_STATUS,IERROR)
         Eelxc=Eelxc+Eelxcslave
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
 call MPI_BARRIER(MPI_COMM_WORLD,mpierror)

!-----------------Madu----------------
!   if(master) then
!   write (*,*) "Madu: MPI version after XC "
!   do i=1,nbasis
!      do j=1,nbasis
!         write (*,*) "MPI: O = ",quick_qm_struct%o(i,j) !Madu
!      enddo
!   enddo
!
!   write (*,*) "XC Energy: ",Eelxc
!   endif


!-----------------Madu----------------

endif

!stop
!----------------------Madu-------------------


 ! Finally, copy lower diagonal to upper diagonal.




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
                        phi=phixiao(Ibas)
                        dphidx=dphidxxiao(Ibas)
                        dphidy=dphidyxiao(Ibas)
                        dphidz=dphidzxiao(Ibas)
                        quicktest = DABS(dphidx+dphidy+dphidz+ &
                              phi)
                        if (quicktest < quick_method%DMCutoff ) then
                           continue
                        else
                           do Jbas=Ibas,nbasis
                              phi2=phixiao(Jbas)
                              dphi2dx=dphidxxiao(Jbas)
                              dphi2dy=dphidyxiao(Jbas)
                              dphi2dz=dphidzxiao(Jbas)
                              temp = phi*phi2
                              tempgx = phi*dphi2dx + phi2*dphidx
                              tempgy = phi*dphi2dy + phi2*dphidy
                              tempgz = phi*dphi2dz + phi2*dphidz
                              quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+ &
                                    xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
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
                        phi=phixiao(Ibas)
                        dphidx=dphidxxiao(Ibas)
                        dphidy=dphidyxiao(Ibas)
                        dphidz=dphidzxiao(Ibas)
                        quicktest = DABS(dphidx+dphidy+dphidz+ &
                              phi)
                        if (quicktest < quick_method%DMCutoff ) then
                           continue
                        else
                           do Jbas=Ibas,nbasis
                              phi2=phixiao(Jbas)
                              dphi2dx=dphidxxiao(Jbas)
                              dphi2dy=dphidyxiao(Jbas)
                              dphi2dz=dphidzxiao(Jbas)
                              temp = phi*phi2
                              tempgx = phi*dphi2dx + phi2*dphidx
                              tempgy = phi*dphi2dy + phi2*dphidy
                              tempgz = phi*dphi2dz + phi2*dphidz
                              quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+ &
                                    xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                           enddo
                        endif
                     enddo
                  endif
               endif
            enddo
         enddo
      enddo
   endif


!   print *,"Eelxc = ", Eelxc

   call cpu_time(timer_end%TEx)

   timer_cumer%TEx=timer_cumer%TEx+timer_end%TEx-timer_begin%TEx

   do Ibas=1,nbasis
      do Jbas=Ibas+1,nbasis
         quick_qm_struct%o(Ibas,Jbas) = quick_qm_struct%o(Jbas,Ibas)
      enddo
   enddo


   !write (ioutfile,'(" TIME of evaluation numerical integral = ",F12.2)') &
    !     T3-T2

if(master) then
   quick_qm_struct%Eel=quick_qm_struct%Eel+Eelxc
!   write(*,*) "E1+E2+Eelxc=",quick_qm_struct%eel
endif


end subroutine MPI_dftoperator

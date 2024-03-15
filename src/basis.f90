#include "util.fh"
!
!	basis.f90
!	new_quick
!
!	Created by Yipu Miao on 3/9/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

subroutine readbasis(natomxiao,natomstart,natomfinal,nbasisstart,nbasisfinal,ierr)
   !
   ! Read in the requested basisfile. This is done twice, once to get sizes and
   ! allocate variables, then again to assign the basis
   !
   use allmod
   use quick_gridpoints_module
   use quick_exception_module

#ifdef CEW
   use quick_cew_module, only: quick_cew
#endif

#ifdef MPIV
   use mpi
#endif

   !
   implicit double precision(a-h,o-z)
   character(len=120) :: line
   character(len=2) :: atom,shell
   logical :: isatom
   logical :: isbasis ! If basis file contains info for a given element
   integer, dimension(0:92)  :: kcontract,kbasis
   logical, dimension(0:92)  :: atmbs,atmbs2
   
   double precision AA(MAXPRIM),BB(MAXPRIM),CC(MAXPRIM)
   integer natomstart,natomfinal,nbasisstart,nbasisfinal
   double precision, allocatable,save, dimension(:) :: aex,gcs,gcp,gcd,gcf,gcg
   integer, intent(inout) :: ierr
   logical :: blngr_test

   ! initialize the arra

   ! =============MPI/ MASTER========================
   masterwork: if (master) then
      ! =============END MPI/MASTER=====================
      ! Alessandro GENONI 03/05/2007
      ! Only for ECP calculations:
      ! * Allocate arrays whose dimensions depend on NATOM (allocateatoms_ecp)
      ! * Read the Effective Core Potentials (ECPs), modify the atomic charges
      !   and the total number of electrons (readecp)
      if (quick_method%ecp)    call readecp
      call quick_open(ibasisfile,basisfilename,'O','F','W',.true.,ierr)
      CHECK_ERROR(ierr)
      iofile = 0
      nshell = 0
      nbasis = 0
      nprim = 0
      kcontract = 0
      quick_basis%kshell = 0
      kbasis = 0
      atmbs=.true.
      atmbs2=.true.
      icont=0

      ! parse the file and find the sizes of things to allocate them in memory
      do while (iofile  == 0 )
         read(ibasisfile,'(A80)',iostat=iofile) line
           read(line,*,iostat=io) atom,ii
         if (io == 0 .and. ii == 0) then
            isatom = .true.
            call upcase(atom,2)

            do i=1,92
               if (symbol(i) == atom) then
                  iat = i
                  atmbs(i)=.false.
                  atmbs2(i)=.false.
                  icont=icont+1
               end if
            enddo
            iatom = 0
            do while (iatom==0)
               read(ibasisfile,'(A80)',iostat=iofile) line
               read(line,*,iostat=iatom) shell,iprim,dnorm

               if (iatom == 0) then
                  quick_basis%kshell(iat) = quick_basis%kshell(iat) +1
                  !kcontract(iat) = kcontract(iat) + iprim
                  if (shell == 'S') then
                     kbasis(iat) = kbasis(iat) + 1
                     kcontract(iat) = kcontract(iat) + iprim
                     elseif (shell == 'P') then
                     kbasis(iat) = kbasis(iat) + 3
                     kcontract(iat) = kcontract(iat) + iprim * 3
                     elseif (shell == 'SP') then
                     kbasis(iat) = kbasis(iat) + 4
                     kcontract(iat) = kcontract(iat) + iprim * 4
                     elseif (shell == 'D') then
                     kbasis(iat) = kbasis(iat) + 6
                     kcontract(iat) = kcontract(iat) + iprim * 6
                     elseif (shell == 'F') then
                     kbasis(iat) = kbasis(iat) + 10
                     kcontract(iat) = kcontract(iat) + iprim * 10
                  end if
                  if (shell == 'SP') then
                     do i=1,iprim
                        read(ibasisfile,'(A80)',iostat=iofile) line
                        read(line,*) a,c1,c2
                     enddo
                  else
                     do i=1,iprim
                        read(ibasisfile,'(A80)',iostat=iofile) line
                        read(line,*) a,d
                     enddo
                  end if
               end if
            enddo
         end if
      enddo
      rewind ibasisfile

      !
      ! Alessandro GENONI: 03/07/2007
      !
      ! This part of the code is important for the ECP calculations.
      ! It allows to read the basis-set from the CUSTOM File for those
      ! elements (in the studied molecule) that don't have the proper
      ! ECP basis-set! (Not for ECP=CUSTOM)
      !
      if ((quick_method%ecp .and. (icont /= 92)) .and. (.not. quick_method%custecp)) then
         open(ibasiscustfile,file=basiscustname,status='old')
         do jj=1,natomxiao
            if (atmbs(quick_molspec%iattype(jj))) then
               atmbs(quick_molspec%iattype(jj))=.false.
               iofile=0
               do while (iofile == 0)
                  read(ibasiscustfile,'(A80)',iostat=iofile) line
                  read(line,*,iostat=io) atom,ii
                  if (io == 0 .and. ii == 0) then
                     call upcase(atom,2)
                     if (symbol(quick_molspec%iattype(jj)) == atom) then
                        iat=quick_molspec%iattype(jj)
                        iatom=0
                        do while (iatom == 0)
                           read(ibasiscustfile,'(A80)',iostat=iofile) line
                           read(line,*,iostat=iatom) shell,iprim,dnorm
                           if (iatom == 0) then
                              quick_basis%kshell(iat) = quick_basis%kshell(iat) +1
                              !kcontract(iat) = kcontract(iat) + iprim
                              if (shell == 'S') then
                                 kbasis(iat) = kbasis(iat) + 1
                                 kcontract(iat) = kcontract(iat) + iprim
                                 elseif (shell == 'P') then
                                 kbasis(iat) = kbasis(iat) + 3
                                 kcontract(iat) = kcontract(iat) + iprim*3
                                 elseif (shell == 'SP') then
                                 kbasis(iat) = kbasis(iat) + 4
                                 kcontract(iat) = kcontract(iat) + iprim*4
                                 elseif (shell == 'D') then
                                 kbasis(iat) = kbasis(iat) + 6
                                 kcontract(iat) = kcontract(iat) + iprim*6
                                 elseif (shell == 'F') then
                                 kbasis(iat) = kbasis(iat) + 10
                                 kcontract(iat) = kcontract(iat) + iprim*10
                              end if
                              if (shell == 'SP') then
                                 do i=1,iprim
                                    read(ibasiscustfile,'(A80)',iostat=iofile) line
                                    read(line,*) a,c1,c2
                                 enddo
                              else
                                 do i=1,iprim
                                    read(ibasiscustfile,'(A80)',iostat=iofile) line
                                    read(line,*) a,d
                                 enddo
                              end if
                           end if
                        end do
                     end if
                  end if
               end do
               rewind ibasiscustfile
            end if
         end do
      end if


      !do i=1,83
      !  if (quick_basis%kshell(i) /= 0) print *, symbol(i),quick_basis%kshell(i),kcontract(i),kbasis(i)
      !enddo

      do i=1,natomxiao

         ! MFCC
         if(i.eq.natomstart)nbasisstart=nbasis+1
         do ixiao=1,npmfcc
            if(matomstart(ixiao).eq.i)then
               matombases(ixiao)=nbasis+1
               !         print*,ixiao,'matombases(ixiao)=',matomstart(ixiao),matombases(ixiao)
            endif
         enddo

         do ixiao=1,npmfcc-1
            if(matomstartcap(ixiao).eq.i)then
               matombasescap(ixiao)=nbasis+1
               !         print*,ixiao,'matombases(ixiao)=',matomstart(ixiao),matombases(ixiao)
            endif
         enddo

         do ixiao=1,kxiaoconnect
            if(matomstartcon(ixiao).eq.i)then
               matombasescon(ixiao)=nbasis+1
               matombasesconi(ixiao)=nbasis+1
               !         print*,ixiao,'matombases(ixiao)=',matomstart(ixiao),matombases(ixiao)
            endif
         enddo

         do ixiao=1,kxiaoconnect
            if(matomstartcon2(ixiao).eq.i)then
               matombasescon2(ixiao)=nbasis+1
               matombasesconj(ixiao)=nbasis+1
               !         print*,ixiao,'matombases(ixiao)=',matomstart(ixiao),matombases(ixiao)
            endif
         enddo

         ! MFCC

         !   print*,nshell,nbasis,nprim,quick_basis%kshell(iattype(i)),kbasis(iattype(i)),kcontract(iattype(i))
 

         nshell = nshell + quick_basis%kshell(quick_molspec%iattype(i))
         nbasis = nbasis + kbasis(quick_molspec%iattype(i))
         nprim = nprim + kcontract(quick_molspec%iattype(i))

         ! MFCC
         if(i.eq.natomfinal)nbasisfinal=nbasis
         do ixiao=1,npmfcc
            if(matomfinal(ixiao).eq.i)matombasef(ixiao)=nbasis
         enddo

         do ixiao=1,npmfcc-1
            if(matomfinalcap(ixiao).eq.i)matombasefcap(ixiao)=nbasis
         enddo

         do ixiao=1,kxiaoconnect
            if(matomfinalcon(ixiao).eq.i)then
               matombasefcon(ixiao)=nbasis
               matombasefconi(ixiao)=nbasis
            endif
         enddo

         do ixiao=1,kxiaoconnect
            if(matomfinalcon2(ixiao).eq.i)then
               matombasefcon2(ixiao)=nbasis
               matombasefconj(ixiao)=nbasis
            endif
         enddo

         ! MFCC

      enddo
      ! =============MPI/MASTER=====================
   endif masterwork
   ! =============END MPI/MASTER=====================

#ifdef MPIV
   ! =============END MPI/ALL NODES=====================
   if (bMPI) then
      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(natom,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(nshell,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(nbasis,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(nprim,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
   endif
#endif

   ! =============END MPI/ALL NODES=====================

   ! Allocate the arrays now that we know the sizes
   if(.not. allocated(Ycutoff)) allocate(Ycutoff(nshell,nshell))
   if(.not. allocated(cutmatrix)) allocate(cutmatrix(nshell,nshell))
   if(.not. allocated(aex)) allocate(aex(nprim ))
   if(.not. allocated(gcs)) allocate(gcs(nprim ))
   if(.not. allocated(gcp)) allocate(gcp(nprim ))
   if(.not. allocated(gcd)) allocate(gcd(nprim ))
   if(.not. allocated(gcf)) allocate(gcf(nprim ))
   if(.not. allocated(gcg)) allocate(gcg(nprim ))

   ! initialize the array values to zero
   Ycutoff      = 0.0d0
   cutmatrix    = 0.0d0
   aex          = 0.0d0
   gcs          = 0.0d0
   gcp          = 0.0d0
   gcd          = 0.0d0
   gcf          = 0.0d0
   gcg          = 0.0d0

   if (quick_method%ecp) then
      nbf12=nbasis*(nbasis+1)/2
      if(.not. allocated(kmin)) allocate(kmin(nshell))
      if(.not. allocated(kmax)) allocate(kmax(nshell))
      if(.not. allocated(eta)) allocate(eta(nprim))
      if(.not. allocated(ecp_int)) allocate(ecp_int(nbf12))
      if(.not. allocated(kvett)) allocate(kvett(nbf12))
      if(.not. allocated(gout)) allocate(gout(25*25))
      if(.not. allocated(ktypecp)) allocate(ktypecp(nshell))
      if(.not. allocated(zlm)) allocate(zlm(lmxdim))
      if(.not. allocated(flmtx)) allocate(flmtx(len_fac,3))
      if(.not. allocated(lf)) allocate(lf(lfdim))
      if(.not. allocated(lmf)) allocate(lmf(lmfdim))
      if(.not. allocated(lml)) allocate(lml(lmfdim))
      if(.not. allocated(lmx)) allocate(lmx(lmxdim))
      if(.not. allocated(lmy)) allocate(lmy(lmxdim))
      if(.not. allocated(lmz)) allocate(lmz(lmxdim))
      if(.not. allocated(mc)) allocate(mc(mc1dim,3))
      if(.not. allocated(mr)) allocate(mr(mc1dim,3))
      if(.not. allocated(dfac)) allocate(dfac(len_dfac))
      if(.not. allocated(dfaci)) allocate(dfaci(len_dfac))
      if(.not. allocated(factorial)) allocate(factorial(len_fac))
      if(.not. allocated(fprod)) allocate(fprod(lfdim,lfdim))
     
      ! initialize the array values to zero 
      kmin      = 0.0d0
      kmax      = 0.0d0
      eta       = 0.0d0
      ecp_int   = 0.0d0
      kvett     = 0.0d0
      gout      = 0.0d0
      ktypecp   = 0.0d0
      zlm       = 0.0d0
      flmtx     = 0.0d0
      lf        = 0.0d0
      lmf       = 0.0d0
      lmx       = 0.0d0
      lmy       = 0.0d0
      lmz       = 0.0d0
      mc        = 0.0d0
      mr        = 0.0d0
      dfac      = 0.0d0
      dfaci     = 0.0d0
      factorial = 0.0d0
      fprod     = 0.0d0

      call vett
   end if

   ! Support for old memory model, to be deleted eventually

   if(.not. allocated(itype)) allocate(itype(3,nbasis))
   if(.not. allocated(ncontract)) allocate(ncontract(nbasis))
   
   ! initialize the array values to zero
   itype     = 0
   ncontract = 0
   
   call alloc(quick_basis,natom,nshell,nbasis)
   
   do ixiao=1,nshell
      quick_basis%gcexpomin(ixiao)=99999.0d0
   enddo
   quick_basis%ncenter = 0

   ! various arrays that depend on the # of basis functions

   !call allocate_quick_gridpoints(nbasis)

   ! xiao He may reconsider this
   call alloc(quick_scratch,nbasis)

   ! do this the stupid way for now
   jbasis=1
   jshell=1
   Ninitial=0
   do i=1,nbasis
      do j=1,3
         quick_basis%KLMN(j,i)=0
      enddo
   enddo
   
   do i = 1,nshell
        do j = 0,3
            quick_basis%Qsbasis(i,j) = 0
            quick_basis%Qfbasis(i,j) = 0
        enddo
   enddo

   quick_method%hasF=.false.

   !====== MPI/MASTER ====================
   masterwork_readfile: if (master) then
      !====== END MPI/MASTER ================

      ! Adding this option to disable normalization if necessary. For testing
      ! long range integrals, one can use normalized contraction coefficients
      ! in CUSTOM basis set and read them in. Set to true for such testing.    
      blngr_test=.false.

      do i=1,natomxiao
         if (.not. atmbs2(quick_molspec%iattype(i))) then
            iofile = 0
            do while (iofile == 0)
               read(ibasisfile,'(A80)',iostat=iofile) line
               read(line,*,iostat=io) atom,ii
               if (io == 0 .and. ii == 0) then
                  call upcase(atom,2)
                  if (symbol(quick_molspec%iattype(i)) == atom) then
                     iatom = 0
                     do while (iatom==0)
                        read(ibasisfile,'(A80)',iostat=iofile) line
                        read(line,*,iostat=iatom) shell,iprim,dnorm
                        if(jshell.le.nshell)then
                           quick_basis%kprim(jshell) = iprim
                        endif
                        if (shell == 'S') then
                           quick_basis%ktype(jshell) = 1
                           quick_basis%katom(jshell) = i
                           quick_basis%kstart(jshell) = jbasis
                           quick_basis%Qnumber(jshell) = 0
                           quick_basis%Qstart(jshell)= 0
                           quick_basis%Qfinal(jshell)=0
                           quick_basis%Qsbasis(jshell,0)=0
                           quick_basis%Qfbasis(jshell,0)=0
                           quick_basis%ksumtype(jshell)=Ninitial+1

                           Ninitial=Ninitial+1
                           quick_basis%cons(Ninitial)=1.0d0
                           if (quick_method%ecp) then
                              kmin(jshell)=1
                              kmax(jshell)=1
                              ktypecp(jshell)=1
                           end if
                           do k=1,iprim
                              read(ibasisfile,'(A80)',iostat=iofile) line
                              read(line,*) AA(k),BB(k)
                              aex(jbasis)=AA(k)
                              gcs(jbasis)=BB(k)
                              jbasis = jbasis+1
                             
                              if(blngr_test) then
                                quick_basis%gccoeff(k,Ninitial)=BB(k)
                              else
                                quick_basis%gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),0,0,0)
                              endif
                              quick_basis%unnorm_gccoeff(k,Ninitial)=BB(k)

                              quick_basis%gcexpo(k,Ninitial)=AA(k)

                              if(quick_basis%gcexpomin(jshell).gt.AA(k))quick_basis%gcexpomin(jshell)=AA(k)
                           enddo
                           xnewtemp=xnewnorm(0,0,0,iprim,quick_basis%gccoeff(1:iprim,Ninitial),quick_basis%gcexpo(1:iprim,Ninitial))
                           do k=1,iprim
                              quick_basis%gccoeff(k,Ninitial)=xnewtemp*quick_basis%gccoeff(k,Ninitial)
                           enddo
                           jshell = jshell+1
                           elseif (shell == 'P') then
                           quick_basis%ktype(jshell) = 3
                           quick_basis%katom(jshell) = i
                           quick_basis%kstart(jshell) = jbasis
                           quick_basis%Qnumber(jshell) = 6
                           quick_basis%Qstart(jshell)= 1
                           quick_basis%Qfinal(jshell)= 1
                           quick_basis%Qsbasis(jshell,1)=0
                           quick_basis%Qfbasis(jshell,1)=2
                           quick_basis%ksumtype(jshell)=Ninitial+1
                           do k=1,iprim
                              read(ibasisfile,'(A80)',iostat=iofile) line
                              read(line,*) AA(k),BB(k)
                              aex(jbasis)=AA(k)
                              gcp(jbasis)=BB(k)
                              jbasis=jbasis+1
                           enddo
                           if (quick_method%ecp) then
                              kmin(jshell)=2
                              kmax(jshell)=4
                              ktypecp(jshell)=2
                           end if
                           do jjj=1,3
                              Ninitial=Ninitial+1
                              quick_basis%cons(Ninitial)=1.0d0
                              quick_basis%KLMN(JJJ,Ninitial)=1
                              do k=1,iprim
                              if(blngr_test) then
                                quick_basis%gccoeff(k,Ninitial)=BB(k)
                              else
                                quick_basis%gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),1,0,0)
                              endif
                              quick_basis%unnorm_gccoeff(k,Ninitial)=BB(k)
                                 !quick_basis%gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),1,0,0)
                                 quick_basis%gcexpo(k,Ninitial)=AA(k)
                                 if(quick_basis%gcexpomin(jshell).gt.AA(k))quick_basis%gcexpomin(jshell)=AA(k)
                              enddo
                           enddo
                           xnewtemp=xnewnorm(1,0,0,iprim,quick_basis%gccoeff(:,Ninitial),quick_basis%gcexpo(:,Ninitial))
                           do iitemp=Ninitial-2,Ninitial
                              do k=1,iprim
                                 quick_basis%gccoeff(k,iitemp)=xnewtemp*quick_basis%gccoeff(k,iitemp)
                              enddo
                           enddo

                           jshell = jshell+1
                           elseif (shell == 'SP') then
                           quick_basis%ktype(jshell) = 4
                           quick_basis%katom(jshell) = i
                           quick_basis%kstart(jshell) = jbasis
                           quick_basis%Qnumber(jshell) = 1
                           quick_basis%Qstart(jshell)= 0
                           quick_basis%Qfinal(jshell)=1
                           quick_basis%Qsbasis(jshell,0)=0
                           quick_basis%Qfbasis(jshell,0)=0
                           quick_basis%Qsbasis(jshell,1)=1
                           quick_basis%Qfbasis(jshell,1)=3
                           quick_basis%ksumtype(jshell)=Ninitial+1

                           do k=1,iprim
                              read(ibasisfile,'(A80)',iostat=iofile) line
                              read(line,*) AA(k),BB(k),CC(k)
                              aex(jbasis)=AA(k)
                              gcs(jbasis)=BB(k)
                              gcp(jbasis)=CC(k)
                              jbasis = jbasis+1
                           enddo
                           Ninitial=Ninitial+1
                           quick_basis%cons(Ninitial)=1.0d0
                           do k=1,iprim
                              !quick_basis%gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),0,0,0)
                              if(blngr_test) then
                                quick_basis%gccoeff(k,Ninitial)=BB(k)
                              else
                                quick_basis%gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),0,0,0)
                              endif
                              quick_basis%unnorm_gccoeff(k,Ninitial)=BB(k)
                              quick_basis%gcexpo(k,Ninitial)=AA(k)
                              if(quick_basis%gcexpomin(jshell).gt.AA(k))quick_basis%gcexpomin(jshell)=AA(k)
                           enddo
                           xnewtemp=xnewnorm(0,0,0,iprim,quick_basis%gccoeff(1:iprim,Ninitial),quick_basis%gcexpo(1:iprim,Ninitial))
                           do k=1,iprim
                              quick_basis%gccoeff(k,Ninitial)=xnewtemp*quick_basis%gccoeff(k,Ninitial)
                           enddo

                           do jjj=1,3
                              Ninitial=Ninitial+1
                              quick_basis%cons(Ninitial)=1.0d0
                              quick_basis%KLMN(JJJ,Ninitial)=1
                              do k=1,iprim
                                 !quick_basis%gccoeff(k,Ninitial)=CC(k)*xnorm(AA(k),1,0,0)
                              if(blngr_test) then
                                quick_basis%gccoeff(k,Ninitial)=CC(k)
                              else
                                quick_basis%gccoeff(k,Ninitial)=CC(k)*xnorm(AA(k),1,0,0)
                              endif
                              quick_basis%unnorm_gccoeff(k,Ninitial)=CC(k)
                                 quick_basis%gcexpo(k,Ninitial)=AA(k)
                                 if(quick_basis%gcexpomin(jshell).gt.AA(k))quick_basis%gcexpomin(jshell)=AA(k)
                              enddo
                           enddo
                           xnewtemp=xnewnorm(1,0,0,iprim,quick_basis%gccoeff(1:iprim,Ninitial),quick_basis%gcexpo(1:iprim,Ninitial))
                           do iitemp=Ninitial-2,Ninitial
                              do k=1,iprim
                                 quick_basis%gccoeff(k,iitemp)=xnewtemp*quick_basis%gccoeff(k,iitemp)
                              enddo
                           enddo

                           jshell = jshell+1
                           elseif (shell == 'D') then
                           quick_basis%ktype(jshell) = 6
                           quick_basis%katom(jshell) = i
                           quick_basis%kstart(jshell) = jbasis
                           quick_basis%Qnumber(jshell) = 2
                           quick_basis%Qstart(jshell)= 2
                           quick_basis%Qfinal(jshell)= 2
                           quick_basis%Qsbasis(jshell,2)=0
                           quick_basis%Qfbasis(jshell,2)=5

                           quick_basis%ksumtype(jshell)=Ninitial+1
                           do k=1,iprim
                              read(ibasisfile,'(A80)',iostat=iofile) line
                              read(line,*) AA(k),BB(k)
                              aex(jbasis)=AA(k)
                              gcd(jbasis)=BB(k)
                              jbasis=jbasis+1
                           enddo
                           do JJJ=1,6
                              Ninitial=Ninitial+1
                              if(JJJ.EQ.1)then
                                 quick_basis%KLMN(1,Ninitial)=2
                                 quick_basis%cons(Ninitial)=1.0D0
                                 elseif(JJJ.EQ.2)then
                                 quick_basis%KLMN(1,Ninitial)=1
                                 quick_basis%KLMN(2,Ninitial)=1
                                 quick_basis%cons(Ninitial)=dsqrt(3.0d0)
                                 elseif(JJJ.EQ.3)then
                                 quick_basis%KLMN(2,Ninitial)=2
                                 quick_basis%cons(Ninitial)=1.0D0
                                 elseif(JJJ.EQ.4)then
                                 quick_basis%KLMN(1,Ninitial)=1
                                 quick_basis%KLMN(3,Ninitial)=1
                                 quick_basis%cons(Ninitial)=dsqrt(3.0d0)
                                 elseif(JJJ.EQ.5)then
                                 quick_basis%KLMN(2,Ninitial)=1
                                 quick_basis%KLMN(3,Ninitial)=1
                                 quick_basis%cons(Ninitial)=dsqrt(3.0d0)
                                 elseif(JJJ.EQ.6)then
                                 quick_basis%KLMN(3,Ninitial)=2
                                 quick_basis%cons(Ninitial)=1.0d0
                              endif

                              do k=1,iprim
                                 !quick_basis%gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),quick_basis%KLMN(1,Ninitial), &
                                 !           quick_basis%KLMN(2,Ninitial),quick_basis%KLMN(3,Ninitial))

                              if(blngr_test) then
                                quick_basis%gccoeff(k,Ninitial)=BB(k)
                              else
                                quick_basis%gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),quick_basis%KLMN(1,Ninitial), &
                                            quick_basis%KLMN(2,Ninitial),quick_basis%KLMN(3,Ninitial))
                              endif
                              quick_basis%unnorm_gccoeff(k,Ninitial)=BB(k)

                                 quick_basis%gcexpo(k,Ninitial)=AA(k)
                                 if(quick_basis%gcexpomin(jshell).gt.AA(k))quick_basis%gcexpomin(jshell)=AA(k)
                              enddo
                           enddo
                           if (quick_method%ecp) then
                              kmin(jshell)=5
                              kmax(jshell)=10
                              ktypecp(jshell)=3
                           end if
                           xnewtemp=xnewnorm(2,0,0,iprim,quick_basis%gccoeff(:,Ninitial-3),quick_basis%gcexpo(:,Ninitial-3))
                           do iitemp=Ninitial-5,Ninitial-3
                              do k=1,iprim
                                 quick_basis%gccoeff(k,iitemp)=xnewtemp*quick_basis%gccoeff(k,iitemp)
                              enddo
                           enddo
                           xnewtemp=xnewnorm(1,1,0,iprim,quick_basis%gccoeff(:,Ninitial),quick_basis%gcexpo(:,Ninitial))
                           do iitemp=Ninitial-2,Ninitial
                              do k=1,iprim
                                 quick_basis%gccoeff(k,iitemp)=xnewtemp*quick_basis%gccoeff(k,iitemp)
                              enddo
                           enddo

                           jshell = jshell+1

                           elseif (shell == 'F') then
                           quick_method%hasF=.true.
                           quick_basis%ktype(jshell) = 10
                           quick_basis%katom(jshell) = i
                           quick_basis%kstart(jshell) = jbasis
                           quick_basis%Qnumber(jshell) = 3
                           quick_basis%Qstart(jshell)= 3
                           quick_basis%Qfinal(jshell)= 3
                           quick_basis%Qsbasis(jshell,3)=0
                           quick_basis%Qfbasis(jshell,3)=9

                           quick_basis%ksumtype(jshell)=Ninitial+1
                           do k=1,iprim
                              read(ibasisfile,'(A80)',iostat=iofile) line
                              read(line,*) AA(k),BB(k)
                              aex(jbasis)=AA(k)
                              gcf(jbasis)=BB(k)
                              jbasis=jbasis+1
                           enddo
                           do JJJ=1,10
                              Ninitial=Ninitial+1
                              if(JJJ.EQ.1)then
                                 quick_basis%KLMN(1,Ninitial)=3
                                 quick_basis%cons(Ninitial)=1.0D0
                                 elseif(JJJ.EQ.2)then
                                 quick_basis%KLMN(1,Ninitial)=2
                                 quick_basis%KLMN(2,Ninitial)=1
                                 quick_basis%cons(Ninitial)=dsqrt(5.0d0)
                                 elseif(JJJ.EQ.3)then
                                 quick_basis%KLMN(1,Ninitial)=1
                                 quick_basis%KLMN(2,Ninitial)=2
                                 quick_basis%cons(Ninitial)=dsqrt(5.0d0)
                                 elseif(JJJ.EQ.4)then
                                 quick_basis%KLMN(2,Ninitial)=3
                                 quick_basis%cons(Ninitial)=1.0d0
                                 elseif(JJJ.EQ.5)then
                                 quick_basis%KLMN(1,Ninitial)=2
                                 quick_basis%KLMN(3,Ninitial)=1
                                 quick_basis%cons(Ninitial)=dsqrt(5.0d0)
                                 elseif(JJJ.EQ.6)then
                                 quick_basis%KLMN(1,Ninitial)=1
                                 quick_basis%KLMN(2,Ninitial)=1
                                 quick_basis%KLMN(3,Ninitial)=1
                                 quick_basis%cons(Ninitial)=dsqrt(5.0d0)*dsqrt(3.0d0)
                                 elseif(JJJ.EQ.7)then
                                 quick_basis%KLMN(2,Ninitial)=2
                                 quick_basis%KLMN(3,Ninitial)=1
                                 quick_basis%cons(Ninitial)=dsqrt(5.0d0)
                                 elseif(JJJ.EQ.8)then
                                 quick_basis%KLMN(1,Ninitial)=1
                                 quick_basis%KLMN(3,Ninitial)=2
                                 quick_basis%cons(Ninitial)=dsqrt(5.0d0)
                                 elseif(JJJ.EQ.9)then
                                 quick_basis%KLMN(2,Ninitial)=1
                                 quick_basis%KLMN(3,Ninitial)=2
                                 quick_basis%cons(Ninitial)=dsqrt(5.0d0)
                                 elseif(JJJ.EQ.10)then
                                 quick_basis%KLMN(3,Ninitial)=3
                                 quick_basis%cons(Ninitial)=1.0d0
                              endif
                              do k=1,iprim
                                 !quick_basis%gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),quick_basis%KLMN(1,Ninitial), &
                                 !       quick_basis%KLMN(2,Ninitial),quick_basis%KLMN(3,Ninitial))
                              if(blngr_test) then
                                quick_basis%gccoeff(k,Ninitial)=BB(k)
                              else
                                quick_basis%gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),quick_basis%KLMN(1,Ninitial), &
                                        quick_basis%KLMN(2,Ninitial),quick_basis%KLMN(3,Ninitial))
                              endif
                              quick_basis%unnorm_gccoeff(k,Ninitial)=BB(k)
                                 quick_basis%gcexpo(k,Ninitial)=AA(k)
                                 if(quick_basis%gcexpomin(jshell).gt.AA(k))quick_basis%gcexpomin(jshell)=AA(k)
                              enddo
                           enddo

                           if (quick_method%ecp) then
                              kmin(jshell)=11
                              kmax(jshell)=20
                              ktypecp(jshell)=4
                           end if

                           jshell = jshell+1

                           elseif (shell == 'G') then
                           ierr = 37   

                        endif
                     enddo
                  endif
               endif
            enddo
            rewind ibasisfile
         end if

         if (atmbs2(quick_molspec%iattype(i)) .and. quick_method%ecp) then
            iofile = 0
            do while (iofile == 0)
               read(ibasiscustfile,'(A80)',iostat=iofile) line
               read(line,*,iostat=io) atom,ii
               if (io == 0 .and. ii == 0) then
                  call upcase(atom,2)
                  if (symbol(quick_molspec%iattype(i)) == atom) then
                     iatom = 0
                     do while (iatom==0)
                        read(ibasiscustfile,'(A80)',iostat=iofile) line
                        read(line,*,iostat=iatom) shell,iprim,dnorm
                        quick_basis%kprim(jshell) = iprim
                        if (shell == 'S') then
                           quick_basis%ktype(jshell) = 1
                           quick_basis%katom(jshell) = i
                           quick_basis%kstart(jshell) = jbasis
                           kmin(jshell) = 1
                           kmax(jshell) = 1
                           ktypecp(jshell)=1
                           do k=1,iprim
                              read(ibasiscustfile,'(A80)',iostat=iofile) line
                              read(line,*) aex(jbasis),gcs(jbasis)
                              jbasis = jbasis+1
                           enddo
                           jshell = jshell+1
                           elseif (shell == 'P') then
                           quick_basis%ktype(jshell) = 3
                           quick_basis%katom(jshell) = i
                           quick_basis%kstart(jshell) = jbasis
                           kmin(jshell) = 2
                           kmax(jshell) = 4
                           ktypecp(jshell)=2
                           do k=1,iprim
                              read(ibasiscustfile,'(A80)',iostat=iofile) line
                              read(line,*) aex(jbasis),gcp(jbasis)
                              jbasis = jbasis+1
                           enddo
                           jshell = jshell+1
                           elseif (shell == 'D') then
                           quick_basis%ktype(jshell) = 6
                           quick_basis%katom(jshell) = i
                           quick_basis%kstart(jshell) = jbasis
                           kmin(jshell) = 5
                           kmax(jshell) = 10
                           ktypecp(jshell)=3
                           do k=1,iprim
                              read(ibasiscustfile,'(A80)',iostat=iofile) line
                              read(line,*) aex(jbasis),gcd(jbasis)
                              jbasis = jbasis+1
                           enddo
                           jshell = jshell+1
                           elseif (shell == 'F') then
                           quick_method%hasF=.true.
                           quick_basis%ktype(jshell) = 10
                           quick_basis%katom(jshell) = i
                           quick_basis%kstart(jshell) = jbasis
                           kmin(jshell) = 11
                           kmax(jshell) = 20
                           ktypecp(jshell)=4
                           do k=1,iprim
                              read(ibasiscustfile,'(A80)',iostat=iofile) line
                              read(line,*) aex(jbasis),gcf(jbasis)
                              jbasis = jbasis+1
                           enddo
                           jshell = jshell+1
                        endif
                     enddo
                  endif
               endif
            enddo
            rewind ibasiscustfile
         end if
      999  enddo


      quick_basis%ksumtype(jshell)=Ninitial+1
      jshell=jshell-1
      jbasis=jbasis-1

      close(ibasisfile)
      close(ibasiscustfile)

      maxcontract = 1

      do i=1,nshell
         if (quick_basis%kprim(i) > maxcontract) maxcontract = quick_basis%kprim(i)
      enddo

      !======== MPI/MASTER ====================
   endif masterwork_readfile
   !======== END MPI/MASTER ================

#ifdef MPIV
   !======== MPI/ALL NODES ====================
   if (bMPI) then
      call MPI_BCAST(maxcontract,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_method%hasF,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
   endif
   !======== END MPI/ALL NODES ================
#endif

#ifndef ENABLEF   
   if(quick_method%hasF) then 
       ierr=36
       return
   endif
#endif

#ifdef CEW
   if(quick_method%hasF .and. quick_cew%use_cew) then
       ierr=38
       return
   endif
#endif

   ! Allocate the arrays now that we know the sizes
   if(.not. quick_method%hasF)then
      if(.not. allocated(Yxiao))        allocate(Yxiao(10000,56,56))
      if(.not. allocated(Yxiaotemp))    allocate(Yxiaotemp(56,56,0:10))
      if(.not. allocated(Yxiaoprim))    allocate(Yxiaoprim(MAXPRIM,MAXPRIM,56,56))
      if(.not. allocated(attraxiao))    allocate(attraxiao(56,56,0:6))
      if(.not. allocated(attraxiaoopt)) allocate(attraxiaoopt(3,56,56,0:5))
   else
      if(.not. allocated(Yxiao))        allocate(Yxiao(10000,120,120))
      if(.not. allocated(Yxiaotemp))    allocate(Yxiaotemp(120,120,0:14))
      if(.not. allocated(Yxiaoprim))    allocate(Yxiaoprim(MAXPRIM,MAXPRIM,120,120))
      if(.not. allocated(attraxiao))    allocate(attraxiao(120,120,0:8))
      if(.not. allocated(attraxiaoopt)) allocate(attraxiaoopt(3,120,120,0:7))
   endif

   Yxiao        = 0.0d0
   Yxiaotemp    = 0.0d0
   Yxiaoprim    = 0.0d0
   attraxiao    = 0.0d0
   attraxiaoopt = 0.0d0

   if(.not. allocated(aexp)) allocate(aexp(maxcontract,nbasis))

   if(.not. allocated(dcoeff)) allocate(dcoeff(maxcontract,nbasis))
   if(.not. allocated(gauss)) allocate(gauss(nbasis))

   !======== MPI/MASTER ====================
   masterwork_setup: if(master) then
      !======== END MPI/MASTER ====================

      ! do i=1,nbasis
      !     allocate(gauss(i)%aexp(maxcontract))
      !     allocate(gauss(i)%dcoeff(maxcontract))
      ! enddo

      ! Still support the old style of storing the basis but only for
      ! S,SP,P, and D
      l = 1
      do i=1,nshell
         do j=1,quick_basis%ktype(i)
            quick_basis%ncenter(l) = quick_basis%katom(i)
            ncontract(l) = quick_basis%kprim(i)
            if (quick_basis%ktype(i) == 1) then
               itype(:,l) = 0
               elseif (quick_basis%ktype(i) == 3) then
               itype(j,l) = 1
               elseif (quick_basis%ktype(i) == 4) then
               if (j> 1) then
                  itype(j-1,l) = 1
               endif
               elseif (quick_basis%ktype(i) == 6) then


               ! New Version for QUICK

               if (j==1) then
                  itype(:,l) = (/2,0,0/)
                  elseif (j==2) then
                  itype(:,l) = (/1,1,0/)
                  elseif(j==3) then
                  itype(:,l) = (/0,2,0/)
                  elseif(j==4) then
                  itype(:,l) = (/1,0,1/)
                  elseif(j==5) then
                  itype(:,l) = (/0,1,1/)
                  elseif(j==6) then
                  itype(:,l) = (/0,0,2/)
               end if

               ! Version for comparison with G03
               !
               !        if (j==1) then
               !          itype(:,l) = (/2,0,0/)
               !        elseif (j==2) then
               !          itype(:,l) = (/0,2,0/)
               !        elseif(j==3) then
               !          itype(:,l) = (/0,0,2/)
               !        elseif(j==4) then
               !          itype(:,l) = (/1,1,0/)
               !        elseif(j==5) then
               !          itype(:,l) = (/1,0,1/)
               !        elseif(j==6) then
               !          itype(:,l) = (/0,1,1/)
               !        end if

               ! Old Version
               !        if (j==1) then
               !          itype(:,l) = (/1,1,0/)
               !        elseif (j==2) then
               !          itype(:,l) = (/0,1,1/)
               !        elseif(j==3) then
               !          itype(:,l) = (/1,0,1/)
               !        elseif(j==4) then
               !          itype(:,l) = (/2,0,0/)
               !        elseif(j==5) then
               !          itype(:,l) = (/0,2,0/)
               !        elseif(j==6) then
               !          itype(:,l) = (/0,0,2/)
               !
               !       endif

               elseif (quick_basis%ktype(i) == 10) then


               ! New Version for QUICK

               if (j==1) then
                  itype(:,l) = (/3,0,0/)
                  elseif (j==2) then
                  itype(:,l) = (/2,1,0/)
                  elseif(j==3) then
                  itype(:,l) = (/1,2,0/)
                  elseif(j==4) then
                  itype(:,l) = (/0,3,0/)
                  elseif(j==5) then
                  itype(:,l) = (/2,0,1/)
                  elseif(j==6) then
                  itype(:,l) = (/1,1,1/)
                  elseif(j==7) then
                  itype(:,l) = (/0,2,1/)
                  elseif(j==8) then
                  itype(:,l) = (/1,0,2/)
                  elseif(j==9) then
                  itype(:,l) = (/0,1,2/)
                  elseif(j==10) then
                  itype(:,l) = (/0,0,3/)
               end if

            endif
            ll = 1
            do k=quick_basis%kstart(i),(quick_basis%kstart(i)+quick_basis%kprim(i))-1
               aexp(ll,l) =  aex(k)
               if (quick_basis%ktype(i) == 1) then
                     dcoeff(ll,l) = gcs(k)
                  elseif (quick_basis%ktype(i) == 3) then
                     dcoeff(ll,l) = gcp(k)
                  elseif (quick_basis%ktype(i) == 4) then
                  if (j==1) then
                     dcoeff(ll,l) = gcs(k)
                  else
                     dcoeff(ll,l) = gcp(k)
                  endif
                  elseif (quick_basis%ktype(i) == 6) then
                  dcoeff(ll,l) = gcd(k)
                  elseif (quick_basis%ktype(i) == 10) then
                  dcoeff(ll,l) = gcf(k)
               endif
               ll = ll + 1
            enddo
            l = l+1
         enddo
      enddo

      ! ifisrt and last_basis_function records the first and last basis set for atom i
      !
      iatm = 1
      is = 0
      quick_basis%first_basis_function(1) = 1
      do i=1,nshell
         is = is + quick_basis%ktype(i)
         if(quick_basis%katom(i) /= iatm) then
            iatm = quick_basis%katom(i)
            quick_basis%first_basis_function(iatm) = is
            quick_basis%last_basis_function(iatm-1) = is -1
         endif
      enddo
      quick_basis%last_basis_function(iatm) = nbasis

      !======== MPI/MASTER ====================
   endif masterwork_setup
   !======== MPI/MASTER ====================

#ifdef MPIV
   !======== MPI/ALL NODES ====================
   if (bMPI) then
      call mpi_setup_basis
      if(.not. allocated(mpi_jshelln)) allocate(mpi_jshelln(0:mpisize-1))
      if(.not. allocated(mpi_jshell)) allocate(mpi_jshell(0:mpisize-1,jshell))

      if(.not. allocated(mpi_nbasisn)) allocate(mpi_nbasisn(0:mpisize-1))
      if(.not. allocated(mpi_nbasis)) allocate(mpi_nbasis(0:mpisize-1,nbasis))

   endif
   !======== END MPI/ALL NODES ====================
#endif

   if (quick_method%debug.and.master) call debugBasis
   
  if (allocated(aex)) deallocate(aex)
  if (allocated(gcs)) deallocate(gcs)
  if (allocated(gcp)) deallocate(gcp)
  if (allocated(gcd)) deallocate(gcd)
  if (allocated(gcf)) deallocate(gcf)
  if (allocated(gcg)) deallocate(gcg)
end subroutine

subroutine store_basis_to_ecp()
   use quick_basis_module
   use quick_ecp_module
   integer iicont,icontb
   iicont=0
   icontb=1
   do i=1,nshell
      do j=1,quick_basis%kprim(i)
         iicont=iicont+1
         eta(iicont)=dcoeff(j,icontb)
      end do
      icontb=icontb+quick_basis%ktype(i)
   end do
end subroutine



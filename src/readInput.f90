! --Yipu Miao 05/09/2010
!********************************************************
! This file lists subroutines to read information from inputfile
! Subroutine List:
! getAtoms(iAtomType)
! readBasis
! readJob
! readecp
! readpdb
!
!********************************************************
! getAtoms(iAtomType)
!--------------------------------------------------------
! Be aware of atomdenssize
! Ken Ayers. 05/26/04
! Subroutines for calculating sizes of the various matrixes in quick
!
    subroutine getAtoms()
    use allmod
    implicit double precision (a-h,o-z)

    character(len=80) :: keyWD

    open(infile,file=inFileName,status='old')
    call PrtAct(iOutFile," Read Atom types ")
    
    iStat = 0
    iAtomType = 1
    do while(iStat.ne.-1)
        iStart = 1
        iFinal = 80
        read (infile,'(A80)',ioStat=iStat) keyWD
        call upcase(keyWD,80)
        call rdword(keyWD,iStart,iFinal)
        if (iStart.ne.0) then
        do i=0,71
            if (keyWD(iStart:iFinal) == symbol(i)) then
                nAtom = nAtom + 1

                if(iAtomType.gt.1)then
                  do itemp=1,iAtomType-1
                    if(symbol(i).eq.atom_type_sym(itemp))goto 111
                  enddo
                endif
                     atom_type_sym(iAtomType)=symbol(i)
                     iAtomType=iAtomType+1                   
111         endif
        enddo
        endif
    enddo

    iAtomType=iAtomType-1
    write(iOutFile,'(a,i4)') "ATOM TYPES=",iAtomType
    write(iOutFile,'(a,i4)') "ATOM NUMBER=",nAtom
    nAtomSave=nAtom
    close(infile)

    end subroutine getatoms
!********************************************************

  Subroutine readecp
!
!********************************************************
! readecp
!--------------------------------------------------------
! Subroutines to read ecp
!
!
! Alessandro GENONI 03/05/2007
! Subroutine to  read the Effective Core Potentials
! The total number of electrons and the nuclear charges are modified too
!
  use quick_constants_module
  use quick_files_module
  use quick_molspec_module
  use quick_ecp_module

  implicit double precision(a-h,o-z)
  character(len=80) :: line
  character(len=2)  :: atom
  character(len=3)  :: pot
  integer, dimension(0:92) :: klmaxecp,kelecp,kprimecp
  logical, dimension(0:92) :: warn 

  open(iecpfile,file=ecpfilename,status='old')

  iofile = 0
  necprim=0
  nelecp=0
  lmaxecp=0
  klmaxecp=0
  kelecp=0
  kprimecp=0
  warn=.true.
 
! Parse the file and find the sizes of arrays to allocate them in memory

  do while (iofile == 0)
    read(iecpfile,'(A80)',iostat=iofile) line
    read(line,*,iostat=io) atom,ii
    if (io == 0 .and. ii == 0) then
      do i=1,92
        if (symbol(i) == atom) then
          iat=i
          warn(i)=.false.
        end if
      end do 
      read(iecpfile,'(A80)',iostat=iofile) line
      read(line,*,iostat=iatom) klmaxecp(iat),kelecp(iat) 
      if (iatom == 0) then
        do while (iatom == 0) 
          read(iecpfile,'(A80)',iostat=iofile) line
          read(line,*,iostat=iatom) iprim,pot
          if (iatom == 0) then
            kprimecp(iat)=kprimecp(iat)+iprim
            do i=1,iprim
              read(iecpfile,'(A80)',iostat=iofile) line
              read(line,*) n,c1,c2
            end do 
          end if        
        end do
      end if
    end if
  end do

  rewind iecpfile

  do i=1,natom
    lmaxecp(i)=klmaxecp(iattype(i))
    nelecp(i)=kelecp(iattype(i))
    chg(i)=chg(i)-nelecp(i)
    necprim=necprim+kprimecp(iattype(i))
    nelec=nelec-nelecp(i)
  end do

!
! Allocation of the arrays whose dimensions depend on NECPRIM
! and of the arrays KFIRST and KLAST
! 
  allocate(clp(necprim))
  allocate(zlp(necprim))
  allocate(nlp(necprim))
  allocate(kfirst(mxproj+1,natom))
  allocate(klast(mxproj+1,natom))
!
! Store the vectors CLP,NLP,ZLP,KFIRST,KLAST
!
  clp=0
  nlp=0
  zlp=0
  kfirst=0
  klast=0
!
  jecprim=0
  do i=1,natom
    iofile=0
    do while (iofile == 0) 
      read(iecpfile,'(A80)',iostat=iofile) line
      read(line,*,iostat=io) atom,ii
      if (io == 0 .and. ii == 0) then
        if (symbol(iattype(i)) == atom) then   
          iatom=0
          do while (iatom==0)
            read(iecpfile,'(A80)',iostat=iofile) line
            read(line,*,iostat=iatom) klmax,nelecore
            if (iatom == 0) then
              jjcont=0
              do while (iatom == 0) 
                read(iecpfile,'(A80)',iostat=iofile) line
                read(line,*,iostat=iatom) iprim,pot
                jjcont=jjcont+1
                if (iatom == 0) then
                  kfirst(jjcont,i)=jecprim+1
                  do j=1,iprim
                    jecprim=jecprim+1
                    read(iecpfile,'(A80)',iostat=iofile) line
                    read(line,*) nlp(jecprim),zlp(jecprim),clp(jecprim)
                  end do
                  klast(jjcont,i)=jecprim
                end if
              end do
            end if
          end do
        end if
      end if
    end do
    rewind iecpfile
!
! Check if the selected ECP exists for each atom in the molecule
!
    if (warn(iattype(i))) then
      write(ioutfile,'("  ")')
      write(ioutfile,'("WARNING: NO ECP FOR ATOM ",A2,I4)') symbol(iattype(i)),i
    end if
!
  end do

  return
 end subroutine

!
!********************************************************
! readbasis
!--------------------------------------------------------
! Subroutines to read basis set
!
!Xiao HE, normalization of D F orbitals

subroutine readbasis(natomxiao,natomstart,natomfinal,nbasisstart,nbasisfinal)
!
! Read in the requested basisfile. This is done twice, once to get sizes and 
! allocate variables, then again to assign the basis 
!
 use allmod
!
 implicit double precision(a-h,o-z)
 character(len=80) :: line
 character(len=2) :: atom,shell
 logical :: isatom
 integer, dimension(0:92)  :: kcontract,kbasis
 logical, dimension(0:92)  :: atmbs,atmbs2
 real*8 AA(8),BB(8),CC(8)
 integer natomstart,natomfinal,nbasisstart,nbasisfinal
 
 include 'mpif.h'
 
 
! =============MPI/ MASTER========================
masterwork: if (master) then
! =============END MPI/MASTER=====================

 open(ibasisfile,file=basisfilename,status='old')
 iofile = 0
 nshell = 0
 nbasis = 0
 nprim = 0
 kcontract = 0
 kshell = 0
 kbasis = 0
 atmbs=.true.
 atmbs2=.true.
 icont=0
 quick_method%ffunxiao=.true.

! parse the file and find the sizes of things to allocate them in memory

 do while (iofile  == 0 ) 
   read(ibasisfile,'(A80)',iostat=iofile) line
   read(line,*,iostat=io) atom,ii
   if (io == 0 .and. ii == 0) then
     isatom = .true.
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
         kshell(iat) = kshell(iat) +1
         kcontract(iat) = kcontract(iat) + iprim
         if (shell == 'S') then
            kbasis(iat) = kbasis(iat) + 1
         elseif (shell == 'P') then
            kbasis(iat) = kbasis(iat) + 3
         elseif (shell == 'SP') then
            kbasis(iat) = kbasis(iat) + 4
         elseif (shell == 'D') then 
            kbasis(iat) = kbasis(iat) + 6
         elseif (shell == 'F') then 
            quick_method%ffunxiao=.false.
            kbasis(iat) = kbasis(iat) + 10
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
     if (atmbs(iattype(jj))) then
       atmbs(iattype(jj))=.false.
       iofile=0
       do while (iofile == 0)
         read(ibasiscustfile,'(A80)',iostat=iofile) line
         read(line,*,iostat=io) atom,ii   
         if (io == 0 .and. ii == 0) then
           if (symbol(iattype(jj)) == atom) then  
             iat=iattype(jj)  
             iatom=0
             do while (iatom == 0)
               read(ibasiscustfile,'(A80)',iostat=iofile) line
               read(line,*,iostat=iatom) shell,iprim,dnorm
               if (iatom == 0) then
                 kshell(iat) = kshell(iat) +1
                 kcontract(iat) = kcontract(iat) + iprim
                 if (shell == 'S') then
                   kbasis(iat) = kbasis(iat) + 1
                 elseif (shell == 'P') then
                   kbasis(iat) = kbasis(iat) + 3
                 elseif (shell == 'SP') then
                   kbasis(iat) = kbasis(iat) + 4
                 elseif (shell == 'D') then
                   kbasis(iat) = kbasis(iat) + 6
                 elseif (shell == 'F') then
                   quick_method%ffunxiao=.false.
                   kbasis(iat) = kbasis(iat) + 10
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
!  if (kshell(i) /= 0) print *, symbol(i),kshell(i),kcontract(i),kbasis(i)
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

!   print*,nshell,nbasis,nprim,kshell(iattype(i)),kbasis(iattype(i)),kcontract(iattype(i))
   nshell = nshell + kshell(iattype(i))
   nbasis = nbasis + kbasis(iattype(i))
   nprim = nprim + kcontract(iattype(i))

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


! =============END MPI/ALL NODES=====================
if (bMPI) then
    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(nshell,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)  
    call MPI_BCAST(nbasis,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)  
    call MPI_BCAST(nprim,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)   
    call MPI_BCAST(quick_method%ffunxiao,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
    call MPI_BCAST(kshell,93,mpi_integer,0,MPI_COMM_WORLD,mpierror)
    call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
endif


! =============END MPI/ALL NODES=====================

! Allocate the arrays now that we know the sizes

if(quick_method%ffunxiao)then
 allocate(Yxiao(12960,56,56))
 allocate(Yxiaotemp(56,56,0:10))
 allocate(Yxiaoprim(8,8,56,56))
 allocate(attraxiao(56,56,0:8))
 allocate(attraxiaoopt(3,56,56,0:5))
else
 allocate(Yxiao(12960,120,120))
 allocate(Yxiaotemp(120,120,0:14))
 allocate(Yxiaoprim(8,8,120,120))
 allocate(attraxiao(120,120,0:8))
 allocate(attraxiaoopt(3,120,120,0:7))
endif

! allocate(Yxiao(1296,35,35))
! allocate(Yxiaotemp(35,35,0:8))
! allocate(Yxiaoprim(6,6,35,35))
!  allocate(Yxiao(81,10,10))
!  allocate(Yxiaotemp(10,10,0:4))
  allocate(Ycutoff(nshell,nshell))
  allocate(cutmatrix(nshell,nshell))
  allocate(allerror(quick_method%maxdiisscf,nbasis,nbasis))
  allocate(alloperator(quick_method%maxdiisscf,nbasis,nbasis))
!  allocate(debug1(nbasis,nbasis))
!  allocate(debug2(nbasis,nbasis))
! allocate(CPMEM(10,10,0:4))
! allocate(MEM(10,10,0:4)) 
 allocate(kstart(nshell))
 allocate(katom(nshell))
 allocate(ktype(nshell))
 allocate(kprim(nshell))
 allocate(Qnumber(nshell))
 allocate(Qstart(nshell))
 allocate(Qfinal(nshell))
 allocate(Qsbasis(nshell,0:3))
 allocate(Qfbasis(nshell,0:3)) 
 allocate(ksumtype(nshell+1))
 allocate(KLMN(3,nbasis))
 allocate(cons(nbasis))
 allocate(gccoeff(8,nbasis))
 allocate(gcexpo(8,nbasis))
 allocate(gcexpomin(nshell))
 allocate(aex(nprim ))
 allocate(gcs(nprim ))
 allocate(gcp(nprim ))
 allocate(gcd(nprim ))
 allocate(gcf(nprim ))
 allocate(gcg(nprim ))

 do ixiao=1,nshell
   gcexpomin(ixiao)=99999.0d0
 enddo 

 nbf12=nbasis*(nbasis+1)/2
 
 if (quick_method%ecp) then
   allocate(kmin(nshell))
   allocate(kmax(nshell))
   allocate(eta(nprim))
   allocate(ecp_int(nbf12))
   allocate(kvett(nbf12))
   allocate(gout(25*25))
   allocate(ktypecp(nshell))
!
   allocate(zlm(lmxdim))
   allocate(flmtx(len_fac,3))
   allocate(lf(lfdim))
   allocate(lmf(lmfdim))
   allocate(lml(lmfdim))
   allocate(lmx(lmxdim))
   allocate(lmy(lmxdim))
   allocate(lmz(lmxdim))
   allocate(mc(mc1dim,3))
   allocate(mr(mc1dim,3))
   allocate(dfac(len_dfac))
   allocate(dfaci(len_dfac))
   allocate(factorial(len_fac))
   allocate(fprod(lfdim,lfdim))
!
   call vett
 end if


!
! Support for old memory model, to be deleted eventually

!allocate(aexp(maxcontract,nbasis))
!allocate(dcoeff(maxcontract,nbasis))
!allocate(gauss(nbasis))
!do i=1,nbasis
!    allocate(gauss(i)%aexp(maxcontract))
!    allocate(gauss(i)%dcoeff(maxcontract))
!enddo
 allocate(itype(3,nbasis))
 allocate(ncenter(nbasis))
 allocate(ncontract(nbasis))

itype = 0
ncenter = 0
ncontract = 0

! various arrays that depend on the # of basis functions

 call allocate_quick_gridpoints(nbasis)

 allocate(Smatrix(nbasis,nbasis))
 allocate(X(nbasis,nbasis))
 allocate(O(nbasis,nbasis))
 allocate(CO(nbasis,nbasis))
 allocate(COB(nbasis,nbasis))
 allocate(VEC(nbasis,nbasis))
 allocate(DENSE(nbasis,nbasis))
 allocate(DENSEB(nbasis,nbasis))
 allocate(DENSEOLD(nbasis,nbasis))
 allocate(DENSESAVE(nbasis,nbasis))
 allocate(Osave(nbasis,nbasis))
 allocate(Osavedft(nbasis,nbasis))
 allocate(V2(3,nbasis))
 allocate(E(nbasis))
 allocate(EB(nbasis))
 allocate(idegen(nbasis))
! xiao He may reconsider this
 allocate(Uxiao(nbasis,nbasis))
! allocate(CPHFA(2*(maxbasis/2)**2,2*(maxbasis/2)**2))
! allocate(CPHFB(2*(maxbasis/2)**2,maxatm*3))
! allocate( B0(2*(maxbasis/2)**2))
! allocate(BU(2*(maxbasis/2)**2))

 allocate(hold(nbasis,nbasis))
 allocate(hold2(nbasis,nbasis))

! do this the stupid way for now

 jbasis=1
 jshell=1
 Ninitial=0
 do i=1,nbasis
   do j=1,3
     KLMN(j,i)=0
   enddo
 enddo

!====== MPI/MASTER ====================
masterwork_readfile: if (master) then
!====== END MPI/MASTER ================

 do i=1,natomxiao
   if (.not. atmbs2(iattype(i))) then
     iofile = 0
!       write(ioutfile,*) 'i = ',i
     do while (iofile == 0) 
!       write(ioutfile,*) 'EXAMPLE'
       read(ibasisfile,'(A80)',iostat=iofile) line
!       write(ioutfile,*) line
       read(line,*,iostat=io) atom,ii
       if (io == 0 .and. ii == 0) then
         if (symbol(iattype(i)) == atom) then
!           write(ioutfile,*) 'COEFFICIENTS=',line
           iatom = 0
           do while (iatom==0)
             read(ibasisfile,'(A80)',iostat=iofile) line
             read(line,*,iostat=iatom) shell,iprim,dnorm
!             write(ioutfile,*)'COEFFICIENTS=',shell,iprim,dnorm
             if(jshell.le.nshell)then
               kprim(jshell) = iprim
             endif
             if (shell == 'S') then
               ktype(jshell) = 1
               katom(jshell) = i
               kstart(jshell) = jbasis
               Qnumber(jshell) = 0
               Qstart(jshell)= 0
               Qfinal(jshell)=0
               Qsbasis(jshell,0)=0
               Qfbasis(jshell,0)=0
               ksumtype(jshell)=Ninitial+1
               Ninitial=Ninitial+1
               cons(Ninitial)=1.0d0
               if (quick_method%ecp) then
                 kmin(jshell)=1
                 kmax(jshell)=1
                 ktypecp(jshell)=1
               end if
               do k=1,iprim
                 read(ibasisfile,'(A80)',iostat=iofile) line
!                 read(line,*) aex(jbasis),gcs(jbasis)
                  read(line,*) AA(k),BB(k)
                  aex(jbasis)=AA(k)
                  gcs(jbasis)=BB(k)
                 jbasis = jbasis+1
!                 print*,iprim,BB(k),AA(k),xnorm(AA(k),0,0,0)
                 gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),0,0,0)
!                 print*,gccoeff(k,Ninitial)
                 gcexpo(k,Ninitial)=AA(k)
                 if(gcexpomin(jshell).gt.AA(k))gcexpomin(jshell)=AA(k)
               enddo
                 xnewtemp=xnewnorm(0,0,0,iprim,gccoeff(1:iprim,Ninitial),gcexpo(1:iprim,Ninitial))
!                 print*,xnewtemp
                 do k=1,iprim
                   gccoeff(k,Ninitial)=xnewtemp*gccoeff(k,Ninitial)
!                 print*,'new=',gccoeff(k,Ninitial)

                 enddo
               jshell = jshell+1
             elseif (shell == 'P') then
               ktype(jshell) = 3
               katom(jshell) = i
               kstart(jshell) = jbasis 
               Qnumber(jshell) = 6
               Qstart(jshell)= 1
               Qfinal(jshell)= 1
               Qsbasis(jshell,1)=0
               Qfbasis(jshell,1)=2
               ksumtype(jshell)=Ninitial+1
!                 do jjj=1,3
!                   Ninitial=Ninitial+1
!                   cons(Ninitial)=1.0d0
!                   KLMN(JJJ,Ninitial)=1
!                 enddo
                 do k=1,iprim
                   read(ibasisfile,'(A80)',iostat=iofile) line
!                   read(line,*) aex(jbasis),gcp(jbasis)
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
                 cons(Ninitial)=1.0d0
                 KLMN(JJJ,Ninitial)=1   
                 do k=1,iprim
                    gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),1,0,0)
                    gcexpo(k,Ninitial)=AA(k)
                 if(gcexpomin(jshell).gt.AA(k))gcexpomin(jshell)=AA(k)
                 enddo
               enddo
                 xnewtemp=xnewnorm(1,0,0,iprim,gccoeff(:,Ninitial),gcexpo(:,Ninitial))
                 do iitemp=Ninitial-2,Ninitial
                   do k=1,iprim
                     gccoeff(k,iitemp)=xnewtemp*gccoeff(k,iitemp)
                   enddo
                 enddo

               jshell = jshell+1
             elseif (shell == 'SP') then
               ktype(jshell) = 4
               katom(jshell) = i
               kstart(jshell) = jbasis 
               Qnumber(jshell) = 1
               Qstart(jshell)= 0
               Qfinal(jshell)=1
               Qsbasis(jshell,0)=0
               Qfbasis(jshell,0)=0
               Qsbasis(jshell,1)=1
               Qfbasis(jshell,1)=3

               ksumtype(jshell)=Ninitial+1
!                 do jjj=1,3
!                   Ninitial=Ninitial+1
!                   cons(Ninitial)=1.0d0
!                   KLMN(JJJ,Ninitial)=1
!                 enddo
               do k=1,iprim
                 read(ibasisfile,'(A80)',iostat=iofile) line
!                 read(line,*) aex(jbasis),gcs(jbasis),gcp(jbasis)
                 read(line,*) AA(k),BB(k),CC(k)
                 aex(jbasis)=AA(k)
                 gcs(jbasis)=BB(k)
                 gcp(jbasis)=CC(k)  
                jbasis = jbasis+1
               enddo
               Ninitial=Ninitial+1
               cons(Ninitial)=1.0d0
               do k=1,iprim
                 gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),0,0,0)
                 gcexpo(k,Ninitial)=AA(k)
                 if(gcexpomin(jshell).gt.AA(k))gcexpomin(jshell)=AA(k)
               enddo
                 xnewtemp=xnewnorm(0,0,0,iprim,gccoeff(1:iprim,Ninitial),gcexpo(1:iprim,Ninitial))
                   do k=1,iprim
                     gccoeff(k,Ninitial)=xnewtemp*gccoeff(k,Ninitial)
                   enddo

               do jjj=1,3
                 Ninitial=Ninitial+1
                 cons(Ninitial)=1.0d0
                 KLMN(JJJ,Ninitial)=1
                 do k=1,iprim
                    gccoeff(k,Ninitial)=CC(k)*xnorm(AA(k),1,0,0)
                    gcexpo(k,Ninitial)=AA(k)
                 if(gcexpomin(jshell).gt.AA(k))gcexpomin(jshell)=AA(k)
                 enddo
               enddo  
                 xnewtemp=xnewnorm(1,0,0,iprim,gccoeff(1:iprim,Ninitial),gcexpo(1:iprim,Ninitial))
                 do iitemp=Ninitial-2,Ninitial
                   do k=1,iprim
                     gccoeff(k,iitemp)=xnewtemp*gccoeff(k,iitemp)
                   enddo
                 enddo
 
               jshell = jshell+1
             elseif (shell == 'D') then
               ktype(jshell) = 6
               katom(jshell) = i
               kstart(jshell) = jbasis
               Qnumber(jshell) = 2
               Qstart(jshell)= 2
               Qfinal(jshell)= 2
               Qsbasis(jshell,2)=0
               Qfbasis(jshell,2)=5

               ksumtype(jshell)=Ninitial+1
                 do k=1,iprim
                   read(ibasisfile,'(A80)',iostat=iofile) line
!                   read(line,*) aex(jbasis),gcd(jbasis)
                    read(line,*) AA(k),BB(k)
                    aex(jbasis)=AA(k)
                    gcd(jbasis)=BB(k)
                    jbasis=jbasis+1
                 enddo
                do JJJ=1,6
                  Ninitial=Ninitial+1
                  if(JJJ.EQ.1)then
                    KLMN(1,Ninitial)=2
                    CONS(Ninitial)=1.0D0
                   elseif(JJJ.EQ.2)then
                    KLMN(1,Ninitial)=1
                    KLMN(2,Ninitial)=1
                    CONS(Ninitial)=dsqrt(3.0d0)
                   elseif(JJJ.EQ.3)then
                    KLMN(2,Ninitial)=2
                    CONS(Ninitial)=1.0D0
                   elseif(JJJ.EQ.4)then
                    KLMN(1,Ninitial)=1
                    KLMN(3,Ninitial)=1
                    CONS(Ninitial)=dsqrt(3.0d0)
                   elseif(JJJ.EQ.5)then
                    KLMN(2,Ninitial)=1
                    KLMN(3,Ninitial)=1
                    CONS(Ninitial)=dsqrt(3.0d0)
                   elseif(JJJ.EQ.6)then
                    KLMN(3,Ninitial)=2
                    CONS(Ninitial)=1.0d0
                  endif

                  do k=1,iprim
                    gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),KLMN(1,Ninitial),KLMN(2,Ninitial),KLMN(3,Ninitial))
                    gcexpo(k,Ninitial)=AA(k)
                 if(gcexpomin(jshell).gt.AA(k))gcexpomin(jshell)=AA(k)
                  enddo
                enddo
               if (quick_method%ecp) then
                 kmin(jshell)=5
                 kmax(jshell)=10
                 ktypecp(jshell)=3
               end if
                 xnewtemp=xnewnorm(2,0,0,iprim,gccoeff(:,Ninitial-3),gcexpo(:,Ninitial-3))
                 do iitemp=Ninitial-5,Ninitial-3
                   do k=1,iprim
                     gccoeff(k,iitemp)=xnewtemp*gccoeff(k,iitemp)
                   enddo
                 enddo
                 xnewtemp=xnewnorm(1,1,0,iprim,gccoeff(:,Ninitial),gcexpo(:,Ninitial))
                 do iitemp=Ninitial-2,Ninitial
                   do k=1,iprim
                     gccoeff(k,iitemp)=xnewtemp*gccoeff(k,iitemp)
                   enddo
                 enddo

!               do k=1,iprim
!                 read(ibasisfile,'(A80)',iostat=iofile) line
!                 read(line,*) aex(jbasis),gcd(jbasis)
!                 jbasis = jbasis+1
!               enddo
               jshell = jshell+1
             elseif (shell == 'F') then
               ktype(jshell) = 10
               katom(jshell) = i
               kstart(jshell) = jbasis
               Qnumber(jshell) = 3
               Qstart(jshell)= 3
               Qfinal(jshell)= 3
               Qsbasis(jshell,3)=0
               Qfbasis(jshell,3)=9

               ksumtype(jshell)=Ninitial+1
                 do k=1,iprim
                   read(ibasisfile,'(A80)',iostat=iofile) line
!                   read(line,*) aex(jbasis),gcf(jbasis)
                    read(line,*) AA(k),BB(k)
                    aex(jbasis)=AA(k)
                    gcf(jbasis)=BB(k)
                    jbasis=jbasis+1
                 enddo
                do JJJ=1,10
                  Ninitial=Ninitial+1
                  if(JJJ.EQ.1)then
                    KLMN(1,Ninitial)=3
                    CONS(Ninitial)=1.0D0
                   elseif(JJJ.EQ.2)then
                    KLMN(1,Ninitial)=2
                    KLMN(2,Ninitial)=1
                    CONS(Ninitial)=dsqrt(5.0d0)
                   elseif(JJJ.EQ.3)then
                    KLMN(1,Ninitial)=1
                    KLMN(2,Ninitial)=2
                    CONS(Ninitial)=dsqrt(5.0d0)
                   elseif(JJJ.EQ.4)then
                    KLMN(2,Ninitial)=3
                    CONS(Ninitial)=1.0d0
                   elseif(JJJ.EQ.5)then
                    KLMN(1,Ninitial)=2
                    KLMN(3,Ninitial)=1
                    CONS(Ninitial)=dsqrt(5.0d0)
                   elseif(JJJ.EQ.6)then
                    KLMN(1,Ninitial)=1
                    KLMN(2,Ninitial)=1
                    KLMN(3,Ninitial)=1
                    CONS(Ninitial)=dsqrt(5.0d0)*dsqrt(3.0d0)
                   elseif(JJJ.EQ.7)then
                    KLMN(2,Ninitial)=2
                    KLMN(3,Ninitial)=1
                    CONS(Ninitial)=dsqrt(5.0d0)
                   elseif(JJJ.EQ.8)then
                    KLMN(1,Ninitial)=1
                    KLMN(3,Ninitial)=2
                    CONS(Ninitial)=dsqrt(5.0d0)
                   elseif(JJJ.EQ.9)then
                    KLMN(2,Ninitial)=1
                    KLMN(3,Ninitial)=2
                    CONS(Ninitial)=dsqrt(5.0d0)
                   elseif(JJJ.EQ.10)then
                    KLMN(3,Ninitial)=3
                    CONS(Ninitial)=1.0d0
                  endif
                  do k=1,iprim
                    gccoeff(k,Ninitial)=BB(k)*xnorm(AA(k),KLMN(1,Ninitial),KLMN(2,Ninitial),KLMN(3,Ninitial))
                    gcexpo(k,Ninitial)=AA(k)
                 if(gcexpomin(jshell).gt.AA(k))gcexpomin(jshell)=AA(k)
                  enddo
               enddo

               if (quick_method%ecp) then
                 kmin(jshell)=11
                 kmax(jshell)=20
                 ktypecp(jshell)=4
               end if

               jshell = jshell+1
             endif
           enddo
         endif
       endif
     enddo
     rewind ibasisfile
   end if

   if (atmbs2(iattype(i)) .and. quick_method%ecp) then
     iofile = 0
     do while (iofile == 0)
       read(ibasiscustfile,'(A80)',iostat=iofile) line
       read(line,*,iostat=io) atom,ii
       if (io == 0 .and. ii == 0) then
         if (symbol(iattype(i)) == atom) then
           iatom = 0
           do while (iatom==0)
             read(ibasiscustfile,'(A80)',iostat=iofile) line
             read(line,*,iostat=iatom) shell,iprim,dnorm
             kprim(jshell) = iprim
             if (shell == 'S') then
               ktype(jshell) = 1
               katom(jshell) = i
               kstart(jshell) = jbasis
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
               ktype(jshell) = 3
               katom(jshell) = i
               kstart(jshell) = jbasis
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
               ktype(jshell) = 6
               katom(jshell) = i
               kstart(jshell) = jbasis
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
               ktype(jshell) = 10
               katom(jshell) = i
               kstart(jshell) = jbasis
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


 ksumtype(jshell)=Ninitial+1
 jshell=jshell-1
 jbasis=jbasis-1

 close(ibasisfile)
 close(ibasiscustfile)

 maxcontract = 1

 do i=1,nshell
    if (kprim(i) > maxcontract) maxcontract = kprim(i)
 enddo

!======== MPI/MASTER ====================
endif masterwork_readfile
!======== END MPI/MASTER ================


!======== MPI/ALL NODES ====================
if (bMPI) then
    call MPI_BCAST(maxcontract,1,mpi_integer,0,MPI_COMM_WORLD,mpierror) 
endif
!======== END MPI/ALL NODES ================

! print*,maxcontract,nbasis

 allocate(aexp(maxcontract,nbasis))
 allocate(dcoeff(maxcontract,nbasis))
 allocate(gauss(nbasis))

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
   do j=1,ktype(i)
     ncenter(l) = katom(i)
     ncontract(l) = kprim(i)
     if (ktype(i) == 1) then
       itype(:,l) = 0
     elseif (ktype(i) == 3) then
       itype(j,l) = 1
     elseif (ktype(i) == 4) then
        if (j> 1) then
          itype(j-1,l) = 1
        endif
     elseif (ktype(i) == 6) then


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

     elseif (ktype(i) == 10) then


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
     do k=kstart(i),(kstart(i)+kprim(i))-1
      aexp(ll,l) =  aex(k) 
      if (ktype(i) == 1) then
        dcoeff(ll,l) = gcs(k)
      elseif (ktype(i) == 3) then
        dcoeff(ll,l) = gcp(k)
      elseif (ktype(i) == 4) then
         if (j==1) then
           dcoeff(ll,l) = gcs(k)
         else
           dcoeff(ll,l) = gcp(k)
         endif
      elseif (ktype(i) == 6) then
        dcoeff(ll,l) = gcd(k)
      elseif (ktype(i) == 10) then
        dcoeff(ll,l) = gcf(k)
      endif
      ll = ll + 1
     enddo
     l = l+1
   enddo
 enddo

! ifisrt and ilast records the first and last basis set for atom i
!
 iatm = 1
 is = 0
 ifirst(1) = 1
 do i=1,nshell
   is = is + ktype(i)
   if(katom(i) /= iatm) then
      iatm = katom(i)
      ifirst(iatm) = is
      ilast(iatm-1) = is -1
   endif
 enddo
 ilast(iatm) = nbasis

!======== MPI/MASTER ====================
endif masterwork_setup
!======== MPI/MASTER ====================

!======== MPI/ALL NODES ====================
if (bMPI) then
    call mpi_setup_basis
    allocate(mpi_jshelln(0:mpisize-1))
    allocate(mpi_jshell(0:mpisize-1,jshell))
    
    allocate(mpi_nbasisn(0:mpisize-1))
    allocate(mpi_nbasis(0:mpisize-1,nbasis))
    
endif
!======== END MPI/ALL NODES ====================

end subroutine

!
!********************************************************
! readJob
!--------------------------------------------------------
! Subroutines to read job
!
    subroutine readJob
    use allmod
    implicit double precision (a-h,o-z)

    character(len=200) :: keyWD
    character(len=20) :: tempstring

if (master) then
    istart = 1
    ifinal = 80
    ibasisstart = 1
    ibasisend = 80

! AG 03/05/2007
    iecpstart=1
    iecpend=80

    molchg=100
    imult=0

! AG 03/05/2007
    itolecp=0


    open(infile,file=inFileName,status='old')
    call PrtAct(iOutFile,"Read Job Type")
    
    read (inFile,'(A200)') keyWD
    call upcase(keyWD,200)
    write(iOutFile,*) " -------------------------------------"
    write(iOutFile,'("KEYWORD=",a)') keyWD
    write(iOutFile,*) " -------------------------------------"
    write(iOutFile,*)

    call rdword(basisdir,ibasisstart,ibasisend)
    call rdword(ecpdir,iecpstart,iecpend) !AG 03/05/2007
    
    ! read method
    call read(quick_method,keyWD)
        
    ! read mol
    call read(quick_molspec,inFile)
    call alloc(quick_molspec)
    call read2(quick_molspec,inFile)  
    call set(quick_molspec)
    
    ! then print
    call print(quick_method,iOutFile)
    call print(quick_molspec,iOutFile)
    
    close(inFile)

!------------------------------------------------------------------
! ECP Basis
! Alessandro GENONI 03/07/2007
!
    if (index(keywd,'ECP=') /= 0) then
       iecp = index(keywd,'ECP=')
       call rdword(keywd,iecp,ifinal)
       ecpfilename = ecpdir(1:iecpend) // '/' // keywd(iecp+4:ifinal)
       basisfilename = basisdir(1:ibasisend) // '/' // keywd(iecp+4:ifinal)
       if (keywd(iecp+4:ifinal) == 'CUSTOM') then
       else
          basiscustname = basisdir(1:ibasisend) // '/CUSTOM'
       END if
    endif
 

!------------------------------------------------------------------
! Gaussian Basis
! Alessandro GENONI 03/07/2007
!
    if (index(keywd,'BASIS=') /= 0) then
       ibasis = index(keywd,'BASIS=')
       call rdword(keywd,ibasis,ifinal)
       basisfilename = basisdir(1:ibasisend) // '/' // keywd(ibasis+6:ifinal)
       write(iOutFile,'("BASIS FILE=",a)') basisfilename
       if (keywd(ibasis+6:ifinal) == 'CUSTOM') then
          write (iOutFile,'("THE BASIS-SET WILL BE READ FROM THE CUSTOM FILE")')
       END if
    endif

! Charge
!
!        if (index(keywd,'CHARGE=') /= 0) then
!            istrt = index(keywd,'CHARGE=')+7
!            iend=istrt+index(keywd(istrt:80),' ')
!            tempstring = keywd(istrt:iend)
!            call rdinum(tempstring,1,molchg,ierror)
!            write (iOutFile,'("TOTAL MOLECULAR CHARGE = ",I4)') molchg
!        endif
!------------------------------------------------------------------
! Multiplicity
!
!        if (index(keywd,'MULT=') /= 0) then
!            istrt = index(keywd,'MULT=')+5
!            iend=istrt+index(keywd(istrt:80),' ')
!            tempstring = keywd(istrt:iend)
!            call rdinum(tempstring,1,imult,ierror)
!            write (iOutFile,'("MULTIPLICITY = ",I4)') imult
!        endif
        
        
!------------------------------------------------------------------
! Experimental Solvantion energy
!
        if (index(keywd,'GSOL=') /= 0) then
            istrt = index(keywd,'GSOL=')+4
            call rdnum(keywd,istrt,Gsolexp,ierror)
            write (iOutFile,'("Experimental solvation energy = ",E10.3)') Gsolexp
        endif
!------------------------------------------------------------------
! ECP integrals prescreening
! Alessandro GENONI 03/05/2007
!
        if (index(keywd,'TOL_ECPINT=') /= 0) then
          istrt = index(keywd,'TOL_ECPINT=')+10
          call rdinum(keywd,istrt,itolecp,ierror)
          tolecp=2.30258d+00*itolecp
          thrshecp=10.0d0**(-1.0d0*itolecp)
          write(iOutFile,'("THRESHOLD FOR ECP-INTEGRALS PRESCREENING = ", &
          & E15.5)')thrshecp
        END if


!------------------------------------------------------------------
! If some quantities have not been set, define a default.

    if (molchg == 100) then
        write (iOutFile,'("TOTAL MOLECULAR CHARGE =  0 (DEFAULT)")')
        molchg=0
    endif

    if (imult == 0) then
        write (iOutFile,'("MULTIPLICITY = 1 (DEFAULT)")')
        imult=1
    endif

    if (ibasis == 2 .AND. .NOT. quick_method%core) then
        write (iOutFile,'("VO BASIS FUNCTIONS ARE USED WITH", &
        & " THE CORE APPROXIMATION.")')
        quick_method%core=.true.
    endif

    if (imult /= 1) quick_method%UNRST= .TRUE. 

    ! Alessandro GENONI 03/05/2007
    if (itolecp == 0) then
      itolecp=12
      tolecp=2.30258d+00*itolecp
      thrshecp=10.0d0**(-1.0d0*itolecp)
      write(iOutFile,'("THRESHOLD FOR ECP-INTEGRALS PRESCREENING = ", &
      & E15.5,"  (DEFAULT)")')thrshecp
    END if

    call check(quick_method,iOutFile)
    
    call PrtAct(iOutFile,"Finish reading job")

    close(infile)
    endif

    !-------------------MPI/ALL NODES---------------------------------------
    if (bMPI) then
      ! communicates nodes and pass job specs to all nodes
      call mpi_setup_job()
    endif
    !-------------------MPI/ALL NODES---------------------------------------

    end

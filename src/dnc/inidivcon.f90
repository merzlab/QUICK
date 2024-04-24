#include "util.fh"
!*******************************************************
! inidivcon
!------------------------------------------------------
! Preparation for divide-and-conquers method
!
! Yipu Miao 10/01/2010:
! Add parallazation option
!
! Yipu Miao 05/21/2010:
! change the output in elegent way. Add 'atombasis' and 'residuebasis' keyword
! if in residue basis mode, pdb file is needed. In atom basis mode, every atom
! is treated as residue.
!
! Yipu Miao 05/20/2010:
! This subroutine is to initialize div and con method.
! Frag or residue will read from pdb file. Some refine will
! do in future so we can get rid of pdb file.
! note: pdb file must contain the same information with in file
! 
! Xiao He 07/12/2008 Reset the allocatable arrays
!
!
! Please send all comment or queries to 
! miao@qtp.ufl.edu
!-------------------------------------------------------
!
!
! np is the fragment number
! ifragbasis =1 Atom Basis (default)
!            =2 Residue Basis (Need extra pdb file to spcify residue, not suggest, test only)
!            =3 Non-Hydrogen Atom Basis (Please use it carefully)
!
!-------------------------------------------------------

subroutine inidivcon(natomsaved)
  use allmod
#ifdef MPIV
  use mpi
#endif
  implicit double precision (a-h,o-z)

  double precision rbuffer1,rbuffer2

  character*6,allocatable:: sn(:)             ! series no.
  double precision,allocatable::coord(:,:)    ! cooridnates
  integer,allocatable::class(:),ttnumber(:)   ! class and residue number
  character*4,allocatable::atomname(:)        ! atom name
  character*3,allocatable::residue(:)         ! residue name

  logical templog
  logical,allocatable:: templog2(:)
  logical,allocatable:: divconmfcc(:,:)
  logical,allocatable:: buffer2log(:,:)
  logical,allocatable:: embedded(:,:)
  integer,allocatable:: temp1d(:),temp2d(:,:)
  integer tempinteger,tempinteger2
  integer natomt,natomsaved
  logical bEliminate  ! if elimination step needed(test only)

  natomt=natomsaved ! avoid modification of important variable natomsaved
  bEliminate=.true.

  allocate(sn(natomt))                     ! ="ATOM"
  allocate(coord(3,natomt))                ! Coordinates, equalance with xyz, but with bohr unit
  allocate(class(natomt))                  ! Residue Name
  allocate(ttnumber(natomt))               ! Atom Serial Number
  allocate(atomname(natomt))               ! AtomName
  allocate(residue(natomt))                ! Residue Serial Number


  !===================================================================
  ! STEP 1. Read Mol Info, including atoms, coorinates, and fragment type
  !         read fragment depending on fragment method
  !===================================================================

  !-------------------MPI/MASTER--------------------------------------
  masterwork_inidivcon_readmol: if (master) then
     !--------------------End MPI/MASTER---------------------------------

     !----------------------------------------------
     ! Read Moleculer information. The residue information also can be read from the pdb file.
     !----------------------------------------------        
     if (quick_method%ifragbasis.eq.2) then
        open(iPDBFile,file=PDBFileName)
        do 99 i=1,natomt
           read(iPDBFile,100)sn(i),ttnumber(i),atomname(i),residue(i),class(i),(coord(j,i),j=1,3)
100        format(a6,1x,I4,1x,a4,1x,a3,3x,I3,4x,3f8.3)
99      enddo
        close(iPDBFile)
     else
        do i=1,natomt
           coord(1,i)=xyz(1,I)*bohr
           coord(2,i)=xyz(2,I)*bohr
           coord(3,i)=xyz(3,I)*bohr
        enddo
     endif

     !----------------------------------------------
     ! Distubute some paramters for different fragment method
     !----------------------------------------------

     select case (quick_method%ifragbasis)    ! define buffer region for different fragment method and number of fragment
     case (1)
        !rbuffer1=7.0d0
        rbuffer1=7.0d0
        rbuffer2=0.0d0
        np=natomt               ! Atom basis
     case (2)
        rbuffer1=5.0d0
        rbuffer2=0.0d0
        np=class(natomt)        ! residue basis: read from pdb file
     case (3)
        rbuffer1=6.0d0
        rbuffer2=0.0d0
        np=quick_molspec%nNonHAtom            ! Non-H Atom basis
     end select

     npsaved=np

     ! Output basic div-con information
     call PrtAct(iOutfile,"Now Begin Div & Con Fragment")
     write(iOutfile,'("NUMBER OF FRAG=",i3)') np
     write(iOutfile,'("RBuffer=",f7.2," A")') rbuffer1

     !-------------------MPI/MASTER---------------------------------------
  endif masterwork_inidivcon_readmol
  !--------------------End MPI/MASTER----------------------------------

#ifdef MPIV
  !-------------------MPI/ALL NODES------------------------------------
  if (bMPI) then
     call MPI_BCAST(np,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
     call MPI_BCAST(npsaved,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
     call MPI_BCAST(NNmax,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
  endif
  !-------------------END MPI/ALL NODES--------------------------------
#endif

  ! Allocate Varibles

  allocate(selectC(np+2),charge(np),spin(np))
  allocate(selectN(0:np+2),selectCA(np+2),ccharge(np))
  allocate(divconmfcc(np,np))
  allocate(buffer2log(np,np))
  allocate(dccore(np,500))
  allocate(dcbuffer1(np,500))
  allocate(dcbuffer2(np,500))
  allocate(dcsub(np,500))
  allocate(dcsubn(np))
  allocate(dccoren(np))
  allocate(dcbuffer1n(np))
  allocate(dcbuffer2n(np))
  allocate(dcoverlap(natomt,natomt))
  allocate(invdcoverlap(natomt,natomt))
  allocate(nbasisdc(np))
  allocate(nelecdcsub(np))
  allocate(disdivmfcc(np,np))
  allocate(selectNN(np))
  allocate(dclogic(np,natomt,natomt))
  allocate(embedded(np,np))
  allocate(templog2(np))
  allocate(temp1d(natomt))
  allocate(temp2d(natomt,natomt))
  allocate(kshells(natom))
  allocate(kshellf(natom))
  allocate(dcconnect(jshell,jshell))
  if (quick_method%mp2) allocate(nelecmp2sub(np))

  !===================================================================
  ! STEP 2. Building fragment and subsystems
  !===================================================================

  !-------------------MPI/MASTER---------------------------------------
  masterwork_inidivcon_buildsystem: if (master) then
     !--------------------End MPI/MASTER---------------------------------

     !---------------------------------------------------------------
     ! initial some varibles
     !---------------------------------------------------------------

     kshells(1)=1
     j=0

     ! kshell : show shell type for atoms
     do i=1,natom
        j=j+quick_basis%kshell(quick_molspec%iattype(i))
        kshellf(i)=j
        if(i.ne.natom)then
           kshells(i+1)=j+1
        endif
     enddo

     ! dcconnect : flag to show the connectivity of fragments
     do i=1,jshell
        do j=1,jshell
           dcconnect(i,j)=0
        enddo
     enddo

     ! divconmfcc and buffer2log : flag to show appliable method
     do i=1,np
        do j=1,np
           divconmfcc(i,j)=.false.
           buffer2log(i,j)=.false.
        enddo
     enddo

     !---------------------------------------------------------------
     ! Mark Nitrogen Atoms.
     !###############################################################
     ! It can treat as residue marker in residue-based
     ! fragment method and will be dummy in atom-based fragment method
     ! j2=amount of N +1
     ! selectN indicates no. of N
     !---------------------------------------------------------------

     j2=1
     selectN(1)=1

     do i=2,natomt
        if(quick_method%ifragbasis.eq.2) then            ! Residue-based method
           if(atomname(i).eq.' N  ')then
              j2=j2+1
              selectN(j2)=i
           endif
        else if(quick_method%ifragbasis.eq.3) then       
           if(quick_molspec%iattype(i).ne.1) then     ! Atom-based method
              j2=j2+1
              selectN(j2)=i
           endif
        else
           j2=j2+1                          ! Non-H Atom based method (in test)
           selectN(j2)=i
        endif
     enddo

     ! Mark the last atom as the last N atom even it was not
     selectN(j2+1)=natomt+1 

     do i=1,j2+1
        selectNN(i)=selectN(i)
     enddo

     !---------------------------------------------------------------
     ! Generate core region contents
     !---------------------------------------------------------------
     do i=1,j2
        icc=0
        do j=selectN(i),selectN(i+1)-1
           icc=icc+1
           !dccore(i,j) stands for the no. of atom for the jth atom in fragment i
           dccore(i,j-selectN(i)+1)=j  
        enddo
        !dccoren(i) stands for the amount of atoms for fragment i.
        dccoren(i)=icc                  
     enddo

     !-----------------------------------------------------------------
     !Searching for the buffer groups.
     !#################################################################
     !Here dcbuffer(i) and dcbuffer1(i,j) stands for the amount of buffer atoms and no. of jth atoms
     !buffer1 means atom distance less than rbuffer1
     !buffer2 means atom distance less than rbuffer2 but larger than rbuffer1
     !-----------------------------------------------------------------
     do i=1,j2
        dcbuffer1n(i)=0
        dcbuffer2n(i)=0
        do ii=1,natomt
           rmin=99.0d0
           templog=.true.
           do j=selectN(i),selectN(i+1)-1
              if(ii.eq.j)then
                 templog=.false.
              endif
           enddo
           if(templog)then
              do j=selectN(i),selectN(i+1)-1
                 dis=dsqrt((coord(1,j)-coord(1,ii))**2.0d0 + &
                      (coord(2,j)-coord(2,ii))**2.0d0 + &
                      (coord(3,j)-coord(3,ii))**2.0d0)
                 rmin=min(rmin,dis)  ! the closest distant between two fragment
              enddo
              !-----------------------------------------------------------------
              ! if rmin<rbuffer1
              !-----------------------------------------------------------------
              if(rmin.le.rbuffer1)then
                 do iiii=1,j2
                    if(ii.ge.selectN(iiii).and.ii.le.(selectN(iiii+1)-1))then
                       divconmfcc(i,iiii)=.true. ! Conjucated atoms
                    endif
                 enddo
              endif

              !-----------------------------------------------------------------
              ! if rbuffer1<rmin<rbuffer2
              !-----------------------------------------------------------------
              if((rmin.le.rbuffer2).and.(rmin.gt.rbuffer1))then
                 do iiii=1,j2
                    if(ii.ge.selectN(iiii).and.ii.le.(selectN(iiii+1)-1))then
                       buffer2log(i,iiii)=.true. ! Buffered atoms
                    endif
                 enddo
              endif
           endif
        enddo
     enddo

     !-----------------------------------------------------------------         
     ! generates all the buffer atoms
     !#################################################################
     ! dcbuffer1n stands for the buffer atom number for fragment i  r < rbuffer1
     ! dcbuffer1  stands for buffer atom for fragment i for         r < rbuffer1
     ! dcbuffer2n stands for the buffer atom number for fragment i  rbuffer1 < r < rbuffer2
     ! dcbuffer2  stands for buffer atom for fragment i             rbuffer1 < r < rbuffer2
     !-----------------------------------------------------------------
     do i=1,j2
        dcbuffer1n(i)=0
        dcbuffer2n(i)=0
        do ii=1,j2
           if(divconmfcc(i,ii).eqv..true.)then
              nntemp=selectN(ii+1)-1
              if(ii.eq.np)nntemp=natomt
              do jnum=selectN(ii),nntemp
                 dcbuffer1n(i)=dcbuffer1n(i)+1
                 dcbuffer1(i,dcbuffer1n(i))=jnum
              enddo
           endif
        enddo

        do ii=1,j2
           if((buffer2log(i,ii).eqv..true.).and.(divconmfcc(i,ii).eqv..false.))then
              nntemp=selectN(ii+1)-1
              if(ii.eq.np)nntemp=natomt
              do jnum=selectN(ii),nntemp
                 dcbuffer2n(i)=dcbuffer2n(i)+1
                 dcbuffer2(i,dcbuffer2n(i))=jnum
              enddo
           endif
        enddo
     enddo


     !-----------------------------------------------------------------         
     ! Now combine core and buffer regions to generate subsystems
     !#################################################################
     ! dcsubn stands for subsystem atom number for fragment i  
     ! dcsub  stands for subsystem atom for fragment i 
     !-----------------------------------------------------------------
     do i=1,np
        dcsubn(i)=0
        do j=1,dccoren(i) 
           dcsubn(i)=dcsubn(i)+1
           dcsub(i,dcsubn(i))=dccore(i,j)
        enddo

        do j=1,dcbuffer1n(i)
           dcsubn(i)=dcsubn(i)+1
           dcsub(i,dcsubn(i))=dcbuffer1(i,j)
        enddo

        do j=1,dcbuffer2n(i)
           dcsubn(i)=dcsubn(i)+1
           dcsub(i,dcsubn(i))=dcbuffer2(i,j)
        enddo
     enddo

     !-----------------------------------------------------------------
     ! Now we are going to make an important step, which we called elimination.
     ! bEliminate is the flag to indicate if it will be taken. 
     ! We will eliminate some embedded subsystem to save calculation cost and avoid
     ! double calculation.
     !-----------------------------------------------------------------

     divcon_elimination: if (bEliminate) then


        !-----------------------------------------------------------------
        ! Before that we output core and buffer infos before elimiantion
        !-----------------------------------------------------------------
        write(iOutfile,'("============================================================")')
        write(iOutfile,'("The Core and Buffer Atoms before elimination")')
        write(iOutfile,'("============================================================")')
        write(iOutfile,'("Frag",7x,"CoreAtom",5x,"CoreTot",5x,"BufAtom",7x,"BufTot",2x,"SubTot")')

        do i=1,np
           write(iOutfile,'(1x,i4)',advance="no") i
           Ftmp(1:dccoren(i))=dccore(i,1:dccoren(i))
           call PrtLab(linetmp,dccoren(i),Ftmp)
           call EffChar(linetmp,1,20,k1,k2)
           write(iOutfile,'(2x,a)',advance="no") linetmp(k1:k1+15)
           write(iOutfile,'(i4)',advance="no") dccoren(i)
           Ftmp(1:dcbuffer1n(i))=dcbuffer1(i,1:dcbuffer1n(i))
           call PrtLab(linetmp,dcbuffer1n(i),Ftmp)
           call EffChar(linetmp,1,20,k1,k2)
           write(iOutfile,'(7x,a)',advance="no") linetmp(k1:k1+15)
           write(iOutfile,'(i4,5x,i4,5x)') dcbuffer1n(i),dcsubn(i)
        enddo
        write(iOutfile,'("-----------------------------------------------------------")')
        write(iOutfile,'("Total Frag=",i4)') np
        write(iOutfile,'("===========================================================")')


        !-----------------------------------------------------------------
        ! First, let's search the embedded subsystems
        !-----------------------------------------------------------------
        do i=1,np
           do j=1,np
              Embedded(i,j)=.true.
              if (i.eq.j) cycle
              do jj=1,dcsubn(i)
                 if ((Any(dcsub(j,1:dcsubn(j)).eq.dcsub(i,jj))).eqv..false.) then
                    Embedded(i,j)=.false.
                 endif
              enddo
           enddo
        enddo

        !-----------------------------------------------------------------
        ! then begin to move them
        !-----------------------------------------------------------------
        do i=1,np
           ! if the embedded number equals 2, then there will be an possibility that two subsystems are
           ! entirely same. In that case just remove any one.
           if (count(embedded(i,1:np).eqv..true.).eq.2) then
              do j=1,np
                 if (i==j) cycle ! don't consider itself
                 if ((dcsubn(j)==dcsubn(i)).and.(j>i)) cycle ! elimiate the subsystem with smaller serier no.
                 if (embedded(i,j)) then
                    ! Move process
                    dcsubn(i)=0
                    dccore(j,dccoren(j)+1:dccoren(j)+dccoren(i))=dccore(i,1:dccoren(i))
                    dccoren(j)=dccoren(j)+dccoren(i)
                    dccoren(i)=0
                    write(iOutfile,*) "move the subsystem ",i," into ",j
                 endif
              enddo
           endif
           ! if the embedded number larger than 3, just try to combine small embedding subsystem
           ! into largest embedded one
           if (count(embedded(i,1:np).eqv..true.)>2) then
              jj=i
              do j=1,np
                 if (i==j) cycle
                 if ((dcsubn(j)==dcsubn(i)).and.(j>i)) cycle
                 if(embedded(i,j)) then
                    if (dcsubn(j)>=dcsubn(jj)) jj=j ! pick up the largest embedded subsystem
                 endif
              enddo
              if (jj.ne.i) then
                 ! Move process
                 dcsubn(i)=0
                 dccore(jj,dccoren(jj)+1:dccoren(jj)+dccoren(i))=dccore(i,1:dccoren(i))
                 dccoren(jj)=dccoren(jj)+dccoren(i)
                 dccoren(i)=0
                 write(iOutfile,*) "move the subsystem ",i," into ",jj
              endif
           endif
        enddo
        call flush(iOutfile)

        !-----------------------------------------------------------------!
        ! Now rebuild and rearrange core and subsystems
        !-----------------------------------------------------------------

        tempinteger=0 ! store fragment number after elimination
        do i=1,np
           if(dccoren(i).ne.0) then
              tempinteger=tempinteger+1
              ! store new dccore and dccoren
              temp1d(tempinteger)=dccoren(i)
              temp2d(tempinteger,1:dccoren(i))=dccore(i,1:dccoren(i)) 
           endif
        enddo

        ! format dccore and dccoren
        do i=1,np
           dccoren(i)=0
           do j=1,natomt
              dccore(i,j)=0
           enddo
        enddo

        ! pass value to new dccoren and dccore
        do i=1,tempinteger
           dccoren(i)=temp1d(i)
        enddo

        do i=1,tempinteger
           do j=1,dccoren(i)
              dccore(i,j)=temp2d(i,j)
           enddo
        enddo

        ! doesn't have much meaning, but just reorder the dccore
        do i=1,tempinteger
           do j=1,dccoren(i)
              temp1d(j)=dccore(i,j)
           enddo
           call iOrder(dccoren(i),temp1d)
           do j=1,dccoren(i)
              dccore(i,j)=temp1d(j)
           enddo
        enddo

        ! we finish dccore, now we will work on dcsub
        tempinteger=0
        do i=1,np
           if(dcsubn(i).ne.0) then
              tempinteger=tempinteger+1
              ! store new dcsub and dcsubn
              temp1d(tempinteger)=dcsubn(i)
              temp2d(tempinteger,1:dcsubn(i))=dcsub(i,1:dcsubn(i))
           endif
        enddo

        ! format dcsub and dcsubn
        do i=1,np
           dcsubn(i)=0
           do j=1,natomt
              dcsub(i,j)=0
           enddo
        enddo

        do i=1,tempinteger
           dcsubn(i)=temp1d(i)
        enddo

        ! pass value to dcsub and dcsubn
        do i=1,tempinteger
           do j=1,dcsubn(i)
              dcsub(i,j)=temp2d(i,j)
           enddo
        enddo

        do i=1,tempinteger
           do j=1,dcsubn(i)
              temp1d(j)=dcsub(i,j)
           enddo
           call iOrder(dcsubn(i),temp1d)
           do j=1,dcsubn(i)
              dcsub(i,j)=temp1d(j)
           enddo
        enddo

        ! finally it's time to rebuild buffers
        np=tempinteger ! get new fragment number
        do i=1,np
           dcbuffer1n(i)=dcsubn(i)-dccoren(i) ! nbuffer=nsub-ndccore
           tempinteger=0
           do j=1,dcsubn(i)
              if(Any(dccore(i,1:dccoren(i)).eq.dcsub(i,j))) cycle ! that atom belongs to core, so cycle
              tempinteger=tempinteger+1
              dcbuffer1(i,tempinteger)=dcsub(i,j)
           enddo
        enddo

     endif divcon_elimination

     !-----------------------------------------------------------------!
     ! Now we finish the elimination step, and calculate elec number and basis set number of frags
     !-----------------------------------------------------------------!
     do i=1,np
        nbasisdc(i)=0
        nelecdcsub(i)=0
        do j=1,dcsubn(i)
           nbasisdc(i)=nbasisdc(i)+quick_basis%last_basis_function(dcsub(i,j))-quick_basis%first_basis_function(dcsub(i,j))+1
           nelecdcsub(i)=nelecdcsub(i)+quick_molspec%iattype(dcsub(i,j))
        enddo
     enddo

     ! Max basis no. of subsystmes
     NNmax=0
     do i=1,np
        NNmax=max(NNmax,nbasisdc(i))
     enddo

     !-----------------------------------------------------------------!
     ! Finally, Time to output them
     !-----------------------------------------------------------------!

     write(iOutfile,'("============================================================")')
     write(iOutfile,'("The Core and Buffer Atoms")')
     write(iOutfile,'("============================================================")')
     write(iOutfile,'("Frag",7x,"CoreAtom",5x,"CoreTot",5x,"BufAtom",7x,"BufTot",5x,"SubAtom",5x,"SubTot",4x,"Elec",4x,"NBasis")')
     do i=1,np
        write(iOutfile,'(1x,i4)',advance="no") i
        Ftmp(1:dccoren(i))=dccore(i,1:dccoren(i))
        call PrtLab(linetmp,dccoren(i),Ftmp)
        call EffChar(linetmp,1,20,k1,k2)
        write(iOutfile,'(2x,a)',advance="no") linetmp(k1:k1+15)
        write(iOutfile,'(i4)',advance="no") dccoren(i)         
        Ftmp(1:dcbuffer1n(i))=dcbuffer1(i,1:dcbuffer1n(i))       
        call PrtLab(linetmp,dcbuffer1n(i),Ftmp)
        call EffChar(linetmp,1,20,k1,k2)
        write(iOutfile,'(7x,a)',advance="no") linetmp(k1:k1+15)
        write(iOutfile,'(i4)',advance="no") dcbuffer1n(i)
        Ftmp(1:dcsubn(i))=dcsub(i,1:dcsubn(i))
        call PrtLab(linetmp,dcsubn(i),Ftmp)
        call EffChar(linetmp,1,20,k1,k2)
        write(iOutfile,'(7x,a)',advance="no") linetmp(k1:k1+15)
        write(iOutfile,'(i4,2x,i4,2x,i4)') dcsubn(i),nelecdcsub(i),nbasisdc(i)
        call flush(iOutfile)
     enddo

     write(iOutfile,'("-----------------------------------------------------------")')
     write(iOutfile,'("Total Frag=",i4)') np
     write(iOutfile,'("NBasis Max=",i4)') NNmax
     write(iOutfile,'("===========================================================")')

     !-------------------MPI/MASTER---------------------------------------
  endif masterwork_inidivcon_buildsystem
  !--------------------End MPI/MASTER----------------------------------

#ifdef MPIV
  !-------------------MPI/ALL NODES------------------------------------
  if (bMPI) then
     call MPI_BCAST(np,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
     call MPI_BCAST(NNmax,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
  endif
  !-------------------END MPI/ALL NODES--------------------------------
#endif

  allocate(Odcsub(np,NNmax,NNmax))
  allocate(Pdcsub(np,NNmax,NNmax))
  allocate(Pdcsubtran(NNmax,NNmax,np))
  allocate(Xdcsub(np,NNmax,NNmax))
  allocate(Smatrixdcsub(np,NNmax,NNmax))
  allocate(evaldcsub(np,NNmax))
  allocate(codcsub(NNmax,NNmax,np))
  allocate(codcsubtran(NNmax,NNmax,np))

  !===================================================================
  ! STEP 3. Set varibles since we know everything about fragment
  !===================================================================

  !-------------------MPI/MASTER---------------------------------------
  masterwork_inidivcon_setvar: if (master) then
     !--------------------End MPI/MASTER---------------------------------

     !-----------------------------------------------------------------
     ! Fisrt, is overlap. Notice, it's not overlap integral
     !
     do i=1,natomt
        do j=1,natomt
           dcoverlap(i,j)=0
        enddo
     enddo

     do i=1,np
        do j=1,dccoren(i)
           itemp=dccore(i,j)
           do j2=1,dccoren(i) 
              jtemp=dccore(i,j2)
              dcoverlap(itemp,jtemp)=dcoverlap(itemp,jtemp)+1 
           enddo
           do j2=1,dcbuffer1n(i) 
              jtemp=dcbuffer1(i,j2)
              dcoverlap(itemp,jtemp)=dcoverlap(itemp,jtemp)+1
              dcoverlap(jtemp,itemp)=dcoverlap(jtemp,itemp)+1
           enddo
        enddo
     enddo

     ! to handle an error which is overlap(i,j) not equals to overlap (j,i)
     do i=1,natomt
        do j=1,natomt
           if(dcoverlap(i,j).ne.dcoverlap(j,i))then
              print*,'error',dcoverlap(i,j),i,j
           endif
        enddo
     enddo

     ! But foundmantally, we need 1/overlap, that is invdcoverlap
     do i=1,natomt
        do j=1,natomt
           if(dcoverlap(i,j).le.0.01d0)then
              invdcoverlap(i,j)=0.0d0
           else
              invdcoverlap(i,j)=1.0d0/real(dcoverlap(i,j))
           endif
        enddo
     enddo

     ! try to block and find connection matrix
     do kk=1,np
        do i=1,natomt
           do j=1,natomt
              dclogic(kk,i,j)=0
           enddo
        enddo
     enddo

     do itt=1,np
        do jtt=1,dcsubn(itt)
           Iblockatom=dcsub(itt,jtt)
           do jtt2=1,dcsubn(itt)
              Jblockatom=dcsub(itt,jtt2)
              do m1=1,dcbuffer1n(itt)
                 m10=dcbuffer1(itt,m1)
                 do m2=1,dcbuffer1n(itt)
                    m20=dcbuffer1(itt,m2)
                    if(Iblockatom.eq.m10.and.Jblockatom.eq.m20)then
                       dclogic(itt,Iblockatom,Jblockatom)=1
                    endif
                 enddo
              enddo

              do itemp=kshells(Iblockatom),kshellf(Iblockatom)
                 do jtemp=kshells(Jblockatom),kshellf(Jblockatom)
                    dcconnect(itemp,jtemp)=1
                    dcconnect(jtemp,itemp)=1
                 enddo
              enddo
           enddo
        enddo
     enddo

     !-------------------MPI/MASTER---------------------------------------
  endif masterwork_inidivcon_setvar
  !--------------------End MPI/MASTER----------------------------------

  !===================================================================
  ! STEP 4. broadcast all the info and variables to other nodes
  !===================================================================

#ifdef MPIV
  allocate(mpi_dc_fragn(0:mpisize-1))       ! frag no. a node has
  allocate(mpi_dc_frag(0:mpisize-1,np)) ! frag a node has
  allocate(mpi_dc_nbasis(0:mpisize-1))  ! total basis set a node has

  ! make it compatible for non-mpi calculation
  mpi_dc_fragn(0)=np
  do i=1,np
     mpi_dc_frag(0,i)=i
  enddo

  !-------------------MPI/ALL NODES------------------------------------
  if (bMPI) then
     call mpi_setup_inidivcon(natomt)
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
  endif
  !-------------------END MPI/ALL NODES--------------------------------
#endif

  if (master) then
     call PrtAct(iOutfile,"Now End Div & Con Fragment")
  endif

  ! Deallocate some varibles
  if (allocated(templog2)) deallocate(templog2)
  if (allocated(divconmfcc)) deallocate(divconmfcc)
  if (allocated(buffer2log)) deallocate(buffer2log)
  if (allocated(embedded)) deallocate(embedded)
  if (allocated(temp1d)) deallocate(temp1d)
  if (allocated(temp2d)) deallocate(temp2d)

  !===================================================================
  ! End of inidivcon
  !===================================================================
  return
end subroutine inidivcon


!*******************************************************
! Xdivided
!-------------------------------------------------------
! To get X matrix from D&C 
!
subroutine Xdivided
  use allmod
  implicit double precision (a-h,o-z)

  do i=1,np
     kstart1=0
     do jxiao=1,dcsubn(i)
        j=dcsub(i,jxiao)
        do itemp=quick_basis%first_basis_function(j),quick_basis%last_basis_function(j)
           kstart2=0
           do jxiao2=1,dcsubn(i)
              j2=dcsub(i,jxiao2)
              do jtemp=quick_basis%first_basis_function(j2),quick_basis%last_basis_function(j2)
                 Xdcsub(i,Kstart1+itemp-quick_basis%first_basis_function(j)+1, &
                        kstart2+jtemp-quick_basis%first_basis_function(j2)+1) &
                      =quick_qm_struct%x(itemp,jtemp)
              enddo
              Kstart2=Kstart2+quick_basis%last_basis_function(j2)-quick_basis%first_basis_function(j2)+1
           enddo
        enddo
        Kstart1=Kstart1+quick_basis%last_basis_function(j)-quick_basis%first_basis_function(j)+1
     enddo
  enddo
end subroutine Xdivided

!*******************************************************
! Odivided
!-------------------------------------------------------
! To get O matrix from D&C 
!

subroutine Odivided
  use allmod
  implicit double precision (a-h,o-z)

  do i=1,np
     kstart1=0
     do jxiao=1,dcsubn(i)
        j=dcsub(i,jxiao)
        do itemp=quick_basis%first_basis_function(j),quick_basis%last_basis_function(j)
           kstart2=0
           do jxiao2=1,dcsubn(i)
              j2=dcsub(i,jxiao2)
              do jtemp=quick_basis%first_basis_function(j2),quick_basis%last_basis_function(j2)
                 Odcsub(i,Kstart1+itemp-quick_basis%first_basis_function(j)+1,& 
                        kstart2+jtemp-quick_basis%first_basis_function(j2)+1) &
                      =quick_qm_struct%o(itemp,jtemp)
              enddo
              Kstart2=Kstart2+quick_basis%last_basis_function(j2)-quick_basis%first_basis_function(j2)+1
           enddo
        enddo
        Kstart1=Kstart1+quick_basis%last_basis_function(j)-quick_basis%first_basis_function(j)+1
     enddo
  enddo

end subroutine Odivided

!*******************************************************
! pdcdivided
!-------------------------------------------------------
! pdc model for D&C 
!
subroutine Pdcdivided
  use allmod
  implicit double precision (a-h,o-z)

  do i=1,np
     kstart1=0
     do jxiao=1,dcsubn(i)
        j=dcsub(i,jxiao)
        do itemp=quick_basis%first_basis_function(j),quick_basis%last_basis_function(j)
           kstart2=0
           do jxiao2=1,dcsubn(i)
              j2=dcsub(i,jxiao2)
              do jtemp=quick_basis%first_basis_function(j2),quick_basis%last_basis_function(j2)
                 Pdcsub(i,Kstart1+itemp-quick_basis%first_basis_function(j)+1, &
                      kstart2+jtemp-quick_basis%first_basis_function(j2)+1) &
                      =quick_qm_struct%dense(itemp,jtemp)
              enddo
              Kstart2=Kstart2+quick_basis%last_basis_function(j2)-quick_basis%first_basis_function(j2)+1
           enddo
        enddo
        Kstart1=Kstart1+quick_basis%last_basis_function(j)-quick_basis%first_basis_function(j)+1
     enddo
  enddo

end subroutine Pdcdivided


!*******************************************************
! divideS
!-------------------------------------------------------
! To get S matrix from D&C 
!
subroutine divideS
  use allmod
  implicit double precision (a-h,o-z)

  do i=1,np
     kstart1=0
     do jxiao=1,dcsubn(i)
        j=dcsub(i,jxiao)
        do itemp=quick_basis%first_basis_function(j),quick_basis%last_basis_function(j)
           kstart2=0
           do jxiao2=1,dcsubn(i)
              j2=dcsub(i,jxiao2)
              do jtemp=quick_basis%first_basis_function(j2),quick_basis%last_basis_function(j2)
                 Smatrixdcsub(i,Kstart1+itemp-quick_basis%first_basis_function(j)+1, & 
                    kstart2+jtemp-quick_basis%first_basis_function(j2)+1)=quick_qm_struct%s(itemp,jtemp)
              enddo
              Kstart2=Kstart2+quick_basis%last_basis_function(j2)-quick_basis%first_basis_function(j2)+1
           enddo
        enddo
        Kstart1=Kstart1+quick_basis%last_basis_function(j)-quick_basis%first_basis_function(j)+1
     enddo
  enddo
end subroutine divideS

!*******************************************************
! divideX
!-------------------------------------------------------
! To get X matrix from D&C (new_method) 
!

subroutine divideX
  use allmod
  implicit double precision(a-h,o-z)

  dimension Sminhalf(nbasis), &
       V(3,nbasis), &
       IDEGEN1(nbasis)

  ! The purpose of this subroutine is to calculate the transformation
  ! matrix X.  The first step is forming the overlap matrix (Smatrix).

  ! Now diagonalize HOLD to generate the eigenvectors and eigenvalues.

  nbasissave=nbasis

  do Itt=1,np

     ! XIAO HE reconsider              

     nbasis=nbasisdc(itt)

     allocate(Odcsubtemp(nbasisdc(itt),nbasisdc(itt)))
     allocate(VECtemp(nbasisdc(itt),nbasisdc(itt)))
     allocate(Vtemp(3,nbasisdc(itt)))
     allocate(EVAL1temp(nbasisdc(itt)))
     allocate(IDEGEN1temp(nbasisdc(itt)))

     do iixiao=1,nbasis
        do jjxiao=1,nbasis
           Odcsubtemp(iixiao,jjxiao)=Smatrixdcsub(itt,iixiao,jjxiao)
        enddo
     enddo

     !    call DIAG(NBASIS,HOLD,NBASIS,TOL,V,Sminhalf,IDEGEN1,Uxiao,IERROR)

     call DIAG(NBASIS,Odcsubtemp,NBASIS,1d-10,Vtemp,EVAL1temp,IDEGEN1temp,VECtemp,IERROR)

     ! Consider the following:

     ! X = U * s^(-.5) * transpose(U)

     ! s^-.5 is a diagonal matrix filled with the eigenvalues of S taken to
     ! to the 1/square root.  If we define an intermediate matrix A for the
     ! purposes of this discussion:

     ! A   = U * s^(-.5)
     ! or Aij = Sum(k=1,m) Uik * s^(-.5)kj

     ! s^(-.5)kj = 0 unless k=j so

     ! Aij = Uij * s^(-.5)jj

     ! X   = A * transpose(U)
     ! Xij = Sum(k=1,m) Aik * transpose(U)kj
     ! Xij = Sum(k=1,m) Uik * s^(-.5)kk * transpose(U)kj
     ! Xij = Sum(k=1,m) Uik * s^(-.5)kk * Ujk

     ! Similarly:
     ! Xji = Sum(k=1,m) Ajk * transpose(U)ki
     ! Xji = Sum(k=1,m) Ujk * s^(-.5)kk * transpose(U)ki
     ! Xji = Sum(k=1,m) Ujk * s^(-.5)kk * Uik

     ! This aggravating little demonstration contains two points:
     ! 1)  X can be calculated without crossing columns in the array
     ! which adds to speed.
     ! 2)  X has to be symmetric. Thus we only have to fill the bottom
     ! half. (Lower Diagonal)

     do I=1,nbasis
        Sminhalf(I) = EVAL1temp(I)**(-.5d0)
        !        print*,'test',I,Sminhalf(I)
     enddo

     ! Transpose U onto X then copy on to U.  Now U contains U transpose.

     do I = 1,nbasis
        do J = 1,nbasis
           quick_scratch%hold(I,J) = VECtemp(J,I)
        enddo
     enddo
     do I = 1,nbasis
        do J = 1,nbasis
           VECtemp(J,I) = quick_scratch%hold(J,I)
        enddo
     enddo

     ! Now calculate X.
     ! Xij = Sum(k=1,m) Transpose(U)kj * s^(-.5)kk * Transpose(U)ki

     do I = 1,nbasis
        do J=I,nbasis
           sum = 0.d0
           do K=1,nbasis
              sum = VECtemp(K,I)*VECtemp(K,J)*Sminhalf(K)+sum
           enddo
           Xdcsub(itt,I,J) = sum
           Xdcsub(itt,J,I) = Xdcsub(itt,I,J)
        enddo
     enddo


     ! At this point we have the transformation matrix (X) which is necessary
     ! to orthogonalize the operator matrix, and the overlap matrix (S) which
     ! is used in the DIIS-SCF procedure.

     if (allocated(Odcsubtemp)) deallocate(Odcsubtemp)
     if (allocated(VECtemp)) deallocate(VECtemp)
     if (allocated(Vtemp)) deallocate(Vtemp)
     if (allocated(EVAL1temp)) deallocate(EVAL1temp)
     if (allocated(IDEGEN1temp)) deallocate(IDEGEN1temp)

  enddo

  nbasis=nbasissave

  return
end subroutine dividex


subroutine wtoscorr
  use allmod
  implicit double precision (a-h,o-z)

  do itt=1,np
     itempcount=0
     do jtt=1,dcsubn(itt)
        k=dcsub(itt,jtt)
        do iii=quick_basis%first_basis_function(k),quick_basis%last_basis_function(k)
           wtospoint(itt,iii)=itempcount+iii-quick_basis%first_basis_function(k)+1
        enddo
        itempcount=itempcount+quick_basis%last_basis_function(k)-quick_basis%first_basis_function(k)+1
     enddo
  enddo

end subroutine wtoscorr

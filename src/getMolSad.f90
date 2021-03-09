!---------------------------------------------------------------------!
! Updated by Madu Manathunga on 06/02/2020                            !
!                                                                     !
! Previous contributors: Yipu Miao, Xio He, Alessandro Genoni,        !
!                         Ken Ayers & Ed Brothers                     !
!                                                                     !
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

#include "util.fh"

subroutine getmolsad(ierr)
   use allmod
   use quick_files_module
   use quick_exception_module

   implicit double precision(a-h,o-z)

   logical :: present,MPIsaved
   double precision:: xyzsaved(3,natom)
   character(len=80) :: keywd
   character(len=20) :: tempstring
   character(len=20) :: sadfile
   integer natomsaved
   type(quick_method_type) quick_method_save
   type(quick_molspec_type) quick_molspec_save
   integer :: ierr

   ! first save some important value
   quick_method_save=quick_method
   quick_molspec_save=quick_molspec
   ! quick_molspec_type has pointers which may lead to memory leaks
   ! therefore, assign individual variable values
!   quick_molspec_save%imult = quick_molspec%imult
!   quick_molspec_save%nelec = quick_molspec%nelec

   natomsaved=natom
   xyzsaved=xyz
   MPIsaved=bMPI

   istart = 1
   ifinal = 80
   ibasisstart = 1
   ibasisend = 80

   ! Then give them new value
   bMPI=.false.
   quick_molspec%imult=0
   quick_method%HF=.true.
   quick_method%DFT=.false.
   quick_method%UNRST=.true.
   quick_method%ZMAT=.false.
   quick_method%divcon=.false.
   quick_method%nodirect=.false.
   call allocate_mol_sad(quick_molspec%iatomtype)


   if (master) then
      call PrtAct(ioutfile,"Begin SAD initial guess")
      !-------------------------------------------
      ! First, find atom type and initialize
      !-------------------------------------------

      natom=1
      do I=1,3
         xyz(I,1) = 0.0d0
      enddo

      do iitemp=1,quick_molspec%iatomtype
         write(ioutfile,'(" For Atom Kind = ",i4)') iitemp

         ! if quick is called through api multiple times, this is necessary
         if(wrtStep .gt. 1) then
           call deallocate_calculated
         endif

         do i=1,90
            if(symbol(i).eq.quick_molspec%atom_type_sym(iitemp))then

               if(mod(i,2).eq.0)then
                  quick_molspec%imult=1
               else
                  quick_molspec%imult=2
               endif
               if(symbol(i).eq.'N ')quick_molspec%imult=4
               if(symbol(i).eq.'O ')quick_molspec%imult=3
               if(symbol(i).eq.'C ')quick_molspec%imult=3
               if(symbol(i).eq.'S ')quick_molspec%imult=3
               if(symbol(i).eq.'P ')quick_molspec%imult=4

               quick_molspec%chg(1)=i
               quick_molspec%iattype(1)=i
               write(ioutfile,'(" ELEMENT = ",a)') symbol(i)
            endif
         enddo
         if (quick_molspec%imult /= 1) quick_method%UNRST= .TRUE.
         quick_molspec%nelec = quick_molspec%iattype(1)
         if ((quick_method%DFT .OR. quick_method%SEDFT).and.quick_method%isg.eq.1) &
               call gridformSG1()
         call check_quick_method_and_molspec(ioutfile,quick_molspec,quick_method)

         !-------------------------------------------
         ! At this point we have the positions and identities of the atoms. We also
         ! have the number of electrons. Now we must assign basis functions. This
         ! is done in a subroutine.
         !-------------------------------------------
         nsenhai=1
         call readbasis(nsenhai,0,0,0,0,ierr)
         CHECK_ERROR(ierr)
         
         atombasis(iitemp)=nbasis
         write (ioutfile,'(" BASIS FUNCTIONS = ",I4)') nbasis

         if(nbasis < 1) then
                call PrtErr(iOutFile,'Unable to find basis set information for this atom.')
                call PrtMsg(iOutFile,'Update the corresponding basis set file or use a different basis set.')
                call quick_exit(iOutFile,1)
         endif

         ! if quick is called through api multiple times, this is necessary
         if(wrtStep .gt. 1) then
           call dealloc(quick_qm_struct)
         endif

         quick_qm_struct%nbasis => nbasis
         call alloc(quick_qm_struct)
         call init(quick_qm_struct)

         ! this following subroutine is as same as normal basis set normlization
         call normalize_basis()
         if (quick_method%ecp) call store_basis_to_ecp()
         if (quick_method%DFT .OR. quick_method%SEDFT) call get_sigrad

         ! Initialize Density arrays. Create initial density matrix guess.
         present = .false.
         if (quick_method%readdmx) inquire (file=dmxfilename,exist=present)
         if (present) then
            return
         else
            ! Initial Guess
            diagelement=dble(quick_molspec%nelec)/dble(nbasis)
            diagelementb=dble(quick_molspec%nelecb)/dble(nbasis)+1.d-8
            do I=1,nbasis
               quick_qm_struct%dense(I,I)=diagelement
               quick_qm_struct%denseb(I,I)=diagelementb
            enddo
         endif
         ! From SCF calculation to get initial density guess
         if(.not. quick_method%readSAD) then
            call getenergy(failed, .true.)
            do i=1,nbasis
               do j=1,nbasis
                  atomdens(iitemp,i,j)=quick_qm_struct%dense(i,j)+quick_qm_struct%denseb(i,j)
               enddo
            enddo

            if(quick_method%writeSAD) then
               sadfile = trim(quick_molspec%atom_type_sym(iitemp))//trim('sad.txt')
               open(212,file=sadfile)
               do i=1,nbasis
                  do j=1,nbasis
                     write(212,*) i,j,atomdens(iitemp,i,j)
                  enddo
               enddo
               close(212)             
            endif           

         else
            sadfile = trim(quick_molspec%atom_type_sym(iitemp))//trim('sad.txt')
            open(213,file=sadfile)  !Read from sadfile
            do i=1,nbasis
               do j=1,nbasis
                  read(213,*) ii,jj,temp
                  atomdens(iitemp,ii,jj)=temp
               enddo
            enddo
            close(213)
         endif

         call deallocate_calculated
         call dealloc(quick_qm_struct)
      enddo
      call PrtAct(ioutfile,"Finish SAD initial guess")
   endif

   natom=natomsaved
   xyz=xyzsaved

   quick_method=quick_method_save
   quick_molspec=quick_molspec_save
!   quick_molspec%imult = quick_molspec_save%imult
!   quick_molspec%nelec = quick_molspec_save%nelec

   bMPI=MPIsaved

   return

end subroutine getmolsad


subroutine allocate_mol_sad(n)
   use quick_basis_module
   integer n

   if(.not. allocated(atomDens))  allocate(atomDens(n,100,100))
   if(.not. allocated(atomBasis)) allocate(atomBasis(n))

end subroutine allocate_mol_sad



subroutine deallocate_mol_sad()
   use quick_basis_module

   if (allocated(atomDens)) deallocate(atomDens)
   if (allocated(atomBasis)) deallocate(atomBasis)

end subroutine deallocate_mol_sad

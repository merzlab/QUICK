! Ed Brothers. 11/26/01
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine getmolsad()
   use allmod
   implicit double precision(a-h,o-z)

   logical :: present,MPIsaved
   double precision:: xyzsaved(3,natom)
   character(len=80) :: keywd
   character(len=20) :: tempstring
   integer natomsaved
   type(quick_method_type) quick_method_save
   type(quick_molspec_type) quick_molspec_save

   ! first save some important value
   quick_method_save=quick_method
   quick_molspec_save=quick_molspec
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
         call readbasis(nsenhai,0,0,0,0)
         atombasis(iitemp)=nbasis

         write (ioutfile,'("BASIS FUNCTIONS = ",I4)') nbasis
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
         if(quick_molspec%atom_type_sym(iitemp).ne.'ZN')then ! if not ZN
            call getenergy(failed)
            do i=1,nbasis
               do j=1,nbasis
                  atomdens(iitemp,i,j)=quick_qm_struct%dense(i,j)+quick_qm_struct%denseb(i,j)
               enddo
            enddo
         else
            ! treat Zinc specially
            open(213,file='znsad.txt')  !Read Zn
            do i=1,39
               do j=1,39
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
   bMPI=MPIsaved
   return
end subroutine getmolsad


subroutine allocate_mol_sad(n)
   use quick_basis_module
   integer n

   allocate(atomDens(n,100,100))
   allocate(atomBasis(n))

end subroutine allocate_mol_sad



subroutine deallocate_mol_sad()
   use quick_basis_module

   deallocate(atomDens)
   deallocate(atomBasis)

end subroutine deallocate_mol_sad

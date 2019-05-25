subroutine ecpint
   !     Contact A. V. Mitin for the ECP integral package
   return
end

Subroutine ecpoperator
   !
   ! Alessandro GENONI 03/12/2007
   ! Suibroutine that adds the ECP integrals to the Fock operator
   ! for ECP calculations
   !
   use quick_basis_module
   use quick_ecp_module
   use quick_calculated_module
   use quick_files_module
   !
   implicit double precision(a - h, o - z)
   !
   ! Add the proper ECP integral to the corresponding Fock matrix element
   !
   do i = 1, nbasis
      do j = 1, nbasis
         ind = kp2(i, j)
         !     write(ioutfile,*) i,j,'---->',ind
         quick_qm_struct%o(i, j) = quick_qm_struct%o(i, j) + ecp_int(ind)
      end do
   end do
   return
end

integer function kp2(ii, jj) result(ind)
   use quick_ecp_module
   implicit none
   integer, intent(in)  :: ii, jj
   !
   ind = kvett(max(ii, jj)) + (ii + jj - max(ii, jj))
   return
end function kp2

Subroutine readecp

   ! Subroutines to read ecp
   !
   !
   ! Alessandro GENONI 03/05/2007
   ! Subroutine to  read the Effective Core Potentials
   ! The total number of electrons and the nuclear charges are modified too
   !
   use allmod

   implicit double precision(a - h, o - z)
   character(len=80) :: line
   character(len=2)  :: atom
   character(len=3)  :: pot
   integer, dimension(0:92) :: klmaxecp, kelecp, kprimecp
   logical, dimension(0:92) :: warn

   open (iecpfile, file=ecpfilename, status='old')

   allocate (nelecp(natom))
   allocate (lmaxecp(natom))

   iofile = 0
   necprim = 0
   nelecp = 0
   lmaxecp = 0
   klmaxecp = 0
   kelecp = 0
   kprimecp = 0
   warn = .true.

   ! Parse the file and find the sizes of arrays to allocate them in memory

   do while (iofile == 0)
      read (iecpfile, '(A80)', iostat=iofile) line
      read (line, *, iostat=io) atom, ii
      if (io == 0 .and. ii == 0) then
         do i = 1, 92
            if (symbol(i) == atom) then
               iat = i
               warn(i) = .false.
            end if
         end do
         read (iecpfile, '(A80)', iostat=iofile) line
         read (line, *, iostat=iatom) klmaxecp(iat), kelecp(iat)
         if (iatom == 0) then
            do while (iatom == 0)
               read (iecpfile, '(A80)', iostat=iofile) line
               read (line, *, iostat=iatom) iprim, pot
               if (iatom == 0) then
                  kprimecp(iat) = kprimecp(iat) + iprim
                  do i = 1, iprim
                     read (iecpfile, '(A80)', iostat=iofile) line
                     read (line, *) n, c1, c2
                  end do
               end if
            end do
         end if
      end if
   end do

   rewind iecpfile

   do i = 1, natom
      lmaxecp(i) = klmaxecp(quick_molspec%iattype(i))
      nelecp(i) = kelecp(quick_molspec%iattype(i))
      quick_molspec%chg(i) = quick_molspec%chg(i) - nelecp(i)
      necprim = necprim + kprimecp(quick_molspec%iattype(i))
      nelec = nelec - nelecp(i)
   end do

   !
   ! Allocation of the arrays whose dimensions depend on NECPRIM
   ! and of the arrays KFIRST and KLAST
   !
   allocate (clp(necprim))
   allocate (zlp(necprim))
   allocate (nlp(necprim))
   allocate (kfirst(mxproj + 1, natom))
   allocate (klast(mxproj + 1, natom))
   !
   ! Store the vectors CLP,NLP,ZLP,KFIRST,KLAST
   !
   clp = 0
   nlp = 0
   zlp = 0
   kfirst = 0
   klast = 0
   !
   jecprim = 0
   do i = 1, natom
      iofile = 0
      do while (iofile == 0)
         read (iecpfile, '(A80)', iostat=iofile) line
         read (line, *, iostat=io) atom, ii
         if (io == 0 .and. ii == 0) then
            if (symbol(quick_molspec%iattype(i)) == atom) then
               iatom = 0
               do while (iatom == 0)
                  read (iecpfile, '(A80)', iostat=iofile) line
                  read (line, *, iostat=iatom) klmax, nelecore
                  if (iatom == 0) then
                     jjcont = 0
                     do while (iatom == 0)
                        read (iecpfile, '(A80)', iostat=iofile) line
                        read (line, *, iostat=iatom) iprim, pot
                        jjcont = jjcont + 1
                        if (iatom == 0) then
                           kfirst(jjcont, i) = jecprim + 1
                           do j = 1, iprim
                              jecprim = jecprim + 1
                              read (iecpfile, '(A80)', iostat=iofile) line
                              read (line, *) nlp(jecprim), zlp(jecprim), clp(jecprim)
                           end do
                           klast(jjcont, i) = jecprim
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
      if (warn(quick_molspec%iattype(i))) then
         write (ioutfile, '("  ")')
         write (ioutfile, '("WARNING: NO ECP FOR ATOM ",A2,I4)') symbol(quick_molspec%iattype(i)), i
      end if
      !
   end do

   return
end subroutine

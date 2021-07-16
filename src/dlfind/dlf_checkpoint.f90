!!****h* main/checkpoint
!!
!! NAME
!! checkpoint
!!
!! FUNCTION
!! Global checkpoint file read and write routines
!!
!!   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!   %%  Global checkpoint file read and write routines                %%
!!   %%  Other files that handle checkpoint information and use this   %%
!!   %%   modue:                                                       %%
!!   %%   dlf_neb.f90                                                  %%
!!   %%   dlf_coords.f90                                               %%
!!   %%   dlf_formstep.f90                                             %%
!!   %%   dlf_lbfgs.f90                                                %%
!!   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!
!!  Issue: An unformatted restart file written by a binary using integer(8)
!!  can not be read by a binary using integer(4) - and vice versa.
!!
!! COPYRIGHT
!!
!!  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Tom Keal (thomas.keal@stfc.ac.uk)
!!
!!  This file is part of DL-FIND.
!!
!!  DL-FIND is free software: you can redistribute it and/or modify
!!  it under the terms of the GNU Lesser General Public License as 
!!  published by the Free Software Foundation, either version 3 of the 
!!  License, or (at your option) any later version.
!!
!!  DL-FIND is distributed in the hope that it will be useful,
!!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!!  GNU Lesser General Public License for more details.
!!
!!  You should have received a copy of the GNU Lesser General Public 
!!  License along with DL-FIND.  If not, see 
!!  <http://www.gnu.org/licenses/>.
!!
!! SOURCE
module dlf_checkpoint
  ! variables
  logical,parameter :: tchkform=.false. ! Checkpoint files formatted?

contains
  ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  subroutine read_separator(unit,name,tok)
    ! read a separator from a checkpoint file and compare it to the
    ! given name. In case they don't match, close the file and 
    ! return tok=false
    use dlf_global, only: glob,stdout,printl
    implicit none
    integer      ,intent(in) :: unit
    character(*) ,intent(in) :: name
    logical      ,intent(out):: tok
    character(20)            :: separator,fname
    ! ******************************************************************
    if(printl >= 4) write(stdout,"('Reading checkpoint: ',a)") &
        name
    if(tchkform) then
      read(unit,'(a20)',end=201,err=200) separator
    else
      read(unit,end=201,err=200) separator
    end if
    write(fname,'(a20)') name
    tok=.true.
    if(separator==fname) return

    write(stdout,10) "Error reading separator "//trim(name)
    close(unit)
    tok=.false.
    return

    ! return on error
200 tok=.false.
    close(unit)
    write(stdout,10) "Error reading file at separator "//trim(name)
    return
201 tok=.false.
    close(unit)
    write(stdout,10) "Error (EOF) reading file at separator "//trim(name)
    return

10  format("Checkpoint reading WARNING: ",a)
  end subroutine read_separator

  ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  subroutine write_separator(unit,name)
    ! read a separator from a checkpoint file and compare it to the
    ! given name. In case they don't match, close the file and 
    ! return tok=false
    implicit none
    integer      ,intent(in) :: unit
    character(*) ,intent(in) :: name
    character(20)            :: name20
    ! ******************************************************************
    if(tchkform) then
      write(unit,'(a20)') name
    else
      write(name20,'(a20)') name
      write(unit) name20
    end if
  end subroutine write_separator

end module dlf_checkpoint
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* checkpoint/dlf_checkpoint_read
!!
!! FUNCTION
!!
!! Read restart information from checkpoint file
!! If an error in reading is encountered, print warnings, but try to 
!! continue execution as if starting from scratch.
!!
!! SYNOPSIS
subroutine dlf_checkpoint_read(status,tok)
!! SOURCE
  use dlf_parameter_module, only: rk,ik
  use dlf_global, only: glob,stdout,printl
  use dlf_stat, only: stat
  use dlf_checkpoint, only: tchkform, read_separator
  implicit none
  integer,intent(out) :: status
  logical,intent(out) :: tok
  logical       :: tchk
  integer       :: nvar
  integer       :: iopt,iline,lbfgs_mem,icoord,nivar,nimage,ncons,nconn
  integer       :: imultistate, needcoupling
  integer       :: po_pop_size
  logical       :: tcoords2
! **********************************************************************
  tok=.false.

  ! check if checkpoint file exists
  INQUIRE(FILE="dlf_global.chk",EXIST=tchk)
  if(.not.tchk) then
    write(stdout,10) "File dlf_global.chk not found"
    return
  end if

  if(tchkform) then
    open(unit=100,file="dlf_global.chk",form="formatted")
  else
    open(unit=100,file="dlf_global.chk",form="unformatted")
  end if
  
  call read_separator(100,"Global sizes",tchk)
  if(.not.tchk) return
  if(tchkform) then
    read(100,*,end=201,err=200) &
        nvar, iopt, iline, lbfgs_mem, icoord, nivar, nimage, ncons, nconn, &
        imultistate, needcoupling
  else
    read(100,end=201,err=200) &
        nvar, iopt, iline, lbfgs_mem, icoord, nivar, nimage, ncons, nconn, &
        imultistate, needcoupling
  end if

  if(nvar/=glob%nvar) then
    write(stdout,10) "Different system size"
    write(stdout,*) "nvar read     ",nvar
    write(stdout,*) "nvar expected ",glob%nvar
    write(stdout,*) "Kind of integers in current code",ik
    if(nvar>huge(1_4).and..not.tchkform ) then
      if(ik==8) then
        write(stdout,*) "This may mean that the checkpoint file was written by a DL-FIND version "
        write(stdout,*) "using integer(4) while the current one uses integer(8)."
        write(stdout,*) "A solution may be to recompile the current code using integer(4)."
      end if
    end if
    
    close(100)
    return
  end if
  if(iopt/=glob%iopt) then
    write(stdout,10) "Different optimiser (iopt)"
    write(stdout,*) "iopt read     ",iopt
    write(stdout,*) "iopt expected ",glob%iopt
    write(stdout,*) "Kind of integers in current code",ik
    if(iopt==0.and..not.tchkform) then
      if(ik==4) then
        write(stdout,*) "This may mean that the checkpoint file was written by a DL-FIND version "
        write(stdout,*) "using integer(8) while the current one uses integer(4)."
        write(stdout,*) "A solution may be to recompile the current code using integer(8)."
      end if
    end if
    close(100)
    return
  end if
  if(iline/=glob%iline) then
    write(stdout,10) "Different line search (iline)"
    close(100)
    return
  end if
  if(lbfgs_mem/=glob%lbfgs_mem) then
    write(stdout,10) "Different memory size of L-BFGS"
    close(100)
    return
  end if
  if(icoord/=glob%icoord) then
    write(stdout,10) "Different coordinate definition (icoord)"
    close(100)
    return
  end if
  if(nivar/=glob%nivar) then
    write(stdout,10) "Different number of internal coordinates"
    close(100)
    return
  end if
  if(nimage/=glob%nimage) then
    write(stdout,10) "Different number of images"
    close(100)
    return
  end if
  if(ncons/=glob%ncons) then
    write(stdout,10) "Different number of constraints"
    close(100)
    return
  end if
  if(nconn/=glob%nconn) then
    write(stdout,10) "Different number of user connections"
    close(100)
    return
  end if
  if(imultistate/=glob%imultistate) then
    write(stdout,10) "Different multistate calculation (imultistate)"
    close(100)
    return
  end if
  if(needcoupling/=glob%needcoupling) then
    write(stdout,10) "Different multistate calculation (needcoupling)"
    close(100)
    return
  end if
    
  call read_separator(100,"Global parameters",tchk)
  if(.not.tchk) return
  
  if(tchkform) then
    read(100,*,end=201,err=200) &
        glob%nat ,printl ,glob%tolerance, glob%energy &
        , glob%oldenergy ,glob%toldenergy ,glob%tinit &
        , glob%tatoms &
        , glob%maxene, glob%maxstep, glob%scalestep &
        , glob%taccepted, glob%nebk &
        , glob%update, glob%maxupd, glob%delta &
        , glob%toldenergy_conv, glob%oldenergy_conv
  else
    read(100,end=201,err=200) &
        glob%nat ,printl ,glob%tolerance, glob%energy &
        , glob%oldenergy ,glob%toldenergy ,glob%tinit &
        , glob%tatoms &
        , glob%maxene, glob%maxstep, glob%scalestep &
        , glob%taccepted, glob%nebk &
        , glob%update, glob%maxupd, glob%delta &
        , glob%toldenergy_conv, glob%oldenergy_conv
  end if

  call read_separator(100,"XYZ data",tchk)
  if(.not.tchk) return

  if(tchkform) then
    read(100,*,end=201,err=200) &
        glob%xcoords, glob%xgradient &
        , glob%weight, glob%mass
    if(glob%tcoords2) read(100,*,end=201,err=200) glob%xcoords2
  else
    read(100,end=201,err=200) &
        glob%xcoords, glob%xgradient &
        , glob%weight, glob%mass
    if(glob%tcoords2) read(100,end=201,err=200) glob%xcoords2
  end if

  call read_separator(100,"internal c data",tchk)
  if(.not.tchk) return
  
  if(tchkform) then
    read(100,*,end=201,err=200) &
        glob%icoords, glob%igradient, glob%step, glob%spec, glob%icons &
        , glob%iconn
  else
    read(100,end=201,err=200) &
        glob%icoords, glob%igradient, glob%step, glob%spec, glob%icons &
        , glob%iconn
  end if

  if (glob%imultistate > 0) then
     call read_separator(100, "Multistate data", tchk)
     if (.not. tchk) return
     if (tchkform) then
        read(100, *, end=201, err=200) &
             glob%state_i, glob%state_j, &
             glob%pf_c1, glob%pf_c2, glob%gp_c3, glob%gp_c4, &
             glob%ln_t1, glob%ln_t2
        read(100, *, end=201, err=200) &
             glob%msenergy, glob%msgradient, glob%mscoupling
     else
        read(100, end=201, err=200) &
             glob%state_i, glob%state_j, &
             glob%pf_c1, glob%pf_c2, glob%gp_c3, glob%gp_c4, &
             glob%ln_t1, glob%ln_t2
        read(100, end=201, err=200) &
             glob%msenergy, glob%msgradient, glob%mscoupling
     endif
  end if

  if (glob%iopt/10 == 5) then
     call read_separator(100, "Parallel opt data", tchk)
     if (.not. tchk) return
     if (tchkform) then
        read(100, *, end=201, err=200) &
             po_pop_size, glob%po_radius_base, &
             glob%po_contraction, glob%po_tol_r_base, glob%po_tolerance_g, &
             glob%po_distribution, glob%po_maxcycle, glob%po_init_pop_size, &
             glob%po_reset, glob%po_mutation_rate, glob%po_death_rate, &
             glob%po_scalefac, glob%po_nsave
     else
        read(100, end=201, err=200) &
             po_pop_size, glob%po_radius_base, &
             glob%po_contraction, glob%po_tol_r_base, glob%po_tolerance_g, &
             glob%po_distribution, glob%po_maxcycle, glob%po_init_pop_size, &
             glob%po_reset, glob%po_mutation_rate, glob%po_death_rate, &
             glob%po_scalefac, glob%po_nsave
     endif
     if (po_pop_size /= glob%po_pop_size) then
        write(stdout,10) "Different population size"
        close(100)
        return
     end if
  end if

  call read_separator(100,"stat module",tchk)
  if(.not.tchk) return

  if(tchkform) then
    read(100,*,end=201,err=200) stat
  else
    read(100,end=201,err=200) stat
  end if

  call read_separator(100,"status",tchk)
  if(.not.tchk) return

  if(tchkform) then
    read(100,*,end=201,err=200) status
  else
    read(100,end=201,err=200) status
  end if

  call read_separator(100,"END",tchk)
  if(.not.tchk) return

  if(printl >= 4) write(stdout,"('Global checkpoint file successfully read')")

  tok=.true.
  close(100)

  call dlf_checkpoint_coords_read(tok)
  if(.not.tok) return
  call dlf_checkpoint_formstep_read(tok)
  if(.not.tok) return
  call dlf_checkpoint_linesearch_read(tok)
  if(.not.tok) return
  call dlf_checkpoint_conint_read(tok)
  if(.not.tok) return

  return

  ! return on error
  close(100)
200 continue
  write(stdout,10) "Error reading global checkpoint file"
  return
201 continue
  write(stdout,10) "Error (EOF) reading file"
  return

10 format("Checkpoint reading WARNING: ",a)
end subroutine dlf_checkpoint_read
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* checkpoint/dlf_checkpoint_write
!!
!! FUNCTION
!!
!! Write restart information to checkpoint file
!!
!! SYNOPSIS
subroutine dlf_checkpoint_write(status)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,printl
  use dlf_stat, only: stat
  use dlf_checkpoint, only: tchkform,write_separator
  implicit none
  integer, intent(in) :: status
  character(20) :: separator
! **********************************************************************

! Only want one processor to do the writing; that processor must know 
! all the necessary information.
  if (glob%iam /= 0) return
  ! Q: Should task-farming really be treated differently?
  ! It might be better for each 
  ! workgroup to write its own checkpoint so task-farming 
  ! runs could restart at any point too (at the cost of extra
  ! I/O and disk space) - this could be useful for a
  ! large standalone FD Hessian calc for example.
  ! If this were done then ntasks should be 
  ! saved to chk files as this would need to be consistent on restart.
  ! In most sophisticated solution each workgroup would write 
  ! only the data which differed between workgroups.
  ! NB: File names would have to be labelled by workgroup too 
  ! as we cannot assume each has its own scratch directory.
  if(tchkform) then
    open(unit=100,file="dlf_global.chk",form="formatted")
    call write_separator(100,"Global sizes")
    write(100,*) &
        glob%nvar, glob%iopt, glob%iline, glob%lbfgs_mem, glob%icoord, &
        glob%nivar, glob%nimage, glob%ncons, glob%nconn, &
        glob%imultistate, glob%needcoupling
    call write_separator(100,"Global parameters")
    write(100,*) &
        glob%nat ,printl ,glob%tolerance, glob%energy &
        , glob%oldenergy ,glob%toldenergy ,glob%tinit &
        , glob%tatoms &
        , glob%maxene, glob%maxstep, glob%scalestep &
        , glob%taccepted, glob%nebk &
        , glob%update, glob%maxupd, glob%delta &
        , glob%toldenergy_conv, glob%oldenergy_conv
    call write_separator(100,"XYZ data")
    write(100,*) glob%xcoords, glob%xgradient &
        , glob%weight, glob%mass
    if(glob%tcoords2)  write(100,*) glob%xcoords2
    call write_separator(100,"internal c data")
    write(100,*) glob%icoords, glob%igradient, glob%step, glob%spec &
        , glob%icons, glob%iconn
    if (glob%imultistate > 0) then
       call write_separator(100,"Multistate data")
       write(100,*) glob%state_i, glob%state_j, &
            glob%pf_c1, glob%pf_c2, glob%gp_c3, glob%gp_c4, &
            glob%ln_t1, glob%ln_t2
       write(100,*) &
            glob%msenergy, glob%msgradient, glob%mscoupling 
    endif
    if (glob%iopt/10 == 5) then
       call write_separator(100, "Parallel opt data")
       write(100, *) &
            glob%po_pop_size, glob%po_radius_base, &
            glob%po_contraction, glob%po_tol_r_base, glob%po_tolerance_g, &
            glob%po_distribution, glob%po_maxcycle, glob%po_init_pop_size, &
            glob%po_reset, glob%po_mutation_rate, glob%po_death_rate, &
            glob%po_scalefac, glob%po_nsave
    end if
    call write_separator(100,"stat module")
    write(100,*) stat
    call write_separator(100,"status")
    write(100,*) status
    call write_separator(100,"END")

  else
    open(unit=100,file="dlf_global.chk",form="unformatted")
    call write_separator(100,"Global sizes")
    write(100) &
        glob%nvar, glob%iopt, glob%iline, glob%lbfgs_mem, glob%icoord, &
        glob%nivar, glob%nimage, glob%ncons, glob%nconn, &
       glob%imultistate, glob%needcoupling
    call write_separator(100,"Global parameters")
    write(100) &
        glob%nat ,printl ,glob%tolerance, glob%energy &
        , glob%oldenergy ,glob%toldenergy ,glob%tinit &
        , glob%tatoms &
        , glob%maxene, glob%maxstep, glob%scalestep &
        , glob%taccepted, glob%nebk &
        , glob%update, glob%maxupd, glob%delta &
        , glob%toldenergy_conv, glob%oldenergy_conv
    call write_separator(100,"XYZ data")
    write(100) glob%xcoords, glob%xgradient &
        , glob%weight, glob%mass
    if(glob%tcoords2) write(100) glob%xcoords2
    call write_separator(100,"internal c data")
    write(100) glob%icoords, glob%igradient, glob%step, glob%spec &
        , glob%icons, glob%iconn
    if (glob%imultistate > 0) then
       call write_separator(100,"Multistate data")
       write(100) glob%state_i, glob%state_j, &
            glob%pf_c1, glob%pf_c2, glob%gp_c3, glob%gp_c4, &
            glob%ln_t1, glob%ln_t2
       write(100) &
            glob%msenergy, glob%msgradient, glob%mscoupling 
    endif
    if (glob%iopt/10 == 5) then
       call write_separator(100, "Parallel opt data")
       write(100) &
            glob%po_pop_size, glob%po_radius_base, &
            glob%po_contraction, glob%po_tol_r_base, glob%po_tolerance_g, &
            glob%po_distribution, glob%po_maxcycle, glob%po_init_pop_size, &
            glob%po_reset, glob%po_mutation_rate, glob%po_death_rate, &
            glob%po_scalefac, glob%po_nsave
    end if
    call write_separator(100,"stat module")
    write(100) stat
    call write_separator(100,"status")
    write(100) status
    call write_separator(100,"END")
  end if

  close(100)

  call dlf_checkpoint_coords_write
  call dlf_checkpoint_formstep_write
  call dlf_checkpoint_linesearch_write
  call dlf_checkpoint_conint_write

end subroutine dlf_checkpoint_write
!!****

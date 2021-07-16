#
#  TCL interface between ChemShell and DL-FIND
#
# COPYRIGHT
#
#  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
#  Tom Keal (thomas.keal@stfc.ac.uk)
#
#  This file is part of DL-FIND.
#
#  DL-FIND is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as 
#  published by the Free Software Foundation, either version 3 of the 
#  License, or (at your option) any later version.
#
#  DL-FIND is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public 
#  License along with DL-FIND.  If not, see 
#  <http://www.gnu.org/licenses/>.
#
# Add dl-find
proc dl-find { args } { 

    set theory_args " "
    set theory mndo
    set result dl-find.result
    set list_option medium
    set restraints undefined
    set printl undefined
    set debug 0
    # we need this default here to add or subtract values
    set icoord 0
    # defaults for imultistate/needcoupling imply a single state optimisation
    set imultistate 0
    set coupled_states 1
    set needcoupling 0
    # for non-microiterative calcs give an error if inner atoms are specified
    set microiterative 0
    set include_res 0

    #
    # Translate command line arguments to Tcl variables
    #
    if { [ parsearg dl-find {theory coords coords2 result result2 \
				 list_option restraints scalestep coordinates neb chainp dimer dimer_extrapolate \
				 dimer_interpolate \
				 tolerance maxcycle maxene optimiser optimizer trustradius maxstep \
				 lbfgs_mem nimage nebk neb_cart dump restart update_method maxupdate delta \
				 soft initial_hessian carthessian tsrelative maxrot tolrot \
				 residues constraints connect active_atoms frozen weights \
				 timestep fric0 fricfac fricp tolerance_e \
				 conint state_i state_j coupled_states pf_c1 pf_c2 gp_c3 gp_c4 ln_t1 ln_t2 \
				 distort minstep maxdump task thermal temperature spec pdb_file mass \
				 qts qtsrate nzero test_delta rate tsplit \
				 stochastic genetic \
				 po_pop_size po_radius po_contraction po_tolerance_r po_tolerance_g \
				 po_distribution po_maxcycle po_init_pop_size po_reset po_mutation_rate \
				 po_death_rate po_scalefac po_nsave \
				 neb_climb_test neb_freeze_test inithessian \
                                 microiterative inner_atoms inner_residues include_res \
                                 maxmicrocycle micro_esp_fit \
			       } \
	       $args ] } {

	chemerr "error in DL-FIND arguments" 
    }

    # A bit of documentation:
    # weights is an array of real numbers, length: number of atoms. It defines the weight
    # for each atom for the distance evaluation in NEB and possibly the dimer direction
    #  (both not implemented yet)
    #
    # spec can be directly input. It is an array of length nat (number of atoms)
    # Should only be used if one knows that the dl-find code is doing with it 
    # it is translated to the first nat entries in the spec array, 0 for a free atom, -1 for a 
    # frozen one, negative values of specific Cartesian components frozen, positive 
    # for residues membership (which is safer to input via keyword residues).

    # Convert the conint keyword to imultistate integer
    # This must be done here so that the dimensions for the scratch energy/gradient matrices
    # are known
    if { [ info exists coupled_states ] == 1 } then {
      set coupled_states [ check_boolean coupled_states ]
    }
    if { [ info exists conint ] == 1 } then {
      switch -glob $conint {
	NO* - no* {}
	pf - PF - Pf {
	  set imultistate 1
	}
	gp - GP - Gp {
	  set imultistate 2
	  if { $coupled_states } { set needcoupling 1 }
        }
        ln - LN - Ln {
          set icoord [ expr $icoord + 10 ]
	  set imultistate 3
	  if { $coupled_states } { set needcoupling 1 }
	  set iopt 40
        }
	default {
	  chemerr "Wrong option to conint"
	}
      }
    }

    #   
    # These settings control the energy/gradient evaluation routines
    # (see futils/eandg.c) and provide workspace
    #
    # Different dimensions are required for (multistate) conical intersection searches
    # and therefore the conint keyword should be evaluated before initialising the matrices
    #
    set tmp_coords  dl-find.coo
    fragment $tmp_coords new persistent
    copy_object from=$coords to=$tmp_coords type=fragment
    #
    set natoms [ get_number_of_atoms coords=$coords ]

    if {$imultistate > 0} {
	set tmp_energy_dimensions {1 2}
	set tmp_gradient_dimensions "3 $natoms 2"
    } else {
	set tmp_energy_dimensions {1 1}
	set tmp_gradient_dimensions "3 $natoms"
    }

    set tmp_energy  dl-find.energy
    matrix  $tmp_energy unknown
    set_matrix_size matrix=$tmp_energy dimensions= $tmp_energy_dimensions \
	datatype=real name= "Scratch energies for dl-find interface"
    #
    set tmp_gradient  dl-find.gradient
    matrix  $tmp_gradient unknown
    set_matrix_size matrix=$tmp_gradient dimensions= $tmp_gradient_dimensions \
	datatype=real name= "Scratch gradient for dl-find interface"

    if {$needcoupling == 1} {
	set tmp_coupling  dl-find.coupling
	matrix  $tmp_coupling unknown
	set_matrix_size matrix=$tmp_coupling dimensions= "3 $natoms" \
	    datatype=real name= "Scratch interstate coupling gradient for dl-find interface"
    }

    switch $printl {
      undefined { 
	switch $list_option {
	  none {
	    set printl 0
	  }
	  medium {
	    set printl 2
	    set printf 2
	  }
	  full {
	    set printl 4
	    set printf 4
	  }
	  debug {
	    set printl 6
	    set printf 6
	    set debug 1
	  }
	}
      }
      default {}
    }
    # Initialise boolean variables
    if { [ info exists carthessian ] == 1 } then {
      set carthessian [ check_boolean carthessian ]
    }
    if { [ info exists tsrelative ] == 1 } then {
      set tsrelative [ check_boolean tsrelative ]
    }
    if { [ info exists restart ] == 1 } then {
      set restart [ check_boolean restart ]
      # check if restart files are present and exit if not
      if { $restart } then {
	if { ! [ file exists dlf_global.chk ] } then {
	  chemerr "Restart requested, but no restart file(s) (dlf_global.chk) exist"
	}
      }
    }
    if { [ info exists dimer ] == 1 } then {
      set dimer [ check_boolean dimer ]
      if { $dimer } then {
	set icoord [ expr $icoord + 220 ]
      }
    }
    # dimer_interpolate and dimer_extrapolate exist for backwards-compatibility
    # reasons. They are equivalent:
    if { [ info exists dimer_interpolate ] == 1 } then {
      set dimer_extrapolate [ check_boolean dimer_interpolate ]
    }
    if { [ info exists dimer_extrapolate ] == 1 } then {
      set dimer_extrapolate [ check_boolean dimer_extrapolate ]
      if { [ info exists dimer ] == 1 } then {
	if { $dimer && ! $dimer_extrapolate } then {
	  set icoord [ expr $icoord - 10 ]
	}
      }
    }
    if { [ info exists neb_cart ] == 1 } then {
      set neb_cart [ check_boolean neb_cart ]
    }
    set microiterative [ check_boolean microiterative ]
    set include_res [ check_boolean include_res ]
    if { [ info exists micro_esp_fit ] == 1 } then {
      set micro_esp_fit [ check_boolean micro_esp_fit ]
    }    

    #
    # initialise any restraints 
    #
    if { $restraints != "undefined" } {
      restraint dl-find_rst \
	  coords = $tmp_coords \
	  energy = $tmp_energy \
	  gradient = $tmp_gradient \
	  restraints = $restraints
    }

    #
    # Convert the "keyword" options to intergers understood by dl-find
    #
    if { [ info exists coordinates ] == 1 } then {
      switch -glob $coordinates {
	cart* - Cart* - CART* {
	  set icoord [ expr $icoord + 0 ]
	}
	mass - Mass - MASS {
	  set massweight 1
	}
	dlc - DLC {
	  set icoord [ expr $icoord + 3 ]
	}
	tc - TC {
	  set icoord [ expr $icoord + 4 ]
	}
	hdlc - HDLC {
	  set icoord [ expr $icoord + 1 ]
	}
	hdlc-tc - HDLC-TC {
	  set icoord [ expr $icoord + 2 ]
	}
	default {
	  chemerr "Wrong option to coordinates"
	}
      }
    }

    if { [ info exists neb ] == 1 } then {
      switch -glob $neb {
	NO* - no* {}
	free - FREE {
	  set icoord [ expr $icoord + 100 ]
	}
	frozen - FROZEN {
	  set icoord [ expr $icoord + 120 ]
	}
	perp* - PERP* {
	  set icoord [ expr $icoord + 110 ]
	}
	default {
	  chemerr "Wrong option to neb, should be: free, frozen, or perp"
	}
      }
      if { [ info exists neb_cart ] == 1 } then {
	if { $neb_cart } then {
	  set icoord [ expr $icoord + 30 ]
	}
      }
    }

    if { [ info exists chainp ] == 1 } then {
      switch -glob $chainp {
	NO* - no* {}
	free - FREE {
	  set icoord [ expr $icoord + 300 ]
	}
	frozen - FROZEN {
	  set icoord [ expr $icoord + 320 ]
	}
	perp* - PERP* {
	  set icoord [ expr $icoord + 310 ]
	}
	default {
	  chemerr "Wrong option to chainp, should be: free, frozen, or perp"
	}
      }
    }

    if { [ info exists optimizer ] == 1 } then {
	if { [ info exists optimiser ] == 0 } then {
	    set optimiser $optimizer
	}
    }

    if { [ info exists optimiser ] == 1 } then {
      switch -glob $optimiser {
	lbfgs - LBFGS - L-BFGS - l-bfgs {
	  set iopt 3
	}
	prfo - PRFO - p-rfo - P-RFO {
	  set iopt 10
	}
	CG - cg {
	  set iopt 2
	}
	SD - sd {
	  set iopt 0
	}
	dyn - Dyn - DYN {
	  set iopt 30
	}
	nr - NR {
	  set iopt 20
	}
	ln - LN {
	  set iopt 40
	}
	default {
	  chemerr "Wrong option to optimiser"
	}
      }
    }

    if { [ info exists trustradius ] == 1 } then {
      switch -glob $trustradius {
	const* - CONST* - Const* {
	  set iline 0
	}
	energy - ENERGY - Energy {
	  set iline 1
	}
	gradient - GRADIENT - gradient {
	  set iline 2
	}
	default {
	  chemerr "Wrong option to trustradius"
	}
      }
    }

    if { [ info exists initial_hessian ] == 1 } then {
      switch -glob $initial_hessian {
	external - EXTERNAL - External {
	  set inithessian 0
	}
	onepoint - ONEPOINT - Onepoint {
	  set inithessian 1
	}
	twopoint - TWOPOINT - Twopoint {
	  set inithessian 2
	}
	diagonal - DIAGONAL - Diagonal {
	  set inithessian 3
	}
	identity - IDENTITY - Identity {
	  set inithessian 4
	}
	default {
	  chemerr "Wrong option to initial_hessian"
	}
      }
    }

    if { [ info exists update_method ] == 1 } then {
      switch -glob $update_method {
	none - NONE - None {
	  set update 0
	}
	pow* - POW* - Pow* {
	  set update 1
	}
	bof* - BOF* - Bof*  {
	  set update 2
	}
	bfgs - BFGS - Bfgs {
	  set update 3
	}
	default {
	  chemerr "Wrong option to update_method"
	}
      }
    }

    # Hessian and thermal corrections only
    if { [ info exists thermal ] == 1 } then {
      if { [ check_boolean thermal ] == 1 } then {
	set iopt 11
      }
    }

    # Hessian and rate for QTST
    if { [ info exists qtsrate ] == 1 } then {
      if { [ check_boolean qtsrate ] == 1 } then {
        set iopt 12
        set qts true
      }
    }

    # TST rate (no tunneling, only via Wigner)
    if { [ info exists rate ] == 1 } then {
      if { [ check_boolean rate ] == 1 } then {
        set iopt 13
      }
    }

    # qTS (instanton) 
    if { [ info exists qts ] == 1 } then {
      if { [ check_boolean qts ] == 1 } then {
	set icoord 190
	set massweight 1
      }
    }

    # qTS (instanton) 
    if { [ info exists tsplit ] == 1 } then {
      if { [ check_boolean tsplit ] == 1 } then {
	set qtsflag 1
      }
    }

    # test_delta
    if { [ info exists test_delta ] == 1 } then {
      if { [ check_boolean test_delta ] == 1 } then {
	set iopt 9
      }
    }

    # img_flag: does the theory understand the argument image= ?
    if { [ info exists img_flag ] == 1 } { unset img_flag }
 #   if { $theory == "mndo" } { set img_flag "" }
    if { $theory == "gamess" } { set img_flag "" }
    if { $theory == "orca" } { set img_flag "" }
    # If extra QM theories are added above, the corresponding switch statement
    # in hybrid2.tcl should also be updated.
    if { $theory == "hybrid" } { set img_flag "" }

    # states array for multistate calculations
    if { [ info exists state_i ] == 1 && [ info exists state_j ] == 1 } then {
	set states " $state_i $state_j "
    } elseif {$imultistate > 0} {
	set states " 1 2 "
    }

    # global minimisation algorithms
    if { [ info exists stochastic ] == 1 } then {
      if { [ check_boolean stochastic ] == 1 } then {
	set iopt 51
      }
    } 
    if { [ info exists genetic ] == 1 } then {
      if { [ check_boolean genetic ] == 1 } then {
	set iopt 52
      }
    } 

    if { [ info exists po_distribution ] == 1 } then {
      switch -glob $po_distribution {
	  random {
	      set po_distrib 1 
	  }
	  force_direction_bias {
	      set po_distrib 2
	  }
	  force_bias {
	      set po_distrib 3
	  }
	  default {
	      chemerr "Wrong option to po_distribution"
	  }
      }
    }

    #
    # There are three ways of specifying active / frozen atoms when residues are used:
    # - residues: the residue named fixed is frozen
    # - active_atoms: all other atoms are frozen
    # - frozen: the specified atoms are frozen
    #

    # convert frozen into active_atoms
    if { [ info exists frozen ] == 1 } then {
      if { [ info exists active_atoms ] != 1 } then {
	set active_atoms {}
      }
      set frozen [ expand_range $frozen ]
      set n [ get_number_of_atoms coords=$coords ]
      for { set iat 1 } { $iat <= $n } { incr iat 1 } {
 	if { [ lsearch $frozen $iat ] == -1 } { lappend active_atoms $iat }
      }
    }
    
    if { [ info exists active_atoms ] == 1 } then {
      set active_atoms [ expand_range $active_atoms ]
    }

    if { [ info exists active_atoms ] == 1 && [ info exists residues ] == 1 } then {

	set active_atoms [ expand_range $active_atoms ]

	if { [ info exists residues ] == 1 } then {

	    set r_names [ lindex $residues 0 ]
	    set r_data  [ lindex $residues 1 ]

	    set r_all [ res_selectall coords=$coords ]

	    if { $debug } {
		puts stdout "sel all $r_all"
	    }
	    # { Molecule } { { <all atoms> } }

	    set tt [ lindex [lindex $r_all 1 ] 0 ]
	    foreach atom $tt { set key($atom) 0 }

	    for { set res 0 } { $res < [ llength $r_names ] } { incr res 1 } {
		set data [ lindex $r_data $res ]
		foreach entry $data { set key($entry) 1	}
	    }

	    set tt3 {}
	    foreach atom $tt {
		if { ! $key($atom) } { lappend tt3 $atom }
	    }

	    if { [ llength $tt3 ] } {
		# These are the atoms not in any of the input residues
		# Add a new residue rest to hold them
		set r_rest [ list rest [ list $tt3 ] ]
		set residues  [ inlist function=merge residues2= $r_rest residues= $residues]
	    }

	    if { $debug } {
		puts stdout "Residues after inital loop $residues"
		puts stdout "Active atoms $active_atoms"
	    }

	    set r_names [ lindex $residues 0 ]
	    set r_data  [ lindex $residues 1 ]

	    catch {unset new_data}
	    catch {unset new_names}
	    set frozen {}

	    for { set res 0 } { $res < [ llength $r_names ] } { incr res 1 } {

		set name [ lindex $r_names $res ]
		set data [ lindex $r_data $res ]

		# puts stdout "$name $data"

		set t1 {}
		set t2 {}

		foreach entry $data {
		    if { [ lsearch $active_atoms $entry ] == -1 } {
			lappend frozen $entry
		    } else {
			lappend t2 $entry
		    }
		}

		# puts stdout " frozen $frozen t2 $t2"

		switch [ llength $t2 ] {
		    0 - 1 - 2 {
			#
			# the remainder (moving part) is to small to be 
			# optimised as a DLC residue, by removing the atoms
			# it will be treated as cartesian 
		    }
		    default {
			lappend new_data  $t2
			lappend new_names $name
		    }
		}
	    }
	    set residues [ list $new_names $new_data ]

	    if { [ lsearch [ lindex $residues 0 ] rest ] != -1} {

		# puts stdout "removing rest $residues"
		# puts stdout " inlist function=remove residues=$residues sets=rest "
		# Sensitive to space before the $residues

		set residues [ inlist function=remove residues= $residues sets=rest ]

		#puts stdout "removing rest done $residues"
	    }

	    set r_fro    [ list fixed [ list  $frozen ] ]
	    set residues   [ inlist function=merge residues= $r_fro residues2= $residues]
	    #puts stdout "Final residue list: $residues"

	} else {

	    set r_act [ list [ list active ] [ list $active_atoms ] ]
	    # { active } { { <active atoms> } }

	    set r_all [ res_selectall coords=$coords ]
	    # { Molecule } { { <all atoms> } }

	    set r_fro [ inlist function=merge residues= $r_act residues2= $r_all ]
	    # { Molecule active} { { <all atoms> } { <active atoms> } }

	    set r_fro [ inlist function=exclude residues= $r_fro \
		    set1=Molecule set2=active target=fixed ]

	    # {fixed active} { {<fixed atoms>} {<active atoms>} }

	    set residues [ inlist function=remove residues= $r_fro \
		    sets={active} ]
	}
    }

    #
    # Microiterative QM/MM optimisation: define inner and outer regions
    #
    if { [ info exists inner_atoms ] == 1 } then {
      
      if { $microiterative == 0 } then {
        chemerr "inner_atoms is not applicable in non-microiterative optimisations"
      }

      if { [ info exists inner_residues ] == 1 } then {
	chemerr "The inner region cannot be specified using inner_atoms and inner_residues at the same time"
      }

      set inner_atoms [ expand_range $inner_atoms ]

      # Handle residues if they are defined.
      # Residues cannot cross the inner/outer region boundary,
      # so residues containing a mix of inner region and outer region atoms
      # must be modified.
      # If include_res is false (default), cross-boundary residues will
      # be split into inner and outer parts.
      # If resulting residues are of size 2 or less they will be deleted and 
      # the atoms inside will be treated as cartesians
      # If include_res is true, all outer atoms in the cross-boundary residue
      # will be converted to inner atoms.
      if { [ info exists residues ] == 1 } then {
        # Go through the residue list and identify cross-boundary residues
        set r_names [ lindex $residues 0 ]
        set r_data  [ lindex $residues 1 ]
        set n_names {}
        set n_data  {}
 
        for { set res 0 } { $res < [ llength $r_names ] } { incr res 1 } {
          set name [ lindex $r_names $res ]
          set data [ lindex $r_data $res ]
          # divide residue into inner and outer part
          set this_inner {}
          set this_outer {}
          if { $name == "fixed" } {
            # outer by definition - check for mistakes in entry
            foreach atom $data {
              if { [ lsearch $inner_atoms $atom ] != -1 } {
                chemerr "Frozen atom $atom selected in inner_atoms list"
              }
            }
            set this_outer $data
          } else {
            foreach atom $data {
              if { [ lsearch $inner_atoms $atom ] != -1 } {
                lappend this_inner $atom
              } else {
                lappend this_outer $atom
              }
            }
          }
          if { $this_inner == {} || $this_outer == {} } {
            # residue does not cross boundary - no action required
            lappend n_names $name
            lappend n_data $data
          } else {
            # we have a mix
            if { $include_res } {
              # convert all atoms in residue to inner atoms
              foreach atom $data {
                if { [ lsearch $inner_atoms $atom ] == -1 } {
                  lappend inner_atoms $atom
                }
              }              
              # residue definition does not change
              lappend n_names $name
              lappend n_data $data
              if { $debug } {
                puts stdout "Residue $name crossed inner/outer boundary"
                puts stdout "Moved to inner region: $data"
              }
            } else {
              # split the residue
              if { $debug } {
                puts stdout "Residue $name crossed inner/outer boundary"
              }
              if { [ llength $this_inner ] >= 2 } {
                lappend n_names $name
                lappend n_data $this_inner
                if { $debug } {
                  puts stdout "New inner region residue: $this_inner"
                }
              } else {
                # new residue too small - do not add to list
                if { $debug } {
                  puts stdout "Inner region atoms treated as cartesians: $this_inner"
                }
              }
              if { [ llength $this_outer ] >= 2 } {
                lappend n_names o_$name
                lappend n_data $this_outer
                if { $debug } {
                  puts stdout "New outer region residue: $this_outer"
                }
              } else {
                # new residue too small - do not add to list
                if { $debug } {
                  puts stdout "Outer region atoms treated as cartesians: $this_outer"
                }
              }
            }
          }

        # end of loop over residues
        }

        # Store new residue list
	set residues [ list $n_names $n_data ]        

      }
    }

    if { [ info exists inner_residues ] == 1 } then {
      # Convert inner_residues specification to an inner_atoms specification
      # which will be read in by DL-FIND

      if { $microiterative == 0 } then {
        chemerr "inner_residues is not applicable in non-microiterative optimisations"
      }

      set inner_atoms { }
      set r_in [ inlist function=pick sets= $inner_residues residues= $residues ]
      set r_names [ lindex $r_in 0 ]
      set r_data  [ lindex $r_in 1 ]
      for { set res 0 } { $res < [ llength $r_names ] } { incr res 1 } {
        set name [ lindex $r_names $res ]
        set data [ lindex $r_data $res ]
        foreach entry $data {
          lappend inner_atoms $entry
        }
      }

    }

    # TODO: mcore (micro_esp_fit) handling (see hdlcopt2.tcl)


    # debug the residue list
    if { [info exists pdb_file] } then {
      write_pdb coords=$tmp_coords file=$pdb_file residues= $residues
    }

   

    ####################################################################
    dlf_c 
    ####################################################################

    delete_object $tmp_coords
    delete_object $tmp_energy
    delete_object $tmp_gradient
    if { $needcoupling == 1 } {
	delete_object $tmp_coupling
    }

    if { $restraints != "undefined" } {
      dl-find_rst destroy
    }

    end_module
}

# This would really belong somewhere else: draw vibration

proc tsmode_xyz { args } {
  set delta 1
  set n 20
  set file tsmode.xyz
  set tsrelative 0
  #
  # Translate command line arguments to Tcl variables
  #
  if { [ parsearg tsmode_xyz { coords coords2 delta n file tsrelative } \
             $args ] } {
    chemerr "error in arguments to tsmode_xyz"
  }
  set tsrelative [ check_boolean tsrelative ]
  fragment cts1 new volatile
  fragment cts2 new volatile
  fragment cp new volatile
  push_banner_flag 0
  copy_object from=$coords  to=cts1 type=fragment
  copy_object from=$coords2 to=cts2 type=fragment
  # calculate distance
  set dist 0.
  set nat [ get_number_of_atoms coords=$coords ]
  for { set iat 1 } { $iat <= $nat } { incr iat } {
    set at2 [ get_atom_entry coords=cts2 atom_number= $iat ]
    if { $tsrelative } then {
      set dist [ expr $dist + \
                     pow ( [ lindex $at2 1 ] ,2 ) + \
                     pow ( [ lindex $at2 2 ] ,2 ) + \
                     pow ( [ lindex $at2 3 ] ,2 ) ]
    } else {
      set at1 [ get_atom_entry coords=cts1  atom_number= $iat ]
      set dist [ expr $dist + \
                     pow ( [ lindex $at1 1 ] - [ lindex $at2 1 ] ,2 ) + \
                     pow ( [ lindex $at1 2 ] - [ lindex $at2 2 ] ,2 ) + \
                     pow ( [ lindex $at1 3 ] - [ lindex $at2 3 ] ,2 ) ]
    }
    #puts "iat $iat nat $nat dist $dist"
  }
  set dist [ expr sqrt( $dist ) ]
  puts "Distance between structures: $dist"
  exec rm -f $file
  copy_object from=cts1 to=cp type=fragment
  for { set i 0 } { $i < $n } { incr i } {
    set elon [ expr $delta / $dist * sin( $i * 6.2831853 / $n ) ]
    #puts "elon $elon"
    for { set iat 1 } { $iat <= $nat } { incr iat } {
      set at1 [ get_atom_entry coords=cts1  atom_number= $iat ]
      set at2 [ get_atom_entry coords=cts2 atom_number= $iat ]
      if { $tsrelative } then {
        set atp [ list [ lindex $at1 0 ] \
                      [ expr [ lindex $at1 1 ] + $elon * [ lindex $at2 1 ] ] \
                      [ expr [ lindex $at1 2 ] + $elon * [ lindex $at2 2 ] ] \
                      [ expr [ lindex $at1 3 ] + $elon * [ lindex $at2 3 ] ] ]
      } else {
        set atp [ list [ lindex $at1 0 ] \
                      [ expr (1. - $elon ) * [ lindex $at1 1 ] + $elon * [ lindex $at2 1 ] ] \
                      [ expr (1. - $elon ) * [ lindex $at1 2 ] + $elon * [ lindex $at2 2 ] ] \
                      [ expr (1. - $elon ) * [ lindex $at1 3 ] + $elon * [ lindex $at2 3 ] ] ]
      }
      replace_atom_entry coords=cp atom_number= $iat atom_entry= $atp
    }
    write_xyz file=tmp.xyz coords=cp
    exec cat tmp.xyz >> $file
    puts "Written frame $i of $n"
  }
  exec rm -f tmp.xyz
  delete_object cts1
  delete_object cts2
  delete_object cp
  pop_banner_flag
}


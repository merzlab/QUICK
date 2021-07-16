 /******************************************************************************
  
            C-interface between ChemShell and DL-FIND

  Provides tcl-callable commands:
    dlf_c

  Provides fortran-callable commands:
    dlf_get_params_
    dlf_error_
    dlf_get_gradient_
    dlf_get_multistate_gradients_
    dlf_get_hessian_
    dlf_put_coords_
    dlf_update_
    dlf_get_procinfo_
    dlf_get_taskfarm_
    dlf_put_procinfo_ (dummy)
    dlf_put_taskfarm_ (dummy)
    
  Calls the fortran command:
    dl_find_

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
  
  $Author: jkk $                           
  $Date: 2006/09/28 09:09:07 $             
  $Revision: 1.5 $                         
  $Source: /c/qcg/cvs/psh/chemsh/src/dlf/interface.c,v $ 
  $State: Exp $                            
  
  ****************************************************************************/

#define debug 0

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <setjmp.h>

#include "objects.h"
#include "tcl.h"
#include "chemsh.h"
#include "dutil.h"
#include "dblock.h"
#include "dfragment.h"
#include "dmatrix.h"

extern Tcl_Interp *chemsh_interp;

#ifdef LINUXF2C
#define dl_find_ dl_find__
#define dlf_get_gradient_ dlf_get_gradient__
#define dlf_get_multistate_gradients_ dlf_get_multistate_gradients__
#define dlf_get_hessian_ dlf_get_hessian__
#define dlf_error_ dlf_error__
#define dlf_put_coords_ dlf_put_coords__
#define dlf_get_params_ dlf_get_params__
#define dlf_update_     dlf_update__
#define dlf_get_procinfo_ dlf_get_procinfo__
#define dlf_get_taskfarm_ dlf_get_taskfarm__
#define dlf_put_procinfo_ dlf_put_procinfo__
#define dlf_put_taskfarm_ dlf_put_taskfarm__
#define dlf_mpi_initialize_ dlf_mpi_initialize__
#endif

#ifdef NOC_
#define dl_find_ dl_find
#define dlf_get_gradient_ dlf_get_gradient
#define dlf_get_multistate_gradients_ dlf_get_multistate_gradients
#define dlf_get_hessian_ dlf_get_hessian
#define dlf_error_ dlf_error
#define dlf_put_coords_ dlf_put_coords
#define dlf_get_params_ dlf_get_params
#define dlf_update_     dlf_update
#define dlf_get_procinfo_ dlf_get_procinfo
#define dlf_get_taskfarm_ dlf_get_taskfarm
#define dlf_put_procinfo_ dlf_put_procinfo
#define dlf_put_taskfarm_ dlf_put_taskfarm
#define dlf_mpi_initialize_ dlf_mpi_initialize
#endif

#if CRAYXX
#define dl_find_ DL_FIND
#define dlf_get_gradient_ DLF_GET_GRADIENT
#define dlf_get_multistate_gradients_ DLF_GET_MULTISTATE_GRADIENTS
#define dlf_get_hessian_ DLF_GET_HESSIAN
#define dlf_error_ DLF_ERROR
#define dlf_put_coords_ DLF_PUT_COORDS
#define dlf_get_params_ DLF_GET_PARAMS
#define dlf_update_     DLF_UPDATE
#define dlf_get_procinfo_ DLF_GET_PROCINFO
#define dlf_get_taskfarm_ DLF_GET_TASKFARM
#define dlf_put_procinfo_ DLF_PUT_PROCINFO
#define dlf_put_taskfarm_ DLF_PUT_TASKFARM
#define dlf_mpi_initialize_ DLF_MPI_INITIALIZE
#endif


jmp_buf dlf_jump_buffer;

int add_dlf_commands(Tcl_Interp *interp)
{
  int DlfCmd( ClientData clientData,
		  Tcl_Interp *interp,
		  int argc,
		  const char *argv[]);

  Tcl_CreateCommand(interp, "dlf_c",DlfCmd, (ClientData) NULL,
		    (Tcl_CmdDeleteProc *) NULL);
  return 0;
}

#define MAXFRAME 100
static Frag f;
static Frag f2[MAXFRAME];
static Frag finp;
static Frag fres;
static Frag fres2;
static Frag ftmp;
static Matrix energy;
static Matrix gradient;
static Matrix hessian;
static Matrix coupling;
static ObjList linp, lres, lres2, ltmp;

static int *map, *map2, *list;
static int img_flag; /* decides if "image=" should be added to the argumetns to $theory.eandg */

/* Array sizes */
static INT nvarin, nspec;
static int nframe_c, nweight_c, nat, nz, ncons, nconn;
/* nmass is always number of atoms */

int DlfCmd(ClientData clientData,
		Tcl_Interp *interp,
		int argc,
		const char *argv[])
{
  const char *s;
  const char *name;
  const char **split1;
  int count;
  INT iccode, i, natoms, natoms_nodum, nb6, lm5, nbig, nincon, nconstr, nrad, nback, iret;
  int retval,iframe,result_flag;

  ObjList lc, lc2[MAXFRAME], le, lg, lisc;
  long errn;

  int icode, mcore, nfrozen, natm;
  INT nvarin2;

  INT master;

  char tclbuff[4096];

  int imultistate, needcoupling;

  FILE *fptest;

  extern void dl_find_ (INT *, INT *, INT *, INT *);
  extern void dlf_mpi_initialize_ ();

#ifdef VALIDATE_MEM
  printf("validating %s %d\n",__FILE__,__LINE__);
  Tcl_ValidateAllMemory(__FILE__,__LINE__);
#endif

  /* Get pointer to input coordinates */
  GetVar(s,"coords");
  linp = get_objlist(s,"fragment",CHEMSH_OBJ_EXISTS,0);
  if(!linp){
    printf("DL-Find: bad fragment tag %s\n",s);
    return TCL_ERROR;
  }
  finp = (Frag) linp->object->data;
  nat = finp->natom;

  /* get second DL-Find coordinates */
  nframe_c=0;
  for(iframe=0;iframe<MAXFRAME;iframe++){
    f2[iframe] = NULL ;
  }
  if(GetOptVar(s,"coords2")){
    Tcl_SplitList(interp,s,&count,&split1);
    nframe_c=count;
    if(nframe_c>MAXFRAME){
      printf("DL-Find WARNING: number of frames in coord2 %d\n",nframe_c);
      printf("   is larger than the maximum of %d. Only %d frames will be used!\n",MAXFRAME,MAXFRAME);
      nframe_c=MAXFRAME;
    }
    for(iframe=0;iframe<nframe_c;iframe++){
      lc2[iframe] = get_objlist(split1[iframe],"fragment",CHEMSH_OBJ_EXISTS,0);
      if(!lc2[iframe]){
	printf("DL-Find: bad fragment tag %s\n",s);
	return TCL_ERROR;
      }
      f2[iframe] = (Frag) lc2[iframe]->object->data;
      if(f2[iframe]->natom != nat) {
	printf("DL-Find: number of atoms in coords and coords2 has to be equal!\n");
	return TCL_ERROR;
      }
    }
  }

  GetVar(name,"tmp_coords");
  lc = get_objlist(name,"fragment",CHEMSH_OBJ_CREATE,CHEMSH_OBJ_VOLATILE);
  if(!lc){
    printf("invalid tmp_coords variable \n");
     return TCL_ERROR;   
  }
  f = (Frag) lc->object->data;

  /* Copy in the structure */
  FRAG_copy(f,finp); 

  GetVar(name,"tmp_energy");
  le = get_objlist(name,"matrix",CHEMSH_OBJ_CREATE,CHEMSH_OBJ_VOLATILE);
  if(!le){
    printf("invalid tmp_energy variable \n");
    return TCL_ERROR;   
  }
  energy = (Matrix) le->object->data;

  GetVar(name,"tmp_gradient");
  lg = get_objlist(name,"matrix",CHEMSH_OBJ_CREATE,CHEMSH_OBJ_VOLATILE);
  if(!lg){
    printf("invalid tmp_gradient variable \n");
    return TCL_ERROR;
  }
  gradient = (Matrix) lg->object->data;

  /* img_flag */
  if(GetOptVar(s,"img_flag")){
    img_flag=1;
  } else {
    img_flag=0;
  }

  /* Interstate couplings for conical intersection searches
     Note in ChemShell TCL procedures, the keyword for storing couplings
     is now 'isc_gradient' instead of 'coupling'.
     This is to avoid a clash in the hybrid procedure where the coupling 
     keyword is used for a different purpose. */
  GetVar(s,"imultistate");
  Tcl_GetInt(interp,s,&imultistate);
  GetVar(s,"needcoupling");
  Tcl_GetInt(interp,s,&needcoupling);

  if (imultistate > 0 && needcoupling) {
    GetVar(name,"tmp_coupling");
    lisc = get_objlist(name,"matrix",CHEMSH_OBJ_CREATE,CHEMSH_OBJ_VOLATILE);
    if(!lisc){
      printf("invalid tmp_coupling variable \n");
      return TCL_ERROR;
    }
    coupling = (Matrix) lisc->object->data;
    strcpy(tclbuff,"if { [ catch {$theory.init $theory_args coords=$tmp_coords energy=$tmp_energy  gradient=$tmp_gradient  isc_gradient=$tmp_coupling states= $states } errmsg ] } then { set errno 1; puts stdout $errmsg } else {set errno 0 } ");
  }
  else if (imultistate > 0) {
    strcpy(tclbuff,"if { [ catch {$theory.init $theory_args coords=$tmp_coords energy=$tmp_energy  gradient=$tmp_gradient states= $states } errmsg ] } then { set errno 1; puts stdout $errmsg } else {set errno 0 } ");
  } else {
    /* default single state initialisation */
    strcpy(tclbuff,"if { [ catch {$theory.init $theory_args coords=$tmp_coords energy=$tmp_energy  gradient=$tmp_gradient} errmsg ] } then { set errno 1; puts stdout $errmsg } else {set errno 0 } ");
  }

  if (Tcl_Eval(interp, tclbuff) != TCL_OK) goto trap;

  strcpy(tclbuff,"$errno");
  iret = Tcl_ExprLong(interp, tclbuff,&errn);

  if(iret != TCL_OK || errn != 0) goto trap;

  GetVar(s,"result");
  /* check if result file exists. If it does, use it; if it doesn't, create a
   * new one */
  fptest=fopen(s,"r");
  if(!fptest){
    result_flag = CHEMSH_OBJ_CREATE;
  } else {
    result_flag = CHEMSH_OBJ_EXISTS;
    fclose(fptest);
  }

  lres = get_objlist(s,"fragment",result_flag,CHEMSH_OBJ_PERSISTENT);
  if(!lres){
    printf("DL-Find can not open result file %s, will create a new one.\n",s);
    result_flag = CHEMSH_OBJ_CREATE;
    /* This trap ensures that the calculation runs if the result file is empty
     * (which it may be after chemshell was killed) */
    lres = get_objlist(s,"fragment",result_flag,CHEMSH_OBJ_PERSISTENT);
    if(!lres){
      printf("DL-Find: bad fragment tag %s\n",s);
      return TCL_ERROR;
    }
  }
  fres = (Frag) lres->object->data;

  /* Copy in the structure in dlf_put_coords (commented out because the file
        should not be overwritten in case dl-find stops early */
  /* FRAG_copy(fres,finp); */

  /* get second DL-Find coordinates */
  if(GetOptVar(s,"result2")){
    /* check if result2 file exists. If it does, us it; if it doesn't, create
     * a new one */
    fptest=fopen(s,"r");
    if(!fptest){
      result_flag = CHEMSH_OBJ_CREATE;
    } else {
      result_flag = CHEMSH_OBJ_EXISTS;
      fclose(fptest);
    }

    lres2 = get_objlist(s,"fragment",result_flag,CHEMSH_OBJ_PERSISTENT);
    if(!lres2){
      printf("DL-Find: bad fragment tag %s\n",s);
      return TCL_ERROR;
    }
    fres2 = (Frag) lres2->object->data;
    /* Copy in the structure in dlf_put_coords*/
    /* FRAG_copy(fres2,finp); */
  } else {
    fres2 = NULL;
  }

  /* Set Array sizes */
  nz=nat;
  ncons=0;
  nconn=0;

  /* Look for constraints */
  if(GetOptVar(s,"constraints")){
    Tcl_SplitList(interp,s,&count,&split1);
    ncons=count;
    printf("Number of constraints %d\n",ncons);
    ckfree((char *) split1);
  }
  
  /* Look for connections */
  if(GetOptVar(s,"connect")){
    Tcl_SplitList(interp,s,&count,&split1);
    nconn=count;
    printf("Number of user-defined connections %d\n",nconn);
    ckfree((char *) split1);
  }
  
  nspec = nat + nz + 5*ncons + 2*nconn + nat;
  nvarin = 3 * nat;

  /* Find out nvarin2 */
  nvarin2= 3*nat*nframe_c + nat;

  /* Look for weights */
  if(GetOptVar(s,"weights")){
    Tcl_SplitList(interp,s,&count,&split1);
    if(count!=nat) {
      printf("DL-Find: array weights has to contain a real number for each atom in coords\n");
      return TCL_ERROR;
    }
    nvarin2=nvarin2+nat;
    nweight_c=nat;
    ckfree((char *) split1);
  }

  icode = setjmp (dlf_jump_buffer);
  if(icode){
    printf("jumped out of DL-Find with code %d\n",icode);
    iret = -icode;
  }else{

    master = 1; /* every instance is treated as master in a parallel run for now */

    /* ChemShell has initialized MPI so this simply passes information
       on processors etc. to DL-FIND. For serial jobs sets MPI variables
       to dummy values */
    dlf_mpi_initialize_();

    /* Call search */
    dl_find_ (&nvarin, &nvarin2, &nspec, &master);

  }

  printf("exit %d\n", (int) iret);
#ifdef VALIDATE_MEM
  printf("validating %s %d\n",__FILE__,__LINE__);
  Tcl_ValidateAllMemory(__FILE__,__LINE__);
#endif

  /*  if(iret)return TCL_ERROR;*/

  fflush(stdout);

  if(iret != 0)printf("exited DL-Find search: iret=%d \n", (int) iret);

  rel_objlist(lres);
  if(fres2) rel_objlist(lres2); 
  for(iframe=0;iframe<MAXFRAME;iframe++){
    if(f2[iframe]) rel_objlist(lc2[iframe]);
  }

  rel_objlist(le);
  rel_objlist(lg);
  if(coupling) rel_objlist(lisc);
  rel_objlist(lc);
  rel_objlist(linp);

  fflush(stdout);

  if (imultistate > 0 && needcoupling) {
  strcpy(tclbuff,"if { [ catch {$theory.kill $theory_args coords=$tmp_coords energy=$tmp_energy  gradient=$tmp_gradient isc_gradient=$tmp_coupling states= $states } errmsg ] } then { set errno 1; puts stdout $errmsg } else {set errno 0 } ");
  } else if (imultistate > 0) {
  strcpy(tclbuff,"if { [ catch {$theory.kill $theory_args coords=$tmp_coords energy=$tmp_energy  gradient=$tmp_gradient states= $states } errmsg ] } then { set errno 1; puts stdout $errmsg } else {set errno 0 } ");
  } else {
  strcpy(tclbuff,"if { [ catch {$theory.kill $theory_args coords=$tmp_coords energy=$tmp_energy  gradient=$tmp_gradient} errmsg ] } then { set errno 1; puts stdout $errmsg } else {set errno 0 } ");
  }

  if (Tcl_Eval(interp, tclbuff) != TCL_OK) goto trap;

#ifdef VALIDATE_MEM
  printf("validating %s %d\n",__FILE__,__LINE__);
  Tcl_ValidateAllMemory(__FILE__,__LINE__);
#endif

  if(iret < 0 ) return TCL_ERROR;
  return TCL_OK;

 trap:
  printf("trap\n");
  fprintf(stderr,"%s\n",Tcl_GetStringResult(interp));
  return TCL_ERROR;

}

void dlf_get_params_(INT *n, 
		     INT *nvar2,
		     INT *nsp,
		     double coords[], 
		     double coords2[],
		     INT spec[],
		     INT *ierr,
		     double *tolerance,
		     INT *printl,
		     INT *maxcycle,
		     INT *maxene,
		     INT *tatoms,
		     INT *icoord,
		     INT *iopt ,
		     INT *iline ,
		     double *maxstep,
		     double *scalestep,
		     INT *lbfgs_mem ,
		     INT *nimage ,
		     double *nebk,
		     INT *dump ,
		     INT *restart ,
		     INT *nz_i ,
		     INT *ncons_i ,
		     INT *nconn_i ,
		     INT *update ,
		     INT *maxupd,
		     double *delta ,
		     double *soft ,
		     INT *inithessian ,
		     INT *carthessian ,
		     INT *tsrel ,
		     INT *maxrot ,
		     double *tolrot ,
		     INT *nframe ,
		     INT *nmass ,
		     INT *nweight ,
		     double *timestep ,
		     double *fric0 ,
		     double *fricfac ,
		     double *fricp ,
		     INT *imultistate,
		     INT *state_i ,
		     INT *state_j ,
		     double *pf_c1 ,
		     double *pf_c2 ,
		     double *gp_c3 ,
		     double *gp_c4 ,
		     double *ln_t1 ,
		     double *ln_t2 ,
		     INT *printfile ,
		     double *tolerance_e,
		     double *distort,
		     INT *massweight,
		     double *minstep,
		     INT *maxdump,
		     INT *task,
		     double *temperature,
		     INT *po_pop_size,
		     double *po_radius_base,
		     double *po_contraction,
		     double *po_tol_r_base,
		     double *po_tolerance_g,
		     INT *po_distribution,
		     INT *po_maxcycle,
		     INT *po_init_pop_size,
		     INT *po_reset,
		     double *po_mutation_rate,
		     double *po_death_rate,
		     double *po_scalefac,
		     INT *po_nsave,
		     INT *ntasks,
		     INT *tdlf_farm,
                     INT *n_po_scaling,
		     double *neb_climb_test,
		     double *neb_freeze_test,
		     INT *nzero,
		     INT *coupled_states,
		     INT *qtsflag,
		     INT *imicroiter,
		     INT *maxmicrocycle,
		     INT *micro_esp_fit
  ){
  int i, retval, loc, istart;
  const char *s;
  const char **split1;
  int count, ivar,iframe;
  double svar;
  Tcl_Interp *interp;
  int *tmp_i;
  /* Variables needed for Residue input in hdlcopt form */
  const char **split2, **split3, **split4;
  int count1, count2, count3, count4, nres, ix, j; /*, j, ncount;*/

  *ierr = 1;
  if(nat * 3 != *n) {
    printf("Error: Number of atoms inconsistent!\n");
    printf("Nat*3= %d     N= %d \n",nat*3, *n);
    *ierr=1;
    return;
  }
  *nz_i = nz;
  *ncons_i = ncons;
  *nconn_i = nconn;
  *nmass = nat;

  interp = chemsh_interp;

  /* defaults are set in dl_find.f90:dlf_default_set */
  if(GetOptVar(s,"tolerance")){
    if(Tcl_GetDouble(interp,s,tolerance)){
      printf("bad tolerance\n");
      return;
    }
  }
  if(GetOptVar(s,"tolerance_e")){
    if(Tcl_GetDouble(interp,s,tolerance_e)){
      printf("bad tolerance_e\n");
      return;
    }
  }
  if(GetOptVar(s,"printl")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad printl\n");
      return;
    }
    *printl=(INT) i;
  }
  if(GetOptVar(s,"printf")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad printf\n");
      return;
    }
    *printfile=(INT) i;
  }
  if(GetOptVar(s,"maxcycle")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad maxcycle\n");
      return;
    }
    *maxcycle=(INT) i;
  }
  if(GetOptVar(s,"maxene")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad maxene\n");
      return;
    }
    *maxene=(INT) i;
  }
  if(GetOptVar(s,"icoord")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad icoord\n");
      return;
    }
    *icoord=(INT) i;
  }
  if(GetOptVar(s,"iopt")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad iopt\n");
      return;
    }
    *iopt=(INT) i;
  }
  if(GetOptVar(s,"iline")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad iline\n");
      return;
    }
    *iline=(INT) i;
  }
  if(GetOptVar(s,"maxstep")){
    if(Tcl_GetDouble(interp,s,maxstep)){
      printf("bad maxstep\n");
      return;
    }
  }
  if(GetOptVar(s,"scalestep")){
    if(Tcl_GetDouble(interp,s,scalestep)){
      printf("bad scalestep\n");
      return;
    }
  }
  if(GetOptVar(s,"lbfgs_mem")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad lbfgs_mem\n");
      return;
    }
    *lbfgs_mem=(INT) i;
  }
  if(GetOptVar(s,"nimage")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad nimage\n");
      return;
    }
    *nimage=(INT) i;
  }
  if(GetOptVar(s,"nebk")){
    if(Tcl_GetDouble(interp,s,nebk)){
      printf("bad nebk\n");
      return;
    }
  }
  if(GetOptVar(s,"dump")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad dump\n");
      return;
    }
    *dump=(INT) i;
  }
  if(GetOptVar(s,"restart")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad restart\n");
      return;
    }
    *restart=(INT) i;
  }
  if(GetOptVar(s,"update")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad update\n");
      return;
    }
    *update=(INT) i;
  }
  if(GetOptVar(s,"maxupdate")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad maxupdate\n");
      return;
    }
    *maxupd=(INT) i;
  }
  if(GetOptVar(s,"delta")){
    if(Tcl_GetDouble(interp,s,delta)){
      printf("bad delta\n");
      return;
    }
  }
  if(GetOptVar(s,"soft")){
    if(Tcl_GetDouble(interp,s,soft)){
      printf("bad soft\n");
      return;
    }
  }
  if(GetOptVar(s,"inithessian")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad inithessian\n");
      return;
    }
    *inithessian=(INT) i;
  }
  if(GetOptVar(s,"carthessian")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad carthessian\n");
      return;
    }
    *carthessian=(INT) i;
  }
  if(GetOptVar(s,"tsrelative")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad tsrelative\n");
      return;
    }
    *tsrel=(INT) i;
  }
  if(GetOptVar(s,"maxrot")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad maxrot\n");
      return;
    }
    *maxrot=(INT) i;
  }
  if(GetOptVar(s,"tolrot")){
    if(Tcl_GetDouble(interp,s,tolrot)){
      printf("bad tolrot\n");
      return;
    }
  }
  if(GetOptVar(s,"timestep")){
    if(Tcl_GetDouble(interp,s,timestep)){
      printf("bad timestep\n");
      return;
    }
  }
  if(GetOptVar(s,"fric0")){
    if(Tcl_GetDouble(interp,s,fric0)){
      printf("bad fric0\n");
      return;
    }
  }
  if(GetOptVar(s,"fricfac")){
    if(Tcl_GetDouble(interp,s,fricfac)){
      printf("bad fricfac\n");
      return;
    }
  }
  if(GetOptVar(s,"fricp")){
    if(Tcl_GetDouble(interp,s,fricp)){
      printf("bad fricp\n");
      return;
    }
  }
  /* Conical intersection search options */
  if(GetOptVar(s,"imultistate")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad imultistate\n");
      return;
    }
    *imultistate=(INT) i;
  }
  if(GetOptVar(s,"state_i")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad state_i\n");
      return;
    }
    *state_i=(INT) i;
  }
  if(GetOptVar(s,"state_j")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad state_j\n");
      return;
    }
    *state_j=(INT) i;
  }
  if(GetOptVar(s,"coupled_states")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad coupled_states\n");
      return;
    }
    *coupled_states=(INT) i;
  }
  if(GetOptVar(s,"pf_c1")){
    if(Tcl_GetDouble(interp,s,pf_c1)){
      printf("bad pf_c1\n");
      return;
    }
  }
  if(GetOptVar(s,"pf_c2")){
    if(Tcl_GetDouble(interp,s,pf_c2)){
      printf("bad pf_c2\n");
      return;
    }
  }
  if(GetOptVar(s,"gp_c3")){
    if(Tcl_GetDouble(interp,s,gp_c3)){
      printf("bad gp_c3\n");
      return;
    }
  }
  if(GetOptVar(s,"gp_c4")){
    if(Tcl_GetDouble(interp,s,gp_c4)){
      printf("bad gp_c4\n");
      return;
    }
  }
  if(GetOptVar(s,"ln_t1")){
    if(Tcl_GetDouble(interp,s,ln_t1)){
      printf("bad ln_t1\n");
      return;
    }
  }
  if(GetOptVar(s,"ln_t2")){
    if(Tcl_GetDouble(interp,s,ln_t2)){
      printf("bad ln_t2\n");
      return;
    }
  }
  /* End of conical intersection search options */
  if(GetOptVar(s,"distort")){
    if(Tcl_GetDouble(interp,s,distort)){
      printf("bad distort\n");
      return;
    }
  }
  if(GetOptVar(s,"massweight")){
    /* it is sufficient if the variable is set */
    *massweight=1;
  }
  if(GetOptVar(s,"minstep")){
    if(Tcl_GetDouble(interp,s,minstep)){
      printf("bad minstep\n");
      return;
    }
  }
  if(GetOptVar(s,"maxdump")){
    /* this is a non-documented command line argument (maxdump) 
       used to test restart options. It causes dump files (restart files)
       only to be written after up to maxdump energy evaluations */
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad maxdump\n");
      return;
    }
    *maxdump=(INT) i;
  }
  if(GetOptVar(s,"task")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad task number\n");
      return;
    }
    *task=(INT) i;
  }
  if(GetOptVar(s,"temperature")){
    if(Tcl_GetDouble(interp,s,temperature)){
      printf("bad temperature\n");
      return;
    }
  }

  /* Global minimisation options */
  if(GetOptVar(s,"po_pop_size")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad po_pop_size\n");
      return;
    }
    *po_pop_size = (INT) i;
  }
  if(GetOptVar(s,"po_radius")){
    if(Tcl_GetDouble(interp,s,po_radius_base)){
      printf("bad po_radius\n");
      return;
    }
  }  
  if(GetOptVar(s,"po_contraction")){
    if(Tcl_GetDouble(interp,s,po_contraction)){
      printf("bad po_contraction\n");
      return;
    }
  } 
  if(GetOptVar(s,"po_tolerance_r")){
    if(Tcl_GetDouble(interp,s,po_tol_r_base)){
      printf("bad po_tolerance_r\n");
      return;
    }
  }
  if(GetOptVar(s,"po_tolerance_g")){
    if(Tcl_GetDouble(interp,s,po_tolerance_g)){
      printf("bad po_tolerance_g\n");
      return;
    }
  }
  if(GetOptVar(s,"po_distrib")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad po_distrib\n");
      return;
    }
    *po_distribution = (INT) i;
  }
  if(GetOptVar(s,"po_maxcycle")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad po_maxcycle\n");
      return;
    }
    *po_maxcycle = (INT) i;
  }
  if(GetOptVar(s,"po_init_pop_size")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad po_init_pop_size\n");
      return;
    }
    *po_init_pop_size = (INT) i;
  }
  if(GetOptVar(s,"po_reset")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad po_reset\n");
      return;
    }
    *po_reset = (INT) i;
  }
  if(GetOptVar(s,"po_mutation_rate")){
    if(Tcl_GetDouble(interp,s,po_mutation_rate)){
      printf("bad po_mutation_rate\n");
      return;
    }
  }
  if(GetOptVar(s,"po_death_rate")){
    if(Tcl_GetDouble(interp,s,po_death_rate)){
      printf("bad po_death_rate\n");
      return;
    }
  }
  if(GetOptVar(s,"po_scalefac")){
    if(Tcl_GetDouble(interp,s,po_scalefac)){
      printf("bad po_scalefac\n");
      return;
    }
  }
  if(GetOptVar(s,"po_nsave")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad po_nsave\n");
      return;
    }
    *po_nsave = (INT) i;
  }
  /* Interface for n_po_scaling != 0 not implemented yet */
  *n_po_scaling = 0;

  if(GetOptVar(s,"neb_climb_test")){
    if(Tcl_GetDouble(interp,s,neb_climb_test)){
      printf("bad neb_climb_test\n");
      return;
    }
  }
  if(GetOptVar(s,"neb_freeze_test")){
    if(Tcl_GetDouble(interp,s,neb_freeze_test)){
      printf("bad neb_freeze_test\n");
      return;
    }
  }

  /* Task farming options */
  /* Number of workgroups (task farms in DL-FIND parlance)
     Should be 1 for a serial build */
#ifdef MASTERSLAVE
  *ntasks = ParNWorkgroups();
#else
  *ntasks = 1;
#endif

  /* ChemShell will set up the task farm */
  *tdlf_farm = 0;

  if(GetOptVar(s,"nzero")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad nzero\n");
      return;
    }
    *nzero=(INT) i;
  }

  if(GetOptVar(s,"qtsflag")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad qtsflag\n");
      return;
    }
    *qtsflag=(INT) i;
  }

  /* entries of residue numbers / frozen atom information */
  if(GetOptVar(s,"spec")){
    Tcl_SplitList(interp,s,&count,&split1);
    if(count != nat){
      printf("Number of elements in spec array must be equal to numer of atoms\n");
      ckfree((char *) split1);
      *ierr=1;
      return;
    }
    for(i=0; i<nat; i++){
      if(Tcl_GetInt(interp,split1[i],&ivar)){
	printf("bad spec entry %d\n",i+1);
	*ierr=1;
	return;
      }
      spec[i]=(INT) ivar;
    }
    ckfree((char *) split1);
  } else {
    for(i=0; i<nat; i++){
      spec[i]=0;
    }
    /* Handle Residues as in HDLCopt */
    if(GetOptVar(s,"residues")){
      Tcl_SplitList(interp,s,&count1,&split1);

      /* First list is the residue names */
      Tcl_SplitList(interp,split1[0],&count2,&split2);

      nres = count2;
      printf("There are %d residues\n",nres);
      if(nres && debug){
	printf("residue names are: ");
	for(i=0;i<nres;i++){
	  printf("%s ",split2[i]);
	}
	printf("\n");
      }

      /*ncount=0;
	nback=natoms-1;*/

      /* Subsequent list is the residue members */
      Tcl_SplitList(interp,split1[1],&count3,&split3);

      for(i=0;i<nres;i++){

	Tcl_SplitList(interp,split3[i],&count4,&split4);
	tmp_i = (int *) ckalloc(count4*sizeof(int));

	if(debug){
	  printf("%s \n",split2[i]);
	}
	count=0;
	for(j=0;j<count4;j++){
	  Tcl_GetInt(interp,split4[j],&ix);
	  
	  if(debug){
	    printf("%d ",ix);
	    count++;
	    if(count > 11){
	      printf("\n");
	      count=0;
	    }
	  }
	  if(ix > nat || ix < 1){
	    printf("error: bad residue member %d\n",ix); 
	    return;
	  }

	  ix--;
	  if(spec[ix]){
	    printf("warning: duplicate entry %d\n",ix); 
	  }else{
	    spec[ix] = (INT) 1+i;
	    if (!strcmp(split2[i],"fixed")){
	      spec[ix] = (INT) -1;
	    }
	  }
	  tmp_i[j]=ix+1;

	}

	/* periodically map atoms of coords= closest to last atom 
	   in each residue */
	if( tmp_i[0] != -1 ) {
	  FRAG_construct_group(f, tmp_i, count4, f->atoms[ix].pos );
	}
	ckfree((char *) tmp_i);

	if(debug)if(count)printf("\n");
	ckfree((char *) split4);
      }
      ckfree((char *) split3);
      ckfree((char *) split2);
      ckfree((char *) split1);
    }
    /* End of residue input in HDLCopt form */

  }
  for(i=0; i<nat; i++){
    spec[i+nat]=f->atoms[i].znum;
  }

  /* Handle user-defined connections */ 
  if(nconn>0){
    if(!GetOptVar(s,"connect")){
      printf("Error getting the connect variable\n");
      return;
    }
    Tcl_SplitList(interp,s,&count1,&split1);
    for(i=0;i<count1;i++){
      Tcl_SplitList(interp,split1[i],&count2,&split2);
      if (count2 != 2 ) {
	printf("Error: number of atoms in connection %d is not 2 (is %d)\n",i+1,count2);
	return;
      }
      /* read the atom numbers */
      for(count3=0;count3<2;count3++){
	j=0;
	Tcl_GetInt(interp,split2[count3],&j);
	if( j>0 && j<=nat) {
	  spec[2*nat+5*ncons+2*i+count3] = (INT) j;
	} else {
	  printf("Error: Atom number %d in connection %d out of range (%d)\n",count3+1,i+1,j);
	  return;
	}
      }
    }
  }

  /* Handle constraints */ 
  if(ncons>0){
    if(!GetOptVar(s,"constraints")){
      printf("Error getting the constraints variable\n");
      return;
    }
    Tcl_SplitList(interp,s,&count1,&split1);
    for(i=0;i<count1;i++){
      Tcl_SplitList(interp,split1[i],&count2,&split2);
      /* count3 is the constraint type */
      if (!strncmp(split2[0],"bond",4)) count3 = 1;
      else if (!strncmp(split2[0],"bonddiff",8)) count3 = 5;
      else if (!strncmp(split2[0],"angle",5)) count3 = 2;
      else if (!strncmp(split2[0],"torsion",7)) count3 = 3;
      else if (!strncmp(split2[0],"cart",4)) count3 = 4;
      else if (!strncmp(split2[0],"diffbond",8)) count3 = 5;
      else {
	printf("Error: bad constraint type: %s \n", split2[0]);
	return;
      }
      for(j=1;j<5;j++){
	spec[2*nat+5*i+j]=0;
      }
      spec[2*nat+5*i]=count3; /* Type of constraint */
      for(j=1;j<count2;j++){
	/* was Tcl_GetInt(interp,split2[j],spec+2*nat+5*i+j); */
	Tcl_GetInt(interp,split2[j],&ivar);
	spec[2*nat+5*i+j]= (INT) ivar;
      }
      ckfree((char *) split2);
    }
    ckfree((char *) split1);
  }

  /* handle active atoms */
  if(GetOptVar(s,"active_atoms")){
    tmp_i = (int *) ckalloc(nat*sizeof(int));
    for(i=0; i<nat; i++){
      tmp_i[i]=0;
    }
    Tcl_SplitList(interp,s,&count1,&split1);
    for(i=0; i<count1; i++){
      Tcl_GetInt(interp,split1[i],&j);
      if(j>0 && j<=nat) tmp_i[j-1]=1;
    } 
    for(i=0; i<nat; i++){
      if(tmp_i[i]==0) spec[i]=(INT)-1;
    }
    ckfree((char *) split1);
    ckfree((char *) tmp_i);
  }
  
  /* Microiterative optimisation */
  if(GetOptVar(s,"microiterative")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad microiterative\n");
      return;
    }
    *imicroiter=(INT) i;
  }
  /* microiterative region specification
     0 = unassigned (non-microiterative) or outer region
     1 = inner region
  */
  tmp_i = (int *) ckalloc(nat*sizeof(int));
  for(i=0; i<nat; i++){
    tmp_i[i]=0;
  }  
  if (GetOptVar(s,"inner_atoms")){ 
    Tcl_SplitList(interp,s,&count1,&split1);
    for(i=0; i<count1; i++){
      Tcl_GetInt(interp,split1[i],&j);
      if(j>0 && j<=nat) tmp_i[j-1] = 1;
    } 
    ckfree((char *) split1);
  }
  istart = nat + nz + 5*ncons + 2*nconn;
  for(i=0; i<nat; i++){
    spec[istart + i] = tmp_i[i];
  }
  ckfree((char *) tmp_i);

  if(GetOptVar(s,"maxmicrocycle")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad maxmicrocycle\n");
      return;
    }
    *maxmicrocycle=(INT) i;
  }
  if(GetOptVar(s,"micro_esp_fit")){
    if(Tcl_GetInt(interp,s,&i)){
      printf("bad micro_esp_fit\n");
      return;
    }
    *micro_esp_fit=(INT) i;
  }


  /* read atom coords */
  for(i=0; i<nat; i++){
    loc=i*3;

    coords[loc]  =f->atoms[i].pos.x[0];
    coords[loc+1]=f->atoms[i].pos.x[1];
    coords[loc+2]=f->atoms[i].pos.x[2];

  }

  /* read set of fragments from coords2 */
  *nframe=nframe_c;
  loc=0;
  for(iframe=0;iframe<nframe_c;iframe++){
    if(!f2[iframe]){
      printf("Error: coords2 transfer wrong!\n");
      *ierr=1;
      return;
    }
    for(i=0; i<nat; i++){
      coords2[loc]  =f2[iframe]->atoms[i].pos.x[0];
      coords2[loc+1]=f2[iframe]->atoms[i].pos.x[1];
      coords2[loc+2]=f2[iframe]->atoms[i].pos.x[2];
      loc=loc+3;
    }
    if(loc>*nvar2){
      printf("Error: coords2 transfer location wrong!\n");
      *ierr=1;
      return;
    }
  }

  /* read weights */
  if(GetOptVar(s,"weights")){
    Tcl_SplitList(interp,s,&count1,&split1);
    for(i=0; i<count1; i++){
      if(Tcl_GetDouble(interp,split1[i],&svar)){
	printf("Error: array weight has to contain real numbers\n");
	return;
      }
      if(3*nat*nframe_c+i >= *nvar2){
	printf("Error in coords2 array\n");
	return;
      }
      coords2[3*nat*nframe_c+i]=svar;
    } 
    ckfree((char *) split1);
    *nweight=nat;
  } else {
    *nweight=0;
    if(nweight_c > 0) {
      printf("Error in reading weights\n");
      *ierr=1;
      return;
    }
  }

  /* set masses */
  if(GetOptVar(s,"mass")){
    Tcl_SplitList(interp,s,&count1,&split1);
    if(count1 != nat){
      printf("Number of elements in mass array must be equal to numer of atoms\n");
      ckfree((char *) split1);
      *ierr=1;
      return;
    }
    for(i=0; i<nat; i++){
      if(Tcl_GetDouble(interp,split1[i],&svar)){
	printf("bad mass entry %d\n",i+1);
	*ierr=1;
	return;
      }
      if(3*nat*nframe_c+*nweight+i >= *nvar2){
        printf("Error writing masses into coords2 array!\n");
        return;
      }
      coords2[3*nat*nframe_c+*nweight+i]=svar;
    }
    ckfree((char *) split1);
  } else {
    for(i=0; i<nat; i++){
/*      TAB_average_mass(f->atoms[i].znum,&svar);*/
      TAB_abundant_mass(f->atoms[i].znum,&svar);
      if(3*nat*nframe_c+*nweight+i >= *nvar2){
        printf("Error writing masses into coords2 array!\n");
        return;
      }
      coords2[3*nat*nframe_c+*nweight+i]=svar;
    }
  }
  /* print out masses - also for future use */
  if(*printl > 4){ /* JK prefers the next line*/
/*  if(*printl >= 4){ */
      printf("Masses used:\n");
      printf("set mass { ");
      for(i=0; i<nat; i++){
	  printf("%15.10f ",coords2[3*nat*nframe_c+*nweight+i]);
      }
      printf("}\n");
      fflush(stdout);
  }

  *ierr = 0;
}

void dlf_error_(){
  fprintf(stderr,"dlf_error called\n");
  fflush(stdout);
  longjmp(dlf_jump_buffer,1);
}

void dlf_get_gradient_( INT *nvar,
			double *coords,
			double *ener,
			double *forces,
			INT *iimage,
			INT *iter,
			INT *ierr
  )
{
  Tcl_Interp *interp;
  char tclbuff[4096];
  int i, nat, loc, iret;
  char *iter_arg;
  const char *qm_args;

#ifdef VALIDATE_MEM
  printf("validating %s %d\n",__FILE__,__LINE__);
  Tcl_ValidateAllMemory(__FILE__,__LINE__);
#endif

  interp = chemsh_interp;

  /* printf("eandg of image %d\n",*iimage); */

  /* Set coordinates */
  nat = f->natom;
  for(i=0; i<nat; i++){
    loc=i*3;

    f->atoms[i].pos.x[0]=coords[loc]  ;
    f->atoms[i].pos.x[1]=coords[loc+1];
    f->atoms[i].pos.x[2]=coords[loc+2];
  }

  /* ESP fit setting */
  if (*iter != -1) {
    /* This hack to set iter is taken from the HDLCOpt interface) */

    if (!*iter)
      printf("\nEnergy calculation with SCF-iteration\n\n");
    else
      printf("\nEnergy calculation without SCF-iteration\n\n");

    Tcl_Eval(interp, "global hybrid");
    qm_args = Tcl_GetVar(interp, "hybrid(qm_theory_args)", 0);
    /*     printf("hybrid(qm_theory_args) = {%s}\n", qm_args); */
    strcpy(tclbuff, qm_args);    

    if ( (iter_arg = strstr(tclbuff, "iter=")) ) {
      /* hybrid(qm_theory_args) already contains iter option. */
      iter_arg[5] = '0' + *iter;
    }
    else {
      /* hybrid(qm_theory_args) does not contain iter option yet. */
      sprintf(tclbuff, "%s iter=%d", qm_args, (int) *iter);
    }

    if (!Tcl_SetVar(interp, "hybrid(qm_theory_args)", tclbuff, 0)) {
      printf("ERROR: Unable to set hybrid(qm_theory_args)!\n");
      *ierr = 1;
      return;
    }    
  }
  
  /* Now evaluate energy and forces */
  if ( img_flag ) {
    /* transferring the NEB image number only works for mndo for the moment! */
    sprintf(tclbuff,"$theory.eandg $theory_args coords=$tmp_coords energy=$tmp_energy  gradient=$tmp_gradient image= %d",*iimage); 
  } else {
    strcpy(tclbuff,"$theory.eandg $theory_args coords=$tmp_coords energy=$tmp_energy  gradient=$tmp_gradient");
  }

  iret = Tcl_Eval(interp, tclbuff);

  if(iret != TCL_OK) {
    printf("DL-FIND: energy/gradient evaluation failed\n");
    fprintf(stderr,"%s\n",Tcl_GetStringResult(interp));
    fflush(stdout);
    fflush(stderr);
    *ierr = 1;
    return;
  }
  
  /* Apply any restraint terms */
  sprintf(tclbuff,"if { $restraints != \"undefined\" } { dl-find_rst accumulate }");
  if(debug)printf("evaluating... %s\n",tclbuff);
  if (Tcl_Eval(interp,tclbuff) != TCL_OK) {
    fprintf(stderr,"%s\n",Tcl_GetStringResult(interp));
    printf("DL-FIND: restraint evaluation or test failed\n");
    fflush(stdout);
    *ierr = 1;
    return;
  }
      
  /* read gradient */

  for(i=0;i<f->natom;i++){
    loc = i*3;
    /*printf("JKc gradient %d - %f %f %f \n",i,gradient->d[3*i+0],gradient->d[3*i+1],gradient->d[3*i+2]);*/
    forces[loc]   = gradient->d[3*i+0];
    forces[loc+1] = gradient->d[3*i+1];
    forces[loc+2] = gradient->d[3*i+2];
  }

/*  for(i=0;i<3*f->natom;i++){
    printf("JKc force %d : %f\n",i,forces[i]);
    }*/
    

#ifdef VALIDATE_MEM
  printf("validating %s %d\n",__FILE__,__LINE__);
  Tcl_ValidateAllMemory(__FILE__,__LINE__);
#endif

  /* get the energy */
  *ener = energy->d[0];

  *ierr = 0;

}


void dlf_get_multistate_gradients_( INT *nvar,
			double *coords,
			double *ener,
			double *forces,
			double *coup,
			INT *needcoupling,
			INT *iimage,
			INT *ierr
  )
{
  /* Caution: the current version of this routine assumes that
     exactly two states are required (as is the case for the 
     PF, GP, and LN algorithms). */

  Tcl_Interp *interp;
  char tclbuff[4096];
  int i, nat, loc, iret;

#ifdef VALIDATE_MEM
  printf("validating %s %d\n",__FILE__,__LINE__);
  Tcl_ValidateAllMemory(__FILE__,__LINE__);
#endif

  interp = chemsh_interp;

  /* Set coordinates */
  nat = f->natom;
  for(i=0; i<nat; i++){
    loc=i*3;

    f->atoms[i].pos.x[0]=coords[loc]  ;
    f->atoms[i].pos.x[1]=coords[loc+1];
    f->atoms[i].pos.x[2]=coords[loc+2];
  }

  /* Now evaluate energies and forces */
  if (*needcoupling) {
    strcpy(tclbuff,"$theory.multegh $theory_args coords=$tmp_coords energy=$tmp_energy  gradient=$tmp_gradient isc_gradient=$tmp_coupling states= $states");
  } else {
    strcpy(tclbuff,"$theory.multeg $theory_args coords=$tmp_coords energy=$tmp_energy  gradient=$tmp_gradient states= $states");
  }
  iret = Tcl_Eval(interp, tclbuff);

  if(iret != TCL_OK) {
    printf("DL-FIND: energy/gradient evaluation failed\n");
    fprintf(stderr,"%s\n",Tcl_GetStringResult(interp));
    fflush(stdout);
    fflush(stderr);
    *ierr = 1;
    return;
  }

  /* read gradient */
  for(i=0;i<f->natom*2;i++){
    loc = i*3;
    forces[loc]   = gradient->d[3*i+0];
    forces[loc+1] = gradient->d[3*i+1];
    forces[loc+2] = gradient->d[3*i+2];
    gradient->d[3*i+0] = 0.;
    gradient->d[3*i+1] = 0.;
    gradient->d[3*i+2] = 0.;
  }

  if (*needcoupling) {
    for (i = 0; i < f->natom; i++) {
      loc = i * 3;
      coup[loc]   = coupling->d[3*i+0];
      coup[loc+1] = coupling->d[3*i+1];
      coup[loc+2] = coupling->d[3*i+2];
    }
  }

  /* get the energy */
  ener[0] = energy->d[0];
  ener[1] = energy->d[1];
  energy->d[0] = 0.;
  energy->d[1] = 0.;

  /* Apply any restraint terms */
  sprintf(tclbuff,"if { $restraints != \"undefined\" } { dl-find_rst accumulate }");
  if(debug)printf("evaluating... %s\n",tclbuff);
  if (Tcl_Eval(interp,tclbuff) != TCL_OK) {
    fprintf(stderr,"%s\n",Tcl_GetStringResult(interp));
    printf("DL-FIND: restraint evaluation or test failed\n");
    fflush(stdout);
    *ierr = 1;
    return;
  }

  /* Add gradient introduced by the restraints. 
     The restraint energy and gradient is only added to the first set 
     of gradient by the routine above, but should be added to both. */
  for(i=0;i<f->natom;i++){
    loc = i*3;
    forces[loc]   += gradient->d[3*i+0];
    forces[loc+1] += gradient->d[3*i+1];
    forces[loc+2] += gradient->d[3*i+2];
    loc = (f->natom * 3) + i*3;
    forces[loc]   += gradient->d[3*i+0];
    forces[loc+1] += gradient->d[3*i+1];
    forces[loc+2] += gradient->d[3*i+2];
  }
  ener[0] += energy->d[0];
  ener[1] += energy->d[0];

  /* Copying back the energy here is not necessary for the calculation, but
     ensures compatibility with the energy tests in the example optimisations.
     The first element will be overwritten by dlf_put_coords anyway. */
  energy->d[0] = ener[0];
  energy->d[1] = ener[1];

#ifdef VALIDATE_MEM
  printf("validating %s %d\n",__FILE__,__LINE__);
  Tcl_ValidateAllMemory(__FILE__,__LINE__);
#endif

  *ierr = 0;

}


void dlf_get_hessian_( INT *nvar,
			double *coords,
			double *hess_arg,
			INT *ierr
  )
{
  Tcl_Interp *interp;
  char tclbuff[4096];
  int i, j, loc, iret;
  ObjList lh;


#ifdef VALIDATE_MEM
  printf("validating %s %d\n",__FILE__,__LINE__);
  Tcl_ValidateAllMemory(__FILE__,__LINE__);
#endif

  /* catch it -IMPROVE */
/*  *ierr = 1;
    return;*/

  interp = chemsh_interp;
  *ierr = 1;

  /* Set coordinates */
  nat = f->natom;
  for(i=0; i<nat; i++){
    loc=i*3;

    f->atoms[i].pos.x[0]=coords[loc]  ;
    f->atoms[i].pos.x[1]=coords[loc+1];
    f->atoms[i].pos.x[2]=coords[loc+2];
  }

  /* initiate a matrix to hold the hessian */
  lh = get_objlist("dl-find.hessian","matrix",CHEMSH_OBJ_CREATE,CHEMSH_OBJ_VOLATILE);
  if(!lh){
    printf("Error initialising Hessian \n");
    return;
  }
  hessian = (Matrix) lh->object->data;


  /* Now evaluate the hessian */
  strcpy(tclbuff,"$theory.hess $theory_args coords=$tmp_coords energy=$tmp_energy  gradient=$tmp_gradient hessian=dl-find.hessian ");
  iret = Tcl_Eval(interp, tclbuff);

  if(iret != TCL_OK) {
    printf("DL-FIND: hessian evaluation failed\n");
    fprintf(stderr,"%s\n",Tcl_GetStringResult(interp));
    rel_objlist(lh);
    fflush(stdout);
    fflush(stderr);
    return;
  }
  
  /* Apply any restraint terms -- THESE ARE MISSED OUT AT PRESENT */
      
  /* read hessian */
  loc=0;
  for(i=0;i<*nvar;i++){
    for(j=0;j<*nvar;j++){
      hess_arg[loc] = hessian->d[loc];
      loc += 1;
    }
  }  
  rel_objlist(lh);

  *ierr = 0;

}

/* write out the coords (specified by the input keyword result) */
/* mode:
   1: optimised coordinates
   2: Transition mode: coordinates displaced relative to those in mode 1
   <0 : NEB image of number -mode

   TWK: iam is related to parallel opts. Currently a dummy argument here. 
*/
void dlf_put_coords_( INT *nvar,
		      INT *mode,
		      double *e0,
		      double *coords,
		      INT *iam
  )
{
  Tcl_Interp *interp;
  int i, nat, loc;
  char buff[4096];

#ifdef VALIDATE_MEM
  printf("validating %s %d\n",__FILE__,__LINE__);
  Tcl_ValidateAllMemory(__FILE__,__LINE__);
#endif

  interp = chemsh_interp;

  /* set dlf energy */
  if (*mode > 0) {
    energy->d[0]= *e0;
  }

  /* Set coordinates 1 of dlf to fragment specified as result*/
  nat = f->natom;
  if(*mode == 1) {
    /* Copy in the structure - only needed in first instance*/
    FRAG_copy(fres,finp);
    for(i=0; i<nat; i++){
      loc=i*3;
      
      fres->atoms[i].pos.x[0]= coords[loc]  ;
      fres->atoms[i].pos.x[1]= coords[loc+1];
      fres->atoms[i].pos.x[2]= coords[loc+2];
    }
    flush_object(lres->name);
  } else if(*mode == 2) {
    if(fres2) {
      /* Copy in the structure - only needed in first instance*/
      FRAG_copy(fres2,finp);
      for(i=0; i<nat; i++){
	loc=i*3;
	
	fres2->atoms[i].pos.x[0]=coords[loc]  ;
	fres2->atoms[i].pos.x[1]=coords[loc+1];
	fres2->atoms[i].pos.x[2]=coords[loc+2];
      }
      flush_object(lres2->name);
    }
  } else if((*mode > 2) && (*mode < 6)) {
    /* a duplicate of the mode==1 branch for now*/
    /* Copy in the structure - only needed in first instance*/
    FRAG_copy(fres,finp);
    for(i=0; i<nat; i++){
      loc=i*3;

      fres->atoms[i].pos.x[0]= coords[loc]  ;
      fres->atoms[i].pos.x[1]= coords[loc+1];
      fres->atoms[i].pos.x[2]= coords[loc+2];
    }
    flush_object(lres->name);
  } else if(*mode < 0) {
    /* create a fragment, put the coords into it, write it to disk 
       and delete it */
    sprintf(buff,"neb_%d.c",- *mode);
    if(debug){ printf("writing file %s\n",buff); }

    ltmp = get_objlist(buff,"fragment",CHEMSH_OBJ_CREATE,CHEMSH_OBJ_PERSISTENT);
    if(!ltmp){
      printf("DL-Find: error allocating temporary fragment %s\n",buff);
      /* no error handling yet - but this is not terribly important */
      return;
    }
    ftmp = (Frag) ltmp->object->data;
    FRAG_copy(ftmp,finp);

    /* set coordinates */
    for(i=0; i<nat; i++){
      loc=i*3;
      
      ftmp->atoms[i].pos.x[0]= coords[loc]  ;
      ftmp->atoms[i].pos.x[1]= coords[loc+1];
      ftmp->atoms[i].pos.x[2]= coords[loc+2];
    }

    flush_object(ltmp->name);

    rel_objlist(ltmp);

  } else {
    printf("Error: Bad mode number");
    /* no error handling yet - but this is not terribly important */
  }

#ifdef VALIDATE_MEM
  printf("validating %s %d\n",__FILE__,__LINE__);
  Tcl_ValidateAllMemory(__FILE__,__LINE__);
#endif

}

/* update the neighbour-list (in case of QM/MM or MM calculations */
void dlf_update_( )
{
  Tcl_Interp *interp;
  char tclbuff[4096];
  int iret;

  interp = chemsh_interp;

  /* Now evaluate energy and forces */
  strcpy(tclbuff,"$theory.update");
  iret = Tcl_Eval(interp, tclbuff);

  if(iret != TCL_OK) {
    printf("DL-FIND: $theory.update failed\n");
    fprintf(stderr,"%s\n",Tcl_GetStringResult(interp));
    fflush(stdout);
    fflush(stderr);
  }
}


/* Parallel interface routines 

   It would be good to add an ierr argument to each of these
   to flag up errors.
   This would require updating GAMESS/CRYSTAL etc. interfaces 
   as well.

 */


/* DL-FIND expects the total number of processors, processor rank, 
   and the global MPI communicator to be passed in here. 
   However, when running under ChemShell DL-FIND only runs on 
   1 node per workgroup. Therefore only these nodes should be 
   passed in. */
void dlf_get_procinfo_(INT *nprocs,
		      INT *rank,
		      INT *global_comm) {

#ifdef MASTERSLAVE
  /* Workgroup slave nodes should be invisible to DL-FIND,
     so number of processors = no of workgroups */
  *nprocs = ParNWorkgroups();
  /* Rank = rank of workgroup */
  *rank = ParWorkgroupID();
  /* Must only communicate between the master node on each workgroup */
  *global_comm = ParCommCounterparts();
#else
  *nprocs = 1;
  *rank = 0;
  *global_comm = -1;
#endif

}

/* Pass information about task farm workgroups to DL-FIND.
   Note that workgroups are called 'task farms' within DL-FIND.
   As DL-FIND only runs on 1 node per workgroup, the workgroup 
   communicator should not be passed in (it is not used anyway) */
void dlf_get_taskfarm_(INT *nworkgroups,
		      INT *workgroupsize,
		      INT *workgroupnode,
		      INT *workgroupid,
		      INT *workgroup_comm,
		      INT *counterparts_comm) {

#ifdef MASTERSLAVE
  *nworkgroups = ParNWorkgroups();
  /* Workgroup slave nodes should be invisible to DL-FIND */
  *workgroupsize = 1;
  /* This should always be 0 */
  *workgroupnode = ParNodeID();
  *workgroupid = ParWorkgroupID();
  /* Set to raise an error if the workgroup communicator
     is ever used in DL-FIND */
  *workgroup_comm = -1;
  /* Communication between the master nodes on each workgroup
     - note for ChemShell this is interchangeable with global_comm,
       but for other programs (e.g. CRYSTAL) global_comm really is
       the global communicator as DL-FIND runs on all nodes */
  *counterparts_comm = ParCommCounterparts();
#else
  *nworkgroups = 1;
  *workgroupsize = 1;
  *workgroupnode = 0;
  *workgroupid = 0;
  *workgroup_comm = -1;
  *counterparts_comm = -1;
#endif

}

/* Dummy routine - ChemShell will always set up the task farm */
void dlf_put_procinfo_(INT *nprocs,
		      INT *rank,
		      INT *global_comm) {

  /* When an ierr argument is added, an error should be returned
     here so that dlf_fail can be called */
  printf("dlf.c: dlf_put_procinfo called in error.\n");
  
}

/* Dummy routine - ChemShell will always set up the task farm */
void dlf_put_taskfarm_(INT *nworkgroups,
		      INT *workgroupsize,
		      INT *workgroupnode,
		      INT *workgroupid,
		      INT *workgroup_comm,
		      INT *counterparts_comm) {

  /* When an ierr argument is added, an error should be returned
     here so that dlf_fail can be called */
  printf("dlf.c: dlf_put_taskfarm called in error.\n");

}

#include <stdlib.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/libcuest-headers/include/cuest.h"
#else
#include <cuest.h>
#endif

#include "helper_status.h"
#include "helper_workspace.h"
#include "quick_cuest.h"
#include "util.h"

static size_t                    i_atom_grids;
static cuestAtomGridParameters_t atom_grid_param;
static cuestAtomGrid_t          *atom_grids;

void
cuest_create_atom_grid_setup ()
{
    i_atom_grids = 0;

    const size_t atom_grids_siz = quick_cuest_data.natom * sizeof (cuestAtomGrid_t);
    atom_grids                  = malloc (atom_grids_siz);
    add_host_alloc (atom_grids_siz);

    checkCuestErrors (cuestParametersCreate (CUEST_ATOMGRID_PARAMETERS, &atom_grid_param));
}

void
cuest_create_atom_grid (int64_t nrad, double *r, double *w, int64_t *nang)
{
    checkCuestErrors (cuestAtomGridCreate (quick_cuest_struct.handle, nrad, r, w, (uint64_t *)nang,
                                           atom_grid_param, &atom_grids[i_atom_grids++]));

#ifdef CUESTDEBUG
    DEBUGLOG ("created atom grid %zu\n", i_atom_grids - 1);
    DEBUGLOG ("\tnrad=%lld\n", nrad);
    DEBUGLOG ("\tr=");
    for (size_t i = 0; i < nrad; ++i)
        DEBUGLOG ("%f ", r[i]);
    DEBUGLOG ("\n");
    DEBUGLOG ("\tw=");
    for (size_t i = 0; i < nrad; ++i)
        DEBUGLOG ("%f ", w[i]);
    DEBUGLOG ("\n");
    DEBUGLOG ("\tnang=");
    for (size_t i = 0; i < nrad; ++i)
        DEBUGLOG ("%lld ", nang[i]);
    DEBUGLOG ("\n");
#endif
}

void
cuest_destroy_atom_grid ()
{

    for (int i = 0; i < quick_cuest_data.natom; ++i)
        checkCuestErrors (cuestAtomGridDestroy (atom_grids[i]));
    free (atom_grids);
    free_host_alloc (quick_cuest_data.natom * sizeof (cuestAtomGrid_t));
    checkCuestErrors (cuestParametersDestroy (CUEST_ATOMGRID_PARAMETERS, atom_grid_param));
}

void
cuest_deinit_xc_grid ()
{
    checkCuestErrors (cuestMolecularGridDestroy (quick_cuest_struct.molgrid));
    freeWorkspace (quick_cuest_struct.persistXCGridWorkspace);
    checkCuestErrors (cuestXCIntPlanDestroy (quick_cuest_struct.XCIntPlan));
    freeWorkspace (quick_cuest_struct.persistXCIntPlanWorkspace);
}

#ifndef OSHELL
#define OSHELL
#include "xc_subs.h"
#endif
#undef OSHELL

#include "xc_subs.h"

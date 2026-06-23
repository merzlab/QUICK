#include <stdint.h>
#include <stdlib.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"
#else
#include <cuest.h>
#endif

#include "helper_status.h"
#include "helper_workspace.h"

#include "quick_cuest.h"

void
cuest_init (uint64_t natom, uint64_t nshell, uint64_t MAXPRIM, double *xyz)
{
    freopen ("cuest.log", "w", stdout);

    // =========== //
    // init handle //
    // =========== //

    cuestHandleParameters_t handle_params;
    checkCuestErrors (cuestParametersCreate (CUEST_HANDLE_PARAMETERS, &handle_params));
    checkCuestErrors (cuestCreate (handle_params, &quick_cuest_struct.handle));
    checkCuestErrors (cuestParametersDestroy (CUEST_HANDLE_PARAMETERS, handle_params));

    // ========================== //
    // init workspace descriptors //
    // ========================== //

    quick_cuest_struct.persistWD = malloc (sizeof (cuestWorkspaceDescriptor_t));
    quick_cuest_struct.tmpWD     = malloc (sizeof (cuestWorkspaceDescriptor_t));

    // ========= //
    // init info //
    // ========= //

    quick_cuest_data.natom   = natom;
    quick_cuest_data.nshell  = nshell;
    quick_cuest_data.MAXPRIM = MAXPRIM;
    quick_cuest_data.xyz     = xyz;
}

void
cuest_deinit ()
{
    checkCuestErrors (cuestOEIntPlanDestroy (quick_cuest_struct.OEIntPlan));
    freeWorkspace (quick_cuest_struct.persistOEIntPlanWorkspace);
    checkCuestErrors (cuestAOPairListDestroy (quick_cuest_struct.AOPairList));
    freeWorkspace (quick_cuest_struct.persistAOPairListWorkspace);
    checkCuestErrors (cuestAOBasisDestroy (quick_cuest_struct.basis));
    freeWorkspace (quick_cuest_struct.persistAOBasisWorkspace);
    free (quick_cuest_struct.persistWD);
    free (quick_cuest_struct.tmpWD);
    checkCuestErrors (cuestDestroy (quick_cuest_struct.handle));
}

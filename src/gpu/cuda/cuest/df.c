#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/fake_cuda_headers/cuda_runtime.h"
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"
#else
#include <cuda_runtime.h>
#include <cuest.h>
#endif

#include "helper_status.h"
#include "helper_workspace.h"

#include "quick_cuest.h"
#include "util.h"

void
cuest_init_dfint_plan (double hyb_coeff)
{
    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD     = quick_cuest_struct.tmpWD;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;

    cuestDFIntPlanParameters_t dfint_plan_parameters;
    checkCuestErrors (cuestParametersCreate (CUEST_DFINTPLAN_PARAMETERS, &dfint_plan_parameters));

    // set hybrid exchange fraction
    checkCuestErrors (cuestParametersConfigure (CUEST_DFINTPLAN_PARAMETERS, dfint_plan_parameters,
                                                CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION,
                                                &hyb_coeff, sizeof (double)));

    checkCuestErrors (cuestDFIntPlanCreateWorkspaceQuery (
        handle, quick_cuest_struct.basis, quick_cuest_struct.auxBasis,
        quick_cuest_struct.AOPairList, dfint_plan_parameters, persistWD, tmpWD,
        &quick_cuest_struct.DFIntPlan));

    MEMLOG ("Density Fit Integral Plan");
    quick_cuest_struct.persistDFIntPlanWorkspace = allocateWorkspace (persistWD);
    cuestWorkspace_t *tmpDFIntPlanWorkspace      = allocateWorkspace (tmpWD);
    checkCuestErrors (cuestDFIntPlanCreate (handle, quick_cuest_struct.basis,
                                            quick_cuest_struct.auxBasis,
                                            quick_cuest_struct.AOPairList, dfint_plan_parameters,
                                            quick_cuest_struct.persistDFIntPlanWorkspace,
                                            tmpDFIntPlanWorkspace, &quick_cuest_struct.DFIntPlan));

    checkCuestErrors (cuestParametersDestroy (CUEST_DFINTPLAN_PARAMETERS, dfint_plan_parameters));
    freeWorkspace (tmpDFIntPlanWorkspace);
}

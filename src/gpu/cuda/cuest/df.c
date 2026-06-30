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

void
cuest_init_dfint_plan ()
{
    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD     = quick_cuest_struct.tmpWD;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;

    cuestDFIntPlanParameters_t dfint_plan_parameters;
    checkCuestErrors (cuestParametersCreate (CUEST_DFINTPLAN_PARAMETERS, &dfint_plan_parameters));
    checkCuestErrors (cuestDFIntPlanCreateWorkspaceQuery (
        handle, quick_cuest_struct.basis, quick_cuest_struct.auxBasis,
        quick_cuest_struct.AOPairList, dfint_plan_parameters, persistWD, tmpWD,
        &quick_cuest_struct.DFIntPlan));

#ifdef CUESTDEBUG
    printf ("%s: density fitting integral plan persistWD allocation size:\t%zu\n", __func__,
            persistWD->deviceBufferSizeInBytes);
    printf ("%s: density fitting integral plan tmpWD allocation size:\t%zu\n", __func__,
            tmpWD->deviceBufferSizeInBytes);
#endif
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

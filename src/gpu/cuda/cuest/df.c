#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/fake_cuda_headers/cuda_runtime.h"
#include "/Users/msun/rehs2026/cuest/libcuest-headers/include/cuest.h"
#else
#include <cuda_runtime.h>
#include <cuest.h>
#endif

#include "helper_status.h"
#include "helper_workspace.h"

#include "cuest_funcs.h"
#include "quick_cuest.h"
#include "util.h"

void
cuest_init_df (double hyb_coeff, int32_t fnl)
{
    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD     = quick_cuest_struct.tmpWD;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;

    double lrc_x_frac = 0, lrc_x_w = 0;

    // SOURCE: ORCA docs: https://www.faccts.de/docs/orca/6.0/manual/contents/detailed/model.html
    switch (fnl) {
        case CUEST_FUNCTIONAL_LCWPBE:
            // hyb_coeff = 0.0;
            lrc_x_frac = 1.0;
            lrc_x_w    = 0.4;
            break;
        case CUEST_FUNCTIONAL_WB97X:
            // hyb_coeff = 0.157706;
            lrc_x_frac = 0.842294;
            lrc_x_w    = 0.3;
            break;
        case CUEST_FUNCTIONAL_WB97XV:
            // hyb_coeff = 0.167;
            lrc_x_frac = 0.833;
            lrc_x_w    = 0.3;
            break;
        case CUEST_FUNCTIONAL_CAMB3LYP:
            // hyb_coeff = 0.19
            lrc_x_frac = 0.46;
            lrc_x_w    = 0.33;
            break;
        case CUEST_FUNCTIONAL_HSE06:
            // hyb_coeff = 0.25
            lrc_x_frac = -0.25;
            lrc_x_w    = 0.11;
            break;
    }

    hyb_coeff -= lrc_x_frac;

#ifdef CUESTDEBUG
    DEBUGLOG ("-------- Functional Info --------\n");
    DEBUGLOG ("hyb_coeff  = %f\n", hyb_coeff);
    DEBUGLOG ("lrc_x_frac = %f\n", lrc_x_frac);
    DEBUGLOG ("lrc_x_w    = %f\n", lrc_x_w);
    DEBUGLOG ("---------------------------------\n");
#endif

    cuestDFIntPlanParameters_t dfint_plan_parameters;
    checkCuestErrors (cuestParametersCreate (CUEST_DFINTPLAN_PARAMETERS, &dfint_plan_parameters));

    checkCuestErrors (cuestParametersConfigure (CUEST_DFINTPLAN_PARAMETERS, dfint_plan_parameters,
                                                CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION,
                                                &hyb_coeff, sizeof (double)));
    checkCuestErrors (cuestParametersConfigure (CUEST_DFINTPLAN_PARAMETERS, dfint_plan_parameters,
                                                CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION,
                                                &lrc_x_frac, sizeof (double)));
    checkCuestErrors (cuestParametersConfigure (CUEST_DFINTPLAN_PARAMETERS, dfint_plan_parameters,
                                                CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA,
                                                &lrc_x_w, sizeof (double)));

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

void
cuest_deinit_df ()
{
    checkCuestErrors (cuestDFIntPlanDestroy (quick_cuest_struct.DFIntPlan));
    freeWorkspace (quick_cuest_struct.persistDFIntPlanWorkspace);
}

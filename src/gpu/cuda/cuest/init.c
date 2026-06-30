#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/fake_cuda_headers/cuda_runtime.h"
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"
#else
#include <cuest.h>
#endif

#include "helper_status.h"
#include "helper_workspace.h"

#include "quick_cuest.h"

void
cuest_init (int64_t natom, int64_t nshell, int64_t nbasis, int64_t nauxshell, int64_t maxcontract,
            int64_t maxcontract_aux, double *xyz, double *chg, int64_t nextatom, double *extxyz,
            double *extchg)
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

    quick_cuest_struct.tmpWD     = malloc (sizeof (cuestWorkspaceDescriptor_t));
    quick_cuest_struct.persistWD = malloc (sizeof (cuestWorkspaceDescriptor_t));

    // ========= //
    // init info //
    // ========= //

    quick_cuest_data.natom           = natom;
    quick_cuest_data.ntotalatom      = natom + nextatom;
    quick_cuest_data.nshell          = nshell;
    quick_cuest_data.nauxshell       = nauxshell;
    quick_cuest_data.maxcontract     = maxcontract;
    quick_cuest_data.maxcontract_aux = maxcontract_aux;
    quick_cuest_data.xyz             = xyz;
    quick_cuest_data.nbasis          = nbasis;

    // =========== //
    // init arrays //
    // =========== //

    quick_cuest_memchk.tmp_o = malloc (nbasis * nbasis * sizeof (double));
    quick_cuest_memchk.tmp_C = NULL;

    size_t chg_siz        = natom * sizeof (double);
    size_t extchg_siz     = nextatom * sizeof (double);
    size_t xyz_gpu_siz    = 3 * chg_siz;
    size_t extxyz_gpu_siz = 3 * extchg_siz;

    // charges on CPU

    if ((quick_cuest_data.allchg = malloc (chg_siz + extchg_siz)) == NULL) {
        fprintf (stderr, "malloc failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    memcpy (quick_cuest_data.allchg, chg, chg_siz);
    memcpy (quick_cuest_data.allchg + natom, extchg, extchg_siz);

    // xyz

    if (cudaMalloc ((void **)&quick_cuest_data.allxyz_gpu, xyz_gpu_siz + extxyz_gpu_siz)
        != cudaSuccess) {
        fprintf (stderr, "cudaMalloc failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    if (cudaMemcpy (quick_cuest_data.allxyz_gpu, xyz, xyz_gpu_siz, cudaMemcpyHostToDevice)
        != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    if (cudaMemcpy ((uint8_t *)quick_cuest_data.allxyz_gpu + xyz_gpu_siz, extxyz, extxyz_gpu_siz,
                    cudaMemcpyHostToDevice)
        != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    // charges

    if (cudaMalloc ((void **)&quick_cuest_data.allchg_gpu, chg_siz + extchg_siz) != cudaSuccess) {
        fprintf (stderr, "cudaMalloc failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    if (cudaMemcpy (quick_cuest_data.allchg_gpu, chg, chg_siz, cudaMemcpyHostToDevice)
        != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    if (cudaMemcpy ((uint8_t *)quick_cuest_data.allchg_gpu + chg_siz, extchg, extchg_siz,
                    cudaMemcpyHostToDevice)
        != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }
}

void
cuest_init_pair_list (double cutoff)
{
    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD     = quick_cuest_struct.tmpWD;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;

    uint64_t natom = quick_cuest_data.natom;
    double  *xyz   = quick_cuest_data.xyz;

    // ================ //
    // set up pair list //
    // ================ //

    cuestAOPairListParameters_t pair_list_params;
    checkCuestErrors (cuestParametersCreate (CUEST_AOPAIRLIST_PARAMETERS, &pair_list_params));
    checkCuestErrors (cuestAOPairListCreateWorkspaceQuery (handle, quick_cuest_struct.basis, natom,
                                                           xyz, cutoff, pair_list_params, persistWD,
                                                           tmpWD, &quick_cuest_struct.AOPairList));

#ifdef CUESTDEBUG
    printf ("%s: pair list persistWD allocation size:\t%zu\n", __func__,
            persistWD->deviceBufferSizeInBytes);
    printf ("%s: pair list tmpWD allocation size:\t%zu\n", __func__,
            tmpWD->deviceBufferSizeInBytes);
#endif
    quick_cuest_struct.persistAOPairListWorkspace = allocateWorkspace (persistWD);
    cuestWorkspace_t *tmpAOPairListWorkspace      = allocateWorkspace (tmpWD);

    checkCuestErrors (
        cuestAOPairListCreate (handle, quick_cuest_struct.basis, natom, xyz, cutoff,
                               pair_list_params, quick_cuest_struct.persistAOPairListWorkspace,
                               tmpAOPairListWorkspace, &quick_cuest_struct.AOPairList));
    checkCuestErrors (cuestParametersDestroy (CUEST_AOPAIRLIST_PARAMETERS, pair_list_params));
    freeWorkspace (tmpAOPairListWorkspace);
    // free (xyz_flat);
}

void
cuest_deinit_oei_plan ()
{
    checkCuestErrors (cuestOEIntPlanDestroy (quick_cuest_struct.OEIntPlan));
    freeWorkspace (quick_cuest_struct.persistOEIntPlanWorkspace);
}

void
cuest_deinit ()
{
    checkCuestErrors (cuestDFIntPlanDestroy (quick_cuest_struct.DFIntPlan));
    freeWorkspace (quick_cuest_struct.persistDFIntPlanWorkspace);
    checkCuestErrors (cuestAOPairListDestroy (quick_cuest_struct.AOPairList));
    freeWorkspace (quick_cuest_struct.persistAOPairListWorkspace);
    checkCuestErrors (cuestAOBasisDestroy (quick_cuest_struct.basis));
    freeWorkspace (quick_cuest_struct.persistAOBasisWorkspace);
    free (quick_cuest_struct.persistWD);
    free (quick_cuest_struct.tmpWD);
    checkCuestErrors (cuestDestroy (quick_cuest_struct.handle));

    free (quick_cuest_data.allchg);
    free (quick_cuest_data.ifshell);
    free (quick_cuest_memchk.chk_katom_ktype_kprim);
    free (quick_cuest_memchk.tmp_o);

    if (cudaFree (quick_cuest_data.allxyz_gpu) != cudaSuccess) {
        fprintf (stderr, "cudaFree failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    if (cudaFree (quick_cuest_data.allchg_gpu) != cudaSuccess) {
        fprintf (stderr, "cudaFree failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }
}

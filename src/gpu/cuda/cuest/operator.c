#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/fake_cuda_headers/cuda_runtime.h"
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"
#else
#include <cuda_runtime.h>
#include <cuest.h>
#endif

#include "correction.h"
#include "helper_status.h"
#include "helper_workspace.h"
#include "quick_cuest.h"
#include "util.h"

/**
 * Initializes the one-electron integrals plan `OEIntPlan` in `quick_cuest_struct`
 *
 * `cuest_init` must have been called before calling `cuest_init_oei_plan`
 */
void
cuest_init_oei_plan ()
{
    // ========================== //
    // one-electron integral plan //
    // ========================== //

    cuestOEIntPlanParameters_t oeint_plan_params;
    checkCuestErrors (cuestParametersCreate (CUEST_OEINTPLAN_PARAMETERS, &oeint_plan_params));
    checkCuestErrors (cuestOEIntPlanCreateWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_struct.basis, quick_cuest_struct.AOPairList,
        oeint_plan_params, quick_cuest_struct.persistWD, quick_cuest_struct.tmpWD,
        &quick_cuest_struct.OEIntPlan));

#ifdef CUESTDEBUG
    printf ("%s: oei plan persistWD allocation size:\t%zu\n", __func__,
            quick_cuest_struct.persistWD->deviceBufferSizeInBytes);
    printf ("%s: oei plan tmpWD allocation size:\t%zu\n", __func__,
            quick_cuest_struct.tmpWD->deviceBufferSizeInBytes);
#endif
    quick_cuest_struct.persistOEIntPlanWorkspace = allocateWorkspace (quick_cuest_struct.persistWD);
    cuestWorkspace_t *tmpOEIntPlanWorkspace      = allocateWorkspace (quick_cuest_struct.tmpWD);
    checkCuestErrors (cuestOEIntPlanCreate (quick_cuest_struct.handle, quick_cuest_struct.basis,
                                            quick_cuest_struct.AOPairList, oeint_plan_params,
                                            quick_cuest_struct.persistOEIntPlanWorkspace,
                                            tmpOEIntPlanWorkspace, &quick_cuest_struct.OEIntPlan));

    checkCuestErrors (cuestParametersDestroy (CUEST_OEINTPLAN_PARAMETERS, oeint_plan_params));
    freeWorkspace (tmpOEIntPlanWorkspace);
}

/**
 * @param o should be `nao` x `nao`.
 */
void
cuest_get_oei_S (double *o)
{
    double *d_S;
    size_t  d_S_siz = quick_cuest_data.nbasis * quick_cuest_data.nbasis * sizeof (double);
    if (cudaMalloc ((void **)&d_S, d_S_siz)) {
        fprintf (stderr, "Failed to allocate device buffer\n");
        exit (EXIT_FAILURE);
    }

    cuestOverlapComputeParameters_t overlap_compute_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_OVERLAPCOMPUTE_PARAMETERS, &overlap_compute_params));
    checkCuestErrors (
        cuestOverlapComputeWorkspaceQuery (quick_cuest_struct.handle, quick_cuest_struct.OEIntPlan,
                                           overlap_compute_params, quick_cuest_struct.tmpWD, d_S));

#ifdef CUESTDEBUG
    printf ("%s: overlap tmpWD allocation size:\t%zu\n", __func__,
            quick_cuest_struct.tmpWD->deviceBufferSizeInBytes);
#endif
    cuestWorkspace_t *tmpSWorkspace = allocateWorkspace (quick_cuest_struct.tmpWD);
    checkCuestErrors (cuestOverlapCompute (quick_cuest_struct.handle, quick_cuest_struct.OEIntPlan,
                                           overlap_compute_params, tmpSWorkspace, d_S));

    freeWorkspace (tmpSWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_OVERLAPCOMPUTE_PARAMETERS, overlap_compute_params));

    // ======================== //
    // copy overlap matrix to o //
    // ======================== //

    if (cudaMemcpy (o, d_S, d_S_siz, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    correct_o (o, CORRECT_REORDER_AND_NORM);

#ifdef CUESTDEBUG
    puts ("-------- S --------");
    for (int i = 0; i < quick_cuest_data.nbasis; ++i) {
        for (int j = 0; j < quick_cuest_data.nbasis; ++j)
            printf ("%16.10f", o[i * quick_cuest_data.nbasis + j]);
        putchar ('\n');
    }
    puts ("------ END S ------");
#endif

    if (cudaFree (d_S) != cudaSuccess) {
        fprintf (stderr, "cudaFree failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }
}

void
cuest_get_oei_T (double *o)
{
    double *d_T;
    size_t  d_T_siz = quick_cuest_data.nbasis * quick_cuest_data.nbasis * sizeof (double);
    if (cudaMalloc ((void **)&d_T, d_T_siz) != cudaSuccess) {
        fprintf (stderr, "cudaMalloc failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    cuestKineticComputeParameters_t kinetic_compute_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_KINETICCOMPUTE_PARAMETERS, &kinetic_compute_params));
    checkCuestErrors (
        cuestKineticComputeWorkspaceQuery (quick_cuest_struct.handle, quick_cuest_struct.OEIntPlan,
                                           kinetic_compute_params, quick_cuest_struct.tmpWD, d_T));

#ifdef CUESTDEBUG
    printf ("%s: kinetic tmpWD allocation size:\t%zu\n", __func__,
            quick_cuest_struct.tmpWD->deviceBufferSizeInBytes);
#endif
    cuestWorkspace_t *tmpTWorkspace = allocateWorkspace (quick_cuest_struct.tmpWD);
    checkCuestErrors (cuestKineticCompute (quick_cuest_struct.handle, quick_cuest_struct.OEIntPlan,
                                           kinetic_compute_params, tmpTWorkspace, d_T));

    freeWorkspace (tmpTWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_KINETICCOMPUTE_PARAMETERS, kinetic_compute_params));

    // ======================== //
    // copy kinetic matrix to o //
    // ======================== //

    if (cudaMemcpy (o, d_T, d_T_siz, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    correct_o (o, CORRECT_REORDER_AND_NORM);

#ifdef CUESTDEBUG
    puts ("-------- T --------");
    for (int i = 0; i < quick_cuest_data.nbasis; ++i) {
        for (int j = 0; j < quick_cuest_data.nbasis; ++j)
            printf ("%16.10f", o[i * quick_cuest_data.nbasis + j]);
        putchar ('\n');
    }
    puts ("------ END T ------");
#endif

    if (cudaFree (d_T) != cudaSuccess) {
        fprintf (stderr, "cudaFree failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }
}

void
cuest_get_oei_V (double *o)
{
    double *d_V;
    size_t  d_V_siz = quick_cuest_data.nbasis * quick_cuest_data.nbasis * sizeof (double);
    if (cudaMalloc ((void **)&d_V, d_V_siz) != cudaSuccess) {
        fprintf (stderr, "cudaMalloc failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    cuestPotentialComputeParameters_t potential_compute_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_POTENTIALCOMPUTE_PARAMETERS, &potential_compute_params));
    checkCuestErrors (cuestPotentialComputeWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_struct.OEIntPlan, potential_compute_params,
        quick_cuest_struct.tmpWD, quick_cuest_data.ntotalatom, quick_cuest_data.allxyz_gpu,
        quick_cuest_data.allchg_gpu, d_V));

#ifdef CUESTDEBUG
    printf ("%s: potential tmpWD allocation size:\t%zu\n", __func__,
            quick_cuest_struct.tmpWD->deviceBufferSizeInBytes);
#endif
    cuestWorkspace_t *tmpVWorkspace = allocateWorkspace (quick_cuest_struct.tmpWD);
    checkCuestErrors (
        cuestPotentialCompute (quick_cuest_struct.handle, quick_cuest_struct.OEIntPlan,
                               potential_compute_params, tmpVWorkspace, quick_cuest_data.ntotalatom,
                               quick_cuest_data.allxyz_gpu, quick_cuest_data.allchg_gpu, d_V));

    freeWorkspace (tmpVWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_POTENTIALCOMPUTE_PARAMETERS, potential_compute_params));

    // ========================== //
    // copy potential matrix to o //
    // ========================== //

    if (cudaMemcpy (o, d_V, d_V_siz, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    correct_o (o, CORRECT_REORDER_AND_NORM);

#ifdef CUESTDEBUG
    puts ("-------- V --------");
    for (int i = 0; i < quick_cuest_data.nbasis; ++i) {
        for (int j = 0; j < quick_cuest_data.nbasis; ++j)
            printf ("%16.10f", o[i * quick_cuest_data.nbasis + j]);
        putchar ('\n');
    }
    puts ("------ END V ------");
#endif

    if (cudaFree (d_V) != cudaSuccess) {
        fprintf (stderr, "cudaFree failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }
}

void
cuest_get_eri_J (double *o, double *P)
{
    double *d_J;
    size_t  d_J_siz = quick_cuest_data.nbasis * quick_cuest_data.nbasis * sizeof (double);
    if (cudaMalloc ((void **)&d_J, d_J_siz) != cudaSuccess) {
        fprintf (stderr, "cudaMalloc failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

#ifdef CUESTDEBUG
    puts ("-------- uncorrected cuEST DENSITY MATRIX --------");
    for (int i = 0; i < quick_cuest_data.nbasis; ++i) {
        for (int j = 0; j < quick_cuest_data.nbasis; ++j)
            printf ("%f ", get (P, i, j, quick_cuest_data.nbasis));
        putchar ('\n');
    }
    puts ("------ END uncorrected cuEST DENSITY MATRIX ------");
#endif

    correct_o (P, CORRECT_REORDER_AND_NORM);

#ifdef CUESTDEBUG
    puts ("-------- uncorrected cuEST DENSITY MATRIX --------");
    for (int i = 0; i < quick_cuest_data.nbasis; ++i) {
        for (int j = 0; j < quick_cuest_data.nbasis; ++j)
            printf ("%f ", get (P, i, j, quick_cuest_data.nbasis));
        putchar ('\n');
    }
    puts ("------ END uncorrected cuEST DENSITY MATRIX ------");
#endif

    // density matrix
    double *d_P;
    size_t  d_P_siz = d_J_siz;
    if (cudaMalloc ((void **)&d_P, d_P_siz)) {
        fprintf (stderr, "cudaMalloc failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    if (cudaMemcpy (d_P, P, d_P_siz, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    correct_o (P, CORRECT_REORDER_AND_NORM);

    cuestDFCoulombComputeParameters_t dfj_compute_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_DFCOULOMBCOMPUTE_PARAMETERS, &dfj_compute_params));
    checkCuestErrors (cuestDFCoulombComputeWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_struct.DFIntPlan, dfj_compute_params,
        quick_cuest_struct.tmpWD, d_P, d_J));

#ifdef CUESTDEBUG
    printf ("%s: coulomb tmpWD allocation size:\t%zu\n", __func__,
            quick_cuest_struct.tmpWD->deviceBufferSizeInBytes);
#endif
    cuestWorkspace_t *tmpDFJWorkspace = allocateWorkspace (quick_cuest_struct.tmpWD);
    checkCuestErrors (cuestDFCoulombCompute (quick_cuest_struct.handle,
                                             quick_cuest_struct.DFIntPlan, dfj_compute_params,
                                             tmpDFJWorkspace, d_P, d_J));

    freeWorkspace (tmpDFJWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_DFCOULOMBCOMPUTE_PARAMETERS, dfj_compute_params));

    // ======================== //
    // copy coulomb matrix to o //
    // ======================== //

    if (cudaMemcpy (o, d_J, d_J_siz, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    correct_o (o, CORRECT_REORDER_AND_NORM);

#ifdef CUESTDEBUG
    puts ("-------- J --------");
    for (int i = 0; i < quick_cuest_data.nbasis; ++i) {
        for (int j = 0; j < quick_cuest_data.nbasis; ++j)
            printf ("%16.10f", o[i * quick_cuest_data.nbasis + j]);
        putchar ('\n');
    }
    puts ("------ END J ------");
#endif

    if (cudaFree (d_J) != cudaSuccess) {
        fprintf (stderr, "cudaFree failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }
}

void
cuest_get_eri_K (double *o, double *C, int64_t nocc)
{
    double *d_K;
    size_t  d_K_siz = quick_cuest_data.nbasis * quick_cuest_data.nbasis * sizeof (double);
    if (cudaMalloc ((void **)&d_K, d_K_siz) != cudaSuccess) {
        fprintf (stderr, "cudaMalloc failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    correct_C (C, nocc, CORRECT_REORDER_AND_NORM);

    double *d_C;
    size_t  d_C_siz = nocc * quick_cuest_data.nbasis * sizeof (double);
    if (cudaMalloc ((void **)&d_C, d_C_siz)) {
        fprintf (stderr, "cudaMalloc failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    if (cudaMemcpy (d_C, C, d_C_siz, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    correct_C (C, nocc, CORRECT_REORDER_AND_NORM);

    cuestDFSymmetricExchangeComputeParameters_t dfk_compute_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS, &dfk_compute_params));

    cuestWorkspaceDescriptor_t *varBufSiz = malloc (sizeof (cuestWorkspaceDescriptor_t));
    varBufSiz->hostBufferSizeInBytes      = 0;
    varBufSiz->deviceBufferSizeInBytes    = 2e9; // TODO(michaelyxsun): adapt this. 2 GB right now
    checkCuestErrors (cuestDFSymmetricExchangeComputeWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_struct.DFIntPlan, dfk_compute_params, varBufSiz,
        quick_cuest_struct.tmpWD, nocc, d_C, d_K));

#ifdef CUESTDEBUG
    printf ("%s: exchange tmpWD allocation size:\t%zu\n", __func__,
            quick_cuest_struct.tmpWD->deviceBufferSizeInBytes);
#endif
    cuestWorkspace_t *tmpDFKWorkspace = allocateWorkspace (quick_cuest_struct.tmpWD);
    checkCuestErrors (cuestDFSymmetricExchangeCompute (
        quick_cuest_struct.handle, quick_cuest_struct.DFIntPlan, dfk_compute_params, varBufSiz,
        tmpDFKWorkspace, nocc, d_C, d_K));

    free (varBufSiz);
    freeWorkspace (tmpDFKWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS, dfk_compute_params));

    // ========================= //
    // copy exchange matrix to o //
    // ========================= //

    if (cudaMemcpy (o, d_K, d_K_siz, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    correct_o (o, CORRECT_REORDER_AND_NORM);

#ifdef CUESTDEBUG
    puts ("-------- K --------");
    for (int i = 0; i < quick_cuest_data.nbasis; ++i) {
        for (int j = 0; j < quick_cuest_data.nbasis; ++j)
            printf ("%16.10f", o[i * quick_cuest_data.nbasis + j]);
        putchar ('\n');
    }
    puts ("------ END K ------");
#endif

    if (cudaFree (d_K) != cudaSuccess) {
        fprintf (stderr, "cudaFree failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }
}

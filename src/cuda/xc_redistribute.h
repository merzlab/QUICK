
#ifndef XC_REDISTRIBUTE
#define XC_REDISTRIBUTE
#ifdef CUDA_MPIV
#ifdef __cplusplus
extern "C" {
#endif

void getAdjustment(int mpisize, int mpirank, int count);


#ifdef __cplusplus
}
#endif
#endif
#endif

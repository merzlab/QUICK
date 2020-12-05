
#ifndef XC_REDISTRIBUTE
#define XC_REDISTRIBUTE
#ifdef CUDA_MPIV
#ifdef __cplusplus
extern "C" {
#endif

int getAdjustment(int mpisize, int mpirank, int count);


#ifdef __cplusplus
}
#endif
#endif
#endif

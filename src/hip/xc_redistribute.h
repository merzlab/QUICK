
#ifndef XC_REDISTRIBUTE
#define XC_REDISTRIBUTE
#ifdef HIP_MPIV
#ifdef __cplusplus
extern "C" {
#endif

int getAdjustment(int mpisize, int mpirank, int count);

void sswderRedistribute(int mpisize, int mpirank, int count, int ncount,
  double *gridx, double *gridy, double *gridz, double *exc, double *quadwt, int *gatm,
  double *ngridx, double *ngridy, double *ngridz, double *nexc, double *nquadwt, int *ngatm);

#ifdef __cplusplus
}
#endif
#endif
#endif
